import os
import json
import time
import uuid

from dotenv import load_dotenv
from flask import Flask, render_template, request, send_file, abort, jsonify
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip

from ai_planner import get_ai_plan, regenerate_style_with_narrative
from renderer import render_video
from camera_config import DEFAULT_CONFIG
from promo_generator import build_promo_video
from metrics_engine import compute_camera_metrics, compute_processing_metrics, compute_cross_video_diversity
from project_model import (
    Project, Effect, TextOverlay, AudioTrack, EXPORT_PRESETS,
    new_project, save_project, load_project,
)
from editing_engine import render_project, detect_silences

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['PROJECT_FOLDER'] = 'projects'
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024

for folder in (app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], app.config['PROJECT_FOLDER']):
    os.makedirs(folder, exist_ok=True)

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")

_METRICS_HISTORY = []  # in-memory log of recent plans, for cross-video diversity metrics


def _project_path(project_id):
    return os.path.join(app.config['PROJECT_FOLDER'], f"{project_id}.json")


def _load_project_or_404(project_id):
    path = _project_path(project_id)
    if not os.path.exists(path):
        abort(404, f"Project {project_id} not found.")
    return load_project(path)


@app.route('/')
def index():
    return render_template('index.html')


# ---------------------------------------------------------------------------
# One-shot cinematic auto-edit (original behaviour)
# ---------------------------------------------------------------------------

@app.route('/process-video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return abort(400, "Missing screen capture video asset.")

    file = request.files['video']
    if file.filename == '':
        return abort(400, "No file selected.")

    job_id = uuid.uuid4().hex[:10]
    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
    output_filename = f"ai_promo_{job_id}.mp4"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    file.save(input_path)

    try:
        t0 = time.time()
        probe_clip = VideoFileClip(input_path)
        duration = probe_clip.duration
        probe_clip.close()

        plan = get_ai_plan(input_path, duration, api_key=NVIDIA_API_KEY, config=DEFAULT_CONFIG)
        render_video(input_path, output_path, plan, config=DEFAULT_CONFIG)
        elapsed = time.time() - t0

        _METRICS_HISTORY.append(plan)
        if len(_METRICS_HISTORY) > 25:
            _METRICS_HISTORY.pop(0)

        return send_file(output_path, as_attachment=True, download_name=f"cinematic_promo_{job_id}.mp4")

    except Exception as e:
        print(f"CRITICAL COMPILATION FAILURE SYSTEM HALT: {str(e)}")
        return abort(500, f"Video render pipeline crash details: {str(e)}")
    finally:
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except OSError:
                pass


@app.route('/plan-preview', methods=['POST'])
def plan_preview():
    if 'video' not in request.files:
        return abort(400, "Missing video asset.")
    file = request.files['video']
    tmp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"preview_{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}")
    file.save(tmp_path)
    try:
        clip = VideoFileClip(tmp_path)
        duration = clip.duration
        clip.close()
        plan = get_ai_plan(tmp_path, duration, api_key=NVIDIA_API_KEY, config=DEFAULT_CONFIG)
        return jsonify(plan)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Promo video generation (NVIDIA NIM image-to-video)
# ---------------------------------------------------------------------------

@app.route('/generate-promo', methods=['POST'])
def generate_promo():
    if 'video' not in request.files:
        return abort(400, "Missing video asset.")
    file = request.files['video']
    job_id = uuid.uuid4().hex[:10]
    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"promo_{job_id}.mp4")
    file.save(input_path)

    title_text = request.form.get("title")
    max_frames = int(request.form.get("max_frames", 4))

    try:
        clip = VideoFileClip(input_path)
        duration = clip.duration
        clip.close()

        plan = get_ai_plan(input_path, duration, api_key=NVIDIA_API_KEY, config=DEFAULT_CONFIG)
        result = build_promo_video(
            input_path, plan, api_key=NVIDIA_API_KEY, output_path=output_path,
            work_dir=os.path.join(app.config['OUTPUT_FOLDER'], f"promo_tmp_{job_id}"),
            max_frames=max_frames, title_text=title_text, config=DEFAULT_CONFIG,
        )
        if result is None:
            return abort(502, "Promo generation unavailable this run (no key frames generated). "
                              "The full cinematic demo endpoint (/process-video) is unaffected.")
        return send_file(result, as_attachment=True, download_name=f"promo_{job_id}.mp4")
    except Exception as e:
        print(f"Promo generation failure: {str(e)}")
        return abort(500, f"Promo generation crash details: {str(e)}")
    finally:
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Project / timeline editor (multi-clip, trim, effects, text, audio, export)
# ---------------------------------------------------------------------------

@app.route('/project', methods=['POST'])
def create_project():
    name = request.json.get("name", "Untitled Project") if request.is_json else "Untitled Project"
    project = new_project(name)
    save_project(project, _project_path(project.id))
    return jsonify(project.to_dict())


@app.route('/project/<project_id>', methods=['GET'])
def get_project(project_id):
    project = _load_project_or_404(project_id)
    return jsonify(project.to_dict())


@app.route('/project/<project_id>/clip', methods=['POST'])
def add_clip(project_id):
    project = _load_project_or_404(project_id)
    if 'video' not in request.files:
        return abort(400, "Missing video file for clip.")
    file = request.files['video']
    filename = secure_filename(file.filename)
    stored_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{project_id}_{uuid.uuid4().hex[:6]}_{filename}")
    file.save(stored_path)

    clip = VideoFileClip(stored_path)
    duration = clip.duration
    clip.close()

    use_ai_camera = request.form.get("cinematic", "true").lower() == "true"
    camera_plan = None
    if use_ai_camera:
        camera_plan = get_ai_plan(stored_path, duration, api_key=NVIDIA_API_KEY, config=DEFAULT_CONFIG)

    new_clip = project.add_clip(stored_path, trim_start=0.0, trim_end=duration, camera_plan=camera_plan)
    save_project(project, _project_path(project_id))
    return jsonify(new_clip.to_dict())


@app.route('/project/<project_id>/clip/<clip_id>/trim', methods=['POST'])
def trim_clip(project_id, clip_id):
    project = _load_project_or_404(project_id)
    data = request.get_json(force=True)
    clip = next((c for c in project.clips if c.id == clip_id), None)
    if clip is None:
        return abort(404, "Clip not found.")
    clip.trim_start = float(data.get("trim_start", clip.trim_start))
    clip.trim_end = float(data.get("trim_end", clip.trim_end))
    save_project(project, _project_path(project_id))
    return jsonify(clip.to_dict())


@app.route('/project/<project_id>/clip/<clip_id>/split', methods=['POST'])
def split_clip(project_id, clip_id):
    project = _load_project_or_404(project_id)
    data = request.get_json(force=True)
    split_time = float(data["split_time"])
    new_clip = project.split_clip(clip_id, split_time)
    if new_clip is None:
        return abort(400, "Invalid split time for this clip.")
    save_project(project, _project_path(project_id))
    return jsonify(new_clip.to_dict())


@app.route('/project/<project_id>/clip/<clip_id>', methods=['DELETE'])
def remove_clip(project_id, clip_id):
    project = _load_project_or_404(project_id)
    project.remove_clip(clip_id)
    save_project(project, _project_path(project_id))
    return jsonify({"removed": clip_id})


@app.route('/project/<project_id>/reorder', methods=['POST'])
def reorder_clips(project_id):
    project = _load_project_or_404(project_id)
    ordered_ids = request.get_json(force=True).get("order", [])
    project.reorder_clips(ordered_ids)
    save_project(project, _project_path(project_id))
    return jsonify(project.to_dict())


@app.route('/project/<project_id>/clip/<clip_id>/effect', methods=['POST'])
def add_effect(project_id, clip_id):
    project = _load_project_or_404(project_id)
    clip = next((c for c in project.clips if c.id == clip_id), None)
    if clip is None:
        return abort(404, "Clip not found.")
    data = request.get_json(force=True)
    clip.effects.append(Effect(kind=data["kind"], params=data.get("params", {}),
                                startTime=data.get("startTime"), endTime=data.get("endTime")))
    save_project(project, _project_path(project_id))
    return jsonify(clip.to_dict())


@app.route('/project/<project_id>/clip/<clip_id>/text', methods=['POST'])
def add_text(project_id, clip_id):
    project = _load_project_or_404(project_id)
    clip = next((c for c in project.clips if c.id == clip_id), None)
    if clip is None:
        return abort(404, "Clip not found.")
    data = request.get_json(force=True)
    clip.text_overlays.append(TextOverlay(
        text=data["text"], startTime=float(data["startTime"]), endTime=float(data["endTime"]),
        position=data.get("position", "bottom"), style=data.get("style", "caption"),
    ))
    save_project(project, _project_path(project_id))
    return jsonify(clip.to_dict())


@app.route('/project/<project_id>/clip/<clip_id>/transition', methods=['POST'])
def set_transition(project_id, clip_id):
    project = _load_project_or_404(project_id)
    clip = next((c for c in project.clips if c.id == clip_id), None)
    if clip is None:
        return abort(404, "Clip not found.")
    data = request.get_json(force=True)
    clip.transition_in = data.get("transition_in", "none")
    clip.transition_duration = float(data.get("transition_duration", 0.6))
    save_project(project, _project_path(project_id))
    return jsonify(clip.to_dict())


@app.route('/project/<project_id>/audio', methods=['POST'])
def add_audio(project_id):
    project = _load_project_or_404(project_id)
    if 'audio' not in request.files:
        return abort(400, "Missing audio file.")
    file = request.files['audio']
    filename = secure_filename(file.filename)
    stored_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{project_id}_audio_{uuid.uuid4().hex[:6]}_{filename}")
    file.save(stored_path)

    volume = float(request.form.get("volume", 1.0))
    duck = request.form.get("duck_under_original", "true").lower() == "true"
    duck_level = float(request.form.get("duck_level", 0.25))

    project.audio_tracks.append(AudioTrack(path=stored_path, volume=volume,
                                            duck_under_original=duck, duck_level=duck_level))
    save_project(project, _project_path(project_id))
    return jsonify(project.to_dict())


@app.route('/project/<project_id>/export-preset', methods=['POST'])
def set_export_preset(project_id):
    project = _load_project_or_404(project_id)
    preset = request.get_json(force=True).get("preset")
    if preset not in EXPORT_PRESETS:
        return abort(400, f"Unknown preset. Options: {list(EXPORT_PRESETS.keys())}")
    project.export_preset = preset
    save_project(project, _project_path(project_id))
    return jsonify(project.to_dict())


@app.route('/project/<project_id>/undo', methods=['POST'])
def undo(project_id):
    project = _load_project_or_404(project_id)
    ok = project.undo()
    save_project(project, _project_path(project_id))
    return jsonify({"undone": ok, "project": project.to_dict()})


@app.route('/project/<project_id>/redo', methods=['POST'])
def redo(project_id):
    project = _load_project_or_404(project_id)
    ok = project.redo()
    save_project(project, _project_path(project_id))
    return jsonify({"redone": ok, "project": project.to_dict()})


@app.route('/project/<project_id>/render', methods=['POST'])
def render_project_route(project_id):
    project = _load_project_or_404(project_id)
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"project_{project_id}.mp4")
    try:
        render_project(project, output_path, DEFAULT_CONFIG)
        return send_file(output_path, as_attachment=True, download_name=f"{project.name}.mp4")
    except Exception as e:
        return abort(500, f"Project render crash details: {str(e)}")


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

@app.route('/metrics/plan', methods=['POST'])
def metrics_for_plan():
    """Computes camera-plan metrics for a single freshly-uploaded video
    without producing a full render - useful for evaluation/benchmarking."""
    if 'video' not in request.files:
        return abort(400, "Missing video asset.")
    file = request.files['video']
    tmp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"metrics_{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}")
    file.save(tmp_path)
    try:
        clip = VideoFileClip(tmp_path)
        duration = clip.duration
        clip.close()
        t0 = time.time()
        plan = get_ai_plan(tmp_path, duration, api_key=NVIDIA_API_KEY, config=DEFAULT_CONFIG)
        elapsed = time.time() - t0

        _METRICS_HISTORY.append(plan)
        if len(_METRICS_HISTORY) > 25:
            _METRICS_HISTORY.pop(0)

        return jsonify({
            "camera_metrics": compute_camera_metrics(plan),
            "processing_metrics": compute_processing_metrics(elapsed, duration),
        })
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route('/metrics/diversity', methods=['GET'])
def metrics_diversity():
    """Cross-video diversity metrics across every plan generated so far in
    this server's runtime - the evidence for 'different videos produce
    different camera behaviour, not a repeated pattern'."""
    return jsonify(compute_cross_video_diversity(_METRICS_HISTORY))


# ---------------------------------------------------------------------------
# Hybrid human/AI editing ("Edit More"): review, override, approve AI
# camera decisions; accept-as-is workflow; multi-take selection; narrative
# intent; silence-based cut suggestions. Every existing endpoint above is
# untouched - this is purely additive.
# ---------------------------------------------------------------------------

def _find_clip_or_404(project, clip_id):
    clip = next((c for c in project.clips if c.id == clip_id), None)
    if clip is None:
        abort(404, "Clip not found.")
    return clip


@app.route('/project/<project_id>/clip/<clip_id>/segments', methods=['GET'])
def list_segments(project_id, clip_id):
    """What the human reviews in 'Edit More': every AI-authored camera
    segment for this clip, with any existing human override/approval
    state merged in, plus which segments came from the AI vs. the
    heuristic fallback (source) so the human can see where the AI was
    confident vs. where it silently degraded."""
    project = _load_project_or_404(project_id)
    clip = _find_clip_or_404(project, clip_id)
    return jsonify({"clipId": clip.id, "segments": clip.effective_segments()})


@app.route('/project/<project_id>/clip/<clip_id>/segment/<int:index>/override', methods=['POST'])
def override_segment(project_id, clip_id, index):
    """Human edits one camera decision (action/focus/zoom/pan/easing/etc.)
    without touching any other segment or the underlying AI output."""
    project = _load_project_or_404(project_id)
    clip = _find_clip_or_404(project, clip_id)
    segments = (clip.camera_plan or {}).get("segments", [])
    if not (0 <= index < len(segments)):
        return abort(404, "Segment index out of range.")
    fields = request.get_json(force=True)
    clip.set_segment_override(index, fields)
    save_project(project, _project_path(project_id))
    return jsonify(clip.effective_segments()[index])


@app.route('/project/<project_id>/clip/<clip_id>/segment/<int:index>/override', methods=['DELETE'])
def clear_segment_override(project_id, clip_id, index):
    """Reverts one segment back to the AI's original decision."""
    project = _load_project_or_404(project_id)
    clip = _find_clip_or_404(project, clip_id)
    clip.clear_segment_override(index)
    save_project(project, _project_path(project_id))
    segments = clip.effective_segments()
    if not (0 <= index < len(segments)):
        return abort(404, "Segment index out of range.")
    return jsonify(segments[index])


@app.route('/project/<project_id>/clip/<clip_id>/segment/<int:index>/approve', methods=['POST'])
def approve_segment(project_id, clip_id, index):
    """Human explicitly signs off on one AI segment as-is (useful for
    tracking review coverage - 'has a human looked at every decision?')."""
    project = _load_project_or_404(project_id)
    clip = _find_clip_or_404(project, clip_id)
    segments = (clip.camera_plan or {}).get("segments", [])
    if not (0 <= index < len(segments)):
        return abort(404, "Segment index out of range.")
    clip.approve_segment(index)
    save_project(project, _project_path(project_id))
    return jsonify(clip.effective_segments()[index])


@app.route('/project/<project_id>/clip/<clip_id>/satisfaction', methods=['POST'])
def set_satisfaction(project_id, clip_id):
    """The core hybrid-workflow switch: 'accepted' means the human is happy
    with the AI's output as-is and can download directly; 'editing' means
    they're using the Edit More tools above. Purely a status flag - every
    editing endpoint works regardless of this value."""
    project = _load_project_or_404(project_id)
    state = request.get_json(force=True).get("state")
    clip = project.set_satisfaction(clip_id, state)
    if clip is None:
        return abort(404, "Clip not found.")
    save_project(project, _project_path(project_id))
    return jsonify(clip.to_dict())


@app.route('/project/<project_id>/clip/<clip_id>/narrative', methods=['POST'])
def set_narrative(project_id, clip_id):
    """Human supplies narrative intent (audience/tone/pacing) the AI has no
    way to infer on its own, and the styling (background/mockup/caption)
    is regenerated to reflect it. Camera segments are untouched here -
    pair with segment overrides above for full creative control."""
    project = _load_project_or_404(project_id)
    clip = _find_clip_or_404(project, clip_id)
    narrative = request.get_json(force=True)
    clip.narrative = narrative

    style = regenerate_style_with_narrative(
        clip.source_path, clip.duration, narrative, api_key=NVIDIA_API_KEY,
    )
    if clip.camera_plan is None:
        clip.camera_plan = {"segments": []}
    clip.camera_plan["background"] = style["background"]
    clip.camera_plan["mockup"] = style["mockup"]
    clip.camera_plan["caption"] = style["caption"]

    save_project(project, _project_path(project_id))
    return jsonify({"narrative": clip.narrative, "style": style})


@app.route('/project/<project_id>/clip/<clip_id>/silences', methods=['GET'])
def get_silences(project_id, clip_id):
    """AI SUGGESTS candidate silent stretches; nothing is cut automatically.
    A human reviews the list and, for any they want removed, calls the
    existing /split and DELETE clip endpoints (or trims around them) -
    this endpoint only proposes."""
    project = _load_project_or_404(project_id)
    clip = _find_clip_or_404(project, clip_id)
    threshold_db = float(request.args.get("threshold_db", -40.0))
    min_duration = float(request.args.get("min_duration", 0.5))
    silences = detect_silences(clip.source_path, threshold_db=threshold_db, min_duration=min_duration)
    return jsonify({"clipId": clip.id, "suggestedCuts": silences})


@app.route('/project/<project_id>/takes', methods=['POST'])
def create_take_group(project_id):
    """Human uploads multiple candidate recordings of the same moment; the
    AI never picks between them - see /takes/<id>/select below."""
    project = _load_project_or_404(project_id)
    if 'videos' not in request.files:
        return abort(400, "Missing candidate video files (field name 'videos').")
    label = request.form.get("label", "Untitled take")
    files = request.files.getlist('videos')
    stored_paths = []
    for f in files:
        filename = secure_filename(f.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], f"{project_id}_take_{uuid.uuid4().hex[:6]}_{filename}")
        f.save(path)
        stored_paths.append(path)
    group = project.add_take_group(label, stored_paths)
    save_project(project, _project_path(project_id))
    return jsonify(group.to_dict())


@app.route('/project/<project_id>/takes/<group_id>/select', methods=['POST'])
def select_take(project_id, group_id):
    """The human's choice of which take becomes the actual timeline clip."""
    project = _load_project_or_404(project_id)
    chosen_path = request.get_json(force=True).get("chosen_path")
    group = next((g for g in project.take_groups if g.id == group_id), None)
    if group is None:
        return abort(404, "Take group not found.")
    if chosen_path not in group.candidate_paths:
        return abort(400, "chosen_path must be one of the group's candidate_paths.")

    clip_obj = VideoFileClip(chosen_path)
    duration = clip_obj.duration
    clip_obj.close()

    use_ai_camera = request.get_json(force=True).get("cinematic", True)
    camera_plan = None
    if use_ai_camera:
        camera_plan = get_ai_plan(chosen_path, duration, api_key=NVIDIA_API_KEY, config=DEFAULT_CONFIG)

    new_clip = project.select_take(group_id, chosen_path, trim_start=0.0, trim_end=duration, camera_plan=camera_plan)
    if new_clip is None:
        return abort(400, "Could not select take.")
    save_project(project, _project_path(project_id))
    return jsonify(new_clip.to_dict())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
