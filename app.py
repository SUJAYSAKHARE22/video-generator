import os
import json
import uuid

from dotenv import load_dotenv
from flask import Flask, render_template, request, send_file, abort, jsonify
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip

from ai_planner import get_ai_plan
from renderer import render_video
from camera_config import DEFAULT_CONFIG

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")


@app.route('/')
def index():
    return render_template('index.html')


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
        probe_clip = VideoFileClip(input_path)
        duration = probe_clip.duration
        probe_clip.close()

        plan = get_ai_plan(input_path, duration, api_key=NVIDIA_API_KEY, config=DEFAULT_CONFIG)
        render_video(input_path, output_path, plan, config=DEFAULT_CONFIG)

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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
