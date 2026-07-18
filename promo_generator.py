"""
promo_generator.py
--------------------
Generates a short AI promo/trailer video as a companion output alongside
the full cinematic demo. Re-uses the same signal the camera director
already computed (the segment plan's `importance`/`reasoning` fields) to
pick the most visually meaningful moments, sends those key frames to an
NVIDIA NIM image-to-video generation model, and stitches the generated
clips into one short promo with transitions, a title card, and optional
background music via `editing_engine`.

NVIDIA NIM's hosted generative-video models (e.g. Stable Video Diffusion,
served at ai.api.nvidia.com) use an async submit -> poll -> fetch contract
rather than a single synchronous call, since video generation takes much
longer than a chat completion. This module implements that contract with
graceful degradation: if a given key frame's generation job fails or times
out, that frame is simply skipped rather than aborting the whole promo.
"""

"""
promo_generator.py
--------------------
Generates a short AI promo/trailer video as a companion output alongside
the full cinematic demo. Re-uses the same signal the camera director
already computed (the segment plan's `importance`/`reasoning` fields) to
pick the most visually meaningful moments, sends those key frames to an
NVIDIA-hosted image/video-conditioned video generation model, and stitches
the generated clips into one short promo with transitions, a title card,
and optional background music via `editing_engine`.

Model choice: NVIDIA's own Cosmos-Predict1 world-foundation models are used
as the primary generator (`nvidia/cosmos-predict1-7b`, falling back to
`nvidia/cosmos-predict1-5b`) rather than third-party community models,
because they are the models NVIDIA currently lists as "Free Endpoint" in
the build.nvidia.com catalog - some previously-available third-party
functions (e.g. `stabilityai/stable-video-diffusion`) can return
"Function ... not found for account" on accounts where that function is
no longer deployed, since the free catalog changes over time. Cosmos is
kept as the last-resort fallback for accounts where it IS still enabled.

Response contract: NVIDIA's hosted generative endpoints vary between a
synchronous 200 response (JSON with a base64 video field, or a raw video
body) and an async 202-with-poll contract (NVCF request-id header, then
poll a status URL until the asset is ready) depending on the model and
how long generation takes. This module handles both shapes per model and
degrades gracefully: if a given key frame's generation job fails, times
out, or the model is unavailable on this account, that frame is simply
skipped rather than aborting the whole promo.
"""

import base64
import os
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional

import cv2
import requests

from camera_config import DEFAULT_CONFIG

NVIDIA_ASSET_API_BASE = "https://ai.api.nvidia.com/v1/genai"


def _default_models():
    return [
        {"model": "nvidia/cosmos-predict1-7b", "shape": "cosmos"},
        {"model": "nvidia/cosmos-predict1-5b", "shape": "cosmos"},
        {"model": "stabilityai/stable-video-diffusion", "shape": "svd"},
    ]


def _models_from_env():
    """NVIDIA's free-tier video-generation catalog changes over time and
    which functions are actually deployed varies per account - hardcoded
    guesses have repeatedly 404'd. Set NVIDIA_VIDEO_GEN_MODELS as a
    comma-separated `org/model:shape` list (shape is "cosmos" or "svd", the
    two payload/response conventions this module knows how to speak) to
    override the built-in guesses with whatever GET /promo/check-models
    reports as actually working on your account."""
    raw = os.environ.get("NVIDIA_VIDEO_GEN_MODELS", "").strip()
    if not raw:
        return _default_models()
    entries = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            model, shape = chunk.rsplit(":", 1)
        else:
            model, shape = chunk, "cosmos"
        entries.append({"model": model.strip(), "shape": shape.strip()})
    return entries or _default_models()


IMG2VID_MODELS = _models_from_env()

POLL_INTERVAL_S = 4
POLL_TIMEOUT_S = 180

DEFAULT_PROMPT = (
    "Smooth, professional camera continuation of a software product demo screen "
    "recording, subtle realistic motion, clean modern UI, no sudden changes."
)


@dataclass
class KeyFrame:
    t: float
    b64_jpeg: str
    importance: float
    reasoning: Optional[str]


@dataclass
class GeneratedClip:
    key_time: float
    local_path: str


def select_key_frames(video_path: str, plan: dict, max_frames: int = 4,
                       max_side: int = 768, quality: int = 80) -> List[KeyFrame]:
    """Picks the highest-importance moments from the already-computed
    camera plan (no separate analysis pass needed) and extracts a frame at
    the midpoint of each, so the promo is generated from the same moments
    the cinematic editor decided mattered most."""
    segments = plan.get("segments", [])
    if not segments:
        return []

    ranked = sorted(segments, key=lambda s: s.get("importance", 0), reverse=True)
    chosen = []
    seen_windows = []
    for seg in ranked:
        mid = (seg["startTime"] + seg["endTime"]) / 2
        if any(abs(mid - w) < 1.5 for w in seen_windows):
            continue
        chosen.append(seg)
        seen_windows.append(mid)
        if len(chosen) >= max_frames:
            break
    chosen.sort(key=lambda s: s["startTime"])

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    key_frames = []
    for seg in chosen:
        mid = (seg["startTime"] + seg["endTime"]) / 2
        frame_idx = min(int(mid * fps), max(total - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            continue
        key_frames.append(KeyFrame(
            t=mid, b64_jpeg=base64.b64encode(buf.tobytes()).decode("utf-8"),
            importance=seg.get("importance", 0.5), reasoning=seg.get("reasoning"),
        ))

    cap.release()
    return key_frames


def _build_payload(shape: str, image_b64: str, prompt: str) -> dict:
    """Each model family expects different field names - this is the one
    place that difference is encoded, so callers just pass a KeyFrame."""
    if shape == "cosmos":
        # nvidia/cosmos-predict1-{5b,7b} Video2World: image + text prompt.
        return {
            "prompt": prompt[:1200],
            "image": f"data:image/jpeg;base64,{image_b64}",
            "seed": 0,
        }
    if shape == "svd":
        # stabilityai/stable-video-diffusion: exact documented schema -
        # only "image" is required; motion_bucket_id only accepts 127.
        return {
            "image": f"data:image/jpeg;base64,{image_b64}",
            "seed": 0,
            "cfg_scale": 1.8,
            "motion_bucket_id": 127,
        }
    return {"image": f"data:image/jpeg;base64,{image_b64}"}


def _extract_video_from_response(resp) -> Optional[bytes]:
    """Handles every response shape NVIDIA's hosted generative endpoints
    can return for a completed (200) response: raw video bytes, or JSON
    with a base64 field under a few possible keys."""
    content_type = resp.headers.get("content-type", "")
    if "video" in content_type:
        return resp.content
    try:
        data = resp.json()
    except ValueError:
        # Not JSON and not tagged as video - only trust it if it's non-trivially large.
        return resp.content if len(resp.content) > 1000 else None

    for key in ("b64_video", "video", "artifact", "output"):
        val = data.get(key)
        if isinstance(val, str) and len(val) > 100:
            try:
                return base64.b64decode(val)
            except Exception:  # noqa: BLE001 - fall through to next key
                continue
    if data.get("video_url"):
        video_resp = requests.get(data["video_url"], timeout=60)
        if video_resp.ok:
            return video_resp.content
    return None


def _submit_job(model: str, shape: str, api_key: str, image_b64: str, prompt: str) -> tuple:
    """Returns ('done', video_bytes), ('pending', request_id), or (None, error_str)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    payload = _build_payload(shape, image_b64, prompt)
    try:
        resp = requests.post(f"{NVIDIA_ASSET_API_BASE}/{model}", headers=headers, json=payload, timeout=60)
    except Exception as exc:  # noqa: BLE001 - best-effort generative call
        return None, f"request exception: {exc}"

    if resp.status_code == 200:
        video = _extract_video_from_response(resp)
        return ("done", video) if video else (None, "200 response but no video payload found")
    if resp.status_code == 202:
        req_id = resp.headers.get("nvcf-reqid")
        if req_id:
            return "pending", req_id
        return None, "202 response but no nvcf-reqid header"
    return None, f"{resp.status_code}: {resp.text[:300]}"


def _poll_job(model: str, api_key: str, request_id: str) -> Optional[bytes]:
    headers = {"Authorization": f"Bearer {api_key}"}
    status_url = f"{NVIDIA_ASSET_API_BASE}/{model}/status/{request_id}"
    waited = 0
    while waited < POLL_TIMEOUT_S:
        try:
            resp = requests.get(status_url, headers=headers, timeout=20)
            if resp.status_code == 200:
                video = _extract_video_from_response(resp)
                if video:
                    return video
                print(f"promo_generator: job {request_id} returned 200 with no extractable video")
                return None
            if resp.status_code not in (202, 404):
                print(f"promo_generator: poll error {resp.status_code}: {resp.text[:200]}")
                return None
        except Exception as exc:  # noqa: BLE001 - best-effort generative call
            print(f"promo_generator: poll exception: {exc}")
        time.sleep(POLL_INTERVAL_S)
        waited += POLL_INTERVAL_S
    print(f"promo_generator: job {request_id} timed out after {POLL_TIMEOUT_S}s")
    return None


def generate_clip_for_keyframe(key_frame: KeyFrame, api_key: str, work_dir: str,
                                seconds: float = 3.0) -> Optional[GeneratedClip]:
    """Tries each model in IMG2VID_MODELS in order until one produces a
    video, handling both synchronous and async (submit->poll) responses.
    Returns None (never raises) if every model fails on this account, so
    the caller can simply skip this moment in the final promo."""
    prompt = (key_frame.reasoning or DEFAULT_PROMPT).strip() or DEFAULT_PROMPT

    for entry in IMG2VID_MODELS:
        model, shape = entry["model"], entry["shape"]
        status, payload = _submit_job(model, shape, api_key, key_frame.b64_jpeg, prompt)

        video_bytes = None
        if status == "done":
            video_bytes = payload
        elif status == "pending":
            video_bytes = _poll_job(model, api_key, payload)
        else:
            print(f"promo_generator: {model} unavailable for t={key_frame.t:.2f}s: {payload}")
            continue

        if video_bytes:
            out_path = os.path.join(work_dir, f"promo_seg_{uuid.uuid4().hex[:8]}.mp4")
            with open(out_path, "wb") as fh:
                fh.write(video_bytes)
            return GeneratedClip(key_time=key_frame.t, local_path=out_path)

    print(f"promo_generator: all models failed for key frame at t={key_frame.t:.2f}s, skipping")
    return None


def build_promo_video(video_path: str, plan: dict, api_key: Optional[str], output_path: str,
                       work_dir: str = "output/promo_tmp", max_frames: int = 4,
                       clip_seconds: float = 3.0, title_text: Optional[str] = None,
                       music_path: Optional[str] = None, config=DEFAULT_CONFIG) -> Optional[str]:
    """Full promo pipeline: pick key frames from the plan the AI director
    already produced, generate a short animated clip per key frame via NIM
    image-to-video, and stitch them into one promo with crossfade
    transitions, an optional title card, and optional background music.
    Returns the output path, or None if no clips could be generated at all
    (e.g. no API key, or every generation job failed) - callers should
    treat that as "promo unavailable this run", not a hard error, since the
    main cinematic demo output is unaffected either way."""
    if not api_key:
        print("promo_generator: no API key, skipping promo generation.")
        return None

    os.makedirs(work_dir, exist_ok=True)

    key_frames = select_key_frames(video_path, plan, max_frames=max_frames)
    if not key_frames:
        print("promo_generator: no key frames available (empty plan), skipping promo.")
        return None

    generated: List[GeneratedClip] = []
    for kf in key_frames:
        clip = generate_clip_for_keyframe(kf, api_key, work_dir, seconds=clip_seconds)
        if clip:
            generated.append(clip)

    if not generated:
        print("promo_generator: no generated clips succeeded, skipping promo.")
        return None

    from moviepy.editor import VideoFileClip, concatenate_videoclips
    from project_model import Project, Clip, TextOverlay

    project = Project(id=uuid.uuid4().hex[:8], name="promo")
    for i, g in enumerate(generated):
        project.add_clip(source_path=g.local_path, trim_start=0.0, trim_end=clip_seconds)
        project.clips[-1].transition_in = "crossfade" if i > 0 else "none"
        project.clips[-1].transition_duration = 0.5
        if title_text and i == 0:
            project.clips[-1].text_overlays.append(
                TextOverlay(text=title_text, startTime=0.0, endTime=min(2.0, clip_seconds), style="title", position="center")
            )
    project.export_preset = "youtube_shorts"

    if music_path and os.path.exists(music_path):
        from project_model import AudioTrack
        project.audio_tracks.append(AudioTrack(path=music_path, volume=0.8, duck_under_original=False))

    from editing_engine import render_project
    render_project(project, output_path, config)

    for g in generated:
        try:
            os.remove(g.local_path)
        except OSError:
            pass

    return output_path


_TINY_TEST_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkI"
    "CQkKDA8MCgsOCwkJDRENDg8QEBEQCgwSExIQEw8QEBD/2wBDAQMDAwQDBAgEBAgQCwkLEBAQEBAQ"
    "EBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBD/wAARCAABAAEDASIA"
    "AhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAj/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEB"
    "AQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX"
    "/9k="
)


def check_model_availability(api_key: str) -> dict:
    """Probes every configured candidate model with a throwaway 1x1 test
    image and reports the exact status for each - no polling, no video is
    generated, and nothing is written to disk. Use this to find out which
    model (if any) is actually callable on your account instead of relying
    on hardcoded guesses, since NVIDIA's free-tier catalog and per-account
    deployments change over time."""
    results = []
    for entry in IMG2VID_MODELS:
        model, shape = entry["model"], entry["shape"]
        url = f"{NVIDIA_ASSET_API_BASE}/{model}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        payload = _build_payload(shape, _TINY_TEST_JPEG_B64, DEFAULT_PROMPT)
        entry_result = {"model": model, "shape": shape, "url": url}
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=20)
            entry_result["status_code"] = resp.status_code
            entry_result["response_preview"] = resp.text[:300]
            entry_result["verdict"] = (
                "usable (200 = returned immediately, 202 = accepted and pollable)"
                if resp.status_code in (200, 202)
                else "not usable on this account/key with this URL"
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic call, never raises
            entry_result["status_code"] = None
            entry_result["response_preview"] = str(exc)
            entry_result["verdict"] = "request failed (network/timeout)"
        results.append(entry_result)

    return {
        "checked": results,
        "note": (
            "Set NVIDIA_VIDEO_GEN_MODELS as a comma-separated org/model:shape list "
            "(shape is 'cosmos' or 'svd') to whichever model(s) above show status "
            "200 or 202, to stop relying on the built-in guesses."
        ),
    }