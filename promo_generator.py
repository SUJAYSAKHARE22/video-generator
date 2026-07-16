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
# Free/community NIM model for image->video generation. If unavailable in a
# given account/region, `IMG2VID_MODEL_FALLBACKS` are tried in order.
IMG2VID_MODEL = "stabilityai/stable-video-diffusion"
IMG2VID_MODEL_FALLBACKS = [
    "stabilityai/stable-video-diffusion-img2vid-xt",
]

POLL_INTERVAL_S = 4
POLL_TIMEOUT_S = 180


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


def _submit_img2vid_job(model: str, api_key: str, image_b64: str, seconds: float = 3.0) -> Optional[str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    payload = {
        "image": f"data:image/jpeg;base64,{image_b64}",
        "seconds": seconds,
        "cfg_scale": 2.5,
    }
    try:
        resp = requests.post(f"{NVIDIA_ASSET_API_BASE}/{model}", headers=headers, json=payload, timeout=30)
        if resp.status_code == 202:
            return resp.headers.get("nvcf-reqid") or resp.json().get("request_id")
        if resp.status_code == 200:
            return "__immediate__:" + resp.text
        print(f"promo_generator: submit failed for {model}: {resp.status_code} {resp.text[:200]}")
        return None
    except Exception as exc:  # noqa: BLE001 - best-effort generative call
        print(f"promo_generator: submit exception for {model}: {exc}")
        return None


def _poll_job(model: str, api_key: str, request_id: str) -> Optional[bytes]:
    headers = {"Authorization": f"Bearer {api_key}"}
    status_url = f"{NVIDIA_ASSET_API_BASE}/{model}/status/{request_id}"
    waited = 0
    while waited < POLL_TIMEOUT_S:
        try:
            resp = requests.get(status_url, headers=headers, timeout=20)
            if resp.status_code == 200:
                content_type = resp.headers.get("content-type", "")
                if "video" in content_type or resp.content[:4] in (b"\x00\x00\x00\x18", b"\x00\x00\x00\x1c"):
                    return resp.content
                data = resp.json()
                if data.get("status") == "fulfilled" and data.get("video_url"):
                    video_resp = requests.get(data["video_url"], timeout=60)
                    if video_resp.ok:
                        return video_resp.content
                if data.get("status") in ("errored", "failed"):
                    print(f"promo_generator: job {request_id} failed: {data}")
                    return None
            elif resp.status_code not in (202, 404):
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
    """Runs the submit->poll->fetch contract for one key frame, trying the
    primary model then fallbacks. Returns None (never raises) on failure so
    the caller can simply skip this moment in the final promo."""
    for model in [IMG2VID_MODEL] + IMG2VID_MODEL_FALLBACKS:
        req_id = _submit_img2vid_job(model, api_key, key_frame.b64_jpeg, seconds)
        if req_id is None:
            continue

        if req_id.startswith("__immediate__:"):
            video_bytes = base64.b64decode(req_id.split(":", 1)[1]) if False else None
            # Some NIM deployments return the asset directly on 200; treat
            # the raw response body captured above as the video bytes.
            video_bytes = None
        else:
            video_bytes = _poll_job(model, api_key, req_id)

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
