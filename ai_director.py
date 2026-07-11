"""
ai_director.py
---------------
The AI "director": given a batch of chronologically-ordered sampled
frames plus structured cursor/scene signals for that time window, asks a
vision-capable LLM to produce a complete cinematic camera plan for the
window - deciding whether/when/how much/where to zoom, pan, track the
cursor, or hold still. Nothing about camera behaviour is hardcoded here;
this module only shapes the request/response contract with the model and
validates/repairs whatever comes back.
"""

import json
import os
import re
from typing import List, Optional

import requests

from camera_config import DEFAULT_CONFIG
from frame_sampler import FrameBatch

NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODEL = "meta/llama-3.2-11b-vision-instruct"

VALID_ACTIONS = {"static", "hold", "zoom_in", "zoom_out", "pan", "track_cursor"}
VALID_EASINGS = {"linear", "easeOutQuart", "easeInQuart", "easeInOutQuart",
                  "easeInOutCubic", "easeOutCubic", "easeInOutSine"}
VALID_PAN_DIRS = {"none", "left", "right", "up", "down", "up-left", "up-right", "down-left", "down-right"}


SEGMENT_SCHEMA_HINT = """{
  "segments": [
    {
      "startTime": number,
      "endTime": number,
      "action": "static|hold|zoom_in|zoom_out|pan|track_cursor",
      "focusX": 0-100,
      "focusY": 0-100,
      "movementEnabled": true|false,
      "movementEndX": 0-100,
      "movementEndY": 0-100,
      "zoomLevel": 1-10,
      "panDirection": "none|left|right|up|down",
      "easing": "easeInOutCubic|easeOutQuart|linear|easeInOutSine",
      "transitionType": "smooth|hold|cut",
      "importance": 0-1,
      "confidence": 0-1,
      "reasoning": "one short sentence, optional"
    }
  ]
}"""


def _build_system_prompt(config) -> str:
    return f"""You are an expert cinematic camera director for software product-demo videos, in the
style of high-quality SaaS marketing/product-launch videos. You are shown a chronological
SEQUENCE of frames sampled from a screen recording (not isolated screenshots) along with
detected cursor/UI events for that same time window. Treat the sequence as a short movie clip.

You decide the camera behaviour as a professional human video editor would:
- whether to zoom, when, how much, and where
- when to pan, when to track the cursor, when to deliberately ignore it
- when to stay static (this is often the correct choice - do not zoom constantly)
- when a modal/dialog/dropdown should override cursor tracking and become the focus
- how long each movement should last, with natural easing (no sudden jumps or jitter)

Rules:
- Prefer STABLE framing. Do not invent unnecessary movement.
- A cursor moving toward a target matters more than its current position - anticipate.
- Hovering ~1s+ over one area signals importance; repeated clicks signal importance.
- A large centered modal/dialog opening should typically become the primary focus and override
  cursor tracking while it is on screen.
- zoomLevel range is 1 (no zoom) to 10 (maximum zoom); most segments should be 2-6.
- Segments must be contiguous or leave brief static gaps; do not overlap.
- Configuration for this render: min_zoom={config.min_zoom}, max_zoom={config.max_zoom},
  tracking_sensitivity={config.tracking_sensitivity}, cursor_influence={config.cursor_influence}.

Respond with ONLY a single JSON object, no markdown fences, no commentary, matching exactly:
{SEGMENT_SCHEMA_HINT}"""


def _user_text_prompt(batch: FrameBatch, signal_summary: str, prev_state: Optional[dict]) -> str:
    frame_times = ", ".join(f"{f.t:.2f}s" for f in batch.frames)
    prev = "none (this is the first window)" if not prev_state else json.dumps(prev_state)
    return (
        f"Time window: {batch.start_t:.2f}s to {batch.end_t:.2f}s.\n"
        f"Frames provided in order, timestamps: {frame_times}.\n"
        f"Detected cursor/UI events in this window: {signal_summary}\n"
        f"Previous camera state at the end of the last window: {prev}\n"
        f"Produce the camera plan segments covering exactly this window."
    )


def _content_with_images(text: str, frames) -> list:
    content = [{"type": "text", "text": text}]
    for f in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{f.b64_jpeg}"},
        })
    return content


def _call_model(api_key, system_prompt, user_text, frames, timeout=60):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    payload = {
        "model": NVIDIA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _content_with_images(user_text, frames) if frames else user_text},
        ],
        "temperature": 0.35,
        "top_p": 1,
        "max_tokens": 1400,
    }
    resp = requests.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=timeout)
    return resp


def request_segment_plan(batch: FrameBatch, signal_summary: str, prev_state: Optional[dict],
                          api_key: str, config=DEFAULT_CONFIG) -> Optional[dict]:
    """Sends one batch of ordered frames (as a temporal sequence) + signal
    context to the vision model and returns the parsed raw JSON plan, or
    None if the model/endpoint could not be used for this batch. Some
    hosted vision endpoints reject multi-image payloads over a certain
    size, so this degrades gracefully: full batch -> half -> single frame
    -> text-only (signals alone, no image) -> give up for this batch."""
    system_prompt = _build_system_prompt(config)
    user_text = _user_text_prompt(batch, signal_summary, prev_state)

    frame_attempts = [
        batch.frames,
        batch.frames[: max(1, len(batch.frames) // 2)],
        batch.frames[:1],
        [],
    ]

    last_error = None
    for attempt_frames in frame_attempts:
        try:
            resp = _call_model(api_key, system_prompt, user_text, attempt_frames)
            if resp.status_code == 400:
                last_error = f"400 Bad Request with {len(attempt_frames)} frame(s): {resp.text[:300]}"
                continue
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            match = re.search(r"\{.*\}", text, re.DOTALL)
            raw = json.loads(match.group(0)) if match else json.loads(text)
            return raw
        except Exception as exc:  # noqa: BLE001 - broad by design, this is a best-effort AI call
            last_error = str(exc)
            continue

    print(f"AI director: batch {batch.start_t:.2f}-{batch.end_t:.2f}s failed, using heuristic fallback ({last_error})")
    return None


def sanitize_segments(raw: dict, window_start: float, window_end: float, config=DEFAULT_CONFIG) -> List[dict]:
    """Validates/clamps whatever the model returned into the exact schema
    the camera engine expects, discarding anything malformed rather than
    trusting the model blindly."""
    out = []
    if not isinstance(raw, dict):
        return out
    segs = raw.get("segments")
    if not isinstance(segs, list):
        return out

    for s in segs:
        if not isinstance(s, dict):
            continue
        try:
            start = max(window_start, min(float(s.get("startTime", window_start)), window_end))
            end = max(start + config.min_segment_duration, min(float(s.get("endTime", start + 1)), window_end + 0.01))
            end = min(end, window_end)
            if end - start < 0.15:
                continue
            action = s.get("action") if s.get("action") in VALID_ACTIONS else "hold"
            zoom_level = max(1.0, min(10.0, float(s.get("zoomLevel", 1))))
            focus_x = max(0.0, min(100.0, float(s.get("focusX", 50))))
            focus_y = max(0.0, min(100.0, float(s.get("focusY", 50))))
            move_x = max(0.0, min(100.0, float(s.get("movementEndX", focus_x))))
            move_y = max(0.0, min(100.0, float(s.get("movementEndY", focus_y))))
            easing = s.get("easing") if s.get("easing") in VALID_EASINGS else config.default_easing
            pan_dir = s.get("panDirection") if s.get("panDirection") in VALID_PAN_DIRS else "none"
            importance = max(0.0, min(1.0, float(s.get("importance", 0.5))))
            confidence = max(0.0, min(1.0, float(s.get("confidence", 0.5))))

            out.append({
                "startTime": round(start, 3),
                "endTime": round(end, 3),
                "action": action,
                "focusX": round(focus_x, 1),
                "focusY": round(focus_y, 1),
                "movementEnabled": bool(s.get("movementEnabled", False)) and action in ("pan", "track_cursor"),
                "movementEndX": round(move_x, 1),
                "movementEndY": round(move_y, 1),
                "zoomLevel": round(zoom_level, 1),
                "panDirection": pan_dir,
                "easing": easing,
                "transitionType": s.get("transitionType") if s.get("transitionType") in ("smooth", "hold", "cut") else "smooth",
                "importance": round(importance, 2),
                "confidence": round(confidence, 2),
                "reasoning": str(s.get("reasoning", ""))[:200] or None,
                "source": "ai",
            })
        except (TypeError, ValueError):
            continue

    out.sort(key=lambda x: x["startTime"])
    return out
