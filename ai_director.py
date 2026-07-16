"""
ai_director.py
---------------
The AI "director" - now a two-stage pipeline instead of one fragile
multi-image request.

Why: the hosted NVIDIA NIM vision endpoint used here only reliably accepts
ONE small image per request; sending a whole sequence of frames in a single
call mostly returned 400 Bad Request, which silently fell back to the
heuristic planner on almost every window - and since the heuristic path was
the one actually running, every video ended up looking similar (it wasn't
really "the AI deciding", it was always the same fallback logic).

New approach:

  Stage 1 - PER-FRAME DESCRIPTION (single image, reliable):
    Each sampled frame in a batch gets ONE small, cheap vision call asking
    for a compact structured description (salient point, notable UI
    change). Near-static frames are skipped entirely (no visual change to
    describe) to save calls and keep the transcript signal-dense rather
    than repetitive.

  Stage 2 - TEXT-ONLY CINEMATIC REASONING (no images, reliable):
    The per-frame descriptions + the real cursor kinematics summary
    (clicks, hovers, redirects, arrivals, scrolls, dominant motion vector)
    + the previous camera state are combined into a single text prompt.
    The model reasons over this like a montage transcript and returns the
    full segment plan for the window. Text-only requests have no image
    payload limits, so this call actually succeeds instead of silently
    degrading - which is what makes different videos produce genuinely
    different plans instead of always falling through to one heuristic.
"""

import json
import re
from typing import List, Optional

import requests

from camera_config import DEFAULT_CONFIG
from frame_sampler import FrameBatch, SampledFrame

NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_VISION_MODEL = "meta/llama-3.2-11b-vision-instruct"
NVIDIA_TEXT_MODEL = "meta/llama-3.1-70b-instruct"

VALID_ACTIONS = {"static", "hold", "zoom_in", "zoom_out", "pan", "track_cursor"}
VALID_EASINGS = {"linear", "easeOutQuart", "easeInQuart", "easeInOutQuart",
                  "easeInOutCubic", "easeOutCubic", "easeInOutSine"}
VALID_PAN_DIRS = {"none", "left", "right", "up", "down",
                   "up-left", "up-right", "down-left", "down-right"}

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

_DESCRIPTION_CACHE = {}


_DESCRIPTION_TIMEOUTS = (20, 35)


def _describe_frame(frame: SampledFrame, api_key: str) -> Optional[str]:
    """Stage 1: single-image, single-frame description. This is the call
    shape the endpoint reliably accepts, so it actually returns useful
    content instead of failing. One retry with a longer timeout before
    giving up on this frame, since a slow/cold endpoint on the first try
    shouldn't blank out the whole transcript."""
    cache_key = frame.b64_jpeg[:64]
    if cache_key in _DESCRIPTION_CACHE:
        return _DESCRIPTION_CACHE[cache_key]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    prompt = (
        "One frame from a screen-recording software demo. In ONE short sentence (<25 words), "
        "describe: the single most visually important point/element right now (roughly where on "
        "screen, e.g. top-left/center/bottom-right), and whether anything looks like a modal, "
        "dropdown, popup, or newly-opened panel. Be concrete, no preamble."
    )
    payload = {
        "model": NVIDIA_VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame.b64_jpeg}"}},
            ],
        }],
        "temperature": 0.2,
        "max_tokens": 80,
    }

    last_error = None
    for timeout in _DESCRIPTION_TIMEOUTS:
        try:
            resp = requests.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            text = " ".join(text.split())[:220]
            _DESCRIPTION_CACHE[cache_key] = text
            return text
        except requests.exceptions.Timeout:
            last_error = f"timed out after {timeout}s"
            continue
        except Exception as exc:  # noqa: BLE001 - best-effort per-frame call
            last_error = str(exc)
            continue

    print(f"ai_director: frame description failed at t={frame.t:.2f}s: {last_error}")
    return None


def _build_frame_transcript(batch: FrameBatch, api_key: str, config) -> str:
    """Runs Stage 1 across the batch, skipping near-static filler frames
    (nothing changed, so nothing worth describing/spending a call on),
    and returns a compact chronological transcript."""
    lines = []
    for frame in batch.frames:
        if frame.near_static and not frame.is_event_anchor:
            continue
        desc = _describe_frame(frame, api_key)
        if desc:
            tag = "[event] " if frame.is_event_anchor else ""
            lines.append(f"t={frame.t:.2f}s {tag}{desc}")
    if not lines:
        return "No frame descriptions available for this window."
    return "\n".join(lines)


def _build_system_prompt(config) -> str:
    return f"""You are an expert cinematic camera director for software product-demo videos, in the
style of high-quality SaaS marketing/product-launch videos.

You do NOT see raw images. Instead you are given, for one time window:
1) a chronological transcript of short frame descriptions (what changed, where, when)
2) the real measured cursor kinematics for that window: clicks, hovers, direction changes
   ("redirect" = attention moving to a new target), decelerations near a target ("arrival" =
   about to interact), scrolls, and the dominant motion direction
3) the camera's state at the end of the previous window (for continuity)

Reason over this exactly like a human editor watching the clip:
- A "redirect" event means the user's attention moved somewhere new - the camera should
  usually retarget toward that new direction, not keep tracking the old point.
- An "arrival" event means the cursor is about to interact - anticipate slightly ahead of it,
  don't lag behind.
- A click is a strong, brief focus point; a hover of 1s+ signals sustained importance.
- A modal/dropdown/popup mentioned in the transcript should usually override cursor tracking
  and become the primary focus while it's the newest thing on screen.
- Two videos with different cursor behaviour MUST get different camera plans - base every
  decision on the specific events and transcript given, never on a generic default pattern.
- Prefer STABLE framing: do not invent movement where the transcript/kinematics show none.
  If the window is genuinely static, a "static" or "hold" segment covering it is correct.
- zoomLevel 1 = no zoom, 10 = maximum zoom; most segments should be 2-6.
- Segments must be contiguous, non-overlapping, and together must exactly cover the window.

Config for this render: min_zoom={config.min_zoom}, max_zoom={config.max_zoom},
tracking_sensitivity={config.tracking_sensitivity}, cursor_influence={config.cursor_influence}.

Respond with ONLY a single JSON object, no markdown fences, no commentary, matching exactly:
{SEGMENT_SCHEMA_HINT}"""


def _user_text_prompt(batch: FrameBatch, transcript: str, signal_summary: str,
                       prev_state: Optional[dict]) -> str:
    prev = "none (this is the first window)" if not prev_state else json.dumps(prev_state)
    return (
        f"Time window: {batch.start_t:.2f}s to {batch.end_t:.2f}s.\n\n"
        f"Frame transcript:\n{transcript}\n\n"
        f"Measured cursor kinematics/events: {signal_summary}\n\n"
        f"Previous camera state at the end of the last window: {prev}\n\n"
        f"Produce the camera plan segments covering exactly this window."
    )


def _call_text_model(api_key, system_prompt, user_text, timeout):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    payload = {
        "model": NVIDIA_TEXT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.4,
        "top_p": 1,
        "max_tokens": 900,
    }
    return requests.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=timeout)


# Retry schedule for the reasoning call: hosted inference endpoints can be
# slow to cold-start or queue under load, so a single 45s timeout with no
# retry meant almost every batch silently fell through to the heuristic
# fallback the moment the endpoint was even slightly slow. Retrying with a
# longer timeout gives the model a real chance before giving up on the
# window.
_REASONING_TIMEOUTS = (60, 90)


def request_segment_plan(batch: FrameBatch, signal_summary: str, prev_state: Optional[dict],
                          api_key: str, config=DEFAULT_CONFIG) -> Optional[dict]:
    """Full two-stage director call for one batch/window. Returns the raw
    parsed JSON plan, or None if both stages could not produce anything
    usable (caller falls back to signal-driven heuristics for this window
    only, not the whole video)."""
    transcript = _build_frame_transcript(batch, api_key, config)
    system_prompt = _build_system_prompt(config)
    user_text = _user_text_prompt(batch, transcript, signal_summary, prev_state)

    last_error = None
    for attempt, timeout in enumerate(_REASONING_TIMEOUTS):
        try:
            resp = _call_text_model(api_key, system_prompt, user_text, timeout)
            if resp.status_code != 200:
                last_error = f"{resp.status_code}: {resp.text[:300]}"
                continue
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            match = re.search(r"\{.*\}", text, re.DOTALL)
            raw = json.loads(match.group(0)) if match else json.loads(text)
            return raw
        except requests.exceptions.Timeout:
            last_error = f"timed out after {timeout}s (attempt {attempt + 1}/{len(_REASONING_TIMEOUTS)})"
            continue
        except Exception as exc:  # noqa: BLE001 - best-effort AI call, caller has a real fallback
            last_error = str(exc)
            continue

    print(f"ai_director: batch {batch.start_t:.2f}-{batch.end_t:.2f}s failed, using heuristic fallback ({last_error})")
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
