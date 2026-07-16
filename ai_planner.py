"""
ai_planner.py
--------------
Top-level entry point used by app.py. Produces the complete render plan:
  - visual styling (background palette, device mockup, optional caption)
  - the full cinematic camera segment timeline (delegated to motion_planner,
    which owns the real cursor-kinematics + two-stage AI director pipeline)

Kept as a thin orchestrator so app.py's interface (`get_ai_plan`) doesn't
change even though the camera planning system underneath it was redesigned.
The styling call now also receives a short activity summary from the
camera plan's own meta so the caption/background choice can react to what
actually happens in THIS video (e.g. "lots of clicks" vs "mostly static")
instead of being decided independently of it.
"""

import os
import json
import base64
import re
import random

import cv2
import requests

from camera_config import DEFAULT_CONFIG
from motion_planner import build_camera_plan

NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_VISION_MODEL = "meta/llama-3.2-11b-vision-instruct"

PALETTES = [
    [{"color": "#FFFBFD", "position": 0}, {"color": "#FFE5F5", "position": 33}, {"color": "#F769EA", "position": 66}, {"color": "#4A1C9B", "position": 100}],
    [{"color": "#0D0B2E", "position": 0}, {"color": "#1A1B63", "position": 25}, {"color": "#2E5BFF", "position": 50}, {"color": "#00D4FF", "position": 75}, {"color": "#E0FFFF", "position": 100}],
    [{"color": "#1B262C", "position": 0}, {"color": "#0F4C75", "position": 25}, {"color": "#3282B8", "position": 50}, {"color": "#BBE1FA", "position": 75}, {"color": "#FFFFFF", "position": 100}],
    [{"color": "#110B2A", "position": 0}, {"color": "#6465F1", "position": 33}, {"color": "#92DAFF", "position": 66}, {"color": "#F9F7F8", "position": 100}],
    [{"color": "#000428", "position": 0}, {"color": "#004E92", "position": 33}, {"color": "#00B4DB", "position": 66}, {"color": "#A8E063", "position": 100}],
    [{"color": "#0f172a", "position": 0}, {"color": "#1e293b", "position": 25}, {"color": "#334155", "position": 50}, {"color": "#475569", "position": 75}, {"color": "#64748b", "position": 100}],
    [{"color": "#e0e7ff", "position": 0}, {"color": "#c7d2fe", "position": 33}, {"color": "#818cf8", "position": 66}, {"color": "#4f46e5", "position": 100}],
    [{"color": "#134E5E", "position": 0}, {"color": "#71B280", "position": 100}],
    [{"color": "#0f0c29", "position": 0}, {"color": "#302b63", "position": 50}, {"color": "#24243e", "position": 100}],
    [{"color": "#232526", "position": 0}, {"color": "#414345", "position": 100}],
    [{"color": "#141E30", "position": 0}, {"color": "#243B55", "position": 100}],
    [{"color": "#2b5876", "position": 0}, {"color": "#4e4376", "position": 100}],
]

FRAME_COLORS_DARK = ["#1e1e1e", "#181818", "#252526", "#0d1117", "#1a1b26", "#282a36"]
FRAME_COLORS_LIGHT = ["#ffffff", "#f3f3f3", "#f6f8fa", "#fafafa", "#e8e8e8"]


def _clamp(v, lo, hi, default):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def _extract_style_frame_b64(video_path, max_side=512, quality=55):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(int(total * 0.5), max(total - 1, 0)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    q = quality
    for _ in range(5):
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok and len(buf) * 4 / 3 < 170_000:
            break
        q = max(20, q - 10)
    return base64.b64encode(buf.tobytes()).decode("utf-8") if ok else None


def _default_style(duration):
    return {
        "background": {"type": "gradient", "paletteIndex": random.randint(0, len(PALETTES) - 1)},
        "mockup": {
            "style": "browser", "darkMode": True, "frameColor": FRAME_COLORS_DARK[0],
            "url": "app.yourproduct.com", "cornerRadius": 14, "padding": 6,
        },
        "caption": None,
    }


def _sanitize_style(raw, duration):
    style = _default_style(duration)
    if not isinstance(raw, dict):
        return style

    bg = raw.get("background")
    if isinstance(bg, dict):
        if bg.get("type") == "solid" and isinstance(bg.get("color"), str) and re.match(r"^#[0-9a-fA-F]{6}$", bg["color"]):
            style["background"] = {"type": "solid", "color": bg["color"]}
        else:
            idx = bg.get("paletteIndex", 0)
            try:
                idx = int(idx) % len(PALETTES)
            except (TypeError, ValueError):
                idx = 0
            style["background"] = {"type": "gradient", "paletteIndex": idx}

    mk = raw.get("mockup")
    if isinstance(mk, dict):
        dark = bool(mk.get("darkMode", True))
        frame_color = mk.get("frameColor")
        if not (isinstance(frame_color, str) and re.match(r"^#[0-9a-fA-F]{6}$", frame_color)):
            frame_color = FRAME_COLORS_DARK[0] if dark else FRAME_COLORS_LIGHT[0]
        style["mockup"] = {
            "style": mk.get("style", "browser") if mk.get("style") in ("browser", "minimal", "none") else "browser",
            "darkMode": dark,
            "frameColor": frame_color,
            "url": str(mk.get("url", "app.yourproduct.com"))[:60] or "app.yourproduct.com",
            "cornerRadius": int(_clamp(mk.get("cornerRadius", 14), 0, 28, 14)),
            "padding": int(_clamp(mk.get("padding", 6), 0, 14, 6)),
        }

    cap = raw.get("caption")
    if isinstance(cap, dict) and isinstance(cap.get("text"), str) and cap["text"].strip():
        st = _clamp(cap.get("startTime", 0), 0, max(duration - 0.5, 0), 0)
        et = _clamp(cap.get("endTime", st + 3), st + 0.5, duration, min(st + 3, duration))
        style["caption"] = {"text": cap["text"].strip()[:90], "startTime": round(st, 2), "endTime": round(et, 2)}
    else:
        style["caption"] = None

    return style


def _build_style_prompt(duration, palette_count, activity_hint):
    return f"""You are an automatic video editing director. You will see one sample frame from a screen-recording of a software product demo, {duration:.1f} seconds long.
Activity detected across the whole recording: {activity_hint}

Decide the visual styling for this demo: a background style, a browser device mockup frame, and one short caption if useful. (Camera zoom/pan timing is handled separately by a dedicated motion-planning system, not by you.)

Respond with ONLY a single JSON object, no markdown, no explanation, matching exactly this schema:
{{
  "background": {{"type": "gradient", "paletteIndex": 0-{palette_count - 1}}},
  "mockup": {{"style": "browser", "darkMode": true/false, "frameColor": "#hex", "url": "short realistic app url", "cornerRadius": 0-28, "padding": 0-14}},
  "caption": {{"text": "short caption or null", "startTime": number, "endTime": number}}
}}

Rules:
- Pick colors/mockup style that visually complement the dominant colors seen in the frame.
- caption can be null if nothing meaningful to say; if you do write one, let it reflect the actual detected activity rather than a generic line.
Return ONLY the JSON object."""


def get_ai_style(video_path, duration, api_key=None, activity_hint="unknown"):
    style = _default_style(duration)
    if not api_key:
        return style

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    try:
        frame_b64 = _extract_style_frame_b64(video_path)
        prompt_text = _build_style_prompt(duration, len(PALETTES), activity_hint)

        if frame_b64:
            content = [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
            ]
        else:
            content = prompt_text

        payload = {
            "model": NVIDIA_VISION_MODEL,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.4,
            "top_p": 1,
            "max_tokens": 512,
        }

        resp = requests.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=60)
        if not resp.ok:
            print(f"NVIDIA NIM style error {resp.status_code}: {resp.text[:2000]}")
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]

        match = re.search(r"\{.*\}", text, re.DOTALL)
        raw_json = json.loads(match.group(0)) if match else json.loads(text)
        style = _sanitize_style(raw_json, duration)
    except Exception as exc:
        print(f"AI styling fallback (using default background/mockup) due to: {exc}")

    return style


def get_ai_plan(video_path, duration, api_key=None, config=DEFAULT_CONFIG):
    """Builds the complete render plan: the cinematic camera segment
    timeline (real cursor kinematics + two-stage AI director + stability
    guards) plus visual styling that reacts to what the camera plan
    actually found in this specific video."""
    api_key = api_key or os.environ.get("NVIDIA_API_KEY")

    try:
        camera = build_camera_plan(video_path, duration, api_key=api_key, config=config)
    except Exception as exc:
        print(f"Camera motion planning failed entirely, falling back to a single static segment: {exc}")
        camera = {
            "segments": [{
                "startTime": 0, "endTime": duration or 1.0, "action": "static",
                "focusX": 50, "focusY": 50, "movementEnabled": False,
                "movementEndX": 50, "movementEndY": 50, "zoomLevel": 1.0,
                "panDirection": "none", "easing": config.default_easing,
                "transitionType": "hold", "importance": 0.0, "confidence": 0.0,
                "reasoning": "planning pipeline error fallback", "source": "error_fallback",
            }],
            "meta": {"totalEvents": 0, "eventTypes": [], "batches": 0, "aiUsed": False},
        }

    meta = camera.get("meta", {})
    activity_hint = (
        f"{meta.get('totalEvents', 0)} cursor/UI events detected "
        f"(types: {', '.join(meta.get('eventTypes', [])) or 'none'})"
    )
    style = get_ai_style(video_path, duration, api_key=api_key, activity_hint=activity_hint)

    plan = {
        "background": style["background"],
        "mockup": style["mockup"],
        "caption": style["caption"],
        "segments": camera["segments"],
        "planningMeta": meta,
    }
    return plan


def regenerate_style_with_narrative(video_path, duration, narrative: dict, api_key=None):
    """Human-in-the-loop restyle: re-runs ONLY the visual-styling call
    (background/mockup/caption), factoring in narrative intent a human
    supplies - target audience, tone, pacing preference - that the AI has
    no way to infer from pixels/cursor signals alone. Does not touch the
    camera segment timeline; combine with per-segment overrides
    (`Clip.set_segment_override`) for full "Edit More" control."""
    api_key = api_key or os.environ.get("NVIDIA_API_KEY")
    audience = narrative.get("audience", "general software users")
    tone = narrative.get("tone", "professional")
    pacing = narrative.get("pacing", "moderate")
    activity_hint = (
        f"Human-specified narrative intent - audience: {audience}; tone: {tone}; "
        f"pacing preference: {pacing}. Reflect this intent in the styling/caption choice."
    )
    return get_ai_style(video_path, duration, api_key=api_key, activity_hint=activity_hint)
