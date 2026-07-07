import os
import json
import base64
import re
import random

import cv2
import requests

from activity_detector import detect_zoom_fragments

NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODEL = "meta/llama-3.2-11b-vision-instruct"  # free NIM vision model

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


def _extract_frames_b64(video_path, count=1, max_side=512, quality=55):
    """Extracts a small number of frames and keeps each base64 payload small.
    The hosted NVIDIA catalog endpoint for this vision model only reliably
    accepts a single, small (<180KB base64) image per request - sending
    multiple/large images causes a 400 Bad Request."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frames = []
    for i in range(count):
        idx = int(total * (i + 0.5) / count)
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, max(total - 1, 0)))
        ok, frame = cap.read()
        if not ok:
            continue
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
        if ok:
            frames.append(base64.b64encode(buf.tobytes()).decode("utf-8"))
    cap.release()
    return frames


def _default_plan(duration):
    frag_len = min(2.2, duration / 3 if duration > 0 else 2.0)
    fragments = []
    if duration > 4:
        s1 = max(0.3, duration * 0.18)
        fragments.append({
            "startTime": round(s1, 2), "endTime": round(min(s1 + frag_len, duration - 0.2), 2),
            "zoomLevel": 4, "speed": 5, "focusX": 50, "focusY": 35,
            "movementEnabled": False, "movementEndX": 50, "movementEndY": 35,
        })
        s2 = max(0.3, duration * 0.62)
        fragments.append({
            "startTime": round(s2, 2), "endTime": round(min(s2 + frag_len, duration - 0.2), 2),
            "zoomLevel": 3, "speed": 5, "focusX": 65, "focusY": 55,
            "movementEnabled": True, "movementEndX": 40, "movementEndY": 50,
        })
    return {
        "background": {"type": "gradient", "paletteIndex": random.randint(0, len(PALETTES) - 1)},
        "mockup": {
            "style": "browser", "darkMode": True, "frameColor": FRAME_COLORS_DARK[0],
            "url": "app.yourproduct.com", "cornerRadius": 14, "padding": 6,
        },
        "zoomFragments": fragments,
        "caption": None,
    }


def _sanitize_plan(raw, duration):
    plan = _default_plan(duration)
    if not isinstance(raw, dict):
        return plan

    bg = raw.get("background")
    if isinstance(bg, dict):
        if bg.get("type") == "solid" and isinstance(bg.get("color"), str) and re.match(r"^#[0-9a-fA-F]{6}$", bg["color"]):
            plan["background"] = {"type": "solid", "color": bg["color"]}
        else:
            idx = bg.get("paletteIndex", 0)
            try:
                idx = int(idx) % len(PALETTES)
            except (TypeError, ValueError):
                idx = 0
            plan["background"] = {"type": "gradient", "paletteIndex": idx}

    mk = raw.get("mockup")
    if isinstance(mk, dict):
        dark = bool(mk.get("darkMode", True))
        frame_color = mk.get("frameColor")
        if not (isinstance(frame_color, str) and re.match(r"^#[0-9a-fA-F]{6}$", frame_color)):
            frame_color = FRAME_COLORS_DARK[0] if dark else FRAME_COLORS_LIGHT[0]
        plan["mockup"] = {
            "style": mk.get("style", "browser") if mk.get("style") in ("browser", "minimal", "none") else "browser",
            "darkMode": dark,
            "frameColor": frame_color,
            "url": str(mk.get("url", "app.yourproduct.com"))[:60] or "app.yourproduct.com",
            "cornerRadius": int(_clamp(mk.get("cornerRadius", 14), 0, 28, 14)),
            "padding": int(_clamp(mk.get("padding", 6), 0, 14, 6)),
        }

    frags = raw.get("zoomFragments")
    if isinstance(frags, list) and frags:
        clean = []
        for f in frags[:6]:
            if not isinstance(f, dict):
                continue
            st = _clamp(f.get("startTime", 0), 0, max(duration - 0.5, 0), 0)
            et = _clamp(f.get("endTime", st + 2), st + 0.4, duration, min(st + 2, duration))
            if et <= st:
                continue
            clean.append({
                "startTime": round(st, 2),
                "endTime": round(et, 2),
                "zoomLevel": int(_clamp(f.get("zoomLevel", 3), 1, 10, 3)),
                "speed": int(_clamp(f.get("speed", 5), 1, 10, 5)),
                "focusX": int(_clamp(f.get("focusX", 50), 0, 100, 50)),
                "focusY": int(_clamp(f.get("focusY", 50), 0, 100, 50)),
                "movementEnabled": bool(f.get("movementEnabled", False)),
                "movementEndX": int(_clamp(f.get("movementEndX", 50), 0, 100, 50)),
                "movementEndY": int(_clamp(f.get("movementEndY", 50), 0, 100, 50)),
            })
        clean.sort(key=lambda x: x["startTime"])
        deoverlapped = []
        last_end = 0
        for f in clean:
            if f["startTime"] < last_end:
                f["startTime"] = last_end + 0.1
            if f["endTime"] <= f["startTime"]:
                continue
            deoverlapped.append(f)
            last_end = f["endTime"]
        if deoverlapped:
            plan["zoomFragments"] = deoverlapped
        # Note: this branch only fires if the caller explicitly passed
        # zoomFragments into _sanitize_plan (e.g. tests). The live AI prompt
        # no longer asks the model to invent zoom fragments - those come
        # from real cursor/UI motion tracking in activity_detector.py,
        # since a single static frame gives the model no way to know where
        # the cursor actually moves or clicks over time.

    cap = raw.get("caption")
    if isinstance(cap, dict) and isinstance(cap.get("text"), str) and cap["text"].strip():
        st = _clamp(cap.get("startTime", 0), 0, max(duration - 0.5, 0), 0)
        et = _clamp(cap.get("endTime", st + 3), st + 0.5, duration, min(st + 3, duration))
        plan["caption"] = {"text": cap["text"].strip()[:90], "startTime": round(st, 2), "endTime": round(et, 2)}
    else:
        plan["caption"] = None

    return plan


def _build_prompt(duration, palette_count):
    return f"""You are an automatic video editing director. You will see one sample frame from a screen-recording of a software product demo, {duration:.1f} seconds long.

Decide the visual styling for this demo: a background style, a browser device mockup frame, and one short caption if useful. (Camera zoom/pan timing is handled separately by motion analysis, not by you.)

Respond with ONLY a single JSON object, no markdown, no explanation, matching exactly this schema:
{{
  "background": {{"type": "gradient", "paletteIndex": 0-{palette_count - 1}}},
  "mockup": {{"style": "browser", "darkMode": true/false, "frameColor": "#hex", "url": "short realistic app url", "cornerRadius": 0-28, "padding": 0-14}},
  "caption": {{"text": "short caption or null", "startTime": number, "endTime": number}}
}}

Rules:
- Pick colors/mockup style that visually complement the dominant colors seen in the frame.
- caption can be null if nothing meaningful to say.
Return ONLY the JSON object."""


def get_ai_plan(video_path, duration, api_key=None):
    api_key = api_key or os.environ.get("NVIDIA_API_KEY")
    plan = _default_plan(duration)

    # Real cursor/UI-activity tracking across the whole timeline - this is
    # what decides WHERE and WHEN to zoom (clicks, cursor moving to another
    # corner, panels opening, etc). A single static frame can never tell us
    # this, so it's computed deterministically instead of guessed by the AI.
    try:
        motion_fragments = detect_zoom_fragments(video_path, duration)
        if motion_fragments:
            plan["zoomFragments"] = motion_fragments
    except Exception as exc:
        print(f"Activity/motion detection failed, using default zoom fragments: {exc}")

    if not api_key:
        return plan

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    try:
        frames_b64 = _extract_frames_b64(video_path, count=1)
        prompt_text = _build_prompt(duration, len(PALETTES))

        if frames_b64:
            content = [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frames_b64[0]}"}},
            ]
        else:
            content = prompt_text

        payload = {
            "model": NVIDIA_MODEL,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.4,
            "top_p": 1,
            "max_tokens": 512,
        }

        resp = requests.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=60)
        if not resp.ok:
            print(f"NVIDIA NIM error {resp.status_code}: {resp.text[:2000]}")
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]

        match = re.search(r"\{.*\}", text, re.DOTALL)
        raw_json = json.loads(match.group(0)) if match else json.loads(text)

        styled_plan = _sanitize_plan(raw_json, duration)
        # Keep the motion-based zoom fragments regardless of what the AI
        # returned - only take background/mockup/caption from the AI.
        styled_plan["zoomFragments"] = plan["zoomFragments"]
        plan = styled_plan
    except Exception as exc:
        print(f"AI styling fallback (using default background/mockup) due to: {exc}")

    return plan
