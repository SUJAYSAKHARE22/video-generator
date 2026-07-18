"""
editing_engine.py
-------------------
The actual video-editor operations that render a `project_model.Project`
into a final video: trimming, multi-clip concatenation with transitions,
color/visual filters, text/title overlays, background music mixing with
optional ducking, and platform export presets. `renderer.py` still owns
turning ONE clip's cinematic camera plan into pixels (the AI zoom/pan
system); this module operates one level up, at the timeline/project level,
which is what a general-purpose editor needs on top of that.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from moviepy.editor import (
    VideoFileClip, VideoClip, CompositeVideoClip, CompositeAudioClip,
    AudioFileClip, concatenate_videoclips, vfx, afx,
)

from project_model import Project, Clip, EXPORT_PRESETS, SUPPORTED_FILTERS
from camera_config import DEFAULT_CONFIG
from camera_engine import compute_camera_state


def _font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Trim / split
# ---------------------------------------------------------------------------

def trim_source(source_path: str, start: float, end: float):
    """Returns a moviepy subclip for [start, end] of the source file. Does
    not write to disk - callers compose this into a timeline first, then
    export once, so trims stay non-destructive/lossless until final export."""
    clip = VideoFileClip(source_path)
    end = min(end, clip.duration)
    start = max(0.0, min(start, end - 0.05))
    return clip.subclip(start, end)


# ---------------------------------------------------------------------------
# Filters (color grade / stylistic)
# ---------------------------------------------------------------------------

def _apply_pixel_filter(frame: np.ndarray, kind: str, params: dict) -> np.ndarray:
    img = Image.fromarray(frame)

    if kind == "brightness":
        factor = float(params.get("factor", 1.15))
        arr = np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return arr

    if kind == "contrast":
        factor = float(params.get("factor", 1.2))
        mean = frame.mean()
        arr = np.clip((frame.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
        return arr

    if kind == "saturation":
        factor = float(params.get("factor", 1.3))
        hsv = np.array(img.convert("HSV"), dtype=np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
        return np.array(Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB"))

    if kind == "grayscale":
        gray = np.array(img.convert("L"))
        return np.stack([gray] * 3, axis=-1)

    if kind == "blur":
        radius = float(params.get("radius", 3))
        return np.array(img.filter(ImageFilter.GaussianBlur(radius)))

    if kind == "sharpen":
        return np.array(img.filter(ImageFilter.SHARPEN))

    if kind == "vignette":
        h, w = frame.shape[:2]
        strength = float(params.get("strength", 0.6))
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w / 2, h / 2
        dist = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
        mask = np.clip(1 - dist * strength, 0, 1)
        return np.clip(frame.astype(np.float32) * mask[..., None], 0, 255).astype(np.uint8)

    return frame


def apply_effects(clip, effects):
    """Applies a list of `project_model.Effect` to a moviepy clip, each
    optionally scoped to a [startTime, endTime] window within the clip."""
    if not effects:
        return clip

    def make_frame(get_frame, t):
        frame = get_frame(t)
        out = frame
        for eff in effects:
            if eff.kind not in SUPPORTED_FILTERS:
                continue
            if eff.startTime is not None and t < eff.startTime:
                continue
            if eff.endTime is not None and t > eff.endTime:
                continue
            out = _apply_pixel_filter(out, eff.kind, eff.params)
        return out

    return clip.fl(make_frame)


# ---------------------------------------------------------------------------
# Text / title overlays
# ---------------------------------------------------------------------------

_STYLE_SIZES = {"caption": 40, "title": 72, "lower_third": 34}


def _draw_text_frame(base_rgb: np.ndarray, text: str, position: str, style: str) -> np.ndarray:
    h, w = base_rgb.shape[:2]
    img = Image.fromarray(base_rgb).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    font = _font(_STYLE_SIZES.get(style, 40))
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 18

    if position == "top":
        y0 = 40
    elif position == "center":
        y0 = h // 2 - th // 2
    else:
        y0 = h - th - 100

    x0 = (w - tw) // 2

    if style != "lower_third":
        draw.rounded_rectangle([x0 - pad, y0 - pad, x0 + tw + pad, y0 + th + pad],
                                radius=14, fill=(10, 10, 15, 200))
        draw.text((x0, y0 - bbox[1]), text, fill=(255, 255, 255, 255), font=font)
    else:
        bar_y0 = y0 - pad
        draw.rectangle([0, bar_y0, w, bar_y0 + th + pad * 2], fill=(20, 20, 25, 190))
        draw.rectangle([0, bar_y0, 6, bar_y0 + th + pad * 2], fill=(80, 140, 255, 255))
        draw.text((40, bar_y0 + pad - bbox[1]), text, fill=(255, 255, 255, 255), font=font)

    return np.array(img.convert("RGB"))


def apply_text_overlays(clip, overlays):
    if not overlays:
        return clip

    def make_frame(get_frame, t):
        frame = get_frame(t)
        for ov in overlays:
            if ov.startTime <= t <= ov.endTime:
                frame = _draw_text_frame(frame, ov.text, ov.position, ov.style)
        return frame

    return clip.fl(make_frame)


# ---------------------------------------------------------------------------
# Per-clip render (trim + camera plan + effects + text)
# ---------------------------------------------------------------------------

def render_clip(clip_model: Clip, config=DEFAULT_CONFIG):
    """Renders one timeline Clip into a moviepy clip: trims the source,
    applies its cinematic camera plan (if present) frame-by-frame, composites
    the result through the SAME device-mockup + gradient-background Shell
    used by the single-shot pipeline (renderer.py) so a clip edited in the
    hybrid "Edit More" flow doesn't silently lose its styling, then draws
    the AI-authored caption (if any) and any manually-added text overlays on
    top. Compositing every clip to the same 1920x1080 canvas here also
    means multi-clip projects concatenate correctly even when the source
    recordings have different native resolutions."""
    from renderer import Shell as _Shell, rounded_mask  # local import: avoids a module-level cycle risk

    raw = trim_source(clip_model.source_path, clip_model.trim_start, clip_model.trim_end)
    # effective_segments() merges any human "Edit More" overrides on top of
    # the AI-authored segments (index-for-index), so a human override here
    # is what actually renders - the AI's original output stays untouched
    # and inspectable in clip_model.camera_plan.
    segments = clip_model.effective_segments()

    style_plan = clip_model.camera_plan or {}
    shell_plan = {
        "background": style_plan.get("background") or {"type": "gradient", "paletteIndex": 0},
        "mockup": style_plan.get("mockup") or {
            "style": "browser", "darkMode": True, "frameColor": "#1e1e1e",
            "url": "app.yourproduct.com", "cornerRadius": 14, "padding": 6,
        },
    }
    caption = style_plan.get("caption")

    video_w, video_h = raw.size
    shell = _Shell(shell_plan, video_w, video_h)
    cx, cy, cw, ch = shell.content_rect
    content_mask = rounded_mask(cw, ch, shell.content_radius, shell.content_corners)

    def make_frame(t):
        frame = raw.get_frame(t)
        h, w = frame.shape[:2]

        if segments:
            scale, fx, fy = compute_camera_state(segments, t, config)
            scale = max(scale, 1.0)
            crop_w, crop_h = w / scale, h / scale
            px, py = w * fx / 100.0, h * fy / 100.0
            x0 = min(max(px - crop_w / 2, 0), w - crop_w)
            y0 = min(max(py - crop_h / 2, 0), h - crop_h)
            cropped_arr = np.array(Image.fromarray(frame).crop(
                (int(x0), int(y0), int(x0 + crop_w), int(y0 + crop_h))
            ))
        else:
            cropped_arr = frame

        for eff in clip_model.effects:
            if eff.kind not in SUPPORTED_FILTERS:
                continue
            if eff.startTime is not None and t < eff.startTime:
                continue
            if eff.endTime is not None and t > eff.endTime:
                continue
            cropped_arr = _apply_pixel_filter(cropped_arr, eff.kind, eff.params)

        content_img = Image.fromarray(cropped_arr).resize((cw, ch), Image.LANCZOS)

        canvas = Image.fromarray(shell.canvas_base.copy()).convert("RGB")
        canvas.paste(content_img, (cx, cy), content_mask)
        canvas_arr = np.array(canvas)

        if caption and caption.get("startTime", 0) <= t <= caption.get("endTime", 0):
            canvas_arr = _draw_text_frame(canvas_arr, caption["text"], "bottom", "caption")

        for ov in clip_model.text_overlays:
            if ov.startTime <= t <= ov.endTime:
                canvas_arr = _draw_text_frame(canvas_arr, ov.text, ov.position, ov.style)

        return canvas_arr

    rendered = VideoClip(make_frame, duration=raw.duration)
    if raw.audio is not None:
        rendered = rendered.set_audio(raw.audio)
    return rendered


# ---------------------------------------------------------------------------
# Multi-clip timeline assembly with transitions
# ---------------------------------------------------------------------------

def _wipe_mask_frame(w, h, progress, direction):
    """Hard-edged directional wipe mask (1 = show incoming clip, 0 = show
    outgoing clip), progress 0..1 across the transition window."""
    mask = np.zeros((h, w), dtype=np.float32)
    edge = int(w * progress) if direction == "wipe_left" else int(w * (1 - progress))
    if direction == "wipe_left":
        mask[:, :edge] = 1.0
    else:
        mask[:, edge:] = 1.0
    return mask


def _build_wipe_transition_clip(prev_tail, next_head, duration, direction):
    """Builds the actual transition WINDOW as its own composite clip: for
    `duration` seconds, a hard vertical edge sweeps across the frame
    revealing `next_head` over `prev_tail` - a real directional wipe, not a
    crossfade standing in for one."""
    w, h = prev_tail.size

    def make_frame(t):
        progress = max(0.0, min(1.0, t / duration))
        mask = _wipe_mask_frame(w, h, progress, direction)
        prev_frame = prev_tail.get_frame(t)
        next_frame = next_head.get_frame(t)
        out = prev_frame * (1 - mask[..., None]) + next_frame * mask[..., None]
        return out.astype(np.uint8)

    transition_clip = VideoClip(make_frame, duration=duration)
    if next_head.audio is not None:
        transition_clip = transition_clip.set_audio(next_head.audio.set_duration(duration))
    return transition_clip


def _apply_transition(prev_clip, next_clip, transition: str, duration: float):
    """Splits `prev_clip`'s tail and `next_clip`'s head into a dedicated
    transition-window clip so each transition type renders as its actual
    visual effect, then returns (prev_body, transition_or_none, next_body)
    for the caller to concatenate in order."""
    duration = min(duration, prev_clip.duration / 2, next_clip.duration / 2)
    if duration <= 0.05 or transition == "none":
        return prev_clip, None, next_clip

    if transition == "crossfade":
        prev_body = prev_clip.subclip(0, prev_clip.duration - duration)
        next_body = next_clip.subclip(duration, next_clip.duration)
        prev_tail = prev_clip.subclip(prev_clip.duration - duration, prev_clip.duration)
        next_head = next_clip.subclip(0, duration).crossfadein(duration)
        window = CompositeVideoClip([prev_tail.crossfadeout(duration), next_head])
        return prev_body, window, next_body

    if transition == "fade_black":
        prev_body = prev_clip.subclip(0, prev_clip.duration - duration).fx(vfx.fadeout, duration / 2)
        next_body = next_clip.subclip(duration, next_clip.duration).fx(vfx.fadein, duration / 2)
        return prev_body, None, next_body

    if transition in ("wipe_left", "wipe_right"):
        prev_body = prev_clip.subclip(0, prev_clip.duration - duration)
        next_body = next_clip.subclip(duration, next_clip.duration)
        prev_tail = prev_clip.subclip(prev_clip.duration - duration, prev_clip.duration)
        next_head = next_clip.subclip(0, duration)
        window = _build_wipe_transition_clip(prev_tail, next_head, duration, transition)
        return prev_body, window, next_body

    return prev_clip, None, next_clip


def assemble_timeline(project: Project, config=DEFAULT_CONFIG):
    """Renders every clip in the project and concatenates them in order,
    inserting each clip's transition-in as its own real transition-window
    clip against the previous clip (crossfade dissolve, fade-to-black, or
    a genuine hard-edged directional wipe)."""
    ordered = sorted(project.clips, key=lambda c: c.order)
    if not ordered:
        raise ValueError("Project has no clips to assemble.")

    rendered = [render_clip(c, config) for c in ordered]

    if len(rendered) == 1:
        return rendered[0]

    sequence = [rendered[0]]
    for i in range(1, len(rendered)):
        prev = sequence[-1]
        cur = rendered[i]
        transition = ordered[i].transition_in
        duration = ordered[i].transition_duration
        prev_body, window, next_body = _apply_transition(prev, cur, transition, duration)
        sequence[-1] = prev_body
        if window is not None:
            sequence.append(window)
        sequence.append(next_body)

    return concatenate_videoclips(sequence, method="chain")


# ---------------------------------------------------------------------------
# Silence detection (human-reviewed cut suggestions - AI proposes, human decides)
# ---------------------------------------------------------------------------

def detect_silences(source_path: str, threshold_db: float = -40.0, min_duration: float = 0.5):
    """Scans the clip's audio track for stretches below `threshold_db` and
    returns them as candidate cut windows. This is deliberately exposed as
    SUGGESTIONS only (`/silences` in app.py) - a human reviews and chooses
    which, if any, to actually cut via `split_clip`/`trim_clip`; nothing
    here removes footage on its own."""
    clip = VideoFileClip(source_path)
    if clip.audio is None:
        clip.close()
        return []

    fps = 22050
    audio = clip.audio.set_fps(fps)
    array = audio.to_soundarray(fps=fps, nbytes=2)
    if array.ndim > 1:
        array = array.mean(axis=1)

    window = max(1, int(fps * 0.05))
    n_windows = len(array) // window
    rms = np.array([
        np.sqrt(np.mean(np.square(array[i * window:(i + 1) * window])) + 1e-12)
        for i in range(n_windows)
    ])
    db = 20 * np.log10(np.maximum(rms, 1e-6))

    silent = db < threshold_db
    silences = []
    start_idx = None
    for i, is_silent in enumerate(silent):
        if is_silent and start_idx is None:
            start_idx = i
        elif not is_silent and start_idx is not None:
            t0, t1 = start_idx * 0.05, i * 0.05
            if t1 - t0 >= min_duration:
                silences.append({"startTime": round(t0, 2), "endTime": round(t1, 2)})
            start_idx = None
    if start_idx is not None:
        t0, t1 = start_idx * 0.05, n_windows * 0.05
        if t1 - t0 >= min_duration:
            silences.append({"startTime": round(t0, 2), "endTime": round(t1, 2)})

    clip.close()
    return silences


# ---------------------------------------------------------------------------
# Audio mixing
# ---------------------------------------------------------------------------

def mix_audio(video_clip, project: Project):
    if not project.audio_tracks:
        return video_clip

    audio_layers = []
    original_audio = video_clip.audio

    for track in project.audio_tracks:
        if not os.path.exists(track.path):
            continue
        music = AudioFileClip(track.path).volumex(track.volume)
        if music.duration < video_clip.duration:
            music = afx.audio_loop(music, duration=video_clip.duration)
        else:
            music = music.subclip(0, video_clip.duration)
        music = music.set_start(track.start_offset)
        if track.duck_under_original and original_audio is not None:
            music = music.volumex(track.duck_level)
        audio_layers.append(music)

    if original_audio is not None:
        audio_layers = [original_audio] + audio_layers

    if not audio_layers:
        return video_clip

    final_audio = CompositeAudioClip(audio_layers) if len(audio_layers) > 1 else audio_layers[0]
    return video_clip.set_audio(final_audio)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_video(video_clip, output_path: str, preset_name: str = "youtube_1080p"):
    preset = EXPORT_PRESETS.get(preset_name, EXPORT_PRESETS["source"])
    clip = video_clip

    if preset["width"] and preset["height"]:
        target_w, target_h = preset["width"], preset["height"]
        src_w, src_h = clip.size
        target_ratio = target_w / target_h
        src_ratio = src_w / src_h

        if abs(target_ratio - src_ratio) > 0.02:
            if src_ratio > target_ratio:
                new_h = src_h
                new_w = int(src_h * target_ratio)
            else:
                new_w = src_w
                new_h = int(src_w / target_ratio)
            x0 = (src_w - new_w) // 2
            y0 = (src_h - new_h) // 2
            clip = clip.crop(x1=x0, y1=y0, x2=x0 + new_w, y2=y0 + new_h)

        clip = clip.resize((target_w, target_h))

    clip.write_videofile(
        output_path, fps=preset["fps"], codec="libx264", audio_codec="aac",
        bitrate=preset["bitrate"], preset="medium", threads=4, logger=None,
    )


def render_project(project: Project, output_path: str, config=DEFAULT_CONFIG):
    """Full pipeline: assemble every clip + transitions, mix audio, export
    to the project's chosen platform preset."""
    timeline = assemble_timeline(project, config)
    timeline = mix_audio(timeline, project)
    export_video(timeline, output_path, project.export_preset)
    return output_path