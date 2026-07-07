import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from moviepy.editor import VideoClip, VideoFileClip

from ai_planner import PALETTES

TARGET_W, TARGET_H = 1920, 1080
MIN_ZOOM, MAX_ZOOM = 1.2, 4.0


def zoom_level_to_factor(level):
    normalized = (level - 1) / 9
    return MIN_ZOOM + (MAX_ZOOM - MIN_ZOOM) * normalized


def speed_to_transition_s(speed):
    min_ms, max_ms = 150, 2000
    normalized = (speed - 1) / 9
    return (max_ms - (max_ms - min_ms) * normalized) / 1000.0


def ease_out_quart(t):
    return 1 - (1 - t) ** 4


def ease_in_out_quart(t):
    if t < 0.5:
        return 8 * t ** 4
    return 1 - ((-2 * t + 2) ** 4) / 2


def compute_zoom_state(fragments, t):
    """Ported from openvid's calculateZoomPhaseState (2D subset)."""
    for frag in fragments:
        start, end = frag["startTime"], frag["endTime"]
        if not (start <= t <= end):
            continue

        total = end - start
        target_scale = zoom_level_to_factor(frag["zoomLevel"])
        transition = speed_to_transition_s(frag["speed"])
        entry_end = start + transition
        exit_start = end - transition
        hold_duration = max(0.0, exit_start - entry_end)

        focus_x, focus_y = frag["focusX"], frag["focusY"]
        move_end_x = frag.get("movementEndX", focus_x)
        move_end_y = frag.get("movementEndY", focus_y)

        if t < entry_end and transition > 0:
            progress = max(0.0, min(1.0, (t - start) / transition))
            eased = ease_out_quart(progress)
            scale = 1 + (target_scale - 1) * eased
        elif t >= exit_start and transition > 0:
            progress = max(0.0, min(1.0, (t - exit_start) / transition))
            eased = ease_out_quart(progress)
            scale = target_scale - (target_scale - 1) * eased
            if frag.get("movementEnabled"):
                focus_x, focus_y = move_end_x, move_end_y
        else:
            scale = target_scale
            if frag.get("movementEnabled") and hold_duration > 0:
                move_progress = max(0.0, min(1.0, (t - entry_end) / hold_duration))
                eased = ease_in_out_quart(move_progress)
                focus_x = frag["focusX"] + (move_end_x - frag["focusX"]) * eased
                focus_y = frag["focusY"] + (move_end_y - frag["focusY"]) * eased

        return scale, focus_x, focus_y

    return 1.0, 50.0, 50.0


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def gradient_array(w, h, stops, angle_deg=135):
    stops = sorted(stops, key=lambda s: s["position"])
    positions = np.array([s["position"] / 100.0 for s in stops])
    colors = np.array([hex_to_rgb(s["color"]) for s in stops], dtype=np.float32)

    theta = np.deg2rad(angle_deg)
    dx, dy = np.cos(theta), np.sin(theta)

    xs, ys = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    proj = xs * dx + ys * dy
    proj -= proj.min()
    proj /= (proj.max() + 1e-9)

    out = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        out[:, :, c] = np.interp(proj, positions, colors[:, c])
    return out.astype(np.uint8)


def rounded_mask(w, h, radius, corners=(True, True, True, True)):
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    tl, tr, bl, br = corners
    draw.rectangle([radius if tl else 0, 0, w - (radius if tr else 0), h], fill=255)
    draw.rectangle([0, radius if tl else 0, w, h - (radius if bl else 0)], fill=255)
    if tl:
        draw.pieslice([0, 0, radius * 2, radius * 2], 180, 270, fill=255)
    if tr:
        draw.pieslice([w - radius * 2, 0, w, radius * 2], 270, 360, fill=255)
    if bl:
        draw.pieslice([0, h - radius * 2, radius * 2, h], 90, 180, fill=255)
    if br:
        draw.pieslice([w - radius * 2, h - radius * 2, w, h], 0, 90, fill=255)
    return mask


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


class Shell:
    """Pre-rendered static background + device-mockup frame. Screen content
    and overlays are composited on top of this per-frame."""

    def __init__(self, plan, video_w, video_h):
        self.plan = plan
        mockup = plan["mockup"]
        padding_pct = mockup["padding"] / 100.0

        bg = plan["background"]
        if bg["type"] == "solid":
            self.background = np.full((TARGET_H, TARGET_W, 3), hex_to_rgb(bg["color"]), dtype=np.uint8)
        else:
            stops = PALETTES[bg["paletteIndex"] % len(PALETTES)]
            self.background = gradient_array(TARGET_W, TARGET_H, stops)

        chrome_h = 44 if mockup["style"] == "browser" else 0
        max_w = int(TARGET_W * (1 - padding_pct * 2))
        max_h = int(TARGET_H * (1 - padding_pct * 2)) - chrome_h
        scale = min(max_w / video_w, max_h / video_h)
        content_w, content_h = int(video_w * scale), int(video_h * scale)

        frame_w, frame_h = content_w, content_h + chrome_h
        frame_x = (TARGET_W - frame_w) // 2
        frame_y = (TARGET_H - frame_h) // 2

        canvas = Image.fromarray(self.background).convert("RGB")

        radius = mockup["cornerRadius"]
        shadow = Image.new("RGBA", (TARGET_W, TARGET_H), (0, 0, 0, 0))
        shadow_mask = rounded_mask(frame_w, frame_h, radius)
        shadow_layer = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 160))
        shadow_layer.putalpha(shadow_mask)
        shadow.paste(shadow_layer, (frame_x, frame_y + 14), shadow_layer)
        shadow = shadow.filter(ImageFilter.GaussianBlur(18))
        canvas.paste(shadow, (0, 0), shadow)

        frame_color = hex_to_rgb(mockup["frameColor"])
        frame_img = Image.new("RGBA", (frame_w, frame_h), frame_color + (255,))
        frame_mask = rounded_mask(frame_w, frame_h, radius)

        if chrome_h:
            draw = ImageDraw.Draw(frame_img)
            dot_colors = [(255, 95, 86), (255, 189, 46), (39, 201, 63)]
            for i, dc in enumerate(dot_colors):
                cx = 18 + i * 20
                draw.ellipse([cx, chrome_h // 2 - 5, cx + 10, chrome_h // 2 + 5], fill=dc)

            pill_x0, pill_x1 = 90, frame_w - 20
            pill_y0, pill_y1 = 10, chrome_h - 10
            pill_color = (50, 50, 55) if mockup["darkMode"] else (235, 235, 238)
            draw.rounded_rectangle([pill_x0, pill_y0, pill_x1, pill_y1], radius=(pill_y1 - pill_y0) // 2, fill=pill_color)
            text_color = (210, 210, 215) if mockup["darkMode"] else (90, 90, 95)
            font = _font(15)
            draw.text((pill_x0 + 14, (pill_y0 + pill_y1) // 2 - 8), mockup["url"], fill=text_color, font=font)

        canvas.paste(frame_img, (frame_x, frame_y), Image.composite(frame_img, Image.new("RGBA", frame_img.size, (0, 0, 0, 0)), frame_mask))

        self.canvas_base = np.array(canvas.convert("RGB"))
        self.content_rect = (frame_x, frame_y + chrome_h, content_w, content_h)
        self.content_radius = max(0, radius - 4)
        self.content_corners = (False, False, True, True) if chrome_h else (True, True, True, True)


def _zoom_crop(frame_rgb, scale, focus_x_pct, focus_y_pct):
    h, w = frame_rgb.shape[:2]
    crop_w, crop_h = w / scale, h / scale
    cx, cy = w * focus_x_pct / 100.0, h * focus_y_pct / 100.0
    x0 = min(max(cx - crop_w / 2, 0), w - crop_w)
    y0 = min(max(cy - crop_h / 2, 0), h - crop_h)
    x1, y1 = x0 + crop_w, y0 + crop_h
    img = Image.fromarray(frame_rgb).crop((int(x0), int(y0), int(x1), int(y1)))
    return img


def render_video(input_path, output_path, plan, progress_callback=None):
    raw_clip = VideoFileClip(input_path)
    duration = raw_clip.duration
    video_w, video_h = raw_clip.size

    shell = Shell(plan, video_w, video_h)
    fragments = plan["zoomFragments"]
    caption = plan.get("caption")
    cx, cy, cw, ch = shell.content_rect
    content_mask = rounded_mask(cw, ch, shell.content_radius, shell.content_corners)
    font = _font(40)

    def make_frame(t):
        t = min(t, max(duration - 1e-3, 0))
        raw_frame = raw_clip.get_frame(t)
        scale, fx, fy = compute_zoom_state(fragments, t)

        cropped = _zoom_crop(raw_frame, scale, fx, fy).resize((cw, ch), Image.LANCZOS)

        canvas = Image.fromarray(shell.canvas_base.copy()).convert("RGB")
        canvas.paste(cropped, (cx, cy), content_mask)

        if caption and caption["startTime"] <= t <= caption["endTime"]:
            draw = ImageDraw.Draw(canvas, "RGBA")
            text = caption["text"]
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            pad = 20
            box_x0 = (TARGET_W - tw) // 2 - pad
            box_y0 = TARGET_H - 140
            box_x1 = (TARGET_W + tw) // 2 + pad
            box_y1 = box_y0 + th + pad * 2
            draw.rounded_rectangle([box_x0, box_y0, box_x1, box_y1], radius=16, fill=(15, 15, 20, 210))
            draw.text(((TARGET_W - tw) // 2, box_y0 + pad - bbox[1]), text, fill=(255, 255, 255, 255), font=font)

        return np.array(canvas)

    final_clip = VideoClip(make_frame, duration=duration)
    if raw_clip.audio is not None:
        final_clip = final_clip.set_audio(raw_clip.audio)

    final_clip.write_videofile(
        output_path,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        threads=4,
        logger=None,
    )

    raw_clip.close()
    final_clip.close()
