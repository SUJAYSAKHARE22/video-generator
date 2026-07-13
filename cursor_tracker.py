"""
cursor_tracker.py
-----------------
Whole-timeline kinematic extraction of cursor/activity signals.

This is the single biggest lever on "every output looks the same": if the
signal stream only reports raw change-magnitude and a noisy centroid, every
video reduces to the same shape (a few blobs of motion) and the planner has
nothing video-specific to react to. This version tracks actual cursor
POSITION, VELOCITY, ACCELERATION and DIRECTION over time and turns that into
a small set of high-signal, semantically distinct events:

  click          - sharp localized spike, velocity collapses to ~0 right after
  hover_start    - sustained near-zero speed in one place for >= threshold
  redirect       - cursor's direction vector changes sharply (attention moved
                   to a new target) - this is what should retarget the camera
  arrival        - cursor decelerates hard while approaching a point (about to
                   click/interact) - lets the planner anticipate rather than lag
  rapid_move     - sustained high speed travel across the screen
  scroll         - vertical band correlation shift

Each event carries the actual measured position/velocity/direction, so two
different videos with different cursor behaviour produce genuinely different
event streams - not the same canned shape.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from camera_config import DEFAULT_CONFIG


@dataclass
class ActivitySample:
    t: float
    magnitude: float           # amount of pixel change vs previous sampled frame
    cx_pct: float               # weighted centroid of change, 0-100
    cy_pct: float
    vx: float = 0.0              # velocity, %/s
    vy: float = 0.0
    speed: float = 0.0           # %/s
    direction_deg: float = 0.0   # 0-360, atan2(vy, vx)
    accel: float = 0.0           # change in speed, %/s^2
    scroll_score: float = 0.0


@dataclass
class CursorEvent:
    t: float
    type: str
    x_pct: float
    y_pct: float
    dx_pct: float = 0.0          # predicted/observed direction, unit-ish vector
    dy_pct: float = 0.0
    speed: float = 0.0
    meta: dict = field(default_factory=dict)


@dataclass
class ActivityTimeline:
    samples: List[ActivitySample]
    events: List[CursorEvent]
    duration: float
    fps: float

    def events_in(self, start: float, end: float) -> List[CursorEvent]:
        return [e for e in self.events if start <= e.t <= end]

    def samples_in(self, start: float, end: float) -> List[ActivitySample]:
        return [s for s in self.samples if start <= s.t <= end]


def _weighted_centroid(mask):
    m = cv2.moments(mask, binaryImage=False)
    if m["m00"] <= 1e-6:
        return None
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


def _estimate_scroll(prev_gray, gray):
    h = gray.shape[0]
    band = gray[int(h * 0.1):int(h * 0.9)]
    prev_band = prev_gray[int(h * 0.1):int(h * 0.9)]
    row_now = band.mean(axis=1)
    row_prev = prev_band.mean(axis=1)

    best_shift, best_score = 0, 0.0
    for shift in range(-14, 15, 2):
        if shift == 0:
            continue
        if shift > 0:
            a, b = row_now[shift:], row_prev[:-shift]
        else:
            a, b = row_now[:shift], row_prev[-shift:]
        if len(a) < 10:
            continue
        a = a - a.mean()
        b = b - b.mean()
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6
        score = float(np.dot(a, b) / denom)
        if score > best_score:
            best_score, best_shift = score, shift
    return best_score if best_score > 0.55 else 0.0


def sample_activity(video_path: str, config=DEFAULT_CONFIG) -> ActivityTimeline:
    """Walks the entire video once at `frame_sampling_fps`, producing a
    dense per-sample kinematic stream: position, velocity, acceleration,
    direction - not just raw diff magnitude."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    duration = total_frames / fps if fps else 0

    step = max(1, int(round(fps / config.frame_sampling_fps)))
    analysis_w = config.analysis_width

    raw_points: List[Tuple[float, float, float, float]] = []  # t, magnitude, cx, cy
    prev_gray = None
    frame_idx = 0
    scroll_scores = {}

    while True:
        ok = cap.grab()
        if not ok:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        ok, frame = cap.retrieve()
        if not ok:
            break

        h, w = frame.shape[:2]
        scale = analysis_w / w
        small = cv2.resize(frame, (analysis_w, max(1, int(h * scale))))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        t = frame_idx / fps

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
            magnitude = float(np.sum(mask > 0))
            centroid = _weighted_centroid(mask.astype(np.float32))
            if centroid is not None:
                cx_pct = max(0.0, min(100.0, centroid[0] / mask.shape[1] * 100.0))
                cy_pct = max(0.0, min(100.0, centroid[1] / mask.shape[0] * 100.0))
            else:
                cx_pct, cy_pct = raw_points[-1][2] if raw_points else 50.0, raw_points[-1][3] if raw_points else 50.0

            scroll_scores[t] = _estimate_scroll(prev_gray, gray) if magnitude > 0 else 0.0
            raw_points.append((t, magnitude, cx_pct, cy_pct))

        prev_gray = gray
        frame_idx += 1

    cap.release()

    samples = _build_kinematics(raw_points, scroll_scores)
    events = _derive_events(samples, config)
    return ActivityTimeline(samples=samples, events=events, duration=duration or 0.0, fps=fps)


def _build_kinematics(raw_points, scroll_scores) -> List[ActivitySample]:
    """Turns raw (t, magnitude, cx, cy) points into a kinematic stream with
    smoothed velocity/acceleration/direction, so downstream logic reasons
    about actual cursor motion instead of frame-to-frame noise."""
    samples: List[ActivitySample] = []
    if not raw_points:
        return samples

    # light smoothing of the centroid path to suppress diff-noise jitter
    xs = np.array([p[2] for p in raw_points], dtype=np.float32)
    ys = np.array([p[3] for p in raw_points], dtype=np.float32)
    if len(xs) >= 5:
        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        xs = np.convolve(xs, kernel, mode="same")
        ys = np.convolve(ys, kernel, mode="same")

    prev_speed = 0.0
    for i, (t, magnitude, _, _) in enumerate(raw_points):
        cx_pct, cy_pct = float(xs[i]), float(ys[i])

        if i > 0:
            dt = max(1e-3, t - raw_points[i - 1][0])
            vx = (cx_pct - float(xs[i - 1])) / dt
            vy = (cy_pct - float(ys[i - 1])) / dt
        else:
            vx = vy = 0.0

        speed = math.hypot(vx, vy)
        direction_deg = (math.degrees(math.atan2(vy, vx)) + 360) % 360
        accel = (speed - prev_speed) / max(1e-3, (t - raw_points[i - 1][0]) if i > 0 else 1.0)
        prev_speed = speed

        samples.append(ActivitySample(
            t=t, magnitude=magnitude, cx_pct=cx_pct, cy_pct=cy_pct,
            vx=vx, vy=vy, speed=speed, direction_deg=direction_deg, accel=accel,
            scroll_score=scroll_scores.get(t, 0.0),
        ))

    return samples


def _angle_delta(a, b):
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


def _derive_events(samples: List[ActivitySample], config) -> List[CursorEvent]:
    events: List[CursorEvent] = []
    if not samples:
        return events

    mags = np.array([s.magnitude for s in samples])
    speeds = np.array([s.speed for s in samples])
    if mags.max() <= 0:
        return events

    mag_mean, mag_std = mags.mean(), mags.std()
    speed_mean, speed_std = speeds.mean(), speeds.std()

    click_threshold = mag_mean + 1.1 * mag_std
    rapid_speed_threshold = speed_mean + 1.0 * speed_std
    idle_speed_threshold = max(2.0, speed_mean * 0.25)

    last_event_t = -1e9
    hover_start_idx = None
    prev_direction = None
    direction_stable_since = 0

    for i, s in enumerate(samples):
        is_scroll = s.scroll_score > 0.6
        is_spike = s.magnitude > click_threshold
        is_rapid = s.speed > rapid_speed_threshold
        is_idle = s.speed < idle_speed_threshold

        # --- scroll -------------------------------------------------
        if is_scroll:
            if s.t - last_event_t > 0.3:
                events.append(CursorEvent(s.t, "scroll", s.cx_pct, s.cy_pct,
                                           speed=s.speed,
                                           meta={"strength": round(s.scroll_score, 2)}))
                last_event_t = s.t
            hover_start_idx = None
            prev_direction = None
            continue

        # --- click: a sharp spike immediately followed by near-zero speed
        if is_spike and s.t - last_event_t > 0.25:
            settles = i + 1 < len(samples) and samples[i + 1].speed < idle_speed_threshold
            events.append(CursorEvent(s.t, "click", s.cx_pct, s.cy_pct, speed=s.speed,
                                       meta={"magnitude": round(s.magnitude, 1), "settles": settles}))
            last_event_t = s.t
            hover_start_idx = None
            prev_direction = None
            continue

        # --- arrival: decelerating hard while previously moving fast
        if s.accel < 0 and abs(s.accel) > (speed_mean + speed_std) and s.speed < speed_mean and i > 2:
            recent_fast = any(samples[j].speed > rapid_speed_threshold for j in range(max(0, i - 4), i))
            if recent_fast and s.t - last_event_t > config.min_gap_between_events * 0.4:
                events.append(CursorEvent(s.t, "arrival", s.cx_pct, s.cy_pct, speed=s.speed,
                                           meta={"decel": round(s.accel, 1)}))
                last_event_t = s.t
                hover_start_idx = None
                prev_direction = None
                continue

        # --- redirect: direction vector changes sharply while moving
        if s.speed > idle_speed_threshold:
            if prev_direction is not None:
                delta = _angle_delta(s.direction_deg, prev_direction)
                if delta > 55 and s.t - last_event_t > 0.3:
                    events.append(CursorEvent(
                        s.t, "redirect", s.cx_pct, s.cy_pct,
                        dx_pct=math.cos(math.radians(s.direction_deg)),
                        dy_pct=math.sin(math.radians(s.direction_deg)),
                        speed=s.speed, meta={"angle_delta": round(delta, 1)},
                    ))
                    last_event_t = s.t
            prev_direction = s.direction_deg
            hover_start_idx = None
        else:
            prev_direction = None

        # --- rapid sustained move -------------------------------------
        if is_rapid and s.t - last_event_t > 0.4:
            events.append(CursorEvent(
                s.t, "rapid_move", s.cx_pct, s.cy_pct,
                dx_pct=math.cos(math.radians(s.direction_deg)) * s.speed,
                dy_pct=math.sin(math.radians(s.direction_deg)) * s.speed,
                speed=s.speed,
            ))
            last_event_t = s.t
            hover_start_idx = None
            continue

        # --- hover / pause ----------------------------------------------
        if is_idle:
            if hover_start_idx is None:
                hover_start_idx = i
            else:
                start_sample = samples[hover_start_idx]
                dwell = s.t - start_sample.t
                if dwell >= config.hover_importance_seconds and s.t - last_event_t > 0.5:
                    events.append(CursorEvent(start_sample.t, "hover_start",
                                               start_sample.cx_pct, start_sample.cy_pct,
                                               speed=0.0, meta={"dwell_so_far": round(dwell, 2)}))
                    last_event_t = s.t
                    hover_start_idx = None
        else:
            hover_start_idx = None

    return events


def dominant_motion_vector(timeline: ActivityTimeline, start: float, end: float) -> Optional[Tuple[float, float]]:
    """Average velocity vector over a window - used to decide pan direction
    when the camera should anticipate where the cursor is heading, rather
    than just centering on where it currently is."""
    window = timeline.samples_in(start, end)
    if not window:
        return None
    vx = float(np.mean([s.vx for s in window]))
    vy = float(np.mean([s.vy for s in window]))
    if math.hypot(vx, vy) < 0.5:
        return None
    return vx, vy


def extrapolate_target(x_pct, y_pct, vx, vy, seconds_ahead=0.6):
    """Anticipates where the cursor is heading, so the camera can lead
    rather than lag: predicted point = current position + velocity * time,
    clamped to frame bounds."""
    tx = max(0.0, min(100.0, x_pct + vx * seconds_ahead))
    ty = max(0.0, min(100.0, y_pct + vy * seconds_ahead))
    return tx, ty


def pan_direction_from_vector(vx, vy) -> str:
    if math.hypot(vx, vy) < 0.5:
        return "none"
    angle = (math.degrees(math.atan2(vy, vx)) + 360) % 360
    dirs = ["right", "down-right", "down", "down-left", "left", "up-left", "up", "up-right"]
    idx = int(((angle + 22.5) % 360) // 45)
    return dirs[idx]


def summarize_for_prompt(timeline: ActivityTimeline, window_start: float, window_end: float) -> str:
    """Compacts events + motion vector within [window_start, window_end]
    into a short natural-language block for the AI director's text-reasoning
    step, keyed on real measured kinematics so different videos produce
    genuinely different summaries."""
    lines = []
    for e in timeline.events_in(window_start, window_end):
        desc = {
            "click": f"t={e.t:.2f}s click at ({e.x_pct:.0f}%,{e.y_pct:.0f}%)",
            "hover_start": f"t={e.t:.2f}s cursor pauses/hovers at ({e.x_pct:.0f}%,{e.y_pct:.0f}%) for {e.meta.get('dwell_so_far')}s+",
            "scroll": f"t={e.t:.2f}s scrolling (strength {e.meta.get('strength')})",
            "rapid_move": f"t={e.t:.2f}s fast cursor travel through ({e.x_pct:.0f}%,{e.y_pct:.0f}%) at speed {e.speed:.0f}%/s",
            "redirect": f"t={e.t:.2f}s cursor sharply changes direction near ({e.x_pct:.0f}%,{e.y_pct:.0f}%), angle change {e.meta.get('angle_delta')}deg - attention retargeting",
            "arrival": f"t={e.t:.2f}s cursor decelerates/arrives near ({e.x_pct:.0f}%,{e.y_pct:.0f}%) - likely about to interact",
        }.get(e.type, f"t={e.t:.2f}s {e.type}")
        lines.append(desc)

    vec = dominant_motion_vector(timeline, window_start, window_end)
    if vec:
        pdir = pan_direction_from_vector(*vec)
        lines.append(f"Dominant motion direction across window: {pdir}")

    if not lines:
        return "No significant cursor/UI events detected in this window; screen largely static."
    return "; ".join(lines)
