"""
cursor_tracker.py
-----------------
Dense, whole-timeline extraction of cursor / activity signals from a screen
recording using classical computer vision (frame differencing + optical
flow). This produces the structured, per-timestamp signal stream that the
motion planner and the AI director both consume - it is the "eyes" of the
system, not the "brain".

No single fixed heuristic decides camera behaviour here; this module only
reports what happened on screen (cursor position, velocity, clicks, hovers,
scrolling) so downstream planning logic can reason over it.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from camera_config import DEFAULT_CONFIG


@dataclass
class ActivitySample:
    t: float
    magnitude: float          # amount of pixel change vs previous sampled frame
    cx_pct: float             # weighted centroid of change, 0-100
    cy_pct: float
    flow_dx: float = 0.0      # dominant optical-flow displacement (px, analysis-res)
    flow_dy: float = 0.0
    scroll_score: float = 0.0  # vertical-band correlated shift => likely scroll


@dataclass
class CursorEvent:
    t: float
    type: str                 # "click" | "hover_start" | "hover_end" | "scroll" | "rapid_move" | "pause"
    x_pct: float
    y_pct: float
    meta: dict = field(default_factory=dict)


@dataclass
class ActivityTimeline:
    samples: List[ActivitySample]
    events: List[CursorEvent]
    duration: float
    fps: float


def _weighted_centroid(mask):
    m = cv2.moments(mask, binaryImage=False)
    if m["m00"] <= 1e-6:
        return None
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


def _estimate_scroll(prev_gray, gray):
    """Cheap scroll detector: correlate row-wise intensity profiles at a
    handful of vertical shifts; a strong best-shift match implies most of
    the frame translated vertically (typical of a scroll)."""
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
    """Walks the entire video once at `frame_sampling_fps` and produces a
    dense per-sample signal stream (change magnitude, centroid, optical
    flow, scroll likelihood)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    duration = total_frames / fps if fps else 0

    step = max(1, int(round(fps / config.frame_sampling_fps)))
    analysis_w = config.analysis_width

    samples: List[ActivitySample] = []
    prev_gray = None
    frame_idx = 0

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
                cx_pct, cy_pct = 50.0, 50.0

            flow_dx = flow_dy = 0.0
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 2, 12, 2, 5, 1.2, 0
                )
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                strong = mag > (mag.mean() + mag.std())
                if np.any(strong):
                    flow_dx = float(np.median(flow[..., 0][strong]))
                    flow_dy = float(np.median(flow[..., 1][strong]))
            except cv2.error:
                pass

            scroll_score = _estimate_scroll(prev_gray, gray) if magnitude > 0 else 0.0

            samples.append(ActivitySample(
                t=t, magnitude=magnitude, cx_pct=cx_pct, cy_pct=cy_pct,
                flow_dx=flow_dx, flow_dy=flow_dy, scroll_score=scroll_score,
            ))

        prev_gray = gray
        frame_idx += 1

    cap.release()

    events = _derive_events(samples, config)
    return ActivityTimeline(samples=samples, events=events, duration=duration or 0.0, fps=fps)


def _derive_events(samples: List[ActivitySample], config) -> List[CursorEvent]:
    """Turns the raw sample stream into semantic events: clicks (sharp
    magnitude spikes), hovers (sustained low motion in one place), scrolls
    (high scroll_score), and rapid moves (large flow magnitude)."""
    events: List[CursorEvent] = []
    if not samples:
        return events

    mags = np.array([s.magnitude for s in samples])
    if mags.max() <= 0:
        return events
    mean, std = mags.mean(), mags.std()
    click_threshold = mean + 1.1 * std
    rapid_threshold = mean + 0.7 * std

    hover_start_idx = None
    last_event_t = -1e9

    for i, s in enumerate(samples):
        is_scroll = s.scroll_score > 0.6
        is_spike = s.magnitude > click_threshold
        is_rapid = s.magnitude > rapid_threshold and (abs(s.flow_dx) + abs(s.flow_dy)) > 3

        if is_scroll:
            if s.t - last_event_t > 0.3:
                events.append(CursorEvent(s.t, "scroll", s.cx_pct, s.cy_pct,
                                           {"strength": round(s.scroll_score, 2)}))
                last_event_t = s.t
            hover_start_idx = None
            continue

        if is_spike and s.t - last_event_t > 0.25:
            events.append(CursorEvent(s.t, "click", s.cx_pct, s.cy_pct,
                                       {"magnitude": round(s.magnitude, 1)}))
            last_event_t = s.t
            hover_start_idx = None
            continue

        if is_rapid and s.t - last_event_t > 0.25:
            events.append(CursorEvent(s.t, "rapid_move", s.cx_pct, s.cy_pct,
                                       {"flow_dx": round(s.flow_dx, 2), "flow_dy": round(s.flow_dy, 2)}))
            last_event_t = s.t
            hover_start_idx = None
            continue

        # low motion => possible hover/pause in roughly the same spot
        if s.magnitude < mean * 0.35:
            if hover_start_idx is None:
                hover_start_idx = i
            else:
                start_sample = samples[hover_start_idx]
                dwell = s.t - start_sample.t
                if dwell >= config.hover_importance_seconds:
                    events.append(CursorEvent(start_sample.t, "hover_start",
                                               start_sample.cx_pct, start_sample.cy_pct,
                                               {"dwell_so_far": round(dwell, 2)}))
                    hover_start_idx = None
        else:
            hover_start_idx = None

    return events


def summarize_for_prompt(timeline: ActivityTimeline, window_start: float, window_end: float) -> str:
    """Compacts events + coarse motion stats within [window_start, window_end]
    into a short natural-language block to give the vision model temporal
    context alongside the sampled frames, without shipping the entire raw
    signal stream as tokens."""
    lines = []
    for e in timeline.events:
        if window_start <= e.t <= window_end:
            desc = {
                "click": f"t={e.t:.2f}s click near ({e.x_pct:.0f}%,{e.y_pct:.0f}%)",
                "hover_start": f"t={e.t:.2f}s cursor pauses/hovers near ({e.x_pct:.0f}%,{e.y_pct:.0f}%)",
                "scroll": f"t={e.t:.2f}s scrolling (strength {e.meta.get('strength')})",
                "rapid_move": f"t={e.t:.2f}s rapid cursor movement toward ({e.x_pct:.0f}%,{e.y_pct:.0f}%)",
            }.get(e.type, f"t={e.t:.2f}s {e.type}")
            lines.append(desc)
    if not lines:
        return "No significant cursor/UI events detected in this window; screen largely static."
    return "; ".join(lines)
