"""
frame_sampler.py
----------------
Samples and encodes frames for the AI director - but, critically, the
sampling grid and batch boundaries are now anchored to what actually
happened in THIS video (its cursor events), not a blind fixed-fps grid.

A fixed grid is one of the reasons different videos used to produce
near-identical plans: every video got the same batch shape regardless of
content. Here, batches are built around event clusters (click, hover,
redirect, arrival, scroll, rapid_move) with a light base-rate grid filling
in the static stretches in between - so a video with three clicks and a
video with a long drag-and-drop sequence get genuinely different batch
timelines.
"""

import base64
from dataclasses import dataclass
from typing import List, Optional

import cv2

from camera_config import DEFAULT_CONFIG
from cursor_tracker import ActivityTimeline


@dataclass
class SampledFrame:
    t: float
    b64_jpeg: str
    is_event_anchor: bool = False
    near_static: bool = False


@dataclass
class FrameBatch:
    start_t: float
    end_t: float
    frames: List[SampledFrame]


def _encode_frame(frame_bgr, max_side, quality):
    h, w = frame_bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        frame_bgr = cv2.resize(frame_bgr, (max(1, int(w * scale)), max(1, int(h * scale))))
    q = quality
    for _ in range(5):
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok and len(buf) * 4 / 3 < 170_000:
            break
        q = max(20, q - 10)
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _event_anchored_times(timeline: ActivityTimeline, duration: float, config) -> List[float]:
    """Builds the set of timestamps worth sampling: every detected event
    (plus a small lead-in so the AI sees the moment just before the event,
    for anticipation) union a sparse base grid so silent/static stretches
    are still covered at a low rate."""
    times = set()

    for e in timeline.events:
        lead_in = max(0.0, e.t - 0.2)
        times.add(round(lead_in, 2))
        times.add(round(e.t, 2))

    base_step = 1.0 / max(config.planning_fps * 0.5, 0.1)  # sparser base grid; events fill in detail
    t = 0.0
    while t < duration:
        times.add(round(t, 2))
        t += base_step

    return sorted(times)


def _near_static(timeline: ActivityTimeline, t: float, radius=0.15) -> bool:
    window = [s for s in timeline.samples if abs(s.t - t) <= radius]
    if not window:
        return False
    avg_mag = sum(s.magnitude for s in window) / len(window)
    all_mags = [s.magnitude for s in timeline.samples] or [0.0]
    mean_mag = sum(all_mags) / len(all_mags)
    return avg_mag < mean_mag * 0.2


def sample_frame_sequence(video_path: str, duration: float, timeline: Optional[ActivityTimeline],
                           config=DEFAULT_CONFIG) -> List[FrameBatch]:
    """Samples frames at event-anchored timestamps (falls back to a plain
    grid if no timeline is available), encodes them, marks which ones sit
    on/near a real detected event vs a static filler frame, and groups the
    sequence into chronological batches for the AI director."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    vid_duration = duration or (total_frames / fps if fps else 0)

    if timeline is not None and timeline.events:
        sample_times = _event_anchored_times(timeline, vid_duration, config)
    else:
        step_t = 1.0 / max(config.planning_fps, 0.1)
        sample_times = []
        t = 0.0
        while t < vid_duration:
            sample_times.append(t)
            t += step_t
        if not sample_times:
            sample_times = [0.0]

    event_times = {round(e.t, 2) for e in (timeline.events if timeline else [])}

    sampled: List[SampledFrame] = []
    for t in sample_times:
        frame_idx = min(int(t * fps), max(total_frames - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        b64 = _encode_frame(frame, config.vision_frame_max_side, config.vision_jpeg_quality)
        if not b64:
            continue
        is_anchor = any(abs(t - et) < 0.05 for et in event_times)
        static = _near_static(timeline, t) if timeline is not None else False
        sampled.append(SampledFrame(t=t, b64_jpeg=b64, is_event_anchor=is_anchor, near_static=static))

    cap.release()

    window = config.ai_context_window
    batches: List[FrameBatch] = []
    for i in range(0, len(sampled), window):
        chunk = sampled[i:i + window]
        if not chunk:
            continue
        batches.append(FrameBatch(start_t=chunk[0].t, end_t=chunk[-1].t, frames=chunk))
        if len(batches) >= config.max_batches:
            break

    return batches
