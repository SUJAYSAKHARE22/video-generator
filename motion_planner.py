"""
motion_planner.py
------------------
Top-level orchestrator for camera motion planning. Ties together:

  cursor_tracker   -> real cursor kinematics: position, velocity, direction,
                      clicks, hovers, redirects, arrivals, scrolls
  scene_analyzer   -> what's semantically important right now (modals, text)
  frame_sampler    -> event-anchored frame batches (per-video, not fixed grid)
  ai_director       -> two-stage description + text-reasoning director

...and applies stability/smoothing guards so the final per-video segment
timeline is production-safe even if the AI is unavailable for a window.

The fallback heuristic used to be "grab the first click/hover in the
window, zoom to a fixed level" - which is why different videos produced
near-identical output whenever the AI call failed (which was most of the
time, due to the old multi-image request shape). This version builds ONE
distinct segment per notable kinematic event in the window, with the
camera action, zoom, pan direction and target genuinely derived from what
that specific event measured (a redirect pans toward the new direction; an
arrival anticipates ahead of the cursor; a hover eases into a slow
sustained zoom; a click punches in and holds; a scroll eases the zoom back
out) - so the fallback path itself is signal-driven, not canned.
"""

from typing import List, Optional

import cv2

from camera_config import DEFAULT_CONFIG
from cursor_tracker import (
    ActivityTimeline, sample_activity, summarize_for_prompt,
    dominant_motion_vector, pan_direction_from_vector, extrapolate_target,
)
from scene_analyzer import analyze_frame
from frame_sampler import sample_frame_sequence, FrameBatch
from ai_director import request_segment_plan, sanitize_segments

_FRAME_CACHE_CAPS = {}


def _grab_frame_at(video_path, t):
    cap = _FRAME_CACHE_CAPS.get(video_path)
    if cap is None:
        cap = cv2.VideoCapture(video_path)
        _FRAME_CACHE_CAPS[video_path] = cap
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(t * fps)))
    ok, frame = cap.read()
    return frame if ok else None


def _event_to_segment(event, next_event_t, batch, config, entry_zoom) -> Optional[dict]:
    """Maps ONE measured kinematic event to a camera segment whose action,
    target, zoom and pan are derived from that event's own measured data -
    not a fixed lookup."""
    start = max(batch.start_t, event.t - 0.15)
    end = min(batch.end_t, next_event_t if next_event_t else batch.end_t)
    if end - start < config.min_segment_duration:
        end = min(batch.end_t, start + config.min_segment_duration)
    if end <= start:
        return None

    base = {
        "startTime": round(start, 3), "endTime": round(end, 3),
        "movementEnabled": False, "movementEndX": round(event.x_pct, 1),
        "movementEndY": round(event.y_pct, 1), "panDirection": "none",
        "easing": config.default_easing, "transitionType": "smooth",
        "confidence": 0.45, "source": "heuristic",
    }

    if event.type == "click":
        base.update(action="zoom_in", focusX=round(event.x_pct, 1), focusY=round(event.y_pct, 1),
                    zoomLevel=min(config.max_zoom, entry_zoom + 2.5), importance=0.75,
                    reasoning=f"click at ({event.x_pct:.0f},{event.y_pct:.0f})")
        return base

    if event.type == "arrival":
        base.update(action="zoom_in", focusX=round(event.x_pct, 1), focusY=round(event.y_pct, 1),
                    zoomLevel=min(config.max_zoom, entry_zoom + 2.0), importance=0.7,
                    reasoning="cursor decelerating - anticipated interaction")
        return base

    if event.type == "redirect":
        tx, ty = extrapolate_target(event.x_pct, event.y_pct, event.dx_pct * event.speed,
                                     event.dy_pct * event.speed, seconds_ahead=0.5)
        pdir = pan_direction_from_vector(event.dx_pct, event.dy_pct)
        base.update(action="pan", focusX=round(event.x_pct, 1), focusY=round(event.y_pct, 1),
                    movementEnabled=True, movementEndX=round(tx, 1), movementEndY=round(ty, 1),
                    panDirection=pdir, zoomLevel=max(config.min_zoom + 0.5, entry_zoom * 0.9),
                    importance=0.65, reasoning=f"attention redirected, panning {pdir}")
        return base

    if event.type == "rapid_move":
        tx, ty = extrapolate_target(event.x_pct, event.y_pct, event.dx_pct, event.dy_pct, seconds_ahead=0.4)
        pdir = pan_direction_from_vector(event.dx_pct, event.dy_pct)
        base.update(action="track_cursor", focusX=round(event.x_pct, 1), focusY=round(event.y_pct, 1),
                    movementEnabled=True, movementEndX=round(tx, 1), movementEndY=round(ty, 1),
                    panDirection=pdir, zoomLevel=max(config.min_zoom + 0.3, entry_zoom * 0.85),
                    importance=0.55, reasoning="tracking fast cursor travel")
        return base

    if event.type == "hover_start":
        base.update(action="zoom_in", focusX=round(event.x_pct, 1), focusY=round(event.y_pct, 1),
                    zoomLevel=min(config.max_zoom, entry_zoom + 1.5), importance=0.6,
                    transitionType="hold", reasoning="sustained hover - slow deliberate zoom")
        return base

    if event.type == "scroll":
        base.update(action="zoom_out", focusX=round(event.x_pct, 1), focusY=round(event.y_pct, 1),
                    zoomLevel=max(config.min_zoom, entry_zoom * 0.7), importance=0.5,
                    reasoning="scrolling - easing zoom back out to follow new content")
        return base

    return None


def _heuristic_segments_for_window(batch: FrameBatch, timeline: ActivityTimeline,
                                    video_path: str, config, entry_zoom: float) -> List[dict]:
    """Signal-driven fallback for one window: a scene scan for an
    overriding modal, otherwise one segment per measured kinematic event in
    the window (chronological), otherwise a static hold if nothing at all
    happened - never a single fixed default point."""
    frame = _grab_frame_at(video_path, (batch.start_t + batch.end_t) / 2)
    if frame is not None:
        regions = analyze_frame(frame)
        if regions and regions[0].kind == "modal":
            m = regions[0]
            return [{
                "startTime": round(batch.start_t, 3), "endTime": round(batch.end_t, 3),
                "action": "zoom_in", "focusX": round(m.x_pct + m.w_pct / 2, 1),
                "focusY": round(m.y_pct + m.h_pct / 2, 1), "movementEnabled": False,
                "movementEndX": 50, "movementEndY": 50, "zoomLevel": min(config.max_zoom, 3.2),
                "panDirection": "none", "easing": config.default_easing, "transitionType": "smooth",
                "importance": 0.8, "confidence": 0.45, "reasoning": "modal/dialog detected",
                "source": "heuristic",
            }]

    events = sorted(timeline.events_in(batch.start_t, batch.end_t), key=lambda e: e.t)
    if not events:
        return [{
            "startTime": round(batch.start_t, 3), "endTime": round(batch.end_t, 3),
            "action": "static", "focusX": 50, "focusY": 50, "movementEnabled": False,
            "movementEndX": 50, "movementEndY": 50, "zoomLevel": max(config.min_zoom, entry_zoom * 0.6),
            "panDirection": "none", "easing": config.default_easing, "transitionType": "hold",
            "importance": 0.1, "confidence": 0.5, "reasoning": "no notable activity in window",
            "source": "heuristic",
        }]

    segments = []
    running_zoom = entry_zoom
    for i, event in enumerate(events):
        next_t = events[i + 1].t if i + 1 < len(events) else None
        seg = _event_to_segment(event, next_t, batch, config, running_zoom)
        if seg:
            segments.append(seg)
            running_zoom = seg["zoomLevel"]

    if not segments:
        segments.append({
            "startTime": round(batch.start_t, 3), "endTime": round(batch.end_t, 3),
            "action": "hold", "focusX": 50, "focusY": 50, "movementEnabled": False,
            "movementEndX": 50, "movementEndY": 50, "zoomLevel": entry_zoom,
            "panDirection": "none", "easing": config.default_easing, "transitionType": "hold",
            "importance": 0.2, "confidence": 0.4, "reasoning": "events present but unmapped",
            "source": "heuristic",
        })

    return segments


def _apply_stability_guards(segments: List[dict], config) -> List[dict]:
    """Removes overlaps, merges/absorbs segments shorter than the minimum
    duration, and damps zoom-level jumps between adjacent segments so the
    final timeline never jump-cuts, regardless of source (AI or heuristic)."""
    if not segments:
        return segments

    segments = sorted(segments, key=lambda s: s["startTime"])
    cleaned = []
    last_end = 0.0
    last_zoom = 1.0

    for seg in segments:
        if seg["startTime"] < last_end:
            seg["startTime"] = round(last_end, 3)
        if seg["endTime"] - seg["startTime"] < config.min_segment_duration:
            if cleaned and seg["importance"] <= cleaned[-1]["importance"]:
                cleaned[-1]["endTime"] = max(cleaned[-1]["endTime"], seg["endTime"])
                continue
            seg["endTime"] = seg["startTime"] + config.min_segment_duration

        zoom_delta = abs(seg["zoomLevel"] - last_zoom)
        if zoom_delta > config.max_zoom_change_per_segment:
            direction = 1 if seg["zoomLevel"] > last_zoom else -1
            seg["zoomLevel"] = last_zoom + direction * config.max_zoom_change_per_segment

        cleaned.append(seg)
        last_end = seg["endTime"]
        last_zoom = seg["zoomLevel"]

    return cleaned


def build_camera_plan(video_path: str, duration: float, api_key: Optional[str],
                       config=DEFAULT_CONFIG) -> dict:
    """Produces the final, stable, cinematic camera segment timeline for
    the whole video: extracts real cursor kinematics across the entire
    timeline, builds event-anchored frame batches (so batch shape reflects
    THIS video's actual activity), requests a two-stage AI plan per batch
    with hand-off of previous camera state, falls back to a kinematics-
    driven heuristic per-window on AI failure, then runs global stability
    guards across the whole timeline."""
    timeline = sample_activity(video_path, config)
    batches = sample_frame_sequence(video_path, duration, timeline, config)

    all_segments: List[dict] = []
    prev_state = None
    running_zoom = 1.0

    for batch in batches:
        signal_summary = summarize_for_prompt(timeline, batch.start_t, batch.end_t)
        raw = None
        if api_key:
            raw = request_segment_plan(batch, signal_summary, prev_state, api_key, config)

        segs = sanitize_segments(raw, batch.start_t, batch.end_t, config) if raw is not None else []

        if not segs:
            segs = _heuristic_segments_for_window(batch, timeline, video_path, config, running_zoom)

        all_segments.extend(segs)
        if segs:
            last = segs[-1]
            running_zoom = last["zoomLevel"]
            prev_state = {
                "focusX": last["focusX"], "focusY": last["focusY"],
                "zoomLevel": last["zoomLevel"], "action": last["action"],
            }

    stable_segments = _apply_stability_guards(all_segments, config)

    return {
        "segments": stable_segments,
        "meta": {
            "totalEvents": len(timeline.events),
            "eventTypes": sorted({e.type for e in timeline.events}),
            "batches": len(batches),
            "aiUsed": bool(api_key),
        },
    }
