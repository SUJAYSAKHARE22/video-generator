"""
motion_planner.py
------------------
Top-level orchestrator for camera motion planning. Ties together:

  cursor_tracker   -> what happened on screen, when (signals)
  scene_analyzer   -> what's semantically important right now (modals, text)
  frame_sampler    -> the frame sequence handed to the AI
  ai_director      -> the actual cinematic decision-making ("the director")

...and applies stability/smoothing guards so the final per-video segment
timeline is production-safe even if the AI is unavailable, rate-limited,
or returns something odd for a given window.

This module owns NO camera-behaviour heuristics of its own beyond a
conservative, cursor/scene-driven fallback used only when the AI call for
a given window fails - the model is meant to be the actual director.
"""

import cv2
from typing import List, Optional

from camera_config import DEFAULT_CONFIG
from cursor_tracker import ActivityTimeline, sample_activity, summarize_for_prompt
from scene_analyzer import analyze_frame
from frame_sampler import sample_frame_sequence, FrameBatch
from ai_director import request_segment_plan, sanitize_segments


def _grab_frame_at(video_path, t, cap_cache={}):
    cap = cap_cache.get(video_path)
    if cap is None:
        cap = cv2.VideoCapture(video_path)
        cap_cache[video_path] = cap
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(t * fps)))
    ok, frame = cap.read()
    return frame if ok else None


def _heuristic_segments_for_window(batch: FrameBatch, timeline: ActivityTimeline,
                                    video_path: str, config) -> List[dict]:
    """Conservative, signal-driven fallback plan for a window where the AI
    call could not be completed. Uses real cursor events and one scene scan
    - never a fixed/static heuristic point - so behaviour still varies with
    what's actually happening on screen."""
    window_events = [e for e in timeline.events if batch.start_t <= e.t <= batch.end_t]
    segments = []

    mid_t = (batch.start_t + batch.end_t) / 2
    frame = _grab_frame_at(video_path, mid_t)
    modal_region = None
    if frame is not None:
        regions = analyze_frame(frame)
        if regions and regions[0].kind == "modal":
            modal_region = regions[0]

    if modal_region is not None:
        segments.append({
            "startTime": round(batch.start_t, 3), "endTime": round(batch.end_t, 3),
            "action": "zoom_in", "focusX": round(modal_region.x_pct + modal_region.w_pct / 2, 1),
            "focusY": round(modal_region.y_pct + modal_region.h_pct / 2, 1),
            "movementEnabled": False, "movementEndX": 50, "movementEndY": 50,
            "zoomLevel": 3.0, "panDirection": "none", "easing": config.default_easing,
            "transitionType": "smooth", "importance": 0.75, "confidence": 0.4,
            "reasoning": "fallback: centered modal detected", "source": "heuristic",
        })
        return segments

    notable = [e for e in window_events if e.type in ("click", "hover_start", "rapid_move")]
    if not notable:
        segments.append({
            "startTime": round(batch.start_t, 3), "endTime": round(batch.end_t, 3),
            "action": "static", "focusX": 50, "focusY": 50, "movementEnabled": False,
            "movementEndX": 50, "movementEndY": 50, "zoomLevel": 1.0, "panDirection": "none",
            "easing": config.default_easing, "transitionType": "hold", "importance": 0.1,
            "confidence": 0.5, "reasoning": "fallback: no notable activity", "source": "heuristic",
        })
        return segments

    cursor = window_events and next((e for e in window_events if e.type == "click"), notable[0])
    seg_end = max(cursor.t + config.hover_importance_seconds, batch.start_t + config.min_segment_duration)
    seg_end = min(seg_end, batch.end_t)
    zoom = 4.0 if cursor.type == "click" else 2.8
    segments.append({
        "startTime": round(max(batch.start_t, cursor.t - 0.3), 3), "endTime": round(seg_end, 3),
        "action": "zoom_in", "focusX": round(cursor.x_pct, 1), "focusY": round(cursor.y_pct, 1),
        "movementEnabled": False, "movementEndX": round(cursor.x_pct, 1), "movementEndY": round(cursor.y_pct, 1),
        "zoomLevel": zoom, "panDirection": "none", "easing": config.default_easing,
        "transitionType": "smooth", "importance": 0.6, "confidence": 0.4,
        "reasoning": f"fallback: {cursor.type} event", "source": "heuristic",
    })
    if seg_end < batch.end_t - config.min_segment_duration:
        segments.append({
            "startTime": round(seg_end, 3), "endTime": round(batch.end_t, 3),
            "action": "hold", "focusX": round(cursor.x_pct, 1), "focusY": round(cursor.y_pct, 1),
            "movementEnabled": False, "movementEndX": round(cursor.x_pct, 1), "movementEndY": round(cursor.y_pct, 1),
            "zoomLevel": zoom, "panDirection": "none", "easing": config.default_easing,
            "transitionType": "hold", "importance": 0.4, "confidence": 0.4,
            "reasoning": "fallback: hold after event", "source": "heuristic",
        })
    return segments


def _apply_stability_guards(segments: List[dict], config) -> List[dict]:
    """Removes overlaps, merges/absorbs segments shorter than the minimum
    duration, and damps zoom-level jumps that would otherwise cause jarring
    cuts between adjacent AI-authored windows."""
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
                # absorb into previous segment rather than keep a sliver
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
    the whole video: samples cursor/UI activity across the entire
    timeline, batches frames for the AI director, requests a plan per
    batch (with signal context + hand-off of previous camera state), falls
    back to signal-driven heuristics per-window on AI failure, then runs
    global stability guards across the whole timeline."""
    timeline = sample_activity(video_path, config)
    batches = sample_frame_sequence(video_path, duration, config)

    all_segments: List[dict] = []
    prev_state = None

    for batch in batches:
        signal_summary = summarize_for_prompt(timeline, batch.start_t, batch.end_t)
        raw = None
        if api_key:
            raw = request_segment_plan(batch, signal_summary, prev_state, api_key, config)

        if raw is not None:
            segs = sanitize_segments(raw, batch.start_t, batch.end_t, config)
        else:
            segs = []

        if not segs:
            segs = _heuristic_segments_for_window(batch, timeline, video_path, config)

        all_segments.extend(segs)
        if segs:
            last = segs[-1]
            prev_state = {
                "focusX": last["focusX"], "focusY": last["focusY"],
                "zoomLevel": last["zoomLevel"], "action": last["action"],
            }

    stable_segments = _apply_stability_guards(all_segments, config)

    return {
        "segments": stable_segments,
        "meta": {
            "totalEvents": len(timeline.events),
            "batches": len(batches),
            "aiUsed": bool(api_key),
        },
    }
