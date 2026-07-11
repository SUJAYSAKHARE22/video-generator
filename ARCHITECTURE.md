# Cinematic Camera Planning System — Architecture

## New modules

| File | Responsibility |
|---|---|
| `camera_config.py` | All tunable knobs (zoom bounds, pan speed, tracking sensitivity, cursor/scene influence, smoothing, inertia, stability guards, sampling rates, AI context window). |
| `cursor_tracker.py` | Whole-timeline signal extraction: frame differencing + optical flow + scroll correlation → `ActivitySample` stream → semantic `CursorEvent`s (click, hover_start, scroll, rapid_move). |
| `scene_analyzer.py` | Single-frame CV scene understanding: modal/dialog detection (contour + centeredness + area heuristics), text/UI density map, optional OCR (pytesseract, degrades gracefully if unavailable). |
| `frame_sampler.py` | Samples the whole video at `planning_fps`, encodes frames, batches them (≤ `ai_context_window` per batch) into chronological `FrameBatch`es for the AI director. |
| `ai_director.py` | Builds the system/user prompt (rich segment schema), sends a batch of ordered frames + signal summary + previous camera state to the NVIDIA NIM vision model, with graceful degradation (full batch → half → 1 frame → text-only) if the endpoint rejects multi-image payloads. Validates/clamps the raw response (`sanitize_segments`). |
| `motion_planner.py` | Orchestrates tracker + analyzer + sampler + director per batch, falls back to a signal-driven (not fixed) heuristic per-window plan if the AI call fails, then applies global stability guards (de-overlap, min segment duration, max zoom-delta damping) across the whole timeline. |
| `camera_engine.py` | Pure interpolation: easing-curve library, per-segment state computation, camera inertia hand-off between segments. No planning logic. |
| `ai_planner.py` | Thin orchestrator: visual styling (background/mockup/caption, one vision call) + delegates camera timeline to `motion_planner.build_camera_plan`. Same `get_ai_plan()` interface as before. |
| `renderer.py` | Unchanged responsibilities, now drives `camera_engine.compute_camera_state(segments, t)` instead of a single fixed-point zoom heuristic. |

## Pipeline flow

```
video ──► cursor_tracker.sample_activity ──► ActivityTimeline (samples + events)
      ──► frame_sampler.sample_frame_sequence ──► FrameBatch[] (chronological, windowed)

for each FrameBatch:
   signal_summary = cursor_tracker.summarize_for_prompt(timeline, window)
   raw = ai_director.request_segment_plan(batch, signal_summary, prev_state, api_key)
   segments = ai_director.sanitize_segments(raw)  OR  motion_planner._heuristic_segments_for_window(...)
   prev_state = last segment's camera state   # hand-off context to next window

all_segments ──► motion_planner._apply_stability_guards ──► final segment timeline

ai_planner.get_ai_plan(): style (background/mockup/caption) + segments ──► plan dict
renderer.render_video(plan): camera_engine.compute_camera_state(segments, t) per frame ──► output mp4
```

## Segment schema

```json
{
  "startTime": 1.87, "endTime": 4.5,
  "action": "zoom_in|zoom_out|pan|track_cursor|hold|static",
  "focusX": 62.0, "focusY": 40.0,
  "movementEnabled": true, "movementEndX": 30.0, "movementEndY": 55.0,
  "zoomLevel": 3.5, "panDirection": "left",
  "easing": "easeInOutCubic", "transitionType": "smooth",
  "importance": 0.6, "confidence": 0.4,
  "reasoning": "cursor rapid-moves toward settings button",
  "source": "ai|heuristic"
}
```

## Stability guards
- Segments < `min_segment_duration` are absorbed into a higher-importance neighbor instead of kept as slivers.
- Overlaps are resolved by shifting the later segment's start forward.
- Zoom-level jumps beyond `max_zoom_change_per_segment` between adjacent segments are damped.
- `camera_engine` blends each segment's entry state with the previous segment's exit state (`camera_inertia`) so batch boundaries never jump-cut.

## Fallback behavior
- No API key / model call fails for a window → `motion_planner._heuristic_segments_for_window` uses real detected cursor events + one scene scan for that window (never a fixed static point).
- Camera planning fails entirely → single full-duration static segment (graceful degradation, never a crash).

## Configuration (env vars, see `camera_config.py`)
`CAM_MIN_ZOOM`, `CAM_MAX_ZOOM`, `CAM_PAN_SPEED`, `CAM_TRACK_SENS`, `CAM_CURSOR_INFLUENCE`, `CAM_SCENE_INFLUENCE`, `CAM_SMOOTHING`, `CAM_INERTIA`, `CAM_MIN_SEG_DUR`, `CAM_MIN_EVENT_GAP`, `CAM_MAX_ZOOM_DELTA`, `CAM_SAMPLE_FPS`, `CAM_PLANNING_FPS`, `CAM_AI_CONTEXT_WINDOW`, `CAM_ANALYSIS_W`, `CAM_VISION_MAX_SIDE`, `CAM_VISION_JPEG_Q`, `CAM_MAX_BATCHES`, `CAM_MODAL_AREA_RATIO`, `CAM_HOVER_IMPORTANCE_S`, `CAM_CLICK_TTL`.

## Files removed
- `activity_detector.py` (logic absorbed/expanded into `cursor_tracker.py`).

## Files added
`camera_config.py`, `cursor_tracker.py`, `scene_analyzer.py`, `frame_sampler.py`, `ai_director.py`, `motion_planner.py`, `camera_engine.py`.

## Future extension points
- Add per-segment `boundingBox` from `scene_analyzer` UI regions into the AI prompt/schema for tighter framing.
- Cache `scene_analyzer` results per keyframe instead of per-batch midpoint.
- Replace Farneback optical flow with a lightweight tracker (e.g. KLT) for lower CPU cost on long videos.
- Feed OCR'd text of the active region into the AI prompt when `pytesseract` is present for caption-worthy content detection.
