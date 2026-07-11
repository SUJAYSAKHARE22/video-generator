"""
Central configuration for the AI-driven cinematic camera pipeline.

All tunable knobs for cursor tracking, scene understanding, AI planning,
motion smoothing and rendering live here so behaviour can be changed
without touching pipeline logic.
"""

import os
from dataclasses import dataclass, field


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class CameraConfig:
    # ---- Zoom bounds -------------------------------------------------
    min_zoom: float = _env_float("CAM_MIN_ZOOM", 1.0)
    max_zoom: float = _env_float("CAM_MAX_ZOOM", 4.5)

    # ---- Motion / pan behaviour ---------------------------------------
    pan_speed: float = _env_float("CAM_PAN_SPEED", 1.0)          # multiplier on movement duration
    tracking_sensitivity: float = _env_float("CAM_TRACK_SENS", 0.6)  # 0..1, how eagerly camera follows cursor
    cursor_influence: float = _env_float("CAM_CURSOR_INFLUENCE", 0.55)  # weight of cursor vs scene signal
    scene_influence: float = _env_float("CAM_SCENE_INFLUENCE", 0.45)
    motion_smoothing: float = _env_float("CAM_SMOOTHING", 0.35)   # 0..1, higher = smoother/slower response
    camera_inertia: float = _env_float("CAM_INERTIA", 0.25)       # resistance to sudden re-targeting

    # ---- Stability guards ----------------------------------------------
    min_segment_duration: float = _env_float("CAM_MIN_SEG_DUR", 0.6)
    min_gap_between_events: float = _env_float("CAM_MIN_EVENT_GAP", 0.9)
    max_zoom_change_per_segment: float = _env_float("CAM_MAX_ZOOM_DELTA", 2.5)
    focus_jitter_threshold_pct: float = _env_float("CAM_JITTER_PX_PCT", 1.5)

    # ---- Sampling / performance -----------------------------------------
    frame_sampling_fps: float = _env_float("CAM_SAMPLE_FPS", 6.0)     # activity/cursor analysis rate
    planning_fps: float = _env_float("CAM_PLANNING_FPS", 2.0)          # frames actually sent to the vision model
    ai_context_window: int = _env_int("CAM_AI_CONTEXT_WINDOW", 8)      # max frames per AI planning batch
    analysis_width: int = _env_int("CAM_ANALYSIS_W", 320)              # low-res width for CV/motion analysis
    vision_frame_max_side: int = _env_int("CAM_VISION_MAX_SIDE", 448)  # resized width/height sent to vision model
    vision_jpeg_quality: int = _env_int("CAM_VISION_JPEG_Q", 55)
    max_batches: int = _env_int("CAM_MAX_BATCHES", 6)                  # cap on number of vision-model calls / video

    # ---- Scene understanding thresholds ----------------------------------
    modal_area_ratio: float = _env_float("CAM_MODAL_AREA_RATIO", 0.18)  # fraction of frame area to call something a modal
    hover_importance_seconds: float = _env_float("CAM_HOVER_IMPORTANCE_S", 1.2)
    click_importance_ttl: float = _env_float("CAM_CLICK_TTL", 1.5)

    # ---- Output ---------------------------------------------------------
    default_easing: str = "easeInOutCubic"

    def clamp_zoom(self, value: float) -> float:
        return max(self.min_zoom, min(self.max_zoom, value))


DEFAULT_CONFIG = CameraConfig()
