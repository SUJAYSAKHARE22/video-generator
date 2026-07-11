"""
camera_engine.py
-----------------
Executes a rich, multi-segment camera motion plan produced by the motion
planner. Responsible ONLY for turning a timeline of segments into a
(scale, focus_x, focus_y) camera state at any timestamp `t` - smooth,
cinematic, and stable. Contains no planning/decision logic itself.
"""

import math

from camera_config import DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Easing curve library
# ---------------------------------------------------------------------------

def linear(t):
    return t


def ease_out_quart(t):
    return 1 - (1 - t) ** 4


def ease_in_quart(t):
    return t ** 4


def ease_in_out_quart(t):
    if t < 0.5:
        return 8 * t ** 4
    return 1 - ((-2 * t + 2) ** 4) / 2


def ease_in_out_cubic(t):
    if t < 0.5:
        return 4 * t ** 3
    return 1 - ((-2 * t + 2) ** 3) / 2


def ease_out_cubic(t):
    return 1 - (1 - t) ** 3


def ease_in_out_sine(t):
    return -(math.cos(math.pi * t) - 1) / 2


EASING_FUNCS = {
    "linear": linear,
    "easeOutQuart": ease_out_quart,
    "easeInQuart": ease_in_quart,
    "easeInOutQuart": ease_in_out_quart,
    "easeInOutCubic": ease_in_out_cubic,
    "easeOutCubic": ease_out_cubic,
    "easeInOutSine": ease_in_out_sine,
}


def get_easing(name):
    return EASING_FUNCS.get(name, ease_in_out_cubic)


def zoom_level_to_factor(level, config=DEFAULT_CONFIG):
    normalized = (max(1.0, min(10.0, level)) - 1) / 9
    return config.min_zoom + (config.max_zoom - config.min_zoom) * normalized


def transition_seconds_for_segment(segment, config=DEFAULT_CONFIG):
    """Transition (ease-in/ease-out) duration for a segment. Prefers an
    explicit `transitionSeconds`; otherwise derives one from segment length
    and pan_speed so short segments don't get transitions longer than the
    segment itself."""
    seg_len = max(0.05, segment["endTime"] - segment["startTime"])
    explicit = segment.get("transitionSeconds")
    if explicit is not None:
        return max(0.05, min(explicit, seg_len / 2))
    base = 0.9 / max(config.pan_speed, 0.1)
    return max(0.12, min(base, seg_len / 2.2))


def _segment_state(segment, t, config=DEFAULT_CONFIG, entry_scale=1.0, entry_focus=(50.0, 50.0)):
    """Camera state produced purely by one segment, ignoring neighbours.
    `entry_scale`/`entry_focus` let the caller hand off smoothly from the
    previous segment's exit state (camera inertia) instead of always
    starting a fresh ease from scale 1.0."""
    start, end = segment["startTime"], segment["endTime"]
    action = segment.get("action", "hold")
    target_scale = config.clamp_zoom(zoom_level_to_factor(segment.get("zoomLevel", 1), config))
    if action == "zoom_out":
        target_scale = min(target_scale, entry_scale)
    transition = transition_seconds_for_segment(segment, config)
    entry_end = start + transition
    exit_start = end - transition
    hold_duration = max(0.0, exit_start - entry_end)

    focus_x = segment.get("focusX", entry_focus[0])
    focus_y = segment.get("focusY", entry_focus[1])
    move_end_x = segment.get("movementEndX", focus_x)
    move_end_y = segment.get("movementEndY", focus_y)
    movement_enabled = segment.get("movementEnabled", False) and action in ("pan", "track_cursor")

    easing_in = get_easing(segment.get("easing", config.default_easing))
    easing_out = get_easing(segment.get("exitEasing", segment.get("easing", config.default_easing)))

    if action == "hold" or action == "static":
        return entry_scale if action == "static" and segment.get("preserveEntryScale") else target_scale, focus_x, focus_y

    if t < entry_end and transition > 0:
        progress = max(0.0, min(1.0, (t - start) / transition))
        eased = easing_in(progress)
        scale = entry_scale + (target_scale - entry_scale) * eased
        fx = entry_focus[0] + (focus_x - entry_focus[0]) * eased
        fy = entry_focus[1] + (focus_y - entry_focus[1]) * eased
        return scale, fx, fy

    if t >= exit_start and transition > 0 and action in ("zoom_in", "zoom_out"):
        progress = max(0.0, min(1.0, (t - exit_start) / transition))
        eased = easing_out(progress)
        scale = target_scale - (target_scale - 1.0) * eased if action == "zoom_in" else target_scale
        return scale, focus_x, focus_y

    scale = target_scale
    if movement_enabled and hold_duration > 0:
        move_progress = max(0.0, min(1.0, (t - entry_end) / hold_duration))
        eased = ease_in_out_cubic(move_progress)
        fx = focus_x + (move_end_x - focus_x) * eased
        fy = focus_y + (move_end_y - focus_y) * eased
        return scale, fx, fy

    return scale, focus_x, focus_y


def compute_camera_state(segments, t, config=DEFAULT_CONFIG):
    """Returns (scale, focus_x_pct, focus_y_pct) for time `t` given the full
    ordered list of camera segments, applying inertia so consecutive
    segments hand off smoothly instead of jump-cutting."""
    if not segments:
        return 1.0, 50.0, 50.0

    entry_scale, entry_focus = 1.0, (50.0, 50.0)
    active = None
    for seg in segments:
        if seg["startTime"] <= t <= seg["endTime"]:
            active = seg
            break
        if seg["endTime"] < t:
            # remember exit state to hand off inertia to the next segment
            exit_scale, exit_fx, exit_fy = _segment_state(
                seg, seg["endTime"], config, entry_scale, entry_focus
            )
            entry_scale, entry_focus = exit_scale, (exit_fx, exit_fy)

    if active is None:
        return entry_scale, entry_focus[0], entry_focus[1]

    inertia = config.camera_inertia
    scale, fx, fy = _segment_state(active, t, config, entry_scale, entry_focus)
    if inertia > 0:
        blend = min(1.0, (t - active["startTime"]) / max(0.05, transition_seconds_for_segment(active, config)))
        damp = inertia * (1 - blend)
        scale = scale * (1 - damp) + entry_scale * damp
        fx = fx * (1 - damp) + entry_focus[0] * damp
        fy = fy * (1 - damp) + entry_focus[1] * damp
    return scale, fx, fy
