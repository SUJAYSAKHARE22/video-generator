import cv2
import numpy as np


def _weighted_centroid(diff_mask):
    m = cv2.moments(diff_mask, binaryImage=False)
    if m["m00"] <= 1e-6:
        return None
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    return cx, cy


def _sample_activity(video_path, sample_fps=4, analysis_w=320):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    duration = total_frames / fps if fps else 0

    step = max(1, int(round(fps / sample_fps)))

    samples = []  # (t, magnitude, cx_pct, cy_pct)
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
            samples.append((t, magnitude, cx_pct, cy_pct))

        prev_gray = gray
        frame_idx += 1

    cap.release()
    return samples, duration


def _select_peaks(samples, max_points, min_gap):
    if not samples:
        return []

    mags = np.array([s[1] for s in samples])
    if mags.max() <= 0:
        return []

    mean, std = mags.mean(), mags.std()
    threshold = mean + 0.6 * std
    candidates = [s for s in samples if s[1] > threshold and s[1] > 0]
    if not candidates:
        # fall back to just the single strongest moment of change
        candidates = [max(samples, key=lambda s: s[1])]

    candidates.sort(key=lambda s: s[1], reverse=True)

    selected = []
    for cand in candidates:
        if all(abs(cand[0] - sel[0]) >= min_gap for sel in selected):
            selected.append(cand)
        if len(selected) >= max_points:
            break

    selected.sort(key=lambda s: s[0])
    return selected


def detect_zoom_fragments(video_path, duration, max_points=4, sample_fps=4, min_gap=1.8,
                          hold_before=0.4, hold_after=1.8):
    """Analyzes real cursor/UI activity across the whole timeline (frame-difference
    tracking) and returns zoom fragments centered on where clicks/movement/UI
    changes actually happen - instead of guessing from a single static frame."""
    samples, detected_duration = _sample_activity(video_path, sample_fps=sample_fps)
    duration = duration or detected_duration
    peaks = _select_peaks(samples, max_points=max_points, min_gap=min_gap)

    fragments = []
    for t, magnitude, cx_pct, cy_pct in peaks:
        start = max(0.0, t - hold_before)
        end = min(duration, t + hold_after)
        if end - start < 1.0:
            continue

        end_sample_time = end - 0.25
        end_cx, end_cy = cx_pct, cy_pct
        closest = None
        for s in samples:
            if s[0] < t:
                continue
            if closest is None or abs(s[0] - end_sample_time) < abs(closest[0] - end_sample_time):
                closest = s
        if closest is not None:
            end_cx, end_cy = closest[2], closest[3]

        moved = ((end_cx - cx_pct) ** 2 + (end_cy - cy_pct) ** 2) ** 0.5

        fragments.append({
            "startTime": round(start, 2),
            "endTime": round(end, 2),
            "zoomLevel": 4,
            "speed": 5,
            "focusX": round(cx_pct, 1),
            "focusY": round(cy_pct, 1),
            "movementEnabled": moved > 6,
            "movementEndX": round(end_cx, 1),
            "movementEndY": round(end_cy, 1),
        })

    clean = []
    last_end = 0.0
    for f in fragments:
        if f["startTime"] < last_end:
            f["startTime"] = last_end + 0.05
        if f["endTime"] <= f["startTime"] + 0.5:
            continue
        clean.append(f)
        last_end = f["endTime"]

    return clean
