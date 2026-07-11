"""
frame_sampler.py
----------------
Extracts and encodes a temporally-ordered sequence of frames from the
source video for the AI director to reason over, batched to respect
vision-model payload limits. This is what turns "isolated screenshots"
into "a movie" for the model: frames are sampled at `planning_fps`,
grouped into batches of at most `ai_context_window` frames, and each
batch is sent as one ordered sequence.
"""

import base64
from dataclasses import dataclass
from typing import List

import cv2

from camera_config import DEFAULT_CONFIG


@dataclass
class SampledFrame:
    t: float
    b64_jpeg: str


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


def sample_frame_sequence(video_path: str, duration: float, config=DEFAULT_CONFIG) -> List[FrameBatch]:
    """Samples the whole video at `planning_fps`, encodes each sampled
    frame, and groups the resulting sequence into batches of at most
    `ai_context_window` frames so the AI director can process the video as
    a chronological sequence rather than disconnected screenshots."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    vid_duration = duration or (total_frames / fps if fps else 0)

    step_t = 1.0 / max(config.planning_fps, 0.1)
    sample_times = []
    t = 0.0
    while t < vid_duration:
        sample_times.append(t)
        t += step_t
    if not sample_times:
        sample_times = [0.0]

    sampled: List[SampledFrame] = []
    for t in sample_times:
        frame_idx = min(int(t * fps), max(total_frames - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        b64 = _encode_frame(frame, config.vision_frame_max_side, config.vision_jpeg_quality)
        if b64:
            sampled.append(SampledFrame(t=t, b64_jpeg=b64))

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
