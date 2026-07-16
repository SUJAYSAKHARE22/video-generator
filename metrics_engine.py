"""
metrics_engine.py
-------------------
Computes objective, reproducible metrics from an actual camera plan and
run, rather than asserting quality claims without evidence. These are the
numbers `EVALUATION.md` reports on, and what a reviewer (or an IEEE
submission) would expect to see backing up any "cinematic"/"intelligent
camera" claim: stability, diversity/adaptiveness across videos, AI vs
heuristic reliance, and processing cost relative to video length.

Every function here takes real data produced by a real run (a `plan` dict
from `ai_planner.get_ai_plan`, or a list of plans from multiple videos) -
nothing in this module fabricates numbers.
"""

import math
import statistics
from collections import Counter
from typing import List, Dict, Any


def _segment_zoom_deltas(segments: List[dict]) -> List[float]:
    deltas = []
    for i in range(1, len(segments)):
        deltas.append(abs(segments[i]["zoomLevel"] - segments[i - 1]["zoomLevel"]))
    return deltas


def compute_camera_metrics(plan: dict) -> Dict[str, Any]:
    """Metrics for ONE video's camera plan."""
    segments = plan.get("segments", [])
    meta = plan.get("planningMeta", plan.get("meta", {}))

    if not segments:
        return {"num_segments": 0}

    durations = [s["endTime"] - s["startTime"] for s in segments]
    zoom_levels = [s["zoomLevel"] for s in segments]
    actions = Counter(s["action"] for s in segments)
    sources = Counter(s.get("source", "unknown") for s in segments)
    zoom_deltas = _segment_zoom_deltas(segments)

    total_ai = sources.get("ai", 0)
    total_seg = len(segments)

    action_probs = [c / total_seg for c in actions.values()]
    action_entropy = -sum(p * math.log2(p) for p in action_probs if p > 0)
    max_entropy = math.log2(len(actions)) if len(actions) > 1 else 1.0

    return {
        "num_segments": total_seg,
        "avg_segment_duration_s": round(statistics.mean(durations), 3),
        "min_segment_duration_s": round(min(durations), 3),
        "max_segment_duration_s": round(max(durations), 3),
        "action_distribution": dict(actions),
        "action_diversity_normalized_entropy": round(action_entropy / max_entropy, 3) if max_entropy else 0.0,
        "zoom_level_mean": round(statistics.mean(zoom_levels), 2),
        "zoom_level_stdev": round(statistics.pstdev(zoom_levels), 2) if len(zoom_levels) > 1 else 0.0,
        "avg_zoom_delta_between_segments": round(statistics.mean(zoom_deltas), 2) if zoom_deltas else 0.0,
        "max_zoom_delta_observed": round(max(zoom_deltas), 2) if zoom_deltas else 0.0,
        "ai_sourced_segment_ratio": round(total_ai / total_seg, 3),
        "heuristic_fallback_ratio": round(1 - (total_ai / total_seg), 3),
        "total_events_detected": meta.get("totalEvents", meta.get("total_events", None)),
        "event_types_detected": meta.get("eventTypes", meta.get("event_types", [])),
        "batches_planned": meta.get("batches", None),
    }


def compute_cross_video_diversity(plans: List[dict]) -> Dict[str, Any]:
    """Metrics ACROSS multiple videos - this is the direct evidence for
    "different videos get different camera behaviour, not a repeated
    pattern": compares the action-sequence and zoom-profile similarity
    between every pair of plans. Lower average similarity = more
    video-specific adaptation (higher is expected only when videos
    genuinely share similar cursor behaviour)."""
    if len(plans) < 2:
        return {"note": "need >= 2 plans to compute cross-video diversity"}

    def action_sequence(plan):
        return [s["action"] for s in plan.get("segments", [])]

    def zoom_profile(plan, buckets=10):
        segs = plan.get("segments", [])
        if not segs:
            return [0.0] * buckets
        total = segs[-1]["endTime"] or 1.0
        profile = [1.0] * buckets
        for s in segs:
            b0 = int(s["startTime"] / total * buckets)
            b1 = int(s["endTime"] / total * buckets)
            for b in range(max(0, b0), min(buckets, b1 + 1)):
                profile[b] = s["zoomLevel"]
        return profile

    def sequence_similarity(a, b):
        # normalized edit distance similarity (1 = identical, 0 = totally different)
        if not a and not b:
            return 1.0
        m, n = len(a), len(b)
        if m == 0 or n == 0:
            return 0.0
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        dist = dp[m][n]
        return 1 - dist / max(m, n)

    def profile_similarity(a, b):
        diffs = [abs(x - y) for x, y in zip(a, b)]
        max_range = 9.0  # zoomLevel range 1-10
        return 1 - (sum(diffs) / len(diffs)) / max_range

    seqs = [action_sequence(p) for p in plans]
    profiles = [zoom_profile(p) for p in plans]

    pair_seq_sims, pair_profile_sims = [], []
    for i in range(len(plans)):
        for j in range(i + 1, len(plans)):
            pair_seq_sims.append(sequence_similarity(seqs[i], seqs[j]))
            pair_profile_sims.append(profile_similarity(profiles[i], profiles[j]))

    return {
        "num_videos_compared": len(plans),
        "num_pairs": len(pair_seq_sims),
        "avg_action_sequence_similarity": round(statistics.mean(pair_seq_sims), 3),
        "avg_zoom_profile_similarity": round(statistics.mean(pair_profile_sims), 3),
        "interpretation": (
            "0 = every video produced a completely different camera plan; "
            "1 = every video produced an identical plan. Values well below 1 "
            "indicate the planner is adapting to each video's own content "
            "rather than repeating a fixed pattern."
        ),
    }


def compute_processing_metrics(processing_seconds: float, video_duration_s: float) -> Dict[str, Any]:
    ratio = processing_seconds / video_duration_s if video_duration_s else None
    return {
        "processing_time_s": round(processing_seconds, 2),
        "video_duration_s": round(video_duration_s, 2),
        "processing_to_duration_ratio": round(ratio, 2) if ratio is not None else None,
        "realtime_factor": round(video_duration_s / processing_seconds, 2) if processing_seconds else None,
    }


def to_markdown_table(metrics: Dict[str, Any], title: str = "Metrics") -> str:
    lines = [f"### {title}", "", "| Metric | Value |", "|---|---|"]
    for k, v in metrics.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)
