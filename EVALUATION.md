# Project Readiness Assessment

## Task 1 — "Is this ready to replace a full video editor?"

**No, not yet as of the previous version** (before this redesign round). It was a
single-purpose auto-zoom/pan pipeline for one input video producing one output
video. A general-purpose editor needs capabilities that pipeline never had. This
redesign adds them; the table below is the gap analysis that drove what was built.

| Capability a video editor needs | Before this round | After this round |
|---|---|---|
| Multi-clip timeline (import >1 clip, arrange, reorder) | ❌ single video in → single video out | ✅ `project_model.Project` + `/project/<id>/clip` |
| Trim / cut / split | ❌ | ✅ `Project.split_clip`, `/clip/<id>/trim` |
| Transitions between clips | ❌ | ✅ crossfade, fade-to-black in `editing_engine._apply_transition` (directional wipes currently approximated as crossfade — see Known Gaps) |
| Visual filters / color grade | ❌ | ✅ brightness/contrast/saturation/grayscale/blur/sharpen/vignette in `editing_engine.apply_effects` |
| Text/title overlays | ❌ (only one auto-generated caption) | ✅ arbitrary per-clip overlays, 3 styles, in `editing_engine.apply_text_overlays` |
| Background music / audio mixing / ducking | ❌ | ✅ `editing_engine.mix_audio`, `AudioTrack` model |
| Export presets for different platforms | ❌ fixed 1920x1080 | ✅ `EXPORT_PRESETS` (YouTube, Shorts, Reels, feed, Twitter, LinkedIn) |
| Undo / redo | ❌ | ✅ `Project.undo/redo`, snapshot-based history |
| Save / reload a project (non-destructive editing) | ❌ | ✅ `save_project`/`load_project` (JSON) |
| AI-assisted cinematic camera movement | ✅ (this was the whole original scope) | ✅ retained, now runs per-clip inside the timeline model |
| Auto-generated promo/trailer | ❌ | ✅ `promo_generator.py` (Task 3) |

**Bottom line:** the redesign closes the largest functional gaps (multi-clip
editing, trims/splits, transitions, filters, text, audio, export presets,
undo/redo, save/reload). It is now a small but real timeline-based editor with
an AI camera-direction feature on top, not just an auto-zoom script. See
**Known Gaps** below for what still separates it from a mature product like
Premiere/CapCut/Descript.

## Task 4 — IEEE-journal publication readiness

This is an **engineering systems paper candidate** (a novel pipeline +
evaluation), not a theoretical-contribution paper. That is a legitimate and
common IEEE paper type (e.g. demo/systems tracks, workshop papers), but it
changes what "ready" means: the bar is a working system + a clear, honest,
reproducible evaluation — not a new algorithm with proofs.

### What is in place
- A defined, documented architecture (`ARCHITECTURE.md`) with a novel
  contribution to point to: an event-anchored, two-stage (vision-description +
  text-reasoning) AI camera director, replacing a fixed-heuristic auto-zoom
  baseline.
- A **quantitative evaluation module** (`metrics_engine.py`) that computes,
  from real runs (not asserted numbers):
  - Per-video: segment count, action distribution, action-diversity entropy,
    zoom-level mean/variance, inter-segment zoom-delta (stability), AI vs.
    heuristic-fallback ratio, detected-event counts.
  - Cross-video: action-sequence similarity (normalized edit distance) and
    zoom-profile similarity between every pair of processed videos — the
    direct evidence for/against "different videos get different camera
    behaviour," which is the paper's central claim.
  - Processing cost: wall-clock time vs. video duration (real-time factor).
- Reproducible endpoints (`/metrics/plan`, `/metrics/diversity`) so numbers in
  a paper can be regenerated from the shipped code, not hand-computed once.

### What is NOT in place yet (must be done before submission)
1. **No human evaluation.** "Cinematic quality" is inherently subjective;
   the entropy/stability metrics are necessary but not sufficient evidence.
   A paper claiming a better viewer experience needs a user study (e.g.
   A/B preference test vs. a fixed-zoom baseline, N≥20 participants, a
   preregistered rubric — smoothness, relevance of focus, perceived
   professionalism).
2. **No baseline comparison implemented.** The old fixed-zoom heuristic is no
   longer in the codebase to compare against numerically. For a paper, keep
   (or reconstruct) that baseline and run the same metrics on it side-by-side
   with the new planner on an identical video set.
3. **No benchmark dataset.** Metrics need to be run across a defined,
   describable set of screen recordings (e.g. N≥15-20 videos spanning
   dashboards, forms, code editors, presentations) with the set itself
   documented (source, length distribution, licensing) so results are
   reproducible by reviewers.
4. **No ablation study.** E.g. AI-director-on vs. heuristic-only vs.
   fixed-zoom, to isolate how much each new component (kinematics tracking,
   event-anchored sampling, two-stage reasoning) contributes.
5. **No failure-mode analysis.** The AI call fails silently to a heuristic
   today (by design, for robustness) — a paper needs the *rate* of that
   fallback reported honestly per video/dataset, not just the capability.
6. **No related-work comparison table.** Needs positioning against existing
   auto-editing/auto-zoom systems (commercial: Descript, Camtasia SmartFocus,
   OBS auto-crop plugins; academic: prior automatic video editing / camera
   planning papers) with a feature/metric comparison table.
7. **Promo-generation feature (Task 3) is unvalidated end-to-end** — it
   depends on an external NIM video-generation endpoint this environment
   cannot reach to test live; the submit/poll/fetch contract is implemented
   defensively (see `promo_generator.py` docstring) but must be run against
   the real endpoint and its output quality assessed before being cited as a
   working contribution.

### Recommended next steps, in order
1. Run `/metrics/plan` across a documented benchmark set; save the raw JSON
   outputs (this is the paper's Table 1/2 source data).
2. Re-add the old fixed-zoom heuristic as an explicit baseline module (not
   deleted, just excluded from the default pipeline) so it can be A/B'd.
3. Run `/metrics/diversity` on the benchmark set for the cross-video
   diversity number.
4. Design and run the human preference study.
5. Only then draft the paper — Claude can help write it once 1-4 produce real
   numbers to report; do not draft results sections before that data exists.

## Metrics reference (computed, not asserted)

`metrics_engine.compute_camera_metrics(plan)` returns:

```
num_segments, avg/min/max_segment_duration_s, action_distribution,
action_diversity_normalized_entropy, zoom_level_mean, zoom_level_stdev,
avg_zoom_delta_between_segments, max_zoom_delta_observed,
ai_sourced_segment_ratio, heuristic_fallback_ratio,
total_events_detected, event_types_detected, batches_planned
```

`metrics_engine.compute_cross_video_diversity(plans)` returns:

```
num_videos_compared, num_pairs, avg_action_sequence_similarity,
avg_zoom_profile_similarity, interpretation
```

`metrics_engine.compute_processing_metrics(processing_seconds, video_duration_s)` returns:

```
processing_time_s, video_duration_s, processing_to_duration_ratio, realtime_factor
```

`metrics_engine.to_markdown_table(metrics, title)` renders any of the above as
a Markdown table for direct inclusion in a paper draft.

## Known gaps (engineering, not evaluation)
- Directional wipe transitions are approximated as crossfades; a true masked
  wipe compositor is a documented future-extension point, not implemented.
- No non-linear-editing UI (timeline scrubber, drag-and-drop) — everything
  above is API-level; a frontend for it does not exist yet.
- No collaborative/multi-user project features.
- No GPU-accelerated rendering path; export is single-threaded-per-clip CPU
  (moviepy/ffmpeg), which will be slow for long timelines.
