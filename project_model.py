"""
project_model.py
-----------------
Non-destructive project/timeline model. This is what turns the pipeline
from "one input video -> one auto-edited output" into an actual editor:
a project is a saveable, reloadable, re-editable timeline of clips, each
with its own trim range, effects stack, and transition into the next
clip - plus a global audio track and export settings. Nothing here
renders pixels; that's `editing_engine.py`. This module only owns state
and its (de)serialization.

Hybrid human/AI editing model
------------------------------
Every clip's AI camera plan is a list of independently addressable
segments. A human never has to accept the whole plan or none of it:
- `Clip.segment_overrides` holds per-segment human edits, keyed by the
  segment's index into `camera_plan["segments"]`. `Clip.effective_segments()`
  merges these on top of the AI segments at render time, so an override is
  additive/reversible, never destructive to what the AI produced.
- `Clip.ai_satisfaction` records the human's verdict for this clip:
  "accepted" (download as-is) or "editing" (opened the "Edit More" tools).
  This is a status flag for the UI/API, not a gate - every editing
  endpoint works regardless of its value.
- `Clip.narrative` holds human-provided narrative intent (audience, tone,
  pacing) that `ai_planner.get_ai_style` can factor into styling/caption
  regeneration - the "judgment/taste" input a human supplies that the AI
  has no way to infer on its own.
- `TakeGroup` lets a human upload multiple candidate recordings of the
  same moment and pick the one that becomes the actual timeline clip -
  the AI never chooses between takes on its own.
"""

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

SUPPORTED_TRANSITIONS = {"none", "crossfade", "fade_black", "wipe_left", "wipe_right"}
SUPPORTED_FILTERS = {"brightness", "contrast", "saturation", "grayscale", "blur", "vignette", "sharpen"}
SATISFACTION_STATES = {"accepted", "editing", None}

EXPORT_PRESETS = {
    "youtube_1080p": {"width": 1920, "height": 1080, "fps": 30, "bitrate": "8000k"},
    "youtube_shorts": {"width": 1080, "height": 1920, "fps": 30, "bitrate": "6000k"},
    "instagram_reel": {"width": 1080, "height": 1920, "fps": 30, "bitrate": "5000k"},
    "instagram_feed": {"width": 1080, "height": 1080, "fps": 30, "bitrate": "5000k"},
    "twitter": {"width": 1280, "height": 720, "fps": 30, "bitrate": "5000k"},
    "linkedin": {"width": 1920, "height": 1080, "fps": 30, "bitrate": "6000k"},
    "source": {"width": None, "height": None, "fps": 30, "bitrate": "8000k"},
}


@dataclass
class Effect:
    kind: str                       # one of SUPPORTED_FILTERS
    params: Dict[str, Any] = field(default_factory=dict)
    startTime: Optional[float] = None   # None = applies to whole clip
    endTime: Optional[float] = None

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return Effect(kind=d["kind"], params=d.get("params", {}),
                       startTime=d.get("startTime"), endTime=d.get("endTime"))


@dataclass
class TextOverlay:
    text: str
    startTime: float
    endTime: float
    position: str = "bottom"        # top | center | bottom
    style: str = "caption"          # caption | title | lower_third

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return TextOverlay(**d)


@dataclass
class Clip:
    id: str
    source_path: str
    trim_start: float
    trim_end: float
    order: int
    camera_plan: Optional[dict] = None      # cinematic segments for this clip (from ai_planner)
    effects: List[Effect] = field(default_factory=list)
    text_overlays: List[TextOverlay] = field(default_factory=list)
    transition_in: str = "none"
    transition_duration: float = 0.6

    # ---- hybrid human/AI editing state -----------------------------------
    segment_overrides: Dict[str, dict] = field(default_factory=dict)   # str(index) -> override fields
    approved_segments: List[int] = field(default_factory=list)         # indices human explicitly signed off on
    narrative: Optional[dict] = None                                    # {"audience":..., "tone":..., "pacing":...}
    ai_satisfaction: Optional[str] = None                               # "accepted" | "editing" | None
    source_take_group: Optional[str] = None                            # TakeGroup.id this clip was chosen from, if any

    @property
    def duration(self):
        return max(0.0, self.trim_end - self.trim_start)

    def effective_segments(self) -> List[dict]:
        """The segments actually used at render time: AI segments with any
        human overrides merged on top, index-for-index. Never mutates
        `camera_plan` itself, so the original AI output stays inspectable/
        revertible at any time."""
        base = list((self.camera_plan or {}).get("segments", []))
        out = []
        for i, seg in enumerate(base):
            merged = dict(seg)
            override = self.segment_overrides.get(str(i))
            if override:
                merged.update(override)
                merged["humanEdited"] = True
            else:
                merged["humanEdited"] = False
            merged["approved"] = i in self.approved_segments
            out.append(merged)
        return out

    def set_segment_override(self, index: int, fields: dict):
        allowed = {"action", "focusX", "focusY", "zoomLevel", "panDirection",
                   "easing", "transitionType", "movementEnabled", "movementEndX", "movementEndY"}
        clean = {k: v for k, v in fields.items() if k in allowed}
        self.segment_overrides[str(index)] = clean
        self.approved_segments = [i for i in self.approved_segments if i != index]

    def clear_segment_override(self, index: int):
        self.segment_overrides.pop(str(index), None)

    def approve_segment(self, index: int):
        if index not in self.approved_segments:
            self.approved_segments.append(index)

    def to_dict(self):
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d):
        clip = Clip(
            id=d["id"], source_path=d["source_path"], trim_start=d["trim_start"],
            trim_end=d["trim_end"], order=d["order"], camera_plan=d.get("camera_plan"),
            transition_in=d.get("transition_in", "none"),
            transition_duration=d.get("transition_duration", 0.6),
            segment_overrides=d.get("segment_overrides", {}),
            approved_segments=d.get("approved_segments", []),
            narrative=d.get("narrative"),
            ai_satisfaction=d.get("ai_satisfaction"),
            source_take_group=d.get("source_take_group"),
        )
        clip.effects = [Effect.from_dict(e) for e in d.get("effects", [])]
        clip.text_overlays = [TextOverlay.from_dict(t) for t in d.get("text_overlays", [])]
        return clip


@dataclass
class AudioTrack:
    path: str
    volume: float = 1.0
    start_offset: float = 0.0
    duck_under_original: bool = True
    duck_level: float = 0.25

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return AudioTrack(**d)


@dataclass
class TakeGroup:
    """A set of candidate recordings of the same moment. The AI never
    picks between them - a human always makes the final selection via
    `Project.select_take`."""
    id: str
    label: str
    candidate_paths: List[str] = field(default_factory=list)
    selected_path: Optional[str] = None
    resulting_clip_id: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return TakeGroup(**d)


@dataclass
class Project:
    id: str
    name: str
    clips: List[Clip] = field(default_factory=list)
    audio_tracks: List[AudioTrack] = field(default_factory=list)
    export_preset: str = "youtube_1080p"
    take_groups: List[TakeGroup] = field(default_factory=list)
    history: List[dict] = field(default_factory=list)   # undo stack of prior serialized states
    redo_stack: List[dict] = field(default_factory=list)

    # ---- clip management -------------------------------------------------
    def add_clip(self, source_path, trim_start=0.0, trim_end=None, camera_plan=None) -> Clip:
        self._snapshot()
        order = len(self.clips)
        clip = Clip(id=uuid.uuid4().hex[:8], source_path=source_path,
                    trim_start=trim_start, trim_end=trim_end if trim_end is not None else trim_start,
                    order=order, camera_plan=camera_plan)
        self.clips.append(clip)
        return clip

    def remove_clip(self, clip_id: str):
        self._snapshot()
        self.clips = [c for c in self.clips if c.id != clip_id]
        self._reorder()

    def split_clip(self, clip_id: str, split_time_abs: float) -> Optional[Clip]:
        """Splits a clip into two at an absolute time (relative to its own
        trimmed range) - the defining "editor" operation. Returns the new
        second half clip."""
        self._snapshot()
        target = next((c for c in self.clips if c.id == clip_id), None)
        if target is None or not (target.trim_start < split_time_abs < target.trim_end):
            return None
        new_clip = Clip(
            id=uuid.uuid4().hex[:8], source_path=target.source_path,
            trim_start=split_time_abs, trim_end=target.trim_end,
            order=target.order + 1, camera_plan=target.camera_plan,
            transition_in=target.transition_in, transition_duration=target.transition_duration,
        )
        target.trim_end = split_time_abs
        self.clips.insert(self.clips.index(target) + 1, new_clip)
        self._reorder()
        return new_clip

    def reorder_clips(self, ordered_ids: List[str]):
        self._snapshot()
        lookup = {c.id: c for c in self.clips}
        self.clips = [lookup[i] for i in ordered_ids if i in lookup]
        self._reorder()

    def _reorder(self):
        for i, c in enumerate(self.clips):
            c.order = i

    # ---- multi-take selection (human chooses, AI never does) ---------------
    def add_take_group(self, label: str, candidate_paths: List[str]) -> TakeGroup:
        self._snapshot()
        group = TakeGroup(id=uuid.uuid4().hex[:8], label=label, candidate_paths=list(candidate_paths))
        self.take_groups.append(group)
        return group

    def select_take(self, group_id: str, chosen_path: str, trim_start=0.0,
                     trim_end=None, camera_plan=None) -> Optional[Clip]:
        group = next((g for g in self.take_groups if g.id == group_id), None)
        if group is None or chosen_path not in group.candidate_paths:
            return None
        clip = self.add_clip(chosen_path, trim_start=trim_start, trim_end=trim_end, camera_plan=camera_plan)
        clip.source_take_group = group.id
        group.selected_path = chosen_path
        group.resulting_clip_id = clip.id
        return clip

    # ---- hybrid satisfaction workflow --------------------------------------
    def set_satisfaction(self, clip_id: str, state: Optional[str]) -> Optional[Clip]:
        if state not in SATISFACTION_STATES:
            raise ValueError(f"Invalid satisfaction state: {state}")
        clip = next((c for c in self.clips if c.id == clip_id), None)
        if clip is None:
            return None
        clip.ai_satisfaction = state
        return clip

    # ---- undo / redo -------------------------------------------------------
    def _snapshot(self):
        self.history.append(self.to_dict(include_history=False))
        self.redo_stack.clear()
        if len(self.history) > 50:
            self.history.pop(0)

    def undo(self) -> bool:
        if not self.history:
            return False
        self.redo_stack.append(self.to_dict(include_history=False))
        prior = self.history.pop()
        restored = Project.from_dict(prior)
        self.clips = restored.clips
        self.audio_tracks = restored.audio_tracks
        self.export_preset = restored.export_preset
        self.take_groups = restored.take_groups
        return True

    def redo(self) -> bool:
        if not self.redo_stack:
            return False
        self.history.append(self.to_dict(include_history=False))
        nxt = self.redo_stack.pop()
        restored = Project.from_dict(nxt)
        self.clips = restored.clips
        self.audio_tracks = restored.audio_tracks
        self.export_preset = restored.export_preset
        self.take_groups = restored.take_groups
        return True

    # ---- (de)serialization --------------------------------------------------
    def to_dict(self, include_history=True) -> dict:
        d = {
            "id": self.id, "name": self.name,
            "clips": [c.to_dict() for c in self.clips],
            "audio_tracks": [a.to_dict() for a in self.audio_tracks],
            "export_preset": self.export_preset,
            "take_groups": [g.to_dict() for g in self.take_groups],
        }
        if include_history:
            d["history"] = self.history
            d["redo_stack"] = self.redo_stack
        return d

    @staticmethod
    def from_dict(d) -> "Project":
        p = Project(id=d.get("id", uuid.uuid4().hex[:8]), name=d.get("name", "Untitled Project"))
        p.clips = [Clip.from_dict(c) for c in d.get("clips", [])]
        p.audio_tracks = [AudioTrack.from_dict(a) for a in d.get("audio_tracks", [])]
        p.export_preset = d.get("export_preset", "youtube_1080p")
        p.take_groups = [TakeGroup.from_dict(g) for g in d.get("take_groups", [])]
        p.history = d.get("history", [])
        p.redo_stack = d.get("redo_stack", [])
        return p


def save_project(project: Project, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(project.to_dict(), fh, indent=2)


def load_project(path: str) -> Project:
    with open(path) as fh:
        return Project.from_dict(json.load(fh))


def new_project(name: str) -> Project:
    return Project(id=uuid.uuid4().hex[:8], name=name)
