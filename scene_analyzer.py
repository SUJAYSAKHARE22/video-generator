"""
scene_analyzer.py
-----------------
Classical computer-vision scene understanding: detects candidate UI
regions (modals/dialogs, panels, buttons, text-dense areas) in a single
frame so the planner can reason about *semantic* importance instead of
only reacting to raw pixel motion or cursor position.

OCR (pytesseract) is used opportunistically when available; the module
degrades gracefully if the tesseract binary isn't installed in the
environment, since it is only ever an optional supporting signal.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

try:
    import pytesseract
    _HAS_TESSERACT = True
except ImportError:
    _HAS_TESSERACT = False


@dataclass
class UIRegion:
    kind: str          # "modal" | "panel" | "text_block" | "button_like" | "toolbar"
    x_pct: float
    y_pct: float
    w_pct: float
    h_pct: float
    importance: float  # 0..1 heuristic score
    text: Optional[str] = None


def _rect_pct(x, y, w, h, frame_w, frame_h):
    return (x / frame_w * 100.0, y / frame_h * 100.0,
            w / frame_w * 100.0, h / frame_h * 100.0)


def detect_modal_or_dialog(frame_bgr) -> Optional[UIRegion]:
    """Looks for a single large, roughly-centered rectangular region with a
    strong contrast border against its surroundings - the classic visual
    signature of a modal / dialog box overlaying the page."""
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = 0.0
    frame_area = w * h

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        area_ratio = area / frame_area
        if area_ratio < 0.12 or area_ratio > 0.85:
            continue
        cx, cy = x + cw / 2, y + ch / 2
        centeredness = 1 - (abs(cx - w / 2) / (w / 2) + abs(cy - h / 2) / (h / 2)) / 2
        if centeredness < 0.35:
            continue
        aspect = cw / max(ch, 1)
        if aspect < 0.3 or aspect > 4.0:
            continue
        score = area_ratio * 0.6 + centeredness * 0.4
        if score > best_score:
            best_score = score
            best = (x, y, cw, ch)

    if best is None:
        return None

    x, y, cw, ch = best
    px, py, pw, ph = _rect_pct(x, y, cw, ch, w, h)
    importance = min(1.0, 0.55 + best_score * 0.5)
    return UIRegion(kind="modal", x_pct=px, y_pct=py, w_pct=pw, h_pct=ph, importance=importance)


def detect_text_density_map(frame_bgr, grid=(6, 4)) -> np.ndarray:
    """Cheap proxy for 'where is text-heavy / UI-dense content' without
    running OCR on every frame: high local-edge density correlates well
    with text, icons and controls."""
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    gh, gw = grid
    density = np.zeros((gh, gw), dtype=np.float32)
    ch, cw = h // gh, w // gw
    for r in range(gh):
        for c in range(gw):
            cell = edges[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
            density[r, c] = float(np.mean(cell > 0))
    if density.max() > 0:
        density /= density.max()
    return density


def density_peak_focus(density: np.ndarray):
    gh, gw = density.shape
    idx = np.unravel_index(np.argmax(density), density.shape)
    r, c = idx
    x_pct = (c + 0.5) / gw * 100.0
    y_pct = (r + 0.5) / gh * 100.0
    return x_pct, y_pct, float(density[r, c])


def ocr_active_region(frame_bgr, region_pct=None) -> Optional[str]:
    """Runs OCR on a region (or the whole frame) if pytesseract/tesseract is
    available. Returns None silently otherwise - OCR is a bonus signal, not
    a hard dependency."""
    if not _HAS_TESSERACT:
        return None
    try:
        h, w = frame_bgr.shape[:2]
        if region_pct:
            x = int(region_pct["x_pct"] / 100 * w)
            y = int(region_pct["y_pct"] / 100 * h)
            rw = int(region_pct["w_pct"] / 100 * w)
            rh = int(region_pct["h_pct"] / 100 * h)
            crop = frame_bgr[max(0, y):y + rh, max(0, x):x + rw]
        else:
            crop = frame_bgr
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config="--psm 6")
        text = " ".join(text.split())
        return text[:200] if text else None
    except Exception:
        return None


def analyze_frame(frame_bgr) -> List[UIRegion]:
    """Full single-frame scene analysis pass producing an ordered list of
    candidate UI regions of interest (most important first)."""
    regions: List[UIRegion] = []

    modal = detect_modal_or_dialog(frame_bgr)
    if modal is not None:
        modal.text = ocr_active_region(frame_bgr, {
            "x_pct": modal.x_pct, "y_pct": modal.y_pct,
            "w_pct": modal.w_pct, "h_pct": modal.h_pct,
        })
        regions.append(modal)

    density = detect_text_density_map(frame_bgr)
    x_pct, y_pct, strength = density_peak_focus(density)
    if strength > 0.15:
        regions.append(UIRegion(
            kind="text_block", x_pct=max(0, x_pct - 12), y_pct=max(0, y_pct - 8),
            w_pct=24, h_pct=16, importance=min(1.0, 0.3 + strength * 0.5),
        ))

    regions.sort(key=lambda r: r.importance, reverse=True)
    return regions
