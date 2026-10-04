"""Shared fixtures: synthetic hands and trajectories (no camera / MediaPipe)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DEFAULT_CONFIG  # noqa: E402
from tracker import HandObservation, normalize_landmarks  # noqa: E402

# Canonical upright right hand in "palm length" units (wrist -> middle MCP == 1),
# image convention: +x right, +y DOWN, so fingers point towards -y.
_MCP = {"index": (-0.35, -0.95), "middle": (0.0, -1.0), "ring": (0.30, -0.95), "pinky": (0.55, -0.85)}
_FINGER_IDX = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
               "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}


def _rot(points: np.ndarray, quarter_turns: int = 0, degrees: float = 0.0) -> np.ndarray:
    out = points.copy()
    for _ in range(quarter_turns % 4):                # (x, y) -> (-y, x)
        out[:, 0], out[:, 1] = -out[:, 1].copy(), out[:, 0].copy()
    if degrees:
        a = np.radians(degrees)
        r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        out[:, :2] = out[:, :2] @ r.T
    return out


def make_hand(fingers=(False, False, False, False), thumb="folded",
              quarter_turns=0, degrees=0.0) -> np.ndarray:
    """(21, 3) wrist-origin landmarks for a given finger pattern.

    thumb: "folded" | "out" | "pinch"
    quarter_turns=1 turns a fist-with-thumb-out into THUMB_UP, 3 into THUMB_DOWN.
    """
    p = np.zeros((21, 3), dtype=np.float32)
    p[1, :2] = (-0.30, -0.15)
    p[2, :2] = (-0.55, -0.35)
    for name, ext in zip(("index", "middle", "ring", "pinky"), fingers):
        mx, my = _MCP[name]
        mcp, pip, dip, tip = _FINGER_IDX[name]
        p[mcp, :2] = (mx, my)
        if ext:
            p[pip, :2] = (mx, my - 0.45)
            p[dip, :2] = (mx, my - 0.75)
            p[tip, :2] = (mx, my - 0.95)
        else:
            p[pip, :2] = (mx, my - 0.35)
            p[dip, :2] = (mx, my - 0.15)
            p[tip, :2] = (mx, -0.60)
    if thumb == "out":
        p[3, :2] = (-0.85, -0.40)
        p[4, :2] = (-1.15, -0.42)
    elif thumb == "pinch":
        p[3, :2] = (-0.60, -0.70)
        p[4, :2] = (-0.45, -1.05)
        p[6, :2] = (-0.40, -1.30)
        p[7, :2] = (-0.45, -1.20)
        p[8, :2] = (-0.40, -1.05)
    else:
        p[3, :2] = (-0.45, -0.55)
        p[4, :2] = (-0.15, -0.60)
    return _rot(p, quarter_turns, degrees)


HANDS = {
    "FIST": make_hand(),
    "OPEN_PALM": make_hand((True,) * 4, "out"),
    "FOUR": make_hand((True,) * 4, "folded"),
    "POINT": make_hand((True, False, False, False)),
    "PEACE": make_hand((True, True, False, False)),
    "THREE": make_hand((True, True, True, False)),
    "PINCH": make_hand(thumb="pinch"),
    "THUMB_UP": make_hand(thumb="out", quarter_turns=1),
    "THUMB_DOWN": make_hand(thumb="out", quarter_turns=3),
}


def observation(pose: str | None, t: float, wrist=(320.0, 200.0), palm=80.0) -> HandObservation:
    """A HandObservation as the tracker would emit it."""
    if pose is None:
        return HandObservation(t)
    px = HANDS[pose] * palm
    px[:, :2] += np.asarray(wrist, dtype=np.float32)
    scaled, scale = normalize_landmarks(px)
    return HandObservation(t, scaled, px[0, :2].copy(), scale, "Right")


@pytest.fixture
def cfg():
    return DEFAULT_CONFIG
