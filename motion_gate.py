"""
Pre-ML motion gate.

A cheap grayscale frame-difference (``cv2.absdiff``) on a 160x90 thumbnail
decides whether MediaPipe needs to run at all. On a static scene the inference
thread skips the neural network entirely and goes back to sleep, which is
where most of the power saving in IDLE / DEEP STANDBY comes from.

The *latch* matters: the wake gesture is a palm held **steady** for 2 s, which
produces almost no frame difference. Once motion is seen (the hand being
raised into view) the gate stays open for ``latch_s`` (> wake hold time) so
the steady palm is still analysed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from config import MotionGateConfig


@dataclass(frozen=True)
class GateDecision:
    run_inference: bool
    motion: bool                 # motion detected on *this* frame
    changed_fraction: float


class MotionGate:
    """Not thread-safe by design: owned exclusively by the inference thread."""

    def __init__(self, cfg: MotionGateConfig) -> None:
        self._cfg = cfg
        self._prev: Optional[np.ndarray] = None
        self._last_motion_t: float = float("-inf")

    def reset(self) -> None:
        self._prev = None
        self._last_motion_t = float("-inf")

    def open_latch(self, now: float) -> None:
        """Force the gate open (e.g. right after a state transition)."""
        self._last_motion_t = now

    def _prepare(self, bgr: np.ndarray) -> np.ndarray:
        small = cv2.resize(bgr, self._cfg.downscale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        k = self._cfg.blur_kernel
        return cv2.GaussianBlur(gray, (k, k), 0) if k > 1 else gray

    def evaluate(self, bgr: np.ndarray, now: float) -> GateDecision:
        gray = self._prepare(bgr)
        prev, self._prev = self._prev, gray
        if prev is None:
            # No reference yet: be conservative and run inference once.
            self._last_motion_t = now
            return GateDecision(True, True, 1.0)

        diff = cv2.absdiff(gray, prev)
        _, mask = cv2.threshold(diff, self._cfg.pixel_threshold, 255, cv2.THRESH_BINARY)
        fraction = float(cv2.countNonZero(mask)) / mask.size
        motion = fraction >= self._cfg.changed_fraction
        if motion:
            self._last_motion_t = now
        latched = (now - self._last_motion_t) <= self._cfg.latch_s
        return GateDecision(run_inference=motion or latched, motion=motion,
                            changed_fraction=fraction)
