"""
Per-stroke swipe telemetry for threshold tuning (``python main.py --swipe-log``).

Every stroke the ``SwipeDetector`` segments - typed, rejected, suppressed or
jitter - becomes one CSV row with the numbers each check looked at, so missed
swipes can be explained and ``GestureConfig.swipe_*`` tuned from real motion.
Written from the inference thread only; each row is flushed immediately so the
file survives a crash or a hard stop.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Optional, TextIO

COLUMNS = (
    "t_start", "t_end", "outcome", "reason", "direction", "pose", "pose_votes", "ended",
    "duration", "frames", "distance", "dx", "dy", "path", "straightness", "axis_ratio",
    "peak_speed", "tip_cos_min", "tip_share_min", "shape_drift", "rigid_share", "palm_px",
)


def _fmt(v) -> str:
    if isinstance(v, float):
        return "inf" if math.isinf(v) else f"{v:.3f}"
    return str(v)


class SwipeCsvLog:
    """Callable sink for ``SwipeDetector.on_stroke``; appends to ``path``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists() or self.path.stat().st_size == 0
        self._fh: Optional[TextIO] = open(self.path, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._fh, fieldnames=COLUMNS, extrasaction="ignore")
        if new:
            self._w.writeheader()
        self.rows = 0

    def __call__(self, stroke: dict) -> None:
        if self._fh is None:
            return
        self._w.writerow({k: _fmt(stroke.get(k, "")) for k in COLUMNS})
        self._fh.flush()
        self.rows += 1

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
