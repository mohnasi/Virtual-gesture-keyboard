"""
Gesture recognition on normalised landmarks.

Layers
------
1. ``classify_pose``      - per-frame static hand shape from scaled landmarks
                            (rotation-tolerant: distances from the wrist, not
                            raw x/y comparisons).
2. ``TemporalBuffer``     - fixed-length ``collections.deque`` (45 frames,
                            ~1.5 s @ 30 FPS) of per-frame records.
3. ``GestureClassifier``  - runs a list of pluggable ``Detector`` objects over
                            the buffer and emits ``GestureEvent``s:
      * ``SwipeDetector``      wrist-trajectory strokes -> SWIPE(direction, pose)
      * ``StaticHoldDetector`` steady pose for N seconds -> HOLD(pose)
      * ``RepetitionDetector`` pose entered K times in a window -> SEQUENCE(pose)

Add a new gesture by subclassing ``Detector`` and passing it to
``GestureClassifier(extra_detectors=[...])``.
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Iterable, List, Optional

import numpy as np

from config import GestureConfig, PoseConfig, SystemState
from tracker import (
    INDEX_MCP, INDEX_PIP, INDEX_TIP, MIDDLE_PIP, MIDDLE_TIP, PINKY_PIP, PINKY_TIP,
    RING_PIP, RING_TIP, THUMB_IP, THUMB_MCP, THUMB_TIP, HandObservation,
)


# --------------------------------------------------------------------------- #
# 1. Static pose classification
# --------------------------------------------------------------------------- #
class Pose(str, Enum):
    UNKNOWN = "UNKNOWN"
    FIST = "FIST"
    OPEN_PALM = "OPEN_PALM"
    POINT = "POINT"
    PEACE = "PEACE"
    THREE = "THREE"
    FOUR = "FOUR"
    PINCH = "PINCH"
    THUMB_UP = "THUMB_UP"
    THUMB_DOWN = "THUMB_DOWN"


_FINGERS = ((INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP),
            (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP))

_PATTERNS = {
    (True, False, False, False): Pose.POINT,
    (True, True, False, False): Pose.PEACE,
    (True, True, True, False): Pose.THREE,
    (False, False, False, False): Pose.FIST,
}


def finger_extension(lm: np.ndarray, cfg: PoseConfig) -> tuple:
    """(thumb_extended, (index, middle, ring, pinky)) from wrist-origin landmarks."""
    dist = np.linalg.norm(lm, axis=1)          # distance of every node from wrist
    fingers = tuple(bool(dist[tip] > dist[pip] * cfg.finger_extended_ratio)
                    for tip, pip in _FINGERS)
    thumb = bool(
        np.linalg.norm(lm[THUMB_TIP] - lm[INDEX_MCP]) > cfg.thumb_extended_dist
        and dist[THUMB_TIP] > dist[THUMB_IP]
    )
    return thumb, fingers


def classify_pose(lm: Optional[np.ndarray], cfg: PoseConfig) -> Optional[Pose]:
    if lm is None:
        return None
    thumb, fingers = finger_extension(lm, cfg)

    if not any(fingers) and thumb:
        v = lm[THUMB_TIP, :2] - lm[THUMB_MCP, :2]
        norm = float(np.linalg.norm(v))
        if norm > 1e-6:
            cos_up = -v[1] / norm              # image y grows downwards
            if cos_up >= cfg.thumb_vertical_cos:
                return Pose.THUMB_UP
            if cos_up <= -cfg.thumb_vertical_cos:
                return Pose.THUMB_DOWN
        return Pose.FIST                       # sideways thumb on a fist

    if (np.linalg.norm(lm[THUMB_TIP] - lm[INDEX_TIP]) < cfg.pinch_dist
            and np.linalg.norm(lm[INDEX_TIP]) > cfg.pinch_min_index_reach):
        return Pose.PINCH

    if all(fingers):
        return Pose.OPEN_PALM if thumb else Pose.FOUR
    return _PATTERNS.get(fingers, Pose.UNKNOWN)


# --------------------------------------------------------------------------- #
# 2. Temporal rolling buffer
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FrameRecord:
    obs: HandObservation
    pose: Optional[Pose]           # None when no hand is present

    @property
    def t(self) -> float:
        return self.obs.timestamp


class TemporalBuffer:
    """Fixed-length history (``deque(maxlen=N)``) of landmark records."""

    def __init__(self, maxlen: int) -> None:
        self._dq: Deque[FrameRecord] = deque(maxlen=maxlen)

    def append(self, rec: FrameRecord) -> None:
        self._dq.append(rec)

    def clear(self) -> None:
        self._dq.clear()

    def __len__(self) -> int:
        return len(self._dq)

    def __iter__(self):
        return iter(self._dq)

    @property
    def maxlen(self) -> int:
        return self._dq.maxlen or 0

    def latest(self) -> Optional[FrameRecord]:
        return self._dq[-1] if self._dq else None

    def since(self, t0: float) -> List[FrameRecord]:
        out: List[FrameRecord] = []
        for rec in reversed(self._dq):       # newest first; stop early
            if rec.t < t0:
                break
            out.append(rec)
        out.reverse()
        return out

    def trajectory(self, t0: float = float("-inf")) -> np.ndarray:
        """(N, 3) array of [t, wrist_x/palm, wrist_y/palm] for present frames."""
        rows = [(r.t, *(r.obs.wrist_px / r.obs.palm_px))
                for r in self.since(t0) if r.obs.present]
        return np.asarray(rows, dtype=np.float64).reshape(-1, 3)


# --------------------------------------------------------------------------- #
# 3. Events and detectors
# --------------------------------------------------------------------------- #
class GestureKind(str, Enum):
    SWIPE = "SWIPE"
    HOLD = "HOLD"
    SEQUENCE = "SEQUENCE"


class Direction(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"


_OPPOSITE = {Direction.LEFT: Direction.RIGHT, Direction.RIGHT: Direction.LEFT,
             Direction.UP: Direction.DOWN, Direction.DOWN: Direction.UP}


@dataclass(frozen=True)
class GestureEvent:
    kind: GestureKind
    pose: Pose
    timestamp: float
    direction: Optional[Direction] = None
    count: int = 1


class Detector:
    """Base class for pluggable temporal detectors."""

    def configure(self, state: SystemState, fps: float) -> None:
        """Adapt to the current power state (frame rate)."""

    def reset(self) -> None:
        """Forget all history (called on state transitions)."""

    def update(self, rec: FrameRecord, buf: TemporalBuffer) -> Optional[GestureEvent]:
        raise NotImplementedError


class SwipeDetector(Detector):
    """Segments wrist-trajectory strokes with a speed hysteresis, then
    validates distance, straightness and axis dominance on the buffered curve.

    Distances are in palm lengths so the same physical motion works whether
    the user sits 40 cm or 1.5 m from the camera.
    """

    MAX_GAP_S = 0.25

    def __init__(self, cfg: GestureConfig) -> None:
        self._cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._prev: Optional[FrameRecord] = None
        self._speed = 0.0
        self._stroke_t0: Optional[float] = None
        self._rest_t: Optional[float] = None      # last frame the hand was ~still
        self._last_t = float("-inf")
        self._last_dir: Optional[Direction] = None

    def update(self, rec, buf):
        if not rec.obs.present:
            self._prev, self._stroke_t0, self._rest_t, self._speed = None, None, None, 0.0
            return None
        prev, self._prev = self._prev, rec
        if prev is None:
            return None
        dt = rec.t - prev.t
        if dt <= 0 or dt > self.MAX_GAP_S:          # gated / dropped frames
            self._stroke_t0, self._rest_t, self._speed = None, None, 0.0
            return None

        palm = 0.5 * (rec.obs.palm_px + prev.obs.palm_px)
        inst = float(np.linalg.norm(rec.obs.wrist_px - prev.obs.wrist_px)) / palm / dt
        a = self._cfg.speed_smoothing
        self._speed = a * inst + (1 - a) * self._speed

        if self._stroke_t0 is None:
            if inst <= self._cfg.swipe_end_speed:
                self._rest_t = rec.t
            elif self._rest_t is None:
                self._rest_t = prev.t
            if self._speed >= self._cfg.swipe_start_speed:
                # The EMA lags a frame or two; anchor the stroke at the last
                # rest point so the full displacement is measured.
                self._stroke_t0 = self._rest_t
            return None

        duration = rec.t - self._stroke_t0
        if self._speed > self._cfg.swipe_end_speed and duration <= self._cfg.swipe_max_duration_s:
            return None                               # stroke still in flight
        t0, self._stroke_t0, self._rest_t = self._stroke_t0, None, rec.t
        return self._evaluate(buf.since(t0), t0, rec.t)

    def _evaluate(self, recs: List[FrameRecord], t0: float, t1: float):
        c = self._cfg
        pts = [r for r in recs if r.obs.present]
        if len(pts) < 2:
            return None
        duration = t1 - t0
        if not (c.swipe_min_duration_s <= duration <= c.swipe_max_duration_s):
            return None

        steps = np.array([(b.obs.wrist_px - a.obs.wrist_px)
                          / (0.5 * (a.obs.palm_px + b.obs.palm_px))
                          for a, b in zip(pts, pts[1:])])
        net = steps.sum(axis=0)
        net_len = float(np.linalg.norm(net))
        path_len = float(np.linalg.norm(steps, axis=1).sum())
        if net_len < c.swipe_min_distance or path_len <= 0:
            return None
        if net_len / path_len < c.swipe_min_straightness:
            return None
        major, minor = sorted((abs(net[0]), abs(net[1])), reverse=True)
        if minor > 0 and major / minor < c.swipe_axis_dominance:
            return None

        if abs(net[0]) >= abs(net[1]):
            direction = Direction.RIGHT if net[0] > 0 else Direction.LEFT
        else:
            direction = Direction.DOWN if net[1] > 0 else Direction.UP

        # Return-stroke suppression: bringing the hand back after a swipe
        # must not type the opposite key.
        if (self._last_dir is not None and direction == _OPPOSITE[self._last_dir]
                and t0 - self._last_t < c.return_suppress_s):
            self._last_dir = None
            return None
        if t0 - self._last_t < c.refractory_s:
            return None

        pose = self._vote(pts)
        self._last_t = t1
        # A FIST stroke is the "clutch" (reposition without typing): it also
        # cancels return suppression so the next stroke is free.
        self._last_dir = None if pose == Pose.FIST else direction
        return GestureEvent(GestureKind.SWIPE, pose, t1, direction)

    def _vote(self, pts: List[FrameRecord]) -> Pose:
        votes = Counter(r.pose for r in pts if r.pose not in (None, Pose.UNKNOWN))
        if not votes:
            return Pose.UNKNOWN
        pose, n = votes.most_common(1)[0]
        known = sum(votes.values())
        return pose if n / known >= self._cfg.pose_vote_min_share and n >= 2 else Pose.UNKNOWN


class StaticHoldDetector(Detector):
    """Fires once when ``pose`` has been held, steady, for ``hold_s``."""

    def __init__(self, pose: Pose, hold_s: float, min_share: float, max_drift: float) -> None:
        self.pose, self.hold_s = pose, hold_s
        self.min_share, self.max_drift = min_share, max_drift
        self.reset()

    def reset(self) -> None:
        self._armed = True

    def update(self, rec, buf):
        if rec.pose != self.pose:
            self._armed = True                # re-arm once the pose is released
            return None
        if not self._armed:
            return None
        window = buf.since(rec.t - self.hold_s)
        if not window or window[-1].t - window[0].t < 0.9 * self.hold_s:
            return None                       # not enough history yet
        share = sum(r.pose == self.pose for r in window) / len(window)
        if share < self.min_share:
            return None
        present = [r for r in window if r.obs.present]
        palm = float(np.median([r.obs.palm_px for r in present]))
        wrists = np.array([r.obs.wrist_px for r in present]) / palm
        drift = float(np.sqrt(wrists.var(axis=0).sum()))
        if drift > self.max_drift:
            return None
        self._armed = False
        return GestureEvent(GestureKind.HOLD, self.pose, rec.t)


class RepetitionDetector(Detector):
    """Counts debounced *entries* into ``pose``; fires after ``reps`` entries
    within ``window_s``. Used for "Off, Off, Off" (thumb down x3) and
    "On, On, On" (thumb up x3). Debounce is expressed in frames so it remains
    meaningful at 1 FPS (where a single frame is all we get)."""

    def __init__(self, pose: Pose, cfg: GestureConfig) -> None:
        self.pose, self._cfg = pose, cfg
        self.reps = cfg.sequence_reps
        self.window_s = cfg.sequence_window_s
        self._need = 1
        self.reset()

    def configure(self, state, fps):
        self.window_s = (self._cfg.sequence_window_standby_s
                         if state == SystemState.DEEP_STANDBY
                         else self._cfg.sequence_window_s)
        self._need = max(1, math.ceil(self._cfg.sequence_min_hold_s * fps))

    def reset(self) -> None:
        self._in_pose = False
        self._match_run = 0
        self._miss_run = 0
        self._entries: Deque[float] = deque()

    def update(self, rec, buf):
        if rec.pose == self.pose:
            self._match_run += 1
            self._miss_run = 0
            if not self._in_pose and self._match_run >= self._need:
                self._in_pose = True
                self._entries.append(rec.t)
        else:
            self._miss_run += 1
            self._match_run = 0
            if self._in_pose and self._miss_run >= self._need:
                self._in_pose = False

        while self._entries and rec.t - self._entries[0] > self.window_s:
            self._entries.popleft()
        if len(self._entries) >= self.reps:
            self._entries.clear()
            return GestureEvent(GestureKind.SEQUENCE, self.pose, rec.t, count=self.reps)
        return None


# --------------------------------------------------------------------------- #
# Façade
# --------------------------------------------------------------------------- #
class GestureClassifier:
    """Feeds observations through the buffer and every registered detector."""

    def __init__(self, gesture_cfg: GestureConfig, pose_cfg: PoseConfig,
                 extra_detectors: Iterable[Detector] = ()) -> None:
        self._pose_cfg = pose_cfg
        self.buffer = TemporalBuffer(gesture_cfg.buffer_len)
        self.detectors: List[Detector] = [
            SwipeDetector(gesture_cfg),
            StaticHoldDetector(Pose.OPEN_PALM, gesture_cfg.wake_hold_s,
                               gesture_cfg.wake_min_pose_share, gesture_cfg.wake_max_drift),
            RepetitionDetector(Pose.THUMB_DOWN, gesture_cfg),
            RepetitionDetector(Pose.THUMB_UP, gesture_cfg),
            *extra_detectors,
        ]

    def configure(self, state: SystemState, fps: float) -> None:
        for d in self.detectors:
            d.configure(state, fps)

    def reset(self) -> None:
        self.buffer.clear()
        for d in self.detectors:
            d.reset()

    def update(self, obs: HandObservation) -> List[GestureEvent]:
        rec = FrameRecord(obs, classify_pose(obs.landmarks, self._pose_cfg))
        self.buffer.append(rec)
        events = []
        for d in self.detectors:
            ev = d.update(rec, self.buffer)
            if ev is not None:
                events.append(ev)
        return events

    @property
    def last_pose(self) -> Optional[Pose]:
        rec = self.buffer.latest()
        return rec.pose if rec else None
