"""
MediaPipe hand tracking + landmark spatial normalisation.

Two interchangeable backends are provided because Google removed the legacy
``mp.solutions.hands`` API in mediapipe 0.10.30:

* ``SolutionsBackend`` - ``mp.solutions.hands.Hands(model_complexity=0)``,
  the Lite landmark model. Preferred; available with mediapipe <= 0.10.21
  (Python 3.10 - 3.12).
* ``TasksBackend``     - ``mediapipe.tasks.vision.HandLandmarker`` in VIDEO
  mode. Used automatically on newer mediapipe / Python 3.13+. The model
  bundle is downloaded once to ``models/``.

Both are created, used and closed on the inference thread only - MediaPipe
graphs are not thread-safe.
"""
from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from config import TrackerConfig

log = logging.getLogger(__name__)

WRIST, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 0, 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20
NUM_LANDMARKS = 21

_MIN_SCALE = 1e-6
_PROJECT_ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Normalisation (pure functions - unit tested without a camera)
# --------------------------------------------------------------------------- #
def to_pixel_space(norm_xyz: np.ndarray, width: int, height: int) -> np.ndarray:
    """MediaPipe x/y are normalised to width/height *independently*; bring
    everything to a common pixel metric (z shares x's scale per MediaPipe)."""
    out = np.asarray(norm_xyz, dtype=np.float32).copy()
    out[:, 0] *= width
    out[:, 1] *= height
    out[:, 2] *= width
    return out


def normalize_landmarks(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """Translate so the wrist (Node 0) is the origin and scale by the
    wrist -> middle-MCP (Node 9) Euclidean distance.

    Returns ``(scaled_points, scale)``; raises ``ValueError`` on a degenerate
    hand (scale ~ 0), which can happen on corrupted detections.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"expected (21, 3) landmarks, got {pts.shape}")
    translated = pts - pts[WRIST]
    scale = float(np.linalg.norm(translated[MIDDLE_MCP]))
    if scale < _MIN_SCALE:
        raise ValueError("degenerate hand: wrist and middle MCP coincide")
    return translated / scale, scale


@dataclass(frozen=True)
class HandObservation:
    timestamp: float
    landmarks: Optional[np.ndarray] = None   # (21, 3) wrist-origin, palm-scaled
    wrist_px: Optional[np.ndarray] = None    # (2,) wrist in pixels (trajectory)
    palm_px: float = 0.0                     # wrist->MCP9 length in pixels
    handedness: Optional[str] = None

    @property
    def present(self) -> bool:
        return self.landmarks is not None

    @staticmethod
    def from_raw(timestamp: float, norm_xyz: np.ndarray, width: int, height: int,
                 handedness: Optional[str] = None) -> "HandObservation":
        px = to_pixel_space(norm_xyz, width, height)
        try:
            scaled, scale = normalize_landmarks(px)
        except ValueError:
            return HandObservation(timestamp)
        return HandObservation(timestamp, scaled, px[WRIST, :2].copy(), scale, handedness)


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class _Backend:
    name = "base"

    def infer(self, rgb: np.ndarray, timestamp: float):
        """Return ((21,3) normalised landmarks, handedness) or (None, None)."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class SolutionsBackend(_Backend):
    name = "solutions"

    def __init__(self, cfg: TrackerConfig) -> None:
        import mediapipe as mp

        hands_mod = getattr(getattr(mp, "solutions", None), "hands", None)
        if hands_mod is None:
            raise ImportError("mediapipe.solutions.hands not available")
        self._hands = hands_mod.Hands(
            static_image_mode=False,
            max_num_hands=cfg.max_num_hands,
            model_complexity=cfg.model_complexity,     # 0 = Lite
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )

    def infer(self, rgb, timestamp):
        res = self._hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None, None
        lm = res.multi_hand_landmarks[0].landmark
        arr = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        label = None
        if res.multi_handedness:
            label = res.multi_handedness[0].classification[0].label
        return arr, label

    def close(self):
        self._hands.close()


class TasksBackend(_Backend):
    name = "tasks"

    def __init__(self, cfg: TrackerConfig) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        model_path = Path(cfg.task_model_path)
        if not model_path.is_absolute():
            model_path = _PROJECT_ROOT / model_path
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            log.info("Downloading hand landmarker model to %s", model_path)
            tmp = model_path.with_suffix(".part")
            urllib.request.urlretrieve(cfg.task_model_url, tmp)
            tmp.replace(model_path)

        self._mp = mp
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=cfg.max_num_hands,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_hand_presence_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(opts)
        self._last_ts_ms = -1

    def infer(self, rgb, timestamp):
        ts_ms = max(int(timestamp * 1000), self._last_ts_ms + 1)  # must increase
        self._last_ts_ms = ts_ms
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(rgb))
        res = self._landmarker.detect_for_video(image, ts_ms)
        if not res.hand_landmarks:
            return None, None
        arr = np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[0]], dtype=np.float32)
        label = res.handedness[0][0].category_name if res.handedness else None
        return arr, label

    def close(self):
        self._landmarker.close()


def create_backend(cfg: TrackerConfig) -> _Backend:
    order = {"solutions": [SolutionsBackend], "tasks": [TasksBackend]}.get(
        cfg.backend, [SolutionsBackend, TasksBackend])
    errors = []
    for cls in order:
        try:
            backend = cls(cfg)
            log.info("MediaPipe backend: %s", backend.name)
            return backend
        except Exception as exc:  # ImportError, AttributeError, download errors
            errors.append(f"{cls.__name__}: {exc}")
    raise RuntimeError("No usable MediaPipe backend:\n  " + "\n  ".join(errors))


class HandTracker:
    """Frame -> ``HandObservation``. Owned by the inference thread."""

    def __init__(self, cfg: TrackerConfig, width: int, height: int,
                 backend: Optional[_Backend] = None) -> None:
        self._w, self._h = width, height
        self._backend = backend or create_backend(cfg)

    def process(self, bgr: np.ndarray, timestamp: float) -> HandObservation:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False            # lets MediaPipe avoid a copy
        raw, label = self._backend.infer(rgb, timestamp)
        if raw is None:
            return HandObservation(timestamp)
        return HandObservation.from_raw(timestamp, raw, self._w, self._h, label)

    def close(self) -> None:
        try:
            self._backend.close()
        except Exception:  # pragma: no cover
            log.exception("Error closing MediaPipe backend")
