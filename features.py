"""
Feature vector for the static ASL letter classifier.

This module is the **single source of truth** for features: the offline
extractor (tools/extract_landmarks.py) and the live app (asl_classifier.py)
both call ``asl_features``. The options used at training time are stored
inside the model bundle and replayed at runtime, so training and inference
can never silently disagree.

Layout (``use_z=True``, the default -> 73 values)
-----------------------------------------------
* 21 landmarks x (x, y, z) after the tracker's normalisation:
  wrist at the origin, scaled by the wrist -> middle-MCP length, so the
  vector does not depend on where the hand is or how far away it is.
* 10 pairwise fingertip distances (thumb, index, middle, ring, pinky). These
  help with E / M / N / S / T, which differ mainly in where the thumb tip sits.

Rotation is deliberately **not** normalised: G/Q, H/U and K/P are largely the
same hand shape pointed in different directions.

Left hands are mirrored to a right hand (x -> -x) when ``mirror_left`` is set,
so one model serves both hands. MediaPipe's handedness label assumes a
mirrored (selfie) image, which is what the live camera produces and what the
extractor feeds it.
"""
from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

from tracker import NUM_LANDMARKS, HandObservation

FEATURE_VERSION = 1
FINGERTIPS = (4, 8, 12, 16, 20)
DEFAULT_OPTIONS = {"mirror_left": True, "use_z": True}

_PAIRS = np.triu_indices(len(FINGERTIPS), 1)          # 10 unordered pairs


def feature_names(options: Optional[Mapping] = None) -> list:
    opts = {**DEFAULT_OPTIONS, **(options or {})}
    axes = ("x", "y", "z") if opts["use_z"] else ("x", "y")
    names = [f"lm{i}_{a}" for i in range(NUM_LANDMARKS) for a in axes]
    names += [f"tip{FINGERTIPS[i]}_tip{FINGERTIPS[j]}" for i, j in zip(*_PAIRS)]
    return names


def feature_size(options: Optional[Mapping] = None) -> int:
    return len(feature_names(options))


def asl_features_from_landmarks(landmarks: np.ndarray, handedness: Optional[str],
                                options: Optional[Mapping] = None) -> np.ndarray:
    """``landmarks``: (21, 3) wrist-origin, palm-scaled (``tracker.normalize_landmarks``)."""
    opts = {**DEFAULT_OPTIONS, **(options or {})}
    lm = np.asarray(landmarks, dtype=np.float32).copy()
    if lm.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"expected (21, 3) landmarks, got {lm.shape}")
    if opts["mirror_left"] and handedness == "Left":
        lm[:, 0] *= -1.0                                # canonical right hand
    if not opts["use_z"]:
        lm = lm[:, :2]
    tips = lm[list(FINGERTIPS)]
    dists = np.linalg.norm(tips[:, None, :] - tips[None, :, :], axis=-1)[_PAIRS]
    return np.concatenate([lm.ravel(), dists]).astype(np.float32)


def asl_features(obs: HandObservation, options: Optional[Mapping] = None) -> np.ndarray:
    if not obs.present:
        raise ValueError("no hand in observation")
    return asl_features_from_landmarks(obs.landmarks, obs.handedness, options)
