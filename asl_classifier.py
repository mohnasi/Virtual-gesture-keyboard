"""
Runtime wrapper around the Random Forest trained by ``tools/train_asl.py``.

The model file is a joblib bundle::

    {
      "model": RandomForestClassifier,      # fitted
      "classes": ["A", ..., "Z"],           # == model.classes_
      "threshold": 0.62,                    # calibrated on held-out signers
      "feature_version": 1,                 # features.FEATURE_VERSION
      "feature_options": {"mirror_left": True, "use_z": True},
      "sklearn_version": "1.x.y",
      ... training metadata (cv scores, sample counts, date)
    }

Security: joblib/pickle files can execute code when loaded. Only load a
model you trained yourself. The app never downloads one.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Union

import numpy as np

from features import FEATURE_VERSION, asl_features, feature_size
from tracker import HandObservation

log = logging.getLogger(__name__)

REQUIRED_KEYS = ("model", "classes", "threshold", "feature_version", "feature_options")
_PROJECT_ROOT = Path(__file__).resolve().parent


class AslModelError(RuntimeError):
    """The bundle is missing, malformed or built for different features."""


class AslClassifier:
    def __init__(self, bundle: Mapping[str, Any]) -> None:
        missing = [k for k in REQUIRED_KEYS if k not in bundle]
        if missing:
            raise AslModelError(f"model bundle is missing keys: {missing}")
        if bundle["feature_version"] != FEATURE_VERSION:
            raise AslModelError(
                f"model was trained on feature_version {bundle['feature_version']}, "
                f"this code produces version {FEATURE_VERSION}; re-run tools/train_asl.py")

        self.model = bundle["model"]
        self.classes: Tuple[str, ...] = tuple(str(c) for c in bundle["classes"])
        self.threshold = float(bundle["threshold"])
        self.options: Dict[str, Any] = dict(bundle["feature_options"])
        self.meta = {k: v for k, v in bundle.items() if k != "model"}

        model_classes = tuple(str(c) for c in getattr(self.model, "classes_", ()))
        if model_classes != self.classes:
            raise AslModelError("bundle 'classes' do not match model.classes_")
        expected = feature_size(self.options)
        n_in = getattr(self.model, "n_features_in_", expected)
        if n_in != expected:
            raise AslModelError(f"model expects {n_in} features, extractor makes {expected}")
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1          # one sample per call: thread pools only add latency
        self.calls = 0

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AslClassifier":
        import joblib

        p = Path(path)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        if not p.exists():
            raise FileNotFoundError(p)
        bundle = joblib.load(p)
        if not isinstance(bundle, Mapping):
            raise AslModelError(f"{p} is not a model bundle created by tools/train_asl.py")
        try:
            import sklearn
            trained = bundle.get("sklearn_version")
            if trained and trained != sklearn.__version__:
                log.warning("ASL model was trained with scikit-learn %s but %s is installed; "
                            "retrain if predictions look wrong", trained, sklearn.__version__)
        except ImportError:  # pragma: no cover - joblib.load would already have failed
            pass
        clf = cls(bundle)
        log.info("ASL model loaded: %d classes, threshold %.2f (%s)",
                 len(clf.classes), clf.threshold, p.name)
        return clf

    def predict_proba(self, obs: HandObservation) -> np.ndarray:
        """Class probabilities, aligned with ``self.classes``."""
        x = asl_features(obs, self.options)[None, :]
        self.calls += 1
        return np.asarray(self.model.predict_proba(x)[0], dtype=np.float64)
