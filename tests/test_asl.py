"""Static ASL pipeline: features, model bundle, LetterDetector, offline tools."""
import dataclasses

import cv2
import joblib
import numpy as np
import pytest

from asl_classifier import AslClassifier, AslModelError
from config import DEFAULT_CONFIG, SystemState
from conftest import HANDS, SYNTH_LETTERS, observation, synthetic_dataset
from features import FEATURE_VERSION, asl_features, feature_names, feature_size
from gestures import GestureClassifier, GestureKind
from tracker import HandObservation, HandTracker, normalize_landmarks

ASL = DEFAULT_CONFIG.asl


# ---------------------------------------------------------------- features -- #
def test_feature_layout():
    assert feature_size() == 21 * 3 + 10 == len(feature_names())
    assert feature_size({"use_z": False}) == 21 * 2 + 10
    assert asl_features(observation("POINT", 0.0)).shape == (73,)


def test_features_ignore_position_and_distance():
    a = asl_features(observation("PEACE", 0.0, wrist=(100, 300), palm=40))
    b = asl_features(observation("PEACE", 0.0, wrist=(500, 120), palm=150))
    assert np.allclose(a, b, atol=1e-5)


def test_left_hand_is_mirrored_to_right():
    right = observation("THREE", 0.0)
    shape = HANDS["THREE"].copy()
    shape[:, 0] *= -1                                   # physical mirror image
    px = (shape * 80.0).astype(np.float32)
    px[:, :2] += (320, 200)
    scaled, scale = normalize_landmarks(px)
    left = HandObservation(0.0, scaled, px[0, :2], scale, "Left")
    assert np.allclose(asl_features(left), asl_features(right), atol=1e-5)
    assert not np.allclose(asl_features(left, {"mirror_left": False}), asl_features(right))


class _RawBackend:
    """Returns fixed MediaPipe-style normalised landmarks."""

    def __init__(self, raw):
        self.raw = raw

    def infer(self, rgb, ts):
        return self.raw, "Right"

    def close(self):
        pass


def test_dataset_photo_and_live_frame_give_identical_features():
    """Parity: the same physical hand in a 200x200 dataset photo and a 640x360
    live frame must produce the same feature vector."""
    px = (HANDS["PINCH"] * 50.0).astype(np.float32)
    feats = []
    for (w, h), offset in (((200, 200), (100, 150)), ((640, 360), (320, 260))):
        p = px.copy()
        p[:, :2] += offset
        raw = p / np.array([w, h, w], np.float32)          # what MediaPipe reports
        tracker = HandTracker(DEFAULT_CONFIG.tracker, 640, 360, backend=_RawBackend(raw))
        obs = tracker.process(np.zeros((h, w, 3), np.uint8), 1.0)
        feats.append(asl_features(obs))
    assert np.allclose(feats[0], feats[1], atol=1e-4)


# ------------------------------------------------------------- classifier -- #
def test_model_bundle_roundtrip(tmp_path, asl_bundle):
    path = tmp_path / "asl.pkl"
    joblib.dump(asl_bundle, path, compress=3)
    clf = AslClassifier.load(path)
    assert clf.classes == tuple(sorted(SYNTH_LETTERS.values()))
    assert clf.model.n_jobs == 1
    probs = clf.predict_proba(observation("POINT", 0.0))
    assert probs.shape == (len(clf.classes),) and np.isclose(probs.sum(), 1.0)


def test_model_recognises_unseen_noisy_samples(asl_model):
    X, y, _ = synthetic_dataset(n_per_class=20, seed=99)
    pred = asl_model.model.predict(X)
    assert (pred == y).mean() > 0.9


@pytest.mark.parametrize("mutate, msg", [
    (lambda b: b.pop("threshold"), "missing"),
    (lambda b: b.update(feature_version=FEATURE_VERSION + 1), "feature_version"),
    (lambda b: b.update(classes=["X"] * len(b["classes"])), "classes"),
    (lambda b: b.update(feature_options={"use_z": False}), "features"),
])
def test_bad_bundles_are_rejected(asl_bundle, mutate, msg):
    b = dict(asl_bundle)
    mutate(b)
    with pytest.raises(AslModelError, match=msg):
        AslClassifier(b)


def test_missing_model_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        AslClassifier.load(tmp_path / "nope.pkl")


# ---------------------------------------------------------- LetterDetector -- #
class Driver:
    def __init__(self, model, asl_cfg=ASL, state=SystemState.ACTIVE, fps=30.0):
        self.clf = GestureClassifier(DEFAULT_CONFIG.gesture, DEFAULT_CONFIG.pose,
                                     letter_model=model, asl_cfg=asl_cfg)
        self.clf.configure(state, fps)
        self.dt, self.t = 1.0 / fps, 0.0
        self.pos = np.array([320.0, 200.0])
        self.rng = np.random.default_rng(3)
        self.letters, self.progress = [], []

    def run(self, pose, seconds, speed=0.0):
        for _ in range(int(round(seconds / self.dt))):
            self.t += self.dt
            self.pos = self.pos + np.array([speed * 80.0 * self.dt, 0.0])
            obs = observation(pose, self.t, wrist=tuple(self.pos), jitter=0.01, rng=self.rng)
            for ev in self.clf.update(obs):
                if ev.kind == GestureKind.LETTER:
                    self.letters.append(ev.label)
            if self.clf.letters is not None:
                self.progress.append(self.clf.letters.progress)
        return self


def test_letter_commits_after_dwell_not_before(asl_model):
    d = Driver(asl_model).run("POINT", 0.35)
    assert d.letters == []
    d.run("POINT", 0.15)
    assert d.letters == ["D"]


def test_holding_does_not_repeat(asl_model):
    assert Driver(asl_model).run("POINT", 3.0).letters == ["D"]


def test_double_letter_after_release(asl_model):
    d = Driver(asl_model).run("POINT", 0.5).run(None, 0.25).run("POINT", 0.5)
    assert d.letters == ["D", "D"]


def test_release_shorter_than_lockout_does_not_repeat(asl_model):
    d = Driver(asl_model).run("POINT", 0.5).run(None, 0.1).run("POINT", 1.0)
    assert d.letters == ["D"]


def test_switching_letters_needs_no_release(asl_model):
    d = Driver(asl_model).run("POINT", 0.5).run("PEACE", 0.5).run("THREE", 0.5)
    assert d.letters == ["D", "V", "W"]


def test_moving_hand_types_nothing(asl_model):
    assert Driver(asl_model).run("POINT", 1.5, speed=3.0).letters == []


def test_control_poses_are_never_letters(asl_model):
    d = Driver(asl_model).run("OPEN_PALM", 1.0).run("THUMB_DOWN", 1.0)
    assert d.letters == [] and d.clf.letters.top == ()


def test_fist_between_thumb_downs_is_not_typed(asl_model):
    """'Off, Off, Off' relaxes into a fist (= S here) between reps."""
    d = Driver(asl_model)
    for _ in range(2):
        d.run("THUMB_DOWN", 0.4).run("FIST", 0.6)
    assert d.letters == []
    d.run("FIST", 1.5)                              # cooldown over -> S is a letter again
    assert d.letters == ["S"]


def test_low_confidence_types_nothing(asl_model):
    strict = dataclasses.replace(ASL, min_confidence=1.01)
    assert Driver(asl_model, strict).run("POINT", 1.0).letters == []


def test_letters_only_in_active(asl_model):
    calls = asl_model.calls
    d = Driver(asl_model, state=SystemState.IDLE, fps=10.0).run("POINT", 2.0)
    assert d.letters == [] and asl_model.calls == calls


def test_prediction_rate_is_capped(asl_model):
    calls = asl_model.calls
    Driver(asl_model).run("POINT", 1.0)
    assert asl_model.calls - calls <= ASL.max_predict_hz + 1


def test_telemetry_progress_and_top3(asl_model):
    d = Driver(asl_model).run("POINT", 0.3)
    top = d.clf.letters.top
    assert len(top) == 3 and top[0][0] == "D" and top[0][1] >= top[1][1] >= top[2][1]
    assert 0.5 < d.progress[-1] < 1.0 and d.clf.letters.candidate == "D"
    d.run("POINT", 0.2)
    assert d.clf.letters.locked == "D" and d.clf.letters.progress == 0.0


# ------------------------------------------------------------------ tools -- #
class ColourTracker:
    """Fake MediaPipe: the image's colour encodes which synthetic hand it shows."""

    POSES = list(SYNTH_LETTERS)

    def process(self, img, ts):
        v = int(img[0, 0, 0])
        if v >= len(self.POSES):
            return HandObservation(ts)                 # "no hand detected"
        return observation(self.POSES[v], ts, jitter=0.03, rng=np.random.default_rng(v + int(ts)))

    def close(self):
        pass


def _write_dataset(root, signers=("s1", "s2"), per_class=12):
    for s in signers:
        for i, pose in enumerate(ColourTracker.POSES):
            folder = root / s / SYNTH_LETTERS[pose]
            folder.mkdir(parents=True, exist_ok=True)
            for k in range(per_class):
                cv2.imwrite(str(folder / f"{k}.png"), np.full((40, 40, 3), i, np.uint8))
        nohand = root / s / "A"
        cv2.imwrite(str(nohand / "blank.png"), np.full((40, 40, 3), 250, np.uint8))
        (root / s / "space").mkdir(exist_ok=True)
        cv2.imwrite(str(root / s / "space" / "x.png"), np.zeros((40, 40, 3), np.uint8))


def test_extract_landmarks_end_to_end(tmp_path):
    from tools import extract_landmarks as ex

    _write_dataset(tmp_path / "raw")
    out = tmp_path / "f.npz"
    rc = ex.main([str(tmp_path / "raw"), "--out", str(out), "--signer-level", "0"],
                 tracker=ColourTracker())
    assert rc == 0
    d = np.load(out)
    assert d["X"].shape == (2 * 7 * 12, 73)
    assert set(d["y"]) == set(SYNTH_LETTERS.values())          # 'space' skipped
    assert set(d["group"]) == {"s1", "s2"}
    assert int(d["feature_version"]) == FEATURE_VERSION


def test_max_per_class_samples_randomly(tmp_path):
    from tools.extract_landmarks import find_images, label_from_path, sample_per_class

    _write_dataset(tmp_path / "raw", signers=("s1",))
    paths = [p for p in find_images(tmp_path / "raw") if label_from_path(p)]
    picked = sample_per_class(paths, 5, seed=1)
    counts = {}
    for p in picked:
        counts[label_from_path(p)] = counts.get(label_from_path(p), 0) + 1
    assert max(counts.values()) == 5


def test_train_asl_end_to_end(tmp_path):
    from tools import extract_landmarks as ex
    from tools import train_asl

    _write_dataset(tmp_path / "raw")
    feats = tmp_path / "f.npz"
    ex.main([str(tmp_path / "raw"), "--out", str(feats), "--signer-level", "0"],
            tracker=ColourTracker())
    model_path = tmp_path / "models" / "asl_rf.pkl"
    rc = train_asl.main([str(feats), "--out", str(model_path), "--trees", "20",
                         "--test", str(feats)])
    assert rc == 0
    clf = AslClassifier.load(model_path)
    assert 0.0 <= clf.threshold < 1.0
    assert "GroupKFold" in clf.meta["cv"]["scheme"]
    assert clf.meta["cv"]["accuracy"] > 0.8
    assert model_path.with_suffix(".txt").exists()
    probs = clf.predict_proba(observation("PEACE", 0.0))
    assert clf.classes[int(np.argmax(probs))] == "V"


def test_train_warns_on_single_signer(tmp_path, capsys):
    from tools import extract_landmarks as ex
    from tools import train_asl

    _write_dataset(tmp_path / "raw", signers=("only",))
    feats = tmp_path / "f.npz"
    ex.main([str(tmp_path / "raw"), "--out", str(feats)], tracker=ColourTracker())
    train_asl.main([str(feats), "--out", str(tmp_path / "m.pkl"), "--trees", "10"])
    assert "OPTIMISTIC" in capsys.readouterr().out


def test_choose_threshold_hits_target_precision():
    from tools.train_asl import choose_threshold

    rng = np.random.default_rng(0)
    n = 2000
    y = rng.integers(0, 3, n)
    conf = rng.uniform(0.34, 1.0, n)
    correct = rng.uniform(size=n) < conf              # higher confidence = more often right
    pred = np.where(correct, y, (y + 1) % 3)
    proba = np.full((n, 3), 0.0)
    proba[np.arange(n), pred] = conf
    proba[np.arange(n), (pred + 1) % 3] = 1 - conf
    thr, prec, cov = choose_threshold(proba, y, 0.9)
    assert prec >= 0.9 and 0 < cov < 1 and 0.5 < thr < 1.0
