"""tools/record_samples.py: session logic, overlay, saving, and the full
record -> train -> load loop with a fake camera, tracker, window and clock."""
import json

import cv2
import numpy as np
import pytest

from asl_classifier import AslClassifier
from conftest import observation
from features import FEATURE_VERSION, asl_features
from tools import record_samples as rs
from tools.record_samples import Phase, RecordingSession, render, save_samples

DT = 1.0 / 30


def drive(session, poses, t=0.0, handedness="Right"):
    """Feed one observation per pose (None = no hand) at 30 FPS."""
    rng = np.random.default_rng(len(poses))
    for p in poses:
        t += DT
        session.step(t, observation(p, t, jitter=0.01, rng=rng, handedness=handedness))
    return t


# ------------------------------------------------------------- session ---- #
def test_countdown_then_records_target_frames():
    s = RecordingSession("D", target=10, countdown_s=1.0, stride=1)
    s.start(0.0)
    t = drive(s, ["POINT"] * 29)                    # < 1 s: still counting down
    assert s.phase == Phase.COUNTDOWN and s.count == 0
    t = drive(s, ["POINT"] * 3, t)
    assert s.phase == Phase.RECORDING
    drive(s, ["POINT"] * 10, t)
    assert s.phase == Phase.LETTER_DONE and len(s.samples["D"]) == 10


def test_frames_without_a_hand_are_skipped_and_flagged():
    s = RecordingSession("D", target=5, countdown_s=0.0, stride=1)
    s.start(0.0)
    t = drive(s, ["POINT"])                          # countdown -> recording
    t = drive(s, [None, None], t)
    assert s.no_hand and s.count == 0
    drive(s, ["POINT", None, "POINT", "POINT", "POINT", "POINT"], t)
    assert s.count == 5 and not s.no_hand


def test_stride_keeps_every_nth_valid_frame():
    s = RecordingSession("D", target=5, countdown_s=0.0, stride=3)
    s.start(0.0)
    drive(s, ["POINT"] * 20)
    assert s.count == 5


def test_steps_through_letters_and_finishes():
    s = RecordingSession("DV", target=4, countdown_s=0.1, done_pause_s=0.1, stride=1)
    s.start(0.0)
    t = 0.0
    while s.letter == "D":
        t = drive(s, ["POINT"], t)
    assert s.letter == "V" and s.phase == Phase.COUNTDOWN and len(s.samples["D"]) == 4
    drive(s, ["PEACE"] * 15, t)
    assert s.finished and s.recorded_letters() == ["D", "V"]


def test_left_hand_samples_match_right_hand_samples():
    right = RecordingSession("D", target=1, countdown_s=0.0, stride=1)
    left = RecordingSession("D", target=1, countdown_s=0.0, stride=1)
    for s, hand in ((right, "Right"), (left, "Left")):
        s.start(0.0)
        s.step(0.1, observation("POINT", 0.1, handedness=hand))   # enter recording
    # A real left hand is the mirror image; asl_features mirrors it back.
    from conftest import HANDS
    from tracker import HandObservation, normalize_landmarks
    shape = HANDS["POINT"].copy()
    shape[:, 0] *= -1
    px = (shape * 80).astype(np.float32)
    px[:, :2] += (320, 200)
    sc, scale = normalize_landmarks(px)
    left.step(0.2, HandObservation(0.2, sc, px[0, :2], scale, "Left"))
    right.step(0.2, observation("POINT", 0.2))
    assert np.allclose(left.samples["D"][0], right.samples["D"][0], atol=1e-5)


def test_redo_back_skip_and_partial_discard():
    s = RecordingSession("DVW", target=4, countdown_s=0.0, done_pause_s=0.0, stride=1)
    s.start(0.0)
    t = drive(s, ["POINT"] * 7)                     # D done, moved on to V
    assert s.letter == "V"
    t = drive(s, ["PEACE"] * 3, t)
    s.redo(t)
    assert s.letter == "V" and s.count == 0 and s.phase == Phase.COUNTDOWN
    s.back(t)
    assert s.letter == "D" and s.samples["D"] == []
    s.skip(t)
    assert s.letter == "V"
    t = drive(s, ["PEACE"] * 3, t)                  # partial V ...
    s.finish()                                      # ... discarded on quit
    assert s.finished and s.recorded_letters() == []


def test_dataset_layout():
    s = RecordingSession("DV", target=10, countdown_s=0.0, done_pause_s=0.0, stride=1)
    s.start(0.0)
    t = drive(s, ["POINT"] * 12)
    drive(s, ["PEACE"] * 12, t)
    d = s.dataset()
    assert d["X"].shape == (20, 73) and d["X"].dtype == np.float32
    assert list(d["y"]) == ["D"] * 10 + ["V"] * 10
    assert set(d["group"]) == {"user"}
    assert sorted(set(d["segment"].tolist())) == [0, 1, 2, 3, 4]
    assert np.allclose(d["X"][0], asl_features(observation("POINT", 0.0)), atol=0.1)


# ------------------------------------------------------------- overlay ---- #
@pytest.mark.parametrize("letter", ["A", "J"])
def test_overlay_draws_prompt_skeleton_and_motion_note(letter):
    frame = np.full((360, 640, 3), 120, np.uint8)
    s = RecordingSession(letter, countdown_s=3.0)
    s.start(0.0)
    obs = observation("POINT", 0.5)
    img = render(frame, s, obs, 0.5)
    assert img.shape == frame.shape and not np.array_equal(img, frame)
    assert np.array_equal(frame, np.full((360, 640, 3), 120, np.uint8))    # not mutated
    note = render(frame, s, None, 0.5)[58:84, 12:400]
    plain_note = render(frame, RecordingSession("A"), None, 0.5)[58:84, 12:400]
    assert (letter == "J") == (not np.array_equal(note, plain_note))


def test_overlay_alerts_when_no_hand():
    frame = np.full((360, 640, 3), 120, np.uint8)
    s = RecordingSession("D", countdown_s=0.0)
    s.start(0.0)
    s.step(0.1, observation("POINT", 0.1))
    s.step(0.2, observation(None, 0.2))
    img = render(frame, s, None, 0.2)
    red = (img[:, :, 2] > 200) & (img[:, :, 1] < 120)
    assert red[150:220].any()                       # red alert text mid-screen


# --------------------------------------------------------------- saving --- #
def _data(letters, n=3, group="user"):
    X = np.random.default_rng(0).normal(size=(n * len(letters), 73)).astype(np.float32)
    y = np.repeat(list(letters), n)
    return {"X": X, "y": y, "group": np.full(len(y), group), "segment": np.zeros(len(y), int)}


def test_save_new_backup_and_append(tmp_path):
    from features import DEFAULT_OPTIONS
    out = tmp_path / "user_samples.npz"
    save_samples(_data("AB"), out, dict(DEFAULT_OPTIONS))
    with np.load(out) as d:
        assert int(d["feature_version"]) == FEATURE_VERSION
        assert json.loads(str(d["feature_options"])) == dict(DEFAULT_OPTIONS)
        assert sorted(set(d["y"])) == ["A", "B"]

    save_samples(_data("BC", n=5), out, dict(DEFAULT_OPTIONS), append=True)
    with np.load(out) as d:
        counts = {c: int((d["y"] == c).sum()) for c in set(d["y"])}
    assert counts == {"A": 3, "B": 5, "C": 5}        # B replaced, A kept, C added

    save_samples(_data("Z"), out, dict(DEFAULT_OPTIONS))     # overwrite -> backup
    assert (tmp_path / "user_samples.bak.npz").exists()


def test_append_refuses_different_features(tmp_path):
    out = tmp_path / "u.npz"
    save_samples(_data("A"), out, {"mirror_left": True, "use_z": True})
    with pytest.raises(SystemExit):
        save_samples(_data("B"), out, {"mirror_left": True, "use_z": False}, append=True)


# ------------------------------------------- record -> train -> load ------ #
class FakeCamera:
    def __init__(self):
        self.released = False

    def read(self):
        return True, np.full((480, 640, 3), 90, np.uint8)     # resized like the daemon

    def release(self):
        self.released = True


class FakeGUI:
    def __init__(self, monkeypatch, keys=()):
        self.keys = list(keys)
        self.frames = 0
        for name in ("namedWindow", "imshow", "waitKey", "getWindowProperty", "destroyWindow"):
            monkeypatch.setattr(cv2, name, getattr(self, name))
        import hud
        monkeypatch.setattr(hud, "display_available", lambda: True)

    def namedWindow(self, *a):
        pass

    def imshow(self, name, img):
        assert img.shape == (360, 640, 3)
        self.frames += 1

    def waitKey(self, d=0):
        return self.keys.pop(0) if self.keys else -1

    def getWindowProperty(self, *a):
        return 1.0

    def destroyWindow(self, *a):
        pass


LETTER_POSE = {"D": "POINT", "V": "PEACE", "W": "THREE", "S": "FIST", "B": "FOUR"}


@pytest.fixture
def scripted(monkeypatch):
    """Fake tracker that shows the right synthetic hand for the current letter."""
    sessions = []

    class Spy(RecordingSession):
        def __post_init__(self):
            super().__post_init__()
            sessions.append(self)

    monkeypatch.setattr(rs, "RecordingSession", Spy)

    class Tracker:
        calls = 0

        def process(self, img, ts):
            Tracker.calls += 1
            s = sessions[-1]
            if Tracker.calls % 7 == 0 or s.letter is None:
                return observation(None, ts)          # occasional dropped detection
            rng = np.random.default_rng(Tracker.calls)
            return observation(LETTER_POSE[s.letter], ts, jitter=0.03, rng=rng)

        def close(self):
            pass

    clock = {"t": 0.0}

    def fake_clock():
        clock["t"] += DT / 2                         # called ~twice per frame
        return clock["t"]

    return Tracker, fake_clock


def test_record_then_train_then_load(tmp_path, monkeypatch, scripted, capsys):
    Tracker, clock = scripted
    gui = FakeGUI(monkeypatch)
    cam = FakeCamera()
    out = tmp_path / "data" / "features" / "user_samples.npz"
    model = tmp_path / "models" / "asl_rf.pkl"
    rc = rs.main(["--letters", "DVWSB", "--frames", "40", "--countdown", "0.3",
                  "--out", str(out), "--train", "--model-out", str(model)],
                 camera=cam, tracker=Tracker(), clock=clock)
    assert rc == 0 and cam.released and gui.frames > 0
    with np.load(out) as d:
        assert d["X"].shape == (200, 73)
        assert sorted(set(d["y"])) == list("BDSVW")
        assert set(d["group"]) == {"user"}

    clf = AslClassifier.load(model)
    assert clf.meta["feature_version"] == FEATURE_VERSION
    assert clf.meta["sklearn_version"]
    assert clf.classes == tuple("BDSVW")
    assert "time block" in clf.meta["cv"]["scheme"]
    probs = clf.predict_proba(observation("THREE", 0.0))
    assert clf.classes[int(np.argmax(probs))] == "W"

    printed = capsys.readouterr().out
    assert "python main.py" in printed and "record_samples.py" in printed


def test_quit_early_saves_completed_letters_only(tmp_path, monkeypatch, scripted):
    Tracker, clock = scripted
    # ~0.3 s countdown + 20 frames for D + pause, then Q midway through V
    FakeGUI(monkeypatch, keys=[-1] * 70 + [ord("q")])
    out = tmp_path / "u.npz"
    rc = rs.main(["--letters", "DV", "--frames", "20", "--countdown", "0.3", "--stride", "1",
                  "--out", str(out)], camera=FakeCamera(), tracker=Tracker(), clock=clock)
    assert rc == 0
    with np.load(out) as d:
        assert set(d["y"]) == {"D"}


def test_train_asl_points_to_recorder_when_samples_missing(tmp_path, capsys):
    from tools import train_asl
    rc = train_asl.main([str(tmp_path / "user_samples.npz"), "--out", str(tmp_path / "m.pkl")])
    assert rc == 2 and "record_samples.py" in capsys.readouterr().err
