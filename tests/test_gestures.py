import dataclasses

import numpy as np

from conftest import observation
from config import SystemState
from gestures import Direction, GestureClassifier, GestureKind, Pose

FPS = 30.0
DT = 1.0 / FPS


class Script:
    """Drives a classifier with a timeline of synthetic observations."""

    def __init__(self, cfg, state=SystemState.ACTIVE, fps=FPS, palm=80.0):
        self.clf = GestureClassifier(cfg.gesture, cfg.pose)
        self.clf.configure(state, fps)
        self.dt = 1.0 / fps
        self.t = 0.0
        self.pos = np.array([320.0, 200.0])
        self.palm = palm
        self.events = []

    def hold(self, pose, seconds, jitter=0.0, rng=None):
        for _ in range(int(round(seconds / self.dt))):
            wrist = self.pos + (rng.normal(0, jitter, 2) if jitter else 0)
            self._feed(pose, wrist)

    def move(self, pose, dx_palms, dy_palms, seconds):
        n = int(round(seconds / self.dt))
        step = np.array([dx_palms, dy_palms]) * self.palm / n
        for _ in range(n):
            self.pos = self.pos + step
            self._feed(pose, self.pos)

    def absent(self, seconds):
        for _ in range(int(round(seconds / self.dt))):
            self._feed(None, self.pos)

    def _feed(self, pose, wrist):
        self.t += self.dt
        self.events += self.clf.update(observation(pose, self.t, wrist, self.palm))

    def kinds(self, kind):
        return [e for e in self.events if e.kind == kind]


# ------------------------------------------------------------------ swipes -- #
def test_single_swipe_right_with_point(cfg):
    s = Script(cfg)
    s.hold("POINT", 0.3)
    s.move("POINT", 2.0, 0.1, 0.25)
    s.hold("POINT", 0.3)
    swipes = s.kinds(GestureKind.SWIPE)
    assert [(e.direction, e.pose) for e in swipes] == [(Direction.RIGHT, Pose.POINT)]


def test_all_four_directions(cfg):
    for dx, dy, want in [(-2, 0, Direction.LEFT), (2, 0, Direction.RIGHT),
                         (0, -2, Direction.UP), (0, 2, Direction.DOWN)]:
        s = Script(cfg)
        s.hold("PEACE", 0.3)
        s.move("PEACE", dx, dy, 0.25)
        s.hold("PEACE", 0.3)
        assert [e.direction for e in s.kinds(GestureKind.SWIPE)] == [want]


def test_swipe_is_distance_invariant(cfg):
    for palm in (30.0, 160.0):                 # far from / close to the lens
        s = Script(cfg, palm=palm)
        s.hold("THREE", 0.3)
        s.move("THREE", 1.8, 0.0, 0.25)
        s.hold("THREE", 0.3)
        assert len(s.kinds(GestureKind.SWIPE)) == 1


def test_return_stroke_is_suppressed_then_next_stroke_allowed(cfg):
    s = Script(cfg)
    s.hold("POINT", 0.3)
    s.move("POINT", 2.0, 0, 0.25)      # 't'
    s.hold("POINT", 0.1)
    s.move("POINT", -2.0, 0, 0.25)     # natural return -> must be swallowed
    s.hold("POINT", 0.6)
    s.move("POINT", -2.0, 0, 0.25)     # deliberate LEFT ('e')
    s.hold("POINT", 0.3)
    assert [e.direction for e in s.kinds(GestureKind.SWIPE)] == [Direction.RIGHT, Direction.LEFT]


def test_fist_clutch_emits_unmapped_stroke_and_cancels_suppression(cfg):
    s = Script(cfg)
    s.hold("POINT", 0.3)
    s.move("POINT", 2.0, 0, 0.25)
    s.hold("FIST", 1.0)                 # a slow return would otherwise type 'e'
    s.move("FIST", -2.0, 0, 0.25)       # reposition with a fist
    s.hold("POINT", 0.4)
    s.move("POINT", 2.0, 0, 0.25)
    s.hold("POINT", 0.3)
    poses = [(e.pose, e.direction) for e in s.kinds(GestureKind.SWIPE)]
    assert poses == [(Pose.POINT, Direction.RIGHT), (Pose.FIST, Direction.LEFT),
                     (Pose.POINT, Direction.RIGHT)]


def test_slow_drift_and_short_jitter_are_not_swipes(cfg):
    s = Script(cfg)
    s.hold("POINT", 0.3)
    s.move("POINT", 1.5, 0, 3.0)        # 0.5 palm/s drift
    s.hold("POINT", 1.0, jitter=3.0, rng=np.random.default_rng(0))
    assert s.kinds(GestureKind.SWIPE) == []


def test_diagonal_stroke_rejected(cfg):
    s = Script(cfg)
    s.hold("POINT", 0.3)
    s.move("POINT", 1.6, 1.6, 0.25)
    s.hold("POINT", 0.3)
    assert s.kinds(GestureKind.SWIPE) == []


# ------------------------------------------------------------- wake (hold) -- #
def test_open_palm_hold_fires_once_at_idle_rate(cfg):
    s = Script(cfg, SystemState.IDLE, fps=5.0)
    s.hold("OPEN_PALM", 1.6, jitter=2.0, rng=np.random.default_rng(1))
    assert s.kinds(GestureKind.HOLD) == []
    s.hold("OPEN_PALM", 2.0, jitter=2.0, rng=np.random.default_rng(2))
    holds = s.kinds(GestureKind.HOLD)
    assert len(holds) == 1 and holds[0].pose == Pose.OPEN_PALM


def test_moving_palm_is_not_a_wake(cfg):
    s = Script(cfg, SystemState.IDLE, fps=5.0)
    s.move("OPEN_PALM", 3.0, 0, 2.5)
    assert s.kinds(GestureKind.HOLD) == []


def test_interrupted_palm_is_not_a_wake(cfg):
    s = Script(cfg, SystemState.IDLE, fps=5.0)
    for _ in range(4):
        s.hold("OPEN_PALM", 0.6)
        s.hold("FIST", 0.4)
    assert s.kinds(GestureKind.HOLD) == []


def test_buffer_is_fixed_length(cfg):
    s = Script(cfg)
    s.hold("POINT", 5.0)
    assert len(s.clf.buffer) == cfg.gesture.buffer_len == 45


# ------------------------------------------------------ "Off x3" / "On x3" -- #
def test_thumb_down_three_times_at_30fps(cfg):
    s = Script(cfg)
    for _ in range(3):
        s.hold("THUMB_DOWN", 0.4)
        s.hold("FIST", 0.4)
    seq = s.kinds(GestureKind.SEQUENCE)
    assert [(e.pose, e.count) for e in seq] == [(Pose.THUMB_DOWN, 3)]


def test_two_repetitions_are_not_enough(cfg):
    s = Script(cfg)
    for _ in range(2):
        s.hold("THUMB_DOWN", 0.4)
        s.hold("FIST", 0.4)
    s.hold("FIST", 1.0)
    assert s.kinds(GestureKind.SEQUENCE) == []


def test_single_glitch_frame_does_not_double_count(cfg):
    s = Script(cfg)
    s.hold("THUMB_DOWN", 0.3)
    s.hold("FIST", DT)                                   # one-frame dropout
    s.hold("THUMB_DOWN", 0.3)
    s.hold("FIST", 0.4)
    s.hold("THUMB_DOWN", 0.3)
    s.hold("FIST", 0.4)
    assert s.kinds(GestureKind.SEQUENCE) == []          # only 2 real entries


def test_repetitions_spread_beyond_window_do_not_fire(cfg):
    s = Script(cfg)
    for _ in range(3):
        s.hold("THUMB_DOWN", 0.3)
        s.hold("FIST", 3.0)
    assert s.kinds(GestureKind.SEQUENCE) == []


def test_thumb_up_three_times_at_1fps_deep_standby(cfg):
    s = Script(cfg, SystemState.DEEP_STANDBY, fps=1.0)
    for _ in range(3):
        s.hold("THUMB_UP", 1.0)
        s.absent(1.0)                    # hand lowered between reps
    seq = s.kinds(GestureKind.SEQUENCE)
    assert [(e.pose, e.count) for e in seq] == [(Pose.THUMB_UP, 3)]


def test_custom_detector_extension_point(cfg):
    from gestures import Detector, GestureEvent

    class PinchDetector(Detector):
        def update(self, rec, buf):
            if rec.pose == Pose.PINCH:
                return GestureEvent(GestureKind.HOLD, Pose.PINCH, rec.t)
            return None

    clf = GestureClassifier(cfg.gesture, cfg.pose, extra_detectors=[PinchDetector()])
    events = clf.update(observation("PINCH", 1.0))
    assert any(e.pose == Pose.PINCH for e in events)


def test_configure_switches_sequence_window(cfg):
    g = dataclasses.replace(cfg.gesture, sequence_window_s=1.0, sequence_window_standby_s=9.0)
    clf = GestureClassifier(g, cfg.pose)
    rep = clf.detectors[2]
    clf.configure(SystemState.DEEP_STANDBY, 1.0)
    assert rep.window_s == 9.0
    clf.configure(SystemState.ACTIVE, 30.0)
    assert rep.window_s == 1.0
