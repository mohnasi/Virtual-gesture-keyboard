import dataclasses

import numpy as np

from conftest import HANDS, _rot, observation
from config import SystemState
from gestures import Direction, GestureClassifier, GestureKind, Pose
from keyboard_output import KeyMapper

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

    def hold(self, pose, seconds, jitter=0.0, rng=None, **kw):
        for _ in range(int(round(seconds / self.dt))):
            wrist = self.pos + (rng.normal(0, jitter, 2) if jitter else 0)
            self._feed(pose, wrist, **kw)

    def move(self, pose, dx_palms, dy_palms, seconds):
        n = int(round(seconds / self.dt))
        step = np.array([dx_palms, dy_palms]) * self.palm / n
        for _ in range(n):
            self.pos = self.pos + step
            self._feed(pose, self.pos)

    def absent(self, seconds):
        for _ in range(int(round(seconds / self.dt))):
            self._feed(None, self.pos)

    def _feed(self, pose, wrist, **kw):
        self.t += self.dt
        self.events += self.clf.update(observation(pose, self.t, wrist, self.palm, **kw))

    def flick(self, pose, deg0, deg1, seconds):
        """Turn the hand about a fixed wrist (fingertips sweep, wrist doesn't)."""
        n = int(round(seconds / self.dt))
        for i in range(1, n + 1):
            self._feed(pose, self.pos, degrees=deg0 + (deg1 - deg0) * i / n)

    def morph(self, a, b, seconds):
        """Blend hand shape a -> b in place (e.g. curl the fingers)."""
        n = int(round(seconds / self.dt))
        for i in range(1, n + 1):
            k = i / n
            self._feed(None, self.pos, shape=(1 - k) * HANDS[a] + k * HANDS[b])

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


# ------------------------------------------------------- fingertip swipes -- #
def _wrist_mode(cfg):
    return dataclasses.replace(
        cfg, gesture=dataclasses.replace(cfg.gesture, swipe_track_point="wrist"))


def test_wrist_flick_is_a_swipe_with_fingertip_tracking(cfg):
    for deg0, deg1, want in [(-20, 20, Direction.RIGHT), (20, -20, Direction.LEFT)]:
        s = Script(cfg)
        s.hold("OPEN_PALM", 0.3, degrees=deg0)
        s.flick("OPEN_PALM", deg0, deg1, 0.2)
        s.hold("OPEN_PALM", 0.3, degrees=deg1)
        swipes = s.kinds(GestureKind.SWIPE)
        assert [(e.direction, e.pose) for e in swipes] == [(want, Pose.OPEN_PALM)]


def test_wrist_flick_is_missed_in_wrist_mode(cfg):
    s = Script(_wrist_mode(cfg))
    s.hold("OPEN_PALM", 0.3, degrees=-20)
    s.flick("OPEN_PALM", -20, 20, 0.2)
    s.hold("OPEN_PALM", 0.3, degrees=20)
    assert s.kinds(GestureKind.SWIPE) == []


def test_wrist_mode_still_detects_arm_sweeps(cfg):
    s = Script(_wrist_mode(cfg))
    s.hold("OPEN_PALM", 0.3)
    s.move("OPEN_PALM", 2.0, 0.1, 0.25)
    s.hold("OPEN_PALM", 0.3)
    assert [e.direction for e in s.kinds(GestureKind.SWIPE)] == [Direction.RIGHT]


def test_quick_finger_curl_in_place_is_not_a_swipe(cfg):
    s = Script(cfg)
    s.hold("OPEN_PALM", 0.3)
    s.hold("FIST", 0.5)                 # tips jump towards the wrist
    s.hold("OPEN_PALM", 0.5)            # ... and back out
    assert s.kinds(GestureKind.SWIPE) == []


def test_slow_finger_curl_is_rejected_as_shape_change(cfg):
    s = Script(cfg)
    s.hold("OPEN_PALM", 0.3)
    s.morph("OPEN_PALM", "FIST", 0.5)   # tips slide ~1.3 palms "down"
    s.hold("FIST", 0.5)
    assert s.kinds(GestureKind.SWIPE) == []
    assert s.clf.swipe.last_reject == "hand shape changed"


def test_tips_disagreeing_is_rejected(cfg):
    # The hand sweeps right while the pinky swings the other way around the
    # wrist (its reach stays the same, so only the coherence check sees it).
    def swung(deg):
        shape = HANDS["OPEN_PALM"].copy()
        shape[[20]] = _rot(shape[[20]], degrees=deg)
        return shape

    s = Script(cfg)
    s.hold("OPEN_PALM", 0.3)
    n = int(round(0.25 / s.dt))
    for i in range(1, n + 1):
        s.pos = s.pos + np.array([2.0 * s.palm / n, 0.0])
        s._feed(None, s.pos, shape=swung(-60.0 * i / n))
    s.hold(None, 0.3, shape=swung(-60.0))
    assert s.kinds(GestureKind.SWIPE) == []
    assert s.clf.swipe.last_reject == "tips incoherent"


def test_swipe_then_relax_into_fist_still_counts(cfg):
    s = Script(cfg)
    s.hold("OPEN_PALM", 0.3)
    s.move("OPEN_PALM", 2.0, 0.0, 0.25)
    s.hold("FIST", 0.5)                 # shape change inside the stroke tail
    swipes = s.kinds(GestureKind.SWIPE)
    assert [(e.direction, e.pose) for e in swipes] == [(Direction.RIGHT, Pose.OPEN_PALM)]


def test_four_finger_swipe_types_like_open_palm(cfg):
    s = Script(cfg)
    s.hold("FOUR", 0.3)
    s.move("FOUR", 2.0, 0.0, 0.25)      # thumb tucked during the sweep
    s.hold("OPEN_PALM", 0.3)
    swipes = s.kinds(GestureKind.SWIPE)
    assert [(e.direction, e.pose) for e in swipes] == [(Direction.RIGHT, Pose.OPEN_PALM)]
    assert KeyMapper(cfg.keyboard).map(swipes[0]) == "<space>"


def test_sweep_out_of_frame_still_counts(cfg):
    s = Script(cfg)
    s.hold("OPEN_PALM", 0.3)
    s.move("OPEN_PALM", 1.6, 0.0, 0.2)
    s.absent(0.5)                       # hand leaves the frame mid-stroke
    swipes = s.kinds(GestureKind.SWIPE)
    assert [(e.direction, e.pose) for e in swipes] == [(Direction.RIGHT, Pose.OPEN_PALM)]


def test_every_stroke_is_reported_to_the_log_hook(cfg):
    s = Script(cfg)
    rows = []
    s.clf.swipe.on_stroke = rows.append
    s.hold("OPEN_PALM", 0.3)
    s.move("OPEN_PALM", 2.0, 0.0, 0.25)     # typed
    s.hold("OPEN_PALM", 0.1)
    s.move("OPEN_PALM", -2.0, 0.0, 0.25)    # natural return -> suppressed
    s.hold("OPEN_PALM", 0.6)
    s.move("OPEN_PALM", 0.8, 0.0, 0.2)      # too short
    s.hold("OPEN_PALM", 0.3)
    assert [(r["outcome"], r["direction"]) for r in rows] == [
        ("swipe", "RIGHT"), ("return_suppressed", "LEFT"), ("rejected", "RIGHT")]
    assert rows[2]["reason"].startswith("distance")
    assert rows[0]["pose"] == "OPEN_PALM" and "OPEN_PALM:" in rows[0]["pose_votes"]
    assert rows[0]["distance"] > cfg.gesture.swipe_min_distance
    assert s.clf.swipe.last_reject == rows[2]["reason"]


def test_slow_drift_before_a_sweep_does_not_inflate_its_duration(cfg):
    s = Script(cfg)
    rows = []
    s.clf.swipe.on_stroke = rows.append
    s.hold("OPEN_PALM", 0.3)
    s.move("OPEN_PALM", 1.5, 0.0, 1.0)      # 1.5 palm/s: above rest, below start
    s.move("OPEN_PALM", 2.0, 0.0, 0.25)     # the actual sweep
    s.hold("OPEN_PALM", 0.3)
    assert [e.direction for e in s.kinds(GestureKind.SWIPE)] == [Direction.RIGHT]
    # One stroke, not a timed-out stroke plus a leftover that happens to pass.
    assert [r["outcome"] for r in rows] == ["swipe"] and rows[0]["duration"] < 0.7


def test_slow_long_sweep_is_accepted(cfg):
    s = Script(cfg)
    s.hold("OPEN_PALM", 0.3)
    s.move("OPEN_PALM", 3.0, 0.0, 1.0)      # 3 palm/s for a full second
    s.hold("OPEN_PALM", 0.3)
    assert [e.direction for e in s.kinds(GestureKind.SWIPE)] == [Direction.RIGHT]


def test_near_miss_reports_reason(cfg):
    s = Script(cfg)
    s.hold("OPEN_PALM", 0.3)
    s.move("OPEN_PALM", 0.9, 0.0, 0.2)  # too short
    s.hold("OPEN_PALM", 0.3)
    assert s.kinds(GestureKind.SWIPE) == []
    assert s.clf.swipe.last_reject.startswith("distance")


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


def test_idle_fps_is_10_and_tolerates_misclassified_frames(cfg):
    """Regression for the wake 'glitch': at 5 FPS a 2 s hold had ~10 samples, so
    a couple of misread frames pushed the open-palm share below 85 % before the
    motion-gate latch (3 s) expired. 10 FPS halves the sampling noise."""
    assert cfg.power.fps[SystemState.IDLE] == 10.0

    def success_rate(fps, p=0.08, trials=80):
        rng = np.random.default_rng(42)
        ok = 0
        for _ in range(trials):
            clf = GestureClassifier(cfg.gesture, cfg.pose)
            clf.configure(SystemState.IDLE, fps)
            t = 0.0
            for _ in range(int(cfg.motion.latch_s * fps)):
                t += 1.0 / fps
                pose = "FIST" if rng.random() < p else "OPEN_PALM"
                if any(e.kind == GestureKind.HOLD for e in clf.update(observation(pose, t))):
                    ok += 1
                    break
        return ok / trials

    at_10 = success_rate(10.0)
    assert at_10 >= 0.95 and at_10 > success_rate(5.0)
