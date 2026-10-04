"""Motion gate, keyboard output and camera lifecycle (with fakes)."""
import threading
import time

import numpy as np

from camera import CaptureWorker, Frame, LatestFrameSlot
from config import BACKSPACE, SPACE
from gestures import Direction, GestureEvent, GestureKind, Pose
from keyboard_output import KeyboardOutput, KeyMapper
from motion_gate import MotionGate


# ------------------------------------------------------------- motion gate -- #
def _scene():
    """A smooth synthetic room: horizontal + vertical gradients."""
    x = np.linspace(40, 200, 640, dtype=np.float32)[None, :]
    y = np.linspace(0, 40, 360, dtype=np.float32)[:, None]
    return np.repeat((x + y)[..., None], 3, axis=2).astype(np.uint8)


def test_gate_skips_static_scene_after_latch(cfg):
    gate = MotionGate(cfg.motion)
    img = _scene()
    assert gate.evaluate(img, 0.0).run_inference           # first frame: no reference
    assert gate.evaluate(img, 1.0).run_inference           # still latched
    d = gate.evaluate(img, 10.0)
    assert not d.motion and not d.run_inference and d.changed_fraction == 0.0


def test_gate_opens_on_motion_and_latch_covers_steady_palm(cfg):
    gate = MotionGate(cfg.motion)
    base = _scene()
    gate.evaluate(base, 0.0)
    gate.evaluate(base, 10.0)
    moved = base.copy()
    moved[100:250, 200:350] = (15, 20, 25)                  # a "hand" appears
    assert gate.evaluate(moved, 10.2).motion
    # hand now held perfectly still for the 2 s wake gesture
    assert gate.evaluate(moved, 11.2).run_inference
    assert gate.evaluate(moved, 12.2).run_inference
    assert not gate.evaluate(moved, 13.5).run_inference      # latch (3 s) expired


def test_gate_ignores_sensor_noise(cfg):
    gate = MotionGate(cfg.motion)
    base = _scene().astype(np.int16)
    rng = np.random.default_rng(5)
    gate.evaluate(base.astype(np.uint8), 0.0)
    noisy = np.clip(base + rng.integers(-6, 7, base.shape), 0, 255).astype(np.uint8)
    assert not gate.evaluate(noisy, 10.0).motion


# ---------------------------------------------------------------- keyboard -- #
def _swipe(pose, direction):
    return GestureEvent(GestureKind.SWIPE, Pose(pose), 0.0, Direction(direction))


def test_mapper_base_layer_and_clutch(cfg):
    m = KeyMapper(cfg.keyboard)
    assert m.map(_swipe("POINT", "RIGHT")) == "t"
    assert m.map(_swipe("OPEN_PALM", "RIGHT")) == SPACE
    assert m.map(_swipe("OPEN_PALM", "LEFT")) == BACKSPACE
    assert m.map(_swipe("FIST", "LEFT")) is None
    assert m.map(_swipe("UNKNOWN", "LEFT")) is None


def test_mapper_alt_layer_is_one_shot(cfg):
    m = KeyMapper(cfg.keyboard)
    assert m.map(_swipe("OPEN_PALM", "UP")) is None and m.alt_layer
    assert m.map(_swipe("POINT", "RIGHT")) == "k"
    assert not m.alt_layer
    assert m.map(_swipe("POINT", "RIGHT")) == "t"


def test_alphabet_coverage(cfg):
    letters = set()
    for layout in (cfg.keyboard.base_layout, cfg.keyboard.alt_layout):
        for keys in layout.values():
            letters |= {k for k in keys.values() if len(k) == 1 and k.isalpha()}
    assert letters == set("abcdefghijklmnopqrstuvwxyz")


class FakeController:
    def __init__(self):
        self.typed = []
        self.event = threading.Event()

    def type(self, s):
        self.typed.append(s)
        self.event.set()

    def tap(self, k):
        self.typed.append(f"<{k}>")
        self.event.set()


def test_keyboard_output_respects_enable_gate(cfg):
    fc = FakeController()
    kb = KeyboardOutput(cfg.keyboard, controller=fc)
    kb.start()
    try:
        assert kb.submit("a") is False                 # disabled by default
        kb.set_enabled(True)
        assert kb.submit("b")
        assert fc.event.wait(1.0)
        assert fc.typed[0] == "b"
    finally:
        kb.stop()
    assert not kb._thread.is_alive()


def test_disabling_flushes_pending_keys(cfg):
    kb = KeyboardOutput(cfg.keyboard, controller=FakeController())  # not started
    kb.set_enabled(True)
    for c in "abc":
        kb.submit(c)
    kb.set_enabled(False)
    assert kb._q.empty()


# ------------------------------------------------------------------ camera -- #
def test_latest_frame_slot_keeps_only_newest():
    slot = LatestFrameSlot()
    img = np.zeros((2, 2, 3), np.uint8)
    for i in range(1, 6):
        slot.publish(Frame(img, float(i), i))
    f = slot.wait_next(0, timeout=0.1)
    assert f.seq == 5
    assert slot.wait_next(5, timeout=0.05) is None
    slot.close()
    assert slot.closed and slot.wait_next(5, timeout=5.0) is None


class FakeCap:
    instances = []

    def __init__(self, *a, fail_after=None):
        self.released = False
        self.reads = 0
        self.fail_after = fail_after
        FakeCap.instances.append(self)

    def isOpened(self):
        return True

    def set(self, *a):
        return True

    def get(self, prop):
        return 0

    def grab(self):
        return True

    def read(self):
        self.reads += 1
        if self.fail_after is not None and self.reads > self.fail_after:
            return False, None
        return True, np.zeros((480, 640, 3), np.uint8)   # wrong size on purpose

    def release(self):
        self.released = True


def test_capture_worker_releases_camera_and_locks_resolution(cfg, monkeypatch):
    import cv2
    FakeCap.instances = []
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a: FakeCap())
    slot, stop = LatestFrameSlot(), threading.Event()
    w = CaptureWorker(cfg.camera, slot, lambda: 30.0, stop)
    w.start()
    f = slot.wait_next(0, timeout=2.0)
    assert f is not None and f.image.shape == (360, 640, 3)
    w.shutdown()
    w.join(2.0)
    assert not w.is_alive()
    assert FakeCap.instances and all(c.released for c in FakeCap.instances)
    assert slot.closed


def test_capture_worker_reopens_dead_camera(cfg, monkeypatch):
    import cv2
    FakeCap.instances = []
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a: FakeCap(fail_after=2))
    slot, stop = LatestFrameSlot(), threading.Event()
    w = CaptureWorker(cfg.camera, slot, lambda: 30.0, stop)
    w.start()
    deadline = time.monotonic() + 5.0
    while len(FakeCap.instances) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    w.shutdown()
    w.join(3.0)
    assert len(FakeCap.instances) >= 2              # reopened after failures
    assert all(c.released for c in FakeCap.instances)


def test_capture_rate_change_wakes_slow_loop(cfg, monkeypatch):
    import cv2
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a: FakeCap())
    fps = {"v": 0.2}                                  # one frame every 5 s
    slot, stop = LatestFrameSlot(), threading.Event()
    w = CaptureWorker(cfg.camera, slot, lambda: fps["v"], stop)
    w.start()
    first = slot.wait_next(0, timeout=2.0)
    fps["v"] = 30.0
    w.notify_rate_change()
    nxt = slot.wait_next(first.seq, timeout=1.0)     # must not wait 5 s
    w.shutdown()
    w.join(2.0)
    assert nxt is not None
