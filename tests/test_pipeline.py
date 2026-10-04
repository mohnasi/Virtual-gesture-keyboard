"""End-to-end: scripted hand observations through the real app wiring."""
import dataclasses

import numpy as np

from camera import Frame
from conftest import observation
from config import SystemState
from main import GestureKeyboardApp, build_config, parse_args
from test_io import FakeController


class ScriptedTracker:
    """Stands in for MediaPipe: returns pre-baked observations per timestamp."""

    def __init__(self):
        self.next = None

    def process(self, image, ts):
        return self.next(ts)

    def close(self):
        pass


def make_app():
    cfg = build_config(parse_args(["--no-tray", "--no-toasts", "--no-hud"]))
    cfg = dataclasses.replace(cfg, motion=dataclasses.replace(cfg.motion, enabled_states=()))
    app = GestureKeyboardApp(cfg, use_tray=False)
    fc = FakeController()
    app.keyboard._controller = fc
    return app, fc


class Driver:
    def __init__(self, app):
        self.app, self.tracker, self.t, self.seq = app, ScriptedTracker(), 100.0, 0
        self.img = np.zeros((360, 640, 3), np.uint8)
        self.pos = np.array([320.0, 200.0])

    def run(self, pose, seconds, dx=0.0, palm=80.0):
        fps = self.app.sm.target_fps
        n = max(1, int(round(seconds * fps)))
        step = np.array([dx * palm / n, 0.0])
        for _ in range(n):
            self.t += 1.0 / fps
            self.seq += 1
            self.pos = self.pos + step
            wrist = self.pos.copy()
            self.tracker.next = lambda ts, p=pose, w=wrist: observation(p, ts, w, palm)
            self.app.inference.process_frame(Frame(self.img, self.t, self.seq), self.tracker)


def test_wake_type_and_standby_cycle():
    app, fc = make_app()
    app.keyboard.start()
    try:
        d = Driver(app)
        assert app.sm.state == SystemState.IDLE
        d.run("OPEN_PALM", 2.4)                         # wake gesture @ 5 FPS
        assert app.sm.state == SystemState.ACTIVE and app.keyboard.enabled

        d.run("POINT", 1.0)                             # past grace period
        d.run("POINT", 0.25, dx=2.0)                    # swipe RIGHT -> 't'
        d.run("POINT", 0.4)
        assert fc.event.wait(1.0)
        assert fc.typed == ["t"]

        for _ in range(3):                              # "Off, Off, Off"
            d.run("THUMB_DOWN", 0.4)
            d.run("FIST", 0.4)
        assert app.sm.state == SystemState.DEEP_STANDBY
        assert not app.keyboard.enabled

        d.run("POINT", 0.3, dx=2.0)                     # no typing in standby
        assert fc.typed == ["t"]

        for _ in range(3):                              # "On, On, On" @ 1 FPS
            d.run("THUMB_UP", 1.0)
            d.run(None, 1.0)
        assert app.sm.state == SystemState.IDLE
    finally:
        app.keyboard.stop()


def test_hand_leaving_returns_to_idle():
    app, _ = make_app()
    d = Driver(app)
    d.run("OPEN_PALM", 2.4)
    assert app.sm.state == SystemState.ACTIVE
    d.run(None, 2.2)
    assert app.sm.state == SystemState.IDLE
    assert not app.keyboard.enabled


def test_shutdown_is_idempotent_without_threads_started():
    app, _ = make_app()
    app.shutdown()
    app.shutdown()
    assert app.stop_event.is_set()


def test_full_daemon_lifecycle_releases_camera(monkeypatch):
    """Real threads, fake camera + fake MediaPipe: start, run, stop cleanly."""
    import threading

    import cv2

    import main as main_mod
    from test_io import FakeCap

    FakeCap.instances = []
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a: FakeCap())

    class NoHandTracker:
        closed = False

        def process(self, image, ts):
            from tracker import HandObservation
            return HandObservation(ts)

        def close(self):
            NoHandTracker.closed = True

    monkeypatch.setattr(main_mod, "HandTracker", lambda *a, **k: NoHandTracker())
    app, _ = make_app()
    threading.Timer(1.0, app.request_shutdown).start()
    assert app.run() == 0
    assert not app.capture.is_alive() and not app.inference.is_alive()
    assert FakeCap.instances and all(c.released for c in FakeCap.instances)
    assert NoHandTracker.closed
    assert app.inference.frames_seen > 0
