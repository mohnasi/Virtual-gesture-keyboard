"""Telemetry HUD: rendering, window placement/drag, loop lifecycle (fake GUI)."""
import dataclasses
import threading

import cv2
import numpy as np
import pytest

from conftest import observation
from config import DEFAULT_CONFIG, SPACE, SystemState
from gestures import Direction, GestureEvent, GestureKind, Pose
from hud import (
    HudChannel, HudLoop, HudRenderer, HudSnapshot, HudWindow, describe_event, key_label,
    landmarks_to_pixels,
)

HUD = DEFAULT_CONFIG.hud
COLORS = DEFAULT_CONFIG.ui.colors


def _frame():
    x = np.linspace(60, 180, 640, dtype=np.float32)[None, :]
    y = np.linspace(0, 30, 360, dtype=np.float32)[:, None]
    return np.repeat((x + y)[..., None], 3, axis=2).astype(np.uint8)


def _snap(**kw):
    base = dict(image=_frame(), timestamp=10.0, state=SystemState.ACTIVE, target_fps=30.0,
                gated=False)
    base.update(kw)
    return HudSnapshot(**base)


# ---------------------------------------------------------------- render -- #
def test_render_size_and_does_not_mutate_source_frame():
    r = HudRenderer(HUD, COLORS)
    src = _frame()
    before = src.copy()
    img = r.render(_snap(image=src))
    assert img.shape == (270, 480, 3)                 # default scale 0.75
    assert np.array_equal(src, before)


def test_skeleton_is_drawn_on_the_joints():
    r = HudRenderer(HUD, COLORS)
    hand = observation("OPEN_PALM", 10.0, wrist=(320.0, 260.0), palm=70.0)
    img = r.render(_snap(hand=hand, pose=Pose.OPEN_PALM))
    plain = r.render(_snap())
    s = 480 / 640
    tip = (landmarks_to_pixels(hand)[12] * s).round().astype(int)   # middle fingertip
    assert tuple(img[tip[1], tip[0]]) == (255, 255, 255)            # filled white tip
    assert not np.array_equal(img, plain)


def test_landmark_round_trip_matches_raw_pixels():
    hand = observation("PEACE", 1.0, wrist=(100.0, 300.0), palm=55.0)
    px = landmarks_to_pixels(hand)
    assert np.allclose(px[0], (100.0, 300.0), atol=1e-3)
    assert np.isclose(np.linalg.norm(px[9] - px[0]), 55.0, rtol=1e-4)


def test_gated_frames_are_dimmed():
    r = HudRenderer(HUD, COLORS)
    awake = r.render(_snap())
    asleep = r.render(_snap(gated=True, state=SystemState.IDLE))
    mid = slice(100, 150)
    assert asleep[mid, 50:150].mean() < awake[mid, 50:150].mean() * 0.7


@pytest.mark.parametrize("state", list(SystemState))
def test_every_state_renders_with_its_colour_border(state):
    r = HudRenderer(HUD, COLORS)
    img = r.render(_snap(state=state, wake_progress=0.5, on_progress=1, off_progress=2))
    bgr = tuple(int(c) for c in COLORS[state][::-1])
    assert tuple(img[135, 0]) == bgr                  # left border, mid-height


def test_event_and_key_labels():
    sw = GestureEvent(GestureKind.SWIPE, Pose.POINT, 1.0, Direction.RIGHT)
    assert describe_event(sw) == "SWIPE RIGHT (POINT)"
    assert describe_event(GestureEvent(GestureKind.SEQUENCE, Pose.THUMB_DOWN, 1.0, count=3)) \
        == "THUMB_DOWN x3"
    assert describe_event(GestureEvent(GestureKind.HOLD, Pose.OPEN_PALM, 1.0)) == "HOLD OPEN_PALM"
    assert key_label(SPACE) == "SPACE" and key_label("t") == "'t'" and key_label(None) == ""


def test_recent_swipe_and_typed_key_render_without_error():
    r = HudRenderer(HUD, COLORS)
    ev = GestureEvent(GestureKind.SWIPE, Pose.POINT, 9.8, Direction.LEFT)
    img = r.render(_snap(last_event=ev, last_key="e", last_key_t=9.8, alt_layer=True,
                         trail=((300.0, 200.0), (340.0, 200.0), (380.0, 200.0))))
    assert img.shape == (270, 480, 3)


# --------------------------------------------------------------- channel -- #
def test_channel_returns_only_new_snapshots():
    ch = HudChannel(visible=True)
    assert ch.latest(0) == (0, None)
    ch.publish(_snap(timestamp=1.0))
    ch.publish(_snap(timestamp=2.0))
    seq, snap = ch.latest(0)
    assert snap.timestamp == 2.0
    assert ch.latest(seq) == (seq, None)
    ch.toggle()
    assert not ch.visible


# ------------------------------------------------------------- fake GUI --- #
class FakeGUI:
    """Records HighGUI calls; simulates a window manager."""

    def __init__(self, monkeypatch, headless=False, close_after=None, stop_after=None,
                 stop_event=None):
        self.windows, self.frames, self.moves, self.props = {}, [], [], []
        self.callback = None
        self.pumps = 0
        self.close_after, self.stop_after, self.stop_event = close_after, stop_after, stop_event
        self.headless = headless
        self.rect_offset = (8, 31)
        import hud as hud_mod
        monkeypatch.setattr(hud_mod, "display_available", lambda: True)
        for name in ("namedWindow", "imshow", "moveWindow", "waitKey", "getWindowProperty",
                     "setWindowProperty", "setMouseCallback", "getWindowImageRect",
                     "destroyWindow"):
            monkeypatch.setattr(cv2, name, getattr(self, name))

    def namedWindow(self, name, flags=0):
        if self.headless:
            raise cv2.error("The function is not implemented (no GUI backend)")
        self.windows[name] = {"open": True, "pos": (0, 0)}

    def imshow(self, name, img):
        self.windows.setdefault(name, {"open": True, "pos": (0, 0)})["open"] = True
        self.frames.append(img.shape)

    def moveWindow(self, name, x, y):
        self.windows[name]["pos"] = (x, y)
        self.moves.append((x, y))

    def waitKey(self, delay=0):
        self.pumps += 1
        if self.close_after is not None and self.pumps == self.close_after:
            for w in self.windows.values():
                w["open"] = False                       # user clicked [X]
        if self.stop_after is not None and self.pumps >= self.stop_after:
            self.stop_event.set()
        return -1

    def getWindowProperty(self, name, prop):
        w = self.windows.get(name)
        return 1.0 if w and w["open"] else -1.0

    def setWindowProperty(self, name, prop, value):
        self.props.append((prop, value))

    def setMouseCallback(self, name, cb, param=None):
        self.callback = cb

    def getWindowImageRect(self, name):
        x, y = self.windows[name]["pos"]
        return (x + self.rect_offset[0], y + self.rect_offset[1], 480, 270)

    def destroyWindow(self, name):
        self.windows.pop(name, None)


def test_window_opens_in_top_right_corner(monkeypatch):
    gui = FakeGUI(monkeypatch)
    w = HudWindow(HUD, (480, 270), screen_provider=lambda: (1920, 1080))
    assert w.open()
    assert w.pos == (1920 - 480 - HUD.margin_px - HudWindow.FRAME_ALLOWANCE, HUD.margin_px)
    assert gui.moves[0] == w.pos
    assert (cv2.WND_PROP_TOPMOST, 1) in gui.props


def test_window_can_be_dragged_with_the_mouse(monkeypatch):
    gui = FakeGUI(monkeypatch)
    w = HudWindow(HUD, (480, 270), screen_provider=lambda: (1920, 1080))
    w.open()
    x0, y0 = w.pos
    gui.callback(cv2.EVENT_LBUTTONDOWN, 100, 50, cv2.EVENT_FLAG_LBUTTON)
    gui.callback(cv2.EVENT_MOUSEMOVE, 70, 90, cv2.EVENT_FLAG_LBUTTON)      # drag left/down
    assert w.pos == (x0 - 30, y0 + 40) and gui.moves[-1] == w.pos
    gui.callback(cv2.EVENT_LBUTTONUP, 70, 90, 0)
    gui.callback(cv2.EVENT_MOUSEMOVE, 0, 0, 0)                            # no button: no move
    assert w.pos == (x0 - 30, y0 + 40)


def test_drag_resyncs_after_title_bar_move(monkeypatch):
    gui = FakeGUI(monkeypatch)
    w = HudWindow(HUD, (480, 270), screen_provider=lambda: (1920, 1080))
    w.open()
    gui.windows[w.name]["pos"] = (200, 300)            # user dragged the title bar
    gui.callback(cv2.EVENT_LBUTTONDOWN, 10, 10, cv2.EVENT_FLAG_LBUTTON)
    gui.callback(cv2.EVENT_MOUSEMOVE, 15, 10, cv2.EVENT_FLAG_LBUTTON)
    assert w.pos == (205, 300)                          # continues from the real spot


def test_loop_shows_snapshots_and_destroys_window_on_stop(monkeypatch):
    stop = threading.Event()
    gui = FakeGUI(monkeypatch, stop_after=5, stop_event=stop)
    ch = HudChannel(visible=True)
    ch.publish(_snap())
    r = HudRenderer(HUD, COLORS)
    loop = HudLoop(HUD, ch, r, HudWindow(HUD, r.size, screen_provider=lambda: (1280, 720)))
    loop.run(stop)
    assert (270, 480, 3) in gui.frames
    assert HUD.window_name not in gui.windows          # destroyed in finally


def test_user_closing_window_hides_hud_but_daemon_keeps_running(monkeypatch):
    stop = threading.Event()
    FakeGUI(monkeypatch, close_after=3)
    ch = HudChannel(visible=True)
    r = HudRenderer(HUD, COLORS)
    loop = HudLoop(HUD, ch, r, HudWindow(HUD, r.size, screen_provider=lambda: (1280, 720)))
    t = threading.Thread(target=loop.run, args=(stop,))
    t.start()
    for _ in range(100):
        if not ch.visible:
            break
        threading.Event().wait(0.01)
    assert not ch.visible and t.is_alive()
    ch.set_visible(True)                               # tray: "Show telemetry HUD"
    for _ in range(100):
        if loop.window.is_open:
            break
        threading.Event().wait(0.01)
    assert loop.window.is_open
    stop.set()
    t.join(2.0)
    assert not t.is_alive()


def test_no_display_server_skips_highgui_entirely(monkeypatch):
    import hud as hud_mod
    called = []
    monkeypatch.setattr(cv2, "namedWindow", lambda *a: called.append(a))
    monkeypatch.setattr(hud_mod.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    w = HudWindow(HUD, (480, 270), screen_provider=lambda: (1280, 720))
    assert not w.open() and called == []


def test_headless_opencv_degrades_gracefully(monkeypatch):
    stop = threading.Event()
    FakeGUI(monkeypatch, headless=True)
    ch = HudChannel(visible=True)
    r = HudRenderer(HUD, COLORS)
    loop = HudLoop(HUD, ch, r, HudWindow(HUD, r.size, screen_provider=lambda: (1280, 720)))
    threading.Timer(0.3, stop.set).start()
    loop.run(stop)                                     # must not raise
    assert not ch.visible and not loop.window.is_open


# ---------------------------------------------------------- integration --- #
def _app(hud=True):
    from main import GestureKeyboardApp, build_config, parse_args
    cfg = build_config(parse_args(["--no-tray", "--no-toasts", "--hud" if hud else "--no-hud"]))
    cfg = dataclasses.replace(cfg, motion=dataclasses.replace(cfg.motion, enabled_states=()))
    return GestureKeyboardApp(cfg, use_tray=False)


def test_enable_hud_false_means_no_channel_and_no_snapshots():
    app = _app(hud=False)
    assert app.hud_channel is None


def test_pipeline_publishes_telemetry_only_while_visible():
    from camera import Frame
    app = _app(hud=True)

    class T:
        def process(self, image, ts):
            return observation("POINT", ts)

    img = np.zeros((360, 640, 3), np.uint8)
    app.inference.process_frame(Frame(img, 1.0, 1), T())
    seq, snap = app.hud_channel.latest(0)
    assert snap is not None and snap.pose == Pose.POINT and snap.state == SystemState.IDLE
    assert snap.hand is not None and snap.infer_ms >= 0.0

    app.hud_channel.set_visible(False)
    app.inference.process_frame(Frame(img, 1.2, 2), T())
    assert app.hud_channel.latest(seq) == (seq, None)


def test_full_daemon_with_hud_shuts_down_cleanly(monkeypatch):
    import main as main_mod
    from test_io import FakeCap
    from tracker import HandObservation

    monkeypatch.setattr(cv2, "VideoCapture", lambda *a: FakeCap())
    gui = FakeGUI(monkeypatch)

    class NoHand:
        def process(self, image, ts):
            return HandObservation(ts)

        def close(self):
            pass

    monkeypatch.setattr(main_mod, "HandTracker", lambda *a, **k: NoHand())
    import hud as hud_mod
    monkeypatch.setattr(hud_mod, "screen_size", lambda: (1920, 1080))
    app = _app(hud=True)
    from test_io import FakeController
    app.keyboard._controller = FakeController()
    threading.Timer(1.0, app.request_shutdown).start()
    assert app.run() == 0
    assert gui.frames and not gui.windows               # frames shown, window destroyed
    assert not app.capture.is_alive() and not app.inference.is_alive()
