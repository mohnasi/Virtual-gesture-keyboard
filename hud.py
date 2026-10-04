"""
Live telemetry HUD: webcam feed, MediaPipe hand skeleton, and system state.

Enabled with ``config.ENABLE_HUD`` (or ``--hud`` / ``--no-hud``). When it is
disabled this module is never imported and OpenCV's GUI is never touched, so
the daemon goes back to pure low-power background mode.

Threading
---------
OpenCV HighGUI is thread-affine (and on macOS it must be the main thread), so
**every** ``cv2.namedWindow / imshow / waitKey / moveWindow`` call happens in
``HudLoop.run`` on the main thread. The inference thread only *publishes*
immutable ``HudSnapshot`` objects into a single-slot ``HudChannel``; the HUD
renders the newest one and drops anything it was too slow to show. The HUD can
therefore never slow down recognition or typing.

Focus safety
------------
Keystrokes go to whichever window owns OS focus. If the HUD stole focus, the
user's typing would land in the HUD. On Windows the window gets
``WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW``: it can be clicked and dragged but
never becomes the foreground window, and it stays out of the taskbar and
Alt-Tab. Whatever window had focus before the HUD opened gets focus back.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np

from config import BACKSPACE, ENTER, SPACE, HudConfig, SystemState
from gestures import Direction, GestureEvent, GestureKind, Pose
from tracker import HandObservation

log = logging.getLogger(__name__)

# MediaPipe's 21-landmark hand topology (defined here so we don't depend on
# mediapipe.solutions.drawing_utils, which newer mediapipe releases removed).
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm base
)
FINGERTIPS = (4, 8, 12, 16, 20)

_KEY_LABELS = {SPACE: "SPACE", BACKSPACE: "BACKSPACE", ENTER: "ENTER"}
_ARROWS = {Direction.LEFT: (-1, 0), Direction.RIGHT: (1, 0),
           Direction.UP: (0, -1), Direction.DOWN: (0, 1)}
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE, _BLACK, _GREY = (255, 255, 255), (0, 0, 0), (170, 170, 170)
_TRAIL = (255, 200, 0)                        # BGR cyan


# --------------------------------------------------------------------------- #
# Data hand-off (inference thread -> main/HUD thread)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HudSnapshot:
    image: np.ndarray                         # BGR frame (never mutated after capture)
    timestamp: float
    state: SystemState
    target_fps: float
    gated: bool                               # motion gate skipped ML on this frame
    hand: Optional[HandObservation] = None
    pose: Optional[Pose] = None
    last_event: Optional[GestureEvent] = None
    last_key: Optional[str] = None
    last_key_t: float = float("-inf")
    alt_layer: bool = False
    trail: Tuple[Tuple[float, float], ...] = ()
    infer_ms: float = 0.0
    wake_progress: float = 0.0                # 0..1 open-palm hold
    off_progress: int = 0                     # thumb-down repetitions so far
    on_progress: int = 0                      # thumb-up repetitions so far
    reps_needed: int = 3


class HudChannel:
    """Single-slot, thread-safe mailbox plus a visibility flag.

    ``visible`` is read by the inference thread (to skip building snapshots
    when nobody is looking) and toggled by the tray or by the user closing the
    window. ``threading.Event`` makes both directions safe without locks.
    """

    def __init__(self, visible: bool = True, trail_s: float = 0.6) -> None:
        self.trail_s = trail_s
        self._lock = threading.Lock()
        self._snap: Optional[HudSnapshot] = None
        self._seq = 0
        self._visible = threading.Event()
        if visible:
            self._visible.set()

    @property
    def visible(self) -> bool:
        return self._visible.is_set()

    def set_visible(self, visible: bool) -> None:
        if visible:
            self._visible.set()
        else:
            self._visible.clear()

    def toggle(self) -> None:
        self.set_visible(not self.visible)

    def publish(self, snap: HudSnapshot) -> None:
        with self._lock:
            self._snap = snap
            self._seq += 1

    def latest(self, after_seq: int = 0) -> Tuple[int, Optional[HudSnapshot]]:
        with self._lock:
            if self._seq == after_seq:
                return after_seq, None
            return self._seq, self._snap


# --------------------------------------------------------------------------- #
# Rendering (pure: snapshot -> image; unit-testable without a display)
# --------------------------------------------------------------------------- #
def key_label(token: Optional[str]) -> str:
    if token is None:
        return ""
    return _KEY_LABELS.get(token, f"'{token}'")


def describe_event(ev: Optional[GestureEvent]) -> str:
    if ev is None:
        return ""
    if ev.kind == GestureKind.SWIPE:
        return f"SWIPE {ev.direction.value} ({ev.pose.value})"
    if ev.kind == GestureKind.HOLD:
        return f"HOLD {ev.pose.value}"
    return f"{ev.pose.value} x{ev.count}"


def landmarks_to_pixels(hand: HandObservation) -> np.ndarray:
    """Invert the wrist-origin / palm-length normalisation -> (21, 2) pixels."""
    return hand.landmarks[:, :2] * hand.palm_px + hand.wrist_px


class HudRenderer:
    def __init__(self, cfg: HudConfig, colors_rgb: Dict[SystemState, Tuple[int, int, int]],
                 frame_size: Tuple[int, int] = (640, 360)) -> None:
        self._cfg = cfg
        w, h = frame_size
        self.size = (max(160, int(round(w * cfg.scale))), max(90, int(round(h * cfg.scale))))
        self._colors = {s: tuple(int(c) for c in rgb[::-1]) for s, rgb in colors_rgb.items()}
        self._k = self.size[0] / 480.0               # layout unit (1.0 at default size)
        self._fs = max(0.33, 0.45 * self._k)         # font scale
        self._last_ts: Optional[float] = None
        self._fps = 0.0

    # ---------------------------------------------------------------- api -- #
    def render(self, snap: HudSnapshot) -> np.ndarray:
        W, H = self.size
        s = W / snap.image.shape[1]
        img = cv2.resize(snap.image, (W, H), interpolation=cv2.INTER_AREA)
        if snap.gated and self._cfg.dim_when_gated:
            img = cv2.convertScaleAbs(img, alpha=0.5)
        color = self._colors.get(snap.state, _WHITE)
        self._update_fps(snap.timestamp)

        self._draw_trail(img, snap.trail, s)
        if snap.hand is not None and snap.hand.present:
            self._draw_skeleton(img, landmarks_to_pixels(snap.hand) * s, color)
        self._draw_swipe_arrow(img, snap, color)
        self._draw_top_bar(img, snap, color)
        self._draw_bottom_bar(img, snap, color)
        if snap.gated:
            self._text_center(img, "ML asleep - waiting for motion", H // 2, _GREY)
        cv2.rectangle(img, (0, 0), (W - 1, H - 1), color, max(2, int(2 * self._k)))
        return img

    # ------------------------------------------------------------ helpers -- #
    def _update_fps(self, ts: float) -> None:
        if self._last_ts is not None and ts > self._last_ts:
            inst = 1.0 / (ts - self._last_ts)
            self._fps = inst if self._fps == 0 else 0.8 * self._fps + 0.2 * inst
        self._last_ts = ts

    def _px(self, v: float) -> int:
        return max(1, int(round(v * self._k)))

    def _text(self, img, text, org, color=_WHITE, scale=1.0) -> None:
        # Hershey glyph advance grows with thickness, so a thick black
        # "outline" would drift; a same-thickness drop shadow stays aligned.
        fs, th = self._fs * scale, self._px(1)
        cv2.putText(img, text, (org[0] + 1, org[1] + 1), _FONT, fs, _BLACK, th, cv2.LINE_AA)
        cv2.putText(img, text, org, _FONT, fs, color, th, cv2.LINE_AA)

    def _text_w(self, text, scale=1.0) -> int:
        return cv2.getTextSize(text, _FONT, self._fs * scale, self._px(1))[0][0]

    def _text_center(self, img, text, y, color) -> None:
        self._text(img, text, ((img.shape[1] - self._text_w(text)) // 2, y), color)

    @staticmethod
    def _shade(img, y0, y1) -> None:
        roi = img[y0:y1]
        img[y0:y1] = cv2.convertScaleAbs(roi, alpha=0.35)

    def _draw_trail(self, img, trail, s) -> None:
        if len(trail) < 2:
            return
        pts = (np.asarray(trail, dtype=np.float32) * s).astype(np.int32)
        n = len(pts)
        for i in range(1, n):                  # older segments thinner
            t = self._px(1 + 3 * i / n)
            cv2.line(img, tuple(pts[i - 1]), tuple(pts[i]), _TRAIL, t, cv2.LINE_AA)

    def _draw_skeleton(self, img, pts, color) -> None:
        p = pts.astype(np.int32)
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, tuple(p[a]), tuple(p[b]), _WHITE, self._px(2), cv2.LINE_AA)
        for i, pt in enumerate(p):
            if i in FINGERTIPS:
                cv2.circle(img, tuple(pt), self._px(5), _WHITE, -1, cv2.LINE_AA)
                cv2.circle(img, tuple(pt), self._px(5), color, self._px(2), cv2.LINE_AA)
            else:
                cv2.circle(img, tuple(pt), self._px(3), color, -1, cv2.LINE_AA)

    def _draw_swipe_arrow(self, img, snap, color) -> None:
        ev = snap.last_event
        if ev is None or ev.kind != GestureKind.SWIPE or snap.timestamp - ev.timestamp > 0.6:
            return
        W, H = self.size
        dx, dy = _ARROWS[ev.direction]
        L = int(min(W, H) * 0.22)
        c = (W // 2, H // 2)
        cv2.arrowedLine(img, (c[0] - dx * L, c[1] - dy * L), (c[0] + dx * L, c[1] + dy * L),
                        color, self._px(6), cv2.LINE_AA, tipLength=0.35)

    def _draw_top_bar(self, img, snap, color) -> None:
        W = self.size[0]
        bar = self._px(26)
        self._shade(img, 0, bar)
        y = bar - self._px(8)
        r = self._px(6)
        cv2.circle(img, (self._px(14), bar // 2), r, color, -1, cv2.LINE_AA)
        self._text(img, snap.state.value.replace("_", " "), (self._px(26), y), color, 1.15)
        ml = "ML skipped" if snap.gated else f"ML {snap.infer_ms:.0f} ms"
        right = f"{self._fps:.0f}/{snap.target_fps:.0f} FPS | {ml}"
        self._text(img, right, (W - self._text_w(right) - self._px(10), y), _WHITE)

    def _draw_bottom_bar(self, img, snap, color) -> None:
        W, H = self.size
        line = self._px(20)
        y0 = H - 2 * line - self._px(8)
        self._shade(img, y0, H)
        x = self._px(10)
        y1, y2 = y0 + line, y0 + 2 * line

        # Line 1: current pose + badges
        if snap.gated:
            pose = "--"
        elif snap.pose is None:
            pose = "no hand"
        else:
            pose = snap.pose.value
        self._text(img, f"Pose: {pose}", (x, y1), _WHITE)
        badge = ""
        if snap.state == SystemState.DEEP_STANDBY:
            badge = "TYPING OFF"
        elif snap.alt_layer:
            badge = "ALT LAYER"
        if badge:
            self._text(img, badge, (W - self._text_w(badge) - x, y1), color)

        # Line 2: last gesture / typed key, or a context hint with progress
        age = snap.timestamp - (snap.last_event.timestamp if snap.last_event else float("-inf"))
        key_age = snap.timestamp - snap.last_key_t
        if snap.last_event is not None and age <= self._cfg.event_display_s:
            msg = f"Gesture: {describe_event(snap.last_event)}"
            if snap.last_key is not None and key_age <= self._cfg.event_display_s:
                msg += f"  -> {key_label(snap.last_key)}"
            self._text(img, msg, (x, y2), _WHITE)
            return

        progress, hint = 0.0, ""
        if snap.state == SystemState.IDLE:
            hint, progress = "Hold open palm 2 s to wake", snap.wake_progress
        elif snap.state == SystemState.ACTIVE:
            if snap.off_progress:
                hint = f"Thumb down {snap.off_progress}/{snap.reps_needed} -> standby"
                progress = snap.off_progress / snap.reps_needed
            else:
                hint = "Swipe to type | thumb down x3 to sleep"
        else:
            hint = f"Thumb up x3 to resume ({snap.on_progress}/{snap.reps_needed})"
            progress = snap.on_progress / snap.reps_needed
        self._text(img, hint, (x, y2), _GREY)
        if progress > 0:
            bw, bh = self._px(70), self._px(8)
            bx, by = W - bw - x, y2 - bh
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), _GREY, 1)
            cv2.rectangle(img, (bx, by), (bx + int(bw * min(1.0, progress)), by + bh), color, -1)


# --------------------------------------------------------------------------- #
# Window management (main thread only)
# --------------------------------------------------------------------------- #
def display_available() -> bool:
    """On Linux, Qt-based OpenCV builds *abort the process* (uncatchable) when
    no display server is reachable, so check before touching HighGUI."""
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def screen_size() -> Tuple[int, int]:
    """Primary-monitor size in the same coordinate space cv2.moveWindow uses."""
    if sys.platform.startswith("win"):
        try:
            import ctypes
            u = ctypes.windll.user32
            return int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1))
        except Exception:
            pass
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        size = int(root.winfo_screenwidth()), int(root.winfo_screenheight())
        root.destroy()
        return size
    except Exception:
        return 1920, 1080


class _Win32:
    """Minimal user32 helpers: keep the HUD on top and focus-neutral."""

    GWL_EXSTYLE = -20
    WS_EX_TOOLWINDOW = 0x00000080
    WS_EX_NOACTIVATE = 0x08000000
    SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE, SWP_FRAMECHANGED = 0x1, 0x2, 0x10, 0x20
    available = sys.platform.startswith("win")

    @classmethod
    def _user32(cls):
        import ctypes
        from ctypes import wintypes
        u = ctypes.windll.user32
        u.FindWindowW.restype = wintypes.HWND
        u.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
        u.GetForegroundWindow.restype = wintypes.HWND
        u.SetForegroundWindow.argtypes = [wintypes.HWND]
        u.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
        u.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
        u.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_uint]
        return u

    @classmethod
    def foreground(cls):
        if not cls.available:
            return None
        try:
            return cls._user32().GetForegroundWindow()
        except Exception:
            return None

    @classmethod
    def restore_foreground(cls, hwnd) -> None:
        if cls.available and hwnd:
            try:
                cls._user32().SetForegroundWindow(hwnd)
            except Exception:
                pass

    @classmethod
    def harden(cls, title: str, topmost: bool, no_activate: bool) -> bool:
        if not cls.available:
            return False
        try:
            u = cls._user32()
            hwnd = u.FindWindowW(None, title)
            if not hwnd:
                return False
            ex = u.GetWindowLongW(hwnd, cls.GWL_EXSTYLE) | cls.WS_EX_TOOLWINDOW
            if no_activate:
                ex |= cls.WS_EX_NOACTIVATE
            u.SetWindowLongW(hwnd, cls.GWL_EXSTYLE, ex)
            flags = cls.SWP_NOSIZE | cls.SWP_NOMOVE | cls.SWP_NOACTIVATE | cls.SWP_FRAMECHANGED
            insert_after = -1 if topmost else 0           # HWND_TOPMOST / HWND_TOP
            u.SetWindowPos(hwnd, insert_after, 0, 0, 0, 0, flags)
            return True
        except Exception:
            log.debug("Win32 HUD hardening failed", exc_info=True)
            return False


class HudWindow:
    """A small borderless-feeling OpenCV window, top-right, draggable."""

    FRAME_ALLOWANCE = 16       # Windows frame width so the window isn't clipped

    def __init__(self, cfg: HudConfig, size: Tuple[int, int],
                 screen_provider: Optional[Callable[[], Tuple[int, int]]] = None) -> None:
        self._cfg = cfg
        self.name = cfg.window_name
        self.size = size
        self._screen_provider = screen_provider
        self.is_open = False
        self.pos = (0, 0)
        self._frame_off: Optional[Tuple[int, int]] = None
        self._drag: Optional[Tuple[int, int]] = None

    def initial_position(self, screen: Tuple[int, int]) -> Tuple[int, int]:
        sw, _ = screen
        x = sw - self.size[0] - self._cfg.margin_px - self.FRAME_ALLOWANCE
        return max(0, x), max(0, self._cfg.margin_px)

    def open(self) -> bool:
        if not display_available():
            log.warning("HUD disabled: no display server (DISPLAY / WAYLAND_DISPLAY unset)")
            return False
        prev_fg = _Win32.foreground()
        try:
            flags = cv2.WINDOW_AUTOSIZE | getattr(cv2, "WINDOW_GUI_NORMAL", 0)
            cv2.namedWindow(self.name, flags)
            cv2.imshow(self.name, np.zeros((self.size[1], self.size[0], 3), np.uint8))
            self.pos = self.initial_position((self._screen_provider or screen_size)())
            cv2.moveWindow(self.name, *self.pos)
            cv2.waitKey(1)
            if self._cfg.always_on_top and hasattr(cv2, "WND_PROP_TOPMOST"):
                try:
                    cv2.setWindowProperty(self.name, cv2.WND_PROP_TOPMOST, 1)
                except cv2.error:
                    pass
            if _Win32.harden(self.name, self._cfg.always_on_top, self._cfg.no_activate):
                _Win32.restore_foreground(prev_fg)       # give focus back to the user's app
            cv2.setMouseCallback(self.name, self._on_mouse)
            self._frame_off = self._measure_frame_offset()
        except cv2.error as exc:
            log.warning("HUD unavailable - this OpenCV build has no GUI support (%s). "
                        "Install 'opencv-contrib-python' (not the -headless variant).", exc)
            self._destroy()
            return False
        self.is_open = True
        log.info("HUD opened at %s (%dx%d)", self.pos, *self.size)
        return True

    def show(self, img: np.ndarray) -> None:
        cv2.imshow(self.name, img)

    def pump(self, delay_ms: int) -> bool:
        """Process GUI events; returns False if the user closed the window."""
        cv2.waitKey(max(1, int(delay_ms)))
        try:
            return cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def close(self) -> None:
        if self.is_open:
            self._destroy()
            self.is_open = False

    # --------------------------------------------------------------- drag -- #
    def _on_mouse(self, event, x, y, flags, _param=None) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag = (x, y)
            self._sync_pos()
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = None
        elif event == cv2.EVENT_MOUSEMOVE and self._drag is not None:
            if not flags & cv2.EVENT_FLAG_LBUTTON:     # button released outside
                self._drag = None
                return
            dx, dy = x - self._drag[0], y - self._drag[1]
            if dx or dy:
                self.pos = (self.pos[0] + dx, self.pos[1] + dy)
                cv2.moveWindow(self.name, *self.pos)

    def _measure_frame_offset(self) -> Optional[Tuple[int, int]]:
        try:
            rx, ry, rw, _ = cv2.getWindowImageRect(self.name)
        except (cv2.error, AttributeError):
            return None
        return (rx - self.pos[0], ry - self.pos[1]) if rw > 0 else None

    def _sync_pos(self) -> None:
        """Re-read the position in case the user moved the window by its title bar."""
        if self._frame_off is None:
            return
        try:
            rx, ry, rw, _ = cv2.getWindowImageRect(self.name)
        except (cv2.error, AttributeError):
            return
        if rw > 0:
            self.pos = (rx - self._frame_off[0], ry - self._frame_off[1])

    def _destroy(self) -> None:
        try:
            cv2.destroyWindow(self.name)
            cv2.waitKey(1)
        except cv2.error:
            pass


class HudLoop:
    """Main-thread loop: show newest snapshot, pump GUI events, honour toggles."""

    def __init__(self, cfg: HudConfig, channel: HudChannel, renderer: HudRenderer,
                 window: Optional[HudWindow] = None) -> None:
        self._cfg = cfg
        self._channel = channel
        self._renderer = renderer
        self.window = window or HudWindow(cfg, renderer.size)
        self._state = SystemState.IDLE

    def run(self, stop_event: threading.Event) -> None:
        last_seq = 0
        try:
            while not stop_event.is_set():
                want = self._channel.visible
                if want and not self.window.is_open and not self.window.open():
                    self._channel.set_visible(False)
                elif not want and self.window.is_open:
                    self.window.close()
                if not self.window.is_open:
                    stop_event.wait(0.25)
                    continue

                last_seq, snap = self._channel.latest(last_seq)
                img = None
                if snap is not None:
                    self._state = snap.state
                    img = self._renderer.render(snap)
                delay = (self._cfg.refresh_ms_active if self._state == SystemState.ACTIVE
                         else self._cfg.refresh_ms_low_power)
                if not self.window.pump(delay):
                    log.info("HUD closed by user - running headless "
                             "(tray menu > Show telemetry HUD to reopen)")
                    self.window.close()
                    self._channel.set_visible(False)
                    continue
                if img is not None:
                    self.window.show(img)
        finally:
            self.window.close()
