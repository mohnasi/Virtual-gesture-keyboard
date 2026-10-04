"""
Video acquisition.

Design rules
------------
* The ``cv2.VideoCapture`` handle is created, read and released **only** on the
  ``CaptureWorker`` thread. Windows camera stacks (DirectShow / Media
  Foundation) are thread-affine; crossing threads is a classic source of
  "camera is in use by another application" locks after exit.
* Frames are handed to the inference thread through ``LatestFrameSlot`` - a
  single-slot, overwrite-on-write mailbox. Capture never blocks on inference
  and inference never processes a stale backlog; memory use is constant.
* The capture rate follows the state machine (30 / 10 / 1 FPS) via a
  ``fps_provider`` callable, so the sensor pipeline itself idles at low power.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from config import CameraConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Frame:
    image: np.ndarray       # BGR, CameraConfig.height x CameraConfig.width
    timestamp: float        # time.monotonic() at acquisition
    seq: int                # strictly increasing sequence number


class LatestFrameSlot:
    """Thread-safe single-slot mailbox holding only the newest frame."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: Optional[Frame] = None
        self._closed = False

    def publish(self, frame: Frame) -> None:
        with self._cond:
            self._frame = frame            # overwrite: drop anything unread
            self._cond.notify_all()

    def wait_next(self, last_seq: int, timeout: float) -> Optional[Frame]:
        """Block until a frame newer than ``last_seq`` exists (or timeout/close)."""
        with self._cond:
            self._cond.wait_for(
                lambda: self._closed
                or (self._frame is not None and self._frame.seq != last_seq),
                timeout=timeout,
            )
            frame = self._frame
            if frame is None or frame.seq == last_seq:
                return None
            return frame

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed


def _resolve_backend(name: str) -> int:
    import cv2

    table = {
        "any": cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "v4l2": cv2.CAP_V4L2,
        "avfoundation": cv2.CAP_AVFOUNDATION,
    }
    if name == "auto":
        # DirectShow opens in ~200 ms (MSMF can take seconds) and honours
        # 640x360 on virtually every UVC webcam.
        return cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    return table.get(name, cv2.CAP_ANY)


def open_capture(cfg: CameraConfig):
    """Open and configure the webcam (shared by the daemon and the tools).

    Returns an opened ``cv2.VideoCapture`` or ``None``. The caller owns it and
    must ``release()`` it on the same thread."""
    import cv2

    try:
        cap = cv2.VideoCapture(cfg.index, _resolve_backend(cfg.backend))
    except Exception:
        log.exception("cv2.VideoCapture raised")
        return None
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    cap.set(cv2.CAP_PROP_FPS, cfg.native_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimise driver-side latency
    log.info(
        "Camera %d opened at %dx%d (requested %dx%d)",
        cfg.index,
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        cfg.width, cfg.height,
    )
    return cap


def prepare_frame(image: np.ndarray, cfg: CameraConfig) -> np.ndarray:
    """Lock the feed to the configured size (640x360) and mirror it."""
    import cv2

    if image.shape[:2] != (cfg.height, cfg.width):
        image = cv2.resize(image, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA)
    if cfg.mirror:
        image = cv2.flip(image, 1)
    return image


class CaptureWorker(threading.Thread):
    """Owns the webcam for its entire lifetime and publishes frames."""

    def __init__(
        self,
        cfg: CameraConfig,
        slot: LatestFrameSlot,
        fps_provider: Callable[[], float],
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="CaptureWorker", daemon=True)
        self._cfg = cfg
        self._slot = slot
        self._fps_provider = fps_provider
        self._stop_event = stop_event
        self._kick = threading.Event()   # wakes the pacing sleep early
        self._seq = 0
        self.frames_captured = 0
        self.camera_open = threading.Event()

    # -- public, thread-safe ------------------------------------------------- #
    def notify_rate_change(self) -> None:
        """Called when the state (and hence target FPS) changes."""
        self._kick.set()

    def shutdown(self) -> None:
        self._stop_event.set()
        self._kick.set()

    # -- thread body ----------------------------------------------------------- #
    def run(self) -> None:
        backoff = self._cfg.reopen_backoff_initial_s
        try:
            while not self._stop_event.is_set():
                cap = self._open()
                if cap is None:
                    log.warning("Camera %d unavailable; retrying in %.1fs",
                                self._cfg.index, backoff)
                    self._sleep(backoff)
                    backoff = min(backoff * 2, self._cfg.reopen_backoff_max_s)
                    continue
                backoff = self._cfg.reopen_backoff_initial_s
                self.camera_open.set()
                try:
                    self._capture_loop(cap)
                finally:
                    # Always hand the device back to the OS - even on exceptions.
                    cap.release()
                    self.camera_open.clear()
                    log.info("Camera released")
        except Exception:  # pragma: no cover - defensive
            log.exception("Capture thread crashed")
            self._stop_event.set()
        finally:
            self._slot.close()

    def _open(self):
        return open_capture(self._cfg)

    def _capture_loop(self, cap) -> None:
        failures = 0
        next_due = time.monotonic()
        while not self._stop_event.is_set():
            interval = 1.0 / max(0.1, float(self._fps_provider()))

            # At low FPS the driver may hold an old frame; discard it so the
            # gesture logic sees "now", not "a second ago".
            if interval > 0.1:
                for _ in range(self._cfg.flush_grabs_when_slow):
                    cap.grab()

            ok, image = cap.read()
            if not ok or image is None:
                failures += 1
                if failures >= 10:
                    log.warning("Camera stopped delivering frames; reopening")
                    return
                self._sleep(0.05)
                continue
            failures = 0

            image = prepare_frame(image, self._cfg)   # 640x360, selfie-mirrored

            self._seq += 1
            self.frames_captured += 1
            self._slot.publish(Frame(image, time.monotonic(), self._seq))

            next_due += interval
            now = time.monotonic()
            if next_due <= now:
                next_due = now      # fell behind: don't burst to catch up
            elif self._sleep(next_due - now):
                next_due = time.monotonic()   # rate changed: re-anchor schedule

    def _sleep(self, seconds: float) -> bool:
        """Interruptible sleep. Returns True if woken early by a kick."""
        if self._stop_event.is_set():
            return True
        kicked = self._kick.wait(seconds)
        self._kick.clear()
        return kicked
