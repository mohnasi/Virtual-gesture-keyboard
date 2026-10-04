"""
Gesture -> keystroke mapping and OS-level injection.

``KeyMapper`` turns a committed ASL letter into a-z, and an OPEN_PALM swipe
into Space / Backspace / Enter (``config.EDIT_SWIPES``).

``KeyboardOutput`` injects keys globally with ``pynput.keyboard.Controller``
(SendInput on Windows, Quartz events on macOS, XTest on X11) into whichever
window owns OS focus. Injection runs on its own worker thread so OS input
latency can never stall the vision pipeline; a separate ``enabled`` gate is
checked at the instant of injection and the queue is flushed whenever the
system leaves ACTIVE, so no key can leak into DEEP STANDBY.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

from config import BACKSPACE, ENTER, SPACE, KeyboardConfig
from gestures import GestureEvent, GestureKind

log = logging.getLogger(__name__)

_STOP = object()


class KeyMapper:
    """LETTER events -> a-z; OPEN_PALM swipes -> Space / Backspace / Enter."""

    def __init__(self, cfg: KeyboardConfig) -> None:
        self._cfg = cfg

    def reset(self) -> None:
        """Stateless; kept so state transitions can call it uniformly."""

    def map(self, event: GestureEvent) -> Optional[str]:
        if event.kind == GestureKind.LETTER:
            label = event.label or ""
            if len(label) != 1 or not label.isalpha():
                return None
            return label.upper() if self._cfg.letter_case == "upper" else label.lower()
        if event.kind == GestureKind.SWIPE and event.direction is not None:
            return self._cfg.edit_swipes.get(event.pose.value, {}).get(event.direction.value)
        return None


class KeyboardOutput:
    def __init__(self, cfg: KeyboardConfig, controller=None) -> None:
        self._q: "queue.Queue[object]" = queue.Queue(maxsize=cfg.max_queue)
        self._enabled = threading.Event()
        self._controller = controller
        self._thread = threading.Thread(target=self._run, name="KeyboardOutput", daemon=True)
        self.keys_sent = 0

    # ----------------------------------------------------------- lifecycle -- #
    def start(self) -> None:
        # Create the controller on the caller's thread so a missing/broken
        # input backend fails fast at start-up instead of inside a worker.
        if self._controller is None:
            from pynput.keyboard import Controller
            self._controller = Controller()
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self.set_enabled(False)
        if not self._thread.is_alive():
            return
        try:
            self._q.put(_STOP, timeout=timeout)
        except queue.Full:  # pragma: no cover - flushed above, cannot happen
            pass
        self._thread.join(timeout)

    # ---------------------------------------------------------- any thread -- #
    def set_enabled(self, enabled: bool) -> None:
        if enabled:
            self._enabled.set()
            return
        self._enabled.clear()
        while True:                      # drop anything still in flight
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            if item is _STOP:            # never swallow a shutdown request
                self._q.put_nowait(_STOP)
                break

    @property
    def enabled(self) -> bool:
        return self._enabled.is_set()

    def submit(self, token: str) -> bool:
        if not self._enabled.is_set():
            return False
        try:
            self._q.put_nowait(token)
            return True
        except queue.Full:
            log.warning("Keystroke queue full; dropping %r", token)
            return False

    # -------------------------------------------------------- worker thread -- #
    def _run(self) -> None:
        from_special = self._special_keys()
        while True:
            token = self._q.get()
            if token is _STOP:
                return
            if not self._enabled.is_set():   # second, independent safety gate
                continue
            try:
                if token in from_special:
                    self._controller.tap(from_special[token])
                else:
                    self._controller.type(token)
                self.keys_sent += 1
                log.debug("Typed %r", token)
            except Exception:
                log.exception("Keystroke injection failed for %r", token)

    def _special_keys(self):
        try:
            from pynput.keyboard import Key
            return {SPACE: Key.space, BACKSPACE: Key.backspace, ENTER: Key.enter}
        except Exception:  # tests / headless CI without an input backend
            return {SPACE: "space", BACKSPACE: "backspace", ENTER: "enter"}
