"""
Non-intrusive feedback: system-tray status icon + native toast notifications.

* The tray icon (pystray + Pillow) runs its own event loop in a daemon thread
  (on macOS AppKit requires the main thread, so ``main.py`` uses
  ``run_blocking`` there instead).
* ``set_state`` may be called from the inference thread; it is serialised with
  a lock and only touches pystray's icon/title setters, which marshal to the
  backend's own loop.
* Toasts are fired on short-lived daemon threads so a slow notification API
  can never stall inference. Backends: winotify (Windows) -> plyer -> log.
"""
from __future__ import annotations

import logging
import sys
import threading
from typing import Callable, Dict, Optional

from config import SystemState, UIConfig
from state_machine import Command

log = logging.getLogger(__name__)

STATE_LABELS = {
    SystemState.ACTIVE: "ACTIVE - typing enabled (30 FPS)",
    SystemState.IDLE: "IDLE - show open palm 2 s to wake (5 FPS)",
    SystemState.DEEP_STANDBY: "DEEP STANDBY - thumb up x3 to resume (1 FPS)",
}


def make_icon_image(rgb, size: int = 64):
    """Coloured status disc with a soft ring, drawn with Pillow."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = max(2, size // 16)
    dark = tuple(max(0, int(c * 0.6)) for c in rgb)
    d.ellipse((pad, pad, size - pad, size - pad), fill=(*rgb, 255), outline=(*dark, 255),
              width=max(2, size // 16))
    inner = size // 3
    d.ellipse((inner, inner, size - inner, size - inner), fill=(255, 255, 255, 200))
    return img


class Notifier:
    def __init__(self, app_name: str, enabled: bool = True) -> None:
        self._app = app_name
        self._enabled = enabled

    def notify(self, title: str, message: str) -> None:
        if not self._enabled:
            return
        threading.Thread(target=self._send, args=(title, message),
                         name="Toast", daemon=True).start()

    def _send(self, title: str, message: str) -> None:
        if sys.platform.startswith("win"):
            try:
                from winotify import Notification
                Notification(app_id=self._app, title=title, msg=message,
                             duration="short").show()
                return
            except Exception as exc:
                log.debug("winotify unavailable: %s", exc)
        try:
            from plyer import notification
            notification.notify(title=title, message=message,
                                app_name=self._app, timeout=4)
            return
        except Exception as exc:
            log.debug("plyer unavailable: %s", exc)
        log.info("[toast] %s: %s", title, message)


class TrayUI:
    def __init__(self, cfg: UIConfig, on_command: Callable[[Command], None],
                 on_quit: Callable[[], None]) -> None:
        self._cfg = cfg
        self._on_command = on_command
        self._on_quit = on_quit
        self._lock = threading.Lock()
        self._state = SystemState.IDLE
        self._icon = None
        self._images: Dict[SystemState, object] = {}
        self._thread: Optional[threading.Thread] = None
        self.available = False
        try:
            self._build()
            self.available = True
        except Exception as exc:  # no display / pystray backend missing
            log.warning("System tray unavailable (%s); running without it", exc)

    def _build(self) -> None:
        import pystray

        self._images = {s: make_icon_image(c, self._cfg.icon_size)
                        for s, c in self._cfg.colors.items()}
        Item = pystray.MenuItem
        menu = pystray.Menu(
            Item(lambda _: f"State: {STATE_LABELS[self._state]}", None, enabled=False),
            pystray.Menu.SEPARATOR,
            Item("Activate now", lambda: self._on_command(Command.ACTIVATE)),
            Item("Pause typing (Deep Standby)", lambda: self._on_command(Command.FORCE_STANDBY)),
            Item("Resume (Idle)", lambda: self._on_command(Command.RESUME)),
            pystray.Menu.SEPARATOR,
            Item("Quit", self._quit),
        )
        self._icon = pystray.Icon("gesture_virtual_keyboard", self._images[self._state],
                                  self._title(self._state), menu)

    def _title(self, state: SystemState) -> str:
        return f"{self._cfg.app_name} - {state.value}"

    def _quit(self, icon=None, item=None) -> None:
        self._on_quit()

    # ------------------------------------------------------------ lifecycle -- #
    def start(self) -> None:
        """Run the tray loop in a daemon thread (Windows / Linux)."""
        if not self.available:
            return
        self._thread = threading.Thread(target=self.run_blocking, name="TrayUI", daemon=True)
        self._thread.start()

    def run_blocking(self) -> None:
        """Run the tray loop on the calling thread (required on macOS)."""
        if not self.available:
            return
        try:
            self._icon.run()
        except Exception:
            log.exception("Tray loop crashed")

    def stop(self) -> None:
        if self._icon is not None:
            try:
                self._icon.stop()
            except Exception:  # pragma: no cover
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ---------------------------------------------------------- any thread -- #
    def set_state(self, state: SystemState) -> None:
        with self._lock:
            self._state = state
            if self._icon is None:
                return
            try:
                self._icon.icon = self._images[state]
                self._icon.title = self._title(state)
                self._icon.update_menu()
            except Exception:
                log.debug("Tray update failed", exc_info=True)
