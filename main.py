"""
Gesture Virtual Keyboard - headless, low-power accessibility daemon.

Thread model
------------
    main thread     : wiring, signals, ordered shutdown, telemetry HUD
                      (OpenCV HighGUI is thread-affine), tray on macOS
    CaptureWorker   : sole owner of cv2.VideoCapture (open / read / release)
    InferenceWorker : sole owner of MediaPipe, motion gate, classifier, state
    KeyboardOutput  : sole caller of pynput (OS keystroke injection)
    TrayUI          : pystray event loop (daemon)
    Toast           : short-lived daemon threads per notification

Run ``python main.py`` (console) or ``pythonw main.py`` (no window, Windows).
"""
from __future__ import annotations

import argparse
import atexit
import dataclasses
import logging
import signal
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from camera import CaptureWorker, LatestFrameSlot
from config import DEFAULT_CONFIG, Config, SystemState
from gestures import GestureClassifier
from keyboard_output import KeyboardOutput, KeyMapper
from motion_gate import MotionGate
from pipeline import InferenceWorker
from state_machine import StateMachine, Transition
from tracker import HandTracker
from ui import Notifier, TrayUI

log = logging.getLogger("gvk")


def load_asl_model(path: str):
    """Load the trained ASL Random Forest; without one the app still runs
    (wake / standby / editing swipes) but cannot type letters."""
    try:
        from asl_classifier import AslClassifier
        return AslClassifier.load(path)
    except FileNotFoundError:
        log.warning("No ASL model at %s - letters disabled. Train one with "
                    "tools/extract_landmarks.py + tools/train_asl.py (see README).", path)
    except ImportError as exc:
        log.warning("scikit-learn/joblib not installed (%s) - letters disabled. "
                    "Run: pip install -r requirements.txt", exc)
    except Exception:
        log.exception("Could not load ASL model %s - letters disabled", path)
    return None


class GestureKeyboardApp:
    JOIN_TIMEOUT_S = 3.0
    MODEL_LOAD_TIMEOUT_S = 120.0     # first run may download the model

    def __init__(self, cfg: Config, use_tray: bool = True) -> None:
        self.cfg = cfg
        self.stop_event = threading.Event()
        self._shutdown_lock = threading.Lock()
        self._shut_down = False

        self.slot = LatestFrameSlot()
        self.sm = StateMachine(cfg.state, cfg.power, now=time.monotonic())
        self.asl_model = load_asl_model(cfg.asl.model_path)
        self.classifier = GestureClassifier(cfg.gesture, cfg.pose,
                                            letter_model=self.asl_model, asl_cfg=cfg.asl)
        self.classifier.configure(self.sm.state, self.sm.target_fps)
        self.gate = MotionGate(cfg.motion)
        self.mapper = KeyMapper(cfg.keyboard)
        self.keyboard = KeyboardOutput(cfg.keyboard)
        self.notifier = Notifier(cfg.ui.app_name, cfg.ui.toasts_enabled)

        # Telemetry HUD: imported only when enabled, so ENABLE_HUD=False keeps
        # OpenCV's GUI completely untouched (pure background mode).
        self.hud_channel = None
        if cfg.hud.enabled:
            from hud import HudChannel
            self.hud_channel = HudChannel(visible=True, trail_s=cfg.hud.trail_s)

        self.capture = CaptureWorker(cfg.camera, self.slot,
                                     fps_provider=lambda: self.sm.target_fps,
                                     stop_event=self.stop_event)
        self.inference = InferenceWorker(
            slot=self.slot, sm=self.sm, classifier=self.classifier, gate=self.gate,
            gated_states=cfg.motion.enabled_states,
            tracker_factory=lambda: HandTracker(cfg.tracker, cfg.camera.width, cfg.camera.height),
            mapper=self.mapper, keyboard=self.keyboard, stop_event=self.stop_event,
            hud=self.hud_channel,
        )
        self.tray = TrayUI(
            cfg.ui, self.sm.request, self.request_shutdown,
            hud_toggle=self.hud_channel.toggle if self.hud_channel else None,
            hud_visible=(lambda: self.hud_channel.visible) if self.hud_channel else None,
        ) if use_tray else None
        self.sm.add_listener(self._on_transition)

    # ------------------------------------------------------------------ #
    # State-change fan-out (runs on the inference thread)
    # ------------------------------------------------------------------ #
    def _on_transition(self, t: Transition) -> None:
        self.keyboard.set_enabled(t.new == SystemState.ACTIVE)   # first: safety
        self.classifier.reset()
        self.classifier.configure(t.new, self.cfg.power.fps[t.new])
        self.mapper.reset()
        self.gate.reset()
        self.gate.open_latch(t.timestamp)     # analyse the first frames of a new state
        self.capture.notify_rate_change()
        if self.tray:
            self.tray.set_state(t.new)
        if t.new == SystemState.DEEP_STANDBY:
            self.notifier.notify("Deep Standby",
                                 "Typing disabled. Thumb up three times to wake.")
        elif t.old == SystemState.DEEP_STANDBY:
            self.notifier.notify("Standby ended",
                                 "Hold an open palm for 2 seconds to start typing.")

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def request_shutdown(self, *_args) -> None:
        self.stop_event.set()

    def _install_signal_handlers(self) -> None:
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):   # SIGBREAK: Ctrl+Break on Windows
            sig = getattr(signal, name, None)
            if sig is not None:
                try:
                    signal.signal(sig, self.request_shutdown)
                except (ValueError, OSError):
                    pass

    def run(self) -> int:
        self._install_signal_handlers()
        atexit.register(self.shutdown)
        log.info("Starting in %s (%.0f FPS)", self.sm.state.value, self.sm.target_fps)

        try:
            self.keyboard.start()
            self.keyboard.set_enabled(self.sm.typing_enabled)
            # Load the model BEFORE touching the camera: if MediaPipe cannot
            # start, the webcam is never opened (no LED, no device lock).
            self.inference.start()
            deadline = time.monotonic() + self.MODEL_LOAD_TIMEOUT_S
            while not self.inference.ready.wait(0.5):      # stay Ctrl+C-responsive
                if self.stop_event.is_set() or time.monotonic() > deadline:
                    break
            if self.stop_event.is_set():
                return 0
            if not self.inference.ready.is_set() or self.inference.error \
                    or not self.inference.is_alive():
                log.error("Hand tracker failed to start; exiting without opening the camera")
                return 1
            self.capture.start()
            if self.tray:
                self.tray.set_state(self.sm.state)

            if self.hud_channel is not None:
                # HighGUI must live on the main thread; the tray gets its own
                # thread (or, on macOS, attaches to the loop the HUD pumps).
                if self.tray:
                    if sys.platform == "darwin":
                        self.tray.start_detached()
                    else:
                        self.tray.start()
                self._run_hud()
            elif self.tray and self.tray.available and sys.platform == "darwin":
                # AppKit insists on the main thread; stop the loop on shutdown.
                threading.Thread(target=self._stop_tray_when_done, daemon=True).start()
                self.tray.run_blocking()
            else:
                if self.tray:
                    self.tray.start()
                # Short waits keep Ctrl+C responsive on Windows consoles.
                while not self.stop_event.wait(0.5):
                    pass
        except Exception:
            log.exception("Fatal start-up error")
            return 1
        finally:
            self.shutdown()
        return 1 if self.inference.error else 0

    def _run_hud(self) -> None:
        from hud import HudLoop, HudRenderer

        renderer = HudRenderer(self.cfg.hud, self.cfg.ui.colors,
                               (self.cfg.camera.width, self.cfg.camera.height))
        HudLoop(self.cfg.hud, self.hud_channel, renderer).run(self.stop_event)

    def _stop_tray_when_done(self) -> None:
        self.stop_event.wait()
        if self.tray:
            self.tray.stop()

    def shutdown(self) -> None:
        """Idempotent, ordered teardown. Releases the camera deterministically."""
        with self._shutdown_lock:
            if self._shut_down:
                return
            self._shut_down = True
        log.info("Shutting down...")
        self.stop_event.set()
        self.keyboard.stop()                       # 1. no more keystrokes
        self.capture.shutdown()                    # 2. camera loop exits ...
        self.slot.close()                          #    ... and inference wakes
        for th in (self.capture, self.inference):  # 3. cap.release() / graph.close()
            if th.is_alive():
                th.join(self.JOIN_TIMEOUT_S)
                if th.is_alive():
                    log.error("%s did not stop within %.1fs", th.name, self.JOIN_TIMEOUT_S)
        if self.tray:                              # 4. UI last
            self.tray.stop()
        log.info("Stopped. frames=%d inferences=%d skipped_by_gate=%d keys=%d",
                 self.inference.frames_seen, self.inference.inferences_run,
                 self.inference.inferences_skipped, self.keyboard.keys_sent)


def setup_logging(level: str) -> None:
    handlers: list[logging.Handler] = []
    log_dir = Path(__file__).resolve().parent / "logs"
    try:
        log_dir.mkdir(exist_ok=True)
        handlers.append(RotatingFileHandler(log_dir / "gvk.log", maxBytes=1_000_000,
                                            backupCount=3, encoding="utf-8"))
    except OSError:
        pass
    if sys.stderr is not None:                     # None under pythonw.exe
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)-7s %(threadName)-15s %(name)s: %(message)s",
                        handlers=handlers)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Low-power gesture virtual keyboard daemon")
    p.add_argument("--camera", type=int, default=DEFAULT_CONFIG.camera.index,
                   help="webcam index (default 0)")
    p.add_argument("--camera-backend", default=DEFAULT_CONFIG.camera.backend,
                   choices=["auto", "dshow", "msmf", "v4l2", "avfoundation", "any"])
    p.add_argument("--tracker", default=DEFAULT_CONFIG.tracker.backend,
                   choices=["auto", "solutions", "tasks"], help="MediaPipe API to use")
    p.add_argument("--no-mirror", action="store_true", help="disable selfie mirroring")
    p.add_argument("--start-active", action="store_true", help="start in ACTIVE instead of IDLE")
    p.add_argument("--no-tray", action="store_true", help="run without the tray icon")
    p.add_argument("--no-toasts", action="store_true", help="disable desktop notifications")
    p.add_argument("--hud", action=argparse.BooleanOptionalAction, default=None,
                   help="show/hide the live telemetry window (default: config.ENABLE_HUD)")
    p.add_argument("--asl-model", default=DEFAULT_CONFIG.asl.model_path,
                   help="trained ASL model bundle (default models/asl_rf.pkl)")
    p.add_argument("--log-level", default=DEFAULT_CONFIG.log_level)
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    c = DEFAULT_CONFIG
    return dataclasses.replace(
        c,
        camera=dataclasses.replace(c.camera, index=args.camera, backend=args.camera_backend,
                                   mirror=not args.no_mirror),
        tracker=dataclasses.replace(c.tracker, backend=args.tracker),
        state=dataclasses.replace(c.state, initial_state=SystemState.ACTIVE
                                  if args.start_active else c.state.initial_state),
        ui=dataclasses.replace(c.ui, toasts_enabled=not args.no_toasts),
        hud=dataclasses.replace(c.hud, enabled=c.hud.enabled if args.hud is None else args.hud),
        asl=dataclasses.replace(c.asl, model_path=args.asl_model),
        log_level=args.log_level,
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    setup_logging(cfg.log_level)
    return GestureKeyboardApp(cfg, use_tray=not args.no_tray).run()


if __name__ == "__main__":
    sys.exit(main())
