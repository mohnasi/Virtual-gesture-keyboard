"""
Inference worker: motion gate -> MediaPipe -> gesture classifier -> state
machine -> keystroke queue.

This thread is the *single owner* of the MediaPipe graph, the motion gate,
the gesture classifier and the state machine's write path - none of those
objects need locks because nothing else touches them.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Callable, Optional

from camera import Frame, LatestFrameSlot
from gestures import GestureClassifier, GestureKind
from keyboard_output import KeyboardOutput, KeyMapper
from motion_gate import MotionGate
from state_machine import StateMachine
from tracker import HandObservation, HandTracker

if TYPE_CHECKING:  # the HUD module is only imported when the HUD is enabled
    from hud import HudChannel

log = logging.getLogger(__name__)


class InferenceWorker(threading.Thread):
    COMMAND_POLL_S = 0.25      # tray commands are honoured within this time

    def __init__(
        self,
        slot: LatestFrameSlot,
        sm: StateMachine,
        classifier: GestureClassifier,
        gate: MotionGate,
        gated_states,
        tracker_factory: Callable[[], HandTracker],
        mapper: KeyMapper,
        keyboard: KeyboardOutput,
        stop_event: threading.Event,
        hud: "Optional[HudChannel]" = None,
    ) -> None:
        super().__init__(name="InferenceWorker", daemon=True)
        self._slot = slot
        self._sm = sm
        self._classifier = classifier
        self._gate = gate
        self._gated_states = frozenset(gated_states)
        self._tracker_factory = tracker_factory
        self._mapper = mapper
        self._keyboard = keyboard
        self._stop_event = stop_event
        self.ready = threading.Event()
        self.error: BaseException | None = None
        self.frames_seen = 0
        self.inferences_run = 0
        self.inferences_skipped = 0
        # telemetry for the HUD (touched only on this thread)
        self._hud = hud
        self._last_event = None
        self._last_key: Optional[str] = None
        self._last_key_t = float("-inf")

    def run(self) -> None:
        tracker = None
        try:
            tracker = self._tracker_factory()   # MediaPipe graph lives on THIS thread
            self.ready.set()
            last_seq = 0
            while not self._stop_event.is_set():
                self._sm.process_commands(time.monotonic())
                frame = self._slot.wait_next(last_seq, timeout=self.COMMAND_POLL_S)
                if frame is None:
                    if self._slot.closed:
                        break
                    continue
                last_seq = frame.seq
                self.process_frame(frame, tracker)
        except BaseException as exc:
            self.error = exc
            log.exception("Inference thread crashed")
            self._stop_event.set()
        finally:
            self.ready.set()                    # never leave main() waiting
            if tracker is not None:
                tracker.close()
                log.info("MediaPipe graph closed")

    def process_frame(self, frame: Frame, tracker: HandTracker) -> None:
        self.frames_seen += 1
        state = self._sm.state
        run = True
        if state in self._gated_states:
            run = self._gate.evaluate(frame.image, frame.timestamp).run_inference

        hand_present = None          # None == "no new evidence" (gated)
        events = []
        obs: Optional[HandObservation] = None
        pose = None
        infer_ms = 0.0
        if run:
            t0 = time.perf_counter()
            obs = tracker.process(frame.image, frame.timestamp)
            infer_ms = (time.perf_counter() - t0) * 1000.0
            self.inferences_run += 1
            hand_present = obs.present
            events = self._classifier.update(obs)
            pose = self._classifier.last_pose   # read before a transition resets it
            for ev in events:
                log.debug("Gesture %s", ev)
        else:
            self.inferences_skipped += 1

        result = self._sm.step(frame.timestamp, hand_present, events)
        for ev in result.typing_events:
            key = self._mapper.map(ev)
            if key is not None:
                self._keyboard.submit(key)
                self._last_key, self._last_key_t = key, frame.timestamp
                if ev.kind == GestureKind.LETTER:
                    log.info("Key %r <- ASL %s (%.0f%%)", key, ev.label, ev.confidence * 100)
                else:
                    log.info("Key %r <- %s + swipe %s", key, ev.pose.value, ev.direction.value)

        if events:
            # Control gestures (HOLD / SEQUENCE) are more informative than a
            # simultaneous swipe, so they win the single "last gesture" slot.
            events_sorted = sorted(events, key=lambda e: e.kind == GestureKind.SWIPE)
            self._last_event = events_sorted[0]
        if self._hud is not None and self._hud.visible:
            self._publish_hud(frame, not run, obs, pose, infer_ms)

    def _publish_hud(self, frame: Frame, gated: bool, obs: Optional[HandObservation],
                     pose, infer_ms: float) -> None:
        from hud import HudSnapshot

        clf = self._classifier
        trail = tuple(
            (float(r.obs.wrist_px[0]), float(r.obs.wrist_px[1]))
            for r in clf.buffer.since(frame.timestamp - self._hud.trail_s) if r.obs.present
        )
        self._hud.publish(HudSnapshot(
            image=frame.image,
            timestamp=frame.timestamp,
            state=self._sm.state,
            target_fps=self._sm.target_fps,
            gated=gated,
            hand=obs,
            pose=pose,
            last_event=self._last_event,
            last_key=self._last_key,
            last_key_t=self._last_key_t,
            letters_enabled=clf.letters is not None,
            letter_top=clf.letters.top if clf.letters else (),
            letter_candidate=clf.letters.candidate if clf.letters else None,
            letter_progress=clf.letters.progress if clf.letters else 0.0,
            letter_locked=clf.letters.locked if clf.letters else None,
            letter_threshold=clf.letters.threshold if clf.letters else 0.0,
            trail=trail,
            infer_ms=infer_ms,
            wake_progress=clf.wake.progress,
            off_progress=clf.off_sequence.progress,
            on_progress=clf.on_sequence.progress,
            reps_needed=clf.off_sequence.reps,
        ))
