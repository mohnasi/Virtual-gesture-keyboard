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
from typing import Callable

from camera import Frame, LatestFrameSlot
from gestures import GestureClassifier
from keyboard_output import KeyboardOutput, KeyMapper
from motion_gate import MotionGate
from state_machine import StateMachine
from tracker import HandTracker

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
        if run:
            obs = tracker.process(frame.image, frame.timestamp)
            self.inferences_run += 1
            hand_present = obs.present
            events = self._classifier.update(obs)
            for ev in events:
                log.debug("Gesture %s", ev)
        else:
            self.inferences_skipped += 1

        result = self._sm.step(frame.timestamp, hand_present, events)
        for ev in result.typing_events:
            key = self._mapper.map(ev)
            if key is not None:
                self._keyboard.submit(key)
                log.info("Key %r <- %s + swipe %s", key, ev.pose.value, ev.direction.value)
