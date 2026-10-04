"""
Three-tier power/interaction state machine.

    ┌────────────┐  open palm held 2 s   ┌────────────┐
    │    IDLE    │ ────────────────────▶ │   ACTIVE   │
    │   10 FPS   │ ◀──────────────────── │   30 FPS   │
    └────────────┘   no hand for 2 s     └────────────┘
          ▲                                     │
          │ "On, On, On"                        │ "Off, Off, Off"
          │ (thumb up x3)    ┌──────────────┐   │ (thumb down x3)
          └───────────────── │ DEEP STANDBY │ ◀─┘
                             │    1 FPS     │
                             └──────────────┘

Threading contract
------------------
* ``step`` / ``process_commands`` are called **only** on the inference thread,
  which is therefore the single writer of ``state``.
* Other threads (tray menu) call ``request(Command)``; commands are queued and
  applied by the inference thread on its next tick.
* ``state`` / ``target_fps`` are plain attribute reads (atomic under the GIL)
  and safe from any thread.
"""
from __future__ import annotations

import logging
import queue
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

from config import PowerConfig, StateMachineConfig, SystemState
from gestures import GestureEvent, GestureKind, Pose

log = logging.getLogger(__name__)


class Command(str, Enum):
    FORCE_STANDBY = "FORCE_STANDBY"   # tray: "Pause typing (Deep Standby)"
    RESUME = "RESUME"                 # tray: "Resume" -> IDLE
    ACTIVATE = "ACTIVATE"             # tray: "Activate now" -> ACTIVE


@dataclass(frozen=True)
class Transition:
    old: SystemState
    new: SystemState
    reason: str
    timestamp: float


@dataclass
class StepResult:
    transitions: List[Transition] = field(default_factory=list)
    typing_events: List[GestureEvent] = field(default_factory=list)


Listener = Callable[[Transition], None]


class StateMachine:
    def __init__(self, cfg: StateMachineConfig, power: PowerConfig, now: float = 0.0) -> None:
        self._cfg = cfg
        self._power = power
        self._state: SystemState = cfg.initial_state
        self._entered_t = now
        self._last_hand_t = now
        self._listeners: List[Listener] = []
        self._commands: "queue.Queue[Command]" = queue.Queue()

    # ------------------------------------------------------------ read-only -- #
    @property
    def state(self) -> SystemState:
        return self._state

    @property
    def target_fps(self) -> float:
        return self._power.fps[self._state]

    @property
    def typing_enabled(self) -> bool:
        return self._state == SystemState.ACTIVE

    # ------------------------------------------------------------ any thread -- #
    def add_listener(self, fn: Listener) -> None:
        self._listeners.append(fn)

    def request(self, cmd: Command) -> None:
        self._commands.put(cmd)

    # ------------------------------------------------------ inference thread -- #
    def process_commands(self, now: float) -> List[Transition]:
        out: List[Transition] = []
        while True:
            try:
                cmd = self._commands.get_nowait()
            except queue.Empty:
                return out
            target = {
                Command.FORCE_STANDBY: SystemState.DEEP_STANDBY,
                Command.RESUME: SystemState.IDLE,
                Command.ACTIVATE: SystemState.ACTIVE,
            }[cmd]
            t = self._transition(target, f"tray command {cmd.value}", now)
            if t:
                out.append(t)

    def step(self, now: float, hand_present: Optional[bool],
             events: List[GestureEvent]) -> StepResult:
        """Advance the machine.

        ``hand_present`` is ``None`` when the motion gate skipped inference -
        "no new evidence", which must NOT be mistaken for "no hand".
        """
        res = StepResult()
        s = self._state

        if s == SystemState.ACTIVE:
            if hand_present:
                self._last_hand_t = now
            if any(e.kind == GestureKind.SEQUENCE and e.pose == Pose.THUMB_DOWN for e in events):
                self._add(res, SystemState.DEEP_STANDBY, '"Off, Off, Off" gesture', now)
                return res
            if hand_present is False and now - self._last_hand_t >= self._cfg.no_hand_timeout_s:
                self._add(res, SystemState.IDLE, "no hand for %.1fs" % self._cfg.no_hand_timeout_s, now)
                return res
            # Letters need a deliberate 0.4 s dwell that can only start after
            # the transition, so they pass at once; swipes wait out the grace
            # period (the hand is often still moving from the wake gesture).
            in_grace = now - self._entered_t < self._cfg.post_transition_grace_s
            res.typing_events = [e for e in events if e.kind == GestureKind.LETTER
                                 or (e.kind == GestureKind.SWIPE and not in_grace)]

        elif s == SystemState.IDLE:
            if any(e.kind == GestureKind.HOLD and e.pose == Pose.OPEN_PALM for e in events):
                self._add(res, SystemState.ACTIVE, "wake gesture (open palm hold)", now)

        elif s == SystemState.DEEP_STANDBY:
            if any(e.kind == GestureKind.SEQUENCE and e.pose == Pose.THUMB_UP for e in events):
                self._add(res, SystemState.IDLE, '"On, On, On" gesture', now)

        return res

    # ------------------------------------------------------------- internal -- #
    def _add(self, res: StepResult, new: SystemState, reason: str, now: float) -> None:
        t = self._transition(new, reason, now)
        if t:
            res.transitions.append(t)

    def _transition(self, new: SystemState, reason: str, now: float) -> Optional[Transition]:
        if new == self._state:
            return None
        t = Transition(self._state, new, reason, now)
        self._state = new
        self._entered_t = now
        self._last_hand_t = now
        log.info("State %s -> %s (%s)", t.old.value, t.new.value, reason)
        for fn in list(self._listeners):
            try:
                fn(t)
            except Exception:  # a broken listener must never wedge the machine
                log.exception("State listener failed")
        return t
