from config import SystemState
from gestures import Direction, GestureEvent, GestureKind, Pose
from state_machine import Command, StateMachine

WAKE = GestureEvent(GestureKind.HOLD, Pose.OPEN_PALM, 0.0)
OFF3 = GestureEvent(GestureKind.SEQUENCE, Pose.THUMB_DOWN, 0.0, count=3)
ON3 = GestureEvent(GestureKind.SEQUENCE, Pose.THUMB_UP, 0.0, count=3)
SWIPE = GestureEvent(GestureKind.SWIPE, Pose.POINT, 0.0, Direction.RIGHT)


def make(cfg, state=SystemState.IDLE):
    import dataclasses
    return StateMachine(dataclasses.replace(cfg.state, initial_state=state), cfg.power, now=0.0)


def test_fps_budget_per_state(cfg):
    sm = make(cfg)
    assert sm.target_fps == 10.0
    sm.step(1.0, True, [WAKE])
    assert sm.target_fps == 30.0
    sm.step(2.0, True, [OFF3])
    assert sm.target_fps == 1.0


def test_full_cycle(cfg):
    sm = make(cfg)
    seen = []
    sm.add_listener(lambda t: seen.append((t.old, t.new)))
    sm.step(1.0, True, [WAKE])
    sm.step(2.0, True, [OFF3])
    sm.step(3.0, True, [WAKE])           # wake ignored in DEEP STANDBY
    sm.step(4.0, True, [ON3])
    assert seen == [(SystemState.IDLE, SystemState.ACTIVE),
                    (SystemState.ACTIVE, SystemState.DEEP_STANDBY),
                    (SystemState.DEEP_STANDBY, SystemState.IDLE)]


def test_idle_listens_only_for_wake(cfg):
    sm = make(cfg)
    r = sm.step(1.0, True, [SWIPE, OFF3, ON3])
    assert sm.state == SystemState.IDLE and r.typing_events == []


def test_active_auto_sleeps_after_two_seconds_without_hand(cfg):
    sm = make(cfg, SystemState.ACTIVE)
    sm.step(0.5, True, [])
    sm.step(2.4, False, [])
    assert sm.state == SystemState.ACTIVE
    sm.step(2.5, False, [])
    assert sm.state == SystemState.IDLE


def test_gated_frames_are_not_evidence_of_absence(cfg):
    sm = make(cfg, SystemState.ACTIVE)
    sm.step(0.5, True, [])
    for t in (1.0, 3.0, 9.0):
        sm.step(t, None, [])             # motion gate skipped inference
    assert sm.state == SystemState.ACTIVE


def test_typing_only_after_grace_period(cfg):
    sm = make(cfg)
    sm.step(1.0, True, [WAKE])
    assert sm.step(1.2, True, [SWIPE]).typing_events == []
    assert sm.step(2.0, True, [SWIPE]).typing_events == [SWIPE]


def test_off_sequence_beats_simultaneous_swipe(cfg):
    sm = make(cfg, SystemState.ACTIVE)
    r = sm.step(5.0, True, [SWIPE, OFF3])
    assert sm.state == SystemState.DEEP_STANDBY and r.typing_events == []


def test_tray_commands_are_queued_and_applied_on_owner_thread(cfg):
    sm = make(cfg)
    sm.request(Command.FORCE_STANDBY)
    assert sm.state == SystemState.IDLE          # nothing happens until processed
    ts = sm.process_commands(1.0)
    assert sm.state == SystemState.DEEP_STANDBY and len(ts) == 1
    sm.request(Command.RESUME)
    sm.request(Command.ACTIVATE)
    sm.process_commands(2.0)
    assert sm.state == SystemState.ACTIVE


def test_broken_listener_does_not_wedge_machine(cfg):
    sm = make(cfg)
    sm.add_listener(lambda t: 1 / 0)
    sm.step(1.0, True, [WAKE])
    assert sm.state == SystemState.ACTIVE
