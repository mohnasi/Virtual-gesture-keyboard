"""
Central configuration for the Gesture Virtual Keyboard.

Every tunable threshold lives here so the rest of the code base stays free of
magic numbers. Values are grouped into small frozen dataclasses; build a
customised profile with ``dataclasses.replace`` (see tests for examples).

Units
-----
* ``*_S``          seconds
* ``palm lengths`` distance divided by the wrist -> middle-MCP span (Node 0 -> 9),
                   i.e. invariant to how far the user sits from the lens.
* ``scaled``       coordinates after wrist-origin translation + palm-length scaling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple

# =========================================================================== #
#  ENABLE_HUD - live telemetry window (webcam feed + hand skeleton + state).
#  True  : a small, draggable, always-on-top window opens in the top-right
#          corner of the screen. Great for learning the gestures and tuning.
#  False : pure low-power background mode - no window is ever created, no
#          frames are resized/drawn, and OpenCV's GUI is never touched.
#  Can also be overridden per launch with ``--hud`` / ``--no-hud``.
# =========================================================================== #
ENABLE_HUD: bool = True


# --------------------------------------------------------------------------- #
# System states and their power budgets
# --------------------------------------------------------------------------- #
class SystemState(str, Enum):
    ACTIVE = "ACTIVE"              # 30 FPS, full inference, typing enabled
    IDLE = "IDLE"                  # 10 FPS, motion-gated, wake gesture only
    DEEP_STANDBY = "DEEP_STANDBY"  # 1 FPS, typing hard-disabled, "On x3" only


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 640                 # feed is locked to 640x360 (resized if the
    height: int = 360                # driver refuses the native mode)
    native_fps: int = 30
    backend: str = "auto"            # "auto" | "dshow" | "msmf" | "v4l2" | "avfoundation" | "any"
    mirror: bool = True              # selfie view: user's right == screen right
    reopen_backoff_initial_s: float = 0.5
    reopen_backoff_max_s: float = 8.0
    flush_grabs_when_slow: int = 2   # discard driver-buffered frames at low FPS


# Capture-rate budget per state. IDLE runs at 10 FPS (it was 5): at 5 FPS the
# 2 s wake hold had only ~10 samples, so one or two motion-blurred frames
# pushed the open-palm share under the 85 % bar and the wake "glitched". At
# 10 FPS the same hold gets ~20 samples. The motion gate still skips
# MediaPipe on a static scene, so the extra cost is mostly camera reads.
ACTIVE_FPS: float = 30.0
IDLE_FPS: float = 10.0
DEEP_STANDBY_FPS: float = 1.0


@dataclass(frozen=True)
class PowerConfig:
    fps: Dict[SystemState, float] = field(default_factory=lambda: {
        SystemState.ACTIVE: ACTIVE_FPS,
        SystemState.IDLE: IDLE_FPS,
        SystemState.DEEP_STANDBY: DEEP_STANDBY_FPS,
    })


@dataclass(frozen=True)
class MotionGateConfig:
    # ACTIVE runs full inference on every frame (a still hand mid-word must
    # stay tracked and swipe onsets must not lose frames). Add ACTIVE here to
    # trade a little responsiveness for extra battery life.
    enabled_states: Tuple[SystemState, ...] = (
        SystemState.IDLE, SystemState.DEEP_STANDBY,
    )
    downscale: Tuple[int, int] = (160, 90)  # diff on a 1/16-area thumbnail
    blur_kernel: int = 5                    # suppress sensor noise
    pixel_threshold: int = 25               # 0-255 grey delta for "changed"
    changed_fraction: float = 0.008         # >0.8% of pixels changed => motion
    # Keep inference running this long after the last motion. MUST exceed the
    # wake-hold duration, otherwise a *steady* palm would be gated out.
    latch_s: float = 3.0


@dataclass(frozen=True)
class TrackerConfig:
    backend: str = "auto"                  # "auto" | "solutions" | "tasks"
    model_complexity: int = 0              # Lite model (Solutions backend)
    max_num_hands: int = 1
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5
    # Tasks backend (mediapipe >= 0.10.30 removed mp.solutions.hands)
    task_model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    )
    task_model_path: str = "models/hand_landmarker.task"


@dataclass(frozen=True)
class PoseConfig:
    finger_extended_ratio: float = 1.12    # |tip| / |pip| from wrist
    thumb_extended_dist: float = 0.55      # scaled dist thumb tip -> index MCP
    thumb_vertical_cos: float = 0.70       # |cos| to image vertical for up/down
    pinch_dist: float = 0.28               # scaled dist thumb tip -> index tip
    pinch_min_index_reach: float = 0.85    # index tip must stand off the palm
                                           # (stops a tight fist reading as PINCH)


@dataclass(frozen=True)
class GestureConfig:
    buffer_len: int = 45                   # ~1.5 s at 30 FPS

    # --- swipe segmentation (wrist trajectory, palm lengths) ---------------- #
    swipe_start_speed: float = 2.2         # palm lengths / s to open a stroke
    swipe_end_speed: float = 1.0           # ... and to close it
    swipe_min_distance: float = 1.1        # palm lengths travelled
    swipe_min_straightness: float = 0.75   # net / path length
    swipe_axis_dominance: float = 1.8      # major axis / minor axis
    swipe_min_duration_s: float = 0.06
    swipe_max_duration_s: float = 0.90
    speed_smoothing: float = 0.5           # EMA alpha on instantaneous speed
    refractory_s: float = 0.25             # dead time after an emitted swipe
    return_suppress_s: float = 0.80        # ignore the opposite stroke this long
    pose_vote_min_share: float = 0.55      # majority pose during a stroke

    # --- static hold (wake gesture) ------------------------------------------ #
    wake_hold_s: float = 2.0
    wake_min_pose_share: float = 0.85
    wake_max_drift: float = 0.35           # palm lengths (wrist std-dev)

    # --- repeated sequences ("Off, Off, Off" / "On, On, On") ----------------- #
    sequence_reps: int = 3
    sequence_window_s: float = 6.0         # at 30 FPS
    sequence_window_standby_s: float = 12.0  # at 1 FPS each rep needs ~2 s
    sequence_min_hold_s: float = 0.15      # debounce; at least 1 frame always


@dataclass(frozen=True)
class StateMachineConfig:
    initial_state: SystemState = SystemState.IDLE
    no_hand_timeout_s: float = 2.0         # ACTIVE -> IDLE
    post_transition_grace_s: float = 0.75  # swallow keys right after a change


# --------------------------------------------------------------------------- #
# Keyboard layout: (pose, swipe direction) -> key
# --------------------------------------------------------------------------- #
# Special key tokens understood by keyboard_output.py
SPACE, BACKSPACE, ENTER, LAYER = "<space>", "<backspace>", "<enter>", "<layer>"

# Base layer: the 20 most frequent English letters + editing keys.
# Directions are from the USER's point of view (mirrored selfie view).
BASE_LAYOUT: Dict[str, Dict[str, str]] = {
    "POINT":     {"LEFT": "e", "RIGHT": "t", "UP": "a", "DOWN": "o"},
    "PEACE":     {"LEFT": "i", "RIGHT": "n", "UP": "s", "DOWN": "h"},
    "THREE":     {"LEFT": "r", "RIGHT": "d", "UP": "l", "DOWN": "c"},
    "FOUR":      {"LEFT": "u", "RIGHT": "m", "UP": "w", "DOWN": "f"},
    "PINCH":     {"LEFT": "g", "RIGHT": "y", "UP": "p", "DOWN": "b"},
    "OPEN_PALM": {"LEFT": BACKSPACE, "RIGHT": SPACE, "UP": LAYER, "DOWN": ENTER},
    # "FIST" is deliberately unmapped: it is the clutch used to reposition.
}

# One-shot alternate layer (OPEN_PALM + swipe UP, then one stroke).
ALT_LAYOUT: Dict[str, Dict[str, str]] = {
    "POINT":     {"LEFT": "v", "RIGHT": "k", "UP": "j", "DOWN": "x"},
    "PEACE":     {"LEFT": "q", "RIGHT": "z", "UP": ".", "DOWN": ","},
    "THREE":     {"LEFT": "1", "RIGHT": "2", "UP": "3", "DOWN": "4"},
    "FOUR":      {"LEFT": "5", "RIGHT": "6", "UP": "7", "DOWN": "8"},
    "PINCH":     {"LEFT": "9", "RIGHT": "0", "UP": "?", "DOWN": "'"},
    "OPEN_PALM": {"LEFT": BACKSPACE, "RIGHT": SPACE, "UP": LAYER, "DOWN": ENTER},
}


@dataclass(frozen=True)
class KeyboardConfig:
    base_layout: Dict[str, Dict[str, str]] = field(default_factory=lambda: BASE_LAYOUT)
    alt_layout: Dict[str, Dict[str, str]] = field(default_factory=lambda: ALT_LAYOUT)
    max_queue: int = 16                    # drop keys rather than lag behind


@dataclass(frozen=True)
class UIConfig:
    app_name: str = "Gesture Virtual Keyboard"
    icon_size: int = 64
    colors: Dict[SystemState, Tuple[int, int, int]] = field(default_factory=lambda: {
        SystemState.ACTIVE: (46, 204, 113),        # green
        SystemState.IDLE: (241, 196, 15),          # yellow
        SystemState.DEEP_STANDBY: (231, 76, 60),   # red
    })
    toasts_enabled: bool = True


@dataclass(frozen=True)
class HudConfig:
    enabled: bool = ENABLE_HUD
    window_name: str = "Gesture Keyboard HUD"
    scale: float = 0.75                    # 640x360 feed -> 480x270 window
    margin_px: int = 24                    # gap from the screen's top-right corner
    always_on_top: bool = True
    no_activate: bool = True               # Windows: never steal keyboard focus
                                           # (typed keys must reach YOUR app)
    trail_s: float = 0.6                   # wrist trajectory drawn behind the hand
    event_display_s: float = 2.0           # how long the last gesture stays visible
    dim_when_gated: bool = True            # darken the feed while ML is asleep
    refresh_ms_active: int = 10            # HighGUI event-pump interval (ACTIVE)
    refresh_ms_low_power: int = 50         # ... in IDLE / DEEP STANDBY


@dataclass(frozen=True)
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    power: PowerConfig = field(default_factory=PowerConfig)
    motion: MotionGateConfig = field(default_factory=MotionGateConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    state: StateMachineConfig = field(default_factory=StateMachineConfig)
    keyboard: KeyboardConfig = field(default_factory=KeyboardConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    hud: HudConfig = field(default_factory=HudConfig)
    log_level: str = "INFO"


DEFAULT_CONFIG = Config()
