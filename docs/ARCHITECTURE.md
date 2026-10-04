# Architecture & Design Review

This document records the architectural self-critique done **before** implementation. It also covers the defects that review (and later cross-version testing) caught.

## 1. Thread model and ownership

| Thread | Owns exclusively | Talks to others via |
|---|---|---|
| `MainThread` | wiring, signal handlers, ordered shutdown (tray loop on macOS) | `stop_event` |
| `CaptureWorker` | `cv2.VideoCapture` (open → read → **release**) | `LatestFrameSlot.publish()` |
| `InferenceWorker` | MediaPipe graph, `MotionGate`, `GestureClassifier`, state-machine writes | `KeyboardOutput.submit()`, state listeners |
| `KeyboardOutput` | `pynput.keyboard.Controller` | bounded `queue.Queue` |
| `TrayUI` | pystray event loop | `StateMachine.request()` (command queue) |
| `Toast` (short-lived) | one notification call | — |

**Rule: one owner per resource.** Nothing that wraps native handles (the camera, the TFLite/MediaPipe graph, the OS input APIs) is touched by more than one thread. That rule removes whole classes of bugs. The two most common are Windows camera stacks refusing to release a device opened on another thread, and MediaPipe's non-thread-safe graph being called concurrently.

### Hand-off primitives

- **`LatestFrameSlot`** is a `Condition`-guarded single slot. Each write overwrites the previous frame. Capture never blocks, and inference never works through a backlog of stale frames, which would show up as input lag. Memory stays constant.
- **Command queue (`StateMachine.request`)**: the tray menu runs on pystray's thread, but only the inference thread writes `state`. Commands are queued and applied at the top of the next inference tick (≤ 250 ms).
- **Atomic reads**: the capture thread reads `sm.target_fps` every frame. That is a single attribute read, which is atomic under the GIL, so it needs no lock.
- **Keystroke queue**: bounded (16). If the OS stalls, keys are dropped rather than replayed late.

### Defence in depth for "typing disabled"

DEEP STANDBY must *guarantee* no keystrokes. Three independent barriers enforce it:

1. The state machine only forwards swipe events while in `ACTIVE`, and only after a 0.75 s grace period.
2. The transition listener calls `keyboard.set_enabled(False)` *first*, before any other fan-out, which also flushes queued keys.
3. The injection worker re-checks the `enabled` event at the moment of injection.

## 2. Camera resource release

- `CaptureWorker.run` wraps each capture session in `try/finally: cap.release()`. Exceptions, read failures and shutdown all go through the same release path.
- One `threading.Event` signals shutdown. It can be set by SIGINT, SIGTERM, SIGBREAK (Windows Ctrl+Break), the tray's **Quit**, or a crash in the inference thread.
- `GestureKeyboardApp.shutdown()` is idempotent and also registered with `atexit`. It tears down in a fixed order:
  1. Disable and stop keyboard output, so no key can fire during teardown.
  2. Stop capture and close the frame slot, which wakes the inference thread.
  3. Join capture (this is when `cap.release()` runs) and inference (`graph.close()`), each with a timeout. Any thread still alive is logged.
  4. Stop the tray.
- **Model before camera:** the MediaPipe graph is created *before* the camera is opened. If the model can't load (missing package, blocked download), the app exits without ever turning on the webcam LED or locking the device.
- **Hot-unplug:** 10 consecutive failed reads trigger release and reopen with exponential backoff (0.5 s → 8 s), instead of spinning on a dead handle.
- **Low-FPS staleness:** with `CAP_PROP_BUFFERSIZE=1`, plus flushing `grab()`s whenever the interval is above 100 ms, a 1 FPS sample shows *now* and not a frame buffered a second ago.

## 3. Issues found during review

| # | Issue | Consequence if missed | Resolution |
|---|---|---|---|
| 1 | The motion gate blocks a **steady** hand | The wake gesture ("open palm held still 2 s") could never be detected | Gate **latch**: after motion, inference continues for 3 s (> 2 s hold) |
| 2 | MediaPipe x and y are normalised to width and height separately | On 640×360 the hand is distorted 16:9, which skews distance ratios | Convert to pixel space (`x·W, y·H, z·W`) **before** the wrist-origin translation and the ÷‖node 9‖ scaling |
| 3 | Wrist-relative landmarks lose position in the frame | Swipes become undetectable | The buffer also stores raw wrist pixels + palm length, and the trajectory is measured in palm-lengths |
| 4 | Return-stroke ghosting | Every swipe also types the opposite key | Refractory period, opposite-direction suppression window, and a FIST "clutch" |
| 5 | A gated frame was treated as "no hand" | A still hand in ACTIVE would be put to sleep | `hand_present=None` means *no new evidence*, which is distinct from `False` |
| 6 | A 1 FPS DEEP STANDBY sequence vs. time-based debounce | "On ×3" would be impossible at 1 FPS | Debounce is counted in **frames** (`ceil(0.15 s × fps)`, minimum 1), and the standby window is 12 s |
| 7 | Gating in ACTIVE drops the first frames of a swipe | Strokes come out short and get rejected | The gate is disabled in ACTIVE by default ("full inference"), and this is configurable |
| 8 | `mp.solutions.hands` was removed in mediapipe ≥ 0.10.30 | `model_complexity=0` crashes on Python 3.13+ | Dual backend: Solutions Lite preferred, Tasks `HandLandmarker` as fallback |
| 9 | A `threading.Thread` subclass attribute named `_stop` | On Python ≤ 3.12 it shadows `Thread._stop()`, so `join()` raises `TypeError` | Renamed to `_stop_event`. Caught by running the suite on 3.10, 3.11, 3.12 and 3.13 |
| 10 | `pythonw.exe` has `sys.stderr = None` | Logging setup crashes when run headless | A rotating file log is always written. The console handler is added only when stderr exists |
| 11 | pystray on macOS must run on the main thread | The tray crashes on macOS | `run_blocking()` runs on the main thread there, and a daemon thread is used elsewhere |

## 4. Power model

| State | Camera reads/s | NN inferences/s (static scene) | NN inferences/s (motion) |
|---|---|---|---|
| ACTIVE | 30 | 30 | 30 |
| IDLE | 5 | 0 (after 3 s latch) | ≤ 5 |
| DEEP STANDBY | 1 | 0 (after 3 s latch) | ≤ 1 |

The motion gate costs one 160×90 resize, a grayscale conversion, a blur, an `absdiff` and `countNonZero`: microseconds per frame.
