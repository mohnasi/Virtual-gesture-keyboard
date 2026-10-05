# Architecture & Design Review

This document records the architectural self-critique done **before** implementation. It also covers the defects that review (and later cross-version testing) caught.

## 1. Thread model and ownership

| Thread | Owns exclusively | Talks to others via |
|---|---|---|
| `MainThread` | wiring, signal handlers, ordered shutdown, **telemetry HUD** (all OpenCV HighGUI calls), tray loop on macOS | `stop_event`, `HudChannel.latest()` |
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
| 12 | OpenCV HighGUI is thread-affine (main thread on macOS) | Windows freeze or crash if `imshow` and `waitKey` run on different threads | Every HighGUI call lives in `HudLoop.run` on the main thread. The inference thread only publishes immutable `HudSnapshot`s to a single-slot `HudChannel` |
| 13 | A HUD window steals keyboard focus | Typed gestures land in the HUD instead of the user's app | On Windows: `WS_EX_NOACTIVATE \| WS_EX_TOOLWINDOW`, and the previous foreground window is restored after the HUD opens. Dragging is done by a mouse callback, which never activates the window |
| 14 | A `-headless` OpenCV build has no GUI | `namedWindow` raises and the daemon would crash | `HudWindow.open()` catches `cv2.error`, logs a hint and hides the HUD. Typing keeps working |
| 15 | Hershey glyph advance grows with stroke thickness | A thick black "outline" drifts away from the text | The overlay uses a same-thickness 1 px drop shadow instead |
| 16 | A state transition resets the classifier in the middle of a frame | The HUD would show "no hand" right after waking | The pose is captured before `sm.step()` runs |
| 17 | The 0.75 s post-wake grace period dropped *all* typing events | A letter committed in that window would be swallowed **and** locked, so it could not be retyped | Grace now applies to swipes only. A letter needs a 0.4 s dwell, which can only start after the transition |
| 18 | Relaxing into a fist between thumb-downs ("Off, Off, Off") | The fist would be typed as A or S | Letters pause for 1.5 s after any THUMB_DOWN frame (`AslConfig.cooldown_s`) |
| 19 | Dataset photos are unrelated to each other | Video-mode tracking reuses the previous hand's region, giving wrong landmarks | Extraction runs MediaPipe in static-image mode (`static_image_mode=True` / Tasks `IMAGE`) |
| 20 | Dataset photos are 200×200, not 640×360 | Pixel-space conversion with the wrong size skews the features | `HandTracker.process` uses each image's real size. A parity test checks a 200×200 photo against a 640×360 frame |
| 21 | Near-duplicate frames from a single signer | Random K-fold reports ~99% while a real webcam gets far less | `GroupKFold` by signer, and a loud warning when only one signer exists |
| 22 | Wrist-only swipe tracking | A natural wrist flick barely moves the wrist, and a tucked thumb turned OPEN_PALM sweeps into unmapped FOUR sweeps | Swipes follow the fingertip centroid (rebuilt in pixels from wrist + palm scale). All four tips must agree, steps where tip-to-wrist distances change fast carry no motion (finger curls, letter changes), slow curls are rejected by a total shape-drift limit, and FOUR votes as OPEN_PALM. Rejected near-misses are shown on the HUD |

## 4. Static ASL letter classifier

```
HandObservation ─► features.asl_features ─► RandomForest.predict_proba ─► EMA ─► LetterDetector ─► LETTER event
 (wrist-origin,     (63 coords + 10 tip-     (≤ 15 calls/s, n_jobs=1,     (α=0.5)  (0.4 s dwell,
  palm-scaled)       tip distances; left      only in ACTIVE, hand still,             0.2 s release
                     hand mirrored to right)  no control pose)                        lockout)
```

- **One feature function.** `features.py` is imported by both `tools/extract_landmarks.py` and the runtime. The feature version and options (`mirror_left`, `use_z`) are stored in the model bundle. `AslClassifier` refuses a bundle whose version, classes or feature count don't match.
- **Threading.** The model is loaded on the main thread at start-up and only called on the inference thread, like everything else in the recognition path, so it needs no locks. `n_jobs` is forced to 1 because a one-sample prediction gains nothing from a thread pool.
- **Cost.** On synthetic 73-feature data, one `predict_proba` call on a 150-tree, depth-20 forest took about 8 ms. The 15 Hz cap and the "hand must be still" rule keep the average CPU cost well below that per frame.
- **Rejection.** There is no catch-all "not a letter" class. Instead, three guards keep non-letters from typing:
  1. the calibrated confidence threshold, applied to EMA-smoothed probabilities;
  2. rule-based suppression of the control poses;
  3. the stillness requirement.
- **Personal calibration.** `tools/record_samples.py` uses the daemon's own `open_capture` and `prepare_frame` (640×360, mirrored), `HandTracker` and `asl_features`. It runs single-threaded, because a recording tool has no power budget. The camera is released in a `finally` block. Each sample stores a `segment` (which fifth of that letter's recording it came from). With a single person, `train_asl.py` runs `GroupKFold` over those time blocks instead of a leaky random split.
- **Threshold floor.** Clean single-person data often calibrates to a threshold near 0, which would type *something* for any hand. `--min-threshold` (default 0.5) puts a floor under it.
- **Safety.** joblib/pickle can execute code on load. The app only loads the configured local path and never downloads models.

## 5. Power model

| State | Camera reads/s | NN inferences/s (static scene) | NN inferences/s (motion) |
|---|---|---|---|
| ACTIVE | 30 | 30 | 30 |
| IDLE | 10 | 0 (after 3 s latch) | ≤ 10 |
| DEEP STANDBY | 1 | 0 (after 3 s latch) | ≤ 1 |

With `ENABLE_HUD = True`, each processed frame adds one 640×360 → 480×270 resize, a few vector draws and an `imshow`. The snapshot hand-off is skipped while the HUD is hidden. With `ENABLE_HUD = False` the HUD module is never imported.

The motion gate costs one 160×90 resize, a grayscale conversion, a blur, an `absdiff` and `countNonZero`: microseconds per frame.
