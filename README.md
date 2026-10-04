<div align="center">

# ✋ Gesture Virtual Keyboard

**Type with your hands — without touching anything.**
A low-power, headless background service that turns mid-air hand gestures into real OS keystrokes, typing into whatever app has focus.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-0097A7?logo=google&logoColor=white)](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/mohnasi/gesture-virtual-keyboard/actions/workflows/ci.yml/badge.svg)](https://github.com/mohnasi/gesture-virtual-keyboard/actions/workflows/ci.yml)

<!-- Record a 10–15 s screen capture (e.g. ScreenToGif) and save it as docs/demo.gif -->
<img src="docs/demo.gif" alt="Execution demo: typing into Notepad with hand gestures" width="720"/>

</div>

---

## Why this exists

Physical keyboards demand two things many people cannot give: **repeated tactile impact** and **fine, isolated finger control**. For people living with severe arthritis, repetitive strain injury (RSI), carpal tunnel syndrome, or recovering from a stroke, every keypress can hurt or simply not be possible. On the other side, technicians in labs, kitchens, workshops or clean rooms often wear soiled or sterile gloves and *shouldn't* touch a shared keyboard at all.

Many of these users still have good **gross arm and hand mobility**: they can raise a hand, hold a shape, and sweep it left or right. Gesture Virtual Keyboard is built around exactly that ability:

- **Tactile-free.** Large, slow, forgiving motions instead of precise key strikes.
- **Works everywhere.** Keystrokes are injected at the OS level (via `pynput`), so it types into any focused app: browser, IDE, terminal, chat.
- **Invisible until needed.** A coloured tray dot shows the state, and toast notifications appear only for major changes. An optional live telemetry HUD helps you learn the gestures, and one config flag turns it off for pure background mode.
- **Gentle on the battery.** A three-tier power state machine and a pre-ML motion gate mean the neural network barely runs when you aren't using it.

---

## How it works

```
                        ┌──────────────────────────── CaptureWorker thread ───┐
   Webcam ──► cv2.VideoCapture (DSHOW) ──► lock 640×360 ──► mirror ──► LatestFrameSlot
              rate = 30 / 10 / 1 FPS (follows the state machine)    (1-slot mailbox,
                        └───────────────────────────────────────── overwrite-on-write)
                                                                         │
 ┌────────────────────────────── InferenceWorker thread ────────────────▼───────────┐
 │                                                                                  │
 │  IDLE / DEEP STANDBY only:                                                       │
 │  ┌───────────────────────────┐  static   ┌───────────────────────────┐           │
 │  │ Motion gate               │──scene───►│ skip ML, sleep until next │           │
 │  │ gray 160×90 · cv2.absdiff │           │ frame (≈0% NN compute)    │           │
 │  │ + 3 s hold-open latch     │           └───────────────────────────┘           │
 │  └────────────┬──────────────┘                                                   │
 │        motion │ (ACTIVE: every frame)                                            │
 │               ▼                                                                  │
 │  MediaPipe Hands · Lite model (model_complexity=0) → 21 × (x, y, z)              │
 │               ▼                                                                  │
 │  Normalise: pixel space → wrist (node 0) = origin → ÷ ‖node 9 − node 0‖          │
 │               ▼                                                                  │
 │  TemporalBuffer deque(maxlen=45)  ≈ 1.5 s @ 30 FPS                               │
 │               ▼                                                                  │
 │  GestureClassifier ── Pose classifier (FIST, POINT, PEACE, … THUMB_UP/DOWN)      │
 │                    ├─ SwipeDetector       (wrist trajectory, palm-length units)  │
 │                    ├─ StaticHoldDetector  (wake: open palm, steady 2 s)          │
 │                    └─ RepetitionDetector  ("Off ×3" / "On ×3")                   │
 │               ▼                                                                  │
 │  ┌─────────────────────────── 3-tier State Machine ───────────────────────────┐  │
 │  │                                                                            │  │
 │  │   ┌──────────┐  open palm held 2 s   ┌──────────┐                          │  │
 │  │   │   IDLE   │ ────────────────────► │  ACTIVE  │──► KeyMapper ──┐         │  │
 │  │   │  10 FPS  │ ◄──────────────────── │  30 FPS  │                │         │  │
 │  │   │  yellow  │    no hand for 2 s    │  green   │                │         │  │
 │  │   └──────────┘                       └────┬─────┘                │         │  │
 │  │        ▲                                  │ "Off, Off, Off"      │         │  │
 │  │        │ "On, On, On"   ┌──────────────┐  │ thumb down ×3        │         │  │
 │  │        └─────────────── │ DEEP STANDBY │◄─┘                      │         │  │
 │  │          thumb up ×3    │  1 FPS · red │                         │         │  │
 │  │                         └──────────────┘                         │         │  │
 │  └──────────────────────────────────────────────────────────────────┼─────────┘  │
 └─────────────────────────────────────────────────────────────────────┼────────────┘
                                                                       ▼
         KeyboardOutput thread ─► pynput.keyboard.Controller ─► focused application
         TrayUI thread (pystray) ◄─ state colour       Toast threads (winotify / plyer)
         Main thread: HudLoop ◄─ HudChannel ◄─ snapshots   (only when ENABLE_HUD = True)
                      └─► OpenCV window: feed + skeleton + state/pose/gesture overlay
```

Each thread is the only owner of the resources it touches: the camera, the MediaPipe graph and the input controller are never shared across threads. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design review covering thread safety, camera release and the edge cases it fixed.

---

## States & power modes

| State | Tray | Capture rate | Motion gate | MediaPipe | Typing | Leaves when… |
|---|---|---|---|---|---|---|
| **ACTIVE** | 🟢 Green | 30 FPS | Off (full inference) | Every frame | ✅ Enabled | No hand for 2 s → IDLE · "Off ×3" → DEEP STANDBY |
| **IDLE** | 🟡 Yellow | 10 FPS | On | Only after motion (+3 s latch) | ❌ | Open palm held steady 2 s → ACTIVE |
| **DEEP STANDBY** | 🔴 Red | 1 FPS | On | Only after motion (+3 s latch) | ⛔ Hard-disabled | "On ×3" → IDLE |

Entering or leaving DEEP STANDBY shows a native desktop notification. The tray menu also has **Activate now**, **Pause typing (Deep Standby)**, **Resume** and **Quit**, for carers or for moments when gestures aren't practical.

---

## Live telemetry HUD

<img src="docs/hud_preview.png" alt="HUD in ACTIVE, IDLE and DEEP STANDBY (rendered from synthetic test data)" width="100%"/>

<sub>The three states, rendered by the HUD from the project's synthetic test hands.</sub>

A small window (480×270 by default) opens in the **top-right corner** of the screen and stays on top. It shows:

| Where | What |
|---|---|
| Video | Live webcam feed with the **21-joint MediaPipe hand skeleton** drawn over your hand, plus the recent wrist trajectory (cyan trail). A large arrow flashes when a swipe is recognised |
| Top bar | **System state** (colour-coded like the tray), measured vs. target FPS, and MediaPipe inference time, or `ML skipped` when the motion gate is sleeping |
| Bottom bar | **Current pose** (`POINT`, `OPEN_PALM`, …), the **last recognised gesture** with the key it typed (e.g. `SWIPE RIGHT (POINT) -> 't'`), the `ALT LAYER` / `TYPING OFF` badges, and progress bars for the wake hold and the ×3 sequences |

- **Move it** by dragging anywhere on the video with the mouse. It also reopens in the top-right corner every time.
- **It never steals focus.** On Windows the HUD is a no-activate tool window, so clicking or dragging it never takes keyboard focus away from the app you're typing into. On macOS and Linux, click back into your app after moving the HUD.
- **Close it** with its ✕ to keep running headless. Bring it back from the tray menu with **Show telemetry HUD**.
- **Turn it off completely** with `ENABLE_HUD = False` at the top of [`config.py`](config.py), or `--no-hud` for a single run. The HUD module is then never imported and no OpenCV window is ever created, so the daemon runs in pure low-power background mode.

---

## Gestures

### Control gestures

| Gesture | How to perform it | Works in | Effect |
|---|---|---|---|
| **Wake** | Raise an **open palm** (all fingers + thumb spread) and hold it still for **2 s** | IDLE | → ACTIVE |
| **Off, Off, Off** | Closed fist, **thumb pointing down**, then relax. Repeat **3×** within 6 s | ACTIVE | → DEEP STANDBY (typing disabled) |
| **On, On, On** | Closed fist, **thumb pointing up**, hold ~1 s, then lower. Repeat **3×** within 12 s | DEEP STANDBY | → IDLE |
| **Clutch** | Move with a **closed fist** | ACTIVE | Reposition your hand without typing |
| **Walk away** | Take your hand out of view for 2 s | ACTIVE | → IDLE (auto-sleep) |

### Typing: hand shape × swipe direction

Hold a hand shape and make one smooth sweep (about 1–2 palm-lengths). The **shape picks the row** and the **direction picks the key**. The 20 most frequent English letters are on the base layer.

| Hand shape held during the swipe | ← Left | → Right | ↑ Up | ↓ Down |
|---|:-:|:-:|:-:|:-:|
| **POINT** (index finger) | `e` | `t` | `a` | `o` |
| **PEACE** (index + middle) | `i` | `n` | `s` | `h` |
| **THREE** (index + middle + ring) | `r` | `d` | `l` | `c` |
| **FOUR** (four fingers, thumb tucked) | `u` | `m` | `w` | `f` |
| **PINCH** (thumb tip touches index tip) | `g` | `y` | `p` | `b` |
| **OPEN PALM** | ⌫ Backspace | ␣ Space | ⇧ Alt layer | ⏎ Enter |

**Alt layer (one-shot).** Open palm + swipe up, then a single stroke from this table:

| Shape | ← | → | ↑ | ↓ |
|---|:-:|:-:|:-:|:-:|
| POINT | `v` | `k` | `j` | `x` |
| PEACE | `q` | `z` | `.` | `,` |
| THREE | `1` | `2` | `3` | `4` |
| FOUR | `5` | `6` | `7` | `8` |
| PINCH | `9` | `0` | `?` | `'` |

*Example:* "hi" is PEACE ↓ then PEACE ←.

Both layouts are plain dictionaries in [`config.py`](config.py), so you can remap them to suit your own mobility or language.

**Built-in ergonomics:**
- Distances are measured in palm-lengths, so the same motion works at 40 cm or 1.5 m from the camera.
- Bringing your hand back after a swipe is recognised as a return stroke and is not typed.
- Keys are suppressed for 0.75 s after any state change, so waking up never types a stray character.

---

## Installation

> **Requirements:** Python 3.10+ and a webcam. Developed for **Windows 10/11**; macOS and Linux (X11) are supported with the notes below.
> Python **3.10–3.12** is recommended: it gets MediaPipe's legacy Lite hand model (`model_complexity=0`), which has the lowest power draw. On 3.13+ the app switches to the MediaPipe Tasks `HandLandmarker` automatically and downloads its model once (~8 MB) into `models/`.

### Windows (PowerShell)

```powershell
git clone https://github.com/mohnasi/gesture-virtual-keyboard.git
cd gesture-virtual-keyboard

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

python main.py            # with a console window (shows logs)
pythonw main.py           # fully headless background service
```

To **start with Windows**, press `Win + R`, type `shell:startup`, and create a shortcut there with the target
`C:\path\to\gesture-virtual-keyboard\.venv\Scripts\pythonw.exe C:\path\to\gesture-virtual-keyboard\main.py`.

### macOS / Linux

```bash
git clone https://github.com/mohnasi/gesture-virtual-keyboard.git
cd gesture-virtual-keyboard
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

- **macOS:** grant your terminal (or Python) **Camera** and **Accessibility** permissions in System Settings → Privacy & Security. Accessibility is what lets it send keystrokes.
- **Linux:** `pynput` needs an **X11** session. Under Wayland, global key injection is restricted by design. The tray icon needs a desktop with AppIndicator / StatusNotifier support.

### Command-line options

| Flag | Default | Purpose |
|---|---|---|
| `--camera N` | `0` | Webcam index |
| `--camera-backend` | `auto` | `auto` (DirectShow on Windows), `dshow`, `msmf`, `v4l2`, `avfoundation`, `any` |
| `--tracker` | `auto` | `solutions` (Lite model), `tasks`, or `auto` fallback |
| `--start-active` | off | Start in ACTIVE instead of IDLE |
| `--no-mirror` | off | Disable the selfie-view mirror |
| `--hud` / `--no-hud` | `ENABLE_HUD` | Show or hide the live telemetry window for this run |
| `--no-tray` / `--no-toasts` | off | Run without the tray icon / notifications |
| `--log-level` | `INFO` | `DEBUG` logs every recognised gesture |

Logs rotate in `logs/gvk.log`. This matters most under `pythonw`, where there's no console to print to.

---

## Project structure

```
gesture-virtual-keyboard/
├── main.py              # entry point: wiring, signals, ordered shutdown
├── config.py            # every threshold, FPS budget and the key layouts
├── camera.py            # CaptureWorker (sole camera owner) + LatestFrameSlot
├── motion_gate.py       # cv2.absdiff motion gate with hold-open latch
├── tracker.py           # MediaPipe backends + landmark normalisation
├── gestures.py          # pose classifier, TemporalBuffer, GestureClassifier
├── state_machine.py     # ACTIVE / IDLE / DEEP_STANDBY transitions
├── keyboard_output.py   # KeyMapper + pynput injection worker
├── ui.py                # pystray tray icon + toast notifications
├── hud.py               # live telemetry HUD (renderer + draggable window)
├── pipeline.py          # InferenceWorker: gate → track → classify → act
├── tests/               # 80 hardware-free tests (synthetic hands, fake camera/GUI)
└── docs/ARCHITECTURE.md # design review & threading model
```

## Testing

The test suite needs **no webcam, no MediaPipe and no keyboard backend**. It drives the real logic with synthetic 21-point hands, scripted trajectories and a fake `VideoCapture`, and also runs the full daemon lifecycle with real threads.

```bash
pip install -r requirements-dev.txt
pytest          # 80 tests
ruff check .
```

CI runs both on Python 3.10, 3.11, 3.12 and 3.13.

## Extending

- **New gesture:** subclass `gestures.Detector`, implement `update(record, buffer)`, and pass it via `GestureClassifier(extra_detectors=[...])`. The buffer gives you the last 45 frames of normalised landmarks and the wrist trajectory.
- **New layout:** edit `BASE_LAYOUT` / `ALT_LAYOUT` in `config.py`.
- **Tuning:** every threshold is a named field in `config.py` (finger-extension ratio, swipe speed and distance, latch time, sequence windows, and so on).

## Roadmap

- [ ] Per-user calibration wizard (records your own pose templates and swipe speed)
- [x] Live telemetry HUD (skeleton, state, pose, gesture)
- [ ] Key-layout cheat-sheet overlay in the HUD for new users
- [ ] Word prediction / auto-complete to cut the strokes needed per word
- [ ] Learned temporal classifier (1D-CNN / GRU over the landmark buffer) as a drop-in `Detector`

## License

[MIT](LICENSE). Free to use, adapt and build on, especially for accessibility work.
