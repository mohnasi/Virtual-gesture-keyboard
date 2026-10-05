# Handover

Context for anyone (human or AI) picking up this repo in a later session.
Last updated: 2026-10-05.

## Project in one paragraph
Gesture Virtual Keyboard: a webcam app that types with your hand. MediaPipe gives
21 hand landmarks per frame (`tracker.py`). A Random Forest classifies static ASL
letters A–Z, and a letter is typed after a 0.4 s hold (`LetterDetector` in
`gestures.py`). Open-palm **swipes** edit text: Right = Space, Left = Backspace,
Down = Enter (`config.EDIT_SWIPES`). A state machine (`state_machine.py`) handles
ACTIVE / IDLE / DEEP_STANDBY:
- hold an open palm for 2 s to wake;
- thumbs-down ×3 to go to standby;
- thumbs-up ×3 to resume.

A HUD window (`hud.py`) shows telemetry. Everything is tunable in `config.py`.
See `README.md` and `docs/ARCHITECTURE.md` (whose numbered "pitfalls" table records
past design decisions).

## Environment
- Windows 11, Python 3.13 (`python`), Git Bash and PowerShell both available.
- Git remote `origin` = https://github.com/mohnasi/gesture-virtual-keyboard, branch `main`
  tracks `origin/main`. The GitHub CLI (`gh`) is **not** installed.
- Dev deps (`requirements-dev.txt`: pytest, ruff, …) were installed into the
  global Python install on 2026-10-05.
- Checks: `python -m pytest -q` (139 tests, hardware-free, synthetic hands in
  `tests/conftest.py`) and `python -m ruff check .`
- The files use LF line endings, and git prints harmless "LF will be replaced by CRLF" warnings.

## Session 2026-10-05: fingertip-based swipe detection

### Problem
Swipes were unreliable. `SwipeDetector` only followed the **wrist** pixel position,
and only typed a key if the stroke's majority pose was exactly `OPEN_PALM`. Likely causes:
1. A natural swipe is a flick from the wrist, so fingertips travel far but the wrist barely moves.
2. The thumb tucks or blurs mid-sweep, so frames read as `FOUR`. The swipe was detected, but
   `EDIT_SWIPES` has no `FOUR` entry, so nothing was typed.
3. Per-frame palm scaling drifted when the hand turned.
4. Rejected strokes were silent, so there was no way to tune.

### Options considered
| Option | Decision |
|---|---|
| A. Lower the wrist thresholds and accept FOUR | Not enough on its own (still misses flicks) |
| **B. Follow the four fingertips and require them to agree** (user's idea) + guards | **Chosen** |
| C. Follow the palm centre (wrist + knuckles) | Same flick blind spot as the wrist |
| D. Trained sequence classifier on landmark history (e.g. the point-history MLP in Kazuhito00/hand-gesture-recognition-using-mediapipe, or an LSTM) | **Deferred.** Needs recorded swipe data per user. Revisit only if B still fails after tuning |

The user chose "B now, D later".

### What was implemented (uncommitted at time of writing; check `git status`)
- `gestures.py`
  - New helpers `fingertips_px(obs)`, `fingertip_reach(obs)` and `track_point(obs, mode)`.
    Fingertip pixels are rebuilt exactly as `wrist_px + landmarks[tips, :2] * palm_px`,
    because normalisation is only a translate + scale. `tracker.py` is unchanged.
  - `SwipeDetector` rewritten around `_step(a, b, palm)`, which returns per-tip travel,
    the per-tip reach change and a `rigid` flag:
    - **Tracked point** = mean of the 4 tips (`swipe_track_point="fingertips"`), or the wrist.
    - **Shape-change masking:** a step where any tip-to-wrist distance changes faster
      than `swipe_max_shape_rate` carries **no motion**. Curling into a fist or
      changing letters can't open a stroke, and relaxing right after a swipe doesn't
      spoil it. (The first attempt checked shape over the whole stroke window and
      rejected real swipes followed by a relax, hence this design.)
    - **One scale per stroke:** the median `palm_px`.
    - **`_check_tips`:** each tip's net travel must have cosine ≥ `swipe_tip_coherence`
      to the mean travel and length ≥ `swipe_tip_min_share` × the mean ("tips incoherent").
      The cumulative reach drift over rigid steps must stay ≤ `swipe_max_shape_change`
      ("hand shape changed", which catches slow curls).
    - **Pose vote** applies `swipe_pose_aliases` (`FOUR → OPEN_PALM`).
    - **Diagnostics:** `last_reject` / `last_reject_t` hold the reason for near-misses.
      Strokes under half the minimum distance count as jitter and aren't reported.
      Rejections are also logged at debug level.
  - `GestureClassifier.swipe` now exposes the detector (with a `track_mode` property).
- `config.py` (`GestureConfig`): new settings `swipe_track_point`, `swipe_tip_coherence`
  (0.80), `swipe_tip_min_share` (0.50), `swipe_max_shape_change` (0.50),
  `swipe_max_shape_rate` (3.0 palm/s) and `swipe_pose_aliases`. The existing distance and
  speed thresholds are unchanged and **not yet tuned on real video**.
- `hud.py` / `pipeline.py`: `HudSnapshot.swipe_reject(_t)`. The bottom bar shows
  "Swipe missed: <reason>" for `event_display_s`, and the trail follows `track_point`.
- Tests:
  - `conftest.observation` accepts `degrees=` (rotate about the wrist) and `shape=`.
  - The `Script` helper in `test_gestures.py` gained `flick`, `morph` and `hold(**kw)`.
  - New tests cover: wrist flick (fingertip mode yes, wrist mode no), wrist-mode arm
    sweep, quick and slow finger curls, tips disagreeing, swipe-then-relax, FOUR → Space,
    the near-miss reason, and the HUD reject message.
- Docs: the README editing-swipes section and HUD row, plus `docs/ARCHITECTURE.md` pitfall #22.

### Gotchas learned
- Index numpy with a **list**, not a tuple: `lm[(8, 12, 16, 20), :2]` is a multi-dim index.
- In synthetic tests, holds before and after a flick must use the flick's start and end angle
  (`hold(..., degrees=…)`). Otherwise the hand snaps back to 0° inside the stroke.
- Coherence can only be tested with tip motion that keeps reach constant (rotate a tip
  about the wrist). Anything that changes reach quickly is masked as shape change first.

## Session 2026-10-05 (later): swipe telemetry for tuning

### Real-camera feedback
The user said fingertip swipes "work but there's a lot of glitches" and some sweeps don't register.
`logs/gvk.log` (18:54–18:56 run) showed:
- RIGHT and DOWN typed, but **LEFT never registered**. Suspects: return-stroke suppression
  (a LEFT within 0.8 s of a RIGHT is swallowed), or short left flicks under the distance threshold.
- Repeated **ACTIVE → IDLE "no hand for 2.0s"** drops, as fast sweeps lose tracking.
- Rejection reasons were DEBUG-only, so they were not in the log.

### What was implemented
- `SwipeDetector` split into `_measure` (every metric) → `_judge` (outcome, reason in check
  order) → `_evaluate` (side effects, HUD reason, hook). Behaviour is unchanged except:
  - **Hand lost mid-stroke** (left frame or tracking dropped): the stroke is now evaluated on the
    frames seen so far (`ended="hand_lost"`) instead of being thrown away.
  - Return-suppressed and refractory strokes now set `last_reject`, so the HUD shows
    "return stroke ignored" or "too soon after last swipe".
- `SwipeDetector.on_stroke` is an optional hook called with one dict per closed stroke.
- New `swipe_log.py` → `SwipeCsvLog` (append, header once, flush per row).
  `python main.py --swipe-log [CSV]` (default `logs/swipes.csv`, git-ignored via `logs/`).
  Columns: outcome, reason, direction, pose, pose_votes, ended, duration, frames, distance,
  dx, dy, path, straightness, axis_ratio, peak_speed, tip_cos_min, tip_share_min,
  shape_drift, rigid_share, palm_px.
- Tests: sweep out of frame, log hook outcomes (swipe / return_suppressed / rejected), CSV
  writer + flag parsing. 142 tests pass.

## Session 2026-10-05 (evening): tuning from the first swipe log

### Data (`logs/swipes.csv`, about 3.5 min, 164 strokes; analysed together with `logs/gvk.log`)
- 50 typed, 72 rejected, 38 jitter, 5 return-suppressed, 2 refractory.
- **Main problem: 37 strokes rejected for duration (0.90–0.95 s), 34 of them closed by timeout.**
  These were otherwise clean: 35 of 37 passed every other check, with median distance 2.7 palm.
  Two causes:
  - the stroke anchor reached back to the last rest frame, so slow drift before a sweep inflated
    the duration;
  - a timed-out stroke always failed the duration check, and its leftover often became a short
    same-direction stroke rejected for distance (a "split stroke").
- LEFT: 11 typed, 23 rejected (15 duration, 6 distance), 3 return-suppressed. LEFT is mostly the same
  timeout problem. Return-suppressed gaps were 0.65–0.77 s, and the user retried right
  after, so those strokes were intended.
- Distance rejects: median 0.75. Many are leftovers or return motions, so they're mostly correct.
- `hand_lost` evaluation (added last session) rescued about 12 typed swipes.
- Typed-stroke poses: 47 OPEN_PALM, 2 FIST, 1 PINCH. The last 3 are unmapped, so nothing typed. Not addressed yet.
- **Letters:** `models/asl_rf.pkl` knows only A, B, C (300 samples, 1 signer). 56 'a' were typed. 17 were
  under 1 s after a swipe and 10 under 0.4 s, which is impossible with a 0.4 s dwell unless the
  sign started during the stroke. A wrist flick barely moves the wrist, so the letter
  speed gate (wrist-based) didn't block them. It's unclear how many of the other 'a's were deliberate.

### Changes
- `GestureConfig`:
  - `swipe_max_duration_s` 0.90 → **1.20**
  - new `swipe_anchor_lookback_s` = **0.20** (stroke t0 = max(last rest, now − 0.2 s))
  - `swipe_min_distance` 1.1 → **1.0**
  - `return_suppress_s` 0.8 → **0.6**
- `AslConfig.after_swipe_s` = **0.40**: `LetterDetector` blocks letters while
  `SwipeDetector.busy(t, after_s)` (stroke in flight or closed < 0.4 s ago). The classifier
  wires `letters.swipe = self.swipe`.
- Tests (each confirmed to fail under the old settings): drift-before-sweep gives one stroke,
  a 1 s slow sweep is accepted, and letters pause around a swipe. 145 tests pass.

## Session 2026-10-05 (night): verification run after tuning
The user reported it "seems much smoother" and liked the letter pause. Session 2 in
`logs/swipes.csv` (rows after the first 164; about 68 s, 24 strokes) compared with session 1:
- Timeouts: 34 → 0. Duration rejects: 37 → 2.
- Mapped directions typed: LEFT 4/5, DOWN 3/5, RIGHT 2/6. 2 of the RIGHT misses were return
  motions after a LEFT; see the next point.
- Remaining rejects look **correct**:
  - 2 UP "hand shape changed" (drift 0.58–0.60; UP is unmapped anyway);
  - short DOWN/UP return motions;
  - 2 RIGHT strokes at 1.24 s. One (d=4.2, right after a LEFT) is clearly the hand coming back.
  **Don't raise `swipe_max_duration_s` further**: the 1.2 s cap is what stops slow return
  sweeps from typing.
- Return-suppressed: 2 RIGHT after LEFT. One was retried at once (intended), the other wasn't
  (a real return). 0.6 s is a reasonable balance.
- Letters: no 'a' within 0.8 s after a swipe (session 1 had 10 within 0.4 s). One 'a' came 0.9 s after a
  Space, which may be accidental (block 0.4 s + dwell 0.4 s). Watch for this; raise `after_swipe_s` if it recurs.
- 3 ACTIVE → IDLE "no hand for 2.0s" drops, which may just be the user lowering their hand.

## Open items / next steps
1. **Done (night session); thresholds look good.** For a further session, re-run `python main.py --swipe-log` (it appends to the same CSV,
   so compare rows after the new session start) and check:
   - duration and timeout rejects are gone;
   - LEFT hit rate improves;
   - no swipes fire during letters;
   - no stray letters appear after swipes.
   Possible follow-ups: alias FIST or PINCH votes during fast strokes, IDLE drops
   (`no_hand_timeout_s`), and retraining the ASL model on more letters and signers.
2. Commit and push the swipe work once you're happy with it (`git push`; the browser login may prompt).
3. Option D, only if needed: extend `tools/record_samples.py` to record short swipe
   sequences (plus negatives: fingerspelling, curls, return strokes), train a sequence
   classifier following `tools/train_asl.py`, and save it under `models/`. Use the
   fingertip trajectory features from `SwipeDetector` as inputs.
