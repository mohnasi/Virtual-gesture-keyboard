#!/usr/bin/env python
"""
Personal ASL calibration: record YOUR hand for letters A-Z with the webcam.

    python tools/record_samples.py                 # all 26 letters
    python tools/record_samples.py --letters MNST --append   # redo weak letters
    python tools/record_samples.py --train         # record, then train immediately

For every letter the window shows "Pose for letter: X", counts down 3 s and
then records ~100 frames of the held handshape (only frames where a hand is
detected count). It uses the same camera settings, MediaPipe tracker and
``features.asl_features`` as the live keyboard, so the samples are exactly
what the classifier will see at runtime. Left hands are mirrored to a right
hand inside ``asl_features`` (the same ``mirror_left`` option that is stored
in the trained model), so either hand works.

Keys while the window has focus:
    R  redo the current letter      B  go back one letter
    S  skip the current letter      Q / Esc  save what you have and quit

Output: ``data/features/user_samples.npz`` with ``X`` (N x 73), ``y`` (letter),
``group`` (default "user") and ``segment`` (which fifth of each letter's
recording a frame came from - train_asl.py uses it for honest validation when
there is only one person).
"""
from __future__ import annotations

import argparse
import json
import string
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from config import DEFAULT_CONFIG  # noqa: E402
from features import DEFAULT_OPTIONS, FEATURE_VERSION, asl_features, feature_size  # noqa: E402
from tracker import HandObservation  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "features" / "user_samples.npz"
WINDOW = "ASL calibration - record your hand"
MOTION_LETTERS = {"J", "Z"}
N_SEGMENTS = 5

_WHITE, _BLACK, _GREY = (255, 255, 255), (0, 0, 0), (170, 170, 170)
_GREEN, _AMBER, _RED = (113, 204, 46), (15, 196, 241), (60, 76, 231)   # BGR
_FONT = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------------- #
# Recording logic (pure: no camera, no window -> unit-testable)
# --------------------------------------------------------------------------- #
class Phase(str, Enum):
    COUNTDOWN = "COUNTDOWN"
    RECORDING = "RECORDING"
    LETTER_DONE = "LETTER_DONE"
    FINISHED = "FINISHED"


@dataclass
class RecordingSession:
    letters: Sequence[str]
    target: int = 100                    # valid frames per letter
    countdown_s: float = 3.0
    done_pause_s: float = 0.8            # "Got it!" before the next letter
    stride: int = 2                      # keep every Nth valid frame (more variety)
    options: Dict = field(default_factory=lambda: dict(DEFAULT_OPTIONS))

    def __post_init__(self) -> None:
        self.letters = [c.upper() for c in self.letters]
        self.idx = 0
        self.phase = Phase.COUNTDOWN
        self.phase_t0: Optional[float] = None
        self.samples: Dict[str, List[np.ndarray]] = {c: [] for c in self.letters}
        self.no_hand = False
        self.last_handedness: Optional[str] = None
        self._valid_seen = 0

    # ------------------------------------------------------------ state -- #
    @property
    def finished(self) -> bool:
        return self.phase == Phase.FINISHED

    @property
    def letter(self) -> Optional[str]:
        return None if self.finished else self.letters[self.idx]

    @property
    def count(self) -> int:
        return len(self.samples[self.letter]) if self.letter else 0

    @property
    def progress(self) -> float:
        return min(1.0, self.count / self.target)

    def countdown_left(self, now: float) -> float:
        if self.phase != Phase.COUNTDOWN or self.phase_t0 is None:
            return 0.0
        return max(0.0, self.countdown_s - (now - self.phase_t0))

    # ---------------------------------------------------------- driving -- #
    def start(self, now: float) -> None:
        self.phase_t0 = now

    def step(self, now: float, obs: HandObservation) -> None:
        if self.finished:
            return
        if self.phase_t0 is None:
            self.phase_t0 = now
        self.no_hand = not obs.present
        if obs.present:
            self.last_handedness = obs.handedness

        if self.phase == Phase.COUNTDOWN:
            if now - self.phase_t0 >= self.countdown_s:
                self._enter(Phase.RECORDING, now)
        elif self.phase == Phase.RECORDING:
            if not obs.present:
                return                                   # skip, alert is drawn
            self._valid_seen += 1
            if (self._valid_seen - 1) % max(1, self.stride):
                return
            self.samples[self.letter].append(asl_features(obs, self.options))
            if self.count >= self.target:
                self._enter(Phase.LETTER_DONE, now)
        elif self.phase == Phase.LETTER_DONE:
            if now - self.phase_t0 >= self.done_pause_s:
                self._goto(self.idx + 1, now)

    # ------------------------------------------------------------- keys -- #
    def redo(self, now: float) -> None:
        if not self.finished:
            self._goto(self.idx, now)

    def back(self, now: float) -> None:
        self._goto(max(0, (len(self.letters) if self.finished else self.idx) - 1), now)

    def skip(self, now: float) -> None:
        if not self.finished:
            self.samples[self.letter] = []
            self._goto(self.idx + 1, now)

    def finish(self) -> None:
        if self.phase == Phase.RECORDING and self.count < self.target:
            self.samples[self.letter] = []               # don't keep a partial letter
        self.phase = Phase.FINISHED

    # ---------------------------------------------------------- results -- #
    def recorded_letters(self) -> List[str]:
        return [c for c in self.letters if self.samples[c]]

    def dataset(self, group: str = "user") -> Dict[str, np.ndarray]:
        X, y, seg = [], [], []
        for c in self.letters:
            rows = self.samples[c]
            for i, row in enumerate(rows):
                X.append(row)
                y.append(c)
                seg.append(i * N_SEGMENTS // max(1, len(rows)))
        n_feat = feature_size(self.options)
        return {
            "X": np.asarray(X, dtype=np.float32).reshape(-1, n_feat),
            "y": np.asarray(y, dtype=str),
            "group": np.full(len(y), group, dtype=object).astype(str),
            "segment": np.asarray(seg, dtype=np.int64),
        }

    # ---------------------------------------------------------- helpers -- #
    def _enter(self, phase: Phase, now: float) -> None:
        self.phase, self.phase_t0 = phase, now

    def _goto(self, idx: int, now: float) -> None:
        if idx >= len(self.letters):
            self.phase = Phase.FINISHED
            return
        self.idx = idx
        self.samples[self.letters[idx]] = []
        self._valid_seen = 0
        self._enter(Phase.COUNTDOWN, now)


# --------------------------------------------------------------------------- #
# Overlay (pure: frame + session -> image)
# --------------------------------------------------------------------------- #
def _text(img, text, org, scale=0.6, color=_WHITE, thick=1) -> None:
    cv2.putText(img, text, (org[0] + 1, org[1] + 1), _FONT, scale, _BLACK, thick, cv2.LINE_AA)
    cv2.putText(img, text, org, _FONT, scale, color, thick, cv2.LINE_AA)


def _center(img, text, y, scale, color, thick=2) -> None:
    w = cv2.getTextSize(text, _FONT, scale, thick)[0][0]
    _text(img, text, ((img.shape[1] - w) // 2, y), scale, color, thick)


def render(frame: np.ndarray, session: RecordingSession, obs: Optional[HandObservation],
           now: float) -> np.ndarray:
    from hud import draw_hand_skeleton, landmarks_to_pixels

    img = frame.copy()
    H, W = img.shape[:2]
    rec_color = {Phase.COUNTDOWN: _AMBER, Phase.RECORDING: _GREEN,
                 Phase.LETTER_DONE: _GREEN}.get(session.phase, _GREY)
    if obs is not None and obs.present:
        draw_hand_skeleton(img, landmarks_to_pixels(obs), rec_color)

    # top banner
    img[0:56] = cv2.convertScaleAbs(img[0:56], alpha=0.35)
    if session.finished:
        _text(img, "All done - saving...", (12, 36), 0.8, _GREEN, 2)
        return img
    letter = session.letter
    n, total = session.idx + 1, len(session.letters)
    _text(img, f"Pose for letter: {letter}", (12, 36), 0.9, _WHITE, 2)
    _text(img, f"{n}/{total}", (W - 70, 36), 0.6, _GREY)

    if letter in MOTION_LETTERS:
        img[58:84] = cv2.convertScaleAbs(img[58:84], alpha=0.35)
        _text(img, f"Hold static pose for {letter}; do not draw in air", (12, 77), 0.55, _AMBER)

    if session.phase == Phase.COUNTDOWN:
        left = session.countdown_left(now)
        _center(img, f"{int(np.ceil(left)) if left > 0 else 0}", H // 2 + 30, 3.0, _AMBER, 6)
        _center(img, "get ready - form the handshape", H // 2 + 70, 0.6, _WHITE, 1)
    elif session.phase == Phase.LETTER_DONE:
        _center(img, f"Got it: {letter}", H // 2 + 20, 1.4, _GREEN, 3)
    elif session.phase == Phase.RECORDING:
        _text(img, "REC", (W - 140, 36), 0.6, _RED, 2)
        cv2.circle(img, (W - 155, 30), 7, _RED, -1, cv2.LINE_AA)
        if session.no_hand:
            _center(img, "No hand detected - keep one hand in view", H // 2, 0.7, _RED, 2)
        elif session.last_handedness == "Left":
            _text(img, "left hand (mirrored to right)", (12, H - 66), 0.5, _GREY)

    # bottom: progress bar + key help
    img[H - 52:H] = cv2.convertScaleAbs(img[H - 52:H], alpha=0.35)
    bx0, bx1, by = 12, W - 110, H - 40
    cv2.rectangle(img, (bx0, by), (bx1, by + 12), _GREY, 1)
    fill = int((bx1 - bx0) * session.progress)
    if fill:
        cv2.rectangle(img, (bx0, by), (bx0 + fill, by + 12), rec_color, -1)
    _text(img, f"{session.count}/{session.target}", (bx1 + 10, by + 11), 0.5, _WHITE)
    _text(img, "R redo   B back   S skip   Q save & quit", (12, H - 10), 0.45, _GREY)
    return img


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #
def save_samples(data: Dict[str, np.ndarray], out: Path, options: Dict,
                 append: bool = False) -> Dict[str, np.ndarray]:
    """Write the .npz. With ``append``, letters recorded now replace the same
    letters in the existing file and every other letter is kept."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        if append:
            with np.load(out, allow_pickle=False) as old:
                if int(old["feature_version"]) != FEATURE_VERSION or \
                        json.loads(str(old["feature_options"])) != options:
                    raise SystemExit(f"{out} was recorded with different features; "
                                     "record without --append")
                keep = ~np.isin(old["y"], np.unique(data["y"]))
                seg_old = old["segment"] if "segment" in old.files else np.zeros(len(old["y"]), np.int64)
                data = {
                    "X": np.concatenate([old["X"][keep], data["X"]]),
                    "y": np.concatenate([old["y"][keep], data["y"]]),
                    "group": np.concatenate([old["group"][keep], data["group"]]),
                    "segment": np.concatenate([seg_old[keep], data["segment"]]),
                }
        else:
            backup = out.with_name(out.stem + ".bak.npz")
            out.replace(backup)
            print(f"previous recording kept as {backup.name}")
    np.savez_compressed(out, **data, feature_version=np.int64(FEATURE_VERSION),
                        feature_options=np.asarray(json.dumps(options)))
    return data


def print_next_steps(out: Path) -> None:
    try:
        rel = out.resolve().relative_to(ROOT)
    except ValueError:
        rel = out
    print("\nNext steps (run from the project folder):")
    print(f"  1. Train:   python tools/train_asl.py {rel.as_posix()}")
    print("     (optionally add dataset features, e.g. data/features/asl_alphabet.npz)")
    print("  2. Run:     python main.py")
    print("  Weak letter? Re-record just those and retrain:")
    print("     python tools/record_samples.py --letters MNST --append")


# --------------------------------------------------------------------------- #
# Interactive loop
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record your own hand for the ASL classifier")
    p.add_argument("--letters", default=string.ascii_uppercase,
                   help="letters to record, in order (default A-Z)")
    p.add_argument("--frames", type=int, default=100, help="valid frames per letter")
    p.add_argument("--countdown", type=float, default=3.0, help="seconds before recording")
    p.add_argument("--stride", type=int, default=2,
                   help="keep every Nth valid frame (2 = more natural variation)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--group", default="user", help="name for this person / session")
    p.add_argument("--append", action="store_true",
                   help="merge into an existing file (re-recorded letters are replaced)")
    p.add_argument("--camera", type=int, default=DEFAULT_CONFIG.camera.index)
    p.add_argument("--tracker", default="auto", choices=["auto", "solutions", "tasks"])
    p.add_argument("--train", action="store_true",
                   help="run tools/train_asl.py on the saved samples when done")
    p.add_argument("--model-out", type=Path, default=ROOT / "models" / "asl_rf.pkl",
                   help="where --train writes the model (default models/asl_rf.pkl)")
    return p.parse_args(argv)


def _open_tracker(backend: str):
    import dataclasses

    from tracker import HandTracker

    cfg = dataclasses.replace(DEFAULT_CONFIG.tracker, backend=backend, max_num_hands=1)
    return HandTracker(cfg, DEFAULT_CONFIG.camera.width, DEFAULT_CONFIG.camera.height)


def main(argv=None, *, camera=None, tracker=None,
         clock: Callable[[], float] = time.monotonic) -> int:
    import dataclasses

    from camera import open_capture, prepare_frame
    from hud import display_available

    args = parse_args(argv)
    letters = [c for c in args.letters.upper() if c in string.ascii_uppercase]
    if not letters:
        print("error: --letters must contain A-Z", file=sys.stderr)
        return 2
    if not display_available():
        print("error: no display available for the recording window", file=sys.stderr)
        return 2

    cam_cfg = dataclasses.replace(DEFAULT_CONFIG.camera, index=args.camera)
    cap = camera or open_capture(cam_cfg)
    if cap is None:
        print(f"error: could not open camera {args.camera}", file=sys.stderr)
        return 1
    own_tracker = tracker is None
    try:
        tracker = tracker or _open_tracker(args.tracker)
    except Exception:
        cap.release()
        raise

    session = RecordingSession(letters, target=args.frames, countdown_s=args.countdown,
                               stride=args.stride)
    print(f"Recording {len(letters)} letter(s): {''.join(letters)}  "
          f"({args.frames} frames each). Press Q in the window to stop early.")
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    failures = 0
    try:
        session.start(clock())
        while not session.finished:
            ok, img = cap.read()
            if not ok or img is None:
                failures += 1
                if failures > 30:
                    print("error: camera stopped delivering frames", file=sys.stderr)
                    session.finish()
                    break
                continue
            failures = 0
            img = prepare_frame(img, cam_cfg)            # 640x360, mirrored like the daemon
            now = clock()
            obs = tracker.process(img, now)
            prev = (session.letter, session.phase)
            session.step(now, obs)
            if (session.letter, session.phase) != prev and session.phase == Phase.LETTER_DONE:
                print(f"  {prev[0]}: {session.count} samples")
            cv2.imshow(WINDOW, render(img, session, obs, now))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                session.finish()
            elif key in (ord("r"), ord("R")):
                session.redo(clock())
            elif key in (ord("b"), ord("B")):
                session.back(clock())
            elif key in (ord("s"), ord("S")):
                session.skip(clock())
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                session.finish()                          # window closed with [X]
    finally:
        cap.release()                                     # always hand the camera back
        if own_tracker:
            tracker.close()
        try:
            cv2.destroyWindow(WINDOW)
            cv2.waitKey(1)
        except cv2.error:
            pass

    recorded = session.recorded_letters()
    if not recorded:
        print("nothing recorded - no file written")
        return 1
    data = save_samples(session.dataset(args.group), args.out, session.options, args.append)
    counts = {c: int((data["y"] == c).sum()) for c in sorted(set(data["y"].tolist()))}
    missing = [c for c in string.ascii_uppercase if c not in counts]
    print(f"\nSaved {len(data['y'])} samples for {len(counts)} letter(s) to {args.out}")
    print("  " + "  ".join(f"{c}:{n}" for c, n in counts.items()))
    if missing:
        print(f"  not recorded yet: {''.join(missing)}")

    if args.train:
        from tools import train_asl
        print("\nTraining ...")
        if train_asl.main([str(args.out), "--out", str(args.model_out)]) == 0:
            return 0                     # train_asl printed the next steps
        print("training failed - see the messages above")
    print_next_steps(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
