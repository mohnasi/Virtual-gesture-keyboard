#!/usr/bin/env python
"""
Step 2 - turn an ASL image dataset into landmark feature vectors.

Each image goes through the *same* MediaPipe tracker and normalisation the
live app uses (``tracker.HandTracker`` -> ``HandObservation.from_raw`` ->
``features.asl_features``), so the model is trained on exactly the numbers
it will see at runtime.

The label is the image's parent folder name (single letters A-Z only; folders
like ``space``, ``del`` or ``nothing`` are skipped because editing uses palm
swipes). The *group* (signer) lets ``train_asl.py`` validate on people the
model has never seen.

Examples (run from the project folder)::

    # Kaggle "ASL Alphabet" (grassknoted): one signer -> one group
    python tools/extract_landmarks.py data/raw/asl-alphabet ^
        --out data/features/asl_alphabet.npz --group asl_alphabet --max-per-class 800

    # "Spelling It Out" (Pugeault & Bowden): signer folders A..E one level down
    python tools/extract_landmarks.py data/raw/spelling-it-out ^
        --out data/features/spelling_it_out.npz --signer-level 1 --include "color_"

Check the folder layout after unzipping; ``--signer-level K`` means "the K-th
folder below the root names the signer" (0 = first folder).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from config import DEFAULT_CONFIG  # noqa: E402
from features import DEFAULT_OPTIONS, FEATURE_VERSION, asl_features, feature_size  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def label_from_path(path: Path) -> Optional[str]:
    name = path.parent.name.strip()
    return name.upper() if len(name) == 1 and name.isalpha() else None


def find_images(root: Path, include: Optional[str] = None,
                exclude: Optional[str] = None) -> List[Path]:
    inc = re.compile(include) if include else None
    exc = re.compile(exclude) if exclude else None
    out = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXTS or not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if inc and not inc.search(rel):
            continue
        if exc and exc.search(rel):
            continue
        out.append(p)
    return out


def signer_of(path: Path, root: Path, level: Optional[int], default: str) -> str:
    if level is None:
        return default
    parts = path.relative_to(root).parts[:-1]          # folders only
    return parts[level] if level < len(parts) else default


def sample_per_class(paths: Sequence[Path], max_per_class: Optional[int],
                     seed: int = 0) -> List[Path]:
    """Random (not first-N) subset per class: datasets are recorded in bursts."""
    if not max_per_class:
        return list(paths)
    by_label: Dict[str, List[Path]] = defaultdict(list)
    for p in paths:
        by_label[label_from_path(p) or "?"].append(p)
    rng = random.Random(seed)
    out: List[Path] = []
    for label in sorted(by_label):
        items = by_label[label]
        out += items if len(items) <= max_per_class else rng.sample(items, max_per_class)
    return sorted(out)


def read_image(path: Path):
    # np.fromfile + imdecode works with non-ASCII Windows paths (cv2.imread does not)
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def extract(paths: Sequence[Path], root: Path, tracker, *, options=None, flip: bool = True,
            signer_level: Optional[int] = None, group: str = "default",
            log_every: int = 500, log: Callable[[str], None] = print) -> dict:
    opts = {**DEFAULT_OPTIONS, **(options or {})}
    X, y, groups, kept = [], [], [], []
    tried, found = Counter(), Counter()
    t0 = time.perf_counter()
    for i, path in enumerate(paths, 1):
        label = label_from_path(path)
        if label is None:
            continue
        tried[label] += 1
        img = read_image(path)
        if img is None:
            continue
        if flip:                      # look like the live, mirrored selfie view
            img = cv2.flip(img, 1)
        obs = tracker.process(img, float(i))
        if not obs.present:
            continue
        found[label] += 1
        X.append(asl_features(obs, opts))
        y.append(label)
        groups.append(signer_of(path, root, signer_level, group))
        kept.append(path.relative_to(root).as_posix())
        if log_every and i % log_every == 0:
            rate = i / (time.perf_counter() - t0)
            log(f"  {i}/{len(paths)} images  ({rate:.0f} img/s, {len(X)} hands found)")
    n_feat = feature_size(opts)
    return {
        "X": np.asarray(X, dtype=np.float32).reshape(-1, n_feat),
        "y": np.asarray(y, dtype=str),
        "group": np.asarray(groups, dtype=str),
        "path": np.asarray(kept, dtype=str),
        "tried": tried,
        "found": found,
        "options": opts,
    }


def save_npz(result: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, X=result["X"], y=result["y"], group=result["group"], path=result["path"],
        feature_version=np.int64(FEATURE_VERSION),
        feature_options=np.asarray(json.dumps(result["options"])),
    )


def report(result: dict, log: Callable[[str], None] = print) -> None:
    log(f"{'class':>5} {'images':>7} {'hands':>7} {'rate':>6}")
    for label in sorted(result["tried"]):
        n, k = result["tried"][label], result["found"][label]
        flag = "   <- low" if k < 50 else ""
        log(f"{label:>5} {n:>7} {k:>7} {k / max(n, 1):>6.0%}{flag}")
    total_n, total_k = sum(result["tried"].values()), sum(result["found"].values())
    log(f"total {total_n:>7} {total_k:>7} {total_k / max(total_n, 1):>6.0%}")
    log(f"groups: {dict(Counter(result['group'].tolist()))}")


def build_tracker(min_confidence: float, backend: str):
    from tracker import HandTracker

    cfg = dataclasses.replace(DEFAULT_CONFIG.tracker, backend=backend,
                              min_detection_confidence=min_confidence)
    # static_images=True: every photo is unrelated, so no frame-to-frame tracking
    return HandTracker(cfg, DEFAULT_CONFIG.camera.width, DEFAULT_CONFIG.camera.height,
                       static_images=True)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("root", type=Path, help="dataset folder (class folders somewhere below)")
    p.add_argument("--out", type=Path, required=True, help="output .npz")
    p.add_argument("--group", default=None,
                   help="signer/group name for every sample (default: root folder name)")
    p.add_argument("--signer-level", type=int, default=None,
                   help="index of the folder (below root) that names the signer")
    p.add_argument("--include", default=None, help="regex a relative path must match")
    p.add_argument("--exclude", default=None, help="regex that drops a relative path")
    p.add_argument("--max-per-class", type=int, default=None,
                   help="random subset per letter (e.g. 800) to save time")
    p.add_argument("--no-flip", action="store_true",
                   help="do not mirror images (only if the dataset is already selfie-view)")
    p.add_argument("--no-mirror-left", action="store_true",
                   help="keep left hands as-is instead of mirroring them to right hands")
    p.add_argument("--no-z", action="store_true", help="drop MediaPipe's noisy z coordinate")
    p.add_argument("--min-confidence", type=float, default=0.5)
    p.add_argument("--tracker", default="auto", choices=["auto", "solutions", "tasks"])
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None, tracker=None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        print(f"error: {root} is not a folder", file=sys.stderr)
        return 2
    paths = [p for p in find_images(root, args.include, args.exclude) if label_from_path(p)]
    skipped = Counter(p.parent.name for p in find_images(root, args.include, args.exclude)
                      if not label_from_path(p))
    if skipped:
        print(f"skipping non-letter folders: {dict(skipped)}")
    paths = sample_per_class(paths, args.max_per_class, args.seed)
    if not paths:
        print("error: no images found under A..Z folders", file=sys.stderr)
        return 2
    print(f"{len(paths)} images from {root}")

    own_tracker = tracker is None
    tracker = tracker or build_tracker(args.min_confidence, args.tracker)
    try:
        result = extract(
            paths, root, tracker,
            options={"mirror_left": not args.no_mirror_left, "use_z": not args.no_z},
            flip=not args.no_flip, signer_level=args.signer_level,
            group=args.group or root.name)
    finally:
        if own_tracker:
            tracker.close()
    report(result)
    if len(result["y"]) == 0:
        print("error: MediaPipe found no hands - check the images", file=sys.stderr)
        return 1
    save_npz(result, args.out)
    print(f"wrote {args.out} ({len(result['y'])} samples x {result['X'].shape[1]} features)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
