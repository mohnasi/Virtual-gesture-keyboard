#!/usr/bin/env python
"""
Step 3 - train the static ASL letter classifier (Random Forest) and save it.

    python tools/train_asl.py data/features/*.npz --out models/asl_rf.pkl

What it does
------------
1. Loads one or more ``.npz`` files from ``extract_landmarks.py`` (they must
   share the same feature version/options).
2. Cross-validates with **GroupKFold by signer** whenever there are >= 2
   signers, so the score reflects people the model has never seen. With a
   single signer it falls back to a stratified split and warns that the score
   is optimistic (near-duplicate frames leak between folds).
3. Picks the confidence threshold: the lowest value at which out-of-fold
   predictions above it are at least ``--target-precision`` correct. The live
   app only types a letter whose smoothed confidence clears it.
4. Fits the final model on all data and writes a joblib bundle (model +
   classes + threshold + feature options + versions) and a text report.

Only load model files you trained yourself: unpickling can execute code.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from features import FEATURE_VERSION  # noqa: E402


def load_features(paths: Iterable[Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    X, y, g, _, options = load_features_ex(paths)
    return X, y, g, options


def load_features_ex(paths: Iterable[Path]):
    """Like ``load_features`` but also returns ``segment`` (recording time
    block from record_samples.py) or ``None`` if any file lacks it."""
    Xs, ys, gs, segs, options = [], [], [], [], None
    for p in paths:
        with np.load(p, allow_pickle=False) as d:
            version = int(d["feature_version"])
            opts = json.loads(str(d["feature_options"]))
            if version != FEATURE_VERSION:
                raise SystemExit(f"{p}: feature_version {version} != {FEATURE_VERSION}; "
                                 "re-run extract_landmarks.py")
            if options is not None and opts != options:
                raise SystemExit(f"{p}: feature options {opts} differ from {options}")
            options = opts
            Xs.append(d["X"])
            ys.append(d["y"])
            gs.append(d["group"])
            segs.append(d["segment"] if "segment" in d.files else None)
    if not Xs:
        raise SystemExit("no feature files given")
    seg = None if any(s is None for s in segs) else np.concatenate(segs)
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs), seg, options


def choose_threshold(proba: np.ndarray, y_idx: np.ndarray,
                     target_precision: float) -> Tuple[float, float, float]:
    """Lowest threshold whose accepted predictions reach the target precision.

    Returns (threshold, precision_at_threshold, coverage)."""
    conf = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y_idx
    best = (0.0, float(correct.mean()), 1.0)
    for thr in np.round(np.arange(0.0, 1.0, 0.01), 2):
        mask = conf >= thr
        if not mask.any():
            break
        prec = float(correct[mask].mean())
        if prec >= target_precision:
            return float(thr), prec, float(mask.mean())
        if prec > best[1]:
            best = (float(thr), prec, float(mask.mean()))
    print(f"warning: precision {target_precision:.0%} not reachable; "
          f"using threshold {best[0]:.2f} ({best[1]:.1%})")
    return best


def top_confusions(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> List[str]:
    pairs = Counter((t, p) for t, p in zip(y_true, y_pred) if t != p)
    totals = Counter(y_true)
    return [f"{t} -> {p}: {n} ({n / totals[t]:.0%} of {t})" for (t, p), n in pairs.most_common(k)]


def make_model(args):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=args.trees, max_depth=args.max_depth, min_samples_leaf=args.min_leaf,
        class_weight="balanced_subsample", n_jobs=-1, random_state=args.seed)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the static ASL Random Forest")
    p.add_argument("features", nargs="+", type=Path, help=".npz files from extract_landmarks.py")
    p.add_argument("--out", type=Path, default=ROOT / "models" / "asl_rf.pkl")
    p.add_argument("--trees", type=int, default=150)
    p.add_argument("--max-depth", type=int, default=20)
    p.add_argument("--min-leaf", type=int, default=2)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--target-precision", type=float, default=0.95)
    p.add_argument("--min-threshold", type=float, default=0.5,
                   help="never type below this confidence, even if validation is perfect "
                        "(guards against relaxed / in-between hands being typed)")
    p.add_argument("--test", type=Path, nargs="*", default=[],
                   help="extra .npz evaluated only (e.g. a different dataset)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> int:
    import joblib
    import sklearn
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict

    args = parse_args(argv)
    missing_files = [p for p in args.features if not p.exists()]
    if missing_files:
        print(f"error: not found: {', '.join(map(str, missing_files))}", file=sys.stderr)
        if any(p.name == "user_samples.npz" for p in missing_files):
            print("record your hand first:  python tools/record_samples.py", file=sys.stderr)
        return 2
    X, y, groups, segments, options = load_features_ex(args.features)
    classes = np.unique(y)
    n_groups = len(np.unique(groups))
    print(f"{len(y)} samples, {len(classes)} classes, {n_groups} signer group(s), "
          f"{X.shape[1]} features")
    lines = [f"ASL Random Forest report - {_dt.datetime.now():%Y-%m-%d %H:%M}",
             f"inputs: {', '.join(str(p) for p in args.features)}",
             f"samples: {len(y)}  classes: {len(classes)}  groups: {n_groups}"]

    min_class = min(Counter(y).values())
    missing_letters = sorted(set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") - set(classes.tolist()))
    if missing_letters:
        print(f"note: no samples for {''.join(missing_letters)} - those letters can't be typed")
    n_segments = len(np.unique(segments)) if segments is not None else 0
    if n_groups >= 2:
        cv = GroupKFold(n_splits=min(args.folds, n_groups))
        scheme = f"GroupKFold({cv.n_splits}) by signer - unseen-signer estimate"
        cv_kwargs = {"groups": groups}
    elif n_segments >= 2:
        # One person (record_samples.py): hold out whole time blocks of each
        # letter's recording, so neighbouring near-identical frames never sit
        # on both sides of the split.
        cv = GroupKFold(n_splits=min(args.folds, n_segments))
        scheme = (f"GroupKFold({cv.n_splits}) by recording time block - estimate for "
                  "THIS person only (not for other people)")
        cv_kwargs = {"groups": segments}
    else:
        cv = StratifiedKFold(n_splits=max(2, min(args.folds, min_class)), shuffle=True,
                             random_state=args.seed)
        scheme = ("StratifiedKFold on ONE signer group - OPTIMISTIC: near-duplicate "
                  "frames leak between folds; add a second signer or your own recordings")
        cv_kwargs = {}
        print(f"warning: {scheme}")

    print(f"cross-validating ({scheme}) ...")
    proba = cross_val_predict(make_model(args), X, y, cv=cv, method="predict_proba",
                              **cv_kwargs)
    y_idx = np.searchsorted(classes, y)
    pred = classes[proba.argmax(axis=1)]
    acc = float(accuracy_score(y, pred))
    thr, prec, coverage = choose_threshold(proba, y_idx, args.target_precision)
    if thr < args.min_threshold:
        # Clean, well-separated data (typical for one person) calibrates to a
        # near-zero threshold, which would type *something* for any hand
        # shape. Keep a floor and report what it costs on validation data.
        thr = args.min_threshold
        conf = proba.max(axis=1)
        mask = conf >= thr
        correct = proba.argmax(axis=1) == y_idx
        prec = float(correct[mask].mean()) if mask.any() else float("nan")
        coverage = float(mask.mean())
        print(f"threshold raised to the --min-threshold floor {thr:.2f}")
    print(f"cross-validated accuracy {acc:.1%}; threshold {thr:.2f} -> "
          f"{prec:.1%} precision on {coverage:.0%} of frames")
    lines += [f"validation: {scheme}", f"accuracy (top-1, no threshold): {acc:.4f}",
              f"threshold: {thr:.2f}  precision above it: {prec:.4f}  coverage: {coverage:.4f}",
              "", "most frequent confusions:", *top_confusions(y, pred), "",
              classification_report(y, pred, digits=3, zero_division=0)]

    print("fitting final model on all samples ...")
    model = make_model(args).fit(X, y)
    test_metrics = {}
    for tp in args.test:
        Xt, yt, _, opts_t = load_features([tp])
        if opts_t != options:
            raise SystemExit(f"{tp}: feature options differ from training data")
        pt = model.predict_proba(Xt)
        known = np.isin(yt, model.classes_)
        pred_t = model.classes_[pt.argmax(axis=1)]
        conf_t = pt.max(axis=1)
        acc_t = float((pred_t[known] == yt[known]).mean()) if known.any() else float("nan")
        acc_mask = known & (conf_t >= thr)
        prec_t = float((pred_t[acc_mask] == yt[acc_mask]).mean()) if acc_mask.any() else float("nan")
        test_metrics[str(tp)] = {"accuracy": acc_t, "precision_at_threshold": prec_t,
                                 "coverage": float(acc_mask.mean())}
        print(f"held-out {tp.name}: accuracy {acc_t:.1%}, "
              f"{prec_t:.1%} precision on {acc_mask.mean():.0%} of samples above threshold")
        lines.append(f"held-out {tp}: {json.dumps(test_metrics[str(tp)])}")

    bundle = {
        "model": model,
        "classes": [str(c) for c in model.classes_],
        "threshold": thr,
        "feature_version": FEATURE_VERSION,
        "feature_options": options,
        "sklearn_version": sklearn.__version__,
        "trained_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "n_samples": int(len(y)),
        "n_groups": int(n_groups),
        "cv": {"scheme": scheme, "accuracy": acc, "precision_at_threshold": prec,
               "coverage": coverage},
        "test": test_metrics,
        "params": {"trees": args.trees, "max_depth": args.max_depth,
                   "min_leaf": args.min_leaf},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out, compress=3)
    report_path = args.out.with_suffix(".txt")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    size_mb = args.out.stat().st_size / 1e6
    print(f"wrote {args.out} ({size_mb:.1f} MB) and {report_path.name}")
    print_next_steps(args.out)
    return 0


def print_next_steps(model_path: Path) -> None:
    default = (ROOT / "models" / "asl_rf.pkl").resolve()
    run = "python main.py" if Path(model_path).resolve() == default \
        else f"python main.py --asl-model {model_path}"
    print("\nNext steps (run from the project folder):")
    print(f"  1. Type with ASL:      {run}")
    print("     Wake with an open palm held 2 s, then hold each letter still for 0.4 s.")
    print("  2. Improve weak letters (see the confusions in the .txt report):")
    print("       python tools/record_samples.py --letters MNST --append")
    print("       python tools/train_asl.py data/features/user_samples.npz")
    print("  3. Calibrate from scratch:  python tools/record_samples.py")


if __name__ == "__main__":
    sys.exit(main())
