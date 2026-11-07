from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cosmofire.features import (  # noqa: E402
    AF_FEATURE_NAMES,
    BS_FEATURE_NAMES,
    extract_af_features,
    extract_bs_features,
    read_mask,
)
from cosmofire.metrics import af_f1, binary_counts, iou_from_counts  # noqa: E402
from cosmofire.model_io import save_native_model  # noqa: E402


def split_af(meta: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    # AF fire_event_id is empty. Group equal spatial cells so repeated dates of the same
    # location cannot leak between train and validation.
    groups = (
        meta["epsg"].astype(str) + "_" + meta["x_min"].astype(str) + "_" + meta["y_min"].astype(str)
    )
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, val_idx = next(splitter.split(meta, groups=groups))
    return meta.iloc[train_idx].reset_index(drop=True), meta.iloc[val_idx].reset_index(drop=True)


def split_bs(meta: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = meta["fire_event_id"].fillna(meta["chip_id"])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, val_idx = next(splitter.split(meta, groups=groups))
    return meta.iloc[train_idx].reset_index(drop=True), meta.iloc[val_idx].reset_index(drop=True)


def collect_af_samples(data_dir: Path, meta: pd.DataFrame, rng: np.random.Generator, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for row in meta.itertuples(index=False):
        x, _ = extract_af_features(data_dir, row.chip_id)
        y = read_mask(data_dir, "af", row.chip_id).reshape(-1)
        valid = y != 255
        positive = np.flatnonzero(valid & (y == 1))
        negative = np.flatnonzero(valid & (y == 0))
        # Hard negatives look hot in I4 and I4-I5.
        hard_score = x[negative, 3] + x[negative, 7]
        hard_n = min(cfg["negative_hard_per_chip"], negative.size)
        hard = negative[np.argpartition(hard_score, -hard_n)[-hard_n:]] if hard_n else np.empty(0, dtype=int)
        remaining = np.setdiff1d(negative, hard, assume_unique=False)
        random_n = min(cfg["negative_random_per_chip"], remaining.size)
        random_neg = rng.choice(remaining, size=random_n, replace=False) if random_n else np.empty(0, dtype=int)
        # Repeat positives to make the rare class visible while preserving hard negatives.
        selected = np.concatenate([positive, positive, positive, hard, random_neg])
        xs.append(x[selected])
        ys.append(y[selected])
    return np.concatenate(xs), np.concatenate(ys).astype(np.uint8)


def train_af(data_dir: Path, weights_dir: Path, cfg: dict, seed: int) -> None:
    meta = pd.read_csv(data_dir / "af" / "meta.csv")
    train_meta, val_meta = split_af(meta, seed)
    rng = np.random.default_rng(seed)
    x_train, y_train = collect_af_samples(data_dir, train_meta, rng, cfg)
    print(f"AF samples: {x_train.shape}, positives={y_train.mean():.5f}, val chips={len(val_meta)}", flush=True)

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=cfg["num_boost_round"],
        learning_rate=cfg["learning_rate"],
        num_leaves=cfg["num_leaves"],
        max_depth=-1,
        min_child_samples=40,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_train, y_train, feature_name=AF_FEATURE_NAMES)

    probabilities, targets = [], []
    for row in val_meta.itertuples(index=False):
        x, _ = extract_af_features(data_dir, row.chip_id)
        y = read_mask(data_dir, "af", row.chip_id).reshape(-1)
        valid = y != 255
        probabilities.append(model.predict_proba(x[valid])[:, 1])
        targets.append(y[valid])

    p = np.concatenate(probabilities)
    y = np.concatenate(targets)
    best = (-1.0, 0.5)
    for threshold in np.linspace(0.05, 0.95, 91):
        score = af_f1(y, p >= threshold)
        if score > best[0]:
            best = (score, float(threshold))

    metadata = {
        "threshold": best[1],
        "feature_names": AF_FEATURE_NAMES,
        "validation_f1": best[0],
        "seed": seed,
    }
    save_native_model(model, weights_dir, "af", metadata)
    print(f"AF validation F1={best[0]:.6f}, threshold={best[1]:.3f}")


def collect_bs_samples(data_dir: Path, meta: pd.DataFrame, rng: np.random.Generator, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    quotas = {
        0: max(1, cfg["background_samples"] // len(meta)),
        1: max(1, cfg["samples_per_class"] // len(meta)),
        2: max(1, cfg["samples_per_class"] // len(meta)),
        3: max(1, cfg["samples_per_class"] // len(meta)),
    }
    xs, ys = [], []
    for row in meta.itertuples(index=False):
        x, _ = extract_bs_features(data_dir, row.chip_id)
        y = read_mask(data_dir, "bs", row.chip_id).reshape(-1)
        for class_id in range(4):
            indices = np.flatnonzero(y == class_id)
            count = min(quotas[class_id], indices.size)
            if count:
                chosen = rng.choice(indices, count, replace=False)
                xs.append(x[chosen])
                ys.append(np.full(count, class_id, dtype=np.uint8))
    return np.concatenate(xs), np.concatenate(ys)


def train_bs(data_dir: Path, weights_dir: Path, cfg: dict, seed: int) -> None:
    meta = pd.read_csv(data_dir / "bs" / "meta.csv")
    train_meta, val_meta = split_bs(meta, seed)
    rng = np.random.default_rng(seed)
    x_train, y_train = collect_bs_samples(data_dir, train_meta, rng, cfg)
    print(f"BS samples: {x_train.shape}, counts={np.bincount(y_train)}, val chips={len(val_meta)}", flush=True)

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=4,
        n_estimators=cfg["num_boost_round"],
        learning_rate=cfg["learning_rate"],
        num_leaves=cfg["num_leaves"],
        min_child_samples=50,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_train, y_train, feature_name=list(BS_FEATURE_NAMES))

    thresholds = np.linspace(0.20, 0.80, 31)
    burn_counts = np.zeros((len(thresholds), 3), dtype=np.int64)
    class_counts = np.zeros((len(thresholds), 3, 3), dtype=np.int64)
    for row in val_meta.itertuples(index=False):
        x, _ = extract_bs_features(data_dir, row.chip_id)
        y = read_mask(data_dir, "bs", row.chip_id).reshape(-1)
        valid = y != 255
        y = y[valid]
        p = model.predict_proba(x[valid])
        severity = np.argmax(p[:, 1:], axis=1) + 1
        burn_probability = 1.0 - p[:, 0]
        for threshold_idx, threshold in enumerate(thresholds):
            pred = np.where(burn_probability >= threshold, severity, 0)
            burn_counts[threshold_idx] += binary_counts(y > 0, pred > 0)
            for class_idx, class_id in enumerate((1, 2, 3)):
                class_counts[threshold_idx, class_idx] += binary_counts(y == class_id, pred == class_id)

    best = (-1.0, 0.5, 0.0, 0.0, [])
    for threshold_idx, threshold in enumerate(thresholds):
        burn_iou = iou_from_counts(*burn_counts[threshold_idx])
        class_ious = [iou_from_counts(*class_counts[threshold_idx, class_idx]) for class_idx in range(3)]
        severity_miou = float(np.mean(class_ious))
        weighted = 0.35 * burn_iou + 0.30 * severity_miou
        if weighted > best[0]:
            best = (weighted, float(threshold), burn_iou, severity_miou, class_ious)

    metadata = {
        "burn_threshold": best[1],
        "feature_names": list(BS_FEATURE_NAMES),
        "validation_burn_iou": best[2],
        "validation_severity_miou": best[3],
        "validation_class_ious": best[4],
        "seed": seed,
    }
    save_native_model(model, weights_dir, "bs", metadata)
    print(
        f"BS val burn IoU={best[2]:.6f}, severity mIoU={best[3]:.6f}, "
        f"classes={best[4]}, threshold={best[1]:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "train", help="Path to train data")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "baseline.json")
    parser.add_argument("--module", choices=["all", "af", "bs"], default="all")
    args = parser.parse_args()

    args.weights_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if args.module in ("all", "af"):
        train_af(args.data_dir, args.weights_dir, cfg["af"], cfg["seed"])
    if args.module in ("all", "bs"):
        train_bs(args.data_dir, args.weights_dir, cfg["bs"], cfg["seed"])


if __name__ == "__main__":
    main()
