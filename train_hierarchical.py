from __future__ import annotations

import argparse
import sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cosmofire.features import (  # noqa: E402
    BS_V2_FEATURE_NAMES,
    extract_bs_aux_masks,
    extract_bs_features_v2,
    read_mask,
)
from cosmofire.metrics import binary_counts, iou_from_counts  # noqa: E402
from cosmofire.model_io import save_native_model  # noqa: E402
from cosmofire.postprocessing import postprocess_bs_prediction  # noqa: E402


def split_bs(meta: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = meta["fire_event_id"].fillna(meta["chip_id"])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, val_idx = next(splitter.split(meta, groups=groups))
    return meta.iloc[train_idx].reset_index(drop=True), meta.iloc[val_idx].reset_index(drop=True)


def collect_samples(
    data_dir: Path, meta: pd.DataFrame, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    burn_x, burn_y, severity_x, severity_y = [], [], [], []
    dnbr_index = BS_V2_FEATURE_NAMES.index("dnbr")
    landcover_index = BS_V2_FEATURE_NAMES.index("landcover")

    for number, row in enumerate(meta.itertuples(index=False), start=1):
        x, _ = extract_bs_features_v2(data_dir, row.chip_id)
        y = read_mask(data_dir, "bs", row.chip_id).reshape(-1)
        valid = y != 255
        background = np.flatnonzero(valid & (y == 0))
        burned = np.flatnonzero(valid & (y > 0))

        # 1. Random background
        random_n = min(1500, background.size)
        random_bg = rng.choice(background, random_n, replace=False) if random_n else np.empty(0, dtype=int)
        remaining = np.setdiff1d(background, random_bg, assume_unique=False)

        # 2. Hard negatives from high dNBR (plowing, phenology)
        hard_n = min(1200, remaining.size)
        hard_bg = (
            remaining[np.argpartition(x[remaining, dnbr_index], -hard_n)[-hard_n:]]
            if hard_n
            else np.empty(0, dtype=int)
        )

        # 3. Cropland hard negatives (landcover == 40)
        crop_bg_candidates = remaining[x[remaining, landcover_index] == 40]
        crop_n = min(600, crop_bg_candidates.size)
        crop_bg = rng.choice(crop_bg_candidates, crop_n, replace=False) if crop_n else np.empty(0, dtype=int)

        # 4. Burn samples
        burn_n = min(2500, burned.size)
        selected_burn = rng.choice(burned, burn_n, replace=False) if burn_n else np.empty(0, dtype=int)

        burn_indices = np.concatenate([random_bg, hard_bg, crop_bg, selected_burn])
        burn_x.append(x[burn_indices])
        burn_y.append((y[burn_indices] > 0).astype(np.uint8))

        # Severity samples
        for class_id in (1, 2, 3):
            indices = np.flatnonzero(y == class_id)
            count = min(1500, indices.size)
            if count:
                chosen = rng.choice(indices, count, replace=False)
                severity_x.append(x[chosen])
                severity_y.append(np.full(count, class_id - 1, dtype=np.uint8))

        if number % 25 == 0:
            print(f"Collected {number}/{len(meta)} train chips", flush=True)

    return (
        np.concatenate(burn_x),
        np.concatenate(burn_y),
        np.concatenate(severity_x),
        np.concatenate(severity_y),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced hierarchical BS training with spatial post-processing")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "train")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.weights_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(args.data_dir / "bs" / "meta.csv")
    train_meta, val_meta = split_bs(meta, args.seed)

    rng = np.random.default_rng(args.seed)
    burn_x, burn_y, severity_x, severity_y = collect_samples(args.data_dir, train_meta, rng)
    print(
        f"Burn samples={burn_x.shape}, counts={np.bincount(burn_y)}; "
        f"severity samples={severity_x.shape}, counts={np.bincount(severity_y)}",
        flush=True,
    )

    categorical_features = ["landcover"]

    burn_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=220,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=args.seed,
        n_jobs=-1,
        verbosity=-1,
    )
    burn_model.fit(burn_x, burn_y, feature_name=BS_V2_FEATURE_NAMES, categorical_feature=categorical_features)
    del burn_x, burn_y

    severity_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=220,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=args.seed,
        n_jobs=-1,
        verbosity=-1,
    )
    severity_model.fit(severity_x, severity_y, feature_name=BS_V2_FEATURE_NAMES, categorical_feature=categorical_features)
    del severity_x, severity_y

    # Validation and threshold search WITH postprocessing
    thresholds = np.linspace(0.40, 0.70, 16)
    min_component_sizes = [10, 15, 20]

    val_caches = []
    print("Pre-caching validation predictions for grid search...", flush=True)
    for number, row in enumerate(val_meta.itertuples(index=False), start=1):
        x, shape = extract_bs_features_v2(args.data_dir, row.chip_id)
        y = read_mask(args.data_dir, "bs", row.chip_id)
        scl_post, landcover = extract_bs_aux_masks(args.data_dir, row.chip_id)
        burn_prob = burn_model.predict_proba(x)[:, 1].reshape(shape)
        sev_probs = severity_model.predict_proba(x).reshape(shape[0], shape[1], 3)
        val_caches.append({
            "y": y,
            "burn_prob": burn_prob,
            "sev_probs": sev_probs,
            "scl_post": scl_post,
            "landcover": landcover,
        })
        if number % 15 == 0:
            print(f"Cached val chip {number}/{len(val_meta)}", flush=True)

    best = (-1.0, 0.55, 15, 0.0, 0.0, [])
    for min_size in min_component_sizes:
        for threshold in thresholds:
            burn_tp, burn_fp, burn_fn = 0, 0, 0
            sev_counts = {1: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0]}

            for item in val_caches:
                y = item["y"]
                valid = y != 255
                pred = postprocess_bs_prediction(
                    burn_prob=item["burn_prob"],
                    severity_logits=item["sev_probs"],
                    burn_threshold=threshold,
                    min_component_size=min_size,
                    scl_post=item["scl_post"],
                    landcover=item["landcover"],
                )
                y_v = y[valid]
                p_v = pred[valid]

                tp, fp, fn = binary_counts(y_v > 0, p_v > 0)
                burn_tp += tp
                burn_fp += fp
                burn_fn += fn

                for c in (1, 2, 3):
                    ctp, cfp, cfn = binary_counts(y_v == c, p_v == c)
                    sev_counts[c][0] += ctp
                    sev_counts[c][1] += cfp
                    sev_counts[c][2] += cfn

            burn_iou = iou_from_counts(burn_tp, burn_fp, burn_fn)
            class_ious = [iou_from_counts(*sev_counts[c]) for c in (1, 2, 3)]
            miou = float(np.mean(class_ious))
            obj = 0.35 * burn_iou + 0.30 * miou

            if obj > best[0]:
                best = (obj, float(threshold), int(min_size), burn_iou, miou, class_ious)
                print(f"New best: obj={obj:.6f}, t={threshold:.2f}, min_size={min_size}, burn_iou={burn_iou:.4f}, miou={miou:.4f}")

    common = {"feature_names": BS_V2_FEATURE_NAMES, "seed": args.seed, "version": "hierarchical_v3_postprocessed"}
    save_native_model(
        burn_model,
        args.weights_dir,
        "bs_burn",
        {
            **common,
            "burn_threshold": best[1],
            "min_component_size": best[2],
            "validation_burn_iou": best[3],
            "validation_severity_miou": best[4],
            "validation_class_ious": best[5],
        },
    )
    save_native_model(severity_model, args.weights_dir, "bs_severity", common)
    print(
        f"\nTRAINING COMPLETE:\nBurn IoU={best[3]:.6f}, severity mIoU={best[4]:.6f}, "
        f"classes={best[5]}, threshold={best[1]:.3f}, min_size={best[2]}, BS obj={best[0]:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
