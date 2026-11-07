"""
Training script for Landcover-Conditioned Mixture of Experts (MoE).
Trains separate models for Cropland (40) vs Nature (Forest/Steppe/Shrubland).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cosmofire.features import (  # noqa: E402
    BS_V2_FEATURE_NAMES,
    extract_bs_aux_masks,
    extract_bs_features_v2,
    read_mask,
)
from cosmofire.mixture_of_experts import LandcoverMoE  # noqa: E402
from train_hierarchical import collect_samples  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Landcover Mixture of Experts (MoE)")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "train")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.weights_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(args.data_dir / "bs" / "meta.csv")
    rng = np.random.default_rng(args.seed)

    print("Collecting multi-source samples across all training chips...", flush=True)
    burn_x, burn_y, sev_x, sev_y = collect_samples(args.data_dir, meta, rng)

    lc_idx = BS_V2_FEATURE_NAMES.index("landcover")
    burn_lc = burn_x[:, lc_idx].astype(int)
    sev_lc = sev_x[:, lc_idx].astype(int)

    # 1. Cropland split (landcover == 40)
    crop_burn_mask = burn_lc == 40
    crop_sev_mask = sev_lc == 40

    # 2. Nature split (forest 10, shrubland 20, grassland 30, etc.)
    nature_burn_mask = ~crop_burn_mask
    nature_sev_mask = ~crop_sev_mask

    print(f"Burn samples: Cropland={crop_burn_mask.sum()}, Nature={nature_burn_mask.sum()}", flush=True)
    print(f"Severity samples: Cropland={crop_sev_mask.sum()}, Nature={nature_sev_mask.sum()}", flush=True)

    # Train Cropland Burn Model
    print("Training Cropland Burn Expert...", flush=True)
    crop_burn_lgb = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=240,
        learning_rate=0.045,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        random_state=args.seed,
        n_jobs=-1,
        verbosity=-1,
    )
    crop_burn_lgb.fit(burn_x[crop_burn_mask], burn_y[crop_burn_mask], feature_name=BS_V2_FEATURE_NAMES)

    # Train Nature Burn Model
    print("Training Nature Burn Expert...", flush=True)
    nature_burn_lgb = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=240,
        learning_rate=0.045,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=args.seed,
        n_jobs=-1,
        verbosity=-1,
    )
    nature_burn_lgb.fit(burn_x[nature_burn_mask], burn_y[nature_burn_mask], feature_name=BS_V2_FEATURE_NAMES)

    # Train Cropland Severity Model
    print("Training Cropland Severity Expert...", flush=True)
    crop_sev_lgb = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=200,
        learning_rate=0.045,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        random_state=args.seed,
        n_jobs=-1,
        verbosity=-1,
    )
    crop_sev_lgb.fit(sev_x[crop_sev_mask], sev_y[crop_sev_mask], feature_name=BS_V2_FEATURE_NAMES)

    # Train Nature Severity Model
    print("Training Nature Severity Expert...", flush=True)
    nature_sev_lgb = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=200,
        learning_rate=0.045,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=args.seed,
        n_jobs=-1,
        verbosity=-1,
    )
    nature_sev_lgb.fit(sev_x[nature_sev_mask], sev_y[nature_sev_mask], feature_name=BS_V2_FEATURE_NAMES)

    # Assemble and save MoE
    moe = LandcoverMoE(
        crop_burn_model=crop_burn_lgb.booster_,
        nature_burn_model=nature_burn_lgb.booster_,
        crop_severity_model=crop_sev_lgb.booster_,
        nature_severity_model=nature_sev_lgb.booster_,
        landcover_idx=lc_idx,
    )
    moe.save(args.weights_dir)
    print("Landcover Mixture of Experts successfully trained and saved!", flush=True)


if __name__ == "__main__":
    main()
