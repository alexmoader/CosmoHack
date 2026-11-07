from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cosmofire.features import AF_FEATURE_NAMES, BS_V2_FEATURE_NAMES  # noqa: E402
from cosmofire.model_io import save_native_model  # noqa: E402
from train import collect_af_samples  # noqa: E402
from train_hierarchical import collect_samples  # noqa: E402


def train_af_full(data_dir: Path, weights_dir: Path, seed: int, cfg: dict) -> None:
    meta = pd.read_csv(data_dir / "af" / "meta.csv")
    rng = np.random.default_rng(seed)
    x, y = collect_af_samples(data_dir, meta, rng, cfg)
    print(f"FINAL AF: chips={len(meta)}, samples={x.shape}, positives={y.mean():.5f}", flush=True)

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=cfg["num_boost_round"],
        learning_rate=cfg["learning_rate"],
        num_leaves=cfg["num_leaves"],
        min_child_samples=40,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x, y, feature_name=AF_FEATURE_NAMES)
    save_native_model(
        model,
        weights_dir,
        "af",
        {
            "threshold": 0.70,
            "feature_names": AF_FEATURE_NAMES,
            "seed": seed,
            "version": "final_full_train",
            "training_scope": f"full_{len(meta)}_chips",
        },
    )


def train_bs_full(data_dir: Path, weights_dir: Path, seed: int) -> None:
    meta = pd.read_csv(data_dir / "bs" / "meta.csv")
    rng = np.random.default_rng(seed)
    burn_x, burn_y, severity_x, severity_y = collect_samples(data_dir, meta, rng)
    print(
        f"FINAL BS: chips={len(meta)}, burn={burn_x.shape}/{np.bincount(burn_y)}, "
        f"severity={severity_x.shape}/{np.bincount(severity_y)}",
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
        random_state=seed,
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
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    severity_model.fit(severity_x, severity_y, feature_name=BS_V2_FEATURE_NAMES, categorical_feature=categorical_features)
    del severity_x, severity_y

    common = {
        "feature_names": BS_V2_FEATURE_NAMES,
        "seed": seed,
        "version": "hierarchical_v3_final_full_train",
        "training_scope": f"full_{len(meta)}_chips",
    }
    save_native_model(
        burn_model,
        weights_dir,
        "bs_burn",
        {
            **common,
            "burn_threshold": 0.53,
            "min_component_size": 15,
        },
    )
    save_native_model(severity_model, weights_dir, "bs_severity", common)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train production models on all labeled chips")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "train")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--module", choices=["all", "af", "bs"], default="all")
    args = parser.parse_args()

    cfg = json.loads((ROOT / "configs" / "baseline.json").read_text(encoding="utf-8"))
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    if args.module in ("all", "af"):
        train_af_full(args.data_dir, args.weights_dir, args.seed, cfg["af"])
    if args.module in ("all", "bs"):
        train_bs_full(args.data_dir, args.weights_dir, args.seed)
    print("FINAL FULL-TRAIN MODELS SAVED SUCCESSFULLY", flush=True)


if __name__ == "__main__":
    main()
