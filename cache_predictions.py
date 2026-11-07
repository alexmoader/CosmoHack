from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cosmofire.features import extract_af_features, extract_bs_aux_masks, extract_bs_features_v2  # noqa: E402
from cosmofire.model_io import load_native_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache test probabilities for fast submission variants")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "test")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "prediction_cache")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    af_dir = args.cache_dir / "af"
    bs_dir = args.cache_dir / "bs"
    af_dir.mkdir(parents=True, exist_ok=True)
    bs_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.data_dir / "meta.csv")
    af_model, af_meta = load_native_model(args.weights_dir, "af")
    burn_model, burn_meta = load_native_model(args.weights_dir, "bs_burn")
    severity_model, severity_meta = load_native_model(args.weights_dir, "bs_severity")

    for index, row in enumerate(meta.itertuples(index=False), start=1):
        if row.kind == "af":
            output = af_dir / f"{row.chip_id}.npz"
            if args.force or not output.exists():
                x, shape = extract_af_features(args.data_dir, row.chip_id)
                probability = af_model.predict(x).reshape(shape).astype(np.float16)
                np.savez(output, probability=probability)
        else:
            output = bs_dir / f"{row.chip_id}.npz"
            if args.force or not output.exists():
                x, shape = extract_bs_features_v2(args.data_dir, row.chip_id)
                burn = burn_model.predict(x).reshape(shape).astype(np.float16)
                severity = severity_model.predict(x).reshape(shape[0], shape[1], 3).astype(np.float16)
                scl_post, landcover = extract_bs_aux_masks(args.data_dir, row.chip_id)
                np.savez(output, burn=burn, severity=severity, scl_post=scl_post, landcover=landcover)

        if index % 20 == 0 or index == len(meta):
            print(f"Cached {index}/{len(meta)} chips", flush=True)

    manifest = {
        "chips": len(meta),
        "af_model_version": af_meta.get("version"),
        "af_training_scope": af_meta.get("training_scope"),
        "bs_model_version": burn_meta.get("version"),
        "bs_training_scope": burn_meta.get("training_scope"),
        "severity_model_version": severity_meta.get("version"),
        "seconds": time.perf_counter() - started,
    }
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    (args.cache_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Prediction cache ready: {args.cache_dir}")


if __name__ == "__main__":
    main()
