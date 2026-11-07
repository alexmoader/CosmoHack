from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT / "src"))

from cosmofire.postprocessing import postprocess_bs_prediction  # noqa: E402
from cosmofire.rle import encode_rle  # noqa: E402


@dataclass(frozen=True)
class Variant:
    name: str
    burn_threshold: float
    af_threshold: float = 0.70
    min_component_size: int = 15
    sev_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    use_postprocessing: bool = True
    description: str = ""


VARIANTS: list[Variant] = [
    # 1. Postprocessed variations with optimal burn threshold
    Variant("v01_clean_t050", burn_threshold=0.50, min_component_size=15, description="Cleaned burn scar, threshold 0.50"),
    Variant("v02_clean_t053", burn_threshold=0.53, min_component_size=15, description="Cleaned burn scar, threshold 0.53 (recommended)"),
    Variant("v03_clean_t055", burn_threshold=0.55, min_component_size=15, description="Cleaned burn scar, threshold 0.55"),
    Variant("v04_clean_t057", burn_threshold=0.57, min_component_size=15, description="Cleaned burn scar, threshold 0.57"),
    Variant("v05_clean_t060", burn_threshold=0.60, min_component_size=15, description="Cleaned burn scar, threshold 0.60"),
    # 2. Area filtering ablations
    Variant("v06_min_size10", burn_threshold=0.53, min_component_size=10, description="Conservative area filter 10 px"),
    Variant("v07_min_size20", burn_threshold=0.53, min_component_size=20, description="Aggressive area filter 20 px"),
    # 3. Severity re-weighting
    Variant("v08_boost_sev1", burn_threshold=0.53, min_component_size=15, sev_weights=(1.15, 1.0, 1.0), description="Boosted low-severity weight 1.15"),
    Variant("v09_boost_sev2", burn_threshold=0.53, min_component_size=15, sev_weights=(1.0, 1.10, 1.0), description="Boosted med-severity weight 1.10"),
    # 4. AF threshold variations
    Variant("v10_af_t065", burn_threshold=0.53, af_threshold=0.65, min_component_size=15, description="AF threshold 0.65 higher recall"),
    Variant("v11_af_t075", burn_threshold=0.53, af_threshold=0.75, min_component_size=15, description="AF threshold 0.75 higher precision"),
]


def load_cache(cache_dir: Path, kind: str, chip_id: str) -> dict[str, np.ndarray]:
    path = cache_dir / kind / f"{chip_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction cache: {path}. Run cache_predictions.py first.")
    with np.load(path) as data:
        return {key: data[key].astype(np.float32) for key in data.files}


def create_variant(
    variant: Variant,
    meta: pd.DataFrame,
    template: pd.DataFrame,
    cache_dir: Path,
    output_dir: Path,
) -> dict[str, str | int | float]:
    encoded: dict[tuple[str, int], str] = {}
    af_pixels, sev1_pixels, sev2_pixels, sev3_pixels = 0, 0, 0, 0

    for row in meta.itertuples(index=False):
        if row.kind == "af":
            cache = load_cache(cache_dir, "af", row.chip_id)
            prob = cache["probability"]
            mask = (prob >= variant.af_threshold).astype(np.uint8)
            af_pixels += int(mask.sum())
            encoded[(row.chip_id, 1)] = encode_rle(mask == 1)
        else:
            cache = load_cache(cache_dir, "bs", row.chip_id)
            burn_prob = cache["burn"]
            sev_probs = cache["severity"]

            if variant.use_postprocessing:
                scl_post = cache.get("scl_post")
                landcover = cache.get("landcover")
                pred = postprocess_bs_prediction(
                    burn_prob=burn_prob,
                    severity_logits=sev_probs,
                    burn_threshold=variant.burn_threshold,
                    min_component_size=variant.min_component_size,
                    scl_post=scl_post,
                    landcover=landcover,
                    severity_weights=variant.sev_weights,
                )
            else:
                sev = np.argmax(sev_probs, axis=-1).astype(np.uint8) + 1
                pred = np.where(burn_prob >= variant.burn_threshold, sev, 0).astype(np.uint8)

            sev1_pixels += int((pred == 1).sum())
            sev2_pixels += int((pred == 2).sum())
            sev3_pixels += int((pred == 3).sum())

            for class_id in (1, 2, 3):
                encoded[(row.chip_id, class_id)] = encode_rle(pred == class_id)

    out_df = template[["chip_id", "class_id"]].copy()
    out_df["rle"] = [encoded[(row.chip_id, int(row.class_id))] for row in out_df.itertuples(index=False)]

    target_path = output_dir / f"{variant.name}.csv"
    out_df.to_csv(target_path, index=False)

    total_burn = sev1_pixels + sev2_pixels + sev3_pixels
    return {
        "variant": variant.name,
        "burn_threshold": variant.burn_threshold,
        "af_threshold": variant.af_threshold,
        "min_component_size": variant.min_component_size,
        "postprocess": variant.use_postprocessing,
        "af_pixels": af_pixels,
        "sev1_pixels": sev1_pixels,
        "sev2_pixels": sev2_pixels,
        "sev3_pixels": sev3_pixels,
        "total_burn_pixels": total_burn,
        "description": variant.description,
        "file": target_path.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fast submission variants from cached predictions")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "prediction_cache")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "test")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submission_variants")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(args.data_dir / "meta.csv")
    template = pd.read_csv(args.data_dir / "sample_submission.csv", dtype={"chip_id": str, "class_id": int})

    stats = []
    print(f"Generating {len(VARIANTS)} submission variants...")
    for var in VARIANTS:
        res = create_variant(var, meta, template, args.cache_dir, args.output_dir)
        stats.append(res)
        print(f"Created {var.name}: burn_px={res['total_burn_pixels']}, af_px={res['af_pixels']}")

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(args.output_dir / "variant_stats.csv", index=False)
    print(f"\nAll variants created in {args.output_dir}/")


if __name__ == "__main__":
    main()
