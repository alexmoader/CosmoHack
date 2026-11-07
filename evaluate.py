from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cosmofire.features import (  # noqa: E402
    extract_af_features,
    extract_bs_aux_masks,
    extract_bs_features,
    extract_bs_features_v2,
    read_mask,
)
from cosmofire.metrics import af_f1, binary_counts, competition_score, iou_from_counts  # noqa: E402
from cosmofire.model_io import load_native_model  # noqa: E402
from cosmofire.postprocessing import postprocess_bs_prediction  # noqa: E402
from train import split_af, split_bs  # noqa: E402


def safe_prf(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CosmoFire models on validation split")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "train")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "evaluation.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-postprocess", action="store_true", help="Disable spatial postprocessing")
    args = parser.parse_args()

    started = time.perf_counter()
    af_model, af_metadata = load_native_model(args.weights_dir, "af")
    af_meta = pd.read_csv(args.data_dir / "af" / "meta.csv")
    _, af_val = split_af(af_meta, args.seed)

    af_true, af_pred = [], []
    for row in af_val.itertuples(index=False):
        x, _ = extract_af_features(args.data_dir, row.chip_id)
        y = read_mask(args.data_dir, "af", row.chip_id).reshape(-1)
        valid = y != 255
        p = af_model.predict(x[valid])
        threshold = af_metadata.get("threshold", 0.70)
        pred = (p >= threshold).astype(np.uint8)
        af_true.append(y[valid].astype(np.uint8))
        af_pred.append(pred)

    af_y = np.concatenate(af_true)
    af_p = np.concatenate(af_pred)
    f1_af = af_f1(af_y, af_p)
    af_details = safe_prf(af_y, af_p)

    # BS evaluation
    bs_meta = pd.read_csv(args.data_dir / "bs" / "meta.csv")
    _, bs_val = split_bs(bs_meta, args.seed)

    hierarchical = (args.weights_dir / "bs_burn_lgbm.txt").exists() and (
        args.weights_dir / "bs_severity_lgbm.txt"
    ).exists()

    burn_tp, burn_fp, burn_fn = 0, 0, 0
    sev_counts = {1: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0]}
    all_y_true = []
    all_y_pred = []

    if hierarchical:
        bs_burn_model, bs_metadata = load_native_model(args.weights_dir, "bs_burn")
        bs_severity_model, _ = load_native_model(args.weights_dir, "bs_severity")
        bs_model_name = "hierarchical_v2"
        burn_threshold = bs_metadata.get("burn_threshold", 0.55)
        min_size = bs_metadata.get("min_component_size", 15)

        for number, row in enumerate(bs_val.itertuples(index=False), start=1):
            x, shape = extract_bs_features_v2(args.data_dir, row.chip_id)
            y = read_mask(args.data_dir, "bs", row.chip_id)
            valid = y != 255

            burn_prob = bs_burn_model.predict(x).reshape(shape)
            severity_logits = bs_severity_model.predict(x).reshape(shape[0], shape[1], 3)

            if not args.no_postprocess:
                scl_post, landcover = extract_bs_aux_masks(args.data_dir, row.chip_id)
                pred = postprocess_bs_prediction(
                    burn_prob=burn_prob,
                    severity_logits=severity_logits,
                    burn_threshold=burn_threshold,
                    min_component_size=min_size,
                    scl_post=scl_post,
                    landcover=landcover,
                )
            else:
                sev = np.argmax(severity_logits, axis=-1).astype(np.uint8) + 1
                pred = np.where(burn_prob >= burn_threshold, sev, 0).astype(np.uint8)

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

            all_y_true.append(y_v)
            all_y_pred.append(p_v)

            if number % 15 == 0 or number == len(bs_val):
                print(f"Evaluated BS {number}/{len(bs_val)} chips", flush=True)

    else:
        bs_model, bs_metadata = load_native_model(args.weights_dir, "bs")
        bs_model_name = "multiclass_baseline"
        burn_threshold = bs_metadata.get("burn_threshold", 0.62)

        for number, row in enumerate(bs_val.itertuples(index=False), start=1):
            x, shape = extract_bs_features(args.data_dir, row.chip_id)
            y = read_mask(args.data_dir, "bs", row.chip_id)
            valid = y != 255

            probs = bs_model.predict(x)
            burn_prob = 1.0 - probs[:, 0]
            sev = np.argmax(probs[:, 1:], axis=1).astype(np.uint8) + 1
            pred = np.where(burn_prob >= burn_threshold, sev, 0).reshape(shape).astype(np.uint8)

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

            all_y_true.append(y_v)
            all_y_pred.append(p_v)

    burn_iou = iou_from_counts(burn_tp, burn_fp, burn_fn)
    class_ious = [iou_from_counts(*sev_counts[c]) for c in (1, 2, 3)]
    severity_miou = float(np.mean(class_ious))

    y_cat = np.concatenate(all_y_true)
    p_cat = np.concatenate(all_y_pred)
    matrix = confusion_matrix(y_cat, p_cat, labels=[0, 1, 2, 3])

    result = {
        "seed": args.seed,
        "af": {
            "validation_chips": len(af_val),
            "threshold": af_metadata["threshold"],
            "f1": f1_af,
            **af_details,
        },
        "bs": {
            "model": bs_model_name,
            "validation_chips": len(bs_val),
            "burn_threshold": burn_threshold,
            "postprocessing_enabled": not args.no_postprocess,
            "burn_iou": burn_iou,
            "severity_miou": severity_miou,
            "class_ious": {"1": class_ious[0], "2": class_ious[1], "3": class_ious[2]},
            "confusion_matrix_rows_true_cols_pred": matrix.tolist(),
        },
        "composite_score": competition_score(f1_af, burn_iou, severity_miou),
        "evaluation_seconds": time.perf_counter() - started,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
