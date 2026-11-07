from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cosmofire.features import (  # noqa: E402
    extract_af_features,
    extract_bs_aux_masks,
    extract_bs_features,
    extract_bs_features_v2,
    bs_paths,
    _read,
    _safe_index,
)
from cosmofire.mixture_of_experts import LandcoverMoE  # noqa: E402
from cosmofire.model_io import load_native_model  # noqa: E402
from cosmofire.postprocessing import postprocess_bs_prediction  # noqa: E402
from cosmofire.rle import encode_rle  # noqa: E402


def predict_af(data_dir: Path, chip_id: str, model, metadata: dict) -> np.ndarray:
    x, shape = extract_af_features(data_dir, chip_id)
    probability = model.predict(x)
    threshold = metadata.get("threshold", 0.70)
    return (probability >= threshold).reshape(shape).astype(np.uint8)


def predict_bs_unet(data_dir: Path, chip_id: str, model, device) -> tuple[np.ndarray, np.ndarray]:
    import torch
    p_pre, p_post, p_s1pre, p_s1post, p_aux = bs_paths(data_dir, chip_id)
    pre = _read(p_pre)[:9].astype(np.float32) / 10000.0
    post = _read(p_post)[:9].astype(np.float32) / 10000.0
    s1pre = _read(p_s1pre).astype(np.float32) / 100.0
    s1post = _read(p_s1post).astype(np.float32) / 100.0
    aux = _read(p_aux).astype(np.float32)

    nbr_pre = _safe_index(pre[6], pre[8])
    nbr_post = _safe_index(post[6], post[8])
    dnbr = nbr_pre - nbr_post
    rdnbr = np.clip(dnbr / np.sqrt(np.maximum(np.abs(nbr_pre), 0.01)), -3, 3)

    channels = [
        pre[1], pre[2], pre[6], pre[7], pre[8],
        post[1], post[2], post[6], post[7], post[8],
        dnbr, rdnbr,
        s1post[0] - s1pre[0], s1post[1] - s1pre[1],
        aux[0] / 1000.0,
        (aux[2] == 40).astype(np.float32),
    ]
    cube = np.stack(channels, axis=0)
    cube[~np.isfinite(cube)] = 0.0

    tensor = torch.from_numpy(cube).float().unsqueeze(0).to(device)
    with torch.no_grad():
        burn_logits, sev_logits = model(tensor)
        burn_prob = torch.sigmoid(burn_logits).squeeze().cpu().numpy()
        sev_probs = torch.softmax(sev_logits, dim=1).squeeze().permute(1, 2, 0).cpu().numpy()

    return burn_prob, sev_probs


def predict_bs(
    data_dir: Path,
    chip_id: str,
    burn_model,
    severity_model,
    moe_model: LandcoverMoE | None,
    unet_model,
    unet_device,
    metadata: dict,
    use_postprocessing: bool = True,
) -> np.ndarray:
    x, shape = extract_bs_features_v2(data_dir, chip_id)
    scl_post, landcover = extract_bs_aux_masks(data_dir, chip_id)

    # 1. Prediction source selection
    if unet_model is not None:
        burn_prob_unet, sev_probs_unet = predict_bs_unet(data_dir, chip_id, unet_model, unet_device)
        if moe_model is not None:
            # Ensembling: 50% CNN U-Net + 50% GBDT MoE
            burn_prob_moe = moe_model.predict_burn_proba(x).reshape(shape)
            sev_probs_moe = moe_model.predict_severity_proba(x).reshape(shape[0], shape[1], 3)
            burn_prob = 0.5 * burn_prob_unet + 0.5 * burn_prob_moe
            sev_probs = 0.5 * sev_probs_unet + 0.5 * sev_probs_moe
        else:
            burn_prob = burn_prob_unet
            sev_probs = sev_probs_unet
    elif moe_model is not None:
        burn_prob = moe_model.predict_burn_proba(x).reshape(shape)
        sev_probs = moe_model.predict_severity_proba(x).reshape(shape[0], shape[1], 3)
    else:
        burn_prob = burn_model.predict(x).reshape(shape)
        sev_probs = severity_model.predict(x).reshape(shape[0], shape[1], 3)

    burn_threshold = metadata.get("burn_threshold", 0.53)
    min_size = metadata.get("min_component_size", 15)

    if use_postprocessing:
        prediction = postprocess_bs_prediction(
            burn_prob=burn_prob,
            severity_logits=sev_probs,
            burn_threshold=burn_threshold,
            min_component_size=min_size,
            scl_post=scl_post,
            landcover=landcover,
        )
    else:
        sev = np.argmax(sev_probs, axis=-1).astype(np.uint8) + 1
        prediction = np.where(burn_prob >= burn_threshold, sev, 0).astype(np.uint8)

    return prediction


def main() -> None:
    parser = argparse.ArgumentParser(description="CosmoHackathon Wildfire Inference Pipeline")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "test")
    parser.add_argument("--output", type=Path, default=ROOT / "submissions" / "submission.csv")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    parser.add_argument("--no-postprocess", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    sample_path = args.data_dir / "sample_submission.csv"
    meta_path = args.data_dir / "meta.csv"

    if not sample_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing sample_submission.csv or meta.csv in {args.data_dir}")

    template = pd.read_csv(sample_path, dtype={"chip_id": str, "class_id": int}, keep_default_na=False)
    meta = pd.read_csv(meta_path)

    # 1. Load AF model
    af_model, af_metadata = load_native_model(args.weights_dir, "af")

    # 2. Check for U-Net
    unet_model, unet_device = None, None
    unet_path = args.weights_dir / "cosmo_unet.pt"
    if unet_path.exists():
        try:
            import torch
            from cosmofire.models.unet import CosmoUNet
            unet_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            unet_model = CosmoUNet(in_channels=16, num_severity_classes=3).to(unet_device)
            unet_model.load_state_dict(torch.load(unet_path, map_location=unet_device))
            unet_model.eval()
            print("Loaded Deep Learning model: CosmoUNet (PyTorch)", flush=True)
        except Exception as e:
            print(f"Notice: U-Net available but could not load ({e}). Using GBDT.", flush=True)

    # 3. Check for MoE
    moe_model = None
    if (args.weights_dir / "moe_crop_burn_lgbm.txt").exists():
        try:
            moe_model = LandcoverMoE.load(args.weights_dir)
            print("Loaded model: Landcover Mixture of Experts (MoE GBDT)", flush=True)
        except Exception as e:
            print(f"Notice: MoE load error ({e})", flush=True)

    # 4. Standard LightGBM fallback
    bs_burn_model, bs_metadata = None, {"burn_threshold": 0.53, "min_component_size": 15}
    bs_severity_model = None
    if (args.weights_dir / "bs_burn_lgbm.txt").exists() and (args.weights_dir / "bs_severity_lgbm.txt").exists():
        bs_burn_model, bs_metadata = load_native_model(args.weights_dir, "bs_burn")
        bs_severity_model, _ = load_native_model(args.weights_dir, "bs_severity")

    encoded: dict[tuple[str, int], str] = {}
    for index, row in enumerate(meta.itertuples(index=False), start=1):
        if row.kind == "af":
            mask = predict_af(args.data_dir, row.chip_id, af_model, af_metadata)
            encoded[(row.chip_id, 1)] = encode_rle(mask == 1)
        elif row.kind == "bs":
            mask = predict_bs(
                args.data_dir,
                row.chip_id,
                burn_model=bs_burn_model,
                severity_model=bs_severity_model,
                moe_model=moe_model,
                unet_model=unet_model,
                unet_device=unet_device,
                metadata=bs_metadata,
                use_postprocessing=not args.no_postprocess,
            )
            for class_id in (1, 2, 3):
                encoded[(row.chip_id, class_id)] = encode_rle(mask == class_id)
        else:
            raise ValueError(f"Unknown kind: {row.kind}")

        if index % 25 == 0 or index == len(meta):
            print(f"Inference progress: {index}/{len(meta)} chips", flush=True)

    output = template[["chip_id", "class_id"]].copy()
    output["rle"] = [encoded[(row.chip_id, int(row.class_id))] for row in output.itertuples(index=False)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    elapsed = time.perf_counter() - started
    print(f"SUCCESS: Wrote {args.output} with {len(output)} rows in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
