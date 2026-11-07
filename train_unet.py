"""
End-to-End Deep Learning Training Pipeline for CosmoFire using Multi-Task U-Net.
Optimizes IoU directly via BCE + Dice Compound Loss on multi-spectral satellite cubes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from cosmofire.models.unet import CosmoUNet, MultiTaskCompoundLoss
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object  # Fallback dummy class to avoid NameError at import time

from cosmofire.features import bs_paths, read_mask, _read, _safe_index


class BSDataset(Dataset):
    def __init__(self, data_dir: Path, meta: pd.DataFrame, patch_size: int = 256, is_train: bool = True):
        self.data_dir = data_dir
        self.meta = meta
        self.patch_size = patch_size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed.")
        row = self.meta.iloc[idx]
        chip_id = row.chip_id

        p_pre, p_post, p_s1pre, p_s1post, p_aux = bs_paths(self.data_dir, chip_id)
        pre = _read(p_pre)[:9].astype(np.float32) / 10000.0
        post = _read(p_post)[:9].astype(np.float32) / 10000.0
        s1pre = _read(p_s1pre).astype(np.float32) / 100.0
        s1post = _read(p_s1post).astype(np.float32) / 100.0
        aux = _read(p_aux).astype(np.float32)

        nbr_pre = _safe_index(pre[6], pre[8])
        nbr_post = _safe_index(post[6], post[8])
        dnbr = nbr_pre - nbr_post
        rdnbr = np.clip(dnbr / np.sqrt(np.maximum(np.abs(nbr_pre), 0.01)), -3, 3)

        # 16-channel tensor
        channels = [
            pre[1], pre[2], pre[6], pre[7], pre[8],   # B3, B4, B8A, B11, B12 pre
            post[1], post[2], post[6], post[7], post[8], # post
            dnbr, rdnbr,
            s1post[0] - s1pre[0], s1post[1] - s1pre[1], # delta VV, VH
            aux[0] / 1000.0, # normalized DEM
            (aux[2] == 40).astype(np.float32), # Cropland indicator
        ]
        cube = np.stack(channels, axis=0)
        cube[~np.isfinite(cube)] = 0.0

        mask = read_mask(self.data_dir, "bs", chip_id).astype(np.int64)

        # Random patch crop for training to fit in memory
        h, w = mask.shape
        if self.is_train and (h > self.patch_size or w > self.patch_size):
            y0 = np.random.randint(0, h - self.patch_size + 1)
            x0 = np.random.randint(0, w - self.patch_size + 1)
            cube = cube[:, y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
            mask = mask[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]

        burn_target = (mask > 0).astype(np.float32)[np.newaxis, ...]
        severity_target = mask.copy()

        return torch.from_numpy(cube).float(), torch.from_numpy(burn_target).float(), torch.from_numpy(severity_target).long()


def train_unet(args) -> None:
    if not TORCH_AVAILABLE:
        print("\n" + "="*60)
        print("ВНИМАНИЕ: Библиотека PyTorch (torch) не установлена!")
        print("В вашей среде Windows используется Python 3.14.")
        print("Для Python 3.14 официальные сборки PyTorch еще не выпущены.")
        print("Рекомендации:")
        print("1. Обучить CosmoUNet в Google Colab или Kaggle (с GPU и Python 3.10-3.11)")
        print("   и скопировать файл cosmo_unet.pt в папку weights/.")
        print("2. Либо использовать обновленный ансамбль CatBoost/HistGradientBoosting/LightGBM MoE.")
        print("="*60 + "\n")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Обучение U-Net на устройстве: {device}", flush=True)

    meta = pd.read_csv(args.data_dir / "bs" / "meta.csv")
    dataset = BSDataset(args.data_dir, meta, patch_size=256, is_train=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = CosmoUNet(in_channels=16, num_severity_classes=3).to(device)
    criterion = MultiTaskCompoundLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Старт обучения U-Net на {len(meta)} чипах на {args.epochs} эпох...")
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        total_loss = 0.0

        for cubes, burn_targets, sev_targets in loader:
            cubes = cubes.to(device)
            burn_targets = burn_targets.to(device)
            sev_targets = sev_targets.to(device)

            optimizer.zero_grad()
            burn_logits, sev_logits = model(cubes)
            loss = criterion(burn_logits, sev_logits, burn_targets, sev_targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        elapsed = time.perf_counter() - started
        avg_loss = total_loss / len(loader)
        print(f"Эпоха {epoch}/{args.epochs} | Loss: {avg_loss:.4f} | Время: {elapsed:.1f}с", flush=True)

    args.weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.weights_dir / "cosmo_unet.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"Веса U-Net успешно сохранены в {weights_path}!", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Multi-Task U-Net for Burn Severity")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "train")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    train_unet(args)


if __name__ == "__main__":
    main()
