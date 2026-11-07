"""
Self-contained Google Colab / Kaggle training script for CosmoFire U-Net.
Run this on a free GPU (T4 / P100) in Google Colab or Kaggle in ~10 minutes.
Directly produces cosmo_unet.pt or full test predictions!
"""
# %% [markdown]
# # CosmoFire — Multi-Task U-Net Training (Google Colab / Kaggle)
# 1. Connect to GPU: Runtime -> Change runtime type -> T4 GPU
# 2. Upload fire-train-renamed.tar (and fire-test-renamed.tar if predicting directly)
# 3. Run all cells below!

# %%
import os
import tarfile
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Check GPU
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %%
# Extract dataset if tar exists in current directory
for tar_name in ["fire-train-renamed.tar", "fire-test-renamed.tar"]:
    if Path(tar_name).exists() and not Path("data").exists():
        print(f"Extracting {tar_name}...")
        with tarfile.open(tar_name, "r") as tar:
            tar.extractall("data")
print("Data directory ready.")

# %%
import rasterio

def _read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read()

def _safe_index(a: np.ndarray, b: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    return (a - b) / (np.abs(a) + np.abs(b) + eps)

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class CosmoUNet(nn.Module):
    def __init__(self, in_channels: int = 16, num_severity_classes: int = 3):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 256))

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256 + 128, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128 + 64, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(64 + 32, 32)
        self.up4 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(32 + 32, 32)

        self.burn_head = nn.Conv2d(32, 1, kernel_size=1)
        self.severity_head = nn.Conv2d(32, num_severity_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        u1 = self.up1(x5)
        d1 = self.conv1(torch.cat([u1, x4], dim=1))
        u2 = self.up2(d1)
        d2 = self.conv2(torch.cat([u2, x3], dim=1))
        u3 = self.up3(d2)
        d3 = self.conv3(torch.cat([u3, x2], dim=1))
        u4 = self.up4(d3)
        d4 = self.conv4(torch.cat([u4, x1], dim=1))

        return self.burn_head(d4), self.severity_head(d4)

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(pred)
        intersection = (prob * target).sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + self.smooth)
        return (1.0 - dice).mean()

class CompoundLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, burn_logits, sev_logits, burn_targets, sev_targets):
        l_bce = self.bce(burn_logits, burn_targets)
        l_dice = self.dice(burn_logits, burn_targets)

        burn_mask = (burn_targets > 0.5).squeeze(1)
        if burn_mask.any():
            target_clamped = torch.clamp(sev_targets[burn_mask] - 1, 0, 2).long()
            preds = sev_logits.permute(0, 2, 3, 1)[burn_mask]
            l_ce = self.ce(preds, target_clamped)
        else:
            l_ce = torch.tensor(0.0, device=burn_logits.device)

        return 0.4 * l_bce + 0.4 * l_dice + 0.2 * l_ce

print("Architecture and loss definitions initialized.")
