"""
Lightweight Multi-Task Convolutional Segmentation Network for Wildfire Burn Mapping.
Multi-input: Sentinel-2 pre/post bands + Sentinel-1 SAR + dNBR + Landcover.
Dual-head:
- Head 1: Binary Burn Boundary Segmentation (BCE + Soft Dice Loss)
- Head 2: 3-class Burn Severity (CrossEntropy + Multi-class Dice Loss)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # Encoder
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        # Bottleneck
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 256))

        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256 + 128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128 + 64, 64)

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(64 + 32, 32)

        self.up4 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(32 + 32, 32)

        # Multi-task heads
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

        burn_logits = self.burn_head(d4)
        severity_logits = self.severity_head(d4)
        return burn_logits, severity_logits

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(pred)
        intersection = (prob * target).sum()
        dice = (2.0 * intersection + self.smooth) / (prob.sum() + target.sum() + self.smooth)
        return 1.0 - dice

class MultiTaskCompoundLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss(ignore_index=255)

    def forward(
        self,
        burn_logits: torch.Tensor,
        severity_logits: torch.Tensor,
        burn_target: torch.Tensor,
        severity_target: torch.Tensor,
    ) -> torch.Tensor:
        bce_loss = self.bce(burn_logits, burn_target)
        dice_loss = self.dice(burn_logits, burn_target)

        # Only compute severity loss on pixels where burn is true
        burn_mask = (burn_target > 0.5).squeeze(1)
        if burn_mask.any():
            # severity_target is 1, 2, 3 -> map to 0, 1, 2
            sev_target_clamped = torch.clamp(severity_target[burn_mask] - 1, 0, 2).long()
            sev_preds = severity_logits.permute(0, 2, 3, 1)[burn_mask]
            ce_loss = self.ce(sev_preds, sev_target_clamped)
        else:
            ce_loss = torch.tensor(0.0, device=burn_logits.device)

        return 0.4 * bce_loss + 0.3 * dice_loss + 0.3 * ce_loss
