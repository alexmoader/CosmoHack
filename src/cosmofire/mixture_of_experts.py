"""
Landcover-Conditioned Mixture of Experts (MoE) for Burn Severity.
Splits the problem into specialized domain models:
- Cropland Expert (Landcover 40): Specialized in separating crop harvest/tillage from stubble burning
- Nature Expert (Landcover 10/20/30/others): Forest, Shrubland, and Steppe biomass burn dynamics
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import lightgbm as lgb
import numpy as np

class LandcoverMoE:
    """
    Mixture of Experts router based on ESA WorldCover landcover classes:
    - 40: Cropland (specialized harvest filter)
    - 10, 20, 30: Forest, Shrubland, Grassland (natural biomass dynamics)
    - others: generic background
    """
    def __init__(
        self,
        crop_burn_model: lgb.Booster | None = None,
        nature_burn_model: lgb.Booster | None = None,
        crop_severity_model: lgb.Booster | None = None,
        nature_severity_model: lgb.Booster | None = None,
        landcover_idx: int = 47,  # default index of landcover in BS_V2
    ):
        self.crop_burn_model = crop_burn_model
        self.nature_burn_model = nature_burn_model
        self.crop_severity_model = crop_severity_model
        self.nature_severity_model = nature_severity_model
        self.landcover_idx = landcover_idx

    def predict_burn_proba(self, x: np.ndarray) -> np.ndarray:
        landcover = x[:, self.landcover_idx].astype(int)
        crop_mask = landcover == 40
        probs = np.zeros(len(x), dtype=np.float32)

        # 1. Cropland expert
        if np.any(crop_mask) and self.crop_burn_model is not None:
            probs[crop_mask] = self.crop_burn_model.predict(x[crop_mask])
        elif np.any(crop_mask) and self.nature_burn_model is not None:
            probs[crop_mask] = self.nature_burn_model.predict(x[crop_mask])

        # 2. Nature expert (forest, steppe, shrubland)
        nature_mask = ~crop_mask
        if np.any(nature_mask) and self.nature_burn_model is not None:
            probs[nature_mask] = self.nature_burn_model.predict(x[nature_mask])
        elif np.any(nature_mask) and self.crop_burn_model is not None:
            probs[nature_mask] = self.crop_burn_model.predict(x[nature_mask])

        return probs

    def predict_severity_proba(self, x: np.ndarray) -> np.ndarray:
        landcover = x[:, self.landcover_idx].astype(int)
        crop_mask = landcover == 40
        probs = np.zeros((len(x), 3), dtype=np.float32)

        if np.any(crop_mask) and self.crop_severity_model is not None:
            probs[crop_mask] = self.crop_severity_model.predict(x[crop_mask])
        elif np.any(crop_mask) and self.nature_severity_model is not None:
            probs[crop_mask] = self.nature_severity_model.predict(x[crop_mask])

        nature_mask = ~crop_mask
        if np.any(nature_mask) and self.nature_severity_model is not None:
            probs[nature_mask] = self.nature_severity_model.predict(x[nature_mask])
        elif np.any(nature_mask) and self.crop_severity_model is not None:
            probs[nature_mask] = self.crop_severity_model.predict(x[nature_mask])

        return probs

    def save(self, weights_dir: Path) -> None:
        weights_dir.mkdir(parents=True, exist_ok=True)
        if self.crop_burn_model is not None:
            self.crop_burn_model.save_model(str(weights_dir / "moe_crop_burn_lgbm.txt"))
        if self.nature_burn_model is not None:
            self.nature_burn_model.save_model(str(weights_dir / "moe_nature_burn_lgbm.txt"))
        if self.crop_severity_model is not None:
            self.crop_severity_model.save_model(str(weights_dir / "moe_crop_severity_lgbm.txt"))
        if self.nature_severity_model is not None:
            self.nature_severity_model.save_model(str(weights_dir / "moe_nature_severity_lgbm.txt"))

        meta = {"landcover_idx": self.landcover_idx, "type": "LandcoverMoE_v1"}
        (weights_dir / "moe_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, weights_dir: Path) -> LandcoverMoE:
        meta_file = weights_dir / "moe_meta.json"
        landcover_idx = 47
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            landcover_idx = meta.get("landcover_idx", 47)

        crop_burn = None
        if (weights_dir / "moe_crop_burn_lgbm.txt").exists():
            crop_burn = lgb.Booster(model_file=str(weights_dir / "moe_crop_burn_lgbm.txt"))

        nature_burn = None
        if (weights_dir / "moe_nature_burn_lgbm.txt").exists():
            nature_burn = lgb.Booster(model_file=str(weights_dir / "moe_nature_burn_lgbm.txt"))

        crop_sev = None
        if (weights_dir / "moe_crop_severity_lgbm.txt").exists():
            crop_sev = lgb.Booster(model_file=str(weights_dir / "moe_crop_severity_lgbm.txt"))

        nature_sev = None
        if (weights_dir / "moe_nature_severity_lgbm.txt").exists():
            nature_sev = lgb.Booster(model_file=str(weights_dir / "moe_nature_severity_lgbm.txt"))

        return cls(crop_burn, nature_burn, crop_sev, nature_sev, landcover_idx=landcover_idx)
