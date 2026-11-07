from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import lightgbm as lgb

def save_native_model(model: Any, weights_dir: Path, kind: str, metadata: dict[str, Any]) -> None:
    """Save a LightGBM model in portable text format plus JSON metadata."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    booster = model.booster_ if hasattr(model, "booster_") else model
    booster.save_model(str(weights_dir / f"{kind}_lgbm.txt"))
    (weights_dir / f"{kind}_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def load_native_model(weights_dir: Path, kind: str) -> tuple[lgb.Booster, dict[str, Any]]:
    model_path = weights_dir / f"{kind}_lgbm.txt"
    metadata_path = weights_dir / f"{kind}_meta.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Не найдены веса {kind}: ожидаются {model_path.name} и {metadata_path.name}"
        )
    model = lgb.Booster(model_file=str(model_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata
