"""
Runner for the CosmoFire GIS & Analytical Service
"""
from __future__ import annotations

import sys
from pathlib import Path
import uvicorn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    print("Запуск CosmoFire GIS Service на http://127.0.0.1:8000 ...")
    uvicorn.run("service.app:app", host="0.0.0.0", port=8000, reload=False)
