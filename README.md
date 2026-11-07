# CosmoFire

Satellite based wildfire monitoring and post fire burn severity mapping. Built by team **Flying Logs** for the CosmoHack 2026 challenge.

## Project

CosmoFire combines two geospatial machine learning tasks with a small web GIS service:

- **Active fire detection:** classify candidate VIIRS thermal observations using spectral, thermal, and contextual features.
- **Burn severity mapping:** identify burned areas and classify severity from Sentinel-1 and Sentinel-2 imagery.
- **Review interface:** expose example fire events and burn contours through a FastAPI service and Leaflet map.

The mapping service includes simulated example events for demonstration. The model pipeline is designed to use the challenge raster data and locally trained model files.

## Approach

The pipeline uses LightGBM models for tabular satellite features, a hierarchical severity classifier, and spatial post-processing. Validation keeps fire events grouped to reduce leakage between training and validation chips. The scoring function combines active-fire F1, burned-area IoU, and severity mIoU.

The team finished just outside the finalist group, with a 0.1 MOE score difference from the finalist threshold, according to the challenge evaluation. This was a close result, not a finalist placement.

## Repository contents

- `src/cosmofire/` — feature engineering, metrics, model I/O, post-processing, RLE, and model code.
- `service/` — FastAPI endpoints and the Leaflet web map.
- `tests/` — tests for scoring, encoding, and pipeline components.
- `configs/` — baseline training parameters.
- Top-level scripts — training, evaluation, inference, and submission validation.

Challenge datasets, trained weights, evaluation logs, and generated submissions are intentionally kept outside this portfolio repository. They are not required to review the source, but are needed to reproduce full model inference and training.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Activate the environment, then install dependencies:
python -m pip install -r requirements.txt
```

The training and inference scripts expect the challenge data and locally available model weights. See each script's `--help` output for its available options.

Run the API and map with:

```bash
python service/run_service.py
```

Then open `http://localhost:8000`. The service uses demonstration data unless the required model/data artifacts are supplied.

## Notes

This repository is a portfolio presentation of a hackathon project. It does not include the original challenge dataset, model weights, or a deployment. Confirm the challenge and team rules before redistributing any restricted materials. No open source license is declared yet.
