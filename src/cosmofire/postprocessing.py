"""
Post-processing module for CosmoFire.
Filters false positives, enforces spatial coherence, and removes speckle noise.
Includes chip-level gate and connected component area filtering.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    binary_opening,
    generate_binary_structure,
    label,
    median_filter,
)

def remove_small_components(mask: np.ndarray, min_size: int = 25) -> np.ndarray:
    """
    Remove connected components with pixel area strictly less than min_size.
    At 20m resolution (BS), 1 pixel = 400 m2 = 0.04 ha.
    A threshold of 25 pixels = 1.0 ha eliminates speckle and sensor noise
    without dropping true fire scars.
    """
    if min_size <= 1 or not np.any(mask):
        return mask

    structure = generate_binary_structure(2, 2)  # 8-connectivity
    labeled, num_features = label(mask > 0, structure=structure)
    if num_features == 0:
        return mask

    component_sizes = np.bincount(labeled.ravel())
    too_small = component_sizes < min_size
    too_small_mask = too_small[labeled]
    cleaned = mask.copy()
    cleaned[too_small_mask] = 0
    return cleaned

def clean_burn_mask(
    binary_mask: np.ndarray,
    min_size: int = 25,
    do_open: bool = True,
    do_close: bool = True,
    do_fill_holes: bool = True,
    hole_max_size: int = 60,
    chip_min_burn: int = 40,
) -> np.ndarray:
    """
    Spatial refinement pipeline for binary burn masks:
    1. Binary opening (removes 1-pixel peninsulas & speckles)
    2. Component area filtering (removes small isolated false positives)
    3. Binary closing (bridges internal sensor dropouts)
    4. Small hole filling (reconstructs contiguous burned areas)
    5. Chip-level gate (if entire chip has < chip_min_burn pixels, zero it out)
    """
    cleaned = binary_mask.astype(bool)

    # 1. Opening to remove single-pixel spikes
    if do_open and np.any(cleaned):
        structure_cross = generate_binary_structure(2, 1)  # 4-connectivity
        cleaned = binary_opening(cleaned, structure=structure_cross)

    # 2. Filter small components
    if min_size > 1 and np.any(cleaned):
        cleaned = remove_small_components(cleaned, min_size=min_size) > 0

    # 3. Closing to bridge internal micro-voids
    if do_close and np.any(cleaned):
        structure_8 = generate_binary_structure(2, 2)
        cleaned = binary_closing(cleaned, structure=structure_8)

    # 4. Fill holes if requested
    if do_fill_holes and np.any(cleaned):
        filled = binary_fill_holes(cleaned)
        holes = filled & (~cleaned)
        if np.any(holes):
            hole_labels, num_holes = label(holes)
            if num_holes > 0:
                hole_sizes = np.bincount(hole_labels.ravel())
                small_holes = (hole_sizes <= hole_max_size)[hole_labels]
                cleaned = cleaned | small_holes

    # 5. Chip-level gate: real fires average >1000 ha.
    # Total chip burn < 40 pixels (1.6 ha) across a 100 km2 chip is 100% false alarm.
    if np.sum(cleaned) < chip_min_burn:
        return np.zeros_like(cleaned, dtype=np.uint8)

    return cleaned.astype(np.uint8)

def filter_by_scl_and_landcover(
    burn_mask: np.ndarray,
    scl_post: np.ndarray | None = None,
    landcover: np.ndarray | None = None,
) -> np.ndarray:
    """
    Domain physics filtering:
    - SCL 6 = water, 8/9 = clouds, 11 = snow
    - WorldCover 80 = water bodies, 50 = built-up
    """
    cleaned = burn_mask.copy()

    if scl_post is not None:
        invalid_scl = np.isin(scl_post, [3, 6, 8, 9, 11])
        cleaned[invalid_scl] = 0

    if landcover is not None:
        water_mask = landcover == 80
        cleaned[water_mask] = 0

    return cleaned

def smooth_severity_map(
    burn_mask: np.ndarray,
    severity: np.ndarray,
    filter_size: int = 3,
) -> np.ndarray:
    """
    Smooth severity predictions within the confirmed burn area.
    Severity classes inside a real fire scar have spatial continuity.
    """
    out = np.zeros_like(severity, dtype=np.uint8)
    burn_bool = burn_mask > 0
    if not np.any(burn_bool):
        return out

    out[burn_bool] = np.clip(severity[burn_bool], 1, 3)

    if filter_size > 1:
        med = median_filter(out, size=filter_size, mode="nearest")
        out[burn_bool] = np.clip(med[burn_bool], 1, 3)

    out[~burn_bool] = 0
    return out

def postprocess_bs_prediction(
    burn_prob: np.ndarray,
    severity_logits: np.ndarray,
    burn_threshold: float = 0.55,
    min_component_size: int = 25,
    chip_min_burn: int = 40,
    scl_post: np.ndarray | None = None,
    landcover: np.ndarray | None = None,
    severity_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Full inference postprocessing pipeline for Burn Severity:
    1. Threshold burn probabilities
    2. Clean binary burn mask (opening, closing, area filter, hole fill, chip gate)
    3. Filter water & cloud artifacts via SCL & Landcover
    4. Compute weighted severity argmax
    5. Smooth severity inside confirmed burn mask
    """
    raw_burn = burn_prob >= burn_threshold

    refined_burn = clean_burn_mask(
        raw_burn,
        min_size=min_component_size,
        chip_min_burn=chip_min_burn,
    )

    refined_burn = filter_by_scl_and_landcover(refined_burn, scl_post=scl_post, landcover=landcover)

    if severity_logits.ndim == 3 and severity_logits.shape[-1] == 3:
        w = np.array(severity_weights, dtype=np.float32)
        weighted_logits = severity_logits * w.reshape(1, 1, 3)
        raw_sev = np.argmax(weighted_logits, axis=-1).astype(np.uint8) + 1
    else:
        raw_sev = severity_logits.astype(np.uint8)

    final_prediction = smooth_severity_map(refined_burn, raw_sev, filter_size=3)
    return final_prediction
