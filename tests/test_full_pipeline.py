import numpy as np
import pytest
from cosmofire.rle import decode_rle, encode_rle
from cosmofire.metrics import af_f1, bs_metrics, competition_score
from cosmofire.postprocessing import (
    clean_burn_mask,
    remove_small_components,
    filter_by_scl_and_landcover,
    smooth_severity_map,
    postprocess_bs_prediction,
)

def test_rle_roundtrip():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:4, 2:5] = 1
    mask[7, 8] = 1
    rle = encode_rle(mask)
    decoded = decode_rle(rle, mask.shape)
    np.testing.assert_array_equal(mask, decoded)

def test_metrics_calculation():
    # True mask: two pixels of class 1, two pixels of class 2
    y_true = np.array([0, 1, 1, 2, 2, 0, 0, 0])
    y_pred = np.array([0, 1, 1, 2, 0, 2, 0, 0])
    burn_iou, miou, class_ious = bs_metrics(y_true, y_pred)
    assert 0.0 < burn_iou <= 1.0
    assert 0.0 < miou <= 1.0
    score = competition_score(0.95, burn_iou, miou)
    assert 0.0 < score <= 1.0

def test_remove_small_components():
    mask = np.zeros((30, 30), dtype=np.uint8)
    # Large component: 5x5 = 25 pixels
    mask[5:10, 5:10] = 1
    # Small component (noise): 2 pixels
    mask[20, 20:22] = 1

    cleaned = remove_small_components(mask, min_size=10)
    assert np.all(cleaned[5:10, 5:10] == 1)
    assert np.all(cleaned[20, 20:22] == 0)

def test_hole_filling_and_closing():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:10, 2:10] = 1
    # Make a small 1-pixel hole
    mask[5, 5] = 0

    cleaned = clean_burn_mask(mask, min_size=5, do_fill_holes=True)
    assert cleaned[5, 5] == 1

def test_domain_filtering():
    mask = np.ones((10, 10), dtype=np.uint8)
    scl = np.full((10, 10), 4, dtype=np.uint8)
    # Pixel (0, 0) is water in SCL (6)
    scl[0, 0] = 6
    # Pixel (1, 1) is cloud in SCL (9)
    scl[1, 1] = 9

    landcover = np.full((10, 10), 30, dtype=np.uint8)
    # Pixel (2, 2) is water body in WorldCover (80)
    landcover[2, 2] = 80

    filtered = filter_by_scl_and_landcover(mask, scl_post=scl, landcover=landcover)
    assert filtered[0, 0] == 0
    assert filtered[1, 1] == 0
    assert filtered[2, 2] == 0
    assert filtered[5, 5] == 1

def test_severity_smoothing():
    burn_mask = np.ones((9, 9), dtype=np.uint8)
    # Mostly class 1
    severity = np.full((9, 9), 1, dtype=np.uint8)
    # Single noise pixel of class 3 in the middle
    severity[4, 4] = 3

    smoothed = smooth_severity_map(burn_mask, severity, filter_size=3)
    assert smoothed[4, 4] == 1  # Successfully smoothed to consensus class 1!
