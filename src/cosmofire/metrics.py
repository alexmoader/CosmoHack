from __future__ import annotations
import numpy as np

def binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int]:
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    return int(np.sum(yt & yp)), int(np.sum(~yt & yp)), int(np.sum(yt & ~yp))

def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 1.0 if denominator == 0 else 2.0 * tp / denominator

def iou_from_counts(tp: int, fp: int, fn: int) -> float:
    denominator = tp + fp + fn
    return 1.0 if denominator == 0 else tp / denominator

def af_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_from_counts(*binary_counts(y_true, y_pred))

def bs_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, list[float]]:
    true_burn = np.asarray(y_true) > 0
    pred_burn = np.asarray(y_pred) > 0
    burn_iou = iou_from_counts(*binary_counts(true_burn, pred_burn))
    class_ious: list[float] = []
    for class_id in (1, 2, 3):
        class_ious.append(
            iou_from_counts(*binary_counts(np.asarray(y_true) == class_id, np.asarray(y_pred) == class_id))
        )
    return burn_iou, float(np.mean(class_ious)), class_ious

def competition_score(f1_af: float, iou_burn: float, miou_severity: float) -> float:
    return 0.35 * f1_af + 0.35 * iou_burn + 0.30 * miou_severity
