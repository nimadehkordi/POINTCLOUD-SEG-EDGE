"""Segmentation and numerical-comparison metrics."""

from __future__ import annotations

import numpy as np

from .data import CLASS_NAMES, NUM_CLASSES


def confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Rows = ground truth, columns = prediction."""
    idx = target.reshape(-1) * num_classes + pred.reshape(-1)
    return np.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)


def iou_from_confusion(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = tp + fp + fn
    return np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)


def summarize(cm: np.ndarray) -> dict:
    iou = iou_from_confusion(cm)
    return {
        "mIoU": float(np.nanmean(iou)),
        "OA": float(np.diag(cm).sum() / cm.sum()),
        **{f"IoU_{name}": float(v) for name, v in zip(CLASS_NAMES, iou)},
    }


def sqnr_db(reference: np.ndarray, approx: np.ndarray) -> float:
    """Signal-to-quantization-noise ratio in dB. Each 6 dB is roughly one extra bit."""
    ref = reference.astype(np.float64).ravel()
    err = ref - approx.astype(np.float64).ravel()
    noise = np.sum(err**2)
    if noise == 0:
        return float("inf")
    return float(10 * np.log10(np.sum(ref**2) / noise + 1e-30))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
