"""Evaluation metrics.

Currently provides ``compute_metrics``. ``tune_thresholds_per_class``
(task 2.5) and ``tta_forward`` (task 2.7) are added later.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    thresholds: np.ndarray,
) -> dict:
    """Compute multi-label classification metrics for one family.

    Args:
        probs: ``[N, C]`` sigmoid probabilities.
        targets: ``[N, C]`` multi-hot ground truth in ``{0, 1}``.
        thresholds: ``[C]`` per-class decision thresholds for binarising
            probabilities. For default 0.5 evaluation, pass
            ``np.full(C, 0.5)``.

    Returns:
        Dict with the following keys (scalars unless noted):
            ``micro_f1``, ``macro_f1``,
            ``precision_micro``, ``precision_macro``,
            ``recall_micro``, ``recall_macro``,
            ``mAP`` (macro-averaged, threshold-free),
            ``per_class_f1`` (``list[float]`` of length ``C``).
    """
    if probs.shape != targets.shape:
        raise ValueError(
            f"probs and targets shape mismatch: {probs.shape} vs {targets.shape}"
        )
    if thresholds.shape != (probs.shape[1],):
        raise ValueError(
            f"thresholds must have shape ({probs.shape[1]},); got {thresholds.shape}"
        )

    preds = (probs >= thresholds[None, :]).astype(np.int64)
    y = targets.astype(np.int64)

    return {
        "micro_f1": float(f1_score(y, preds, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y, preds, average="macro", zero_division=0)),
        "precision_micro": float(
            precision_score(y, preds, average="micro", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(y, preds, average="macro", zero_division=0)
        ),
        "recall_micro": float(recall_score(y, preds, average="micro", zero_division=0)),
        "recall_macro": float(recall_score(y, preds, average="macro", zero_division=0)),
        "mAP": float(average_precision_score(y, probs, average="macro")),
        "per_class_f1": f1_score(y, preds, average=None, zero_division=0).tolist(),
    }
