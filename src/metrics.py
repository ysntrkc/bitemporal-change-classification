from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


_DEFAULT_STEPS = np.arange(0.05, 0.96, 0.02)


def compute_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    thresholds: np.ndarray,
) -> dict:
    """Multi-label F1/precision/recall (micro+macro) and mAP, plus per-class breakdowns."""
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

    # AP is undefined when a class has no positives; NaN distinguishes that from F1=0.
    support = y.sum(axis=0).astype(np.int64)
    per_class_ap: list[float] = []
    for k in range(y.shape[1]):
        if support[k] == 0:
            per_class_ap.append(float("nan"))
        else:
            per_class_ap.append(
                float(average_precision_score(y[:, k], probs[:, k]))
            )

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
        "per_class_precision": precision_score(
            y, preds, average=None, zero_division=0
        ).tolist(),
        "per_class_recall": recall_score(
            y, preds, average=None, zero_division=0
        ).tolist(),
        "per_class_ap": per_class_ap,
        "per_class_support": support.tolist(),
    }


def tune_thresholds_per_class(
    probs: np.ndarray,
    targets: np.ndarray,
    steps: np.ndarray = _DEFAULT_STEPS,
) -> np.ndarray:
    """Per-class argmax_t F1_c(t) sweep — tune on val, freeze, apply on test."""
    if probs.shape != targets.shape:
        raise ValueError(
            f"probs and targets shape mismatch: {probs.shape} vs {targets.shape}"
        )
    if steps.ndim != 1:
        raise ValueError(f"steps must be 1-D; got shape {steps.shape}")

    n, c = probs.shape
    t = steps.shape[0]
    y = targets.astype(np.int64)

    # Per-class loop keeps peak memory to O(N*T) instead of O(N*C*T).
    best = np.empty(c, dtype=np.float64)
    for k in range(c):
        p = probs[:, k][:, None] >= steps[None, :]
        yk = y[:, k][:, None]
        tp = (p & (yk == 1)).sum(axis=0).astype(np.float64)
        fp = (p & (yk == 0)).sum(axis=0).astype(np.float64)
        fn = (~p & (yk == 1)).sum(axis=0).astype(np.float64)
        denom = 2.0 * tp + fp + fn
        f1 = np.where(denom > 0, 2.0 * tp / np.maximum(denom, 1.0), 0.0)
        best[k] = float(steps[int(np.argmax(f1))])

    logger.info(
        "tuned thresholds: n=%d, c=%d, t=%d steps; min=%.2f median=%.2f max=%.2f",
        n, c, t, float(best.min()), float(np.median(best)), float(best.max()),
    )
    return best


_TTA_OPS = ("orig", "hflip", "vflip", "rot180")


def _apply_tta(x: torch.Tensor, op: str) -> torch.Tensor:
    if op == "orig":
        return x
    if op == "hflip":
        return torch.flip(x, dims=[-1])
    if op == "vflip":
        return torch.flip(x, dims=[-2])
    if op == "rot180":
        return torch.flip(x, dims=[-2, -1])
    raise ValueError(f"unknown TTA op {op!r}; supported: {_TTA_OPS}")


@torch.no_grad()
def tta_forward(
    model: torch.nn.Module,
    batch: dict,
    tta_ops: Sequence[str],
) -> dict:
    """Average sigmoid probs over TTA ops. Returns probs_X/prob_X for each logits_X/logit_X.
    Caller must put the model in eval() and wrap with autocast if desired.
    """
    if len(tta_ops) == 0:
        raise ValueError("tta_ops must be non-empty")
    unknown = [op for op in tta_ops if op not in _TTA_OPS]
    if unknown:
        raise ValueError(f"unknown TTA ops: {unknown}; supported: {_TTA_OPS}")

    a = batch["A"]
    b = batch["B"]
    sums: dict[str, torch.Tensor] = {}

    for op in tta_ops:
        out = model(_apply_tta(a, op), _apply_tta(b, op))
        for key, logit in out.items():
            if key.startswith("logits_"):
                out_key = "probs_" + key[len("logits_") :]
            elif key.startswith("logit_"):
                out_key = "prob_" + key[len("logit_") :]
            else:
                continue
            prob = torch.sigmoid(logit.float())
            sums[out_key] = prob if out_key not in sums else sums[out_key] + prob

    n = float(len(tta_ops))
    return {k: v / n for k, v in sums.items()}
