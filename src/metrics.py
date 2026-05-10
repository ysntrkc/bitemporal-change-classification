"""Evaluation metrics.

Provides ``compute_metrics``, ``tune_thresholds_per_class``, and
``tta_forward``.
"""

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


def tune_thresholds_per_class(
    probs: np.ndarray,
    targets: np.ndarray,
    steps: np.ndarray = _DEFAULT_STEPS,
) -> np.ndarray:
    """Pick the per-class decision threshold that maximises F1 on the given set.

    For each class ``c`` independently, sweep ``t`` over ``steps`` and pick
    ``argmax_t F1_c(t)``. Run on the validation set; the returned thresholds
    are then frozen and applied to test.

    Args:
        probs: ``[N, C]`` sigmoid probabilities.
        targets: ``[N, C]`` multi-hot ground truth in ``{0, 1}``.
        steps: 1-D array of candidate thresholds. Defaults to
            ``np.arange(0.05, 0.96, 0.02)``.

    Returns:
        ``[C]`` array of best-F1 thresholds (``float64``).

    Notes:
        Vectorised over both ``N`` and ``T``. Memory is ``O(N * T)`` per
        class — for our val sizes (a few thousand × ~46 steps) this is
        well under a megabyte. Classes with zero positives in ``targets``
        receive the first step (``steps[0]``) since every threshold ties
        at F1 = 0; this has no effect on macro-F1.
    """
    if probs.shape != targets.shape:
        raise ValueError(
            f"probs and targets shape mismatch: {probs.shape} vs {targets.shape}"
        )
    if steps.ndim != 1:
        raise ValueError(f"steps must be 1-D; got shape {steps.shape}")

    n, c = probs.shape
    t = steps.shape[0]
    y = targets.astype(np.int64)

    # preds[i, j, k] = (probs[i, k] >= steps[j])
    # Build per-class to keep memory bounded: peak [N, T] per class.
    best = np.empty(c, dtype=np.float64)
    for k in range(c):
        p = probs[:, k][:, None] >= steps[None, :]      # [N, T] bool
        yk = y[:, k][:, None]                            # [N, 1]
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
    """Average sigmoid probabilities over geometric TTA passes.

    For each op in ``tta_ops``, apply the same spatial transform to both
    ``A`` and ``B`` (preserving correspondence), forward through
    ``model``, sigmoid the outputs, and accumulate. The class labels are
    invariant under the supported ops (h-flip, v-flip, 180° rotation),
    so averaging is pure noise reduction.

    Supported ops: ``"orig"``, ``"hflip"``, ``"vflip"``, ``"rot180"``.

    Args:
        model: a Phase-1 (or compatible) model returning a dict with
            ``logits_family`` (``[B, C]``) and ``logit_nochg`` (``[B]``).
        batch: dict with at least ``"A"`` and ``"B"`` tensors of shape
            ``[B, 3, H, W]``, already on the model's device.
        tta_ops: non-empty sequence of op names. Order is irrelevant
            (the result is a mean).

    Returns:
        Dict with:
            ``probs_family``: ``[B, C]`` averaged sigmoid probs (float32).
            ``prob_nochg``:    ``[B]`` averaged sigmoid prob (float32).

    Notes:
        Caller is responsible for ``model.eval()`` and any autocast
        context. ``no_grad`` is enforced by the decorator.
    """
    if len(tta_ops) == 0:
        raise ValueError("tta_ops must be non-empty")
    unknown = [op for op in tta_ops if op not in _TTA_OPS]
    if unknown:
        raise ValueError(f"unknown TTA ops: {unknown}; supported: {_TTA_OPS}")

    a = batch["A"]
    b = batch["B"]

    probs_family_sum: torch.Tensor | None = None
    prob_nochg_sum: torch.Tensor | None = None

    for op in tta_ops:
        out = model(_apply_tta(a, op), _apply_tta(b, op))
        pf = torch.sigmoid(out["logits_family"].float())
        pn = torch.sigmoid(out["logit_nochg"].float())
        probs_family_sum = pf if probs_family_sum is None else probs_family_sum + pf
        prob_nochg_sum = pn if prob_nochg_sum is None else prob_nochg_sum + pn

    assert probs_family_sum is not None and prob_nochg_sum is not None
    n = float(len(tta_ops))
    return {
        "probs_family": probs_family_sum / n,
        "prob_nochg": prob_nochg_sum / n,
    }
