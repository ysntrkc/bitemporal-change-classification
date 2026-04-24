"""Loss functions for Phase 1 and Phase 2.

Currently provides ``AsymmetricLoss`` (Ridnik et al., ICCV 2021).
``UncertaintyWeightedLoss`` (Kendall et al., CVPR 2018) is added in task 3.4.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification.

    Ridnik et al., "Asymmetric Loss for Multi-Label Classification",
    ICCV 2021. Decouples focusing parameters for positives and negatives,
    with probability clipping on the negative branch to discard
    easy/mislabeled negatives.

    Defaults follow PROJECT_PLAN.md §4.2: ``gamma_neg=4``, ``gamma_pos=0``,
    ``clip=0.05``.

    Args:
        gamma_neg: Focusing parameter for negative class (γ⁻). Higher
            values down-weight easy negatives more aggressively.
        gamma_pos: Focusing parameter for positive class (γ⁺).
        clip: Probability clipping margin on the negative branch. The
            negative probability becomes ``clamp(1 - σ(x) + clip, max=1)``,
            which zeroes out the loss for negatives with ``σ(x) ≤ clip``.
        eps: Small value to avoid ``log(0)``.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        if logits.shape != targets.shape:
            raise ValueError(
                f"logits and targets shape mismatch: {tuple(logits.shape)} "
                f"vs {tuple(targets.shape)}"
            )

        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = xs_pos * targets + xs_neg * (1.0 - targets)
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            loss = loss * ((1.0 - pt) ** one_sided_gamma)

        return -loss.mean()
