"""Loss functions for Phase 1 and Phase 2.

Phase 1 uses ``AsymmetricLoss`` (Ridnik et al., ICCV 2021) on the
single family head plus a fixed-weight BCE on the no-change head.
Phase 2 uses ``UncertaintyWeightedLoss`` (Kendall, Gal & Cipolla,
CVPR 2018) to balance the four task losses (object / event /
attribute ASL + no-change BCE) with 4 learnable log-σ parameters.
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

    Defaults follow Ridnik et al.: ``gamma_neg=4``, ``gamma_pos=0``,
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


class UncertaintyWeightedLoss(nn.Module):
    """4-task uncertainty-weighted loss (Kendall, Gal & Cipolla, CVPR 2018).

    Owns four learnable log-σ² parameters — one per task: object,
    event, attribute, no-change — and combines pre-computed scalar
    task losses as

        L = 0.5 * Σᵢ exp(-log_σᵢ²) · Lᵢ  +  0.5 * Σᵢ log_σᵢ²

    The first term down-weights tasks the model is currently noisy on;
    the second penalises arbitrarily large σ² (which would otherwise
    drive Lᵢ contributions to zero).

    Despite the attribute name ``log_sigma``, the parameter represents
    log-variance (log σ²): ``exp(-log_sigma)`` plays the role of 1/σ²
    (the "precision" of each task).
    """

    _TASKS = ("obj", "evt", "attr", "nochg")

    def __init__(
        self,
        init_log_sigma: float = 0.0,
        log_sigma_min: float = -2.0,
        log_sigma_max: float = 2.0,
    ):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.full((4,), float(init_log_sigma)))
        # Clamping range guards against the degenerate equilibrium where the
        # regularizer ``+0.5·Σ log_σ²`` dominates and the model "trains" by
        # shrinking log_σ instead of fitting tasks.
        self.log_sigma_min = float(log_sigma_min)
        self.log_sigma_max = float(log_sigma_max)

    def _clamped(self) -> Tensor:
        return self.log_sigma.clamp(self.log_sigma_min, self.log_sigma_max)

    def forward(self, losses: dict[str, Tensor]) -> Tensor:
        missing = [t for t in self._TASKS if t not in losses]
        if missing:
            raise KeyError(f"missing task losses: {missing}; got keys {list(losses)}")
        per_task = torch.stack([losses[t] for t in self._TASKS])   # [4]
        log_sigma = self._clamped()
        precision = torch.exp(-log_sigma)                           # [4]
        return 0.5 * (precision * per_task).sum() + 0.5 * log_sigma.sum()

    def task_weights(self) -> dict[str, float]:
        """Current effective per-task weight (exp(-log_σ²)/2). For logging."""
        with torch.no_grad():
            w = (0.5 * torch.exp(-self._clamped())).tolist()
        return dict(zip(self._TASKS, w))


class FixedWeightLoss(nn.Module):
    """Fixed per-task scalar weights, no learnable parameters.

    Phase-1-style aggregation: ``L = Σᵢ wᵢ · Lᵢ`` where weights are
    config-driven. Used as the no-UWL fallback in Phase 2 (mirrors
    the Phase-1 default of ``1·ASL_family + 0.2·BCE_nochg`` when
    instantiated with ``weights={'obj': 1, 'evt': 1, 'attr': 1,
    'nochg': 0.2}``). Has the same forward/task_weights interface as
    ``UncertaintyWeightedLoss`` so callers can swap implementations.
    """

    _TASKS = ("obj", "evt", "attr", "nochg")

    def __init__(self, weights: dict[str, float]):
        super().__init__()
        missing = [t for t in self._TASKS if t not in weights]
        if missing:
            raise KeyError(f"weights missing tasks: {missing}; got {list(weights)}")
        self._weights = {t: float(weights[t]) for t in self._TASKS}

    def forward(self, losses: dict[str, Tensor]) -> Tensor:
        missing = [t for t in self._TASKS if t not in losses]
        if missing:
            raise KeyError(f"missing task losses: {missing}; got keys {list(losses)}")
        return sum(self._weights[t] * losses[t] for t in self._TASKS)

    def task_weights(self) -> dict[str, float]:
        return dict(self._weights)
