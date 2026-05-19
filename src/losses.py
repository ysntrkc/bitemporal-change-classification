"""Loss functions for Phase 1 and Phase 2.

Phase 1 uses ``AsymmetricLoss`` (Ridnik et al., ICCV 2021) on the
single family head plus a fixed-weight BCE on the no-change head.
``DistributionBalancedLoss`` (Wu et al., ECCV 2020) is a drop-in
alternative for the same head, targeted at long-tailed multi-label
problems. Phase 2 uses ``UncertaintyWeightedLoss`` (Kendall, Gal &
Cipolla, CVPR 2018) to balance the four task losses with 4 learnable
log-σ parameters.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F
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


class DistributionBalancedLoss(nn.Module):
    """Distribution-Balanced Loss for multi-label long-tail classification.

    Wu et al., "Distribution-Balanced Loss for Multi-Label Classification
    in Long-Tailed Datasets", ECCV 2020.

    Combines two corrections on top of standard binary cross-entropy:

    1. **Rebalanced positive weighting (R-BCE).** Class ``c`` with
       ``n_c`` positives in ``N`` training samples gets a positive
       weight ``w_c = ((n_max / n_c) ** kappa)``, capped at
       ``max_pos_weight``. This compensates for the over-suppression of
       rare classes that arises in multi-label co-occurrence.

    2. **Negative-tolerant regularisation (NTR).** Each class has a
       fixed log-prior bias ``ν_c = log(n_c / (N - n_c))``. On the
       negative branch the logit is shifted and scaled — the loss
       becomes ``-log(1 - σ(λ * (z_c - ν_c))) / λ``. The shift makes
       "easy" majority negatives (already matching the prior) contribute
       near-zero gradient; the ``1/λ`` factor keeps overall scale
       comparable to plain BCE.

    Reduces to BCE in the degenerate case ``n_c = N/2 ∀ c`` and
    ``kappa = 0``, ``λ = 1``.

    Args:
        class_freq: ``[C]`` positive counts per class on the training set
            (integer or float; cast to float internally).
        n_train: total training-set size (``N``).
        kappa: exponent on the rebalanced-positive weight (paper default
            ~0.05–0.1; lower = less aggressive rebalancing).
        neg_scale: ``λ`` on the negative branch (paper default 2.0).
        max_pos_weight: cap on ``w_c`` to keep extreme tails from
            dominating the gradient (default 10.0).
        eps: numerical floor on ``n_c`` so log-prior is finite when a
            class has zero training positives (rare but possible after
            split-leakage filtering).
        nochg_bias: optional extra bias added to all negative-branch
            logits (set to a non-zero value to break ties on cold
            starts; default 0.0).

    Notes:
        * ``class_freq`` and ``n_train`` are required at construction
          and frozen — they're computed once over the training split.
        * Returns the **mean** over (batch × classes), matching the
          convention used by ``AsymmetricLoss``.
    """

    def __init__(
        self,
        class_freq: Tensor,
        n_train: int,
        kappa: float = 0.05,
        neg_scale: float = 2.0,
        max_pos_weight: float = 10.0,
        eps: float = 1.0,
        nochg_bias: float = 0.0,
    ):
        super().__init__()
        if class_freq.ndim != 1:
            raise ValueError(f"class_freq must be 1-D; got shape {tuple(class_freq.shape)}")
        if n_train <= 0:
            raise ValueError(f"n_train must be positive; got {n_train}")
        if neg_scale <= 0:
            raise ValueError(f"neg_scale must be > 0; got {neg_scale}")

        n_c = class_freq.float().clamp(min=eps)              # [C]
        n_neg = (float(n_train) - n_c).clamp(min=eps)        # [C]
        n_max = float(n_c.max().item())

        # Rebalanced positive weight, capped.
        pos_weight = (n_max / n_c).pow(kappa).clamp(max=max_pos_weight)   # [C]
        # Per-class negative-branch logit bias prior (log-odds of positive).
        bias_prior = torch.log(n_c / n_neg) + nochg_bias                  # [C]

        # Buffers so they ride device.to(...) and are saved into the
        # state dict (which lets eval reconstruct identical objects).
        self.register_buffer("pos_weight", pos_weight, persistent=True)
        self.register_buffer("bias_prior", bias_prior, persistent=True)
        self.neg_scale = float(neg_scale)
        self.kappa = float(kappa)
        self.max_pos_weight = float(max_pos_weight)
        self.n_train = int(n_train)

        logger.info(
            "DBLoss init | C=%d N=%d κ=%.3f λ=%.2f | "
            "pos_weight: min=%.3f median=%.3f max=%.3f | "
            "bias_prior: min=%.3f median=%.3f max=%.3f",
            int(class_freq.numel()), n_train, kappa, neg_scale,
            float(pos_weight.min()), float(pos_weight.median()), float(pos_weight.max()),
            float(bias_prior.min()), float(bias_prior.median()), float(bias_prior.max()),
        )

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        if logits.shape != targets.shape:
            raise ValueError(
                f"logits and targets shape mismatch: {tuple(logits.shape)} "
                f"vs {tuple(targets.shape)}"
            )
        if logits.shape[-1] != self.pos_weight.shape[0]:
            raise ValueError(
                f"logits last-dim {logits.shape[-1]} != C={self.pos_weight.shape[0]}"
            )

        # Positive branch: w_c * y * (-log σ(z))
        log_p = F.logsigmoid(logits)                             # [B, C]
        pos_loss = self.pos_weight * targets * (-log_p)          # broadcast over batch

        # Negative branch with NTR shift+scale: -(1/λ) * (1-y) * log(1 - σ(λ(z - ν)))
        shifted = self.neg_scale * (logits - self.bias_prior)    # [B, C]
        log_1mp = F.logsigmoid(-shifted)                          # log(1 - σ(λ(z - ν)))
        neg_loss = (1.0 - targets) * (-log_1mp) / self.neg_scale

        return (pos_loss + neg_loss).mean()


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
