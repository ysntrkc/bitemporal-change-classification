from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class AsymmetricLoss(nn.Module):
    """ASL from Ridnik et al. (ICCV 2021); defaults gamma_neg=4, gamma_pos=0, clip=0.05."""

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
    """DBLoss from Wu et al. (ECCV 2020): rebalanced positive weights + negative-tolerant shift."""

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

        log_p = F.logsigmoid(logits)
        pos_loss = self.pos_weight * targets * (-log_p)

        shifted = self.neg_scale * (logits - self.bias_prior)
        log_1mp = F.logsigmoid(-shifted)
        neg_loss = (1.0 - targets) * (-log_1mp) / self.neg_scale

        return (pos_loss + neg_loss).mean()


class UncertaintyWeightedLoss(nn.Module):
    """Kendall, Gal & Cipolla (CVPR 2018) 4-task uncertainty-weighted loss."""

    _TASKS = ("obj", "evt", "attr", "nochg")

    def __init__(
        self,
        init_log_sigma: float = 0.0,
        log_sigma_min: float = -2.0,
        log_sigma_max: float = 2.0,
    ):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.full((4,), float(init_log_sigma)))
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
        with torch.no_grad():
            w = (0.5 * torch.exp(-self._clamped())).tolist()
        return dict(zip(self._TASKS, w))


class FixedWeightLoss(nn.Module):
    """L = Σᵢ wᵢ · Lᵢ — fixed task weights, drop-in for UncertaintyWeightedLoss."""

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
