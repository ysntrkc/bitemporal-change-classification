"""Exponential Moving Average of model weights.

Simple, self-contained implementation. Decay ramps from ~0 to the
configured value over the first ~``warmup_steps`` steps so early, noisy
weights don't dominate the average.
"""

from __future__ import annotations

import copy
import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class ModelEma:
    """Maintain an EMA copy of a model's parameters and buffers.

    Usage::

        ema = ModelEma(model, decay=0.999, warmup_steps=1000)
        ...
        loss.backward(); optimizer.step()
        ema.update(model)
        ...
        # at eval time
        ema.module.eval()
        out = ema.module(inputs)
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        warmup_steps: int = 1000,
    ):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self._step = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend current model weights into the EMA copy."""
        self._step += 1
        # Ramp: d grows from ~0 at step 1 to target `decay` around `warmup_steps`.
        d = min(self.decay, (1.0 + self._step) / (10.0 + self._step))
        for ema_p, p in zip(self.module.parameters(), model.parameters()):
            ema_p.mul_(d).add_(p.detach(), alpha=1.0 - d)
        # Copy buffers (BN running mean/var, etc.) directly.
        for ema_b, b in zip(self.module.buffers(), model.buffers()):
            ema_b.copy_(b)

    def state_dict(self) -> dict:
        return self.module.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.module.load_state_dict(sd)
