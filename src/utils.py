"""Training utilities: seeding, optimizer (LLRD), scheduler, checkpoint I/O."""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.AdamW:
    """AdamW with layer-wise LR decay on the backbone, head LR for everything else.

    Uses two hyperparameters from ``cfg['train']``: ``backbone_lr`` and
    ``head_lr``. Parameters under ``encoder.backbone`` are assigned a
    depth — ``stem → 0``, ``stages.i → i + 1`` — and receive
    ``backbone_lr * llrd^(max_depth - depth)``. All other trainable
    parameters (fusion, heads, Phase-2 transformer, queries) get
    ``head_lr``. Weight decay is zeroed for 1-D parameters (LayerNorm,
    bias) per standard practice.
    """
    train_cfg = cfg["train"]
    wd = float(train_cfg.get("wd", 0.05))
    head_lr = float(train_cfg["head_lr"])
    backbone_lr = float(train_cfg["backbone_lr"])
    llrd = float(train_cfg.get("llrd", 0.8))

    backbone = model.encoder.backbone
    n_stages = len(getattr(backbone, "stages", []))
    max_depth = n_stages

    groups: dict[tuple[float, float], list[torch.Tensor]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder.backbone."):
            rel = name[len("encoder.backbone."):]
            if rel.startswith("stem"):
                depth = 0
            elif rel.startswith("stages."):
                depth = int(rel.split(".")[1]) + 1
            else:
                depth = max_depth
            lr = backbone_lr * (llrd ** (max_depth - depth))
        else:
            lr = head_lr
        wd_val = 0.0 if (p.ndim == 1 or name.endswith(".bias")) else wd
        groups.setdefault((lr, wd_val), []).append(p)

    param_groups = [
        {"params": params, "lr": lr, "weight_decay": wd_val}
        for (lr, wd_val), params in groups.items()
    ]
    return torch.optim.AdamW(param_groups, lr=head_lr, betas=(0.9, 0.999), eps=1e-8)


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: dict, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine decay to ``min_lr / head_lr`` ratio.

    Each parameter group decays proportionally from its initial LR toward
    ``min_lr`` (so backbone groups end up even lower, preserving the LLRD
    ratio throughout training).
    """
    train_cfg = cfg["train"]
    epochs = int(train_cfg["epochs"])
    warmup_epochs = int(train_cfg.get("warmup_epochs", 3))
    min_lr = float(train_cfg.get("min_lr", 1e-7))
    head_lr = float(train_cfg["head_lr"])

    warmup_steps = max(1, int(total_steps * warmup_epochs / max(1, epochs)))
    eta_ratio = min_lr / head_lr

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_ratio + (1.0 - eta_ratio) * cos

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    ema: Optional[Any] = None,
    epoch: int = 0,
    metrics: Optional[dict] = None,
) -> None:
    """Write a single-file checkpoint. Creates parent dirs as needed."""
    state: dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if ema is not None:
        state["ema"] = ema.state_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str) -> dict:
    """Load a checkpoint onto CPU. Caller moves tensors to device as needed."""
    return torch.load(path, map_location="cpu", weights_only=False)
