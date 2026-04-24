"""Model components: Siamese encoder, fusion, classification heads.

Currently contains ``SiameseEncoder``. ``Phase1Model``, ``BITFusion``,
``Query2LabelHead``, and ``Phase2Model`` are added in tasks 1.11 and 3.1–3.3.
"""

from __future__ import annotations

import logging

import timm
import timm.data
import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class SiameseEncoder(nn.Module):
    """Shared-weight Siamese encoder around a timm backbone.

    A and B are concatenated along the batch dimension and forwarded
    through a single backbone call (efficiency + identical BatchNorm
    statistics, per CLAUDE.md §5.2). Outputs are split back into
    ``(fA, fB)``.

    For the default ``convnextv2_tiny.fcmae_ft_in22k_in1k`` checkpoint
    at 224² input, ``fA`` and ``fB`` have shape ``[B, 768, 7, 7]``.

    Attributes:
        backbone: Underlying timm model, no classifier, no global pool.
        dim: Number of output channels (``backbone.num_features``).
        name: timm checkpoint identifier.
    """

    def __init__(
        self,
        name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
        pretrained: bool = True,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            name,
            pretrained=pretrained,
            features_only=False,
            num_classes=0,
            global_pool="",
            drop_path_rate=drop_path_rate,
        )
        self.name = name
        self.dim = self.backbone.num_features

    def forward(self, a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
        if a.shape != b.shape:
            raise ValueError(f"A and B shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
        x = torch.cat([a, b], dim=0)
        f = self.backbone.forward_features(x)
        fa, fb = f.chunk(2, dim=0)
        return fa, fb

    def norm_stats(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return ``(mean, std)`` for input normalization, from the backbone's timm config."""
        cfg = timm.data.resolve_model_data_config(self.backbone)
        return tuple(cfg["mean"]), tuple(cfg["std"])
