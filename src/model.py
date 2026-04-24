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

        # The backbone's ``head`` module (pre-logits LayerNorm + classifier
        # FC) is bypassed by ``forward_features``. Freeze it so its weights
        # are not touched by optimizer weight-decay updates.
        if hasattr(self.backbone, "head"):
            for p in self.backbone.head.parameters():
                p.requires_grad_(False)

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


def _enhanced_4way_fusion(fa_vec: Tensor, fb_vec: Tensor) -> Tensor:
    """Return ``concat(fA, fB, |fA - fB|, fA ⊙ fB)`` along the last dim."""
    return torch.cat([fa_vec, fb_vec, (fa_vec - fb_vec).abs(), fa_vec * fb_vec], dim=-1)


class Phase1Model(nn.Module):
    """Phase 1 single-task model.

    Siamese encoder → global average pool → 4-way fusion
    ``[fA; fB; |fA - fB|; fA ⊙ fB]`` → FusionMLP → family head (``n_classes``
    sigmoid logits) + auxiliary no-change head (1 sigmoid logit).

    The same class template is reused for Object (12), Event (12), and
    Attribute (24) — only ``cfg['experiment']['n_classes']`` differs.

    Expects the following cfg keys:
        ``experiment.n_classes``: int, head output width.
        ``model.backbone``: timm checkpoint name.
        ``model.pretrained``: bool.
        ``model.droppath``: float (StochasticDepth rate).
        ``model.fusion_dim``: int (FusionMLP hidden width).
        ``model.dropout``: float (FusionMLP dropout).
    """

    def __init__(self, cfg: dict):
        super().__init__()
        model_cfg = cfg.get("model", {})
        n_classes = cfg["experiment"]["n_classes"]

        self.encoder = SiameseEncoder(
            name=model_cfg.get("backbone", "convnextv2_tiny.fcmae_ft_in22k_in1k"),
            pretrained=model_cfg.get("pretrained", True),
            drop_path_rate=model_cfg.get("droppath", 0.1),
        )
        dim = self.encoder.dim
        fusion_dim = model_cfg.get("fusion_dim", 512)
        dropout = model_cfg.get("dropout", 0.3)

        self.fusion = nn.Sequential(
            nn.Linear(4 * dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_family = nn.Linear(fusion_dim, n_classes)
        self.head_nochg = nn.Linear(fusion_dim, 1)

    def forward(self, a: Tensor, b: Tensor) -> dict[str, Tensor]:
        fa, fb = self.encoder(a, b)
        fa_vec = fa.mean(dim=(2, 3))
        fb_vec = fb.mean(dim=(2, 3))
        v = _enhanced_4way_fusion(fa_vec, fb_vec)
        feat = self.fusion(v)
        return {
            "logits_family": self.head_family(feat),
            "logit_nochg": self.head_nochg(feat).squeeze(-1),
        }
