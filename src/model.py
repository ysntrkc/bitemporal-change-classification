"""Model components: Siamese encoder, fusion, classification heads.

Phase 1: ``SiameseEncoder`` → 4-way fusion → ``Phase1Model``.
Phase 2: adds ``BITFusion`` (token transformer + cross-attend),
``Query2LabelHead`` (transformer-decoder classifier), and ``Phase2Model``.
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


class BITFusion(nn.Module):
    """Bitemporal token transformer + cross-attended spatial refinement.

    Each phase's spatial map ``f ∈ [B, dim, H, W]`` is summarised into
    ``L`` learnable tokens via a shared 1×1-conv attention (softmax over
    HW). The concatenated ``[tA ; tB] ∈ [B, 2L, dim]`` (with a learnable
    positional + temporal embedding) is jointly refined by a small pre-LN
    transformer encoder. Each phase's spatial features then cross-attend
    to *its* refined tokens, yielding spatially-aware refined feature
    maps with the original ``[B, dim, H, W]`` shape.

    Returns ``(fa', fb', refined_tokens)`` where ``refined_tokens ∈
    [B, 2L, dim]`` (first L are A-tokens, last L are B-tokens) — useful
    as decoder memory for ``Query2LabelHead``.
    """

    def __init__(
        self,
        dim: int = 768,
        L: int = 4,
        nhead: int = 8,
        depth: int = 2,
        dim_ff: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.L = L
        if dim_ff is None:
            dim_ff = 2 * dim

        # Shared per-phase tokenizer: 1×1 conv → L attention maps over HW.
        self.tokenizer = nn.Conv2d(dim, L, kernel_size=1)

        # Learnable position + temporal embedding for [tA ; tB].
        self.pos_emb = nn.Parameter(torch.zeros(2 * L, dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Shared cross-attention from spatial-feature queries to refined tokens.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=nhead, dropout=dropout, batch_first=True,
        )
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def _tokenize(self, f: Tensor) -> Tensor:
        """``f: [B, dim, H, W]`` → tokens ``[B, L, dim]`` via spatial-softmax weighted sum."""
        attn = self.tokenizer(f).flatten(2)        # [B, L, HW]
        attn = attn.softmax(dim=-1)
        flat = f.flatten(2)                        # [B, dim, HW]
        return torch.einsum("blp,bcp->blc", attn, flat)

    def _cross_refine(self, f: Tensor, tokens: Tensor) -> Tensor:
        """Cross-attend spatial features to their refined tokens; residual add."""
        b, d, h, w = f.shape
        flat = f.flatten(2).transpose(1, 2)        # [B, HW, dim]
        q = self.norm_q(flat)
        kv = self.norm_kv(tokens)
        out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        return (flat + out).transpose(1, 2).reshape(b, d, h, w)

    def forward(self, fa: Tensor, fb: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        ta = self._tokenize(fa)                    # [B, L, dim]
        tb = self._tokenize(fb)                    # [B, L, dim]
        tokens = torch.cat([ta, tb], dim=1) + self.pos_emb   # broadcast on batch
        refined = self.encoder(tokens)             # [B, 2L, dim]
        ta_r, tb_r = refined[:, : self.L], refined[:, self.L :]
        return self._cross_refine(fa, ta_r), self._cross_refine(fb, tb_r), refined


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
