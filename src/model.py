from __future__ import annotations

import logging

import timm
import timm.data
import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class SiameseEncoder(nn.Module):
    """Shared timm backbone applied to A and B in a single batched forward."""

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

        # forward_features bypasses backbone.head; freeze it so weight-decay leaves it alone.
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
        cfg = timm.data.resolve_model_data_config(self.backbone)
        return tuple(cfg["mean"]), tuple(cfg["std"])


def _enhanced_4way_fusion(fa_vec: Tensor, fb_vec: Tensor) -> Tensor:
    # concat(fA, fB, |fA - fB|, fA ⊙ fB)
    return torch.cat([fa_vec, fb_vec, (fa_vec - fb_vec).abs(), fa_vec * fb_vec], dim=-1)


class BITFusion(nn.Module):
    """L tokens per phase → joint pre-LN transformer → cross-attend back to spatial maps.
    Returns (fa', fb', refined_tokens) where refined_tokens is [B, 2L, dim] (A then B).
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


class Query2LabelHead(nn.Module):
    """One learnable query per class cross-attends to a shared memory → sigmoid logit each."""

    def __init__(
        self,
        n_classes: int,
        dim: int = 512,
        nhead: int = 8,
        dim_ff: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if dim_ff is None:
            dim_ff = 2 * dim
        self.n_classes = n_classes
        self.queries = nn.Parameter(torch.zeros(n_classes, dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        # Bias is a separate parameter so set_class_prior() can init it to log(p/(1-p))
        # without leaking O(1) random projection noise into the starting logits.
        self.proj = nn.Linear(dim, 1, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.class_bias = nn.Parameter(torch.zeros(n_classes))

    def forward(self, memory: Tensor) -> Tensor:
        b = memory.shape[0]
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        out = self.decoder(tgt=q, memory=memory)
        return self.proj(out).squeeze(-1) + self.class_bias

    @torch.no_grad()
    def set_class_prior(self, p_pos: Tensor, eps: float = 1e-3) -> None:
        if p_pos.shape != (self.n_classes,):
            raise ValueError(
                f"p_pos shape {tuple(p_pos.shape)} != ({self.n_classes},)"
            )
        p = p_pos.clamp(min=eps, max=1.0 - eps)
        self.class_bias.data.copy_(torch.log(p / (1.0 - p)))


class Phase1Model(nn.Module):
    """Siamese → GAP → 4-way fusion → MLP → family head (n_classes) + nochg head (1)."""

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


class Phase2Model(nn.Module):
    """Siamese → BITFusion → 3 Q2L heads (per family, shared memory) + nochg head."""

    _FAMILY_KEY = {"object": "logits_obj", "event": "logits_evt", "attribute": "logits_attr"}

    def __init__(self, cfg: dict):
        super().__init__()
        import json as _json
        from pathlib import Path as _Path

        exp_cfg = cfg["experiment"]
        model_cfg = cfg.get("model", {})
        fusion_cfg = model_cfg.get("fusion", {})
        heads_cfg = model_cfg.get("heads", {})

        families = list(exp_cfg.get("families", ["object", "event", "attribute"]))
        vocab_path = exp_cfg.get("label_vocab", "configs/label_vocab.json")
        vocab = _json.loads(_Path(vocab_path).read_text())
        n_classes = {fam: len(vocab[fam]) for fam in families}

        self.encoder = SiameseEncoder(
            name=model_cfg.get("backbone", "convnextv2_tiny.fcmae_ft_in22k_in1k"),
            pretrained=model_cfg.get("pretrained", True),
            drop_path_rate=model_cfg.get("droppath", 0.1),
        )
        backbone_dim = self.encoder.dim
        fusion_dim = int(model_cfg.get("fusion_dim", 512))
        fusion_dropout = float(model_cfg.get("dropout", 0.3))
        head_dim = int(heads_cfg.get("dim", fusion_dim))
        head_nhead = int(heads_cfg.get("nhead", 8))
        head_dropout = float(heads_cfg.get("dropout", 0.1))

        head_type = heads_cfg.get("type", "query2label")
        if head_type not in ("query2label", "linear"):
            raise ValueError(f"unknown model.heads.type {head_type!r}")
        self.head_type = head_type

        fusion_type = fusion_cfg.get("type", "bit")
        if fusion_type not in ("bit", "passthrough"):
            raise ValueError(f"unknown model.fusion.type {fusion_type!r}")
        self.fusion_type = fusion_type
        if fusion_type == "bit":
            self.bit_fusion = BITFusion(
                dim=backbone_dim,
                L=int(fusion_cfg.get("L", 4)),
                nhead=int(fusion_cfg.get("nhead", 8)),
                depth=int(fusion_cfg.get("depth", 2)),
                dim_ff=fusion_cfg.get("dim_ff"),
                dropout=float(fusion_cfg.get("dropout", 0.1)),
            )
        else:
            # passthrough is incompatible with Q2L (no refined tokens to use as memory).
            if head_type == "query2label":
                raise ValueError(
                    "model.fusion.type='passthrough' is incompatible with "
                    "model.heads.type='query2label' (no refined tokens for "
                    "decoder memory). Use heads.type='linear'."
                )
            self.bit_fusion = None

        self.fusion_mlp = nn.Sequential(
            nn.Linear(4 * backbone_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
        )

        if head_type == "query2label":
            self.token_proj = nn.Linear(backbone_dim, head_dim)
            self.feat_proj = (
                nn.Identity() if fusion_dim == head_dim
                else nn.Linear(fusion_dim, head_dim)
            )
            self.heads = nn.ModuleDict({
                fam: Query2LabelHead(
                    n_classes=n_classes[fam],
                    dim=head_dim,
                    nhead=head_nhead,
                    dropout=head_dropout,
                )
                for fam in families
            })
        else:
            self.heads = nn.ModuleDict({
                fam: nn.Linear(fusion_dim, n_classes[fam])
                for fam in families
            })

        self.head_nochg = nn.Linear(fusion_dim, 1)
        self.families = families

    def forward(self, a: Tensor, b: Tensor) -> dict[str, Tensor]:
        fa, fb = self.encoder(a, b)
        refined_tokens: Tensor | None = None
        if self.fusion_type == "bit":
            assert self.bit_fusion is not None
            fa_ref, fb_ref, refined_tokens = self.bit_fusion(fa, fb)
        else:
            fa_ref, fb_ref = fa, fb

        fa_vec = fa_ref.mean(dim=(2, 3))
        fb_vec = fb_ref.mean(dim=(2, 3))
        v = _enhanced_4way_fusion(fa_vec, fb_vec)
        feat = self.fusion_mlp(v)

        if self.head_type == "query2label":
            assert refined_tokens is not None
            feat_q = self.feat_proj(feat).unsqueeze(1)
            tokens_q = self.token_proj(refined_tokens)
            memory = torch.cat([feat_q, tokens_q], dim=1)
            out: dict[str, Tensor] = {
                self._FAMILY_KEY[fam]: self.heads[fam](memory)
                for fam in self.families
            }
        else:
            out = {
                self._FAMILY_KEY[fam]: self.heads[fam](feat)
                for fam in self.families
            }
        out["logit_nochg"] = self.head_nochg(feat).squeeze(-1)
        return out

    def set_class_priors(self, priors: dict[str, Tensor]) -> None:
        for fam, p in priors.items():
            if fam not in self.heads:
                raise KeyError(f"family {fam!r} not in heads ({list(self.heads)})")
            head = self.heads[fam]
            if self.head_type == "query2label":
                head.set_class_prior(p)
            else:
                eps = 1e-3
                p_clamped = p.clamp(min=eps, max=1.0 - eps)
                head.bias.data.copy_(torch.log(p_clamped / (1.0 - p_clamped)))


def build_model(cfg: dict) -> Phase1Model | Phase2Model:
    phase = cfg["experiment"]["phase"]
    if phase == 1:
        return Phase1Model(cfg)
    if phase == 2:
        return Phase2Model(cfg)
    raise ValueError(f"experiment.phase must be 1 or 2, got {phase!r}")
