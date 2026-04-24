"""Augmentation pipeline for bitemporal training samples.

Provides ``PairAug`` (per-sample spatial + photometric) and
``cutmix_pair`` (batch-level pair-wise CutMix). Used for the train
split only. See PROJECT_PLAN.md §6.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Sequence

import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import functional as TF

logger = logging.getLogger(__name__)


class PairAug:
    """Pair-aware augmentation for ``(A, B)`` bitemporal pairs.

    Identical spatial transforms (horizontal flip, vertical flip,
    90/180/270° rotation, small affine) are applied to both phases so
    correspondence is preserved. Photometric jitter (brightness /
    contrast / saturation) is sampled independently per phase. Outputs
    are resized, converted to tensors, and normalized.

    Construct with ``PairAug.from_cfg(cfg, mean, std)`` to respect the
    schema in PROJECT_PLAN.md §10.1.
    """

    def __init__(
        self,
        img_size: int = 224,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
        p_hflip: float = 0.5,
        p_vflip: float = 0.5,
        p_rot90: float = 0.5,
        p_affine: float = 0.5,
        affine_deg: float = 5.0,
        affine_trans: float = 0.05,
        affine_scale: tuple[float, float] = (0.95, 1.05),
        p_jitter: float = 0.8,
        jitter_brightness: float = 0.2,
        jitter_contrast: float = 0.2,
        jitter_saturation: float = 0.2,
    ):
        self.img_size = img_size
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_rot90 = p_rot90
        self.p_affine = p_affine
        self.affine_deg = affine_deg
        self.affine_trans = affine_trans
        self.affine_scale = affine_scale
        self.p_jitter = p_jitter
        self.jitter = transforms.ColorJitter(
            brightness=jitter_brightness,
            contrast=jitter_contrast,
            saturation=jitter_saturation,
        )
        self.normalize = transforms.Normalize(mean=list(mean), std=list(std))

    @classmethod
    def from_cfg(
        cls, cfg: dict, mean: Sequence[float], std: Sequence[float]
    ) -> "PairAug":
        """Build from a config dict per PROJECT_PLAN.md §10.1."""
        aug = cfg.get("augment", {})
        spatial = aug.get("spatial", {})
        photo = aug.get("photometric", {})
        data = cfg.get("data", {})
        return cls(
            img_size=data.get("img_size", 224),
            mean=mean,
            std=std,
            p_hflip=spatial.get("hflip", 0.5),
            p_vflip=spatial.get("vflip", 0.5),
            p_rot90=spatial.get("rot90", 0.5),
            p_affine=spatial.get("affine", 0.5),
            affine_deg=spatial.get("affine_deg", 5.0),
            affine_trans=spatial.get("affine_trans", 0.05),
            p_jitter=photo.get("p", 0.8),
            jitter_brightness=photo.get("brightness", 0.2),
            jitter_contrast=photo.get("contrast", 0.2),
            jitter_saturation=photo.get("saturation", 0.2),
        )

    def __call__(self, a: Image.Image, b: Image.Image) -> tuple[Tensor, Tensor]:
        a = a.resize((self.img_size, self.img_size), Image.BILINEAR)
        b = b.resize((self.img_size, self.img_size), Image.BILINEAR)

        if random.random() < self.p_hflip:
            a, b = TF.hflip(a), TF.hflip(b)
        if random.random() < self.p_vflip:
            a, b = TF.vflip(a), TF.vflip(b)
        if random.random() < self.p_rot90:
            angle = random.choice([90, 180, 270])
            a = TF.rotate(a, angle)
            b = TF.rotate(b, angle)
        if random.random() < self.p_affine:
            angle = random.uniform(-self.affine_deg, self.affine_deg)
            translate = [
                random.uniform(-self.affine_trans, self.affine_trans) * self.img_size,
                random.uniform(-self.affine_trans, self.affine_trans) * self.img_size,
            ]
            scale = random.uniform(*self.affine_scale)
            a = TF.affine(a, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0])
            b = TF.affine(b, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0])

        if random.random() < self.p_jitter:
            a = self.jitter(a)
        if random.random() < self.p_jitter:
            b = self.jitter(b)

        return self.normalize(TF.to_tensor(a)), self.normalize(TF.to_tensor(b))


class EvalTransform:
    """Deterministic resize + normalize for val and test (no augmentation).

    Same signature as ``PairAug.__call__``: takes two PIL images, returns
    two normalized tensors.
    """

    def __init__(
        self,
        img_size: int = 224,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ):
        self.img_size = img_size
        self.normalize = transforms.Normalize(mean=list(mean), std=list(std))

    def __call__(self, a: Image.Image, b: Image.Image) -> tuple[Tensor, Tensor]:
        a = a.resize((self.img_size, self.img_size), Image.BILINEAR)
        b = b.resize((self.img_size, self.img_size), Image.BILINEAR)
        return self.normalize(TF.to_tensor(a)), self.normalize(TF.to_tensor(b))


def cutmix_pair(batch: dict, p: float = 0.3, alpha: float = 1.0) -> dict:
    """Apply pair-wise CutMix to an already-collated training batch.

    With probability ``p``, sample ``λ ∼ Beta(α, α)``, pick a random
    rectangle covering roughly ``1 - λ`` of the image area, and paste
    it from a shuffled permutation of the batch onto both ``A`` and
    ``B`` (same rectangle, same permutation, preserving correspondence).
    Labels in ``y_obj`` / ``y_evt`` / ``y_attr`` / ``is_change`` are
    mixed linearly using the *actual* area fraction (which differs from
    the sampled λ due to integer pixel rounding).

    The batch is mutated in place and also returned.
    """
    if random.random() >= p:
        return batch

    n = batch["A"].shape[0]
    perm = torch.randperm(n)

    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    h, w = batch["A"].shape[2], batch["A"].shape[3]

    cut_ratio = math.sqrt(1.0 - lam)
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)
    cx = random.randint(0, w - 1)
    cy = random.randint(0, h - 1)
    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(w, cx + cut_w // 2)
    y2 = min(h, cy + cut_h // 2)

    for key in ("A", "B"):
        batch[key][:, :, y1:y2, x1:x2] = batch[key][perm, :, y1:y2, x1:x2]

    lam_actual = 1.0 - ((x2 - x1) * (y2 - y1)) / (h * w)
    for key in ("y_obj", "y_evt", "y_attr", "is_change"):
        if key in batch:
            batch[key] = lam_actual * batch[key] + (1.0 - lam_actual) * batch[key][perm]

    return batch
