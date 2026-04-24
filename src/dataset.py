"""Data utilities for the bitemporal change classification project.

Currently provides label-vocabulary construction. The dataset class,
dataloader builder, leakage report, and EDA helpers are added in later
tasks.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import functional as TF

logger = logging.getLogger(__name__)

LABEL_FAMILIES: tuple[str, ...] = ("object", "event", "attribute")
_NONE_LABEL = "none"


def build_label_vocab(json_path: str | Path) -> dict[str, list[str]]:
    """Build per-family label vocabularies from a ``dataset.json`` file.

    Scans every record in the ``images`` array and collects the unique label
    strings for each family. The sentinel ``"none"`` is excluded; no-change
    samples are represented downstream as all-zeros multi-hot vectors in
    every family.

    Args:
        json_path: Path to ``dataset.json``.

    Returns:
        A dict with keys ``"object"``, ``"event"``, ``"attribute"`` mapping
        to alphabetically sorted label lists (deterministic across runs).
    """
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        blob = json.load(f)
    records = blob["images"]

    collected: dict[str, set[str]] = {family: set() for family in LABEL_FAMILIES}
    for rec in records:
        for family in LABEL_FAMILIES:
            for label in rec.get(f"{family}_labels", []):
                if label != _NONE_LABEL:
                    collected[family].add(label)

    return {family: sorted(collected[family]) for family in LABEL_FAMILIES}


def leakage_report(json_path: str | Path) -> dict:
    """Detect base filenames that span more than one split.

    Some train records are pre-computed augmentations named
    ``<base>_random_augment.png``. They are expected to share their base
    filename with an un-augmented record in the **same** split. A leakage
    is any base filename that appears across two or more of
    ``{train, val, test}``.

    Args:
        json_path: Path to ``dataset.json``.

    Returns:
        A dict with keys ``n_violations`` (int) and ``examples`` (list of
        up to 50 entries, each ``{"base", "splits", "filenames"}``).
        Deterministic: violations are sorted by base filename.
    """
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        blob = json.load(f)
    records = blob["images"]

    grouped: dict[str, dict] = {}
    for rec in records:
        base = rec["filename"].replace("_random_augment", "")
        entry = grouped.setdefault(base, {"splits": set(), "filenames": []})
        entry["splits"].add(rec["split"])
        entry["filenames"].append(rec["filename"])

    violations = [
        {
            "base": base,
            "splits": sorted(entry["splits"]),
            "filenames": sorted(entry["filenames"]),
        }
        for base, entry in sorted(grouped.items())
        if len(entry["splits"]) > 1
    ]

    return {"n_violations": len(violations), "examples": violations[:50]}


class BitempDataset(Dataset):
    """Bitemporal change classification dataset backed by ``dataset.json``.

    Each ``__getitem__`` returns a dict with keys ``"A"``, ``"B"`` (float32
    tensors shaped ``[3, img_size, img_size]``), ``"y_obj"``, ``"y_evt"``,
    ``"y_attr"`` (float32 multi-hot vectors), ``"is_change"`` (float32
    scalar), and ``"sample_id"`` (str).

    The ``transform`` callable, if given, is invoked as ``transform(A, B)``
    on the two PIL images and must return a pair of tensors — it owns
    resize and normalization. When ``transform`` is ``None``, a minimal
    resize + ToTensor fallback is used (un-normalized, ``[0, 1]``); this
    path is intended for smoke tests only — production loaders wire in a
    ``PairAug``-style callable.
    """

    def __init__(
        self,
        json_path: str,
        root: str,
        split: Literal["train", "val", "test"],
        label_vocab: dict[str, list[str]],
        transform: Optional[Callable] = None,
        img_size: int = 224,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be one of train/val/test, got {split!r}")
        for family in LABEL_FAMILIES:
            if family not in label_vocab:
                raise KeyError(f"label_vocab missing family {family!r}")

        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.img_size = img_size

        with Path(json_path).open("r", encoding="utf-8") as f:
            blob = json.load(f)
        self.records: list[dict] = [r for r in blob["images"] if r["split"] == split]

        self._label_to_idx: dict[str, dict[str, int]] = {
            family: {lbl: i for i, lbl in enumerate(label_vocab[family])}
            for family in LABEL_FAMILIES
        }
        self._n_classes: dict[str, int] = {
            family: len(label_vocab[family]) for family in LABEL_FAMILIES
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        a_pil = Image.open(self.root / rec["rgb_A"]).convert("RGB")
        b_pil = Image.open(self.root / rec["rgb_B"]).convert("RGB")

        if self.transform is not None:
            a_tensor, b_tensor = self.transform(a_pil, b_pil)
        else:
            a_pil = a_pil.resize((self.img_size, self.img_size), Image.BILINEAR)
            b_pil = b_pil.resize((self.img_size, self.img_size), Image.BILINEAR)
            a_tensor = TF.to_tensor(a_pil)
            b_tensor = TF.to_tensor(b_pil)

        return {
            "A": a_tensor,
            "B": b_tensor,
            "y_obj": self._encode_family(rec, "object"),
            "y_evt": self._encode_family(rec, "event"),
            "y_attr": self._encode_family(rec, "attribute"),
            "is_change": torch.tensor(float(rec["changeflag"]), dtype=torch.float32),
            "sample_id": rec["sample_id"],
        }

    def _encode_family(self, rec: dict, family: str) -> torch.Tensor:
        vec = torch.zeros(self._n_classes[family], dtype=torch.float32)
        for label in rec.get(f"{family}_labels", []):
            idx = self._label_to_idx[family].get(label)
            if idx is not None:
                vec[idx] = 1.0
        return vec


def _compute_sampler_weights(
    records: list[dict], phase: int, family: Optional[str]
) -> torch.Tensor:
    """Per-sample weights for ``WeightedRandomSampler``.

    Uses ``w_i = 1 / sqrt(1 + n_positives_i)`` per PROJECT_PLAN.md §4.3.
    In Phase 1, ``n_positives`` counts positive labels in the target
    family only; in Phase 2, it sums positives across all three families.
    """
    if phase == 1 and family is None:
        raise ValueError("phase 1 requires cfg['experiment']['family']")

    weights = torch.empty(len(records), dtype=torch.double)
    for i, rec in enumerate(records):
        if phase == 1:
            n_pos = sum(
                1 for label in rec.get(f"{family}_labels", []) if label != _NONE_LABEL
            )
        else:
            n_pos = sum(
                1
                for fam in LABEL_FAMILIES
                for label in rec.get(f"{fam}_labels", [])
                if label != _NONE_LABEL
            )
        weights[i] = 1.0 / math.sqrt(1.0 + n_pos)
    return weights


def _worker_init_fn(worker_id: int) -> None:
    """Deterministic per-worker seeding."""
    seed = torch.utils.data.get_worker_info().seed % (2**32)
    random.seed(seed + worker_id)
    np.random.seed((seed + worker_id) % (2**32))


def build_dataloaders(
    cfg: dict,
    transform_train: Optional[Callable] = None,
    transform_eval: Optional[Callable] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build ``(train, val, test)`` loaders from a config dict.

    Train uses ``WeightedRandomSampler`` (see ``_compute_sampler_weights``).
    Val and test iterate in dataset-order, no shuffling. The caller supplies
    augmentation + normalization via ``transform_train`` and eval-time
    resize + normalization via ``transform_eval``; both default to ``None``
    so the ``BitempDataset`` fallback (resize + ToTensor, un-normalized)
    runs — useful for smoke tests only.

    Args:
        cfg: Nested dict with ``data``, ``train``, and ``experiment`` keys
            (see PROJECT_PLAN.md §10 for the schema).
        transform_train: Optional ``(PIL, PIL) -> (Tensor, Tensor)`` callable
            applied to the train split.
        transform_eval: Optional eval-time transform applied to val and test.

    Returns:
        ``(train_loader, val_loader, test_loader)``.
    """
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    exp_cfg = cfg["experiment"]

    with Path(data_cfg["vocab_path"]).open("r", encoding="utf-8") as f:
        vocab = json.load(f)

    img_size = data_cfg.get("img_size", 224)
    num_workers = data_cfg.get("num_workers", 4)
    pin_memory = data_cfg.get("pin_memory", True)
    batch_size = train_cfg["batch_size"]
    seed = exp_cfg["seed"]
    phase = exp_cfg.get("phase", 1)
    family = exp_cfg.get("family")

    common = {
        "json_path": data_cfg["json_path"],
        "root": data_cfg["root"],
        "label_vocab": vocab,
        "img_size": img_size,
    }
    train_ds = BitempDataset(split="train", transform=transform_train, **common)
    val_ds = BitempDataset(split="val", transform=transform_eval, **common)
    test_ds = BitempDataset(split="test", transform=transform_eval, **common)

    weights = _compute_sampler_weights(train_ds.records, phase=phase, family=family)
    sampler_gen = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(train_ds),
        replacement=True,
        generator=sampler_gen,
    )
    loader_gen = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        worker_init_fn=_worker_init_fn,
        generator=loader_gen,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, test_loader


def _cmd_build_vocab(args: argparse.Namespace) -> int:
    vocab = build_label_vocab(args.json)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {out_path}")
    for family, labels in vocab.items():
        print(f"  {family:9s} {len(labels):3d} labels")
    return 0


def _cmd_leakage_check(args: argparse.Namespace) -> int:
    report = leakage_report(args.json)
    n = report["n_violations"]
    if n == 0:
        print("leakage-check: OK — every base filename is confined to a single split")
    else:
        print(f"leakage-check: FAIL — {n} base filename(s) span multiple splits")
        for v in report["examples"][:10]:
            print(f"  {v['base']}: {', '.join(v['splits'])}")
        if n > 10:
            print(f"  … and {n - 10} more")
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        print(f"wrote {out_path}")
    return 0 if n == 0 else 1


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for `python -m src.dataset`."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="src.dataset")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_vocab = sub.add_parser("build-vocab", help="build per-family label vocabulary")
    p_vocab.add_argument("--json", required=True, help="path to dataset.json")
    p_vocab.add_argument("--out", required=True, help="output path for label_vocab.json")
    p_vocab.set_defaults(func=_cmd_build_vocab)

    p_leak = sub.add_parser("leakage-check", help="detect base filenames that span splits")
    p_leak.add_argument("--json", required=True, help="path to dataset.json")
    p_leak.add_argument("--out", default=None, help="optional path to persist the JSON report")
    p_leak.set_defaults(func=_cmd_leakage_check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
