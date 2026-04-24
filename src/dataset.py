"""Data utilities for the bitemporal change classification project.

Currently provides label-vocabulary construction. The dataset class,
dataloader builder, leakage report, and EDA helpers are added in later
tasks.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Callable, Literal, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
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
