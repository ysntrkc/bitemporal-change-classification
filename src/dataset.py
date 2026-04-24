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


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for `python -m src.dataset`."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="src.dataset")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_vocab = sub.add_parser("build-vocab", help="build per-family label vocabulary")
    p_vocab.add_argument("--json", required=True, help="path to dataset.json")
    p_vocab.add_argument("--out", required=True, help="output path for label_vocab.json")
    p_vocab.set_defaults(func=_cmd_build_vocab)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
