"""Evaluation entry point for the Phase-2 unified multi-task model.

Forwards the chosen split through the loaded checkpoint, optionally
with TTA (orig/hflip/vflip/rot180 averaged in probability space) and
optionally with the no-change gate applied (multiply each family prob
by ``P(change) = sigmoid(logit_nochg)``). Saves per-family + mean
metrics to ``metrics_<split>[_tta][_gate].json`` next to the checkpoint.

Examples::

    # Default eval (test, no TTA, gate per cfg.inference.nochg_gate)
    python eval_phase2.py --ckpt results/phase2_unified/seed42/best_ema.pth \\
                          --config configs/phase2_unified.yaml

    # Test + TTA + gate (the canonical Phase-2 number)
    python eval_phase2.py --ckpt results/phase2_unified/seed42/best_ema.pth \\
                          --config configs/phase2_unified.yaml --tta

    # Same but disable the gate (for the gate ablation)
    python eval_phase2.py --ckpt results/phase2_unified/seed42/best_ema.pth \\
                          --config configs/phase2_unified.yaml --tta --no-gate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import yaml

from src.augment import EvalTransform
from src.dataset import build_dataloaders
from src.metrics import compute_metrics
from src.model import Phase2Model
from src.utils import load_checkpoint, seed_everything

logger = logging.getLogger(__name__)

FAMILY_LOGITS_KEY = {"object": "logits_obj", "event": "logits_evt", "attribute": "logits_attr"}
FAMILY_Y_KEY = {"object": "y_obj", "event": "y_evt", "attribute": "y_attr"}
_TTA_OPS = ("orig", "hflip", "vflip", "rot180")


def _apply_tta(x: torch.Tensor, op: str) -> torch.Tensor:
    if op == "orig":
        return x
    if op == "hflip":
        return torch.flip(x, dims=[-1])
    if op == "vflip":
        return torch.flip(x, dims=[-2])
    if op == "rot180":
        return torch.flip(x, dims=[-2, -1])
    raise ValueError(f"unknown TTA op {op!r}; supported: {_TTA_OPS}")


@torch.no_grad()
def collect_probs_phase2(
    model: torch.nn.Module,
    loader,
    families: list[str],
    tta_ops: Sequence[str],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    """Forward the loader with TTA averaging in probability space.

    Returns ``(family_probs, change_probs, family_targets)`` where
    ``family_probs[fam]`` is ``[N, C_fam]`` and ``change_probs`` is ``[N]``.
    """
    if len(tta_ops) == 0:
        raise ValueError("tta_ops must be non-empty")
    unknown = [op for op in tta_ops if op not in _TTA_OPS]
    if unknown:
        raise ValueError(f"unknown TTA ops: {unknown}; supported: {_TTA_OPS}")

    model.eval()
    fam_probs: dict[str, list[np.ndarray]] = {fam: [] for fam in families}
    fam_targets: dict[str, list[np.ndarray]] = {fam: [] for fam in families}
    change_probs_list: list[np.ndarray] = []
    n = float(len(tta_ops))

    for batch in loader:
        a = batch["A"].to(device, non_blocking=True)
        b = batch["B"].to(device, non_blocking=True)

        fam_sum: dict[str, Optional[torch.Tensor]] = {fam: None for fam in families}
        change_sum: Optional[torch.Tensor] = None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for op in tta_ops:
                out = model(_apply_tta(a, op), _apply_tta(b, op))
                p_change = torch.sigmoid(out["logit_nochg"].float())
                change_sum = p_change if change_sum is None else change_sum + p_change
                for fam in families:
                    p = torch.sigmoid(out[FAMILY_LOGITS_KEY[fam]].float())
                    fam_sum[fam] = p if fam_sum[fam] is None else fam_sum[fam] + p

        change_probs_list.append((change_sum / n).cpu().numpy())  # type: ignore[operator]
        for fam in families:
            fam_probs[fam].append((fam_sum[fam] / n).cpu().numpy())  # type: ignore[operator]
            fam_targets[fam].append(batch[FAMILY_Y_KEY[fam]].cpu().numpy())

    out_probs = {fam: np.concatenate(fam_probs[fam], axis=0) for fam in families}
    out_targets = {fam: np.concatenate(fam_targets[fam], axis=0) for fam in families}
    out_change = np.concatenate(change_probs_list, axis=0)
    return out_probs, out_change, out_targets


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(prog="eval_phase2.py")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--tta", action="store_true",
                        help="average orig/hflip/vflip/rot180 probabilities")
    parser.add_argument("--no-gate", action="store_true",
                        help="override config: disable the no-change gate")
    parser.add_argument("--gate", action="store_true",
                        help="override config: force the no-change gate on")
    parser.add_argument("--output", type=str, default=None,
                        help="output JSON path (default: next to ckpt)")
    args = parser.parse_args(argv)

    if args.no_gate and args.gate:
        parser.error("--no-gate and --gate are mutually exclusive")

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = int(cfg["experiment"].get("seed", 42))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    families = list(cfg["experiment"]["families"])
    cfg_gate = bool(cfg.get("inference", {}).get("nochg_gate", False))
    if args.no_gate:
        gate = False
    elif args.gate:
        gate = True
    else:
        gate = cfg_gate

    tta_ops = list(_TTA_OPS) if args.tta else ["orig"]

    model = Phase2Model(cfg).to(device)
    ckpt = load_checkpoint(args.ckpt)
    model.load_state_dict(ckpt["model"])
    mean, std = model.encoder.norm_stats()
    transform = EvalTransform(img_size=int(cfg["data"].get("img_size", 224)),
                              mean=mean, std=std)
    train_loader, val_loader, test_loader = build_dataloaders(
        cfg, transform_train=transform, transform_eval=transform
    )
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]
    logger.info("split=%s | tta=%s | gate=%s | n_batches=%d",
                args.split, tta_ops, gate, len(loader))

    fam_probs, change_probs, fam_targets = collect_probs_phase2(
        model, loader, families, tta_ops, device
    )

    metrics: dict = {
        "ckpt": str(args.ckpt),
        "config": str(args.config),
        "split": args.split,
        "tta": tta_ops,
        "nochg_gate": gate,
    }
    macro_f1s: list[float] = []
    for fam in families:
        probs = fam_probs[fam]
        if gate:
            probs = probs * change_probs[:, None]
        n_classes = probs.shape[1]
        m = compute_metrics(probs, fam_targets[fam], thresholds=np.full(n_classes, 0.5))
        metrics[fam] = m
        macro_f1s.append(m["macro_f1"])
        logger.info("  %s: macro_f1=%.4f micro_f1=%.4f mAP=%.4f",
                    fam, m["macro_f1"], m["micro_f1"], m["mAP"])
    metrics["macro_f1_mean"] = float(np.mean(macro_f1s))
    logger.info("mean macro_f1 across families = %.4f", metrics["macro_f1_mean"])

    if args.output:
        out_path = Path(args.output)
    else:
        ckpt_dir = Path(args.ckpt).parent
        suffix = ""
        if args.tta:
            suffix += "_tta"
        if gate:
            suffix += "_gate"
        out_path = ckpt_dir / f"metrics_{args.split}{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    logger.info("saved -> %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
