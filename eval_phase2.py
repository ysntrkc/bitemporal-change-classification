from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from src.augment import EvalTransform
from src.config import load_config
from src.dataset import build_dataloaders
from src.metrics import compute_metrics, tta_forward
from src.model import build_model
from src.utils import load_checkpoint, seed_everything

logger = logging.getLogger(__name__)

FAMILY_Y_KEY = {"object": "y_obj", "event": "y_evt", "attribute": "y_attr"}
FAMILY_PROB_KEY = {"object": "probs_obj", "event": "probs_evt", "attribute": "probs_attr"}
_TTA_OPS = ("orig", "hflip", "vflip", "rot180")


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
    model.eval()
    fam_probs: dict[str, list[np.ndarray]] = {fam: [] for fam in families}
    fam_targets: dict[str, list[np.ndarray]] = {fam: [] for fam in families}
    change_probs_list: list[np.ndarray] = []

    for batch in loader:
        a = batch["A"].to(device, non_blocking=True)
        b = batch["B"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = tta_forward(model, {"A": a, "B": b}, tta_ops)
        change_probs_list.append(out["prob_nochg"].cpu().numpy())
        for fam in families:
            fam_probs[fam].append(out[FAMILY_PROB_KEY[fam]].cpu().numpy())
            fam_targets[fam].append(batch[FAMILY_Y_KEY[fam]].cpu().numpy())

    return (
        {fam: np.concatenate(fam_probs[fam], axis=0) for fam in families},
        np.concatenate(change_probs_list, axis=0),
        {fam: np.concatenate(fam_targets[fam], axis=0) for fam in families},
    )


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

    cfg = load_config(args.config)
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

    model = build_model(cfg).to(device)
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
