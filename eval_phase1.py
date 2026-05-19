"""Evaluation entry point for Phase 1 (Phase 2 reuses unchanged).

Two modes:

* ``--mode tune-thresholds``: forward the chosen split (default ``val``)
  through the loaded checkpoint, sweep per-class F1 thresholds, save
  ``thresholds.json`` next to the checkpoint.
* ``--mode metrics`` (default): forward the chosen split (default
  ``test``), optionally with TTA and pre-tuned thresholds, save
  ``metrics_<split>[_tta][_thr].json``.

Examples::

    # Tune per-class thresholds on val
    python eval_phase1.py --ckpt results/phase1_object/seed42/best_ema.pth \\
                   --config configs/phase1_object.yaml \\
                   --mode tune-thresholds --split val

    # Final test metrics with TTA + tuned thresholds
    python eval_phase1.py --ckpt results/phase1_object/seed42/best_ema.pth \\
                   --config configs/phase1_object.yaml \\
                   --tta \\
                   --apply-thresholds results/phase1_object/seed42/thresholds.json \\
                   --split test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

from src.augment import EvalTransform
from src.dataset import build_dataloaders
from src.metrics import compute_metrics, tta_forward, tune_thresholds_per_class
from src.model import Phase1Model
from src.utils import load_checkpoint, seed_everything

logger = logging.getLogger(__name__)

FAMILY_Y_KEY = {"object": "y_obj", "event": "y_evt", "attribute": "y_attr"}


def load_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> torch.nn.Module:
    phase = cfg["experiment"]["phase"]
    if phase == 1:
        return Phase1Model(cfg)
    raise NotImplementedError(
        f"phase {phase} not handled by eval_phase1.py; use eval_phase2.py"
    )


def collect_probs(
    model: torch.nn.Module,
    loader,
    fam_key: str,
    tta_ops: list[str],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward the loader once; return ``(probs_family, prob_nochg, targets_family)``.

    Uses ``tta_forward`` so a single ``orig`` op gives the plain forward
    pass. Autocasts to bf16 for parity with training.
    """
    model.eval()
    probs_fam: list[np.ndarray] = []
    probs_nc: list[np.ndarray] = []
    tgts: list[np.ndarray] = []

    for batch in loader:
        a = batch["A"].to(device, non_blocking=True)
        b = batch["B"].to(device, non_blocking=True)
        y_fam = batch[fam_key]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = tta_forward(model, {"A": a, "B": b}, tta_ops)

        probs_fam.append(out["probs_family"].cpu().numpy())
        probs_nc.append(out["prob_nochg"].cpu().numpy())
        tgts.append(y_fam.cpu().numpy())

    return (
        np.concatenate(probs_fam, axis=0),
        np.concatenate(probs_nc, axis=0),
        np.concatenate(tgts, axis=0),
    )


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(prog="eval_phase1.py")
    parser.add_argument("--ckpt", required=True, help="checkpoint path (.pth)")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument(
        "--mode",
        choices=["metrics", "tune-thresholds"],
        default="metrics",
        help="metrics: compute metrics on --split; tune-thresholds: sweep on --split",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default=None,
        help="default: val for tune-thresholds, test for metrics",
    )
    parser.add_argument(
        "--tta", action="store_true", help="enable TTA using cfg.eval.tta ops"
    )
    parser.add_argument(
        "--apply-thresholds",
        type=str,
        default=None,
        help="path to thresholds.json (metrics mode only)",
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="multiplicative no-change gate: probs_family *= prob_nochg "
             "(metrics mode only; mirrors Phase 2 gating)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="output JSON path (default: alongside checkpoint)",
    )
    args = parser.parse_args(argv)

    if args.split is None:
        args.split = "val" if args.mode == "tune-thresholds" else "test"
    if args.apply_thresholds and args.mode != "metrics":
        parser.error("--apply-thresholds is only valid in metrics mode")
    if args.gate and args.mode != "metrics":
        parser.error("--gate is only valid in metrics mode")

    cfg = load_config(args.config)
    seed_everything(int(cfg["experiment"]["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    family = cfg["experiment"]["family"]
    fam_key = FAMILY_Y_KEY[family]
    n_classes = int(cfg["experiment"]["n_classes"])

    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(args.ckpt)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        logger.warning(
            "load_state_dict: %d missing, %d unexpected", len(missing), len(unexpected)
        )
    logger.info("loaded ckpt %s (epoch=%s)", args.ckpt, ckpt.get("epoch", "?")
                if isinstance(ckpt, dict) else "?")

    mean, std = model.encoder.norm_stats()
    eval_transform = EvalTransform(
        img_size=int(cfg["data"].get("img_size", 224)), mean=mean, std=std
    )
    train_loader, val_loader, test_loader = build_dataloaders(
        cfg, transform_train=eval_transform, transform_eval=eval_transform
    )
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]

    cfg_tta = cfg.get("eval", {}).get("tta", ["orig", "hflip", "vflip", "rot180"])
    tta_ops = list(cfg_tta) if args.tta else ["orig"]
    logger.info(
        "eval | family=%s split=%s n_classes=%d tta=%s",
        family, args.split, n_classes, tta_ops,
    )

    probs_fam, probs_nc, targets = collect_probs(model, loader, fam_key, tta_ops, device)
    logger.info("collected probs: shape=%s", probs_fam.shape)

    if args.gate:
        # Multiplicative no-change gate, mirroring Phase 2: each family
        # probability is downweighted by the auxiliary head's P(changed).
        # Soft (no hard threshold) so calibration of ``head_nochg`` carries
        # through directly to the downstream decision.
        probs_fam = probs_fam * probs_nc[:, None]
        logger.info("gate applied: mean P(chg)=%.3f", float(probs_nc.mean()))

    ckpt_dir = Path(args.ckpt).parent

    if args.mode == "tune-thresholds":
        thr = tune_thresholds_per_class(probs_fam, targets)
        out_path = Path(args.output) if args.output else ckpt_dir / "thresholds.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "family": family,
                    "n_classes": n_classes,
                    "split": args.split,
                    "tta": args.tta,
                    "thresholds": thr.tolist(),
                },
                f,
                indent=2,
            )
        m = compute_metrics(probs_fam, targets, thr)
        m_default = compute_metrics(probs_fam, targets, np.full(n_classes, 0.5))
        logger.info(
            "thresholds saved -> %s | tuned macro_f1=%.4f (vs default 0.5: %.4f)",
            out_path, m["macro_f1"], m_default["macro_f1"],
        )
    else:
        if args.apply_thresholds:
            with Path(args.apply_thresholds).open("r", encoding="utf-8") as f:
                tdat = json.load(f)
            if tdat.get("family") != family:
                raise ValueError(
                    f"threshold family {tdat.get('family')!r} does not match "
                    f"model family {family!r}"
                )
            thr = np.array(tdat["thresholds"], dtype=np.float64)
            if thr.shape != (n_classes,):
                raise ValueError(
                    f"thresholds shape {thr.shape} != ({n_classes},)"
                )
        else:
            thr = np.full(n_classes, 0.5, dtype=np.float64)

        m = compute_metrics(probs_fam, targets, thr)
        suffix = ""
        if args.tta:
            suffix += "_tta"
        if args.apply_thresholds:
            suffix += "_thr"
        if args.gate:
            suffix += "_gate"
        out_path = (
            Path(args.output)
            if args.output
            else ckpt_dir / f"metrics_{args.split}{suffix}.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "ckpt": str(args.ckpt),
                    "config": str(args.config),
                    "family": family,
                    "n_classes": n_classes,
                    "split": args.split,
                    "tta": args.tta,
                    "tta_ops": tta_ops,
                    "thresholds_path": args.apply_thresholds,
                    "thresholds": thr.tolist(),
                    "nochg_gate": bool(args.gate),
                    **m,
                },
                f,
                indent=2,
            )
        logger.info(
            "metrics saved -> %s | macro_f1=%.4f micro_f1=%.4f mAP=%.4f",
            out_path, m["macro_f1"], m["micro_f1"], m["mAP"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
