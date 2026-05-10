"""Training entry point for the Phase-2 unified multi-task model.

Usage:
    python train_phase2.py --config configs/phase2_unified.yaml --seed 42

Differs from ``train.py`` (Phase 1) in four ways:
  1. Builds ``Phase2Model`` (BIT fusion + 3x Q2L heads + no-change head).
  2. Combines four task losses (3x ASL + 1x BCE) via
     ``UncertaintyWeightedLoss`` with 4 learnable log-sigma parameters.
  3. Implements real gradient accumulation (Phase-1 used ``grad_accum=1``).
  4. Tracks per-family val metrics and early-stops on their mean
     macro-F1; optionally applies the no-change gate during val.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

from src.augment import EvalTransform, PairAug, cutmix_pair
from src.dataset import build_dataloaders
from src.ema import ModelEma
from src.losses import AsymmetricLoss, UncertaintyWeightedLoss
from src.metrics import compute_metrics
from src.model import Phase2Model
from src.utils import build_optimizer, build_scheduler, save_checkpoint, seed_everything

logger = logging.getLogger(__name__)

FAMILY_Y_KEY = {"object": "y_obj", "event": "y_evt", "attribute": "y_attr"}
FAMILY_LOGITS_KEY = {"object": "logits_obj", "event": "logits_evt", "attribute": "logits_attr"}
FAMILY_LOSS_KEY = {"object": "obj", "event": "evt", "attribute": "attr"}


def load_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_n_classes(cfg: dict, families: list[str]) -> dict[str, int]:
    vocab_path = cfg["experiment"].get("label_vocab", "configs/label_vocab.json")
    vocab = json.loads(Path(vocab_path).read_text())
    return {fam: len(vocab[fam]) for fam in families}


def _compute_task_losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    asl: AsymmetricLoss,
    families: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Forward outputs + batch -> 4 scalar task losses keyed obj/evt/attr/nochg."""
    losses: dict[str, torch.Tensor] = {}
    for fam in families:
        logits = out[FAMILY_LOGITS_KEY[fam]].float()
        targets = batch[FAMILY_Y_KEY[fam]].to(device, non_blocking=True)
        losses[FAMILY_LOSS_KEY[fam]] = asl(logits, targets)
    logit_nc = out["logit_nochg"].float()
    y_nc = batch["is_change"].to(device, non_blocking=True)
    losses["nochg"] = F.binary_cross_entropy_with_logits(logit_nc, y_nc)
    return losses


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    asl: AsymmetricLoss,
    uwl: UncertaintyWeightedLoss,
    cfg: dict,
    ema: Optional[ModelEma],
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    global_step: int,
) -> tuple[float, int]:
    model.train()
    families = list(cfg["experiment"]["families"])
    cutmix_p = float(cfg.get("augment", {}).get("cutmix", {}).get("p", 0.3))
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))
    grad_accum = max(1, int(cfg["train"].get("grad_accum", 1)))

    total_loss = 0.0
    n_batches = 0
    t0 = time.time()

    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        batch = cutmix_pair(batch, p=cutmix_p)
        a = batch["A"].to(device, non_blocking=True)
        b = batch["B"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(a, b)
            task_losses = _compute_task_losses(out, batch, asl, families, device)
            loss = uwl(task_losses) / grad_accum

        loss.backward()
        # Track the un-scaled loss for logging.
        total_loss += loss.item() * grad_accum
        n_batches += 1

        # Optimizer step every grad_accum micro-batches.
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % 50 == 0:
                lrs = [g["lr"] for g in optimizer.param_groups]
                writer.add_scalar("train/loss_step", loss.item() * grad_accum, global_step)
                writer.add_scalar("train/lr_max", max(lrs), global_step)
                writer.add_scalar("train/lr_min", min(lrs), global_step)
                for fam_key in ("obj", "evt", "attr", "nochg"):
                    writer.add_scalar(f"train/loss_{fam_key}", task_losses[fam_key].item(),
                                      global_step)

    # Flush any remaining gradients from a partial final accumulation window.
    if (n_batches % grad_accum) != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

    dt = time.time() - t0
    avg = total_loss / max(1, n_batches)
    logger.info("epoch %d train | loss=%.4f | %d batches in %.1fs",
                epoch, avg, n_batches, dt)
    return avg, global_step


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    asl: AsymmetricLoss,
    uwl: UncertaintyWeightedLoss,
    cfg: dict,
    device: torch.device,
    n_classes: dict[str, int],
) -> dict:
    """Returns nested dict: ``{family: {macro_f1, micro_f1, ...}, "loss": ..., "macro_f1_mean": ...}``."""
    model.eval()
    families = list(cfg["experiment"]["families"])
    nochg_gate = bool(cfg.get("inference", {}).get("nochg_gate", False))

    fam_probs: dict[str, list[np.ndarray]] = {fam: [] for fam in families}
    fam_targets: dict[str, list[np.ndarray]] = {fam: [] for fam in families}
    nc_probs_list: list[np.ndarray] = []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        a = batch["A"].to(device, non_blocking=True)
        b = batch["B"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(a, b)
            task_losses = _compute_task_losses(out, batch, asl, families, device)
            loss = uwl(task_losses)

        total_loss += loss.item()
        n_batches += 1

        # Collect raw per-family probs and the no-change prob (for optional gating).
        p_nc = torch.sigmoid(out["logit_nochg"].float()).cpu().numpy()
        nc_probs_list.append(p_nc)
        for fam in families:
            p = torch.sigmoid(out[FAMILY_LOGITS_KEY[fam]].float()).cpu().numpy()
            fam_probs[fam].append(p)
            fam_targets[fam].append(batch[FAMILY_Y_KEY[fam]].cpu().numpy())

    nc_probs = np.concatenate(nc_probs_list, axis=0)             # [N]
    metrics: dict = {"loss": total_loss / max(1, n_batches)}
    macro_f1s: list[float] = []
    for fam in families:
        probs = np.concatenate(fam_probs[fam], axis=0)            # [N, C]
        targets = np.concatenate(fam_targets[fam], axis=0)
        if nochg_gate:
            probs = probs * (1.0 - nc_probs[:, None])
        m = compute_metrics(probs, targets, thresholds=np.full(n_classes[fam], 0.5))
        metrics[fam] = m
        macro_f1s.append(m["macro_f1"])
    metrics["macro_f1_mean"] = float(np.mean(macro_f1s))
    return metrics


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(prog="train_phase2.py")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--seed", type=int, default=None, help="override experiment.seed")
    parser.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    parser.add_argument("--output", type=str, default=None, help="override experiment.output_dir")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.output is not None:
        cfg["experiment"]["output_dir"] = args.output

    seed = int(cfg["experiment"]["seed"])
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)

    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    shutil.copy(args.config, output_dir / "config_source.yaml")
    writer = SummaryWriter(log_dir=str(output_dir / "tb"))

    model = Phase2Model(cfg).to(device)
    mean, std = model.encoder.norm_stats()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("model built: %s trainable params", f"{n_trainable:,}")

    img_size = int(cfg["data"].get("img_size", 224))
    train_transform = PairAug.from_cfg(cfg, mean, std)
    eval_transform = EvalTransform(img_size=img_size, mean=mean, std=std)
    train_loader, val_loader, test_loader = build_dataloaders(
        cfg, transform_train=train_transform, transform_eval=eval_transform
    )
    logger.info("loaders: train=%d val=%d test=%d batches",
                len(train_loader), len(val_loader), len(test_loader))

    asl = AsymmetricLoss(**cfg["loss"].get("asl", {})).to(device)
    init_log_sigma = float(cfg["loss"].get("init_log_sigma", 0.0))
    uwl = UncertaintyWeightedLoss(init_log_sigma=init_log_sigma).to(device)

    # Optimizer must include the uncertainty-weighting log-sigma parameters.
    optimizer = build_optimizer(model, cfg)
    head_lr = float(cfg["train"]["head_lr"])
    optimizer.add_param_group({"params": list(uwl.parameters()), "lr": head_lr,
                               "weight_decay": 0.0})

    grad_accum = max(1, int(cfg["train"].get("grad_accum", 1)))
    steps_per_epoch = max(1, len(train_loader) // grad_accum)
    total_steps = steps_per_epoch * int(cfg["train"]["epochs"])
    scheduler = build_scheduler(optimizer, cfg, total_steps=total_steps)

    ema: Optional[ModelEma] = None
    ema_decay = float(cfg["train"].get("ema_decay", 0.0))
    if ema_decay > 0:
        ema = ModelEma(
            model,
            decay=ema_decay,
            warmup_steps=int(cfg["train"].get("ema_warmup_steps", 1000)),
        )

    families = list(cfg["experiment"]["families"])
    n_classes = _resolve_n_classes(cfg, families)
    patience = int(cfg["train"].get("early_stop_patience", 10))

    best_macro_f1_mean = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    global_step = 0
    epoch = 0
    val_metrics: dict | None = None

    log_path = output_dir / "train_log.csv"
    csv_cols = ["epoch", "train_loss", "val_loss", "val_macro_f1_mean"]
    for fam in families:
        csv_cols += [f"val_macro_f1_{fam}", f"val_micro_f1_{fam}", f"val_map_{fam}"]
    csv_cols += [f"log_sigma_{k}" for k in ("obj", "evt", "attr", "nochg")]
    csv_cols += ["lr_head", "lr_bb_max"]

    with log_path.open("w", newline="", encoding="utf-8") as lf:
        log_writer = csv.writer(lf)
        log_writer.writerow(csv_cols)

        for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
            train_loss, global_step = train_one_epoch(
                model, train_loader, optimizer, scheduler, asl, uwl, cfg, ema, device,
                epoch=epoch, writer=writer, global_step=global_step,
            )

            val_model = ema.module if ema is not None else model
            val_metrics = evaluate(val_model, val_loader, asl, uwl, cfg, device, n_classes)

            lrs = [g["lr"] for g in optimizer.param_groups]
            row: list = [epoch, train_loss, val_metrics["loss"], val_metrics["macro_f1_mean"]]
            for fam in families:
                m = val_metrics[fam]
                row += [m["macro_f1"], m["micro_f1"], m["mAP"]]
            for v in uwl.log_sigma.detach().cpu().tolist():
                row.append(v)
            row += [max(lrs), min(lrs)]
            log_writer.writerow(row)
            lf.flush()

            writer.add_scalar("val/loss", val_metrics["loss"], epoch)
            writer.add_scalar("val/macro_f1_mean", val_metrics["macro_f1_mean"], epoch)
            for fam in families:
                m = val_metrics[fam]
                writer.add_scalar(f"val/macro_f1_{fam}", m["macro_f1"], epoch)
                writer.add_scalar(f"val/micro_f1_{fam}", m["micro_f1"], epoch)
                writer.add_scalar(f"val/mAP_{fam}", m["mAP"], epoch)
            for i, fam_key in enumerate(("obj", "evt", "attr", "nochg")):
                writer.add_scalar(f"train/log_sigma_{fam_key}",
                                  uwl.log_sigma[i].item(), epoch)

            per_fam_f1 = " ".join(
                f"{fam[:3]}={val_metrics[fam]['macro_f1']:.4f}" for fam in families
            )
            logger.info(
                "epoch %d val   | loss=%.4f | mean_f1=%.4f | %s",
                epoch, val_metrics["loss"], val_metrics["macro_f1_mean"], per_fam_f1,
            )

            if val_metrics["macro_f1_mean"] > best_macro_f1_mean:
                best_macro_f1_mean = val_metrics["macro_f1_mean"]
                best_epoch = epoch
                epochs_no_improve = 0
                save_checkpoint(
                    str(output_dir / "best.pth"), model, optimizer, scheduler,
                    ema, epoch, val_metrics,
                )
                if ema is not None:
                    torch.save({"model": ema.state_dict(), "metrics": val_metrics},
                               str(output_dir / "best_ema.pth"))
                logger.info("  new best macro_f1_mean=%.4f saved", best_macro_f1_mean)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info("early stopping: no improvement for %d epochs", patience)
                    break

    if val_metrics is not None:
        save_checkpoint(str(output_dir / "last.pth"), model, optimizer, scheduler, ema,
                        epoch, val_metrics)
    writer.close()
    logger.info("done. best val macro_f1_mean=%.4f at epoch %d",
                best_macro_f1_mean, best_epoch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
