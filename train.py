"""Training entry point for Phase 1 single-task models.

Usage:
    python train.py --config configs/phase1_object.yaml --seed 42

Phase 2 support is added in a later task; for now, only Phase 1
(``experiment.phase == 1``) is wired up.
"""

from __future__ import annotations

import argparse
import csv
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
from src.losses import AsymmetricLoss
from src.metrics import compute_metrics
from src.model import Phase1Model
from src.utils import build_optimizer, build_scheduler, save_checkpoint, seed_everything

logger = logging.getLogger(__name__)

FAMILY_Y_KEY = {"object": "y_obj", "event": "y_evt", "attribute": "y_attr"}


def load_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> torch.nn.Module:
    phase = cfg["experiment"]["phase"]
    if phase == 1:
        return Phase1Model(cfg)
    raise NotImplementedError(f"phase {phase} not yet implemented in train.py")


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: AsymmetricLoss,
    cfg: dict,
    ema: Optional[ModelEma],
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    global_step: int,
) -> tuple[float, int]:
    model.train()
    family = cfg["experiment"]["family"]
    fam_key = FAMILY_Y_KEY[family]
    nochg_weight = float(cfg["loss"].get("nochg_weight", 0.2))
    cutmix_p = float(cfg.get("augment", {}).get("cutmix", {}).get("p", 0.3))
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))

    total_loss = 0.0
    n_batches = 0
    t0 = time.time()

    for step, batch in enumerate(loader):
        batch = cutmix_pair(batch, p=cutmix_p)

        a = batch["A"].to(device, non_blocking=True)
        b = batch["B"].to(device, non_blocking=True)
        y_fam = batch[fam_key].to(device, non_blocking=True)
        y_nc = batch["is_change"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(a, b)
        logits_fam = out["logits_family"].float()
        logit_nc = out["logit_nochg"].float()
        loss = loss_fn(logits_fam, y_fam) + nochg_weight * F.binary_cross_entropy_with_logits(
            logit_nc, y_nc
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        n_batches += 1
        global_step += 1

        if step % 50 == 0:
            lrs = [g["lr"] for g in optimizer.param_groups]
            writer.add_scalar("train/loss_step", loss.item(), global_step)
            writer.add_scalar("train/lr_max", max(lrs), global_step)
            writer.add_scalar("train/lr_min", min(lrs), global_step)

    dt = time.time() - t0
    avg = total_loss / max(1, n_batches)
    logger.info("epoch %d train | loss=%.4f | %d batches in %.1fs", epoch, avg, n_batches, dt)
    return avg, global_step


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    loss_fn: AsymmetricLoss,
    cfg: dict,
    device: torch.device,
    n_classes: int,
) -> dict:
    model.eval()
    family = cfg["experiment"]["family"]
    fam_key = FAMILY_Y_KEY[family]
    nochg_weight = float(cfg["loss"].get("nochg_weight", 0.2))

    probs_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        a = batch["A"].to(device, non_blocking=True)
        b = batch["B"].to(device, non_blocking=True)
        y_fam = batch[fam_key].to(device, non_blocking=True)
        y_nc = batch["is_change"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(a, b)
        logits_fam = out["logits_family"].float()
        logit_nc = out["logit_nochg"].float()
        loss = loss_fn(logits_fam, y_fam) + nochg_weight * F.binary_cross_entropy_with_logits(
            logit_nc, y_nc
        )
        total_loss += loss.item()
        n_batches += 1

        probs_list.append(torch.sigmoid(logits_fam).cpu().numpy())
        targets_list.append(y_fam.cpu().numpy())

    probs = np.concatenate(probs_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    metrics = compute_metrics(probs, targets, thresholds=np.full(n_classes, 0.5))
    metrics["loss"] = total_loss / max(1, n_batches)
    return metrics


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(prog="train.py")
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

    model = build_model(cfg).to(device)
    mean, std = model.encoder.norm_stats()
    logger.info("model built: %s trainable params", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    img_size = int(cfg["data"].get("img_size", 224))
    train_transform = PairAug.from_cfg(cfg, mean, std)
    eval_transform = EvalTransform(img_size=img_size, mean=mean, std=std)
    train_loader, val_loader, test_loader = build_dataloaders(
        cfg, transform_train=train_transform, transform_eval=eval_transform
    )
    logger.info(
        "loaders: train=%d val=%d test=%d batches", len(train_loader), len(val_loader), len(test_loader)
    )

    loss_fn = AsymmetricLoss(**cfg["loss"].get("asl", {}))
    optimizer = build_optimizer(model, cfg)
    total_steps = len(train_loader) * int(cfg["train"]["epochs"])
    scheduler = build_scheduler(optimizer, cfg, total_steps=total_steps)

    ema: Optional[ModelEma] = None
    ema_decay = float(cfg["train"].get("ema_decay", 0.0))
    if ema_decay > 0:
        ema = ModelEma(
            model,
            decay=ema_decay,
            warmup_steps=int(cfg["train"].get("ema_warmup_steps", 1000)),
        )

    n_classes = int(cfg["experiment"]["n_classes"])
    patience = int(cfg["train"].get("early_stop_patience", 10))
    best_macro_f1 = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    global_step = 0

    log_path = output_dir / "train_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as lf:
        log_writer = csv.writer(lf)
        log_writer.writerow(
            ["epoch", "train_loss", "val_loss", "val_macro_f1", "val_micro_f1",
             "val_map", "lr_head", "lr_bb_max"]
        )

        for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
            train_loss, global_step = train_one_epoch(
                model, train_loader, optimizer, scheduler, loss_fn, cfg, ema, device,
                epoch=epoch, writer=writer, global_step=global_step,
            )

            val_model = ema.module if ema is not None else model
            val_metrics = evaluate(val_model, val_loader, loss_fn, cfg, device, n_classes)

            lrs = [g["lr"] for g in optimizer.param_groups]
            row = [
                epoch, train_loss, val_metrics["loss"],
                val_metrics["macro_f1"], val_metrics["micro_f1"], val_metrics["mAP"],
                max(lrs), min(lrs),
            ]
            log_writer.writerow(row)
            lf.flush()

            writer.add_scalar("val/loss", val_metrics["loss"], epoch)
            writer.add_scalar("val/macro_f1", val_metrics["macro_f1"], epoch)
            writer.add_scalar("val/micro_f1", val_metrics["micro_f1"], epoch)
            writer.add_scalar("val/mAP", val_metrics["mAP"], epoch)

            logger.info(
                "epoch %d val   | loss=%.4f | macro_f1=%.4f | micro_f1=%.4f | mAP=%.4f",
                epoch, val_metrics["loss"], val_metrics["macro_f1"],
                val_metrics["micro_f1"], val_metrics["mAP"],
            )

            if val_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = val_metrics["macro_f1"]
                best_epoch = epoch
                epochs_no_improve = 0
                save_checkpoint(
                    str(output_dir / "best.pth"), model, optimizer, scheduler,
                    ema, epoch, val_metrics,
                )
                if ema is not None:
                    # best_ema.pth stores just the EMA module's state_dict for easy eval load.
                    torch.save({"model": ema.state_dict(), "metrics": val_metrics},
                               str(output_dir / "best_ema.pth"))
                logger.info("  new best macro_f1=%.4f saved", best_macro_f1)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info("early stopping: no improvement for %d epochs", patience)
                    break

    # Final "last.pth" for resumability.
    save_checkpoint(str(output_dir / "last.pth"), model, optimizer, scheduler, ema,
                    epoch, val_metrics)
    writer.close()
    logger.info("done. best val macro_f1=%.4f at epoch %d", best_macro_f1, best_epoch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
