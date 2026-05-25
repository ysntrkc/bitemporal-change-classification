from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.augment import EvalTransform
from src.dataset import build_dataloaders
from src.metrics import compute_metrics
from src.model import Phase1Model, Phase2Model
from src.utils import load_checkpoint, seed_everything

FAMILIES = ("object", "event", "attribute")
SEEDS = (42, 1337, 2024)
TTA_OPS = ("orig", "hflip", "vflip", "rot180")

FAM_LOGITS = {"object": "logits_obj", "event": "logits_evt", "attribute": "logits_attr"}
FAM_Y = {"object": "y_obj", "event": "y_evt", "attribute": "y_attr"}


def _apply_tta(x: torch.Tensor, op: str) -> torch.Tensor:
    return {
        "orig": x,
        "hflip": torch.flip(x, dims=[-1]),
        "vflip": torch.flip(x, dims=[-2]),
        "rot180": torch.flip(x, dims=[-2, -1]),
    }[op]


@torch.no_grad()
def _collect_phase1(model, loader, family, device):
    fam_key = {"object": "y_obj", "event": "y_evt", "attribute": "y_attr"}[family]
    probs, tgts, is_change = [], [], []
    n = float(len(TTA_OPS))
    for batch in loader:
        a = batch["A"].to(device); b = batch["B"].to(device)
        psum = None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for op in TTA_OPS:
                out = model(_apply_tta(a, op), _apply_tta(b, op))
                p = torch.sigmoid(out["logits_family"].float())
                psum = p if psum is None else psum + p
        probs.append((psum / n).cpu().numpy())
        tgts.append(batch[fam_key].cpu().numpy())
        is_change.append(batch["is_change"].cpu().numpy())
    return (np.concatenate(probs), np.concatenate(tgts), np.concatenate(is_change))


@torch.no_grad()
def _collect_phase2(model, loader, device):
    fam_probs = {fam: [] for fam in FAMILIES}
    fam_tgts = {fam: [] for fam in FAMILIES}
    p_change_list, is_change_list = [], []
    n = float(len(TTA_OPS))
    for batch in loader:
        a = batch["A"].to(device); b = batch["B"].to(device)
        sums = {fam: None for fam in FAMILIES}
        change_sum = None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for op in TTA_OPS:
                out = model(_apply_tta(a, op), _apply_tta(b, op))
                change_sum = (torch.sigmoid(out["logit_nochg"].float()) if change_sum is None
                              else change_sum + torch.sigmoid(out["logit_nochg"].float()))
                for fam in FAMILIES:
                    p = torch.sigmoid(out[FAM_LOGITS[fam]].float())
                    sums[fam] = p if sums[fam] is None else sums[fam] + p
        p_change_list.append((change_sum / n).cpu().numpy())
        is_change_list.append(batch["is_change"].cpu().numpy())
        for fam in FAMILIES:
            fam_probs[fam].append((sums[fam] / n).cpu().numpy())
            fam_tgts[fam].append(batch[FAM_Y[fam]].cpu().numpy())
    return (
        {fam: np.concatenate(fam_probs[fam]) for fam in FAMILIES},
        np.concatenate(p_change_list),
        {fam: np.concatenate(fam_tgts[fam]) for fam in FAMILIES},
        np.concatenate(is_change_list),
    )


def _macro_f1_on_subset(probs: np.ndarray, targets: np.ndarray,
                       mask: np.ndarray, threshold: float = 0.5) -> float:
    p_sub = probs[mask]
    t_sub = targets[mask]
    m = compute_metrics(p_sub, t_sub, thresholds=np.full(probs.shape[1], threshold))
    return float(m["macro_f1"])


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Phase 1: one ckpt per (family, seed) ----
    p1_rows: dict[tuple[str, int], dict[str, float]] = {}
    for fam in FAMILIES:
        cfg = yaml.safe_load(Path(f"configs/phase1_main_{fam}.yaml").read_text())
        for seed in SEEDS:
            ckpt_path = Path(f"results/phase1_{fam}/seed{seed}/best_ema.pth")
            if not ckpt_path.exists():
                continue
            seed_everything(seed)
            model = Phase1Model(cfg).to(device)
            ckpt = load_checkpoint(str(ckpt_path))
            model.load_state_dict(ckpt["model"])
            model.eval()
            mean, std = model.encoder.norm_stats()
            transform = EvalTransform(img_size=224, mean=mean, std=std)
            _, _, test_loader = build_dataloaders(
                cfg, transform_train=transform, transform_eval=transform
            )
            probs, tgts, is_change = _collect_phase1(model, test_loader, fam, device)
            full = _macro_f1_on_subset(probs, tgts, np.ones_like(is_change, dtype=bool))
            sub = _macro_f1_on_subset(probs, tgts, is_change.astype(bool))
            p1_rows[(fam, seed)] = {"full": full, "changed": sub}
            print(f"P1 {fam} seed{seed}: full={full:.4f}  changed-only={sub:.4f}")
            del model
            torch.cuda.empty_cache()

    # ---- Phase 2 BIT-only canonical ----
    p2_rows: dict[int, dict[str, dict[str, float]]] = {}
    cfg2 = yaml.safe_load(Path("configs/phase2_main.yaml").read_text())
    for seed in SEEDS:
        ckpt_path = Path(f"results/phase2_bit_only/seed{seed}/best_ema.pth")
        if not ckpt_path.exists():
            continue
        seed_everything(seed)
        model = Phase2Model(cfg2).to(device)
        ckpt = load_checkpoint(str(ckpt_path))
        model.load_state_dict(ckpt["model"])
        model.eval()
        mean, std = model.encoder.norm_stats()
        transform = EvalTransform(img_size=224, mean=mean, std=std)
        _, _, test_loader = build_dataloaders(
            cfg2, transform_train=transform, transform_eval=transform
        )
        fp, pc, ft, ic = _collect_phase2(model, test_loader, device)
        p2_rows[seed] = {}
        for fam in FAMILIES:
            gated = fp[fam] * pc[:, None]
            full = _macro_f1_on_subset(gated, ft[fam], np.ones_like(ic, dtype=bool))
            sub = _macro_f1_on_subset(gated, ft[fam], ic.astype(bool))
            p2_rows[seed][fam] = {"full": full, "changed": sub}
            print(f"P2 {fam} seed{seed}: full={full:.4f}  changed-only={sub:.4f}")
        del model
        torch.cuda.empty_cache()

    # ---- Aggregate ----
    lines = []
    lines.append("# Changed-subset macro-F1 (test split, EMA, TTA [+ gate for P2])")
    lines.append("")
    lines.append("Two columns per family: full = all test pairs (1190 of which "
                 "~71% have a change), changed-only = restricted to is_change=1.")
    lines.append("Mean ± std over 3 seeds.")
    lines.append("")
    lines.append("| model | object full | object changed | event full | event changed | attr full | attr changed |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    # Phase 1
    cells_p1 = ["P1 (default 0.5 + TTA)"]
    for fam in FAMILIES:
        full_vals = np.array([p1_rows[(fam, s)]["full"]
                              for s in SEEDS if (fam, s) in p1_rows])
        sub_vals = np.array([p1_rows[(fam, s)]["changed"]
                             for s in SEEDS if (fam, s) in p1_rows])
        cells_p1.append(f"{full_vals.mean():.4f} ± {full_vals.std(ddof=1):.4f}")
        cells_p1.append(f"{sub_vals.mean():.4f} ± {sub_vals.std(ddof=1):.4f}")
    lines.append("| " + " | ".join(cells_p1) + " |")

    # Phase 2
    cells_p2 = ["P2 BIT-only (+TTA + gate)"]
    for fam in FAMILIES:
        full_vals = np.array([p2_rows[s][fam]["full"] for s in SEEDS if s in p2_rows])
        sub_vals = np.array([p2_rows[s][fam]["changed"] for s in SEEDS if s in p2_rows])
        cells_p2.append(f"{full_vals.mean():.4f} ± {full_vals.std(ddof=1):.4f}")
        cells_p2.append(f"{sub_vals.mean():.4f} ± {sub_vals.std(ddof=1):.4f}")
    lines.append("| " + " | ".join(cells_p2) + " |")

    out = Path("results/changed_subset_table.md")
    out.write_text("\n".join(lines) + "\n")
    print()
    print(f"saved -> {out}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
