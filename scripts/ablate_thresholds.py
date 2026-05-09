"""One-off ablation: compare four threshold strategies on Phase-1 ckpts.

Strategies (all evaluated against the same TTA'd test probs):
    1. default 0.5            — flat 0.5 threshold for every class
    2. tuned per-class        — argmax F1 per class on val (current default)
    3. tuned per-class, clipped to [0.3, 0.7]   — smooth extreme thresholds
    4. tuned global           — single threshold for the family, argmax macro-F1 on val

Each (family, seed) ckpt is forwarded once on val and once on test with
TTA (orig / hflip / vflip / rot180); the four strategies share those
probabilities. Saves per-ckpt rows + an aggregated table to JSON.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score

from eval import FAMILY_Y_KEY, collect_probs
from src.augment import EvalTransform
from src.dataset import build_dataloaders
from src.metrics import compute_metrics, tune_thresholds_per_class
from src.model import Phase1Model
from src.utils import load_checkpoint, seed_everything


_STEPS = np.arange(0.05, 0.96, 0.02)


def tune_threshold_global(probs: np.ndarray, targets: np.ndarray) -> float:
    """Pick the single threshold maximising macro-F1 across all classes."""
    y = targets.astype(np.int64)
    best_t, best_f1 = 0.5, -1.0
    for t in _STEPS:
        preds = (probs >= t).astype(np.int64)
        f1 = f1_score(y, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t


def main() -> None:
    device = torch.device("cuda")
    tta_ops = ["orig", "hflip", "vflip", "rot180"]
    results = []

    for fam in ["object", "event", "attribute"]:
        cfg = yaml.safe_load(open(f"configs/phase1_{fam}.yaml"))
        n_classes = int(cfg["experiment"]["n_classes"])
        fam_key = FAMILY_Y_KEY[fam]

        for seed in [42, 1337, 2024]:
            ckpt_path = f"results/phase1_{fam}/seed{seed}/best_ema.pth"
            print(f">>> {fam}/seed{seed}")
            seed_everything(seed)

            model = Phase1Model(cfg).to(device)
            ckpt = load_checkpoint(ckpt_path)
            model.load_state_dict(ckpt["model"])
            mean, std = model.encoder.norm_stats()
            transform = EvalTransform(img_size=224, mean=mean, std=std)
            _, val_loader, test_loader = build_dataloaders(
                cfg, transform_train=transform, transform_eval=transform
            )

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                val_probs, _, val_tgts = collect_probs(
                    model, val_loader, fam_key, tta_ops, device
                )
                test_probs, _, test_tgts = collect_probs(
                    model, test_loader, fam_key, tta_ops, device
                )

            strategies: dict[str, np.ndarray] = {
                "default 0.5":          np.full(n_classes, 0.5),
                "tuned per-class":      tune_thresholds_per_class(val_probs, val_tgts),
            }
            strategies["tuned clip [0.3,0.7]"] = np.clip(
                strategies["tuned per-class"], 0.3, 0.7
            )
            t_global = tune_threshold_global(val_probs, val_tgts)
            strategies["tuned global"] = np.full(n_classes, t_global)
            print(f"    global t* = {t_global:.2f}")

            for name, thr in strategies.items():
                m = compute_metrics(test_probs, test_tgts, thr)
                results.append({
                    "family": fam, "seed": seed, "strategy": name,
                    "macro_f1": m["macro_f1"],
                    "micro_f1": m["micro_f1"],
                    "mAP": m["mAP"],
                })

            del model
            torch.cuda.empty_cache()

    # ---- aggregate ----
    agg: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in results:
        agg[r["family"]][r["strategy"]].append(r["macro_f1"])

    order = ["default 0.5", "tuned per-class", "tuned clip [0.3,0.7]", "tuned global"]
    print()
    print(f'{"family":10s} | ' + " | ".join(f"{s:>22s}" for s in order))
    print("-" * (12 + 26 * len(order)))
    for fam in ["object", "event", "attribute"]:
        cells = []
        for s in order:
            vals = agg[fam][s]
            cells.append(f"{np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}")
        print(f'{fam:10s} | ' + " | ".join(f"{c:>22s}" for c in cells))

    Path("results/phase1_threshold_ablation.json").write_text(
        json.dumps(results, indent=2)
    )
    print("\nsaved -> results/phase1_threshold_ablation.json")


if __name__ == "__main__":
    main()
