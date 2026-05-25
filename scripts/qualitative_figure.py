from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

from src.augment import EvalTransform
from src.dataset import build_dataloaders
from src.model import Phase2Model
from src.utils import load_checkpoint, seed_everything

CKPT = "results/phase2_bit_only/seed42/best_ema.pth"
CONFIG = "configs/phase2_main.yaml"
DATASET_JSON = "dataset/dataset.json"
DATASET_ROOT = Path("dataset")
OUT_PATH = Path("results/phase2_qualitative.png")
TTA_OPS = ("orig", "hflip", "vflip", "rot180")

FAMILIES = ("object", "event", "attribute")
FAMILY_LOGITS_KEY = {"object": "logits_obj", "event": "logits_evt",
                     "attribute": "logits_attr"}
FAMILY_Y_KEY = {"object": "y_obj", "event": "y_evt", "attribute": "y_attr"}


def _per_sample_macro_f1(probs: np.ndarray, targets: np.ndarray,
                         threshold: float = 0.5) -> np.ndarray:
    preds = (probs >= threshold).astype(np.int64)
    y = targets.astype(np.int64)
    tp = ((preds == 1) & (y == 1)).sum(axis=1)
    fp = ((preds == 1) & (y == 0)).sum(axis=1)
    fn = ((preds == 0) & (y == 1)).sum(axis=1)
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return f1.astype(np.float32)


def _apply_tta(x: torch.Tensor, op: str) -> torch.Tensor:
    return {
        "orig": x,
        "hflip": torch.flip(x, dims=[-1]),
        "vflip": torch.flip(x, dims=[-2]),
        "rot180": torch.flip(x, dims=[-2, -1]),
    }[op]


def _label_text(labels: list[str], probs: np.ndarray, vocab: list[str],
                threshold: float = 0.5) -> str:
    above = [(i, probs[i]) for i in range(len(vocab)) if probs[i] >= threshold]
    above.sort(key=lambda t: -t[1])
    if not above:
        return "(no predictions)"
    return "\n".join(f"{vocab[i]} (p={p:.2f})" for i, p in above)


def main() -> None:
    cfg = yaml.safe_load(Path(CONFIG).read_text())
    seed_everything(int(cfg["experiment"].get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = json.loads(Path(cfg["data"]["vocab_path"]).read_text())

    model = Phase2Model(cfg).to(device)
    ckpt = load_checkpoint(CKPT)
    model.load_state_dict(ckpt["model"])
    model.eval()
    mean, std = model.encoder.norm_stats()
    transform = EvalTransform(img_size=int(cfg["data"].get("img_size", 224)),
                              mean=mean, std=std)
    _, _, test_loader = build_dataloaders(
        cfg, transform_train=transform, transform_eval=transform
    )

    # 1) Forward the test split (TTA-averaged probs, like canonical eval).
    all_probs = {fam: [] for fam in FAMILIES}
    all_targets = {fam: [] for fam in FAMILIES}
    all_change = []
    all_sample_ids: list[str] = []
    n_tta = float(len(TTA_OPS))
    with torch.no_grad():
        for batch in test_loader:
            a = batch["A"].to(device, non_blocking=True)
            b = batch["B"].to(device, non_blocking=True)
            fam_sum = {fam: None for fam in FAMILIES}
            change_sum = None
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for op in TTA_OPS:
                    out = model(_apply_tta(a, op), _apply_tta(b, op))
                    p_change = torch.sigmoid(out["logit_nochg"].float())
                    change_sum = p_change if change_sum is None else change_sum + p_change
                    for fam in FAMILIES:
                        p = torch.sigmoid(out[FAMILY_LOGITS_KEY[fam]].float())
                        fam_sum[fam] = p if fam_sum[fam] is None else fam_sum[fam] + p
            ch = (change_sum / n_tta).cpu().numpy()
            all_change.append(ch)
            for fam in FAMILIES:
                all_probs[fam].append((fam_sum[fam] / n_tta).cpu().numpy())
                all_targets[fam].append(batch[FAMILY_Y_KEY[fam]].cpu().numpy())
            all_sample_ids.extend(batch["sample_id"])

    change_probs = np.concatenate(all_change, axis=0)
    fam_probs = {fam: np.concatenate(all_probs[fam], axis=0) for fam in FAMILIES}
    fam_targets = {fam: np.concatenate(all_targets[fam], axis=0) for fam in FAMILIES}
    # Apply the no-change gate
    for fam in FAMILIES:
        fam_probs[fam] = fam_probs[fam] * change_probs[:, None]

    # 2) Per-sample multi-family F1 (mean over 3 families) for ranking.
    per_fam_f1 = {fam: _per_sample_macro_f1(fam_probs[fam], fam_targets[fam])
                  for fam in FAMILIES}
    has_pos = np.zeros(len(all_sample_ids), dtype=bool)
    for fam in FAMILIES:
        has_pos |= fam_targets[fam].sum(axis=1) > 0
    sample_f1 = np.mean([per_fam_f1[fam] for fam in FAMILIES], axis=0)

    # Successes: highest F1 among samples with positives.
    candidate_succ = np.where(has_pos)[0]
    top_succ = candidate_succ[np.argsort(-sample_f1[candidate_succ])[:2]]
    # Failures: lowest F1 among samples with positives (to avoid trivial cases).
    bot_fail = candidate_succ[np.argsort(sample_f1[candidate_succ])[:2]]
    picks = list(top_succ) + list(bot_fail)
    labels = ["success", "success", "failure", "failure"]
    print("picks:")
    for idx, lab in zip(picks, labels):
        print(f"  [{lab}] {all_sample_ids[idx]}  per-sample F1={sample_f1[idx]:.3f}  "
              f"p_change={change_probs[idx]:.2f}")

    # 3) Lookup raw image paths from dataset.json by sample_id.
    blob = json.loads(Path(DATASET_JSON).read_text())
    by_id = {r["sample_id"]: r for r in blob["images"]}

    # 4) Render
    n_rows = len(picks)
    fig, axes = plt.subplots(n_rows, 4, figsize=(14, 3.2 * n_rows),
                             gridspec_kw={"width_ratios": [1, 1, 1.1, 1.6]})
    for row, (idx, lab) in enumerate(zip(picks, labels)):
        rec = by_id[all_sample_ids[idx]]
        a_img = Image.open(DATASET_ROOT / rec["rgb_A"]).convert("RGB")
        b_img = Image.open(DATASET_ROOT / rec["rgb_B"]).convert("RGB")

        gt_text_lines = []
        pred_text_lines = []
        for fam in FAMILIES:
            gt_labels = rec.get(f"{fam}_labels", []) or []
            gt_text_lines.append(f"[{fam}] {', '.join(gt_labels) if gt_labels else '—'}")
            pred_text_lines.append(
                f"[{fam}]\n" + _label_text(gt_labels, fam_probs[fam][idx], vocab[fam])
            )
        gt_text = "\n".join(gt_text_lines)
        pred_text = "\n\n".join(pred_text_lines)

        ax_a, ax_b, ax_gt, ax_pred = axes[row]
        ax_a.imshow(a_img); ax_a.set_xticks([]); ax_a.set_yticks([])
        ax_b.imshow(b_img); ax_b.set_xticks([]); ax_b.set_yticks([])
        ax_a.set_ylabel(f"{lab}\np_change={change_probs[idx]:.2f}", fontsize=10,
                        labelpad=8)
        if row == 0:
            ax_a.set_title("A (pre)")
            ax_b.set_title("B (post)")
            ax_gt.set_title("ground truth", loc="left")
            ax_pred.set_title("predictions (TTA + gate, threshold 0.5)", loc="left")

        ax_gt.axis("off")
        ax_gt.text(0.0, 0.95, gt_text, va="top", ha="left", fontsize=9,
                   family="monospace")
        ax_pred.axis("off")
        ax_pred.text(0.0, 0.95, pred_text, va="top", ha="left", fontsize=9,
                     family="monospace")

    fig.suptitle("Phase-2 BIT-only qualitative examples (test split, seed 42)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
