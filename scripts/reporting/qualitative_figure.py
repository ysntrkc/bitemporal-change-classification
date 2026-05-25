from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from PIL import Image

from src.augment import EvalTransform
from src.config import load_config
from src.dataset import build_dataloaders
from src.metrics import tta_forward
from src.model import build_model
from src.utils import load_checkpoint, seed_everything

from scripts.reporting.teaser_figure import (
    FAMILIES,
    FAMILY_PROB_KEY,
    FAMILY_Y_KEY,
    _gather_chips,
    _per_sample_macro_f1,
    render_card_figure,
)

CKPT = "results/phase2_bit_only/seed42/best_ema.pth"
CONFIG = "configs/phase2_main.yaml"
DATASET_JSON = "dataset/dataset.json"
DATASET_ROOT = Path("dataset")
OUT_PATH = Path("reports/figs/qualitative.png")
TTA_OPS = ("orig", "hflip", "vflip", "rot180")


def main() -> None:
    cfg = load_config(CONFIG)
    seed_everything(int(cfg["experiment"].get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = json.loads(Path(cfg["data"]["vocab_path"]).read_text())

    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(CKPT)
    model.load_state_dict(ckpt["model"])
    model.eval()
    mean, std = model.encoder.norm_stats()
    transform = EvalTransform(img_size=int(cfg["data"].get("img_size", 224)),
                              mean=mean, std=std)
    _, _, test_loader = build_dataloaders(
        cfg, transform_train=transform, transform_eval=transform
    )

    all_probs: dict[str, list[np.ndarray]] = {fam: [] for fam in FAMILIES}
    all_targets: dict[str, list[np.ndarray]] = {fam: [] for fam in FAMILIES}
    all_change: list[np.ndarray] = []
    all_sample_ids: list[str] = []
    for batch in test_loader:
        a = batch["A"].to(device, non_blocking=True)
        b = batch["B"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = tta_forward(model, {"A": a, "B": b}, list(TTA_OPS))
        all_change.append(out["prob_nochg"].cpu().numpy())
        for fam in FAMILIES:
            all_probs[fam].append(out[FAMILY_PROB_KEY[fam]].cpu().numpy())
            all_targets[fam].append(batch[FAMILY_Y_KEY[fam]].cpu().numpy())
        all_sample_ids.extend(batch["sample_id"])

    change_probs = np.concatenate(all_change, axis=0)
    fam_probs = {fam: np.concatenate(all_probs[fam], axis=0) for fam in FAMILIES}
    fam_targets = {fam: np.concatenate(all_targets[fam], axis=0) for fam in FAMILIES}
    for fam in FAMILIES:
        fam_probs[fam] = fam_probs[fam] * change_probs[:, None]

    per_fam_f1 = {fam: _per_sample_macro_f1(fam_probs[fam], fam_targets[fam])
                  for fam in FAMILIES}
    has_pos = np.zeros(len(all_sample_ids), dtype=bool)
    for fam in FAMILIES:
        has_pos |= fam_targets[fam].sum(axis=1) > 0
    sample_f1 = np.mean([per_fam_f1[fam] for fam in FAMILIES], axis=0)

    candidates = np.where(has_pos)[0]
    # top 2 success + bottom 2 failure (all with at least one positive label)
    order = np.argsort(sample_f1[candidates])
    bottom_idx = candidates[order[:2]].tolist()
    top_idx = candidates[order[-2:]][::-1].tolist()
    picks_idx = top_idx + bottom_idx
    row_kinds = ["success", "success", "failure", "failure"]

    blob = json.loads(Path(DATASET_JSON).read_text())
    by_id = {r["sample_id"]: r for r in blob["images"]}

    picks_data: list[dict] = []
    for idx, row_kind in zip(picks_idx, row_kinds):
        rec = by_id[all_sample_ids[idx]]
        a_img = Image.open(DATASET_ROOT / rec["rgb_A"]).convert("RGB")
        b_img = Image.open(DATASET_ROOT / rec["rgb_B"]).convert("RGB")
        chips_per_family = {
            fam: _gather_chips(fam_probs[fam][idx], fam_targets[fam][idx], vocab[fam])
            for fam in FAMILIES
        }
        picks_data.append({
            "sample_id": all_sample_ids[idx],
            "f1": float(sample_f1[idx]),
            "p_change": float(change_probs[idx]),
            "row_kind": row_kind,
            "a_img": a_img,
            "b_img": b_img,
            "chips_per_family": chips_per_family,
        })
        print(f"[{row_kind}] {all_sample_ids[idx]}  "
              f"F1={sample_f1[idx]:.3f}  P_chg={change_probs[idx]:.3f}")

    render_card_figure(picks_data, OUT_PATH)


if __name__ == "__main__":
    main()
