"""No-change handling diagnostic for Phase-1 checkpoints.

For each ``(prefix, seed)`` checkpoint, runs a single forward pass over
the test split (no TTA) and computes two views:

1. **No-change head accuracy.** ``head_nochg`` is trained as an
   auxiliary BCE task on ``is_change`` (1 = changed, 0 = no-change).
   We never used it in canonical eval, so this is the first look at
   whether the auxiliary head actually learned the task.

2. **Family head behaviour on no-change samples.** No-change ground
   truth has every family label = 0. We count, for each family, how
   often the family head predicts at least one positive (``any
   sigmoid > 0.5``) and the mean number of positives — both on
   no-change samples and on changed samples for contrast. The first
   number is the false-positive-on-no-change rate that drags
   precision down.

Outputs ``results/no_change_metrics.json`` (full structured dump
keyed by prefix) and ``results/no_change_metrics.md`` (compact table).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

# Make the project root importable when this is run from /scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from src.augment import EvalTransform
from src.dataset import build_dataloaders
from src.model import Phase1Model
from src.utils import load_checkpoint, seed_everything

logger = logging.getLogger(__name__)

FAMILY_Y_KEY = {"object": "y_obj", "event": "y_evt", "attribute": "y_attr"}
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SEEDS = (42, 1337, 2024)


def _load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _stats(values: list[float]) -> tuple[float, float]:
    clean = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return float("nan"), float("nan")
    mean = sum(clean) / len(clean)
    var = sum((v - mean) ** 2 for v in clean) / len(clean)
    return mean, math.sqrt(var)


def _fmt(mean: float, std: float, ndigits: int = 4) -> str:
    if math.isnan(mean):
        return "—"
    return f"{mean:.{ndigits}f} ± {std:.{ndigits}f}"


@torch.no_grad()
def diagnose_one(ckpt_path: Path, cfg_path: Path, device: torch.device) -> dict[str, Any]:
    """One checkpoint -> diagnostic numbers as a flat dict."""
    cfg = _load_cfg(cfg_path)
    family = cfg["experiment"]["family"]
    fam_key = FAMILY_Y_KEY[family]
    seed_everything(int(cfg["experiment"]["seed"]))

    model = Phase1Model(cfg).to(device)
    ckpt = load_checkpoint(str(ckpt_path))
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()

    mean, std = model.encoder.norm_stats()
    eval_t = EvalTransform(img_size=int(cfg["data"].get("img_size", 224)), mean=mean, std=std)
    _train, _val, test_loader = build_dataloaders(
        cfg, transform_train=eval_t, transform_eval=eval_t
    )

    probs_fam: list[np.ndarray] = []
    probs_nc:  list[np.ndarray] = []
    targets:   list[np.ndarray] = []
    is_change: list[np.ndarray] = []

    for batch in test_loader:
        a = batch["A"].to(device, non_blocking=True)
        b = batch["B"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(a, b)
        probs_fam.append(torch.sigmoid(out["logits_family"].float()).cpu().numpy())
        probs_nc.append(torch.sigmoid(out["logit_nochg"].float()).cpu().numpy())
        targets.append(batch[fam_key].cpu().numpy())
        is_change.append(batch["is_change"].cpu().numpy())

    pf = np.concatenate(probs_fam, axis=0)            # [N, C], P(class present)
    pn = np.concatenate(probs_nc,  axis=0)            # [N]   , P(changed)
    y_fam = np.concatenate(targets, axis=0)           # [N, C]
    y_ch  = np.concatenate(is_change, axis=0).astype(np.int64)   # [N]

    # --- view 1: head_nochg metrics ---
    # The training target was is_change directly (1=changed, 0=no-change).
    # We report metrics with "changed" as the positive class.
    pred_ch = (pn >= 0.5).astype(np.int64)
    tp = int(((pred_ch == 1) & (y_ch == 1)).sum())
    fp = int(((pred_ch == 1) & (y_ch == 0)).sum())
    tn = int(((pred_ch == 0) & (y_ch == 0)).sum())
    fn = int(((pred_ch == 0) & (y_ch == 1)).sum())
    n = int(y_ch.shape[0])
    acc = (tp + tn) / n if n else float("nan")
    prec_ch = tp / (tp + fp) if (tp + fp) else 0.0
    rec_ch = tp / (tp + fn) if (tp + fn) else 0.0
    f1_ch = 2 * prec_ch * rec_ch / (prec_ch + rec_ch) if (prec_ch + rec_ch) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0   # nochg recall

    # --- view 2: family-head behaviour split by is_change ---
    pred_fam = (pf >= 0.5).astype(np.int64)
    any_pos = pred_fam.any(axis=1).astype(np.int64)              # [N]
    n_pos   = pred_fam.sum(axis=1).astype(np.float64)            # [N] (per-sample positive count)
    n_nc = int((y_ch == 0).sum())
    n_chg = int((y_ch == 1).sum())

    fp_rate_on_nc = (
        float((any_pos[y_ch == 0] == 1).sum()) / n_nc if n_nc else float("nan")
    )
    mean_pos_on_nc = float(n_pos[y_ch == 0].mean()) if n_nc else float("nan")
    mean_pos_on_chg = float(n_pos[y_ch == 1].mean()) if n_chg else float("nan")

    return {
        "family": family,
        "n_test": n,
        "n_no_change": n_nc,
        "n_changed": n_chg,
        # nochg head
        "nochg_acc": acc,
        "nochg_f1_change": f1_ch,
        "nochg_specificity": spec,   # recall on no-change class
        "nochg_prob_mean_on_nc": float(pn[y_ch == 0].mean()) if n_nc else float("nan"),
        "nochg_prob_mean_on_chg": float(pn[y_ch == 1].mean()) if n_chg else float("nan"),
        # family head behaviour
        "family_fp_rate_on_nc": fp_rate_on_nc,
        "family_mean_pos_on_nc": mean_pos_on_nc,
        "family_mean_pos_on_chg": mean_pos_on_chg,
    }


METRIC_KEYS = (
    "nochg_acc", "nochg_f1_change", "nochg_specificity",
    "nochg_prob_mean_on_nc", "nochg_prob_mean_on_chg",
    "family_fp_rate_on_nc", "family_mean_pos_on_nc", "family_mean_pos_on_chg",
)


def aggregate(prefix: str, device: torch.device) -> dict[str, Any]:
    cfg_path = ROOT / "configs" / f"{prefix}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config missing: {cfg_path}")

    per_seed: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        ckpt = RESULTS / prefix / f"seed{seed}" / "best_ema.pth"
        if not ckpt.exists():
            logger.warning("missing %s — skip", ckpt)
            continue
        logger.info("eval %s seed %d", prefix, seed)
        per_seed[seed] = diagnose_one(ckpt, cfg_path, device)

    if not per_seed:
        raise FileNotFoundError(f"no checkpoints found under results/{prefix}")

    family = next(iter(per_seed.values()))["family"]
    summary: dict[str, dict[str, float]] = {}
    for key in METRIC_KEYS:
        vals = [per_seed[s][key] for s in per_seed]
        m, sd = _stats(vals)
        summary[key] = {"mean": m, "std": sd}

    return {
        "prefix": prefix,
        "family": family,
        "seeds": sorted(per_seed),
        "n_test":      per_seed[next(iter(per_seed))]["n_test"],
        "n_no_change": per_seed[next(iter(per_seed))]["n_no_change"],
        "n_changed":   per_seed[next(iter(per_seed))]["n_changed"],
        "per_seed": per_seed,
        "summary": summary,
    }


def render_markdown(payloads: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# No-change diagnostic (Phase-1, test split, no TTA)")
    lines.append("")
    lines.append(
        "All numbers are mean ± std over the seeds available for each prefix. "
        "`nochg_*` rows describe the auxiliary head (P(changed)); `family_*` "
        "rows describe what the per-family head predicts split by ground-truth "
        "is_change."
    )
    lines.append("")
    header = "| prefix | family | seeds | nochg acc | nochg F1(chg) | nochg spec | E[P(chg)\\|nc] | E[P(chg)\\|chg] | family FP-rate on nc | mean pos / nc | mean pos / chg |"
    sep    = "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines.append(header)
    lines.append(sep)
    for p in payloads:
        s = p["summary"]
        cells = [
            p["prefix"],
            p["family"],
            ",".join(str(x) for x in p["seeds"]),
            _fmt(s["nochg_acc"]["mean"], s["nochg_acc"]["std"]),
            _fmt(s["nochg_f1_change"]["mean"], s["nochg_f1_change"]["std"]),
            _fmt(s["nochg_specificity"]["mean"], s["nochg_specificity"]["std"]),
            _fmt(s["nochg_prob_mean_on_nc"]["mean"], s["nochg_prob_mean_on_nc"]["std"]),
            _fmt(s["nochg_prob_mean_on_chg"]["mean"], s["nochg_prob_mean_on_chg"]["std"]),
            _fmt(s["family_fp_rate_on_nc"]["mean"], s["family_fp_rate_on_nc"]["std"]),
            _fmt(s["family_mean_pos_on_nc"]["mean"], s["family_mean_pos_on_nc"]["std"]),
            _fmt(s["family_mean_pos_on_chg"]["mean"], s["family_mean_pos_on_chg"]["std"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(prog="no_change_report.py")
    parser.add_argument(
        "--prefixes",
        nargs="+",
        default=["phase1_object", "phase1_event", "phase1_attribute"],
        help="result subdirectory prefixes (default: 3 Phase-1 ASL families)",
    )
    parser.add_argument(
        "--out-json", default=str(RESULTS / "no_change_metrics.json"),
    )
    parser.add_argument(
        "--out-md", default=str(RESULTS / "no_change_metrics.md"),
    )
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        logger.warning("CUDA not available — running on CPU (slow)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payloads: list[dict[str, Any]] = []
    for prefix in args.prefixes:
        try:
            payloads.append(aggregate(prefix, device))
        except FileNotFoundError as e:
            logger.warning("%s — skipping", e)

    if not payloads:
        logger.error("no prefixes yielded data")
        return 1

    Path(args.out_json).write_text(
        json.dumps({"prefixes": {p["prefix"]: p for p in payloads}}, indent=2),
        encoding="utf-8",
    )
    Path(args.out_md).write_text(render_markdown(payloads), encoding="utf-8")
    logger.info("wrote %s and %s", args.out_json, args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
