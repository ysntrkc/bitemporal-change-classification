"""Generate Phase-2 report artifacts (mirrors scripts/phase1_report.py).

Produces:
    results/phase2_table.md          — main results table (mean ± std over 3 seeds)
                                       with a side-by-side delta vs Phase 1.
    results/phase2_<fam>_per_class_f1.png — per-class F1 bar charts (3-seed mean ± std)
    results/phase2_<fam>_curves.png       — train/val loss + per-family + mean macro-F1

Reads ``metrics_test_tta_gate.json`` (TTA + no-change gate — our canonical
config) and ``train_log.csv`` from each seed directory under
``results/phase2_bit_only/seed<k>/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FAMILIES = ["object", "event", "attribute"]
SEEDS = [42, 1337, 2024]
RESULTS_DIR = Path("results")
RUN_DIR = RESULTS_DIR / "phase2_bit_only"
VOCAB = json.loads(Path("configs/label_vocab.json").read_text())


def _load_metrics(seed: int) -> dict:
    return json.loads(
        (RUN_DIR / f"seed{seed}" / "metrics_test_tta_gate.json").read_text()
    )


def _load_phase1_table() -> dict[str, dict[str, tuple[float, float]]]:
    """Parse the existing Phase-1 markdown table back into numbers."""
    text = (RESULTS_DIR / "phase1_table.md").read_text().splitlines()
    result: dict[str, dict[str, tuple[float, float]]] = {}
    metric_keys = ["macro_f1", "micro_f1", "precision_macro", "recall_macro", "mAP"]
    for line in text:
        if not line.startswith("|") or "---" in line or "family" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        fam = cells[0].strip().strip("*")
        if fam not in (*FAMILIES, "mean"):
            continue
        vals: dict[str, tuple[float, float]] = {}
        for k, c in zip(metric_keys, cells[1:]):
            mu_str, sd_str = c.split("±")
            vals[k] = (float(mu_str), float(sd_str))
        result[fam] = vals
    return result


def build_table() -> None:
    metric_keys = ["macro_f1", "micro_f1", "precision_macro", "recall_macro", "mAP"]
    metric_pretty = {
        "macro_f1": "macro-F1",
        "micro_f1": "micro-F1",
        "precision_macro": "P (macro)",
        "recall_macro": "R (macro)",
        "mAP": "mAP",
    }

    rows: list[dict] = []
    for fam in FAMILIES:
        runs = [_load_metrics(s)[fam] for s in SEEDS]
        row = {"family": fam}
        for k in metric_keys:
            vals = np.array([r[k] for r in runs])
            row[k] = (float(vals.mean()), float(vals.std(ddof=1)))
        rows.append(row)

    overall = {"family": "**mean**"}
    for k in metric_keys:
        means = np.array([r[k][0] for r in rows])
        overall[k] = (float(means.mean()), float(means.std(ddof=1)))
    rows.append(overall)

    p1 = _load_phase1_table()

    lines: list[str] = []
    lines.append("# Phase-2 results — BIT-only (TTA + no-change gate, EMA, test split)")
    lines.append("")
    lines.append("Mean ± std over 3 seeds (42, 1337, 2024). All numbers higher is better.")
    lines.append("")
    header = "| family    | " + " | ".join(metric_pretty[k] for k in metric_keys) + " |"
    sep = "|" + "|".join(["---"] + ["---:"] * len(metric_keys)) + "|"
    lines.append(header)
    lines.append(sep)
    for row in rows:
        cells = [row["family"].ljust(9)]
        for k in metric_keys:
            mu, sd = row[k]
            cells.append(f"{mu:.4f} ± {sd:.4f}")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Phase-2 vs Phase-1 (delta in absolute points)")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for row in rows:
        fam = row["family"].strip("*")
        cells = [row["family"].ljust(9)]
        ref = p1.get(fam)
        for k in metric_keys:
            mu, _ = row[k]
            if ref and k in ref:
                ref_mu = ref[k][0]
                delta = mu - ref_mu
                sign = "+" if delta >= 0 else ""
                cells.append(f"{sign}{delta:.4f}")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")

    out = RESULTS_DIR / "phase2_table.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"[table] -> {out}")
    print()
    print("\n".join(lines))
    print()


def plot_per_class_f1() -> None:
    """Per-family horizontal bars: mean per-class F1 across 3 seeds, sorted by mean."""
    for fam in FAMILIES:
        labels = VOCAB[fam]
        per_class = np.array([_load_metrics(s)[fam]["per_class_f1"] for s in SEEDS])
        mean = per_class.mean(axis=0)
        std = per_class.std(axis=0, ddof=1)
        order = np.argsort(mean)
        sorted_labels = [labels[i] for i in order]
        sorted_mean = mean[order]
        sorted_std = std[order]

        height = max(3.0, 0.30 * len(labels))
        fig, ax = plt.subplots(figsize=(7.5, height))
        ax.barh(range(len(labels)), sorted_mean, xerr=sorted_std,
                color="#c0504d", ecolor="#5a2f2f", capsize=3)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(sorted_labels, fontsize=9)
        ax.set_xlim(0, max(0.6, float(sorted_mean.max() + sorted_std.max() + 0.05)))
        ax.set_xlabel("F1 (test, TTA + no-change gate, mean ± std over 3 seeds)")
        ax.set_title(f"Phase-2 BIT-only per-class F1 — {fam} ({len(labels)} classes)")
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        fig.tight_layout()
        out = RESULTS_DIR / f"phase2_{fam}_per_class_f1.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[per-class] {fam} -> {out}")


def plot_train_curves() -> None:
    """One unified figure per seed: train_loss / val_loss / per-family + mean macro-F1."""
    frames: list[pd.DataFrame] = []
    for seed in SEEDS:
        csv = RUN_DIR / f"seed{seed}" / "train_log.csv"
        df = pd.read_csv(csv).set_index("epoch")
        df["seed"] = seed
        frames.append(df)

    cols_to_avg = ["train_loss", "val_loss", "val_macro_f1_mean",
                   "val_macro_f1_object", "val_macro_f1_event", "val_macro_f1_attribute"]
    combined: dict[str, pd.DataFrame] = {}
    for col in cols_to_avg:
        wide = pd.concat({s: f[col] for s, f in zip(SEEDS, frames)}, axis=1)
        combined[col] = wide

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
    panels = [
        ("loss", ["train_loss", "val_loss"], ["#4472c4", "#c0504d"]),
        ("mean macro-F1", ["val_macro_f1_mean"], ["#4472c4"]),
        ("per-family macro-F1",
         ["val_macro_f1_object", "val_macro_f1_event", "val_macro_f1_attribute"],
         ["#4472c4", "#c0504d", "#9bbb59"]),
    ]
    for ax, (title, cols, colors) in zip(axes, panels):
        for col, color in zip(cols, colors):
            wide = combined[col]
            mean = wide.mean(axis=1, skipna=True)
            std = wide.std(axis=1, ddof=1, skipna=True)
            ax.plot(wide.index, mean, color=color, linewidth=1.7, label=col.replace("val_macro_f1_", "").replace("_", " "))
            ax.fill_between(wide.index, mean - std, mean + std, color=color, alpha=0.18)
        ax.set_xlabel("epoch")
        ax.set_title(title)
        ax.grid(linestyle=":", alpha=0.5)
        if len(cols) > 1:
            ax.legend(fontsize=8)
    fig.suptitle("Phase-2 BIT-only training curves (mean ± std, 3 seeds)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = RESULTS_DIR / "phase2_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[curves] -> {out}")


def main() -> None:
    build_table()
    plot_per_class_f1()
    plot_train_curves()


if __name__ == "__main__":
    main()
