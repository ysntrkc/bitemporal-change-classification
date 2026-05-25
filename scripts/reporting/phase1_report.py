from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FAMILIES = ["object", "event", "attribute"]
SEEDS = [42, 1337, 2024]
RESULTS_DIR = Path("results")
VOCAB = json.loads(Path("configs/label_vocab.json").read_text())


def _load_metrics(fam: str, seed: int) -> dict:
    return json.loads(
        (RESULTS_DIR / f"phase1_{fam}" / f"seed{seed}" / "metrics_test_tta.json").read_text()
    )


def build_table() -> None:
    metric_keys = ["macro_f1", "micro_f1", "precision_macro", "recall_macro", "mAP"]
    metric_pretty = {
        "macro_f1": "macro-F1",
        "micro_f1": "micro-F1",
        "precision_macro": "P (macro)",
        "recall_macro": "R (macro)",
        "mAP": "mAP",
    }

    rows: list[dict[str, str | tuple[float, float]]] = []
    for fam in FAMILIES:
        runs = [_load_metrics(fam, s) for s in SEEDS]
        row: dict[str, str | tuple[float, float]] = {"family": fam}
        for k in metric_keys:
            vals = np.array([r[k] for r in runs])
            row[k] = (float(vals.mean()), float(vals.std(ddof=1)))
        rows.append(row)

    overall: dict[str, str | tuple[float, float]] = {"family": "**mean**"}
    for k in metric_keys:
        means = np.array([r[k][0] for r in rows])
        overall[k] = (float(means.mean()), float(means.std(ddof=1)))
    rows.append(overall)

    lines = []
    lines.append("# Phase-1 results: default 0.5 + TTA, EMA weights (test split)")
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

    out = RESULTS_DIR / "phase1_table.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"[2.9] table -> {out}")
    print()
    print("\n".join(lines))
    print()


def plot_per_class_f1() -> None:
    for fam in FAMILIES:
        labels = VOCAB[fam]
        per_class: list[list[float]] = []  # [n_seeds][n_classes]
        for seed in SEEDS:
            per_class.append(_load_metrics(fam, seed)["per_class_f1"])
        arr = np.asarray(per_class)            # [3, C]
        mean = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=1)
        order = np.argsort(mean)
        sorted_labels = [labels[i] for i in order]
        sorted_mean = mean[order]
        sorted_std = std[order]

        height = max(3.0, 0.30 * len(labels))
        fig, ax = plt.subplots(figsize=(7.5, height))
        ax.barh(range(len(labels)), sorted_mean, xerr=sorted_std,
                color="#4472c4", ecolor="#2f3a5a", capsize=3)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(sorted_labels, fontsize=9)
        ax.set_xlim(0, max(0.6, float(sorted_mean.max() + sorted_std.max() + 0.05)))
        ax.set_xlabel("F1 (test, default 0.5 + TTA, mean ± std over 3 seeds)")
        ax.set_title(f"Phase-1 per-class F1: {fam} ({len(labels)} classes)")
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        fig.tight_layout()
        out = RESULTS_DIR / f"phase1_{fam}_per_class_f1.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[2.10] {fam} bars -> {out}")


def plot_train_curves() -> None:
    for fam in FAMILIES:
        frames: list[pd.DataFrame] = []
        for seed in SEEDS:
            csv = RESULTS_DIR / f"phase1_{fam}" / f"seed{seed}" / "train_log.csv"
            df = pd.read_csv(csv)
            df = df.set_index("epoch")
            df["seed"] = seed
            frames.append(df)
        # Outer-join on epoch so seeds that early-stopped become NaN past their last epoch.
        combined: dict[str, pd.DataFrame] = {}
        for col in ["train_loss", "val_loss", "val_macro_f1"]:
            wide = pd.concat({s: f[col] for s, f in zip(SEEDS, frames)}, axis=1)
            combined[col] = wide

        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
        for ax, col in zip(axes, ["train_loss", "val_loss", "val_macro_f1"]):
            wide = combined[col]
            mean = wide.mean(axis=1, skipna=True)
            std = wide.std(axis=1, ddof=1, skipna=True)
            ax.plot(wide.index, mean, color="#4472c4", linewidth=1.7)
            ax.fill_between(wide.index, mean - std, mean + std,
                            color="#4472c4", alpha=0.18)
            ax.set_xlabel("epoch")
            ax.set_title(col.replace("_", " "))
            ax.grid(linestyle=":", alpha=0.5)
        axes[0].set_ylabel("loss")
        axes[2].set_ylabel("macro-F1")
        fig.suptitle(f"Phase-1 training curves: {fam} (mean ± std, 3 seeds)",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out = RESULTS_DIR / f"phase1_{fam}_curves.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[2.11] {fam} curves -> {out}")


def main() -> None:
    build_table()
    plot_per_class_f1()
    plot_train_curves()


if __name__ == "__main__":
    main()
