from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RESULTS = Path("results")
FAMILIES = ("object", "event", "attribute")
SEEDS = (42, 1337, 2024)


@dataclass
class RowSpec:
    label: str
    json_pattern: str  # accepts {family} and {seed} placeholders
    seeds: tuple[int, ...]
    note: str = ""
    families: tuple[str, ...] = FAMILIES

    def family_paths(self, family: str) -> list[Path]:
        return [
            RESULTS / self.json_pattern.format(family=family, seed=s)
            for s in self.seeds
        ]


def _load_macro_f1(path: Path, family: str | None = None) -> float | None:
    if not path.exists():
        return None
    blob = json.loads(path.read_text())
    if family is not None and family in blob:
        return float(blob[family]["macro_f1"])
    if "macro_f1" in blob:
        return float(blob["macro_f1"])
    return None


def _aggregate_row(row: RowSpec) -> dict[str, tuple[float, float] | None]:
    out: dict[str, tuple[float, float] | None] = {}
    fam_means: list[float] = []
    for fam in FAMILIES:
        if fam not in row.families:
            out[fam] = None
            continue
        vals: list[float] = []
        for path in row.family_paths(fam):
            v = _load_macro_f1(path, family=fam if "phase2" in str(path) else None)
            if v is not None:
                vals.append(v)
        if not vals:
            out[fam] = None
            continue
        arr = np.array(vals)
        out[fam] = (float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0)
        fam_means.append(float(arr.mean()))
    # Only compute mean across families if the row covers ALL families.
    if len(row.families) == len(FAMILIES) and fam_means:
        out["mean"] = (float(np.mean(fam_means)),
                       float(np.std(fam_means, ddof=1)) if len(fam_means) > 1 else 0.0)
    else:
        out["mean"] = None
    return out


def _fmt(v: tuple[float, float] | None, multi_seed: bool) -> str:
    if v is None:
        return "—"
    mu, sd = v
    if multi_seed:
        return f"{mu:.4f} ± {sd:.4f}"
    return f"{mu:.4f}"


ROWS: list[RowSpec] = [
    RowSpec("A1: ResNet-50 + ASL  (backbone swap)",
            "phase1_{family}_resnet50/seed{seed}/metrics_test_tta.json", SEEDS),
    RowSpec("P1: ConvNeXt-V2 + ASL, no TTA",
            "phase1_{family}/seed{seed}/metrics_test.json", SEEDS),
    RowSpec("P1: ConvNeXt-V2 + ASL, +TTA  (canonical)",
            "phase1_{family}/seed{seed}/metrics_test_tta.json", SEEDS),
    RowSpec("P1: canonical, +TTA, +gate",
            "phase1_{family}/seed{seed}/metrics_test_tta_gate.json", SEEDS,
            note="multiplicative gate from head_nochg"),
    RowSpec("P1: DBLoss, default 0.5 +TTA  (uncalibrated)",
            "phase1_{family}_dbloss/seed{seed}/metrics_test_tta.json", SEEDS,
            note="object only (270:1 imbalance)",
            families=("object",)),
    RowSpec("P1: DBLoss, tuned thr +TTA  (calibrated)",
            "phase1_{family}_dbloss/seed{seed}/metrics_test_tta_thr.json", SEEDS,
            note="object only (270:1 imbalance)",
            families=("object",)),
    RowSpec("P2: BIT, linear heads, fixed weights  (canonical)",
            "phase2_bit_only/seed{seed}/metrics_test_tta_gate.json", SEEDS),
    RowSpec("P2: no BIT, linear heads, fixed weights",
            "phase2_no_bit/seed{seed}/metrics_test_tta_gate.json", SEEDS,
            note="fusion ablation"),
    RowSpec("P2: BIT, Q2L heads, UWL  (full stack)",
            "phase2_unified/seed{seed}/metrics_test_tta_gate.json", SEEDS,
            note="head + loss ablation"),
]


def main() -> None:
    rows_data: list[tuple[RowSpec, dict]] = [(r, _aggregate_row(r)) for r in ROWS]

    lines: list[str] = []
    lines.append("# Ablation table: macro-F1 on test split (EMA weights)")
    lines.append("")
    lines.append("All rows: mean ± std over 3 seeds (42, 1337, 2024). "
                 "DBLoss row is object-only (heaviest 270:1 imbalance).")
    lines.append("")
    header = "| variant | object | event | attribute | **mean** | notes |"
    sep = "|---|---:|---:|---:|---:|---|"
    lines.append(header)
    lines.append(sep)
    for row, data in rows_data:
        multi = len(row.seeds) > 1
        cells = [
            row.label,
            _fmt(data["object"], multi),
            _fmt(data["event"], multi),
            _fmt(data["attribute"], multi),
            "**" + _fmt(data["mean"], multi) + "**",
            row.note or ("3-seed mean ± std" if multi else "single seed"),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    out_path = RESULTS / "ablation_table.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"saved -> {out_path}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
