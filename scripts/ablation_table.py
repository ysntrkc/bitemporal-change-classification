"""Build the Phase-1 + Phase-2 ablation table (Table II).

Combines every saved ``metrics_test*.json`` into a single markdown table.
Phase-1 rows aggregate mean ± std over 3 seeds × 3 families. Phase-2
ablation rows that only ran one seed are reported as single-seed
numbers (clearly marked). The canonical Phase-2 BIT-only row spans 3
seeds and gets mean ± std.

Output: ``results/ablation_table.md``.
"""

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
    """One ablation row.

    ``json_pattern`` accepts ``{family}`` and ``{seed}`` placeholders. ``seeds``
    is the seed list to aggregate; pass a single-element list for one-seed rows.
    """
    label: str
    json_pattern: str
    seeds: tuple[int, ...]
    note: str = ""

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
    """Return {family: (mean, std)} plus 'mean' across families. None on missing."""
    out: dict[str, tuple[float, float] | None] = {}
    fam_means: list[float] = []
    for fam in FAMILIES:
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
    if fam_means:
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
    # Phase-1 ablations (4 flavors × 3 seeds × 3 families)
    RowSpec("P1: default 0.5, no TTA",
            "phase1_{family}/seed{seed}/metrics_test.json", SEEDS),
    RowSpec("P1: default 0.5, +TTA  (canonical)",
            "phase1_{family}/seed{seed}/metrics_test_tta.json", SEEDS),
    RowSpec("P1: tuned thr, no TTA",
            "phase1_{family}/seed{seed}/metrics_test_thr.json", SEEDS),
    RowSpec("P1: tuned thr, +TTA",
            "phase1_{family}/seed{seed}/metrics_test_tta_thr.json", SEEDS),
    # Phase-2 ablations (variant labels emit different files; family axis runs
    # inside the same JSON, so we re-use the same path for all families)
    RowSpec("P2: no BIT, linear heads, fixed weights",
            "phase2_no_bit/seed{seed}/metrics_test_tta_gate.json", (42,),
            note="single seed"),
    RowSpec("P2: BIT, linear heads, fixed weights  (canonical)",
            "phase2_bit_only/seed{seed}/metrics_test_tta_gate.json", SEEDS),
    RowSpec("P2: BIT, Q2L heads, UWL  (full stack)",
            "phase2_unified/seed{seed}/metrics_test_tta_gate.json", (42,),
            note="single seed"),
]


def main() -> None:
    rows_data: list[tuple[RowSpec, dict]] = [(r, _aggregate_row(r)) for r in ROWS]

    lines: list[str] = []
    lines.append("# Ablation table — macro-F1 on test split (EMA weights)")
    lines.append("")
    lines.append("Phase-1 rows: mean ± std over 3 seeds (42, 1337, 2024). "
                 "Phase-2 rows marked 'single seed' use seed 42 only — these are "
                 "ablation probes, not headline numbers.")
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
