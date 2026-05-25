from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
VOCAB_PATH = ROOT / "configs" / "label_vocab.json"
SEEDS = (42, 1337, 2024)


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


def _family_from_prefix(prefix: str) -> str:
    # phase1_object_dbloss -> object
    stem = prefix.removeprefix("phase1_")
    return stem.split("_", 1)[0]


def collect_family(prefix: str, vocab: dict[str, list[str]]) -> dict[str, Any]:
    family = _family_from_prefix(prefix)
    if family not in vocab:
        raise KeyError(f"family {family!r} not in label_vocab.json")
    class_names = vocab[family]
    n_classes = len(class_names)

    seed_payloads: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        path = RESULTS / prefix / f"seed{seed}" / "metrics_test_tta.json"
        if not path.exists():
            logger.warning("missing %s, skipping seed %d", path, seed)
            continue
        with path.open("r", encoding="utf-8") as f:
            seed_payloads[seed] = json.load(f)

    if not seed_payloads:
        raise FileNotFoundError(
            f"no metrics_test_tta.json found under results/{prefix}/seed*/"
        )

    required = ("per_class_f1", "per_class_precision", "per_class_recall",
                "per_class_ap", "per_class_support")
    for seed, d in seed_payloads.items():
        missing = [k for k in required if k not in d]
        if missing:
            raise KeyError(
                f"{prefix}/seed{seed}: metrics JSON missing keys {missing}; "
                f"re-run eval for this checkpoint"
            )
        for k in required:
            if len(d[k]) != n_classes:
                raise ValueError(
                    f"{prefix}/seed{seed}: {k} length {len(d[k])} != "
                    f"n_classes {n_classes}"
                )

    classes: list[dict[str, Any]] = []
    for c in range(n_classes):
        f1_vals = [seed_payloads[s]["per_class_f1"][c] for s in seed_payloads]
        p_vals  = [seed_payloads[s]["per_class_precision"][c] for s in seed_payloads]
        r_vals  = [seed_payloads[s]["per_class_recall"][c] for s in seed_payloads]
        ap_vals = [seed_payloads[s]["per_class_ap"][c] for s in seed_payloads]
        sup_vals = [seed_payloads[s]["per_class_support"][c] for s in seed_payloads]

        f1_m, f1_s = _stats(f1_vals)
        p_m,  p_s  = _stats(p_vals)
        r_m,  r_s  = _stats(r_vals)
        ap_m, ap_s = _stats(ap_vals)
        # support is deterministic across seeds; sanity-check.
        if len(set(sup_vals)) > 1:
            logger.warning(
                "%s class %s: support varies across seeds %s",
                prefix, class_names[c], sup_vals,
            )

        classes.append({
            "name": class_names[c],
            "support_test": int(sup_vals[0]),
            "f1": {"mean": f1_m, "std": f1_s, "per_seed": dict(zip(seed_payloads, f1_vals))},
            "precision": {"mean": p_m, "std": p_s, "per_seed": dict(zip(seed_payloads, p_vals))},
            "recall": {"mean": r_m, "std": r_s, "per_seed": dict(zip(seed_payloads, r_vals))},
            "ap": {"mean": ap_m, "std": ap_s, "per_seed": dict(zip(seed_payloads, ap_vals))},
        })

    # also aggregate top-level macro/micro for sanity
    def _agg(key: str) -> dict[str, float]:
        vals = [seed_payloads[s][key] for s in seed_payloads]
        m, sd = _stats(vals)
        return {"mean": m, "std": sd}

    return {
        "prefix": prefix,
        "family": family,
        "seeds_used": list(seed_payloads.keys()),
        "n_classes": n_classes,
        "summary": {
            "macro_f1":  _agg("macro_f1"),
            "micro_f1":  _agg("micro_f1"),
            "precision_macro": _agg("precision_macro"),
            "recall_macro":    _agg("recall_macro"),
            "mAP":       _agg("mAP"),
        },
        "classes": classes,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    family = payload["family"]
    prefix = payload["prefix"]
    seeds = payload["seeds_used"]
    summary = payload["summary"]

    lines: list[str] = []
    lines.append(f"## {prefix} (family = {family})")
    lines.append("")
    lines.append(f"Seeds aggregated: {seeds}. Canonical config: default 0.5 + TTA, EMA weights, test split.")
    lines.append("")
    lines.append(
        f"Aggregate macro-F1 **{_fmt(summary['macro_f1']['mean'], summary['macro_f1']['std'])}** | "
        f"micro-F1 {_fmt(summary['micro_f1']['mean'], summary['micro_f1']['std'])} | "
        f"mAP {_fmt(summary['mAP']['mean'], summary['mAP']['std'])}"
    )
    lines.append("")
    lines.append("| class | support | F1 (mean ± std) | precision | recall | AP |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    # sort by descending support so rare classes pop to the bottom
    rows = sorted(payload["classes"], key=lambda c: -c["support_test"])
    for c in rows:
        lines.append(
            f"| {c['name']} | {c['support_test']} | "
            f"{_fmt(c['f1']['mean'], c['f1']['std'])} | "
            f"{_fmt(c['precision']['mean'], c['precision']['std'])} | "
            f"{_fmt(c['recall']['mean'], c['recall']['std'])} | "
            f"{_fmt(c['ap']['mean'], c['ap']['std'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(prog="per_class_report.py")
    parser.add_argument(
        "--prefixes",
        nargs="+",
        default=["phase1_object", "phase1_event", "phase1_attribute"],
        help="result subdirectory prefixes (default: 3 Phase-1 families)",
    )
    parser.add_argument(
        "--out-json",
        default=str(RESULTS / "per_class_metrics.json"),
    )
    parser.add_argument(
        "--out-md",
        default=str(RESULTS / "per_class_metrics.md"),
    )
    args = parser.parse_args(argv)

    with VOCAB_PATH.open("r", encoding="utf-8") as f:
        vocab = json.load(f)

    big: dict[str, Any] = {"prefixes": {}}
    md_blocks: list[str] = ["# Per-class metrics (Phase-1 canonical: TTA + default 0.5)", ""]
    for prefix in args.prefixes:
        payload = collect_family(prefix, vocab)
        big["prefixes"][prefix] = payload
        md_blocks.append(render_markdown(payload))
        logger.info(
            "%s: %d classes, %d seeds, macro-F1 %.4f ± %.4f",
            prefix, payload["n_classes"], len(payload["seeds_used"]),
            payload["summary"]["macro_f1"]["mean"],
            payload["summary"]["macro_f1"]["std"],
        )

    Path(args.out_json).write_text(json.dumps(big, indent=2), encoding="utf-8")
    Path(args.out_md).write_text("\n".join(md_blocks), encoding="utf-8")
    logger.info("wrote %s and %s", args.out_json, args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
