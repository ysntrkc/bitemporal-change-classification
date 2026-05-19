#!/usr/bin/env bash
# Generic launcher for class-aware-sampler ablation rows.
# Pass one or more prefix names; each gets 3-seed train + canonical eval.
# Resume-safe: skips any (prefix, seed) whose best_ema.pth already exists.
#
# Supported prefixes (must have a matching configs/<prefix>.yaml):
#   phase1_object_classaware       -> 40 epochs, train_phase1.py
#   phase1_event_classaware        -> 40 epochs, train_phase1.py
#   phase1_attribute_classaware    -> 40 epochs, train_phase1.py
#   phase2_bit_only_classaware     -> 50 epochs, train_phase2.py
#
# Usage:
#   bash scripts/run_classaware_suite.sh phase1_event_classaware phase1_attribute_classaware
#   bash scripts/run_classaware_suite.sh phase2_bit_only_classaware
#
# Wallclock estimate per prefix: Phase 1 ~3.5 h, Phase 2 ~4 h.

set -euo pipefail
cd "$(dirname "$0")/.."

source /home/ysn/miniconda3/etc/profile.d/conda.sh
conda activate blm5135

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <prefix> [<prefix> ...]" >&2
  exit 2
fi

SEEDS=(42 1337 2024)

select_trainer() {
  case "$1" in
    phase1_*) echo "train_phase1.py" ;;
    phase2_*) echo "train_phase2.py" ;;
    *) echo "unknown phase for prefix $1" >&2; exit 2 ;;
  esac
}

select_evaluator() {
  case "$1" in
    phase1_*) echo "eval_phase1.py --tta --split test" ;;
    phase2_*) echo "eval_phase2.py --tta --gate" ;;
    *) echo "unknown phase for prefix $1" >&2; exit 2 ;;
  esac
}

eval_output_file() {
  case "$1" in
    phase1_*) echo "metrics_test_tta.json" ;;
    phase2_*) echo "metrics_test_tta_gate.json" ;;
  esac
}

for prefix in "$@"; do
  cfg="configs/${prefix}.yaml"
  if [[ ! -f "${cfg}" ]]; then
    echo "[skip-prefix] ${prefix}: ${cfg} missing"
    continue
  fi
  trainer=$(select_trainer "${prefix}")
  evaluator=$(select_evaluator "${prefix}")
  eval_file=$(eval_output_file "${prefix}")

  # ---- Train ----
  for seed in "${SEEDS[@]}"; do
    out="results/${prefix}/seed${seed}"
    if [[ -f "${out}/best_ema.pth" ]]; then
      echo "[skip-train] ${prefix}/seed${seed}: best_ema.pth exists"
      continue
    fi
    mkdir -p "${out}"
    log="${out}/run.log"
    echo "=========================================="
    echo "[train] ${prefix}/seed${seed} via ${trainer} -> ${log}"
    echo "=========================================="
    python "${trainer}" --config "${cfg}" --seed "${seed}" --output "${out}" 2>&1 | tee "${log}"
  done

  # ---- Canonical eval ----
  echo
  echo "=========================================="
  echo "[eval] canonical ${prefix}"
  echo "=========================================="
  for seed in "${SEEDS[@]}"; do
    ckpt="results/${prefix}/seed${seed}/best_ema.pth"
    if [[ ! -f "${ckpt}" ]]; then
      echo "[skip-eval] ${prefix}/seed${seed}: ckpt missing"
      continue
    fi
    if [[ -f "results/${prefix}/seed${seed}/${eval_file}" ]]; then
      echo "[skip-eval] ${prefix}/seed${seed}: ${eval_file} exists"
      continue
    fi
    # shellcheck disable=SC2086
    python ${evaluator} --ckpt "${ckpt}" --config "${cfg}"
  done
done

# ---- Refresh tables ----
echo
echo "[report] ablation table + per-class refresh"
python scripts/ablation_table.py || echo "[warn] ablation_table.py failed"

echo "[done] class-aware suite complete"
