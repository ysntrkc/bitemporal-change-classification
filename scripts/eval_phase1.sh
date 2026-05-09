#!/usr/bin/env bash
# For each Phase-1 checkpoint:
#   1) Tune per-class thresholds on val      -> thresholds.json
#   2) Test metrics with TTA + tuned thr     -> metrics_test_tta_thr.json
#   3) Test metrics with default 0.5, no TTA -> metrics_test.json (for the
#      tuned-vs-default ablation, task 2.12)
#
# Uses best_ema.pth (the EMA weights, our canonical eval target).
#
# Usage: bash scripts/eval_phase1.sh

set -euo pipefail
cd "$(dirname "$0")/.."

source /home/ysn/miniconda3/etc/profile.d/conda.sh
conda activate blm5135

FAMILIES=(object event attribute)
SEEDS=(42 1337 2024)

for fam in "${FAMILIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    ckpt="results/phase1_${fam}/seed${seed}/best_ema.pth"
    cfg="configs/phase1_${fam}.yaml"
    thr="results/phase1_${fam}/seed${seed}/thresholds.json"
    if [[ ! -f "${ckpt}" ]]; then
      echo "[skip] ${fam}/seed${seed}: ${ckpt} missing"
      continue
    fi
    echo "=========================================="
    echo "=== ${fam}/seed${seed}"
    echo "=========================================="

    echo "--- 1/3 tune thresholds on val ---"
    python eval.py --ckpt "${ckpt}" --config "${cfg}" \
      --mode tune-thresholds --split val

    echo "--- 2/3 test metrics: TTA + tuned thresholds ---"
    python eval.py --ckpt "${ckpt}" --config "${cfg}" \
      --tta --apply-thresholds "${thr}" --split test

    echo "--- 3/3 test metrics: default 0.5, no TTA (baseline) ---"
    python eval.py --ckpt "${ckpt}" --config "${cfg}" --split test
  done
done

echo
echo "[eval-done] all phase-1 evaluations complete"
