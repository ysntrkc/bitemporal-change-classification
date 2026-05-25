#!/usr/bin/env bash
# Per Phase 1 checkpoint: tune thresholds on val, then eval test with TTA+tuned and default 0.5.

set -euo pipefail
cd "$(dirname "$0")/.."

source /home/ysn/miniconda3/etc/profile.d/conda.sh
conda activate blm5135

FAMILIES=(object event attribute)
SEEDS=(42 1337 2024)

for fam in "${FAMILIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    ckpt="results/phase1_${fam}/seed${seed}/best_ema.pth"
    cfg="configs/phase1_main_${fam}.yaml"
    thr="results/phase1_${fam}/seed${seed}/thresholds.json"
    [[ -f "${ckpt}" ]] || { echo "[skip] ${fam}/seed${seed}: no ckpt"; continue; }

    python eval_phase1.py --ckpt "${ckpt}" --config "${cfg}" \
      --mode tune-thresholds --split val
    python eval_phase1.py --ckpt "${ckpt}" --config "${cfg}" \
      --tta --apply-thresholds "${thr}" --split test
    python eval_phase1.py --ckpt "${ckpt}" --config "${cfg}" --split test
  done
done
