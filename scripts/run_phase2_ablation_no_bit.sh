#!/usr/bin/env bash
# Phase 2 no-BIT ablation: 3 seeds, fusion=passthrough, linear heads, fixed weights.

set -euo pipefail
cd "$(dirname "$0")/.."

source /home/ysn/miniconda3/etc/profile.d/conda.sh
conda activate blm5135

CONFIG=configs/phase2_ablation_no_bit.yaml
NAME=phase2_no_bit
SEEDS=(42 1337 2024)

for seed in "${SEEDS[@]}"; do
  out="results/${NAME}/seed${seed}"
  if [[ -f "${out}/best_ema.pth" ]]; then
    echo "[skip-train] seed${seed}"
    continue
  fi
  mkdir -p "${out}"
  python train_phase2.py --config "${CONFIG}" --seed "${seed}" --output "${out}" 2>&1 \
    | tee "${out}/run.log"
done

for seed in "${SEEDS[@]}"; do
  ckpt="results/${NAME}/seed${seed}/best_ema.pth"
  [[ -f "${ckpt}" ]] || { echo "[skip-eval] seed${seed}"; continue; }
  python eval_phase2.py --ckpt "${ckpt}" --config "${CONFIG}" --tta --gate
done

python scripts/ablation_table.py
