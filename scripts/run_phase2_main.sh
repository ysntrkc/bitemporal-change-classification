#!/usr/bin/env bash
# Train + evaluate the canonical Phase 2 model across 3 seeds.

set -euo pipefail
cd "$(dirname "$0")/.."

source /home/ysn/miniconda3/etc/profile.d/conda.sh
conda activate blm5135

CONFIG=configs/phase2_main.yaml
SEEDS=(42 1337 2024)

for seed in "${SEEDS[@]}"; do
  out="results/phase2_bit_only/seed${seed}"
  if [[ -f "${out}/best_ema.pth" ]]; then
    echo "[skip-train] seed${seed}"
    continue
  fi
  mkdir -p "${out}"
  python train_phase2.py --config "${CONFIG}" --seed "${seed}" --output "${out}" 2>&1 \
    | tee "${out}/run.log"
done

for seed in "${SEEDS[@]}"; do
  ckpt="results/phase2_bit_only/seed${seed}/best_ema.pth"
  [[ -f "${ckpt}" ]] || { echo "[skip-eval] seed${seed}: no ckpt"; continue; }
  python eval_phase2.py --ckpt "${ckpt}" --config "${CONFIG}" --tta --gate
done

python scripts/phase2_report.py
