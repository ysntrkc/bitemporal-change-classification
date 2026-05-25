#!/usr/bin/env bash
# Train 3 families × 3 seeds of the canonical Phase 1 model.
# Skips runs whose best.pth already exists.

set -euo pipefail
cd "$(dirname "$0")/.."

FAMILIES=(object event attribute)
SEEDS=(42 1337 2024)

for fam in "${FAMILIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    out="results/phase1_${fam}/seed${seed}"
    if [[ -f "${out}/best.pth" ]]; then
      echo "[skip] ${fam}/seed${seed}"
      continue
    fi
    mkdir -p "${out}"
    python train_phase1.py \
      --config "configs/phase1_main_${fam}.yaml" \
      --seed "${seed}" \
      --output "${out}" 2>&1 | tee "${out}/run.log"
  done
done
