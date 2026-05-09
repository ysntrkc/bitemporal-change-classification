#!/usr/bin/env bash
# Bulk launcher: train all 9 Phase-1 runs (3 families × 3 seeds).
# Skips a run if its best.pth already exists, so re-running resumes
# the queue rather than overwriting completed runs. Exit on first
# failure; tee each run's stdout to its own run.log.
#
# Usage: bash scripts/run_phase1.sh

set -euo pipefail

cd "$(dirname "$0")/.."

FAMILIES=(object event attribute)
SEEDS=(42 1337 2024)

for fam in "${FAMILIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    out="results/phase1_${fam}/seed${seed}"
    if [[ -f "${out}/best.pth" ]]; then
      echo "[skip] ${fam}/seed${seed}: ${out}/best.pth already exists"
      continue
    fi
    mkdir -p "${out}"
    log="${out}/run.log"
    echo "[run]  ${fam}/seed${seed} -> ${log}"
    python train.py \
      --config "configs/phase1_${fam}.yaml" \
      --seed "${seed}" \
      --output "${out}" 2>&1 | tee "${log}"
    echo "[done] ${fam}/seed${seed}"
  done
done

echo "[all-done] 9 Phase-1 runs complete"
