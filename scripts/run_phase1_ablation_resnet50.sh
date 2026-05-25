#!/usr/bin/env bash
# A1 ablation: ResNet-50 backbone × 3 families × 3 seeds.

set -euo pipefail
cd "$(dirname "$0")/.."

source /home/ysn/miniconda3/etc/profile.d/conda.sh
conda activate blm5135

FAMILIES=(object event attribute)
SEEDS=(42 1337 2024)

for fam in "${FAMILIES[@]}"; do
  cfg="configs/phase1_ablation_resnet50_${fam}.yaml"
  name="phase1_${fam}_resnet50"
  for seed in "${SEEDS[@]}"; do
    out="results/${name}/seed${seed}"
    if [[ -f "${out}/best_ema.pth" ]]; then
      echo "[skip-train] ${fam}/seed${seed}"
      continue
    fi
    mkdir -p "${out}"
    python train_phase1.py --config "${cfg}" --seed "${seed}" --output "${out}" 2>&1 \
      | tee "${out}/run.log"
  done
  for seed in "${SEEDS[@]}"; do
    ckpt="results/${name}/seed${seed}/best_ema.pth"
    [[ -f "${ckpt}" ]] || { echo "[skip-eval] ${fam}/seed${seed}"; continue; }
    python eval_phase1.py --ckpt "${ckpt}" --config "${cfg}" --tta --split test
  done
done

python scripts/ablation_table.py
