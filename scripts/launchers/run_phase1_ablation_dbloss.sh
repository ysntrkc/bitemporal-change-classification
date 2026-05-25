#!/usr/bin/env bash
# Phase 1 object family, DBLoss ablation: train 3 seeds, eval with TTA + default 0.5.

set -euo pipefail
cd "$(dirname "$0")/.."

source /home/ysn/miniconda3/etc/profile.d/conda.sh
conda activate blm5135

NAME="phase1_object_dbloss"
CFG="configs/phase1_ablation_dbloss_object.yaml"
SEEDS=(42 1337 2024)

for seed in "${SEEDS[@]}"; do
  out="results/${NAME}/seed${seed}"
  if [[ -f "${out}/best_ema.pth" ]]; then
    echo "[skip-train] seed${seed}"
    continue
  fi
  mkdir -p "${out}"
  python train_phase1.py --config "${CFG}" --seed "${seed}" --output "${out}" 2>&1 \
    | tee "${out}/run.log"
done

for seed in "${SEEDS[@]}"; do
  ckpt="results/${NAME}/seed${seed}/best_ema.pth"
  [[ -f "${ckpt}" ]] || { echo "[skip-eval] seed${seed}: no ckpt"; continue; }
  python eval_phase1.py --ckpt "${ckpt}" --config "${CFG}" --tta --split test
done

python scripts/reporting/ablation_table.py
python scripts/reporting/per_class_report.py \
  --prefixes phase1_object phase1_event phase1_attribute phase1_object_dbloss
