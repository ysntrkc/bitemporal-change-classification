#!/usr/bin/env bash
# Tune-then-apply per-class thresholds for the DBLoss object checkpoints.

set -euo pipefail
cd "$(dirname "$0")/.."

source /home/ysn/miniconda3/etc/profile.d/conda.sh
conda activate blm5135

NAME="phase1_object_dbloss"
CFG="configs/phase1_ablation_dbloss_object.yaml"
SEEDS=(42 1337 2024)

for seed in "${SEEDS[@]}"; do
  out="results/${NAME}/seed${seed}"
  ckpt="${out}/best_ema.pth"
  thr="${out}/thresholds.json"
  [[ -f "${ckpt}" ]] || { echo "[skip] seed${seed}: no ckpt"; continue; }

  python eval_phase1.py --ckpt "${ckpt}" --config "${CFG}" \
    --mode tune-thresholds --split val
  python eval_phase1.py --ckpt "${ckpt}" --config "${CFG}" \
    --tta --apply-thresholds "${thr}" --split test
done

python scripts/reporting/ablation_table.py
python scripts/reporting/per_class_report.py \
  --prefixes phase1_object phase1_event phase1_attribute phase1_object_dbloss
