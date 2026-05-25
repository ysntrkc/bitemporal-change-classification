#!/usr/bin/env bash
# Eval Phase 1 checkpoints with TTA + multiplicative no-change gate.
# Args: 0 or more prefix names from {phase1_object, phase1_event, phase1_attribute,
# phase1_object_dbloss}. Default = the 3 main families.

set -euo pipefail
cd "$(dirname "$0")/.."

source /home/ysn/miniconda3/etc/profile.d/conda.sh
conda activate blm5135

prefix_to_cfg() {
  case "$1" in
    phase1_object)        echo "configs/phase1_main_object.yaml" ;;
    phase1_event)         echo "configs/phase1_main_event.yaml" ;;
    phase1_attribute)     echo "configs/phase1_main_attribute.yaml" ;;
    phase1_object_dbloss) echo "configs/phase1_ablation_dbloss_object.yaml" ;;
    *) echo ""; return 1 ;;
  esac
}

if [[ $# -gt 0 ]]; then
  PREFIXES=("$@")
else
  PREFIXES=(phase1_object phase1_event phase1_attribute)
fi
SEEDS=(42 1337 2024)

for prefix in "${PREFIXES[@]}"; do
  cfg=$(prefix_to_cfg "${prefix}") || { echo "[skip] unknown prefix ${prefix}"; continue; }
  for seed in "${SEEDS[@]}"; do
    ckpt="results/${prefix}/seed${seed}/best_ema.pth"
    [[ -f "${ckpt}" ]] || { echo "[skip] ${prefix}/seed${seed}: no ckpt"; continue; }
    python eval_phase1.py --ckpt "${ckpt}" --config "${cfg}" --tta --gate --split test
  done
done

python scripts/ablation_table.py
