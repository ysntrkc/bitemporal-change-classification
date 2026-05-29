#!/usr/bin/env bash
# Train + eval 3 families × 3 seeds of Phase 1 with BIT fusion.
# Skips a run whose best_ema.pth exists, eval whose metrics JSON exists.

set -euo pipefail
cd "$(dirname "$0")/../.."

source /home/ysn/miniconda3/etc/profile.d/conda.sh
conda activate blm5135

FAMILIES=(object event attribute)
SEEDS=(42 1337 2024)

mkdir -p results/_logs
CHAIN_LOG="results/_logs/phase1_ablation_bit_chain_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${CHAIN_LOG}") 2>&1

echo "[chain] start $(date -Is)"

for fam in "${FAMILIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    out="results/phase1_${fam}_bit/seed${seed}"
    if [[ -f "${out}/best_ema.pth" ]]; then
      echo "[skip-train] ${fam}/seed${seed}"
      continue
    fi
    mkdir -p "${out}"
    echo "[train] ${fam} seed=${seed}  $(date -Is)"
    python train_phase1.py \
      --config "configs/phase1_ablation_bit_${fam}.yaml" \
      --seed "${seed}" \
      --output "${out}" 2>&1 | tee "${out}/run.log"
  done
done

for fam in "${FAMILIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    out="results/phase1_${fam}_bit/seed${seed}"
    ckpt="${out}/best_ema.pth"
    metrics="${out}/metrics_test_tta.json"
    [[ -f "${ckpt}" ]] || { echo "[skip-eval] ${fam}/seed${seed}: no ckpt"; continue; }
    [[ -f "${metrics}" ]] && { echo "[skip-eval] ${fam}/seed${seed}: metrics exist"; continue; }
    echo "[eval]  ${fam} seed=${seed}  $(date -Is)"
    python eval_phase1.py \
      --ckpt "${ckpt}" \
      --config "configs/phase1_ablation_bit_${fam}.yaml" \
      --tta --split test 2>&1 | tee -a "${out}/run.log"
  done
done

echo "[chain] done  $(date -Is)"
