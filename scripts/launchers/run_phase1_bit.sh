#!/usr/bin/env bash
# Train + eval 3 families × 3 seeds of Phase 1 with BIT fusion.
# Sequential on a single GPU. Resumable: skips a run if best_ema.pth
# already exists for it; skips the eval if its metrics JSON exists.

set -euo pipefail
cd "$(dirname "$0")/../.."

source /home/ysn/miniconda3/etc/profile.d/conda.sh
conda activate blm5135

FAMILIES=(object event attribute)
SEEDS=(42 1337 2024)

mkdir -p results/_logs
CHAIN_LOG="results/_logs/phase1_bit_chain_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${CHAIN_LOG}") 2>&1

echo "[chain] start $(date -Is)"
echo "[chain] log = ${CHAIN_LOG}"

# -------- Training: 3 families × 3 seeds = 9 runs --------
for fam in "${FAMILIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    out="results/phase1_bit_${fam}/seed${seed}"
    if [[ -f "${out}/best_ema.pth" ]]; then
      echo "[skip-train] ${fam}/seed${seed} (best_ema.pth exists)"
      continue
    fi
    mkdir -p "${out}"
    echo "[train] phase1_bit_${fam} seed=${seed}  $(date -Is)"
    python train_phase1.py \
      --config "configs/phase1_bit_${fam}.yaml" \
      --seed "${seed}" \
      --output "${out}" 2>&1 | tee "${out}/run.log"
  done
done

# -------- Evaluation: TTA + default 0.5 + no gate (Phase 1 canonical) --------
for fam in "${FAMILIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    out="results/phase1_bit_${fam}/seed${seed}"
    ckpt="${out}/best_ema.pth"
    metrics="${out}/metrics_test_tta.json"
    if [[ ! -f "${ckpt}" ]]; then
      echo "[skip-eval] ${fam}/seed${seed} (no ckpt)"
      continue
    fi
    if [[ -f "${metrics}" ]]; then
      echo "[skip-eval] ${fam}/seed${seed} (metrics exist)"
      continue
    fi
    echo "[eval]  phase1_bit_${fam} seed=${seed}  $(date -Is)"
    python eval_phase1.py \
      --ckpt "${ckpt}" \
      --config "configs/phase1_bit_${fam}.yaml" \
      --tta --split test 2>&1 | tee -a "${out}/run.log"
  done
done

echo "[chain] done  $(date -Is)"
