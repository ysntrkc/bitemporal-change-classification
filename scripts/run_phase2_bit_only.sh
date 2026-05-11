#!/usr/bin/env bash
# Train Phase-2 BIT-only seeds 1337 and 2024 (seed 42 already done),
# evaluate all 3 ckpts on the test split with TTA + no-change gate, and
# regenerate the aggregated phase2_table.md / per-class plots / curves.
#
# Usage: bash scripts/run_phase2_bit_only.sh
#
# Each training run is ~70 min on the RTX 5070 (50 epochs, bs=16,
# grad_accum=2). Eval per ckpt is < 30s. Total ~140 min unattended.

set -euo pipefail
cd "$(dirname "$0")/.."

source /home/ysn/miniconda3/etc/profile.d/conda.sh
conda activate blm5135

CONFIG=configs/phase2_bit_only.yaml

for seed in 1337 2024; do
  out="results/phase2_bit_only/seed${seed}"
  if [[ -f "${out}/best_ema.pth" ]]; then
    echo "[skip] seed${seed}: ${out}/best_ema.pth already exists"
    continue
  fi
  mkdir -p "${out}"
  log="${out}/run.log"
  echo "=========================================="
  echo "[train] seed${seed} -> ${log}"
  echo "=========================================="
  python train_phase2.py \
    --config "${CONFIG}" \
    --seed "${seed}" \
    --output "${out}" 2>&1 | tee "${log}"
  echo "[done] train seed${seed}"
done

echo
echo "=========================================="
echo "[eval] canonical metrics: test + TTA + gate"
echo "=========================================="
for seed in 42 1337 2024; do
  ckpt="results/phase2_bit_only/seed${seed}/best_ema.pth"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[skip] seed${seed}: ${ckpt} missing"
    continue
  fi
  python eval_phase2.py --ckpt "${ckpt}" --config "${CONFIG}" --tta --gate
done

echo
echo "=========================================="
echo "[report] aggregating phase2_table.md + plots"
echo "=========================================="
python scripts/phase2_report.py

echo
echo "[all-done] Phase-2 BIT-only multi-seed pipeline complete"
