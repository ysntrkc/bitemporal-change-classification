#!/usr/bin/env bash
# Phase-1 object family, DBLoss ablation row.
# Train 3 seeds × 40 epochs, then canonical eval (TTA, default 0.5).
# Resume-safe: skips any seed whose best_ema.pth already exists.
#
# Usage:
#   bash scripts/run_phase1_dbloss.sh
#
# Wallclock estimate: 3 × ~70 min train + 3 × ~30 s eval ≈ 3.5 h.

set -euo pipefail
cd "$(dirname "$0")/.."

source /home/ysn/miniconda3/etc/profile.d/conda.sh
conda activate blm5135

NAME="phase1_object_dbloss"
CFG="configs/${NAME}.yaml"
SEEDS=(42 1337 2024)

# ---- Train ----
for seed in "${SEEDS[@]}"; do
  out="results/${NAME}/seed${seed}"
  if [[ -f "${out}/best_ema.pth" ]]; then
    echo "[skip-train] ${NAME}/seed${seed}: best_ema.pth exists"
    continue
  fi
  mkdir -p "${out}"
  log="${out}/run.log"
  echo "=========================================="
  echo "[train] ${NAME}/seed${seed} -> ${log}"
  echo "=========================================="
  python train_phase1.py --config "${CFG}" --seed "${seed}" --output "${out}" 2>&1 | tee "${log}"
done

# ---- Canonical eval ----
echo
echo "=========================================="
echo "[eval] canonical TTA + default 0.5 ${NAME}"
echo "=========================================="
for seed in "${SEEDS[@]}"; do
  ckpt="results/${NAME}/seed${seed}/best_ema.pth"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[skip-eval] ${NAME}/seed${seed}: ckpt missing"
    continue
  fi
  python eval_phase1.py --ckpt "${ckpt}" --config "${CFG}" --tta --split test
done

# ---- Refresh ablation + per-class tables ----
echo
echo "=========================================="
echo "[report] ablation table + per-class JSON refresh"
echo "=========================================="
python scripts/ablation_table.py || echo "[warn] ablation_table.py failed; run manually"
python scripts/per_class_report.py \
  --prefixes phase1_object phase1_event phase1_attribute phase1_object_dbloss \
  || echo "[warn] per_class_report.py failed; run manually"

echo
echo "[all-done] phase1_object_dbloss suite complete"
