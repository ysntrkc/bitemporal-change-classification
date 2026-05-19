#!/usr/bin/env bash
# Apply the multiplicative no-change gate (--gate) on top of canonical
# (TTA, default 0.5) Phase-1 evaluation. Produces metrics_test_tta_gate.json
# per (family, seed). Optional positional args restrict the prefixes (default
# is the 3 ASL families; pass phase1_object_dbloss to cover the DBLoss run
# once it finishes).
#
# Usage:
#   bash scripts/eval_phase1_gate.sh
#   bash scripts/eval_phase1_gate.sh phase1_object_dbloss

set -euo pipefail
cd "$(dirname "$0")/.."

source /home/ysn/miniconda3/etc/profile.d/conda.sh
conda activate blm5135

if [[ $# -gt 0 ]]; then
  PREFIXES=("$@")
else
  PREFIXES=(phase1_object phase1_event phase1_attribute)
fi
SEEDS=(42 1337 2024)

for prefix in "${PREFIXES[@]}"; do
  cfg="configs/${prefix}.yaml"
  for seed in "${SEEDS[@]}"; do
    ckpt="results/${prefix}/seed${seed}/best_ema.pth"
    if [[ ! -f "${ckpt}" ]]; then
      echo "[skip] ${prefix}/seed${seed}: ${ckpt} missing"
      continue
    fi
    if [[ ! -f "${cfg}" ]]; then
      echo "[skip] ${prefix}/seed${seed}: ${cfg} missing"
      continue
    fi
    echo "=== ${prefix}/seed${seed} ==="
    python eval_phase1.py --ckpt "${ckpt}" --config "${cfg}" \
      --tta --gate --split test
  done
done

echo
echo "[refresh] ablation table"
python scripts/ablation_table.py || echo "[warn] ablation_table.py failed"

echo "[done] gate pass complete"
