#!/usr/bin/env bash
# Re-runs canonical Phase-1 eval (default 0.5 + TTA) on every existing
# best_ema.pth checkpoint to refresh metrics_test_tta.json with the
# extended per-class precision / recall / AP / support fields.
#
# Cheap: each pass is a single forward + TTA over the test split.
#
# Usage:
#   bash scripts/regen_canonical_eval.sh
#   bash scripts/regen_canonical_eval.sh phase1_object_dbloss     # subset

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
  fam="${prefix#phase1_}"
  fam="${fam%%_*}"
  cfg_name="configs/${prefix}.yaml"
  for seed in "${SEEDS[@]}"; do
    ckpt="results/${prefix}/seed${seed}/best_ema.pth"
    if [[ ! -f "${ckpt}" ]]; then
      echo "[skip] ${prefix}/seed${seed}: ${ckpt} missing"
      continue
    fi
    if [[ ! -f "${cfg_name}" ]]; then
      echo "[skip] ${prefix}/seed${seed}: ${cfg_name} missing"
      continue
    fi
    echo "=== ${prefix}/seed${seed} (family=${fam}) ==="
    python eval_phase1.py --ckpt "${ckpt}" --config "${cfg_name}" --tta --split test
  done
done

echo "[regen] done"
