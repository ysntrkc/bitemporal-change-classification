#!/usr/bin/env bash
# Threshold-tune + apply pass for the phase1_object_dbloss checkpoints.
# Runs after run_phase1_dbloss.sh completes the canonical (default 0.5 + TTA)
# eval. Produces metrics_test_tta_thr.json per seed so the ablation table
# can show DBLoss with both calibration variants.
#
# Usage: bash scripts/eval_phase1_dbloss_thr.sh

set -euo pipefail
cd "$(dirname "$0")/.."

source /home/ysn/miniconda3/etc/profile.d/conda.sh
conda activate blm5135

NAME="phase1_object_dbloss"
CFG="configs/${NAME}.yaml"
SEEDS=(42 1337 2024)

for seed in "${SEEDS[@]}"; do
  out="results/${NAME}/seed${seed}"
  ckpt="${out}/best_ema.pth"
  thr="${out}/thresholds.json"

  if [[ ! -f "${ckpt}" ]]; then
    echo "[skip] seed${seed}: ${ckpt} missing"
    continue
  fi

  echo "=========================================="
  echo "=== ${NAME}/seed${seed}"
  echo "=========================================="

  echo "--- 1/2 tune thresholds on val ---"
  python eval_phase1.py --ckpt "${ckpt}" --config "${CFG}" \
    --mode tune-thresholds --split val

  echo "--- 2/2 test metrics: TTA + tuned thresholds ---"
  python eval_phase1.py --ckpt "${ckpt}" --config "${CFG}" \
    --tta --apply-thresholds "${thr}" --split test
done

# Refresh tables (will now have both default-0.5 and tuned-thr DBLoss rows).
echo
echo "[report] refreshing ablation table + per-class JSON"
python scripts/ablation_table.py || echo "[warn] ablation_table.py failed"
python scripts/per_class_report.py \
  --prefixes phase1_object phase1_event phase1_attribute phase1_object_dbloss \
  || echo "[warn] per_class_report.py failed"

echo
echo "[done] dbloss threshold pass complete"
