#!/usr/bin/env bash
# Overnight chain:
#   1. Wait for the currently-running phase1_object_classaware suite to finish.
#   2. Run Phase 2 class-aware (BIT-only base + class-aware sampler) end-to-end.
#   3. Run tuned-threshold eval for both class-aware variants (object + phase2)
#      so the ablation table has both default-0.5 and tuned-thr rows.
#   4. Final ablation + per-class table refresh.
#
# Sentinel for step 1 is the seed-2024 canonical-eval JSON, which is the last
# artefact the object suite writes before its ablation refresh. A short grace
# period lets the table-refresh complete before we proceed.
#
# Usage: bash scripts/overnight_classaware_chain.sh

set -euo pipefail
cd "$(dirname "$0")/.."

LOG="results/overnight_classaware_chain.log"
exec >> "${LOG}" 2>&1

echo
echo "[chain] $(date '+%F %T'): start. waiting for phase1_object_classaware to finish."

source /home/ysn/miniconda3/etc/profile.d/conda.sh
conda activate blm5135

SENTINEL="results/phase1_object_classaware/seed2024/metrics_test_tta.json"
while [[ ! -f "${SENTINEL}" ]]; do
  sleep 90
done
echo "[chain] $(date '+%F %T'): sentinel detected (${SENTINEL}). 60s grace period."
sleep 60

# ---- Step 2: Phase 2 class-aware ----
echo
echo "[chain] $(date '+%F %T'): launching Phase 2 class-aware suite."
bash scripts/run_classaware_suite.sh phase2_bit_only_classaware

# ---- Step 3: tuned-threshold pass on both class-aware prefixes ----
echo
echo "[chain] $(date '+%F %T'): tuned-threshold eval on class-aware variants."
SEEDS=(42 1337 2024)

# Phase 1 object: uses eval_phase1.py
P1_NAME="phase1_object_classaware"
P1_CFG="configs/${P1_NAME}.yaml"
for seed in "${SEEDS[@]}"; do
  ckpt="results/${P1_NAME}/seed${seed}/best_ema.pth"
  thr="results/${P1_NAME}/seed${seed}/thresholds.json"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[chain] skip ${P1_NAME}/seed${seed}: ckpt missing"
    continue
  fi
  echo "=== ${P1_NAME}/seed${seed}: tune-thresholds on val ==="
  python eval_phase1.py --ckpt "${ckpt}" --config "${P1_CFG}" \
    --mode tune-thresholds --split val
  echo "=== ${P1_NAME}/seed${seed}: test + TTA + tuned thr ==="
  python eval_phase1.py --ckpt "${ckpt}" --config "${P1_CFG}" \
    --tta --apply-thresholds "${thr}" --split test
done

# Phase 2's canonical eval (TTA + gate) is already produced by the suite
# launcher. eval_phase2.py does not yet implement tune-thresholds, so
# we skip the tuned-thr pass for Phase 2 to avoid adding scope here.

# ---- Step 4: table refresh ----
echo
echo "[chain] $(date '+%F %T'): refreshing tables."
python scripts/ablation_table.py || echo "[warn] ablation_table.py failed"
python scripts/per_class_report.py \
  --prefixes phase1_object phase1_event phase1_attribute \
             phase1_object_dbloss phase1_object_classaware \
  || echo "[warn] per_class_report.py failed"

echo
echo "[chain] $(date '+%F %T'): done."
