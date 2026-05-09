#!/usr/bin/env bash
# Wait for results/phase1_object/seed42 to finish cleanly, then launch
# the bulk Phase-1 script (which skips seed42 since best.pth exists and
# trains the remaining 8 runs).
#
# Logs to results/phase1_handoff.log.

cd "$(dirname "$0")/.."

LOG="results/phase1_handoff.log"
SEED42_LOG="results/phase1_object/seed42/run.log"

exec >> "${LOG}" 2>&1
echo
echo "[handoff] $(date '+%F %T'): start. waiting for seed42 to finish."

source /home/ysn/miniconda3/etc/profile.d/conda.sh
conda activate blm5135

while true; do
  if grep -qE "done\. best val|Traceback|Error:|FAILED|Killed|CUDA out of memory" "${SEED42_LOG}" 2>/dev/null; then
    break
  fi
  sleep 60
done

echo "[handoff] $(date '+%F %T'): seed42 wait complete."

if grep -q "done\. best val" "${SEED42_LOG}"; then
  echo "[handoff] $(date '+%F %T'): seed42 OK. launching bulk."
  bash scripts/run_phase1.sh
  rc=$?
  echo "[handoff] $(date '+%F %T'): bulk script exited rc=${rc}."
  exit ${rc}
else
  echo "[handoff] $(date '+%F %T'): seed42 did NOT finish cleanly. aborting."
  echo "[handoff] tail of seed42 log:"
  tail -30 "${SEED42_LOG}"
  exit 1
fi
