#!/usr/bin/env bash
# Train every ResNet-50 variant (Phase-1 with both ASL and BCE, Phase-2
# BIT-only) across all 3 seeds, then evaluate each with the canonical
# protocol (TTA; Phase-2 also applies the no-change gate), and finally
# regenerate the ablation table.
#
# Resume-safe: skips any (variant, seed) whose best_ema.pth already exists.
#
# Usage: bash scripts/run_resnet50_suite.sh
#
# Total wallclock: ~20 runs × ~60 min on RTX 5070 ≈ 20 h unattended.

set -euo pipefail
cd "$(dirname "$0")/.."

source /home/ysn/miniconda3/etc/profile.d/conda.sh
conda activate blm5135

SEEDS=(42 1337 2024)

# (variant_name, config_path, runner, eval_extra_args)
PHASE1_VARIANTS=(
  "phase1_object_resnet50:configs/phase1_object_resnet50.yaml"
  "phase1_event_resnet50:configs/phase1_event_resnet50.yaml"
  "phase1_attribute_resnet50:configs/phase1_attribute_resnet50.yaml"
  "phase1_object_resnet50_bce:configs/phase1_object_resnet50_bce.yaml"
  "phase1_event_resnet50_bce:configs/phase1_event_resnet50_bce.yaml"
  "phase1_attribute_resnet50_bce:configs/phase1_attribute_resnet50_bce.yaml"
)

PHASE2_VARIANTS=(
  "phase2_bit_only_resnet50:configs/phase2_bit_only_resnet50.yaml"
)

run_train_phase1() {
  local name="$1" cfg="$2" seed="$3"
  local out="results/${name}/seed${seed}"
  if [[ -f "${out}/best_ema.pth" ]]; then
    echo "[skip-train] ${name}/seed${seed}: best_ema.pth exists"
    return
  fi
  mkdir -p "${out}"
  local log="${out}/run.log"
  echo "=========================================="
  echo "[train] ${name}/seed${seed} -> ${log}"
  echo "=========================================="
  python train.py --config "${cfg}" --seed "${seed}" --output "${out}" 2>&1 | tee "${log}"
}

run_train_phase2() {
  local name="$1" cfg="$2" seed="$3"
  local out="results/${name}/seed${seed}"
  if [[ -f "${out}/best_ema.pth" ]]; then
    echo "[skip-train] ${name}/seed${seed}: best_ema.pth exists"
    return
  fi
  mkdir -p "${out}"
  local log="${out}/run.log"
  echo "=========================================="
  echo "[train] ${name}/seed${seed} -> ${log}"
  echo "=========================================="
  python train_phase2.py --config "${cfg}" --seed "${seed}" --output "${out}" 2>&1 | tee "${log}"
}

run_eval_phase1() {
  local name="$1" cfg="$2" seed="$3"
  local out="results/${name}/seed${seed}"
  local ckpt="${out}/best_ema.pth"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[skip-eval] ${name}/seed${seed}: ckpt missing"
    return
  fi
  if [[ -f "${out}/metrics_test_tta.json" ]]; then
    echo "[skip-eval] ${name}/seed${seed}: metrics_test_tta.json exists"
    return
  fi
  python eval.py --ckpt "${ckpt}" --config "${cfg}" --tta --split test
}

run_eval_phase2() {
  local name="$1" cfg="$2" seed="$3"
  local out="results/${name}/seed${seed}"
  local ckpt="${out}/best_ema.pth"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[skip-eval] ${name}/seed${seed}: ckpt missing"
    return
  fi
  if [[ -f "${out}/metrics_test_tta_gate.json" ]]; then
    echo "[skip-eval] ${name}/seed${seed}: metrics_test_tta_gate.json exists"
    return
  fi
  python eval_phase2.py --ckpt "${ckpt}" --config "${cfg}" --tta --gate
}

# ---- Train all ----
for entry in "${PHASE1_VARIANTS[@]}"; do
  name="${entry%%:*}"; cfg="${entry##*:}"
  for seed in "${SEEDS[@]}"; do
    run_train_phase1 "${name}" "${cfg}" "${seed}"
  done
done

for entry in "${PHASE2_VARIANTS[@]}"; do
  name="${entry%%:*}"; cfg="${entry##*:}"
  for seed in "${SEEDS[@]}"; do
    run_train_phase2 "${name}" "${cfg}" "${seed}"
  done
done

# ---- Eval all ----
echo
echo "=========================================="
echo "[eval-all] canonical metrics for all variants"
echo "=========================================="
for entry in "${PHASE1_VARIANTS[@]}"; do
  name="${entry%%:*}"; cfg="${entry##*:}"
  for seed in "${SEEDS[@]}"; do
    run_eval_phase1 "${name}" "${cfg}" "${seed}"
  done
done
for entry in "${PHASE2_VARIANTS[@]}"; do
  name="${entry%%:*}"; cfg="${entry##*:}"
  for seed in "${SEEDS[@]}"; do
    run_eval_phase2 "${name}" "${cfg}" "${seed}"
  done
done

# ---- Regenerate ablation table ----
echo
echo "=========================================="
echo "[report] regenerating ablation table"
echo "=========================================="
python scripts/ablation_table.py

echo
echo "[all-done] ResNet-50 suite complete"
