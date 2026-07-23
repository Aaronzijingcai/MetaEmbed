#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
TRAIN_WRAPPER="$SCRIPT_DIR/run_train.sh"
RUN_ROOT="$PROJECT_DIR/experiments/2026-07-08/runs"

INTERACTION_STRATEGY=${INTERACTION_STRATEGY:-q2d_topk48_mean}
PROBE_DATA_START_STEP=${PROBE_DATA_START_STEP:-982}
PROBE_DATA_END_STEP=${PROBE_DATA_END_STEP:-986}
PROBE_SUFFIX=${PROBE_SUFFIX:-probe_data${PROBE_DATA_START_STEP}_${PROBE_DATA_END_STEP}_$(date +%Y%m%d_%H%M%S)}

if [[ "$PROBE_DATA_END_STEP" -lt "$PROBE_DATA_START_STEP" ]]; then
  echo "[probe-984] end step must be >= start step" >&2
  exit 2
fi

case "$INTERACTION_STRATEGY" in
  q2d_topk48_mean|q2d)
    SOURCE_RUN=rhc_mmeb_vidore_q2d_topk48_mean_from_base
    ;;
  adaptive_bidirectional_topk48_mean|adaptive)
    SOURCE_RUN=rhc_mmeb_vidore_bi_topk48_adaptive_mean_from_base
    ;;
  *)
    echo "[probe-984] expected q2d or adaptive, got $INTERACTION_STRATEGY" >&2
    exit 2
    ;;
esac

RESUME_CKPT=${RESUME_CKPT:-$RUN_ROOT/$SOURCE_RUN/checkpoint-950}
for required in trainer_state.json optimizer.pt scheduler.pt folder_homo.pt adapter_model.bin; do
  if [[ ! -f "$RESUME_CKPT/$required" ]]; then
    echo "[probe-984] missing $RESUME_CKPT/$required" >&2
    exit 2
  fi
done

PROBE_BATCH_COUNT=$((PROBE_DATA_END_STEP - PROBE_DATA_START_STEP + 1))
STOP_AFTER_STEP=$((950 + PROBE_BATCH_COUNT))
DEBUG_END_STEP=$((STOP_AFTER_STEP - 1))

echo "[probe-984] strategy=$INTERACTION_STRATEGY checkpoint=$RESUME_CKPT"
echo "[probe-984] data_window=$PROBE_DATA_START_STEP-$PROBE_DATA_END_STEP global_window=950-$DEBUG_END_STEP"

RESUME_CKPT="$RESUME_CKPT" \
RUN_SUFFIX="$PROBE_SUFFIX" \
MAX_STEPS=60000 \
SAVE_STEPS=950 \
STOP_AFTER_STEP="$STOP_AFTER_STEP" \
SKIP_EVAL=1 \
IGNORE_DATA_SKIP=0 \
MURE_PROBE_DATA_START_STEP="$PROBE_DATA_START_STEP" \
CONTRASTIVE_DEBUG_STEPS="950-$DEBUG_END_STEP" \
MURE_GATHER_WITH_GRAD_MODE="${MURE_GATHER_WITH_GRAD_MODE:-torch}" \
TORCH_DISTRIBUTED_TIMEOUT="${TORCH_DISTRIBUTED_TIMEOUT:-900}" \
NCCL_TIMEOUT="${NCCL_TIMEOUT:-900}" \
INTERACTION_STRATEGY="$INTERACTION_STRATEGY" \
"$TRAIN_WRAPPER"

if [[ "${DRY_RUN:-0}" == "1" || "${DRY_RUN:-0}" == "true" || "${DRY_RUN:-0}" == "TRUE" ]]; then
  echo "[probe-984] dry run validated; no training was executed"
  exit 0
fi

echo "[probe-984] passed data window $PROBE_DATA_START_STEP-$PROBE_DATA_END_STEP"
