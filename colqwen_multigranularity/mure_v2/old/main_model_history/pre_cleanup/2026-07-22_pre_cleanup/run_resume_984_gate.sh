#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
TRAIN_WRAPPER="$SCRIPT_DIR/run_train.sh"
RUN_ROOT="$PROJECT_DIR/experiments/2026-07-08/runs"

INTERACTION_STRATEGY=${INTERACTION_STRATEGY:-q2d_topk48_mean}
GATE_STOP_STEP=${GATE_STOP_STEP:-990}
GATE_SUFFIX=${GATE_SUFFIX:-resume950_gate_$(date +%Y%m%d_%H%M%S)}

case "$INTERACTION_STRATEGY" in
  q2d_topk48_mean|q2d)
    SOURCE_RUN=rhc_mmeb_vidore_q2d_topk48_mean_from_base
    ;;
  adaptive_bidirectional_topk48_mean|adaptive)
    SOURCE_RUN=rhc_mmeb_vidore_bi_topk48_adaptive_mean_from_base
    ;;
  *)
    echo "[resume-984-gate] expected q2d or adaptive, got $INTERACTION_STRATEGY" >&2
    exit 2
    ;;
esac

RESUME_CKPT=${RESUME_CKPT:-$RUN_ROOT/$SOURCE_RUN/checkpoint-950}
for required in trainer_state.json optimizer.pt scheduler.pt folder_homo.pt adapter_model.bin; do
  if [[ ! -f "$RESUME_CKPT/$required" ]]; then
    echo "[resume-984-gate] missing $RESUME_CKPT/$required" >&2
    exit 2
  fi
done

echo "[resume-984-gate] strategy=$INTERACTION_STRATEGY checkpoint=$RESUME_CKPT"
echo "[resume-984-gate] gather=${MURE_GATHER_WITH_GRAD_MODE:-torch} stop=$GATE_STOP_STEP suffix=$GATE_SUFFIX"

RESUME_CKPT="$RESUME_CKPT" \
RUN_SUFFIX="$GATE_SUFFIX" \
MAX_STEPS=60000 \
SAVE_STEPS=950 \
STOP_AFTER_STEP="$GATE_STOP_STEP" \
SKIP_EVAL=1 \
CONTRASTIVE_DEBUG_STEPS="${CONTRASTIVE_DEBUG_STEPS:-978-990}" \
MURE_GATHER_WITH_GRAD_MODE="${MURE_GATHER_WITH_GRAD_MODE:-torch}" \
TORCH_DISTRIBUTED_TIMEOUT="${TORCH_DISTRIBUTED_TIMEOUT:-900}" \
NCCL_TIMEOUT="${NCCL_TIMEOUT:-900}" \
INTERACTION_STRATEGY="$INTERACTION_STRATEGY" \
"$TRAIN_WRAPPER"

if [[ "${DRY_RUN:-0}" == "1" || "${DRY_RUN:-0}" == "true" || "${DRY_RUN:-0}" == "TRUE" ]]; then
  echo "[resume-984-gate] dry run validated; no training was executed"
  exit 0
fi

echo "[resume-984-gate] passed through checkpoint-$GATE_STOP_STEP"
