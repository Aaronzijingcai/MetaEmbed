#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
TRAIN_WRAPPER="$SCRIPT_DIR/run_train.sh"

INTERACTION_STRATEGY=${INTERACTION_STRATEGY:-adaptive}
PROBE_DATA_START_STEP=${PROBE_DATA_START_STEP:-57}
PROBE_STEPS=${PROBE_STEPS:-1}
PROBE_SUFFIX=${PROBE_SUFFIX:-deep_audit_data${PROBE_DATA_START_STEP}_${PROBE_STEPS}step_$(date +%Y%m%d_%H%M%S)}

if [[ "$PROBE_DATA_START_STEP" -lt 0 || "$PROBE_STEPS" -lt 1 ]]; then
  echo "[deep-audit] PROBE_DATA_START_STEP must be >= 0 and PROBE_STEPS >= 1" >&2
  exit 2
fi

case "$INTERACTION_STRATEGY" in
  adaptive|adaptive_bidirectional_topk48_mean)
    RUN_PREFIX=rhc_mmeb_vidore_bi_topk48_adaptive_mean_from_base
    ;;
  q2d|q2d_topk48_mean)
    RUN_PREFIX=rhc_mmeb_vidore_q2d_topk48_mean_from_base
    ;;
  *)
    echo "[deep-audit] expected adaptive or q2d, got $INTERACTION_STRATEGY" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_PREFIX}_${PROBE_SUFFIX}"
RUN_DIR="$PROJECT_DIR/experiments/2026-07-08/runs/$RUN_NAME"
DEBUG_STEPS="0-$((PROBE_STEPS - 1))"

echo "[deep-audit] strategy=$INTERACTION_STRATEGY data_start=$PROBE_DATA_START_STEP steps=$PROBE_STEPS"
echo "[deep-audit] run=$RUN_NAME"

RUN_SUFFIX="$PROBE_SUFFIX" \
MAX_STEPS=60000 \
SAVE_STEPS="$PROBE_STEPS" \
STOP_AFTER_STEP="$PROBE_STEPS" \
SKIP_EVAL=1 \
RESUME_CKPT="" \
WARM_START_ADAPTER_PATH="" \
MURE_PROBE_DATA_START_STEP="$PROBE_DATA_START_STEP" \
CONTRASTIVE_DEBUG_STEPS="$DEBUG_STEPS" \
MURE_DEEP_AUDIT=1 \
MURE_DEEP_AUDIT_STEPS="$DEBUG_STEPS" \
INTERACTION_STRATEGY="$INTERACTION_STRATEGY" \
"$TRAIN_WRAPPER"

if [[ "${DRY_RUN:-0}" == "1" || "${DRY_RUN:-0}" == "true" || "${DRY_RUN:-0}" == "TRUE" ]]; then
  exit 0
fi

python3 "$SCRIPT_DIR/analyze_deep_audit.py" \
  "$RUN_DIR/debug/contrastive" \
  --expected-ranks 8 \
  --expected-steps "$PROBE_STEPS" \
  | tee "$RUN_DIR/debug/deep_audit_summary.json"
