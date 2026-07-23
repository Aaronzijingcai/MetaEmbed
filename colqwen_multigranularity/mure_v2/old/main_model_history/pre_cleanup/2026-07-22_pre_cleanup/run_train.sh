#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
LEGACY_MAIN="$PROJECT_DIR/experiments/2026-07-08/run_full_main_models.sh"

INTERACTION_STRATEGY=${INTERACTION_STRATEGY:-q2d_topk48_mean}

case "$INTERACTION_STRATEGY" in
  standard_directed_maxsim|standard|maxsim)
    export RUNS=standard
    ;;
  q2d_topk48_mean|q2d)
    export RUNS=q2d
    ;;
  adaptive_bidirectional_topk48_mean|adaptive)
    export RUNS=adaptive
    ;;
  all)
    export RUNS=all
    ;;
  *)
    echo "[main_model] unknown INTERACTION_STRATEGY=$INTERACTION_STRATEGY" >&2
    echo "[main_model] expected: standard_directed_maxsim, q2d_topk48_mean, adaptive_bidirectional_topk48_mean, all" >&2
    exit 2
    ;;
esac

export MAX_STEPS=${MAX_STEPS:-60000}
export SAVE_STEPS=${SAVE_STEPS:-1000}
export LOGGING_STEPS=${LOGGING_STEPS:-10}
export LEARNING_RATE=${LEARNING_RATE:-1e-4}
export LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-linear}
export TRAIN_BSZ=${TRAIN_BSZ:-8}
export INTERLEAVED_BSZ=${INTERLEAVED_BSZ:-8}
export BUDGETS=${BUDGETS:-"128 128 128"}
export GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
export TOPK=${TOPK:-48}
export ADAPTIVE_LAMBDA=${ADAPTIVE_LAMBDA:-0.8}
export QUERY_CHUNK_SIZE=${QUERY_CHUNK_SIZE:-64}
export DOC_CHUNK_SIZE=${DOC_CHUNK_SIZE:-128}
export DDP_FIND_UNUSED_PARAMETERS=${DDP_FIND_UNUSED_PARAMETERS:-1}
export MURE_GATHER_WITH_GRAD_MODE=${MURE_GATHER_WITH_GRAD_MODE:-torch}
export CONTRASTIVE_DEBUG_STEPS=${CONTRASTIVE_DEBUG_STEPS:-978-990}
export STOP_AFTER_STEP=${STOP_AFTER_STEP:-0}

if [[ ! -f "$LEGACY_MAIN" ]]; then
  echo "[main_model] missing legacy launcher: $LEGACY_MAIN" >&2
  exit 2
fi

echo "[main_model] RHC train wrapper"
echo "[main_model] interaction_strategy=$INTERACTION_STRATEGY RUNS=$RUNS"
echo "[main_model] delegating to $LEGACY_MAIN"
exec "$LEGACY_MAIN"
