#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)

CHECKPOINT=${1:-${CHECKPOINT:-}}
if [[ -z "$CHECKPOINT" ]]; then
  echo "Usage: CHECKPOINT=/path/to/checkpoint bash $0" >&2
  exit 2
fi
if [[ ! -d "$CHECKPOINT" ]]; then
  echo "checkpoint directory not found: $CHECKPOINT" >&2
  exit 2
fi

RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
NUM_GPUS=${NUM_GPUS:-8}
CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}
BUDGETS=${BUDGETS:-160 160 160}
BATCH_QUERY=${BATCH_QUERY:-8}
BATCH_PASSAGE=${BATCH_PASSAGE:-8}
BATCH_SCORE=${BATCH_SCORE:-32}
NUM_WORKERS=${NUM_WORKERS:-0}
EVAL_MAX_QUERIES=${EVAL_MAX_QUERIES:-16}
EVAL_MAX_CORPUS=${EVAL_MAX_CORPUS:-96}
QUERY_AUGMENTATION_REPEATS=${QUERY_AUGMENTATION_REPEATS:-10}
TRAIN_LOG=${TRAIN_LOG:-}
MIN_V2=${MIN_V2:-0.45}
MIN_MMEB=${MIN_MMEB:-0.60}
MIN_STAGE_COUNT=${MIN_STAGE_COUNT:-1.0}
MIN_AUX_RATIO=${MIN_AUX_RATIO:-0.002}
MAX_AUX_RATIO=${MAX_AUX_RATIO:-0.08}

RUN_DIR=$(cd "$(dirname "$CHECKPOINT")" && pwd)
OUT_DIR=${OUT_DIR:-$RUN_DIR/eval/quick_gate_${RUN_TAG}}
LOG_DIR=${LOG_DIR:-$RUN_DIR/logs}
mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "[quick_gate] checkpoint=$CHECKPOINT"
echo "[quick_gate] out_dir=$OUT_DIR"

env \
  EVAL_MODE=smoke \
  NUM_GPUS="$NUM_GPUS" \
  CUDA_DEVICE_LIST="$CUDA_DEVICE_LIST" \
  BUDGETS="$BUDGETS" \
  BATCH_QUERY="$BATCH_QUERY" \
  BATCH_PASSAGE="$BATCH_PASSAGE" \
  BATCH_SCORE="$BATCH_SCORE" \
  NUM_WORKERS="$NUM_WORKERS" \
  EVAL_MAX_QUERIES="$EVAL_MAX_QUERIES" \
  EVAL_MAX_CORPUS="$EVAL_MAX_CORPUS" \
  QUERY_AUGMENTATION_REPEATS="$QUERY_AUGMENTATION_REPEATS" \
  OUT_DIR="$OUT_DIR" \
  LOG_DIR="$LOG_DIR" \
  LOG_FILE="$LOG_DIR/quick_gate_eval_${RUN_TAG}.log" \
  bash "$PROJECT_DIR/experiments/exp_stagecompress/folder_homo/eval_3sets.sh" "$CHECKPOINT"

SUMMARY_ARGS=(
  --checkpoint "$CHECKPOINT"
  --eval-dir "$OUT_DIR"
  --output-json "$OUT_DIR/quick_gate_summary.json"
  --output-md "$OUT_DIR/quick_gate_summary.md"
  --min-v2 "$MIN_V2"
  --min-mmeb "$MIN_MMEB"
  --min-stage-count "$MIN_STAGE_COUNT"
  --min-aux-ratio "$MIN_AUX_RATIO"
  --max-aux-ratio "$MAX_AUX_RATIO"
)
if [[ -n "$TRAIN_LOG" ]]; then
  SUMMARY_ARGS+=(--train-log "$TRAIN_LOG")
fi

python3 "$PROJECT_DIR/experiments/exp_stagecompress/analysis/quick_gate.py" \
  "${SUMMARY_ARGS[@]}"

echo "[quick_gate] summary=$OUT_DIR/quick_gate_summary.md"
