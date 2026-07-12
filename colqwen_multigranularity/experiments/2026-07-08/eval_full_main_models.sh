#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
EXP_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
EVAL_SH="$EXP_DIR/2026-07-01/MaxSim交互/run_eval_vidorev2_worst10.sh"

RUN_NAME=${RUN_NAME:?RUN_NAME is required}
CHECKPOINT=${CHECKPOINT:?CHECKPOINT is required}
SCORERS=${SCORERS:?SCORERS is required}

CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}
NUM_GPUS=${NUM_GPUS:-8}
BATCH_QUERY=${BATCH_QUERY:-32}
BATCH_PASSAGE=${BATCH_PASSAGE:-32}
BATCH_SCORE=${BATCH_SCORE:-128}
NUM_WORKERS=${NUM_WORKERS:-0}

RUN_DIR="$SCRIPT_DIR/runs/$RUN_NAME"
OUT_DIR=${OUT_DIR:-$RUN_DIR/eval/maxsim_vidorev2_worst10_$(basename "$CHECKPOINT")}
LOG_DIR=${LOG_DIR:-$RUN_DIR/logs/maxsim_vidorev2_worst10_$(basename "$CHECKPOINT")}

if [[ ! -f "$EVAL_SH" ]]; then
  echo "[2026-07-08 eval] missing eval script: $EVAL_SH" >&2
  exit 2
fi
if [[ ! -d "$CHECKPOINT" ]]; then
  echo "[2026-07-08 eval] missing checkpoint: $CHECKPOINT" >&2
  exit 2
fi

mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "[2026-07-08 eval] run=$RUN_NAME checkpoint=$CHECKPOINT scorers=$SCORERS"
env \
  CHECKPOINT="$CHECKPOINT" \
  OUT_DIR="$OUT_DIR" \
  LOG_DIR="$LOG_DIR" \
  SCORERS="$SCORERS" \
  RUN_MMEB=1 \
  RUN_VIDORE=1 \
  CUDA_DEVICE_LIST="$CUDA_DEVICE_LIST" \
  NUM_GPUS="$NUM_GPUS" \
  BATCH_QUERY="$BATCH_QUERY" \
  BATCH_PASSAGE="$BATCH_PASSAGE" \
  BATCH_SCORE="$BATCH_SCORE" \
  NUM_WORKERS="$NUM_WORKERS" \
  "$EVAL_SH"

