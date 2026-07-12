#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)
MMEB_DIR="$SCRIPT_DIR/../MMEB全量"

BASE_CHECKPOINT=${BASE_CHECKPOINT:-$MMEB_DIR/runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000}
DIAG_NAME=${DIAG_NAME:-vqa_hard}
CONTINUE_STEPS=${CONTINUE_STEPS:-500}
BASE_STEP=${BASE_STEP:-}
SUBSET_CONFIG=${SUBSET_CONFIG:-$SCRIPT_DIR/configs/train_vqa_hard.yaml}
RUN_NAME=${RUN_NAME:-taskcurr_${DIAG_NAME}_from_sym160_s${CONTINUE_STEPS}}
RUN_DIR=${RUN_DIR:-$SCRIPT_DIR/runs/$RUN_NAME}

if [[ ! -d "$BASE_CHECKPOINT" ]]; then
  echo "BASE_CHECKPOINT not found: $BASE_CHECKPOINT" >&2
  exit 2
fi
if [[ ! -f "$SUBSET_CONFIG" ]]; then
  echo "SUBSET_CONFIG not found: $SUBSET_CONFIG" >&2
  exit 2
fi
if [[ ! -x "$MMEB_DIR/run_train_full.sh" ]]; then
  echo "Missing MMEB train launcher: $MMEB_DIR/run_train_full.sh" >&2
  exit 2
fi

if [[ -z "$BASE_STEP" ]]; then
  ckpt_name=$(basename "$BASE_CHECKPOINT")
  BASE_STEP=${ckpt_name#checkpoint-}
fi
if ! [[ "$BASE_STEP" =~ ^[0-9]+$ ]]; then
  echo "Cannot infer numeric BASE_STEP from BASE_CHECKPOINT=$BASE_CHECKPOINT. Set BASE_STEP explicitly." >&2
  exit 2
fi

TARGET_MAX_STEPS=$((BASE_STEP + CONTINUE_STEPS))
SAVE_STEPS=${SAVE_STEPS:-$CONTINUE_STEPS}
if [[ "$SAVE_STEPS" -le 0 ]]; then
  SAVE_STEPS=$CONTINUE_STEPS
fi

mkdir -p "$RUN_DIR/logs"

echo "[taskcurr] BASE_CHECKPOINT=$BASE_CHECKPOINT"
echo "[taskcurr] SUBSET_CONFIG=$SUBSET_CONFIG"
echo "[taskcurr] RUN_DIR=$RUN_DIR"
echo "[taskcurr] BASE_STEP=$BASE_STEP CONTINUE_STEPS=$CONTINUE_STEPS TARGET_MAX_STEPS=$TARGET_MAX_STEPS"

cd "$MMEB_DIR"

RUN_NAME="$RUN_NAME" \
RUN_DIR="$RUN_DIR" \
OUTPUT_DIR="$RUN_DIR" \
SUBSET_CONFIG="$SUBSET_CONFIG" \
RESUME_CKPT="$BASE_CHECKPOINT" \
IGNORE_DATA_SKIP=1 \
MAX_STEPS="$TARGET_MAX_STEPS" \
SAVE_STEPS="$SAVE_STEPS" \
RUN_EVAL=0 \
BUDGETS="${BUDGETS:-160 160 160}" \
TRAIN_BSZ="${TRAIN_BSZ:-4}" \
EVAL_BSZ="${EVAL_BSZ:-4}" \
INTERLEAVED_BSZ="${INTERLEAVED_BSZ:-4}" \
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}" \
DOC_CHUNK_SIZE="${DOC_CHUNK_SIZE:-128}" \
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-512}" \
NUM_GPUS="${NUM_GPUS:-8}" \
CUDA_DEVICE_LIST="${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}" \
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}" \
bash run_train_full.sh
