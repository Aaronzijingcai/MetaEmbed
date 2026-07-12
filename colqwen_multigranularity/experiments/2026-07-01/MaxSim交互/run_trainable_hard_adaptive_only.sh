#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
EXP_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)
TRAIN_SH="$EXP_DIR/MMEB全量/run_train_full.sh"
EVAL_SH="$SCRIPT_DIR/run_eval_vidorev2_worst10.sh"
SUBSET_CONFIG=${SUBSET_CONFIG:-$SCRIPT_DIR/configs/train_vidore_mmeb_hard_core4.yaml}
MODEL_PATH=${MODEL_PATH:-$PROJECT_DIR/models/colqwen2.5-base}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}
export NUM_GPUS=${NUM_GPUS:-8}

RUN_NAME=${RUN_NAME:-vidore_mmeb_bi_topk_mean48_hard_adaptive_s1k_from_base}
TRAIN_MODE=${TRAIN_MODE:-bi_query_topk_hard_adaptive}
EVAL_SCORER=${EVAL_SCORER:-bi_topk_mean48_hard_adaptive}
MAX_STEPS=${MAX_STEPS:-1000}
SAVE_STEPS=${SAVE_STEPS:-1000}
LEARNING_RATE=${LEARNING_RATE:-2e-4}
TRAIN_BSZ=${TRAIN_BSZ:-8}
INTERLEAVED_BSZ=${INTERLEAVED_BSZ:-8}
EVAL_BSZ=${EVAL_BSZ:-4}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
BATCH_QUERY=${BATCH_QUERY:-16}
BATCH_PASSAGE=${BATCH_PASSAGE:-16}
BATCH_SCORE=${BATCH_SCORE:-64}
NUM_WORKERS=${NUM_WORKERS:-0}
TOPK=${TOPK:-48}
BI_LAMBDA=${BI_LAMBDA:-0.5}
ADAPTIVE_RATIO=${ADAPTIVE_RATIO:-1.5}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29673}

RUN_DIR="$SCRIPT_DIR/runs/$RUN_NAME"
TRAIN_LOG="$RUN_DIR/logs/train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$RUN_DIR/logs"

if [[ ! -f "$SUBSET_CONFIG" ]]; then
  echo "[hard_adaptive_only] missing SUBSET_CONFIG=$SUBSET_CONFIG" >&2
  exit 2
fi

echo "[hard_adaptive_only] started at $(date +%Y-%m-%d\ %H:%M:%S)"
echo "[hard_adaptive_only] from base model; RESUME_CKPT and WARM_START_ADAPTER_PATH are intentionally empty"
echo "[hard_adaptive_only] run=$RUN_NAME mode=$TRAIN_MODE scorer=$EVAL_SCORER"
echo "[hard_adaptive_only] train data=$SUBSET_CONFIG"
echo "[hard_adaptive_only] topk=$TOPK lambda=$BI_LAMBDA adaptive_ratio=$ADAPTIVE_RATIO train_bsz=$TRAIN_BSZ interleaved_bsz=$INTERLEAVED_BSZ num_gpus=$NUM_GPUS"

if [[ -d "$RUN_DIR/checkpoint-$SAVE_STEPS" ]]; then
  echo "[hard_adaptive_only] skip existing checkpoint: $RUN_DIR/checkpoint-$SAVE_STEPS"
else
  env \
    RUN_NAME="$RUN_NAME" \
    RUN_DIR="$RUN_DIR" \
    OUTPUT_DIR="$RUN_DIR" \
    LOG_FILE="$TRAIN_LOG" \
    MODEL_PATH="$MODEL_PATH" \
    RESUME_CKPT= \
    WARM_START_ADAPTER_PATH= \
    SUBSET_CONFIG="$SUBSET_CONFIG" \
    CUDA_DEVICE_LIST="$CUDA_DEVICE_LIST" \
    NUM_GPUS="$NUM_GPUS" \
    MAIN_PROCESS_PORT="$MAIN_PROCESS_PORT" \
    MAX_STEPS="$MAX_STEPS" \
    SAVE_STEPS="$SAVE_STEPS" \
    LEARNING_RATE="$LEARNING_RATE" \
    LR_SCHEDULER_TYPE=constant \
    WARMUP_RATIO=0 \
    WARMUP_STEPS=0 \
    TRAIN_BSZ="$TRAIN_BSZ" \
    EVAL_BSZ="$EVAL_BSZ" \
    INTERLEAVED_BSZ="$INTERLEAVED_BSZ" \
    GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS" \
    BUDGETS="160 160 160" \
    COMPRESS_STAGES=all \
    MARC_ENABLED=0 \
    INTERACTION_LOSS_MODE="$TRAIN_MODE" \
    INTERACTION_BI_LAMBDA="$BI_LAMBDA" \
    INTERACTION_QUERY_TOPK="$TOPK" \
    INTERACTION_ADAPTIVE_RATIO="$ADAPTIVE_RATIO" \
    INTERACTION_GLOBAL_WEIGHT=0.0 \
    INTERACTION_FACTORIZED_LOCAL_WEIGHT=1.0 \
    INTERACTION_GLOBAL_AUX_WEIGHT=0.0 \
    "$TRAIN_SH"
fi

CKPT="$RUN_DIR/checkpoint-$SAVE_STEPS"
if [[ ! -d "$CKPT" ]]; then
  echo "[hard_adaptive_only] missing checkpoint after train: $CKPT" >&2
  exit 1
fi

echo "[hard_adaptive_only] START eval scorer=$EVAL_SCORER"
env \
  CHECKPOINT="$CKPT" \
  OUT_DIR="$RUN_DIR/eval/maxsim_vidorev2_worst10_checkpoint-$SAVE_STEPS" \
  LOG_DIR="$RUN_DIR/logs/maxsim_vidorev2_worst10_checkpoint-$SAVE_STEPS" \
  SCORERS="$EVAL_SCORER" \
  RUN_MMEB=1 \
  RUN_VIDORE=1 \
  CUDA_DEVICE_LIST="$CUDA_DEVICE_LIST" \
  NUM_GPUS="$NUM_GPUS" \
  BATCH_QUERY="$BATCH_QUERY" \
  BATCH_PASSAGE="$BATCH_PASSAGE" \
  BATCH_SCORE="$BATCH_SCORE" \
  NUM_WORKERS="$NUM_WORKERS" \
  "$EVAL_SH"

echo "[hard_adaptive_only] finished at $(date +%Y-%m-%d\ %H:%M:%S)"
