#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
EXP_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)
TRAIN_SH="$EXP_DIR/MMEB全量/run_train_full.sh"
EVAL_SH="$SCRIPT_DIR/run_eval_vidorev2_worst10.sh"
RUN_ROOT=${RUN_ROOT:-$SCRIPT_DIR/runs}

export PYTHONPATH="$SCRIPT_DIR:$PROJECT_DIR/vendor:$(cd "$PROJECT_DIR/.." && pwd):${PYTHONPATH:-}"
if [[ -d /opt/conda/bin ]]; then
  export PATH="/opt/conda/bin:$PATH"
fi

export CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}
export NUM_GPUS=${NUM_GPUS:-8}
export WANDB_MODE=${WANDB_MODE:-offline}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MODEL_PATH=${MODEL_PATH:-$PROJECT_DIR/models/colqwen2.5-base}
SUBSET_CONFIG=${SUBSET_CONFIG:-$PROJECT_DIR/configs/train/moca_data_ratios_v3_full.yaml}
MAX_STEPS=${MAX_STEPS:-1000}
SAVE_STEPS=${SAVE_STEPS:-1000}
LEARNING_RATE=${LEARNING_RATE:-2e-4}
TRAIN_BSZ=${TRAIN_BSZ:-12}
EVAL_BSZ=${EVAL_BSZ:-4}
INTERLEAVED_BSZ=${INTERLEAVED_BSZ:-12}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
DOC_CHUNK_SIZE=${DOC_CHUNK_SIZE:-128}
QUERY_CHUNK_SIZE=${QUERY_CHUNK_SIZE:-512}

EVAL_BATCH_QUERY=${EVAL_BATCH_QUERY:-32}
EVAL_BATCH_PASSAGE=${EVAL_BATCH_PASSAGE:-32}
EVAL_BATCH_SCORE=${EVAL_BATCH_SCORE:-128}
EVAL_NUM_WORKERS=${EVAL_NUM_WORKERS:-0}

run_one() {
  local run_name=$1
  local scorer=$2
  local interaction_mode=$3
  local bi_lambda=$4
  local run_dir="$RUN_ROOT/$run_name"
  local train_log="$run_dir/logs/train_$(date +%Y%m%d_%H%M%S).log"
  local ckpt="$run_dir/checkpoint-$MAX_STEPS"

  mkdir -p "$run_dir/logs"
  echo "[machine_b] $(date '+%F %T') run=$run_name scorer=$scorer mode=$interaction_mode bi_lambda=$bi_lambda"
  echo "[machine_b] MODEL_PATH=$MODEL_PATH SUBSET_CONFIG=$SUBSET_CONFIG"

  if [[ -f "$ckpt/folder_homo.pt" ]]; then
    echo "[machine_b] skip train, checkpoint exists: $ckpt"
  else
    RUN_NAME="$run_name" \
    RUN_DIR="$run_dir" \
    OUTPUT_DIR="$run_dir" \
    LOG_FILE="$train_log" \
    MODEL_PATH="$MODEL_PATH" \
    RESUME_CKPT= \
    WARM_START_ADAPTER_PATH= \
    SUBSET_CONFIG="$SUBSET_CONFIG" \
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
    DOC_CHUNK_SIZE="$DOC_CHUNK_SIZE" \
    QUERY_CHUNK_SIZE="$QUERY_CHUNK_SIZE" \
    BUDGETS="160 160 160" \
    COMPRESS_STAGES=all \
    MARC_ENABLED=0 \
    INTERACTION_LOSS_MODE="$interaction_mode" \
    INTERACTION_BI_LAMBDA="$bi_lambda" \
    INTERACTION_QUERY_TOPK=48 \
    INTERACTION_GLOBAL_WEIGHT=0.0 \
    INTERACTION_FACTORIZED_LOCAL_WEIGHT=1.0 \
    INTERACTION_GLOBAL_AUX_WEIGHT=0.0 \
    "$TRAIN_SH"
  fi

  if [[ ! -f "$ckpt/folder_homo.pt" ]]; then
    echo "[machine_b] missing checkpoint after train: $ckpt" >&2
    exit 1
  fi

  local eval_root="$run_dir/eval/maxsim_vidorev2_worst10"
  local eval_log_dir="$run_dir/logs/maxsim_vidorev2_worst10"
  if [[ -f "$eval_root/mmeb_worst10/$scorer/mmeb_full_summary.json" && -f "$eval_root/vidore_v2/$scorer/vidore_v2.json" ]]; then
    echo "[machine_b] skip eval, outputs exist: $eval_root"
  else
    echo "[machine_b] START eval run=$run_name scorer=$scorer"
    OUT_DIR="$eval_root" \
    LOG_DIR="$eval_log_dir" \
    SCORERS="$scorer" \
    CUDA_DEVICE_LIST="$CUDA_DEVICE_LIST" \
    NUM_GPUS="$NUM_GPUS" \
    BATCH_QUERY="$EVAL_BATCH_QUERY" \
    BATCH_PASSAGE="$EVAL_BATCH_PASSAGE" \
    BATCH_SCORE="$EVAL_BATCH_SCORE" \
    NUM_WORKERS="$EVAL_NUM_WORKERS" \
    "$EVAL_SH" "$ckpt"
  fi
}

echo "[machine_b] queue started at $(date '+%F %T')"
echo "[machine_b] from base only: RESUME_CKPT and WARM_START_ADAPTER_PATH are intentionally empty"

run_one vidore_mmeb_bi_mean_lam07_s1k_from_base bi_mean_lam07 bi_mean 0.7
run_one vidore_mmeb_bi_adaptive_s1k_from_base bi_adaptive_lam08 bi_adaptive 0.8

echo "[machine_b] queue finished at $(date '+%F %T')"
