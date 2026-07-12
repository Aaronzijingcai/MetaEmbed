#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
EXP_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
TRAIN_SH="$EXP_DIR/2026-07-01/MMEB全量/run_train_full.sh"
EVAL_SH="$SCRIPT_DIR/eval_full_main_models.sh"

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}
export NUM_GPUS=${NUM_GPUS:-8}

DEFAULT_CACHE_ROOT="$PROJECT_DIR/.cache"
DEFAULT_HF_DATASETS_CACHE="$DEFAULT_CACHE_ROOT/huggingface/datasets"
if [[ -d /MURE-V2/env ]]; then
  DEFAULT_CACHE_ROOT="/MURE-V2/env/mure_cache/colqwen_multigranularity"
  DEFAULT_HF_DATASETS_CACHE="/MURE-V2/env/hf_datasets_cache"
fi
export MURE_CACHE_ROOT=${MURE_CACHE_ROOT:-$DEFAULT_CACHE_ROOT}
export HF_HOME=${HF_HOME:-$MURE_CACHE_ROOT/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$DEFAULT_HF_DATASETS_CACHE}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
export TMPDIR=${TMPDIR:-$MURE_CACHE_ROOT/tmp}
mkdir -p "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TMPDIR"

MODEL_PATH=${MODEL_PATH:-$PROJECT_DIR/models/colqwen2.5-base}
SUBSET_CONFIG=${SUBSET_CONFIG:-$PROJECT_DIR/configs/train/moca_data_ratios_v3_full.yaml}

MAX_STEPS=${MAX_STEPS:-90000}
SAVE_STEPS=${SAVE_STEPS:-3000}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
TRAIN_BSZ=${TRAIN_BSZ:-10}
INTERLEAVED_BSZ=${INTERLEAVED_BSZ:-10}
EVAL_BSZ=${EVAL_BSZ:-4}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
TOPK=${TOPK:-48}
ADAPTIVE_LAMBDA=${ADAPTIVE_LAMBDA:-0.8}
ADAPTIVE_RATIO=${ADAPTIVE_RATIO:-1.5}
MAIN_PROCESS_PORT_BASE=${MAIN_PROCESS_PORT_BASE:-29780}
RUNS=${RUNS:-all}
DRY_RUN=${DRY_RUN:-0}
RUN_SUFFIX=${RUN_SUFFIX:-}
SKIP_EVAL=${SKIP_EVAL:-0}
RESUME_CKPT=${RESUME_CKPT:-}
WARM_START_ADAPTER_PATH=${WARM_START_ADAPTER_PATH:-}

run_cmd() {
  if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "TRUE" ]]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

should_run() {
  local name=$1
  [[ "$RUNS" == "all" || "$RUNS" == "$name" || "$RUNS" == *",$name,"* || "$RUNS" == "$name,"* || "$RUNS" == *",$name" ]]
}

run_one() {
  local run_key=$1
  local run_name=$2
  local train_mode=$3
  local bi_lambda=$4
  local eval_scorer=$5
  local port=$6
  if [[ -n "$RUN_SUFFIX" ]]; then
    run_name="${run_name}_${RUN_SUFFIX}"
  fi

  local run_dir="$SCRIPT_DIR/runs/$run_name"
  local train_log="$run_dir/logs/train_$(date +%Y%m%d_%H%M%S).log"
  mkdir -p "$run_dir/logs"

  if [[ -d "$run_dir/checkpoint-$MAX_STEPS" ]]; then
    echo "[2026-07-08] skip existing final checkpoint: $run_dir/checkpoint-$MAX_STEPS"
  else
    echo "[2026-07-08] START train key=$run_key run=$run_name mode=$train_mode scorer=$eval_scorer"
    run_cmd env \
      RUN_NAME="$run_name" \
      RUN_DIR="$run_dir" \
      OUTPUT_DIR="$run_dir" \
      LOG_FILE="$train_log" \
      MODEL_PATH="$MODEL_PATH" \
      RESUME_CKPT="$RESUME_CKPT" \
      WARM_START_ADAPTER_PATH="$WARM_START_ADAPTER_PATH" \
      SUBSET_CONFIG="$SUBSET_CONFIG" \
      CUDA_DEVICE_LIST="$CUDA_DEVICE_LIST" \
      NUM_GPUS="$NUM_GPUS" \
      MAIN_PROCESS_PORT="$port" \
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
      USE_PEFT=1 \
      USE_LIGER_KERNEL="${USE_LIGER_KERNEL:-0}" \
      GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}" \
      DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-0}" \
      DO_GATHER="${DO_GATHER:-1}" \
      DO_PADDING="${DO_PADDING:-1}" \
      INTERACTION_LOSS_MODE="$train_mode" \
      INTERACTION_BI_LAMBDA="$bi_lambda" \
      INTERACTION_QUERY_TOPK="$TOPK" \
      INTERACTION_ADAPTIVE_RATIO="$ADAPTIVE_RATIO" \
      INTERACTION_GLOBAL_WEIGHT=0.0 \
      INTERACTION_FACTORIZED_LOCAL_WEIGHT=1.0 \
      INTERACTION_GLOBAL_AUX_WEIGHT=0.0 \
      "$TRAIN_SH"
  fi

  local ckpt="$run_dir/checkpoint-$MAX_STEPS"
  if [[ "$DRY_RUN" != "1" && "$DRY_RUN" != "true" && "$DRY_RUN" != "TRUE" && ! -d "$ckpt" ]]; then
    echo "[2026-07-08] missing final checkpoint after train: $ckpt" >&2
    exit 1
  fi

  if [[ "$SKIP_EVAL" == "1" || "$SKIP_EVAL" == "true" || "$SKIP_EVAL" == "TRUE" ]]; then
    echo "[2026-07-08] SKIP eval run=$run_name scorer=$eval_scorer"
  else
    echo "[2026-07-08] START eval run=$run_name scorer=$eval_scorer"
    run_cmd env \
      RUN_NAME="$run_name" \
      CHECKPOINT="$ckpt" \
      SCORERS="$eval_scorer" \
      CUDA_DEVICE_LIST="$CUDA_DEVICE_LIST" \
      NUM_GPUS="$NUM_GPUS" \
      "$EVAL_SH"
  fi
}

if [[ ! -f "$TRAIN_SH" ]]; then
  echo "[2026-07-08] missing train script: $TRAIN_SH" >&2
  exit 2
fi
if [[ ! -f "$SUBSET_CONFIG" ]]; then
  echo "[2026-07-08] missing subset config: $SUBSET_CONFIG" >&2
  exit 2
fi

mkdir -p "$SCRIPT_DIR/runs"

echo "[2026-07-08] queue started at $(date +%Y-%m-%d\ %H:%M:%S)"
echo "[2026-07-08] train data=$SUBSET_CONFIG"
echo "[2026-07-08] model=$MODEL_PATH"
echo "[2026-07-08] cache root=$MURE_CACHE_ROOT hf_datasets=$HF_DATASETS_CACHE tmp=$TMPDIR"
echo "[2026-07-08] max_steps=$MAX_STEPS save_steps=$SAVE_STEPS lr=$LEARNING_RATE train_bsz=$TRAIN_BSZ interleaved_bsz=$INTERLEAVED_BSZ grad_accum=$GRAD_ACCUM_STEPS topk=$TOPK run_suffix=$RUN_SUFFIX skip_eval=$SKIP_EVAL"
if [[ -n "$RESUME_CKPT" ]]; then
  echo "[2026-07-08] resume same run from checkpoint: RESUME_CKPT=$RESUME_CKPT"
else
  echo "[2026-07-08] from base model; RESUME_CKPT is empty"
fi
if [[ -n "$WARM_START_ADAPTER_PATH" ]]; then
  echo "[2026-07-08] WARNING: WARM_START_ADAPTER_PATH is set: $WARM_START_ADAPTER_PATH"
else
  echo "[2026-07-08] WARM_START_ADAPTER_PATH is intentionally empty"
fi

if should_run q2d; then
  run_one \
    q2d \
    full_mmeb_vidore_q2d_topk48_mean_from_base \
    q2d_query_topk \
    0.5 \
    q2d_query_topk48 \
    "$MAIN_PROCESS_PORT_BASE"
fi

if should_run adaptive; then
  run_one \
    adaptive \
    full_mmeb_vidore_bi_topk48_adaptive_mean_from_base \
    bi_query_topk_adaptive \
    "$ADAPTIVE_LAMBDA" \
    bi_topk_mean48_adaptive_lam08 \
    "$((MAIN_PROCESS_PORT_BASE + 1))"
fi

echo "[2026-07-08] queue finished at $(date +%Y-%m-%d\ %H:%M:%S)"
