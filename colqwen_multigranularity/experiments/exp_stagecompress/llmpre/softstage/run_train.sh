#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../../../.." && pwd)
REPO_ROOT=$(cd "$PROJECT_DIR/.." && pwd)

export PYTHONPATH="$PROJECT_DIR/vendor:$REPO_ROOT:${PYTHONPATH:-}"
if [[ -d /opt/conda/bin ]]; then
  export PATH="/opt/conda/bin:$PATH"
fi

ACCELERATE_BIN=${ACCELERATE_BIN:-accelerate}
CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}
NUM_GPUS=${NUM_GPUS:-8}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-0}
MAX_STEPS=${MAX_STEPS:-4000}
SAVE_STEPS=${SAVE_STEPS:-500}
TRAIN_BSZ=${TRAIN_BSZ:-4}
EVAL_BSZ=${EVAL_BSZ:-4}
INTERLEAVED_BSZ=${INTERLEAVED_BSZ:-4}
USE_PEFT=${USE_PEFT:-1}
DDP_FIND_UNUSED_PARAMETERS=${DDP_FIND_UNUSED_PARAMETERS:-1}
SOFTSTAGE_KEEP_RATIOS=${SOFTSTAGE_KEEP_RATIOS:-1.0,0.5,0.25}
SOFTSTAGE_TEMPERATURE=${SOFTSTAGE_TEMPERATURE:-0.1}
SOFTSTAGE_MIN_MASK_VALUE=${SOFTSTAGE_MIN_MASK_VALUE:-0.0}
RUN_NAME=${RUN_NAME:-softstage_8gpu_nommE5_textquery_focus_4k}
RUN_DIR=${RUN_DIR:-$PROJECT_DIR/experiments/exp_stagecompress/llmpre/softstage/runs/$RUN_NAME}
MODEL_PATH=${MODEL_PATH:-$PROJECT_DIR/models/colqwen2.5-base}
OUTPUT_DIR=${OUTPUT_DIR:-$RUN_DIR}
LOG_FILE=${LOG_FILE:-$RUN_DIR/logs/train_$(date +%Y%m%d_%H%M%S).log}
RESUME_CKPT=${RESUME_CKPT:-}
SUBSET_CONFIG=${SUBSET_CONFIG:-$PROJECT_DIR/configs/train/moca_data_ratios_v3_nommE5.yaml}

export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-$RUN_DIR/wandb}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export MURE_CACHE_ROOT=${MURE_CACHE_ROOT:-$PROJECT_DIR/.cache}
export HF_HOME=${HF_HOME:-$MURE_CACHE_ROOT/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
export TMPDIR=${TMPDIR:-$MURE_CACHE_ROOT/tmp}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export DATASET_NUM_PROC=${DATASET_NUM_PROC:-1}
export DATASET_SHUFFLE_BUFFER=${DATASET_SHUFFLE_BUFFER:-1024}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-7200}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}
mkdir -p "$OUTPUT_DIR" "$(dirname "$LOG_FILE")" "$WANDB_DIR" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TMPDIR"

EXTRA_ARGS=(
  --softstage-keep-ratios "$SOFTSTAGE_KEEP_RATIOS"
  --softstage-temperature "$SOFTSTAGE_TEMPERATURE"
  --softstage-min-mask-value "$SOFTSTAGE_MIN_MASK_VALUE"
)
if [[ "$USE_PEFT" == "1" || "$USE_PEFT" == "true" || "$USE_PEFT" == "TRUE" ]]; then
  EXTRA_ARGS+=(--use-peft)
fi
if [[ "$DDP_FIND_UNUSED_PARAMETERS" == "1" || "$DDP_FIND_UNUSED_PARAMETERS" == "true" || "$DDP_FIND_UNUSED_PARAMETERS" == "TRUE" ]]; then
  EXTRA_ARGS+=(--ddp-find-unused-parameters)
fi
if [[ -n "$RESUME_CKPT" ]]; then
  EXTRA_ARGS+=(--resume-from-checkpoint "$RESUME_CKPT")
fi

{
  echo "[launcher] $(date +%Y-%m-%d\ %H:%M:%S) starting SoftStage training"
  echo "[launcher] OUTPUT_DIR=$OUTPUT_DIR"
  echo "[launcher] LOG_FILE=$LOG_FILE"
  echo "[launcher] CUDA_DEVICE_LIST=$CUDA_DEVICE_LIST NUM_GPUS=$NUM_GPUS MAIN_PROCESS_PORT=$MAIN_PROCESS_PORT"
  echo "[launcher] protocol=MRL_Main no_learnable_global_tokens=1"
  echo "[launcher] SOFTSTAGE_KEEP_RATIOS=$SOFTSTAGE_KEEP_RATIOS SOFTSTAGE_TEMPERATURE=$SOFTSTAGE_TEMPERATURE SOFTSTAGE_MIN_MASK_VALUE=$SOFTSTAGE_MIN_MASK_VALUE"
  echo "[launcher] MAX_STEPS=$MAX_STEPS SAVE_STEPS=$SAVE_STEPS"
} >> "$LOG_FILE"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_LIST" \
PYTHONUNBUFFERED=1 \
"$ACCELERATE_BIN" launch \
  --num_machines 1 \
  --num_processes "$NUM_GPUS" \
  --main_process_port "$MAIN_PROCESS_PORT" \
  --mixed_precision bf16 \
  -m colqwen_multigranularity.experiments.exp_stagecompress.llmpre.softstage.train_softstage \
  --model-name-or-path "$MODEL_PATH" \
  --processor-name-or-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --subset-config "$SUBSET_CONFIG" \
  --eval-vidore-v1-config "$PROJECT_DIR/configs/eval/test_data_vidore_v1_v2_mmeb_textquery_focus.yaml" \
  --eval-vidore-v2-config "$PROJECT_DIR/configs/eval/test_data_vidore_v1_v2_mmeb_textquery_focus.yaml" \
  --eval-mmeb-config "$PROJECT_DIR/configs/eval/test_data_vidore_v1_v2_mmeb_textquery_focus.yaml" \
  --granularities 1 2 4 \
  --max-steps "$MAX_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --logging-steps 10 \
  --learning-rate 1e-4 \
  --lr-scheduler-type linear \
  --warmup-ratio 0.03 \
  --per-device-train-batch-size "$TRAIN_BSZ" \
  --per-device-eval-batch-size "$EVAL_BSZ" \
  --gradient-accumulation-steps 1 \
  --interleaved-batch-size "$INTERLEAVED_BSZ" \
  --dataloader-num-workers 0 \
  --num-negative 1 \
  --num-shards 128 \
  --attn-implementation flash_attention_2 \
  --query-augmentation-repeats 0 \
  --document-augmentation-repeats 0 \
  --no-normalize-scores \
  "${EXTRA_ARGS[@]}" \
  >> "$LOG_FILE" 2>&1
