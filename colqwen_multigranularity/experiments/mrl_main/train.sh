#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
REPO_ROOT=$(cd "$PROJECT_DIR/.." && pwd)

export PYTHONPATH="$PROJECT_DIR/vendor:$REPO_ROOT:${PYTHONPATH:-}"
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-$PROJECT_DIR/runs/wandb}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export MURE_CACHE_ROOT=${MURE_CACHE_ROOT:-$PROJECT_DIR/.cache}
export HF_HOME=${HF_HOME:-$MURE_CACHE_ROOT/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
export TMPDIR=${TMPDIR:-$MURE_CACHE_ROOT/tmp}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export TORCH_NCCL_DESYNC_DEBUG=${TORCH_NCCL_DESYNC_DEBUG:-1}
export TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}
export DATASET_NUM_PROC=${DATASET_NUM_PROC:-1}
export DATASET_SHUFFLE_BUFFER=${DATASET_SHUFFLE_BUFFER:-1024}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1800}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}
mkdir -p "$PROJECT_DIR/runs/logs" "$WANDB_DIR" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TMPDIR"

if [[ -d /opt/conda/bin ]]; then
  export PATH="/opt/conda/bin:$PATH"
fi
ACCELERATE_BIN=${ACCELERATE_BIN:-accelerate}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-0}
CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}
NUM_GPUS=${NUM_GPUS:-8}
MAX_STEPS=${MAX_STEPS:-4000}
SAVE_STEPS=${SAVE_STEPS:-500}
TRAIN_BSZ=${TRAIN_BSZ:-4}
EVAL_BSZ=${EVAL_BSZ:-4}
INTERLEAVED_BSZ=${INTERLEAVED_BSZ:-4}
DOC_CHUNK_SIZE=${DOC_CHUNK_SIZE:-128}
QUERY_CHUNK_SIZE=${QUERY_CHUNK_SIZE:-512}
MAX_NUM_VISUAL_TOKENS=${MAX_NUM_VISUAL_TOKENS:-1024}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_DIR/runs/mrl_main_full_lora}
EVAL_VIDORE_V1_CONFIG=${EVAL_VIDORE_V1_CONFIG:-$PROJECT_DIR/configs/eval/test_data_vidore_beir.yaml}
EVAL_VIDORE_V2_CONFIG=${EVAL_VIDORE_V2_CONFIG:-$PROJECT_DIR/configs/eval/test_data_mast_v2.yaml}
EVAL_MMEB_CONFIG=${EVAL_MMEB_CONFIG:-$PROJECT_DIR/configs/eval/test_data_mast_mmeb_v3.yaml}
RUN_EVAL=${RUN_EVAL:-0}
RESUME_CKPT=${RESUME_CKPT:-}
LOG_FILE=${LOG_FILE:-$PROJECT_DIR/runs/logs/mrl_main_full_lora_$(date +%Y%m%d_%H%M%S).log}

EXTRA_ARGS=()
if [[ "$RUN_EVAL" == "1" || "$RUN_EVAL" == "true" || "$RUN_EVAL" == "TRUE" ]]; then
  EXTRA_ARGS+=(--run-eval)
fi
if [[ -n "$RESUME_CKPT" ]]; then
  EXTRA_ARGS+=(--resume-from-checkpoint "$RESUME_CKPT")
fi

{
  echo "[launcher] $(date +%Y-%m-%d\ %H:%M:%S) starting training"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_LIST" PYTHONUNBUFFERED=1 "$ACCELERATE_BIN" launch   --num_machines 1   --num_processes "$NUM_GPUS"   --main_process_port "$MAIN_PROCESS_PORT"   --mixed_precision bf16   "$PROJECT_DIR/train.py"   --model-name-or-path "$PROJECT_DIR/models/colqwen2.5-base"   --processor-name-or-path "$PROJECT_DIR/models/colqwen2.5-base"   --output-dir "$OUTPUT_DIR"   --subset-config "$PROJECT_DIR/configs/train/moca_data_ratios_v3_nommE5.yaml"   --eval-vidore-v1-config "$EVAL_VIDORE_V1_CONFIG"   --eval-vidore-v2-config "$EVAL_VIDORE_V2_CONFIG"   --eval-mmeb-config "$EVAL_MMEB_CONFIG"   --granularities 1 2 4   --max-steps "$MAX_STEPS"   --save-steps "$SAVE_STEPS"   --logging-steps 10   --learning-rate 1e-4   --lr-scheduler-type linear   --warmup-ratio 0.03   --per-device-train-batch-size "$TRAIN_BSZ"   --per-device-eval-batch-size "$EVAL_BSZ"   --gradient-accumulation-steps 1   --interleaved-batch-size "$INTERLEAVED_BSZ"   --dataloader-num-workers 0   --num-negative 1   --num-shards 128   --doc-chunk-size "$DOC_CHUNK_SIZE"   --query-chunk-size "$QUERY_CHUNK_SIZE"   --max-num-visual-tokens "$MAX_NUM_VISUAL_TOKENS"   --attn-implementation flash_attention_2   --use-peft   --ddp-find-unused-parameters   "${EXTRA_ARGS[@]}"
  status=$?
  echo "[launcher] $(date +%Y-%m-%d\ %H:%M:%S) training exited with status ${status}"
  exit ${status}
} >> "$LOG_FILE" 2>&1
