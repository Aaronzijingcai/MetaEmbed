#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)
REPO_ROOT=$(cd "$PROJECT_DIR/.." && pwd)

export PYTHONPATH="$SCRIPT_DIR:$PROJECT_DIR/vendor:$REPO_ROOT:${PYTHONPATH:-}"
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
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
DOC_CHUNK_SIZE=${DOC_CHUNK_SIZE:-128}
QUERY_CHUNK_SIZE=${QUERY_CHUNK_SIZE:-512}
MAX_NUM_VISUAL_TOKENS=${MAX_NUM_VISUAL_TOKENS:-1024}
USE_PEFT=${USE_PEFT:-1}
DDP_FIND_UNUSED_PARAMETERS=${DDP_FIND_UNUSED_PARAMETERS:-1}
IGNORE_DATA_SKIP=${IGNORE_DATA_SKIP:-0}
TRAIN_COMPRESSOR_ONLY=${TRAIN_COMPRESSOR_ONLY:-0}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}
COMPRESS_STAGES=${COMPRESS_STAGES:-all}
IMPORTANCE_MODE=${IMPORTANCE_MODE:-mlp}
IMPORTANCE_BLEND=${IMPORTANCE_BLEND:-1.0}
RUN_NAME=${RUN_NAME:-folder_importance_v1_${IMPORTANCE_MODE}_b160_160_160_4k}
RUN_DIR=${RUN_DIR:-$SCRIPT_DIR/runs/$RUN_NAME}
BUDGETS=(${BUDGETS:-160 160 160})
NOVELTY_WEIGHT=${NOVELTY_WEIGHT:-1.0}
GATE_STRENGTH=${GATE_STRENGTH:-0.25}
FOLDER_ALPHA=${FOLDER_ALPHA:-1.0}
PAGERANK_DAMPING=${PAGERANK_DAMPING:-0.85}
PAGERANK_ITERS=${PAGERANK_ITERS:-8}
MODEL_PATH=${MODEL_PATH:-$PROJECT_DIR/models/colqwen2.5-base}
OUTPUT_DIR=${OUTPUT_DIR:-$RUN_DIR}
LOG_FILE=${LOG_FILE:-$RUN_DIR/logs/train_$(date +%Y%m%d_%H%M%S).log}
RESUME_CKPT=${RESUME_CKPT:-}
WARM_START_ADAPTER_PATH=${WARM_START_ADAPTER_PATH:-}
SUBSET_CONFIG=${SUBSET_CONFIG:-$PROJECT_DIR/configs/train/moca_data_ratios_v3_nommE5.yaml}

if [[ "${#BUDGETS[@]}" -ne 3 ]]; then
  echo "BUDGETS must contain exactly 3 integers, got: ${BUDGETS[*]}" >&2
  exit 2
fi

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
  --importance-enabled
  --importance-compress-stages "$COMPRESS_STAGES"
  --importance-budgets "${BUDGETS[@]}"
  --importance-mode "$IMPORTANCE_MODE"
  --importance-blend "$IMPORTANCE_BLEND"
  --importance-novelty-weight "$NOVELTY_WEIGHT"
  --importance-gate-strength "$GATE_STRENGTH"
  --importance-folder-alpha "$FOLDER_ALPHA"
  --importance-pagerank-damping "$PAGERANK_DAMPING"
  --importance-pagerank-iters "$PAGERANK_ITERS"
)
if [[ "$USE_PEFT" == "1" || "$USE_PEFT" == "true" || "$USE_PEFT" == "TRUE" ]]; then
  EXTRA_ARGS+=(--use-peft)
fi
if [[ "$DDP_FIND_UNUSED_PARAMETERS" == "1" || "$DDP_FIND_UNUSED_PARAMETERS" == "true" || "$DDP_FIND_UNUSED_PARAMETERS" == "TRUE" ]]; then
  EXTRA_ARGS+=(--ddp-find-unused-parameters)
fi
if [[ "$GRADIENT_CHECKPOINTING" == "0" || "$GRADIENT_CHECKPOINTING" == "false" || "$GRADIENT_CHECKPOINTING" == "FALSE" ]]; then
  EXTRA_ARGS+=(--no-gradient-checkpointing)
fi
if [[ "$TRAIN_COMPRESSOR_ONLY" == "1" || "$TRAIN_COMPRESSOR_ONLY" == "true" || "$TRAIN_COMPRESSOR_ONLY" == "TRUE" ]]; then
  EXTRA_ARGS+=(--importance-train-compressor-only)
fi
if [[ -n "$RESUME_CKPT" ]]; then
  EXTRA_ARGS+=(--resume-from-checkpoint "$RESUME_CKPT")
fi
if [[ -n "$WARM_START_ADAPTER_PATH" ]]; then
  EXTRA_ARGS+=(--warm-start-adapter-path "$WARM_START_ADAPTER_PATH")
fi
if [[ "$IGNORE_DATA_SKIP" == "1" || "$IGNORE_DATA_SKIP" == "true" || "$IGNORE_DATA_SKIP" == "TRUE" ]]; then
  EXTRA_ARGS+=(--ignore-data-skip)
fi

{
  echo "[importance_launcher] $(date +%Y-%m-%d\ %H:%M:%S) starting importance ablation training"
  echo "[importance_launcher] OUTPUT_DIR=$OUTPUT_DIR LOG_FILE=$LOG_FILE"
  echo "[importance_launcher] CUDA_DEVICE_LIST=$CUDA_DEVICE_LIST NUM_GPUS=$NUM_GPUS MAIN_PROCESS_PORT=$MAIN_PROCESS_PORT"
  echo "[importance_launcher] BUDGETS=${BUDGETS[*]} COMPRESS_STAGES=$COMPRESS_STAGES IMPORTANCE_MODE=$IMPORTANCE_MODE IMPORTANCE_BLEND=$IMPORTANCE_BLEND"
  echo "[importance_launcher] MAX_STEPS=$MAX_STEPS SAVE_STEPS=$SAVE_STEPS TRAIN_BSZ=$TRAIN_BSZ INTERLEAVED_BSZ=$INTERLEAVED_BSZ"
  echo "[importance_launcher] MODEL_PATH=$MODEL_PATH USE_PEFT=$USE_PEFT TRAIN_COMPRESSOR_ONLY=$TRAIN_COMPRESSOR_ONLY"
} >> "$LOG_FILE"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_LIST" \
PYTHONUNBUFFERED=1 \
"$ACCELERATE_BIN" launch \
  --num_machines 1 \
  --num_processes "$NUM_GPUS" \
  --main_process_port "$MAIN_PROCESS_PORT" \
  --mixed_precision bf16 \
  "$SCRIPT_DIR/train_importance.py" \
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
  --gradient-accumulation-steps "$GRAD_ACCUM_STEPS" \
  --interleaved-batch-size "$INTERLEAVED_BSZ" \
  --dataloader-num-workers 0 \
  --num-negative 1 \
  --num-shards 128 \
  --doc-chunk-size "$DOC_CHUNK_SIZE" \
  --query-chunk-size "$QUERY_CHUNK_SIZE" \
  --max-num-visual-tokens "$MAX_NUM_VISUAL_TOKENS" \
  --attn-implementation flash_attention_2 \
  "${EXTRA_ARGS[@]}" \
  >> "$LOG_FILE" 2>&1
