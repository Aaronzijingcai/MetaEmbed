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
QUERY_BUDGETS=(${QUERY_BUDGETS:-160 160 160})
DOC_BUDGETS=(${DOC_BUDGETS:-160 160 160})
RUN_NAME=${RUN_NAME:-folder_homo_mmeb_budget_q${QUERY_BUDGETS[0]}_${QUERY_BUDGETS[1]}_${QUERY_BUDGETS[2]}_d${DOC_BUDGETS[0]}_${DOC_BUDGETS[1]}_${DOC_BUDGETS[2]}_4k}
RUN_DIR=${RUN_DIR:-$SCRIPT_DIR/runs/$RUN_NAME}
OUTPUT_DIR=${OUTPUT_DIR:-$RUN_DIR}
LOG_FILE=${LOG_FILE:-$RUN_DIR/logs/train_$(date +%Y%m%d_%H%M%S).log}
MODEL_PATH=${MODEL_PATH:-$PROJECT_DIR/models/colqwen2.5-base}
SUBSET_CONFIG=${SUBSET_CONFIG:-$PROJECT_DIR/configs/train/moca_data_ratios_v3_full.yaml}
EVAL_MMEB_CONFIG=${EVAL_MMEB_CONFIG:-$PROJECT_DIR/configs/eval/test_data_mast_mmeb_v3.yaml}
EVAL_VIDORE_V1_CONFIG=${EVAL_VIDORE_V1_CONFIG:-$PROJECT_DIR/configs/eval/test_data_vidore_beir.yaml}
EVAL_VIDORE_V2_CONFIG=${EVAL_VIDORE_V2_CONFIG:-$PROJECT_DIR/configs/eval/test_data_mast_v2.yaml}
RUN_EVAL=${RUN_EVAL:-0}
RESUME_CKPT=${RESUME_CKPT:-}
USE_PEFT=${USE_PEFT:-1}
DDP_FIND_UNUSED_PARAMETERS=${DDP_FIND_UNUSED_PARAMETERS:-1}
IGNORE_DATA_SKIP=${IGNORE_DATA_SKIP:-0}
TRAIN_COMPRESSOR_ONLY=${TRAIN_COMPRESSOR_ONLY:-0}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}
COMPRESS_STAGES=${COMPRESS_STAGES:-all}
NOVELTY_WEIGHT=${NOVELTY_WEIGHT:-1.0}
GATE_STRENGTH=${GATE_STRENGTH:-0.25}
FOLDER_ALPHA=${FOLDER_ALPHA:-1.0}
MARC_ENABLED=${MARC_ENABLED:-0}

if [[ "${#QUERY_BUDGETS[@]}" -ne 3 ]]; then
  echo "QUERY_BUDGETS must contain exactly 3 integers, got: ${QUERY_BUDGETS[*]}" >&2
  exit 2
fi
if [[ "${#DOC_BUDGETS[@]}" -ne 3 ]]; then
  echo "DOC_BUDGETS must contain exactly 3 integers, got: ${DOC_BUDGETS[*]}" >&2
  exit 2
fi
if [[ "$MARC_ENABLED" == "1" || "$MARC_ENABLED" == "true" || "$MARC_ENABLED" == "TRUE" ]]; then
  echo "MMEB budget experiments intentionally disable MARC; use MARC_ENABLED=0." >&2
  exit 2
fi

export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-$RUN_DIR/wandb}
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
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-7200}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}
mkdir -p "$OUTPUT_DIR" "$(dirname "$LOG_FILE")" "$WANDB_DIR" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TMPDIR"

choose_port() {
  if [[ "$MAIN_PROCESS_PORT" != "0" ]]; then
    echo "$MAIN_PROCESS_PORT"
    return
  fi
  python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()"
}
MAIN_PROCESS_PORT=$(choose_port)
export MAIN_PROCESS_PORT

EXTRA_ARGS=(
  --folder-homo-enabled
  --folder-homo-compress-stages "$COMPRESS_STAGES"
  --folder-homo-budgets "${DOC_BUDGETS[@]}"
  --mmeb-query-budgets "${QUERY_BUDGETS[@]}"
  --mmeb-doc-budgets "${DOC_BUDGETS[@]}"
  --folder-homo-novelty-weight "$NOVELTY_WEIGHT"
  --folder-homo-gate-strength "$GATE_STRENGTH"
  --folder-homo-folder-alpha "$FOLDER_ALPHA"
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
  EXTRA_ARGS+=(--folder-homo-train-compressor-only)
fi
if [[ "$RUN_EVAL" == "1" || "$RUN_EVAL" == "true" || "$RUN_EVAL" == "TRUE" ]]; then
  EXTRA_ARGS+=(--run-eval)
fi
if [[ -n "$RESUME_CKPT" ]]; then
  EXTRA_ARGS+=(--resume-from-checkpoint "$RESUME_CKPT")
fi
if [[ "$IGNORE_DATA_SKIP" == "1" || "$IGNORE_DATA_SKIP" == "true" || "$IGNORE_DATA_SKIP" == "TRUE" ]]; then
  EXTRA_ARGS+=(--ignore-data-skip)
fi

{
  echo "[mmeb_budget_train] $(date +%Y-%m-%d\ %H:%M:%S) starting MMEB budget train"
  echo "[mmeb_budget_train] PROJECT_DIR=$PROJECT_DIR OUTPUT_DIR=$OUTPUT_DIR LOG_FILE=$LOG_FILE"
  echo "[mmeb_budget_train] CUDA_DEVICE_LIST=$CUDA_DEVICE_LIST NUM_GPUS=$NUM_GPUS MAIN_PROCESS_PORT=$MAIN_PROCESS_PORT"
  echo "[mmeb_budget_train] SUBSET_CONFIG=$SUBSET_CONFIG MAX_STEPS=$MAX_STEPS SAVE_STEPS=$SAVE_STEPS"
  echo "[mmeb_budget_train] TRAIN_BSZ=$TRAIN_BSZ INTERLEAVED_BSZ=$INTERLEAVED_BSZ GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS"
  echo "[mmeb_budget_train] QUERY_BUDGETS=${QUERY_BUDGETS[*]} DOC_BUDGETS=${DOC_BUDGETS[*]} COMPRESS_STAGES=$COMPRESS_STAGES"
} >> "$LOG_FILE"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_LIST" \
PYTHONUNBUFFERED=1 \
"$ACCELERATE_BIN" launch \
  --num_machines 1 \
  --num_processes "$NUM_GPUS" \
  --main_process_port "$MAIN_PROCESS_PORT" \
  --mixed_precision bf16 \
  "$SCRIPT_DIR/train_mmeb_budget.py" \
  --model-name-or-path "$MODEL_PATH" \
  --processor-name-or-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --subset-config "$SUBSET_CONFIG" \
  --eval-vidore-v1-config "$EVAL_VIDORE_V1_CONFIG" \
  --eval-vidore-v2-config "$EVAL_VIDORE_V2_CONFIG" \
  --eval-mmeb-config "$EVAL_MMEB_CONFIG" \
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
