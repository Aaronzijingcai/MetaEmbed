#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
REPO_ROOT=$(cd "$PROJECT_DIR/.." && pwd)

cd "$REPO_ROOT"

if [[ -d /opt/conda/bin ]]; then
  export PATH="/opt/conda/bin:$PATH"
fi

ACCELERATE_BIN=${ACCELERATE_BIN:-accelerate}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-0}
export PYTHONPATH="$PROJECT_DIR/vendor:$REPO_ROOT:${PYTHONPATH:-}"
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-$PROJECT_DIR/runs/wandb}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export MURE_CACHE_ROOT=${MURE_CACHE_ROOT:-$PROJECT_DIR/.cache}
export HF_HOME=${HF_HOME:-$MURE_CACHE_ROOT/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
export TMPDIR=${TMPDIR:-$MURE_CACHE_ROOT/tmp}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-7200}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "$PROJECT_DIR/runs/logs" "$WANDB_DIR" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TMPDIR"

CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1,2,3}
NUM_GPUS=${NUM_GPUS:-4}
MODEL_PATH=${MODEL_PATH:-$PROJECT_DIR/models/colqwen2.5-base}
PROCESSOR_PATH=${PROCESSOR_PATH:-$MODEL_PATH}
SUBSET_CONFIG=${SUBSET_CONFIG:-$PROJECT_DIR/configs/train/moca_data_ratios_v3_nommE5.yaml}

MAX_STEPS=${MAX_STEPS:-8000}
SAVE_STEPS=${SAVE_STEPS:-1000}
LOGGING_STEPS=${LOGGING_STEPS:-10}
PER_GPU_BSZ=${PER_GPU_BSZ:-1}
EVAL_BSZ=${EVAL_BSZ:-1}
INTERLEAVED_BSZ=${INTERLEAVED_BSZ:-4}
GRAD_ACCUM=${GRAD_ACCUM:-1}
NUM_NEGATIVE=${NUM_NEGATIVE:-1}
NUM_SHARDS=${NUM_SHARDS:-128}
DOC_CHUNK_SIZE=${DOC_CHUNK_SIZE:-256}
TRUNCATION_LEN=${TRUNCATION_LEN:-16384}
ATTN_IMPL=${ATTN_IMPL:-flash_attention_2}
CROP_RESIZE_MODE=${CROP_RESIZE_MODE:-stretch}
USE_PEFT=${USE_PEFT:-1}
RESUME_CKPT=${RESUME_CKPT:-}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}
DDP_FIND_UNUSED_PARAMETERS=${DDP_FIND_UNUSED_PARAMETERS:-1}

SOFTASSIGN_STAGES=${SOFTASSIGN_STAGES:-all}
SOFTASSIGN_BUDGETS=(${SOFTASSIGN_BUDGETS:-512 1024 2048})
SOFTASSIGN_KEEP_RATIO=${SOFTASSIGN_KEEP_RATIO:-0.25}
SOFTASSIGN_TEMPERATURE=${SOFTASSIGN_TEMPERATURE:-0.1}
LEARNABLE_TEMPERATURE=${LEARNABLE_TEMPERATURE:-0}
DEBUG_SHAPES=${DEBUG_SHAPES:-0}

if [[ "${#SOFTASSIGN_BUDGETS[@]}" -ne 3 ]]; then
  echo "SOFTASSIGN_BUDGETS must contain three integers, got: ${SOFTASSIGN_BUDGETS[*]}" >&2
  exit 2
fi

RUN_NAME=${RUN_NAME:-strategy1_softassign_full_4gpu_${SOFTASSIGN_STAGES}_kr${SOFTASSIGN_KEEP_RATIO}_${SOFTASSIGN_BUDGETS[0]}-${SOFTASSIGN_BUDGETS[1]}-${SOFTASSIGN_BUDGETS[2]}_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_DIR/runs/$RUN_NAME}
LOG_FILE=${LOG_FILE:-$PROJECT_DIR/runs/logs/${RUN_NAME}.log}

SOFTASSIGN_ARGS=(
  --strategy1_softassign-enabled
  --strategy1_softassign-compress-stages "$SOFTASSIGN_STAGES"
  --strategy1_softassign-budgets "${SOFTASSIGN_BUDGETS[@]}"
  --strategy1_softassign-keep-ratio "$SOFTASSIGN_KEEP_RATIO"
  --strategy1_softassign-temperature "$SOFTASSIGN_TEMPERATURE"
)
if [[ "$LEARNABLE_TEMPERATURE" == "1" || "$LEARNABLE_TEMPERATURE" == "true" ]]; then
  SOFTASSIGN_ARGS+=(--strategy1_softassign-learnable-temperature)
fi
if [[ "$DEBUG_SHAPES" == "1" || "$DEBUG_SHAPES" == "true" ]]; then
  SOFTASSIGN_ARGS+=(--strategy1_softassign-debug-shapes)
fi
if [[ "$USE_PEFT" == "1" || "$USE_PEFT" == "true" ]]; then
  SOFTASSIGN_ARGS+=(--use-peft)
else
  SOFTASSIGN_ARGS+=(--no-use-peft)
fi
if [[ -n "$RESUME_CKPT" ]]; then
  SOFTASSIGN_ARGS+=(--resume-from-checkpoint "$RESUME_CKPT")
fi
if [[ "$GRADIENT_CHECKPOINTING" == "1" || "$GRADIENT_CHECKPOINTING" == "true" ]]; then
  SOFTASSIGN_ARGS+=(--gradient-checkpointing)
else
  SOFTASSIGN_ARGS+=(--no-gradient-checkpointing)
fi
if [[ "$DDP_FIND_UNUSED_PARAMETERS" == "1" || "$DDP_FIND_UNUSED_PARAMETERS" == "true" ]]; then
  SOFTASSIGN_ARGS+=(--ddp-find-unused-parameters)
fi

echo "[strategy1_softassign_train_4gpu] cuda=$CUDA_DEVICE_LIST num_gpus=$NUM_GPUS"
echo "[strategy1_softassign_train_4gpu] main_process_port=$MAIN_PROCESS_PORT"
echo "[strategy1_softassign_train_4gpu] model=$MODEL_PATH"
echo "[strategy1_softassign_train_4gpu] subset_config=$SUBSET_CONFIG"
echo "[strategy1_softassign_train_4gpu] output_dir=$OUTPUT_DIR"
echo "[strategy1_softassign_train_4gpu] max_steps=$MAX_STEPS save_steps=$SAVE_STEPS per_gpu_bsz=$PER_GPU_BSZ interleaved_bsz=$INTERLEAVED_BSZ grad_accum=$GRAD_ACCUM"
echo "[strategy1_softassign_train_4gpu] budgets=${SOFTASSIGN_BUDGETS[*]} stages=$SOFTASSIGN_STAGES keep_ratio=$SOFTASSIGN_KEEP_RATIO temperature=$SOFTASSIGN_TEMPERATURE"
echo "[strategy1_softassign_train_4gpu] use_peft=$USE_PEFT doc_chunk_size=$DOC_CHUNK_SIZE truncation_len=$TRUNCATION_LEN"
echo "[strategy1_softassign_train_4gpu] gradient_checkpointing=$GRADIENT_CHECKPOINTING"
echo "[strategy1_softassign_train_4gpu] ddp_find_unused_parameters=$DDP_FIND_UNUSED_PARAMETERS"
echo "[strategy1_softassign_train_4gpu] resume_ckpt=${RESUME_CKPT:-none}"
echo "[strategy1_softassign_train_4gpu] log=$LOG_FILE"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_LIST" \
PYTHONUNBUFFERED=1 \
"$ACCELERATE_BIN" launch \
  --num_machines 1 \
  --num_processes "$NUM_GPUS" \
  --main_process_port "$MAIN_PROCESS_PORT" \
  --mixed_precision bf16 \
  -m colqwen_multigranularity.experiments.strategy1_softassign.train_strategy1_softassign \
  --model-name-or-path "$MODEL_PATH" \
  --processor-name-or-path "$PROCESSOR_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --subset-config "$SUBSET_CONFIG" \
  --granularities 1 2 4 \
  --max-steps "$MAX_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --logging-steps "$LOGGING_STEPS" \
  --learning-rate 1e-4 \
  --lr-scheduler-type linear \
  --warmup-ratio 0.03 \
  --per-device-train-batch-size "$PER_GPU_BSZ" \
  --per-device-eval-batch-size "$EVAL_BSZ" \
  --vidore-eval-batch-size "$EVAL_BSZ" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --interleaved-batch-size "$INTERLEAVED_BSZ" \
  --dataloader-num-workers 0 \
  --num-negative "$NUM_NEGATIVE" \
  --num-shards "$NUM_SHARDS" \
  --doc-chunk-size "$DOC_CHUNK_SIZE" \
  --truncation-len "$TRUNCATION_LEN" \
  --crop-resize-mode "$CROP_RESIZE_MODE" \
  --attn-implementation "$ATTN_IMPL" \
  "${SOFTASSIGN_ARGS[@]}" \
  2>&1 | tee -a "$LOG_FILE"
