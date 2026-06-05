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

VISIONZIP_STAGES=${VISIONZIP_STAGES:-all}
VISIONZIP_BUDGETS=(${VISIONZIP_BUDGETS:-64 128 256})
VISIONZIP_KEEP_RATIO=${VISIONZIP_KEEP_RATIO:-}
VISIONZIP_KEEP_RATIOS=(${VISIONZIP_KEEP_RATIOS:-})
VISIONZIP_SCOPE=${VISIONZIP_SCOPE:-crop}
VISIONZIP_CROP_BUDGET_MODE=${VISIONZIP_CROP_BUDGET_MODE:-proportional}
VISIONZIP_DOMINANT_RATIO=${VISIONZIP_DOMINANT_RATIO:-0.75}
VISIONZIP_ATTENTION_SOURCE=${VISIONZIP_ATTENTION_SOURCE:-self_similarity}
VISIONZIP_VISUAL_ATTN_LAYER=${VISIONZIP_VISUAL_ATTN_LAYER:--2}
VISIONZIP_TARGET_SELECT=${VISIONZIP_TARGET_SELECT:-uniform}
VISIONZIP_MERGE_METRIC=${VISIONZIP_MERGE_METRIC:-cosine}
VISIONZIP_RANDOM_SEED=${VISIONZIP_RANDOM_SEED:-0}
PRESERVE_INPUT_RMS=${PRESERVE_INPUT_RMS:-1}
DEBUG_SHAPES=${DEBUG_SHAPES:-0}

if [[ "${#VISIONZIP_BUDGETS[@]}" -ne 3 ]]; then
  echo "VISIONZIP_BUDGETS must contain three integers, got: ${VISIONZIP_BUDGETS[*]}" >&2
  exit 2
fi
if [[ "${#VISIONZIP_KEEP_RATIOS[@]}" -ne 0 && "${#VISIONZIP_KEEP_RATIOS[@]}" -ne 3 ]]; then
  echo "VISIONZIP_KEEP_RATIOS must be empty or contain three floats, got: ${VISIONZIP_KEEP_RATIOS[*]}" >&2
  exit 2
fi
if [[ "$VISIONZIP_ATTENTION_SOURCE" == "visual_attn" && "$ATTN_IMPL" != "eager" ]]; then
  echo "VISIONZIP_ATTENTION_SOURCE=visual_attn requires ATTN_IMPL=eager, got: $ATTN_IMPL" >&2
  exit 2
fi

RUN_NAME=${RUN_NAME:-strategy2_visionzip_${VISIONZIP_SCOPE}_${VISIONZIP_STAGES}_${VISIONZIP_BUDGETS[0]}-${VISIONZIP_BUDGETS[1]}-${VISIONZIP_BUDGETS[2]}_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_DIR/runs/$RUN_NAME}
LOG_FILE=${LOG_FILE:-$PROJECT_DIR/runs/logs/${RUN_NAME}.log}

VISIONZIP_ARGS=(
  --strategy2_visionzip-enabled
  --strategy2_visionzip-compress-stages "$VISIONZIP_STAGES"
  --strategy2_visionzip-budgets "${VISIONZIP_BUDGETS[@]}"
  --strategy2_visionzip-compression-scope "$VISIONZIP_SCOPE"
  --strategy2_visionzip-crop-budget-mode "$VISIONZIP_CROP_BUDGET_MODE"
  --strategy2_visionzip-dominant-ratio "$VISIONZIP_DOMINANT_RATIO"
  --strategy2_visionzip-attention-source "$VISIONZIP_ATTENTION_SOURCE"
  --strategy2_visionzip-visual-attn-layer "$VISIONZIP_VISUAL_ATTN_LAYER"
  --strategy2_visionzip-target-select "$VISIONZIP_TARGET_SELECT"
  --strategy2_visionzip-merge-metric "$VISIONZIP_MERGE_METRIC"
  --strategy2_visionzip-random-seed "$VISIONZIP_RANDOM_SEED"
)
if [[ -n "$VISIONZIP_KEEP_RATIO" ]]; then
  VISIONZIP_ARGS+=(--strategy2_visionzip-keep-ratio "$VISIONZIP_KEEP_RATIO")
fi
if [[ "${#VISIONZIP_KEEP_RATIOS[@]}" -eq 3 ]]; then
  VISIONZIP_ARGS+=(--strategy2_visionzip-keep-ratios "${VISIONZIP_KEEP_RATIOS[@]}")
fi
if [[ "$PRESERVE_INPUT_RMS" != "1" && "$PRESERVE_INPUT_RMS" != "true" ]]; then
  VISIONZIP_ARGS+=(--strategy2_visionzip-no-preserve-input-rms)
fi
if [[ "$DEBUG_SHAPES" == "1" || "$DEBUG_SHAPES" == "true" ]]; then
  VISIONZIP_ARGS+=(--strategy2_visionzip-debug-shapes)
fi
if [[ "$USE_PEFT" == "1" || "$USE_PEFT" == "true" ]]; then
  VISIONZIP_ARGS+=(--use-peft)
else
  VISIONZIP_ARGS+=(--no-use-peft)
fi
if [[ -n "$RESUME_CKPT" ]]; then
  VISIONZIP_ARGS+=(--resume-from-checkpoint "$RESUME_CKPT")
fi
if [[ "$GRADIENT_CHECKPOINTING" == "1" || "$GRADIENT_CHECKPOINTING" == "true" ]]; then
  VISIONZIP_ARGS+=(--gradient-checkpointing)
else
  VISIONZIP_ARGS+=(--no-gradient-checkpointing)
fi
if [[ "$DDP_FIND_UNUSED_PARAMETERS" == "1" || "$DDP_FIND_UNUSED_PARAMETERS" == "true" ]]; then
  VISIONZIP_ARGS+=(--ddp-find-unused-parameters)
fi

{
  echo "[strategy2_visionzip_train_4gpu] cuda=$CUDA_DEVICE_LIST num_gpus=$NUM_GPUS"
  echo "[strategy2_visionzip_train_4gpu] main_process_port=$MAIN_PROCESS_PORT"
  echo "[strategy2_visionzip_train_4gpu] model=$MODEL_PATH"
  echo "[strategy2_visionzip_train_4gpu] processor=$PROCESSOR_PATH"
  echo "[strategy2_visionzip_train_4gpu] subset_config=$SUBSET_CONFIG"
  echo "[strategy2_visionzip_train_4gpu] output_dir=$OUTPUT_DIR"
  echo "[strategy2_visionzip_train_4gpu] max_steps=$MAX_STEPS save_steps=$SAVE_STEPS logging_steps=$LOGGING_STEPS"
  echo "[strategy2_visionzip_train_4gpu] per_gpu_bsz=$PER_GPU_BSZ eval_bsz=$EVAL_BSZ interleaved_bsz=$INTERLEAVED_BSZ grad_accum=$GRAD_ACCUM"
  echo "[strategy2_visionzip_train_4gpu] num_negative=$NUM_NEGATIVE num_shards=$NUM_SHARDS doc_chunk_size=$DOC_CHUNK_SIZE truncation_len=$TRUNCATION_LEN"
  echo "[strategy2_visionzip_train_4gpu] attn_impl=$ATTN_IMPL crop_resize_mode=$CROP_RESIZE_MODE"
  echo "[strategy2_visionzip_train_4gpu] visionzip_scope=$VISIONZIP_SCOPE stages=$VISIONZIP_STAGES budgets=${VISIONZIP_BUDGETS[*]}"
  echo "[strategy2_visionzip_train_4gpu] crop_budget_mode=$VISIONZIP_CROP_BUDGET_MODE dominant_ratio=$VISIONZIP_DOMINANT_RATIO"
  echo "[strategy2_visionzip_train_4gpu] attention_source=$VISIONZIP_ATTENTION_SOURCE visual_attn_layer=$VISIONZIP_VISUAL_ATTN_LAYER target_select=$VISIONZIP_TARGET_SELECT merge_metric=$VISIONZIP_MERGE_METRIC random_seed=$VISIONZIP_RANDOM_SEED"
  echo "[strategy2_visionzip_train_4gpu] keep_ratio=${VISIONZIP_KEEP_RATIO:-none} keep_ratios=${VISIONZIP_KEEP_RATIOS[*]:-none} preserve_input_rms=$PRESERVE_INPUT_RMS debug_shapes=$DEBUG_SHAPES"
  echo "[strategy2_visionzip_train_4gpu] use_peft=$USE_PEFT gradient_checkpointing=$GRADIENT_CHECKPOINTING ddp_find_unused_parameters=$DDP_FIND_UNUSED_PARAMETERS"
  echo "[strategy2_visionzip_train_4gpu] resume_ckpt=${RESUME_CKPT:-none}"
  echo "[strategy2_visionzip_train_4gpu] log=$LOG_FILE"
} | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_LIST" \
PYTHONUNBUFFERED=1 \
"$ACCELERATE_BIN" launch \
  --num_machines 1 \
  --num_processes "$NUM_GPUS" \
  --main_process_port "$MAIN_PROCESS_PORT" \
  --mixed_precision bf16 \
  -m colqwen_multigranularity.experiments.strategy2_visionzip.train_visionzip \
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
  "${VISIONZIP_ARGS[@]}" \
  2>&1 | tee -a "$LOG_FILE"
