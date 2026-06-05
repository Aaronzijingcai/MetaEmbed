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

SOFTASSIGN_STAGES=${SOFTASSIGN_STAGES:-g2g3}
SOFTASSIGN_BUDGETS=(${SOFTASSIGN_BUDGETS:-512 1024 2048})
SOFTASSIGN_TEMPERATURE=${SOFTASSIGN_TEMPERATURE:-0.1}
LEARNABLE_TEMPERATURE=${LEARNABLE_TEMPERATURE:-0}
DEBUG_SHAPES=${DEBUG_SHAPES:-0}

SERIAL_RUN_PREFIX=${SERIAL_RUN_PREFIX:-softassign_textquery_focus_perstage_8k_$(date +%Y%m%d_%H%M%S)}

if [[ "${#SOFTASSIGN_BUDGETS[@]}" -ne 3 ]]; then
  echo "SOFTASSIGN_BUDGETS must contain three integers, got: ${SOFTASSIGN_BUDGETS[*]}" >&2
  exit 2
fi

resolve_local_path() {
  local label="$1"
  local input_path="$2"
  local basename_path
  basename_path=$(basename "$input_path")

  if [[ -d "$input_path" ]]; then
    echo "$input_path"
    return 0
  fi

  local candidates=(
    "$PROJECT_DIR/models/$basename_path"
    "$REPO_ROOT/colqwen_multigranularity/models/$basename_path"
    "$REPO_ROOT/models/$basename_path"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -d "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done

  if [[ "$input_path" = /* ]]; then
    echo "$label directory not found: $input_path" >&2
    echo "Checked fallback locations:" >&2
    printf '  %s\n' "${candidates[@]}" >&2
    return 2
  fi

  echo "$input_path"
}

MODEL_PATH=$(resolve_local_path "MODEL_PATH" "$MODEL_PATH")
PROCESSOR_PATH=$(resolve_local_path "PROCESSOR_PATH" "$PROCESSOR_PATH")
if [[ -d "$MODEL_PATH" && ! -f "$MODEL_PATH/config.json" ]]; then
  echo "MODEL_PATH exists but config.json is missing: $MODEL_PATH" >&2
  exit 2
fi
if [[ -d "$PROCESSOR_PATH" && ! -f "$PROCESSOR_PATH/preprocessor_config.json" ]]; then
  echo "PROCESSOR_PATH exists but preprocessor_config.json is missing: $PROCESSOR_PATH" >&2
  exit 2
fi

run_experiment() {
  local label="$1"
  local g1_keep="$2"
  local g2_keep="$3"
  local g3_keep="$4"
  local run_name="${SERIAL_RUN_PREFIX}_${label}"
  local output_dir="$PROJECT_DIR/runs/$run_name"
  local log_file="$PROJECT_DIR/runs/logs/${run_name}.log"

  local softassign_args=(
    --strategy1_softassign-enabled
    --strategy1_softassign-compress-stages "$SOFTASSIGN_STAGES"
    --strategy1_softassign-budgets "${SOFTASSIGN_BUDGETS[@]}"
    --strategy1_softassign-keep-ratios "$g1_keep" "$g2_keep" "$g3_keep"
    --strategy1_softassign-temperature "$SOFTASSIGN_TEMPERATURE"
  )
  if [[ "$LEARNABLE_TEMPERATURE" == "1" || "$LEARNABLE_TEMPERATURE" == "true" ]]; then
    softassign_args+=(--strategy1_softassign-learnable-temperature)
  fi
  if [[ "$DEBUG_SHAPES" == "1" || "$DEBUG_SHAPES" == "true" ]]; then
    softassign_args+=(--strategy1_softassign-debug-shapes)
  fi
  if [[ "$USE_PEFT" == "1" || "$USE_PEFT" == "true" ]]; then
    softassign_args+=(--use-peft)
  else
    softassign_args+=(--no-use-peft)
  fi
  if [[ -n "$RESUME_CKPT" ]]; then
    softassign_args+=(--resume-from-checkpoint "$RESUME_CKPT")
  fi
  if [[ "$GRADIENT_CHECKPOINTING" == "1" || "$GRADIENT_CHECKPOINTING" == "true" ]]; then
    softassign_args+=(--gradient-checkpointing)
  else
    softassign_args+=(--no-gradient-checkpointing)
  fi
  if [[ "$DDP_FIND_UNUSED_PARAMETERS" == "1" || "$DDP_FIND_UNUSED_PARAMETERS" == "true" ]]; then
    softassign_args+=(--ddp-find-unused-parameters)
  fi

  echo "[strategy1_softassign_b2_b3_b4_serial] start label=$label run_name=$run_name"
  echo "[strategy1_softassign_b2_b3_b4_serial] cuda=$CUDA_DEVICE_LIST num_gpus=$NUM_GPUS"
  echo "[strategy1_softassign_b2_b3_b4_serial] main_process_port=$MAIN_PROCESS_PORT"
  echo "[strategy1_softassign_b2_b3_b4_serial] model=$MODEL_PATH"
  echo "[strategy1_softassign_b2_b3_b4_serial] subset_config=$SUBSET_CONFIG"
  echo "[strategy1_softassign_b2_b3_b4_serial] output_dir=$output_dir"
  echo "[strategy1_softassign_b2_b3_b4_serial] max_steps=$MAX_STEPS save_steps=$SAVE_STEPS per_gpu_bsz=$PER_GPU_BSZ interleaved_bsz=$INTERLEAVED_BSZ grad_accum=$GRAD_ACCUM"
  echo "[strategy1_softassign_b2_b3_b4_serial] budgets=${SOFTASSIGN_BUDGETS[*]} stages=$SOFTASSIGN_STAGES keep_ratios=$g1_keep $g2_keep $g3_keep temperature=$SOFTASSIGN_TEMPERATURE"
  echo "[strategy1_softassign_b2_b3_b4_serial] use_peft=$USE_PEFT doc_chunk_size=$DOC_CHUNK_SIZE truncation_len=$TRUNCATION_LEN"
  echo "[strategy1_softassign_b2_b3_b4_serial] gradient_checkpointing=$GRADIENT_CHECKPOINTING"
  echo "[strategy1_softassign_b2_b3_b4_serial] ddp_find_unused_parameters=$DDP_FIND_UNUSED_PARAMETERS"
  echo "[strategy1_softassign_b2_b3_b4_serial] resume_ckpt=${RESUME_CKPT:-none}"
  echo "[strategy1_softassign_b2_b3_b4_serial] log=$log_file"

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_LIST" \
  PYTHONUNBUFFERED=1 \
  "$ACCELERATE_BIN" launch \
    --num_machines 1 \
    --num_processes "$NUM_GPUS" \
    --main_process_port "$MAIN_PROCESS_PORT" \
    --mixed_precision bf16 \
    -m colqwen_multigranularity.experiments.strategy1_softassign.train_softassign \
    --model-name-or-path "$MODEL_PATH" \
    --processor-name-or-path "$PROCESSOR_PATH" \
    --output-dir "$output_dir" \
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
    "${softassign_args[@]}" \
    2>&1 | tee -a "$log_file"

  echo "[strategy1_softassign_b2_b3_b4_serial] done label=$label"
}

run_experiment "b2_g1-1.0_g2-0.75_g3-0.40" "1.0" "0.75" "0.40"
run_experiment "b3_g1-1.0_g2-0.65_g3-0.35" "1.0" "0.65" "0.35"
run_experiment "b4_g1-1.0_g2-0.50_g3-0.35" "1.0" "0.50" "0.35"
