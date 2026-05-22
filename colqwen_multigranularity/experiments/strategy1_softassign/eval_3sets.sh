#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
REPO_ROOT=$(cd "$PROJECT_DIR/.." && pwd)

cd "$REPO_ROOT"

if [[ -d /opt/conda/bin ]]; then
  export PATH="/opt/conda/bin:$PATH"
fi

ACCELERATE_BIN=${ACCELERATE_BIN:-$(command -v accelerate || true)}
if [[ -z "$ACCELERATE_BIN" ]]; then
  echo "accelerate executable not found; set ACCELERATE_BIN=/path/to/accelerate" >&2
  exit 2
fi

export PYTHONPATH="$PROJECT_DIR/vendor:$REPO_ROOT:${PYTHONPATH:-}"
export WANDB_MODE=${WANDB_MODE:-offline}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export MURE_CACHE_ROOT=${MURE_CACHE_ROOT:-$PROJECT_DIR/.cache}
export HF_HOME=${HF_HOME:-$MURE_CACHE_ROOT/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
export TMPDIR=${TMPDIR:-$MURE_CACHE_ROOT/tmp}
export WANDB_DIR=${WANDB_DIR:-$PROJECT_DIR/runs/wandb}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-7200}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "$PROJECT_DIR/runs/logs" "$PROJECT_DIR/runs/eval" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TMPDIR" "$WANDB_DIR"

CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0}
NUM_GPUS=${NUM_GPUS:-1}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-0}
MODEL_PATH=${MODEL_PATH:-$PROJECT_DIR/models/colqwen2.5-base}
PROCESSOR_PATH=${PROCESSOR_PATH:-$MODEL_PATH}
CHECKPOINT=${1:-${CHECKPOINT:-$PROJECT_DIR/runs/strategy1_softassign_full_4gpu_all_64-64-128_20260515_161253}}
OUT_DIR=${OUT_DIR:-$PROJECT_DIR/runs/eval/strategy1_softassign_full_4gpu_all_64-64-128_20260515_161253}
RUN_NAME=${RUN_NAME:-strategy1_softassign_eval_3sets_$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-$PROJECT_DIR/runs/logs/${RUN_NAME}.log}
EVAL_CKPTS=${EVAL_CKPTS:-}

BATCH_QUERY=${BATCH_QUERY:-1}
BATCH_PASSAGE=${BATCH_PASSAGE:-1}
BATCH_SCORE=${BATCH_SCORE:-4}
NUM_WORKERS=${NUM_WORKERS:-2}
TRUNCATION_LEN=${TRUNCATION_LEN:-16384}
ATTN_IMPL=${ATTN_IMPL:-flash_attention_2}
CROP_RESIZE_MODE=${CROP_RESIZE_MODE:-stretch}
USE_V2_RETRIEVER=${USE_V2_RETRIEVER:-1}
V2_DO_PADDING=${V2_DO_PADDING:-1}
INCLUDE_MULTILINGUAL=${INCLUDE_MULTILINGUAL:-0}
SOFTASSIGN_BUDGETS=(${SOFTASSIGN_BUDGETS:-64 64 128})
SOFTASSIGN_KEEP_RATIO=${SOFTASSIGN_KEEP_RATIO:-}
SOFTASSIGN_STAGES=${SOFTASSIGN_STAGES:-all}
SOFTASSIGN_TEMPERATURE=${SOFTASSIGN_TEMPERATURE:-0.1}

VIDORE_V1_CONFIG=${VIDORE_V1_CONFIG:-$PROJECT_DIR/configs/eval/test_data_vidore_beir.yaml}
VIDORE_V2_CONFIG=${VIDORE_V2_CONFIG:-$PROJECT_DIR/configs/eval/test_data_mast_v2.yaml}
MMEB_CONFIG=${MMEB_CONFIG:-$PROJECT_DIR/configs/eval/test_data_mast_mmeb_v3.yaml}

if [[ "${#SOFTASSIGN_BUDGETS[@]}" -ne 3 ]]; then
  echo "SOFTASSIGN_BUDGETS must contain three integers, got: ${SOFTASSIGN_BUDGETS[*]}" >&2
  exit 2
fi

mkdir -p "$OUT_DIR" "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"

run_eval() {
  local name="$1"
  local config="$2"
  local format="$3"
  local output="$4"
  shift 4
  echo "[strategy1_softassign_eval] name=$name config=$config format=$format output=$output" | tee -a "$LOG_FILE"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_LIST" \
  PYTHONUNBUFFERED=1 \
  "$ACCELERATE_BIN" launch \
    --num_machines 1 \
    --num_processes "$NUM_GPUS" \
    --main_process_port "$MAIN_PROCESS_PORT" \
    --mixed_precision bf16 \
    -m colqwen_multigranularity.experiments.strategy1_softassign.eval_softassign \
    "$@" \
    --eval-config "$config" \
    --dataset-format "$format" \
    --output-path "$output" \
    2>&1 | tee -a "$LOG_FILE"
}

run_checkpoint() {
  local checkpoint="$1"
  local strategy1_softassign_path="${SOFTASSIGN_PATH:-$checkpoint}"
  local ckpt_name
  ckpt_name=$(basename "$checkpoint")
  local output_dir="$OUT_DIR"
  if [[ -n "$EVAL_CKPTS" ]]; then
    output_dir="$OUT_DIR/$ckpt_name"
  fi

  if [[ ! -d "$checkpoint" ]]; then
    echo "checkpoint directory not found: $checkpoint" >&2
    exit 2
  fi
  if [[ ! -f "$checkpoint/adapter_config.json" ]]; then
    echo "adapter_config.json not found under checkpoint: $checkpoint" >&2
    exit 2
  fi
  if [[ ! -f "$strategy1_softassign_path/strategy1_softassign.bin" || ! -f "$strategy1_softassign_path/strategy1_softassign_config.json" ]]; then
    echo "strategy1_softassign weights/config not found under SOFTASSIGN_PATH: $strategy1_softassign_path" >&2
    exit 2
  fi

  mkdir -p "$output_dir"

  local common_flags=(
    --model-name-or-path "$MODEL_PATH"
    --processor-name-or-path "$PROCESSOR_PATH"
    --adapter-path "$checkpoint"
    --strategy1_softassign-path "$strategy1_softassign_path"
    --strategy1_softassign-enabled
    --strategy1_softassign-compress-stages "$SOFTASSIGN_STAGES"
    --strategy1_softassign-budgets "${SOFTASSIGN_BUDGETS[@]}"
    --strategy1_softassign-temperature "$SOFTASSIGN_TEMPERATURE"
    --granularities 1 2 4
    --truncation-len "$TRUNCATION_LEN"
    --crop-resize-mode "$CROP_RESIZE_MODE"
    --attn-implementation "$ATTN_IMPL"
    --batch-query "$BATCH_QUERY"
    --batch-passage "$BATCH_PASSAGE"
    --batch-score "$BATCH_SCORE"
    --num-workers "$NUM_WORKERS"
  )
  if [[ -n "$SOFTASSIGN_KEEP_RATIO" ]]; then
    common_flags+=(--strategy1_softassign-keep-ratio "$SOFTASSIGN_KEEP_RATIO")
  fi
  if [[ "$USE_V2_RETRIEVER" == "1" || "$USE_V2_RETRIEVER" == "true" ]]; then
    common_flags+=(--use-v2-retriever)
  else
    common_flags+=(--no-use-v2-retriever)
  fi
  if [[ "$V2_DO_PADDING" == "1" || "$V2_DO_PADDING" == "true" ]]; then
    common_flags+=(--v2-do-padding)
  else
    common_flags+=(--no-v2-do-padding)
  fi
  if [[ "$INCLUDE_MULTILINGUAL" == "1" || "$INCLUDE_MULTILINGUAL" == "true" ]]; then
    common_flags+=(--include-multilingual)
  fi

  {
    echo "[strategy1_softassign_eval] checkpoint=$checkpoint"
    echo "[strategy1_softassign_eval] strategy1_softassign_path=$strategy1_softassign_path"
    echo "[strategy1_softassign_eval] out_dir=$output_dir"
    echo "[strategy1_softassign_eval] cuda=$CUDA_DEVICE_LIST num_gpus=$NUM_GPUS batch_query=$BATCH_QUERY batch_passage=$BATCH_PASSAGE batch_score=$BATCH_SCORE"
    echo "[strategy1_softassign_eval] budgets=${SOFTASSIGN_BUDGETS[*]} stages=$SOFTASSIGN_STAGES keep_ratio=${SOFTASSIGN_KEEP_RATIO:-from_config}"
  } | tee -a "$LOG_FILE"

  run_eval vidore_v1 "$VIDORE_V1_CONFIG" beir "$output_dir/vidore_v1.json" "${common_flags[@]}"
  run_eval vidore_v2 "$VIDORE_V2_CONFIG" beir "$output_dir/vidore_v2.json" "${common_flags[@]}"
  run_eval mmeb "$MMEB_CONFIG" mmeb "$output_dir/mmeb.json" "${common_flags[@]}"
}

if [[ -n "$EVAL_CKPTS" ]]; then
  for checkpoint in $EVAL_CKPTS; do
    run_checkpoint "$checkpoint"
  done
else
  run_checkpoint "$CHECKPOINT"
fi

echo "[strategy1_softassign_eval] done" | tee -a "$LOG_FILE"
