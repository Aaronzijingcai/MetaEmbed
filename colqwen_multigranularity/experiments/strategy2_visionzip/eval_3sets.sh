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
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
export TMPDIR=${TMPDIR:-$MURE_CACHE_ROOT/tmp}

mkdir -p "$PROJECT_DIR/runs/logs" "$PROJECT_DIR/runs/eval" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TMPDIR"

CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0}
NUM_GPUS=${NUM_GPUS:-1}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-0}
MODEL_PATH=${MODEL_PATH:-$PROJECT_DIR/models/colqwen2.5-base}
PROCESSOR_PATH=${PROCESSOR_PATH:-$MODEL_PATH}
CHECKPOINT=${1:-${CHECKPOINT:-}}
OUT_DIR=${OUT_DIR:-$PROJECT_DIR/runs/eval/strategy2_visionzip_3sets}
RUN_NAME=${RUN_NAME:-strategy2_visionzip_eval_3sets_$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-$PROJECT_DIR/runs/logs/${RUN_NAME}.log}

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

VISIONZIP_BUDGETS=(${VISIONZIP_BUDGETS:-64 128 256})
VISIONZIP_STAGES=${VISIONZIP_STAGES:-all}
VISIONZIP_SCOPE=${VISIONZIP_SCOPE:-crop}
VISIONZIP_CROP_BUDGET_MODE=${VISIONZIP_CROP_BUDGET_MODE:-proportional}
VISIONZIP_DOMINANT_RATIO=${VISIONZIP_DOMINANT_RATIO:-0.75}
VISIONZIP_ATTENTION_SOURCE=${VISIONZIP_ATTENTION_SOURCE:-self_similarity}
VISIONZIP_VISUAL_ATTN_LAYER=${VISIONZIP_VISUAL_ATTN_LAYER:--2}
VISIONZIP_TARGET_SELECT=${VISIONZIP_TARGET_SELECT:-uniform}
VISIONZIP_MERGE_METRIC=${VISIONZIP_MERGE_METRIC:-cosine}

VIDORE_V1_CONFIG=${VIDORE_V1_CONFIG:-$PROJECT_DIR/configs/eval/test_data_vidore_beir.yaml}
VIDORE_V2_CONFIG=${VIDORE_V2_CONFIG:-$PROJECT_DIR/configs/eval/test_data_mast_v2.yaml}
MMEB_CONFIG=${MMEB_CONFIG:-$PROJECT_DIR/configs/eval/test_data_mast_mmeb_v3.yaml}

if [[ "${#VISIONZIP_BUDGETS[@]}" -ne 3 ]]; then
  echo "VISIONZIP_BUDGETS must contain three integers, got: ${VISIONZIP_BUDGETS[*]}" >&2
  exit 2
fi
if [[ -z "$CHECKPOINT" ]]; then
  echo "Set CHECKPOINT or pass a checkpoint path as the first argument." >&2
  exit 2
fi
if [[ ! -d "$CHECKPOINT" ]]; then
  echo "checkpoint directory not found: $CHECKPOINT" >&2
  exit 2
fi
if [[ "$VISIONZIP_ATTENTION_SOURCE" == "visual_attn" && "$ATTN_IMPL" != "eager" ]]; then
  echo "VISIONZIP_ATTENTION_SOURCE=visual_attn requires ATTN_IMPL=eager, got: $ATTN_IMPL" >&2
  exit 2
fi

mkdir -p "$OUT_DIR" "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"

COMMON_FLAGS=(
  --model-name-or-path "$MODEL_PATH"
  --processor-name-or-path "$PROCESSOR_PATH"
  --adapter-path "$CHECKPOINT"
  --strategy2_visionzip-path "$CHECKPOINT"
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
  --granularities 1 2 4
  --truncation-len "$TRUNCATION_LEN"
  --crop-resize-mode "$CROP_RESIZE_MODE"
  --attn-implementation "$ATTN_IMPL"
  --batch-query "$BATCH_QUERY"
  --batch-passage "$BATCH_PASSAGE"
  --batch-score "$BATCH_SCORE"
  --num-workers "$NUM_WORKERS"
)
if [[ "$USE_V2_RETRIEVER" == "1" || "$USE_V2_RETRIEVER" == "true" ]]; then
  COMMON_FLAGS+=(--use-v2-retriever)
else
  COMMON_FLAGS+=(--no-use-v2-retriever)
fi
if [[ "$V2_DO_PADDING" == "1" || "$V2_DO_PADDING" == "true" ]]; then
  COMMON_FLAGS+=(--v2-do-padding)
else
  COMMON_FLAGS+=(--no-v2-do-padding)
fi
if [[ "$INCLUDE_MULTILINGUAL" == "1" || "$INCLUDE_MULTILINGUAL" == "true" ]]; then
  COMMON_FLAGS+=(--include-multilingual)
fi

run_eval() {
  local name="$1"
  local config="$2"
  local format="$3"
  local output="$4"
  echo "[strategy2_visionzip_eval] name=$name config=$config format=$format output=$output" | tee -a "$LOG_FILE"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_LIST" \
  PYTHONUNBUFFERED=1 \
  "$ACCELERATE_BIN" launch \
    --num_machines 1 \
    --num_processes "$NUM_GPUS" \
    --main_process_port "$MAIN_PROCESS_PORT" \
    --mixed_precision bf16 \
    -m colqwen_multigranularity.experiments.strategy2_visionzip.eval_visionzip \
    "${COMMON_FLAGS[@]}" \
    --eval-config "$config" \
    --dataset-format "$format" \
    --output-path "$output" \
    2>&1 | tee -a "$LOG_FILE"
}

run_eval vidore_v1 "$VIDORE_V1_CONFIG" beir "$OUT_DIR/vidore_v1.json"
run_eval vidore_v2 "$VIDORE_V2_CONFIG" beir "$OUT_DIR/vidore_v2.json"
run_eval mmeb "$MMEB_CONFIG" mmeb "$OUT_DIR/mmeb.json"

echo "[strategy2_visionzip_eval] done" | tee -a "$LOG_FILE"
