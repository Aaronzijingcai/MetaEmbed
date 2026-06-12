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
CHECKPOINT=${1:-${CHECKPOINT:-}}
OUT_DIR=${OUT_DIR:-$PROJECT_DIR/runs/eval/strategy2_visionzip_3sets}
RUN_NAME=${RUN_NAME:-strategy2_visionzip_eval_3sets_$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-$PROJECT_DIR/runs/logs/${RUN_NAME}.log}
EVAL_CKPTS=${EVAL_CKPTS:-}
RESUME_EXISTING=${RESUME_EXISTING:-0}
FORCE_RERUN=${FORCE_RERUN:-0}

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
if [[ -z "$CHECKPOINT" && -z "$EVAL_CKPTS" ]]; then
  echo "Set CHECKPOINT, pass a checkpoint path as the first argument, or set EVAL_CKPTS." >&2
  exit 2
fi
if [[ "$VISIONZIP_ATTENTION_SOURCE" == "visual_attn" && "$ATTN_IMPL" != "eager" ]]; then
  echo "VISIONZIP_ATTENTION_SOURCE=visual_attn requires ATTN_IMPL=eager, got: $ATTN_IMPL" >&2
  exit 2
fi
if [[ "$NUM_GPUS" -gt 1 && "$NUM_WORKERS" -gt 0 ]]; then
  echo "[strategy2_visionzip_eval] warning: NUM_GPUS=$NUM_GPUS and NUM_WORKERS=$NUM_WORKERS starts workers per rank; if DataLoader SIGBUS occurs, rerun with NUM_WORKERS=0." >&2
fi

mkdir -p "$OUT_DIR" "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"

json_is_valid() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  python3 - "$path" >/dev/null 2>&1 <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("r", encoding="utf-8") as f:
    data = json.load(f)
if not isinstance(data, dict) or not data:
    raise SystemExit(1)
PY
}

should_skip_eval() {
  local name="$1"
  local output="$2"
  if [[ "$RESUME_EXISTING" != "1" && "$RESUME_EXISTING" != "true" ]]; then
    return 1
  fi
  if [[ "$FORCE_RERUN" == "1" || "$FORCE_RERUN" == "true" ]]; then
    return 1
  fi
  if json_is_valid "$output"; then
    echo "[strategy2_visionzip_eval] skip_existing name=$name output=$output" | tee -a "$LOG_FILE"
    return 0
  fi
  if [[ -e "$output" ]]; then
    echo "[strategy2_visionzip_eval] existing output is not valid JSON; rerun name=$name output=$output" | tee -a "$LOG_FILE"
  fi
  return 1
}

run_eval() {
  local name="$1"
  local config="$2"
  local format="$3"
  local output="$4"
  shift 4
  if should_skip_eval "$name" "$output"; then
    return 0
  fi
  echo "[strategy2_visionzip_eval] name=$name config=$config format=$format output=$output" | tee -a "$LOG_FILE"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_LIST" \
  PYTHONUNBUFFERED=1 \
  "$ACCELERATE_BIN" launch \
    --num_machines 1 \
    --num_processes "$NUM_GPUS" \
    --main_process_port "$MAIN_PROCESS_PORT" \
    --mixed_precision bf16 \
    -m colqwen_multigranularity.experiments.strategy2_visionzip.eval_visionzip \
    "$@" \
    --eval-config "$config" \
    --dataset-format "$format" \
    --output-path "$output" \
    2>&1 | tee -a "$LOG_FILE"
}

run_checkpoint() {
  local checkpoint="$1"
  local strategy2_visionzip_path="${VISIONZIP_PATH:-$checkpoint}"
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
  if [[ ! -f "$strategy2_visionzip_path/strategy2_visionzip_config.json" ]]; then
    echo "strategy2_visionzip_config.json not found under VISIONZIP_PATH: $strategy2_visionzip_path" >&2
    exit 2
  fi
  if grep -q '"attention_source"[[:space:]]*:[[:space:]]*"visual_attn"' "$strategy2_visionzip_path/strategy2_visionzip_config.json" && [[ "$ATTN_IMPL" != "eager" ]]; then
    echo "checkpoint uses attention_source=visual_attn, so ATTN_IMPL must be eager; got: $ATTN_IMPL" >&2
    exit 2
  fi

  mkdir -p "$output_dir"

  local common_flags=(
    --model-name-or-path "$MODEL_PATH"
    --processor-name-or-path "$PROCESSOR_PATH"
    --adapter-path "$checkpoint"
    --strategy2_visionzip-path "$strategy2_visionzip_path"
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
    echo "[strategy2_visionzip_eval] checkpoint=$checkpoint"
    echo "[strategy2_visionzip_eval] strategy2_visionzip_path=$strategy2_visionzip_path"
    echo "[strategy2_visionzip_eval] out_dir=$output_dir"
    echo "[strategy2_visionzip_eval] cuda=$CUDA_DEVICE_LIST num_gpus=$NUM_GPUS batch_query=$BATCH_QUERY batch_passage=$BATCH_PASSAGE batch_score=$BATCH_SCORE"
    echo "[strategy2_visionzip_eval] resume_existing=$RESUME_EXISTING force_rerun=$FORCE_RERUN"
    echo "[strategy2_visionzip_eval] attn_impl=$ATTN_IMPL crop_resize_mode=$CROP_RESIZE_MODE"
    echo "[strategy2_visionzip_eval] fallback_scope=$VISIONZIP_SCOPE stages=$VISIONZIP_STAGES budgets=${VISIONZIP_BUDGETS[*]} attention_source=$VISIONZIP_ATTENTION_SOURCE"
    echo "[strategy2_visionzip_eval] config is loaded from strategy2_visionzip_path when strategy2_visionzip_config.json exists"
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

echo "[strategy2_visionzip_eval] done" | tee -a "$LOG_FILE"
