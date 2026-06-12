#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

# Stage-level VisionZip checkpoint trained with Qwen2.5-VL visual attention saliency.
# The checkpoint config is loaded from CHECKPOINT/strategy2_visionzip_config.json.
DEFAULT_CHECKPOINT="$PROJECT_DIR/runs/strategy2_visionzip_stage_visualattn_l-2_b64-128-256_20260530_201752"
CHECKPOINT=${1:-${CHECKPOINT:-$DEFAULT_CHECKPOINT}}

export CHECKPOINT
export ATTN_IMPL=${ATTN_IMPL:-eager}
export VISIONZIP_SCOPE=${VISIONZIP_SCOPE:-stage}
export VISIONZIP_BUDGETS="${VISIONZIP_BUDGETS:-64 128 256}"
export VISIONZIP_CROP_BUDGET_MODE=${VISIONZIP_CROP_BUDGET_MODE:-proportional}
export VISIONZIP_DOMINANT_RATIO=${VISIONZIP_DOMINANT_RATIO:-0.75}
export VISIONZIP_ATTENTION_SOURCE=${VISIONZIP_ATTENTION_SOURCE:-visual_attn}
export VISIONZIP_VISUAL_ATTN_LAYER=${VISIONZIP_VISUAL_ATTN_LAYER:--2}
export VISIONZIP_TARGET_SELECT=${VISIONZIP_TARGET_SELECT:-uniform}
export VISIONZIP_MERGE_METRIC=${VISIONZIP_MERGE_METRIC:-cosine}
export OUT_DIR=${OUT_DIR:-$PROJECT_DIR/runs/eval/$(basename "$CHECKPOINT")_3sets}
export RUN_NAME=${RUN_NAME:-$(basename "$CHECKPOINT")_eval_3sets_$(date +%Y%m%d_%H%M%S)}

exec "$SCRIPT_DIR/eval_3sets.sh" "$CHECKPOINT"
