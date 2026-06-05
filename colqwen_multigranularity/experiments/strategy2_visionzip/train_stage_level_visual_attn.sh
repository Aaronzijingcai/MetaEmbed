#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# Stage-level VisionZip with Qwen2.5-VL vision encoder attention saliency.
# Requires eager attention because flash_attention_2/sdpa do not expose full attention weights.
# Attention scores are collected once from the selected visual block, aligned back to image_embeds,
# then split by stage. Dominant tokens are selected inside each stage and emitted in original index order.
export ATTN_IMPL=${ATTN_IMPL:-eager}
export VISIONZIP_SCOPE=${VISIONZIP_SCOPE:-stage}
export VISIONZIP_BUDGETS="${VISIONZIP_BUDGETS:-64 128 256}"
export VISIONZIP_CROP_BUDGET_MODE=${VISIONZIP_CROP_BUDGET_MODE:-proportional}
export VISIONZIP_DOMINANT_RATIO=${VISIONZIP_DOMINANT_RATIO:-0.75}
export VISIONZIP_ATTENTION_SOURCE=${VISIONZIP_ATTENTION_SOURCE:-visual_attn}
export VISIONZIP_VISUAL_ATTN_LAYER=${VISIONZIP_VISUAL_ATTN_LAYER:--2}
export VISIONZIP_TARGET_SELECT=${VISIONZIP_TARGET_SELECT:-uniform}
export VISIONZIP_MERGE_METRIC=${VISIONZIP_MERGE_METRIC:-cosine}
export PRESERVE_INPUT_RMS=${PRESERVE_INPUT_RMS:-1}
export MAX_STEPS=${MAX_STEPS:-4000}
export RUN_NAME=${RUN_NAME:-strategy2_visionzip_stage_visualattn_l${VISIONZIP_VISUAL_ATTN_LAYER}_b64-128-256_$(date +%Y%m%d_%H%M%S)}

exec "$SCRIPT_DIR/train_4gpu.sh"
