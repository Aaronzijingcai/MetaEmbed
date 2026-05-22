#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/MURE-V2/code/MetaEmbed
RUN_DIR=${RUN_DIR:-$REPO_ROOT/colqwen_multigranularity/runs/softassign_textquery_focus_4gpu_all_kr0.25_8k_20260519_120509}
OUT_DIR=${OUT_DIR:-$REPO_ROOT/colqwen_multigranularity/runs/eval/textquery_focus_softassign_8k_5000-8000}
RUN_NAME=${RUN_NAME:-softassign_textquery_focus_eval_5000-8000_$(date +%Y%m%d_%H%M%S)}
MMEB_EVAL_CONFIG=${MMEB_EVAL_CONFIG:-$REPO_ROOT/colqwen_multigranularity/configs/eval/test_data_mast_mmeb_v3.yaml}

cd "$REPO_ROOT"

EVAL_CKPTS="${EVAL_CKPTS:-$RUN_DIR/checkpoint-5000 $RUN_DIR/checkpoint-6000 $RUN_DIR/checkpoint-7000 $RUN_DIR/checkpoint-8000}" \
OUT_DIR="$OUT_DIR" \
RUN_NAME="$RUN_NAME" \
MMEB_EVAL_CONFIG="$MMEB_EVAL_CONFIG" \
CUDA_DEVICE_LIST="${CUDA_DEVICE_LIST:-0}" \
NUM_GPUS="${NUM_GPUS:-1}" \
BATCH_QUERY="${BATCH_QUERY:-1}" \
BATCH_PASSAGE="${BATCH_PASSAGE:-1}" \
BATCH_SCORE="${BATCH_SCORE:-4}" \
SOFTASSIGN_KEEP_RATIO="${SOFTASSIGN_KEEP_RATIO:-0.25}" \
bash "$REPO_ROOT/colqwen_multigranularity/experiments/strategy1_softassign/eval_textquery_focus.sh"
