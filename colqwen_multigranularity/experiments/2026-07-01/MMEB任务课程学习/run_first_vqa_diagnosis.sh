#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# First diagnostic run:
# continue sym160 checkpoint for 500 steps on trainable VQA-hard subsets only.

BASE_CHECKPOINT=${BASE_CHECKPOINT:-$SCRIPT_DIR/../MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000}
DIAG_NAME=${DIAG_NAME:-vqa_hard}
CONTINUE_STEPS=${CONTINUE_STEPS:-500}
SUBSET_CONFIG=${SUBSET_CONFIG:-$SCRIPT_DIR/configs/train_vqa_hard.yaml}
RUN_NAME=${RUN_NAME:-taskcurr_vqa_hard_from_sym160_s${CONTINUE_STEPS}}

BASE_CHECKPOINT="$BASE_CHECKPOINT" \
DIAG_NAME="$DIAG_NAME" \
CONTINUE_STEPS="$CONTINUE_STEPS" \
SUBSET_CONFIG="$SUBSET_CONFIG" \
RUN_NAME="$RUN_NAME" \
bash "$SCRIPT_DIR/run_continue_diagnosis.sh"

