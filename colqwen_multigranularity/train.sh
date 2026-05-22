#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR="$SCRIPT_DIR"

CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}
NUM_GPUS=${NUM_GPUS:-8}
MAX_STEPS=${MAX_STEPS:-4000}
SAVE_STEPS=${SAVE_STEPS:-500}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_DIR/runs/mrl_main_full_lora}
RUN_EVAL=${RUN_EVAL:-0}
RESUME_CKPT=${RESUME_CKPT:-}

usage() {
  echo "Usage: bash colqwen_multigranularity/train.sh [--gpus ids] [--num-gpus n] [--steps n] [--save-steps n] [--output-dir path] [--resume ckpt] [--run-eval]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --gpus) CUDA_DEVICE_LIST="$2"; shift 2 ;;
    --num-gpus) NUM_GPUS="$2"; shift 2 ;;
    --steps) MAX_STEPS="$2"; shift 2 ;;
    --save-steps) SAVE_STEPS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --resume) RESUME_CKPT="$2"; shift 2 ;;
    --run-eval) RUN_EVAL=1; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

CUDA_DEVICE_LIST="$CUDA_DEVICE_LIST" NUM_GPUS="$NUM_GPUS" MAX_STEPS="$MAX_STEPS" SAVE_STEPS="$SAVE_STEPS" OUTPUT_DIR="$OUTPUT_DIR" RUN_EVAL="$RUN_EVAL" RESUME_CKPT="$RESUME_CKPT" bash "$PROJECT_DIR/experiments/mrl_main/train.sh"
