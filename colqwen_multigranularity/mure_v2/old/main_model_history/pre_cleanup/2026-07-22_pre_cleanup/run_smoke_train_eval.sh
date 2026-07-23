#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
TRAIN_WRAPPER="$SCRIPT_DIR/run_train.sh"
MMEB_EVAL="$PROJECT_DIR/experiments/2026-07-01/MMEB全量/eval_mmeb_full.sh"

INTERACTION_STRATEGY=${INTERACTION_STRATEGY:-q2d_topk48_mean}
SMOKE_STEPS=${SMOKE_STEPS:-3}
SMOKE_SAVE_STEPS=${SMOKE_SAVE_STEPS:-3}
SMOKE_SUFFIX=${SMOKE_SUFFIX:-main_model_smoke_$(date +%Y%m%d_%H%M%S)}
SMOKE_RUN_EVAL=${SMOKE_RUN_EVAL:-1}

case "$INTERACTION_STRATEGY" in
  standard_directed_maxsim|standard|maxsim)
    TRAIN_RUN_KEY=standard
    BASE_RUN_NAME=rhc_mmeb_vidore_q2d_sum_from_base
    EVAL_INTERACTION=q2d
    EVAL_SCORER_NAME=q2d
    EVAL_QUERY_AGG=sum
    EVAL_BI_LAMBDA=0.5
    ;;
  q2d_topk48_mean|q2d)
    TRAIN_RUN_KEY=q2d
    BASE_RUN_NAME=rhc_mmeb_vidore_q2d_topk48_mean_from_base
    EVAL_INTERACTION=q2d_query_topk
    EVAL_SCORER_NAME=q2d_query_topk48
    EVAL_QUERY_AGG=mean
    EVAL_BI_LAMBDA=0.5
    ;;
  adaptive_bidirectional_topk48_mean|adaptive)
    TRAIN_RUN_KEY=adaptive
    BASE_RUN_NAME=rhc_mmeb_vidore_bi_topk48_adaptive_mean_from_base
    EVAL_INTERACTION=bi_query_topk_adaptive
    EVAL_SCORER_NAME=bi_topk_mean48_adaptive_lam08
    EVAL_QUERY_AGG=mean
    EVAL_BI_LAMBDA=0.8
    ;;
  *)
    echo "[main_model_smoke] unknown INTERACTION_STRATEGY=$INTERACTION_STRATEGY" >&2
    exit 2
    ;;
esac

if [[ ! -f "$TRAIN_WRAPPER" ]]; then
  echo "[main_model_smoke] missing train wrapper: $TRAIN_WRAPPER" >&2
  exit 2
fi
if [[ ! -f "$MMEB_EVAL" ]]; then
  echo "[main_model_smoke] missing MMEB eval backend: $MMEB_EVAL" >&2
  exit 2
fi

echo "[main_model_smoke] strategy=$INTERACTION_STRATEGY suffix=$SMOKE_SUFFIX"
INTERACTION_STRATEGY="$TRAIN_RUN_KEY" \
MAX_STEPS="$SMOKE_STEPS" \
SAVE_STEPS="$SMOKE_SAVE_STEPS" \
LOGGING_STEPS=1 \
RUN_SUFFIX="$SMOKE_SUFFIX" \
SKIP_EVAL=1 \
TRAIN_BSZ="${TRAIN_BSZ:-2}" \
INTERLEAVED_BSZ="${INTERLEAVED_BSZ:-2}" \
EVAL_BSZ="${EVAL_BSZ:-1}" \
NUM_GPUS="${NUM_GPUS:-1}" \
CUDA_DEVICE_LIST="${CUDA_DEVICE_LIST:-0}" \
MAIN_PROCESS_PORT_BASE="${MAIN_PROCESS_PORT_BASE:-29890}" \
"$TRAIN_WRAPPER"

RUN_NAME="${BASE_RUN_NAME}_${SMOKE_SUFFIX}"
RUN_DIR="$PROJECT_DIR/experiments/2026-07-08/runs/$RUN_NAME"
CHECKPOINT="$RUN_DIR/checkpoint-$SMOKE_STEPS"

if [[ "${DRY_RUN:-0}" == "1" || "${DRY_RUN:-0}" == "true" || "${DRY_RUN:-0}" == "TRUE" ]]; then
  echo "[main_model_smoke] dry run done; expected checkpoint would be $CHECKPOINT"
  exit 0
fi

if [[ ! -d "$CHECKPOINT" ]]; then
  echo "[main_model_smoke] missing smoke checkpoint: $CHECKPOINT" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT/folder_homo.pt" ]]; then
  echo "[main_model_smoke] missing folder_homo.pt in checkpoint: $CHECKPOINT" >&2
  exit 1
fi

if [[ "$SMOKE_RUN_EVAL" == "0" || "$SMOKE_RUN_EVAL" == "false" || "$SMOKE_RUN_EVAL" == "FALSE" ]]; then
  echo "[main_model_smoke] skip eval by SMOKE_RUN_EVAL=$SMOKE_RUN_EVAL"
  exit 0
fi

echo "[main_model_smoke] eval checkpoint=$CHECKPOINT scorer=$EVAL_SCORER_NAME"
CHECKPOINT="$CHECKPOINT" \
OUT_DIR="$RUN_DIR/eval/smoke_$EVAL_SCORER_NAME" \
LOG_DIR="$RUN_DIR/logs" \
EVAL_MODE=smoke \
ONLY_EVAL_KEYWORDS="MMEB-eval-VisDial-beir" \
AVG_METRIC=recall_at_1 \
RUN_VIDORE=0 \
BATCH_QUERY="${BATCH_QUERY:-1}" \
BATCH_PASSAGE="${BATCH_PASSAGE:-1}" \
BATCH_SCORE="${BATCH_SCORE:-4}" \
NUM_WORKERS=0 \
CUDA_DEVICE_LIST="${CUDA_DEVICE_LIST:-0}" \
NUM_GPUS="${NUM_GPUS:-1}" \
MAXSIM_INTERACTION="$EVAL_INTERACTION" \
MAXSIM_QUERY_AGG="$EVAL_QUERY_AGG" \
MAXSIM_QUERY_TOPK=48 \
MAXSIM_BI_LAMBDA="$EVAL_BI_LAMBDA" \
bash "$MMEB_EVAL" "$CHECKPOINT"

echo "[main_model_smoke] done run=$RUN_NAME checkpoint=$CHECKPOINT"
