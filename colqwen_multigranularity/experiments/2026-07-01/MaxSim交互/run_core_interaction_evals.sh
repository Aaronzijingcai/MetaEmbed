#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMEB_DIR="$(cd "$SCRIPT_DIR/../MMEB全量" && pwd)"

CHECKPOINT=${1:-${CHECKPOINT:-$MMEB_DIR/runs/folder_homo_mmeb_full_train_b160_160_160_4k/checkpoint-4000}}
RUN_DIR=$(cd "$(dirname "$CHECKPOINT")" && pwd)
BASE_OUT_DIR=${BASE_OUT_DIR:-$RUN_DIR/eval/maxsim_interaction}
BASE_LOG_DIR=${BASE_LOG_DIR:-$RUN_DIR/logs}
INCLUDE_P1=${INCLUDE_P1:-1}
INCLUDE_P2=${INCLUDE_P2:-0}
SKIP_LEGACY=${SKIP_LEGACY:-0}

if [[ ! -d "$CHECKPOINT" ]]; then
  echo "checkpoint directory not found: $CHECKPOINT" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT/folder_homo.pt" ]]; then
  echo "folder_homo.pt not found under checkpoint: $CHECKPOINT" >&2
  echo "This launcher is for FolderHomo checkpoints only." >&2
  exit 2
fi

run_eval() {
  local name="$1"
  local interaction="$2"
  local query_agg="$3"
  local bi_lambda="$4"
  local topk="$5"
  local global_weight="$6"

  export CHECKPOINT
  export AVG_METRIC="${AVG_METRIC:-recall_at_1}"
  export MAXSIM_INTERACTION="$interaction"
  export MAXSIM_QUERY_AGG="$query_agg"
  export MAXSIM_BI_LAMBDA="$bi_lambda"
  export MAXSIM_QUERY_TOPK="$topk"
  export MAXSIM_GLOBAL_WEIGHT="$global_weight"
  export OUT_DIR="$BASE_OUT_DIR/$name"
  export LOG_DIR="$BASE_LOG_DIR"
  export LOG_FILE="$BASE_LOG_DIR/eval_maxsim_${name}_$(date +%Y%m%d_%H%M%S).log"
  unset ASYM_QUERY_IMAGE_BUDGETS || true

  echo "[maxsim_interaction] running $name"
  echo "[maxsim_interaction] interaction=$interaction query_agg=$query_agg bi_lambda=$bi_lambda topk=$topk global_weight=$global_weight"
  bash "$MMEB_DIR/eval_mmeb_full.sh" "$CHECKPOINT"

  unset OUT_DIR LOG_FILE MAXSIM_INTERACTION MAXSIM_QUERY_AGG MAXSIM_BI_LAMBDA MAXSIM_QUERY_TOPK MAXSIM_GLOBAL_WEIGHT
}

# P0: scorer-only ablations with fixed FolderHomo sym160 budget.
if [[ "$SKIP_LEGACY" != "1" ]]; then
  run_eval "legacy_q2d_sum_sym160" "q2d" "sum" "0.5" "0" "0.0"
fi
run_eval "q2d_mean_sym160" "q2d" "mean" "0.5" "0" "0.0"
run_eval "bi_mean_sym160" "bi_mean" "mean" "0.5" "0" "0.0"

if [[ "$INCLUDE_P1" == "1" ]]; then
  run_eval "global_local_bi_mean_sym160" "bi_mean" "mean" "0.5" "0" "${MAXSIM_GLOBAL_WEIGHT_P1:-0.2}"
  run_eval "bi_topk_mean_sym160" "bi_topk_mean" "mean" "0.5" "${MAXSIM_TOPK_P1:-4}" "0.0"
fi

if [[ "$INCLUDE_P2" == "1" ]]; then
  run_eval "bi_lse_sym160" "bi_lse" "mean" "0.5" "0" "0.0"
fi

echo "[maxsim_interaction] done"
echo "[maxsim_interaction] summaries can be compared with:"
echo "python3 $MMEB_DIR/compare_mmeb_runs.py $BASE_OUT_DIR/*/mmeb_full_summary.json"
