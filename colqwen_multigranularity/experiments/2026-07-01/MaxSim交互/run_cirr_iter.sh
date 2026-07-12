#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMEB_DIR="$(cd "$SCRIPT_DIR/../MMEB全量" && pwd)"

CHECKPOINT=${1:-${CHECKPOINT:-$MMEB_DIR/runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000}}
RUN_DIR=$(cd "$(dirname "$CHECKPOINT")" && pwd)
BASE_OUT_DIR=${BASE_OUT_DIR:-$RUN_DIR/eval/maxsim_cirr_iter}
BASE_LOG_DIR=${BASE_LOG_DIR:-$RUN_DIR/logs}
BATCH_QUERY=${BATCH_QUERY:-32}
BATCH_PASSAGE=${BATCH_PASSAGE:-32}
BATCH_SCORE=${BATCH_SCORE:-96}
NUM_WORKERS=${NUM_WORKERS:-0}
AVG_METRIC=${AVG_METRIC:-recall_at_1}
ONLY_EVAL_KEYWORDS="MMEB-eval-CIRR-beir"

if [[ ! -d "$CHECKPOINT" ]]; then
  echo "checkpoint directory not found: $CHECKPOINT" >&2
  exit 2
fi

run_eval() {
  local name="$1"
  local interaction="$2"
  local query_agg="$3"
  local bi_lambda="$4"
  local topk="$5"
  local global_weight="$6"
  local hit_penalty_weight="$7"
  local hit_penalty_threshold="$8"
  local out_dir="$BASE_OUT_DIR/$name"
  local summary="$out_dir/mmeb_full_summary.json"

  if [[ -f "$summary" ]]; then
    echo "[maxsim_cirr] skip existing $name: $summary"
    return 0
  fi

  export CHECKPOINT
  export OUT_DIR="$out_dir"
  export LOG_DIR="$BASE_LOG_DIR"
  export LOG_FILE="$BASE_LOG_DIR/eval_maxsim_cirr_${name}_$(date +%Y%m%d_%H%M%S).log"
  export ONLY_EVAL_KEYWORDS
  export AVG_METRIC
  export MAXSIM_INTERACTION="$interaction"
  export MAXSIM_QUERY_AGG="$query_agg"
  export MAXSIM_BI_LAMBDA="$bi_lambda"
  export MAXSIM_QUERY_TOPK="$topk"
  export MAXSIM_GLOBAL_WEIGHT="$global_weight"
  export MAXSIM_HIT_PENALTY_WEIGHT="$hit_penalty_weight"
  export MAXSIM_HIT_PENALTY_THRESHOLD="$hit_penalty_threshold"
  export BATCH_QUERY BATCH_PASSAGE BATCH_SCORE NUM_WORKERS
  unset ASYM_QUERY_IMAGE_BUDGETS || true
  unset VIS_OUTPUT_DIR || true

  echo "[maxsim_cirr] running name=$name interaction=$interaction query_agg=$query_agg lambda=$bi_lambda topk=$topk global=$global_weight hit_penalty=$hit_penalty_weight threshold=$hit_penalty_threshold"
  bash "$MMEB_DIR/eval_mmeb_full.sh" "$CHECKPOINT"

  unset OUT_DIR LOG_FILE MAXSIM_INTERACTION MAXSIM_QUERY_AGG MAXSIM_BI_LAMBDA MAXSIM_QUERY_TOPK
  unset MAXSIM_GLOBAL_WEIGHT MAXSIM_HIT_PENALTY_WEIGHT MAXSIM_HIT_PENALTY_THRESHOLD
}

run_eval "legacy_q2d_sum_sym160" "q2d" "sum" "0.5" "0" "0.0" "0.0" "0.35"
run_eval "q2d_mean_sym160" "q2d" "mean" "0.5" "0" "0.0" "0.0" "0.35"
run_eval "q2d_query_topk64_sym160" "q2d_query_topk" "mean" "0.5" "64" "0.0" "0.0" "0.35"
run_eval "q2d_query_topk128_sym160" "q2d_query_topk" "mean" "0.5" "128" "0.0" "0.0" "0.35"
run_eval "bi_mean_lam05_sym160" "bi_mean" "mean" "0.5" "0" "0.0" "0.0" "0.35"
run_eval "bi_mean_lam07_sym160" "bi_mean" "mean" "0.7" "0" "0.0" "0.0" "0.35"
run_eval "bi_mean_lam09_sym160" "bi_mean" "mean" "0.9" "0" "0.0" "0.0" "0.35"
run_eval "bi_query_topk64_lam07_sym160" "bi_query_topk" "mean" "0.7" "64" "0.0" "0.0" "0.35"
run_eval "bi_query_topk128_lam07_sym160" "bi_query_topk" "mean" "0.7" "128" "0.0" "0.0" "0.35"
run_eval "global_q2d_mean_w02_sym160" "q2d" "mean" "0.5" "0" "0.2" "0.0" "0.35"
run_eval "global_q2d_mean_w05_sym160" "q2d" "mean" "0.5" "0" "0.5" "0.0" "0.35"
run_eval "global_bi_mean_lam07_w02_sym160" "bi_mean" "mean" "0.7" "0" "0.2" "0.0" "0.35"
run_eval "q2d_mean_hitpen_w02_sym160" "q2d" "mean" "0.5" "0" "0.0" "0.2" "0.35"
run_eval "q2d_mean_hitpen_w05_sym160" "q2d" "mean" "0.5" "0" "0.0" "0.5" "0.35"

python3 "$MMEB_DIR/compare_mmeb_runs.py" \
  "$BASE_OUT_DIR"/*/mmeb_full_summary.json \
  --output-path "$BASE_OUT_DIR/cirr_compare.md" || true

echo "[maxsim_cirr] done"
echo "[maxsim_cirr] compare: $BASE_OUT_DIR/cirr_compare.md"
