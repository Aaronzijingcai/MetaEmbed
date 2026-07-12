#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMEB_DIR="$(cd "$SCRIPT_DIR/../MMEB全量" && pwd)"

CHECKPOINT=${1:-${CHECKPOINT:-$MMEB_DIR/runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000}}
RUN_DIR=$(cd "$(dirname "$CHECKPOINT")" && pwd)
BASE_OUT_DIR=${BASE_OUT_DIR:-$RUN_DIR/eval/maxsim_server_b_weak_bi_global}
BASE_LOG_DIR=${BASE_LOG_DIR:-$RUN_DIR/logs}

BATCH_QUERY=${BATCH_QUERY:-16}
BATCH_PASSAGE=${BATCH_PASSAGE:-16}
BATCH_SCORE=${BATCH_SCORE:-64}
NUM_WORKERS=${NUM_WORKERS:-0}
AVG_METRIC=${AVG_METRIC:-recall_at_1}
QUERY_TOPK=${QUERY_TOPK:-64}

if [[ ! -d "$CHECKPOINT" ]]; then
  echo "checkpoint directory not found: $CHECKPOINT" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT/folder_homo.pt" ]]; then
  echo "folder_homo.pt not found under checkpoint: $CHECKPOINT" >&2
  exit 2
fi

WORST10_KEYWORDS=(
  MMEB-eval-FashionIQ-beir
  MMEB-eval-CIRR-beir
  MMEB-eval-Country211-beir
  MMEB-eval-GQA-beir
  MMEB-eval-ScienceQA-beir
  MMEB-eval-InfographicsVQA-beir
  MMEB-eval-A-OKVQA-beir
  MMEB-eval-Visual7W-beir
  MMEB-eval-OK-VQA-beir
  MMEB-eval-ChartQA-beir
)

RETENTION_KEYWORDS=(
  MMEB-eval-ImageNet-1K-beir
  MMEB-eval-VOC2007-beir
  MMEB-eval-VisualNews_i2t-beir
  MMEB-eval-VisualNews_t2i-beir
  MMEB-eval-MSCOCO_i2t-beir
  MMEB-eval-MSCOCO_t2i-beir
  MMEB-eval-WebQA-beir
  MMEB-eval-VisDial-beir
  MMEB-eval-RefCOCO-Matching-beir
)

write_plan() {
  mkdir -p "$BASE_OUT_DIR"
  cat > "$BASE_OUT_DIR/PLAN.md" <<PLAN
# Server B Weak-Bi / Global Scorer-Only Eval

Checkpoint: \`$CHECKPOINT\`

This launcher does not train or warm-start any model. It only changes the MMEB late-interaction scorer at eval time.

| Run | MAXSIM_INTERACTION | MAXSIM_BI_LAMBDA | MAXSIM_QUERY_TOPK | MAXSIM_GLOBAL_WEIGHT | Scope |
| --- | --- | ---: | ---: | ---: | --- |
| B1 \`bi_query_topk64_lam07\` | \`bi_query_topk\` | 0.7 | $QUERY_TOPK | 0.0 | worst10 + retention |
| B2 \`bi_query_topk64_lam09\` | \`bi_query_topk\` | 0.9 | $QUERY_TOPK | 0.0 | worst10 + retention |
| B3 \`q2d_query_topk64_global_w02\` | \`q2d_query_topk\` | 0.5 | $QUERY_TOPK | 0.2 | worst10 + retention |

Runtime defaults: BATCH_QUERY=$BATCH_QUERY, BATCH_PASSAGE=$BATCH_PASSAGE, BATCH_SCORE=$BATCH_SCORE, NUM_WORKERS=$NUM_WORKERS.
PLAN
}

run_eval() {
  local scope="$1"
  local keywords_name="$2"
  local name="$3"
  local interaction="$4"
  local bi_lambda="$5"
  local global_weight="$6"
  local out_dir="$BASE_OUT_DIR/$scope/$name"
  local summary="$out_dir/mmeb_full_summary.json"

  if [[ -f "$summary" ]]; then
    echo "[server_b_scorer] skip existing scope=$scope name=$name summary=$summary"
    return 0
  fi

  local -n keywords_ref="$keywords_name"
  export CHECKPOINT
  export OUT_DIR="$out_dir"
  export LOG_DIR="$BASE_LOG_DIR"
  export LOG_FILE="$BASE_LOG_DIR/eval_server_b_${scope}_${name}_$(date +%Y%m%d_%H%M%S).log"
  export ONLY_EVAL_KEYWORDS="${keywords_ref[*]}"
  export AVG_METRIC
  export MAXSIM_INTERACTION="$interaction"
  export MAXSIM_QUERY_AGG="mean"
  export MAXSIM_QUERY_TOPK="$QUERY_TOPK"
  export MAXSIM_BI_LAMBDA="$bi_lambda"
  export MAXSIM_GLOBAL_WEIGHT="$global_weight"
  export MAXSIM_HIT_PENALTY_WEIGHT="0.0"
  export MAXSIM_HIT_PENALTY_THRESHOLD="0.35"
  export BATCH_QUERY BATCH_PASSAGE BATCH_SCORE NUM_WORKERS
  unset ASYM_QUERY_IMAGE_BUDGETS || true
  unset VIS_OUTPUT_DIR || true

  echo "[server_b_scorer] $(date '+%F %T') running scope=$scope name=$name"
  echo "[server_b_scorer] CHECKPOINT=$CHECKPOINT"
  echo "[server_b_scorer] interaction=$interaction query_agg=mean topk=$QUERY_TOPK bi_lambda=$bi_lambda global_weight=$global_weight"
  echo "[server_b_scorer] keywords=${keywords_ref[*]}"
  bash "$MMEB_DIR/eval_mmeb_full.sh" "$CHECKPOINT"

  unset OUT_DIR LOG_FILE ONLY_EVAL_KEYWORDS MAXSIM_INTERACTION MAXSIM_QUERY_AGG
  unset MAXSIM_QUERY_TOPK MAXSIM_BI_LAMBDA MAXSIM_GLOBAL_WEIGHT
  unset MAXSIM_HIT_PENALTY_WEIGHT MAXSIM_HIT_PENALTY_THRESHOLD
}

run_scope() {
  local scope="$1"
  local keywords_name="$2"
  run_eval "$scope" "$keywords_name" "B1_bi_query_topk64_lam07" "bi_query_topk" "0.7" "0.0"
  run_eval "$scope" "$keywords_name" "B2_bi_query_topk64_lam09" "bi_query_topk" "0.9" "0.0"
  run_eval "$scope" "$keywords_name" "B3_q2d_query_topk64_global_w02" "q2d_query_topk" "0.5" "0.2"
}

compare_scope() {
  local scope="$1"
  local out_md="$BASE_OUT_DIR/${scope}_compare.md"
  python3 "$MMEB_DIR/compare_mmeb_runs.py" \
    "$BASE_OUT_DIR"/"$scope"/*/mmeb_full_summary.json \
    --output-path "$out_md" || true
  echo "[server_b_scorer] compare $scope: $out_md"
}

write_plan
run_scope "worst10" "WORST10_KEYWORDS"
compare_scope "worst10"
run_scope "retention" "RETENTION_KEYWORDS"
compare_scope "retention"

echo "[server_b_scorer] done BASE_OUT_DIR=$BASE_OUT_DIR"
