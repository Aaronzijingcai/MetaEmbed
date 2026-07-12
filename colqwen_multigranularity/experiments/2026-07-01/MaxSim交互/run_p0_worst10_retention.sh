#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMEB_DIR="$(cd "$SCRIPT_DIR/../MMEB全量" && pwd)"

CHECKPOINT=${1:-${CHECKPOINT:-$MMEB_DIR/runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000}}
RUN_DIR=$(cd "$(dirname "$CHECKPOINT")" && pwd)
BASE_OUT_DIR=${BASE_OUT_DIR:-$RUN_DIR/eval/maxsim_p0_worst10_retention}
BASE_LOG_DIR=${BASE_LOG_DIR:-$RUN_DIR/logs}
QUERY_TOPK=${QUERY_TOPK:-64}
BATCH_QUERY=${BATCH_QUERY:-16}
BATCH_PASSAGE=${BATCH_PASSAGE:-16}
BATCH_SCORE=${BATCH_SCORE:-64}
NUM_WORKERS=${NUM_WORKERS:-0}

if [[ ! -d "$CHECKPOINT" ]]; then
  echo "checkpoint directory not found: $CHECKPOINT" >&2
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

run_eval() {
  local scope="$1"
  local keywords_name="$2"
  local name="$3"
  local interaction="$4"
  local query_agg="$5"
  local topk="$6"
  local out_dir="$BASE_OUT_DIR/$scope/$name"
  local summary="$out_dir/mmeb_full_summary.json"

  if [[ -f "$summary" ]]; then
    echo "[maxsim_p0] skip existing $scope/$name: $summary"
    return 0
  fi

  local -n keywords_ref="$keywords_name"
  export CHECKPOINT
  export OUT_DIR="$out_dir"
  export LOG_DIR="$BASE_LOG_DIR"
  export LOG_FILE="$BASE_LOG_DIR/eval_maxsim_p0_${scope}_${name}_$(date +%Y%m%d_%H%M%S).log"
  export AVG_METRIC="${AVG_METRIC:-recall_at_1}"
  export MAXSIM_INTERACTION="$interaction"
  export MAXSIM_QUERY_AGG="$query_agg"
  export MAXSIM_QUERY_TOPK="$topk"
  export MAXSIM_BI_LAMBDA="${MAXSIM_BI_LAMBDA:-0.5}"
  export MAXSIM_GLOBAL_WEIGHT="0.0"
  export BATCH_QUERY BATCH_PASSAGE BATCH_SCORE NUM_WORKERS
  export ONLY_EVAL_KEYWORDS="${keywords_ref[*]}"
  unset ASYM_QUERY_IMAGE_BUDGETS || true

  echo "[maxsim_p0] running scope=$scope name=$name interaction=$interaction query_agg=$query_agg topk=$topk"
  echo "[maxsim_p0] keywords=${keywords_ref[*]}"
  bash "$MMEB_DIR/eval_mmeb_full.sh" "$CHECKPOINT"

  unset OUT_DIR LOG_FILE ONLY_EVAL_KEYWORDS MAXSIM_INTERACTION MAXSIM_QUERY_AGG MAXSIM_QUERY_TOPK MAXSIM_GLOBAL_WEIGHT
}

run_scope() {
  local scope="$1"
  local keywords_name="$2"
  run_eval "$scope" "$keywords_name" "q2d_mean_sym160" "q2d" "mean" "0"
  run_eval "$scope" "$keywords_name" "bi_mean_sym160" "bi_mean" "mean" "0"
  run_eval "$scope" "$keywords_name" "q2d_query_topk64_sym160" "q2d_query_topk" "mean" "$QUERY_TOPK"
  run_eval "$scope" "$keywords_name" "bi_query_topk64_sym160" "bi_query_topk" "mean" "$QUERY_TOPK"
}

run_scope "worst10" "WORST10_KEYWORDS"
run_scope "retention" "RETENTION_KEYWORDS"

python3 "$MMEB_DIR/compare_mmeb_runs.py" \
  "$BASE_OUT_DIR"/worst10/*/mmeb_full_summary.json \
  --output-path "$BASE_OUT_DIR/worst10_compare.md" || true
python3 "$MMEB_DIR/compare_mmeb_runs.py" \
  "$BASE_OUT_DIR"/retention/*/mmeb_full_summary.json \
  --output-path "$BASE_OUT_DIR/retention_compare.md" || true

echo "[maxsim_p0] done"
echo "[maxsim_p0] worst10 compare: $BASE_OUT_DIR/worst10_compare.md"
echo "[maxsim_p0] retention compare: $BASE_OUT_DIR/retention_compare.md"
