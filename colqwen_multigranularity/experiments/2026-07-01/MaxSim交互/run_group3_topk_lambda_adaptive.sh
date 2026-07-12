#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMEB_DIR="$(cd "$SCRIPT_DIR/../MMEB全量" && pwd)"

CHECKPOINT=${1:-${CHECKPOINT:-$MMEB_DIR/runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000}}
RUN_DIR=$(cd "$(dirname "$CHECKPOINT")" && pwd)
BASE_OUT_DIR=${BASE_OUT_DIR:-$RUN_DIR/eval/maxsim_group3_topk_lambda_adaptive}
BASE_LOG_DIR=${BASE_LOG_DIR:-$RUN_DIR/logs}

BATCH_QUERY=${BATCH_QUERY:-16}
BATCH_PASSAGE=${BATCH_PASSAGE:-16}
BATCH_SCORE=${BATCH_SCORE:-64}
NUM_WORKERS=${NUM_WORKERS:-0}
AVG_METRIC=${AVG_METRIC:-recall_at_1}

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
# MaxSim Group 3: topK / lambda / adaptive scorer-only eval

Checkpoint: \`$CHECKPOINT\`

This launcher does not train or warm-start any model. It only changes the MMEB late-interaction scorer at eval time.

| Run | MAXSIM_INTERACTION | MAXSIM_BI_LAMBDA | MAXSIM_QUERY_TOPK | Scope |
| --- | --- | ---: | ---: | --- |
| \`bi_query_topk32_lam05\` | \`bi_query_topk\` | 0.5 | 32 | worst10 + retention |
| \`bi_query_topk32_lam07\` | \`bi_query_topk\` | 0.7 | 32 | worst10 + retention |
| \`bi_query_topk64_lam05\` | \`bi_query_topk\` | 0.5 | 64 | worst10 + retention |
| \`bi_query_topk64_lam07\` | \`bi_query_topk\` | 0.7 | 64 | worst10 + retention |
| \`bi_query_topk32_adaptive_lam08\` | \`bi_query_topk_adaptive\` | 0.8 | 32 | worst10 + retention |
| \`bi_query_topk64_adaptive_lam08\` | \`bi_query_topk_adaptive\` | 0.8 | 64 | worst10 + retention |

Runtime defaults: BATCH_QUERY=$BATCH_QUERY, BATCH_PASSAGE=$BATCH_PASSAGE, BATCH_SCORE=$BATCH_SCORE, NUM_WORKERS=$NUM_WORKERS.
PLAN
}

run_eval() {
  local scope="$1"
  local keywords_name="$2"
  local name="$3"
  local interaction="$4"
  local bi_lambda="$5"
  local topk="$6"
  local out_dir="$BASE_OUT_DIR/$scope/$name"
  local summary="$out_dir/mmeb_full_summary.json"
  local json_out="$out_dir/mmeb_full.json"

  if [[ -f "$summary" ]]; then
    echo "[maxsim_group3] skip existing scope=$scope name=$name summary=$summary"
    return 0
  fi
  if [[ -f "$json_out" ]]; then
    echo "[maxsim_group3] found existing json without summary, rebuilding summary: $json_out"
    python3 "$MMEB_DIR/analyze_mmeb.py" "$json_out" \
      --metric "$AVG_METRIC" \
      --output-path "$summary"
    return 0
  fi

  local -n keywords_ref="$keywords_name"
  export CHECKPOINT
  export OUT_DIR="$out_dir"
  export LOG_DIR="$BASE_LOG_DIR"
  export LOG_FILE="$BASE_LOG_DIR/eval_group3_${scope}_${name}_$(date +%Y%m%d_%H%M%S).log"
  export ONLY_EVAL_KEYWORDS="${keywords_ref[*]}"
  export AVG_METRIC
  export MAXSIM_INTERACTION="$interaction"
  export MAXSIM_QUERY_AGG="mean"
  export MAXSIM_QUERY_TOPK="$topk"
  export MAXSIM_BI_LAMBDA="$bi_lambda"
  export MAXSIM_GLOBAL_WEIGHT="0.0"
  export MAXSIM_HIT_PENALTY_WEIGHT="0.0"
  export MAXSIM_HIT_PENALTY_THRESHOLD="0.35"
  export BATCH_QUERY BATCH_PASSAGE BATCH_SCORE NUM_WORKERS
  unset ASYM_QUERY_IMAGE_BUDGETS || true
  unset VIS_OUTPUT_DIR || true

  echo "[maxsim_group3] $(date '+%F %T') running scope=$scope name=$name"
  echo "[maxsim_group3] CHECKPOINT=$CHECKPOINT"
  echo "[maxsim_group3] interaction=$interaction query_agg=mean topk=$topk bi_lambda=$bi_lambda global_weight=0.0"
  echo "[maxsim_group3] keywords=${keywords_ref[*]}"
  set +e
  bash "$MMEB_DIR/eval_mmeb_full.sh" "$CHECKPOINT"
  local status=$?
  set -e

  if [[ "$status" -ne 0 && ! -f "$json_out" ]]; then
    echo "[maxsim_group3] eval failed before writing json: scope=$scope name=$name status=$status" >&2
    exit "$status"
  fi
  if [[ ! -f "$summary" && -f "$json_out" ]]; then
    echo "[maxsim_group3] rebuilding summary after eval status=$status: $json_out"
    python3 "$MMEB_DIR/analyze_mmeb.py" "$json_out" \
      --metric "$AVG_METRIC" \
      --output-path "$summary"
  fi

  unset OUT_DIR LOG_FILE ONLY_EVAL_KEYWORDS MAXSIM_INTERACTION MAXSIM_QUERY_AGG
  unset MAXSIM_QUERY_TOPK MAXSIM_BI_LAMBDA MAXSIM_GLOBAL_WEIGHT
  unset MAXSIM_HIT_PENALTY_WEIGHT MAXSIM_HIT_PENALTY_THRESHOLD
}

run_scope() {
  local scope="$1"
  local keywords_name="$2"
  run_eval "$scope" "$keywords_name" "bi_query_topk32_lam05" "bi_query_topk" "0.5" "32"
  run_eval "$scope" "$keywords_name" "bi_query_topk32_lam07" "bi_query_topk" "0.7" "32"
  run_eval "$scope" "$keywords_name" "bi_query_topk64_lam05" "bi_query_topk" "0.5" "64"
  run_eval "$scope" "$keywords_name" "bi_query_topk64_lam07" "bi_query_topk" "0.7" "64"
  run_eval "$scope" "$keywords_name" "bi_query_topk32_adaptive_lam08" "bi_query_topk_adaptive" "0.8" "32"
  run_eval "$scope" "$keywords_name" "bi_query_topk64_adaptive_lam08" "bi_query_topk_adaptive" "0.8" "64"
}

write_plan
run_scope "worst10" "WORST10_KEYWORDS"
run_scope "retention" "RETENTION_KEYWORDS"

echo "[maxsim_group3] done BASE_OUT_DIR=$BASE_OUT_DIR"
