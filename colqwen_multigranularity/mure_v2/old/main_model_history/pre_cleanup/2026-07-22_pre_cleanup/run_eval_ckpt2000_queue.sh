#!/usr/bin/env bash
set +e

PROJECT_DIR=${PROJECT_DIR:-/MURE-V2/code/MetaEmbed/colqwen_multigranularity}
cd "$PROJECT_DIR" || exit 1

export MURE_CACHE_ROOT=/MURE-V2/env/mure_cache/colqwen_multigranularity
export HF_HOME="$MURE_CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE=/MURE-V2/env/hf_datasets_cache
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TMPDIR="$MURE_CACHE_ROOT/tmp"
mkdir -p "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TMPDIR"

run_one() {
  local strategy="$1"
  local checkpoint="$2"
  local output_root="$3"
  local interaction="$4"
  local scorer="$5"
  local bi_lambda="$6"

  mkdir -p "$output_root/logs"
  printf 'started=%s\ncheckpoint=%s\nstrategy=%s\ngpus=0,1,2,3,4,5,6,7\nhf_datasets_cache=%s\n' \
    "$(date -Is)" "$checkpoint" "$strategy" "$HF_DATASETS_CACHE" > "$output_root/status.txt"

  env \
    CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 \
    NUM_GPUS=8 \
    OUT_DIR="$output_root/mmeb_full" \
    LOG_DIR="$output_root/logs" \
    LOG_FILE="$output_root/logs/mmeb_full.log" \
    BUDGETS="128 128 128" \
    MAXSIM_INTERACTION="$interaction" \
    MAXSIM_QUERY_AGG=mean \
    MAXSIM_QUERY_TOPK=48 \
    MAXSIM_BI_LAMBDA="$bi_lambda" \
    MAXSIM_ADAPTIVE_RATIO=1.5 \
    EVAL_MODE=full \
    bash experiments/2026-07-01/MMEB全量/eval_mmeb_full.sh "$checkpoint"
  local mmeb_status=$?
  printf 'mmeb_exit=%s\nmmeb_finished=%s\n' "$mmeb_status" "$(date -Is)" >> "$output_root/status.txt"

  env \
    CHECKPOINT="$checkpoint" \
    OUT_DIR="$output_root" \
    LOG_DIR="$output_root/logs" \
    SCORERS="$scorer" \
    RUN_MMEB=0 \
    RUN_VIDORE=1 \
    CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 \
    NUM_GPUS=8 \
    BUDGETS="128 128 128" \
    BATCH_QUERY=32 \
    BATCH_PASSAGE=32 \
    BATCH_SCORE=128 \
    EVAL_MODE=full \
    bash experiments/2026-07-01/MaxSim交互/run_eval_vidorev2_worst10.sh
  local vidore_status=$?
  printf 'vidore_v2_exit=%s\nvidore_v2_finished=%s\n' "$vidore_status" "$(date -Is)" >> "$output_root/status.txt"

  return $((mmeb_status != 0 || vidore_status != 0))
}

Q2D_CHECKPOINT=/MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/2026-07-08/runs/rhc_mmeb_vidore_q2d_topk48_mean_from_base/checkpoint-2000
ADAPTIVE_CHECKPOINT=/MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/2026-07-08/runs/rhc_mmeb_vidore_bi_topk48_adaptive_mean_from_base_formal_adaptive_bsz8_b128_20260719_2057/checkpoint-2000

Q2D_OUTPUT_ROOT="$(dirname "$Q2D_CHECKPOINT")/eval/checkpoint-2000_q2d_topk48_mean"
ADAPTIVE_OUTPUT_ROOT="$(dirname "$ADAPTIVE_CHECKPOINT")/eval/checkpoint-2000_adaptive_bidirectional_topk48_mean"

echo "[queue] q2d start $(date -Is)"
run_one q2d_topk48_mean "$Q2D_CHECKPOINT" "$Q2D_OUTPUT_ROOT" q2d_query_topk q2d_topk_mean48 0.5
q2d_status=$?
echo "[queue] q2d exit=$q2d_status $(date -Is)"

echo "[queue] adaptive start $(date -Is)"
run_one adaptive_bidirectional_topk48_mean "$ADAPTIVE_CHECKPOINT" "$ADAPTIVE_OUTPUT_ROOT" bi_query_topk_adaptive bi_topk_mean48_adaptive_lam08 0.8
adaptive_status=$?
echo "[queue] adaptive exit=$adaptive_status $(date -Is)"

echo "[queue] all done q2d=$q2d_status adaptive=$adaptive_status $(date -Is)"
exit $((q2d_status != 0 || adaptive_status != 0))
