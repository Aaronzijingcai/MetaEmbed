#!/usr/bin/env bash
set +e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=${PROJECT_DIR:-/MURE-V2/code/MetaEmbed/colqwen_multigranularity}
cd "$PROJECT_DIR" || exit 1

export MURE_CACHE_ROOT=/MURE-V2/env/mure_cache/colqwen_multigranularity
export HF_HOME="$MURE_CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE=/MURE-V2/env/hf_datasets_cache
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

run_one() {
  local strategy="$1"
  local checkpoint="$2"
  local output_root="$3"
  local interaction="$4"
  local bi_lambda="$5"
  local vidore_output_name="$6"

  mkdir -p "$output_root"
  printf 'started=%s\ncheckpoint=%s\nstrategy=%s\nmode=isolated-per-dataset\ngpus=0,1,2,3,4,5,6,7\n' \
    "$(date -Is)" "$checkpoint" "$strategy" > "$output_root/isolated_status.txt"

  env \
    BENCHMARK=mmeb \
    CHECKPOINT="$checkpoint" \
    OUT_DIR="$output_root/mmeb_full" \
    MAXSIM_INTERACTION="$interaction" \
    MAXSIM_QUERY_AGG=mean \
    MAXSIM_QUERY_TOPK=48 \
    MAXSIM_BI_LAMBDA="$bi_lambda" \
    MAXSIM_ADAPTIVE_RATIO=1.5 \
    CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 \
    BUDGETS="128 128 128" \
    bash "$SCRIPT_DIR/run_isolated_benchmark.sh"
  local mmeb_status=$?
  printf 'mmeb_exit=%s\nmmeb_finished=%s\n' "$mmeb_status" "$(date -Is)" >> "$output_root/isolated_status.txt"

  env \
    BENCHMARK=vidore_v2 \
    CHECKPOINT="$checkpoint" \
    OUT_DIR="$output_root/vidore_v2/$vidore_output_name" \
    MAXSIM_INTERACTION="$interaction" \
    MAXSIM_QUERY_AGG=mean \
    MAXSIM_QUERY_TOPK=48 \
    MAXSIM_BI_LAMBDA="$bi_lambda" \
    MAXSIM_ADAPTIVE_RATIO=1.5 \
    CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 \
    BUDGETS="128 128 128" \
    bash "$SCRIPT_DIR/run_isolated_benchmark.sh"
  local vidore_status=$?
  printf 'vidore_v2_exit=%s\nvidore_v2_finished=%s\n' "$vidore_status" "$(date -Is)" >> "$output_root/isolated_status.txt"

  return $((mmeb_status != 0 || vidore_status != 0))
}

Q2D_CHECKPOINT=/MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/2026-07-08/runs/rhc_mmeb_vidore_q2d_topk48_mean_from_base/checkpoint-2000
ADAPTIVE_CHECKPOINT=/MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/2026-07-08/runs/rhc_mmeb_vidore_bi_topk48_adaptive_mean_from_base_formal_adaptive_bsz8_b128_20260719_2057/checkpoint-2000
Q2D_OUTPUT_ROOT="$(dirname "$Q2D_CHECKPOINT")/eval/checkpoint-2000_q2d_topk48_mean"
ADAPTIVE_OUTPUT_ROOT="$(dirname "$ADAPTIVE_CHECKPOINT")/eval/checkpoint-2000_adaptive_bidirectional_topk48_mean"

echo "[isolated-queue] q2d start $(date -Is)"
run_one q2d_topk48_mean "$Q2D_CHECKPOINT" "$Q2D_OUTPUT_ROOT" q2d_query_topk 0.5 q2d_topk_mean48
q2d_status=$?
echo "[isolated-queue] q2d exit=$q2d_status $(date -Is)"

echo "[isolated-queue] adaptive start $(date -Is)"
run_one adaptive_bidirectional_topk48_mean "$ADAPTIVE_CHECKPOINT" "$ADAPTIVE_OUTPUT_ROOT" bi_query_topk_adaptive 0.8 bi_topk_mean48_adaptive_lam08
adaptive_status=$?
echo "[isolated-queue] adaptive exit=$adaptive_status $(date -Is)"

echo "[isolated-queue] all done q2d=$q2d_status adaptive=$adaptive_status $(date -Is)"
exit $((q2d_status != 0 || adaptive_status != 0))
