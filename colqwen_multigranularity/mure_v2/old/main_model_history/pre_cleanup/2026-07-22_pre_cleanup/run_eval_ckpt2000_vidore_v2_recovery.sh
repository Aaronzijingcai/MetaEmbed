#!/usr/bin/env bash
set +e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=${PROJECT_DIR:-/MURE-V2/code/MetaEmbed/colqwen_multigranularity}
ACTIVE_PID_FILE="$SCRIPT_DIR/logs/eval_ckpt2000_fault_isolated_world8_20260721.pid"
RUNNER=${RUNNER:-$SCRIPT_DIR/run_isolated_benchmark_v2_fixed.sh}

cd "$PROJECT_DIR" || exit 1

export MURE_CACHE_ROOT=/MURE-V2/env/mure_cache/colqwen_multigranularity
export HF_HOME="$MURE_CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE=/MURE-V2/env/hf_datasets_cache
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

if [[ -f "$ACTIVE_PID_FILE" ]]; then
  active_pid=$(cat "$ACTIVE_PID_FILE")
  while kill -0 "$active_pid" 2>/dev/null; do
    echo "[v2-recovery] waiting for active queue pid=$active_pid $(date -Is)"
    sleep 30
  done
fi

run_v2() {
  local checkpoint="$1"
  local out_dir="$2"
  local interaction="$3"
  local bi_lambda="$4"

  env \
    BENCHMARK=vidore_v2 \
    CHECKPOINT="$checkpoint" \
    OUT_DIR="$out_dir" \
    MAXSIM_INTERACTION="$interaction" \
    MAXSIM_QUERY_AGG=mean \
    MAXSIM_QUERY_TOPK=48 \
    MAXSIM_BI_LAMBDA="$bi_lambda" \
    MAXSIM_ADAPTIVE_RATIO=1.5 \
    CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 \
    EVAL_WORLD_SIZE=8 \
    BUDGETS="128 128 128" \
    MAX_RETRIES=1 \
    bash "$RUNNER"
}

Q2D_CHECKPOINT=/MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/2026-07-08/runs/rhc_mmeb_vidore_q2d_topk48_mean_from_base/checkpoint-2000
ADAPTIVE_CHECKPOINT=/MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/2026-07-08/runs/rhc_mmeb_vidore_bi_topk48_adaptive_mean_from_base_formal_adaptive_bsz8_b128_20260719_2057/checkpoint-2000
Q2D_OUT="$(dirname "$Q2D_CHECKPOINT")/eval/checkpoint-2000_q2d_topk48_mean/vidore_v2/q2d_topk_mean48"
ADAPTIVE_OUT="$(dirname "$ADAPTIVE_CHECKPOINT")/eval/checkpoint-2000_adaptive_bidirectional_topk48_mean/vidore_v2/bi_topk_mean48_adaptive_lam08"

echo "[v2-recovery] q2d start $(date -Is)"
run_v2 "$Q2D_CHECKPOINT" "$Q2D_OUT" q2d_query_topk 0.5
q2d_status=$?
echo "[v2-recovery] q2d exit=$q2d_status $(date -Is)"

echo "[v2-recovery] adaptive start $(date -Is)"
run_v2 "$ADAPTIVE_CHECKPOINT" "$ADAPTIVE_OUT" bi_query_topk_adaptive 0.8
adaptive_status=$?
echo "[v2-recovery] adaptive exit=$adaptive_status $(date -Is)"

exit $((q2d_status != 0 || adaptive_status != 0))
