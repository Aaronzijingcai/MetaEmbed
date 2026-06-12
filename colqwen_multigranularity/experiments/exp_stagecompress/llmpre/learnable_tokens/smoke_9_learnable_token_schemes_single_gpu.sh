#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../../../.." && pwd)
TS=${TS:-$(date +%Y%m%d_%H%M%S)}
SUMMARY_DIR=${SUMMARY_DIR:-$PROJECT_DIR/experiments/exp_stagecompress/llmpre/learnable_tokens/runs/smoke_9_schemes_$TS}
SUMMARY_FILE=${SUMMARY_FILE:-$SUMMARY_DIR/summary.tsv}
mkdir -p "$SUMMARY_DIR"

export CUDA_DEVICE_LIST=${CUDA_DEVICE_LIST:-0}
export NUM_GPUS=${NUM_GPUS:-1}
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-0}
export MAX_STEPS=${MAX_STEPS:-1}
export SAVE_STEPS=${SAVE_STEPS:-1}
export TRAIN_BSZ=${TRAIN_BSZ:-1}
export EVAL_BSZ=${EVAL_BSZ:-1}
export INTERLEAVED_BSZ=${INTERLEAVED_BSZ:-1}
export EVAL_MODE=${EVAL_MODE:-smoke}
export EVAL_MAX_QUERIES=${EVAL_MAX_QUERIES:-2}
export EVAL_MAX_CORPUS=${EVAL_MAX_CORPUS:-8}
export BATCH_QUERY=${BATCH_QUERY:-1}
export BATCH_PASSAGE=${BATCH_PASSAGE:-1}
export BATCH_SCORE=${BATCH_SCORE:-4}
export NUM_WORKERS=${NUM_WORKERS:-0}
export ORTH_LAMBDA=${ORTH_LAMBDA:-0.0}
export ORTH_MODE=${ORTH_MODE:-per_stage}

printf "scheme\tplacement\tquery_tokens\tdoc_tokens\tmrl_groups\tstatus\trun_dir\n" > "$SUMMARY_FILE"

prefix_groups() {
  local q="$1" d="$2"
  IFS=, read -r q1 q2 q3 <<< "$q"
  IFS=, read -r d1 d2 d3 <<< "$d"
  local qg1=$((q1)) qg2=$((q1 + q2)) qg3=$((q1 + q2 + q3))
  local dg1=$((d1)) dg2=$((d1 + d2)) dg3=$((d1 + d2 + d3))
  echo "$qg1,$dg1,1.0;$qg2,$dg2,1.0;$qg3,$dg3,1.0"
}

run_stage_scheme() {
  local scheme="$1" q="$2" d="$3"
  local run_name="stage_interleaved_${scheme}_q${q//,/_}_d${d//,/_}_single_gpu_smoke_$TS"
  local run_dir="$PROJECT_DIR/experiments/exp_stagecompress/llmpre/learnable_tokens/runs/$run_name"
  local groups
  groups=$(prefix_groups "$q" "$d")
  echo "[smoke9] start $scheme stage q=$q d=$d groups=$groups run=$run_name"
  if RUN_NAME="$run_name" RUN_DIR="$run_dir" QUERY_STAGE_MRL_TOKENS="$q" DOC_STAGE_MRL_TOKENS="$d" MRL_GROUPS="$groups" \
    bash "$SCRIPT_DIR/smoke_stage_interleaved_single_gpu_train_eval.sh"; then
    printf "%s\tstage\t%s\t%s\t%s\tPASS\t%s\n" "$scheme" "$q" "$d" "$groups" "$run_dir" >> "$SUMMARY_FILE"
    echo "[smoke9] PASS $scheme"
  else
    local rc=$?
    printf "%s\tstage\t%s\t%s\t%s\tFAIL_%s\t%s\n" "$scheme" "$q" "$d" "$groups" "$rc" "$run_dir" >> "$SUMMARY_FILE"
    echo "[smoke9] FAIL $scheme rc=$rc"
  fi
}

run_tail_scheme() {
  local scheme="P8_tail"
  local q="2,4,8" d="8,16,32"
  local groups
  groups=$(prefix_groups "$q" "$d")
  local q_total=14 d_total=56
  local run_name="global_tail_${scheme}_q2_4_8_d8_16_32_single_gpu_smoke_$TS"
  local run_dir="$PROJECT_DIR/experiments/exp_stagecompress/llmpre/learnable_tokens/runs/$run_name"
  echo "[smoke9] start $scheme tail q_total=$q_total d_total=$d_total groups=$groups run=$run_name"
  if RUN_NAME="$run_name" RUN_DIR="$run_dir" NUM_QUERY_MRL_TOKENS="$q_total" NUM_DOC_MRL_TOKENS="$d_total" MRL_GROUPS="$groups" \
    bash "$SCRIPT_DIR/run_train.sh" && \
    RUN_DIR="$run_dir" ADAPTER_PATH="$run_dir/checkpoint-$MAX_STEPS" NUM_QUERY_MRL_TOKENS="$q_total" NUM_DOC_MRL_TOKENS="$d_total" MRL_GROUPS="$groups" \
    bash "$SCRIPT_DIR/eval_3sets.sh"; then
    printf "%s\ttail\t%s\t%s\t%s\tPASS\t%s\n" "$scheme" "$q" "$d" "$groups" "$run_dir" >> "$SUMMARY_FILE"
    echo "[smoke9] PASS $scheme"
  else
    local rc=$?
    printf "%s\ttail\t%s\t%s\t%s\tFAIL_%s\t%s\n" "$scheme" "$q" "$d" "$groups" "$rc" "$run_dir" >> "$SUMMARY_FILE"
    echo "[smoke9] FAIL $scheme rc=$rc"
  fi
}

run_stage_scheme P0_smoke 2,4,8 8,16,32
run_stage_scheme P1_T2_main 2,4,8 8,16,32
run_stage_scheme P2_T3_capacity_up 2,4,8 16,32,64
run_stage_scheme P3_T1_capacity_down 2,4,8 4,8,16
run_stage_scheme P4_T4_upper_bound 2,4,8 32,64,128
run_stage_scheme P5_Q1_query_down 1,2,4 8,16,32
run_stage_scheme P6_Q3_query_up 4,8,16 8,16,32
run_stage_scheme P7_Q4_symmetric 8,16,32 8,16,32
run_tail_scheme

echo "[smoke9] done summary=$SUMMARY_FILE"
cat "$SUMMARY_FILE"
