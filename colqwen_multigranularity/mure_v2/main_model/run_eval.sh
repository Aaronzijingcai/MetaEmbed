#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CHECKPOINT=${CHECKPOINT:?CHECKPOINT is required}
BENCHMARK=${BENCHMARK:?BENCHMARK must be mmeb or vidore_v2}
CHECKPOINT_NAME=$(basename "$CHECKPOINT")
RUN_NAME=$(basename "$(dirname "$CHECKPOINT")")
OUT_DIR=${OUT_DIR:-$SCRIPT_DIR/evaluations/$RUN_NAME/$CHECKPOINT_NAME/$BENCHMARK}

export CHECKPOINT BENCHMARK OUT_DIR
export MAXSIM_INTERACTION=${MAXSIM_INTERACTION:-bi_query_topk_adaptive}
export MAXSIM_QUERY_AGG=${MAXSIM_QUERY_AGG:-mean}
export MAXSIM_QUERY_TOPK=${MAXSIM_QUERY_TOPK:-48}
export MAXSIM_BI_LAMBDA=${MAXSIM_BI_LAMBDA:-0.8}
export BUDGETS=${BUDGETS:-"128 128 128"}

exec "$SCRIPT_DIR/tools/run_isolated_benchmark.sh"
