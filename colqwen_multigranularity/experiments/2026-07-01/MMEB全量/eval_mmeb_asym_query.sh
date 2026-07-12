#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

CHECKPOINT=${1:-${CHECKPOINT:-$SCRIPT_DIR/runs/folder_homo_mmeb_full_train_b160_160_160_4k/checkpoint-4000}}
ASYM_QUERY_BUDGET_SETS_STR=${ASYM_QUERY_BUDGET_SETS:-"80,80,80 40,40,40"}
read -r -a ASYM_QUERY_BUDGET_SETS_ARRAY <<< "$ASYM_QUERY_BUDGET_SETS_STR"
BASE_OUT_DIR=${BASE_OUT_DIR:-}
export MAXSIM_INTERACTION=${MAXSIM_INTERACTION:-q2d}
export MAXSIM_QUERY_AGG=${MAXSIM_QUERY_AGG:-mean}

if [[ ! -d "$CHECKPOINT" ]]; then
  echo "checkpoint directory not found: $CHECKPOINT" >&2
  exit 2
fi

for budget_set in "${ASYM_QUERY_BUDGET_SETS_ARRAY[@]}"; do
  IFS=',' read -r b1 b2 b3 <<< "$budget_set"
  if [[ -z "${b1:-}" || -z "${b2:-}" || -z "${b3:-}" ]]; then
    echo "Invalid ASYM_QUERY_BUDGET_SETS entry: $budget_set" >&2
    exit 2
  fi
  suffix="q${b1}_${b2}_${b3}_doc160_160_160"
  run_dir=$(cd "$(dirname "$CHECKPOINT")" && pwd)
  export CHECKPOINT
  export ASYM_QUERY_IMAGE_BUDGETS="$b1 $b2 $b3"
  if [[ -n "$BASE_OUT_DIR" ]]; then
    export OUT_DIR="$BASE_OUT_DIR/mmeb_full_asym_${suffix}"
  else
    export OUT_DIR="$run_dir/eval/mmeb_full_asym_${suffix}"
  fi
  export LOG_DIR="${LOG_DIR:-$run_dir/logs}"
  export LOG_FILE="$LOG_DIR/eval_mmeb_full_asym_${suffix}_$(date +%Y%m%d_%H%M%S).log"
  echo "[mmeb_asym_query] running query-image budgets=$b1/$b2/$b3 doc budgets=160/160/160 interaction=$MAXSIM_INTERACTION query_agg=$MAXSIM_QUERY_AGG"
  bash "$SCRIPT_DIR/eval_mmeb_full.sh" "$CHECKPOINT"
  unset OUT_DIR LOG_FILE ASYM_QUERY_IMAGE_BUDGETS
done
