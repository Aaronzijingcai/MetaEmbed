#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RUN_ROOT="$SCRIPT_DIR/runs"
EVAL_ROOT="$SCRIPT_DIR/evaluations"

echo "[main_model] local run root: $RUN_ROOT"
if [[ -d "$RUN_ROOT" ]]; then
  find "$RUN_ROOT" -maxdepth 4 -type f \
    \( -name run_manifest.json -o -name trainer_state.json -o -name train.log \) \
    -print | sort
else
  echo "[main_model] no local runs"
fi

echo "[main_model] local evaluation root: $EVAL_ROOT"
if [[ -d "$EVAL_ROOT" ]]; then
  find "$EVAL_ROOT" -maxdepth 6 -type f \
    \( -name mmeb_full.json -o -name vidore_v2.json -o -name '*.status' \) \
    -print | sort
else
  echo "[main_model] no local evaluations"
fi
