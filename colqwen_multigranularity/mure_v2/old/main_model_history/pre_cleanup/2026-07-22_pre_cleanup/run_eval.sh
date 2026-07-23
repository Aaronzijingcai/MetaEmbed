#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
LEGACY_EVAL="$PROJECT_DIR/experiments/2026-07-08/eval_full_main_models.sh"

if [[ ! -f "$LEGACY_EVAL" ]]; then
  echo "[main_model] missing legacy eval launcher: $LEGACY_EVAL" >&2
  exit 2
fi

echo "[main_model] RHC eval wrapper"
echo "[main_model] RUN_NAME=${RUN_NAME:-<required by legacy eval>} CHECKPOINT=${CHECKPOINT:-<required by legacy eval>} SCORERS=${SCORERS:-<required by legacy eval>}"
exec "$LEGACY_EVAL"
