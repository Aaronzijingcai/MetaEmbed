#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SUITE="$SCRIPT_DIR/../ablations/suite.py"

exec python3 "$SUITE" train \
  --config "$SCRIPT_DIR/experiment.json" \
  --variant adaptive_bidirectional_topk48_mean \
  "$@"
