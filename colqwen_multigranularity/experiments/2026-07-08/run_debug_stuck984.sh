#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
REPO_ROOT=$(cd "$PROJECT_DIR/.." && pwd)

export PYTHONPATH="$PROJECT_DIR/vendor:$REPO_ROOT:${PYTHONPATH:-}"
if [[ -d /opt/conda/bin ]]; then
  export PATH="/opt/conda/bin:$PATH"
fi

DEFAULT_CACHE_ROOT="$PROJECT_DIR/.cache"
DEFAULT_HF_DATASETS_CACHE="$DEFAULT_CACHE_ROOT/huggingface/datasets"
if [[ -d /MURE-V2/env ]]; then
  DEFAULT_CACHE_ROOT="/MURE-V2/env/mure_cache/colqwen_multigranularity"
  DEFAULT_HF_DATASETS_CACHE="/MURE-V2/env/hf_datasets_cache"
fi
export MURE_CACHE_ROOT=${MURE_CACHE_ROOT:-$DEFAULT_CACHE_ROOT}
export HF_HOME=${HF_HOME:-$MURE_CACHE_ROOT/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$DEFAULT_HF_DATASETS_CACHE}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
export TMPDIR=${TMPDIR:-$MURE_CACHE_ROOT/tmp}
export DATA_DIR=${DATA_DIR:-$PROJECT_DIR/data_dir/}
export DATASET_NUM_PROC=${DATASET_NUM_PROC:-1}
export DATASET_SHUFFLE_BUFFER=${DATASET_SHUFFLE_BUFFER:-1024}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

NUM_PROCS=${NUM_PROCS:-8}
MASTER_PORT=${MASTER_PORT:-29984}
START_STEP=${START_STEP:-970}
END_STEP=${END_STEP:-990}
TRAIN_BSZ=${TRAIN_BSZ:-10}
INTERLEAVED_BSZ=${INTERLEAVED_BSZ:-10}
NUM_SHARDS=${NUM_SHARDS:-128}
COLLATE=${COLLATE:-0}
OUTPUT_DIR=${OUTPUT_DIR:-$SCRIPT_DIR/runs/debug_stuck984_$(date +%Y%m%d_%H%M%S)}

mkdir -p "$OUTPUT_DIR" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TMPDIR"

args=(
  "$SCRIPT_DIR/debug_stuck984_batches.py"
  --subset-config "$PROJECT_DIR/configs/train/moca_data_ratios_v3_full.yaml"
  --processor-name-or-path "$PROJECT_DIR/models/colqwen2.5-base"
  --output-dir "$OUTPUT_DIR"
  --start-step "$START_STEP"
  --end-step "$END_STEP"
  --per-device-train-batch-size "$TRAIN_BSZ"
  --interleaved-batch-size "$INTERLEAVED_BSZ"
  --num-shards "$NUM_SHARDS"
  --dataset-num-proc "$DATASET_NUM_PROC"
  --dataset-shuffle-buffer "$DATASET_SHUFFLE_BUFFER"
)

if [[ "$COLLATE" == "1" || "$COLLATE" == "true" || "$COLLATE" == "TRUE" ]]; then
  args+=(--collate)
fi

{
  echo "[debug-stuck984] start $(date +%Y-%m-%d\ %H:%M:%S)"
  echo "[debug-stuck984] output=$OUTPUT_DIR"
  echo "[debug-stuck984] cache=$MURE_CACHE_ROOT hf_datasets=$HF_DATASETS_CACHE tmp=$TMPDIR"
  echo "[debug-stuck984] steps=$START_STEP-$END_STEP train_bsz=$TRAIN_BSZ interleaved_bsz=$INTERLEAVED_BSZ num_procs=$NUM_PROCS collate=$COLLATE"
} | tee "$OUTPUT_DIR/launch.log"

python -m torch.distributed.run \
  --standalone \
  --nnodes 1 \
  --nproc_per_node "$NUM_PROCS" \
  --master_port "$MASTER_PORT" \
  "${args[@]}" 2>&1 | tee -a "$OUTPUT_DIR/launch.log"

python - "$OUTPUT_DIR" <<'PY' | tee "$OUTPUT_DIR/summary_table.tsv"
from __future__ import annotations
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("rank*_steps*.jsonl")):
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
print("rank\tstep\tcollate_seconds\tsubset_counts\tmax_qry_len\tmax_pos_len\timage_pairs")
for row in sorted(rows, key=lambda x: (x["step"], x["rank"])):
    examples = row["examples"]
    max_q = max((x.get("qry_len") or 0) for x in examples)
    max_p = max((x.get("pos_text_len") or 0) for x in examples)
    image_pairs = sum(int(x["qry_image"]["has"]) + int(x["pos_image"]["has"]) for x in examples)
    print(
        f'{row["rank"]}\t{row["step"]}\t{row.get("collate_seconds", "")}\t'
        f'{row["subset_counts"]}\t{max_q}\t{max_p}\t{image_pairs}'
    )
PY

echo "[debug-stuck984] done $(date +%Y-%m-%d\ %H:%M:%S)"
