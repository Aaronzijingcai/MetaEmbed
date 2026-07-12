#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
MMEB_DIR=$(cd "$SCRIPT_DIR/../MMEB全量" && pwd)
CHECKPOINT=${CHECKPOINT:-$MMEB_DIR/runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000}
BASE_OUT=${BASE_OUT:-$MMEB_DIR/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_cirr_text_priority_iter}
LOG_DIR=${LOG_DIR:-$MMEB_DIR/runs/folder_homo_mmeb_budget_sym160_4k/logs/maxsim_cirr_text_priority_iter}
mkdir -p "$BASE_OUT" "$LOG_DIR"

run_eval() {
  local name="$1"
  local interaction="$2"
  local query_agg="$3"
  local drop_prefix="$4"
  local drop_suffix="$5"
  local bi_lambda="$6"
  local topk="$7"
  local out_dir="$BASE_OUT/$name"
  if [[ -f "$out_dir/mmeb_full_summary.json" ]]; then
    echo "[cirr_text_priority] skip existing $name"
    return
  fi
  echo "[cirr_text_priority] running name=$name interaction=$interaction query_agg=$query_agg drop_prefix=$drop_prefix drop_suffix=$drop_suffix lambda=$bi_lambda topk=$topk"
  OUT_DIR="$out_dir" \
  LOG_FILE="$LOG_DIR/${name}.log" \
  ONLY_EVAL_KEYWORDS="MMEB-eval-CIRR-beir" \
  MAXSIM_INTERACTION="$interaction" \
  MAXSIM_QUERY_AGG="$query_agg" \
  MAXSIM_QUERY_DROP_PREFIX="$drop_prefix" \
  MAXSIM_QUERY_DROP_SUFFIX="$drop_suffix" \
  MAXSIM_BI_LAMBDA="$bi_lambda" \
  MAXSIM_QUERY_TOPK="$topk" \
  MAXSIM_GLOBAL_WEIGHT=0.0 \
  MAXSIM_HIT_PENALTY_WEIGHT=0.0 \
  BATCH_QUERY=${BATCH_QUERY:-16} \
  BATCH_PASSAGE=${BATCH_PASSAGE:-16} \
  BATCH_SCORE=${BATCH_SCORE:-128} \
  NUM_WORKERS=${NUM_WORKERS:-0} \
  NUM_GPUS=${NUM_GPUS:-8} \
  bash "$MMEB_DIR/eval_mmeb_full.sh" "$CHECKPOINT"
}

# Baseline/control; q2d mean is equivalent to legacy ranking for CIRR but cheaper to compare against slice variants.
run_eval "q2d_mean_drop0" "q2d" "mean" "0" "0" "0.5" "0"

# CIRR query = reference image tokens first, modification text later. Drop-prefix diagnoses whether reference-image tokens dominate.
for p in 80 160 240 320 400 480 560 640 800 960; do
  run_eval "q2d_mean_drop_prefix_${p}" "q2d" "mean" "$p" "0" "0.5" "0"
done

# Drop suffix controls whether repeated query augmentation/end tokens create noise.
for s in 16 32 64 128; do
  run_eval "q2d_mean_drop_suffix_${s}" "q2d" "mean" "0" "$s" "0.5" "0"
done

# If text-priority helps, test a conservative reciprocal version after the same prefix crop.
for p in 160 320 480 640; do
  run_eval "bi_lam09_drop_prefix_${p}" "bi_mean" "mean" "$p" "0" "0.9" "0"
done

python3 - <<'PY'
import json
from pathlib import Path
base=Path(__import__('os').environ.get('BASE_OUT', ''))
if not str(base):
    base=Path('')
rows=[]
for p in sorted(base.glob('*/mmeb_full_summary.json')):
    data=json.load(open(p))
    rows.append((p.parent.name, data.get('overall')))
rows.sort(key=lambda x: (-999 if x[1] is None else -x[1], x[0]))
md=['| Run | CIRR P@1 |','| --- | ---: |']
for name,val in rows:
    md.append(f'| `{name}` | {val:.3f} |' if val is not None else f'| `{name}` | NA |')
text='\n'.join(md)+'\n'
(base/'cirr_text_priority_compare.md').write_text(text)
print(text)
PY
