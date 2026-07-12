#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
MMEB_DIR=$(cd "$SCRIPT_DIR/../MMEB全量" && pwd)
CHECKPOINT=${CHECKPOINT:-$MMEB_DIR/runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000}
BASE_OUT=${BASE_OUT:-$MMEB_DIR/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_cirr_prompt_iter}
LOG_DIR=${LOG_DIR:-$MMEB_DIR/runs/folder_homo_mmeb_budget_sym160_4k/logs/maxsim_cirr_prompt_iter}
mkdir -p "$BASE_OUT" "$LOG_DIR"
run_eval() {
  local name="$1"; local qaug="$2"; local strip="$3"; local dropdoc="$4"; local interaction="$5"; local lambda="$6"
  local out_dir="$BASE_OUT/$name"
  if [[ -f "$out_dir/mmeb_full_summary.json" ]]; then echo "[cirr_prompt] skip $name"; return; fi
  echo "[cirr_prompt] running $name qaug=$qaug strip=$strip dropdoc=$dropdoc interaction=$interaction lambda=$lambda"
  OUT_DIR="$out_dir" \
  LOG_FILE="$LOG_DIR/${name}.log" \
  ONLY_EVAL_KEYWORDS="MMEB-eval-CIRR-beir" \
  QUERY_AUGMENTATION_REPEATS="$qaug" \
  STRIP_CIRR_QUERY_INSTRUCTION="$strip" \
  DROP_DOC_TEXT_IF_IMAGE="$dropdoc" \
  MAXSIM_INTERACTION="$interaction" \
  MAXSIM_QUERY_AGG=mean \
  MAXSIM_BI_LAMBDA="$lambda" \
  MAXSIM_QUERY_TOPK=0 \
  MAXSIM_QUERY_DROP_PREFIX=0 \
  MAXSIM_QUERY_DROP_SUFFIX=0 \
  BATCH_QUERY=${BATCH_QUERY:-16} \
  BATCH_PASSAGE=${BATCH_PASSAGE:-16} \
  BATCH_SCORE=${BATCH_SCORE:-128} \
  NUM_WORKERS=${NUM_WORKERS:-0} \
  NUM_GPUS=${NUM_GPUS:-8} \
  bash "$MMEB_DIR/eval_mmeb_full.sh" "$CHECKPOINT"
}
run_eval "q2d_mean_qaug10_original" 10 0 0 q2d 0.5
run_eval "q2d_mean_qaug0" 0 0 0 q2d 0.5
run_eval "q2d_mean_strip_instruction" 10 1 0 q2d 0.5
run_eval "q2d_mean_strip_instruction_qaug0" 0 1 0 q2d 0.5
run_eval "q2d_mean_strip_qaug0_dropdoc" 0 1 1 q2d 0.5
run_eval "bi_lam09_strip_qaug0_dropdoc" 0 1 1 bi_mean 0.9
python3 - <<'PY'
import json, os
from pathlib import Path
base=Path(os.environ.get('BASE_OUT',''))
if not str(base): base=Path('')
rows=[]
for p in sorted(base.glob('*/mmeb_full_summary.json')):
    data=json.load(open(p)); rows.append((p.parent.name,data.get('overall')))
rows.sort(key=lambda x:(-1 if x[1] is None else -x[1], x[0]))
md=['| Run | CIRR P@1 |','| --- | ---: |']+[f'| `{n}` | {v:.3f} |' for n,v in rows]
(base/'cirr_prompt_compare.md').write_text('\n'.join(md)+'\n')
print('\n'.join(md))
PY
