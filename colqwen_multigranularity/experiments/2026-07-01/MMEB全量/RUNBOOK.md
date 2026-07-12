# MMEB Runbook

Updated: 2026-07-02

This file records operational knowledge for MMEB full and subset evals.

## Directory Rule

The 2026-07-01 MMEB work is intentionally split into three directories:

- `MMEB全量/`: full MMEB eval, asymmetric query budget eval, result aggregation, and this runbook.
- `MaxSim交互/`: scorer-level and interaction-mechanism diagnosis without retraining.
- `MMEB任务课程学习/`: 500-step low-cost training-side diagnosis and curriculum experiments.

Do not create a fourth `MMEB问题诊断/` directory. Cross-cutting notes should be merged into one of the three directories above.

## Known OOM Cases

### 2026-07-02 sym160 full MMEB eval OOM

Run:

```text
checkpoint: experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000
log: experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/logs/eval_mmeb_budget_20260702_210305.log
eval config: configs/eval/test_data_mast_mmeb_v3.yaml
metric: recall_at_1
batch-query: 16
batch-passage: 32
batch-score: 256
num-workers: 0
query budgets: 160 160 160
doc budgets: 160 160 160
```

Status before failure:

```text
completed: 29 / 36
last completed dataset: MMEB-eval-TextVQA-beir
last completed TextVQA metric: R@1 0.214, R@5 0.357
```

Failure point:

```text
next dataset: MMEB-eval-OVEN-beir
stage: score processing
time: 2026-07-02 21:48:36
```

Error:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 64.98 GiB.
GPU total: 79.25 GiB
process memory in use: about 76.25 GiB
allocated by PyTorch: about 75.10 GiB
```

Interpretation:

The failure is not from encoding. Query and passage encoding completed for OVEN. The OOM happened during `score_multi_vector_dist` / `einsum`, so the score matrix for OVEN is too large for `BATCH_SCORE=256` under sym160.

Action:

- Do not rerun full MMEB from the beginning with the same high score batch.
- Resume by evaluating remaining datasets only.
- Use smaller `BATCH_SCORE`, likely 64 first; if still OOM, use 32 or 16.

Remaining datasets after TextVQA:

```text
MMEB-eval-OVEN-beir
MMEB-eval-FashionIQ-beir
MMEB-eval-EDIS-beir
MMEB-eval-Wiki-SS-NQ-beir
MMEB-eval-Visual7W-Pointing-beir
MMEB-eval-RefCOCO-beir
MMEB-eval-RefCOCO-Matching-beir
```

Suggested recovery command:

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/2026-07-01/MMEB全量

CHECKPOINT=runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000 \
OUT_DIR=runs/folder_homo_mmeb_budget_sym160_4k/eval/mmeb_budget_sym160_4k_remaining_oven_onward_bs64 \
LOG_FILE=runs/folder_homo_mmeb_budget_sym160_4k/logs/eval_mmeb_budget_remaining_oven_onward_bs64_$(date +%Y%m%d_%H%M%S).log \
BATCH_QUERY=16 \
BATCH_PASSAGE=32 \
BATCH_SCORE=64 \
NUM_WORKERS=0 \
ONLY_EVAL_KEYWORDS="MMEB-eval-OVEN-beir MMEB-eval-FashionIQ-beir MMEB-eval-EDIS-beir MMEB-eval-Wiki-SS-NQ-beir MMEB-eval-Visual7W-Pointing-beir MMEB-eval-RefCOCO-beir MMEB-eval-RefCOCO-Matching-beir" \
bash eval_mmeb_full.sh
```

If OVEN still OOMs:

```bash
BATCH_SCORE=32
```

If it still OOMs:

```bash
BATCH_SCORE=16
```

## Current Safe Defaults

For full MMEB on sym160:

```text
BATCH_QUERY=16
BATCH_PASSAGE=16
BATCH_SCORE=64
NUM_WORKERS=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

For smaller targeted eval subsets:

```text
BATCH_QUERY=16
BATCH_PASSAGE=32
BATCH_SCORE=128
NUM_WORKERS=0
```

Avoid using `BATCH_SCORE=256` for full MMEB unless the remaining large score-matrix subsets are excluded or separately tested.

## Logging Rule

When an OOM happens, append:

- date and time
- checkpoint
- eval config
- output dir
- log file
- batch-query / batch-passage / batch-score
- last completed dataset
- failing dataset
- stage: query, passage, scoring, or metric
- exact allocation request
- recovery command

## Result Rule

If a full eval OOMs after partial completion:

- Treat the partial log as partial evidence only.
- Do not report `avg_recall_at_1` unless all intended datasets are complete.
- It is acceptable to merge partial results and remaining-subset results only if both use the same checkpoint, same metric, same token budgets, and same scorer.

