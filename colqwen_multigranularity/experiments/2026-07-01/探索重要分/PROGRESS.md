# 探索重要分 Progress

Last updated: 2026-07-01

## Scope

研究 FolderHomo / FOLDER 中 unary importance/protect score 的来源。固定相似分、budget、训练数据和评测口径，只比较重要分定义。

Formal launcher policy:

```text
run_train.sh default MAX_STEPS=3000
run_train.sh default RUN_NAME=folder_importance_v1_<IMPORTANCE_MODE>_b160_160_160_3k
BUDGETS=160/160/160
Eval entry=eval_3sets.sh
```

Removed lightweight-check policy:

```text
Lightweight-check training/testing code and temporary run artifacts have been removed from this folder.
Do not use temporary-check metrics for reporting.
If a future temporary-check is needed, create a temporary external command or restore intentionally.
```

## Current Files

| File | Role |
| --- | --- |
| `config.py` | importance mode config |
| `modeling_importance.py` | local model/compressor implementation |
| `train_importance.py` | formal training entry |
| `eval_importance.py` | formal eval entry |
| `run_train.sh` | 8-GPU formal training launcher, default 3k |
| `eval_3sets.sh` | ViDoReV1 / ViDoReV2 / MMEB eval launcher |
| `README.md` | method motivation and experiment design |

Removed:

| Removed item | Reason |
| --- | --- |
| `removed temporary-check train/eval launcher` | temporary check code cleanup |
| `removed temporary-check run artifacts` | temporary check artifact cleanup |

## Experiment Matrix

Baseline target:

| Method | Budget | ViDoReV1 | ViDoReV2 | MMEB | Avg | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| FolderHomo residual160 / V1 | 160/160/160 | 89.34 | 60.28 | 76.43 | 75.35 | reference |

Importance experiments:

| Priority | IMPORTANCE_MODE | Formal train | Formal eval | Result | TODO |
| --- | --- | --- | --- | --- | --- |
| P0 | `mlp` | TODO | TODO | TODO | Run 3k formal train, then eval 3 sets. |
| P0 | `mha_attn` | PARTIAL | TODO | TODO | Existing dirs: `folder_importance_v1_mha_attn_b160_160_160_3k`, `folder_importance_v1_mha_attn_b160_160_160_4k`; verify checkpoint completion and run eval. |
| P0 | `learned_gate` | TODO | TODO | TODO | Run 3k formal train, then eval 3 sets. |
| P1 | `mha_pagerank` | TODO | TODO | TODO | Run only if P0 attention route is promising or time permits. |

## TODO

1. Confirm whether `folder_importance_v1_mha_attn_b160_160_160_3k` reached `checkpoint-3000`; if not, resume or rerun.
2. Evaluate `mha_attn` at the valid checkpoint with `eval_3sets.sh`.
3. Launch formal 3k `mlp` run for independent implementation alignment.
4. Launch formal 3k `learned_gate` run.
5. Decide whether `mha_pagerank` is worth running after P0 results.
6. Fill the result table in `README.md` and this file after each eval.

## Commands

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity

IMPORTANCE_MODE=mlp bash experiments/2026-07-01/探索重要分/run_train.sh
IMPORTANCE_MODE=mha_attn bash experiments/2026-07-01/探索重要分/run_train.sh
IMPORTANCE_MODE=learned_gate bash experiments/2026-07-01/探索重要分/run_train.sh
IMPORTANCE_MODE=mha_pagerank bash experiments/2026-07-01/探索重要分/run_train.sh
```

Eval example:

```bash
IMPORTANCE_MODE=mha_attn \
  bash experiments/2026-07-01/探索重要分/eval_3sets.sh \
  experiments/2026-07-01/探索重要分/runs/folder_importance_v1_mha_attn_b160_160_160_3k/checkpoint-3000
```
