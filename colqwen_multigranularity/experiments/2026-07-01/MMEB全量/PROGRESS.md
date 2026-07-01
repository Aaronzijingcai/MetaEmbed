# MMEB 全量 Progress

Last updated: 2026-07-01

## Scope

研究 MMEB full setting 下 MaxSim 非对称性问题，尤其是 query 端含图时是否应该在训练阶段把 query image tokens 压得比 target/doc 更短。

Formal launcher policy:

```text
MMEB run_train_full.sh default MAX_STEPS=4000
MMEB run_train_budget.sh default MAX_STEPS=4000
Run names use _4k suffix by default
Train config=configs/train/moca_data_ratios_v3_full.yaml
Eval config=configs/eval/test_data_mast_mmeb_v3.yaml
```

Removed lightweight-check policy:

```text
Lightweight-check training/testing code and temporary run artifacts have been removed from this folder.
Before cleanup, all 7 budget settings passed 8-GPU 2-step training and all 36 MMEB-dataset truncated eval.
Those temporary-check outputs were intentionally deleted and should not be reported as results.
```

## Current Files

| File | Role |
| --- | --- |
| `run_train_full.sh` | original full MMEB FolderHomo train launcher, default 4k |
| `eval_mmeb.py` | original MMEB eval entry, includes historical eval-only asym support |
| `eval_mmeb_full.sh` | original MMEB eval launcher |
| `config_mmeb_budget.py` | query/doc budget config for formal 7-run plan |
| `modeling_mmeb_budget.py` | local copy/wrapper of FolderHomo model with training-stage query/doc budgets |
| `loss_mmeb_budget.py` | MaxSim loss masks with query budget and doc budget separated |
| `train_mmeb_budget.py` | formal training entry for budget-aware MMEB experiments |
| `eval_mmeb_budget.py` | formal eval entry for budget-aware MMEB experiments |
| `run_train_budget.sh` | 8-GPU formal training launcher, default 4k |
| `eval_mmeb_budget_full.sh` | full MMEB eval launcher for budget-aware checkpoints |
| `analyze_mmeb.py` | aggregation by IND/OOD and task class |
| `README.md` | method motivation and experiment design |

Removed:

| Removed item | Reason |
| --- | --- |
| `removed temporary-check train/eval launcher` | temporary check code cleanup |
| `removed 7-budget temporary-check launcher` | temporary check code cleanup |
| `removed temporary-check run artifacts` | temporary check artifact cleanup |

## Experiment Matrix

Baseline/reference:

| Method | Train steps | Eval | Status | TODO |
| --- | ---: | --- | --- | --- |
| `run_train_full.sh`, FolderHomo 160/160/160 full MMEB | 4000 | MMEB full | TODO | Optional reference if not superseded by `sym160` budget run. |

Formal 7-run budget plan:

| Experiment | QUERY_BUDGETS | DOC_BUDGETS | Formal train | Formal eval | Result | TODO |
| --- | --- | --- | --- | --- | --- | --- |
| `sym160` | `160 160 160` | `160 160 160` | TODO | TODO | TODO | Run 4k train, then full MMEB eval. |
| `sym80` | `80 80 80` | `80 80 80` | TODO | TODO | TODO | Run 4k train, then full MMEB eval. |
| `sym40` | `40 40 40` | `40 40 40` | TODO | TODO | TODO | Run 4k train, then full MMEB eval. |
| `sym20` | `20 20 20` | `20 20 20` | TODO | TODO | TODO | Run 4k train, then full MMEB eval. |
| `asym_q80_d160` | `80 80 80` | `160 160 160` | TODO | TODO | TODO | Run 4k train, then full MMEB eval. |
| `asym_q40_d160` | `40 40 40` | `160 160 160` | TODO | TODO | TODO | Run 4k train, then full MMEB eval. |
| `asym_q20_d160` | `20 20 20` | `160 160 160` | TODO | TODO | TODO | Run 4k train, then full MMEB eval. |

## TODO

1. Launch formal 4k `sym160` run.
2. Launch formal 4k `sym80` run.
3. Launch formal 4k `sym40` run.
4. Launch formal 4k `sym20` run.
5. Launch formal 4k `asym_q80_d160` run.
6. Launch formal 4k `asym_q40_d160` run.
7. Launch formal 4k `asym_q20_d160` run.
8. Run `eval_mmeb_budget_full.sh` for each `checkpoint-4000`.
9. Run `analyze_mmeb.py` summaries and compare Classification / VQA / Retrieval / Visual Grounding and IND/OOD groups.
10. Fill the result table in `README.md` and this file after each eval.

## Commands

Single formal train example:

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/2026-07-01/MMEB全量
QUERY_BUDGETS="80 80 80" DOC_BUDGETS="160 160 160" \
RUN_NAME=folder_homo_mmeb_budget_asym_q80_d160_4k \
bash run_train_budget.sh
```

Single formal eval example:

```bash
QUERY_BUDGETS="80 80 80" DOC_BUDGETS="160 160 160" \
CHECKPOINT=runs/folder_homo_mmeb_budget_asym_q80_d160_4k/checkpoint-4000 \
bash eval_mmeb_budget_full.sh
```

Suggested run-name map:

```text
sym160        folder_homo_mmeb_budget_sym160_4k
sym80         folder_homo_mmeb_budget_sym80_4k
sym40         folder_homo_mmeb_budget_sym40_4k
sym20         folder_homo_mmeb_budget_sym20_4k
asym_q80      folder_homo_mmeb_budget_asym_q80_d160_4k
asym_q40      folder_homo_mmeb_budget_asym_q40_d160_4k
asym_q20      folder_homo_mmeb_budget_asym_q20_d160_4k
```
