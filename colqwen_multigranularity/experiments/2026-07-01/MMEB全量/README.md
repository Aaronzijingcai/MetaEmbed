# MMEB 全量与 Query-Side Budget 实验

Updated: 2026-07-03

## 2026-07-03 from-base rerun plan

本目录从今天起新增一个有效训练基线：`fullmix_flat_sym160_s500_from_base`。

目的不是追求最终收敛，而是建立一个干净的 500-step MMEB full-mix baseline，用来和任务课程学习、MaxSim interaction loss 两条线公平比较。

统一配置：

| 项 | 设置 |
| --- | --- |
| Initialization | `models/colqwen2.5-base` |
| Resume checkpoint | none |
| Warm-start adapter | none |
| Model | FolderHomo real-token compressor |
| Budget | `160/160/160` |
| Compress stages | `all` |
| Loss mode | `flat` |
| Train data | full-mix MMEB/MoCa config, recorded per run |
| Steps | `500` |
| LR | `1e-4` |
| LR scheduler | `constant` |
| Warmup | `0` |
| MARC | off |

Run name:

```text
fullmix_flat_sym160_s500_from_base
```

有效 checkpoint:

```text
runs/fullmix_flat_sym160_s500_from_base/checkpoint-500
```

评估优先级：

1. `worst10` MMEB P@1。
2. `retention` MMEB P@1。
3. 如果 worst10 或 retention 明显优于其他 P0 run，再跑完整 36 子集 MMEB。

旧的 `sym160_full` / `asym_q80_t160` / `asym_q40_t160` 是 eval-only 诊断，不受 continued-tuning 问题影响；但它们不能回答“from-base 训练策略是否有效”。

## 实验目的

本目录回答三个问题：

1. 当前 FolderHomo `sym160` checkpoint 在完整 MMEB v3 36 个子集上的 P@1 表现如何？
2. 当 query 端包含图像时，把 query 视觉 token 从 `160/160/160` 压到 `80/80/80` 或 `40/40/40`，是否能缓解 MMEB 中 image-query / multimodal-query 的 MaxSim 失败？
3. 从 `colqwen2.5-base` 直接训练 FolderHomo 500 steps 时，full-mix 数据是否已经能产生可靠早期信号？

这里使用的是当前 FolderHomo / FOLDER 主模型，不是 MRL-main。所谓“沿用 MRL-main”只指训练集和测试集口径。

## 实现方案

固定项：

| 项目 | 设置 |
| --- | --- |
| Model | FolderHomo real-token compressor |
| Base checkpoint | `runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000` 或明确指定 checkpoint |
| Document/target budget | `160/160/160` |
| Compress stages | `all` |
| Eval config | `configs/eval/test_data_mast_mmeb_v3.yaml` |
| Metric reported in docs | `P@1` |
| Underlying evaluator field | `recall_at_1`，MMEB evaluator 注释说明实际是 hit@1 |

唯一变量：

| 实验 | Query image budget | Target budget | MaxSim setting |
| --- | ---: | ---: | --- |
| `sym160_full` | `160/160/160` | `160/160/160` | 默认 full eval |
| `asym_q80_t160` | `80/80/80` | `160/160/160` | 默认 `q2d_mean` |
| `asym_q40_t160` | `40/40/40` | `160/160/160` | 默认 `q2d_mean` |

`sym80/sym40/sym20` 暂不属于本目录 P0。它们会同时改变 query 和 target 两端 token 数，不能直接回答 query-side 非对称压缩是否有效。

## 优先级

| Priority | Run | 目的 |
| --- | --- | --- |
| P0 | `sym160_full` | 固定当前 checkpoint，建立完整 MMEB 36 子集基线。 |
| P0 | `fullmix_flat_sym160_s500_from_base` | 从 base 训练 500 steps，建立干净 full-mix training baseline。 |
| P0 | `asym_q80_t160` | 中等 query 端压缩，测试 image-query 噪声是否下降。 |
| P0 | `asym_q40_t160` | 强 query 端压缩，测试收益和信息损失边界。 |
| P1 | `asym_q20_t160` | 只有 q40 明显有效但仍不够时再跑。 |
| P2 | `sym80/sym40/sym20` | 只有 MaxSim 机制和课程学习都无法解决时再考虑。 |

## 训练集

当前历史基线 checkpoint 来自：

```text
configs/train/moca_data_ratios_v3_full.yaml 或同口径 full MMEB/MoCa train config
```

记录结果时必须填写实际训练配置：

| Run | Train config | Train samples | Steps | Global batch | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `sym160_full` | TODO | 581k if confirmed | 4000 | 32 | TODO |
| `fullmix_flat_sym160_s500_from_base` | TODO | TODO | 500 | TODO | from base, no warmup, constant LR |
| `asym_q80_t160` | no retrain | - | - | - | eval-only |
| `asym_q40_t160` | no retrain | - | - | - | eval-only |

## 测试集

完整 MMEB 使用：

```text
configs/eval/test_data_mast_mmeb_v3.yaml
```

共 36 个子集。完整结果必须填写下面的大表，只记录 P@1。

## 运行命令

Full eval:

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/2026-07-01/MMEB全量

CHECKPOINT=runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000 \
BATCH_QUERY=16 \
BATCH_PASSAGE=16 \
BATCH_SCORE=64 \
NUM_WORKERS=0 \
bash eval_mmeb_full.sh
```

From-base 500-step full-mix training:

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/2026-07-01/MMEB全量

RUN_NAME=fullmix_flat_sym160_s500_from_base \
MAX_STEPS=500 \
SAVE_STEPS=500 \
MODEL_PATH=../../../models/colqwen2.5-base \
RESUME_CKPT= \
WARM_START_ADAPTER_PATH= \
BUDGETS="160 160 160" \
INTERACTION_LOSS_MODE=flat \
LR_SCHEDULER_TYPE=constant \
WARMUP_RATIO=0 \
WARMUP_STEPS=0 \
LEARNING_RATE=1e-4 \
TRAIN_BSZ=16 \
INTERLEAVED_BSZ=16 \
bash run_train_full.sh
```

Query-side asymmetric budget:

```bash
CHECKPOINT=runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000 \
BATCH_QUERY=16 \
BATCH_PASSAGE=16 \
BATCH_SCORE=64 \
NUM_WORKERS=0 \
bash eval_mmeb_asym_query.sh
```

单独跑 q80:

```bash
ASYM_QUERY_BUDGET_SETS="80,80,80" CHECKPOINT=... bash eval_mmeb_asym_query.sh
```

单独跑 q40:

```bash
ASYM_QUERY_BUDGET_SETS="40,40,40" CHECKPOINT=... bash eval_mmeb_asym_query.sh
```

## 文件说明

| File | Role |
| --- | --- |
| `run_train_full.sh` | FolderHomo full train launcher。 |
| `eval_mmeb.py` | MMEB eval model-loading wrapper。 |
| `eval_mmeb_full.sh` | Full/subset MMEB eval launcher。 |
| `eval_mmeb_asym_query.sh` | Query-side asymmetric budget launcher。 |
| `analyze_mmeb.py` | 生成 group/class/per-dataset summary。 |
| `compare_mmeb_runs.py` | 汇总多个 `mmeb_full_summary.json`。 |
| `RUNBOOK.md` | OOM、恢复命令和运行事故记录。 |

## 实验结果: 总表

| Run | Checkpoint | Query budget | Target budget | Interaction | Eval scope | P@1 overall | IND | OOD | Classification | VQA | Retrieval | Grounding | Status |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `sym160_full` | ckpt-4000 | 160/160/160 | 160/160/160 | q2d_sum | 36/36 | 0.4611 | 0.4827 | 0.4341 | 0.5433 | 0.1958 | 0.5858 | 0.5450 | DONE |
| `fullmix_flat_sym160_s500_from_base` | ckpt-500 | 160/160/160 | 160/160/160 | q2d_mean / scorer matrix | worst10 + retention first | TODO | TODO | TODO | TODO | TODO | TODO | TODO | P0 planned |
| `asym_q80_t160` | TODO | 80/80/80 | 160/160/160 | q2d_mean | 36/36 or subset | TODO | TODO | TODO | TODO | TODO | TODO | TODO | RUNNING |
| `asym_q40_t160` | TODO | 40/40/40 | 160/160/160 | q2d_mean | 36/36 or subset | TODO | TODO | TODO | TODO | TODO | TODO | TODO | RUNNING |

## 实验结果: MMEB 36 子集 P@1

| Dataset | Type | `sym160_full` | `asym_q80_t160` | `asym_q40_t160` | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| ImageNet-1K | Classification | 0.649 | TODO | TODO |  |
| N24News | Classification | 0.624 | TODO | TODO |  |
| HatefulMemes | Classification | 0.595 | TODO | TODO |  |
| SUN397 | Classification | 0.645 | TODO | TODO |  |
| VOC2007 | Classification | 0.848 | TODO | TODO |  |
| InfographicsVQA | VQA | 0.141 | TODO | TODO | hard |
| ChartQA | VQA | 0.171 | TODO | TODO | hard |
| A-OKVQA | VQA | 0.183 | TODO | TODO | hard |
| DocVQA | VQA | 0.272 | TODO | TODO | hard |
| OK-VQA | VQA | 0.216 | TODO | TODO | hard |
| Visual7W | VQA | 0.145 | TODO | TODO | hard |
| VisDial | Retrieval | 0.545 | TODO | TODO |  |
| CIRR | Retrieval | 0.103 | TODO | TODO | compositional hard |
| NIGHTS | Retrieval | 0.652 | TODO | TODO |  |
| WebQA | Retrieval | 0.889 | TODO | TODO |  |
| VisualNews_i2t | Retrieval | 0.625 | TODO | TODO | image-to-text |
| VisualNews_t2i | Retrieval | 0.630 | TODO | TODO | text-to-image |
| MSCOCO_i2t | Retrieval | 0.605 | TODO | TODO | image-to-text |
| MSCOCO_t2i | Retrieval | 0.686 | TODO | TODO | text-to-image |
| MSCOCO | Visual Grounding | 0.430 | TODO | TODO |  |
| Place365 | Classification | 0.358 | TODO | TODO | OOD |
| ImageNet-A | Classification | 0.415 | TODO | TODO | OOD |
| ImageNet-R | Classification | 0.679 | TODO | TODO | OOD |
| ObjectNet | Classification | 0.531 | TODO | TODO | OOD |
| Country211 | Classification | 0.089 | TODO | TODO | hard |
| ScienceQA | VQA | 0.197 | TODO | TODO | OOD hard |
| VizWiz | VQA | 0.264 | TODO | TODO | OOD |
| GQA | VQA | 0.154 | TODO | TODO | OOD hard |
| TextVQA | VQA | 0.215 | TODO | TODO | OOD |
| OVEN | Retrieval | 0.636 | TODO | TODO | image+text target |
| FashionIQ | Retrieval | 0.024 | TODO | TODO | compositional hard |
| EDIS | Retrieval | 0.844 | TODO | TODO | OOD |
| Wiki-SS-NQ | Retrieval | 0.790 | TODO | TODO | OOD |
| Visual7W-Pointing | Visual Grounding | 0.520 | TODO | TODO | hard |
| RefCOCO | Visual Grounding | 0.445 | TODO | TODO | hard |
| RefCOCO-Matching | Visual Grounding | 0.785 | TODO | TODO | image+text-to-image+text |

## 实验结论

待完整结果填充。

判断规则：

| 观察 | 结论 |
| --- | --- |
| q80/q40 主要提升 image-query / multimodal-query，text-query 基本不掉 | query 端视觉 token 冗余是 MMEB 失败的重要因素。 |
| q40 比 q80 更好 | 视觉 query 端需要强压缩，后续可考虑更小 query budget。 |
| q40 明显掉，q80 稳定 | query token 不能过度压缩，q80 是更合理折中。 |
| q80/q40 都不能改善 VQA/compositional hard subsets | token 数量不是主因，应转向 `MaxSim交互/` 和 `MMEB任务课程学习/`。 |
