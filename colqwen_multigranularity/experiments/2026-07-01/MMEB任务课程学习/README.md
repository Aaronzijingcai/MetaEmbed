# MMEB 任务课程学习实验

Updated: 2026-07-03

## 2026-07-04 Server B from-base 结果

Server B 已完成三组 `from_base_s500` 训练与 `worst10 + retention` 评测。所有 run 均从 `models/colqwen2.5-base` 初始化，未使用任何历史 checkpoint 或 adapter：

```text
RESUME_CKPT=
WARM_START_ADAPTER_PATH=
TRAIN_BSZ=12
INTERLEAVED_BSZ=12
LR=1e-4
LR scheduler=constant
warmup=0
MARC=off
```

### 总表

| Run | Train config | Loss mode | Steps | Worst10 P@1 | Retention P@1 | Status | Conclusion |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `core4_flat_sym160_s500_from_base` | `train_worst10_core4.yaml` | `flat` | 500 | 0.1437 | 0.3056 | DONE | 基本打平历史 worst10 baseline，没有明显解决 MMEB 低分问题。 |
| `core4_factorized_local_sym160_s500_from_base` | `train_worst10_core4.yaml` | `factorized_local` | 500 | 0.1370 | 0.2772 | DONE | 分通道 local interaction 没有收益，整体弱于 flat。 |
| `compositional_flat_sym160_s500_from_base` | `train_compositional_hard.yaml` | `flat` | 500 | 0.0490 | 0.2702 | DONE | 只训 compositional 会严重伤 VQA/分类，不适合作为统一方案。 |

### Worst10 P@1 明细

| Dataset | `core4_flat` | `core4_factorized_local` | `compositional_flat` |
| --- | ---: | ---: | ---: |
| FashionIQ | 0.036 | 0.048 | 0.042 |
| CIRR | 0.121 | 0.116 | 0.130 |
| Country211 | 0.040 | 0.030 | 0.017 |
| GQA | 0.169 | 0.192 | 0.075 |
| ScienceQA | 0.186 | 0.133 | 0.072 |
| InfographicsVQA | 0.105 | 0.101 | 0.013 |
| A-OKVQA | 0.211 | 0.220 | 0.036 |
| Visual7W | 0.182 | 0.143 | 0.040 |
| OK-VQA | 0.232 | 0.251 | 0.060 |
| ChartQA | 0.155 | 0.136 | 0.005 |
| **Average** | **0.1437** | **0.1370** | **0.0490** |

### Retention P@1 明细

| Dataset | `core4_flat` | `core4_factorized_local` | `compositional_flat` |
| --- | ---: | ---: | ---: |
| ImageNet-1K | 0.459 | 0.353 | 0.242 |
| VOC2007 | 0.525 | 0.667 | 0.721 |
| VisualNews_i2t | 0.290 | 0.367 | 0.163 |
| VisualNews_t2i | 0.205 | 0.079 | 0.234 |
| MSCOCO_i2t | 0.463 | 0.513 | 0.279 |
| MSCOCO_t2i | 0.482 | 0.390 | 0.433 |
| WebQA | 0.160 | 0.096 | 0.362 |
| VisDial | 0.333 | 0.140 | 0.328 |
| ChartQA | 0.155 | 0.136 | 0.005 |
| GQA | 0.169 | 0.192 | 0.075 |
| CIRR | 0.121 | 0.116 | 0.130 |
| **Average** | **0.3056** | **0.2772** | **0.2702** |

### 当前解释

这批结果说明，MMEB 训练侧问题不能简单归因于旧实验的 warm-start/continued-tuning 漂移。即使从 base 重新训练，`core4_flat` 也只能回到历史 worst10 baseline 附近；`factorized_local` 没有证明分通道 local interaction 是有效方向；`compositional_flat` 只对 CIRR/FashionIQ 有轻微局部信号，但大幅破坏 VQA/分类类任务。

因此，当前不建议继续扩大同类 `core4` 或 compositional-only 500-step 训练。更合理的下一步是等待 Server A 的 `fullmix_flat`、`global_local`、`vqa_hard` 结果，然后优先分析：

- full-mix 是否明显强于 core4 小课程；
- VQA-hard from-base 是否能单独修复短答案任务；
- global-local 是否比 factorized-local 更适合 interaction-loss 训练；
- scorer-only 的 `bi_query_topk64` 为什么能显著改善 VQA，而训练式 factorized loss 没有复现该收益。

## 2026-07-03 from-base rerun plan

本目录旧的 `from_sym160` / `continued-training` 实验不再作为方法结论使用。它们只说明继续训练可能产生快速漂移，不能证明任务课程本身无效。

新的有效实验全部从 `models/colqwen2.5-base` 开始训练 FolderHomo：

```text
colqwen2.5-base -> FolderHomo 160/160/160 -> checkpoint-500
```

统一固定项：

| 项 | 设置 |
| --- | --- |
| Initialization | `models/colqwen2.5-base` |
| Resume checkpoint | none |
| Warm-start adapter | none |
| Model | FolderHomo real-token compressor |
| Budget | `160/160/160` |
| Loss mode | `flat` |
| Steps | `500` |
| LR | `1e-4` |
| LR scheduler | `constant` |
| Warmup | `0` |
| MARC | off |

P0 from-base runs:

| Priority | Run | Train config | 目的 |
| --- | --- | --- | --- |
| P0 | `core4_flat_sym160_s500_from_base` | `configs/train_worst10_core4.yaml` 或等价 core4 config | 用最小 hard-task mix 判断数据课程是否有整体信号。 |
| P0 | `vqa_hard_flat_sym160_s500_from_base` | `configs/train_vqa_hard.yaml` | 判断 VQA 类低分是否主要来自监督不足。 |
| P0 | `compositional_flat_sym160_s500_from_base` | `configs/train_compositional_hard.yaml` | 判断 CIRR/FashionIQ 类组合检索是否需要单独课程。 |

每个 run 完成后立即评估 `worst10` 与 `retention`，并把 10 个子集明细和平均值写入本文件。

## 实验目的

本目录从训练角度诊断 MMEB 失败：

> 低分子集是否因为全量混合训练中的任务冲突、方向冲突或困难子集监督不足而学不好？

这里不做大规模重训。所有 P0/P1 都是从当前 FolderHomo `sym160` checkpoint 出发的 500-step 小训练，用最低成本判断训练侧是否有希望。

2026-07-03 修正：上面这一路 continued-training 只保留为历史诊断。有效方法实验改为 from-base 500-step 小训练，避免 checkpoint、LR、数据分布和目标函数不一致带来的混杂。

## 实现方案

固定项：

| 项目 | 设置 |
| --- | --- |
| Initialization | `models/colqwen2.5-base` |
| Resume / warm start | forbidden for method conclusions |
| Token budget | `160/160/160` |
| Compress stages | `all` |
| Steps | P0 默认 500 from-base steps |
| LR schedule | constant LR, no warmup |
| Metric in docs | `P@1` |
| Eval launcher | `eval_diagnosis.sh` |

唯一变量：

```text
from-base training subset / curriculum / replay
```

## 优先级

| Priority | Run | Train data | Steps | 目的 |
| --- | --- | --- | ---: | --- |
| P0 | `core4_flat_sym160_s500_from_base` | `configs/train_worst10_core4.yaml` | 500 | 最小 hard-task mix，先看训练课程总方向是否有效。 |
| P0 | `vqa_hard_flat_sym160_s500_from_base` | `configs/train_vqa_hard.yaml` | 500 | 判断 VQA-hard 是否可被 targeted training 快速提升。 |
| P0 | `compositional_flat_sym160_s500_from_base` | `configs/train_compositional_hard.yaml` | 500 | 针对 CIRR/FashionIQ/EDIS 等 compositional hard subsets。 |
| P1 | replay / retention repair | `configs/train_vqa_hard_replay20.yaml` 等 | 500 | 如果 target 提升但 retention 掉，才测试 replay。 |
| P2 | direction curriculum | `train_text_image_warmup.yaml` + staged configs | >=1000 | 只有 from-base 500-step 诊断有效后再设计。 |

## 训练集

| Config | 用途 | 预计覆盖 |
| --- | --- | --- |
| `configs/train_vqa_hard.yaml` | VQA-hard targeted diagnosis | InfographicsVQA / ChartQA / VQA-like tasks where available |
| `configs/train_vqa_hard_replay20.yaml` | VQA-hard + replay | hard subsets + 20% general replay |
| `configs/train_compositional_hard.yaml` | compositional targeted diagnosis | CIRR / FashionIQ / EDIS-like tasks where available |
| `configs/train_text_image_warmup.yaml` | direction curriculum warmup | text->image / stable retrieval |

记录结果时必须写实际可加载的训练集和样本数。配置名不等于真实样本覆盖，必须以服务器可读数据为准。

## 测试集

本目录先不跑完整 MMEB。每个小训练先评估两个 scope：

| Scope | Datasets | 用途 |
| --- | --- | --- |
| `vqa_hard` | InfographicsVQA, ChartQA, A-OKVQA, DocVQA, OK-VQA, Visual7W, GQA, TextVQA, ScienceQA, VizWiz | 看 target hard subsets 是否提升。 |
| `retention` | ImageNet-1K, VOC2007, VisualNews_i2t/t2i, MSCOCO_i2t/t2i, WebQA, VisDial, ChartQA, GQA, CIRR | 看是否灾难性遗忘。 |
| `compositional` | CIRR, NIGHTS, FashionIQ, EDIS | 看 compositional retrieval 是否可学。 |
| `full` | MMEB 36 子集 | 只有 S1/S2 有清晰改善后再跑。 |

## 运行命令

P0 core4 from-base 500 steps:

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/2026-07-01/MMEB任务课程学习

RUN_NAME=core4_flat_sym160_s500_from_base \
MAX_STEPS=500 \
SAVE_STEPS=500 \
SUBSET_CONFIG=configs/train_worst10_core4.yaml \
INTERACTION_LOSS_MODE=flat \
LR_SCHEDULER_TYPE=constant \
WARMUP_RATIO=0 \
WARMUP_STEPS=0 \
LEARNING_RATE=1e-4 \
TRAIN_BSZ=16 \
INTERLEAVED_BSZ=16 \
bash ../MMEB全量/run_train_full.sh
```

Evaluate target:

```bash
CHECKPOINT=runs/core4_flat_sym160_s500_from_base/checkpoint-500 \
SCOPE=vqa_hard \
BATCH_QUERY=16 \
BATCH_PASSAGE=16 \
BATCH_SCORE=64 \
NUM_WORKERS=0 \
bash eval_diagnosis.sh
```

Evaluate retention:

```bash
CHECKPOINT=runs/core4_flat_sym160_s500_from_base/checkpoint-500 \
SCOPE=retention \
BATCH_QUERY=16 \
BATCH_PASSAGE=16 \
BATCH_SCORE=64 \
NUM_WORKERS=0 \
bash eval_diagnosis.sh
```

## 文件说明

| File | Role |
| --- | --- |
| `run_continue_diagnosis.sh` | Historical continued-training launcher; not for method conclusions. |
| `run_first_vqa_diagnosis.sh` | Historical convenience wrapper; not for method conclusions. |
| `eval_diagnosis.sh` | Target/retention/compositional/full eval wrapper。 |
| `summarize_eval_log.py` | 从日志里快速提取 metrics。 |
| `configs/*.yaml` | 小训练或小评测配置。 |

## 实验结果: 总表

| Run | Base ckpt | Train config | Steps | Eval target P@1 | Eval retention P@1 | Full MMEB P@1 | Status | Conclusion |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `core4_flat_sym160_s500_from_base` | `colqwen2.5-base` | `train_worst10_core4.yaml` | 500 | 0.1437 | 0.3056 | - | DONE | 基本打平历史 worst10 baseline，没有明显解决 MMEB 低分问题。 |
| `vqa_hard_flat_sym160_s500_from_base` | `colqwen2.5-base` | `train_vqa_hard.yaml` | 500 | TODO | TODO | - | P0 planned | TODO |
| `compositional_flat_sym160_s500_from_base` | `colqwen2.5-base` | `train_compositional_hard.yaml` | 500 | 0.0490 | 0.2702 | - | DONE | 只训 compositional 会严重伤 VQA/分类，不适合作为统一方案。 |
| historical `*_from_sym160_*` | `sym160 checkpoint-4000` | various | 500 continued | invalid for method | invalid for method | - | diagnostic only | continued tuning drift |

## 实验结果: VQA-hard P@1

| Dataset | Baseline sym160 | `vqa_hard_500` | `vqa_hard_replay20_500` | Notes |
| --- | ---: | ---: | ---: | --- |
| InfographicsVQA | TODO | TODO | TODO | hard |
| ChartQA | TODO | TODO | TODO | hard |
| A-OKVQA | TODO | TODO | TODO | hard |
| DocVQA | TODO | TODO | TODO | hard |
| OK-VQA | TODO | TODO | TODO | hard |
| Visual7W | TODO | TODO | TODO | hard |
| GQA | TODO | TODO | TODO | OOD hard |
| TextVQA | TODO | TODO | TODO | OOD |
| ScienceQA | TODO | TODO | TODO | OOD hard |
| VizWiz | TODO | TODO | TODO | OOD |

## 实验结果: Retention P@1

| Dataset | Baseline sym160 | `vqa_hard_500` | `vqa_hard_replay20_500` | Notes |
| --- | ---: | ---: | ---: | --- |
| ImageNet-1K | TODO | TODO | TODO | classification |
| VOC2007 | TODO | TODO | TODO | classification |
| VisualNews_i2t | TODO | TODO | TODO | image-to-text |
| VisualNews_t2i | TODO | TODO | TODO | text-to-image |
| MSCOCO_i2t | TODO | TODO | TODO | image-to-text |
| MSCOCO_t2i | TODO | TODO | TODO | text-to-image |
| WebQA | TODO | TODO | TODO | strong retrieval |
| VisDial | TODO | TODO | TODO | retrieval |
| ChartQA | TODO | TODO | TODO | hard overlap |
| GQA | TODO | TODO | TODO | hard overlap |
| CIRR | TODO | TODO | TODO | compositional |

## 实验结论

待结果填充。

判断规则：

| 观察 | 结论 |
| --- | --- |
| Target hard P@1 +5 points 以上，retention 下降 <2 points | 任务课程学习值得继续。 |
| Target 提升但 retention 明显下降 | 加 replay，而不是直接 full finetune。 |
| Target 不提升 | 低分更可能来自 scorer 机制、数据格式或表示能力，不应继续堆训练。 |
| 只有 R@5 或候选 recall 变好但 P@1 不变 | 排序校准问题，不能认为表示已解决。 |
