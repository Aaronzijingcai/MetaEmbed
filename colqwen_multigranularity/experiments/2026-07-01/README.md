# 2026-07-01 实验总览

Updated: 2026-07-04

老板汇报用的集中结果表见 [`实验记录与汇报表.md`](./实验记录与汇报表.md)。本文件保留研究边界、目录职责和统一记录规则。

## 2026-07-04 Server A TopK Sweep 结果

在 `core4_global_local_sym160_s500_from_base/checkpoint-500` 上做免训练 MaxSim scorer sweep，固定 FolderHomo `160/160/160`，只改变 query-side TopK 数量。测试范围为 `worst10 + retention`，指标统一为 `P@1`。

| Scorer | Worst10 P@1 | Retention P@1 | 结论 |
| --- | ---: | ---: | --- |
| `q2d_query_topk32_sym160` | **0.3105** | 0.4048 | 当前最强；比 topK64 继续提升，说明 query 端视觉冗余仍然很重。 |
| `q2d_query_topk48_sym160` | 0.2977 | 0.3914 | 好于 topK64，但弱于 topK32。 |
| `q2d_query_topk64_sym160` | 0.2850 | 0.3998 | 上一轮最强 baseline。 |
| `q2d_query_topk96_sym160` | 0.2620 | **0.4077** | retention 略高，但 worst10 明显下降，说明保留过多 query token 会重新引入噪声。 |

Worst10 明细：

| Dataset | TopK32 | TopK48 | TopK64 | TopK96 |
| --- | ---: | ---: | ---: | ---: |
| FashionIQ | 0.037 | 0.038 | 0.038 | 0.039 |
| CIRR | 0.117 | 0.116 | 0.116 | 0.120 |
| Country211 | 0.131 | 0.139 | 0.137 | 0.104 |
| GQA | 0.251 | 0.252 | 0.244 | 0.228 |
| ScienceQA | 0.341 | 0.319 | 0.321 | 0.331 |
| OK-VQA | 0.361 | 0.341 | 0.324 | 0.299 |
| A-OKVQA | 0.384 | 0.380 | 0.358 | 0.339 |
| Visual7W | 0.448 | 0.439 | 0.425 | 0.393 |
| ChartQA | 0.516 | 0.488 | 0.476 | 0.430 |
| InfographicsVQA | 0.519 | 0.465 | 0.411 | 0.337 |
| **Average** | **0.3105** | **0.2977** | **0.2850** | **0.2620** |

当前核心结论：

1. MMEB worst10 的主要可修复问题不是简单训练步数不足，而是 query 端视觉 token 冗余会放大 directed MaxSim 的局部噪声。
2. `q2d_query_topk32` 比 `topk64` 更强，说明推理时只保留最有响应的少量 query tokens 能显著缓解 VQA / Chart / Infographic 类失败。
3. `topk96` 在 retention 上略高，但 worst10 明显下降，因此不能简单扩大 query token 数；下一步训练目标应优先对齐 `q2d_query_topk32`。
4. CIRR/FashionIQ 仍然低，说明 compositional retrieval 不是 query-side TopK 能单独解决的问题，后续需要单独分析文本修改关系、global anchor 或任务构造。

下一步 P0 训练决策已经收敛到 `MaxSim交互/` 的 6 组 1k from-base 训练。为了避免继续扫 TopK 数值，凡是使用 TopK 的机制统一固定为 `K=48`。训练数据必须包含 ViDoRe train / visual-document train，同时加入 MMEB hard/core4 train，目标是在保住 ViDoRe 主线的前提下缓解 MMEB worst10。

统一设置：

```text
Base model=models/colqwen2.5-base
Model=FolderHomo
Budget=160/160/160
Steps=1000
LR=2e-4
LR scheduler=constant
Warmup=0
Resume / warm start=forbidden
Eval=MMEB worst10 P@1 + ViDoRe v2 nDCG@5
```

三台机器分工：

| Machine | Runs | 机制组 |
| --- | --- | --- |
| A | `vidore_mmeb_q2d_mean_s1k_from_base`, `vidore_mmeb_qtopk48_s1k_from_base` | 单向基线组 |
| B | `vidore_mmeb_bi_mean_lam07_s1k_from_base`, `vidore_mmeb_bi_adaptive_s1k_from_base` | 双向无 TopK 组 |
| C | `vidore_mmeb_bi_qtopk48_lam07_s1k_from_base`, `vidore_mmeb_bi_qtopk48_adaptive_s1k_from_base` | TopK + 双向组合组 |

## 2026-07-04 Server B from-base 结果

Server B 三组 `from_base_s500` 已完成。所有训练均从 `models/colqwen2.5-base` 初始化，`RESUME_CKPT=`、`WARM_START_ADAPTER_PATH=` 为空，`TRAIN_BSZ=12`，`INTERLEAVED_BSZ=12`，`LR=1e-4`，`constant` schedule，`warmup=0`，`MARC=off`。

| Run | Train config | Loss mode | Worst10 P@1 | Retention P@1 | 结论 |
| --- | --- | --- | ---: | ---: | --- |
| `core4_flat_sym160_s500_from_base` | `train_worst10_core4.yaml` | `flat` | 0.1437 | 0.3056 | 基本打平历史 worst10 baseline，没有明显解决 MMEB 低分问题。 |
| `core4_factorized_local_sym160_s500_from_base` | `train_worst10_core4.yaml` | `factorized_local` | 0.1370 | 0.2772 | 分通道 local interaction 没有收益，整体弱于 flat。 |
| `compositional_flat_sym160_s500_from_base` | `train_compositional_hard.yaml` | `flat` | 0.0490 | 0.2702 | 只训 compositional 会严重伤 VQA/分类，不适合作为统一方案。 |

当前结论：这批结果排除了“之前失败完全是 warm-start/continued tuning 漂移”的单一解释。即使从 base 重新训练 500 step，`core4` 课程只能恢复到历史 worst10 baseline 附近，`factorized_local` 也没有带来 interaction-loss 层面的有效改善。后续不应继续简单堆同类 `core4` 或 compositional-only 500-step 训练，应优先比较 Server A 的 `fullmix_flat`、`global_local`、`vqa_hard` 结果，并回到 scorer-only、数据格式/答案表征、以及更合理的分组 full-mix sampler。

## 2026-07-03 from-base rerun plan

今天开始，所有用于下结论的 MMEB 训练实验统一改为 `from_base_s500`：

```text
models/colqwen2.5-base -> FolderHomo train -> checkpoint-500
```

不再使用 `sym160 checkpoint-4000` 做 continued tuning。之前所有 `from_sym160` / `continue` / `checkpoint-4500` 训练结果只作为“继续训练不稳定”的诊断记录，不能用于判断 loss、数据课程或训练策略本身好坏。

统一固定项：

| 项 | 设置 |
| --- | --- |
| Base model | `models/colqwen2.5-base` |
| 主模型 | FolderHomo / FOLDER-Homo real-token compressor |
| Token budget | `160/160/160` |
| Compress stages | `all` |
| Trainable modules | LLM LoRA + `custom_text_proj` + `folder_homo` compressor |
| LoRA target | `down_proj/gate_proj/up_proj/k_proj/q_proj/v_proj/o_proj` |
| Steps | `500` |
| LR policy | `constant`, no warmup, no linear decay |
| LR | `1e-4` unless explicitly recorded |
| MARC | off |
| Resume / warm start | forbidden for method conclusions |

三大 MMEB 版块重新定位：

| 版块 | 研究变量 | P0 from-base runs |
| --- | --- | --- |
| `MMEB全量/` | 全量混合训练是否能在 500 step 内产生有效 MMEB 信号 | `fullmix_flat_sym160_s500_from_base` |
| `MMEB任务课程学习/` | 训练数据 / 任务课程是否是 MMEB 失败主因 | `core4_flat_sym160_s500_from_base`, `vqa_hard_flat_sym160_s500_from_base`, `compositional_flat_sym160_s500_from_base` |
| `MaxSim交互/` | loss / interaction 机制是否能修复 directed MaxSim 失败 | `core4_global_local_sym160_s500_from_base`, `core4_factorized_local_sym160_s500_from_base` |

所有 P0 run 先评估 `worst10` MMEB 子集和 `retention` 子集，只记录 `P@1`。只有 top-2 方法再跑完整 36 个 MMEB 子集。

## 研究边界

这一批实验必须保持精简。图像 token 压缩、多粒度同质化和 FolderHomo real-token compression 已经是当前主线基础，不再在这里继续扩散新压缩家族。当前要解决的核心问题只有一个：

> 当前 FolderHomo 模型在 MMEB 上仍存在明显失败，尤其是 VQA、image-query、image+text-to-image+text 和 grounding 相关子集。我们需要用最小成本判断失败来自 MaxSim 机制、query/token 非对称，还是训练任务课程。

MetaEmbed / learnable tokens 是外部 baseline 和需要打败的对象，不放入本目录内部消融。本目录只研究 FolderHomo 压缩后的真实视觉 token。

## 目录结构

| 子目录 | 问题 | 实验类型 | 当前优先级 |
| --- | --- | --- | --- |
| `探索重要分/` | FOLDER merge 时哪些 token 应该被保护？ | 训练消融 | 已基本完成，记录结果即可 |
| `增益分/` | finer token 相对 coarse anchors 的增益应如何定义？ | 训练消融 | 正在跑/记录结果 |
| `MMEB全量/` | FolderHomo 在 MMEB full setting 的真实表现和失败边界是什么？ | from-base full-mix train / full eval / query-side budget | P0 |
| `MaxSim交互/` | directed MaxSim 是否是 MMEB 失败主因？ | scorer-only + from-base interaction-loss train | P0 |
| `MMEB任务课程学习/` | 低分是否来自任务混合冲突或训练课程缺失？ | from-base 500-step 小训练诊断 | P0 |

不要再创建 `MMEB问题诊断/` 这类重复入口。跨目录结论写在本文件；每个具体实验的配置、命令和结果写在对应子目录的 `README.md`。

## 当前策略

旧 P0 免训练诊断仍然有效：

- `MMEB全量/`: 先完成 `sym160` full MMEB P@1 记录。
- `MMEB全量/`: 只做 `q80/doc160`、`q40/doc160` 两个 query-side asymmetric budget eval。
- `MaxSim交互/`: 固定 `sym160` checkpoint，跑 `q2d_sum`、`q2d_mean`、`bi_mean`。

新 P0 from-base 训练：

- 当前主线只推进 `MaxSim交互/` 的 6 组 `vidore_mmeb_*_s1k_from_base`。
- 旧的 `core4_flat`、`vqa_hard_flat`、`compositional_flat`、`factorized_local` 已作为诊断完成或归档，不再继续扩展同类训练。
- `MMEB全量/` 的 fullmix 直接随机 interleave 曾出现 step-0 卡住；后续如需 fullmix，必须先做 grouped/bucketed sampler。

P1:

- `MaxSim交互/`: 若 6 组 P0 中 TopK + adaptive 仍不足，再考虑 `bi_lse`、coverage 正则或 global anchor。
- `MMEB任务课程学习/`: replay / staged curriculum。只有 P0 from-base target 提升但 ViDoRe 掉点时再跑。
- `MMEB全量/`: grouped full-mix sampler。当前 fullmix 直接混合在 `BSZ=12` 下出现 step-0 卡住，现象与样本长度/模态结构差异过大一致。后续如果重新做 fullmix，不再把全部任务直接 interleave 到同一个 batch，而是先分成 3 个大组；每个 training step 先随机抽一个大组，再从该大组内部抽一个任务/小组组成 8 卡 batch，以保持 query/doc 长度和模态结构更一致。

P2:

- `bi_lse`。
- `sym80/sym40/sym20`。
- full-scale row+col loss、coverage loss 重训。

`sym80/sym40/sym20` 暂时后置。它们可以缓解视觉 query 太长的问题，但如果 MaxSim 机制或任务课程本身没有解决，它们更像治标方案。只有当 `MaxSim交互/` 和 `MMEB任务课程学习/` 都无法给出清晰改善时，再把更小 budget 作为最后补充。

## 统一记录规则

每个子目录的 `README.md` 必须包含：

- 实验目的。
- 实现方案。
- P0/P1/P2 优先级。
- 训练集。
- 测试集。
- 变量控制。
- 运行命令。
- 实验结果表。
- 实验结论。

MMEB 结果只记录 `P@1`。当前 evaluator 底层字段叫 `recall_at_1`，但 MMEB evaluator 注释说明它实际是 local candidate set 上的 hit@1；在本批实验文档里统一写作 `P@1`，避免继续混用 `recall@1`、`recall@5` 和旧表口径。

如果是完整 MMEB eval，必须记录 36 个子集的 P@1，并记录：

- checkpoint。
- train config。
- eval config。
- query/doc budget。
- MaxSim interaction。
- query aggregation。
- batch query / passage / score。
- 是否 full MMEB 或只跑 subset。

不要只报 macro avg。MMEB 的目标是定位失败子集，36 子集表格再大也要保留。

## 当前结果索引

| 实验线 | Run | Status | 结果文件 | 备注 |
| --- | --- | --- | --- | --- |
| `探索重要分/` | `mlp` | TODO | `探索重要分/README.md` | 记录 ViDoReV1/ViDoReV2/MMEB 三组 |
| `探索重要分/` | `mha_attn` | TODO | `探索重要分/README.md` | P0 |
| `探索重要分/` | `learned_gate` | RUNNING/DONE | `探索重要分/README.md` | 以实际服务器状态为准 |
| `增益分/` | `learned_anchor_gate` | RUNNING | `增益分/README.md` | 用户确认正在跑 |
| `MMEB全量/` | `asym_q80_t160` | RUNNING | `MMEB全量/README.md` | 用户确认正在跑 |
| `MMEB全量/` | `asym_q40_t160` | RUNNING | `MMEB全量/README.md` | 用户确认正在跑 |
| `MaxSim交互/` | `legacy_q2d_sum_sym160` | DONE | `MaxSim交互/README.md` | full MMEB P@1=0.4611 |
| `MaxSim交互/` | `q2d_mean_sym160` | RUNNING | `MaxSim交互/README.md` | 2026-07-02 23:08 接续启动 |

更适合汇报的最新结果矩阵请以 `实验记录与汇报表.md` 为准；该文件只记录索引，不重复维护完整数值表。

## 写作原则

论文表述要收敛为：

1. FolderHomo 解决的是多粒度视觉 token 的真实 token 冗余，不是 learnable token 替代。
2. MMEB 失败暴露的是 directed MaxSim 在 image-query / pair-to-pair 场景中的结构性边界。
3. 我们用最小成本实验先诊断 scorer 机制和任务课程，再决定是否需要更激进 token budget。

不要把单次小幅差异写成“显著”。没有多 seed 或统计检验时，只能写 “suggests / indicates / is consistent with”。
