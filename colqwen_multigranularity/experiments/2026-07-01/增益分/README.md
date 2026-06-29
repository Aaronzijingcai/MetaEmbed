# 2026-07-01 增益分 Only: Trainable relative gain ablation

## 实验目的

当前 FolderHomo V1 的 residual gain 定义是：

```text
gain(x) = 1 - max_a cos(x, a)
```

其中 `x` 是当前粒度 residual token，`a` 是已经保留的 coarse anchor token。这个定义稳定，但本质上是一个固定的 hard similarity heuristic：只要某个 coarse anchor 和当前 token 很像，就认为当前 token 没有新增信息。

这对多粒度文档检索可能过硬。重复 OCR、表格行、页眉、数字附近 layout token 看起来相似，但 MaxSim 可能仍然需要这些重复 anchors 来排序。因此本组实验不再继续手写 top-k / coverage / MMR 规则，而是把 gain 改成：

```text
relative + trainable + coarse-anchor-aware
```

目标是找出一种能平行替换 `1 - max similarity` 的可训练信息增益定义。

## 调研依据

近两年 VLM / LMM token compression 的主要信号是：

| 论文 | 借鉴点 | 对本实验的启发 |
| --- | --- | --- |
| LLaVA-PruMerge, 2024 | 先保留重要 token，再基于 key similarity merge 冗余 token。 | merge 比 pure pruning 更适合保留视觉证据，但 similarity metric 不一定要固定。 |
| VisionZip, 2024 | 区分 informative token 和 contextual token，并强调 visual redundancy。 | gain 不能只看单个 token，要看相对上下文/已有视觉证据。 |
| SparseVLM, 2024 | text-guided token sparsification，并回收被剪 token。 | 对检索任务，query-free gain 容易失配；需要可学习信号补偿固定规则。 |
| TokenPacker, 2024 | coarse-to-fine injection，让高分辨率局部 cue 被 coarse query 吸收。 | 我们的 `G1 -> R2 -> R3` 可以看成 coarse anchor 对 finer evidence 的吸收/解释过程。 |
| PAR, 2024 | prompt-aware token reduction，区分 external/internal redundancy。 | redundant 不是绝对概念，应由上下文关系判断。 |
| FastV, 2024 | 早层注意力确定后续视觉 token 稀疏化。 | compression signal 应该来自模型内部可适配的交互，而不是纯后处理。 |
| PyramidDrop, 2024 | 深层 redundancy 逐渐增加，分阶段降低 visual tokens。 | 多粒度 residual compression 的阶段式设计是合理的，但每阶段 gain 应可学习。 |
| FastVLM, 2024 | 通过视觉编码结构减少 tokens，而不是只做后处理 pruning。 | 固定 rule 的上限有限，最好让压缩模块参与训练。 |
| LLaVA-Mini, 2025 | modality pre-fusion 后极端压缩 vision tokens。 | “吸收到已有表示里”比“简单丢掉”更稳，支持 reconstruction/absorption gain。 |
| LVPruning / AdaptPrune / KVTP, 2025 | language/cue/frame relevance 自适应压缩。 | 多图/多帧场景需要相对已有上下文的动态保留率，而不是固定阈值。 |

因此本实验只选可训练、相对 coarse anchors 的 gain 方案。

## 实验变量控制

固定项：

| 变量 | 设置 |
| --- | --- |
| 基础方法 | FolderHomo V1 / residual160 |
| Token budget | `160 / 160 / 160` |
| 压缩阶段 | `G1 + R2 + R3`, `COMPRESS_STAGES=all` |
| 重要分 | 原始 MLP saliency |
| 相似分 | 原始 FOLDER cosine merge score |
| 训练数据 | 与 baseline 相同的 MoCa / MetaEmbed 配置 |
| 训练预算 | 先用 4k step 筛选；若超过或接近 baseline，再考虑更长训练 |
| 评测 | ViDoReV1, ViDoReV2, MMEB text-query focus 三组 |

唯一变量：

```text
coarse-to-fine gain definition
```

不在本次变量内：

| 方法 | 原因 |
| --- | --- |
| `soft_topk_residual` | 仍然是 top-k heuristic，不符合“相对可训练”的目标。 |
| `anchor_subspace_residual` | 虽然是 residual 思路，但核心依赖 top-k anchor selection。 |
| `geo_coverage` | 是 coverage 加项，不是对 `1-max` 的直接可训练替换；已有结果 Avg 约 73.30。 |
| `mmr` | 是同阶段 diversity 加项，不是 coarse-to-fine gain 定义；已有结果 Avg 约 74.39。 |
| `residual_mass` | 主要改变 crop/stage budget allocation，不是单 token gain 的平行替换。 |

## 三种平替方案

### Control: `hard_max`

```text
gain_i = normalize(1 - max_a cos(x_i, a))
```

这是原始 FolderHomo V1 的 gain 定义，用作可复现实验锚点。

### P0-1: `learned_metric_residual`

```text
q_i = W_q x_i
k_a = W_k a
coverage_i = softmax(q_i k_a / tau) · tanh(q_i k_a)
gain_i = sigmoid(b - softplus(s) * coverage_i)
```

动机：保留“相对 coarse anchors 的已解释程度”这个思想，但把 cosine metric 改成可训练 metric。它是和 `1-max similarity` 最接近的平替，因此优先级最高。

预期：如果原始 hard cosine 不是最佳 metric，这个方案应该能提升 ViDoReV2，同时不大幅伤 MMEB。

### P0-2: `learned_anchor_gate`

```text
context_i = CrossAttn(x_i, anchors)
gain_i = MLP([x_i, context_i, x_i - context_i, x_i * context_i, hard_gain_i, mean_sim_i, entropy_i])
```

动机：借鉴 prompt-aware / language-guided / multi-cue pruning 的思想，让模型自己判断当前 token 和 coarse anchors 的关系。它不只问“像不像”，还显式看到 residual difference、interaction product、hard gain、平均相似度和 anchor attention entropy。

预期：适合处理“相似但仍有用”的 OCR/layout 重复 anchors，是最有希望超过 fixed hard gain 的方案。

### P1: `learned_reconstruction_residual`

```text
context_i = CrossAttn(x_i, anchors)
recon_i = MLP([x_i, context_i, x_i - context_i])
gain_i = normalize(||x_i - recon_i||_2)
```

动机：借鉴 TokenPacker / LLaVA-Mini 的 absorption 思想：如果当前 token 能被 coarse anchors 吸收/重构，则它的信息增益低；如果不能重构，则应该被保护。

优先级设为 P1，因为它参数更多，训练不稳定风险比前两个 P0 高。

## 运行命令

Baseline 复现：

```bash
cd /Users/czj/MetaEmbed_github_20260623/colqwen_multigranularity/experiments/2026-07-01/增益分
GAIN_MODE=hard_max RUN_NAME=folder_gain_only_v1_hard_max_b160_160_160_4k bash run_train.sh
```

P0-1 trainable metric residual：

```bash
GAIN_MODE=learned_metric_residual GAIN_TAU=0.07 \
RUN_NAME=folder_gain_only_v1_learned_metric_residual_b160_160_160_4k bash run_train.sh
```

P0-2 cross-anchor gate：

```bash
GAIN_MODE=learned_anchor_gate \
RUN_NAME=folder_gain_only_v1_learned_anchor_gate_b160_160_160_4k bash run_train.sh
```

P1 reconstruction residual：

```bash
GAIN_MODE=learned_reconstruction_residual \
RUN_NAME=folder_gain_only_v1_learned_reconstruction_residual_b160_160_160_4k bash run_train.sh
```

Smoke 训练和测试：

```bash
GAIN_MODE=learned_metric_residual bash smoke_train_eval.sh
GAIN_MODE=learned_anchor_gate bash smoke_train_eval.sh
GAIN_MODE=learned_reconstruction_residual bash smoke_train_eval.sh
```

评测：

```bash
GAIN_MODE=learned_metric_residual bash eval_3sets.sh runs/folder_gain_only_v1_learned_metric_residual_b160_160_160_4k/checkpoint-4000
GAIN_MODE=learned_anchor_gate bash eval_3sets.sh runs/folder_gain_only_v1_learned_anchor_gate_b160_160_160_4k/checkpoint-4000
GAIN_MODE=learned_reconstruction_residual bash eval_3sets.sh runs/folder_gain_only_v1_learned_reconstruction_residual_b160_160_160_4k/checkpoint-4000
```

## 实验结果

当前 baseline 锚点：

| 方法 | ViDoReV1 | ViDoReV2 | MMEB | Avg | 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| FolderHomo V1 / residual160, `160/160/160` | 89.34 | 60.28 | 76.43 | 75.35 | 已有 |
| FolderHomo V1 / residual160, ckpt2500 variant | 89.20 | 61.60 | 75.55 | 75.45 | 已有 |

本实验待填：

| Priority | Gain mode | ViDoReV1 | ViDoReV2 | MMEB | Avg | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Control | `hard_max` | TBD | TBD | TBD | TBD | 复现当前 `1-max sim` |
| P0 | `learned_metric_residual` | TBD | TBD | TBD | TBD | 待跑 |
| P0 | `learned_anchor_gate` | TBD | TBD | TBD | TBD | 待跑 |
| P1 | `learned_reconstruction_residual` | TBD | TBD | TBD | TBD | 有时间再跑 |

## 实验结论

如果两个 P0 都不能超过 `hard_max` 或 residual160，结论应收缩为：

> FolderHomo 的收益主要来自 coarse-to-fine residual token path 和 FOLDER merge 的结构性压缩，而不是复杂 gain proxy；`1-max similarity` 虽然简单，但作为 query-free residual proxy 已经足够稳。

如果 `learned_anchor_gate` 或 `learned_metric_residual` 超过 baseline，则下一步只保留胜出方向做更长训练和 budget curve，不再同时扩多种 gain 定义。

## References

1. Shang et al., [LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models](https://arxiv.org/abs/2403.15388), 2024.
2. Yang et al., [VisionZip: Longer is Better but Not Necessary in Vision Language Models](https://arxiv.org/abs/2412.04467), 2024.
3. Zhang et al., [SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference](https://arxiv.org/abs/2410.04417), 2024.
4. Li et al., [TokenPacker: Efficient Visual Projector for Multimodal LLM](https://arxiv.org/abs/2407.02392), 2024.
5. Liu et al., [PAR: Prompt-Aware Token Reduction Method for Efficient Large Multimodal Models](https://arxiv.org/abs/2410.07278), 2024.
6. Chen et al., [An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models](https://arxiv.org/abs/2403.06764), 2024.
7. Xing et al., [PyramidDrop: Accelerating Your Large Vision-Language Models via Pyramid Visual Redundancy Reduction](https://arxiv.org/abs/2410.17247), 2024.
8. Vasu et al., [FastVLM: Efficient Vision Encoding for Vision Language Models](https://arxiv.org/abs/2412.13303), 2024.
9. Zhang et al., [LLaVA-Mini: Efficient Image and Video Large Multimodal Models with One Vision Token](https://arxiv.org/abs/2501.03895), 2025.
10. Sun et al., [LVPruning: An Effective yet Simple Language-Guided Vision Token Pruning Approach for Multi-modal Large Language Models](https://arxiv.org/abs/2501.13652), 2025.
11. Luan et al., [Multi-Cue Adaptive Visual Token Pruning for Large Vision-Language Models](https://arxiv.org/abs/2503.08019), 2025.
12. Liu et al., [Keyframe-oriented Vision Token Pruning: Enhancing Efficiency of Large Vision Language Models on Long-Form Video Processing](https://arxiv.org/abs/2503.10742), 2025.
