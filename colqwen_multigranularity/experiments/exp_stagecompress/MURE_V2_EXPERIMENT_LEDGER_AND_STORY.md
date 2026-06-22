# MURE-V2 实验汇总与论文叙述路线

更新时间：2026-06-15

本文档汇总当前 MURE-V2 / `exp_stagecompress` 阶段已经完成、正在运行、暂停和后续需要补充的实验，并按照论文叙述逻辑重新组织：从单粒度验证，到 Oracle 上限，到 MRL-main，再到单图压缩，最后进入多图/多粒度同质化压缩。

## 0. 当前技术路线一句话

```text
X-VisEmbed 启发的多粒度采样
-> 多粒度感知
-> 单粒度 / 单图 token 冗余压缩横向比较
-> 发现 merge-based FOLDER 最适合视觉文档检索
-> 以 FOLDER 作为基础压缩 primitive
-> 设计跨粒度 residual homogeneity compression
-> 用 MRL 监督 G1, G1+R2, G1+R2+R3
```

核心论文主张：

```text
多粒度视觉文档检索不仅需要更多视角，还需要去除跨视角重复证据。
真实视觉 token 的 merge-based residual compression 可以在保留 OCR/layout 证据的同时降低 token 数量，并缓解 MaxSim 中冗余 token 带来的噪声。
```

## 1. 论文叙述主线

### 1.1 单粒度验证

目的：验证 Qwen/ColQwen 系列下，不同视觉粒度是否仍然能捕获不同信息。

| 方法 | 视图 | ViDoReV1 | ViDoReV2 | MMEB | Step | 结论 |
|---|---|---:|---:|---:|---:|---|
| `g1` | `1x1` | 86.7 | 56.9 | 61.6 | 16k | 全局布局有效 |
| `g2` | `1x2/2x1` | 85.5 | 53.8 | 53.7 | 16k | 当前中尺度方案不是最优 |
| `g3` | `2x2` | 86.0 | 59.4 | 51.1 | 16k | 细粒度对 ViDoReV2 更有帮助 |

关键观察：

```text
g1/g2/g3 各自有不同优势，说明多粒度方向仍然有意义。
但 1x2/2x1 方案可能存在拉伸和信息利用不足问题。
```

[TODO] 后续可以在大实验阶段重新检查采样策略，例如 `1x1 + 2x2`、更稳定的非拉伸中尺度 crop，或动态长宽比 crop。

### 1.2 Oracle 上限

目的：验证多粒度之间是否真的有互补信息。

| 方法 | ViDoReV1 | ViDoReV2 | MMEB | Step | 结论 |
|---|---:|---:|---:|---:|---|
| Oracle | 91.4 | 70.2 | 71.8 | 16k | 明显高于任一单粒度 |

关键观察：

```text
Oracle 大幅高于单粒度，说明不同粒度之间存在互补视觉证据。
```

论文可写：

```text
This gap motivates a multi-granularity retrieval representation, but also raises the question of how to fuse and compress redundant visual evidence across granularities.
```

### 1.3 MRL-main 验证

目的：验证基础 MRL 结构能否把多粒度整合成一个可训练的 retrieval representation。

| 方法 | 输入/输出组织 | ViDoReV1 | ViDoReV2 | MMEB | Step | 结论 |
|---|---|---:|---:|---:|---:|---|
| MRL no text | `g1/g2/g3` | 87.6 | 58.4 | 35.6 | 16k | ViDoRe 尚可，MMEB 崩 |
| MRL with text | text + visual | 82.4 | 50.0 | 37.1 | 16k | 简单拼 text 没有收益 |
| MRL-main baseline | 当前强 baseline | 89.8 | 61.0 | 75.8 | 4k | 当前强无压缩参考 |

关键观察：

```text
MRL 在 ViDoRe 上可行，但 MaxSim 在 MMEB / 图文配对场景下存在明显边界。
```

原因分析：

```text
MaxSim 更适合 query 短、target 长的少配多场景。
当 query 长、target 短，或多配多/多配少时，MaxSim 的非对称性会放大噪声。
```

当前决策：

```text
本文主线不直接修改 MaxSim，而是先通过 token 压缩减少冗余和长度不均衡带来的噪声。
```

[TODO] 可学习 token 路线单独推进，用于解决多配多/多配少下 MaxSim 长度不均衡问题。

## 2. 参考 Baseline

| 方法 | ViDoReV1 | ViDoReV2 | MMEB | 角色 |
|---|---:|---:|---:|---|
| MURE-V1 | 87.0 | 59.5 | - | 旧版本多粒度参考 |
| ColPali | 84.9 | 54.5 | 34.9 | 典型 late-interaction 多向量参考 |
| MetaEmbed | 88.7 | 60.3 | 69.1 | 固定 token / 学习 token 思路参考 |
| MRL-main baseline | 89.8 | 61.0 | 75.8 | 当前无压缩强 baseline |

论文叙述建议：

```text
MetaEmbed 的稳定性说明固定长度 token 表征可能缓解 MaxSim 长度不均衡；
但本文当前主线聚焦在多粒度视觉文档索引冗余，因此优先研究真实视觉 token 的 residual compression。
```

## 3. 单图 / 单粒度压缩横向比较

这一阶段先不考虑 `g1/g2/g3` 之间的同质化关系，只把每个粒度内部看成独立 token sequence，比较主流视觉 token 压缩思想。

| 方法 | 类型 | 压缩位置 | ViDoReV1 | ViDoReV2 | MMEB | Avg | 状态 | 结论 |
|---|---|---|---:|---:|---:|---:|---|---|
| MRL-main baseline | 无压缩 | - | 89.8 | 61.0 | 75.8 | 75.5 | DONE | 参考上限 |
| MetaEmbed | 无压缩 / fixed tokens | LLM 前 | 83.1 | 52.9 | 73.6 | 69.9 | DONE | 参考 |
| SoftAssign / strategy1 | 合并 | MLP 后 | 81.2 | 47.4 | 72.1 | 66.9 | DONE | 弱 |
| PruMerge / strategy3 | 剪枝+合并 | MLP 后 | 87.9 | 58.3 | 75.3 | 73.8 | DONE | 强参考 |
| VisionZip / strategy4 | 剪枝+合并 | MLP 后 | 87.7 | 57.8 | 73.3 | 72.9 | DONE | 强参考 |
| FOLDER / strategy5 | 合并 | MLP 后 | 89.6 | 58.8 | 75.1 | 74.5 | DONE | 单图压缩最佳锚点 |
| SCOPE / strategy6 | 剪枝 | MLP 后 | 88.6 | 57.2 | 75.1 | 73.6 | DONE | 强参考 |
| Stage Resampler / strategy7 | 可学习 token | MLP 后 | 80.8 | 45.0 | 70.1 | 65.3 | DONE | 弱 |

阶段结论：

```text
在视觉文档检索任务中，merge-based 方法比 pure pruning 更稳。
FOLDER 是最强的单图压缩 anchor。
```

论文叙述建议：

```text
Unlike generation-oriented token pruning, visual document retrieval is sensitive to rare OCR and layout evidence. Pure pruning may delete critical evidence, while merge-based compression can preserve information by absorbing redundant tokens into retained tokens.
```

图表建议：

| 图 | 内容 |
|---|---|
| 单图压缩散点/圆圈图 | x 轴为方法类别，y 轴为 Avg，圆圈大小为 token 数或压缩率 |
| bar chart | PruMerge / VisionZip / FOLDER / SCOPE 横向对比 |
| method taxonomy | pruning, pruning+merge, merge, learnable token |

[TODO] 统一所有单图压缩方法的 token 数 / 压缩比记录，方便画圆圈图。

## 4. LLM-pre / 可训练压缩早期探索

这一阶段探索了更激进的可训练压缩，主要包括 learnable token、soft/hard prune 和 LLM-pre 压缩。

| 方法 | 类型 | 压缩位置 | ViDoReV1 | ViDoReV2 | MMEB | Avg | 状态 | 结论 |
|---|---|---|---:|---:|---:|---:|---|---|
| Learnable Global MRL Tokens | 可学习 token | LLM 前 | 78.0 | 46.1 | 70.6 | 64.9 | DONE | 训练难，效果弱 |
| Learnable Global MRL Tokens + TwigStage | 可学习 token + 剪枝 | LLM 前 | 81.6 | 49.6 | 69.1 | 66.8 | DONE | 有提升但仍弱 |
| SoftStageMRL | 剪枝 | LLM 前 | 74.3 | 43.4 | 68.1 | 61.9 | DONE | 弱 |
| VisionSelectorMRL | 软训练硬推理剪枝 | LLM 前 | 16.9 | 14.6 | 2.8 | 11.4 | DONE/FAILED | 当前实现失败 |
| TwigMRL | 剪枝 | LLM 浅层 | TODO | TODO | TODO | TODO | PAUSED | loss 高、训练成本过大 |
| VisionZipMRL | 剪枝+合并 | LLM 前 | TODO | TODO | TODO | TODO | PAUSED | 暂不主推 |

阶段结论：

```text
可学习 token 和 LLM-pre 剪枝理论上可以更直接解决 token 长度问题，但当前训练难度高，结果不如 MLP-post FOLDER 路线稳定。
```

当前决策：

```text
可学习 token 不作为当前 FOLDER/HomoFolder 主论文的必要组件。
它更适合作为第二条路线，用来解决多配多/多配少下 MaxSim 长度不均衡问题。
```

[TODO] 低成本实现 `Learnable Residual Tokens`：`T1`, `T2`, `T3` residual resampler，对应 `G1/R2/R3` 思想，但输出固定长度 learnable tokens。

[TODO] 若 learnable residual token 在 MMEB 多配多/多配少上显著提升，考虑拆成第二篇论文。

## 5. Training-free / Qwen-pre 压缩探索

这一阶段主要参考 AngelSlim / DART / DivPrune / FastV / HiPrune / VisionZip / SCOPE 等思想，尝试在 Qwen2.5VL 上做免训练压缩。

| 方向 | 状态 | 当前结论 |
|---|---|---|
| AngelSlim Qwen-pre 多方法适配 | 已做大量 smoke / full 尝试 | 效果整体不理想，且与当前论文主线偏离 |
| training-free group budget `160/320/640` | 已尝试 | 难以达到 trainable HomoFolder 效果 |
| `r=0.5` / `r=0.9` retention ratio | 已尝试 | 过强压缩容易明显伤害效果 |
| DART / DivPrune / FastV / HiPrune / VisionZip / SCOPE Qwen-pre | 已有 run 目录 | 暂作为归档参考，不进入主线 |

阶段结论：

```text
training-free 方法能作为 baseline，但不能让表示空间适配压缩后的 token 结构。
当前主论文应聚焦 trainable residual homogeneity compression。
```

[TODO] 如果论文需要补充 training-free baseline，可以只选 1-2 个代表方法，而不是完整展开 AngelSlim 大矩阵。

## 6. 多图 / 多粒度同质化压缩

### 6.1 问题定义

单图压缩只能去掉每个视图内部冗余，但 MURE-V2 的输入天然是多粒度图像集合：

```text
g1: 全局视图
g2: 中尺度视图
g3: 细尺度视图
```

它们会重复编码相同 OCR、表格、布局块。因此需要跨粒度同质化压缩：

```text
Level 1: G1
Level 2: G1 + R2
Level 3: G1 + R2 + R3
```

### 6.2 当前实现逻辑

`g1` 压缩：

```text
g1 -> internal saliency/gate -> FOLDER merge -> G1
```

`g1` 没有更粗 anchor，因此只使用内部图像信号。

`g2/g3` 压缩：

```text
R2 = FOLDER(g2, internal saliency + novelty to G1)
R3 = FOLDER(g3, internal saliency + novelty to G1+R2)
```

所以 `g2/g3` 不是只靠 `g1` 信号，而是：

```text
内部图像重要性 + 跨粒度新增信息
```

当前正在跑的 `Global-Guided HomoFolder` 进一步加入 crop-level guidance：

```text
G1 -> crop commander -> 给 g2/g3 的不同 crop 分配 residual budget
```

### 6.3 同质化结果表

| 方法 | 类型 | Token | ViDoReV1 | ViDoReV2 | MMEB | Avg | 状态 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|---|
| MLP-post FOLDER | 单图 merge | 1120 | 89.6 | 58.8 | 75.1 | 74.5 | DONE | 同质化路线基础 |
| FolderHomo v1 | trainable homogeneity | 1120 | 89.27 | 59.20 | 75.88 | 74.78 | DONE | 第一版可训练同质化成功 |
| FolderHomo residual160 | trainable homogeneity | 480 | 89.34 | 60.28 | 76.43 | 75.35 | DONE | 当前最强完成结果 |
| FolderHomo v1 80/80/80 | trainable homogeneity | 240 | 88.44 | 56.10 | 74.53 | 73.02 | DONE | 3k 强压缩 ablation；ViDoReV2 下滑，240-token 下限可用但不替换主线 |
| FolderHomo prefix L1 | prefix ablation | 160 | 88.36 | 59.09 | 74.93 | 74.12 | DONE | `G1` only |
| FolderHomo prefix L2 | prefix ablation | 320 | 89.20 | 59.73 | 75.75 | 74.89 | DONE | `G1+R2`, 接近 480 |
| Global-Guided HomoFolder / V2 | trainable homogeneity | 480 | 89.08 | 58.44 | 73.75 | 73.76 | DONE | 未超过 residual160，保留为 crop-level guidance ablation |
| FolderGainHomo geo_coverage | gain-based homogeneity | 480 | 88.98 | 56.46 | 74.45 | 73.30 | DONE | coverage gain ablation；ViDoReV2 下降明显 |
| FolderGainHomo residual_mass | gain-based homogeneity | 480 | 88.83 | 59.32 | 74.78 | 74.31 | DONE | 动态残差信息量预算；优于 geo，但仍低于 residual160 |
| FolderGainHomo V5 / MMR | gain-based homogeneity | 480 | 88.96 | 58.76 | 75.45 | 74.39 | DONE | MMR/diversity gain ablation，低于 residual160 主线 |
| FolderGainHomo V6 / residual_mass_mmr | gain-based homogeneity | 480 | 88.27 | 59.42 | 74.25 | 73.98 | DONE | residual_mass 预算 + MMR 去冗余；低于单独 residual_mass/MMR |
| DART-Pivot HomoFolder | trainable homogeneity | 480 | TODO | TODO | TODO | TODO | SMOKE PASSED | P1 formal TODO |
| GlobalCom-DART Fusion | trainable homogeneity | 480 | TODO | TODO | TODO | TODO | SMOKE PASSED | P1 formal TODO |

最关键结论：

```text
FolderHomo residual160 用 480 tokens 达到 75.35 Avg，高于 1120-token FolderHomo v1 的 74.78。
```

这说明：

```text
多粒度 token 不是越多越好。
跨粒度重复证据会增加 MaxSim 噪声。
合理 residual compression 可以同时降低 token 数和提升效果。
```

2026-06-14 prefix ablation：

| Prefix | Token | ViDoReV1 | ViDoReV2 | MMEB | Avg | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `G1` | 160 | 88.36 | 59.09 | 74.93 | 74.12 | 已经可用，但仍损失细节 |
| `G1 + R2` | 320 | 89.20 | 59.73 | 75.75 | 74.89 | 主要边际收益段，效率点好 |
| `G1 + R2 + R3` | 480 | 89.34 | 60.28 | 76.43 | 75.35 | 最强完整表征 |

补充判断：

```text
320 tokens 已经接近 480 tokens，但 480 仍然最好。
因此后续不能简单假设 token 越少越能解决 MaxSim；
更合理的目标是找到“更少且更干净”的同质化 token。
```

2026-06-15 gain-based homogeneity 结果补充：

```text
FolderGainHomo geo_coverage: ViDoReV1 88.98 / ViDoReV2 56.46 / MMEB 74.45 / Avg 73.30
Run: experiments/exp_stagecompress/runs/folder_gain_homo_geo_coverage_b160_160_160_bsz4_gc_20260614_120847
Eval: eval/folder_gain_homo_geo_coverage_ckpt4000_full_8gpu_b160_160_160_workers0_20260615_175913

FolderGainHomo residual_mass: ViDoReV1 88.83 / ViDoReV2 59.32 / MMEB 74.78 / Avg 74.31
Run: experiments/exp_stagecompress/runs/folder_gain_homo_residual_mass_native_qwen25_lora_linear_gain_b160_160_160_bsz4_gc_20260614_120913
Eval: eval/folder_gain_homo_residual_mass_ckpt4000_full_8gpu_b160_160_160_workers0_20260615_180120

FolderGainHomo MMR: ViDoReV1 88.96 / ViDoReV2 58.76 / MMEB 75.45 / Avg 74.39
Run: experiments/exp_stagecompress/runs/folder_gain_homo_mmr_native_qwen25_lora_linear_gain_b160_160_160_bsz4_gc_20260614_124236
Eval: eval/folder_gain_homo_mmr_full_3sets

FolderGainHomo V6 residual_mass_mmr: ViDoReV1 88.27 / ViDoReV2 59.42 / MMEB 74.25 / Avg 73.98
Run: experiments/exp_stagecompress/runs/folder_gain_homo_residual_mass_mmr_native_qwen25_lora_linear_gain_b160_160_160_bsz4_gc_20260615_223238
Eval: eval/folder_gain_homo_residual_mass_mmr_ckpt3000_full_8gpu_b160_160_160_mmr025_workers0_20260616_195600
```

结论是，空间 coverage 可以作为 residual gain 的合理定义，但单独使用时不如 residual160。`residual_mass` 和 `mmr` 分别改善了 ViDoReV2 或 MMEB，但单独使用仍未超过 residual160。V6 `residual_mass_mmr` 将二者组合后 Avg=73.98，低于单独 `residual_mass`/`mmr`，说明动态预算与 MMR 去冗余存在目标冲突，论文中建议作为负向组合消融报告。

## 7. 当前正在运行的实验

截至 2026-06-16 20:51 CST，`FolderGainHomo V6 / residual_mass_mmr` 已完成 `checkpoint-3000` 训练与 8 卡 full eval。该实验保持原数据采样权重，步数为 3000。

| Run | 方法 | 状态 | 结果摘要 |
|---|---|---|---|
| `folder_global_homo_native_qwen25_lora_linear_global_b160_160_160_bsz4_gc_20260612_221751` | Global-Guided HomoFolder / V2 | DONE | 89.08 / 58.44 / 73.75 / Avg 73.76 |
| `folder_gain_homo_geo_coverage_b160_160_160_bsz4_gc_20260614_120847` | FolderGainHomo `geo_coverage` | DONE | 88.98 / 56.46 / 74.45 / Avg 73.30 |
| `folder_gain_homo_residual_mass_native_qwen25_lora_linear_gain_b160_160_160_bsz4_gc_20260614_120913` | FolderGainHomo `residual_mass` | DONE | 88.83 / 59.32 / 74.78 / Avg 74.31 |
| `folder_gain_homo_mmr_native_qwen25_lora_linear_gain_b160_160_160_bsz4_gc_20260614_124236` | FolderGainHomo `mmr` | DONE | 88.96 / 58.76 / 75.45 / Avg 74.39 |
| `folder_gain_homo_residual_mass_mmr_native_qwen25_lora_linear_gain_b160_160_160_bsz4_gc_20260615_223238` | FolderGainHomo `residual_mass_mmr` | DONE | 88.27 / 59.42 / 74.25 / Avg 73.98 |

当前 V6 训练和评测已完成，GPU 已空闲。

## 8. 推荐论文结构

### 8.1 Introduction

可按以下逻辑写：

1. Visual document retrieval 需要同时理解全局布局和局部细节。
2. 多粒度采样可以扩大视觉证据覆盖范围。
3. 但多粒度带来大量重复视觉 token，尤其是跨 `g1/g2/g3` 的同质化证据。
4. MaxSim late interaction 对冗余 token 和 token 长度不均衡较敏感。
5. 因此需要一种 query-free、MRL-compatible 的多粒度视觉 token 压缩机制。

### 8.2 Preliminary / Motivation

建议展示三组证据：

| 证据 | 说明 |
|---|---|
| 单粒度 vs Oracle | 多粒度确实存在互补信息 |
| MRL-main vs MMEB failure | MaxSim 对长 query / 多配多场景有边界 |
| 单图压缩横向比较 | merge-based 方法更适合 VDR |

### 8.3 Method

可以组织成四个模块：

```text
Hierarchical Sampling
Hierarchical Perception
Hierarchical Compression
Hierarchical MRL Supervision
```

其中 Hierarchical Compression 写成：

```text
G1 = Folder(g1)
R2 = ResidualFolder(g2 | G1)
R3 = ResidualFolder(g3 | G1, R2)
```

进一步解释：

```text
g1 只用内部 saliency 和 FOLDER merge；
g2/g3 联合内部 saliency 和跨粒度 novelty；
Global-Guided 版本额外用 G1 给 crop 分配 residual budget。
```

### 8.4 Experiments

建议实验表分成四块：

1. Single granularity and Oracle.
2. MRL-main and baselines.
3. Single-image compression comparison.
4. Homogeneity compression comparison.

图建议：

| 图 | 内容 |
|---|---|
| Fig.1 | 总体框架：层级采样 -> 层级感知 -> 层级压缩 -> MRL |
| Fig.2 | 单图压缩横向比较圆圈图 |
| Fig.3 | `G1/R2/R3` residual homogeneity 示意图 |
| Fig.4 | token 数量 vs Avg performance |
| Fig.5 | badcase / redundancy visualization |

## 9. 后续 TODO

### P0: 当前主线必须完成

- [DONE] `Global-Guided HomoFolder` 已训练到 `checkpoint-4000` 并完成 8 卡 full eval。
- [DONE] `FolderGainHomo geo_coverage` 已训练到 `checkpoint-4000` 并完成 8 卡 full eval。
- [DONE] 已补充 Global-Guided 与 geo_coverage 的 ViDoReV1 / ViDoReV2 / MMEB / Avg。
- [DONE] 当前主方法选择：保留 FolderHomo residual160。Global-Guided 与 geo_coverage 均未超过 residual160。
- [DONE] 补齐 `residual_mass` 和 `mmr` gain-based full eval。
- [DONE] V6 `residual_mass_mmr` 的 `checkpoint-3000` 已完成 3-set full eval：88.27 / 59.42 / 74.25 / Avg 73.98，低于 residual160、residual_mass、MMR。

### P0: 论文实验表需要补齐

- [TODO] 统一 MMEB metric 命名，是 `recall@1`、`precision@1` 还是内部 Avg。
- [TODO] 为所有关键方法补 token 数和压缩率。
- [TODO] 将 `MRL-main baseline` 的具体 run 路径、checkpoint、eval 路径补入表格。
- [TODO] 将 `g1/g2/g3/Oracle/MRL_text/MRL_notext` 的 run 路径补入表格。

### P1: 同质化增强实验

- [DONE] V6 `residual_mass_mmr` 未超过 residual160，也未超过单独 residual_mass/MMR；训练型同质化增强线建议收缩。
- [TODO] 若 V6 仍低于 residual160，收缩同质化增强矩阵，DART-Pivot / GlobalCom-DART Fusion 只作为附录或不再 formal。
- [TODO] 对最佳同质化模型做 eval-only query augmentation / MaxSim token mask 对比，优先验证 badcase 中发现的 prompt/special token 噪声。

### P1: 图和分析

- [TODO] 画单图压缩圆圈图：method family / Avg / token count。
- [TODO] 画同质化压缩图：token count vs performance。
- [TODO] 做 ViDoReV2 badcase：看压缩是否损失细粒度 OCR / layout。
- [TODO] 做 MMEB badcase：看 token 冗余是否影响 MaxSim。

### P2: token budget 研究

- [TODO] 在最佳模型上做少量 budget ablation：
  - `160/160/160`
  - `160/80/160`
  - `160/80/80`
  - `80/80/80` DONE：FolderHomo v1 3k，240 tokens，V1 88.44 / V2 56.10 / MMEB 74.53 / Avg 73.02
- [PARTIAL DONE] 已完成 `80/80/80` 下限测试；其余 budget 只在结构确定后继续，避免扩大实验矩阵。

### P2: Learnable token 第二路线

- [TODO] 设计 `Learnable Residual Tokens`：

```text
T1 = Resampler(Q1, g1)
T2 = ResidualResampler(Q2, g2 | T1)
T3 = ResidualResampler(Q3, g3 | T1+T2)
```

- [TODO] 先做低成本 smoke + 小评估。
- [TODO] 若 VDR 和 MMEB 均明显提升，再考虑作为当前论文增强版。
- [TODO] 若主要解决多配多/多配少 MaxSim 失效，则拆成第二篇论文。

### P2: 写作边界

- [TODO] 当前论文不要过度声称解决所有 MaxSim 非对称问题。
- [TODO] 当前论文主 claim 限定为：

```text
query-free cross-granularity homogeneity compression for multi-vector visual document retrieval
```

- [TODO] 可学习 token 路线作为 limitation / future work，除非后续实验强到足以合并。


## 10. 后续路线决策矩阵（2026-06-15）

后续不再无差别扩大 compressor 矩阵，而围绕两个变量做决策：

```text
同质化是否继续提升？
可学习 token 是否能显著提升 MMEB / 缓解 MaxSim 长度非对称？
```

| 情况 | 主判断 | 论文 / 实验路线 |
|---|---|---|
| 同质化好，可学习 token 好 | 同质化是通用原则 | 主推最强 HomoFolder；learnable residual tokens 作为第二实现或增强；测 budget + bidirectional MaxSim |
| 同质化好，可学习 token 不好 | 最稳预期 | 主论文聚焦 real-token residual homogeneity；learnable token 写作 ablation / limitation |
| 同质化不好，可学习 token 好 | 切换主线 | 转向 fixed-budget residual learnable tokens，HomoFolder 作为 real-token baseline |
| 同质化不好，可学习 token 不好 | 压缩不是主要瓶颈 | 转向 scoring / interaction：bidirectional MaxSim、length normalization、top-k MaxSim mean、weighted MaxSim |

可学习 token 若效果不好，并不必然是负面结果。它可以支撑以下解释：

```text
固定 token budget 有利于 MaxSim 长度控制，
但视觉文档检索依赖 OCR / layout / table 的真实局部证据，
纯 learned abstraction 可能难以替代 real visual tokens。
```

## 11. 下一阶段执行顺序（2026-06-15）

| 优先级 | 动作 | 说明 |
|---|---|---|
| P0 | 等三组同质化模型完成并统一 full eval | 先判断最强同质化结构 |
| P0 | 与 FolderHomo residual160 baseline 对比 | 决定是否替换主方法 |
| P1 | 只在最强 1-2 个同质化结构上做 budget ablation | 避免实验矩阵过宽 |
| P1 | 优先 budget：`160/80/80`, `120/60/60`；`80/80/80` 已完成 | 检查更强压缩是否提升 MMEB / Avg；当前 240-token 下限 Avg 73.02 |
| P1 | 做 bidirectional MaxSim eval-only | 低成本验证非对称交互瓶颈 |
| P2 | 设计 residual learnable tokens | 继承 `G1/R2/R3` 同质化思想，而不是单独 global resampler |

双向 MaxSim 建议公式：

```text
score_qd = MaxSim(query_tokens -> doc_tokens)
score_dq = MaxSim(doc_tokens -> query_tokens)
score = alpha * score_qd + (1 - alpha) * score_dq
```

先测：

```text
alpha = 0.5, 0.7, 0.3
```

优先对象：

```text
FolderHomo ckpt4000 level2 / 320 tokens
FolderHomo ckpt4000 level3 / 480 tokens
后续最强同质化模型的最佳 budget
```

## 12. 当前最稳结论（更新）

当前阶段最稳的结论是：

```text
1. 多粒度仍然有效，Oracle 证明不同尺度有互补信息。
2. MRL-main 可以建立强 baseline，但 MaxSim 在 MMEB / 多配多 / 多配少场景存在边界。
3. 单图压缩横向比较显示，merge-based FOLDER 比 pure pruning 更适合 VDR。
4. 以 FOLDER 为基础的 residual homogeneity compression 是目前最稳主线。
5. FolderHomo residual160 已经用 480 tokens 达到 75.35 Avg，说明压缩不仅省 token，还可能减少同质化噪声。
6. Prefix ablation 显示 320 tokens 是很好的效率点，但 480 tokens 仍然最好。
7. V6 residual_mass_mmr 已完成但未带来增益，后续应转向最强 residual160 的 query augmentation / MaxSim token mask / budget 对比。
```

## 2026-06-21 Stage-Interleaved Learnable Tokens P1 Run

Rationale: after the homogeneity variants underperformed residual160, the next formal 8-GPU slot moved to the learnable-token line. This tests whether stage-inserted fixed-length learnable tokens can address MaxSim token-count imbalance more directly than further query-free homogeneity objectives.

Run:

```text
Session: stage_p1_q2t2_8gpu
Run name: stage_interleaved_P1_Q2T2_8gpu_nommE5_textquery_focus_4k_orth0_20260621_012104
Run dir: experiments/exp_stagecompress/llmpre/learnable_tokens/runs/stage_interleaved_P1_Q2T2_8gpu_nommE5_textquery_focus_4k_orth0_20260621_012104
Method: Stage-Interleaved Learnable Tokens, P1/Q2-T2
Query inserted tokens: 2,4,8
Document inserted tokens: 8,16,32
MRL groups: 2,8,1.0;6,24,1.0;14,56,1.0
Orthogonality lambda: 0.0
Max steps: 4000
Save steps: 500
GPUs: 8
```

Early health check:

```text
Step 10 loss: 9.4184
mrl_q2_d8: 4.8328
mrl_q6_d24: 11.2982
mrl_q14_d56: 18.5091
GPU status: all 8 GPUs active at 100% utilization during startup check
```

Next actions after completion: evaluate checkpoint-4000 on the standard full 3-set config. If P1 is competitive, run P2/T3 capacity-up and P3/T1 capacity-down to build the token-count vs performance curve, then run P8 tail-placement control.
