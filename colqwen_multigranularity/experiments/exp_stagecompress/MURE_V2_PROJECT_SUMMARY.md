# MURE-V2 项目总纲与阶段总结

更新时间：2026-06-15

本文档用于整理 `exp_stagecompress` 阶段的核心逻辑、实验结论、方法筛选和后续推进计划。它不是单个实验 README，而是论文导向的项目总纲。

## 1. 项目定位

MURE-V1 已经验证了多分辨率视觉编码在视觉文档检索中的有效性：不同粒度可以分别捕获全局布局、局部 OCR、表格结构和细节区域，从而提升 VDR 表征能力。

MURE-V2 的核心问题不再是“多粒度是否有效”，而是：

```text
多粒度视觉 token 如何更好融合、压缩和索引？
```

MURE-V2 的升级可以分成三层：

| 层面 | 内容 | 论文角色 |
|---|---|---|
| 基座升级 | Qwen2VL / Qwen2.5VL / Qwen3VL 采样与 ColQwen 检索适配 | 工程基础，不作为主要创新点 |
| 任务升级 | 从 VDR 扩展到 VDR + 图文配对 / MMEB 风格任务 | 暴露 MaxSim 和多向量检索的新失效边界 |
| 方法升级 | 多粒度 token 的融合与压缩 | 主要 novelty |

当前最适合的论文叙事是：

```text
MURE-V2 研究多粒度视觉文档检索中由跨尺度重复观测带来的 token 同质化冗余，并提出可训练的 residual MRL 压缩机制，在保留多向量 MaxSim 框架的同时降低 token 数量并稳定检索效果。
```

## 2. 当前主链路

当前链路为：

```text
image
-> 多粒度采样：1x1, 1x2/2x1, 2x2
-> Qwen2.5VL / ColQwen2.5
-> LLM 上下文建模
-> MLP / projection 得到检索 token
-> MaxSim late interaction
```

三组视觉粒度记为：

| 记号 | 视图 | 作用 |
|---|---|---|
| `g1` | `1x1` 全局图 | 全局布局、页面整体语义 |
| `g2` | `1x2` 或 `2x1`，按长宽比自适应 | 中尺度结构 |
| `g3` | `2x2` crop | 局部 OCR 和细节证据 |

原始 MRL 表征：

```text
Level 1: g1
Level 2: g1 + g2
Level 3: g1 + g2 + g3
```

同质化 residual 压缩后的目标表征：

```text
Level 1: G1
Level 2: G1 + R2
Level 3: G1 + R2 + R3
```

其中：

| 记号 | 含义 |
|---|---|
| `G1` | 压缩后的全局基础视觉证据 |
| `R2` | 相对 `G1` 的中尺度残差信息 |
| `R3` | 相对 `G1 + R2` 的细尺度残差信息 |

## 3. Preliminary Study 结论

### 3.1 多粒度仍然有潜力

早期实验说明，单粒度并不能覆盖全部信息，Oracle 上限明显更高。

| 方法 | ViDoReV1 nDCG@5 | ViDoReV2 nDCG@5 | MMEB | Step |
|---|---:|---:|---:|---:|
| `g1` `1x1` | 86.7 | 56.9 | 61.6 | 16k |
| `g2` `1x2/2x1` | 85.5 | 53.8 | 53.7 | 16k |
| `g3` `2x2` | 86.0 | 59.4 | 51.1 | 16k |
| Oracle | 91.4 | 70.2 | 71.8 | 16k |
| MRL no text | 87.6 | 58.4 | 35.6 | 16k |
| MRL with text | 82.4 | 50.0 | 37.1 | 16k |
| MURE-V1 | 87.0 | 59.5 | - | - |
| ColPali | 84.9 | 54.5 | 34.9 | - |
| MetaEmbed | 88.7 | 60.3 | 69.1 | - |

关键结论：

1. Oracle 高于单粒度，说明多粒度之间确实有互补信息。
2. 简单加入 text token 没有收益，甚至会破坏视觉 token 表征。
3. ViDoRe 上 MRL 较稳定，但 MMEB 上基础 MRL 明显下降，说明多向量 MaxSim 在图文配对任务上有失效边界。

### 3.2 MaxSim 失效边界

MMEB 效果差不是单纯代码问题。类似现象也可以在 ColPali / VLM2Vec 风格多向量设置中观察到。

当前判断：

```text
MaxSim 是非对称、query-dominated 的交互机制。
它适合 query 短、target 长的检索场景。
当 query 端较长、target 端较短，或两端长度不均衡时，MaxSim 容易放大噪声。
```

MetaEmbed 效果相对稳定的一个重要原因是：它用固定数量可学习 token 统一 query / target 表征长度。

因此存在两条路线：

| 路线 | 内容 | 当前选择 |
|---|---|---|
| 改 MaxSim | 重新设计长 query / 短 target 下的交互机制 | 暂不作为主线，风险更大 |
| 保持 MaxSim，压缩 token | 控制 query / target token 数量，减少冗余和噪声 | 当前主线 |

## 4. 压缩问题拆解

MURE-V2 的压缩问题不同于普通 LVLM token compression。

普通 LVLM 压缩一般是：

```text
one image -> one visual token sequence -> generation / VQA acceleration
```

MURE-V2 是：

```text
one document -> multiple granularity/crop views -> multi-vector retrieval index
```

因此有两类冗余：

| 冗余类型 | 含义 | 当前状态 |
|---|---|---|
| 单图冗余 | 每个粒度内部 token 重复 | 已探索 SoftAssign / PruMerge / VisionZip / FOLDER / SCOPE / Stage Resampler，FOLDER 最强 |
| 多图同质化 | `g1/g2/g3` 跨粒度重复表达相同证据 | 当前主线，使用 `G1/R2/R3` residual MRL |

从是否可训练看：

| 类型 | 含义 | 当前判断 |
|---|---|---|
| 免训练压缩 | 在已训练模型上推理时直接剪枝/合并 | 可做 baseline，但无法让表示空间适配压缩，已暂停 |
| 可训练压缩 | 训练时联合优化 LoRA / projection / compressor | 当前主线 |

从压缩位置看：

```text
Pic -> Vision Encoder -> Adapter -> LLM -> MLP/projection -> retrieval tokens
```

| 位置 | 优点 | 风险 | 当前角色 |
|---|---|---|---|
| Vision Encoder 深层 | 可减少更早的视觉 token 成本 | 对 Qwen2.5VL 侵入大 | 参考路线 |
| LLM 前 / Adapter 后 | 可减少 LLM 输入长度 | 训练更难，语义未对齐 retrieval | 历史探索 |
| LLM 浅层 | 适合 generation acceleration | 破坏 MRL prefix 风险大 | 暂不主推 |
| MLP 后 | 最稳定，直接对齐检索目标 | 不减少 LLM 计算 | 当前成功路线 |

## 5. 主要实验结果

下表是当前阶段的清理版结果。MMEB 的具体 metric 命名在论文中需要统一，目前笔记中存在 `precision@1` / `recall@1` / `MMEB Avg` 混用。

| 方法 | 类型 | 压缩位置 | Token | ViDoReV1 | ViDoReV2 | MMEB | Avg | 状态 |
|---|---|---|---:|---:|---:|---:|---:|---|
| MRL-main baseline | 无压缩 | - | full | 89.8 | 61.0 | 75.8 | 75.5 | 强 baseline |
| MetaEmbed baseline | 无压缩 / learned tokens | LLM 前 | fixed | 83.1 | 52.9 | 73.6 | 69.9 | 参考 |
| SoftAssign | 合并 | MLP 后 | - | 81.2 | 47.4 | 72.1 | 66.9 | 弱 |
| PruMerge / strategy3 | 剪枝+合并 | MLP 后 | - | 87.9 | 58.3 | 75.3 | 73.8 | 已完成，历史强参考 |
| VisionZip / strategy4 | 剪枝+合并 | MLP 后 | - | 87.7 | 57.8 | 73.3 | 72.9 | 已完成 |
| FOLDER / strategy5 | 合并 | MLP 后 | 1120 | 89.6 | 58.8 | 75.1 | 74.5 | 旧策略最强锚点 |
| SCOPE / strategy6 | 剪枝 | MLP 后 | - | 88.6 | 57.2 | 75.1 | 73.6 | 已完成 |
| Stage Resampler / strategy7 | 可学习 token | MLP 后 | - | 80.8 | 45.0 | 70.1 | 65.3 | 弱 |
| Learnable Global MRL Tokens | 可学习 token | LLM 前 | - | 78.0 | 46.1 | 70.6 | 64.9 | 弱 |
| Learnable Global MRL Tokens + TwigStage | 可学习 token + 剪枝 | LLM 前 | - | 81.6 | 49.6 | 69.1 | 66.8 | 弱 |
| SoftStageMRL | 剪枝 | LLM 前 | - | 74.3 | 43.4 | 68.1 | 61.9 | 弱 |
| VisionSelectorMRL | 软训练硬推理剪枝 | LLM 前 | - | 16.9 | 14.6 | 2.8 | 11.4 | 失败，暂停 |
| FolderHomo v1 | 可训练同质化 | MLP 后 | 1120 | 89.27 | 59.20 | 75.88 | 74.78 | 第一版同质化成功 |
| FolderHomo residual160 | 可训练同质化 | MLP 后 | 480 | 89.34 | 60.28 | 76.43 | 75.35 | 当前最强完成结果 |
| FolderHomo v1 80/80/80 | 可训练同质化强压缩 | MLP 后 | 240 | 88.44 | 56.10 | 74.53 | 73.02 | DONE，240-token budget ablation，质量下降明显但未崩 |
| FolderHomo residual160 prefix L1 | 可训练同质化 prefix eval | MLP 后 | 160 | 88.36 | 59.09 | 74.93 | 74.12 | ckpt4000 prefix ablation |
| FolderHomo residual160 prefix L2 | 可训练同质化 prefix eval | MLP 后 | 320 | 89.20 | 59.73 | 75.75 | 74.89 | ckpt4000 prefix ablation |
| Global-Guided HomoFolder / V2 | 可训练同质化 | MLP 后 | 480 | 89.08 | 58.44 | 73.75 | 73.76 | DONE，未超过 residual160 |
| FolderGainHomo geo_coverage | 可训练增益同质化 | MLP 后 | 480 | 88.98 | 56.46 | 74.45 | 73.30 | DONE，coverage 增益 ablation，ViDoReV2 下降明显 |
| FolderGainHomo V5 / MMR | 可训练同质化增益 | MLP 后 | 480 | 88.96 | 58.76 | 75.45 | 74.39 | V5 已完成，MMR gain ablation |

最重要的阶段结论：

```text
FolderHomo residual160 将 token 从 1120 降到 480，效果反而从 74.78 提升到 75.35。
```

这说明当前任务中“更多 token 不一定更好”。跨粒度重复 token 可能会增加 MaxSim 噪声，合理 residual 压缩反而能提升效果。

2026-06-14 补充的 prefix ablation 进一步说明，`G1/R2/R3` 的边际收益是递减但仍然单调的：

| Prefix | Visual tokens | ViDoReV1 | ViDoReV2 | MMEB | Avg | 说明 |
|---|---:|---:|---:|---:|---:|---|
| `G1` | 160 | 88.36 | 59.09 | 74.93 | 74.12 | 只用全局基础证据 |
| `G1 + R2` | 320 | 89.20 | 59.73 | 75.75 | 74.89 | 加入中尺度 residual，主要收益段 |
| `G1 + R2 + R3` | 480 | 89.34 | 60.28 | 76.43 | 75.35 | 完整 residual160 表征 |

关键观察：

```text
160 -> 320 的 Avg 提升约 +0.77，320 -> 480 的 Avg 提升约 +0.46。
320 tokens 已经接近 480 tokens，但 480 仍然最好。
```

2026-06-15 补充的 gain-based homogeneity full eval：

| 方法 | Gain mode | Run | Token | ViDoReV1 | ViDoReV2 | MMEB | Avg | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---|
| FolderGainHomo geo_coverage | `geo_coverage` | `folder_gain_homo_geo_coverage_b160_160_160_bsz4_gc_20260614_120847` | 480 | 88.98 | 56.46 | 74.45 | 73.30 | 空间覆盖增益没有超过 residual160；主要短板在 ViDoReV2 |

这说明只把“增益”定义为空间覆盖 / local coverage 还不够。它在 ViDoReV1 和 MMEB 上仍保持可用，但会牺牲 ViDoReV2 中更细粒度、跨语言或专业文档的匹配稳定性。论文中可以把它作为 gain definition ablation，而不是主方法。

因此，“更少 token 自动解决 MaxSim”不是当前数据能直接支持的结论；更准确的判断是：

```text
只有当 token 同时更少且更干净时，压缩才可能缓解 MaxSim 的非对称噪声。
```

## 6. 方法筛选结论

### 6.1 为什么 FOLDER 是当前锚点

FOLDER 的优势是保留真实视觉 token，并通过 merge 回收被压缩 token 的信息。文档检索对 OCR、表格、局部 layout 很敏感，纯剪枝容易删掉关键证据，而 merge 更保守。

因此后续方法不应该完全抛弃 FOLDER，而应该在 FOLDER 的基础上解决：

```text
哪些 token 是跨粒度重复信息？
哪些 token 是相对粗粒度的新信息？
```

### 6.2 为什么 learnable token 不是当前主线

Learnable token 理论上能解决固定长度表征问题，但当前结果较弱。

可能原因：

1. 可学习 token 需要从头学习 OCR / layout / 局部证据吸收，训练难度高。
2. MaxSim 更依赖真实局部 token，合成 token 可能损失细粒度可匹配性。
3. 只训练 compressor 或 token 模块不够，成功实验需要 LLM LoRA、projection、compressor 联合训练。

因此 learnable token 目前作为副线保留，不是主论文方法。

### 6.3 为什么可训练同质化是主线

成功配方是：

```text
native Qwen2.5VL / ColQwen2.5 base
+ LLM LoRA
+ custom_text_proj / retrieval projection
+ trainable FOLDER-style homogeneity compressor
+ MRL prefix supervision
```

只训练压缩模块会导致 loss 降不下来。LLM LoRA 必须参与训练，让语言侧上下文空间适配压缩后的视觉证据。

## 7. 当前主线

`exp_stagecompress` 当前收缩为两条主线：

| 主线 | 目录 | 状态 | 决策 |
|---|---|---|---|
| Learnable tokens | `mainlines/learnable_tokens/` | 副线 / 暂缓 | 保留参考，不投入大算力 |
| Homogeneity | `mainlines/homogeneity/` | 主线 | 当前论文路线 |

同质化主线的方法优先级：

| 优先级 | 方法 | 目录 | 状态 | 目的 |
|---|---|---|---|---|
| P0 | Residual HomoFolder | `folder_homo/` | DONE | 建立 `160/160/160` 480-token residual MRL baseline |
| P0 | Global-Guided Residual HomoFolder | `folder_global_homo/` | DONE / eval done | 用全局视图决定局部 crop residual budget；结果未超过 residual160 |
| P1 | Gain-based GeoCoverage HomoFolder | `folder_gain_homo/` | DONE / eval done | 用空间 coverage 定义 residual gain；结果未超过 residual160，作为 ablation |
| P1 | DART-Pivot Residual HomoFolder | `folder_dart_pivot/` | smoke passed, formal TODO | 用 coarse visual pivots 判断 token duplication |
| P1 | GlobalCom-DART Fusion | `folder_global_dart_homo/` | smoke passed, formal TODO | 同时建模 crop-level 和 token-level 同质化 |
| P2 | Redundancy regularization | best P0/P1 | TODO | 显式惩罚 residual token 重复 coarse evidence |
| P2 | Token budget ablation | Residual HomoFolder / best P0/P1 | PARTIAL DONE | `80/80/80` 已完成；后续探索 `160/80/160`, `160/80/80`, `120/60/60` |

## 8. 最近 8 卡训练 / 评测状态

截至 2026-06-16，已完成两组 480-token 同质化增强 full eval，并补充 FolderHomo 240-token 强压缩 ablation：

| Run | 方法 | ViDoReV1 | ViDoReV2 | MMEB | Avg | 状态 |
|---|---|---:|---:|---:|---:|---|
| `folder_global_homo_native_qwen25_lora_linear_global_b160_160_160_bsz4_gc_20260612_221751` | Global-Guided HomoFolder / V2 | 89.08 | 58.44 | 73.75 | 73.76 | 训练完成，8 卡评测完成 |
| `folder_gain_homo_geo_coverage_b160_160_160_bsz4_gc_20260614_120847` | FolderGainHomo `geo_coverage` | 88.98 | 56.46 | 74.45 | 73.30 | 训练完成，8 卡评测完成 |
| `folder_homo_v1_b80_80_80_native_qwen25_lora_linear_folder_bsz4_gc_3k_20260615_231152` | FolderHomo v1 `80/80/80` | 88.44 | 56.10 | 74.53 | 73.02 | 3k 训练完成，8 卡评测完成 |

评测口径：ViDoReV1 / ViDoReV2 使用 `avg_ndcg_at_5`，MMEB 使用 `avg_recall_at_1`；`geo_coverage` 评测使用 `BATCH_QUERY=32, BATCH_PASSAGE=32, BATCH_SCORE=128, NUM_WORKERS=0`。`80/80/80` full eval 使用 `BATCH_QUERY=4, BATCH_PASSAGE=4, BATCH_SCORE=16, NUM_WORKERS=0`。

当前判断：这些补充实验都没有超过 FolderHomo residual160 的 75.35 Avg，因此主方法暂不替换。`geo_coverage` 更适合作为“增益定义方式”的负向或边界 ablation：coverage-based residual selection 可行，但对 ViDoReV2 不够稳。`80/80/80` 将 visual token 压到 240 后 Avg 降到 73.02，主要问题同样是 ViDoReV2，下限可接受但不作为主结果。

## 9. strategy3 状态

`strategy3_prumerge` 已经是完成的 MLP-post 历史参考，不是当前训练任务。

| 方法 | 类型 | 位置 | ViDoReV1 | ViDoReV2 | MMEB | Avg | 状态 |
|---|---|---|---:|---:|---:|---:|---|
| `strategy3_prumerge` | 剪枝+合并 | MLP 后 | 87.9 | 58.3 | 75.3 | 73.8 | completed / archived |

它证明剪枝+合并是可行方向，但当前更强的主线是 FOLDER -> FolderHomo -> Global-Guided HomoFolder。

## 10. 论文故事建议

建议按下面逻辑写：

1. 多粒度视觉文档检索有效，但会引入跨粒度重复观测。
2. MaxSim 在 query / target token 长度不均衡时存在失效边界。
3. 现有 LVLM token compression 多面向单图、生成/VQA 加速，很少研究 query-free 的多粒度检索索引压缩。
4. MURE-V2 将问题定义为 cross-granularity homogeneity compression。
5. 我们不直接替换 MaxSim，而是在保持 MRL + MaxSim 框架的前提下压缩冗余视觉证据。
6. 方法上使用 `G1/R2/R3` residual MRL，保留真实视觉 token，并通过 FOLDER-style merge 减少信息损失。
7. 实验上 480-token 的 FolderHomo residual160 已经接近或超过无压缩/大 token 版本，说明“去除同质化冗余”本身可以提升检索稳定性。

可以使用的核心 claim：

```text
MURE-V2 identifies cross-granularity homogeneity as a key redundancy source in multi-vector visual document retrieval and introduces trainable residual MRL compression to preserve real visual evidence while reducing token count and maintaining retrieval quality.
```

中文版本：

```text
MURE-V2 发现多粒度视觉文档检索中存在跨粒度同质化冗余，并提出可训练的 residual MRL 压缩机制，在保留真实视觉证据和 MaxSim 多向量检索框架的同时，降低 token 数量并提升检索稳定性。
```

## 11. 风险与待补充

| 风险 | 原因 | 动作 |
|---|---|---|
| MMEB metric 命名混乱 | 笔记中有 precision@1 / recall@1 / Avg 混用 | 论文表格前统一口径 |
| ViDoReV2 敏感 | 某些方法会提升 MMEB 但损害 ViDoReV2 | 三个 benchmark 必须分开报 |
| token budget 混杂因素 | 480 token 提升可能部分来自预算正则化 | 模型确定后做少量 budget ablation |
| MLP-post 不减少 LLM 计算 | 当前主要减少 index / MaxSim 成本 | 论文中明确贡献边界 |
| novelty 需要谨慎 | 可能已有多图压缩相关工作 | claim 限定为 query-free cross-granularity multi-vector VDR |
| learnable token 负结果解释 | reviewer 可能质疑训练不足 | 作为对照和动机，不作为攻击重点 |

## 12. 近期推进计划

| 优先级 | 动作 | 目的 |
|---|---|---|
| P0 | 等当前 Global-Guided HomoFolder 跑到 4000 | 得到正式 checkpoint |
| P0 | 对 checkpoint-4000 做 8 卡全量评估 | 和 FolderHomo residual160 对比 |
| P0 | 如果效果提升，将 Global-Guided 写成 crop-level homogeneity compression | 形成更强故事 |
| P1 | 如果 Global-Guided 不提升，保留 residual160 作为主方法并跑 DART-Pivot | 测 token-level duplication awareness |
| P1 | 只在必要时跑 Fusion | 控制算力，不做过宽 sweep |
| P2 | 模型确定后再做 token budget 研究 | 构建质量/成本 frontier |
| P2 | 做 ViDoReV2 / MMEB badcase | 解释 residual compression 帮助和失效位置 |

## 13. 当前推荐结论

当前阶段最稳的结论是：

```text
可训练的多粒度 residual 压缩是有效方向。
FOLDER-style merge 是目前最稳的基础算子。
LLM LoRA + projection + compressor 必须联合训练。
480-token residual MRL 已经能达到甚至超过 1120-token 版本。
后续应优先验证 Global-Guided / DART-Pivot 是否能进一步改善跨粒度同质化压缩。
```

不要过度声称已经完全解决 MaxSim 非对称问题。当前更准确的说法是：

```text
通过控制多粒度 token 冗余和长度，MURE-V2 可以缓解 MaxSim 在多向量检索中的噪声放大问题，并提高多粒度视觉表征的效率和稳定性。
```


## 14. 后续研究决策矩阵（2026-06-15）

当前后续路线不应继续无差别扩大实验矩阵，而应围绕两个核心变量决策：

```text
同质化路线是否继续变强？
可学习 token 路线是否能显著改善 MMEB / MaxSim 长度不均衡？
```

这里“同质化好”定义为：在 480 或更低 token 下，Avg / MMEB 超过 FolderHomo residual160，且 ViDoReV2 不明显下降。  
这里“可学习 token 好”定义为：在固定较少 token 下，MMEB 明显提升，且 ViDoReV1 / ViDoReV2 不崩。

| 情况 | 判断 | 主线选择 | 后续动作 |
|---|---|---|---|
| 同质化好，可学习 token 好 | 最理想 | 同质化作为原则，real-token merge 与 learned-token resampling 双实现 | 主推最强同质化；learnable residual tokens 作为增强版；测 budget + 双向 MaxSim |
| 同质化好，可学习 token 不好 | 当前最稳预期 | 主论文聚焦 real-token residual homogeneity compression | 把 learnable token 写作 ablation / limitation；在最强同质化上做强压缩 budget 与双向 MaxSim |
| 同质化不好，可学习 token 好 | 需要切主线 | 转向 fixed-budget residual learnable tokens | HomoFolder 作为 real-token baseline；重点解释固定 token budget 对 MMEB / MaxSim 的价值 |
| 同质化不好，可学习 token 不好 | 压缩不是主要瓶颈 | 转向 interaction / scoring 机制 | 做 bidirectional MaxSim、length-normalized MaxSim、top-k MaxSim mean、weighted MaxSim；MMEB 作为边界分析 |

当前主观判断：最可能、也最稳的是第二种情况，即同质化继续有效，而可学习 token 暂时较弱。这个结果并不坏，反而可以支撑：

```text
视觉文档检索依赖真实 OCR / layout / table token 的局部可匹配性，
纯 learned abstraction 难以替代真实视觉证据。
```

## 15. 下一阶段执行顺序（2026-06-15）

`geo_coverage` 已完成 full eval，后续应继续补齐剩余增益同质化结构，并在确定最强结构后再做预算缩放。推荐顺序：

| Step | 动作 | 目的 |
|---|---|---|
| 1 | 补齐剩余增益同质化结构的 `160/160/160` full eval | 先确定哪个同质化结构最强；`geo_coverage` 已完成 |
| 2 | 与 FolderHomo residual160 baseline 对比 | 判断是否替换主方法 |
| 3 | 只选最强 1-2 个做 budget ablation | 避免在弱结构上浪费算力 |
| 4 | 强压缩 budget：`160/80/80`, `120/60/60`；`80/80/80` 已完成 | 检查更少、更干净 token 是否提升 MMEB / Avg；当前 240-token 下限 Avg 73.02 |
| 5 | 在最佳 budget 上做 bidirectional MaxSim eval-only | 低成本验证 MaxSim 非对称是否是 MMEB 瓶颈 |
| 6 | 再决定是否投入 learnable residual tokens | 让 learnable token 继承同质化原则，而不是单独盲训 |

双向 MaxSim 的低成本验证形式：

```text
score_qd = MaxSim(query_tokens -> doc_tokens)
score_dq = MaxSim(doc_tokens -> query_tokens)
score = alpha * score_qd + (1 - alpha) * score_dq
```

建议先测：

```text
alpha = 0.5, 0.7, 0.3
```

优先评估对象：

```text
FolderHomo ckpt4000 level2 / 320 tokens
FolderHomo ckpt4000 level3 / 480 tokens
后续最强同质化模型的最佳 budget
```

写作边界保持谨慎：当前论文不宣称已经彻底解决 MaxSim 非对称问题，而是声明通过同质化压缩减少冗余 token 和噪声放大；若 bidirectional MaxSim 或 learnable residual tokens 后续显著提升 MMEB，再作为扩展机制写入。
