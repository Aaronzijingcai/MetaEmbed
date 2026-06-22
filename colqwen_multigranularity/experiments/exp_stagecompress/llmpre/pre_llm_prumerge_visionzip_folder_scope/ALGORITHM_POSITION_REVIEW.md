# 四种算法迁移到 LLM 前的位置考察

更新时间：2026-06-19

本文只讨论 PruMerge、VisionZip、FOLDER、SCOPE 在当前 MURE-V2 / ColQwen2.5 多粒度检索代码中的合适插入位置。更完整的公共接口和训练计划见 `TECHNICAL_DESIGN.md`，`adapter_pre` 的精确定义见 `ADAPTER_PRE_POSITION.md`。

## 位置候选

| 代号 | 名称 | 本仓库位置 | 适用性 |
|---|---|---|---|
| P1 | `vision_late` | Qwen2.5-VL visual backbone 内部末层或输出前 | 最贴近部分视觉压缩论文，但工程耦合高。 |
| P2 | `adapter_pre` | `base_model.visual(...)` 输出写入 `inputs_embeds` 后，`language_model.layers[0]` 前 | 本次推荐主位置。真正减少 LLM token 数，且已有 `visionzip_mrl` 可复用。 |
| P3 | `llm_early` | 先跑 LLM 前 K 层，再压缩视觉 token | 可做对照，但不严格属于 LLM 层前。 |
| P4 | `mlp_post` | `custom_text_proj` 后 | 已有归档实现；不减少 LLM 计算，不满足本次目标。 |

## 总体推荐

| 算法 | 推荐第一位置 | 第二位置 | 不建议作为本次主位置 | 判断 |
|---|---|---|---|---|
| VisionZip | P2 `adapter_pre` | P3 `llm_early` | P4 `mlp_post` | 最自然的 LLM 前压缩算法，优先实现。 |
| PruMerge | P2 `adapter_pre` | P1 `vision_late` / P3 `llm_early` | P4 `mlp_post` | 可迁移为 PruMerge-style，注意 Qwen2.5-VL 没有直接等价 CLIP CLS attention。 |
| FOLDER | P2 `adapter_pre` 第一版 | P1 `vision_late` 忠实版 | P4 `mlp_post` 仅作已有参考 | merge-preserving 对文档检索友好；若追求视觉 encoder 加速，再做 P1。 |
| SCOPE | P2 `adapter_pre` | P3 `llm_early` | P4 `mlp_post` | 纯 pruning baseline，适合位置清晰但质量风险最高。 |

## VisionZip

推荐位置：P2 `adapter_pre`。

最新参考：CVPR 2025 版本；官方仓库已发布 Qwen2.5VL 相关代码。VisionZip 的核心目标就是在视觉 token 进入 LLM 前保留 dominant tokens，并把部分 residual 信息合并到 contextual tokens 中，因此和 `adapter_pre` 匹配度最高。

在本仓库中的落点：

```text
_build_inputs_embeds()
-> image_embeds 已经替换 image_token 占位
-> per-crop VisionZip 压缩
-> _run_language_layers(start_layer=0, end_layer=num_layers)
```

建议实现判断：

1. 第一版直接复用 `llmpre/visionzip_mrl/modeling_visionzip_mrl.py` 的 `adapter_pre` 逻辑。
2. 训练阶段使用 soft mask / soft merge，保持原始序列长度。
3. 评估阶段使用 hard prune / hard merge，报告进入 LLM 的实际视觉 token 数。
4. per-crop 处理，不跨 crop merge，避免局部 OCR/table 信息被别的 crop 吞掉。

风险：

1. uniform contextual tokens 对视觉文档 OCR 的保护不一定充分。
2. g3 强压缩时，表格、小字、页脚日期等细粒度证据容易丢。

优先级：P0，首个实现。

## PruMerge

推荐位置：P2 `adapter_pre`。

最新参考：ICCV 2025 LLaVA-PruMerge，讨论新版时包含 PruMerge+。原方法依赖 CLIP visual encoder 的 CLS-to-patch attention 和 key similarity；当前 Qwen2.5-VL / ColQwen2.5 路径不保证能低成本拿到完全等价的 CLS attention，因此第一版应写作 PruMerge-style。

在本仓库中的落点：

```text
per-crop adapter_pre image tokens
-> learned saliency 或可提取 attention score
-> top-k important anchors
-> PruMerge+ style spatial supplement
-> pruned tokens 按相似度合并回 anchors
-> 短视觉序列进入 LLM
```

建议实现判断：

1. 第一版用 trainable scorer 代替 CLIP CLS attention，保证和现有 MRL 训练体系兼容。
2. 合并算子可以从 `mlppost/strategies/common.py::_forward_prumerge_impl()` 迁移，但输出不要在 LLM 前强制归一化到 retrieval embedding 风格。
3. 必须加 spatial quota / uniform supplement，接近 PruMerge+ 的防丢布局思想。
4. hard eval 时重建 `attention_mask` 和 `position_ids`，保留被合并 anchor 的 position。

风险：

1. 若没有原始 CLS attention，不能声称严格复现 PruMerge。
2. top-k saliency 容易偏向标题、大图、粗线框，漏掉小字号 OCR。

优先级：P1，VisionZip 之后实现。

## FOLDER

推荐位置：第一版 P2 `adapter_pre`，第二版 P1 `vision_late`。

最新参考：ICCV 2025 FOLDER。原论文更强调在 visual backbone 中集成 fold merge，能减少视觉侧后续 block 和 LLM 的 token 成本。当前本仓库如果先做 `adapter_pre`，只能减少 LLM 侧成本，但工程上最稳。

P2 第一版落点：

```text
per-crop adapter_pre image tokens
-> bipartite fold matching
-> redundant source tokens merge into destination tokens
-> 重复直到目标 token 数
-> merged visual tokens 进入完整 LLM
```

P1 第二版落点：

```text
Qwen2.5-VL visual backbone late block
-> fold merge
-> 更新后续 visual blocks 的 token sequence / position 相关信息
-> 输出更短 image_embeds
-> LLM
```

建议实现判断：

1. 若目标是先得到四算法 LLM 前对照，做 P2。
2. 若目标是复现 FOLDER 的论文加速边界，做 P1。
3. P2 可以迁移 `mlppost/strategies/common.py::_forward_folder_impl()` 的 fold merge 思想，但要保留 LLM hidden-state 尺度，不做 retrieval embedding normalization。
4. 对 visual document retrieval，FOLDER 是四者中最适合作为主 baseline 的 merge primitive，因为它不直接删除所有冗余信息。

风险：

1. P2 不能节省 visual encoder 成本。
2. P1 要处理 Qwen visual backbone 内部的 RoPE / spatial merge / grid 对齐，调试成本高。
3. merged token 的 position 继承策略会影响 LLM 解释局部空间信息。

优先级：P2。如果以检索质量为主，FOLDER-style adapter_pre 很值得做；如果以系统加速叙事为主，再推进 vision_late。

## SCOPE

推荐位置：P2 `adapter_pre`。

最新参考：NeurIPS 2025 / OpenReview 版本。SCOPE 的核心是 saliency + coverage 的视觉 token 选择，它不依赖生成新的 merged token，适合作为纯 pruning baseline。

在本仓库中的落点：

```text
per-crop adapter_pre image tokens
-> saliency score
-> token-token relationship / coverage gain
-> greedy select saliency * coverage gain
-> selected tokens 进入 LLM
```

建议实现判断：

1. 第一版保持纯 selection，不做 merge，作为 pruning baseline。
2. per-crop 做 coverage，避免 g1/g2/g3 或不同局部 crop 之间相互覆盖。
3. 加 minimum quota、spatial quota、duplicate quota ablation，避免对 OCR 重复锚点过度去重。
4. O(N^2) token relationship 在 g3 上成本较高，必要时先降采样或分块。

风险：

1. 纯 pruning 没有信息回收，强压缩时比 FOLDER / PruMerge / VisionZip 更容易掉质量。
2. coverage 目标可能把文档中重复但有检索价值的文本行、表格列、页眉页脚当作冗余。

优先级：P3，作为对照实验。

## 推荐实现顺序

| 顺序 | 内容 | 原因 |
|---:|---|---|
| 1 | VisionZip adapter_pre | 已有 `visionzip_mrl`，最容易整理成可运行 baseline。 |
| 2 | PruMerge-style adapter_pre | 与 VisionZip 共用 scorer、sequence packing、merge 回收。 |
| 3 | FOLDER-style adapter_pre | 最符合当前检索质量经验，但 merged-token position 需要单独验证。 |
| 4 | SCOPE adapter_pre | 纯 pruning 对照，质量风险高但实现清晰。 |
| 5 | FOLDER vision_late | 需要改 visual backbone，等 P2 结果有价值后再做。 |

## 统一报告口径

四个算法都必须报告：

1. `adapter_pre` 进入 LLM 前的视觉 token 数。
2. LLM forward / prefill 时间或 tokens/s。
3. `custom_text_proj` 后 compact doc token 数。
4. ViDoReV1、ViDoReV2、MMEB 三组检索指标。
5. soft train 与 hard eval 的差距。

这样才能和旧 `mlppost` 结果区分开：旧结果主要说明 index / MaxSim 压缩，本目录目标是证明 LLM 前 token 计算也被压缩。
