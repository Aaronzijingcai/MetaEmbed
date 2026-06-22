# PruMerge / VisionZip / FOLDER / SCOPE 迁移到 LLM 层前的技术设计

更新时间：2026-06-19

目标：把目前 `mlppost/strategies/` 中作为归档参考的 PruMerge、VisionZip、FOLDER、SCOPE 思路，迁移为真正位于 LLM decoder 前的视觉 token 压缩方案。新的方案应减少进入 Qwen2.5-VL language model 的视觉 token 数，从而降低 LLM prefill / attention 成本；不能只压缩 `custom_text_proj` 之后的检索 embedding。

## 参考来源

| 方法 | 最新采用版本 | 关键来源 | 对本方案的启发 |
|---|---|---|---|
| PruMerge | LLaVA-PruMerge, ICCV 2025；需要讨论新版时包括 PruMerge+ | [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Shang_LLaVA-PruMerge_Adaptive_Token_Reduction_for_Efficient_Large_Multimodal_Models_ICCV_2025_paper.html), [arXiv](https://arxiv.org/abs/2403.15388), [official repo](https://github.com/42Shawn/LLaVA-PruMerge), [project page](https://llava-prumerge.github.io/) | 先选重要视觉 token，再把被剪 token 合并回保留 token；PruMerge+ 的空间补充思想适合防止文档布局丢失。 |
| VisionZip | VisionZip, CVPR 2025；Qwen2.5VL 相关实现优先参考官方仓库 | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_VisionZip_Longer_is_Better_but_Not_Necessary_in_Vision_Language_CVPR_2025_paper.html), [arXiv](https://arxiv.org/abs/2412.04467), [official repo](https://github.com/dvlab-research/VisionZip) | 直接为输入 LLM 的视觉 token 做 dominant token 保留和 contextual token 合并，是四者里最自然的 LLM 前方案。 |
| FOLDER | FOLDER, ICCV 2025 | [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_FOLDER_Accelerating_Multi-Modal_Large_Language_Models_with_Enhanced_Performance_ICCV_2025_paper.html), [arXiv](https://arxiv.org/abs/2501.02430), [official repo](https://github.com/anakin-skywalker-Joseph/Folder) | 原方法更偏视觉 backbone 末层内合并；迁移到 LLM 前时应保留 bipartite fold merge 这个核心算子。 |
| SCOPE | SCOPE, NeurIPS 2025 / OpenReview | [OpenReview](https://openreview.net/forum?id=oUghNi5XWc), [NeurIPS 2025 abstract](https://papers.nips.cc/paper_files/paper/2025/hash/ec6b4456c2bdfd04002d7984043c4936-Abstract-Conference.html), [arXiv](https://arxiv.org/abs/2510.24214), [official repo](https://github.com/kinredon/SCOPE) | 用 saliency + coverage 做纯选择，适合做 LLM 前 pruning baseline，但要注意文档 OCR 重复锚点不能被过度去重。 |

## 本仓库的输入路径

当前 MRL 主路径可以概括为：

```text
image / crops
-> processor 生成 pixel_values 与 image_grid_thw
-> base_model.visual(pixel_values, grid_thw=image_grid_thw)
-> image_embeds 替换 input_ids 中的 image_token 占位
-> language_model decoder layers
-> custom_text_proj
-> compact / MRL late interaction loss
```

关键本地参考：

| 文件 | 可复用内容 |
|---|---|
| `colqwen_multigranularity/core.py` | 标准 MRL forward：`inner_forward -> custom_text_proj`。 |
| `llmpre/visionzip_mrl/modeling_visionzip_mrl.py` | 已有 `adapter_pre` 与 `llm_early` 两种位置、stage/crop map、hard prune sequence packing、soft mask training。 |
| `llmpre/softstage/modeling_softstage.py` | LLM 前 soft mask 写回 `inputs_embeds` 的简单参考。 |
| `mlppost/strategies/common.py` | 四类算法的 MLP-post 版本算子，可迁移为 `compress_crop()` 的原型。 |

## 候选位置定义

| 位置 | 路径位置 | 是否满足“LLM 层前” | 优点 | 风险 |
|---|---|---:|---|---|
| P1: vision_late | Qwen visual backbone 末层内或输出前 | 是 | 最接近 FOLDER 原论文；可能减少部分视觉 backbone 后段成本。 | 需要改 `base_model.visual` 内部，和 Qwen2.5-VL / FlashAttention / grid 逻辑耦合高。 |
| P2: adapter_pre | `base_model.visual(...)` 输出后，进入 language_model decoder 前 | 是 | 最清晰、最可控；真正减少 LLM token 数；已有 VisionZipMRL 可复用。 | 不减少 visual encoder 本身成本；hard prune 会改变序列长度，需要同步 attention mask / position ids。 |
| P3: llm_early | 先跑 K 层 LLM，再压缩，继续跑剩余 LLM | 否，严格说已经进入 LLM | 可以用早期跨模态上下文评分，质量可能更稳。 | 不符合本次“LLM 层前”的主目标；节省少量前 K 层之后的 LLM 成本。 |
| P4: mlp_post | `custom_text_proj` 后 | 否 | 现有代码成熟，检索 embedding 压缩有效。 | 不减少 LLM 计算；不是本次目标。 |

本次主目标采用 P2 `adapter_pre`。FOLDER 单独保留 P1 作为长期增强路线，因为它原始设计更贴近视觉 backbone 内合并。

`adapter_pre` 的精确定义见 `ADAPTER_PRE_POSITION.md`。简要说，它位于：

```text
base_model.visual(pixel_values, grid_thw=image_grid_thw)
-> image_embeds 写入 inputs_embeds 中 image_token 占位
-> adapter_pre compression
-> base_model.model.language_model.layers[0]
```

因此它能减少进入 LLM decoder 的视觉 token 数，但不减少 Qwen visual encoder 本身的计算。

## 总体结论

| 方法 | 推荐主位置 | 可选位置 | 结论 |
|---|---|---|---|
| VisionZip | P2 adapter_pre | P3 llm_early | 最适合先实现。论文目标就是筛选输入 LLM 的视觉 token，本仓库已有同类实现。 |
| PruMerge | P2 adapter_pre | P1 vision_late / P3 llm_early | 适合迁移，但 Qwen2.5-VL 没有直接等价的 CLIP CLS attention 时，应改用 learned saliency + visual similarity。 |
| SCOPE | P2 adapter_pre | P3 llm_early | 适合做纯 pruning baseline；需要保护文档检索中的重复 OCR/layout 锚点。 |
| FOLDER | P2 adapter_pre 作为第一版；P1 vision_late 作为更贴近论文的第二版 | P3 llm_early | 对检索最友好的 merge primitive。若只做 LLM 前，P2 足够；若要严格复现论文加速边界，应做 P1。 |

## 共用实现骨架建议

建议不要为四个算法各写一套完整模型。先抽一个公共 PreLLM 压缩框架，再把四个算法作为 crop-level strategy plug-in。

建议目录：

```text
llmpre/pre_llm_algorithms/
  __init__.py
  common.py
  modeling_pre_llm_compress.py
  strategies/
    prumerge.py
    visionzip.py
    folder.py
    scope.py
  train_pre_llm_compress.py
  eval_pre_llm_compress.py
  run_train.sh
  eval_3sets.sh
```

建议公共接口：

```python
class PreLLMCompressionStrategy(nn.Module):
    def forward_crop(
        self,
        tokens: torch.Tensor,
        *,
        text_context: torch.Tensor | None,
        stage_index: int,
        crop_index: int,
        mode: str,  # "soft" or "hard"
    ) -> CompressionOutput:
        ...

@dataclass
class CompressionOutput:
    tokens: torch.Tensor
    keep_mask: torch.BoolTensor | None
    stats: dict
```

公共模型负责：

1. 用 `base_model.visual(pixel_values, grid_thw=image_grid_thw)` 得到 image embeds。
2. 根据 `image_grid_thw` 和 stage spec 构建 `stage_map` / `crop_map`。
3. 在 P2 位置对每个 crop 调用 strategy。
4. 训练时默认 soft mode，保持原序列长度，避免 MRL loss 的 `input_ids` mask 对齐问题。
5. eval / inference 可启用 hard mode，真正删除视觉 token，并同步重建 `attention_mask` 和 `position_ids`。
6. 跑完整 language model，再做 `custom_text_proj` 和 MRL compact。

## 训练与 hard prune 约束

当前 `MRLInBatchNegativeLoss` 的 g1/g2/g3 mask 是从原始 `input_ids` 推导出来的。训练时如果直接 hard prune，输出序列长度会和原始 `input_ids` 不一致，MRL mask 会错位。

因此第一版统一采用：

| 阶段 | 压缩方式 | 原因 |
|---|---|---|
| train | soft mask / differentiable merge，保持序列长度不变 | 稳定 DDP 和 MRL mask，对齐已有训练路径。 |
| eval smoke | 同时跑 soft 与 hard | 检查 hard prune 真实路径和 soft 训练目标是否一致。 |
| formal eval | hard prune 为主，soft eval 为对照 | hard prune 才代表真实 LLM 前压缩收益。 |

如果后续要 hard prune training，需要重写 loss，使其接收压缩后的 `output_mask`、stage labels 或 level masks，而不是只从原始 `input_ids` 推导。

## 每个算法的位置考察

### VisionZip

推荐位置：P2 `adapter_pre`。

理由：

1. 原论文目标就是减少输入 LLM 的视觉 token。
2. 算法只需要视觉 token 表征、token score、token-token 相似度，不强依赖 LLM 隐状态。
3. 本仓库已有 `llmpre/visionzip_mrl`，包括 `adapter_pre` 和 `llm_early` 两种模式、per-crop 压缩和 hard sequence packing。

建议实现：

```text
image_embeds / inputs_embeds image span
-> per crop:
   dominant tokens = top-k(score)
   contextual target tokens = uniform sample from residual tokens
   merged contextual tokens = residual-to-contextual similarity merge
-> replace image spans in inputs_embeds
-> language_model
```

建议默认参数：

| 参数 | 建议值 | 说明 |
|---|---:|---|
| stage keep ratios | `1.0,0.5,0.25` | g1 不压，g2/g3 渐进压缩。 |
| dominant ratio | `0.65` | 沿用当前 llmpre/visionzip_mrl 默认。 |
| contextual ratio | `0.05` | 保留少量上下文 token 做 merge target。 |
| train mode | soft | 保持长度。 |
| eval mode | hard prune | 验证真实 LLM token 减少。 |

风险：

- VisionZip 的 uniform contextual sampling 对文档局部 OCR 不一定最优。
- 如果 g3 压得太狠，ViDoReV2 report-style 数据容易丢表格和小字。

优先级：第一优先级。先复用并清理 `llmpre/visionzip_mrl`，再扩展成统一四策略框架。

### PruMerge

推荐位置：P2 `adapter_pre`。

可选位置：P1 `vision_late`，但只适合第二阶段。

理由：

1. PruMerge 的 prune+merge 结构适合在 LLM 前减少 token。
2. 原论文依赖 CLIP visual encoder 中 CLS-to-patch attention 的稀疏性；Qwen2.5-VL visual 输出不一定提供同等可用的 CLS attention。
3. 在 P2 使用 learned saliency + visual similarity 更符合本仓库已有训练体系，也能复用 MLP-post 的 `StagePatchScorer` 思路。

建议实现：

```text
per crop tokens
-> score tokens
-> keep top-k important tokens
-> optional spatial quota / PruMerge+ style uniform supplement
-> assign pruned tokens to kept anchors by cosine/key similarity
-> update anchors with merged residual information
-> output kept/updated anchors
```

建议默认策略：

| 组件 | 建议 |
|---|---|
| saliency | 第一版用 learned scorer；如果能低风险取 Qwen visual attention，再做 attention scorer ablation。 |
| merge | cosine similarity over image embeds；soft training，hard eval。 |
| spatial supplement | 建议加入，尤其保护 g2/g3 crop 的角落、页眉页脚、表格边缘。 |
| crop scope | 先 per-crop，不跨 crop 合并。 |

风险：

- 不使用原始 CLS attention 时，不能声称是精确 PruMerge 复现，应写作 PruMerge-style。
- 如果只选 top saliency，容易过度集中到大标题或显著图块，丢掉小字号 OCR。

优先级：第二优先级。它和 VisionZip 共用大量 sequence packing 和 merge 代码。

### FOLDER

推荐位置：

```text
第一版：P2 adapter_pre FOLDER-style
第二版：P1 vision_late FOLDER-faithful
```

理由：

1. FOLDER 原论文更强调在视觉 backbone 中后段做 token merge，理论上比 P2 更贴近原始方法。
2. 但 P1 需要改 Qwen2.5-VL visual backbone 内部，风险明显高于 P2。
3. 本项目的视觉文档检索结果显示 merge-preserving 比纯 pruning 更稳；所以即使只做 P2，FOLDER-style 仍然很值得作为 LLM 前主力 baseline。

P2 第一版建议：

```text
per crop tokens after base_model.visual
-> split into bipartite partitions
-> compute similarity / redundancy
-> merge redundant source tokens into destination tokens
-> maintain token_size
-> repeat until target count
-> run language_model on shorter visual sequence
```

P1 第二版建议：

```text
进入 base_model.visual 内部
-> 在末层或倒数第 N 个 visual block 后插入 fold merge
-> 更新后续 visual block 的 token sequence 和 grid/position 相关信息
-> 输出更短 image_embeds 给 LLM
```

风险：

- P2 FOLDER 不能节省 visual encoder 成本，只节省 LLM 成本。
- P1 FOLDER 会触碰 Qwen2.5-VL visual 的内部结构，和 `image_grid_thw`、spatial merge、RoPE/position 逻辑耦合，调试成本高。
- FOLDER 会产生 merged token；要检查 merged token 对 ColQwen MaxSim 是否仍像真实视觉 token 一样稳定。

优先级：第三优先级。若目标是尽快形成 LLM 前对照，先做 P2；若目标是论文级 acceleration story，再做 P1。

### SCOPE

推荐位置：P2 `adapter_pre`。

理由：

1. SCOPE 是 saliency + coverage 的 selection/pruning 方法，天然适合在输入 LLM 前裁剪视觉 token。
2. 它不需要 merge token，因此实现最干净。
3. 但视觉文档检索里，重复 OCR、表格行、页眉页脚可能是 MaxSim 需要的锚点；SCOPE 的 coverage/diversity 目标可能把这些当作冗余删掉。

建议实现：

```text
per crop tokens
-> compute saliency score
-> compute pairwise token relationship
-> greedy select token with max saliency * incremental coverage gain
-> output selected tokens only
```

建议默认保护规则：

| 保护项 | 说明 |
|---|---|
| g1 minimum quota | 全局页信息不能被压得过低。 |
| per-crop minimum quota | 防止某个局部 crop 被 coverage 贪心完全忽略。 |
| spatial quota | 每个 crop 的边界/角落保留少量 token，减少 OCR/table 丢失。 |
| duplicate quota ablation | 不做完全去重，至少保留 2 个近重复代表作为 report-style 数据 ablation。 |

风险：

- 纯 pruning 没有信息回收，强压缩时会比 FOLDER/PruMerge/VisionZip 更容易掉质量。
- coverage 贪心是 O(N^2) 关系计算，g3 token 很多时要控制 crop 内计算规模。

优先级：第四优先级，适合作为 pure pruning 对照和 coverage ablation。

## 多粒度与 crop 处理规则

第一版必须 per-crop 压缩，不做跨 crop 合并：

| Stage | Crop 数 | 第一版规则 |
|---|---:|---|
| g1 | 1 | 可少压或不压，保留全页上下文。 |
| g2 | 2 | 每个 crop 独立压缩，压缩后按原 crop 顺序拼回。 |
| g3 | 4 | 每个 crop 独立压缩，避免不同局部区域相互吞并。 |

不建议第一版做跨 g1/g2/g3 homogeneity merge。LLM 前压缩先要证明单 crop / 单 stage 稳定，再把 `folder_homo` 的 residual 思路迁进来。

## 推荐实验顺序

| 阶段 | 内容 | 通过标准 |
|---|---|---|
| S0 | 从 `visionzip_mrl` 抽公共 P2 框架 | soft train 和 hard eval 都能跑通 smoke。 |
| S1 | VisionZip P2 | 2-GPU smoke + 小数据 hard eval 无 shape/mask 错位。 |
| S2 | PruMerge P2 | 与 VisionZip 共用 scorer/packing，确认 merge 后 token norm 稳定。 |
| S3 | FOLDER P2 | 检查 merged token 的 MaxSim 行为，重点看 ViDoReV2 report-style。 |
| S4 | SCOPE P2 | 作为 pruning baseline，加入 spatial/min-quota 保护。 |
| S5 | FOLDER P1 | 只有在 P2 结果有价值且需要视觉 backbone 加速故事时启动。 |

## 评估口径

必须同时报告：

| 指标 | 原因 |
|---|---|
| 进入 LLM 的 visual token 数 | 证明不是 MLP-post 压缩。 |
| 完整 compact 后 doc token 数 | 对应 index / MaxSim 成本。 |
| Prefill 时间或 forward tokens/s | 证明 LLM 前压缩有效。 |
| ViDoReV1 NDCG@5 | 旧主指标。 |
| ViDoReV2 NDCG@5 | 报告类 OCR/table 风险最高。 |
| MMEB Recall@1 | 检查通用多模态检索能力。 |
| hard vs soft eval gap | 判断 soft training 是否能代表 hard prune。 |

建议先用：

```text
EVAL_MODE=smoke
VIDORE_V1: syntheticDocQA_energy
VIDORE_V2: esg_reports_human_labeled_v2
MMEB: MMEB-eval-VisDial-beir
```

正式结果再跑三套完整 eval。

## 命名建议

为了避免和历史 `mlppost`、`visionzip_mrl` 混淆，建议新实现统一命名：

```text
prellm_visionzip
prellm_prumerge
prellm_folder
prellm_scope
```

运行目录建议：

```text
experiments/exp_stagecompress/llmpre/pre_llm_algorithms/runs/
```

run name 示例：

```text
prellm_visionzip_adapterpre_softtrain_hardeval_g1-1_g2-05_g3-025_8gpu_4k
prellm_prumerge_adapterpre_spatialplus_g1-1_g2-05_g3-025_8gpu_4k
prellm_folder_adapterpre_foldmerge_g1-1_g2-05_g3-025_8gpu_4k
prellm_scope_adapterpre_saliencycoverage_quota_g1-1_g2-05_g3-025_8gpu_4k
```

## 当前决策

1. 本次迁移的主落点是 P2 `adapter_pre`，即视觉编码器输出后、LLM decoder 前。
2. VisionZip 是第一实现对象，因为它和本地历史实现最贴近。
3. PruMerge、FOLDER、SCOPE 都应写成同一个 PreLLM 框架下的 strategy，而不是复制四套 model/trainer。
4. FOLDER 有两条路线：先做 P2 可控版本，再考虑 P1 视觉 backbone 内版本。
5. 训练阶段先不 hard prune；hard prune 用于 eval/inference，除非同步重写 MRL loss mask。
6. 文档检索场景不能盲目追求去重。重复 OCR/table/layout token 可能是 MaxSim ranking 的有效锚点，所有 pruning/coverage 方法都要做 quota 或 anchor-preservation ablation。
