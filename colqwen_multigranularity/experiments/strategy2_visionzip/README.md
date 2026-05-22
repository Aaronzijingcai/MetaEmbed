# VisionZip Technical Design

## 1. 目标

本目录给出一个适配当前 `MetaEmbed / ColQwen2.5 / MRL` 训练框架的 VisionZip 方案设计文档。

核心目标不是在 LLM 内部剪枝，而是在视觉 token 进入下游 late-interaction 表征之前完成压缩：

```text
pixel_values
-> visual encoder / projector
-> stage-wise VisionZip compression
-> text + C1 / text + C1 + C2 / text + C1 + C2 + C3
-> multi-granularity retrieval loss
```

这和当前仓库 `exp_stagecompress` 的总体插入点一致，只是把压缩策略明确收敛到 VisionZip 风格。

## 2. 仓库现状

主流程已经在 `exp_stagecompress` 中打通，关键位置如下：

- `modeling_stagecompress.py`
  - `StageCompressMRLColQwen2_5.forward()`
  - 先调用 `_project_hidden_states(...)` 取出文本 token 和图像 token 的投影结果
  - 再进入 `stage_compressor(...)`
- `StageCompressor.forward()`
  - 按三段 `G1/G2/G3` 拆分图像 token
  - 对每段分别调用压缩 block
  - 最后重组为 `text + C1 + C2 + C3`
- `strategies/common.py`
  - 提供 `StagePatchScorer`
  - 提供 `_forward_visionzip_impl(...)`
- `strategies/strategy4_visionzip.py`
  - 已经有一个可运行的 `strategy4_visionzip` baseline

也就是说，当前仓库并不缺 VisionZip 的“接入点”，缺的是一份更清晰、更适配编码器差异的正式设计。

## 3. 当前实现和原始 VisionZip 思路的差异

当前 `exp_stagecompress/strategies/strategy4_visionzip.py` 的实现是一个 **VisionZip 风格 baseline**，但不是“直接读取视觉编码器原生 attention” 的完全版。

当前实现的主导分数来源：

- 不是直接读取视觉编码器里 CLS token 的注意力
- 也不是直接读取每层 self-attention map 的全局平均
- 而是先用 `StagePatchScorer` 对 stage token 做一次轻量 MHA 增强
- 再由一个可训练 score head 产生 `saliency`

然后在 `common.py::_forward_visionzip_impl(...)` 中做：

1. `topk(saliency)` 选 dominant tokens
2. 在剩余 token 里均匀采样 contextual anchors
3. 用余弦相似度把其余 token 分配到 contextual anchors
4. 聚合得到 contextual tokens

所以当前版本更准确地说是：

`VisionZip-style stage compressor`

而不是：

`strict encoder-attention VisionZip`

## 4. 适配当前仓库的推荐方案

建议把 `strategy2_visionzip` 定义成一个 **两级可适配框架**：

### 4.1 一级：统一外部接口

无论底层视觉编码器有没有 CLS，都统一输出：

- `dominant_tokens`
- `contextual_tokens`
- `compressed_tokens = [dominant ; contextual]`

这样上层 `StageCompressor`、训练入口、评估入口都不用改。

### 4.2 二级：可切换的主导 token 打分后端

把 dominant token selection 的分数来源设计成可插拔后端：

1. `encoder_cls_attn`
   - 适用于带 CLS 的编码器，例如 CLIP 类结构
   - 直接使用 CLS -> patch attention 作为 importance

2. `encoder_mean_attn`
   - 适用于无 CLS 的编码器，例如 SigLIP 类结构
   - 对 patch-to-patch attention 做全局平均，得到每个 token 的被关注强度

3. `learned_saliency`
   - 适用于当前 ColQwen2.5 这条工程链路
   - 不强依赖底层视觉 encoder 暴露原生 attention
   - 直接复用现有 `StagePatchScorer`

推荐默认优先级：

```text
如果能稳定拿到 encoder attention:
    优先 encoder_cls_attn / encoder_mean_attn
否则:
    回退到 learned_saliency
```

这样既保留 VisionZip 的方法语义，也和当前仓库的可实现性对齐。

## 5. 推荐算法定义

设单个 stage 的输入 token 为：

```text
X = {x_1, ..., x_N}, x_i in R^d
```

压缩预算为 `K`，其中：

- `K_dom = round(K * r_dom)`
- `K_ctx = K - K_dom`

默认建议：

- `r_dom = 0.75 ~ 0.9`
- 当前仓库 baseline 已使用 `0.9 / 0.1`
- 更稳妥的工程默认值建议用 `0.8 / 0.2`

因为当前 MRL 三段结构里，后两段 token 数量大、信息密度不均，`0.8 / 0.2` 往往比极端 `0.9 / 0.1` 更抗退化。

### 5.1 Step A: Dominant Token Selection

先得到每个 token 的 importance score `s_i`。

三种可适配计算方式：

#### A. 带 CLS 的视觉编码器

若视觉编码器某层 attention 为：

```text
A in R^(H x T x T)
```

其中 `token 0` 为 CLS，则：

```text
s_i = mean_h A[h, CLS, i]
```

选取 `TopK(s, K_dom)` 作为 dominant tokens。

#### B. 无 CLS 的视觉编码器

若没有 CLS，则改用全局平均关注：

```text
s_i = mean_h mean_j A[h, j, i]
```

即 token `i` 被其它 token 关注的平均强度。

#### C. 当前仓库兼容回退

若上游拿不到原生 attention，则使用：

```text
enhanced, saliency = StagePatchScorer(tokens)
s_i = saliency_i
```

这也是当前 `strategy4_visionzip` 的做法。

### 5.2 Step B: Contextual Tokens Merging

对非 dominant tokens 记为 residual set：

```text
R = X \ D
```

其中 `|D| = K_dom`。

再从 `R` 中选 `K_ctx` 个上下文目标 token，作为 merge anchors。

推荐提供三种 anchor 选法：

1. `uniform`
   - 在 residual token 序列里均匀采样
   - 最贴近当前代码，最稳定

2. `random`
   - 随机选 `K_ctx` 个 token
   - 最贴近你描述里的“随机选择 target 个 token”

3. `coverage`
   - 用 farthest point / greedy coverage 选 anchor
   - 质量更高，但实现更重

对于当前仓库，建议默认：

```text
anchor_method = uniform
```

因为它和现有 `common.py::_select_uniform_indices(...)` 直接一致。

随后对其余 residual token 做聚类式归并：

```text
q_m = normalize(anchor_m)
k_i = normalize(key_i)
assign(i) = argmax_m <k_i, q_m>
```

其中 `key_i` 推荐取：

- `enhanced token feature`
- 或单独加一层 `Linear(d, d)` 得到 merge key

聚类后对每组取平均：

```text
c_m = mean({x_i | assign(i)=m})
```

若希望保留强 token 的贡献，可做加权平均：

```text
w_i = softmax(s_i / tau)
c_m = sum_i w_i x_i / sum_i w_i
```

当前仓库的 `strategy4_visionzip` 已经接近这个形式，只是它先保留 anchor token 本身，再加上归并残差。

## 6. 推荐的工程化版本

为了适配现在的代码结构，建议把 VisionZip 拆成三个逻辑层。

### 6.1 Scorer 层

职责：只负责输出 token importance。

建议接口：

```python
scores = scorer(tokens, encoder_attn=None, has_cls=False, text_context=None)
```

支持：

- `encoder_cls_attn`
- `encoder_mean_attn`
- `learned_saliency`

### 6.2 Selector/Merger 层

职责：在单个 stage 内做 VisionZip 压缩。

建议接口：

```python
compressed = visionzip_block(
    tokens,
    scores,
    enhanced_tokens=None,
    dominant_ratio=0.8,
    anchor_method="uniform",
    tau=1.0,
)
```

输出始终为 `K x d`。

### 6.3 Stage Wrapper 层

职责：嵌回当前 `StageCompressor`。

它只需要做三件事：

1. 拆出 `G1/G2/G3`
2. 对激活的 stage 调用 VisionZip block
3. 重组为 `text + C1 + C2 + C3`

这层当前仓库已经有，不建议再改接口。

## 7. 与 strategy1 的关系

可以直接借鉴 `strategy1_softassign` 的三点工程经验：

1. 训练入口和评估入口保持共享
   - 不单开一套 trainer
2. 压缩 block 只处理单 stage token
   - 不把复杂逻辑散到 loss 或 processor
3. 文档和脚本命名跟 method 名严格对齐
   - 方便复现实验矩阵

VisionZip 和 `strategy1_softassign` 的关键区别是：

- `strategy1`：所有 token 都被软分配进 prototype
- `visionzip`：先显式保留 dominant evidence，再压缩剩余 context

因此在文档定位上，VisionZip 更适合作为：

`结构保真型压缩基线`

而不是纯 pooling 基线。

## 8. 推荐默认配置

面向当前三阶段 MRL 结构，建议默认配置如下：

```text
method = strategy2_visionzip
compress_stages = all
budgets = 160 320 640
dominant_ratio = 0.8
anchor_method = uniform
score_source = learned_saliency
tau = 1.0
use_text_context = false
```

说明：

- `budgets = 160 320 640` 与现有 `exp_stagecompress` 主实验保持一致
- `dominant_ratio = 0.8` 比当前 baseline 的 `0.9` 更平衡
- `score_source = learned_saliency` 是当前仓库最稳的落地路径
- `use_text_context = false` 更符合 VisionZip “无文本依赖” 的初衷

如果未来要做更“原教旨”的 VisionZip，再切到：

```text
score_source = encoder_cls_attn / encoder_mean_attn
```

## 9. 推荐实现草案

建议在当前仓库里把 `strategy2_visionzip` 实现成下面这个形态。

### 9.1 配置项

建议新增：

```text
--stagecompress-visionzip-score-source
--stagecompress-visionzip-dominant-ratio
--stagecompress-visionzip-anchor-method
--stagecompress-visionzip-use-weighted-merge
```

可选值建议：

```text
score_source: learned_saliency | encoder_cls_attn | encoder_mean_attn
anchor_method: uniform | random | coverage
```

### 9.2 Block 伪代码

```python
def forward(tokens, encoder_attn=None, text_context=None):
    if len(tokens) <= budget:
        return normalize(tokens)

    enhanced = feature_enhancer(tokens)
    scores = build_scores(
        tokens=tokens,
        enhanced=enhanced,
        encoder_attn=encoder_attn,
        text_context=text_context,
    )

    k_dom, k_ctx = split_budget(budget, dominant_ratio)

    dominant_idx = topk(scores, k_dom)
    dominant = tokens[dominant_idx]

    residual = tokens[~dominant_idx]
    residual_feat = enhanced[~dominant_idx]
    residual_scores = scores[~dominant_idx]

    anchor_idx = select_anchors(residual_feat, k_ctx, method=anchor_method)
    anchors = residual[anchor_idx]
    anchor_feat = residual_feat[anchor_idx]

    others = residual[~anchor_idx]
    others_feat = residual_feat[~anchor_idx]
    others_scores = residual_scores[~anchor_idx]

    assign = argmax(normalize(others_feat) @ normalize(anchor_feat).T)
    context = merge_by_cluster(anchors, others, assign, others_scores, tau=tau)

    return normalize(concat(dominant, context))
```

## 10. 适配 CLIP / SigLIP / ColQwen2.5 的建议

### 10.1 CLIP 类编码器

推荐：

- `score_source = encoder_cls_attn`
- 直接使用 CLS attention 做 dominant token 排序

### 10.2 SigLIP 类编码器

推荐：

- `score_source = encoder_mean_attn`
- 使用全局平均 attention 替代 CLS

### 10.3 ColQwen2.5 当前工程链路

推荐：

- `score_source = learned_saliency`
- 先不强求读取底层视觉 encoder attention
- 直接复用 `StagePatchScorer`

这是最现实的原因：

当前 `StageCompressMRLColQwen2_5.forward()` 拿到的是已经投影后的 `hidden_states`，并不是天然就暴露视觉 encoder 每层 attention map。若强行把原生 attention 取出来，会牵动底层模型前向接口和缓存逻辑，工程入侵明显更大。

## 11. 推荐落地顺序

如果下一步要真正实现代码，建议按下面顺序推进：

1. 先把当前 `strategy4_visionzip` 视为 `learned_saliency` 版本的基线
2. 在此基础上补齐配置项
3. 把 dominant ratio 从硬编码改成可配置
4. 把 contextual anchor 从固定 `uniform` 改成可切换 `uniform/random/coverage`
5. 最后再研究是否值得从视觉 encoder 暴露原生 attention

这样可以先复用现有训练/评估链路，避免一开始就大改底层模型。

## 12. 结论

面向这个仓库，最合适的 VisionZip 方案不是“完全照搬论文描述”，而是：

- 保留 VisionZip 的两步结构
  - dominant token selection
  - contextual token merging
- 把 dominant score 设计成可插拔后端
  - `CLS attention`
  - `global mean attention`
  - `learned saliency`
- 在当前 ColQwen2.5 / MRL 代码里先走 `learned_saliency` 版本
- 在接口上为后续切换到 encoder-native attention 预留空间

这条路线兼顾了三件事：

1. 和 VisionZip 方法语义一致
2. 和当前 `exp_stagecompress` 主体代码兼容
3. 后续可以自然扩展到 CLIP / SigLIP / 其它视觉编码器

## 13. 相关代码参考

- `experiments/strategy1_softassign/README.md`
- `experiments/exp_stagecompress/modeling_stagecompress.py`
- `experiments/exp_stagecompress/compression.py`
- `experiments/exp_stagecompress/strategies/common.py`
- `experiments/exp_stagecompress/strategies/strategy4_visionzip.py`
