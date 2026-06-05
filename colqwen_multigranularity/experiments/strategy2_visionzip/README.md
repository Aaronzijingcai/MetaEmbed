# Strategy2 VisionZip

本文档记录 `strategy2_visionzip` 当前可运行的 VisionZip 视觉 token 压缩策略。当前策略在视觉编码器输出后、LLM 输入前压缩视觉 token，并使用 Qwen2.5-VL vision encoder 内部 attention 分数选择 dominant token。

## 当前入口

当前只保留两个 attention 训练入口，二者均直接调用 `train_4gpu.sh`：

```bash
# 方案 A：crop-first，逐 crop 压缩后汇总到 g1/g2/g3
bash colqwen_multigranularity/experiments/strategy2_visionzip/train_crop_first_visual_attn.sh

# 方案 B：stage-level，先按 stage 汇总 crop，再逐 stage 压缩
bash colqwen_multigranularity/experiments/strategy2_visionzip/train_stage_level_visual_attn.sh
```

两个脚本的共同默认配置：

```text
ATTN_IMPL=eager
VISIONZIP_ATTENTION_SOURCE=visual_attn
VISIONZIP_VISUAL_ATTN_LAYER=-2
VISIONZIP_BUDGETS=64 128 256
VISIONZIP_DOMINANT_RATIO=0.75
VISIONZIP_TARGET_SELECT=uniform
VISIONZIP_MERGE_METRIC=cosine
PRESERVE_INPUT_RMS=1
MAX_STEPS=4000
```

`visual_attn` 必须使用 `attn_implementation="eager"`。如果使用 `flash_attention_2` 或 `sdpa`，主代码会直接报错，不会静默退回 `self_similarity`。

## 目标

- 在 `base_model.visual(pixel_values, grid_thw=image_grid_thw)` 得到视觉 embedding 后、`base_model.inner_forward(...)` 前压缩视觉 token。
- 使用视觉 encoder 内部 attention 分数选择 dominant token，不依赖文本 query。
- 支持 `--granularities 1 2 4` 对应的 g1/g2/g3 三个阶段。
- 使用绝对 token 预算，而不是 keep ratio：g1=64、g2=128、g3=256。
- 保持 compact sequence、`compact_image_grid_thw`、MRL group mask 三者对齐。

不做的事情：

- 不在 LLM 内部剪枝。
- 不引入额外可训练 scorer。
- 不改 processor 的多 crop 生成逻辑。

## 算法流程

每个样本的视觉编码器输出：

```python
image_embeds: Tensor[raw_visual_tokens, hidden_size]
image_grid_thw: LongTensor[total_crops, 3]
```

当前固定三阶段 layout：

```text
g1: 1 crop
g2: 2 crops
g3: 4 crops
total_crops = 7
```

每个 crop 的 merged visual token 数由：

```python
crop_tokens = image_grid_thw.prod(dim=1) // (spatial_merge_size * spatial_merge_size)
```

得到 crop spans 后，再按 `1/2/4` crop 数构造 stage spans。

### Attention 分数

当前 `attention_source=visual_attn` 的分数来源：

```text
base_model.visual.blocks[visual_attn_layer].attn
默认 visual_attn_layer = -2
```

主代码 monkey patch 选定 vision block 的 attention，在 eager path 中显式计算：

```python
raw_scores = (q @ k.transpose(-2, -1)) * scaling
attn_probs = softmax(raw_scores)
token_saliency = attn_probs.mean(dim=(0, 1, 2))
```

含义：

```text
每个 key token 的 saliency = 被所有视觉 query token、所有 attention head 平均关注的强度
```

当前使用 softmax 后的 `attn_probs`，不是 softmax 前的 `raw_scores`。

Qwen2.5-VL 的 attention 是 patch-level，而 `image_embeds` 是 spatial merge 后的 visual token。代码会：

1. 按 `visual.spatial_merge_unit` 把 patch-level saliency 平均到 merged-token saliency。
2. 用 `reverse_indices = argsort(window_index)` 对齐回 `image_embeds` 顺序。
3. 再按 crop span 或 stage span 同步切分 token 和 saliency。

如果 `visual_attn_layer` 对应 window attention block，则 saliency 反映窗口内 attention；如果要强制全局 attention，应选择模型 `fullatt_block_indexes` 中的层。

### 两步压缩

对每个压缩单元独立执行。压缩单元由 `VISIONZIP_SCOPE` 决定：

```text
crop  -> 单个 crop
stage -> 单个 g1/g2/g3 stage
```

1. Dominant Token Selection
   - 根据 saliency 取 `dominant_budget` 个最高分 token。
   - 这些 token 直接保留，不参与平均合并。
   - `topk` 只决定选哪些 token；输出时按原始 token index 升序排列，不按 attention 分数降序排列。

2. Contextual Tokens Merging
   - 对非 dominant token 生成 `contextual_budget` 个上下文 token。
   - `target_select=uniform` 时，从 residual token 中均匀选择 target token。
   - 用 embedding key 的 cosine 相似度把剩余 residual token 分配到最近 target。
   - 每个 cluster 做朴素平均，平均包含 target token 自身。
   - contextual token 输出顺序跟 target index 顺序一致。

单个压缩单元最终输出：

```python
compressed = concat([dominant_tokens, contextual_tokens], dim=0)
compressed = compressed[:unit_budget]
```

## 两种实验模式

### 方案 A：Crop-First

入口：

```bash
bash colqwen_multigranularity/experiments/strategy2_visionzip/train_crop_first_visual_attn.sh
```

核心配置：

```text
VISIONZIP_SCOPE=crop
VISIONZIP_CROP_BUDGET_MODE=proportional
```

流程：

```text
image_embeds
  -> split_by_crop(image_grid_thw)
  -> VisionZip(crop_0)
  -> VisionZip(crop_1)
  -> ...
  -> aggregate compressed crops by original stage
  -> C1, C2, C3
```

预算：

- 先得到 stage budget：g1=64、g2=128、g3=256。
- 再把每个 stage budget 按 crop token 数比例分配到该 stage 内的 crop。
- 同一 stage 内，压缩后的 crop 按原 crop 顺序拼接。

特点：

- 每个 crop 内独立按 attention saliency 选 dominant token。
- 不会在压缩阶段混合不同 crop。
- 每个 crop 有机会保留信息，但不能跨 crop 去重。

### 方案 B：Stage-Level

入口：

```bash
bash colqwen_multigranularity/experiments/strategy2_visionzip/train_stage_level_visual_attn.sh
```

核心配置：

```text
VISIONZIP_SCOPE=stage
```

流程：

```text
image_embeds
  -> split_by_stage(image_grid_thw)
  -> VisionZip(g1)
  -> VisionZip(g2)
  -> VisionZip(g3)
  -> C1, C2, C3
```

预算：

```text
g1 = 64
g2 = 128
g3 = 256
```

特点：

- 每个 stage 内统一按 attention saliency 选 dominant token。
- 可以在同一 stage 内跨 crop 去重。
- 同一 stage 内不同 crop 会竞争 dominant token，弱信息 crop 可能被压缩得更狠。

## Budget 规则

每个 stage 的有效预算：

```python
stage_budget = min(config_budget, original_stage_tokens)
```

dominant/contextual 划分：

```python
def partition_budget(budget: int, token_count: int, dominant_ratio: float) -> tuple[int, int]:
    budget = min(int(budget), int(token_count))
    if budget <= 0:
        return 0, 0
    if budget == 1:
        return 1, 0
    dominant = max(1, int(round(budget * dominant_ratio)))
    dominant = min(dominant, budget - 1)
    contextual = budget - dominant
    return dominant, contextual
```

默认 `dominant_ratio=0.75`。例如 g1 预算 64 时，约 48 个 dominant token、16 个 contextual token。`0.9` 可以作为 ablation，但小 budget 下 contextual token 会偏少。

## Compact Sequence 与 MRL 对齐

无论 crop-first 还是 stage-level，最终都重建为：

```text
text_without_vision_markers
+ <vision_start> C1 <vision_end>
+ <vision_start> C2 <vision_end>
+ <vision_start> C3 <vision_end>
```

每个非空 `Ci` 对应一行 compact grid：

```python
compact_grid_i = [1, spatial_merge_size, len(Ci) * spatial_merge_size]
```

必须满足：

```python
sum(stage_raw_lengths) == image_embeds.shape[0]
stage_compact_lengths[i] == Ci.shape[0]
sum(Ci.shape[0] for Ci in [C1, C2, C3]) == compact_visual_tokens
```

MRL loss 的 level mask 语义：

```text
g1 mask: text + C1
g2 mask: text + C1 + C2
g3 mask: text + C1 + C2 + C3
```

`<vision_start>/<vision_end>` marker 会计入非空 stage 的长度。

## 训练日志检查

启动时日志应出现：

```text
attn_impl=eager
attention_source=visual_attn
visual_attn_layer=-2
visionzip_scope=crop 或 stage
budgets=64 128 256
max_steps=4000
```

训练中应检查：

```text
strategy2_visionzip_doc_compact_visual_tokens
strategy2_visionzip_doc_g1_tokens
strategy2_visionzip_doc_g2_tokens
strategy2_visionzip_doc_g3_tokens
strategy2_visionzip_mrl_g1/g2/g3
strategy2_visionzip_mrl_active_ratio_g1/g2/g3
```

正常情况下：

```text
doc_compact_visual_tokens == doc_g1_tokens + doc_g2_tokens + doc_g3_tokens
doc_g1_tokens ~= 64
doc_g2_tokens ~= 128
doc_g3_tokens ~= 256
mrl_active_ratio_g1/g2/g3 == 1.0
```

如果日志里出现：

```text
attn_impl=flash_attention_2
attention_source=self_similarity
```

说明跑的是 self-similarity 版本，不是 attention 分数版本。

## 评测

评测脚本：

```bash
bash colqwen_multigranularity/experiments/strategy2_visionzip/eval_3sets.sh /path/to/checkpoint
```

如果 checkpoint 使用 `visual_attn`，评测也必须设置：

```bash
ATTN_IMPL=eager
VISIONZIP_ATTENTION_SOURCE=visual_attn
```

`eval_3sets.sh` 会读取 checkpoint 中的 `strategy2_visionzip_config.json`，并顺序评测 Vidore v1、Vidore v2、MMEB。

## 已知限制

### Eager Attention 成本

`visual_attn` 需要 `ATTN_IMPL=eager`，速度和显存开销明显高于 `flash_attention_2`/`sdpa`。这是拿到完整 attention matrix 的代价。

### RoPE Grid 是线性近似

压缩后 token 不再对应真实 2D patch grid。当前复用线性 grid：

```python
[1, spatial_merge_size, length * spatial_merge_size]
```

这保证 `get_rope_index()` 可运行，但不是精确空间位置恢复。

### Loss 侧 Stage Length 估算

当前 `VisionZipMRLInBatchNegativeLoss` 只能看到原始 `input_ids/attention_mask`，不能直接看到每个样本的 `image_grid_thw`。默认训练脚本使用：

```text
CROP_RESIZE_MODE=stretch
granularities=1 2 4
```

在这个默认设置下，每个 crop 的视觉 token 数一致，loss 侧按 `1:2:4` crop 数比例估算 stage raw length 能与 model 端对齐。若后续改成动态分辨率或不同 crop token 数，需要把真实 per-stage compact lengths 或 `image_grid_thw` 传入 loss，否则 MRL group mask 可能错位。

## 文件说明

```text
compression.py
  VisionZipConfig、crop/stage split、budget 分配、dominant/contextual 压缩。

modeling.py
  pre-LLM 压缩接入、visual attention saliency hook、compact sequence 重建。

loss.py
  VisionZipMRLInBatchNegativeLoss，构造 g1/g2/g3 group mask 和 token 统计。

train_visionzip.py
  训练入口模块，解析 VisionZip 参数并构建模型/loss。

train_4gpu.sh
  通用 4 GPU launcher，打印关键配置并调用 train_visionzip。

train_crop_first_visual_attn.sh
  crop-first attention 实验入口。

train_stage_level_visual_attn.sh
  stage-level attention 实验入口。

eval_visionzip.py / eval_3sets.sh
  VisionZip checkpoint 评测入口。

smoke_validate.py
  压缩长度和 MRL mask 对齐的轻量验证。
```
