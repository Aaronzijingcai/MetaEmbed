## Soft Assignment 视觉压缩方案（面向 MetaEmbed / colqwen_multigranularity）

### 结论先行

这次压缩只针对 image token，不压缩 text token。

推荐方案是：

- 在 `Qwen2.5-VL` 的 `visual encoder` 输出之后、进入 `language model` 之前插入压缩模块
- 按 `g1 / g2 / g3` 三个粒度分别做 `soft assignment`
- 保持 `D1 / D2 / D3` 的嵌套结构不变
- 暂不直接改主线 `colqwen_multigranularity/train.py`，先以隔离实验分支实现

目标链路：

```text
vision encoder -> stage-wise soft assignment -> compact multimodal sequence -> llm
```

而不是当前 `ReMaG-Trim` 的：

```text
llm -> custom_text_proj -> token compression -> loss
```

后者只能减少检索 token 和 loss 成本，不能减少 LLM 注意力成本；本方案要求的是前者。

---

## 1. 背景与约束

在当前 `MetaEmbed/colqwen_multigranularity` 中：

- 文本先被 tokenizer 编码成 `input_ids`
- 文本 token 再通过 `embed_tokens` 变成 `inputs_embeds`
- 图像通过 `self.visual(pixel_values, grid_thw=image_grid_thw)` 变成视觉 embedding
- 视觉 embedding 替换掉序列中 `<|image_pad|>` 对应的位置
- 替换后的混合序列进入 `Qwen2.5-VL` 的 `language_model`

因此，如果要真正减少上下文长度和 LLM 计算，压缩必须插在：

```text
self.visual(...) 之后
self.model(...) 之前
```

也就是压缩“视觉 embedding 序列”，而不是压缩最终的检索 embedding。

---

## 2. 为什么只压 image token

这个选择是合理的，而且比“图文一起压缩”更稳。

### 2.1 主要收益来自图像侧

在文档检索场景里，序列长度的主要开销来自 image token：

- 文本 token 数通常较少
- image token 数远大于文本
- 多粒度展开后，`g1 + g2 + g3` 会进一步放大视觉 token 数

因此只压 image token，就能抓住大头，同时最小化对文本语义的破坏。

### 2.2 文本作为稳定锚点应保留

MRL 训练里，文本部分承担两个重要角色：

- query / document 的语义锚点
- `D1 / D2 / D3` 三层中始终共享的稳定部分

如果把 text token 也一起压缩，会引入两类额外风险：

- 文本语义被压平，影响 query-doc 对齐
- 各层共享的公共语义底座不稳定，影响 `D1 / D2 / D3` 之间的嵌套关系

所以本方案固定：

- `text token` 原样保留
- `image token` 才进入压缩器

---

## 3. 为什么要按 g1 / g2 / g3 分别压缩

不能把所有视觉 token 一起做一个全局 `T -> K` 压缩。

### 3.1 当前 loss 依赖层级结构

当前 `MRL` 的核心不是“只有一组视觉 token”，而是：

```text
D1 = text + g1
D2 = text + g1 + g2
D3 = text + g1 + g2 + g3
```

如果把全部 image token 一锅压成 `K` 个 token，那么：

- `g1 / g2 / g3` 的边界消失
- `D1 / D2 / D3` 无法重建
- 现有 MRL 训练目标不成立

### 3.2 分 stage 压缩可以保留嵌套结构

因此推荐对三个 stage 分别做压缩：

```text
G1: [T1, D] -> SA(P1, K1) -> C1: [K1, D]
G2: [T2, D] -> SA(P2, K2) -> C2: [K2, D]
G3: [T3, D] -> SA(P3, K3) -> C3: [K3, D]
```

最终仍然保持：

```text
D1 = [text ; C1]
D2 = [text ; C1 ; C2]
D3 = [text ; C1 ; C2 ; C3]
```

这和当前 `MRLInBatchNegativeLoss` 的使用方式是一致的，也便于后续改造 loss mask。

---

## 4. Soft Assignment 是否可学习

结论：**是可学习的，而且适合端到端训练。**

### 4.1 单个 stage 的定义

设某个 stage 的视觉 token 为：

```text
X_i in R^{T_i x D}
```

该 stage 的可学习 prototype 为：

```text
P_i in R^{K_i x D}
```

其中：

- `T_i` 是该 stage 压缩前的视觉 token 数
- `K_i` 是压缩后的预算
- `D` 是 hidden size

### 4.2 计算流程

对每个 stage：

```text
S_i = norm(X_i) @ norm(P_i)^T              # [T_i, K_i]
A_i = softmax(S_i / tau)                  # [T_i, K_i]
Z_i = A_i^T @ X_i                         # [K_i, D]
C_i = l2_norm(Z_i)                        # [K_i, D]
```

其中：

- `S_i` 是 token 到 prototype 的相似度
- `A_i` 是软分配矩阵
- `Z_i` 是聚合结果
- `C_i` 是进入 LLM 的压缩视觉 token

### 4.3 为什么它是可微的

整个流程由以下操作组成：

- 向量归一化
- 矩阵乘法
- softmax
- 加权求和

这些都是可微的，因此梯度可以从最终 retrieval loss 一路回传到：

1. `prototype P_i`
2. 输入视觉 token `X_i`
3. 视觉编码器参数

梯度路径可写成：

```text
retrieval loss
  -> llm hidden states
  -> compact visual tokens C_i
  -> assignment matrix A_i
  -> prototypes P_i
  -> raw visual tokens X_i
  -> visual encoder
```

所以这不是训练时的“启发式剪枝”，而是真正的可学习压缩。

### 4.4 为什么分 stage 后仍然可学习

按 `g1 / g2 / g3` 分开做 soft assignment 不会破坏可学习性，反而更稳定：

- 每个 stage 只负责自己的信息尺度
- coarse / medium / fine token 不会抢同一组 prototype
- `D1 / D2 / D3` 的层级监督会自然回传到对应 stage

也就是说：

- `P1` 学 coarse prototype
- `P2` 学 medium prototype
- `P3` 学 fine prototype

这和 MRL 的设计目标是对齐的。

---

## 5. 这种方案的主要优势

### 5.1 真正减少 LLM 成本

因为压缩发生在 `language_model` 之前，减少的是：

- LLM 输入长度
- Attention 计算量
- 中间激活显存
- 后续检索 token 数

### 5.2 保持 text 不变，训练稳定

文本 token 不参与压缩，避免了：

- 文本语义漂移
- query/doc 对齐不稳
- 多层共用语义底座丢失

### 5.3 保留 MRL 嵌套结构

分 stage 压缩后仍能构造：

```text
D1 <= D2 <= D3
```

这对当前多粒度训练是必要条件。

---

## 6. 需要注意的学习风险

虽然它是可学习的，但不是无风险。

### 6.1 Prototype collapse

表现：

- 多个 prototype 学成几乎一样
- 实际只用了少数 prototype

风险：

- 压缩后表达能力下降
- 预算 `K_i` 没有真正被利用

建议监控：

- prototype usage
- 每个 prototype 被分配到的平均 token 数
- prototype 相似度矩阵

可选正则：

- prototype 去相关 / 正交约束
- usage balance regularization

### 6.2 Assignment 过于均匀

表现：

- `A_i` 接近均匀分布
- 所有输出 token 趋同

常见原因：

- `tau` 太大

建议：

- 从 `tau = 0.1` 起步
- 训练初期可以略大，后期 anneal 变小
- 或把 `tau` 设成可学习参数，但需要 clamp

### 6.3 Assignment 过于尖锐

表现：

- 接近硬分配
- 梯度变差
- 某些 prototype 死掉

常见原因：

- `tau` 太小

建议：

- 前期不要上来就做近似 one-hot
- 推理阶段如需硬分配，单独开关，不影响训练

### 6.4 预算过小导致信息瓶颈

如果 `K1 / K2 / K3` 太小，会导致：

- coarse token 不够表达全局结构
- fine token 不够保留细节

所以预算应按 stage 分配，而不是平均切。

建议：

- `g1` 预算最小，但不能过低
- `g3` 预算最大
- 先做 128 / 256 / 512 三档实验

---

## 7. 与当前 MetaEmbed 代码的适配策略

### 7.1 不直接改主线入口

当前主线：

- `colqwen_multigranularity/core.py`
- `colqwen_multigranularity/train.py`
- `colqwen_multigranularity/eval.py`

已经稳定承载基础 MRL 训练。  
这次改动涉及：

- visual token 压缩
- multimodal sequence 重建
- compact 后的 `position_ids / rope`
- 新的 MRL mask

因此建议：

```text
先做隔离实验分支，不直接污染主线
```

建议新增目录：

```text
colqwen_multigranularity/experiments/soft_assignment/
├── compression.py
├── modeling.py
├── loss.py
├── train_softassign.py
├── eval_softassign.py
└── README.md
```

### 7.2 复用哪些现有模块

可直接复用：

- 数据集与 batch 构造：`colpali_engine/utils/mm_dataset_transformation.py`
- 多粒度裁剪与 prompt 构造：`colqwen_multigranularity/core.py`
- 训练器：`ContrastiveTrainerV2`
- 评测器：`eval.py` / `external_evaluate_dataset_loader`

需要新建：

- pre-LLM soft assignment compressor
- pre-LLM 重建后的模型封装
- 与压缩布局匹配的 MRL loss

---

## 8. 插入位置的精确定义

当前 `ColQwen2_5.inner_forward()` 的关键顺序是：

```text
input_ids
  -> embed_tokens
  -> inputs_embeds

pixel_values + image_grid_thw
  -> self.visual(...)
  -> image_embeds

image_embeds 替换 <image_pad> 对应位置
  -> self.model(...)
```

本方案的插入点是：

```text
self.visual(...)
之后
self.model(...)
之前
```

即：

```text
raw image_embeds
  -> split by stage
  -> stage-wise soft assignment
  -> compact image embeddings
  -> rebuild compact multimodal sequence
  -> self.model(...)
```

---

## 9. Stage 划分不建议再用“比例近似”，建议用 image_grid_thw 精确恢复

现有一些实现会按 `[1, 2, 4]` 比例估计 stage 边界。  
这在 loss 层做近似还可以，但 pre-LLM 压缩时不够稳。

更好的方式是：

- processor 输出的 crop 顺序是确定的：`g1 -> g2 -> g3`
- `image_grid_thw` 给出了每个 crop 对应的 token 网格
- 因此可以按 crop 顺序精确累计出每个 stage 的视觉 token 范围

建议：

```text
stage boundary = sum(tokens_per_crop) over crops in that stage
```

这样切分更准确，也不依赖“每个 crop token 数完全相同”的强假设。

---

## 10. Sequence 重建是必要步骤

压缩前：

- `input_ids` 中 image token 数量 = 原始视觉 token 数
- `image_embeds.shape[0] = T`

压缩后：

- 视觉 token 变成 `K1 + K2 + K3`

如果不重建 sequence，会出现：

- `input_ids` 和视觉 embedding 数量不匹配
- `rope` 位置不匹配
- 序列长度没真正变短

因此必须重建：

1. `compact_input_ids`
2. `compact_attention_mask`
3. `compact_inputs_embeds`
4. `compact_image_grid_thw`

### 推荐的重建方式

对每个 stage 构造一个新的视觉段：

```text
<vision_start> [K1 个 image token] <vision_end>
<vision_start> [K2 个 image token] <vision_end>
<vision_start> [K3 个 image token] <vision_end>
```

并为每段分配新的伪 `image_grid_thw`，例如：

```text
g1 -> [1, 1, K1]
g2 -> [1, 1, K2]
g3 -> [1, 1, K3]
```

这里的 `grid_thw` 不再代表真实二维 patch 结构，而是“压缩后的视觉序列布局”。

这是一种工程上的伪网格，但足以让：

- `get_rope_index`
- `self.model(...)`

继续按现有接口工作。

---

## 11. 推荐的参数化方式

### 11.1 每个 stage 独立一组 prototype

建议：

```text
P1 for g1
P2 for g2
P3 for g3
```

而不是一组 prototype 给所有 stage 共用。

原因：

- coarse / medium / fine 的信息密度不同
- 共用 prototype 容易让 coarse/fine 互相干扰
- 独立 prototype 更贴合 MRL 的层级设计

### 11.2 query / doc 共享同一套 stage prototype

建议同一个 stage 的 prototype 在 query 和 doc 两侧共享。

原因：

- query/doc 走同一个 backbone
- 共享 prototype 可以保持压缩后的表示空间更一致
- 参数更少，训练更稳

---

## 12. Loss 层需要什么变化

不能直接原样复用当前 `MRLInBatchNegativeLoss` 的 mask 构造逻辑。

新 loss 需要基于压缩后的布局重新构造：

```text
D1 = text + K1
D2 = text + K1 + K2
D3 = text + K1 + K2 + K3
```

对 text-only 样本：

```text
D1 = D2 = D3 = text
```

也就是说，loss 要认的是：

- 文本 token 原样保留
- image token 已被 stage-wise soft assignment 替换
- 各层长度由 `K1/K2/K3` 决定

---

## 13. 日志与监控要求

除了现有训练 loss，建议额外记录：

### 13.1 token 预算

- `raw_visual_tokens_g1/g2/g3`
- `compact_visual_tokens_g1/g2/g3`
- `keep_ratio_g1/g2/g3`

### 13.2 assignment 统计

- `assignment_entropy_g1/g2/g3`
- `assignment_max_prob_mean_g1/g2/g3`
- `prototype_usage_g1/g2/g3`
- `prototype_norm_g1/g2/g3`

### 13.3 梯度与学习状态

- `prototype_grad_norm_g1/g2/g3`
- `visual_grad_norm`

### 13.4 结构正确性

- `D1 <= D2 <= D3`
- text-only 样本三层长度相等
- image row 的 visual length 非 0

---

## 14. 参数保存与加载策略

Soft Assignment 是可学习模块，因此不仅要考虑“能训练”，还要考虑：

1. 训练中 checkpoint 是否完整
2. 训练结束导出是否完整
3. LoRA 开启时新参数会不会丢
4. 推理加载时是否能恢复同样的压缩结构

### 14.1 需要保存哪些参数

如果按推荐方案实现，每个 stage 都有独立 prototype，那么至少需要保存：

- `P1`：`g1` 的 prototype
- `P2`：`g2` 的 prototype
- `P3`：`g3` 的 prototype

如果温度 `tau` 是可学习的，还要保存：

- `tau1 / tau2 / tau3`，或等价的 `log_tau`

如果压缩器内部还有额外模块，例如：

- 小投影层
- gate
- normalization 仿射参数

这些参数也要一起保存。

### 14.2 推荐的模块组织方式

不建议把 prototype 直接散挂成若干裸 `nn.Parameter`。  
建议封装成一个命名明确的 `nn.Module`，例如：

```text
self.soft_assignment
```

内部包含：

- `g1_prototypes`
- `g2_prototypes`
- `g3_prototypes`
- 可选 `temperature` / `log_temperature`
- 其他辅助层

这样做的原因：

- 参数结构清晰
- `state_dict()` 自然完整
- LoRA 场景下更容易加入 `modules_to_save`
- 保存 / 加载逻辑更容易维护

### 14.3 当前主线保存逻辑的风险

当前 `MRLColQwen2_5` 的保存逻辑只会调用底层 `base_model.save_pretrained(...)`。

这意味着：

- 如果 Soft Assignment 参数挂在 wrapper 上
- 但不改 `save_pretrained()`

那么这些新参数可能不会被保存到最终导出目录里。

因此，Soft Assignment 版本的模型必须覆盖默认保存行为，不能直接沿用当前 wrapper 的简单透传。

### 14.4 推荐的保存方案

推荐使用“主模型 / adapter 与 soft assignment 分开保存”的方式。

输出目录建议结构：

```text
output_dir/
├── adapter_model.safetensors / pytorch_model.bin
├── adapter_config.json                # 如果使用 LoRA / PEFT
├── config.json
├── tokenizer_config.json
├── preprocessor_config.json
├── soft_assignment.bin
└── soft_assignment_config.json
```

其中：

- `adapter_model.safetensors` 或 `pytorch_model.bin`
  - 保存 backbone / LoRA / `custom_text_proj`
- `soft_assignment.bin`
  - 保存 `soft_assignment.state_dict()`
- `soft_assignment_config.json`
  - 保存压缩器结构配置

这样做的好处：

- 与主模型结构解耦
- 推理加载更明确
- 不依赖 `base_model.save_pretrained()` 是否知道新模块
- 实验分支和主线目录结构更清楚

### 14.5 必须保存的配置项

仅保存 prototype 权重是不够的。  
还需要保存 Soft Assignment 的结构超参数，否则推理时无法正确恢复模块。

建议保存：

```json
{
  "enabled": true,
  "compress_stages": "all",
  "budgets": [64, 64, 128],
  "temperature": 0.1,
  "learnable_temperature": false,
  "normalize_inputs": true,
  "normalize_outputs": true,
  "share_query_doc_prototypes": true,
  "stage_count": 3
}
```

至少应覆盖：

- 是否启用压缩
- 压哪些 stage
- 每个 stage 的预算 `K1/K2/K3`
- 温度及其是否可学习
- 是否输入归一化 / 输出归一化
- query/doc 是否共享 prototype

### 14.6 LoRA / PEFT 场景下的要求

如果训练时使用 LoRA，必须确保 Soft Assignment 模块不会在导出 adapter 时被漏掉。

推荐策略：

- backbone 投影层继续走 LoRA
- `custom_text_proj` 全量保存
- `soft_assignment` 全量保存

这意味着在 PEFT 配置中，`modules_to_save` 里至少应包含：

```text
custom_text_proj
soft_assignment
```

否则会出现这种风险：

- 训练时 Soft Assignment 参数参与更新
- 但 adapter 导出时没有把它们带上
- 最终推理结果与训练时不一致

### 14.7 中间 checkpoint 的要求

训练中断点保存时，也必须把 Soft Assignment 参数带上。

一个完整 checkpoint 至少应包含：

```text
checkpoint-XXXX/
├── pytorch_model.bin / adapter_model.safetensors
├── optimizer.pt
├── scheduler.pt
├── trainer_state.json
├── training_args.bin
├── soft_assignment.bin
└── soft_assignment_config.json
```

其中：

- `optimizer.pt` 会自然包含 Soft Assignment 参数的优化器状态，只要这些参数已注册并加入 optimizer
- `soft_assignment.bin` 用于确保压缩器权重在恢复训练时一致

### 14.8 加载策略

加载建议分两步：

#### Step 1：加载主模型

- 加载 `base_model`
- 如有 adapter，则加载 `adapter_path`

#### Step 2：加载 Soft Assignment

- 读取 `soft_assignment_config.json`
- 按配置构造 `soft_assignment` 模块
- 加载 `soft_assignment.bin`

即：

```text
base model / adapter
    +
soft assignment config
    +
soft assignment weights
    ->
完整推理模型
```

### 14.9 推荐原则

保存与加载策略的原则是：

```text
Soft Assignment 参数不能只依赖 base_model.save_pretrained()
Soft Assignment 结构配置必须显式保存
LoRA 开启时必须把 soft_assignment 纳入 modules_to_save
```

否则容易出现“训练到了，但导出丢了”的问题。

---

## 15. 实现顺序建议

### Phase 1：最小可运行

1. `compression.py`
   - `SoftAssignmentConfig`
   - `StageSoftAssignmentCompressor`
   - `SoftAssignmentCompressor`

2. `modeling.py`
   - `SoftAssignmentColQwen2_5`
   - pre-LLM sequence rebuild

3. `loss.py`
   - `SoftAssignmentMRLInBatchNegativeLoss`

### Phase 2：训练评测入口

4. `train_softassign.py`
5. `eval_softassign.py`

### Phase 3：验证

6. 1-step / 5-step smoke train
7. quick eval
8. 观察 assignment 和 prototype 日志

---

## 16. 验收标准

满足以下条件才算方案落地正确：

1. 图像 token 压缩发生在 LLM 前，而不是 LLM 后
2. text token 完全不压缩
3. `g1 / g2 / g3` 分别压缩，保留 `D1 / D2 / D3` 嵌套
4. 梯度能回传到：
   - prototype
   - raw image embeddings
   - visual encoder
5. query/doc 无图像时能自动旁路压缩模块
6. quick eval 能正常跑通
7. 日志里可见：
   - token 数下降
   - assignment 非塌缩
   - retrieval loss 正常收敛

---

## 17. 当前建议的最终方案

最终建议如下：

```text
只压 image token
+ 按 g1/g2/g3 分别做 soft assignment
+ 插在 visual encoder 后、LLM 前
+ 用 stage-specific prototypes
+ query/doc 共享同一套 stage prototypes
+ 重建 compact multimodal sequence
+ 新写与 compact 布局匹配的 MRL loss
+ 先在 experiments/soft_assignment 中隔离实现
```

这是当前仓库里：

- 与 `MetaEmbed` 多粒度结构最一致
- 与 `Qwen2.5-VL` 前向接口最兼容
- 同时又真正减少 LLM 计算的方案

---

## 18. 明确不做的事

本阶段先不做：

- 图文一起压缩
- 训练时硬分配
- 直接改主线 `train.py` / `core.py`
- 后 LLM 压缩替代前 LLM 压缩

这些都可以后续再做，但不是当前推荐落点。
