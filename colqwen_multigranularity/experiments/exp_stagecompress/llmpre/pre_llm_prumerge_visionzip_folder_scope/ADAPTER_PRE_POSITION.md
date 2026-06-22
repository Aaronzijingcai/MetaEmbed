# adapter_pre 位置说明

更新时间：2026-06-19

## 一句话定义

`adapter_pre` 是视觉 token 已经经过 Qwen2.5-VL 的 visual encoder / visual merger，已经变成可以写入 LLM `inputs_embeds` 的 `image_embeds`，但还没有进入 `language_model` decoder layers 的位置。

它不是 PEFT adapter 的前后位置，也不是 `custom_text_proj` 前的位置。这里的 `adapter` 更接近多模态模型里的视觉适配/投影输出，即视觉侧输出适配到 LLM hidden size 之后、LLM decoder 之前。

## 本仓库中的实际张量流

标准 MRL 路径在 `colqwen_multigranularity/core.py` 中大致是：

```text
input_ids + pixel_values + image_grid_thw
-> base_model.inner_forward(...)
   -> embed text tokens
   -> base_model.visual(pixel_values, grid_thw=image_grid_thw)
   -> image_embeds 替换 input_ids 中的 image_token 占位
   -> language_model decoder layers
-> custom_text_proj(last_hidden_states)
-> normalize + compact
```

`adapter_pre` 把压缩插入到 `base_model.visual(...)` 和 `language_model decoder layers` 之间：

```text
input_ids
-> base_model._embed_tokens(input_ids) 得到 inputs_embeds
-> base_model.visual(pixel_values, grid_thw=image_grid_thw) 得到 image_embeds
-> masked_scatter: image_embeds 写入 inputs_embeds 中 image_token 对应位置
-> adapter_pre compression: 对 image spans 做选择/合并/剪枝
-> language_model decoder layers
-> custom_text_proj
-> retrieval multi-vector embeddings
```

## 代码定位

已有可参考实现位于：

```text
llmpre/visionzip_mrl/modeling_visionzip_mrl.py
```

关键函数：

| 函数 | 作用 |
|---|---|
| `_build_inputs_embeds()` | 调 `base_model.visual(...)`，把 `image_embeds` 写入 `inputs_embeds` 的 image token 位置。 |
| `_stage_and_crop_maps()` | 根据 `input_ids` 和 `image_grid_thw` 建立每个视觉 token 属于 g1/g2/g3、哪个 crop 的映射。 |
| `_compress_visionzip_sequence()` | hard prune / hard merge 后重建短序列、`attention_mask` 和 `position_ids`。 |
| `_soft_visionzip_sequence()` | 训练时保持原序列长度，只在 image token 位置做 soft mask / soft merge。 |
| `_project_hidden_states_with_mask()` | `adapter_pre` 分支的主入口。先构建 `inputs_embeds`，再压缩，再跑完整 LLM。 |

`adapter_pre` 分支的核心顺序是：

```text
inputs_embeds = _build_inputs_embeds(...)
position_ids = base_model.get_rope_index(...)

if has_images and visionzip_position == "adapter_pre":
    stage_map, crop_map = _stage_and_crop_maps(...)
    if eval hard prune:
        inputs_embeds, attention_mask, position_ids = _compress_visionzip_sequence(...)
    else:
        inputs_embeds = _soft_visionzip_sequence(...)

hidden_states = _run_language_layers(
    hidden_states=inputs_embeds,
    start_layer=0,
    end_layer=num_layers,
)
proj = custom_text_proj(hidden_states)
```

## 和其他位置的区别

| 位置 | 插入点 | 是否进入 LLM 前 | 能否减少 LLM token 计算 | 备注 |
|---|---|---:|---:|---|
| `vision_late` | visual backbone 末层内部或输出前 | 是 | 是 | 更贴近 FOLDER 原论文，但要改 Qwen visual 内部。 |
| `adapter_pre` | visual 输出写入 `inputs_embeds` 后、LLM decoder 前 | 是 | 是 | 本次四算法迁移的推荐主位置。 |
| `llm_early` | 先跑若干层 LLM，再压缩 | 否 | 只能减少后续层 | 可利用早期跨模态上下文，但不严格满足“LLM 层前”。 |
| `mlp_post` | `custom_text_proj` 后 | 否 | 否 | 只减少 retrieval embedding / MaxSim / index 成本。 |

## 为什么 adapter_pre 适合这四个算法

PruMerge、VisionZip、FOLDER、SCOPE 都以视觉 token 的选择、合并或覆盖为核心。`adapter_pre` 已经拿到了 LLM hidden size 的视觉 token，且尚未进入 LLM，因此：

1. 可以真实减少 LLM prefill / attention 的视觉 token 数。
2. 不需要改 Qwen2.5-VL visual backbone 内部，工程风险低于 `vision_late`。
3. 可以复用现有 `image_grid_thw`、stage/crop map、sequence packing 逻辑。
4. 训练时可以先用 soft mask 保持序列长度，避免 MRL mask 和原始 `input_ids` 错位。
5. 评估时再使用 hard prune / hard merge，报告真实进入 LLM 的 token 数。

## 需要注意的限制

`adapter_pre` 不减少 visual encoder 本身的计算。它节省的是从 LLM decoder 开始的 token 计算，以及后续 `custom_text_proj` / compact 后的序列处理成本。

如果目标是“视觉 encoder 也加速”，需要走 `vision_late`，尤其是 FOLDER 更接近这个路线。但 `vision_late` 要处理 visual RoPE、spatial merge、`image_grid_thw` 与内部 block 输出长度的同步，风险明显更高。

## 对当前任务的结论

本次把四个算法“更新到 LLM 层前”时，`adapter_pre` 应作为第一实现位置：

```text
base_model.visual(...) 输出后
image_embeds 写入 inputs_embeds 后
language_model.layers[0] 之前
```

其中 VisionZip、PruMerge、SCOPE 都直接采用 `adapter_pre`。FOLDER 第一版也采用 `adapter_pre` 做 FOLDER-style merge；只有在需要更贴近论文的 visual-backbone 加速时，再做第二版 `vision_late`。
