# FolderHomo V1 Importance-Only Ablation

## 1. 实验目的

本实验只研究 FOLDER-style homogeneity compressor 中的 token 重要分如何计算。

当前最强同质化基线是 FolderHomo residual160 / V1：

| Baseline | Budget | ViDoReV1 | ViDoReV2 | MMEB | Avg |
|---|---:|---:|---:|---:|---:|
| FolderHomo residual160 / V1 | 160/160/160 | 89.34 | 60.28 | 76.43 | 75.35 |

本实验目标不是重新证明 FOLDER 是否有效，而是在完全固定 FOLDER 相似分、token budget 和训练配置的前提下，只替换 unary importance/protect score，寻找是否存在优于当前 learned saliency head 的重要分来源。

核心问题：

```text
在 post-MLP retrieval token 空间中，哪些 token 应该被 FOLDER merge 保护？
```

成功标准：

```text
Avg 超过 75.35，且 ViDoReV2 不明显下降。
```

## 2. 实验方案

FOLDER 原本包含两个分数：

| 分数 | 作用 | 本实验是否改变 |
|---|---|---|
| Similarity score | 决定哪些 token 彼此相似、可以 merge | 不改变 |
| Importance / protect score | 决定哪些 token 不应该被 merge 掉 | 只改变这一项 |

相似分仍然使用原 FOLDER 的 post-MLP token cosine matching。本实验只替换每个 token 的一维 importance/protect score。

考虑到时间和算力约束，首批只保留差异性最大的 4 种方法：

| Priority | Mode | 重要分来源 | 选择原因 |
|---|---|---|---|
| P0 | `mlp` / `mlp_saliency` | learned saliency head | 必须保留。作为 FolderHomo V1 风格 baseline，也检查独立实现是否对齐。 |
| P0 | `mha_attn` / `mha_received_attn` | scorer 内部 MHA received attention | 直接回答核心假设：MHA attention 能不能替代额外 saliency MLP。 |
| P0 | `learned_gate` | scorer gate head 的 sigmoid 输出 | 与 saliency/attention 差异最大，测试 value/retention gate 是否足以表达 token 保留重要性。 |
| P1 | `mha_pagerank` | MHA attention graph 上的 PageRank centrality | 如果 P0 结果显示 attention 路线有希望，再测试二阶 graph centrality。 |

暂不优先跑：

| Mode | 原因 |
|---|---|
| `mha_entropy_confidence` | 与 `mha_attn` 同属 attention 去噪变体，差异性不足。代码保留，首批不占 8 卡实验。 |
| `coverage_gain` | 这是 coverage/gain objective，不是纯 importance score，应放入 `../增益分/`。 |

## 3. 参考论文与设计来源

本实验不是任意枚举 importance score，而是把已有 token compression 文献中的几类重要性信号迁移到 MURE-V2 的 post-MLP retrieval token 空间中。

| 本实验 mode | 参考来源 | 对应关系 |
|---|---|---|
| `mlp` / `mlp_saliency` | learned scorer / learned importance line，包括 DynamicViT 的 lightweight prediction module[^dynamicvit] | 用一个可训练 saliency head 直接预测 token 是否应被保护。 |
| `mha_attn` / `mha_received_attn` | attention-based token importance line，包括 PruMerge 的 attention-based important token selection[^prumerge] 和 SparseVLM 的 attention-matrix-based visual-token significance[^sparsevlm] | 不再从 vision encoder 或 LLM 内部取 attention，而是从 post-MLP scorer 自己的 MHA 中取 received attention。 |
| `learned_gate` | gated pruning / learned token retention line，和 DynamicViT 类方法共享“由模型学习 token 保留价值”的思想[^dynamicvit] | 不额外使用 saliency head，而是直接测试 scorer 内已有 gate 是否能表达保留价值。 |
| `mha_pagerank` | graph centrality / PageRank[^pagerank] | 把 MHA attention matrix 看成 token graph，用二阶中心性替代 raw received attention。 |

其他相关工作：

- VisionZip 说明 visual tokens 存在大量冗余，并将 token 分成 informative/dominant 与 contextual 部分进行压缩[^visionzip]。这支持“重要 token + 冗余 token merge”的总体实验动机。
- PruMerge 在保留重要 token 后再按 key similarity 合并被裁 token[^prumerge]，和本实验保持 FOLDER 相似分不动、只替换 protect score 的设定相近。
- SparseVLM 使用 attention 矩阵估计 visual token significance[^sparsevlm]，支持 `mha_attn` 这类 attention-score 方案。

需要注意：这些工作多数在 vision encoder、VLM interface 或 LLM 内部做 token pruning/compression；本实验的区别是所有 importance score 都在 `custom_text_proj` 之后的 retrieval token 空间中计算，并服务于 MaxSim multi-vector retrieval，而不是 generation acceleration。

## 4. 实验变量控制

所有变量都对齐 FolderHomo residual160 / V1：

| 变量 | 设置 |
|---|---|
| Base model | native Qwen2.5VL / ColQwen2.5 base |
| Trainable modules | LLM LoRA + `custom_text_proj` + post-MLP FOLDER compressor |
| Granularity | `1 2 4` |
| MRL supervision | `G1`, `G1+R2`, `G1+R2+R3` |
| Token budget | `160/160/160` |
| Total visual tokens | 480 |
| Compress stages | `all` |
| Similarity score | 原 FOLDER post-MLP cosine matching，不改变 |
| Novelty weight | `1.0` |
| Gate strength | `0.25` |
| Folder alpha | `1.0` |
| Training data | MoCa no-mmE5 text-query-focus config |
| Eval data | ViDoReV1 / ViDoReV2 / MMEB standard 3-set config |
| Max steps | 默认 `4000` |
| Batch setting | 8 GPU, per-device train batch size 4 |

默认训练命令已经固定 `BUDGETS=160 160 160`，不需要额外指定预算。

Smoke 训练和测试：

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity

IMPORTANCE_MODE=mlp \
  bash experiments/2026-07-01/探索重要分/smoke_train_eval.sh
```

默认 smoke 设置为单卡、2 training steps、保存 `checkpoint-2`，随后用 `EVAL_MODE=smoke` 评测 1 个 ViDoReV1 子集、1 个 ViDoReV2 子集和 1 个 MMEB 子集，每个子集只取少量 query/corpus。它只用于验证训练保存、`folder_importance.pt` 加载、评测入口和 smoke 限流是否正常，不用于汇报指标。

P0 训练：

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity

IMPORTANCE_MODE=mlp \
  bash experiments/2026-07-01/探索重要分/run_train.sh

IMPORTANCE_MODE=mha_attn \
  bash experiments/2026-07-01/探索重要分/run_train.sh

IMPORTANCE_MODE=learned_gate \
  bash experiments/2026-07-01/探索重要分/run_train.sh
```

P1 训练：

```bash
IMPORTANCE_MODE=mha_pagerank \
  bash experiments/2026-07-01/探索重要分/run_train.sh
```

评测示例：

```bash
IMPORTANCE_MODE=mha_attn \
  bash experiments/2026-07-01/探索重要分/eval_3sets.sh \
  experiments/2026-07-01/探索重要分/runs/folder_importance_v1_mha_attn_b160_160_160_4k/checkpoint-4000
```

## 5. 实验结果

结果记录规则：

| 项目 | 必须填写 |
|---|---|
| Train config | 训练 yaml 或数据配比 |
| Eval config | ViDoReV1 / ViDoReV2 / MMEB 对应配置 |
| Checkpoint | 具体 checkpoint 路径 |
| Token budget | 固定 `160/160/160`，共 480 visual tokens |
| Metric | ViDoRe / MMEB 三组沿用该实验原有表格口径 |

| Priority | Importance mode | Tokens | ViDoReV1 | ViDoReV2 | MMEB | Avg | Status | Notes |
|---|---|---:|---:|---:|---:|---:|---|---|
| Baseline | FolderHomo residual160 / V1 | 480 | 89.34 | 60.28 | 76.43 | 75.35 | DONE | 当前目标基线 |
| P0 | `mlp` | 480 | TODO | TODO | TODO | TODO | TODO | 独立实现对齐 learned saliency |
| P0 | `mha_attn` | 480 | TODO | TODO | TODO | TODO | TODO | MHA received attention |
| P0 | `learned_gate` | 480 | TODO | TODO | TODO | TODO | TODO | gate-based importance |
| P1 | `mha_pagerank` | 480 | TODO | TODO | TODO | TODO | TODO | 有时间再跑 |

## 6. 实验结论

当前结论待实验结果填充。

预设判断规则：

| 结果 | 结论 |
|---|---|
| 某个 mode 超过 Avg 75.35 且 ViDoReV2 不下降 | 可以作为新的 FolderHomo importance 主候选 |
| `mha_attn` 接近或超过 `mlp` | post-MLP scorer attention 可以作为更可解释的重要分来源 |
| `learned_gate` 接近或超过 `mlp` | 现有 gate 已捕获 token 保留价值，可简化 scorer |
| P0 中 attention 路线有效 | 再跑 P1 `mha_pagerank` |
| P0 全部不如 baseline | 当前 learned saliency head 仍是最稳的重要分设计，后续不继续扩 importance 矩阵 |

## 7. 文件说明

| File | Role |
|---|---|
| `config.py` | 实验配置和 importance mode 选项 |
| `modeling_importance.py` | 独立模型与 compressor 实现 |
| `train_importance.py` | 独立训练入口 |
| `eval_importance.py` | 独立评测入口 |
| `run_train.sh` | 8-GPU 训练脚本 |
| `eval_3sets.sh` | ViDoReV1 / ViDoReV2 / MMEB 评测脚本 |
| `smoke_train_eval.sh` | 单卡 smoke 训练 + smoke 评测链路检查 |

## References

[^prumerge]: Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, Yan Yan. "LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models." arXiv:2403.15388, 2024. https://arxiv.org/abs/2403.15388

[^visionzip]: Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, Jiaya Jia. "VisionZip: Longer is Better but Not Necessary in Vision Language Models." arXiv:2412.04467, 2024. https://arxiv.org/abs/2412.04467

[^sparsevlm]: Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, Shanghang Zhang. "SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference." arXiv:2410.04417, 2024. https://arxiv.org/abs/2410.04417

[^dynamicvit]: Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, Cho-Jui Hsieh. "DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification." arXiv:2106.02034, 2021. https://arxiv.org/abs/2106.02034

[^pagerank]: Lawrence Page, Sergey Brin, Rajeev Motwani, Terry Winograd. "The PageRank Citation Ranking: Bringing Order to the Web." Stanford InfoLab technical report, 1998. https://ilpubs.stanford.edu:8090/422/
