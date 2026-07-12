# 2026-07-08 主模型训练计划：FolderHomo + MaxSim 交互机制

## 目标

本目录用于启动论文主模型训练。基于 2026-07-01 的 MaxSim 机制探索结果，当前最值得进入全量训练的两条路线是：

1. `q2d_query_top48`：单向 Query TopK48 + Mean，当前最强、最稳定的主线。
2. `bi_topk_mean48_adaptive_lam08`：自适应双向 TopK48 + Mean，用于验证更有解释性的 adaptive MaxSim 是否能在全量训练后超过固定单向策略。

这两组都使用同一个主模型框架：

- Backbone: `models/colqwen2.5-base`
- Model: `FolderHomo`
- Token budget: `160 160 160`
- Train data: `configs/train/moca_data_ratios_v3_full.yaml`
- Data coverage: MMEB 全量训练任务 + visual document retrieval 大规模语料 `tevatron_colpali`、`visrag_ind`
- Scheduler: `constant`
- Warmup: `0`
- LR: 默认 `1e-4`
- Batch: 当前主跑目标为 `8 GPUs x TRAIN_BSZ 10`，`INTERLEAVED_BSZ=10`，`GRAD_ACCUM_STEPS=1`
- Steps: 正式目标为 `MAX_STEPS=90000`，每 `SAVE_STEPS=3000` 保存一次完整 checkpoint。
- 默认从 base model 重新训练，不从 `sym160` 或 7 月 1 日 checkpoint 继续训练。

注意：当前脚本不是线性学习率衰减，实际传入 `LR_SCHEDULER_TYPE=constant` 且 `WARMUP_STEPS=0`。由于不做 warmup，默认学习率从 `2e-4` 下调到 `1e-4`。

## 训练长度与断点续训策略

MetaEmbed 原模型使用更长训练长度。当前主线训练按完整 `90k` step 设置，但每 `3k` step 保存一次 checkpoint，便于阶段性评估和异常恢复。

1. 默认 `MAX_STEPS=90000`，`SAVE_STEPS=3000`。
2. 训练会保存 `checkpoint-3000/6000/9000/.../90000`。
3. 如果中途需要停下来评估，直接停止进程，使用最近的 checkpoint 继续训练。
4. 继续训练必须使用同一个 run 的 `RESUME_CKPT`，而不是重新加载 adapter warm start。
5. 每个阶段重点评估 MMEB worst10、MMEB 36 任务和 ViDoRe v2。如果指标平台期或 MMEB 明显退化，则停止继续烧算力。

断点继续训练只能使用同一个 run 的 `RESUME_CKPT`，例如：

```bash
RUNS=q2d \
MAX_STEPS=90000 \
SAVE_STEPS=3000 \
RESUME_CKPT=experiments/2026-07-08/runs/full_mmeb_vidore_q2d_topk48_mean_from_base/checkpoint-3000 \
experiments/2026-07-08/run_full_main_models.sh
```

`MAX_STEPS` 表示最终全局 step，不是“再训练多少 step”。例如从 `checkpoint-3000` 继续时仍然写 `MAX_STEPS=90000`，训练会从第 3000 step 继续到第 90000 step。

`WARM_START_ADAPTER_PATH` 默认保持为空。除非明确做 warm-start 对照实验，否则不要用 7 月 1 日或其他旧 checkpoint 作为 warm start。主线续训只使用 `RESUME_CKPT`，因为它会恢复 optimizer/scheduler/global step 等完整训练状态；只加载 LoRA adapter 不等价于连续训练。

2026-07-09 口径：

- MaxSim 机制探索已经在 `experiments/2026-07-01/MaxSim交互` 中收敛。
- 本目录只记录下一阶段主模型全量训练，不再归档探索阶段的负向对照和 scorer sweep 结果。
- 当前进入全量训练的两条机制为 `q2d_query_top48` 与 `bi_topk_mean48_adaptive_lam08`。

## 2026-07-09 BSZ 调试、等价性检查与最终主跑配置

为支持更大 batch，对 TopK MaxSim 的执行路径做了两类不改变数学定义的优化：

1. `query_chunk_size` 正确传入 `FolderHomoMRLInBatchNegativeLoss`，使 `QUERY_CHUNK_SIZE` 只控制矩阵乘分块大小。
2. 在 TopK reducer 内按 `query_mask/doc_mask` 的最大有效前缀裁掉无效尾部 token。MRL 的 group mask 是前缀式，因此被裁掉的 token 本来全部被 mask 掉，不参与 score。

已在远端用随机 embeddings 做等价性测试，并故意把无效尾部 token 放大 50 倍以检查 mask 是否真的生效。结果：

| 检查项 | 结果 |
|---|---|
| g1/g2/g3 `q2d_query_topk` mean | `max_abs_diff=0` |
| g1/g2/g3 `q2d_query_topk_sum` | `max_abs_diff=0` |
| g1/g2/g3 negative diagonal q2d mean | `max_abs_diff=0` |
| g1/g2/g3 bi adaptive pairwise | `max_abs_diff=0` |
| g1/g2/g3 bi adaptive diagonal | `max_abs_diff=0` |

结论：这些优化是严格等价的执行优化，不会损害原始 MaxSim/MRL 训练目标。

随后对同一条 `q2d_query_topk48` 训练链路做了 `TRAIN_BSZ=8/10/12` 短测，均使用 8 卡、`GRAD_ACCUM_STEPS=1`、`QUERY_CHUNK_SIZE=64`、`DOC_CHUNK_SIZE=128`。

| 单卡 BSZ | 全局 batch | 2 step 结果 | Trainer 吞吐 | 结论 |
|---:|---:|---|---:|---|
| 8 | 64 | 148s 完成 | 1.338 samples/s | 稳定 |
| 10 | 80 | 156s 完成 | 1.534 samples/s | 最快，作为主跑配置 |
| 12 | 96 | 288s 仍停在 0/2，无 checkpoint | 不可用 | 长序列 batch 下显存接近满载且吞吐崩 |

结论：正式全量主模型训练使用 `TRAIN_BSZ=10`、`INTERLEAVED_BSZ=10`、`GRAD_ACCUM_STEPS=1`。`TRAIN_BSZ=12` 不再作为主跑配置。

## 运行配置

默认脚本会顺序训练两组实验：

| Run | 训练 mode | 评估 scorer | 含义 |
|---|---|---|---|
| `full_mmeb_vidore_q2d_topk48_mean_from_base` | `q2d_query_topk` | `q2d_query_topk48` | 单向 Query TopK48 + Mean |
| `full_mmeb_vidore_bi_topk48_adaptive_mean_from_base` | `bi_query_topk_adaptive` | `bi_topk_mean48_adaptive_lam08` | 自适应双向 TopK48 + Mean，lambda max = 0.8 |

## 启动

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity

setsid env \
  MAX_STEPS=90000 \
  SAVE_STEPS=3000 \
  LEARNING_RATE=1e-4 \
  TRAIN_BSZ=10 \
  INTERLEAVED_BSZ=10 \
  GRAD_ACCUM_STEPS=1 \
  QUERY_CHUNK_SIZE=64 \
  DOC_CHUNK_SIZE=128 \
  EVAL_BSZ=4 \
  CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 \
  NUM_GPUS=8 \
  MAIN_PROCESS_PORT_BASE=29780 \
  experiments/2026-07-08/run_full_main_models.sh \
  > experiments/2026-07-08/runs/launch_full_main_models_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
```

如果单机只想跑其中一组：

```bash
RUNS=q2d experiments/2026-07-08/run_full_main_models.sh
RUNS=adaptive experiments/2026-07-08/run_full_main_models.sh
```

短测建议：

```bash
RUNS=q2d MAX_STEPS=2 SAVE_STEPS=1 TRAIN_BSZ=10 INTERLEAVED_BSZ=10 \
  GRAD_ACCUM_STEPS=1 QUERY_CHUNK_SIZE=64 DOC_CHUNK_SIZE=128 \
  RUN_SUFFIX=probe_bsz10_acc1_q64_prefix SKIP_EVAL=1 \
  experiments/2026-07-08/run_full_main_models.sh
```

## 评估

训练脚本默认每个 run 训练完成后会调用：

```bash
experiments/2026-07-08/eval_full_main_models.sh
```

评估集沿用当前主线：

- MMEB worst10: Precision@1
- ViDoRe v2: nDCG@5

如果后续需要补全完整 MMEB 36 任务或完整 ViDoRe，可以在评估脚本里扩展，不改训练目录结构。

## 记录口径

- 7 月 8 日目录只记录主模型训练，不再混入探索性 scorer-only 表格。
- 7 月 1 日结果作为方法筛选依据。
- 本目录结果用于论文主表或核心 ablation。

## 2026-07-12 step 984 卡顿诊断

背景：`q2d_query_topk48_mean` 与 `bi_topk_mean48_adaptive_lam08` 两组正式训练都在约 `984/4000` step 附近长时间无进展。由于两台服务器表现一致，优先排查数据/cache 与固定样本顺序，而不是直接改训练目标。

### Cache 修正

训练脚本已统一改为使用共享 cache，避免继续写项目目录 `.cache` 或 `/root/.cache`：

```bash
HF_DATASETS_CACHE=/MURE-V2/env/hf_datasets_cache
HF_HOME=/MURE-V2/env/mure_cache/colqwen_multigranularity/huggingface
MURE_CACHE_ROOT=/MURE-V2/env/mure_cache/colqwen_multigranularity
TMPDIR=/MURE-V2/env/mure_cache/colqwen_multigranularity/tmp
```

修正前，正式训练实际写到了项目目录：

```bash
/MURE-V2/code/MetaEmbed/colqwen_multigranularity/.cache/huggingface/datasets
```

服务器上该目录已经膨胀到约 `305G`，说明之前确实存在 cache 位置不统一的问题。后续主训练必须显式使用上面的共享 cache 路径。

### 样本顺序复现

用 `debug_stuck984_batches.py` 抽取 `970-1000` step，连续执行两次：

- pass1: `debug_970_1000_pass1_20260712_101812`
- pass2: `debug_970_1000_pass2_20260712_103038`

对比结果：

| 检查项 | 结果 |
|---|---|
| records | `248 vs 248` |
| missing rank-step | `0` |
| sample order diff | `0` |

结论：当前数据顺序是可复现的；同样配置下，`970-1000` step 两次抽到的 `sample_id` 完全一致。因此可以定位 `984` 附近的具体样本。

### 984 附近样本分析

在修正 image token 统计后，对 `980-986` step 做 collate 精查：

- run: `debug_980_986_tokens_20260712_103528`
- ranks: `8/8`
- records: `56`
- examples: `560`
- collate: 使用正式 `MultimodalRetrieverCollator`
- train batch: `TRAIN_BSZ=10`, `INTERLEAVED_BSZ=10`

最慢 batch：

| rank | step | collate 秒 | max image tokens | max input len | 主要子集 |
|---:|---:|---:|---:|---:|---|
| 7 | 984 | 9.19 | 5236 | 5283 | `tevatron_colpali`, `Visual7W`, `WebQA`, `MSCOCO`, `visrag_ind` |
| 3 | 981 | 8.70 | 5313 | 5339 | `InfographicsVQA`, `VisualNews_t2i`, `visrag_ind`, `ChartQA`, `OK-VQA`, `VOC2007`, `tevatron_colpali`, `ArxivQA` |
| 7 | 981 | 8.61 | 5376 | 5426 | `visrag_ind`, `tevatron_colpali`, `A-OKVQA`, `Visual7W`, `InfoSeek_it2t`, `N24News` |
| 5 | 984 | 7.87 | 5320 | 5388 | `visrag_ind`, `ArxivQA` |

代表性长样本：

| sample_id | subset | rank/step | q image tokens | d image tokens | neg image tokens | max input len |
|---|---|---:|---:|---:|---:|---:|
| `N24News:7851` | `N24News` | 2/981 | 5320 | 0 | 0 | 5429 |
| `Visual7W:40531` | `Visual7W` | 7/981 | 5376 | 0 | 0 | 5426 |
| `N24News:47445` | `N24News` | 4/985 | 5320 | 0 | 0 | 5420 |
| `SUN397:5728` | `SUN397` | 1/981 | 5376 | 0 | 0 | 5416 |
| `visrag_ind:119815` | `visrag_ind` | 0/982 | 0 | 5376 | 5376 | 5402 |
| `visrag_ind:120185` | `visrag_ind` | 0/982 | 0 | 5376 | 5376 | 5402 |

当前判断：

1. `984` 不是由某一个单独样本造成的“无限卡死”。在 `980-986` 精查中，984 附近 collate 最慢约 9 秒，属于长样本慢 batch，但不是分钟级卡死。
2. 这段确实有大量接近 `5.3k-5.4k` input length / image tokens 的样本，集中在 `N24News`、`Visual7W`、`SUN397`、`InfographicsVQA`、`visrag_ind` 等子集。
3. 两次抽样顺序完全一致，说明如果后续训练再次卡在相同步数，可以稳定复现并定位同一批样本。
4. 更可疑的问题是 cache 路径不统一和大 cache 反复构建/读取。后续正式训练应先清理或至少避开项目目录 `.cache`，并统一使用 `/MURE-V2/env/hf_datasets_cache`。

### 后续执行建议

不要为了越过 `984` 直接改变训练目标或关闭 gather。下一步应保持正式训练配置不变，只做：

1. 使用共享 cache 路径重启正式训练。
2. 从 `0` 或已确认可靠的 checkpoint 重新跑，观察是否仍在 `984` 附近卡住。
3. 若仍卡住，优先检查 cache/IO/dataset worker 状态；其次再考虑对 `N24News`、`Visual7W`、`SUN397`、`InfographicsVQA`、`visrag_ind` 的极长样本做专项统计或上限裁剪。
