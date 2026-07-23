# 2026-07-08 主模型训练计划：FolderHomo + MaxSim 交互机制

## 目标

本目录用于启动论文主模型训练。基于 2026-07-01 的 MaxSim 机制探索结果，当前最值得进入全量训练的两条路线是：

1. `q2d_query_top48`：单向 Query TopK48 + Mean，当前最强、最稳定的主线。
2. `bi_topk_mean48_adaptive_lam08`：自适应双向 TopK48 + Mean，用于验证更有解释性的 adaptive MaxSim 是否能在全量训练后超过固定单向策略。

这两组都使用同一个主模型框架：

- Backbone: `models/colqwen2.5-base`
- Model: `FolderHomo`
- Token budget: `128 128 128`
- Train data: `configs/train/moca_data_ratios_v3_full.yaml`
- Data coverage: MMEB 全量训练任务 + visual document retrieval 大规模语料 `tevatron_colpali`、`visrag_ind`
- Scheduler: `linear`
- Warmup: `0`
- LR: 默认 `1e-4`
- Batch: 当前主跑目标为 `8 GPUs x TRAIN_BSZ=8`，`INTERLEAVED_BSZ=8`，`GRAD_ACCUM_STEPS=1`，全局 batch 为 64。
- Steps: 正式目标为 `MAX_STEPS=60000`，每 `SAVE_STEPS=1000` 保存一次完整 checkpoint。
- 默认从 base model 重新训练，不从 `sym160` 或 7 月 1 日 checkpoint 继续训练。

注意：当前脚本使用线性学习率衰减，实际传入 `LR_SCHEDULER_TYPE=linear` 且 `WARMUP_STEPS=0`。学习率从 `1e-4` 开始，并在 `MAX_STEPS=60000` 时衰减到 0。

## 训练长度与断点续训策略

MetaEmbed 原模型使用更长训练长度。当前主线训练按完整 `60k` step 设置，但每 `1000` step 保存一次 checkpoint，便于阶段性评估和异常恢复。

1. 默认 `MAX_STEPS=60000`，`SAVE_STEPS=1000`。
2. 训练会保存 `checkpoint-1000/2000/3000/...`，并在最终训练结束保存 `checkpoint-60000`。Trainer 当前设置 `save_total_limit=4`，只保留最近 4 个 checkpoint。
3. 如果中途需要停下来评估，直接停止进程，使用最近的 checkpoint 继续训练。
4. 继续训练必须使用同一个 run 的 `RESUME_CKPT`，而不是重新加载 adapter warm start。
5. 每个阶段重点评估 MMEB worst10、MMEB 36 任务和 ViDoRe v2。如果指标平台期或 MMEB 明显退化，则停止继续烧算力。

断点继续训练只能使用同一个 run 的 `RESUME_CKPT`，例如：

```bash
RUNS=q2d \
MAX_STEPS=60000 \
SAVE_STEPS=1000 \
RESUME_CKPT=experiments/2026-07-08/runs/rhc_mmeb_vidore_q2d_topk48_mean_from_base/checkpoint-1000 \
experiments/2026-07-08/run_full_main_models.sh
```

`MAX_STEPS` 表示最终全局 step，不是“再训练多少 step”。例如从 `checkpoint-1000` 继续时仍然写 `MAX_STEPS=60000`，训练会从第 1000 step 继续到第 60000 step。

`WARM_START_ADAPTER_PATH` 默认保持为空。除非明确做 warm-start 对照实验，否则不要用 7 月 1 日或其他旧 checkpoint 作为 warm start。主线续训只使用 `RESUME_CKPT`，因为它会恢复 optimizer/scheduler/global step 等完整训练状态；只加载 LoRA adapter 不等价于连续训练。在线性衰减设置下，断点续训会继承 scheduler 的 `last_epoch/global_step`，不会把学习率重新拉回最高值。

### 历史记录：2026-07-13 线性学习率衰减 smoke

> 本节记录当时的 `SAVE_STEPS=950` / `BUDGETS=160 160 160` 实验，仅用于追溯，不是当前主跑配置。

本轮将主训练口径从 `90k/3k/constant` 调整为 `60k/950/linear`：

- `MAX_STEPS=60000`
- `SAVE_STEPS=950`
- `LEARNING_RATE=1e-4`
- `LR_SCHEDULER_TYPE=linear`
- `WARMUP_STEPS=0`

远端已完成单卡快速 smoke，验证点如下：

1. 从 base 训练 `MAX_STEPS=3/SAVE_STEPS=3/LOGGING_STEPS=1`，生成 `checkpoint-3`，LR 日志为 `1e-4 -> 6.67e-5 -> 3.33e-5`，证明 linear scheduler 生效。
2. 严格 resume smoke 使用同一个 run：先以 `MAX_STEPS=6/SAVE_STEPS=3` 生成 `checkpoint-3` 和 `checkpoint-6`，确认原始 LR 曲线为 `1e-4 -> 8.33e-5 -> 6.67e-5 -> 5e-5 -> 3.33e-5 -> 1.67e-5`。
3. 删除该 smoke 的 `checkpoint-6` 后，从 `RESUME_CKPT=.../checkpoint-3` 继续到 `checkpoint-6`。日志明确显示 `Continuing training from global step 3`，续训 LR 为 `5e-5 -> 3.33e-5 -> 1.67e-5`，没有重新回到 `1e-4`。
4. `checkpoint-6` 包含 `folder_homo.pt`、`optimizer.pt`、`scheduler.pt`、`trainer_state.json`；读取状态显示 `global_step=6`、`scheduler_last_epoch=6`、`scheduler_base_lrs=[0.0001, 0.0001]`。
5. `eval_full_main_models.sh` 对 `checkpoint-6` 做 dry-run 通过，能正确展开 `q2d_query_topk48` scorer、`160 160 160` budget 与 MMEB worst10 / ViDoRe v2 评估入口。

### 历史记录：2026-07-16 8卡断点续训与评估 smoke

> 本节保留当时 `TRAIN_BSZ=10` 的原始实验数据，不代表当前 `TRAIN_BSZ=8` 配置。

本轮按正式 8 卡配置验证主训练的断点语义，重点确认两件事：从 base 首训不加载旧 adapter；从 checkpoint 续训时恢复 optimizer/scheduler/RNG/Trainer state，并跳过已消费 batch。

phase1 首训配置：

- `INTERACTION_STRATEGY=q2d_topk48_mean`
- `RUN_SUFFIX=resume_smoke_20260716_155827`
- `MAX_STEPS=6`
- `SAVE_STEPS=3`
- `LOGGING_STEPS=1`
- `LEARNING_RATE=1e-4`
- `LR_SCHEDULER_TYPE=linear`
- `TRAIN_BSZ=10`
- `INTERLEAVED_BSZ=10`
- `GRAD_ACCUM_STEPS=1`
- `WARM_START_ADAPTER_PATH=` and `RESUME_CKPT=`

phase1 结果：

- 输出目录：`experiments/2026-07-08/runs/rhc_mmeb_vidore_q2d_topk48_mean_from_base_resume_smoke_20260716_155827`
- 已保存：`checkpoint-3`、`checkpoint-6`
- LR 日志：`1e-4 -> 8.33e-5 -> 6.67e-5 -> 5e-5 -> 3.33e-5 -> 1.67e-5`

phase2 续训配置：

- `RESUME_CKPT=experiments/2026-07-08/runs/rhc_mmeb_vidore_q2d_topk48_mean_from_base_resume_smoke_20260716_155827/checkpoint-3`
- `RUN_SUFFIX=resume_smoke_phase2_from_ckpt3_20260716_160530`
- 其他训练参数与 phase1 相同。

phase2 结果：

- 输出目录：`experiments/2026-07-08/runs/rhc_mmeb_vidore_q2d_topk48_mean_from_base_resume_smoke_phase2_from_ckpt3_20260716_160530`
- 日志明确显示：`Continuing training from global step 3`
- 日志明确显示：`Will skip the first 0 epochs then the first 3 batches in the first epoch.`
- 续训 LR 日志：step 4 `5e-5`，step 5 `3.33e-5`，step 6 `1.67e-5`，没有重置回 `1e-4`
- `checkpoint-6` 内含 `folder_homo.pt`、`optimizer.pt`、`scheduler.pt`、`trainer_state.json`、`rng_state_0.pth` ... `rng_state_7.pth`
- 读取状态显示：`trainer_global_step=6`，`scheduler_last_epoch=6`，`optimizer_exists=True`

sample-id 级别数据顺序检查：

- 连续数据流：`debug_resume_data_seq_full6_20260716_164256`，记录 rank 0-7 的 step 0-5。
- 断点后半段数据流：`debug_resume_data_seq_resume3_20260716_164256`，记录 rank 0-7 的 step 3-5。
- 比较范围：8 ranks x 3 steps x 10 samples/rank/step。
- 比较字段：`sample_id = subset_name:data_idx`。
- 结果：`full_records=48`，`resume_records=24`，`missing=[]`，`diff_count=0`。
- 结论：在当前正式数据构造参数下，连续跑 6 step 的后半段与跳过前 3 step 后继续得到的 step 3-5 sample_id 序列完全一致。Trainer resume 日志中的 data skip 与实际样本序列相匹配。

评估 smoke：

- checkpoint：phase2 `checkpoint-6`
- eval：`EVAL_MODE=smoke`，`ONLY_EVAL_KEYWORDS=MMEB-eval-VisDial-beir`
- scorer：`MAXSIM_INTERACTION=q2d_query_topk`，`MAXSIM_QUERY_AGG=mean`，`MAXSIM_QUERY_TOPK=48`
- 输出：`eval_smoke/mmeb_visdial_q2d_topk48/mmeb_full.json` 和 `mmeb_full_summary.json`
- 结论：续训 checkpoint 可被 MMEB eval 正常加载并完成落盘。smoke 只使用 2 queries / 8 local docs，不作为性能结果。

## 2026-07-19 最终稳定性结论（替代此前 local_slice/cache 推断）

本轮使用当前正式配置做了 exact-data-step 定位：8 卡、每卡 BSZ=8、`[1,2,4] = 7` crops、每 crop 最多 1024 visual tokens、budget `128/128/128`、adaptive TopK48、lambda 0.8、完整 differentiable gather 和 hard negative。没有截断、缩图、跳样本或修改 MaxSim/negative 语义。

根因不是单一坏样本、collate、cache 或 MaxSim 公式，而是以下组合：

1. PEFT 在 gradient checkpointing 启用前完成包装，之后的 hotpatched Trainer 只调用 `gradient_checkpointing_enable()`，没有为自定义 ColQwen 文本/视觉入口启用 input gradients。reentrant 模式会因此静默漏掉 192 个视觉 LoRA parameter tensors 的梯度。
2. Transformers 4.55 的 Qwen2.5-VL vision tower 虽设置 `gradient_checkpointing` 标志，但其 forward 直接遍历 32 个 vision blocks，未实际调用 activation checkpoint。
3. data57 的 rank7 doc 分支打包约 167k raw visual rows。视觉 LoRA 梯度修复后，整批 backward 峰值约 80.6GB 并停滞；其他 rank 随后等待同步，所以表面上像 NCCL/DDP timeout。
4. `local_slice` 可以绕开部分分布式反向，但会丢掉其他 rank loss 对本 rank document embedding 的梯度，不是合法正式修复。

正式默认修复：

- 官方 `facebookresearch/MetaEmbed` commit `5abd6b4` 已做运行时 PEFT 审计；当前目标集合与官方宽正则一致。正式 launcher 在首个 batch 前强制验证 `504 language LoRA + 192 visual LoRA + 2 custom_text_proj + 66 folder_homo = 764 tensors / 74,967,174 parameters`。
- PEFT 使用 reentrant gradient checkpointing，并在 ColQwen 实际使用的 `_embed_tokens` 与 vision `patch_embed` 边界启用 input gradients。
- 对 32 个 vision blocks 启用 PyTorch activation checkpoint；不改变 block、权重或 state_dict。
- 编码器按样本做最多 60k raw visual rows 的显存微批，随后拼回原 BSZ=8 embeddings，再计算原始全局 gather、hard negatives 和 MaxSim loss。全局 batch 与负样本集合不变。
- DDP backward 在 `no_sync()` 内完成，所有 rank 到达 CPU barrier 后按确定性参数桶平均梯度，避免 DDP reducer 与 differentiable gather 的 collective 交错。

验证结果：

| 门禁 | 结果 |
|---|---|
| exact data12 单步 | 8/8 rank 完成，764/764 trainable parameter tensors 有梯度，无 warning/OOM/timeout |
| exact data57 单步 | 8/8 rank 完成，rank7 峰值约 76.2GB；loss 15.0999，grad norm 94.6245 |
| Q2D exact data57 单步 | 8/8 rank 完成，764/764 trainable parameter tensors 有梯度；loss 15.9534，grad norm 91.0405 |
| MetaEmbed 对象门禁 + adaptive exact data57 | 启动对象计数完全匹配 504/192/2/66；8/8 rank 完成，loss 15.0999，grad norm 94.5738 |
| 连续 data11-13 | 3/3 optimizer steps 完成；loss 15.0375 / 12.1474 / 11.7188 |
| 连续 data55-58 | 4/4 optimizer steps 完成；关键 data57 loss 11.9117、grad norm 71.0849，data58 随后完成 |

2026-07-19 深度打点门禁进一步验证了训练中的静默错误风险：

- adaptive exact data57 与 Q2D exact data57 均通过 8 卡审计；每个 rank 都完成 embedding/loss finite 检查、764/764 梯度检查、同步后梯度指纹检查和 optimizer 参数差分检查。
- adaptive 连续 data55-58 共完成 `8 ranks x 4 steps` 的梯度和参数更新门禁；每步跨 rank 梯度指纹最大差异均为 `0.0`。
- 第 1 步只有 LoRA-B 产生非零更新，因此 language/visual 分别更新 252/96 个张量；第 2 步起 LoRA-A/B 全部更新，达到 504/192。这符合 LoRA 的零初始化，不是漏梯度。
- `custom_text_proj` 的 2/2 张量与 `folder_homo` 的 66/66 张量每步都有非零梯度和实际参数变化。
- 连续窗口每步结束后的 CUDA allocated memory 为 `8.03GB -> 8.61GB -> 8.61GB -> 8.61GB`；第一次 optimizer state 分配后保持稳定，未观察到逐步显存泄漏。
- 连续窗口最重步为 data57，rank7 峰值 `74.83GB`。该步 doc/negative encoder 均拆成 3 个完整样本组；gather 约 4ms、CPU barrier 约 3.6ms、固定 bucket 梯度同步约 38ms，长耗时仍来自视觉 forward/backward，而不是通信 collective。

可重复入口：

```bash
# exact data57
PROBE_DATA_START_STEP=57 PROBE_STEPS=1 INTERACTION_STRATEGY=adaptive \
  experiments/main_model/run_deep_audit.sh

# 连续 data55-58
PROBE_DATA_START_STEP=55 PROBE_STEPS=4 INTERACTION_STRATEGY=adaptive \
  experiments/main_model/run_deep_audit.sh
```

深度审计通过 `MURE_DEEP_AUDIT=1` 显式开启，正常正式训练默认关闭，以免 CPU 参数快照拖慢训练。审计失败会直接抛错，不会由 Trainer 的 NaN 日志过滤继续运行。

对应远端 run：

- `runs/rhc_mmeb_vidore_bi_topk48_adaptive_mean_from_base_finalfix_microbatch60k_probe_data12_adaptive_bsz8_b128_20260719`
- `runs/rhc_mmeb_vidore_bi_topk48_adaptive_mean_from_base_finalfix_microbatch60k_probe_data57_adaptive_bsz8_b128_20260719`
- `runs/rhc_mmeb_vidore_q2d_topk48_mean_from_base_finalgate_default_data57_q2d_bsz8_b128_20260719`
- `runs/rhc_mmeb_vidore_bi_topk48_adaptive_mean_from_base_finalgate_default_data11_13_adaptive_bsz8_b128_20260719`
- `runs/rhc_mmeb_vidore_bi_topk48_adaptive_mean_from_base_finalgate_default_data55_58_adaptive_bsz8_b128_20260719`

编码器微批会改变 BF16/FlashAttention 的 kernel reduction grouping，因此与未拆分 forward 不保证 bitwise identical；它不改变样本、视觉 token、crop、embedding 集合或 loss 定义。不能把打补丁前的 checkpoint 与新 run 当作严格相同数值轨迹混用。

### 历史记录：2026-07-16 DDP / gather 稳定性排查（已被 2026-07-19 结论替代）

此前正式训练在 984 step 附近出现过 NCCL collective timeout。当时曾把 `local_slice` 作为候选修复；2026-07-19 已确认它会改变跨 rank 梯度，只能用于诊断，不能用于正式训练。当时脚本传入：

- `MURE_GATHER_WITH_GRAD_MODE=local_slice`
- `DDP_FIND_UNUSED_PARAMETERS=1`
- `TRAIN_BSZ=10`
- `INTERLEAVED_BSZ=10`
- `QUERY_CHUNK_SIZE=64`
- `DOC_CHUNK_SIZE=128`

为确认不是数据样本、adaptive MaxSim 或 DDP reducer 本身的问题，补充了三类 smoke：

| 检查 | 配置 | 结果 | 结论 |
|---|---|---|---|
| 正式 Trainer DDP from-base smoke | `RUNS=adaptive`, `MAX_STEPS=3`, `SAVE_STEPS=3`, `bi_query_topk_adaptive`, `TopK=48`, `lambda=0.8` | 完成 3/3 step，保存 `checkpoint-3`，`train_runtime=163.634s` | 主训练链路、全量 26 子集配置、DDP、local-slice gather、checkpoint 保存均可闭环 |
| 单点 DDP replay | `START_STEP=984`, `END_STEP=984`, `MODEL_REPLAY=1`, `BACKWARD=1`, `DDP_WRAP=1` | 8 个 rank 均完成 `query_forward -> doc_forward -> gather -> neg_doc_forward -> loss -> backward -> zero_grad` | 984 batch 本身不是坏样本，adaptive TopK48 在真实 DDP backward 下可通过 |
| 邻域 DDP replay | `START_STEP=980`, `END_STEP=986`, `MODEL_REPLAY=1`, `BACKWARD=1`, `DDP_WRAP=1` | 980-986 全部 step、全部 rank 完成 backward/zero_grad | 984 附近窗口稳定，未复现 DDP reducer 死锁 |

对应远端记录：

- `runs/full_mmeb_vidore_bi_topk48_adaptive_mean_from_base_adaptive_ddp_smoke_frombase_3step_20260716_112737`
- `runs/debug_984_adaptive_ddpwrap_20260716_113812`
- `runs/debug_980_986_adaptive_ddpwrap_20260716_114459`

该阶段只排除了单一坏样本和 adaptive TopK MaxSim 公式，未覆盖 PEFT input-gradient、视觉 LoRA 和视觉 backward 峰值问题。

### 历史记录：2026-07-17 checkpoint-950 复盘与训练门禁

> 本节和 `experiments/main_model/run_*984*.sh` 针对已删除的旧 `BSZ=10/SAVE_STEPS=950/BUDGETS=160` run，不能用于当前 `8/1000/128` 主跑。

7 月 16 日的两条正式训练均保存 `checkpoint-950`，Q2D 随后在显示 984 后进入 DDP reducer timeout，adaptive 在同一进度附近因实例终止而停止。此前 raw DDP replay 不能替代正式 TrainerV2 续训，原因包括未加载 checkpoint-950 的模型/优化器/RNG 状态，以及 replay 中 gradient-checkpointing 路径出现输入不需要梯度的警告。

重新审计后的约束如下：

- `[1, 2, 4]` 表示 `1 + 2 + 4 = 7` crops：`1x1`、自适应 `1x2|2x1`、`2x2`。
- 正式训练使用 `ContrastiveTrainerV2`，诊断埋点必须加在 V2 路径。
- `torch` differentiable gather 保留完整跨 rank 文档梯度，恢复为正式默认值。
- `local_slice` 会丢弃其他 rank loss 对本 rank document embedding 的梯度，只保留为显式 A/B 诊断模式。
- 正式长训前必须从真实 `checkpoint-950` 续训到 990，并保留 `MAX_STEPS=60000`，避免改变 linear scheduler。
- 978-990 的 TrainerV2 阶段日志写入每个 run 的 `debug/contrastive/rank*.jsonl`。

门禁入口：

```bash
bash experiments/main_model/run_resume_984_gate.sh
```

只有生成完整 `checkpoint-990` 且 8 个 rank 均记录 `training_step_done` 后，才允许继续正式长训。

更快的数据窗口探针：

```bash
bash experiments/main_model/run_probe_984_window.sh
```

该探针加载真实 checkpoint-950，但使用 `ignore_data_skip` 并将每个 rank 的 iterable dataset 直接跳到 data step 982，仅执行 982-986 五个 batch。它用于快速隔离固定数据窗口与完整 TrainerV2 backward，通常只需数分钟；由于省略了 951-981 的参数更新，它不能替代上面的连续 950-990 门禁。

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

当时结论是 `TRAIN_BSZ=10` 吞吐更高，但后续长训稳定性排查后，当前正式配置已固定为 `TRAIN_BSZ=8`、`INTERLEAVED_BSZ=8`、`GRAD_ACCUM_STEPS=1`。

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
  MAX_STEPS=60000 \
  SAVE_STEPS=1000 \
  LEARNING_RATE=1e-4 \
  LR_SCHEDULER_TYPE=linear \
  TRAIN_BSZ=8 \
  INTERLEAVED_BSZ=8 \
  BUDGETS="128 128 128" \
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
RUNS=q2d MAX_STEPS=2 SAVE_STEPS=1 TRAIN_BSZ=8 INTERLEAVED_BSZ=8 \
  GRAD_ACCUM_STEPS=1 QUERY_CHUNK_SIZE=64 DOC_CHUNK_SIZE=128 \
  BUDGETS="128 128 128" RUN_SUFFIX=probe_bsz8_acc1_q64_prefix SKIP_EVAL=1 \
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

当时判断（已被 2026-07-19 的 backward/gradient 证据替代）：

1. `984` 不是由某一个单独样本造成的“无限卡死”。在 `980-986` 精查中，984 附近 collate 最慢约 9 秒，属于长样本慢 batch，但不是分钟级卡死。
2. 这段确实有大量接近 `5.3k-5.4k` input length / image tokens 的样本，集中在 `N24News`、`Visual7W`、`SUN397`、`InfographicsVQA`、`visrag_ind` 等子集。
3. 两次抽样顺序完全一致，说明如果后续训练再次卡在相同步数，可以稳定复现并定位同一批样本。
4. 更可疑的问题是 cache 路径不统一和大 cache 反复构建/读取。后续正式训练应先清理或至少避开项目目录 `.cache`，并统一使用 `/MURE-V2/env/hf_datasets_cache`。

### 后续执行建议

不要为了越过 `984` 直接改变训练目标或关闭 gather。下一步应保持正式训练配置不变，只做：

1. 使用共享 cache 路径重启正式训练。
2. 从 `0` 或已确认可靠的 checkpoint 重新跑，观察是否仍在 `984` 附近卡住。
3. 若仍卡住，优先检查 cache/IO/dataset worker 状态；其次再考虑对 `N24News`、`Visual7W`、`SUN397`、`InfographicsVQA`、`visrag_ind` 的极长样本做专项统计或上限裁剪。
