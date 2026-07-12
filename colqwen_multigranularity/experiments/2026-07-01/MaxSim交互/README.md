# MaxSim 交互机制实验

Updated: 2026-07-07

## 当前定位

这一部分只研究 **MaxSim 交互机制本身**，不再混入任务课程学习、非对称 token 预算压缩、retention 或 full-MMEB 大评测。

2026-07-06 更新：这里需要严格区分两类实验。**免训练 scorer-only 验证**使用已经训练好的 `FolderHomo sym160 checkpoint-4000`，只改打分函数；**可训练 MaxSim 机制实验**必须从原生 `colqwen2.5-base` 开始训练，不 warm-start `sym160`，否则会把 continued tuning 漂移和 MaxSim loss 本身混在一起。

后续所有默认测试集固定为：

1. **ViDoRe v2**：报告 `nDCG@5`，用于确认视觉文档主线不能被破坏。
2. **MMEB worst10**：报告 `P@1`，实现里对应 `recall_at_1`，用于专门解决 MMEB 上最差的 10 个任务。

固定模型默认使用已经训练好的 `FolderHomo sym160`：

```text
Checkpoint=../MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000
Budget=160/160/160
Training=no retrain, scorer-only eval first
```

## 研究问题

原始 ColBERT/ColQwen 的 MaxSim 是 directed scoring：

```text
score(Q, D) = sum_i max_j q_i dot d_j
```

它只要求 query token 被 target 覆盖，不要求 target token 也被 query 覆盖。这个机制适合“短文本 query 查长文档”，但在 MMEB 的 image/text/image+text 混合任务里会出现两个问题：

1. **query 端噪声过多**：图像 query 或图文 query 里很多 token 不该参与最终匹配。
2. **target coverage 缺失**：只匹配 query 端局部证据，容易让 VQA、Chart、InfoGraphics 等任务被局部高分误导。

因此当前只看两条可落地路线：

| 路线 | 思路 | 代表 scorer |
| --- | --- | --- |
| 保留 directed bias | 仍然 query->target，但只保留 query 端最有用的 topK token 参与打分。 | `q2d_query_topkK` |
| 补 target coverage | 在 query->target 外加入 target->query，检查 reciprocal matching 是否能救 VQA/Chart 类任务。 | `bi_mean_lamXX`, `bi_query_topkK_lamXX` |

## 默认测试集合

### ViDoRe v2

配置文件：

```text
configs/eval/test_data_mast_v2.yaml
```

包含：

```text
esg_reports_human_labeled_v2
esg_reports_v2_multilingual
esg_reports_v2
biomedical_lectures_v2
biomedical_lectures_v2_multilingual
economics_reports_v2
economics_reports_v2_multilingual
```

### MMEB worst10

固定 10 个困难数据集：

```text
FashionIQ
CIRR
Country211
GQA
ScienceQA
InfographicsVQA
A-OKVQA
Visual7W
OK-VQA
ChartQA
```

## Scorer-only Cluster C 结果记录

更新时间：2026-07-07。

这一组是 **免训练 scorer-only 验证**，使用已经训练好的 `FolderHomo sym160 checkpoint-4000`，只改 MaxSim 聚合方式：

```text
Checkpoint = experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000
Budget = 160/160/160
Eval = MMEB worst10 P@1 + ViDoRe v2 nDCG@5
BATCH_QUERY = 32
BATCH_PASSAGE = 32
BATCH_SCORE = 128
```

运行命令：

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity

RUN_LOG=experiments/2026-07-01/MaxSim交互/runs/scorer_only_missing3_shardC_$(date +%Y%m%d_%H%M%S).log
setsid env SCORERS="bi_topk_sum32_lam05 bi_topk_sum32_lam07 bi_topk_sum64_lam05 bi_topk_sum64_lam07" \
  CHECKPOINT=experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000 \
  BATCH_QUERY=32 BATCH_PASSAGE=32 BATCH_SCORE=128 NUM_WORKERS=0 \
  experiments/2026-07-01/MaxSim交互/run_eval_vidorev2_worst10.sh > "$RUN_LOG" 2>&1 < /dev/null &
```

完成日志：

```text
experiments/2026-07-01/MaxSim交互/runs/scorer_only_missing3_shardC_20260707_014451.log
```

结果汇总：

| Scorer | MMEB worst10 avg P@1 | ViDoRe v2 avg nDCG@5 | 结论 |
| --- | ---: | ---: | --- |
| `bi_topk_sum32_lam05` | 0.0996 | 0.4805 | lam=0.5 下 sum 聚合明显不稳，MMEB 和 ViDoRe 都较弱。 |
| `bi_topk_sum32_lam07` | **0.2016** | 0.5152 | Cluster C 中 MMEB 最好，但仍弱于 `q2d_query_topk32` / `q2d_topk_sum32`。 |
| `bi_topk_sum64_lam05` | 0.1294 | 0.4781 | 增大 topK 到 64 不能弥补 lam=0.5 的损伤。 |
| `bi_topk_sum64_lam07` | 0.1800 | **0.5262** | Cluster C 中 ViDoRe 最好，但 MMEB 不如 topK32+lam07。 |

10 个困难集细分结果：

| Dataset | `bi_topk_sum32_lam05` | `bi_topk_sum32_lam07` | `bi_topk_sum64_lam05` | `bi_topk_sum64_lam07` |
| --- | ---: | ---: | ---: | ---: |
| FashionIQ | 0.0220 | 0.0220 | 0.0220 | 0.0210 |
| CIRR | 0.0850 | 0.0870 | 0.0830 | 0.0880 |
| Country211 | 0.0890 | 0.1400 | 0.1070 | 0.1140 |
| GQA | 0.0800 | 0.1510 | 0.1110 | 0.1500 |
| ScienceQA | 0.0530 | 0.1850 | 0.1330 | 0.2500 |
| InfographicsVQA | 0.0340 | 0.2560 | 0.0980 | 0.2530 |
| A-OKVQA | 0.2700 | 0.3090 | 0.2220 | 0.2230 |
| Visual7W | 0.0620 | 0.2450 | 0.1350 | 0.1740 |
| OK-VQA | 0.2430 | 0.3330 | 0.2470 | 0.2540 |
| ChartQA | 0.0580 | 0.2880 | 0.1360 | 0.2730 |
| **Average** | 0.0996 | **0.2016** | 0.1294 | 0.1800 |
| **ViDoRe v2 avg nDCG@5** | 0.4805 | 0.5152 | 0.4781 | **0.5262** |

阶段性判断：

- `lam07` 明显优于 `lam05`，说明这类双向 sum 聚合不能过度强化 target->query，否则会同时伤害 MMEB 和 ViDoRe。
- `topK32` 更偏向 MMEB，`topK64` 更偏向 ViDoRe retention；但二者都没有超过此前最强的 query-side topK 路线。
- 对于 scorer-only 设置，`bi_topk_sum*` 不应作为下一阶段主线；更值得保留的是 `q2d_query_topk32/48` 或训练版 `q2d_topk48_mean`。

### Cluster C: `bi_topk_mean64` lambda sweep

更新时间：2026-07-07。

这一组继续使用已经训练完成的 `sym160` checkpoint 做 scorer-only 评估，不重新训练模型，目标是检查 **mean 聚合下的 TopK64 弱双向项** 是否比前面的 sum 聚合更稳。

运行信息：

```bash
SCORERS="bi_topk_mean64_lam05 bi_topk_mean64_lam07 bi_topk_mean64_lam09"
CHECKPOINT=experiments/2026-07-01/MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000
BATCH_QUERY=32 BATCH_PASSAGE=32 BATCH_SCORE=128 NUM_WORKERS=0
experiments/2026-07-01/MaxSim交互/run_eval_vidorev2_worst10.sh
```

日志文件：

```text
experiments/2026-07-01/MaxSim交互/runs/scorer_only_clusterC_mean64_20260707_102549.log
```

汇总结果：

| Scorer | MMEB worst10 avg P@1 | ViDoRe v2 avg nDCG@5 | 结论 |
| --- | ---: | ---: | --- |
| `bi_topk_mean64_lam05` | **0.2438** | 0.5076 | MMEB 最好，说明更强 target coverage 对 worst10 有帮助，但 ViDoRe retention 较弱。 |
| `bi_topk_mean64_lam07` | 0.2345 | 0.5430 | 折中配置：MMEB 略降，ViDoRe 明显恢复。 |
| `bi_topk_mean64_lam09` | 0.2146 | **0.5519** | 最接近单向 q2d，ViDoRe 最好，但 MMEB worst10 收益继续下降。 |

10 个困难集细分结果：

| Dataset | `bi_topk_mean64_lam05` | `bi_topk_mean64_lam07` | `bi_topk_mean64_lam09` |
| --- | ---: | ---: | ---: |
| FashionIQ | 0.0220 | 0.0210 | 0.0230 |
| CIRR | 0.0830 | 0.0880 | 0.0880 |
| Country211 | 0.1410 | 0.1360 | 0.1230 |
| GQA | 0.1680 | 0.1660 | 0.1550 |
| ScienceQA | 0.2670 | 0.2800 | 0.2860 |
| InfographicsVQA | 0.3770 | 0.3760 | 0.3570 |
| A-OKVQA | 0.2980 | 0.2770 | 0.2240 |
| Visual7W | 0.3290 | 0.2860 | 0.2340 |
| OK-VQA | 0.3250 | 0.3050 | 0.2690 |
| ChartQA | 0.4280 | 0.4100 | 0.3870 |
| **Average** | **0.2438** | 0.2345 | 0.2146 |
| **ViDoRe v2 avg nDCG@5** | 0.5076 | 0.5430 | **0.5519** |

阶段性判断：

- 与 `bi_topk_sum64` 相比，`bi_topk_mean64` 明显更稳，尤其在 VQA 类困难集上更接近有效 scorer-only 路线。
- lambda 越靠近 q2d，ViDoRe retention 越好；lambda 越接近双向均衡，MMEB worst10 越好。这说明当前冲突仍然是 MMEB hard/core4 与 ViDoRe text-to-page 之间的 scoring 偏好冲突，而不是单纯 topK 大小问题。
- 如果继续做 scorer-only sweep，`bi_topk_mean64_lam07` 可以作为折中基线；如果目标是 MMEB worst10，优先看 `lam05`；如果目标是保住 ViDoRe，优先看 `lam09`。

## 可训练 MaxSim 结果记录

更新时间：2026-07-06。

这一组是 **从 base 训练** 的 MaxSim 机制实验，不使用 `sym160` warm start：

```text
Base model = models/colqwen2.5-base
Model = FolderHomo
Budget = 160/160/160
Train data = ViDoRe train + MMEB hard/core4
Steps = 1000
LR = 2e-4
Scheduler = constant
Warmup = 0
Eval = MMEB worst10 P@1 + ViDoRe v2 nDCG@5
```

已形成正式 `checkpoint-1000` 并完成评估的模型如下：

| Method | MMEB worst10 avg P@1 | ViDoRe v2 avg nDCG@5 | 结论 |
| --- | ---: | ---: | --- |
| `q2d_mean` | 0.1380 | 0.5649 | 单向 mean 训练没有解决 MMEB worst10。 |
| `q2d_topk48_mean` | **0.3852** | **0.5635** | 当前最均衡：MMEB 大幅提升，ViDoRe 基本不掉。 |
| `bi_mean_lam07` | 0.2271 | 0.5449 | 双向本身有收益，但不如 TopK48。 |
| `bi_adaptive_lam08` | 0.3347 | 0.5144 | 自适应双向能提升 MMEB，但 ViDoRe 损伤明显。 |
| `bi_topk48_mean_lam07` | 0.3769 | 0.5375 | 强结果，但整体略弱于单向 TopK48。 |
| `bi_topk_mean48_hard_adaptive` | 0.3487 | 0.5417 | 训练版 hard adaptive 明显强于免训练 hard routing，但仍弱于 `q2d_topk48_mean` 和固定双向 TopK48。 |

10 个困难集细分结果：

| Dataset | `q2d_mean` | `q2d_topk48_mean` | `bi_mean_lam07` | `bi_adaptive_lam08` | `bi_topk48_mean_lam07` | `bi_topk_mean48_hard_adaptive` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FashionIQ | 0.0310 | 0.0790 | 0.0240 | 0.0310 | 0.0480 | 0.0190 |
| CIRR | 0.1220 | 0.2150 | 0.1020 | 0.1160 | 0.1920 | 0.1180 |
| Country211 | 0.0300 | 0.2240 | 0.1270 | 0.1650 | 0.1390 | 0.1060 |
| GQA | 0.1620 | 0.4010 | 0.2260 | **0.5260** | 0.3880 | 0.4250 |
| ScienceQA | 0.1700 | 0.3140 | 0.2740 | 0.3310 | **0.3360** | 0.2960 |
| InfographicsVQA | 0.0690 | 0.5840 | 0.3240 | 0.4470 | **0.6110** | 0.5980 |
| A-OKVQA | 0.2090 | **0.5010** | 0.2760 | 0.4140 | 0.4870 | 0.4570 |
| Visual7W | 0.1720 | 0.4590 | 0.2540 | 0.4090 | **0.4770** | 0.4620 |
| OK-VQA | 0.2240 | 0.5320 | 0.3070 | 0.4590 | **0.5520** | 0.5160 |
| ChartQA | 0.1910 | **0.5430** | 0.3570 | 0.4490 | 0.5390 | 0.4900 |
| **Average** | 0.1380 | **0.3852** | 0.2271 | 0.3347 | 0.3769 | 0.3487 |
| **ViDoRe v2 avg nDCG@5** | **0.5649** | 0.5635 | 0.5449 | 0.5144 | 0.5375 | 0.5417 |

补充说明：

- `q2d_topk48_mean` 是当前最适合作为下一阶段主线的训练目标。它说明 MMEB 的主要问题更像是 **query 侧视觉/混合 token 噪声过多**，而不是单纯需要完全双向化。
- `bi_topk48_mean_lam07` 在 VQA/Chart 类任务上也很强，说明 target coverage 有价值，但双向项会带来 ViDoRe 下降。
- `bi_topk_mean48_hard_adaptive` 训练版把免训练 hard routing 的 MMEB worst10 从 `0.1820` 提到 `0.3487`，说明 hard routing 是可学习的；但它没有超过固定 `bi_topk48_mean_lam07` 和单向 `q2d_topk48_mean`。
- `bi_query_topk48_adaptive_s1k_from_base` 没有正式 `checkpoint-1000`。历史日志最多到约 822/1000 step，loss 约 1.1184，但没有可评估 checkpoint，因此不纳入正式结果表。

## 主脚本

免训练统一评测入口：

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/2026-07-01/MaxSim交互

bash ./run_eval_vidorev2_worst10.sh \
  ../MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000
```

默认会顺序跑：

```text
legacy_q2d_sum
q2d_mean
q2d_query_topk16
q2d_query_topk32
q2d_query_topk48
q2d_query_topk64
q2d_query_topk96
q2d_query_topk128
bi_mean_lam05
bi_mean_lam07
bi_mean_lam09
bi_query_topk32_lam05
bi_query_topk32_lam07
bi_query_topk64_lam05
bi_query_topk64_lam07
```

三台 8 卡机器分片跑时使用：

```bash
# Machine 1
SCORER_GROUP=1 bash ./run_eval_vidorev2_worst10.sh /path/to/checkpoint-4000

# Machine 2
SCORER_GROUP=2 bash ./run_eval_vidorev2_worst10.sh /path/to/checkpoint-4000

# Machine 3
SCORER_GROUP=3 bash ./run_eval_vidorev2_worst10.sh /path/to/checkpoint-4000
```

当前三组划分：

| Group | Methods |
| --- | --- |
| 1 | `legacy_q2d_sum`, `q2d_mean`, `q2d_query_topk16/32/48/64` |
| 2 | `q2d_query_topk96/128`, `bi_mean_lam05/07/09`, `bi_adaptive_lam08` |
| 3 | `bi_query_topk32_lam05/07`, `bi_query_topk64_lam05/07`, `bi_query_topk32_adaptive_lam08`, `bi_query_topk64_adaptive_lam08` |

`adaptive` 双向的定义是：仍然计算 `q2d` 和 `d2q`，但双向权重不固定，而是根据 query/doc 的有效 token 长度自动调节：

```text
lambda = clamp(doc_len / (query_len + doc_len), min=0.5, max=0.8)
score = lambda * q2d + (1 - lambda) * d2q
```

直觉是：当 query 明显短于 target 时保留更多 directed q2d；当两侧长度接近时回到更对称的 0.5/0.5。

可通过 `SCORERS` 只跑指定方法：

```bash
SCORERS="q2d_query_topk32 q2d_query_topk48 q2d_query_topk64" \
bash ./run_eval_vidorev2_worst10.sh /path/to/checkpoint-4000
```

输出：

```text
eval/maxsim_vidorev2_worst10/
  mmeb_worst10/<scorer>/mmeb_full.json
  mmeb_worst10/<scorer>/mmeb_full_summary.json
  mmeb_worst10_summary.md
  vidore_v2/<scorer>/vidore_v2.json
  vidore_v2_summary.md
  maxsim_12row_table.md
```

默认 batch：

```text
BATCH_QUERY=32
BATCH_PASSAGE=32
BATCH_SCORE=128
```

如果显存不足，优先把 `BATCH_SCORE` 降到 64。

## 方法说明

正式机制空间按三个正交维度组织：

```text
Direction: q2d / bidirectional
Aggregation: sum / mean
TopK: off / on
```

因此核心 scorer-only 消融是 2 x 2 x 2，共 8 类。`adaptive`、`LSE`、`global` 属于额外 P1/P2 变体，不放进基础 8 类。

| Family | Direction | Aggregation | TopK | Scorer examples | 解释 |
| --- | --- | --- | --- | --- | --- |
| q2d-sum | 单向 | Sum | 否 | `legacy_q2d_sum` | 原始 ColBERT/ColQwen MaxSim：每个 query token 对 target 取 max，最后对 query token 求和。 |
| q2d-mean | 单向 | Mean | 否 | `q2d_mean` | 单向 MaxSim，但最后对 query token 求均值，消除 query 长度尺度影响。 |
| q2d-topK-sum | 单向 | Sum | 是 | `q2d_topk_sum48` | 先得到每个 query token 的 MaxSim 分数，只保留 topK，再求和。 |
| q2d-topK-mean | 单向 | Mean | 是 | `q2d_topk_mean48`, legacy alias `q2d_query_topk48` | 先得到每个 query token 的 MaxSim 分数，只保留 topK，再求均值。当前历史 topK 结果默认属于这一类。 |
| bi-sum | 双向 | Sum | 否 | `bi_sum_lam07` | 同时算 q2d-sum 和 d2q-sum，再按 lambda 加权。 |
| bi-mean | 双向 | Mean | 否 | `bi_mean_lam07` | 同时算 q2d-mean 和 d2q-mean，再按 lambda 加权。 |
| bi-topK-sum | 双向 | Sum | 是 | `bi_topk_sum48_lam07` | q2d 和 d2q 两个方向都只保留 topK token 分数，再求和并加权。 |
| bi-topK-mean | 双向 | Mean | 是 | `bi_topk_mean48_lam07`, legacy alias `bi_query_topk48_lam07` | q2d 和 d2q 两个方向都只保留 topK token 分数，再求均值并加权。 |

统一评测 checkpoint：

```text
FolderHomo sym160 = ../MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000
Budget = 160/160/160
Eval = MMEB worst10 P@1 + ViDoRe v2 nDCG@5
```

推荐先跑基础 8 类：

```bash
SCORER_GROUP=base8 bash ./run_eval_vidorev2_worst10.sh \
  ../MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000
```

TopK sweep：

```bash
SCORER_GROUP=topk_sweep bash ./run_eval_vidorev2_worst10.sh /path/to/checkpoint-4000
SCORER_GROUP=bi_topk_sweep bash ./run_eval_vidorev2_worst10.sh /path/to/checkpoint-4000
```

## 免训练验证全景计划

所有免训练实验都固定：

```text
Checkpoint=FolderHomo sym160 checkpoint-4000
Train=no
Eval=MMEB worst10 P@1 + ViDoRe v2 nDCG@5
```

### P0: 必须完成的正交消融

P0 目标是把 MaxSim 机制拆成 `Direction x Aggregation x TopK`，先完成最小但完备的机制矩阵。

注意：当前历史结果里的 `q2d_query_topkK` 和 `bi_query_topkK` 在代码中都是 **TopK 后取 mean**，因此它们对应表中的 `Mean + TopK`，不是 `原始/Sum + TopK`。

| ID | 方向 | 打分聚合 | TopK | Scorer | Status | MMEB P@1 / ViDoRe v2 |
| --- | --- | --- | --- | --- | --- | --- |
| P0-1 | 单向 | 原始 Sum | 否 | `legacy_q2d_sum` | DONE | 14.23 / 54.77 |
| P0-2 | 单向 | Mean | 否 | `q2d_mean` | DONE | 14.26 / 54.77 |
| P0-3 | 单向 | 原始 Sum | 是 | `q2d_topk_sum{16,32,48,64,96,128}` | TODO-RUN | TODO |
| P0-4 | 单向 | Mean | 是 | `q2d_topk_meanK`, legacy `q2d_query_topkK` | DONE | K16 22.15/37.65; K32 26.98/51.93; K48 23.44/54.74; K64 20.03/54.77; K96 15.71/54.77; K128 13.89/54.77 |
| P0-5 | 双向 | 原始 Sum | 否 | `bi_sum_lam{05,07,09}` | TODO-RUN | TODO |
| P0-6 | 双向 | Mean | 否 | `bi_mean_lam{05,07,09}` | DONE | lam05 20.57/49.12; lam07 18.44/55.16; lam09 15.83/55.41 |
| P0-7 | 双向 | 原始 Sum | 是 | `bi_topk_sum{32,64}_lam{05,07}` | TODO-RUN | TODO |
| P0-8 | 双向 | Mean | 是 | `bi_topk_meanK_lamXX`, legacy `bi_query_topkK_lamXX` | PARTIAL-DONE | 已有 worst10/retention；ViDoRe v2 需要统一补测 |

因此，为了补齐 8 格主表，当前只需要补 3 类实验：

1. `q2d_topk_sumK`：单向 / 原始 Sum / TopK。
2. `bi_sum_lamXX`：双向 / 原始 Sum / 无 TopK。
3. `bi_topk_sumK_lamXX`：双向 / 原始 Sum / TopK。

可直接运行：

```bash
SCORER_GROUP=base8 bash ./run_eval_vidorev2_worst10.sh /path/to/checkpoint-4000
SCORER_GROUP=missing3 bash ./run_eval_vidorev2_worst10.sh /path/to/checkpoint-4000
SCORER_GROUP=topk_sweep bash ./run_eval_vidorev2_worst10.sh /path/to/checkpoint-4000
SCORER_GROUP=bi_topk_sweep bash ./run_eval_vidorev2_worst10.sh /path/to/checkpoint-4000
```

### P1: 已有代码支持的补充机制

P1 不作为第一张主表，但如果 P0 结果还不够清楚，应补跑。

| ID | Scorer group / scorer | Status | 要回答的问题 |
| --- | --- | --- | --- |
| P1-1 | `bi_adaptive_lam08` | DONE | 根据 query/doc 长度自适应双向权重是否比固定 lambda 更稳。 |
| P1-2 | `bi_topk_mean{32,64}_adaptive_lam08` / legacy `bi_query_topkK_adaptive_lam08` | DONE for 32/64 | TopK + adaptive 是否超过固定 lambda。 |
| P1-3 | `bi_topk_mean48_adaptive_lam08` | DONE | 补齐当前关键 K=48 下的 soft length adaptive。 |
| P1-4 | `bi_topk_sum48_adaptive_lam08` | DONE | 检查 adaptive 是否应该建立在原始 Sum 聚合上。 |
| P1-5 | `bi_topk_mean48_hard_adaptive` | DONE | 长短差距大时走单向，长度接近时走 0.5/0.5 双向。 |
| P1-6 | `lse_beta20`, `bi_lse_beta20_lam05` | TODO | 用 LogSumExp 软化 hard max，判断 winner-take-all 是否是主要问题。 |
| P1-7 | `bi_topk_mean_k4_lam05`, `bi_topk_mean_k8_lam05` | TODO | 每个 token 不只取 top-1，而是取 top-k 平均，检查局部多匹配是否有帮助。 |
| P1-8 | `q2d_sum_lennorm_a025/a050/a075/a100` | TODO | 原始 sum 按 query 长度做连续归一化，检查 sum 与 mean 之间是否有更优折中。 |
| P1-9 | `q2d_mean_hitpen_w02/w05`, `q2d_topk_mean48_hitpen_w02/w05` | TODO | 惩罚少数 target token 被大量 query token 命中，检查 hit concentration 是否导致伪匹配。 |
| P1-10 | `q2d_query_topk64_global_w02` | TODO | 加 pooled global score，检查全局语义能否缓解 CIRR/FashionIQ。 |
| P1-11 | `bi_adaptive_sum_lam08` | TODO / NOT-RUN | 精确的“无 TopK + 自适应 soft + Sum 聚合”尚未跑；不要和 `bi_topk_sum48_adaptive_lam08` 混淆。 |
| P1-12 | `bi_adaptive_hard_lam08` | TODO / NOT-RUN | 精确的“无 TopK + 自适应 hard + Mean 聚合”尚未跑；不要和 `bi_topk_mean48_hard_adaptive` 混淆。 |

可直接运行：

```bash
SCORER_GROUP=adaptive3 bash ./run_eval_vidorev2_worst10.sh /path/to/checkpoint-4000
SCORER_GROUP=p1 bash ./run_eval_vidorev2_worst10.sh /path/to/checkpoint-4000
SCORER_GROUP=length_norm bash ./run_eval_vidorev2_worst10.sh /path/to/checkpoint-4000
SCORER_GROUP=hit_penalty bash ./run_eval_vidorev2_worst10.sh /path/to/checkpoint-4000
```

### Adaptive 机制单独规划

Adaptive 不应该只保留一个当前的 length-aware 版本。它本质上是在回答：当 query/target 的信息量和 token 长度不同，MaxSim 应该如何选择方向和权重。

当前先不急着确定 adaptive 的底层打分和 TopK 策略。应先跑完 P0 的 8 格主表，确定 `Sum/Mean`、`TopK/no-TopK` 哪个底座最稳，再把 adaptive 套到最有希望的 1-2 个底座上。

| Adaptive 类型 | Status | 机制 | 适合回答的问题 |
| --- | --- | --- | --- |
| Soft lambda by length | READY / current `bi_adaptive_lam08` | 一定计算双向，然后根据有效 token 长度自动调 lambda。 | 是否能用连续权重在 ViDoRe 稳定性和 MMEB 修复之间折中。 |
| Soft lambda by length + Sum | NOT-RUN / exact `bi_adaptive_sum_lam08` | 无 TopK，底层聚合从 mean 改为 sum，再按长度自适应调 lambda。 | 判断 adaptive 的收益是否依赖 mean 归一化；当前精确配置尚未跑。 |
| Soft lambda by length + Sum TopK | READY / `bi_topk_sum48_adaptive_lam08` | 在 TopK48 后用 Sum 聚合，再按长度自适应调 lambda。 | 如果 P0 证明 Sum+TopK 更好，adaptive 是否也应该用 Sum 底座。 |
| Hard length routing | READY / `bi_topk_mean48_hard_adaptive` | 根据 query/doc 有效 token 长度硬选择 `q2d`、`d2q` 或 `bi`。长短差距超过 1.5 倍时走单向，长度接近时走双向。 | 是否应该显式保留不同任务方向的非对称归纳偏置。 |
| Hard length routing without TopK | NOT-RUN / exact `bi_adaptive_hard_lam08` | 无 TopK，Mean 聚合，按长度硬路由选择单向或双向。 | 检查 hard adaptive 本身是否有效；当前精确配置尚未跑。 |
| Hard modality routing | TODO | 根据 query/target 是否含图像，硬选择 `q2d`、`d2q` 或 `bi`。 | 是否“有图像”比“token 长度”更能解释 MaxSim 非对称失效。 |
| Soft lambda by modality | TODO | 一定计算双向，但 lambda 由 query/target 是否有 image/text 决定，而不只是长度。 | 是否“有图像”比“token 长度”更能解释 MaxSim 非对称失效。 |
| Soft lambda by score confidence | TODO | 根据 q2d/d2q 分数差、hit concentration 或 score entropy 动态调 lambda。 | 是否能在局部伪匹配强时自动增加反向 coverage。 |
| Adaptive TopK | TODO | K 根据 query/token 长度或图像存在动态选择，如 text query 不裁、image query 用 K=32/48。 | 是否能让 TopK 只修复图像 query 噪声，而不伤文本 query。 |

### Adaptive 已跑结果汇总

更新时间：2026-07-07。

这里按实际跑过的 scorer 名称记录，不强行映射到临时命名。注意：当前已跑的是 `bi_adaptive_lam08` 以及若干 `bi_topk_*_adaptive` 变体；精确的无 TopK `bi_adaptive_sum_lam08` 和 `bi_adaptive_hard_lam08` 仍未跑。

| Scorer | Adaptive 类型 | 聚合 | TopK | MMEB worst10 avg P@1 | ViDoRe v2 avg nDCG@5 | 结论 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `bi_adaptive_lam08` | soft length adaptive | Mean | - | 0.2057 | 0.5576 | ViDoRe 最稳，但 MMEB 修复强度有限。 |
| `bi_topk_mean32_adaptive_lam08` / legacy `bi_query_topk32_adaptive_lam08` | soft length adaptive | Mean | 32 | 0.2715 | 0.6372 | MMEB 明显提升，但不超过固定 `bi_query_topk32_lam07`。 |
| `bi_topk_mean64_adaptive_lam08` / legacy `bi_query_topk64_adaptive_lam08` | soft length adaptive | Mean | 64 | 0.2436 | 0.6692 | retention 强，但 MMEB 不如 topK32。 |
| `bi_topk_mean48_adaptive_lam08` | soft length adaptive | Mean | 48 | 0.2594 | 0.5453 | 中间 K 的 adaptive，不如训练版 TopK48。 |
| `bi_topk_sum48_adaptive_lam08` | soft length adaptive | Sum | 48 | 0.1249 | 0.5420 | Sum 底座不稳，MMEB worst10 明显较差。 |
| `bi_topk_mean48_hard_adaptive` | hard length routing | Mean | 48 | 0.1820 | 0.5474 | 免训练 scorer-only：hard routing 没有带来收益，MMEB 明显弱于 soft TopK adaptive。 |
| trained `bi_topk_mean48_hard_adaptive` | hard length routing | Mean | 48 | 0.3487 | 0.5417 | 从 base 训练 1k step 后显著提升 MMEB，说明 hard routing 可学习，但仍未超过固定 TopK48 主线。 |

阶段性判断：

- Adaptive 的有效形式目前主要是 `Mean + TopK + soft length adaptive`；无 TopK 的 `bi_adaptive_lam08` 更像 ViDoRe 保守增强，不能充分解决 MMEB worst10。
- `bi_topk_sum48_adaptive_lam08` 证明 Sum 聚合不适合作为 adaptive 底座，至少在当前 sym160 checkpoint 的 scorer-only 评估下如此。
- 免训练 `bi_topk_mean48_hard_adaptive` 结果偏弱，但训练版达到 `0.3487 / 0.5417`，说明 hard routing 本身不是完全无效，而是需要通过训练适配；后续若继续做 hard adaptive，更应该引入 modality routing 或 score confidence，而不是只看长度。
- 精确的 `bi_adaptive_sum_lam08`、`bi_adaptive_hard_lam08` 仍是 NOT-RUN；如果后续要跑，必须先在脚本里明确实现/命名，避免和 TopK48 变体混淆。

当前建议：

1. 先完成 P0 8 格表，尤其补齐 `q2d_topk_sumK`、`bi_sum_lamXX`、`bi_topk_sumK_lamXX`。
2. 如果 `Mean + TopK` 最稳，则 adaptive 基于 `bi_topk_mean` 做。
3. 如果 `Sum + TopK` 更稳，则 adaptive 基于 `bi_topk_sum` 做。
4. 如果双向无 TopK 已经足够，则 adaptive 只需要调 lambda，不需要调 K。
5. Hard routing 是 P1/P2 之间的实验，只有当 soft lambda 解释不了 CIRR/FashionIQ 或 VQA/ViDoRe 冲突时再实现。

### P2 / TODO: 尚未实现或暂不优先的想法

这些机制有论文写作价值，但当前不应先实现，除非 P0/P1 都不能解释问题。

| ID | Mechanism | Status | 备注 |
| --- | --- | --- | --- |
| TODO-1 | modality-aware MaxSim: TT / TI / IT / II 分通道 scorer-only | TODO | 需要 eval embeddings 保留文本/图像 token mask；适合解释 image+text to image+text。 |
| TODO-2 | query-side learned importance scorer at eval time | TODO | 需要额外训练或复用 FolderHomo saliency，免训练不够公平。 |
| TODO-3 | target-side coverage entropy score | TODO | 比 hit penalty 更细，但要稳定定义 coverage 分布。 |
| TODO-4 | Sinkhorn / OT rerank | TODO | 计算更贵，适合作为小规模 rerank 上界，不适合作为第一阶段。 |
| TODO-5 | CIRR/FashionIQ prompt-only scorer variants | TODO | 可做错误分析，但不作为统一 MaxSim 机制主线。 |
| TODO-6 | image-query-only special routing | TODO | 和统一检索目标有冲突，除非最终证明统一 scorer 无法处理。 |
| TODO-7 | global-local learned weight / per-task adaptive lambda | TODO | 需要训练或验证集调参，免训练公平性较弱。 |
| TODO-8 | score calibration by query modality / length bucket | TODO | 可能有效，但容易变成规则系统，不优先。 |

| Scorer | 解释 | 目的 |
| --- | --- | --- |
| `legacy_q2d_sum` | 原始 query->target MaxSim，query token 分数求和。 | 原始基线。 |
| `q2d_mean` | query->target MaxSim，query token 分数取均值。 | 去掉 query 长度尺度影响。 |
| `q2d_query_topkK` | query->target MaxSim 后，只保留 query 侧 topK token 分数。 | 保留 directed 检索假设，同时减少图像 query 噪声。 |
| `bi_mean_lamXX` | `lambda * q2d + (1-lambda) * d2q`。 | 补 target coverage，检查双向互覆盖是否有效。 |
| `bi_query_topkK_lamXX` | query 和 target 两侧都做 topK，再双向加权。 | 同时做 query 去噪和 reciprocal coverage。 |
| `bi_adaptive_lam08` | 根据 query/doc token 长度动态选择 q2d 权重，范围 0.5 到 0.8。 | 保留短 query 检索优势，同时在图文/图像 query 上自动增强双向。 |
| `bi_query_topkK_adaptive_lam08` | query topK 去噪后再做 adaptive 双向。 | 当前最值得验证的折中方案之一。 |
| `lse_beta20` | 用 LogSumExp 平滑替代 hard max。 | P1，对比 hard winner-take-all 是否是核心问题。 |
| `bi_topk_mean_k4_lam05` | 每个 token 对另一侧取 top-k 均值，再双向平均。 | P1，弱化单个 winner token 劫持。 |

## 已有 worst10 历史结果

以下结果是 scorer-only，基于 `FolderHomo sym160 checkpoint-4000`，只改 MaxSim scorer，不重新训练。

| Dataset | legacy_q2d_sum | q2d_mean | q2d_query_topk64 | bi_mean | bi_query_topk64 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FashionIQ | 0.024 | 0.024 | 0.023 | 0.021 | 0.022 |
| CIRR | 0.103 | 0.103 | 0.090 | 0.095 | 0.083 |
| Country211 | 0.089 | 0.089 | 0.116 | 0.126 | 0.140 |
| GQA | 0.154 | 0.154 | 0.147 | 0.184 | 0.168 |
| ScienceQA | 0.197 | 0.197 | 0.285 | 0.224 | 0.271 |
| InfographicsVQA | 0.141 | 0.142 | 0.338 | 0.238 | 0.371 |
| A-OKVQA | 0.183 | 0.183 | 0.202 | 0.257 | 0.299 |
| Visual7W | 0.145 | 0.145 | 0.205 | 0.292 | 0.331 |
| OK-VQA | 0.216 | 0.216 | 0.244 | 0.313 | 0.325 |
| ChartQA | 0.171 | 0.171 | 0.354 | 0.315 | 0.426 |
| **Average** | **0.1423** | **0.1424** | **0.2004** | **0.2065** | **0.2436** |

结论：

1. `q2d_mean` 基本不改变结果，说明问题不是简单 query 长度归一化。
2. `q2d_query_topk64` 显著提升 VQA/Chart 类任务，说明 query 端 token 去噪是有效方向。
3. `bi_mean` 与 `bi_query_topk64` 进一步提升 VQA/Chart/OK-VQA，说明 target coverage 确实有价值。
4. FashionIQ/CIRR 没有被解决，说明组合式图像检索更依赖全局语义或训练目标，不能只靠双向 MaxSim。

## Group 2 完整结果

运行时间：2026-07-04 23:10 到 2026-07-05 03:43。

运行配置：

```text
Checkpoint=../MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7
NUM_GPUS=8
BATCH_QUERY=32
BATCH_PASSAGE=32
BATCH_SCORE=128
```

输出位置：

```text
../MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_vidorev2_worst10/
```

Group 2 包含：

```text
q2d_query_topk96
q2d_query_topk128
bi_mean_lam05
bi_mean_lam07
bi_mean_lam09
bi_adaptive_lam08
```

平均结果：

| Scorer | MMEB worst10 avg P@1 | ViDoRe v2 avg nDCG@5 | 备注 |
| --- | ---: | ---: | --- |
| `q2d_query_topk96` | 0.1571 | 0.5477 | topK 过大，MMEB 提升有限；ViDoRe 基本保持。 |
| `q2d_query_topk128` | 0.1389 | 0.5477 | 接近原始 q2d，MMEB worst10 无明显收益。 |
| `bi_mean_lam05` | 0.2057 | 0.4912 | MMEB 有收益，但 ViDoRe 明显掉点。 |
| `bi_mean_lam07` | 0.1844 | 0.5516 | 折中较稳，ViDoRe 保持，MMEB 中等提升。 |
| `bi_mean_lam09` | 0.1583 | 0.5541 | 偏向 q2d 后 MMEB 收益减弱，ViDoRe 略好。 |
| `bi_adaptive_lam08` | 0.2057 | 0.5576 | Group 2 中最均衡：MMEB 与 `bi_mean_lam05` 持平，ViDoRe 最好。 |

MMEB worst10 逐数据集结果：

| Dataset | q2d_topk96 | q2d_topk128 | bi_lam05 | bi_lam07 | bi_lam09 | adaptive_lam08 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FashionIQ | 0.0210 | 0.0210 | 0.0200 | 0.0230 | 0.0250 | 0.0200 |
| CIRR | 0.0870 | 0.0900 | 0.0950 | 0.1020 | 0.1080 | 0.0950 |
| Country211 | 0.1060 | 0.0990 | 0.1260 | 0.1100 | 0.0980 | 0.1260 |
| GQA | 0.1330 | 0.1260 | 0.1830 | 0.1850 | 0.1650 | 0.1830 |
| ScienceQA | 0.2430 | 0.2170 | 0.2220 | 0.2370 | 0.2090 | 0.2220 |
| InfographicsVQA | 0.2140 | 0.1600 | 0.2380 | 0.1980 | 0.1590 | 0.2380 |
| A-OKVQA | 0.1640 | 0.1470 | 0.2570 | 0.2230 | 0.1950 | 0.2570 |
| Visual7W | 0.1580 | 0.1340 | 0.2910 | 0.2210 | 0.1860 | 0.2910 |
| OK-VQA | 0.1970 | 0.1840 | 0.3090 | 0.2810 | 0.2360 | 0.3090 |
| ChartQA | 0.2480 | 0.2110 | 0.3160 | 0.2640 | 0.2020 | 0.3160 |
| **Average** | **0.1571** | **0.1389** | **0.2057** | **0.1844** | **0.1583** | **0.2057** |

ViDoRe v2 逐数据集结果：

| Dataset | q2d_topk96 | q2d_topk128 | bi_lam05 | bi_lam07 | bi_lam09 | adaptive_lam08 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| biomedical_lectures_v2 | 0.5994 | 0.5994 | 0.5303 | 0.5926 | 0.6122 | 0.6049 |
| biomedical_lectures_v2_multilingual | 0.5836 | 0.5836 | 0.4639 | 0.5484 | 0.5799 | 0.5727 |
| economics_reports_v2 | 0.5937 | 0.5937 | 0.5911 | 0.6213 | 0.5974 | 0.6060 |
| economics_reports_v2_multilingual | 0.5037 | 0.5037 | 0.4918 | 0.5229 | 0.5113 | 0.5196 |
| esg_reports_human_labeled_v2 | 0.5384 | 0.5384 | 0.4934 | 0.5892 | 0.5644 | 0.5818 |
| esg_reports_v2 | 0.5399 | 0.5399 | 0.4545 | 0.5162 | 0.5309 | 0.5339 |
| esg_reports_v2_multilingual | 0.4754 | 0.4754 | 0.4137 | 0.4707 | 0.4828 | 0.4840 |
| **Average** | **0.5477** | **0.5477** | **0.4912** | **0.5516** | **0.5541** | **0.5576** |

结论：

1. `q2d_query_topk96/128` 过于接近原始 directed MaxSim，ViDoRe 稳定但 MMEB worst10 收益不足。
2. `bi_mean_lam05` 证明 target coverage 能明显救 VQA/Chart/Visual7W/OK-VQA，但 0.5/0.5 对 ViDoRe 伤害较大。
3. `bi_mean_lam07` 和 `bi_mean_lam09` 说明更偏向 q2d 可以保住 ViDoRe，但 MMEB 收益会随 lambda 增大而下降。
4. `bi_adaptive_lam08` 是 Group 2 最值得保留的机制：MMEB 平均达到 0.2057，同时 ViDoRe v2 达到 0.5576。

## Group 3 完整结果

运行时间：2026-07-04 23:41 到 2026-07-05 02:21。

运行配置：

```text
Checkpoint=../MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7
NUM_GPUS=8
BATCH_QUERY=16
BATCH_PASSAGE=16
BATCH_SCORE=64
```

输出位置：

```text
../MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_group3_topk_lambda_adaptive/
```

Group 3 包含：

```text
bi_query_topk32_lam05
bi_query_topk32_lam07
bi_query_topk64_lam05
bi_query_topk64_lam07
bi_query_topk32_adaptive_lam08
bi_query_topk64_adaptive_lam08
```

平均结果：

| Scorer | MMEB worst10 avg P@1 | Retention avg P@1 | 备注 |
| --- | ---: | ---: | --- |
| `bi_query_topk32_lam05` | 0.2715 | 0.6312 | worst10 明显强于 topK64，但 retention 较低。 |
| `bi_query_topk32_lam07` | **0.2817** | 0.6521 | 当前 Group 3 中 worst10 最好。 |
| `bi_query_topk64_lam05` | 0.2436 | 0.6581 | 与历史 `bi_query_topk64` 基本一致。 |
| `bi_query_topk64_lam07` | 0.2340 | **0.6856** | retention 最好，适合保守保留原能力。 |
| `bi_query_topk32_adaptive_lam08` | 0.2715 | 0.6372 | adaptive 未超过固定 lambda。 |
| `bi_query_topk64_adaptive_lam08` | 0.2436 | 0.6692 | adaptive 未超过 `topK64_lam07`。 |

MMEB worst10 逐数据集结果：

| Dataset | bi_query_topk32_lam05 | bi_query_topk32_lam07 | bi_query_topk64_lam05 | bi_query_topk64_lam07 | bi_query_topk32_adaptive_lam08 | bi_query_topk64_adaptive_lam08 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FashionIQ | 0.025 | 0.025 | 0.022 | 0.023 | 0.025 | 0.022 |
| CIRR | 0.086 | 0.091 | 0.083 | 0.085 | 0.086 | 0.083 |
| Country211 | 0.141 | 0.147 | 0.140 | 0.137 | 0.141 | 0.140 |
| GQA | 0.191 | 0.197 | 0.168 | 0.165 | 0.191 | 0.168 |
| ScienceQA | 0.245 | 0.294 | 0.271 | 0.279 | 0.245 | 0.271 |
| A-OKVQA | 0.340 | 0.332 | 0.299 | 0.278 | 0.340 | 0.299 |
| OK-VQA | 0.364 | 0.368 | 0.325 | 0.304 | 0.364 | 0.325 |
| Visual7W | 0.377 | 0.373 | 0.331 | 0.283 | 0.377 | 0.331 |
| InfographicsVQA | 0.467 | 0.506 | 0.371 | 0.373 | 0.467 | 0.371 |
| ChartQA | 0.479 | 0.484 | 0.426 | 0.413 | 0.479 | 0.426 |
| **Average** | **0.2715** | **0.2817** | **0.2436** | **0.2340** | **0.2715** | **0.2436** |

Retention 逐数据集结果：

| Dataset | bi_query_topk32_lam05 | bi_query_topk32_lam07 | bi_query_topk64_lam05 | bi_query_topk64_lam07 | bi_query_topk32_adaptive_lam08 | bi_query_topk64_adaptive_lam08 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VisualNews_i2t | 0.315 | 0.424 | 0.460 | 0.577 | 0.315 | 0.460 |
| VisDial | 0.514 | 0.525 | 0.521 | 0.530 | 0.522 | 0.532 |
| MSCOCO_i2t | 0.534 | 0.555 | 0.571 | 0.607 | 0.534 | 0.571 |
| MSCOCO_t2i | 0.598 | 0.611 | 0.634 | 0.661 | 0.624 | 0.678 |
| VisualNews_t2i | 0.614 | 0.620 | 0.614 | 0.635 | 0.621 | 0.634 |
| ImageNet-1K | 0.660 | 0.672 | 0.670 | 0.682 | 0.660 | 0.670 |
| RefCOCO-Matching | 0.708 | 0.710 | 0.717 | 0.722 | 0.713 | 0.719 |
| VOC2007 | 0.854 | 0.859 | 0.859 | 0.861 | 0.854 | 0.859 |
| WebQA | 0.884 | 0.893 | 0.877 | 0.895 | 0.892 | 0.900 |
| **Average** | **0.6312** | **0.6521** | **0.6581** | **0.6856** | **0.6372** | **0.6692** |

结论：

1. `bi_query_topk32_lam07` 是 Group 3 中最好的 MMEB worst10 scorer，平均 `0.2817`。
2. `bi_query_topk64_lam07` 的 retention 最好，平均 `0.6856`，说明更大的 topK 更能保留原有强项。
3. `topK32` 比 `topK64` 更利于 worst10，继续支持“减少 query 端参与 MaxSim 的噪声 token”这个判断。
4. `adaptive_lam08` 没有带来额外收益，基本等同或弱于固定 lambda；后续优先保留 `topK32 + lam07` 和 `topK64 + lam07` 两个方向。

## 2026-07-05 免训练正式汇总

这一组是当前 MaxSim 交互机制的主表。所有方法都基于同一个模型做 **scorer-only eval**，不重新训练。

运行配置：

```text
Checkpoint=../MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000
Budget=160/160/160
Training=no retrain
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7
NUM_GPUS=8
BATCH_QUERY=32
BATCH_PASSAGE=32
BATCH_SCORE=128
```

测试集固定为：

```text
MMEB worst10: report P@1 / recall_at_1
ViDoRe v2: report average nDCG@5
```

完整 12 行结果如下：

| Dataset | bi_adaptive_lam08 | bi_mean_lam05 | bi_mean_lam07 | bi_mean_lam09 | legacy_q2d_sum | q2d_mean | q2d_query_topk128 | q2d_query_topk16 | q2d_query_topk32 | q2d_query_topk48 | q2d_query_topk64 | q2d_query_topk96 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FashionIQ | 0.0200 | 0.0200 | 0.0230 | 0.0250 | 0.0250 | 0.0250 | 0.0210 | 0.0320 | 0.0240 | 0.0250 | 0.0240 | 0.0210 |
| CIRR | 0.0950 | 0.0950 | 0.1020 | 0.1080 | 0.1040 | 0.1040 | 0.0900 | 0.0830 | 0.0880 | 0.0890 | 0.0890 | 0.0870 |
| Country211 | 0.1260 | 0.1260 | 0.1100 | 0.0980 | 0.0880 | 0.0880 | 0.0990 | 0.1670 | 0.1560 | 0.1310 | 0.1170 | 0.1060 |
| GQA | 0.1830 | 0.1830 | 0.1850 | 0.1650 | 0.1540 | 0.1540 | 0.1260 | 0.1450 | 0.1880 | 0.1630 | 0.1490 | 0.1330 |
| ScienceQA | 0.2220 | 0.2220 | 0.2370 | 0.2090 | 0.1970 | 0.1970 | 0.2170 | 0.2300 | 0.3200 | 0.3020 | 0.2830 | 0.2430 |
| InfographicsVQA | 0.2380 | 0.2380 | 0.1980 | 0.1590 | 0.1370 | 0.1390 | 0.1600 | 0.4060 | 0.5100 | 0.4360 | 0.3330 | 0.2140 |
| A-OKVQA | 0.2570 | 0.2570 | 0.2230 | 0.1950 | 0.1820 | 0.1820 | 0.1470 | 0.1990 | 0.2990 | 0.2440 | 0.2010 | 0.1640 |
| Visual7W | 0.2910 | 0.2910 | 0.2210 | 0.1860 | 0.1470 | 0.1470 | 0.1340 | 0.3020 | 0.3260 | 0.2520 | 0.2070 | 0.1580 |
| OK-VQA | 0.3090 | 0.3090 | 0.2810 | 0.2360 | 0.2150 | 0.2150 | 0.1840 | 0.2120 | 0.3310 | 0.2800 | 0.2440 | 0.1970 |
| ChartQA | 0.3160 | 0.3160 | 0.2640 | 0.2020 | 0.1740 | 0.1750 | 0.2110 | 0.4390 | 0.4560 | 0.4220 | 0.3560 | 0.2480 |
| **MMEB-worst10-average** | **0.2057** | **0.2057** | **0.1844** | **0.1583** | **0.1423** | **0.1426** | **0.1389** | **0.2215** | **0.2698** | **0.2344** | **0.2003** | **0.1571** |
| **ViDoRe-v2-average** | **0.5576** | **0.4912** | **0.5516** | **0.5541** | **0.5477** | **0.5477** | **0.5477** | **0.3765** | **0.5193** | **0.5474** | **0.5477** | **0.5477** |

阶段性结论：

1. `q2d_mean` 与 `legacy_q2d_sum` 几乎一致，说明简单 query 长度归一化不是关键。
2. `q2d_query_topk32` 是 MMEB worst10 最强的免训练修复：平均从 0.1423 提升到 0.2698，但 ViDoRe v2 从 0.5477 降到 0.5193。
3. `q2d_query_topk48` 是当前最均衡的免训练方案：MMEB worst10 从 0.1423 提升到 0.2344，同时 ViDoRe v2 基本不掉点，0.5474 vs 0.5477。
4. `q2d_query_topk16` 过于激进，虽然 MMEB 有提升，但 ViDoRe v2 明显下降到 0.3765，不适合作为主方案。
5. `bi_mean_lam05` 能提升 MMEB worst10 到 0.2057，但 ViDoRe v2 掉到 0.4912，说明纯 0.5/0.5 双向互覆盖会破坏视觉文档主线。
6. `bi_adaptive_lam08` 同时提升 MMEB worst10 和 ViDoRe v2，说明 adaptive 双向有价值，但 MMEB 修复强度不如 `q2d_query_topk32/48`。

当前推荐：

| 用途 | 推荐方法 | 原因 |
| --- | --- | --- |
| 主论文默认免训练修复 | `q2d_query_topk48` | MMEB worst10 明显提升，ViDoRe v2 几乎不损失。 |
| MMEB 诊断上界 | `q2d_query_topk32` | 当前 worst10 平均最高，用于证明 query token 去噪确实有效。 |
| ViDoRe 保守增强 | `bi_adaptive_lam08` | ViDoRe v2 最高，同时 MMEB 有中等提升。 |

因此，当前最明确的机制结论是：**MMEB 的主要问题不是 MaxSim 求和尺度，而是图像/图文 query 端 token 噪声过多；保留 directed MaxSim，但只让高响应 query token 参与打分，是最有效且最稳的第一阶段修复。**

## 历史 TopK sweep 结果

以下结果基于 `core4_global_local_sym160_s500_from_base/checkpoint-500` 的 scorer-only sweep，只用于说明“query TopK 有早期信号”。它不是正式 `FolderHomo sym160 checkpoint-4000` 的公平消融结果，不进入论文主表；正式表必须用上文统一 checkpoint 和统一测试集重跑。

| Dataset | topK32 | topK48 | topK64 | topK96 |
| --- | ---: | ---: | ---: | ---: |
| FashionIQ | 0.020 | 0.021 | 0.022 | 0.023 |
| CIRR | 0.117 | 0.116 | 0.116 | 0.120 |
| Country211 | 0.223 | 0.191 | 0.150 | 0.128 |
| GQA | 0.238 | 0.223 | 0.206 | 0.181 |
| ScienceQA | 0.311 | 0.312 | 0.309 | 0.299 |
| InfographicsVQA | 0.423 | 0.416 | 0.408 | 0.393 |
| A-OKVQA | 0.355 | 0.344 | 0.327 | 0.295 |
| Visual7W | 0.326 | 0.303 | 0.284 | 0.253 |
| OK-VQA | 0.361 | 0.333 | 0.294 | 0.237 |
| ChartQA | 0.731 | 0.718 | 0.734 | 0.691 |
| **Average** | **0.3105** | **0.2977** | **0.2850** | **0.2620** |

临时结论：更小的 query topK 倾向于改善 worst10，尤其是 VQA/Chart/InfoGraphics。但由于 checkpoint 不统一，后续只能作为选题动机，不能作为最终证据。

## 下一步判据

一个 scorer 只有同时满足以下条件才进入训练阶段：

1. MMEB worst10 平均明显高于 `legacy_q2d_sum` 和 `q2d_mean`。
2. ViDoRe v2 平均 `nDCG@5` 不明显下降。
3. 提升不是只来自单个数据集，而是在 VQA/Chart/InfoGraphics 等多个困难任务上稳定出现。

当前优先级：

| Priority | Scorers |
| --- | --- |
| P0 | `q2d_query_topk16/32/48/64/96/128`, `bi_query_topk32_lam05/07`, `bi_query_topk64_lam05/07`, `bi_mean_lam05/07/09`, `bi_adaptive_lam08`, `bi_query_topk32/64_adaptive_lam08` |
| P1 | `lse_beta20`, `bi_lse_beta20_lam05`, `bi_topk_mean_k4_lam05`, `bi_topk_mean_k8_lam05`, `q2d_query_topk64_global_w02` |

## 下一阶段: 6 组 1k from-base 训练计划

免训练实验已经说明：query-side TopK 与双向 target coverage 都有信号。下一阶段统一从原生 `colqwen2.5-base` 开始训练 1k step，判断 MaxSim loss 机制能否在较低成本下真正学出来。这里不使用 `sym160` warm-start。

统一固定项：

```text
Base model=models/colqwen2.5-base
Model=FolderHomo / FOLDER-Homo real-token compressor
Budget=160/160/160
Resume checkpoint=none
Warm-start adapter=none
Trainable=LLM LoRA + custom_text_proj + folder_homo compressor
Steps=1000
Learning rate=2e-4
LR scheduler=constant
Warmup=0
MARC=off
TopK if used=48
Eval=MMEB worst10 P@1 + ViDoRe v2 nDCG@5
```

训练数据原则：

```text
ViDoRe train / visual-document train
+ MMEB hard/core4 train
```

这里必须包含 ViDoRe 训练集，避免为了修 MMEB 把视觉文档主线训坏。MMEB 部分优先覆盖 VQA/Chart/Infographic hard tasks，并保留少量 classification/compositional 信号。不要使用完全随机 fullmix；若出现长度不匹配或 DDP 卡住，使用 grouped/bucketed sampler：每个 step 先选任务大类，再在同类样本内部组 batch。

6 个实验如下：

| ID | Run | Scorer / loss target | 机制 | 目的 |
| --- | --- | --- | --- | --- |
| A1 | `vidore_mmeb_q2d_mean_s1k_from_base` | `q2d_mean` | 原始单向 MaxSim mean | 训练基线，确认同一数据和 1k step 下原始机制能到哪里。 |
| A2 | `vidore_mmeb_qtopk48_s1k_from_base` | `q2d_query_topk48` | 单向 query TopK48 | 验证 query-side 去噪能否通过训练进一步放大收益。 |
| B1 | `vidore_mmeb_bi_mean_lam07_s1k_from_base` | `bi_mean_lam07` | 固定双向，不裁 token | 验证 target coverage 本身是否有效。 |
| B2 | `vidore_mmeb_bi_adaptive_s1k_from_base` | `bi_adaptive_lam08` | 自适应双向，不裁 token | 验证长度自适应权重是否比固定双向更稳。 |
| C1 | `vidore_mmeb_bi_qtopk48_lam07_s1k_from_base` | `bi_query_topk48_lam07` | 固定双向 + TopK48 | 验证 query 去噪 + target coverage 的固定组合。 |
| C2 | `vidore_mmeb_bi_qtopk48_adaptive_s1k_from_base` | `bi_query_topk48_adaptive_lam08` | 自适应双向 + TopK48 | 验证 query 去噪 + adaptive coverage 是否最稳。 |

三台机器分配：

| Machine | Runs | 作用 |
| --- | --- | --- |
| A | `vidore_mmeb_q2d_mean_s1k_from_base`, `vidore_mmeb_qtopk48_s1k_from_base` | 单向基线组，回答 TopK48 是否比原始单向 MaxSim 更好。 |
| B | `vidore_mmeb_bi_mean_lam07_s1k_from_base`, `vidore_mmeb_bi_adaptive_s1k_from_base` | 双向无 TopK 组，回答 target coverage 和 adaptive 是否有效。 |
| C | `vidore_mmeb_bi_qtopk48_lam07_s1k_from_base`, `vidore_mmeb_bi_qtopk48_adaptive_s1k_from_base` | TopK + 双向组合组，回答组合机制是否超过单独 TopK48。 |

关键对比关系：

| 对比 | 回答的问题 |
| --- | --- |
| A1 vs A2 | TopK48 query 去噪是否比原始 MaxSim 更适合 MMEB。 |
| A1 vs B1 | 只加双向 target coverage 是否有效。 |
| B1 vs B2 | adaptive 权重是否比固定 lambda 更稳。 |
| A2 vs C1 | 在 TopK48 基础上加入反向 coverage 是否有额外收益。 |
| C1 vs C2 | adaptive 是否能改善固定双向 TopK 的稳定性。 |
| A2 vs C2 | 最终是单向去噪更好，还是去噪 + 自适应双向更好。 |

判定标准：

| 条件 | 说明 |
| --- | --- |
| MMEB worst10 明显高于 `q2d_mean` 训练基线 | 否则训练后的机制没有修复 MMEB。 |
| ViDoRe v2 不明显低于同批 `q2d_mean` | 否则该方法不适合作为视觉文档主线方案。 |
| VQA/Chart/Infographic 多个子集共同提升 | 防止只由单个数据集拉高平均。 |
| CIRR/FashionIQ 单独报告但不作为唯一成败标准 | 这类 compositional retrieval 需要全局组合语义，MaxSim scorer 可能不能单独解决。 |

## 历史脚本

早期探索脚本已归档到 `legacy/`。它们可能包含 CIRR-only、retention、full-MMEB、continued training 等历史实验入口，但不再作为当前主线使用。
