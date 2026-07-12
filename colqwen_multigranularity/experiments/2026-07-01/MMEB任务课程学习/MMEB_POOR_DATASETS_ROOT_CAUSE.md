# MMEB 低分数据集问题归因诊断

更新时间：2026-07-03

## 当前结论

MMEB 上最差数据集不是同一种问题。当前证据显示至少有两类失效：

1. **图像/图文 query 到极短文本 target**：Country211、InfographicsVQA、Visual7W、GQA、ChartQA、A-OKVQA、ScienceQA、OK-VQA、DocVQA 等大多是 `image + text question -> short text answer/label`。评测 corpus 的 target 往往只有 5-15 个字符左右，例如 `14`、`pinterest`、`cigarette`、`Fiji`。训练 debug 中 VQA-hard batch 的 query embedding 约 540-560 tokens，而 doc 只有 14-27 tokens。
2. **图像 query 到图像 target，但文本提示不可区分候选**：FashionIQ/CIRR 是 image-query -> image-corpus，corpus 文本全部或几乎全部是 `Represent the given image.`，真正差异全在图像 token 上。因此普通文本提示不能帮助区分候选，必须依赖图像 token 交互和图像表征本身。

因此，低分不只是“训练不足”，而是任务形式和当前单向 MaxSim/MRL token 压缩之间存在结构性冲突。

## 关键证据

### 1. 训练样本结构

原始 MoCa 训练审计文件：`MMEB任务课程学习/data_audit/raw_sample_audit.md`

VQA-hard 训练子集包括 InfographicsVQA、ChartQA、A-OKVQA、DocVQA、OK-VQA、Visual7W。审计结果：

- query 全部有图：`query_has_image = checked`
- positive target 全部无图：`pos_has_image = 0`
- negative target 也无图
- 正样本 target 多为短答案，如 `serif fonts`、`Yes`、`2014`、`cab`、`5`

训练日志进一步确认：

- `mrl_query_has_images_ratio = 1.0`
- `mrl_doc_has_images_ratio = 0.0`
- step=2 时 `query_emb=(32, 538-563, 128)`，`doc_emb_local=(32, 20, 128)`，`neg_doc_emb=(32, 14-27, 128)`

这意味着当前损失在训练一个极端非对称问题：数百个图像/问题 query tokens 对十几个答案 tokens 做 q->d MaxSim。

### 2. 评测集结构

评测审计文件：`MMEB任务课程学习/data_audit/eval_beir_audit.md`

最差 10 个中的主要结构：

| 类型 | 数据集 | 评测形态 | 主要问题 |
| --- | --- | --- | --- |
| 图像/问题 -> 短文本 | Country211, InfographicsVQA, Visual7W, GQA, ChartQA, A-OKVQA, ScienceQA, OK-VQA | query 有图，corpus 无图 | target 太短，候选文本语义区分弱，像 VQA/分类而不是普通检索 |
| 图像/文本 -> 图像 | FashionIQ, CIRR | query 有图，corpus 有图 | corpus 文本几乎恒为 `Represent the given image.`，候选区分只能依赖图像 token |

### 3. Scorer-only 对照

结果文件：`MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/eval/maxsim_worst10_direction_diagnosis/worst10_maxsim_diagnosis_summary.md`

在不重新训练、只改 scorer 的情况下：

- worst10 baseline 平均 P@1：0.143
- `q2d_mean` 平均：0.143，几乎不变
- `bi_mean` 平均：0.206，明显提升
- VQA/短文本类提升明显：ChartQA 0.174 -> 0.316，Visual7W 0.147 -> 0.291，OK-VQA 0.214 -> 0.310
- FashionIQ/CIRR 下降：FashionIQ 0.025 -> 0.020，CIRR 0.105 -> 0.095

这说明：

- 对 VQA/短文本类，问题和 MaxSim 非对称性交互强相关；target->query coverage 能补一部分信号。
- 对 FashionIQ/CIRR，简单双向 MaxSim 不是解法，可能需要图像 token 重要性、图像 query 剪枝、或专门的组合式图像检索训练。

### 4. Targeted tuning 对照

已跑短训结果：

| 方法 | target avg | retention avg | 结论 |
| --- | ---: | ---: | --- |
| vqa_hard_s64 | 0.0045 | 0.0465 | 直接崩，严重灾难性遗忘 |
| vqa_hard_replay20_s64 | 0.0046 | 0.0468 | replay 20% 无法修复 |
| compositional_s64 | 0.1873 | 0.0464 | NIGHTS 有信号，但整体 retention 仍崩 |
| vqa_hard_s64_lr1e-5 | 0.0046 | 0.0467 | 低学习率、大 batch 仍无法学起 VQA-hard，且 retention 仍崩 |

这说明不能简单“专门训差集”。当前 adapter/folder_homo 部分非常敏感，短训就会破坏原本 4k checkpoint 的通用能力。


### 5. MMEB 评测是 local candidate 排序，不是全库召回

源码确认：`MMEBEvaluator` 会读取 query 的 `local-did`，并在 `_get_local_retrieval_results_and_metrics` 中只对该 query 的局部候选排序；`local-did[0]` 是 ground-truth doc。

因此，最差数据集的 P@1 低不能简单解释为 corpus 太大。它们是在给定局部候选池内仍然无法把正例排到第一，说明模型对这些局部 hard candidates 的判别信号不足。

这进一步支持两个判断：

- VQA/分类类：短答案/短标签之间局部区分弱，模型需要图像理解和答案判别能力，不只是 retrieval alignment。
- FashionIQ/CIRR：局部候选都是图像，文本提示基本恒定，必须靠图像 token 和组合式变化理解。


### 6. 局部候选池难度审计

候选池审计文件：`MMEB任务课程学习/data_audit/local_candidate_audit.md`

关键统计：

| Dataset | local candidates | target text | local text uniqueness | 解释 |
| --- | ---: | --- | ---: | --- |
| FashionIQ | 1000 | 全部 `Represent the given image.` | 0.001 | 完全依赖图像候选判别，文本无区分信号 |
| CIRR | 1000 | 全部 `Represent the given image.` | 0.001 | 完全依赖图像候选判别和组合式变化理解 |
| ChartQA | 1000 | median 4 chars | 0.998 | 在 1000 个短答案中选正确数值/文本 |
| OK-VQA | 1000 | median 6 chars | 1.000 | 在 1000 个短答案中选正确答案 |
| InfographicsVQA | 1000 | median 5 chars | 1.000 | OCR/图表/视觉理解 + 短答案判别 |
| Visual7W | 1000 | median 8 chars | 0.997 | 图像问答式短答案判别 |
| Country211 | 211 | median 8 chars | 1.000 | 图像分类到国家标签 |

这说明 P@1 低的任务并不是普通“检索相关文档”，而是 local candidate classification / answer selection。当前多向量 MaxSim 模型被迫把图像理解问题转化成 token-level retrieval score，因此会明显弱于专门的 VQA/分类目标。

## 当前最可能的问题链条

1. **任务目标错位**：MMEB 中不少低分任务本质更接近 VQA answer retrieval / image classification，而不是文档/图像检索。target 是短标签或短答案，信息量非常低。
2. **单向 q->d MaxSim 对长图像 query 不友好**：图像 query 有数百个视觉 tokens，doc 只有十几个文本 tokens。q->d 要求每个 query token 找 target winner，容易让背景/冗余视觉 token 参与损失，梯度噪声很重。
3. **target token 太少，coverage 约束弱**：短答案端被动承接，无法表达充分语义，也很难区分 `Yes/No/14/2014` 这类候选。
4. **训练配方覆盖不足但不是唯一原因**：GQA/ScienceQA/FashionIQ/EDIS 等不在当前 full train config 中；但已经在训练中的 ChartQA/OK-VQA/Visual7W 等仍差，说明不是纯粹 OOD。
5. **专项训练容易破坏通用压缩器**：只训 VQA-hard 或 compositional-hard 64 step，retention 直接掉到约 0.046，说明当前 folder_homo adapter 在多任务平衡上很脆弱。

## 下一步实验判断

### A. 低学习率 targeted tuning

已完成训练与 VQA-hard 目标集评测：`taskcurr_vqa_hard_warm_sym160_eq_b32_s64_lr1e-5`

目的：验证之前 target-only 训练崩坏是否主要由 `1e-4` 学习率过大导致。

训练配置：

- warm start：`MMEB全量/runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000`
- subset：VQA-hard 训练集
- steps：64
- lr：`1e-5`
- per-device train batch：32，8 卡，有效 batch 256
- 训练显存：约 64.3GB/卡
- 最终 train loss：约 5.954

VQA-hard 目标集评测结果：

| Dataset | P@1 |
| --- | ---: |
| Visual7W | 0.000 |
| TextVQA | 0.001 |
| ChartQA | 0.002 |
| InfographicsVQA | 0.004 |
| A-OKVQA | 0.005 |
| DocVQA | 0.005 |
| OK-VQA | 0.005 |
| VizWiz | 0.005 |
| GQA | 0.009 |
| ScienceQA | 0.010 |
| **Average** | **0.0046** |

结论：

- 低学习率和大 batch 没有解决 target-only 崩坏问题。
- 该结果远低于 `sym160_4k` baseline 中 VQA/短答案类数据集的 0.1-0.3 区间表现，说明专项训练不是“没有充分训练”，而是训练目标/数据形式与当前 MaxSim 表征发生了严重错配。
- retention eval 已完成，说明该 checkpoint 对原本图文/图像检索能力也有严重破坏。

Retention 评测结果：

| Dataset | P@1 |
| --- | ---: |
| VisDial | 0.001 |
| MSCOCO_t2i | 0.001 |
| ChartQA | 0.002 |
| VisualNews_t2i | 0.005 |
| GQA | 0.009 |
| ImageNet-1K | 0.039 |
| MSCOCO_i2t | 0.050 |
| VisualNews_i2t | 0.051 |
| CIRR | 0.085 |
| VOC2007 | 0.115 |
| WebQA | 0.156 |
| **Average** | **0.0467** |

这个结果和 `lr=1e-4`、`replay20`、`compositional` 短训的 retention collapse 基本一致。因此，当前证据不支持继续沿着 target-only curriculum tuning 方向投入算力。

### B. VQA 类应优先尝试 scorer / MaxSim 机制改造

优先级：

1. scorer-only 的 `bi_mean` 已证明 VQA 类有效，应做 full-MMEB `bi_mean` 完整评测，检查平均收益和副作用。
2. 重点比较 `q2d`、`bi_mean`、`bi_topk_mean`、`global_local_bi_mean` 等不重训 scorer，先判断收益来自交互机制还是训练。
3. 评测必须按任务类型拆开看：VQA/短答案、classification、image-text retrieval、image-image/compositional retrieval。不能只看 overall，因为 `bi_mean` 对 VQA 有明显收益，但对 FashionIQ/CIRR 可能下降。
4. 当前先不继续训练侧 target-only 或 replay 实验；如果 scorer-only 结论稳定，再考虑是否需要训练。

### C. 短答案 target 表征改造

短答案任务的另一个核心问题是 target 过短。当前 corpus target 可能只有 `14`、`yes`、`Fiji`、`cigarette` 这类 token，和长图像 query 做 MaxSim 时语义承载力不足。

建议优先做不重训的 answer prompt expansion 评测：

| 原始 target | 改造 target 示例 | 目的 |
| --- | --- | --- |
| `14` | `The answer is 14.` | 增加可匹配的语义上下文 |
| `yes` | `The answer is yes.` | 避免极短 token 单点匹配 |
| `Fiji` | `The class label is Fiji.` | 分类标签显式化 |
| `cigarette` | `The object/answer is cigarette.` | 将标签转成自然语言 answer |

这一路的关键是先只改评测 corpus 文本构造，不改模型参数。若 prompt expansion 能显著提升 VQA/分类类 P@1，说明短 target 表征是主要瓶颈之一；若提升有限，则问题更偏向视觉理解能力或 MaxSim 交互本身。

### D. FashionIQ/CIRR 应单独处理

这类不是短文本 answer retrieval，而是组合式图像检索。简单 `bi_mean` 会下降，说明需要另一路：

- image-query token pruning / importance weighting
- adaptive MaxSim：image-image 用更平衡或图像专用交互，image-text 用短端主导
- 构造同类 hard negatives，而不是把它和 VQA 短答案混在一个课程里

## 临时结论

当前最有解释力的归因是：**MMEB 低分来自任务类型混杂导致的交互目标错配，而不是单一数据未见过或训练步数不足。** 其中 VQA/分类类主要是“长图像 query -> 极短文本 target”的 MaxSim 非对称和短 target 表征问题；FashionIQ/CIRR 主要是图像候选区分和组合式检索问题。

## 当前阶段决策：暂不继续 target-only 训练

截至 2026-07-03，target-only 课程学习已经被多组结果基本证伪：

- `lr=1e-4`：target avg 约 0.0045，retention avg 约 0.0465。
- `replay20`：target avg 约 0.0046，retention avg 约 0.0468。
- `lr=1e-5 + 大 batch`：target avg 0.0046，retention avg 0.0467。

因此，后续优先级调整为：

1. **先做 scorer-only / MaxSim 机制完整评测**：重点是 `bi_mean`，因为它在 worst10 scorer-only 诊断中将平均 P@1 从 0.143 提到 0.206，且 VQA/短答案类提升最明显。
2. **再做短答案 target 表征改造**：通过 answer prompt expansion 判断短 target 是否是主要瓶颈。
3. **暂缓混合 full replay 训练**：除非前两类不重训实验已经证明方向有效，否则继续训练容易把问题和遗忘混在一起，解释性较弱。
