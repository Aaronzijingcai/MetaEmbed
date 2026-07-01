# 2026-07-01 MMEB 全量实验

## 实验目的

这一组使用当前主线 FolderHomo / FOLDER 模型，不是 MRL-main 模型。这里“沿用 MRL-main”只指训练集和测试集口径：使用最早的 full train 配置和 MMEB full eval 配置，系统检查当前 FolderHomo 在 MMEB 全量任务上的表现和失效边界。

做 MMEB 全量的核心动机不是单纯补一个大评测，而是重新验证 MaxSim 非对称性问题是否能被视觉 token 压缩缓解。我们的理想目标原本是做 full MMEB，但直接用 MRL-main 时 MMEB 效果非常差。后续分析发现，late-interaction MaxSim 更适合“短 query 对长 document”的少配多场景；当 MMEB 的 query 本身包含图像时，query 端视觉 token 过长会破坏这种非对称假设，使 query-document 两侧都变成大量视觉 token 的多配多匹配，MaxSim 更容易被冗余局部 token 和 spurious matching 干扰。

因此，本实验的假设是：如果 FolderHomo 已经把多粒度视觉 token 压到 `160+160+160=480`，query 端图像 token 的冗余被显著减少，那么 MRL-main 在 MMEB 上暴露出的 MaxSim 非对称性问题可能会得到一定程度缓解。这个实验用于判断：当前 480-token FolderHomo 是否已经足够支持重新回到 MMEB full setting，还是 MMEB 中的图像 query / 多配多任务仍然不是当前 MaxSim 检索范式的合适主战场。

目标：

1. 用 FolderHomo `160/160/160` 和 `moca_data_ratios_v3_full.yaml` 跑完整训练配置。
2. 在 `test_data_mast_mmeb_v3.yaml` 上跑 MMEB full eval。
3. 比较压缩后 FolderHomo 是否相对缓解 MRL-main 在 MMEB full setting 上的失败。
4. 进一步做 query-side asymmetric compression：当 query 端有图像时，把 query 视觉 token 压到 `80/80/80` 或 `40/40/40`，target/doc 端仍保持 `160/160/160`。
5. 对 30+ 个 MMEB 子任务做 IND/OOD 和任务类型聚合，找出 MaxSim 仍然不适合的子任务。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `run_train_full.sh` | FolderHomo full train launcher，默认使用 `configs/train/moca_data_ratios_v3_full.yaml` 和 `160/160/160`。 |
| `eval_mmeb.py` | 复用 `experiments/exp_stagecompress/folder_homo/eval_folder_homo.py` 的模型加载逻辑，只负责 MMEB full eval。 |
| `eval_mmeb_full.sh` | MMEB full eval launcher，默认输出 `mmeb_full.json` 和 `mmeb_full_summary.json`。 |
| `eval_mmeb_asym_query.sh` | 用同一个 checkpoint 跑 query-image asymmetric budget：`80/80/80` 和 `40/40/40`。 |
| `analyze_mmeb.py` | 对 MMEB eval json 按 IND/OOD、Classification/VQA/Retrieval/Visual Grounding 聚合。 |

## 训练配置

默认训练配置：

```text
SUBSET_CONFIG=configs/train/moca_data_ratios_v3_full.yaml
MAX_STEPS=4000
NUM_GPUS=8
TRAIN_BSZ=4
INTERLEAVED_BSZ=4
GRAD_ACCUM_STEPS=1
MODEL_PATH=models/colqwen2.5-base
BUDGETS=160/160/160
COMPRESS_STAGES=all
```

正式训练：

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/2026-07-01/MMEB全量
RUN_NAME=folder_homo_mmeb_full_train_b160_160_160_4k bash run_train_full.sh
```

如果要跑 1 epoch，需要按当前有效训练样本数重设：

```bash
MAX_STEPS=18156 RUN_NAME=folder_homo_mmeb_full_train_b160_160_160_1epoch bash run_train_full.sh
```

## MMEB 全量评测

默认评测：

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/2026-07-01/MMEB全量
CHECKPOINT=runs/folder_homo_mmeb_full_train_b160_160_160_4k/checkpoint-4000 bash eval_mmeb_full.sh
```

输出：

```text
runs/<run>/eval/mmeb_full/mmeb_full.json
runs/<run>/eval/mmeb_full/mmeb_full_summary.json
```

默认主指标是 `recall_at_1`。如需和旧 `mrl_main/eval_mrl.py` 口径一致，也可以改为：

```bash
AVG_METRIC=recall_at_5 CHECKPOINT=... bash eval_mmeb_full.sh
```

只评测少数 MMEB 子任务：

```bash
ONLY_EVAL_KEYWORDS="MMEB-eval-VisDial-beir MMEB-eval-WebQA-beir MMEB-eval-MSCOCO_t2i-beir" \
CHECKPOINT=... bash eval_mmeb_full.sh
```

## Query 端非对称压缩（历史 eval-only 入口）

这一节是早期 eval-only 入口，已不作为最新 7 组正式实验口径。最新口径见上面的训练阶段 MMEB budget 实验。这里两个补充实验不重新训练模型，只在 MMEB eval 时改变 query 端预算：

| 实验 | Query 有图像时 | Target/Doc 端 | 目的 |
| --- | --- | --- | --- |
| `asym_q80_doc160` | `80/80/80` | `160/160/160` | 中等强度压缩 query image tokens，测试是否更接近短 query 场景。 |
| `asym_q40_doc160` | `40/40/40` | `160/160/160` | 强压缩 query image tokens，测试 MaxSim 非对称性是否进一步恢复，或是否因 query 信息损失过大而下降。 |

运行两组：

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/2026-07-01/MMEB全量
CHECKPOINT=runs/folder_homo_mmeb_full_train_b160_160_160_4k/checkpoint-4000 bash eval_mmeb_asym_query.sh
```

单独跑其中一组：

```bash
ASYM_QUERY_IMAGE_BUDGETS="80 80 80" CHECKPOINT=... bash eval_mmeb_full.sh
ASYM_QUERY_IMAGE_BUDGETS="40 40 40" CHECKPOINT=... bash eval_mmeb_full.sh
```

实现细节：`eval_mmeb.py` 在模型 `forward(is_query=True)` 且 batch 存在 `pixel_values/image_grid_thw` 时，临时把 FolderHomo block budgets 切到 query budget；forward 结束后立即恢复 `160/160/160`。因此 document/target 端始终使用原始 `160/160/160`，文本 query 也不受影响。

## 临时检查代码清理

临时检查训练和测试代码及 run 产物已删除；正式进度请看 `PROGRESS.md`。

## 2026-07-01 更新：训练阶段的 7 组 MMEB budget 实验

MMEB budget experiments keep `MAX_STEPS=4000` by default and use `_4k` run-name suffixes. This is intentionally different from the homogeneity experiments, whose formal launcher defaults to 3k steps.


最新口径不再把 query-side asymmetric compression 当作 eval-only trick，而是在训练阶段就区分 query/doc 预算。实现已单独放在本目录，不改 `exp_stagecompress/folder_homo` 主线：

| 文件 | 作用 |
| --- | --- |
| `config_mmeb_budget.py` | MMEB query/doc 双预算配置。 |
| `modeling_mmeb_budget.py` | 从 FolderHomo 模型代码复制出的本地版本，`forward(is_query=True)` 且 query batch 有图像时使用 query budget，doc/target/negative 使用 doc budget。 |
| `loss_mmeb_budget.py` | 训练 loss 的 MaxSim group mask 同步区分 query budget 和 doc budget，避免训练阶段只改模型、不改 mask。 |
| `train_mmeb_budget.py` / `run_train_budget.sh` | 单独的 MMEB budget 训练入口。 |
| `eval_mmeb_budget.py` / `eval_mmeb_budget_full.sh` | 单独的 MMEB budget 验证入口，和训练使用同一套 query/doc budget。 |

7 组正式实验为：

| 实验 | Query 有图像时 | Target/Doc 有图像时 | 说明 |
| --- | --- | --- | --- |
| `sym160` | `160/160/160` | `160/160/160` | query/doc 同质压缩，不考虑 MaxSim 非对称性。 |
| `sym80` | `80/80/80` | `80/80/80` | query/doc 同质压缩。 |
| `sym40` | `40/40/40` | `40/40/40` | query/doc 同质压缩。 |
| `sym20` | `20/20/20` | `20/20/20` | query/doc 同质压缩。 |
| `asym_q80_d160` | `80/80/80` | `160/160/160` | MaxSim 非对称性：query 短、target/doc 长。 |
| `asym_q40_d160` | `40/40/40` | `160/160/160` | MaxSim 非对称性：query 更短。 |
| `asym_q20_d160` | `20/20/20` | `160/160/160` | MaxSim 非对称性：query 最短。 |

运行单组训练：

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/2026-07-01/MMEB全量
QUERY_BUDGETS="80 80 80" DOC_BUDGETS="160 160 160" \
RUN_NAME=folder_homo_mmeb_budget_asym_q80_d160_4k bash run_train_budget.sh
```

运行单组验证：

```bash
QUERY_BUDGETS="80 80 80" DOC_BUDGETS="160 160 160" \
CHECKPOINT=runs/folder_homo_mmeb_budget_asym_q80_d160_4k/checkpoint-4000 \
bash eval_mmeb_budget_full.sh
```

7 组正式训练/验证命令和 TODO 已移到 `PROGRESS.md` 维护。

## 论文写法

这组实验不是新方法 ablation，而是检验视觉 token 压缩是否能缓解 MaxSim 非对称性，并定位当前 FolderHomo/FOLDER 模型在 MMEB 全量口径上的边界：

```text
MRL-main performs poorly on the full MMEB setting, suggesting that the asymmetric MaxSim assumption is less suitable when the query side also contains many visual tokens. We therefore revisit MMEB after compressing multi-granularity visual tokens with FolderHomo. This experiment tests whether reducing the query-side visual redundancy to 160+160+160 tokens can partially restore the short-query-to-long-document regime preferred by late interaction, and where the MaxSim assumption still breaks down.
```

后续重点看：

| 分组 | 解释 |
| --- | --- |
| Classification | 多为多类分类/识别，target 往往短，MaxSim 可能不占优。 |
| VQA | 如果 query 和 target 都包含较多语义，MaxSim 非对称性可能放大噪声。 |
| Retrieval | 更接近少配多或图文检索，可能相对稳定。 |
| Visual Grounding | 依赖局部视觉定位，可能与多粒度 token 更相关。 |
| IND/OOD | 判断 full train 是否只是记住常见 MMEB 分布，还是有跨任务泛化。 |
