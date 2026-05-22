# Exp MaxSim Report

## 1. Final Setting

This report only keeps the current final one-stage formulation.

Final scoring design:

- primary score: `query -> doc`
- auxiliary score: `doc -> query`
- reverse branch uses **top-k doc token mean** instead of all-token mean
- fused score:

`score = (w_q * score_q2d + w_d * score_d2q_topk) / (w_q + w_d)`

Current default recommendation:

- `query_score_weight = 0.9`
- `doc_score_weight = 0.1`
- `doc_topk_ratio = 0.1`
- `doc_topk_min_tokens = 8`

Deprecated and removed from the final report:

- full symmetric BiMax as a one-stage main score
- all-token reverse averaging as a formal method

## 2. Relevant Files

Main experiment directory:

`/MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/exp_maxsim`

Main code:

- `symmetric_maxsim.py`
- `train_symmetric_maxsim.py`
- `eval_symmetric_maxsim.py`
- `train_symmetric_maxsim.sh`
- `eval_symmetric_maxsim.sh`

Length analysis code:

- `collect_lengths.py`
- `collect_train_lengths.py`
- `plot_lengths.py`
- `plot_train_lengths.py`

Main result files:

- test lengths:
  - `/MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/exp_maxsim/results/all_lengths.json`
  - `/MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/exp_maxsim/results/all_lengths.md`
- train lengths:
  - `/MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/exp_maxsim/results/train_lengths.json`
  - `/MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/exp_maxsim/results/train_lengths.md`

## 3. Formal Training Package

Formal training script:

`train_symmetric_maxsim.sh`

Formal runs to execute:

### Run A: weak reverse

- `query_score_weight = 0.9`
- `doc_score_weight = 0.1`
- `doc_topk_ratio = 0.1`

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/exp_maxsim
SCORE_MODE=bimax QUERY_SCORE_WEIGHT=0.9 DOC_SCORE_WEIGHT=0.1 DOC_TOPK_RATIO=0.1 \
OUTPUT_DIR=/MURE-V2/code/MetaEmbed/colqwen_multigranularity/runs/exp_maxsim/formal_bimax_09_01 \
bash train_symmetric_maxsim.sh
```

Result:

- `[TODO]`

### Run B: medium reverse

- `query_score_weight = 0.7`
- `doc_score_weight = 0.3`
- `doc_topk_ratio = 0.1`

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/exp_maxsim
SCORE_MODE=bimax QUERY_SCORE_WEIGHT=0.7 DOC_SCORE_WEIGHT=0.3 DOC_TOPK_RATIO=0.1 \
OUTPUT_DIR=/MURE-V2/code/MetaEmbed/colqwen_multigranularity/runs/exp_maxsim/formal_bimax_07_03 \
bash train_symmetric_maxsim.sh
```

Result:

- `[TODO]`

### Run C: strong reverse

- `query_score_weight = 0.5`
- `doc_score_weight = 0.5`
- `doc_topk_ratio = 0.1`

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/exp_maxsim
SCORE_MODE=bimax QUERY_SCORE_WEIGHT=0.5 DOC_SCORE_WEIGHT=0.5 DOC_TOPK_RATIO=0.1 \
OUTPUT_DIR=/MURE-V2/code/MetaEmbed/colqwen_multigranularity/runs/exp_maxsim/formal_bimax_05_05 \
bash train_symmetric_maxsim.sh
```

Result:

- `[TODO]`

## 4. Formal Evaluation Package

Formal evaluation script:

`eval_symmetric_maxsim.sh`

Evaluation template:

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/exp_maxsim
CHECKPOINT=/MURE-V2/code/MetaEmbed/colqwen_multigranularity/runs/exp_maxsim/[CHECKPOINT_DIR] \
SCORE_MODE=bimax QUERY_SCORE_WEIGHT=[Q_WEIGHT] DOC_SCORE_WEIGHT=[D_WEIGHT] DOC_TOPK_RATIO=0.1 \
bash eval_symmetric_maxsim.sh
```

Metrics to report:

- `ViDoRe v1`: `ndcg_at_5`
- `ViDoRe v2`: `ndcg_at_5`
- `MMEB`: `recall_at_5`

Formal results:

- `ViDoRe v1 = [TODO]`
- `ViDoRe v2 = [TODO]`
- `MMEB = [TODO]`

## 5. Test Length Analysis

All test groups are completed:

- `ViDoRe v1 = 10 / 10`
- `ViDoRe v2 = 7 / 7`
- `MMEB = 36 / 36`

Test figures:

- `/MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/exp_maxsim/plots/length_asymmetry_dumbbell.png`
- `/MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/exp_maxsim/plots/length_asymmetry_ratio.png`
- `/MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/exp_maxsim/plots/length_asymmetry_table.csv`

### 5.1 Most asymmetric test datasets overall

| Group | Dataset | target/query p50 ratio | target/query mean ratio |
|---|---|---:|---:|
| vidore_v1 | docvqa_subsampled | 193.78 | 183.86 |
| vidore_v2 | esg_reports_human_labeled_v2 | 178.48 | 175.97 |
| vidore_v1 | infovqa_subsampled | 166.97 | 156.05 |
| vidore_v1 | syntheticDocQA_artificial_intelligence_test | 163.50 | 160.03 |
| vidore_v1 | syntheticDocQA_government_reports | 163.50 | 158.33 |
| vidore_v1 | syntheticDocQA_energy | 158.55 | 149.19 |

Interpretation:

- `ViDoRe v1 / v2` are dominated by target-long asymmetry.
- `docvqa_subsampled` is the strongest target-long case in the test collection.
- This supports the diagnosis that reverse aggregation can be fragile in one-stage retrieval.

### 5.2 MMEB asymmetry pattern

MMEB is heterogeneous.

Strongest target-long MMEB subsets:

| Dataset | target/query p50 ratio |
|---|---:|
| MMEB-eval-Wiki-SS-NQ-beir | 128.20 |
| MMEB-eval-MSCOCO_t2i-beir | 19.53 |
| MMEB-eval-VisualNews_t2i-beir | 18.90 |
| MMEB-eval-WebQA-beir | 17.74 |
| MMEB-eval-EDIS-beir | 17.51 |

Strongest query-long MMEB subsets:

| Dataset | target/query p50 ratio |
|---|---:|
| MMEB-eval-N24News-beir | 0.0013 |
| MMEB-eval-ObjectNet-beir | 0.0016 |
| MMEB-eval-VizWiz-beir | 0.0017 |
| MMEB-eval-TextVQA-beir | 0.0017 |
| MMEB-eval-HatefulMemes-beir | 0.0019 |

Interpretation:

- MMEB mixes opposite length regimes.
- A single naive symmetric rule is unlikely to be optimal across all MMEB subsets.

## 6. Train Length Analysis

All train subsets are completed:

- `26 / 26`

Train figures:

- `/MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/exp_maxsim/plots/train_length_asymmetry_dumbbell.png`
- `/MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/exp_maxsim/plots/train_length_asymmetry_ratio.png`
- `/MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/exp_maxsim/plots/train_length_asymmetry_table.csv`

### 6.1 Most target-long train subsets

| Subset | positive/query p50 ratio | positive/query mean ratio |
|---|---:|---:|
| tevatron_colpali | 153.94 | 148.04 |
| visrag_ind | 129.80 | 122.69 |
| MSCOCO_t2i | 20.59 | 20.50 |
| VisualNews_t2i | 19.72 | 17.59 |
| WebQA | 17.83 | 17.97 |
| VisDial | 14.19 | 14.07 |
| InfoSeek_it2it | 1.28 | 1.26 |
| NIGHTS | 0.97 | 0.97 |

Interpretation:

- Large-scale corpora like `tevatron_colpali` and `visrag_ind` are strongly target-long.
- Retrieval-style training subsets are the closest to the test-set target-long regime.
- These subsets are likely the most relevant when analyzing one-stage reverse aggregation risk.

### 6.2 Most query-long train subsets

| Subset | positive/query p50 ratio | positive/query mean ratio |
|---|---:|---:|
| N24News | 0.0013 | 0.0015 |
| InfographicsVQA | 0.0017 | 0.0019 |
| ChartQA | 0.0019 | 0.0029 |
| DocVQA | 0.0021 | 0.0023 |
| TAT-DQA | 0.0021 | 0.0026 |
| HatefulMemes | 0.0026 | 0.0024 |
| SUN397 | 0.0029 | 0.0028 |
| A-OKVQA | 0.0032 | 0.0031 |

Interpretation:

- Many VQA / classification style training subsets are query-long rather than target-long.
- The training set is not homogeneous.
- This means the final scoring rule should not be reasoned about from only one task family.

## 7. Confirmed Diagnostic Findings

### 7.1 Small test-set scorer sweep

Result file:

- `/MURE-V2/code/MetaEmbed/colqwen_multigranularity/runs/eval/exp_maxsim/sweep_docvqa_cached.json`

| Variant | ndcg_at_5 | recall_at_5 | mrr |
|---|---:|---:|---:|
| query | 0.04487 | 0.07317 | 0.05049 |
| bimax_0.7_0.3 | 0.01732 | 0.02901 | 0.02808 |
| bimax_0.5_0.5 | 0.01384 | 0.02458 | 0.02420 |
| bimax_0.3_0.7 | 0.01079 | 0.02236 | 0.01978 |
| doc | 0.00575 | 0.01127 | 0.01662 |

Conclusion:

- all-token reverse averaging is harmful in one-stage retrieval
- larger reverse weight causes larger degradation

### 7.2 Rerank check

Result file:

- `/MURE-V2/code/MetaEmbed/colqwen_multigranularity/runs/eval/exp_maxsim/rerank_docvqa_cached.json`

Conclusion:

- reranking is more reasonable than using full BiMax as the one-stage score
- however, current reverse scoring still does not beat `query_full`
- weak reverse fusion remains the safer one-stage choice

## 8. TODO

- formal Run A result = `[TODO]`
- formal Run B result = `[TODO]`
- formal Run C result = `[TODO]`
- formal ViDoRe v1 metrics = `[TODO]`
- formal ViDoRe v2 metrics = `[TODO]`
- formal MMEB metrics = `[TODO]`
- optional smoke re-validation for the new doc-topk scorer = `[TODO]`
