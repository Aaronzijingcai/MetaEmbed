# Symmetric MaxSim Experiments

This directory contains an isolated experiment path for one-stage retrieval with query-dominant MaxSim and weak reverse-side assistance.

## Motivation

The default ColBERT/MaxSim score in the current codebase is query-centric:

- compute token similarities between query and target
- for each query token, take the max over target tokens
- sum or mean over query tokens

The current diagnosis shows that fully symmetric scoring is too aggressive for one-stage retrieval. Therefore the updated design keeps query-to-doc as the main score and only uses a weak doc-to-query auxiliary branch on top-k document tokens.

## BiMax formulation

We borrow the key idea from `reference/EmbDA` and adapt it to token-level retrieval:

- query-to-doc branch: for each query token, max over doc tokens, then mean over valid query tokens
- doc-to-query branch: for each doc token, max over query tokens, then keep only top-k doc tokens before averaging
- fused score:

`score = (w_q * score_q2d + w_d * score_d2q_topk) / (w_q + w_d)`

Special cases:

- `--score-mode query`: recover the original asymmetric MaxSim
- `--score-mode doc`: reverse-only scoring
- `--score-mode bimax`: bidirectional fused scoring

## Files

- `symmetric_maxsim.py`: core scorer, distributed scorer patch, and training loss
- `train_symmetric_maxsim.py`: isolated training entrypoint
- `eval_symmetric_maxsim.py`: isolated evaluation entrypoint
- `train_symmetric_maxsim.sh`: recommended multi-GPU launcher
- `eval_symmetric_maxsim.sh`: recommended evaluation launcher
- `reference/EmbDA`: cloned reference repository

## Recommended protocol

1. Reproduce the original asymmetric baseline inside this experiment path.

Use:

`SCORE_MODE=query QUERY_SCORE_WEIGHT=1 DOC_SCORE_WEIGHT=0`

2. Run one-stage weak reverse fusion.

Recommended default:

`SCORE_MODE=bimax QUERY_SCORE_WEIGHT=0.9 DOC_SCORE_WEIGHT=0.1 DOC_TOPK_RATIO=0.1`

3. Sweep directional fusion weights.

Suggested sweep:

- `0.9 / 0.1`
- `0.7 / 0.3`
- `0.5 / 0.5`

4. Compare on all three suites with extra focus on MMEB.

Primary metric:

- ViDoRe v1/v2: `ndcg_at_5`
- MMEB: `recall_at_5`

## Example commands

Train one-stage weak reverse fusion:

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/exp_maxsim
bash train_symmetric_maxsim.sh
```

Train asymmetric baseline in the same code path:

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/exp_maxsim
SCORE_MODE=query QUERY_SCORE_WEIGHT=1 DOC_SCORE_WEIGHT=0 bash train_symmetric_maxsim.sh
```

Evaluate a checkpoint with symmetric 50/50:

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/exp_maxsim
CHECKPOINT=/MURE-V2/code/MetaEmbed/colqwen_multigranularity/runs/exp_maxsim/bimax_main/checkpoint-4000 \
SCORE_MODE=bimax QUERY_SCORE_WEIGHT=0.9 DOC_SCORE_WEIGHT=0.1 DOC_TOPK_RATIO=0.1 \
bash eval_symmetric_maxsim.sh
```
