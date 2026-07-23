# MARC-v1 3k Result and MaxSim-Margin Follow-up Plan

Updated: 2026-06-18

This note records the completed MARC-v1 formal run and the narrowed follow-up direction. The follow-up is constrained to an end-to-end, single-training-run method: no teacher model, no offline teacher distillation, no extra query-document joint LLM input, and no inference-time change to late interaction.

## 1. Run Identity

| Item | Value |
|---|---|
| Method | MARC-v1: MaxSim-Aware Residual Compression |
| Run name | `folder_homo_marc_v1_b160_160_160_3k` |
| Checkpoint | `experiments/exp_stagecompress/runs/folder_homo_marc_v1_b160_160_160_3k/checkpoint-3000` |
| Budgets | `160 160 160` |
| Steps | `3000` |
| MARC config | `MARC_ENABLED=1`, `MARC_WEIGHT=0.1`, `MARC_BETA=20` |
| Evaluation mode | full 3-set evaluation, 8 GPUs |
| Eval output | `experiments/exp_stagecompress/runs/folder_homo_marc_v1_b160_160_160_3k/eval/folder_homo_marc_v1_ckpt3000_full_8gpu_b160_160_160_bq16_bp24_bs64_workers0_20260618_101003` |

## 2. Training Signal

The run finished at `3000/3000` steps. Final logged values:

| Signal | Value |
|---|---:|
| final loss | 0.9207 |
| train_loss | 0.6771 |
| mrl_g1 | 0.1287 |
| mrl_g2 | 0.1136 |
| mrl_g3 | 0.1006 |
| marc_utility | 0.003641 |
| marc_stage_count | 11.125 |
| marc_weighted | 0.000364 |
| runtime | 53269s, about 14.8h |

Interpretation: MARC-v1 was active, but the auxiliary term was very small. At the final step, `marc_weighted / loss` is about `0.04%`. Even if the target is conceptually aligned with MaxSim, this scale is unlikely to materially change the main ranking behavior.

## 3. Full Evaluation Results

Overall metrics:

| Split | Metric | MARC-v1 ckpt3000 |
|---|---|---:|
| ViDoReV1 | avg nDCG@5 | 88.80 |
| ViDoReV2 | avg nDCG@5 | 57.55 |
| MMEB | avg Recall@1 | 75.15 |
| Macro average | average of the three rows | 73.84 |

ViDoReV1 detail:

| Dataset | nDCG@5 | Recall@1 | Recall@5 |
|---|---:|---:|---:|
| syntheticDocQA_energy | 96.00 | 95.00 | 97.00 |
| syntheticDocQA_healthcare_industry | 97.69 | 95.00 | 100.00 |
| syntheticDocQA_artificial_intelligence_test | 98.39 | 96.00 | 100.00 |
| syntheticDocQA_government_reports | 96.52 | 93.00 | 99.00 |
| infovqa_subsampled | 92.10 | 88.87 | 95.08 |
| docvqa_subsampled | 61.19 | 54.18 | 67.15 |
| arxivqa_subsampled | 87.47 | 82.00 | 92.00 |
| tabfquad_subsampled | 92.76 | 87.50 | 97.14 |
| tatdqa | 78.10 | 66.80 | 87.62 |
| shift_project | 87.80 | 76.00 | 97.00 |

ViDoReV2 detail:

| Dataset | nDCG@5 | Recall@1 | Recall@5 |
|---|---:|---:|---:|
| esg_reports_human_labeled_v2 | 66.89 | 45.38 | 71.70 |
| esg_reports_v2_multilingual | 57.12 | 24.55 | 60.78 |
| esg_reports_v2 | 56.80 | 28.05 | 57.63 |
| biomedical_lectures_v2 | 63.14 | 36.10 | 66.95 |
| biomedical_lectures_v2_multilingual | 59.90 | 34.04 | 63.64 |
| economics_reports_v2 | 52.28 | 5.29 | 30.95 |
| economics_reports_v2_multilingual | 46.73 | 5.17 | 25.40 |

MMEB detail:

| Dataset | Recall@1 | Recall@5 |
|---|---:|---:|
| MMEB-eval-VisDial-beir | 69.70 | 90.90 |
| MMEB-eval-WebQA-beir | 92.00 | 98.90 |
| MMEB-eval-VisualNews_t2i-beir | 67.10 | 83.30 |
| MMEB-eval-MSCOCO_t2i-beir | 71.80 | 92.90 |

## 4. Position Among Homogeneity Runs

The table below uses the existing homogeneity comparison snapshot and adds the completed MARC-v1 result.

| Method | ViDoReV1 | ViDoReV2 | MMEB | Avg |
|---|---:|---:|---:|---:|
| FolderHomo residual160 ckpt2500 | 89.20 | 61.60 | 75.55 | 75.45 |
| FolderHomo residual160 ckpt4000 | 89.34 | 60.28 | 76.42 | 75.35 |
| Gain residual_mass | 88.83 | 59.32 | 74.78 | 74.31 |
| Gain MMR | 88.96 | 58.76 | 75.45 | 74.39 |
| Gain residual_mass_mmr | 88.27 | 59.42 | 74.25 | 73.98 |
| Global-Guided | 89.08 | 58.44 | 73.75 | 73.76 |
| MARC-v1 ckpt3000 | 88.80 | 57.55 | 75.15 | 73.84 |
| Gain geo_coverage | 88.88 | 56.48 | 74.55 | 73.30 |

Relative to `FolderHomo residual160 ckpt2500`, MARC-v1 is close on ViDoReV1 and MMEB, but loses heavily on ViDoReV2:

| Split | Delta |
|---|---:|
| ViDoReV1 | -0.40 |
| ViDoReV2 | -4.05 |
| MMEB | -0.40 |
| Avg | -1.61 |

This means the failure is not a full model collapse. The weak point remains report-style retrieval, especially ESG/economics pages where exact text, numbers, table cells, and repeated layout anchors matter.

## 5. Why MARC-v1 Failed

1. The auxiliary loss was too weak. The final weighted MARC term was only about `0.00036`, roughly `0.04%` of the main logged loss.
2. The target was positive-only. It asked which positive tokens are touched by MaxSim, but not whether those tokens enlarge the margin against hard negatives.
3. The target reused the model's own current MaxSim winners. If the current winners are generic report/layout tokens, MARC-v1 can reinforce those choices.
4. Query-free homogeneity can remove repeated-looking OCR/table anchors. MaxSim often needs these anchors, especially on ViDoReV2 reports.
5. The optimization target was still a token-saliency proxy, not the final ranking margin `S(q,d+) > S(q,d-)`.

## 6. Narrowed Decision

Do not continue broad exploration. The next method should be a single line of work:

- Keep query and document encoded separately.
- Keep inference as late interaction: `score(q,d)=sum_i max_j sim(q_i,d_j)`.
- Do not use teacher models or offline distillation.
- Do not change dataset sampling for this step.
- Do not keep adding query-side scoring ablations; qaug/top-k/hit-penalty ablations already looked worse.
- Train end-to-end once, with the loss directly aware of MaxSim margin.

## 7. Implemented First Version: MARC-v2 Margin-Aware MaxSim Compression

MARC-v2 should replace positive-only utility with margin-aware utility. The key idea is simple: a source token is useful only if it helps the positive document beat the most confusing negatives under MaxSim.

For a query token `q_i`, define:

```text
s_i+ = max_j q_i^T d_j+
s_i- = max_{d-} max_k q_i^T d_k-
```

For a positive source token `x_j+` before compression, build a utility target using the query tokens whose positive evidence is threatened by negatives:

```text
v_i = softplus((s_i- + margin - s_i+) / tau)
p_ij = softmax_beta(q_i^T x_j+)
u_j = sum_i v_i * p_ij
L_marc2 = CE_or_KL(softmax(stage_logits), normalize(u))
L_total = L_rank + lambda * L_marc2
```

This keeps the training end-to-end. It does not require a teacher. The loss only uses the current batch's positive and negative documents, which are already present in contrastive training.

Implemented first-version behavior:

- Cache positive document source tokens as MARC-v1 already does.
- Use current compressed positive document embeddings and the hardest negative from in-batch documents plus the paired hard negative to compute `s_i+` and `s_i-`.
- Use `v_i` to weight the positive source-token target.
- Log `marc2_loss`, `marc2_weighted`, `marc2_stage_count`, `marc2_margin_violation`, `marc2_margin_gap`, `marc2_pos_token_score`, and `marc2_neg_token_score`.
- Use `MARC_MODE=margin` to enable this path; default `MARC_MODE=positive` preserves MARC-v1 behavior.

A more aggressive version can cache negative-document source tokens and explicitly suppress false-positive negative tokens, but that should not be the first formal run because it expands code and memory risk.

Initial command-line configuration:

```bash
MARC_ENABLED=1 MARC_MODE=margin MARC_WEIGHT=0.02 MARC_BETA=20 MARC_MARGIN=0.02 MARC_TAU=0.05
```

The initial `MARC_WEIGHT=0.02` is conservative because MARC-v2 uses a stronger KL reduction than MARC-v1. Use the quick gate to adjust the weight so the auxiliary/main ratio is visible but not dominant.

## 8. Minimal Verification Plan

1. Smoke run: 50 steps on 8 GPUs to verify shapes, nonzero `marc2_stage_count`, finite loss, and auxiliary/main loss ratio.
2. Early run: 300 or 500 steps, then run `experiments/exp_stagecompress/analysis/run_quick_gate.sh`.
3. Formal run: one 8-GPU, 3k-step run with `BUDGETS=160 160 160` only if the quick gate is healthy.
4. Evaluation: same full 3-set evaluation as MARC-v1 at `checkpoint-3000`.
5. Success criterion: ViDoReV2 should recover most of the MARC-v1 gap. The practical target is to beat MARC-v1 clearly and approach or surpass residual160 on macro average.

## 9. Current MARC-v2 Verification Log

### 9.1 50-step sanity run

| Item | Value |
|---|---|
| Run name | `folder_homo_marc_v2_margin_b160_160_160_sanity50_20260618_1324` |
| Checkpoint | `experiments/exp_stagecompress/runs/folder_homo_marc_v2_margin_b160_160_160_sanity50_20260618_1324/checkpoint-50` |
| Steps | `50` |
| Budgets | `160 160 160` |
| Config | `MARC_ENABLED=1`, `MARC_MODE=margin`, `MARC_WEIGHT=0.02`, `MARC_BETA=20`, `MARC_MARGIN=0.02`, `MARC_TAU=0.05` |

Final 50-step training signal:

| Signal | Value |
|---|---:|
| loss | 3.4595 |
| mrl_g1 | 1.1429 |
| mrl_g2 | 1.1279 |
| mrl_g3 | 1.1329 |
| marc2_loss | 1.7509 |
| marc2_weighted | 0.0350 |
| marc2_weighted / loss | 1.01% |
| marc2_stage_count | 10.875 |
| marc2_margin_violation | 0.0957 |
| marc2_margin_gap | 0.0708 |

Interpretation: the MARC-v2 path is active and numerically stable. The auxiliary signal is no longer negligible: `marc2_weighted / loss` is about `1.01%`, compared with about `0.04%` for MARC-v1 at the end of the 3k run.

### 9.2 50-step quick gate

Quick gate output:

`experiments/exp_stagecompress/runs/folder_homo_marc_v2_margin_b160_160_160_sanity50_20260618_1324/eval/quick_gate_sanity50_20260618_1437`

| Split | Metric | Value |
|---|---|---:|
| ViDoReV1 | avg nDCG@5 | 0.7308 |
| ViDoReV2 | avg nDCG@5 | 0.0471 |
| MMEB | avg Recall@1 | 0.3125 |

Gate status: `fail`, due to low ViDoReV2 and MMEB smoke metrics.

Interpretation: this is not enough evidence to discard MARC-v2, because the checkpoint has only 50 training steps. The useful signal from this run is implementation sanity: the loss is finite, the margin-aware auxiliary is nonzero, and its scale is in the intended 1%-level range.

### 9.3 Active 300-step quick run

Started and completed a more meaningful early validation run:

| Item | Value |
|---|---|
| Run name | `folder_homo_marc_v2_margin_b160_160_160_quick300_20260618_1442` |
| Output dir | `experiments/exp_stagecompress/runs/folder_homo_marc_v2_margin_b160_160_160_quick300_20260618_1442` |
| Training tmux session | `marc_v2_q300` |
| Auto quick-gate tmux session | `marc_v2_q300_watch` |
| Steps | `300` |
| Save steps | `100` |
| Config | `MARC_ENABLED=1`, `MARC_MODE=margin`, `MARC_WEIGHT=0.02`, `MARC_BETA=20`, `MARC_MARGIN=0.02`, `MARC_TAU=0.05` |

Training completed at `2026-06-18 16:48 CST` and saved `checkpoint-100`, `checkpoint-200`, and `checkpoint-300`.

Final 300-step training signal:

| Signal | Value |
|---|---:|
| final loss | 1.3904 |
| train_loss | 2.1823 |
| mrl_g1 | 0.5102 |
| mrl_g2 | 0.5130 |
| mrl_g3 | 0.4705 |
| marc2_loss | 1.9569 |
| marc2_weighted | 0.0391 |
| marc2_weighted / loss | 2.81% |
| marc2_stage_count | 11.250 |
| marc2_margin_violation | 0.0481 |
| marc2_margin_gap | -0.0265 |

Interpretation: the optimization path itself is healthy. The auxiliary term is visible but not dominant, and the margin gap becomes negative by the end, meaning positive token scores are higher than negative token scores on the logged batches.

### 9.4 300-step quick-gate results

Auto quick gate evaluated `checkpoint-100`, `checkpoint-200`, and `checkpoint-300` with the same smoke setting:

`EVAL_MAX_QUERIES=16`, `EVAL_MAX_CORPUS=96`, `BUDGETS=160 160 160`

| Checkpoint | ViDoReV1 avg nDCG@5 | ViDoReV2 avg nDCG@5 | MMEB avg Recall@1 | Gate |
|---|---:|---:|---:|---|
| checkpoint-100 | 0.9688 | 0.1145 | 0.6250 | fail |
| checkpoint-200 | 1.0000 | 0.1006 | 0.6250 | fail |
| checkpoint-300 | 1.0000 | 0.1096 | 0.6250 | fail |

All three checkpoints fail the current gate for the same reason: ViDoReV2 smoke performance is far below the `0.45` threshold. MMEB is acceptable under this small smoke gate, and ViDoReV1 is very high. However, the baseline calibration below shows that this ViDoReV2 smoke setting is not a valid hard filter.

Calibration run:

| Method | Checkpoint | ViDoReV1 avg nDCG@5 | ViDoReV2 avg nDCG@5 | MMEB avg Recall@1 |
|---|---|---:|---:|---:|
| residual160 baseline | checkpoint-2500 | 1.0000 | 0.1117 | 1.0000 |
| MARC-v2 | checkpoint-300 | 1.0000 | 0.1096 | 0.6250 |

This baseline was evaluated with the same quick-gate setting. Since the known strong residual160 checkpoint also gets only `0.1117` on the ViDoReV2 smoke split, the current ViDoReV2 quick gate is not a reliable hard rejection signal. The small `16 query / 96 corpus` smoke subset is useful for verifying that the evaluation pipeline runs, but it is too noisy or too harsh for judging ViDoReV2 quality.

### 9.5 ViDoReV2 partial comparison with a valid baseline

Because the `16 query / 96 corpus` ViDoReV2 quick gate was not discriminative, a larger ViDoReV2 partial comparison was run with:

`EVAL_MAX_QUERIES=64`, `EVAL_MAX_CORPUS=0`, `BUDGETS=160 160 160`

Output directory:

`experiments/exp_stagecompress/runs/vidore_v2_partial_compare_marcv2_vs_res160_20260618_1752`

| Model | Checkpoint | Avg nDCG@5 |
|---|---|---:|
| residual160 baseline | `checkpoint-2500` | 0.6360 |
| MARC-v2 | `checkpoint-300` | 0.5633 |
| Delta | - | -0.0726 |

Dataset-level detail:

| Dataset | residual160 nDCG@5 | MARC-v2 nDCG@5 | Delta | residual160 R@1 | MARC-v2 R@1 |
|---|---:|---:|---:|---:|---:|
| esg_reports_human_labeled_v2 | 0.7248 | 0.5782 | -0.1466 | 0.4849 | 0.3263 |
| biomedical_lectures_v2 | 0.6095 | 0.5766 | -0.0328 | 0.3338 | 0.3091 |
| economics_reports_v2 | 0.5736 | 0.5351 | -0.0385 | 0.0591 | 0.0654 |

This partial comparison is more reliable than the quick gate because the residual160 baseline recovers a normal ViDoReV2 level. Under this setting, MARC-v2 is clearly below the current best mainline, with the largest drop on `esg_reports_human_labeled_v2`.

Updated interpretation:

1. Do not launch a formal 3k MARC-v2 run from this checkpoint family yet.
2. The failure is unlikely to be a pure code-path failure, because training is stable and MaxSim-margin signals are active.
3. The larger ViDoReV2 partial evaluation gives a real negative signal: MARC-v2 currently hurts report-style retrieval relative to residual160.
4. The next algorithmic change should specifically address report/OCR/table anchor preservation instead of only using local in-batch MaxSim margin.

## 10. Paper Story If It Works

The current negative result is still useful: query-free or positive-only homogeneity improves token organization but does not necessarily improve late-interaction ranking. The refined contribution becomes:

> Since late interaction ranks documents by per-query-token MaxSim margins, document compression should be trained to preserve tokens that increase the positive-negative MaxSim margin, rather than tokens that are merely salient inside the positive document.

This is a clean story: same inference mechanism, same single training run, but a training objective better matched to the retrieval endpoint.

## 11. References Used For Reasoning

- ColBERT: late interaction and MaxSim retrieval. https://arxiv.org/abs/2004.12832
- ColBERTv2: lightweight late interaction with residual compression and supervision. https://arxiv.org/abs/2112.01488
- COIL: contextualized exact lexical matching. https://arxiv.org/abs/2104.07186
- SPLADE: sparse lexical expansion signals for retrieval. https://arxiv.org/abs/2107.05720
