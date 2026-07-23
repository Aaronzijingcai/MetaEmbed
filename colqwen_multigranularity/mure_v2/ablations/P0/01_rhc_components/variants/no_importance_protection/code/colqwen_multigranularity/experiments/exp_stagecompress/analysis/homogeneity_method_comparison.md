# Homogeneity Methods Comparative Analysis

Generated from completed full-eval JSON files on 2026-06-16. Metrics are ViDoRe nDCG@5 and MMEB recall@1, reported in percent.

## Overall Results

| Method | ViDoReV1 | ViDoReV2 | MMEB | Avg | Delta Avg vs residual160-2500 |
|---|---:|---:|---:|---:|---:|
| FolderHomo residual160 ckpt2500 | 89.20 | 61.60 | 75.55 | 75.45 | +0.00 |
| FolderHomo residual160 ckpt4000 | 89.34 | 60.28 | 76.42 | 75.35 | -0.10 |
| Global-Guided | 89.08 | 58.44 | 73.75 | 73.76 | -1.69 |
| Gain geo_coverage | 88.88 | 56.48 | 74.55 | 73.30 | -2.15 |
| Gain residual_mass | 88.83 | 59.32 | 74.78 | 74.31 | -1.14 |
| Gain MMR | 88.96 | 58.76 | 75.45 | 74.39 | -1.06 |
| Gain residual_mass_mmr | 88.27 | 59.42 | 74.25 | 73.98 | -1.47 |

## Largest Dataset-Level Regressions

| Split | Dataset | Residual160 | Avg delta of new methods | Worst method | Worst score |
|---|---|---:|---:|---|---:|
| vidore_v2 | esg_reports_human_labeled_v2 | 72.75 | -4.75 | Gain residual_mass_mmr | 64.31 |
| vidore_v2 | esg_reports_v2 | 61.56 | -4.40 | Gain geo_coverage | 53.59 |
| vidore_v2 | economics_reports_v2 | 58.42 | -3.76 | Gain geo_coverage | 51.15 |
| vidore_v2 | economics_reports_v2_multilingual | 54.29 | -3.74 | Gain geo_coverage | 48.71 |
| vidore_v2 | esg_reports_v2_multilingual | 58.92 | -3.03 | Gain geo_coverage | 51.09 |
| mmeb | MMEB-eval-VisDial-beir | 72.10 | -2.42 | Global-Guided | 69.10 |
| vidore_v2 | biomedical_lectures_v2 | 64.45 | -1.67 | Global-Guided | 60.00 |
| vidore_v1 | shift_project | 87.78 | -0.98 | Gain residual_mass_mmr | 84.14 |
| vidore_v1 | syntheticDocQA_government_reports | 97.02 | -0.73 | Gain geo_coverage | 94.20 |
| vidore_v1 | docvqa_subsampled | 61.69 | -0.64 | Gain residual_mass | 60.79 |
| mmeb | MMEB-eval-VisualNews_t2i-beir | 67.30 | -0.60 | Global-Guided | 65.80 |
| vidore_v1 | infovqa_subsampled | 93.02 | -0.60 | Gain residual_mass_mmr | 91.72 |

## Positive or Near-Neutral Cases

| Split | Dataset | Residual160 | Best new method | Best score | Delta |
|---|---|---:|---|---:|---:|
| vidore_v1 | arxivqa_subsampled | 87.81 | Gain residual_mass | 88.76 | +0.95 |
| vidore_v1 | tabfquad_subsampled | 92.08 | Gain MMR | 93.13 | +1.05 |
| vidore_v1 | syntheticDocQA_healthcare_industry | 97.89 | Gain residual_mass_mmr | 98.52 | +0.63 |
| vidore_v1 | tatdqa | 79.12 | Gain residual_mass | 79.62 | +0.51 |
| vidore_v1 | syntheticDocQA_artificial_intelligence_test | 98.89 | Gain residual_mass_mmr | 99.13 | +0.24 |
| mmeb | MMEB-eval-MSCOCO_t2i-beir | 71.40 | Gain MMR | 71.80 | +0.40 |
| vidore_v1 | syntheticDocQA_energy | 96.65 | Global-Guided | 96.52 | -0.12 |
| vidore_v2 | biomedical_lectures_v2_multilingual | 60.83 | Gain MMR | 61.66 | +0.83 |
| mmeb | MMEB-eval-WebQA-beir | 91.40 | Gain MMR | 91.70 | +0.30 |
| vidore_v1 | infovqa_subsampled | 93.02 | Gain residual_mass | 92.98 | -0.04 |

## Interpretation

1. The main regression is not uniform. ViDoReV1 changes are small, while the largest drops concentrate on ViDoReV2 report-style datasets, especially ESG and economics reports. Compared with residual160-2500, the new methods lose about 3-5 points on `esg_reports_human_labeled_v2`, `esg_reports_v2`, and `economics_reports_v2`. This pattern suggests that the issue is tied to long report pages with dense OCR, tables, and layout-specific evidence rather than a general failure of the backbone.

2. Residual160 remains strong because it keeps a simple coarse-to-fine residual path: `G1` preserves page-level context, while `R2/R3` keep additional evidence only when it is not already covered by coarser stages. The later variants add extra objectives such as spatial coverage, dynamic crop mass, or MMR diversity. These objectives are plausible, but they are query-free. They can preserve visually diverse tokens that are not useful for the current query, and they can suppress repeated-looking OCR/layout tokens that are actually semantically important.

3. The ViDoReV2 drops are strongest on ESG/economics report datasets. The sampled query/corpus structure shows that these queries ask for specific report facts and the corpus contains page images from long PDFs. Such tasks depend on exact text, tables, headings, dates, and nearby layout context. A coverage or MMR objective may treat similar text blocks, repeated table rows, or boilerplate report sections as redundant, but MaxSim retrieval can need those repeated lexical anchors to match the query.

4. MMR has the best gain-mode average, mainly because it preserves MMEB better than the other gain variants, but it still loses ViDoReV2. This means generic diversity helps image-style retrieval more than report QA-style retrieval. Conversely, residual_mass helps ViDoReV2 more than geo_coverage/MMR, but it reduces MMEB. The combined `residual_mass_mmr` does not improve over either component, which suggests the two objectives can conflict: dynamic mass allocation selects where to spend budget, while MMR changes which tokens survive inside that budget. Combining them can remove the dense repeated evidence that residual_mass alone was trying to keep.

5. The most likely algorithmic defect is objective mismatch. The added homogeneity methods optimize document-side token cleanliness without seeing the query. Retrieval quality, however, is determined by query-document MaxSim. Query-free diversity and coverage are therefore only proxies. When the proxy is wrong, it reduces token redundancy but also removes repeated anchors that MaxSim uses for ranking.



## Sensitive-Set Sample Signal

The most affected datasets also differ structurally from the more stable ViDoReV1 subsets. They contain long-report pages, multi-page relevance, and queries that ask for exact company facts, percentages, disclosure IDs, or cross-page comparisons.

| Dataset | Queries | Corpus pages | Qrels | Rel/query | Example query |
|---|---:|---:|---:|---:|---|
| esg_reports_human_labeled_v2 | 52 | 1538 | 128 | 2.46 | Who is responsible for integrating climate consideration into Jack in the box governance? |
| esg_reports_v2 | 57 | 1538 | 222 | 3.89 | What are the RBI brands mentioned in the report? |
| economics_reports_v2 | 58 | 452 | 907 | 15.64 | How do the fiscal challenges in small states compare to those in larger economies? |
| economics_reports_v2_multilingual | 232 | 452 | 3628 | 15.64 | How do the fiscal challenges in small states compare to those in larger economies? |
| biomedical_lectures_v2 | 160 | 1016 | 515 | 3.22 | What are the specific outcomes of using autologous chondrocyte implantation in canine studies? |
| arxivqa_subsampled | 500 | 500 | 500 | 1.00 | Based on the graph, what is the impact of correcting for fspec not equal to 1 on the surface... |
| tabfquad_subsampled | 280 | 70 | 280 | 1.00 | Quelles etaient les principales composantes du benefice net par action... |
| docvqa_subsampled | 451 | 500 | 500 | 1.11 | What is the dividend payout in 2012? |

This supports the same explanation as the metric table. The vulnerable sets are not simply harder; they require ranking among many pages from the same report family where negatives share similar boilerplate and visual layout. Query-free homogeneity can improve token diversity but reduce the margin between true pages and nearby hard negatives if it removes repeated table rows, section headers, company names, dates, or disclosure numbers.

## Scoring-Ablation Signal

Eval-only scoring ablations were run on `FolderHomo residual160 checkpoint-2500` to check whether the observed badcases mainly come from query-side MaxSim noise. The results do not support simply removing query augmentation or replacing MaxSim-sum with top-k mean.

| Scoring variant | ViDoReV1 | ViDoReV2 | MMEB | Avg |
|---|---:|---:|---:|---:|
| baseline_qaug10 | 89.17 | 60.50 | 75.48 | 75.05 |
| qaug0 | 88.37 | 59.16 | 73.15 | 73.56 |
| qaug2 | 88.60 | 59.61 | 74.08 | 74.09 |
| qaug0_trim_suffix8 | 88.36 | 59.02 | 73.15 | 73.51 |
| qaug0_topk8_mean | 68.09 | 28.84 | 19.88 | 38.94 |
| qaug0_hitpenalty | 88.37 | 59.16 | 71.65 | 73.06 |

This changes the badcase interpretation. Repeated query augmentation is not merely harmful prompt noise; in the current scorer it appears to stabilize query-token matching, especially on MMEB. Top-k mean and hit-concentration penalties are too aggressive because they discard much of the MaxSim evidence accumulated over query tokens. Therefore, the main issue is more likely document-side: query-free homogeneity objectives can remove or downweight repeated OCR/layout anchors that are useful for MaxSim ranking. The sensitive failure mode is not redundant tokens in general, but removing the wrong redundant-looking tokens.

## Working Hypotheses To Verify

- H1: On ESG/economics failures, positive pages lose margin because useful OCR/table tokens are merged or de-emphasized, while hard negatives retain generic report tokens that match the query prompt.
- H2: Query augmentation should not be removed blindly. The qaug0/qaug2 ablations underperform qaug10, so query-side tokens likely provide useful matching mass; the remaining badcases are more consistent with document-side compression removing useful anchors.
- H3: MMR reduces same-stage token duplication, but some apparent duplication corresponds to repeated semantically meaningful OCR patterns in report pages.
- H4: Spatial coverage over-preserves visually distributed but semantically irrelevant regions, hurting fact-centric report retrieval.

## Recommended Next Step

Do not expand training variants further. Use the best residual160 model as the main homogeneity result. Do not expand query-side scoring variants based on qaug removal, top-k mean, or hit penalties because the completed ablations are worse. If more evidence is needed for the paper, run a small qualitative visualization on ESG/economics report pages: compare which OCR/table/layout tokens are kept by residual160, geo_coverage, residual_mass, and MMR on the same failed query. This should verify whether useful repeated anchors are being treated as redundant by query-free homogeneity objectives.

## Duplicate-Token MaxSim Probe

Motivation: recent homogeneity variants suggest that repeated-looking document tokens are not always harmful. In report-style ViDoReV2 datasets, repeated OCR/table/header/date/company tokens may provide the exact anchors that MaxSim uses to separate a positive page from visually similar hard negatives.

Diagnostic script:

```bash
bash experiments/exp_stagecompress/analysis/run_maxsim_duplicate_probe.sh
```

Current probe target:

```text
Checkpoint: folder_homo_residual160 checkpoint-2500
Datasets: esg_reports_human_labeled_v2, economics_reports_v2
Metric idea: identify near-duplicate compressed document tokens, then measure how much positive MaxSim score and positive-vs-hard-negative margin comes from query tokens whose winner is a near-duplicate document token.
```

Reading rule:

```text
If positive_score_duplicate_fraction is high and duplicate_margin_component is positive, then repeated-looking tokens are not merely redundant noise. They are part of the evidence used by MaxSim ranking, so homogeneity objectives that aggressively remove duplicates can hurt report-style retrieval.
```

Smoke signal on two ESG queries already showed positive_score_duplicate_fraction around 0.55-0.60 and positive duplicate margin contribution against the hardest sampled negative. A larger probe is running in tmux session dup_probe_v2; fill the final numbers after completion.

### Completed Probe Result: residual160 ckpt2500

Run directory:

```text
experiments/exp_stagecompress/runs/maxsim_duplicate_probe_v2_reports_fast_20260618_210901
```

Setting:

```text
Datasets: esg_reports_human_labeled_v2, economics_reports_v2
Queries: 8 per dataset, 16 total
Candidate negatives: 48 per query
Hard negatives used for margin decomposition: top 3 per query
Duplicate definition: compressed document tokens with cosine similarity >= 0.88 to another token
Text/special prefix: skipped; probe focuses on compressed visual evidence
```

Main result:

| Signal | Overall | ESG | Economics |
|---|---:|---:|---:|
| Positive MaxSim hits on duplicate tokens | 66.62% | 71.40% | 61.85% |
| Positive MaxSim score from duplicate tokens | 65.52% | 68.71% | 62.33% |
| Duplicate-token margin component | +4.12 | +6.98 | +1.26 |
| Non-duplicate-token margin component | -1.04 | -1.31 | -0.77 |

Interpretation:

```text
This supports the key hypothesis: in ViDoReV2 report-style retrieval, near-duplicate compressed visual tokens are not simply noise. A large fraction of the positive MaxSim evidence lands on these repeated-looking tokens, and the duplicate-token component contributes positively to the positive-vs-hard-negative margin. Therefore, homogeneity objectives that aggressively suppress duplicate-looking OCR/table/layout evidence can reduce the exact anchors used by MaxSim, especially on ESG/economics pages.
```

Method implication:

```text
The next compression objective should not be generic de-duplication. It should be residual compression with anchor preservation: repeated tokens may be compressed only when their MaxSim/margin contribution is low, while repeated OCR/table/header/number-like anchors should be protected.
```

## Duplicate Quota Sweep

Purpose: test whether the retrieval quality prefers complete duplicate removal, full retention, or an intermediate quota. This is an eval-only diagnostic on residual160 checkpoint-2500; it does not retrain the model.

Run directory:

```text
experiments/exp_stagecompress/runs/duplicate_quota_sweep_v2_reports_20260619_020151
```

Setting:

```text
Datasets: esg_reports_human_labeled_v2, economics_reports_v2
Queries: 8 per dataset
Candidate corpus: first 128 pages plus qrels positives
Duplicate definition: compressed document tokens with cosine similarity >= 0.88
Quota: maximum retained tokens per near-duplicate cluster
```

Results:

| Quota | Macro nDCG@5 | Macro Recall@1 | Macro Recall@5 | Mean Doc Tokens |
|---|---:|---:|---:|---:|
| 1 | 0.7937 | 0.3957 | 0.6939 | 202.6 |
| 2 | 0.8337 | 0.4082 | 0.7021 | 264.1 |
| 4 | 0.8278 | 0.4082 | 0.6952 | 319.9 |
| all | 0.8244 | 0.4082 | 0.7006 | 503.0 |

Dataset-level reading:

| Dataset | Best quota | Reading |
|---|---|---|
| ESG human-labeled | all, but only +0.0026 nDCG over quota 1/2/4 | Duplicate retention gives a small benefit; aggressive quota is not catastrophic on this tiny subset. |
| Economics reports | quota 2 | Hard quota 1 loses clearly, while full retention is also below quota 2. This directly supports a balance point. |

Interpretation:

```text
The sweep supports the balancing story. Complete duplicate removal is too aggressive, especially on economics_reports_v2, where quota=1 drops nDCG@5 to 0.5981 while quota=2 reaches 0.6782. Full retention is not always optimal either: economics quota=2 is above all-token retention by +0.0213 nDCG@5 while using about 46% fewer document tokens. Therefore, the right objective is not generic de-duplication or full retention, but controlled preservation of useful repeated anchors.
```

Method implication:

```text
A practical next method should learn or approximate a dynamic duplicate quota. Repeated OCR/table/layout anchors should keep multiple representatives when they contribute to MaxSim evidence, while generic duplicate background/layout tokens can be merged more aggressively.
```

## Anchor-Aware Duplicate Quota Sweep

Purpose: test whether preserving the repeated tokens that are actually used by MaxSim is better than a generic fixed duplicate quota. This is an oracle-style eval-only diagnostic: it uses query-side MaxSim hits to rank tokens inside each duplicate cluster, so it is not a valid offline indexing method by itself. Its role is to test whether anchor-aware retention is worth turning into a trainable query-free objective.

Run directory:

```text
experiments/exp_stagecompress/runs/duplicate_quota_sweep_anchor_v2_reports_20260619_021815
```

Setting:

```text
Datasets: esg_reports_human_labeled_v2, economics_reports_v2
Queries: 8 per dataset
Candidate corpus: first 128 pages plus qrels positives
Duplicate definition: compressed document tokens with cosine similarity >= 0.88
anchor2 / anchor4: keep top-k tokens in each duplicate cluster by accumulated MaxSim hit strength from the evaluated queries
```

Results:

| Quota | Macro nDCG@5 | Macro Recall@1 | Macro Recall@5 | Mean Doc Tokens |
|---|---:|---:|---:|---:|
| 2 | 0.7628 | 0.3868 | 0.6684 | 264.1 |
| all | 0.8244 | 0.4082 | 0.7006 | 503.0 |
| anchor2 | 0.8116 | 0.4082 | 0.6859 | 264.1 |
| anchor4 | 0.8193 | 0.3957 | 0.6929 | 319.9 |

Dataset-level reading:

| Dataset | quota2 | all | anchor2 | anchor4 |
|---|---:|---:|---:|---:|
| ESG human-labeled nDCG@5 | 0.9893 | 0.9919 | 0.9893 | 0.9919 |
| Economics reports nDCG@5 | 0.5363 | 0.6569 | 0.6338 | 0.6468 |

Interpretation:

```text
Anchor-aware retention is useful. On economics_reports_v2, anchor2 recovers most of the gap between generic quota=2 and full retention while using the same token budget as quota=2: 0.6338 vs 0.5363 nDCG@5 at about 270 tokens. anchor4 gets closer to full retention with about 323 tokens. This suggests that the key is not just how many duplicate tokens are kept, but which repeated tokens are preserved.
```

Caveat:

```text
This diagnostic is query-aware, so it should not be used directly as the deployment-time compression policy. The trainable method should approximate this behavior offline by learning an anchor-preservation score from document-side evidence and training-time MaxSim/margin signals.
```

Method implication:

```text
The next trainable variant should add an anchor-preservation objective to FolderHomo: tokens inside duplicate clusters should be protected when they repeatedly receive MaxSim hits or contribute positive-negative margin during training. Generic duplicate clusters without anchor evidence can still be merged aggressively.
```

## Anchor-Balance 3k Attempt: Early Stop Result

Date: 2026-06-20/21.

Purpose: turn the duplicate-token diagnostic into a trainable FolderHomo objective. The intended rule was:

```text
High duplicate + high MaxSim/margin contribution -> protect.
High duplicate + low contribution -> allow compression.
Low duplicate + contribution -> keep normally.
```

Implementation setting:

```text
Method: MARC anchor_balance
Budget: 160/160/160
Training: quick300 warm-start + 2700 additional steps, equivalent to about 3k total update steps
Checkpoint used for evaluation: final run root of folder_homo_anchor_balance_b160_160_160_warm300_plus2700_20260619_121805
Eval setting: ViDoReV2 partial 3-set, max 64 queries, 8 GPUs, same config as residual160 ckpt3000 partial eval
```

Operational note:

```text
The first eval attempt pointed to checkpoint-2700 and failed before scoring because the warm-start checkpoint directory did not contain adapter_config.json. The final run root contained adapter_config.json, adapter_model.safetensors, and folder_homo.pt, so the target eval was restarted from the run root. Baseline residual160 ckpt3000 was already completed and reused.
```

Measured result:

| Dataset | residual160 ckpt3000 nDCG@5 | anchor_balance warm300+2700 nDCG@5 | Delta |
|---|---:|---:|---:|
| esg_reports_human_labeled_v2 | 0.70239 | 0.48841 | -0.21398 |
| biomedical_lectures_v2 | 0.60931 | 0.56786 | -0.04145 |
| economics_reports_v2 | 0.56956 | 0.52040 | -0.04916 |
| Average | 0.62709 | 0.52556 | -0.10153 |

The full partial JSON was eventually written before the tmux job was stopped. The 3-set partial result confirms that the failure is not limited to one ESG subset: the average nDCG@5 is 0.52556, which is -0.10153 below residual160 ckpt3000. It is also almost unchanged from the quick300 result, whose partial average was 0.52758. This is sufficient to classify the trainable anchor-balance variant as failed under the current formulation, and it weakens the hypothesis that the quick300 result only failed because of insufficient training.

Interpretation:

1. The diagnostic conclusion still stands: repeated tokens can be useful MaxSim anchors. However, the current trainable objective does not successfully preserve the right anchors.
2. The auxiliary loss is still an indirect proxy. It reweights token targets during training, but inference remains query-free compression. This can over-protect duplicate tokens that look useful under minibatch positives/negatives while still failing to preserve the exact page-level OCR/table anchors needed by ESG ranking.
3. Warm-starting from a weak 300-step anchor-balance adapter and training 2700 more steps did not recover residual160 behavior. The poor ESG result suggests the auxiliary objective perturbs the learned residual compression policy rather than only adding a small correction.
4. The most important practical lesson is not that duplicate balancing is wrong, but that making it a learned end-to-end objective is fragile. The evidence is stronger as an analysis/interpretability finding than as a new mainline method at this stage.

## Current Mainline Decision

Use `FolderHomo residual160` as the homogeneity mainline. It remains the strongest completed method and has stable full-eval results:

| Model | ViDoReV1 | ViDoReV2 | MMEB | Avg |
|---|---:|---:|---:|---:|
| residual160 ckpt2500 | 89.20 | 61.60 | 75.55 | 75.45 |
| residual160 ckpt4000 | 89.34 | 60.28 | 76.42 | 75.35 |

For the paper, the homogeneity section should be framed as follows:

```text
FolderHomo residual compression is the reliable compression mechanism. Additional query-free diversity or coverage objectives do not consistently improve retrieval because MaxSim relies on repeated OCR/layout anchors in long-report pages. The failed variants are useful ablations: they show that naive de-duplication, coverage, MMR-style diversity, and the current anchor-balance proxy can remove or distort useful repeated evidence. The duplicate-token probes explain why residual compression works better than aggressive homogeneity regularization.
```

Recommended next actions:

1. Stop training new homogeneity variants unless there is a very small diagnostic-only reason.
2. Keep residual160 ckpt2500 as the main homogeneity result, with ckpt4000 as a training-length sensitivity point.
3. Use duplicate-token MaxSim probe and quota sweep as analysis evidence, not as a claim that anchor_balance improves performance.
4. If the paper needs a positive contribution beyond residual160, focus on the original MURE/X-VisEmbed multi-granularity design and use homogeneity variants as ablations/limitations.
5. For future optimization, prefer eval-time or lightweight post-hoc diagnostics before any 8-GPU full training. The current quick300 and partial eval pipeline should remain the gate before full runs.

## Concrete Failure Examples and Next Experiment Direction

### Example 1: repeated anchors are useful, not just redundant

In `esg_reports_human_labeled_v2`, query 3 asks:

```text
Which projects does Shake Shack endorse to support the LGBTQ+ community?
```

The relevant pages are all from the same Shake Shack 2023 ESG report: corpus IDs 1197, 1200, 1206, 1214, and 1218. The answer depends on repeated report anchors such as company name, Pride campaign text, LGBTQ+ terms, The Trevor Project, donation descriptions, and nearby layout context. In the duplicate-token MaxSim probe on residual160 ckpt2500, the positive page for this query had:

```text
positive_score_duplicate_fraction = 0.9259
positive_hit_duplicate_fraction   = 0.9231
hard-negative duplicate_margin_component examples = +15.16, +12.80, +15.93
hard-negative nonduplicate_margin_component examples = -8.73, -5.67, -8.30
```

This means most useful MaxSim evidence for this example lands on near-duplicate compressed visual tokens. The repeated tokens are not simply noise; they are the lexical/layout anchors that distinguish the correct Shake Shack pages from other visually similar report pages.

### Example 2: which duplicate tokens survive changes ranking

In `economics_reports_v2`, query 5 asks:

```text
What are the differences in financial sector resilience between lower-income and higher-income EMDEs?
```

Relevant pages include corpus IDs 111, 115, 121, 122, 134, 136, and 299. The answer requires comparing repeated terms and table/chart anchors such as lower-income EMDEs, higher-income EMDEs, financial sector risk, resilience, supervisory frameworks, and bank capital buffers. In the duplicate quota sweep, the same query changed substantially under different duplicate retention policies:

| Policy | nDCG@5 | Top-5 retrieved corpus IDs |
|---|---:|---|
| quota=2 generic | 0.5148 | 108, 111, 115, 117, 134 |
| all duplicates | 0.8688 | 111, 115, 121, 122, 123 |
| anchor2 oracle | 0.8539 | 111, 115, 121, 108, 134 |
| anchor4 oracle | 0.8688 | 111, 115, 121, 122, 108 |

The generic quota policy loses because it keeps too few or the wrong representatives inside duplicate clusters. The oracle-style anchor policy recovers most of the ranking while keeping about the same token budget as quota=2. This supports the analysis that duplicate control must preserve query-useful anchors. It does not imply that the current trainable anchor-balance loss works; the 3k anchor-balance training result shows that our current proxy fails to learn this behavior robustly.

### Why the trainable anchor-balance attempt failed

The likely failure is an objective mismatch rather than a simple lack of training steps. The quick300 partial average was 0.52758, and the warm300+2700 result was 0.52556, both far below residual160 ckpt3000 at 0.62709. Additional training did not recover the baseline.

The current auxiliary objective tries to protect duplicated tokens using training-time MaxSim/margin signals, but deployment-time compression remains query-free. This creates three problems:

1. **Proxy leakage across queries.** A token that is useful for one training query may be protected globally, but a query-free document encoder cannot know which duplicate anchor will matter for a future query.
2. **Residual-policy perturbation.** Residual160 already learns a stable coarse-to-fine compression policy. The auxiliary loss changes the target distribution and can disturb the base residual behavior even when its scalar weight is small.
3. **Hard-negative mismatch.** ESG/economics failures often involve pages from the same report family. Generic minibatch negatives may not match the exact same-report hard negatives that determine ViDoReV2 ranking, so the learned protection score does not transfer cleanly.

### Direction decision

Do not spend the next 8-GPU slot on another homogeneity variant. The homogeneity result is already sufficient for the paper if framed correctly: residual160 is the positive mainline; coverage/MMR/MARC/anchor-balance are ablations showing that query-free diversity and duplicate suppression can damage MaxSim evidence.

The next 8-GPU experiment should move to the other active paper line: **stage-interleaved learnable tokens**. This directly targets the remaining limitation that MaxSim is sensitive to token-count imbalance and multi-vector noise, and it also matches the existing TODO about learnable-token experiments and token-number-vs-performance curves.

Planned next formal run:

```text
Method: Stage-Interleaved Learnable Tokens
Priority: P1 / Q2-T2 main setting
Query inserted tokens: 2,4,8
Document inserted tokens: 8,16,32
Cumulative MRL groups: 2,8; 6,24; 14,56
Orthogonality: 0.0 first, because this should be the controlled baseline before diversity ablations
Max steps: 4000
GPUs: 8
```

If this run is stable, the next 8-GPU runs should be:

1. `P2/T3 capacity-up`: query 2,4,8; document 16,32,64. Tests whether page-side capacity is the bottleneck.
2. `P3/T1 capacity-down`: query 2,4,8; document 4,8,16. Builds the token-count vs performance curve.
3. `P8 tail-placement control`: same token counts as P1 but tokens placed at the tail. Isolates whether stage insertion matters beyond token budget.
4. Orthogonality ablation on P1 only: `ORTH_LAMBDA=0.01` then `0.05` if P1 is competitive.

