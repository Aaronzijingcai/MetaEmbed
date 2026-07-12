# MMEB 36 Dataset Results: sym160 4k

Updated: 2026-07-02

## Run Metadata

| Item | Value |
| --- | --- |
| Run | `folder_homo_mmeb_budget_sym160_4k` |
| Checkpoint | `runs/folder_homo_mmeb_budget_sym160_4k/checkpoint-4000` |
| Query budget | `160/160/160` |
| Target budget | `160/160/160` |
| Compress stages | `all` |
| Metric | `recall_at_1` / `recall_at_5` |
| Eval config | `configs/eval/test_data_mast_mmeb_v3.yaml` |
| Stable eval batch | `BATCH_QUERY=16`, `BATCH_PASSAGE=32`, `BATCH_SCORE=64`, `NUM_WORKERS=0` |

This result is merged from:

1. `runs/folder_homo_mmeb_budget_sym160_4k/logs/eval_mmeb_budget_20260702_210305.log`
2. `runs/folder_homo_mmeb_budget_sym160_4k/logs/eval_mmeb_budget_remaining_oven_onward_bs64_20260702_220802.log`

The first full eval completed 29/36 datasets and OOMed at OVEN with `BATCH_SCORE=256`. The remaining 7 datasets were recovered with `BATCH_SCORE=64`. The checkpoint, token budgets, scorer, and metric are the same, so the merged 36-dataset table is the valid current full result.

## Overall

| Metric | Value |
| --- | ---: |
| Number of datasets | 36 |
| Avg R@1 | 0.462083 |
| Avg R@5 | 0.676333 |

## Per-Dataset Results

| Dataset | Type | R@1 | R@5 | Note |
| --- | --- | ---: | ---: | --- |
| MMEB-eval-ImageNet-1K-beir | Classification | 0.648 | 0.874 |  |
| MMEB-eval-N24News-beir | Classification | 0.624 | 0.900 |  |
| MMEB-eval-HatefulMemes-beir | Classification | 0.595 | 1.000 |  |
| MMEB-eval-SUN397-beir | Classification | 0.643 | 0.900 |  |
| MMEB-eval-VOC2007-beir | Classification | 0.848 | 0.980 | strong |
| MMEB-eval-InfographicsVQA-beir | VQA | 0.137 | 0.334 | hard |
| MMEB-eval-ChartQA-beir | VQA | 0.174 | 0.364 | hard |
| MMEB-eval-A-OKVQA-beir | VQA | 0.182 | 0.395 | hard |
| MMEB-eval-DocVQA-beir | VQA | 0.274 | 0.493 | hard |
| MMEB-eval-OK-VQA-beir | VQA | 0.214 | 0.434 | hard |
| MMEB-eval-Visual7W-beir | VQA | 0.147 | 0.372 | hard |
| MMEB-eval-VisDial-beir | Retrieval | 0.547 | 0.805 |  |
| MMEB-eval-CIRR-beir | Retrieval | 0.105 | 0.437 | compositional hard |
| MMEB-eval-NIGHTS-beir | Retrieval | 0.648 | 0.978 |  |
| MMEB-eval-WebQA-beir | Retrieval | 0.893 | 0.976 | strong |
| MMEB-eval-VisualNews_i2t-beir | Retrieval | 0.628 | 0.810 | image-to-text |
| MMEB-eval-VisualNews_t2i-beir | Retrieval | 0.628 | 0.812 | text-to-image |
| MMEB-eval-MSCOCO_i2t-beir | Retrieval | 0.603 | 0.865 | image-to-text |
| MMEB-eval-MSCOCO_t2i-beir | Retrieval | 0.680 | 0.912 | text-to-image |
| MMEB-eval-MSCOCO-beir | Visual Grounding | 0.429 | 0.616 |  |
| MMEB-eval-Place365-beir | Classification | 0.354 | 0.651 | OOD |
| MMEB-eval-ImageNet-A-beir | Classification | 0.414 | 0.615 | OOD |
| MMEB-eval-ImageNet-R-beir | Classification | 0.683 | 0.902 | OOD strong |
| MMEB-eval-ObjectNet-beir | Classification | 0.531 | 0.722 | OOD |
| MMEB-eval-Country211-beir | Classification | 0.088 | 0.204 | hard |
| MMEB-eval-ScienceQA-beir | VQA | 0.198 | 0.451 | OOD hard |
| MMEB-eval-VizWiz-beir | VQA | 0.266 | 0.469 | OOD |
| MMEB-eval-GQA-beir | VQA | 0.155 | 0.357 | OOD hard |
| MMEB-eval-TextVQA-beir | VQA | 0.214 | 0.357 | OOD hard |
| MMEB-eval-OVEN-beir | Retrieval | 0.635 | 0.835 | recovered with `BATCH_SCORE=64` |
| MMEB-eval-FashionIQ-beir | Retrieval | 0.025 | 0.104 | compositional hard, worst |
| MMEB-eval-EDIS-beir | Retrieval | 0.846 | 0.971 | strong |
| MMEB-eval-Wiki-SS-NQ-beir | Retrieval | 0.790 | 0.955 | strong |
| MMEB-eval-Visual7W-Pointing-beir | Visual Grounding | 0.527 | 0.756 |  |
| MMEB-eval-RefCOCO-beir | Visual Grounding | 0.446 | 0.742 |  |
| MMEB-eval-RefCOCO-Matching-beir | Visual Grounding | 0.816 | 1.000 | strong |

## Worst 10 by R@1

| Dataset | R@1 | R@5 |
| --- | ---: | ---: |
| MMEB-eval-FashionIQ-beir | 0.025 | 0.104 |
| MMEB-eval-Country211-beir | 0.088 | 0.204 |
| MMEB-eval-CIRR-beir | 0.105 | 0.437 |
| MMEB-eval-InfographicsVQA-beir | 0.137 | 0.334 |
| MMEB-eval-Visual7W-beir | 0.147 | 0.372 |
| MMEB-eval-GQA-beir | 0.155 | 0.357 |
| MMEB-eval-ChartQA-beir | 0.174 | 0.364 |
| MMEB-eval-A-OKVQA-beir | 0.182 | 0.395 |
| MMEB-eval-ScienceQA-beir | 0.198 | 0.451 |
| MMEB-eval-OK-VQA-beir | 0.214 | 0.434 |

## Best 10 by R@1

| Dataset | R@1 | R@5 |
| --- | ---: | ---: |
| MMEB-eval-WebQA-beir | 0.893 | 0.976 |
| MMEB-eval-VOC2007-beir | 0.848 | 0.980 |
| MMEB-eval-EDIS-beir | 0.846 | 0.971 |
| MMEB-eval-RefCOCO-Matching-beir | 0.816 | 1.000 |
| MMEB-eval-Wiki-SS-NQ-beir | 0.790 | 0.955 |
| MMEB-eval-ImageNet-R-beir | 0.683 | 0.902 |
| MMEB-eval-MSCOCO_t2i-beir | 0.680 | 0.912 |
| MMEB-eval-NIGHTS-beir | 0.648 | 0.978 |
| MMEB-eval-ImageNet-1K-beir | 0.648 | 0.874 |
| MMEB-eval-SUN397-beir | 0.643 | 0.900 |

## Preliminary Reading

The current `sym160` compression can recover strong results on several classification, retrieval, and grounding subsets. The remaining failures are concentrated in compositional image retrieval and VQA-hard subsets, especially `FashionIQ`, `CIRR`, `InfographicsVQA`, `Visual7W`, `GQA`, and `ChartQA`. These are the primary targets for the follow-up `MMEB任务课程学习` and MaxSim interaction diagnostics.
