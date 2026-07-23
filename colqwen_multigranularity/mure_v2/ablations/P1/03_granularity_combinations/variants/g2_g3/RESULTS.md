# Experiment Results: g2_g3

<!-- BEGIN GENERATED CONFIGURATION -->
> This block is generated from the experiment configuration. Do not edit it manually.

## Experiment Definition

| Field | Value |
|---|---|
| Suite | granularity_combinations |
| Variant | g2_g3 |
| Priority | P1 |
| Status | ready |
| Backend | rhc |
| Purpose | Intermediate and fine views under a matched total budget. |
| Experiment config | mure_v2/ablations/P1/03_granularity_combinations/experiment.json |
| MaxSim formulation | Adaptive bidirectional Top-K MaxSim with mean aggregation |

## Primary Configuration

| Setting | Resolved value |
|---|---|
| Model | ${PROJECT_DIR}/models/colqwen2.5-base |
| Training mixture | ${PROJECT_DIR}/configs/train/moca_data_ratios_v3_full.yaml |
| Maximum training steps | 60000 |
| Checkpoint interval | 1000 |
| Logging interval | 10 |
| Learning rate | 0.0001 |
| LR scheduler | linear |
| Per-device train batch | 8 |
| Gradient accumulation | 1 |
| Number of GPUs | 8 |
| Global batch | 64 |
| Maximum visual tokens | 1024 |
| Compression stages | all |
| Per-stage token budgets | 0 192 192 |
| MRL loss weights | 0 1 1 |
| Importance weight (alpha) | 1.0 |
| Gain weight (beta) | 1.0 |
| Value-modulation strength | 0.25 |
| Interaction mode | bi_query_topk_adaptive (Adaptive bidirectional Top-K MaxSim with mean aggregation) |
| Interaction Top-K | 48 |
| Bidirectional lambda cap | 0.8 |
| Document scoring chunk | 128 |
| Query scoring chunk | 64 |
| PEFT enabled | 1 |
| Gradient checkpointing | 1 |
| Distributed gather | 1 |

## Training and Evaluation Data

Training uses `${PROJECT_DIR}/configs/train/moca_data_ratios_v3_full.yaml`.
The evaluation protocols are fixed as follows:

| Benchmark | Configuration and metric |
|---|---|
| MMEB | `configs/eval/test_data_mast_mmeb_v3.yaml` (36 tasks; Precision@1) |
| ViDoRe V1 | `configs/eval/test_data_vidore_beir.yaml` (10 subsets; NDCG@5) |
| ViDoRe V2 | `configs/eval/test_data_mast_v2.yaml` (7 subsets; NDCG@5) |

## Complete Resolved Environment

This table is the authoritative record of all configured launcher overrides for this variant.

| Variable | Value |
|---|---|
| `BUDGETS` | 0 192 192 |
| `COMPRESS_STAGES` | all |
| `CONTRASTIVE_DEBUG_STEPS` |  |
| `CUDA_DEVICE_LIST` | 0,1,2,3,4,5,6,7 |
| `DDP_FIND_UNUSED_PARAMETERS` | 1 |
| `DOC_CHUNK_SIZE` | 128 |
| `DO_GATHER` | 1 |
| `DO_PADDING` | 1 |
| `EVAL_BSZ` | 4 |
| `FOLDER_ALPHA` | 1.0 |
| `GATE_STRENGTH` | 0.25 |
| `GLOBAL_BATCH_SIZE` | 64 |
| `GRADIENT_CHECKPOINTING` | 1 |
| `GRAD_ACCUM_STEPS` | 1 |
| `GRANULARITY_LOSS_WEIGHTS` | 0 1 1 |
| `IGNORE_DATA_SKIP` | 0 |
| `INCLUDED_STAGES` | g2+g3 |
| `INTERACTION_BI_LAMBDA` | 0.8 |
| `INTERACTION_FACTORIZED_LOCAL_WEIGHT` | 1.0 |
| `INTERACTION_GLOBAL_AUX_WEIGHT` | 0.0 |
| `INTERACTION_GLOBAL_WEIGHT` | 0.0 |
| `INTERACTION_LOSS_MODE` | bi_query_topk_adaptive |
| `INTERACTION_QUERY_TOPK` | 48 |
| `INTERLEAVED_BSZ` | 8 |
| `LEARNING_RATE` | 0.0001 |
| `LOGGING_STEPS` | 10 |
| `LR_SCHEDULER_TYPE` | linear |
| `MAIN_PROCESS_PORT` | 29912 |
| `MARC_ENABLED` | 0 |
| `MAX_NUM_VISUAL_TOKENS` | 1024 |
| `MAX_STEPS` | 60000 |
| `MODEL_PATH` | ${PROJECT_DIR}/models/colqwen2.5-base |
| `MURE_DEEP_AUDIT` | 0 |
| `MURE_DEFER_DDP_BUCKET_MB` | 64 |
| `MURE_DEFER_DDP_REDUCE` | 1 |
| `MURE_ENCODER_BACKWARD_MODE` | full |
| `MURE_ENCODER_MAX_VISUAL_ROWS` | 60000 |
| `MURE_EXPECTED_CUSTOM_TEXT_PROJ_TENSORS` | 2 |
| `MURE_EXPECTED_FOLDER_HOMO_TENSORS` | 66 |
| `MURE_EXPECTED_LANGUAGE_LORA_TENSORS` | 504 |
| `MURE_EXPECTED_TRAINABLE_NUMEL` | 74967174 |
| `MURE_EXPECTED_TRAINABLE_TENSORS` | 764 |
| `MURE_EXPECTED_VISUAL_LORA_TENSORS` | 192 |
| `MURE_GATHER_WITH_GRAD_MODE` | torch |
| `MURE_GRADIENT_CHECKPOINTING_REENTRANT` | 1 |
| `NOVELTY_WEIGHT` | 1.0 |
| `NUM_GPUS` | 8 |
| `QUERY_CHUNK_SIZE` | 64 |
| `RUN_EVAL` | 0 |
| `SAVE_STEPS` | 1000 |
| `STOP_AFTER_STEP` | 0 |
| `SUBSET_CONFIG` | ${PROJECT_DIR}/configs/train/moca_data_ratios_v3_full.yaml |
| `TRAIN_BSZ` | 8 |
| `TRAIN_COMPRESSOR_ONLY` | 0 |
| `USE_LIGER_KERNEL` | 0 |
| `USE_PEFT` | 1 |
| `WARMUP_RATIO` | 0 |
| `WARMUP_STEPS` | 0 |
<!-- END GENERATED CONFIGURATION -->

## Run Status and Artifacts

- Training status: [TODO]
- Evaluation status: [TODO]
- Run directory: `[TODO]`
- Selected checkpoint: `[TODO]`
- Training log: `[TODO]`
- MMEB result directory: `[TODO]`
- ViDoRe V1 result directory: `[TODO]`
- ViDoRe V2 result directory: `[TODO]`
- Evaluation date and code snapshot: `[TODO]`

## Paper-Level Summary

Report percentages after multiplying raw metrics in [0, 1] by 100. VDR Avg. is the mean of the ViDoRe V1 and V2 macro averages; MMEB Avg. is the macro average over all 36 MMEB tasks.

| Variant | Checkpoint | Step | V1 | V2 | VDR Avg. | CLS | VQA | RET | VG | MMEB Avg. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `g2_g3` | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |

## ViDoRe V1 Results

| Paper label | Dataset key | NDCG@5 (%) | Result artifact |
|---|---|---:|---|
| ArxivQ | `arxivqa_subsampled` | [TODO] | [TODO] |
| DocQ | `docvqa_subsampled` | [TODO] | [TODO] |
| InfoQ | `infovqa_subsampled` | [TODO] | [TODO] |
| TabF | `tabfquad_subsampled` | [TODO] | [TODO] |
| TATQ | `tatdqa` | [TODO] | [TODO] |
| Shift | `shift_project` | [TODO] | [TODO] |
| AI | `syntheticDocQA_artificial_intelligence_test` | [TODO] | [TODO] |
| Energy | `syntheticDocQA_energy` | [TODO] | [TODO] |
| Gov. | `syntheticDocQA_government_reports` | [TODO] | [TODO] |
| Health | `syntheticDocQA_healthcare_industry` | [TODO] | [TODO] |

- ViDoRe V1 macro average: **[TODO]**

## ViDoRe V2 Results

| Paper label | Dataset key | NDCG@5 (%) | Result artifact |
|---|---|---:|---|
| ESGHuman | `esg_reports_human_labeled_v2` | [TODO] | [TODO] |
| ESGSyn_Mul | `esg_reports_v2_multilingual` | [TODO] | [TODO] |
| ESGSyn | `esg_reports_v2` | [TODO] | [TODO] |
| Bio | `biomedical_lectures_v2` | [TODO] | [TODO] |
| BioMul | `biomedical_lectures_v2_multilingual` | [TODO] | [TODO] |
| Eco | `economics_reports_v2` | [TODO] | [TODO] |
| EcoMul | `economics_reports_v2_multilingual` | [TODO] | [TODO] |

- ViDoRe V2 macro average: **[TODO]**

## MMEB Results

The paper reports Precision@1. Some evaluation artifacts name this single-positive metric `recall_at_1`; record the paper-facing value here as Precision@1.

| Category | Dataset | Precision@1 (%) | Result artifact |
|---|---|---:|---|
| Classification | `ImageNet-1K` | [TODO] | [TODO] |
| Classification | `N24News` | [TODO] | [TODO] |
| Classification | `HatefulMemes` | [TODO] | [TODO] |
| Classification | `VOC2007` | [TODO] | [TODO] |
| Classification | `SUN397` | [TODO] | [TODO] |
| Classification | `Place365` | [TODO] | [TODO] |
| Classification | `ImageNet-A` | [TODO] | [TODO] |
| Classification | `ImageNet-R` | [TODO] | [TODO] |
| Classification | `ObjectNet` | [TODO] | [TODO] |
| Classification | `Country211` | [TODO] | [TODO] |
| VQA | `OK-VQA` | [TODO] | [TODO] |
| VQA | `A-OKVQA` | [TODO] | [TODO] |
| VQA | `DocVQA` | [TODO] | [TODO] |
| VQA | `InfographicsVQA` | [TODO] | [TODO] |
| VQA | `ChartQA` | [TODO] | [TODO] |
| VQA | `Visual7W` | [TODO] | [TODO] |
| VQA | `ScienceQA` | [TODO] | [TODO] |
| VQA | `VizWiz` | [TODO] | [TODO] |
| VQA | `GQA` | [TODO] | [TODO] |
| VQA | `TextVQA` | [TODO] | [TODO] |
| Retrieval | `VisDial` | [TODO] | [TODO] |
| Retrieval | `CIRR` | [TODO] | [TODO] |
| Retrieval | `VisualNews_t2i` | [TODO] | [TODO] |
| Retrieval | `VisualNews_i2t` | [TODO] | [TODO] |
| Retrieval | `MSCOCO_t2i` | [TODO] | [TODO] |
| Retrieval | `MSCOCO_i2t` | [TODO] | [TODO] |
| Retrieval | `NIGHTS` | [TODO] | [TODO] |
| Retrieval | `WebQA` | [TODO] | [TODO] |
| Retrieval | `FashionIQ` | [TODO] | [TODO] |
| Retrieval | `Wiki-SS-NQ` | [TODO] | [TODO] |
| Retrieval | `OVEN` | [TODO] | [TODO] |
| Retrieval | `EDIS` | [TODO] | [TODO] |
| Grounding | `MSCOCO` | [TODO] | [TODO] |
| Grounding | `RefCOCO` | [TODO] | [TODO] |
| Grounding | `RefCOCO-Matching` | [TODO] | [TODO] |
| Grounding | `Visual7W-Pointing` | [TODO] | [TODO] |

| MMEB category | Average (%) |
|---|---:|
| Classification | [TODO] |
| VQA | [TODO] |
| Retrieval | [TODO] |
| Grounding | [TODO] |
| **MMEB Avg.** | **[TODO]** |

## Observations

- Main finding: [TODO]
- Comparison with the reference variant: [TODO]
- Failures, warnings, or protocol deviations: [TODO]
- Paper table/figure destination: [TODO]
