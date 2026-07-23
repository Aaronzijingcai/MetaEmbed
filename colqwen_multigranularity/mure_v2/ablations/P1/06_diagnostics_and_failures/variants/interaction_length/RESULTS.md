# Experiment Results: interaction_length

<!-- BEGIN GENERATED CONFIGURATION -->
> This block is generated from the experiment configuration. Do not edit it manually.

## Experiment Definition

| Field | Value |
|---|---|
| Suite | diagnostics_and_failures |
| Variant | interaction_length |
| Priority | P1 |
| Status | eval_only |
| Backend | diagnostics |
| Purpose | Query modality and query-candidate length-asymmetry diagnostics. |
| Experiment config | mure_v2/ablations/P1/06_diagnostics_and_failures/experiment.json |
| MaxSim formulation | Defined by the source checkpoint or evaluation command |

## Primary Configuration

| Setting | Resolved value |
|---|---|
| Model | - |
| Training mixture | - |
| Maximum training steps | - |
| Checkpoint interval | - |
| Logging interval | - |
| Learning rate | - |
| LR scheduler | - |
| Per-device train batch | - |
| Gradient accumulation | - |
| Number of GPUs | - |
| Global batch | - |
| Maximum visual tokens | - |
| Compression stages | - |
| Per-stage token budgets | - |
| MRL loss weights | - |
| Importance weight (alpha) | - |
| Gain weight (beta) | - |
| Value-modulation strength | - |
| Interaction mode | - (Defined by the source checkpoint or evaluation command) |
| Interaction Top-K | - |
| Bidirectional lambda cap | - |
| Document scoring chunk | - |
| Query scoring chunk | - |
| PEFT enabled | - |
| Gradient checkpointing | - |
| Distributed gather | - |

## Training and Evaluation Data

This is an evaluation-only analysis; record its source checkpoint and scoring command below.
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
| N/A | Evaluation-only derived analysis |
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
| `interaction_length` | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |

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
