# Ablation Results: compression_operators

<!-- BEGIN GENERATED VARIANT REGISTRY -->
> This registry is generated from `experiment.json`. Detailed configurations and per-dataset results are stored in each linked variant record.

- Suite: `compression_operators`
- Configuration: `mure_v2/ablations/P0/03_compression_operators/experiment.json`
- Number of variants: 7

| Variant record | Resolved design |
|---|---|
| [`full_token_mrl`](variants/full_token_mrl/RESULTS.md) | Priority: P0<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Uncompressed MRL reference under the complete data and training protocol. |
| [`visionzip`](variants/visionzip/RESULTS.md) | Priority: P0<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Post-projection VisionZip at the matched 384-token budget. |
| [`folder`](variants/folder/RESULTS.md) | Priority: P0<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Post-projection FOLDER at the matched 384-token budget. |
| [`scope`](variants/scope/RESULTS.md) | Priority: P0<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Post-projection SCOPE at the matched 384-token budget. |
| [`stage_resampler`](variants/stage_resampler/RESULTS.md) | Priority: P0<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Learnable stage resampler at the matched 384-token budget. |
| [`light_colpali`](variants/light_colpali/RESULTS.md) | Priority: P0<br>Status: eval_only<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Light-ColPali-style post-MLLM clustering at the matched 384-token budget. |
| [`rhc`](variants/rhc/RESULTS.md) | Priority: P0<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: RHC reference at the matched 384-token budget. |
<!-- END GENERATED VARIANT REGISTRY -->

## Paper-Level Comparison

Fill this table from the corresponding detailed variant records. Values are percentages.

| Variant | Checkpoint | Step | V1 | V2 | VDR Avg. | CLS | VQA | RET | VG | MMEB Avg. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`full_token_mrl`](variants/full_token_mrl/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`visionzip`](variants/visionzip/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`folder`](variants/folder/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`scope`](variants/scope/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`stage_resampler`](variants/stage_resampler/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`light_colpali`](variants/light_colpali/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`rhc`](variants/rhc/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |

## Group-Level Conclusion

- Primary comparison: [TODO]
- Supported claim: [TODO]
- Paper table/figure destination: [TODO]
- Protocol deviations or exclusions: [TODO]
