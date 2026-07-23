# Ablation Results: late_interaction

<!-- BEGIN GENERATED VARIANT REGISTRY -->
> This registry is generated from `experiment.json`. Detailed configurations and per-dataset results are stored in each linked variant record.

- Suite: `late_interaction`
- Configuration: `mure_v2/ablations/P0/06_late_interaction/experiment.json`
- Number of variants: 3

| Variant record | Resolved design |
|---|---|
| [`directed_maxsim`](variants/directed_maxsim/RESULTS.md) | Priority: P0<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Standard directed MaxSim (query-to-candidate sum)<br>Purpose: Standard directed MaxSim using query-to-candidate sum aggregation. |
| [`directed_topk48_mean`](variants/directed_topk48_mean/RESULTS.md) | Priority: P0<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Directed Top-K MaxSim with mean aggregation<br>Purpose: Directed TopK-mean over the strongest 48 query-side matches. |
| [`adaptive_bidirectional_topk48_mean`](variants/adaptive_bidirectional_topk48_mean/RESULTS.md) | Priority: P0<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Final adaptive bidirectional TopK-mean interaction. |
<!-- END GENERATED VARIANT REGISTRY -->

## Paper-Level Comparison

Fill this table from the corresponding detailed variant records. Values are percentages.

| Variant | Checkpoint | Step | V1 | V2 | VDR Avg. | CLS | VQA | RET | VG | MMEB Avg. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`directed_maxsim`](variants/directed_maxsim/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`directed_topk48_mean`](variants/directed_topk48_mean/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`adaptive_bidirectional_topk48_mean`](variants/adaptive_bidirectional_topk48_mean/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |

## Group-Level Conclusion

- Primary comparison: [TODO]
- Supported claim: [TODO]
- Paper table/figure destination: [TODO]
- Protocol deviations or exclusions: [TODO]
