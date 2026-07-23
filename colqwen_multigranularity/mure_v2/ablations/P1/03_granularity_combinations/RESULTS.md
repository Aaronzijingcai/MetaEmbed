# Ablation Results: granularity_combinations

<!-- BEGIN GENERATED VARIANT REGISTRY -->
> This registry is generated from `experiment.json`. Detailed configurations and per-dataset results are stored in each linked variant record.

- Suite: `granularity_combinations`
- Configuration: `mure_v2/ablations/P1/03_granularity_combinations/experiment.json`
- Number of variants: 4

| Variant record | Resolved design |
|---|---|
| [`g1_g2`](variants/g1_g2/RESULTS.md) | Priority: P1<br>Status: ready<br>Budget: 192 192 0<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Global and intermediate views under a matched total budget. |
| [`g1_g3`](variants/g1_g3/RESULTS.md) | Priority: P1<br>Status: ready<br>Budget: 192 0 192<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Global and fine views under a matched total budget. |
| [`g2_g3`](variants/g2_g3/RESULTS.md) | Priority: P1<br>Status: ready<br>Budget: 0 192 192<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Intermediate and fine views under a matched total budget. |
| [`g1_g2_g3`](variants/g1_g2_g3/RESULTS.md) | Priority: P1<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Complete three-view representation under the same total budget. |
<!-- END GENERATED VARIANT REGISTRY -->

## Paper-Level Comparison

Fill this table from the corresponding detailed variant records. Values are percentages.

| Variant | Checkpoint | Step | V1 | V2 | VDR Avg. | CLS | VQA | RET | VG | MMEB Avg. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`g1_g2`](variants/g1_g2/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`g1_g3`](variants/g1_g3/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`g2_g3`](variants/g2_g3/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`g1_g2_g3`](variants/g1_g2_g3/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |

## Group-Level Conclusion

- Primary comparison: [TODO]
- Supported claim: [TODO]
- Paper table/figure destination: [TODO]
- Protocol deviations or exclusions: [TODO]
