# Ablation Results: hierarchy_and_mrl

<!-- BEGIN GENERATED VARIANT REGISTRY -->
> This registry is generated from `experiment.json`. Detailed configurations and per-dataset results are stored in each linked variant record.

- Suite: `hierarchy_and_mrl`
- Configuration: `mure_v2/ablations/P0/02_hierarchy_mrl/experiment.json`
- Number of variants: 4

| Variant record | Resolved design |
|---|---|
| [`hierarchical_with_mrl`](variants/hierarchical_with_mrl/RESULTS.md) | Priority: P0<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Hierarchical residual compression with supervision on all three prefixes. |
| [`hierarchical_without_mrl`](variants/hierarchical_without_mrl/RESULTS.md) | Priority: P0<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Hierarchical residual compression supervised only at the complete prefix. |
| [`flat_with_mrl`](variants/flat_with_mrl/RESULTS.md) | Priority: P0<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Flat matched-budget compression with supervision on all three prefixes. |
| [`flat_without_mrl`](variants/flat_without_mrl/RESULTS.md) | Priority: P0<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Flat matched-budget compression supervised only at the complete prefix. |
<!-- END GENERATED VARIANT REGISTRY -->

## Paper-Level Comparison

Fill this table from the corresponding detailed variant records. Values are percentages.

| Variant | Checkpoint | Step | V1 | V2 | VDR Avg. | CLS | VQA | RET | VG | MMEB Avg. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`hierarchical_with_mrl`](variants/hierarchical_with_mrl/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`hierarchical_without_mrl`](variants/hierarchical_without_mrl/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`flat_with_mrl`](variants/flat_with_mrl/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`flat_without_mrl`](variants/flat_without_mrl/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |

## Group-Level Conclusion

- Primary comparison: [TODO]
- Supported claim: [TODO]
- Paper table/figure destination: [TODO]
- Protocol deviations or exclusions: [TODO]
