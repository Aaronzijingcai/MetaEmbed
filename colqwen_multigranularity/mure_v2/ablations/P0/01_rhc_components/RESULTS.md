# Ablation Results: rhc_components

<!-- BEGIN GENERATED VARIANT REGISTRY -->
> This registry is generated from `experiment.json`. Detailed configurations and per-dataset results are stored in each linked variant record.

- Suite: `rhc_components`
- Configuration: `mure_v2/ablations/P0/01_rhc_components/experiment.json`
- Number of variants: 4

| Variant record | Resolved design |
|---|---|
| [`full_rhc`](variants/full_rhc/RESULTS.md) | Priority: P0<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Complete RHC reference under the final training and interaction protocol. |
| [`no_importance_protection`](variants/no_importance_protection/RESULTS.md) | Priority: P0<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Remove importance protection from the merge-selection score. |
| [`no_cross_granularity_gain`](variants/no_cross_granularity_gain/RESULTS.md) | Priority: P0<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Remove the cross-granularity residual gain from token protection. |
| [`no_value_modulation`](variants/no_value_modulation/RESULTS.md) | Priority: P0<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Disable differentiable value modulation while retaining hard merge selection. |
<!-- END GENERATED VARIANT REGISTRY -->

## Paper-Level Comparison

Fill this table from the corresponding detailed variant records. Values are percentages.

| Variant | Checkpoint | Step | V1 | V2 | VDR Avg. | CLS | VQA | RET | VG | MMEB Avg. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`full_rhc`](variants/full_rhc/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`no_importance_protection`](variants/no_importance_protection/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`no_cross_granularity_gain`](variants/no_cross_granularity_gain/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`no_value_modulation`](variants/no_value_modulation/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |

## Group-Level Conclusion

- Primary comparison: [TODO]
- Supported claim: [TODO]
- Paper table/figure destination: [TODO]
- Protocol deviations or exclusions: [TODO]
