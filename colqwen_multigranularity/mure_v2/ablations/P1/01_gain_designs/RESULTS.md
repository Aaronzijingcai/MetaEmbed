# Ablation Results: gain_designs

<!-- BEGIN GENERATED VARIANT REGISTRY -->
> This registry is generated from `experiment.json`. Detailed configurations and per-dataset results are stored in each linked variant record.

- Suite: `gain_designs`
- Configuration: `mure_v2/ablations/P1/01_gain_designs/experiment.json`
- Number of variants: 4

| Variant record | Resolved design |
|---|---|
| [`no_gain`](variants/no_gain/RESULTS.md) | Priority: P1<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Remove cross-granularity gain. |
| [`one_minus_maxsim`](variants/one_minus_maxsim/RESULTS.md) | Priority: P1<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Final 1-MaxSim residual gain. |
| [`learned_metric_residual`](variants/learned_metric_residual/RESULTS.md) | Priority: P1<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Learned metric residual from the archived gain exploration. |
| [`learned_anchor_gate`](variants/learned_anchor_gate/RESULTS.md) | Priority: P1<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Learned cross-anchor gate from the archived gain exploration. |
<!-- END GENERATED VARIANT REGISTRY -->

## Paper-Level Comparison

Fill this table from the corresponding detailed variant records. Values are percentages.

| Variant | Checkpoint | Step | V1 | V2 | VDR Avg. | CLS | VQA | RET | VG | MMEB Avg. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`no_gain`](variants/no_gain/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`one_minus_maxsim`](variants/one_minus_maxsim/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`learned_metric_residual`](variants/learned_metric_residual/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`learned_anchor_gate`](variants/learned_anchor_gate/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |

## Group-Level Conclusion

- Primary comparison: [TODO]
- Supported claim: [TODO]
- Paper table/figure destination: [TODO]
- Protocol deviations or exclusions: [TODO]
