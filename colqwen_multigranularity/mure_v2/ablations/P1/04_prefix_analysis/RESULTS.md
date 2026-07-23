# Ablation Results: prefix_analysis

<!-- BEGIN GENERATED VARIANT REGISTRY -->
> This registry is generated from `experiment.json`. Detailed configurations and per-dataset results are stored in each linked variant record.

- Suite: `prefix_analysis`
- Configuration: `mure_v2/ablations/P1/04_prefix_analysis/experiment.json`
- Number of variants: 6

| Variant record | Resolved design |
|---|---|
| [`mrl_e1`](variants/mrl_e1/RESULTS.md) | Priority: P1<br>Status: eval_only<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: First prefix from the MRL-trained model. |
| [`mrl_e2`](variants/mrl_e2/RESULTS.md) | Priority: P1<br>Status: eval_only<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Second prefix from the MRL-trained model. |
| [`mrl_e3`](variants/mrl_e3/RESULTS.md) | Priority: P1<br>Status: eval_only<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Complete prefix from the MRL-trained model. |
| [`no_mrl_e1`](variants/no_mrl_e1/RESULTS.md) | Priority: P1<br>Status: eval_only<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: First prefix from the final-prefix-only model. |
| [`no_mrl_e2`](variants/no_mrl_e2/RESULTS.md) | Priority: P1<br>Status: eval_only<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Second prefix from the final-prefix-only model. |
| [`no_mrl_e3`](variants/no_mrl_e3/RESULTS.md) | Priority: P1<br>Status: eval_only<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Complete prefix from the final-prefix-only model. |
<!-- END GENERATED VARIANT REGISTRY -->

## Paper-Level Comparison

Fill this table from the corresponding detailed variant records. Values are percentages.

| Variant | Checkpoint | Step | V1 | V2 | VDR Avg. | CLS | VQA | RET | VG | MMEB Avg. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`mrl_e1`](variants/mrl_e1/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`mrl_e2`](variants/mrl_e2/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`mrl_e3`](variants/mrl_e3/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`no_mrl_e1`](variants/no_mrl_e1/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`no_mrl_e2`](variants/no_mrl_e2/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`no_mrl_e3`](variants/no_mrl_e3/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |

## Group-Level Conclusion

- Primary comparison: [TODO]
- Supported claim: [TODO]
- Paper table/figure destination: [TODO]
- Protocol deviations or exclusions: [TODO]
