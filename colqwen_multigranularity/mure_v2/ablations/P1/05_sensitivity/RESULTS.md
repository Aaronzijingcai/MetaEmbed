# Ablation Results: sensitivity

<!-- BEGIN GENERATED VARIANT REGISTRY -->
> This registry is generated from `experiment.json`. Detailed configurations and per-dataset results are stored in each linked variant record.

- Suite: `sensitivity`
- Configuration: `mure_v2/ablations/P1/05_sensitivity/experiment.json`
- Number of variants: 14

| Variant record | Resolved design |
|---|---|
| [`alpha_0_5`](variants/alpha_0_5/RESULTS.md) | Priority: P1<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Importance coefficient alpha=0.5. |
| [`alpha_1_0`](variants/alpha_1_0/RESULTS.md) | Priority: P1<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Importance coefficient alpha=1.0. |
| [`alpha_2_0`](variants/alpha_2_0/RESULTS.md) | Priority: P1<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Importance coefficient alpha=2.0. |
| [`beta_0_5`](variants/beta_0_5/RESULTS.md) | Priority: P1<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Gain coefficient beta=0.5. |
| [`beta_1_0`](variants/beta_1_0/RESULTS.md) | Priority: P1<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Gain coefficient beta=1.0. |
| [`beta_2_0`](variants/beta_2_0/RESULTS.md) | Priority: P1<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Gain coefficient beta=2.0. |
| [`topk_16`](variants/topk_16/RESULTS.md) | Priority: P1<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Adaptive bidirectional interaction with K=16. |
| [`topk_32`](variants/topk_32/RESULTS.md) | Priority: P1<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Adaptive bidirectional interaction with K=32. |
| [`topk_48`](variants/topk_48/RESULTS.md) | Priority: P1<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Adaptive bidirectional interaction with K=48. |
| [`topk_64`](variants/topk_64/RESULTS.md) | Priority: P1<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Adaptive bidirectional interaction with K=64. |
| [`rho_0_7`](variants/rho_0_7/RESULTS.md) | Priority: P1<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Adaptive bidirectional direction cap rho=0.7. |
| [`rho_0_8`](variants/rho_0_8/RESULTS.md) | Priority: P1<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Adaptive bidirectional direction cap rho=0.8. |
| [`rho_0_9`](variants/rho_0_9/RESULTS.md) | Priority: P1<br>Status: ready<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Adaptive bidirectional direction cap rho=0.9. |
| [`source_destination_split`](variants/source_destination_split/RESULTS.md) | Priority: P1<br>Status: pending_implementation<br>Budget: 128 128 128<br>MaxSim: Adaptive bidirectional Top-K MaxSim with mean aggregation<br>Purpose: Parity, random, and spatial source-destination partitions. |
<!-- END GENERATED VARIANT REGISTRY -->

## Paper-Level Comparison

Fill this table from the corresponding detailed variant records. Values are percentages.

| Variant | Checkpoint | Step | V1 | V2 | VDR Avg. | CLS | VQA | RET | VG | MMEB Avg. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`alpha_0_5`](variants/alpha_0_5/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`alpha_1_0`](variants/alpha_1_0/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`alpha_2_0`](variants/alpha_2_0/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`beta_0_5`](variants/beta_0_5/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`beta_1_0`](variants/beta_1_0/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`beta_2_0`](variants/beta_2_0/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`topk_16`](variants/topk_16/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`topk_32`](variants/topk_32/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`topk_48`](variants/topk_48/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`topk_64`](variants/topk_64/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`rho_0_7`](variants/rho_0_7/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`rho_0_8`](variants/rho_0_8/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`rho_0_9`](variants/rho_0_9/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| [`source_destination_split`](variants/source_destination_split/RESULTS.md) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |

## Group-Level Conclusion

- Primary comparison: [TODO]
- Supported claim: [TODO]
- Paper table/figure destination: [TODO]
- Protocol deviations or exclusions: [TODO]
