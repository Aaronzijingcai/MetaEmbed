# P0: Required Experiments

Run these families before optional analyses:

1. `01_rhc_components`: causal controls for the RHC scoring and aggregation components.
2. `02_hierarchy_mrl`: hierarchy and nested-supervision controls.
3. `03_compression_operators`: matched-budget compression baselines.
4. `04_multi_granularity_oracle`: evidence-complementarity analysis.
5. `05_token_budget`: 16/32/64/128/256 tokens per granularity.
6. `06_late_interaction`: directed MaxSim, directed TopK-mean, and the final adaptive bidirectional rule.

Variants marked `pending_*` are intentionally blocked; their presence records
the required experiment without claiming that the implementation is ready.
