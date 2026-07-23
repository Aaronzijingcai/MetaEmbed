# Main-Model Audit Artifacts

This directory owns retained diagnostics produced while validating the MURE
main-model training path. Date directories may contain test scripts, rank-level
JSONL traces, and immutable log snapshots.

- `2026-07-19/scripts/`: focused gradient, gather, DDP, and MetaEmbed LoRA audits.
- `2026-07-19/deep_audit_jsonl/`: eight-rank gradient and optimizer audit traces.
- `2026-07-20/formal_adaptive_step1300.log`: a point-in-time copy of the formal
  adaptive run log used for loss-trend analysis.

These files are evidence, not launcher inputs. Current training code remains in
the owning source modules and `experiments/main_model/` launchers.
