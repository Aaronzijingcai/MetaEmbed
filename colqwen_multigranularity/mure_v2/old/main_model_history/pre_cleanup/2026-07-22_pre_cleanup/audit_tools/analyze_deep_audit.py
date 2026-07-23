#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


REQUIRED_GROUPS = (
    "language_lora",
    "visual_lora",
    "custom_text_proj",
    "folder_homo",
)


def read_events(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("debug_dir", type=Path)
    parser.add_argument("--expected-ranks", type=int, default=8)
    parser.add_argument("--expected-steps", type=int, required=True)
    args = parser.parse_args()

    failures = []
    summary = {"ranks": {}, "failures": failures}
    for rank in range(args.expected_ranks):
        audit_path = args.debug_dir / f"rank{rank}.audit.jsonl"
        trace_path = args.debug_dir / f"rank{rank}.jsonl"
        if not audit_path.is_file() or not trace_path.is_file():
            failures.append(f"rank{rank}: missing audit or trace file")
            continue
        audit_events = read_events(audit_path)
        trace_events = read_events(trace_path)
        gradient_events = [
            event for event in audit_events if event["stage"] == "gradient_integrity"
        ]
        update_events = [
            event
            for event in audit_events
            if event["stage"] == "optimizer_parameter_updates"
        ]
        done_events = [
            event for event in trace_events if event["stage"] == "training_step_done"
        ]
        for label, events in (
            ("gradient", gradient_events),
            ("optimizer", update_events),
            ("training_done", done_events),
        ):
            if len(events) != args.expected_steps:
                failures.append(
                    f"rank{rank}: expected {args.expected_steps} {label} events, got {len(events)}"
                )
        for event in gradient_events:
            if event["cross_rank_max_abs_diff"] > 1e-6:
                failures.append(f"rank{rank}: gradient fingerprints differ across ranks")
            for group in REQUIRED_GROUPS:
                stats = event["groups"][group]
                if stats["grad_tensors"] != stats["tensors"]:
                    failures.append(f"rank{rank}: {group} has missing gradients")
                if stats["nonzero_grad_tensors"] == 0:
                    failures.append(f"rank{rank}: {group} has all-zero gradients")
        for event in update_events:
            for group in REQUIRED_GROUPS:
                if event["groups"][group]["changed_tensors"] == 0:
                    failures.append(f"rank{rank}: {group} did not update")
        summary["ranks"][str(rank)] = {
            "gradient_events": len(gradient_events),
            "optimizer_events": len(update_events),
            "training_done_events": len(done_events),
            "peak_allocated_mb": max(
                (event.get("cuda_peak_allocated_mb", 0) for event in trace_events),
                default=0,
            ),
        }

    summary["passed"] = not failures
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
