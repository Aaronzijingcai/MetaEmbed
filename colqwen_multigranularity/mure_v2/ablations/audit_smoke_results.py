#!/usr/bin/env python3
"""Audit isolated smoke checkpoints and evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from suite import find_project_root, load_config, resolve_variant


STATE_FILES = {
    "rhc": "folder_homo.pt",
    "mlppost": "stage_compressor.pt",
    "gain_design": "folder_gain_only.pt",
    "importance_design": "folder_importance.pt",
}


def complete_run_dirs(variant_root: Path, backend: str) -> list[Path]:
    state_file = STATE_FILES.get(backend)
    if state_file is None:
        return []
    complete: list[Path] = []
    for run_dir in sorted((variant_root / "runs").glob("smoke_*")):
        checkpoints = sorted(run_dir.glob("checkpoint-*"))
        if not checkpoints:
            continue
        checkpoint = checkpoints[-1]
        required = (
            checkpoint / "trainer_state.json",
            checkpoint / "adapter_config.json",
            checkpoint / state_file,
        )
        if all(path.is_file() for path in required):
            complete.append(run_dir)
    return complete


def audit(root: Path) -> dict[str, Any]:
    project_root = find_project_root(root)
    rows: list[dict[str, Any]] = []
    for priority in ("P0", "P1"):
        for config_path in sorted((root / priority).glob("*/experiment.json")):
            config = load_config(config_path)
            for variant in sorted(config["variants"]):
                resolved = resolve_variant(config, variant, project_root)
                variant_root = config_path.parent / "variants" / variant
                backend = str(resolved.get("backend", ""))
                runs = complete_run_dirs(variant_root, backend)
                evaluations = sorted((variant_root / "evaluations").glob("**/smoke.json"))
                configured_status = str(resolved.get("status", "pending"))
                if configured_status == "ready":
                    smoke_status = "pass" if runs and evaluations else "fail"
                elif configured_status == "eval_only":
                    smoke_status = "pass" if evaluations else "not_run"
                else:
                    smoke_status = "pending"
                rows.append(
                    {
                        "priority": priority,
                        "family": config_path.parent.name,
                        "variant": variant,
                        "configured_status": configured_status,
                        "backend": backend,
                        "smoke_status": smoke_status,
                        "complete_runs": [str(path.relative_to(project_root)) for path in runs],
                        "evaluations": [str(path.relative_to(project_root)) for path in evaluations],
                    }
                )

    counts = Counter(row["smoke_status"] for row in rows)
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "counts": dict(sorted(counts.items())),
        "rows": rows,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# MURE-V2 Smoke Audit",
        "",
        f"Generated: `{report['created_at']}`",
        "",
        "## Summary",
        "",
    ]
    for status, count in report["counts"].items():
        lines.append(f"- `{status}`: {count}")
    lines.extend(
        [
            "",
            "## Variants",
            "",
            "| Priority | Family | Variant | Config | Smoke | Runs | Evals |",
            "|---|---|---|---|---|---:|---:|",
        ]
    )
    for row in report["rows"]:
        lines.append(
            "| {priority} | {family} | {variant} | {configured_status} | "
            "{smoke_status} | {run_count} | {eval_count} |".format(
                **row,
                run_count=len(row["complete_runs"]),
                eval_count=len(row["evaluations"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output_dir = args.output_dir or root / "logs" / datetime.now().strftime("audit_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = audit(root)
    (output_dir / "smoke_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(report, output_dir / "smoke_audit.md")
    print(json.dumps(report["counts"], sort_keys=True))
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
