#!/usr/bin/env python3
"""Run ready ablation smoke gates sequentially without changing training semantics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from suite import find_project_root, load_config, resolve_variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priority", choices=("P0", "P1"), required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--benchmark", choices=("mmeb", "vidore_v1", "vidore_v2"), default="vidore_v2")
    parser.add_argument("--max-queries", type=int, default=2)
    parser.add_argument("--max-corpus", type=int, default=8)
    parser.add_argument("--skip", action="append", default=[], help="family/variant to skip")
    parser.add_argument("--tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    project_root = find_project_root(root)
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)
    matrix_log = log_dir / f"{args.priority.lower()}_{args.tag}.jsonl"
    failures = 0

    for config_path in sorted((root / args.priority).glob("*/experiment.json")):
        config = load_config(config_path)
        family = config_path.parent.name
        for variant in sorted(config["variants"]):
            resolved = resolve_variant(config, variant, project_root)
            key = f"{family}/{variant}"
            if resolved.get("status") != "ready" or key in args.skip:
                continue
            run_id = f"smoke_{args.steps}step_{args.tag}"
            command = [
                sys.executable,
                str(root / "suite.py"),
                "smoke",
                "--config",
                str(config_path),
                "--variant",
                variant,
                "--run-id",
                run_id,
                "--steps",
                str(args.steps),
                "--benchmark",
                args.benchmark,
                "--max-queries",
                str(args.max_queries),
                "--max-corpus",
                str(args.max_corpus),
            ]
            started = datetime.now().astimezone().isoformat()
            print(f"[START] {key} run_id={run_id}", flush=True)
            returncode = subprocess.run(command, cwd=project_root, check=False).returncode
            finished = datetime.now().astimezone().isoformat()
            record = {
                "family": family,
                "variant": variant,
                "run_id": run_id,
                "started_at": started,
                "finished_at": finished,
                "returncode": returncode,
            }
            with matrix_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(f"[{'PASS' if returncode == 0 else 'FAIL'}] {key} rc={returncode}", flush=True)
            failures += int(returncode != 0)

    print(f"matrix_log={matrix_log}")
    print(f"failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
