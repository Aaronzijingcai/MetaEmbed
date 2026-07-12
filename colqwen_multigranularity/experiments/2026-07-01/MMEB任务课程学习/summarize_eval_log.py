#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


PATTERN = re.compile(r"Metrics for (?P<name>MMEB-eval-[^:]+): (?P<metrics>\{.*\})")


def parse_log(path: Path) -> dict:
    rows = {}
    for line in path.read_text(errors="replace").splitlines():
        match = PATTERN.search(line)
        if not match:
            continue
        name = match.group("name")
        metrics = ast.literal_eval(match.group("metrics"))
        rows[name] = metrics
    if rows:
        for metric in ("recall_at_1", "recall_at_5"):
            values = [float(row[metric]) for row in rows.values() if metric in row]
            if values:
                rows[f"avg_{metric}"] = sum(values) / len(values)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()

    summary = parse_log(args.log_path)
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

