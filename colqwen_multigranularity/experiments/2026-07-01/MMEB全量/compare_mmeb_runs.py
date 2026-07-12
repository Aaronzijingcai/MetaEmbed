from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _infer_name(path: Path) -> str:
    parent = path.parent.name
    if parent and parent not in {"mmeb_full", "eval"}:
        return parent
    return path.stem


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def _load_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    return {
        "run": _infer_name(path),
        "metric": data.get("report_metric") or data.get("metric", ""),
        "overall": data.get("overall"),
        "IND": data.get("group", {}).get("IND"),
        "OOD": data.get("group", {}).get("OOD"),
        "Classification": data.get("class", {}).get("Classification"),
        "VQA": data.get("class", {}).get("VQA"),
        "Retrieval": data.get("class", {}).get("Retrieval"),
        "Visual Grounding": data.get("class", {}).get("Visual Grounding"),
        "num_datasets": data.get("num_datasets"),
        "path": str(path),
    }


def _to_markdown(rows: list[dict[str, Any]]) -> str:
    columns = [
        "run",
        "P@1 overall",
        "IND",
        "OOD",
        "Classification",
        "VQA",
        "Retrieval",
        "Visual Grounding",
        "num_datasets",
    ]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = [row.get("overall") if col == "P@1 overall" else row.get(col) for col in columns]
        lines.append("| " + " | ".join(_fmt(value) for value in values) + " |")
    return "\n".join(lines)


def _to_tsv(rows: list[dict[str, Any]]) -> str:
    columns = [
        "run",
        "metric",
        "overall",
        "IND",
        "OOD",
        "Classification",
        "VQA",
        "Retrieval",
        "Visual Grounding",
        "num_datasets",
        "path",
    ]
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(_fmt(row.get(col)) for col in columns))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_json", nargs="+", help="mmeb_full_summary.json files")
    parser.add_argument("--format", choices=["markdown", "tsv", "json"], default="markdown")
    parser.add_argument("--output-path", type=str, default=None)
    args = parser.parse_args()

    rows = [_load_summary(Path(path)) for path in args.summary_json]
    rows.sort(key=lambda row: (-float(row["overall"] or -1.0), row["run"]))

    if args.format == "json":
        text = json.dumps(rows, indent=2, ensure_ascii=False)
    elif args.format == "tsv":
        text = _to_tsv(rows)
    else:
        text = _to_markdown(rows)

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text)
    print(text)
