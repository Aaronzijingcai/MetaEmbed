from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SKIP_KEYS = {
    "average",
    "avg",
    "macro_avg",
    "micro_avg",
    "overall",
    "report_metric",
    "metric",
    "group",
    "class",
    "num_datasets",
}


def _find_json_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".json":
            files.append(path)
        elif path.is_dir():
            for candidate in sorted(path.rglob("*.json")):
                if candidate.name.endswith("_summary.json"):
                    continue
                if candidate.name in {"summary.json", "compare.json"}:
                    continue
                files.append(candidate)
    return files


def _run_name(path: Path) -> str:
    if path.name in {"mmeb_full.json", "vidore_v2.json"}:
        return path.parent.name
    return path.stem


def _metric_from_dict(value: dict[str, Any], metric: str) -> float | None:
    candidates = [
        metric,
        f"avg_{metric}",
        metric.replace("precision_at_", "recall_at_"),
        metric.replace("recall_at_", "precision_at_"),
    ]
    for key in candidates:
        score = value.get(key)
        if isinstance(score, (int, float)):
            return float(score)
    for nested in value.values():
        if isinstance(nested, dict):
            score = _metric_from_dict(nested, metric)
            if score is not None:
                return score
    return None


def _extract_scores(path: Path, metric: str) -> dict[str, float]:
    data = json.loads(path.read_text())
    scores: dict[str, float] = {}
    if not isinstance(data, dict):
        return scores

    direct = _metric_from_dict(data, metric)
    if direct is not None and not any(isinstance(v, dict) for v in data.values()):
        scores[path.stem] = direct
        return scores

    for name, value in data.items():
        if name in SKIP_KEYS or not isinstance(value, dict):
            continue
        score = _metric_from_dict(value, metric)
        if score is not None:
            scores[name] = score
    return scores


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _to_markdown(run_scores: dict[str, dict[str, float]]) -> str:
    runs = sorted(run_scores)
    datasets = sorted({dataset for scores in run_scores.values() for dataset in scores})
    rows: list[tuple[str, list[float | None]]] = []
    for dataset in datasets:
        rows.append((dataset, [run_scores[run].get(dataset) for run in runs]))
    averages = []
    for run in runs:
        values = list(run_scores[run].values())
        averages.append(sum(values) / len(values) if values else None)

    header = ["Dataset", *runs]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---", *[":---:" for _ in runs]]) + " |",
    ]
    for dataset, values in rows:
        lines.append("| " + " | ".join([dataset, *[_fmt(v) for v in values]]) + " |")
    lines.append("| **Average** | " + " | ".join(_fmt(v) for v in averages) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="JSON files or directories to summarize.")
    parser.add_argument("--metric", default="recall_at_1")
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    json_files = _find_json_files([Path(path) for path in args.paths])
    run_scores: dict[str, dict[str, float]] = {}
    for path in json_files:
        scores = _extract_scores(path, args.metric)
        if scores:
            run_scores[_run_name(path)] = scores

    text = _to_markdown(run_scores)
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
