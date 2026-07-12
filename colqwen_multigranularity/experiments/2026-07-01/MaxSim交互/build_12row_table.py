from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


WORST10 = [
    "MMEB-eval-FashionIQ-beir",
    "MMEB-eval-CIRR-beir",
    "MMEB-eval-Country211-beir",
    "MMEB-eval-GQA-beir",
    "MMEB-eval-ScienceQA-beir",
    "MMEB-eval-InfographicsVQA-beir",
    "MMEB-eval-A-OKVQA-beir",
    "MMEB-eval-Visual7W-beir",
    "MMEB-eval-OK-VQA-beir",
    "MMEB-eval-ChartQA-beir",
]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _metric_from_dict(value: dict[str, Any], metric: str) -> float | None:
    keys = [
        metric,
        f"avg_{metric}",
        metric.replace("precision_at_", "recall_at_"),
        metric.replace("recall_at_", "precision_at_"),
    ]
    for key in keys:
        score = value.get(key)
        if isinstance(score, (int, float)):
            return float(score)
    for nested in value.values():
        if isinstance(nested, dict):
            score = _metric_from_dict(nested, metric)
            if score is not None:
                return score
    return None


def _load_mmeb_scores(path: Path) -> dict[str, float]:
    data = _load_json(path)
    if not isinstance(data, dict):
        return {}
    scores: dict[str, float] = {}
    for dataset in WORST10:
        value = data.get(dataset)
        if isinstance(value, dict):
            score = _metric_from_dict(value, "recall_at_1")
            if score is not None:
                scores[dataset] = score
    return scores


def _load_vidore_average(path: Path) -> float | None:
    data = _load_json(path)
    if not isinstance(data, dict):
        return None
    scores: list[float] = []
    for value in data.values():
        if not isinstance(value, dict):
            continue
        score = _metric_from_dict(value, "ndcg_at_5")
        if score is not None:
            scores.append(score)
    return sum(scores) / len(scores) if scores else None


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_root", help="Directory containing mmeb_worst10/ and vidore_v2/.")
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    root = Path(args.eval_root)
    mmeb_root = root / "mmeb_worst10"
    vidore_root = root / "vidore_v2"
    scorers = sorted(
        {
            path.name
            for base in [mmeb_root, vidore_root]
            if base.exists()
            for path in base.iterdir()
            if path.is_dir()
        }
    )

    table: dict[str, dict[str, float | None]] = {}
    for scorer in scorers:
        mmeb_scores = _load_mmeb_scores(mmeb_root / scorer / "mmeb_full.json")
        values = list(mmeb_scores.values())
        table[scorer] = {dataset: mmeb_scores.get(dataset) for dataset in WORST10}
        table[scorer]["MMEB-worst10-average"] = sum(values) / len(values) if values else None
        table[scorer]["ViDoRe-v2-average"] = _load_vidore_average(vidore_root / scorer / "vidore_v2.json")

    rows = [*WORST10, "MMEB-worst10-average", "ViDoRe-v2-average"]
    header = ["Dataset", *scorers]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---", *[":---:" for _ in scorers]]) + " |",
    ]
    for row in rows:
        values = [_fmt(table[scorer].get(row)) for scorer in scorers]
        label = row.replace("MMEB-eval-", "").replace("-beir", "")
        if row.endswith("average"):
            label = f"**{label}**"
        lines.append("| " + " | ".join([label, *values]) + " |")

    text = "\n".join(lines)
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
