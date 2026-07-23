#!/usr/bin/env python3
"""Aggregate g1/g2/g3 per-query eval outputs into an oracle report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


BENCHMARKS = {
    "vidore_v1": "ndcg_at_5",
    "vidore_v2": "ndcg_at_5",
    "mmeb": "recall_at_1",
}
GRANULARITIES = ("g1", "g2", "g3")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing eval result: {path}")
    return json.loads(path.read_text())


def _per_query_files(input_dir: Path, granularity: str, benchmark: str) -> list[Path]:
    root = input_dir / granularity / benchmark
    if not root.exists():
        raise FileNotFoundError(f"Missing per-query directory: {root}")
    files = sorted(root.glob("*.per_query.json"))
    if not files:
        raise FileNotFoundError(f"No per-query files found under: {root}")
    return files


def analyze(input_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "note": "Strict per-query oracle: for each query, choose the best of g1/g2/g3 by the benchmark main metric.",
        "benchmarks": {},
    }

    all_queries: list[dict[str, Any]] = []
    benchmark_summaries: list[dict[str, Any]] = []
    all_best = Counter()

    for benchmark, main_metric in BENCHMARKS.items():
        per_dataset: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
        for g in GRANULARITIES:
            for path in _per_query_files(input_dir, g, benchmark):
                dataset = path.name.removesuffix(".per_query.json")
                per_dataset.setdefault(dataset, {})[g] = _load_json(path)

        rows = {}
        best_counter = Counter()
        benchmark_queries = []
        for dataset, by_g in sorted(per_dataset.items()):
            missing = [g for g in GRANULARITIES if g not in by_g]
            if missing:
                raise KeyError(f"{benchmark}/{dataset} missing granularities: {missing}")

            query_ids = sorted(set.intersection(*(set(by_g[g].keys()) for g in GRANULARITIES)))
            if not query_ids:
                raise ValueError(f"No common query ids for {benchmark}/{dataset}")

            query_rows = []
            for query_id in query_ids:
                values = {
                    g: float(by_g[g][query_id].get(main_metric, 0.0))
                    for g in GRANULARITIES
                }
                best_g, best_value = max(values.items(), key=lambda item: item[1])
                best_counter[best_g] += 1
                row = {
                    "benchmark": benchmark,
                    "dataset": dataset,
                    "query_id": query_id,
                    "metric": main_metric,
                    **values,
                    "oracle_best_granularity": best_g,
                    "oracle": best_value,
                    "oracle_gain_over_g3": best_value - values["g3"],
                }
                query_rows.append(row)
                benchmark_queries.append(row)
                all_queries.append(row)

            rows[dataset] = {
                "num_queries": len(query_rows),
                "avg_g1": mean(row["g1"] for row in query_rows),
                "avg_g2": mean(row["g2"] for row in query_rows),
                "avg_g3": mean(row["g3"] for row in query_rows),
                "avg_oracle": mean(row["oracle"] for row in query_rows),
                "avg_oracle_gain_over_g3": mean(row["oracle_gain_over_g3"] for row in query_rows),
                "oracle_choice_counts": dict(Counter(row["oracle_best_granularity"] for row in query_rows)),
            }

        dataset_rows = list(rows.values())
        summary = {
            "metric": main_metric,
            "num_datasets": len(rows),
            "num_queries": len(benchmark_queries),
            "average_mode": "dataset_mean",
            "avg_g1": mean(row["avg_g1"] for row in dataset_rows),
            "avg_g2": mean(row["avg_g2"] for row in dataset_rows),
            "avg_g3": mean(row["avg_g3"] for row in dataset_rows),
            "avg_oracle": mean(row["avg_oracle"] for row in dataset_rows),
            "avg_oracle_gain_over_g3": mean(row["avg_oracle_gain_over_g3"] for row in dataset_rows),
            "query_weighted_avg_g1": mean(row["g1"] for row in benchmark_queries),
            "query_weighted_avg_g2": mean(row["g2"] for row in benchmark_queries),
            "query_weighted_avg_g3": mean(row["g3"] for row in benchmark_queries),
            "query_weighted_avg_oracle": mean(row["oracle"] for row in benchmark_queries),
            "query_weighted_avg_oracle_gain_over_g3": mean(
                row["oracle_gain_over_g3"] for row in benchmark_queries
            ),
            "oracle_choice_counts": dict(best_counter),
            "oracle_choice_ratio": {
                g: best_counter[g] / len(benchmark_queries) for g in GRANULARITIES
            },
        }
        benchmark_summaries.append(summary)
        all_best.update(best_counter)
        report["benchmarks"][benchmark] = {
            "summary": summary,
            "datasets": rows,
        }

    report["overall"] = {
        "num_queries": len(all_queries),
        "average_mode": "benchmark_mean",
        "avg_g1": mean(row["avg_g1"] for row in benchmark_summaries),
        "avg_g2": mean(row["avg_g2"] for row in benchmark_summaries),
        "avg_g3": mean(row["avg_g3"] for row in benchmark_summaries),
        "avg_oracle": mean(row["avg_oracle"] for row in benchmark_summaries),
        "avg_oracle_gain_over_g3": mean(
            row["avg_oracle_gain_over_g3"] for row in benchmark_summaries
        ),
        "query_weighted_avg_oracle": mean(row["oracle"] for row in all_queries),
        "query_weighted_avg_oracle_gain_over_g3": mean(row["oracle_gain_over_g3"] for row in all_queries),
        "oracle_choice_counts": dict(all_best),
        "oracle_choice_ratio": {
            g: all_best[g] / len(all_queries) for g in GRANULARITIES
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    report = analyze(args.input_dir)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["overall"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
