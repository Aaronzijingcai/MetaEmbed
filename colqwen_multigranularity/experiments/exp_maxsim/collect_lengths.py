from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, median
from typing import Any

import configue

from colqwen_multigranularity import eval as base_eval

CONFIGS = [
    ("vidore_v1", "colqwen_multigranularity/configs/eval/test_data_vidore_beir.yaml"),
    ("vidore_v2", "colqwen_multigranularity/configs/eval/test_data_mast_v2.yaml"),
    ("mmeb", "colqwen_multigranularity/configs/eval/test_data_mast_mmeb_v3.yaml"),
]
CONFIG_MAP = {name: path for name, path in CONFIGS}


def summarize(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {"count": 0}
    s = sorted(int(x) for x in lengths)
    n = len(s)

    def pct(p: float) -> int:
        idx = min(n - 1, max(0, round((n - 1) * p)))
        return s[idx]

    return {
        "count": n,
        "min": s[0],
        "max": s[-1],
        "mean": round(mean(s), 4),
        "median": median(s),
        "p10": pct(0.10),
        "p25": pct(0.25),
        "p50": pct(0.50),
        "p75": pct(0.75),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
    }


def get_col(dataset, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in dataset.column_names:
            return name
    return None


def chunk_pairs(texts: list[Any], images: list[Any], batch_size: int):
    for i in range(0, len(texts), batch_size):
        yield texts[i : i + batch_size], images[i : i + batch_size]


def lengths_from_batch(batch) -> list[int]:
    return batch["attention_mask"].sum(dim=1).tolist()


def collect_query_lengths(processor, dataset, batch_size: int) -> list[int]:
    queries = dataset["queries"]
    text_key = get_col(queries, ["query", "query_txt", "text", "qry", "content", "caption"])
    image_key = get_col(queries, ["image", "query_img", "qry_image", "query_image", "img"])
    if text_key is None and image_key is None:
        raise ValueError(f"No usable query columns: {queries.column_names}")

    texts = list(queries[text_key]) if text_key else [None] * len(queries)
    images = list(queries[image_key]) if image_key else [None] * len(queries)
    out: list[int] = []
    for batch_texts, batch_images in chunk_pairs(texts, images, batch_size):
        batch = processor.process_mm_queries(batch_texts, batch_images, is_train=False)
        out.extend(lengths_from_batch(batch))
    return out


def collect_doc_lengths(processor, dataset, batch_size: int) -> list[int]:
    corpus = dataset["corpus"]
    text_key = get_col(corpus, ["text", "txt", "content", "doc_text", "caption"])
    image_key = get_col(corpus, ["image", "img", "doc_image"])
    if text_key is None and image_key is None:
        raise ValueError(f"No usable corpus columns: {corpus.column_names}")

    texts = list(corpus[text_key]) if text_key else [None] * len(corpus)
    images = list(corpus[image_key]) if image_key else [None] * len(corpus)
    out: list[int] = []
    for batch_texts, batch_images in chunk_pairs(texts, images, batch_size):
        batch = processor.process_mm_documents(batch_texts, batch_images, is_train=False)
        out.extend(lengths_from_batch(batch))
    return out


def write_md(results: dict[str, dict[str, Any]], output_md: Path) -> None:
    lines = ["# MRL Sequence Length Statistics", ""]
    for group_name, group_results in results.items():
        lines.append(f"## {group_name}")
        lines.append("")
        lines.append("| Dataset | Query N | Query Mean | Query P50 | Query P90 | Query Max | Target N | Target Mean | Target P50 | Target P90 | Target Max |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for ds_name, stats in group_results.items():
            q = stats["query_summary"]
            d = stats["target_summary"]
            lines.append(
                f"| {ds_name} | {q['count']} | {q['mean']} | {q['p50']} | {q['p90']} | {q['max']} | {d['count']} | {d['mean']} | {d['p50']} | {d['p90']} | {d['max']} |"
            )
        lines.append("")
    output_md.write_text("\n".join(lines) + "\n")


def build_processor_args(batch_query: int, batch_doc: int) -> argparse.Namespace:
    return argparse.Namespace(
        model_name_or_path="/MURE-V2/code/MetaEmbed/colqwen_multigranularity/models/colqwen2.5-base",
        processor_name_or_path="/MURE-V2/code/MetaEmbed/colqwen_multigranularity/models/colqwen2.5-base",
        adapter_path=None,
        mrl_state_dict_path=None,
        eval_config=None,
        dataset_format="beir",
        output_path="",
        vis_output_dir=None,
        batch_query=batch_query,
        batch_passage=batch_doc,
        batch_score=16,
        num_workers=0,
        avg_metric=None,
        granularities=[1, 2, 4],
        truncation_len=16384,
        processor_max_length=None,
        query_augmentation_repeats=10,
        document_augmentation_repeats=0,
        include_multilingual=True,
        drop_query_text_if_image=False,
        drop_doc_text_if_image=False,
        attn_implementation="flash_attention_2",
        use_simple_prompt=True,
        resize_crops_to_page=True,
        crop_resize_mode=None,
        use_v2_retriever=True,
        v2_do_padding=True,
    )


def run_single_dataset(group_name: str, config_path: str, ds_name: str, batch_query: int, batch_doc: int) -> tuple[str, str, dict[str, Any]]:
    print(f"[collect_lengths] start dataset={ds_name} group={group_name}", flush=True)
    loader = configue.load(Path(config_path))
    dataset = loader[ds_name]()
    processor = base_eval.build_processor(build_processor_args(batch_query, batch_doc))
    qlens = collect_query_lengths(processor, dataset, batch_query)
    print(f"[collect_lengths] queries done dataset={ds_name} n={len(qlens)}", flush=True)
    dlens = collect_doc_lengths(processor, dataset, batch_doc)
    print(f"[collect_lengths] docs done dataset={ds_name} n={len(dlens)}", flush=True)
    result = {
        "query_lengths": qlens,
        "target_lengths": dlens,
        "query_summary": summarize(qlens),
        "target_summary": summarize(dlens),
    }
    print(f"[collect_lengths] summary dataset={ds_name} query={result['query_summary']} target={result['target_summary']}", flush=True)
    return group_name, ds_name, result


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_progress(results: dict[str, dict[str, Any]], output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    write_md(results, output_md)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-json", default="colqwen_multigranularity/experiments/exp_maxsim/results/all_lengths.json")
    ap.add_argument("--output-md", default="colqwen_multigranularity/experiments/exp_maxsim/results/all_lengths.md")
    ap.add_argument("--batch-query", type=int, default=64)
    ap.add_argument("--batch-doc", type=int, default=16)
    ap.add_argument("--max-workers", type=int, default=min(4, max(1, (os.cpu_count() or 1) // 2)))
    ap.add_argument("--groups", nargs='+', default=[name for name, _ in CONFIGS], choices=[name for name, _ in CONFIGS])
    ap.add_argument("--dataset-names", nargs='*', default=None)
    ap.add_argument("--resume", action='store_true')
    args = ap.parse_args()

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    all_results: dict[str, dict[str, Any]] = load_existing(output_json) if args.resume else {}

    for group_name in args.groups:
        config_path = CONFIG_MAP[group_name]
        print(f"[collect_lengths] group={group_name} config={config_path}", flush=True)
        loader = configue.load(Path(config_path))
        dataset_names = list(loader.keys())
        if args.dataset_names:
            requested = set(args.dataset_names)
            dataset_names = [name for name in dataset_names if name in requested]

        existing_group = all_results.get(group_name, {})
        pending = [name for name in dataset_names if name not in existing_group]
        if not pending:
            print(f"[collect_lengths] group={group_name} already complete", flush=True)
            continue

        group_results = dict(existing_group)
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(run_single_dataset, group_name, config_path, ds_name, args.batch_query, args.batch_doc)
                for ds_name in pending
            ]
            for future in as_completed(futures):
                finished_group, ds_name, result = future.result()
                group_results[ds_name] = result
                all_results[finished_group] = {name: group_results[name] for name in dataset_names if name in group_results}
                save_progress(all_results, output_json, output_md)
                print(f"[collect_lengths] checkpoint-saved group={finished_group} dataset={ds_name}", flush=True)

        all_results[group_name] = {name: group_results[name] for name in dataset_names if name in group_results}
        save_progress(all_results, output_json, output_md)

    print(f"[collect_lengths] wrote json={output_json} md={output_md}", flush=True)


if __name__ == '__main__':
    main()
