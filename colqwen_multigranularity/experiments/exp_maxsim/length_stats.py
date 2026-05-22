from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import configue
import torch
from datasets import Dataset

from colqwen_multigranularity import eval as base_eval
from colqwen_multigranularity import train as base_train
from colpali_engine.utils.dataset_transformation import TestSetFactoryBEIR


def valid_lengths(embeddings: torch.Tensor) -> list[int]:
    return embeddings.abs().sum(dim=-1).ne(0).sum(dim=1).tolist()


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


def batched(iterable: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


def dataset_to_lists(dataset: Dataset, text_key: str, image_key: str) -> tuple[list[Any], list[Any]]:
    texts = dataset[text_key] if text_key in dataset.column_names else [None] * len(dataset)
    images = dataset[image_key] if image_key in dataset.column_names else [None] * len(dataset)
    return list(texts), list(images)


def encode_queries(model, processor, dataset: dict[str, Dataset], batch_size: int) -> list[int]:
    queries = dataset["queries"]
    if "query" in queries.column_names:
        text_key = "query"
    elif "query_txt" in queries.column_names:
        text_key = "query_txt"
    elif "text" in queries.column_names:
        text_key = "text"
    else:
        raise ValueError(f"Unknown query text columns: {queries.column_names}")
    image_key = "image" if "image" in queries.column_names else ("query_img" if "query_img" in queries.column_names else None)
    texts, images = dataset_to_lists(queries, text_key, image_key) if image_key else (list(queries[text_key]), [None] * len(queries))
    out = []
    for chunk in batched(list(zip(texts, images)), batch_size):
        t = [x[0] for x in chunk]
        i = [x[1] for x in chunk]
        batch = processor.process_mm_queries(t, i, is_train=False).to(model.device)
        with torch.inference_mode():
            emb = model(**batch)
        out.extend(valid_lengths(emb))
    return out


def encode_docs(model, processor, dataset: dict[str, Dataset], batch_size: int) -> list[int]:
    corpus = dataset["corpus"]
    if "text" in corpus.column_names:
        text_key = "text"
    elif "txt" in corpus.column_names:
        text_key = "txt"
    elif "content" in corpus.column_names:
        text_key = "content"
    else:
        text_key = None
    image_key = "image" if "image" in corpus.column_names else ("img" if "img" in corpus.column_names else None)
    texts = list(corpus[text_key]) if text_key else [None] * len(corpus)
    images = list(corpus[image_key]) if image_key else [None] * len(corpus)
    out = []
    for chunk in batched(list(zip(texts, images)), batch_size):
        t = [x[0] for x in chunk]
        i = [x[1] for x in chunk]
        batch = processor.process_mm_documents(t, i, is_train=False).to(model.device)
        with torch.inference_mode():
            emb = model(**batch)
        out.extend(valid_lengths(emb))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name-or-path", default="/MURE-V2/code/MetaEmbed/colqwen_multigranularity/models/colqwen2.5-base")
    ap.add_argument("--processor-name-or-path", default="/MURE-V2/code/MetaEmbed/colqwen_multigranularity/models/colqwen2.5-base")
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-query", type=int, default=8)
    ap.add_argument("--batch-doc", type=int, default=4)
    ap.add_argument("--granularities", type=int, nargs="+", default=[1, 2, 4])
    args = ap.parse_args()

    base_train._maybe_init_distributed()
    try:
        eval_args = argparse.Namespace(
            model_name_or_path=args.model_name_or_path,
            processor_name_or_path=args.processor_name_or_path,
            adapter_path=None,
            mrl_state_dict_path=None,
            eval_config=args.config,
            dataset_format="beir",
            output_path=args.output,
            vis_output_dir=None,
            batch_query=args.batch_query,
            batch_passage=args.batch_doc,
            batch_score=16,
            num_workers=0,
            avg_metric=None,
            granularities=args.granularities,
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
        model = base_eval.build_model(eval_args)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        processor = base_eval.build_processor(eval_args)
        dataset_loader = configue.load(Path(args.config))

        results = {}
        for name, factory in dataset_loader.items():
            dataset = factory()
            qlens = encode_queries(model, processor, dataset, args.batch_query)
            dlens = encode_docs(model, processor, dataset, args.batch_doc)
            results[name] = {
                "query_lengths": qlens,
                "target_lengths": dlens,
                "query_summary": summarize(qlens),
                "target_summary": summarize(dlens),
            }
            print(name, results[name]["query_summary"], results[name]["target_summary"], flush=True)

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
