from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import configue
import torch
import torch.distributed as dist
from peft import PeftModel

_PROJECT_DIR = Path(__file__).resolve().parents[3]
_ROOT_DIR = _PROJECT_DIR.parent
_VENDOR_DIR = _PROJECT_DIR / "vendor"
if _VENDOR_DIR.exists():
    _VENDOR_PATH = str(_VENDOR_DIR)
    if _VENDOR_PATH in sys.path:
        sys.path.remove(_VENDOR_PATH)
    sys.path.insert(0, _VENDOR_PATH)
if str(_ROOT_DIR) not in sys.path:
    sys.path.append(str(_ROOT_DIR))

os.environ.setdefault("MURE_CACHE_ROOT", str(_PROJECT_DIR / ".cache"))
os.environ.setdefault("HF_HOME", str(Path(os.environ["MURE_CACHE_ROOT"]) / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(Path(os.environ["HF_HOME"]) / "datasets"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
os.environ.setdefault("TMPDIR", str(Path(os.environ["MURE_CACHE_ROOT"]) / "tmp"))
os.environ.setdefault("DATA_DIR", str(_PROJECT_DIR / "data_dir") + "/")
os.environ.setdefault("CACHED_DATA_DIR", str(_PROJECT_DIR / "cached_data_dir"))

from colpali_engine.models.qwen2_5.colqwen2_5.mm_processing_colqwen2_5 import MultimodalColQwen2_5_Processor
from colpali_engine.models.qwen2_5.lastqwen2_5.modeling_lastqwen2_5_new import LastQwen2_5
from colpali_engine.trainer.eval_utils import external_evaluate_dataset_loader


def _maybe_init_distributed() -> None:
    if dist.is_available() and (not dist.is_initialized()) and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(backend=backend, init_method="env://", world_size=world_size, rank=rank)


def _is_rank0() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def _parse_mrl_groups(raw: str) -> list[tuple[int, int, float]]:
    groups = []
    for chunk in str(raw).replace(";", " ").split():
        values = [value.strip() for value in chunk.split(",") if value.strip()]
        if len(values) not in {2, 3}:
            raise ValueError(f"Invalid MRL group {chunk!r}; expected q,d or q,d,weight.")
        q_tokens = int(values[0])
        d_tokens = int(values[1])
        weight = float(values[2]) if len(values) == 3 else 1.0
        groups.append((q_tokens, d_tokens, weight))
    if not groups:
        raise ValueError("At least one MRL group is required.")
    return groups


def _materialize_smoke_dataset(dataset):
    if not hasattr(dataset, "flatten_indices"):
        return dataset
    try:
        return dataset.flatten_indices(keep_in_memory=True, load_from_cache_file=False)
    except TypeError:
        try:
            return dataset.flatten_indices(keep_in_memory=True)
        except TypeError:
            return dataset.flatten_indices()


def _select_dataset_rows(dataset, indices):
    if hasattr(dataset, "select"):
        try:
            selected = dataset.select(indices, keep_in_memory=True)
        except TypeError:
            selected = dataset.select(indices)
        return _materialize_smoke_dataset(selected)
    return [dataset[index] for index in indices]


def _limit_eval_dataset(dataset: dict[str, Any], *, max_queries: int, max_corpus: int, dataset_format: str) -> dict[str, Any]:
    if max_queries <= 0 and max_corpus <= 0:
        return dataset
    corpus = dataset["corpus"]
    queries = dataset["queries"]
    qrels = dataset.get("qrels")
    query_count = len(queries) if max_queries <= 0 else min(max_queries, len(queries))
    limited_queries = _select_dataset_rows(queries, list(range(query_count)))

    if dataset_format == "mmeb":
        candidate_dids = []
        if "local-did" in limited_queries.column_names:
            local_did_lists = []
            for row in limited_queries["local-did"]:
                row = [str(value) for value in row]
                if max_corpus > 0:
                    row = row[:max_corpus]
                local_did_lists.append(row)
                candidate_dids.extend(row)
            limited_queries = limited_queries.remove_columns("local-did").add_column("local-did", local_did_lists)
        did_column = "corpus-id" if "corpus-id" in corpus.column_names else "did"
        keep, seen = [], set()
        for index, did in enumerate(corpus[did_column]):
            did = str(did)
            if did in candidate_dids and did not in seen:
                keep.append(index)
                seen.add(did)
        if max_corpus > 0:
            for index in range(min(max_corpus, len(corpus))):
                if index not in keep:
                    keep.append(index)
        if not keep:
            keep = list(range(min(max_corpus if max_corpus > 0 else len(corpus), len(corpus))))
        return {"corpus": _select_dataset_rows(corpus, keep), "queries": limited_queries, "qrels": qrels}

    query_id_column = "query-id"
    corpus_id_column = "corpus-id"
    selected_query_ids = {str(value) for value in limited_queries[query_id_column]}
    relevant_corpus_ids = []
    limited_qrels = qrels
    if qrels is not None:
        qrel_keep = []
        for index, row in enumerate(qrels):
            if str(row[query_id_column]) in selected_query_ids:
                qrel_keep.append(index)
                relevant_corpus_ids.append(str(row[corpus_id_column]))
        limited_qrels = _select_dataset_rows(qrels, qrel_keep)

    keep, seen = [], set()
    for index, corpus_id in enumerate([str(value) for value in corpus[corpus_id_column]]):
        if corpus_id in relevant_corpus_ids and corpus_id not in seen:
            keep.append(index)
            seen.add(corpus_id)
    if max_corpus > 0:
        for index in range(min(max_corpus, len(corpus))):
            if index not in keep:
                keep.append(index)
    if not keep:
        keep = list(range(min(max_corpus if max_corpus > 0 else len(corpus), len(corpus))))
    return {"corpus": _select_dataset_rows(corpus, keep), "queries": limited_queries, "qrels": limited_qrels}


def _build_smoke_limited_loader(eval_dataset_loader: dict, *, max_queries: int, max_corpus: int, dataset_format: str) -> dict:
    if max_queries <= 0 and max_corpus <= 0:
        return eval_dataset_loader
    limited = {}
    for name, factory in eval_dataset_loader.items():
        def _factory(factory=factory):
            return _limit_eval_dataset(factory(), max_queries=max_queries, max_corpus=max_corpus, dataset_format=dataset_format)
        limited[name] = _factory
    return limited


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default=str(_PROJECT_DIR / "models/colqwen2.5-base"))
    parser.add_argument("--processor-name-or-path", type=str, default=str(_PROJECT_DIR / "models/colqwen2.5-base"))
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--eval-config", type=str, required=True)
    parser.add_argument("--dataset-format", type=str, default="beir")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--vis-output-dir", type=str, default=None)
    parser.add_argument("--batch-query", type=int, default=4)
    parser.add_argument("--batch-passage", type=int, default=4)
    parser.add_argument("--batch-score", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--avg-metric", type=str, default=None)
    parser.add_argument("--truncation-len", type=int, default=16384)
    parser.add_argument("--max-num-visual-tokens", type=int, default=1024)
    parser.add_argument("--query-augmentation-repeats", type=int, default=0)
    parser.add_argument("--document-augmentation-repeats", type=int, default=0)
    parser.add_argument("--include-multilingual", action="store_true")
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    parser.add_argument("--num-query-prompt-tokens", type=int, default=16)
    parser.add_argument("--num-doc-prompt-tokens", type=int, default=64)
    parser.add_argument("--shared-query-doc-prompt", action="store_true", default=False)
    parser.add_argument("--mrl-groups", type=str, default="1,1,1.0;2,4,1.0;4,8,1.0;8,16,1.0;16,64,1.0")
    parser.add_argument("--only-eval-keywords", type=str, nargs="*", default=None)
    parser.add_argument("--smoke-eval-max-queries", type=int, default=0)
    parser.add_argument("--smoke-eval-max-corpus", type=int, default=0)
    parser.add_argument("--use-v2-retriever", action="store_true", dest="use_v2_retriever")
    parser.add_argument("--no-use-v2-retriever", action="store_false", dest="use_v2_retriever")
    parser.add_argument("--v2-do-padding", action="store_true", dest="v2_do_padding")
    parser.add_argument("--no-v2-do-padding", action="store_false", dest="v2_do_padding")
    parser.set_defaults(use_v2_retriever=True, v2_do_padding=True)
    return parser.parse_args()


def build_processor(args: argparse.Namespace):
    return MultimodalColQwen2_5_Processor.from_pretrained(
        args.processor_name_or_path,
        max_num_visual_tokens=args.max_num_visual_tokens,
        use_simple_prompt=True,
        truncation_len=args.truncation_len,
        query_augmentation_repeats=args.query_augmentation_repeats,
        document_augmentation_repeats=args.document_augmentation_repeats,
    )


def build_model(args: argparse.Namespace):
    model = LastQwen2_5.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        attn_implementation=args.attn_implementation,
        use_liger_kernel=False,
        dim=-1,
        num_query_prompt_tokens=args.num_query_prompt_tokens,
        num_doc_prompt_tokens=args.num_doc_prompt_tokens,
        shared_query_doc_prompt=args.shared_query_doc_prompt,
    )
    model = PeftModel.from_pretrained(model, Path(args.adapter_path))
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    _maybe_init_distributed()
    try:
        mrl_groups = _parse_mrl_groups(args.mrl_groups)
        if _is_rank0():
            print(
                "[OriginalMetaEmbed eval] "
                f"adapter={args.adapter_path} eval_config={args.eval_config} format={args.dataset_format} "
                f"groups={[(q, d) for q, d, _ in mrl_groups]} smoke=({args.smoke_eval_max_queries}, {args.smoke_eval_max_corpus})",
                flush=True,
            )
        model = build_model(args)
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
            model.to(device)

        processor = build_processor(args)
        eval_dataset_loader = configue.load(Path(args.eval_config))
        if args.only_eval_keywords:
            eval_dataset_loader = {
                k: v for k, v in eval_dataset_loader.items()
                if any(keyword in k for keyword in args.only_eval_keywords)
            }
        eval_dataset_loader = _build_smoke_limited_loader(
            eval_dataset_loader,
            max_queries=int(args.smoke_eval_max_queries),
            max_corpus=int(args.smoke_eval_max_corpus),
            dataset_format=args.dataset_format,
        )
        avg_metric = args.avg_metric
        if avg_metric is None and args.dataset_format == "mmeb":
            avg_metric = "recall_at_1"
        no_eval_keywords = [] if args.include_multilingual else None
        metrics = external_evaluate_dataset_loader(
            model=model,
            processor=processor,
            eval_dataset_loader=eval_dataset_loader,
            format=args.dataset_format,
            batch_query=args.batch_query,
            batch_passage=args.batch_passage,
            batch_score=args.batch_score,
            num_workers=args.num_workers,
            avg_metric=avg_metric or "ndcg_at_5",
            no_eval_keywords=no_eval_keywords,
            mrl_groups=[(q, d) for q, d, _ in mrl_groups],
            is_last_model=True,
            use_v2_retriever=args.use_v2_retriever,
            vis_output_dir=args.vis_output_dir,
            v2_do_padding=args.v2_do_padding,
        )
        output_path = Path(args.output_path)
        if _is_rank0():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
            print(json.dumps(metrics, indent=2, ensure_ascii=False))
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
