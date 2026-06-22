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

_PROJECT_DIR = Path(__file__).resolve().parents[4]
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

from colpali_engine.trainer.eval_utils import external_evaluate_dataset_loader
from colqwen_multigranularity import train as base_train
from colqwen_multigranularity.core import normalize_granularities
from colqwen_multigranularity.processing import MultiGranularityColQwen2_5Processor
from colqwen_multigranularity.experiments.exp_stagecompress.llmpre.visionselector_mrl.modeling_visionselector_mrl import build_visionselector_mrl_model


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


def _parse_keep_ratios(raw: str) -> list[float]:
    values = [float(value.strip()) for value in str(raw).replace(";", ",").split(",") if value.strip()]
    if len(values) != 3:
        raise ValueError(f"Expected exactly three keep ratios for g1/g2/g3, got {values}.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default=str(_PROJECT_DIR / "models/colqwen2.5-base"))
    parser.add_argument("--processor-name-or-path", type=str, default=str(_PROJECT_DIR / "models/colqwen2.5-base"))
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--visionselector-mrl-state-path", type=str, default=None)
    parser.add_argument("--eval-config", type=str, required=True)
    parser.add_argument("--dataset-format", type=str, default="beir")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--vis-output-dir", type=str, default=None)
    parser.add_argument("--batch-query", type=int, default=4)
    parser.add_argument("--batch-passage", type=int, default=4)
    parser.add_argument("--batch-score", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--avg-metric", type=str, default=None)
    parser.add_argument("--granularities", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--truncation-len", type=int, default=16384)
    parser.add_argument("--processor-max-length", type=int, default=None)
    parser.add_argument("--query-augmentation-repeats", type=int, default=10)
    parser.add_argument("--document-augmentation-repeats", type=int, default=0)
    parser.add_argument("--include-multilingual", action="store_true")
    parser.add_argument("--drop-query-text-if-image", action="store_true", default=False)
    parser.add_argument("--drop-doc-text-if-image", action="store_true", default=False)
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    parser.add_argument("--visionselector-mode", type=str, choices=["mask", "prune"], default="prune")
    parser.add_argument("--visionselector-position", type=str, choices=["adapter_pre"], default="adapter_pre")
    parser.add_argument("--visionselector-keep-ratios", type=str, default="1.0,0.5,0.25")
    parser.add_argument("--visionselector-scorer-hidden-dim", type=int, default=1792)
    parser.add_argument("--visionselector-init-scale", type=float, default=1e-4)
    parser.add_argument("--only-eval-keywords", type=str, nargs="*", default=None)
    parser.add_argument("--smoke-eval-max-queries", type=int, default=0)
    parser.add_argument("--smoke-eval-max-corpus", type=int, default=0)
    parser.add_argument("--use-simple-prompt", action="store_true", dest="use_simple_prompt")
    parser.add_argument("--no-use-simple-prompt", action="store_false", dest="use_simple_prompt")
    parser.add_argument("--resize-crops-to-page", action="store_true", dest="resize_crops_to_page")
    parser.add_argument("--no-resize-crops-to-page", action="store_false", dest="resize_crops_to_page")
    parser.add_argument("--crop-resize-mode", type=str, default=None, choices=["stretch", "none"])
    parser.add_argument("--use-v2-retriever", action="store_true", dest="use_v2_retriever")
    parser.add_argument("--no-use-v2-retriever", action="store_false", dest="use_v2_retriever")
    parser.add_argument("--v2-do-padding", action="store_true", dest="v2_do_padding")
    parser.add_argument("--no-v2-do-padding", action="store_false", dest="v2_do_padding")
    parser.set_defaults(use_simple_prompt=True, resize_crops_to_page=True, use_v2_retriever=True, v2_do_padding=True)
    return parser.parse_args()


def build_processor(args: argparse.Namespace) -> MultiGranularityColQwen2_5Processor:
    kwargs = {
        "granularities": normalize_granularities(args.granularities),
        "truncation_len": args.truncation_len,
        "use_simple_prompt": args.use_simple_prompt,
        "resize_crops_to_page": args.resize_crops_to_page,
        "crop_resize_mode": args.crop_resize_mode,
        "query_augmentation_repeats": args.query_augmentation_repeats,
        "document_augmentation_repeats": args.document_augmentation_repeats,
        "drop_query_text_if_image": args.drop_query_text_if_image,
        "drop_doc_text_if_image": args.drop_doc_text_if_image,
    }
    if args.processor_max_length is not None:
        kwargs["processor_max_length"] = args.processor_max_length
    return MultiGranularityColQwen2_5Processor.from_pretrained(args.processor_name_or_path, **kwargs)


def build_model(args: argparse.Namespace):
    state_path = args.visionselector_mrl_state_path
    if state_path is None and args.adapter_path is not None:
        candidate = Path(args.adapter_path) / "visionselector_mrl_selector.pt"
        if candidate.exists():
            state_path = str(candidate)
    return build_visionselector_mrl_model(
        args.model_name_or_path,
        granularities=normalize_granularities(args.granularities),
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        adapter_path=args.adapter_path,
        visionselector_mrl_state_path=state_path,
        eval_mode=True,
        compact_query_tokens=True,
        visionselector_mode=args.visionselector_mode,
        visionselector_position=args.visionselector_position,
        visionselector_keep_ratios=_parse_keep_ratios(args.visionselector_keep_ratios),
        visionselector_scorer_hidden_dim=args.visionselector_scorer_hidden_dim,
        visionselector_init_scale=args.visionselector_init_scale,
    )


def _split_mixed_eval_loader(eval_dataset_loader: dict):
    beir_loader = {}
    mmeb_loader = {}
    for name, factory in eval_dataset_loader.items():
        if str(name).startswith("MMEB-eval-"):
            mmeb_loader[name] = factory
        else:
            beir_loader[name] = factory
    return beir_loader, mmeb_loader


def _is_rank0() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


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


def main() -> None:
    args = parse_args()
    base_train._maybe_init_distributed()
    model = build_model(args)

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        model.to(device)

    processor = build_processor(args)
    eval_dataset_loader = configue.load(Path(args.eval_config))
    no_eval_keywords = [] if args.include_multilingual else None
    if args.only_eval_keywords:
        eval_dataset_loader = {
            name: factory
            for name, factory in eval_dataset_loader.items()
            if any(keyword in name for keyword in args.only_eval_keywords)
        }
    eval_dataset_loader = _build_smoke_limited_loader(
        eval_dataset_loader,
        max_queries=int(args.smoke_eval_max_queries),
        max_corpus=int(args.smoke_eval_max_corpus),
        dataset_format=args.dataset_format,
    )
    beir_loader, mmeb_loader = _split_mixed_eval_loader(eval_dataset_loader)

    if args.dataset_format == "beir" and mmeb_loader and beir_loader:
        metrics = {}
        beir_metrics = external_evaluate_dataset_loader(
            model=model,
            processor=processor,
            eval_dataset_loader=beir_loader,
            format="beir",
            batch_query=args.batch_query,
            batch_passage=args.batch_passage,
            batch_score=args.batch_score,
            num_workers=args.num_workers,
            avg_metric=args.avg_metric or "ndcg_at_5",
            no_eval_keywords=no_eval_keywords,
            use_v2_retriever=args.use_v2_retriever,
            vis_output_dir=args.vis_output_dir,
            v2_do_padding=args.v2_do_padding,
        )
        metrics.update(beir_metrics)
        mmeb_metrics = external_evaluate_dataset_loader(
            model=model,
            processor=processor,
            eval_dataset_loader=mmeb_loader,
            format="mmeb",
            batch_query=args.batch_query,
            batch_passage=args.batch_passage,
            batch_score=args.batch_score,
            num_workers=args.num_workers,
            avg_metric=args.avg_metric or "recall_at_1",
            no_eval_keywords=no_eval_keywords,
            use_v2_retriever=args.use_v2_retriever,
            vis_output_dir=args.vis_output_dir,
            v2_do_padding=args.v2_do_padding,
        )
        metrics.update(mmeb_metrics)
    else:
        avg_metric = args.avg_metric
        if avg_metric is None and args.dataset_format == "mmeb":
            avg_metric = "recall_at_1"
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
            use_v2_retriever=args.use_v2_retriever,
            vis_output_dir=args.vis_output_dir,
            v2_do_padding=args.v2_do_padding,
        )

    output_path = Path(args.output_path)
    is_rank0 = _is_rank0()
    if is_rank0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if dist.is_initialized():
        dist.barrier()
        base_train._cleanup_distributed()


if __name__ == "__main__":
    main()
