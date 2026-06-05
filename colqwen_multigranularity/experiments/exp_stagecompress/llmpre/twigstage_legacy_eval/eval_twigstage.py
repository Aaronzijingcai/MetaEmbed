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

from colpali_engine.trainer.eval_utils import external_evaluate_dataset_loader
from colqwen_multigranularity import eval as base_eval
from colqwen_multigranularity import train as base_train
from colqwen_multigranularity.core import normalize_granularities

from .modeling_twigstage import build_twigstage_model, load_global_mrl_token_state, load_twigstage_state
from .train_twigstage import _parse_keep_ratios, _parse_mrl_groups, _validate_mrl_groups


def _is_rank0() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--num-query-mrl-tokens", type=int, default=16)
    parser.add_argument("--num-doc-mrl-tokens", type=int, default=64)
    parser.add_argument("--shared-query-doc-mrl-tokens", action="store_true", default=False)
    parser.add_argument("--mrl-groups", type=str, default="1,1,1.0;2,4,1.0;4,8,1.0;8,16,1.0;16,64,1.0")
    parser.add_argument("--global-mrl-token-path", type=str, default=None)
    parser.add_argument("--twigstage-state-path", type=str, default=None)
    parser.add_argument("--twigstage-mode", type=str, choices=["mask", "prune"], default="mask")
    parser.add_argument("--twigstage-exit-layer", type=int, default=2)
    parser.add_argument("--twigstage-keep-ratios", type=str, default="1.0,0.5,0.25")
    parser.add_argument("--twigstage-temperature", type=float, default=0.1)
    parser.add_argument("--twigstage-min-mask-value", type=float, default=0.0)
    parser.add_argument("--twigstage-train-prune", action="store_true", default=False)
    parser.add_argument("--twigstage-no-context", action="store_true", default=False)
    parser.add_argument("--only-eval-keywords", type=str, nargs="*", default=None)
    parser.add_argument("--smoke-eval-max-queries", type=int, default=0)
    parser.add_argument("--smoke-eval-max-corpus", type=int, default=0)
    custom_args, remaining = parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + remaining
        args = base_eval.parse_args()
    finally:
        sys.argv = original_argv

    for key, value in vars(custom_args).items():
        setattr(args, key, value)
    if "--query-augmentation-repeats" not in remaining:
        args.query_augmentation_repeats = 0
    if "--document-augmentation-repeats" not in remaining:
        args.document_augmentation_repeats = 0
    return args


def build_model(args: argparse.Namespace):
    twigstage_keep_ratios = _parse_keep_ratios(args.twigstage_keep_ratios)
    model = build_twigstage_model(
        args.model_name_or_path,
        granularities=normalize_granularities(args.granularities),
        num_query_mrl_tokens=args.num_query_mrl_tokens,
        num_doc_mrl_tokens=args.num_doc_mrl_tokens,
        shared_query_doc_mrl_tokens=args.shared_query_doc_mrl_tokens,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        adapter_path=args.adapter_path,
        global_mrl_token_path=args.global_mrl_token_path,
        twigstage_state_path=args.twigstage_state_path,
        eval_mode=True,
        twigstage_mode=args.twigstage_mode,
        twigstage_exit_layer=args.twigstage_exit_layer,
        twigstage_keep_ratios=twigstage_keep_ratios,
        twigstage_temperature=args.twigstage_temperature,
        twigstage_min_mask_value=args.twigstage_min_mask_value,
        twigstage_train_prune=args.twigstage_train_prune,
        twigstage_use_context=not args.twigstage_no_context,
    )
    if args.mrl_state_dict_path is not None:
        state_dict = torch.load(args.mrl_state_dict_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
    if args.global_mrl_token_path is None and args.adapter_path is not None:
        candidate = Path(args.adapter_path) / "global_mrl_tokens.pt"
        if candidate.exists():
            load_global_mrl_token_state(model, candidate, map_location="cpu")
    if args.twigstage_state_path is None and args.adapter_path is not None:
        candidate = Path(args.adapter_path) / "twigstage_selector.pt"
        if candidate.exists():
            load_twigstage_state(model, args.adapter_path, map_location="cpu")
    return model


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
            return _limit_eval_dataset(
                factory(),
                max_queries=max_queries,
                max_corpus=max_corpus,
                dataset_format=dataset_format,
            )

        limited[name] = _factory
    return limited


def main() -> None:
    args = parse_args()
    base_train._maybe_init_distributed()
    try:
        mrl_groups = _parse_mrl_groups(args.mrl_groups)
        _validate_mrl_groups(mrl_groups, num_query_tokens=args.num_query_mrl_tokens, num_doc_tokens=args.num_doc_mrl_tokens)
        if _is_rank0():
            print(
                "[TwigStage eval] "
                f"mode={args.twigstage_mode} adapter={args.adapter_path} eval_config={args.eval_config} "
                f"format={args.dataset_format} groups={[(q, d) for q, d, _ in mrl_groups]} "
                f"smoke=({args.smoke_eval_max_queries}, {args.smoke_eval_max_corpus})",
                flush=True,
            )
        model = build_model(args)
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
            model.to(device)

        processor = base_eval.build_processor(args)
        eval_dataset_loader = configue.load(Path(args.eval_config))
        only_eval_keywords = args.only_eval_keywords if args.only_eval_keywords else None
        if only_eval_keywords is not None:
            eval_dataset_loader = {
                key: value for key, value in eval_dataset_loader.items() if any(keyword in key for keyword in only_eval_keywords)
            }
        eval_dataset_loader = _build_smoke_limited_loader(
            eval_dataset_loader,
            max_queries=int(args.smoke_eval_max_queries),
            max_corpus=int(args.smoke_eval_max_corpus),
            dataset_format=args.dataset_format,
        )
        avg_metric = args.avg_metric
        if avg_metric is None and args.dataset_format == "mmeb":
            avg_metric = "recall_at_5"
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
        base_train._cleanup_distributed()


if __name__ == "__main__":
    main()
