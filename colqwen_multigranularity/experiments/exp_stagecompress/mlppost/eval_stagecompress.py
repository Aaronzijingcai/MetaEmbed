from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import configue
import torch

from colpali_engine.trainer.eval_utils import external_evaluate_dataset_loader
from colqwen_multigranularity import eval as base_eval
from colqwen_multigranularity import train as base_train
from colqwen_multigranularity.core import normalize_granularities

from .compression import StageCompressConfig, canonicalize_stagecompress_method
from .modeling_stagecompress import build_stagecompress_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--stagecompress-enabled', action='store_true', default=False)
    parser.add_argument('--stagecompress-compress-stages', type=str, default='none')
    parser.add_argument('--stagecompress-budgets', type=int, nargs=3, default=[0, 0, 0])
    parser.add_argument(
        '--stagecompress-method',
        type=str,
        default='strategy1_softassign',
        choices=['strategy1_softassign', 'strategy3_prumerge', 'strategy4_visionzip', 'strategy5_folder', 'strategy6_scope', 'strategy7_stage_resampler'],
    )
    parser.add_argument('--stagecompress-tau', type=float, default=1.0)
    parser.add_argument('--stagecompress-use-text-context', action='store_true', default=False)
    parser.add_argument('--stagecompress-scorer-heads', type=int, default=8)
    parser.add_argument('--stagecompress-scorer-dropout', type=float, default=0.1)
    parser.add_argument('--stagecompress-debug-shapes', action='store_true', default=False)
    parser.add_argument('--only-eval-keywords', type=str, nargs='*', default=None)
    parser.add_argument('--stagecompress-skip-save', action='store_true', default=False)
    parser.add_argument('--smoke-eval-max-queries', type=int, default=0)
    parser.add_argument('--smoke-eval-max-corpus', type=int, default=0)
    sc_args, remaining = parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + remaining
        args = base_eval.parse_args()
    finally:
        sys.argv = original_argv

    for k, v in vars(sc_args).items():
        setattr(args, k, v)
    return args


def build_config(args: argparse.Namespace) -> StageCompressConfig:
    return StageCompressConfig(
        enabled=bool(args.stagecompress_enabled),
        budgets=tuple(int(v) for v in args.stagecompress_budgets),
        compress_stages=args.stagecompress_compress_stages,
        method=canonicalize_stagecompress_method(args.stagecompress_method),
        tau=float(args.stagecompress_tau),
        use_text_context=bool(args.stagecompress_use_text_context),
        scorer_heads=int(args.stagecompress_scorer_heads),
        scorer_dropout=float(args.stagecompress_scorer_dropout),
        debug_shapes=bool(args.stagecompress_debug_shapes),
    )


def build_model(args: argparse.Namespace, compress_config: StageCompressConfig):
    model = build_stagecompress_model(
        args.model_name_or_path,
        granularities=normalize_granularities(args.granularities),
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        adapter_path=args.adapter_path,
        eval_mode=True,
        compress_config=compress_config,
    )
    stagecompress_dir = None
    if args.adapter_path is not None:
        stagecompress_dir = Path(args.adapter_path)
    elif args.mrl_state_dict_path is not None:
        stagecompress_dir = Path(args.mrl_state_dict_path).parent

    if args.mrl_state_dict_path is not None:
        state_dict = torch.load(args.mrl_state_dict_path, map_location='cpu')
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict, strict=False)

    if stagecompress_dir is not None:
        extra_state = stagecompress_dir / 'stage_compressor.pt'
        if extra_state.exists():
            for name, submodule in model.named_modules():
                if name.endswith('stage_compressor'):
                    submodule.load_state_dict(torch.load(extra_state, map_location='cpu'), strict=False)
                    break
    return model




def _materialize_smoke_dataset(dataset):
    if not hasattr(dataset, 'flatten_indices'):
        return dataset
    try:
        return dataset.flatten_indices(keep_in_memory=True, load_from_cache_file=False)
    except TypeError:
        try:
            return dataset.flatten_indices(keep_in_memory=True)
        except TypeError:
            return dataset.flatten_indices()


def _select_dataset_rows(dataset, indices):
    if hasattr(dataset, 'select'):
        try:
            selected = dataset.select(indices, keep_in_memory=True)
        except TypeError:
            selected = dataset.select(indices)
        return _materialize_smoke_dataset(selected)
    return [dataset[index] for index in indices]


def _limit_eval_dataset(dataset: dict[str, Any], *, max_queries: int, max_corpus: int, dataset_format: str) -> dict[str, Any]:
    if max_queries <= 0 and max_corpus <= 0:
        return dataset

    corpus = dataset['corpus']
    queries = dataset['queries']
    qrels = dataset.get('qrels')

    query_count = len(queries) if max_queries <= 0 else min(max_queries, len(queries))
    query_indices = list(range(query_count))
    limited_queries = _select_dataset_rows(queries, query_indices)

    if dataset_format == 'mmeb':
        candidate_dids = []
        if 'local-did' in limited_queries.column_names:
            local_did_lists = []
            for row in limited_queries['local-did']:
                row = [str(value) for value in row]
                if max_corpus > 0:
                    row = row[:max_corpus]
                local_did_lists.append(row)
                candidate_dids.extend(row)
            limited_queries = limited_queries.remove_columns('local-did').add_column('local-did', local_did_lists)
        did_column = 'corpus-id' if 'corpus-id' in corpus.column_names else 'did'
        keep = []
        seen = set()
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
        return {
            'corpus': _select_dataset_rows(corpus, keep),
            'queries': limited_queries,
            'qrels': qrels,
        }

    query_id_column = 'query-id'
    corpus_id_column = 'corpus-id'
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

    keep = []
    seen = set()
    corpus_ids = [str(value) for value in corpus[corpus_id_column]]
    for index, corpus_id in enumerate(corpus_ids):
        if corpus_id in relevant_corpus_ids and corpus_id not in seen:
            keep.append(index)
            seen.add(corpus_id)
    if max_corpus > 0:
        for index in range(min(max_corpus, len(corpus))):
            if index not in keep:
                keep.append(index)
    if not keep:
        keep = list(range(min(max_corpus if max_corpus > 0 else len(corpus), len(corpus))))

    return {
        'corpus': _select_dataset_rows(corpus, keep),
        'queries': limited_queries,
        'qrels': limited_qrels,
    }


def _build_smoke_limited_loader(eval_dataset_loader: dict, *, max_queries: int, max_corpus: int, dataset_format: str) -> dict:
    if max_queries <= 0 and max_corpus <= 0:
        return eval_dataset_loader

    limited = {}
    for name, factory in eval_dataset_loader.items():
        def _factory(factory=factory):
            dataset = factory()
            return _limit_eval_dataset(
                dataset,
                max_queries=max_queries,
                max_corpus=max_corpus,
                dataset_format=dataset_format,
            )
        limited[name] = _factory
    return limited

def main() -> None:
    args = parse_args()
    base_train._maybe_init_distributed()
    model = build_model(args, build_config(args))
    if torch.cuda.is_available():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        model.to(device)
    processor = base_eval.build_processor(args)
    eval_dataset_loader = configue.load(Path(args.eval_config))
    avg_metric = args.avg_metric
    only_eval_keywords = args.only_eval_keywords if args.only_eval_keywords else None
    if avg_metric is None and args.dataset_format == 'mmeb':
        avg_metric = 'recall_at_5'
    no_eval_keywords = [] if args.include_multilingual else None
    if only_eval_keywords is not None:
        eval_dataset_loader = {k: v for k, v in eval_dataset_loader.items() if any(keyword in k for keyword in only_eval_keywords)}
    eval_dataset_loader = _build_smoke_limited_loader(
        eval_dataset_loader,
        max_queries=int(args.smoke_eval_max_queries),
        max_corpus=int(args.smoke_eval_max_corpus),
        dataset_format=args.dataset_format,
    )

    metrics = external_evaluate_dataset_loader(model=model, processor=processor, eval_dataset_loader=eval_dataset_loader, format=args.dataset_format, batch_query=args.batch_query, batch_passage=args.batch_passage, batch_score=args.batch_score, num_workers=args.num_workers, avg_metric=avg_metric or 'ndcg_at_5', no_eval_keywords=no_eval_keywords, use_v2_retriever=args.use_v2_retriever, vis_output_dir=args.vis_output_dir, v2_do_padding=args.v2_do_padding)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
