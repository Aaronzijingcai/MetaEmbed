from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import configue
import torch
from torch import distributed as dist

from colpali_engine.trainer.eval_utils import external_evaluate_dataset_loader
from colqwen_multigranularity import eval as base_eval
from colqwen_multigranularity import train as base_train
from colqwen_multigranularity.core import normalize_granularities

from .config import FolderGlobalHomoConfig
from .modeling_folder_global_homo import build_folder_global_homo_model


def _maybe_init_distributed() -> None:
    if dist.is_available() and (not dist.is_initialized()) and 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = 'nccl'
        else:
            backend = 'gloo'
        dist.init_process_group(backend=backend, init_method='env://', world_size=world_size, rank=rank)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--folder-global-homo-enabled', action='store_true', default=False)
    parser.add_argument('--folder-global-homo-compress-stages', type=str, default='all')
    parser.add_argument('--folder-global-homo-budgets', type=int, nargs=3, default=[160, 160, 160])
    parser.add_argument('--folder-global-homo-novelty-weight', type=float, default=1.0)
    parser.add_argument('--folder-global-homo-global-guidance-weight', type=float, default=0.5)
    parser.add_argument('--folder-global-homo-global-min-budget-ratio', type=float, default=0.6)
    parser.add_argument('--folder-global-homo-gate-strength', type=float, default=0.25)
    parser.add_argument('--folder-global-homo-folder-alpha', type=float, default=1.0)
    parser.add_argument('--folder-global-homo-tau', type=float, default=1.0)
    parser.add_argument('--folder-global-homo-detach-anchors', action='store_true', default=True)
    parser.add_argument('--folder-global-homo-no-detach-anchors', action='store_false', dest='folder_global_homo_detach_anchors')
    parser.add_argument('--folder-global-homo-use-text-context', action='store_true', default=False)
    parser.add_argument('--folder-global-homo-scorer-heads', type=int, default=8)
    parser.add_argument('--folder-global-homo-scorer-dropout', type=float, default=0.1)
    parser.add_argument('--folder-global-homo-debug-shapes', action='store_true', default=False)
    parser.add_argument('--only-eval-keywords', type=str, nargs='*', default=None)
    parser.add_argument('--smoke-eval-max-queries', type=int, default=0)
    parser.add_argument('--smoke-eval-max-corpus', type=int, default=0)
    homo_args, remaining = parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + remaining
        args = base_eval.parse_args()
    finally:
        sys.argv = original_argv

    for key, value in vars(homo_args).items():
        setattr(args, key, value)
    return args


def build_config(args: argparse.Namespace) -> FolderGlobalHomoConfig:
    return FolderGlobalHomoConfig(
        enabled=bool(args.folder_global_homo_enabled),
        budgets=tuple(int(value) for value in args.folder_global_homo_budgets),
        compress_stages=args.folder_global_homo_compress_stages,
        novelty_weight=float(args.folder_global_homo_novelty_weight),
        global_guidance_weight=float(args.folder_global_homo_global_guidance_weight),
        global_min_budget_ratio=float(args.folder_global_homo_global_min_budget_ratio),
        gate_strength=float(args.folder_global_homo_gate_strength),
        folder_alpha=float(args.folder_global_homo_folder_alpha),
        tau=float(args.folder_global_homo_tau),
        detach_anchors=bool(args.folder_global_homo_detach_anchors),
        use_text_context=bool(args.folder_global_homo_use_text_context),
        scorer_heads=int(args.folder_global_homo_scorer_heads),
        scorer_dropout=float(args.folder_global_homo_scorer_dropout),
        debug_shapes=bool(args.folder_global_homo_debug_shapes),
    )


def build_model(args: argparse.Namespace, folder_global_homo_config: FolderGlobalHomoConfig):
    model = build_folder_global_homo_model(
        args.model_name_or_path,
        granularities=normalize_granularities(args.granularities),
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        adapter_path=args.adapter_path,
        eval_mode=True,
        folder_global_homo_config=folder_global_homo_config,
    )
    folder_global_homo_dir: Optional[Path] = None
    if args.adapter_path is not None:
        folder_global_homo_dir = Path(args.adapter_path)
    elif args.mrl_state_dict_path is not None:
        folder_global_homo_dir = Path(args.mrl_state_dict_path).parent

    if args.mrl_state_dict_path is not None:
        state_dict = torch.load(args.mrl_state_dict_path, map_location='cpu')
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict, strict=False)

    if folder_global_homo_dir is not None:
        extra_state = folder_global_homo_dir / 'folder_global_homo.pt'
        if extra_state.exists():
            for name, submodule in model.named_modules():
                if name.endswith('folder_global_homo'):
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
        for idx, did in enumerate(corpus[did_column]):
            did = str(did)
            if did in candidate_dids and did not in seen:
                keep.append(idx)
                seen.add(did)
        limited_corpus = _select_dataset_rows(corpus, keep)
        return {'queries': limited_queries, 'corpus': limited_corpus, 'qrels': qrels}
    corpus_count = len(corpus) if max_corpus <= 0 else min(max_corpus, len(corpus))
    limited_corpus = _select_dataset_rows(corpus, list(range(corpus_count)))
    if qrels is not None:
        limited_query_ids = set(limited_queries['query_id']) if 'query_id' in limited_queries.column_names else None
        if limited_query_ids is not None:
            qrels = {qid: rels for qid, rels in qrels.items() if qid in limited_query_ids}
    return {'queries': limited_queries, 'corpus': limited_corpus, 'qrels': qrels}


def _build_smoke_limited_loader(eval_dataset_loader: dict, *, max_queries: int, max_corpus: int, dataset_format: str):
    if max_queries <= 0 and max_corpus <= 0:
        return eval_dataset_loader
    limited = {}
    for name, factory in eval_dataset_loader.items():
        def _factory(factory=factory):
            dataset = factory()
            return _limit_eval_dataset(dataset, max_queries=max_queries, max_corpus=max_corpus, dataset_format=dataset_format)
        limited[name] = _factory
    return limited


def _run_eval(args: argparse.Namespace, model, processor, eval_dataset_loader: dict, *, dataset_format: str, avg_metric: str):
    return external_evaluate_dataset_loader(
        model=model,
        processor=processor,
        eval_dataset_loader=eval_dataset_loader,
        format=dataset_format,
        batch_query=args.batch_query,
        batch_passage=args.batch_passage,
        batch_score=args.batch_score,
        num_workers=args.num_workers,
        avg_metric=avg_metric,
        no_eval_keywords=[] if args.include_multilingual else None,
        use_v2_retriever=args.use_v2_retriever,
        vis_output_dir=args.vis_output_dir,
        v2_do_padding=args.v2_do_padding,
    )


def main() -> None:
    args = parse_args()
    base_train._maybe_init_distributed()
    folder_global_homo_config = build_config(args)
    model = build_model(args, folder_global_homo_config)
    if torch.cuda.is_available():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        model.to(device)
    processor = base_eval.build_processor(args)
    eval_dataset_loader = configue.load(Path(args.eval_config))
    only_eval_keywords = args.only_eval_keywords if args.only_eval_keywords else None
    if only_eval_keywords is not None:
        eval_dataset_loader = {key: value for key, value in eval_dataset_loader.items() if any(keyword in key for keyword in only_eval_keywords)}
    eval_dataset_loader = _build_smoke_limited_loader(
        eval_dataset_loader,
        max_queries=int(args.smoke_eval_max_queries),
        max_corpus=int(args.smoke_eval_max_corpus),
        dataset_format=args.dataset_format,
    )
    is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
    if is_rank0:
        print(json.dumps({
            'event': 'folder_global_homo_eval_start',
            'compress_stages': folder_global_homo_config.compress_stages,
            'budgets': folder_global_homo_config.budgets,
            'novelty_weight': folder_global_homo_config.novelty_weight,
            'gate_strength': folder_global_homo_config.gate_strength,
            'eval_config': str(args.eval_config),
            'dataset_format': args.dataset_format,
            'only_eval_keywords': only_eval_keywords,
        }, ensure_ascii=False))
    metrics = _run_eval(args, model, processor, eval_dataset_loader, dataset_format=args.dataset_format, avg_metric=args.avg_metric or ('recall_at_5' if args.dataset_format == 'mmeb' else 'ndcg_at_5'))
    if is_rank0:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if dist.is_initialized():
        dist.barrier()
        base_train._cleanup_distributed()


if __name__ == '__main__':
    main()
