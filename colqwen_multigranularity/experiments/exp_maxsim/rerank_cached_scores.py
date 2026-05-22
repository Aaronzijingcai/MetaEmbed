from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import configue
import pytrec_eval
import torch

from colqwen_multigranularity.experiments.exp_maxsim.sweep_cached_scores import (
    _build_qrels,
    _encode_passages,
    _encode_queries,
    _load_model,
    _load_processor,
    _normalize_metric_names,
)
from colqwen_multigranularity.experiments.exp_maxsim.symmetric_maxsim import SymmetricMaxSimConfig, score_multi_vector_symmetric


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter-path', type=str, required=True)
    parser.add_argument('--eval-config', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--model-name-or-path', type=str, default='/MURE-V2/code/MetaEmbed/colqwen_multigranularity/models/colqwen2.5-base')
    parser.add_argument('--processor-name-or-path', type=str, default='/MURE-V2/code/MetaEmbed/colqwen_multigranularity/models/colqwen2.5-base')
    parser.add_argument('--granularities', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--attn-implementation', type=str, default='flash_attention_2')
    parser.add_argument('--batch-query', type=int, default=8)
    parser.add_argument('--batch-passage', type=int, default=8)
    parser.add_argument('--batch-score', type=int, default=32)
    parser.add_argument('--dataset-name', type=str, default=None)
    parser.add_argument('--topk-values', type=int, nargs='+', default=[20, 50])
    return parser


def _compute_metrics(qrels, results):
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {
        'ndcg_cut.1,3,5,10,20,50,100',
        'map_cut.1,3,5,10,20,50,100',
        'recall.1,3,5,10,20,50,100',
        'P.1,3,5,10,20,50,100',
        'recip_rank',
    })
    per_query = evaluator.evaluate(results)
    metrics = defaultdict(float)
    n = max(len(per_query), 1)
    for row in per_query.values():
        for k, v in row.items():
            metrics[k] += v
    return _normalize_metric_names({k: v / n for k, v in metrics.items()})


def _full_results(query_ids, passage_ids, scores):
    return {
        qid: {pid: float(scores[i, j]) for j, pid in enumerate(passage_ids)}
        for i, qid in enumerate(query_ids)
    }


def _rerank_results(query_ids, passage_ids, baseline_scores, query_embeddings, passage_embeddings, *, topk, config, batch_score, device):
    results = {}
    for i, qid in enumerate(query_ids):
        row = baseline_scores[i]
        top_idx = torch.topk(row, k=min(topk, row.shape[0]), dim=0).indices.tolist()
        cand_passages = [passage_embeddings[j] for j in top_idx]
        rerank_scores = score_multi_vector_symmetric(
            [query_embeddings[i]],
            cand_passages,
            batch_size=min(batch_score, max(1, len(cand_passages))),
            device=device,
            config=config,
        )[0]
        order = torch.argsort(rerank_scores, descending=True).tolist()
        results[qid] = {
            passage_ids[top_idx[idx]]: float(rerank_scores[idx])
            for idx in order
        }
    return results


def main():
    args = build_parser().parse_args()
    model = _load_model(args)
    processor = _load_processor(args)
    loader = configue.load(Path(args.eval_config))
    dataset_name = args.dataset_name or next(iter(loader.keys()))
    dataset = loader[dataset_name]()

    query_ids, query_embeddings = _encode_queries(model, processor, dataset['queries'], args.batch_query)
    passage_ids, passage_embeddings = _encode_passages(model, processor, dataset['corpus'], args.batch_passage)
    qrels = _build_qrels(dataset['qrels'])

    baseline_config = SymmetricMaxSimConfig(score_mode='query', query_weight=1.0, doc_weight=0.0, doc_chunk_size=args.batch_score)
    baseline_scores = score_multi_vector_symmetric(
        query_embeddings,
        passage_embeddings,
        batch_size=args.batch_score,
        device=model.device,
        config=baseline_config,
    )

    report = {
        'dataset': dataset_name,
        'variants': {}
    }

    baseline_results = _full_results(query_ids, passage_ids, baseline_scores)
    report['variants']['query_full'] = _compute_metrics(qrels, baseline_results)

    bimax_configs = {
        'bimax_0.7_0.3': SymmetricMaxSimConfig(score_mode='bimax', query_weight=0.7, doc_weight=0.3, doc_chunk_size=args.batch_score),
        'bimax_0.5_0.5': SymmetricMaxSimConfig(score_mode='bimax', query_weight=0.5, doc_weight=0.5, doc_chunk_size=args.batch_score),
        'bimax_0.3_0.7': SymmetricMaxSimConfig(score_mode='bimax', query_weight=0.3, doc_weight=0.7, doc_chunk_size=args.batch_score),
    }

    for topk in args.topk_values:
        for name, config in bimax_configs.items():
            key = f'rerank_top{topk}_{name}'
            rerank = _rerank_results(
                query_ids,
                passage_ids,
                baseline_scores,
                query_embeddings,
                passage_embeddings,
                topk=topk,
                config=config,
                batch_score=args.batch_score,
                device=model.device,
            )
            report['variants'][key] = _compute_metrics(qrels, rerank)
            metrics = report['variants'][key]
            print(key, metrics.get('ndcg_at_5'), metrics.get('recall_at_5'), metrics.get('mrr'))

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
