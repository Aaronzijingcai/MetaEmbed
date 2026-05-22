from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import configue
import pytrec_eval
import torch
from peft import PeftModel

from colqwen_multigranularity import eval as base_eval
from colqwen_multigranularity.core import build_colqwen2_5_mrl_model, normalize_granularities
from colqwen_multigranularity.experiments.exp_maxsim.symmetric_maxsim import SymmetricMaxSimConfig, score_multi_vector_symmetric


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default="/MURE-V2/code/MetaEmbed/colqwen_multigranularity/models/colqwen2.5-base")
    parser.add_argument("--processor-name-or-path", type=str, default="/MURE-V2/code/MetaEmbed/colqwen_multigranularity/models/colqwen2.5-base")
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--eval-config", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--granularities", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    parser.add_argument("--batch-query", type=int, default=8)
    parser.add_argument("--batch-passage", type=int, default=8)
    parser.add_argument("--batch-score", type=int, default=32)
    parser.add_argument("--dataset-name", type=str, default=None)
    return parser


def _load_model(args):
    model = build_colqwen2_5_mrl_model(
        args.model_name_or_path,
        granularities=normalize_granularities(args.granularities),
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        eval_mode=False,
    )
    model = PeftModel.from_pretrained(model, Path(args.adapter_path))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def _load_processor(args):
    eval_args = argparse.Namespace(
        model_name_or_path=args.model_name_or_path,
        processor_name_or_path=args.processor_name_or_path,
        adapter_path=None,
        mrl_state_dict_path=None,
        eval_config=args.eval_config,
        dataset_format="beir",
        output_path=args.output_path,
        vis_output_dir=None,
        batch_query=args.batch_query,
        batch_passage=args.batch_passage,
        batch_score=args.batch_score,
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
        attn_implementation=args.attn_implementation,
        use_simple_prompt=True,
        resize_crops_to_page=True,
        crop_resize_mode=None,
        use_v2_retriever=False,
        v2_do_padding=False,
    )
    return base_eval.build_processor(eval_args)


def _encode_queries(model, processor, ds_queries, batch_size):
    query_ids = [str(x) for x in ds_queries["query-id"]]
    texts = list(ds_queries["query"])
    images = list(ds_queries["image"]) if "image" in ds_queries.column_names else [None] * len(ds_queries)
    outputs = []
    for start in range(0, len(texts), batch_size):
        batch = processor.process_mm_queries(texts[start:start+batch_size], images[start:start+batch_size], is_train=False).to(model.device)
        with torch.inference_mode():
            emb = model(**batch).to("cpu")
        outputs.extend(list(torch.unbind(emb)))
    return query_ids, outputs


def _encode_passages(model, processor, ds_corpus, batch_size):
    passage_ids = [str(x) for x in ds_corpus["corpus-id"]]
    texts = list(ds_corpus["text"]) if "text" in ds_corpus.column_names else [None] * len(ds_corpus)
    images = list(ds_corpus["image"]) if "image" in ds_corpus.column_names else [None] * len(ds_corpus)
    outputs = []
    for start in range(0, len(images), batch_size):
        batch = processor.process_mm_documents(texts[start:start+batch_size], images[start:start+batch_size], is_train=False).to(model.device)
        with torch.inference_mode():
            emb = model(**batch).to("cpu")
        outputs.extend(list(torch.unbind(emb)))
    return passage_ids, outputs


def _build_qrels(ds_qrels):
    qrels = defaultdict(dict)
    for row in ds_qrels:
        qrels[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])
    return qrels


def _results_from_scores(query_ids, passage_ids, scores):
    results = {}
    for i, qid in enumerate(query_ids):
        results[qid] = {pid: float(scores[i, j]) for j, pid in enumerate(passage_ids)}
    return results


def _compute_metrics(qrels, results):
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {
        "ndcg_cut.1,3,5,10,20,50,100",
        "map_cut.1,3,5,10,20,50,100",
        "recall.1,3,5,10,20,50,100",
        "P.1,3,5,10,20,50,100",
        "recip_rank",
    })
    per_query = evaluator.evaluate(results)
    metrics = defaultdict(float)
    n = max(len(per_query), 1)
    for row in per_query.values():
        for k, v in row.items():
            metrics[k] += v
    return {k: v / n for k, v in metrics.items()}


def _normalize_metric_names(metrics):
    out = {}
    for k, v in metrics.items():
        if k.startswith("ndcg_cut_"):
            out[k.replace("ndcg_cut_", "ndcg_at_")] = round(v, 5)
        elif k.startswith("map_cut_"):
            out[k.replace("map_cut_", "map_at_")] = round(v, 5)
        elif k.startswith("recall_"):
            out[k.replace("recall_", "recall_at_")] = round(v, 5)
        elif k.startswith("P_"):
            out[k.replace("P_", "precision_at_")] = round(v, 5)
        elif k == "recip_rank":
            out["mrr"] = round(v, 5)
        else:
            out[k] = round(v, 5)
    return out


def main():
    args = build_parser().parse_args()
    model = _load_model(args)
    processor = _load_processor(args)
    loader = configue.load(Path(args.eval_config))

    if args.dataset_name is None:
        dataset_name = next(iter(loader.keys()))
    else:
        dataset_name = args.dataset_name
    dataset = loader[dataset_name]()

    query_ids, query_embeddings = _encode_queries(model, processor, dataset["queries"], args.batch_query)
    passage_ids, passage_embeddings = _encode_passages(model, processor, dataset["corpus"], args.batch_passage)
    qrels = _build_qrels(dataset["qrels"])

    variants = [
        ("query", SymmetricMaxSimConfig(score_mode="query", query_weight=1.0, doc_weight=0.0, doc_chunk_size=args.batch_score)),
        ("bimax_0.7_0.3", SymmetricMaxSimConfig(score_mode="bimax", query_weight=0.7, doc_weight=0.3, doc_chunk_size=args.batch_score)),
        ("bimax_0.5_0.5", SymmetricMaxSimConfig(score_mode="bimax", query_weight=0.5, doc_weight=0.5, doc_chunk_size=args.batch_score)),
        ("bimax_0.3_0.7", SymmetricMaxSimConfig(score_mode="bimax", query_weight=0.3, doc_weight=0.7, doc_chunk_size=args.batch_score)),
        ("doc", SymmetricMaxSimConfig(score_mode="doc", query_weight=0.0, doc_weight=1.0, doc_chunk_size=args.batch_score)),
    ]

    report = {"dataset": dataset_name, "variants": {}}
    for name, config in variants:
        scores = score_multi_vector_symmetric(query_embeddings, passage_embeddings, batch_size=args.batch_score, device=model.device, config=config)
        results = _results_from_scores(query_ids, passage_ids, scores)
        metrics = _normalize_metric_names(_compute_metrics(qrels, results))
        report["variants"][name] = metrics
        print(name, metrics.get("ndcg_at_5"), metrics.get("recall_at_5"))

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
