from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml
from datasets import load_dataset

from colqwen_multigranularity import train as base_train


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


def chunk_pairs(texts: list[Any], images: list[Any], batch_size: int):
    for i in range(0, len(texts), batch_size):
        yield texts[i : i + batch_size], images[i : i + batch_size]


def lengths_from_batch(batch) -> list[int]:
    return batch['attention_mask'].sum(dim=1).tolist()


def write_md(results: dict[str, Any], output_md: Path) -> None:
    lines = ['# Train Sequence Length Statistics', '']
    lines.append('| Subset | Query N | Query Mean | Query P50 | Query P90 | Query Max | Positive N | Positive Mean | Positive P50 | Positive P90 | Positive Max | Ratio P50 | Ratio Mean |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for subset, stats in results.items():
        q = stats['query_summary']
        d = stats['positive_summary']
        ratio_p50 = round(d['p50'] / max(q['p50'], 1), 4) if q.get('count', 0) else None
        ratio_mean = round(d['mean'] / max(q['mean'], 1e-9), 4) if q.get('count', 0) else None
        lines.append(
            f"| {subset} | {q['count']} | {q['mean']} | {q['p50']} | {q['p90']} | {q['max']} | {d['count']} | {d['mean']} | {d['p50']} | {d['p90']} | {d['max']} | {ratio_p50} | {ratio_mean} |"
        )
    output_md.write_text('\n'.join(lines) + '\n')


def load_existing(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_progress(results: dict[str, Any], output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    write_md(results, output_md)


def build_processor_args(batch_query: int, batch_doc: int) -> argparse.Namespace:
    return argparse.Namespace(
        model_name_or_path='/MURE-V2/code/MetaEmbed/colqwen_multigranularity/models/colqwen2.5-base',
        processor_name_or_path='/MURE-V2/code/MetaEmbed/colqwen_multigranularity/models/colqwen2.5-base',
        output_dir='',
        subset_config='',
        eval_vidore_v1_config='',
        eval_vidore_v2_config='',
        eval_mmeb_config='',
        granularities=[1, 2, 4],
        max_num_visual_tokens=None,
        granularity_loss_weights=None,
        max_steps=0,
        learning_rate=1e-4,
        lr_scheduler_type='linear',
        warmup_ratio=0.0,
        warmup_steps=0,
        resume_from_checkpoint=None,
        per_device_train_batch_size=batch_query,
        per_device_eval_batch_size=batch_doc,
        vidore_eval_batch_size=batch_doc,
        gradient_accumulation_steps=1,
        dataloader_num_workers=0,
        logging_steps=10,
        save_steps=100,
        num_negative=1,
        interleaved_batch_size=0,
        num_shards=1,
        stopping_strategy='all_exhausted',
        truncation_len=16384,
        processor_max_length=None,
        query_augmentation_repeats=10,
        document_augmentation_repeats=0,
        wandb_project='MetaEmbed',
        attn_implementation='flash_attention_2',
        temperature=0.03,
        normalize_scores=True,
        doc_chunk_size=256,
        use_liger_kernel=False,
        use_v2_trainer=True,
        use_v2_retriever=True,
        do_gather=False,
        do_padding=True,
        run_eval=False,
        use_peft=False,
        use_simple_prompt=True,
        resize_crops_to_page=True,
        crop_resize_mode=None,
        compact_query_tokens=True,
        drop_query_text_if_image=False,
        drop_doc_text_if_image=False,
        ddp_find_unused_parameters=False,
    )


def collect_subset_lengths(subset: str, num_samples: int, processor, batch_query: int, batch_doc: int, base_path: str) -> dict[str, Any]:
    ds = load_dataset(base_path, subset, split='original', streaming=True)
    query_texts, query_images = [], []
    pos_texts, pos_images = [], []
    qlens, dlens = [], []

    count = 0
    for example in ds:
        if num_samples is not None and count >= num_samples:
            break
        query_text = example['qry'].replace('<|image_1|>\n', '').replace('<|image_1|>', '') if isinstance(example.get('qry'), str) else None
        query_img = example.get('qry_image')
        pos_text = example['pos_text'].replace('<|image_1|>\n', '').replace('<|image_1|>', '') if isinstance(example.get('pos_text'), str) else None
        pos_img = example.get('pos_image')
        query_texts.append(query_text)
        query_images.append(query_img)
        pos_texts.append(pos_text)
        pos_images.append(pos_img)
        count += 1

        if len(query_texts) >= batch_query:
            batch = processor.process_mm_queries(query_texts, query_images, is_train=False)
            qlens.extend(lengths_from_batch(batch))
            query_texts, query_images = [], []
        if len(pos_texts) >= batch_doc:
            batch = processor.process_mm_documents(pos_texts, pos_images, is_train=False)
            dlens.extend(lengths_from_batch(batch))
            pos_texts, pos_images = [], []

    if query_texts:
        batch = processor.process_mm_queries(query_texts, query_images, is_train=False)
        qlens.extend(lengths_from_batch(batch))
    if pos_texts:
        batch = processor.process_mm_documents(pos_texts, pos_images, is_train=False)
        dlens.extend(lengths_from_batch(batch))

    return {
        'query_lengths': qlens,
        'positive_lengths': dlens,
        'query_summary': summarize(qlens),
        'positive_summary': summarize(dlens),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset-config', default='/MURE-V2/code/MetaEmbed/colqwen_multigranularity/configs/train/moca_data_ratios_v3_full.yaml')
    ap.add_argument('--output-json', default='/MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/exp_maxsim/results/train_lengths.json')
    ap.add_argument('--output-md', default='/MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/exp_maxsim/results/train_lengths.md')
    ap.add_argument('--batch-query', type=int, default=64)
    ap.add_argument('--batch-doc', type=int, default=16)
    ap.add_argument('--max-samples-per-subset', type=int, default=5000)
    ap.add_argument('--dataset-names', nargs='*', default=None)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    subset2meta = yaml.safe_load(Path(args.subset_config).read_text())
    subset_names = list(subset2meta.keys())
    if args.dataset_names:
        requested = set(args.dataset_names)
        subset_names = [name for name in subset_names if name in requested]

    results = load_existing(Path(args.output_json)) if args.resume else {}
    processor = base_train.build_processor(build_processor_args(args.batch_query, args.batch_doc))
    base_path = '/MURE-V2/code/MetaEmbed/data_dir/MoCa_train_with_image'

    for subset in subset_names:
        if subset in results:
            print(f'[collect_train_lengths] skip existing subset={subset}', flush=True)
            continue
        limit = subset2meta[subset].get('num_samples', args.max_samples_per_subset)
        limit = min(limit, args.max_samples_per_subset) if args.max_samples_per_subset is not None else limit
        print(f'[collect_train_lengths] start subset={subset} limit={limit}', flush=True)
        result = collect_subset_lengths(subset, limit, processor, args.batch_query, args.batch_doc, base_path)
        results[subset] = result
        save_progress(results, Path(args.output_json), Path(args.output_md))
        print(f"[collect_train_lengths] done subset={subset} query={result['query_summary']} positive={result['positive_summary']}", flush=True)

    print(f"[collect_train_lengths] wrote json={args.output_json} md={args.output_md}", flush=True)


if __name__ == '__main__':
    main()
