#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from datasets import DatasetDict, concatenate_datasets, load_dataset
from datasets.distributed import split_dataset_by_node

from colpali_engine.collators.mm_collator import MultimodalRetrieverCollator
from colpali_engine.utils.dist_utils import rank0_print
from colpali_engine.utils.hf_dataset_utils import interleave_datasets
from colqwen_multigranularity.core import MRLColQwen2_5Processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locate deterministic stuck batches around step 984.")
    parser.add_argument("--subset-config", default="configs/train/moca_data_ratios_v3_full.yaml")
    parser.add_argument("--processor-name-or-path", default="models/colqwen2.5-base")
    parser.add_argument("--output-dir", default="experiments/2026-07-08/runs/debug_stuck984")
    parser.add_argument("--start-step", type=int, default=970)
    parser.add_argument("--end-step", type=int, default=990)
    parser.add_argument("--per-device-train-batch-size", type=int, default=10)
    parser.add_argument("--interleaved-batch-size", type=int, default=10)
    parser.add_argument("--num-shards", type=int, default=128)
    parser.add_argument("--eval-size", type=int, default=100)
    parser.add_argument("--stopping-strategy", default="all_exhausted", choices=["all_exhausted", "first_exhausted"])
    parser.add_argument("--collate", action="store_true", help="Run the real multimodal collator and record timing/shapes.")
    parser.add_argument("--max-num-visual-tokens", type=int, default=1024)
    parser.add_argument("--truncation-len", type=int, default=16384)
    parser.add_argument("--query-augmentation-repeats", type=int, default=10)
    parser.add_argument("--document-augmentation-repeats", type=int, default=0)
    parser.add_argument("--dataset-num-proc", type=int, default=1)
    parser.add_argument("--dataset-shuffle-buffer", type=int, default=1024)
    return parser.parse_args()


def init_dist() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    if "RANK" not in os.environ:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29991")
    dist.init_process_group(backend="gloo", init_method="env://")
    return dist.get_rank(), dist.get_world_size()


def load_subset_config(path: str) -> dict[str, dict[str, Any]]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _as_len(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        return len(value)
    return 1


def _image_summary(value: Any) -> dict[str, Any]:
    if value is None:
        return {"has": False}
    if isinstance(value, list):
        return {"has": len(value) > 0, "type": "list", "count": len(value)}
    size = getattr(value, "size", None)
    mode = getattr(value, "mode", None)
    return {
        "has": True,
        "type": type(value).__name__,
        "size": list(size) if size is not None else None,
        "mode": mode,
    }


def _summarize_example(example: dict[str, Any]) -> dict[str, Any]:
    neg_text = example.get("neg_text")
    neg_count = len(neg_text) if isinstance(neg_text, list) else (1 if neg_text is not None else 0)
    neg_img_count = 0
    for key, value in example.items():
        if key.startswith("neg_image_") and value is not None:
            neg_img_count += 1
    subset_id = example.get("subset_idx", example.get("__subset_id"))
    subset_name = example.get("subset_name", example.get("__subset"))
    row_idx = example.get("data_idx", example.get("__row_idx"))
    return {
        "sample_id": f"{subset_name}:{row_idx}",
        "subset_id": subset_id,
        "subset": subset_name,
        "row_idx": row_idx,
        "qry_len": _as_len(example.get("qry")),
        "pos_text_len": _as_len(example.get("pos_text")),
        "neg_count": neg_count,
        "neg_img_count": neg_img_count,
        "qry_image": _image_summary(example.get("qry_image")),
        "pos_image": _image_summary(example.get("pos_image")),
        "neg_text_lens": [_as_len(x) for x in neg_text[:3]] if isinstance(neg_text, list) else [],
    }


def _tensor_shapes(batch: dict[str, Any]) -> dict[str, Any]:
    shapes = {}
    for key, value in batch.items():
        shape = getattr(value, "shape", None)
        if shape is not None:
            shapes[key] = list(shape)
        elif isinstance(value, list):
            shapes[key] = {"list_len": len(value)}
    return shapes


def _row_lengths(mask: Any) -> list[int]:
    if not isinstance(mask, torch.Tensor) or mask.ndim < 2:
        return []
    return [int(x) for x in mask.long().sum(dim=1).detach().cpu().tolist()]


def _image_token_counts(input_ids: Any, image_token_id: int | None) -> list[int]:
    if image_token_id is None or not isinstance(input_ids, torch.Tensor) or input_ids.ndim < 2:
        return []
    return [int(x) for x in input_ids.eq(int(image_token_id)).long().sum(dim=1).detach().cpu().tolist()]


def _grid_visual_tokens(image_grid_thw: Any) -> list[int]:
    if not isinstance(image_grid_thw, torch.Tensor) or image_grid_thw.numel() == 0:
        return []
    grid = image_grid_thw.detach().cpu().long()
    if grid.ndim == 1:
        grid = grid.unsqueeze(0)
    # Qwen-style image_grid_thw records one row per image. prod(t, h, w)
    # is the closest processor-side proxy for visual token count.
    return [int(row.prod().item()) for row in grid]


def _attach_processed_counts(
    examples: list[dict[str, Any]],
    collated: dict[str, Any],
    image_token_id: int | None,
) -> None:
    fields = [
        ("query", "query_input_ids", "query_attention_mask", "query_image_grid_thw"),
        ("doc", "doc_input_ids", "doc_attention_mask", "doc_image_grid_thw"),
        ("neg_doc", "neg_doc_input_ids", "neg_doc_attention_mask", "neg_doc_image_grid_thw"),
    ]
    for prefix, ids_key, mask_key, grid_key in fields:
        ids = collated.get(ids_key)
        mask = collated.get(mask_key)
        token_counts = _image_token_counts(ids, image_token_id)
        row_lens = _row_lengths(mask)
        for idx, example in enumerate(examples):
            if idx < len(token_counts):
                example[f"{prefix}_image_token_count"] = token_counts[idx]
            if idx < len(row_lens):
                example[f"{prefix}_input_len"] = row_lens[idx]
        grid_counts = _grid_visual_tokens(collated.get(grid_key))
        if grid_counts:
            # Grid rows are emitted only for examples that contain images. Attach
            # them to image-bearing rows in order, which matches processor order
            # for the one-image-per-field MoCA/MMEB training format.
            pos = 0
            for example in examples:
                has_image = int(example.get(f"{prefix}_image_token_count", 0)) > 0
                if has_image and pos < len(grid_counts):
                    example[f"{prefix}_grid_visual_tokens"] = grid_counts[pos]
                    pos += 1


def build_debug_dataset(args: argparse.Namespace, rank: int, world_size: int):
    os.environ.setdefault("DATA_DIR", str(Path("data_dir").resolve()) + "/")
    base_path = os.environ.get("DATA_DIR", "./data_dir/")
    ds_path = "MoCa_train_with_image"
    subset2meta = load_subset_config(args.subset_config)
    ds_list = []
    eval_ds_list = []
    weight_list = []

    for subset_id, (subset, meta_info) in enumerate(subset2meta.items()):
        rank0_print(f"[debug-data] load {subset_id + 1}/{len(subset2meta)} {subset} {meta_info}")
        split = f"original[:{meta_info['num_samples']}]" if "num_samples" in meta_info else "original"
        if world_size > 1 and rank != 0:
            dist.barrier()
        ds = load_dataset(base_path + ds_path, subset, num_proc=args.dataset_num_proc, split=split)
        if world_size > 1 and rank == 0:
            dist.barrier()

        ds_eval = ds.select(range(len(ds) - args.eval_size, len(ds)))
        ds = ds.select(range(0, len(ds) - args.eval_size))
        num_rows = ds.num_rows
        ds = ds.add_column("subset_idx", [subset_id] * num_rows)
        ds = ds.add_column("subset_name", [subset] * num_rows)
        ds = ds.add_column("data_idx", list(range(num_rows)))
        ds = ds.to_iterable_dataset(num_shards=args.num_shards)
        ds = ds.shuffle(buffer_size=args.dataset_shuffle_buffer, seed=42 + subset_id)
        setattr(ds, "num_rows", num_rows)
        ds_list.append(ds)
        eval_ds_list.append(ds_eval)
        weight_list.append(float(meta_info["weight"]))

    probs = [weight / sum(weight_list) for weight in weight_list]
    train_ds = interleave_datasets(
        ds_list,
        probs,
        batch_size=args.interleaved_batch_size,
        seed=42 + rank,
        stopping_strategy=args.stopping_strategy,
    )
    train_ds = split_dataset_by_node(train_ds, rank=rank, world_size=world_size)
    train_ds = train_ds.shuffle(seed=42)
    return DatasetDict({"train": train_ds, "test": concatenate_datasets(eval_ds_list)}), subset2meta


def main() -> None:
    args = parse_args()
    os.environ["DATASET_NUM_PROC"] = str(args.dataset_num_proc)
    os.environ["DATASET_SHUFFLE_BUFFER"] = str(args.dataset_shuffle_buffer)
    rank, world_size = init_dist()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"rank{rank}_steps{args.start_step}-{args.end_step}.jsonl"
    summary_file = output_dir / f"rank{rank}_summary.json"

    t0 = time.time()
    ds, subset2meta = build_debug_dataset(args, rank, world_size)
    build_seconds = time.time() - t0

    collator = None
    image_token_id = None
    if args.collate:
        processor = MRLColQwen2_5Processor.from_pretrained(
            args.processor_name_or_path,
            max_num_visual_tokens=args.max_num_visual_tokens,
            truncation_len=args.truncation_len,
            granularities=[1, 2, 4],
            resize_crops_to_page=True,
            query_augmentation_repeats=args.query_augmentation_repeats,
            document_augmentation_repeats=args.document_augmentation_repeats,
            use_simple_prompt=True,
        )
        image_token_id = int(getattr(processor, "image_token_id", -1))
        collator = MultimodalRetrieverCollator(
            processor=processor,
            num_negative=1,
            corpus_format="moca",
        )

    iterator = iter(ds["train"])
    records = []
    batch_size = args.per_device_train_batch_size
    for step in range(args.end_step + 1):
        batch = [next(iterator) for _ in range(batch_size)]
        if step < args.start_step:
            continue

        record: dict[str, Any] = {
            "rank": rank,
            "world_size": world_size,
            "step": step,
            "batch_size": len(batch),
            "examples": [_summarize_example(x) for x in batch],
        }
        subset_counts: dict[str, int] = {}
        for item in record["examples"]:
            subset = str(item["subset"])
            subset_counts[subset] = subset_counts.get(subset, 0) + 1
        record["subset_counts"] = subset_counts

        if collator is not None:
            t1 = time.time()
            try:
                collated = collator(batch)
                record["collate_seconds"] = time.time() - t1
                record["collate_shapes"] = _tensor_shapes(collated)
                _attach_processed_counts(record["examples"], collated, image_token_id)
            except Exception as exc:
                record["collate_error"] = repr(exc)
                record["collate_seconds"] = time.time() - t1

        records.append(record)
        with output_file.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"[debug-stuck984] rank={rank} step={step} subsets={subset_counts} "
            f"collate_s={record.get('collate_seconds')}",
            flush=True,
        )

    summary = {
        "rank": rank,
        "world_size": world_size,
        "build_seconds": build_seconds,
        "output_file": str(output_file),
        "num_records": len(records),
        "subset_order": list(subset2meta.keys()),
        "args": vars(args),
    }
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
