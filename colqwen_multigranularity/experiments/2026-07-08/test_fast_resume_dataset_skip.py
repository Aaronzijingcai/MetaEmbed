from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch.distributed as dist

from debug_stuck984_batches import build_debug_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-step", type=int, default=950)
    parser.add_argument("--num-steps", type=int, default=3)
    parser.add_argument("--per-device-train-batch-size", type=int, default=10)
    parser.add_argument("--interleaved-batch-size", type=int, default=10)
    parser.add_argument("--num-shards", type=int, default=128)
    parser.add_argument("--dataset-num-proc", type=int, default=1)
    parser.add_argument("--dataset-shuffle-buffer", type=int, default=1024)
    parser.add_argument("--eval-size", type=int, default=100)
    parser.add_argument("--stopping-strategy", default="all_exhausted")
    return parser.parse_args()


def sample_id(example: dict) -> str:
    return f"{example['subset_name']}:{example['data_idx']}"


def take_batches(dataset, *, skip_rows: int, num_steps: int, batch_size: int) -> list[list[str]]:
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(0)
    iterator = iter(dataset)
    for _ in range(skip_rows):
        next(iterator)
    return [
        [sample_id(next(iterator)) for _ in range(batch_size)]
        for _ in range(num_steps)
    ]


def main() -> None:
    args = parse_args()
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    os.environ["DATASET_NUM_PROC"] = str(args.dataset_num_proc)
    os.environ["DATASET_SHUFFLE_BUFFER"] = str(args.dataset_shuffle_buffer)

    skip_rows = args.start_step * args.per_device_train_batch_size
    args.pre_split_skip_rows_per_rank = 0
    sequential_dict, _ = build_debug_dataset(args, rank, world_size)
    args.pre_split_skip_rows_per_rank = skip_rows
    skipped_dict, _ = build_debug_dataset(args, rank, world_size)
    sequential_dataset = sequential_dict["train"].select_columns(["subset_name", "data_idx"])
    skipped_dataset = skipped_dict["train"].select_columns(["subset_name", "data_idx"])

    sequential = take_batches(
        sequential_dataset,
        skip_rows=skip_rows,
        num_steps=args.num_steps,
        batch_size=args.per_device_train_batch_size,
    )
    skipped = take_batches(
        skipped_dataset,
        skip_rows=0,
        num_steps=args.num_steps,
        batch_size=args.per_device_train_batch_size,
    )
    mismatches = [
        {"step": args.start_step + index, "sequential": left, "skipped": right}
        for index, (left, right) in enumerate(zip(sequential, skipped))
        if left != right
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "rank": rank,
        "world_size": world_size,
        "start_step": args.start_step,
        "num_steps": args.num_steps,
        "batch_size": args.per_device_train_batch_size,
        "skip_rows": skip_rows,
        "global_skip_rows": skip_rows * world_size,
        "mismatches": mismatches,
        "sequential": sequential,
        "skipped": skipped,
    }
    (output_dir / f"rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")

    dist.barrier()
    if rank == 0:
        all_results = [
            json.loads((output_dir / f"rank{item_rank}.json").read_text())
            for item_rank in range(world_size)
        ]
        summary = {
            "world_size": world_size,
            "compared_batches": world_size * args.num_steps,
            "compared_samples": world_size * args.num_steps * args.per_device_train_batch_size,
            "mismatch_count": sum(len(item["mismatches"]) for item in all_results),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
