#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
from datasets import DatasetDict, concatenate_datasets, load_dataset
from datasets.distributed import split_dataset_by_node
from peft import get_peft_model

from colpali_engine.collators.mm_collator import MultimodalRetrieverCollator
from colpali_engine.utils.dist_utils import rank0_print
from colpali_engine.utils.dist_utils import all_gather_with_padding_select_dim, gather_with_grad_torch, pad_to_max_len_right
from colpali_engine.utils.hf_dataset_utils import interleave_datasets
from colqwen_multigranularity.core import MRLColQwen2_5Processor
from colqwen_multigranularity.experiments.exp_stagecompress.folder_homo.config import FolderHomoConfig
from colqwen_multigranularity.experiments.exp_stagecompress.folder_homo.loss import FolderHomoMRLInBatchNegativeLoss
from colqwen_multigranularity.experiments.exp_stagecompress.folder_homo.modeling_folder_homo import build_folder_homo_model
from colqwen_multigranularity.experiments.exp_stagecompress.folder_homo.train_folder_homo import build_peft_config


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
    parser.add_argument("--dist-backend", default="gloo", choices=["gloo", "nccl"])
    parser.add_argument("--model-replay", action="store_true", help="Run FolderHomo model forward/gather/loss on selected batches.")
    parser.add_argument("--model-name-or-path", default="models/colqwen2.5-base")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--use-peft", action="store_true", default=False)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False)
    parser.add_argument("--ddp-wrap", action="store_true", help="Wrap the replay model with DistributedDataParallel.")
    parser.add_argument("--ddp-find-unused-parameters", action="store_true", default=False)
    parser.add_argument("--backward", action="store_true", help="Also run loss.backward() for each replayed batch.")
    parser.add_argument("--do-gather", action="store_true", default=False)
    parser.add_argument("--do-padding", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--normalize-scores", action="store_true", default=True)
    parser.add_argument("--doc-chunk-size", type=int, default=512)
    parser.add_argument("--query-chunk-size", type=int, default=512)
    parser.add_argument("--folder-homo-budgets", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--folder-homo-compress-stages", default="all")
    parser.add_argument("--interaction-loss-mode", default="q2d_query_topk")
    parser.add_argument("--interaction-bi-lambda", type=float, default=0.5)
    parser.add_argument("--interaction-query-topk", type=int, default=48)
    return parser.parse_args()


def init_dist(backend: str = "gloo") -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    if "RANK" not in os.environ:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29991")
    if backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group(backend=backend, init_method="env://")
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


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def _model_inputs(inputs: dict[str, Any], prefix: str) -> dict[str, Any]:
    blocked_key = f"{prefix}_has_images"
    prefix_with_sep = f"{prefix}_"
    prefix_len = len(prefix_with_sep)
    return {
        key[prefix_len:]: value
        for key, value in inputs.items()
        if key.startswith(prefix_with_sep) and key != blocked_key
    }


def _extract_has_images(
    *,
    input_ids: torch.Tensor,
    has_images: torch.Tensor | None = None,
    pixel_values: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
) -> torch.Tensor:
    if has_images is not None:
        return has_images.to(device=input_ids.device, dtype=torch.bool)
    has_visuals = (
        pixel_values is not None
        and image_grid_thw is not None
        and getattr(pixel_values, "numel", lambda: 0)() > 0
        and getattr(image_grid_thw, "numel", lambda: 0)() > 0
    )
    return torch.full((input_ids.shape[0],), bool(has_visuals), dtype=torch.bool, device=input_ids.device)


def _gather_bool_rows(flags: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return flags
    gathered = [torch.zeros_like(flags) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, flags)
    return torch.cat(gathered, dim=0)


def _gather_2d_tensor_rows(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return tensor
    gathered, _ = all_gather_with_padding_select_dim(tensor, dist.get_world_size(), pad_dim=1)
    return gathered


def _sync_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def build_folder_homo_config(args: argparse.Namespace) -> FolderHomoConfig:
    return FolderHomoConfig(
        enabled=True,
        budgets=tuple(int(value) for value in args.folder_homo_budgets),
        compress_stages=args.folder_homo_compress_stages,
        interaction_loss_mode=str(args.interaction_loss_mode),
        interaction_bi_lambda=float(args.interaction_bi_lambda),
        interaction_query_topk=int(args.interaction_query_topk),
    )


def build_replay_objects(args: argparse.Namespace, processor: MRLColQwen2_5Processor, device: torch.device):
    folder_homo_config = build_folder_homo_config(args)
    model = build_folder_homo_model(
        args.model_name_or_path,
        granularities=(1, 2, 4),
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        use_liger_kernel=False,
        compact_query_tokens=True,
        folder_homo_config=folder_homo_config,
        adapter_path=None,
    )
    if args.use_peft:
        model = get_peft_model(model, build_peft_config())
        for name, param in model.named_parameters():
            if "folder_homo" in name:
                param.requires_grad = True
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "config"):
            model.config.use_cache = False
    model.to(device)
    model.train()
    if args.ddp_wrap:
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            raise RuntimeError("--ddp-wrap requires a distributed multi-process run.")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        ddp_kwargs: dict[str, Any] = {"find_unused_parameters": bool(args.ddp_find_unused_parameters)}
        if device.type == "cuda":
            ddp_kwargs.update({"device_ids": [local_rank], "output_device": local_rank})
        model = DDP(model, **ddp_kwargs)

    loss_func = FolderHomoMRLInBatchNegativeLoss(
        image_token_id=processor.image_token_id,
        folder_homo_config=folder_homo_config,
        temperature=args.temperature,
        granularities=(1, 2, 4),
        level_weights=None,
        normalize_scores=args.normalize_scores,
        doc_chunk_size=args.doc_chunk_size,
        query_chunk_size=args.query_chunk_size,
    )
    return model, loss_func


def replay_model_batch(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    loss_func: FolderHomoMRLInBatchNegativeLoss,
    collated: dict[str, Any],
    rank: int,
    step: int,
    device: torch.device,
) -> dict[str, Any]:
    inputs = _to_device(collated, device)
    report: dict[str, Any] = {
        "model_replay": True,
        "device": str(device),
        "interaction_loss_mode": args.interaction_loss_mode,
        "do_gather": bool(args.do_gather),
        "do_padding": bool(args.do_padding),
        "backward": bool(args.backward),
        "ddp_wrap": bool(args.ddp_wrap),
        "stages": [],
    }

    def stage(name: str, start: float, **extra: Any) -> float:
        now = _sync_time()
        row = {"stage": name, "seconds": now - start}
        row.update(extra)
        report["stages"].append(row)
        print(f"[debug-model] rank={rank} step={step} stage={name} seconds={row['seconds']:.4f} extra={extra}", flush=True)
        return now

    autocast_enabled = device.type == "cuda"
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled)
    with autocast_ctx:
        started = _sync_time()
        query_outputs = model(**_model_inputs(inputs, "query"))
        last = stage("query_forward", started, shape=list(query_outputs.shape))

        doc_outputs = model(**_model_inputs(inputs, "doc"))
        last = stage("doc_forward", last, shape=list(doc_outputs.shape))

    query_has_images = _extract_has_images(
        input_ids=inputs["query_input_ids"],
        has_images=inputs.get("query_has_images"),
        pixel_values=inputs.get("query_pixel_values"),
        image_grid_thw=inputs.get("query_image_grid_thw"),
    )
    doc_has_images_local = _extract_has_images(
        input_ids=inputs["doc_input_ids"],
        has_images=inputs.get("doc_has_images"),
        pixel_values=inputs.get("doc_pixel_values"),
        image_grid_thw=inputs.get("doc_image_grid_thw"),
    )

    additional_loss_kwargs: dict[str, Any] = {}
    query_embeddings = query_outputs
    doc_embeddings = doc_outputs
    doc_has_images = doc_has_images_local
    if args.do_gather:
        if args.do_padding:
            doc_outputs, _ = pad_to_max_len_right(doc_outputs, dist.get_world_size())
            last = stage("pad_doc", last, shape=list(doc_outputs.shape))
        doc_embeddings = gather_with_grad_torch(doc_outputs)
        last = stage("gather_doc_embeddings", last, shape=list(doc_embeddings.shape))
        doc_has_images = _gather_bool_rows(doc_has_images_local)

    if getattr(loss_func, "needs_has_images", False):
        additional_loss_kwargs["query_has_images"] = query_has_images
        additional_loss_kwargs["doc_has_images"] = doc_has_images
    if getattr(loss_func, "needs_input_ids", False):
        additional_loss_kwargs.update(
            {
                "query_input_ids": inputs.get("query_input_ids"),
                "query_attention_mask": inputs.get("query_attention_mask"),
                "doc_input_ids": _gather_2d_tensor_rows(inputs.get("doc_input_ids")) if args.do_gather else inputs.get("doc_input_ids"),
                "doc_attention_mask": _gather_2d_tensor_rows(inputs.get("doc_attention_mask")) if args.do_gather else inputs.get("doc_attention_mask"),
            }
        )
        last = stage(
            "gather_doc_inputs" if args.do_gather else "prepare_doc_inputs",
            last,
            doc_input_ids=list(additional_loss_kwargs["doc_input_ids"].shape),
        )

    neg_doc_embeddings = None
    if "neg_doc_input_ids" in inputs:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            neg_doc_embeddings = model(**_model_inputs(inputs, "neg_doc"))
        last = stage("neg_doc_forward", last, shape=list(neg_doc_embeddings.shape))
        if getattr(loss_func, "needs_has_images", False):
            additional_loss_kwargs["neg_doc_has_images"] = _extract_has_images(
                input_ids=inputs["neg_doc_input_ids"],
                has_images=inputs.get("neg_doc_has_images"),
                pixel_values=inputs.get("neg_doc_pixel_values"),
                image_grid_thw=inputs.get("neg_doc_image_grid_thw"),
            )
        if getattr(loss_func, "needs_input_ids", False):
            additional_loss_kwargs.update(
                {
                    "neg_doc_input_ids": inputs.get("neg_doc_input_ids"),
                    "neg_doc_attention_mask": inputs.get("neg_doc_attention_mask"),
                }
            )

    loss_started = last
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        if neg_doc_embeddings is not None:
            loss = loss_func(query_embeddings, doc_embeddings, neg_doc_embeddings, **additional_loss_kwargs)
        else:
            loss = loss_func(query_embeddings, doc_embeddings, **additional_loss_kwargs)
    if isinstance(loss, tuple):
        loss, loss_stats = loss
        report["loss_stats"] = {
            key: (float(value.detach().float().cpu().item()) if isinstance(value, torch.Tensor) and value.numel() == 1 else str(value))
            for key, value in loss_stats.items()
        }
    else:
        loss_stats = None
    last = stage("loss", loss_started, loss=float(loss.detach().float().cpu().item()))

    if args.backward:
        loss.backward()
        last = stage("backward", last)
        model.zero_grad(set_to_none=True)
        stage("zero_grad", last)

    report["loss"] = float(loss.detach().float().cpu().item())
    return report


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
    pre_split_skip_rows_per_rank = int(
        getattr(args, "pre_split_skip_rows_per_rank", 0)
    )
    if pre_split_skip_rows_per_rank > 0:
        distributed = train_ds._distributed
        train_ds._distributed = None
        train_ds = train_ds.skip(pre_split_skip_rows_per_rank * world_size)
        train_ds._distributed = distributed
    return DatasetDict({"train": train_ds, "test": concatenate_datasets(eval_ds_list)}), subset2meta


def main() -> None:
    args = parse_args()
    os.environ["DATASET_NUM_PROC"] = str(args.dataset_num_proc)
    os.environ["DATASET_SHUFFLE_BUFFER"] = str(args.dataset_shuffle_buffer)
    if args.model_replay:
        args.collate = True
        args.dist_backend = "nccl"
        args.do_gather = True
        args.do_padding = True
        args.use_peft = True
        args.gradient_checkpointing = True
    rank, world_size = init_dist(args.dist_backend)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"rank{rank}_steps{args.start_step}-{args.end_step}.jsonl"
    summary_file = output_dir / f"rank{rank}_summary.json"

    t0 = time.time()
    ds, subset2meta = build_debug_dataset(args, rank, world_size)
    build_seconds = time.time() - t0

    collator = None
    image_token_id = None
    processor = None
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

    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}" if args.model_replay else "cpu")
    model = None
    loss_func = None
    if args.model_replay:
        if processor is None:
            raise RuntimeError("--model-replay requires a processor/collator.")
        model, loss_func = build_replay_objects(args, processor, device)

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
                if args.model_replay:
                    if model is None or loss_func is None:
                        raise RuntimeError("model replay objects were not initialized")
                    t_model = time.time()
                    record["model_replay"] = replay_model_batch(
                        args=args,
                        model=model,
                        loss_func=loss_func,
                        collated=collated,
                        rank=rank,
                        step=step,
                        device=device,
                    )
                    record["model_replay_seconds"] = time.time() - t_model
                    if torch.cuda.is_available():
                        record["cuda_max_memory_allocated_gb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                        record["cuda_max_memory_reserved_gb"] = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
            except Exception as exc:
                record["collate_error"] = repr(exc)
                record["traceback"] = traceback.format_exc()
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
