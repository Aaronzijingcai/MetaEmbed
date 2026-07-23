# V2 version contrastive trainer uses gather for all query, docs and optionally neg_dows
# additional support over v1 includes:
# 1) support for multiple negatives (*DONE)
# 2) remove padding for ColBERT dim-1
# 3) disable random sampler to enable interleaved batching
# 4) only gather for positive documents
import contextlib
import inspect
import json
import os
import time

import types

import torch
import torch.distributed as dist
import transformers

from colpali_engine.trainer.hotpatch_hf_trainer import (
    _inner_training_loop,
    _inner_training_loop_4_55_0,
    set_initial_training_values,
    set_initial_training_values_4_55_0,
)
from colpali_engine.utils.dist_utils import (
    all_gather_with_padding_select_dim,
    gather_with_grad_torch,
    pad_to_max_len_right,
    rank0_print,
    sync_gradients_after_backward,
    wait_for_all_ranks_cpu,
)
from colpali_engine.utils.trainer_utils import hack_callbacks_and_replace

from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback

from transformers.trainer_utils import seed_worker


def _audit_step_enabled(state) -> bool:
    if os.environ.get("MURE_DEEP_AUDIT", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
    }:
        return False
    spec = os.environ.get("MURE_DEEP_AUDIT_STEPS", "").strip()
    if not spec:
        return True
    step = int(getattr(state, "global_step", -1))
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            if int(start) <= step <= int(end):
                return True
        elif step == int(part):
            return True
    return False


def _trainable_group(name: str) -> str:
    if ".language_model." in name and ".lora_" in name:
        return "language_lora"
    if ".visual." in name and ".lora_" in name:
        return "visual_lora"
    if "custom_text_proj" in name:
        return "custom_text_proj"
    if "folder_homo" in name:
        return "folder_homo"
    return "other"


def _write_audit_event(state, stage: str, payload: dict) -> None:
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    event = {
        "rank": rank,
        "step": int(getattr(state, "global_step", -1)),
        "stage": stage,
        "time": time.time(),
        **payload,
    }
    line = json.dumps(event, ensure_ascii=True, sort_keys=True)
    print(f"[mure-deep-audit] {line}", flush=True)
    debug_dir = os.environ.get("CONTRASTIVE_DEBUG_DIR", "").strip()
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        path = os.path.join(debug_dir, f"rank{rank}.audit.jsonl")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class MureDeepAuditOptimizerCallback(TrainerCallback):
    """Verify that every intended trainable group changes at optimizer.step()."""

    def __init__(self):
        self._before = None

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if not _audit_step_enabled(state):
            return
        self._before = {
            name: parameter.detach().to(device="cpu", copy=True)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        _write_audit_event(
            state,
            "optimizer_snapshot_before",
            {"trainable_tensors": len(self._before)},
        )

    def on_optimizer_step(self, args, state, control, model=None, **kwargs):
        if self._before is None:
            return
        stats = {
            group: {
                "tensors": 0,
                "changed_tensors": 0,
                "changed_elements": 0,
                "delta_l2_sq": 0.0,
                "delta_max_abs": 0.0,
            }
            for group in (
                "language_lora",
                "visual_lora",
                "custom_text_proj",
                "folder_homo",
                "other",
            )
        }
        for name, parameter in model.named_parameters():
            if name not in self._before:
                continue
            before = self._before.pop(name)
            after = parameter.detach().to(device="cpu", copy=True)
            delta = after.float().sub_(before.float())
            group_stats = stats[_trainable_group(name)]
            group_stats["tensors"] += 1
            changed = delta.ne(0)
            changed_elements = int(changed.sum().item())
            if changed_elements:
                group_stats["changed_tensors"] += 1
                group_stats["changed_elements"] += changed_elements
                group_stats["delta_l2_sq"] += float(delta.square().sum().item())
                group_stats["delta_max_abs"] = max(
                    group_stats["delta_max_abs"], float(delta.abs().max().item())
                )
        if self._before:
            raise RuntimeError(
                "Deep audit lost trainable parameters across optimizer step: "
                f"{list(self._before)[:8]}"
            )
        self._before = None
        for group_stats in stats.values():
            group_stats["delta_l2"] = group_stats.pop("delta_l2_sq") ** 0.5
        required_groups = (
            "language_lora",
            "visual_lora",
            "custom_text_proj",
            "folder_homo",
        )
        unchanged = [
            group for group in required_groups if stats[group]["changed_tensors"] == 0
        ]
        if unchanged:
            raise RuntimeError(
                f"Deep audit found trainable groups unchanged after optimizer step: {unchanged}"
            )
        _write_audit_event(state, "optimizer_parameter_updates", {"groups": stats})


class DataLoaderWithSetEpoch(DataLoader):
    BUFFER_SIZE = 10000

    def set_epoch(self, epoch: int):
        if hasattr(self.dataset, "set_epoch"):
            rank0_print(f"Setting epoch {epoch} for {self.dataset}")
            self.dataset.set_epoch(epoch)
            # self.dataset.shuffle_data_sources()
        else:
            # print available methods of self.dataset
            rank0_print(f"You should never see this: {self.dataset}")


class ContrastiveTrainerV2(Trainer):
    def __init__(
        self,
        loss_func,
        do_gather,
        use_is_query,
        do_padding=False,
        ignored_fsdp_keys=None,
        *args,
        **kwargs,
    ):
        self.ignored_fsdp_keys = ignored_fsdp_keys  # __init__ -> create_accelerator_and_postprocess so this has to be set before
        super().__init__(*args, **kwargs)
        self.loss_func = loss_func
        self.do_gather = do_gather
        self.do_padding = do_padding
        self.use_new_loss = (
            "offset" in inspect.signature(self.loss_func.forward).parameters
        )
        self.needs_has_images = bool(getattr(self.loss_func, "needs_has_images", False))
        if self.use_new_loss:
            print(
                f"Using new loss {self.loss_func.__class__.__name__} with offset -- this is only compatible with Gather"
            )
        # post-check loss compatibility
        self.use_is_query = True
        if self.do_gather:
            print(f"Using Gather with {self.loss_func.__class__.__name__}")

        # self.freeze_keys = freeze_keys
        self.callback_handler = hack_callbacks_and_replace(self.callback_handler)
        if os.environ.get("MURE_DEEP_AUDIT", "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }:
            self.add_callback(MureDeepAuditOptimizerCallback())

        # for successful resuming from IterableDataset, hotpatch two functions
        if transformers.__version__ == "4.51.3":
            self._inner_training_loop = types.MethodType(_inner_training_loop, self)
            self.set_initial_training_values = types.MethodType(
                set_initial_training_values, self
            )
        elif transformers.__version__ == "4.55.0":
            self._inner_training_loop = types.MethodType(
                _inner_training_loop_4_55_0, self
            )
            self.set_initial_training_values = types.MethodType(
                set_initial_training_values_4_55_0, self
            )
        else:
            raise NotImplementedError(
                f"Transformer version {transformers.__version__} not supported for patching!"
            )

    def _get_train_sampler(self):
        raise NotImplementedError("ContrastiveTrainer v2 should never invoke this")

    def get_train_dataloader(self) -> DataLoader:
        """
        override original trainer's method to disable self.accelerator.prepare since it will wrap DataLoaderDispatcher and lead to
        (1) `RuntimeError: You can't use batches of different size with `dispatch_batches=True` or when using an `IterableDataset`.`
        (2) all outputs of dataloader must be tensors
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        train_dataset = self._remove_unused_columns(
            train_dataset, description="training"
        )
        self._mure_dataset_level_resume_skip_batches = int(
            getattr(train_dataset, "_mure_dataset_level_resume_skip_batches", 0)
        )
        configured_skip_batches = int(
            os.environ.get("MURE_DATASET_RESUME_SKIP_BATCHES", "0")
        )
        if self._mure_dataset_level_resume_skip_batches != configured_skip_batches:
            raise RuntimeError(
                "Dataset-level resume skip marker mismatch: "
                f"dataset={self._mure_dataset_level_resume_skip_batches}, "
                f"configured={configured_skip_batches}."
            )
        if configured_skip_batches > 0:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            print(
                f"[trainer-v2-fast-resume] rank={rank} "
                f"skip_batches={configured_skip_batches} "
                f"rows_per_rank={getattr(train_dataset, '_mure_dataset_level_resume_skip_rows_per_rank', 0)}",
                flush=True,
            )
        probe_start_step = int(os.environ.get("MURE_PROBE_DATA_START_STEP", "0"))
        if probe_start_step < 0:
            raise ValueError("MURE_PROBE_DATA_START_STEP must be non-negative")
        if probe_start_step > 0:
            if self._mure_dataset_level_resume_skip_batches > 0:
                raise ValueError("MURE_PROBE_DATA_START_STEP cannot be combined with checkpoint resume")
            if self.args.ignore_data_skip:
                raise ValueError(
                    "MURE_PROBE_DATA_START_STEP manages resume skipping itself; "
                    "do not enable --ignore-data-skip"
                )
            expected_skip_rows = probe_start_step * int(self._train_batch_size)
            positioned_skip_rows = int(
                getattr(train_dataset, "_mure_probe_data_skip_rows_per_rank", 0)
            )
            if positioned_skip_rows != expected_skip_rows:
                raise RuntimeError(
                    "Probe dataset position mismatch: "
                    f"dataset={positioned_skip_rows}, expected={expected_skip_rows}."
                )
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            print(
                f"[trainer-v2-probe] rank={rank} data_start_step={probe_start_step} "
                f"skip_rows={positioned_skip_rows}",
                flush=True,
            )
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        else:
            dataloader_params["sampler"] = None
            dataloader_params["shuffle"] = False
            dataloader_params["drop_last"] = True
            dataloader_params["prefetch_factor"] = (
                None  # # tune on both prefetch_factor and persistent_workers will cause hang at epoch2
            )

        # remove accelrateor prepare: we will take over the data sampler
        # however we need to keep set_epoch to ensure they sample differently
        return DataLoaderWithSetEpoch(train_dataset, **dataloader_params)

    def _debug_step_enabled(self) -> bool:
        spec = os.environ.get("CONTRASTIVE_DEBUG_STEPS", "").strip()
        if not spec:
            return False
        step = int(getattr(self.state, "global_step", -1))
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = part.split("-", 1)
                if int(start) <= step <= int(end):
                    return True
            elif step == int(part):
                return True
        return False

    @staticmethod
    def _debug_tensor_shape(value):
        shape = getattr(value, "shape", None)
        return None if shape is None else list(shape)

    @staticmethod
    def _debug_values(value, limit: int = 16):
        if torch.is_tensor(value):
            value = value.detach().flatten()[:limit].cpu().tolist()
        elif isinstance(value, (list, tuple)):
            value = list(value[:limit])
        else:
            return None
        return [item.item() if hasattr(item, "item") else item for item in value]

    def _debug_print(self, stage: str, extra: dict | None = None, *, sync_cuda: bool = False) -> float:
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.time()
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        payload = {
            "rank": rank,
            "step": int(getattr(self.state, "global_step", -1)),
            "stage": stage,
            "time": now,
        }
        if torch.cuda.is_available():
            payload.update(
                {
                    "cuda_allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 2),
                    "cuda_reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 2),
                    "cuda_peak_allocated_mb": round(
                        torch.cuda.max_memory_allocated() / 2**20, 2
                    ),
                }
            )
        if extra:
            payload.update(extra)
        line = json.dumps(payload, ensure_ascii=True, sort_keys=True)
        print(f"[contrastive-debug-v2] {line}", flush=True)
        debug_dir = os.environ.get("CONTRASTIVE_DEBUG_DIR", "").strip()
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            with open(os.path.join(debug_dir, f"rank{rank}.jsonl"), "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return now

    def _deep_audit_enabled(self) -> bool:
        return _audit_step_enabled(self.state)

    def _deep_audit_tensor(self, name: str, value: torch.Tensor) -> None:
        if not self._deep_audit_enabled():
            return
        if not torch.is_tensor(value):
            raise RuntimeError(f"Deep audit expected tensor for {name}")
        if not torch.isfinite(value).all():
            nonfinite = int((~torch.isfinite(value)).sum().item())
            raise FloatingPointError(
                f"Deep audit found {nonfinite} non-finite values in {name}"
            )

    def _deep_audit_gradients(self, model) -> None:
        if not self._deep_audit_enabled():
            return
        groups = {
            group: {
                "tensors": 0,
                "grad_tensors": 0,
                "nonzero_grad_tensors": 0,
                "numel": 0,
                "grad_l2_sq": 0.0,
                "grad_sum": 0.0,
                "grad_max_abs": 0.0,
            }
            for group in (
                "language_lora",
                "visual_lora",
                "custom_text_proj",
                "folder_homo",
                "other",
            )
        }
        missing = []
        nonfinite = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            group_stats = groups[_trainable_group(name)]
            group_stats["tensors"] += 1
            group_stats["numel"] += parameter.numel()
            if parameter.grad is None:
                missing.append(name)
                continue
            grad = parameter.grad.detach().float()
            group_stats["grad_tensors"] += 1
            if not torch.isfinite(grad).all():
                nonfinite.append(name)
                continue
            grad_abs_max = float(grad.abs().max().item())
            if grad_abs_max > 0:
                group_stats["nonzero_grad_tensors"] += 1
            group_stats["grad_l2_sq"] += float(grad.square().sum().item())
            group_stats["grad_sum"] += float(grad.sum().item())
            group_stats["grad_max_abs"] = max(
                group_stats["grad_max_abs"], grad_abs_max
            )
        if missing:
            raise RuntimeError(
                f"Deep audit found {len(missing)} trainable tensors without gradients: "
                f"{missing[:16]}"
            )
        if nonfinite:
            raise FloatingPointError(
                f"Deep audit found non-finite gradients in {len(nonfinite)} tensors: "
                f"{nonfinite[:16]}"
            )
        required_groups = (
            "language_lora",
            "visual_lora",
            "custom_text_proj",
            "folder_homo",
        )
        zero_groups = [
            group
            for group in required_groups
            if groups[group]["nonzero_grad_tensors"] == 0
        ]
        if zero_groups:
            raise RuntimeError(f"Deep audit found all-zero gradient groups: {zero_groups}")
        for group_stats in groups.values():
            group_stats["grad_l2"] = group_stats.pop("grad_l2_sq") ** 0.5

        fingerprint = torch.tensor(
            [
                value
                for group in required_groups
                for value in (
                    groups[group]["grad_sum"],
                    groups[group]["grad_l2"],
                    groups[group]["grad_max_abs"],
                )
            ],
            dtype=torch.float64,
            device=next(model.parameters()).device,
        )
        cross_rank_max_abs_diff = 0.0
        if dist.is_available() and dist.is_initialized():
            gathered = [torch.empty_like(fingerprint) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, fingerprint)
            reference = gathered[0]
            cross_rank_max_abs_diff = max(
                float((rank_value - reference).abs().max().item())
                for rank_value in gathered
            )
            tolerance = float(os.environ.get("MURE_DEEP_AUDIT_SYNC_ATOL", "1e-6"))
            if cross_rank_max_abs_diff > tolerance:
                raise RuntimeError(
                    "Deep audit found inconsistent synchronized gradient fingerprints: "
                    f"max_abs_diff={cross_rank_max_abs_diff}, tolerance={tolerance}"
                )
        _write_audit_event(
            self.state,
            "gradient_integrity",
            {
                "groups": groups,
                "cross_rank_max_abs_diff": cross_rank_max_abs_diff,
            },
        )

    @staticmethod
    def _safe_shape(tensor):
        return None if tensor is None else tuple(tensor.shape)

    def _debug_batch_summary(self, inputs, query_outputs, doc_outputs, neg_doc_outputs=None):
        if os.environ.get("MURE_TRAINER_DEBUG_BATCH", "").strip().lower() not in {"1", "true", "yes", "y"}:
            return
        step = getattr(self.state, "global_step", -1)
        if step > 2:
            return
        rank = dist.get_rank() if dist.is_initialized() else 0

        def _sum_bool(name: str):
            value = inputs.get(name)
            if value is None:
                return None
            if value.dtype == torch.bool:
                return int(value.sum().item())
            return None

        print(
            f"[debug-batch] step={step} rank={rank} "
            f"query_ids={self._safe_shape(inputs.get('query_input_ids'))} "
            f"doc_ids={self._safe_shape(inputs.get('doc_input_ids'))} "
            f"neg_doc_ids={self._safe_shape(inputs.get('neg_doc_input_ids'))} "
            f"query_emb={tuple(query_outputs.shape)} doc_emb_local={tuple(doc_outputs.shape)} "
            f"neg_doc_emb={self._safe_shape(neg_doc_outputs)} "
            f"query_has_images={_sum_bool('query_has_images')} "
            f"doc_has_images={_sum_bool('doc_has_images')} "
            f"neg_doc_has_images={_sum_bool('neg_doc_has_images')}"
        )

    @staticmethod
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
        return torch.full(
            (input_ids.shape[0],),
            bool(has_visuals),
            dtype=torch.bool,
            device=input_ids.device,
        )

    @staticmethod
    def _gather_bool_rows(flags: torch.Tensor) -> torch.Tensor:
        if not dist.is_initialized():
            return flags
        gathered = [torch.zeros_like(flags) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, flags)
        return torch.cat(gathered, dim=0)

    @staticmethod
    def _model_inputs(inputs: dict, prefix: str) -> dict:
        blocked_key = f"{prefix}_has_images"
        prefix_with_sep = f"{prefix}_"
        prefix_len = len(prefix_with_sep)
        return {
            key[prefix_len:]: value
            for key, value in inputs.items()
            if key.startswith(prefix_with_sep) and key != blocked_key
        }

    def _encoder_forward(self, model, model_inputs: dict, stage_name: str):
        max_visual_rows = int(os.environ.get("MURE_ENCODER_MAX_VISUAL_ROWS", "60000"))
        pixel_values = model_inputs.get("pixel_values")
        image_grid_thw = model_inputs.get("image_grid_thw")
        if (
            max_visual_rows <= 0
            or pixel_values is None
            or image_grid_thw is None
            or image_grid_thw.numel() == 0
        ):
            return model(**model_inputs)

        input_ids = model_inputs["input_ids"]
        batch_size = input_ids.shape[0]
        unwrapped_model = model.module if hasattr(model, "module") else model
        if hasattr(unwrapped_model, "get_base_model"):
            unwrapped_model = unwrapped_model.get_base_model()
        config = getattr(unwrapped_model, "config", None)
        image_token_id = getattr(config, "image_token_id", None)
        vision_config = getattr(config, "vision_config", None)
        spatial_merge_size = int(getattr(vision_config, "spatial_merge_size", 0))
        if image_token_id is None or spatial_merge_size <= 0:
            raise RuntimeError(
                "Encoder visual microbatching requires image_token_id and spatial_merge_size"
            )

        merge_area = spatial_merge_size * spatial_merge_size
        raw_rows_per_grid = image_grid_thw.long().prod(dim=1).cpu().tolist()
        image_tokens_per_sample = (
            input_ids.eq(int(image_token_id)).sum(dim=1).cpu().tolist()
        )
        sample_spans = []
        grid_cursor = 0
        pixel_cursor = 0
        for sample_index, target_tokens in enumerate(image_tokens_per_sample):
            grid_start = grid_cursor
            pixel_start = pixel_cursor
            assigned_tokens = 0
            while assigned_tokens < target_tokens:
                if grid_cursor >= len(raw_rows_per_grid):
                    raise RuntimeError(
                        f"{stage_name} visual grids ended while assigning sample {sample_index}"
                    )
                raw_rows = int(raw_rows_per_grid[grid_cursor])
                if raw_rows % merge_area != 0:
                    raise RuntimeError(
                        f"{stage_name} grid rows {raw_rows} are not divisible by merge area {merge_area}"
                    )
                assigned_tokens += raw_rows // merge_area
                pixel_cursor += raw_rows
                grid_cursor += 1
            if assigned_tokens != target_tokens:
                raise RuntimeError(
                    f"{stage_name} sample {sample_index} image-token mismatch: "
                    f"input_ids={target_tokens}, grids={assigned_tokens}"
                )
            sample_spans.append(
                (grid_start, grid_cursor, pixel_start, pixel_cursor)
            )
        if grid_cursor != len(raw_rows_per_grid) or pixel_cursor != pixel_values.shape[0]:
            raise RuntimeError(
                f"{stage_name} visual packing mismatch: grids={grid_cursor}/{len(raw_rows_per_grid)} "
                f"pixels={pixel_cursor}/{pixel_values.shape[0]}"
            )

        groups = []
        group_start = 0
        group_rows = 0
        for sample_index, span in enumerate(sample_spans):
            sample_rows = span[3] - span[2]
            if sample_index > group_start and group_rows + sample_rows > max_visual_rows:
                groups.append((group_start, sample_index, group_rows))
                group_start = sample_index
                group_rows = 0
            group_rows += sample_rows
        groups.append((group_start, batch_size, group_rows))
        if len(groups) == 1:
            return model(**model_inputs)

        if self._debug_step_enabled():
            self._debug_print(
                f"{stage_name}_encoder_microbatch_plan",
                {
                    "max_visual_rows": max_visual_rows,
                    "groups": [
                        {"start": start, "end": end, "visual_rows": rows}
                        for start, end, rows in groups
                    ],
                },
            )

        outputs = []
        for start, end, _ in groups:
            grid_start = sample_spans[start][0]
            grid_end = sample_spans[end - 1][1]
            pixel_start = sample_spans[start][2]
            pixel_end = sample_spans[end - 1][3]
            micro_inputs = {}
            for key, value in model_inputs.items():
                if key == "pixel_values":
                    micro_inputs[key] = value[pixel_start:pixel_end]
                elif key == "image_grid_thw":
                    micro_inputs[key] = value[grid_start:grid_end]
                elif torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size:
                    micro_inputs[key] = value[start:end]
                else:
                    micro_inputs[key] = value
            outputs.append(model(**micro_inputs))

        max_output_length = max(output.shape[1] for output in outputs)
        padded_outputs = [
            torch.nn.functional.pad(
                output,
                (0, 0, 0, max_output_length - output.shape[1]),
            )
            for output in outputs
        ]
        return torch.cat(padded_outputs, dim=0)

    @staticmethod
    def _gather_2d_tensor_rows(tensor: torch.Tensor) -> torch.Tensor:
        if not dist.is_initialized():
            return tensor
        gathered, _ = all_gather_with_padding_select_dim(
            tensor,
            dist.get_world_size(),
            pad_dim=1,
        )
        return gathered

    def training_step(self, model, inputs, *args, **kwargs):
        debug_enabled = self._debug_step_enabled()
        if debug_enabled and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if debug_enabled:
            self._debug_print(
                "training_step_start",
                {
                    "input_keys": sorted(inputs),
                    "subset_idx": self._debug_values(inputs.get("subset_idx")),
                    "data_idx": self._debug_values(inputs.get("data_idx")),
                },
            )
        gather_mode = os.environ.get("MURE_GATHER_WITH_GRAD_MODE", "torch").strip().lower()
        defer_setting = os.environ.get("MURE_DEFER_DDP_REDUCE", "auto").strip().lower()
        if defer_setting not in {"auto", "1", "true", "yes", "0", "false", "no"}:
            raise ValueError(
                "MURE_DEFER_DDP_REDUCE must be 'auto', true, or false; "
                f"got {defer_setting!r}"
            )
        full_gradient_gather = gather_mode in {"torch", "torch_all_gather", "full"}
        defer_ddp_reduce = (
            self.do_gather
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
            and full_gradient_gather
            and defer_setting not in {"0", "false", "no"}
        )
        if defer_setting in {"1", "true", "yes"} and not full_gradient_gather:
            raise ValueError(
                "MURE_DEFER_DDP_REDUCE requires full-gradient gather mode; "
                f"got MURE_GATHER_WITH_GRAD_MODE={gather_mode!r}"
            )
        if defer_ddp_reduce and not hasattr(model, "no_sync"):
            raise RuntimeError("Deferred DDP gradient sync requires a DDP-wrapped model")

        sync_context = model.no_sync if defer_ddp_reduce else contextlib.nullcontext
        with sync_context():
            self._mure_staged_encoder_backwards = []
            result = super().training_step(model, inputs, *args, **kwargs)
            staged_backwards = self._mure_staged_encoder_backwards
            for stage_name, encoder_outputs, embedding_leaf in staged_backwards:
                if embedding_leaf.grad is None:
                    raise RuntimeError(
                        f"Staged {stage_name} backward did not produce an embedding gradient"
                    )
                if debug_enabled:
                    self._debug_print(f"staged_{stage_name}_backward_start", sync_cuda=True)
                staged_gradient = embedding_leaf.grad.detach().clone()
                torch.autograd.backward(encoder_outputs, staged_gradient)
                if debug_enabled:
                    self._debug_print(f"staged_{stage_name}_backward_done", sync_cuda=True)
            if staged_backwards:
                trainable_parameters = [
                    (name, parameter)
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                ]
                parameters_with_grad = sum(
                    parameter.grad is not None
                    for _, parameter in trainable_parameters
                )
                if parameters_with_grad == 0:
                    raise RuntimeError(
                        "Staged encoder backward produced no trainable parameter gradients; "
                        "for PEFT with reentrant gradient checkpointing, input gradients "
                        "must be enabled before the forward pass"
                    )
                if debug_enabled:
                    self._debug_print(
                        "staged_parameter_gradients_ready",
                        {
                            "trainable_parameters": len(trainable_parameters),
                            "parameters_with_grad": parameters_with_grad,
                            "missing_gradient_parameters": [
                                name
                                for name, parameter in trainable_parameters
                                if parameter.grad is None
                            ][:24],
                        },
                        sync_cuda=True,
                    )
            self._mure_staged_encoder_backwards = []

            if debug_enabled and not staged_backwards:
                trainable_parameters = [
                    (name, parameter)
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                ]
                self._debug_print(
                    "parameter_gradients_ready",
                    {
                        "trainable_parameters": len(trainable_parameters),
                        "parameters_with_grad": sum(
                            parameter.grad is not None
                            for _, parameter in trainable_parameters
                        ),
                        "missing_gradient_parameters": [
                            name
                            for name, parameter in trainable_parameters
                            if parameter.grad is None
                        ][:24],
                    },
                    sync_cuda=True,
                )

        if defer_ddp_reduce and self.accelerator.sync_gradients:
            if debug_enabled:
                self._debug_print("pre_grad_sync_cpu_barrier_start", sync_cuda=True)
            wait_for_all_ranks_cpu()
            if debug_enabled:
                self._debug_print("pre_grad_sync_cpu_barrier_done")
                self._debug_print("deferred_grad_sync_start")
            bucket_mb = int(os.environ.get("MURE_DEFER_DDP_BUCKET_MB", "64"))
            sync_gradients_after_backward(model, bucket_bytes=bucket_mb * 1024 * 1024)
            if debug_enabled:
                self._debug_print("deferred_grad_sync_done", sync_cuda=True)
        self._deep_audit_gradients(model)
        self._deep_audit_tensor("training_loss", result)
        if debug_enabled:
            loss_value = None
            if torch.is_tensor(result) and result.numel() == 1:
                loss_value = float(result.detach().float().cpu().item())
            self._debug_print("training_step_done", {"loss": loss_value}, sync_cuda=True)
        return result

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        loss_stats = None
        encoder_backward_mode = os.environ.get(
            "MURE_ENCODER_BACKWARD_MODE", "full"
        ).strip().lower()
        if encoder_backward_mode not in {"full", "staged"}:
            raise ValueError(
                "MURE_ENCODER_BACKWARD_MODE must be 'full' or 'staged'; "
                f"got {encoder_backward_mode!r}"
            )
        staged_encoder_backward = encoder_backward_mode == "staged" and model.training
        if staged_encoder_backward and not hasattr(self, "_mure_staged_encoder_backwards"):
            raise RuntimeError("Staged encoder backward must run through training_step")
        debug_enabled = self._debug_step_enabled()
        last_debug_time = None
        if debug_enabled:
            last_debug_time = self._debug_print(
                "compute_loss_start",
                {
                    "query_input_ids": self._debug_tensor_shape(inputs.get("query_input_ids")),
                    "doc_input_ids": self._debug_tensor_shape(inputs.get("doc_input_ids")),
                    "neg_doc_input_ids": self._debug_tensor_shape(inputs.get("neg_doc_input_ids")),
                    "query_pixel_values": self._debug_tensor_shape(inputs.get("query_pixel_values")),
                    "doc_pixel_values": self._debug_tensor_shape(inputs.get("doc_pixel_values")),
                    "neg_doc_pixel_values": self._debug_tensor_shape(inputs.get("neg_doc_pixel_values")),
                    "subset_idx": self._debug_values(inputs.get("subset_idx")),
                    "data_idx": self._debug_values(inputs.get("data_idx")),
                },
            )

        if self.use_is_query:
            inputs["query_is_query"] = True
        if debug_enabled:
            self._debug_print("query_forward_start")
        query_outputs = self._encoder_forward(
            model, self._model_inputs(inputs, "query"), "query"
        )
        self._deep_audit_tensor("query_outputs", query_outputs)
        if debug_enabled:
            last_debug_time = self._debug_print(
                "query_forward_done",
                {"seconds": time.time() - last_debug_time, "output": self._debug_tensor_shape(query_outputs)},
                sync_cuda=True,
            )
        inputs.pop("query_is_query", None)
        if staged_encoder_backward:
            query_embeddings = query_outputs.detach().requires_grad_(True)
            self._mure_staged_encoder_backwards.append(
                ("query", query_outputs, query_embeddings)
            )
        else:
            query_embeddings = query_outputs

        if debug_enabled:
            self._debug_print("doc_forward_start")
        doc_outputs = self._encoder_forward(
            model, self._model_inputs(inputs, "doc"), "doc"
        )
        self._deep_audit_tensor("doc_outputs", doc_outputs)
        if debug_enabled:
            last_debug_time = self._debug_print(
                "doc_forward_done",
                {"seconds": time.time() - last_debug_time, "output": self._debug_tensor_shape(doc_outputs)},
                sync_cuda=True,
            )

        additional_loss_kwargs = {}
        query_has_images = self._extract_has_images(
            input_ids=inputs["query_input_ids"],
            has_images=inputs.get("query_has_images"),
            pixel_values=inputs.get("query_pixel_values"),
            image_grid_thw=inputs.get("query_image_grid_thw"),
        )
        doc_has_images_local = self._extract_has_images(
            input_ids=inputs["doc_input_ids"],
            has_images=inputs.get("doc_has_images"),
            pixel_values=inputs.get("doc_pixel_values"),
            image_grid_thw=inputs.get("doc_image_grid_thw"),
        )
        neg_doc_has_images = None

        if self.do_gather:
            if self.do_padding:
                if debug_enabled:
                    self._debug_print("pad_doc_start")
                doc_outputs, doc_local_lens = pad_to_max_len_right(
                    doc_outputs, dist.get_world_size()
                )
                if debug_enabled:
                    last_debug_time = self._debug_print(
                        "pad_doc_done",
                        {"seconds": time.time() - last_debug_time, "output": self._debug_tensor_shape(doc_outputs)},
                        sync_cuda=True,
                    )
                if os.environ.get("MURE_TRAINER_DEBUG_BATCH", "").strip().lower() in {"1", "true", "yes", "y"} and getattr(self.state, "global_step", -1) <= 2:
                    print(
                        f"[debug-pad] step={getattr(self.state, 'global_step', -1)} rank={dist.get_rank()} "
                        f"doc_local_lens={[int(x.item()) for x in doc_local_lens]} padded_doc={tuple(doc_outputs.shape)}"
                    )
            if staged_encoder_backward:
                doc_outputs_for_loss = doc_outputs.detach().requires_grad_(True)
                self._mure_staged_encoder_backwards.append(
                    ("doc", doc_outputs, doc_outputs_for_loss)
                )
            else:
                doc_outputs_for_loss = doc_outputs
            # print(
            #     f"rank: {dist.get_rank()} doc_embeddings before gather: {doc_outputs.shape}"
            # )
            if debug_enabled:
                self._debug_print("gather_doc_start")
            doc_embeddings = gather_with_grad_torch(
                doc_outputs_for_loss
            )  # [B * num_gpus, Nd, D]
            self._deep_audit_tensor("gathered_doc_embeddings", doc_embeddings)
            if debug_enabled:
                last_debug_time = self._debug_print(
                    "gather_doc_done",
                    {"seconds": time.time() - last_debug_time, "output": self._debug_tensor_shape(doc_embeddings)},
                    sync_cuda=True,
                )
            if os.environ.get("MURE_TRAINER_DEBUG_BATCH", "").strip().lower() in {"1", "true", "yes", "y"} and getattr(self.state, "global_step", -1) <= 2:
                print(
                    f"[debug-gather] step={getattr(self.state, 'global_step', -1)} rank={dist.get_rank()} "
                    f"gathered_doc={tuple(doc_embeddings.shape)}"
                )
            doc_has_images = self._gather_bool_rows(doc_has_images_local)
            # print(
            #     f"rank: {dist.get_rank()} doc_embeddings after gather: {doc_embeddings.shape}"
            # )
            if self.use_new_loss:
                additional_loss_kwargs["offset"] = (
                    dist.get_rank() * inputs["query_input_ids"].shape[0]
                )
        else:
            if staged_encoder_backward:
                doc_embeddings = doc_outputs.detach().requires_grad_(True)
                self._mure_staged_encoder_backwards.append(
                    ("doc", doc_outputs, doc_embeddings)
                )
            else:
                doc_embeddings = doc_outputs
            doc_has_images = doc_has_images_local

        if self.needs_has_images:
            additional_loss_kwargs["query_has_images"] = query_has_images
            additional_loss_kwargs["doc_has_images"] = doc_has_images
        if getattr(self.loss_func, "needs_input_ids", False):
            additional_loss_kwargs.update(
                {
                    "query_input_ids": inputs.get("query_input_ids"),
                    "query_attention_mask": inputs.get("query_attention_mask"),
                    "doc_input_ids": self._gather_2d_tensor_rows(inputs.get("doc_input_ids"))
                    if self.do_gather
                    else inputs.get("doc_input_ids"),
                    "doc_attention_mask": self._gather_2d_tensor_rows(inputs.get("doc_attention_mask"))
                    if self.do_gather
                    else inputs.get("doc_attention_mask"),
                }
            )

        if "neg_doc_input_ids" in inputs:
            if debug_enabled:
                self._debug_print("neg_doc_forward_start")
            neg_doc_outputs = self._encoder_forward(
                model, self._model_inputs(inputs, "neg_doc"), "neg_doc"
            )
            self._deep_audit_tensor("neg_doc_outputs", neg_doc_outputs)
            if debug_enabled:
                last_debug_time = self._debug_print(
                    "neg_doc_forward_done",
                    {"seconds": time.time() - last_debug_time, "output": self._debug_tensor_shape(neg_doc_outputs)},
                    sync_cuda=True,
                )

            if staged_encoder_backward:
                neg_doc_embeddings = neg_doc_outputs.detach().requires_grad_(True)
                self._mure_staged_encoder_backwards.append(
                    ("neg_doc", neg_doc_outputs, neg_doc_embeddings)
                )
            else:
                neg_doc_embeddings = neg_doc_outputs
            if debug_enabled:
                self._debug_print(
                    "encoder_backward_mode",
                    {"mode": encoder_backward_mode},
                )
            self._debug_batch_summary(inputs, query_embeddings, doc_outputs, neg_doc_outputs=neg_doc_embeddings)
            neg_doc_has_images = self._extract_has_images(
                input_ids=inputs["neg_doc_input_ids"],
                has_images=inputs.get("neg_doc_has_images"),
                pixel_values=inputs.get("neg_doc_pixel_values"),
                image_grid_thw=inputs.get("neg_doc_image_grid_thw"),
            )
            if self.needs_has_images:
                additional_loss_kwargs["neg_doc_has_images"] = neg_doc_has_images

            if getattr(self.loss_func, "needs_input_ids", False):
                additional_loss_kwargs.update(
                    {
                        "neg_doc_input_ids": inputs.get("neg_doc_input_ids"),
                        "neg_doc_attention_mask": inputs.get("neg_doc_attention_mask"),
                    }
                )
            # never gather neg_doc_embeddings, they are only useful to current query

            # DEBUG: dump to local pt files and examine the inputs & embeds
            # torch.save(
            #     {
            #         "subset_idx": inputs["subset_idx"],
            #         "data_idx": inputs["data_idx"],
            #     },
            #     f"../debug_tensors/train_stats_{self.state.global_step}_{dist.get_rank()}.pt",
            # )
            # if self.state.global_step < 15:
            # torch.save(
            #     {
            #         "inputs": inputs,
            #         "query_embeddings": query_embeddings,
            #         "doc_embeddings": doc_embeddings,
            #         "neg_doc_embeddings": neg_doc_embeddings,
            #     },
            #     f"../debug_tensors/train_loop_{self.state.global_step}_{dist.get_rank()}.pt",
            # )
            # if self.state.global_step == 2:
            #     if dist.get_rank() == 0:
            #         breakpoint()
            #     dist.barrier()
            # if dist.get_rank() == 0:
            #     breakpoint()
            # dist.barrier()
            if debug_enabled:
                self._debug_print("loss_start")
            loss = self.loss_func(
                query_embeddings,
                doc_embeddings,
                neg_doc_embeddings,
                **additional_loss_kwargs,
            )
            if debug_enabled:
                raw_loss = loss[0] if isinstance(loss, tuple) else loss
                self._debug_print(
                    "loss_done",
                    {"seconds": time.time() - last_debug_time, "loss": float(raw_loss.detach().float().cpu().item())},
                    sync_cuda=True,
                )
            if isinstance(loss, tuple):
                loss, loss_stats = loss
            self._deep_audit_tensor("hard_negative_loss", loss)

            # inject loss_stats to self.state.loss_stats
            if loss_stats and self.state.global_step % self.args.logging_steps == 0:
                for k, v in loss_stats.items():
                    self.state.loss_stats[k] = self._nested_gather(v).mean().item()

            return (
                (loss, (query_embeddings, doc_embeddings, neg_doc_embeddings))
                if return_outputs
                else loss
            )

        self._debug_batch_summary(inputs, query_embeddings, doc_outputs)
        loss = self.loss_func(
            query_embeddings,
            doc_embeddings,
            **additional_loss_kwargs,
        )

        if isinstance(loss, tuple):
            loss, loss_stats = loss
        self._deep_audit_tensor("contrastive_loss", loss)

        # inject loss_stats to self.state.loss_stats
        if loss_stats and self.state.global_step % self.args.logging_steps == 0:
            for k, v in loss_stats.items():
                self.state.loss_stats[k] = self._nested_gather(v).mean().item()

        return (loss, (query_embeddings, doc_embeddings)) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=True):
        """This function is used to generate predictions and return the loss for the given inputs."""
        if not prediction_loss_only:
            raise ValueError(
                "prediction_step is only called with prediction_loss_only=True"
            )

        with torch.no_grad():
            if self.use_is_query:
                inputs["query_is_query"] = True
            query_outputs = model(**self._model_inputs(inputs, "query"))
            inputs.pop("query_is_query", None)
            # feed only kwargs with 'doc_' prefix
            doc_outputs = model(**self._model_inputs(inputs, "doc"))
            additional_loss_kwargs = {}
            if self.needs_has_images:
                additional_loss_kwargs["query_has_images"] = self._extract_has_images(
                    input_ids=inputs["query_input_ids"],
                    has_images=inputs.get("query_has_images"),
                    pixel_values=inputs.get("query_pixel_values"),
                    image_grid_thw=inputs.get("query_image_grid_thw"),
                )
                additional_loss_kwargs["doc_has_images"] = self._extract_has_images(
                    input_ids=inputs["doc_input_ids"],
                    has_images=inputs.get("doc_has_images"),
                    pixel_values=inputs.get("doc_pixel_values"),
                    image_grid_thw=inputs.get("doc_image_grid_thw"),
                )
            if getattr(self.loss_func, "needs_input_ids", False):
                additional_loss_kwargs.update(
                    {
                        "query_input_ids": inputs.get("query_input_ids"),
                        "query_attention_mask": inputs.get("query_attention_mask"),
                        "doc_input_ids": self._gather_2d_tensor_rows(inputs.get("doc_input_ids"))
                        if self.do_gather
                        else inputs.get("doc_input_ids"),
                        "doc_attention_mask": self._gather_2d_tensor_rows(inputs.get("doc_attention_mask"))
                        if self.do_gather
                        else inputs.get("doc_attention_mask"),
                    }
                )
            if "neg_doc_input_ids" in inputs:
                # stuck here! it seems neg_doc does not have padded images
                # shape_info = {
                #     k: v.shape for k, v in inputs.items() if k.startswith("neg_doc")
                # }
                # print(f"rank: {dist.get_rank()} all_shapes: {shape_info}")
                neg_doc_outputs = model(
                    **self._model_inputs(inputs, "neg_doc")
                )
                if self.needs_has_images:
                    additional_loss_kwargs["neg_doc_has_images"] = self._extract_has_images(
                        input_ids=inputs["neg_doc_input_ids"],
                        has_images=inputs.get("neg_doc_has_images"),
                        pixel_values=inputs.get("neg_doc_pixel_values"),
                        image_grid_thw=inputs.get("neg_doc_image_grid_thw"),
                    )

                if getattr(self.loss_func, "needs_input_ids", False):
                    additional_loss_kwargs.update(
                        {
                            "neg_doc_input_ids": inputs.get("neg_doc_input_ids"),
                            "neg_doc_attention_mask": inputs.get("neg_doc_attention_mask"),
                        }
                    )
                loss = self.loss_func(
                    query_outputs,
                    doc_outputs,
                    neg_doc_outputs,
                    **additional_loss_kwargs,
                )

                # if dist.get_rank() == 0:
                #     breakpoint()
                # dist.barrier()

                if isinstance(loss, tuple):
                    loss, loss_stats = loss
                return loss, None, None

            loss = self.loss_func(query_outputs, doc_outputs, **additional_loss_kwargs)

            if isinstance(loss, tuple):
                loss, loss_stats = loss

            return loss, None, None
