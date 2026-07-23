import inspect

import os
import random
import time

import torch
import torch.distributed as dist
from colpali_engine.utils.dist_utils import (
    all_gather_with_padding_select_dim,
    gather_with_grad_torch,
    pad_to_max_len_right,
    rank0_print,
)
from colpali_engine.utils.trainer_utils import hack_callbacks_and_replace

from transformers import Trainer


class ContrastiveTrainer(Trainer):
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
            rank0_print(
                f"Using new loss {self.loss_func.__class__.__name__} with offset -- this is only compatible with Gather"
            )
        # post-check loss compatibility
        self.use_is_query = use_is_query
        if self.use_is_query:
            rank0_print(
                "Using is_query in the model forward pass -- useful when playing with LastXXX. "
            )
        if self.do_gather:
            rank0_print(f"Using Gather with {self.loss_func.__class__.__name__}")

        # self.freeze_keys = freeze_keys

        self.callback_handler = hack_callbacks_and_replace(self.callback_handler)

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
        if shape is None:
            return None
        return list(shape)

    @staticmethod
    def _debug_tensor_values(value, limit: int = 16):
        if not torch.is_tensor(value):
            return None
        flat = value.detach().flatten()[:limit].cpu()
        return [int(x.item()) if x.numel() == 1 else x for x in flat]

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
        if extra:
            payload.update(extra)
        print(f"[contrastive-debug] {payload}", flush=True)
        return now

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

    # DEPRECATED: we do not plan to use FSDP in the future
    def hack_fsdp_plugin(self):
        def iterative_getattr(obj, attr):
            for key in attr.split("."):
                obj = getattr(obj, key)
            return obj

        # WARNGING: this only works when no LoRA is used on the model's vision part
        if self.is_fsdp_enabled and self.ignored_fsdp_keys is not None:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            # hack `fsdp_plugin` to use `ignored_modules` -- as long as it happens before preparing the model
            ignored_modules = [
                iterative_getattr(
                    self.model, "model." + key
                )  # "model." compensate for LoRA
                for key in self.ignored_fsdp_keys
            ]
            setattr(fsdp_plugin, "ignored_modules", ignored_modules)
            rank0_print(f"Setting ignored modules: {ignored_modules} for FSDP")

    def training_step(self, model, inputs, *args, **kwargs):
        debug_enabled = self._debug_step_enabled()
        if debug_enabled:
            self._debug_print(
                "training_step_start",
                {
                    "input_keys": sorted(list(inputs.keys())),
                    "subset_idx": self._debug_tensor_values(inputs.get("subset_idx")),
                    "data_idx": self._debug_tensor_values(inputs.get("data_idx")),
                },
            )
        result = super().training_step(model, inputs, *args, **kwargs)
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
                    "subset_idx": self._debug_tensor_values(inputs.get("subset_idx")),
                    "data_idx": self._debug_tensor_values(inputs.get("data_idx")),
                },
            )

        if self.use_is_query:
            inputs["query_is_query"] = True
        if debug_enabled:
            self._debug_print("query_forward_start")
        query_outputs = model(**self._model_inputs(inputs, "query"))
        if debug_enabled:
            now = self._debug_print(
                "query_forward_done",
                {"seconds": time.time() - (last_debug_time or time.time()), "query_outputs": self._debug_tensor_shape(query_outputs)},
                sync_cuda=True,
            )
            last_debug_time = now
        inputs.pop("query_is_query", None)
        if debug_enabled:
            self._debug_print("doc_forward_start")
        doc_outputs = model(**self._model_inputs(inputs, "doc"))
        if debug_enabled:
            now = self._debug_print(
                "doc_forward_done",
                {"seconds": time.time() - (last_debug_time or time.time()), "doc_outputs": self._debug_tensor_shape(doc_outputs)},
                sync_cuda=True,
            )
            last_debug_time = now

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

        if self.do_gather:
            query_embeddings = query_outputs  # [B, Nq, D]
            if self.do_padding:
                if debug_enabled:
                    self._debug_print("pad_doc_start")
                doc_outputs, _ = pad_to_max_len_right(
                    doc_outputs, dist.get_world_size()
                )
                if debug_enabled:
                    now = self._debug_print(
                        "pad_doc_done",
                        {"seconds": time.time() - (last_debug_time or time.time()), "doc_outputs": self._debug_tensor_shape(doc_outputs)},
                        sync_cuda=True,
                    )
                    last_debug_time = now
            if debug_enabled:
                self._debug_print("gather_doc_start")
            doc_embeddings = gather_with_grad_torch(
                doc_outputs
            )  # [B * num_gpus, Nd, D]
            if debug_enabled:
                now = self._debug_print(
                    "gather_doc_done",
                    {"seconds": time.time() - (last_debug_time or time.time()), "doc_embeddings": self._debug_tensor_shape(doc_embeddings)},
                    sync_cuda=True,
                )
                last_debug_time = now
            doc_has_images = self._gather_bool_rows(doc_has_images_local)
            if self.use_new_loss:
                additional_loss_kwargs["offset"] = (
                    dist.get_rank() * inputs["query_input_ids"].shape[0]
                )
        else:
            query_embeddings = query_outputs
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
            neg_doc_outputs = model(
                **self._model_inputs(inputs, "neg_doc")
            )
            if debug_enabled:
                now = self._debug_print(
                    "neg_doc_forward_done",
                    {"seconds": time.time() - (last_debug_time or time.time()), "neg_doc_outputs": self._debug_tensor_shape(neg_doc_outputs)},
                    sync_cuda=True,
                )
                last_debug_time = now

            neg_doc_embeddings = neg_doc_outputs
            if self.needs_has_images:
                additional_loss_kwargs["neg_doc_has_images"] = self._extract_has_images(
                    input_ids=inputs["neg_doc_input_ids"],
                    has_images=inputs.get("neg_doc_has_images"),
                    pixel_values=inputs.get("neg_doc_pixel_values"),
                    image_grid_thw=inputs.get("neg_doc_image_grid_thw"),
                )

            # Some research losses need access to input_ids/attention_mask to build token pools.
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
            #         "inputs": inputs,
            #         "query_embeddings": query_embeddings,
            #         "doc_embeddings": doc_embeddings,
            #         "neg_doc_embeddings": neg_doc_embeddings,
            #     },
            #     f"debug_tensors/train_loop_{self.state.global_step}_{dist.get_rank()}.pt",
            # )
            # if self.state.global_step == 2:
            #     if dist.get_rank() == 0:
            #         breakpoint()
            #     dist.barrier()

            if debug_enabled:
                self._debug_print("loss_start")
            loss = self.loss_func(
                query_embeddings,
                doc_embeddings,
                neg_doc_embeddings,
                **additional_loss_kwargs,
            )
            if debug_enabled:
                now = self._debug_print(
                    "loss_done",
                    {"seconds": time.time() - (last_debug_time or time.time()), "loss_shape": self._debug_tensor_shape(loss[0] if isinstance(loss, tuple) else loss)},
                    sync_cuda=True,
                )
                last_debug_time = now
            if isinstance(loss, tuple):
                loss, loss_stats = loss

            # inject loss_stats to self.state.loss_stats
            if loss_stats and self.state.global_step % self.args.logging_steps == 0:
                for k, v in loss_stats.items():
                    self.state.loss_stats[k] = self._nested_gather(v).mean().item()

            return (
                (loss, (query_embeddings, doc_embeddings, neg_doc_embeddings))
                if return_outputs
                else loss
            )
        # if self.state.global_step % 100 == 0:
        #     rank0_print(f"saving debug tensors at step {self.state.global_step}")
        #     torch.save(
        #         {
        #             "inputs": inputs,
        #             "query_embeddings": query_embeddings,
        #             "doc_embeddings": doc_embeddings,
        #             # "neg_doc_embeddings": neg_doc_embeddings,
        #         },
        #         f"debug_tensors/train_loop_{self.state.global_step}_{dist.get_rank()}.pt",
        #     )
        # if self.state.global_step == 2:
        #     if dist.get_rank() == 0:
        #         breakpoint()
        #     dist.barrier()

        if debug_enabled:
            self._debug_print("loss_start")
        loss = self.loss_func(
            query_embeddings,
            doc_embeddings,
            **additional_loss_kwargs,
        )
        if debug_enabled:
            self._debug_print(
                "loss_done",
                {"seconds": time.time() - (last_debug_time or time.time()), "loss_shape": self._debug_tensor_shape(loss[0] if isinstance(loss, tuple) else loss)},
                sync_cuda=True,
            )

        if isinstance(loss, tuple):
            loss, loss_stats = loss

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

                if isinstance(loss, tuple):
                    loss, loss_stats = loss
                return loss, None, None

            loss = self.loss_func(query_outputs, doc_outputs, **additional_loss_kwargs)

            if isinstance(loss, tuple):
                loss, loss_stats = loss

            return loss, None, None
