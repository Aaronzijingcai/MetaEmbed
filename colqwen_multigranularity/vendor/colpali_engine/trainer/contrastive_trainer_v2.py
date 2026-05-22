# V2 version contrastive trainer uses gather for all query, docs and optionally neg_dows
# additional support over v1 includes:
# 1) support for multiple negatives (*DONE)
# 2) remove padding for ColBERT dim-1
# 3) disable random sampler to enable interleaved batching
# 4) only gather for positive documents
import inspect

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
)
from colpali_engine.utils.trainer_utils import hack_callbacks_and_replace

from torch.utils.data import DataLoader
from transformers import Trainer

from transformers.trainer_utils import seed_worker


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

    @staticmethod
    def _safe_shape(tensor):
        return None if tensor is None else tuple(tensor.shape)

    def _debug_batch_summary(self, inputs, query_outputs, doc_outputs, neg_doc_outputs=None):
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

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        loss_stats = None

        if self.use_is_query:
            inputs["query_is_query"] = True
        query_outputs = model(**self._model_inputs(inputs, "query"))
        inputs.pop("query_is_query", None)

        doc_outputs = model(**self._model_inputs(inputs, "doc"))

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
            query_embeddings = query_outputs  # [B, Nq, D]
            if self.do_padding:
                doc_outputs, doc_local_lens = pad_to_max_len_right(
                    doc_outputs, dist.get_world_size()
                )
                if getattr(self.state, "global_step", -1) <= 2:
                    print(
                        f"[debug-pad] step={getattr(self.state, 'global_step', -1)} rank={dist.get_rank()} "
                        f"doc_local_lens={[int(x.item()) for x in doc_local_lens]} padded_doc={tuple(doc_outputs.shape)}"
                    )
            # print(
            #     f"rank: {dist.get_rank()} doc_embeddings before gather: {doc_outputs.shape}"
            # )
            doc_embeddings = gather_with_grad_torch(
                doc_outputs
            )  # [B * num_gpus, Nd, D]
            if getattr(self.state, "global_step", -1) <= 2:
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
            neg_doc_outputs = model(
                **self._model_inputs(inputs, "neg_doc")
            )

            neg_doc_embeddings = neg_doc_outputs
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
            loss = self.loss_func(
                query_embeddings,
                doc_embeddings,
                neg_doc_embeddings,
                **additional_loss_kwargs,
            )
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

        self._debug_batch_summary(inputs, query_embeddings, doc_outputs)
        loss = self.loss_func(
            query_embeddings,
            doc_embeddings,
            **additional_loss_kwargs,
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
