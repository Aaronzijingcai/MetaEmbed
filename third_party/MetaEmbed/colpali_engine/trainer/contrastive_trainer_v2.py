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
    all_gather_with_padding,
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

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        loss_stats = None

        if self.use_is_query:
            inputs["query_is_query"] = True
        query_outputs = model(
            **{k[6:]: v for k, v in inputs.items() if k.startswith("query")}
        )
        inputs.pop("query_is_query", None)

        doc_outputs = model(
            **{k[4:]: v for k, v in inputs.items() if k.startswith("doc")}
        )

        additional_loss_kwargs = {}

        if self.do_gather:
            query_embeddings = query_outputs  # [B, Nq, D]
            if self.do_padding:
                doc_outputs, _ = pad_to_max_len_right(
                    doc_outputs, dist.get_world_size()
                )
            # print(
            #     f"rank: {dist.get_rank()} doc_embeddings before gather: {doc_outputs.shape}"
            # )
            doc_embeddings = gather_with_grad_torch(
                doc_outputs
            )  # [B * num_gpus, Nd, D]
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

        if "neg_doc_input_ids" in inputs:
            neg_doc_outputs = model(
                **{k[8:]: v for k, v in inputs.items() if k.startswith("neg_doc")}
            )

            neg_doc_embeddings = neg_doc_outputs
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
            query_outputs = model(
                **{k[6:]: v for k, v in inputs.items() if k.startswith("query")}
            )
            inputs.pop("query_is_query", None)
            # feed only kwargs with 'doc_' prefix
            doc_outputs = model(
                **{k[4:]: v for k, v in inputs.items() if k.startswith("doc")}
            )
            if "neg_doc_input_ids" in inputs:
                # stuck here! it seems neg_doc does not have padded images
                # shape_info = {
                #     k: v.shape for k, v in inputs.items() if k.startswith("neg_doc")
                # }
                # print(f"rank: {dist.get_rank()} all_shapes: {shape_info}")
                neg_doc_outputs = model(
                    **{k[8:]: v for k, v in inputs.items() if k.startswith("neg_doc")}
                )

                loss = self.loss_func(query_outputs, doc_outputs, neg_doc_outputs)

                # if dist.get_rank() == 0:
                #     breakpoint()
                # dist.barrier()

                if isinstance(loss, tuple):
                    loss, loss_stats = loss
                return loss, None, None

            loss = self.loss_func(query_outputs, doc_outputs)

            if isinstance(loss, tuple):
                loss, loss_stats = loss

            return loss, None, None
