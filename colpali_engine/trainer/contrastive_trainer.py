import inspect

import random

import torch
import torch.distributed as dist
from colpali_engine.utils.dist_utils import gather_with_grad_torch, rank0_print
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
            doc_embeddings = gather_with_grad_torch(
                doc_outputs
            )  # [B * num_gpus, Nd, D]
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
                # input_ids=inputs["query_input_ids"],
                # attention_mask=inputs["query_attention_mask"],
                **{k[6:]: v for k, v in inputs.items() if k.startswith("query")}
            )
            inputs.pop("query_is_query", None)
            # feed only kwargs with 'doc_' prefix
            doc_outputs = model(
                **{k[4:]: v for k, v in inputs.items() if k.startswith("doc")}
            )
            if "neg_doc_input_ids" in inputs:
                neg_doc_outputs = model(
                    **{k[8:]: v for k, v in inputs.items() if k.startswith("neg_doc")}
                )
                loss = self.loss_func(query_outputs, doc_outputs, neg_doc_outputs)

                if isinstance(loss, tuple):
                    loss, loss_stats = loss
                return loss, None, None

            loss = self.loss_func(query_outputs, doc_outputs)

            if isinstance(loss, tuple):
                loss, loss_stats = loss

            return loss, None, None
