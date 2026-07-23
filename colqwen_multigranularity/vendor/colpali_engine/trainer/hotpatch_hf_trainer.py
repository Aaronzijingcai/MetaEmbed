# hotpatch huggingface trainer using this PR, which correctly handles
# resume_from_checkpoint when using IterableDataset
# https://github.com/huggingface/transformers/pull/33544
# patches only `_inner_training_loop` with transformers 4.51.3

import contextlib
import copy
import functools
import importlib.metadata
import math, sys
import os, shutil
import time
import types
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

import transformers
from packaging import version
from torch import nn
from transformers import Trainer
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations.deepspeed import (
    deepspeed_init,
    deepspeed_load_checkpoint,
    is_deepspeed_available,
)
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.modeling_utils import (
    is_peft_available,
    load_sharded_checkpoint,
    PreTrainedModel,
    unwrap_model,
)
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    ExportableState,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import get_model_param_count
from transformers.trainer_utils import has_length, speed_metrics, TrainOutput
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    is_accelerate_available,
    is_sagemaker_mp_enabled,
    is_torch_xla_available,
    logging,
)

logger = logging.get_logger(__name__)

if is_peft_available():
    from peft import PeftModel

if is_accelerate_available():
    from accelerate import skip_first_batches
    from accelerate.utils import DistributedType

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCALER_NAME = "scaler.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"


def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False


def _activate_gradient_checkpointing(self, args):
    """Enable checkpointing without silently dropping PEFT gradients."""
    trainable_parameters = [
        (name, parameter)
        for name, parameter in self.model.named_parameters()
        if parameter.requires_grad
    ]
    trainable_groups = {
        "language_lora": [
            name
            for name, _ in trainable_parameters
            if ".language_model." in name and ".lora_" in name
        ],
        "visual_lora": [
            name
            for name, _ in trainable_parameters
            if ".visual." in name and ".lora_" in name
        ],
        "custom_text_proj": [
            name for name, _ in trainable_parameters if "custom_text_proj" in name
        ],
        "folder_homo": [
            name for name, _ in trainable_parameters if "folder_homo" in name
        ],
    }
    expected = {
        "tensors": int(os.environ.get("MURE_EXPECTED_TRAINABLE_TENSORS", "0")),
        "numel": int(os.environ.get("MURE_EXPECTED_TRAINABLE_NUMEL", "0")),
        "language_lora": int(
            os.environ.get("MURE_EXPECTED_LANGUAGE_LORA_TENSORS", "0")
        ),
        "visual_lora": int(
            os.environ.get("MURE_EXPECTED_VISUAL_LORA_TENSORS", "0")
        ),
        "custom_text_proj": int(
            os.environ.get("MURE_EXPECTED_CUSTOM_TEXT_PROJ_TENSORS", "0")
        ),
        "folder_homo": int(
            os.environ.get("MURE_EXPECTED_FOLDER_HOMO_TENSORS", "0")
        ),
    }
    actual_tensors = len(trainable_parameters)
    actual_numel = sum(parameter.numel() for _, parameter in trainable_parameters)
    if expected["tensors"] > 0 and actual_tensors != expected["tensors"]:
        raise RuntimeError(
            f"Trainable tensor count mismatch: expected={expected['tensors']}, actual={actual_tensors}"
        )
    if expected["numel"] > 0 and actual_numel != expected["numel"]:
        raise RuntimeError(
            f"Trainable parameter count mismatch: expected={expected['numel']}, actual={actual_numel}"
        )
    for group_name, names in trainable_groups.items():
        expected_count = expected[group_name]
        if expected_count > 0 and len(names) != expected_count:
            raise RuntimeError(
                f"Trainable {group_name} tensor count mismatch: "
                f"expected={expected_count}, actual={len(names)}"
            )
    if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
        logger.info(
            "Verified trainable set: tensors=%d numel=%d language_lora=%d "
            "visual_lora=%d custom_text_proj=%d folder_homo=%d",
            actual_tensors,
            actual_numel,
            len(trainable_groups["language_lora"]),
            len(trainable_groups["visual_lora"]),
            len(trainable_groups["custom_text_proj"]),
            len(trainable_groups["folder_homo"]),
        )

    if not args.gradient_checkpointing:
        return

    checkpoint_kwargs = args.gradient_checkpointing_kwargs
    self.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs=checkpoint_kwargs
    )

    use_reentrant = True
    if checkpoint_kwargs is not None:
        use_reentrant = checkpoint_kwargs.get("use_reentrant", True)
    if not use_reentrant or not _is_peft_model(self.model):
        return
    if getattr(self.model, "_mure_input_require_grads_enabled", False):
        return

    base_model = self.model.get_base_model()
    checkpoint_model = next(
        (
            module
            for module in base_model.modules()
            if isinstance(module, PreTrainedModel)
            and hasattr(module, "enable_input_require_grads")
        ),
        None,
    )
    if checkpoint_model is None:
        raise RuntimeError(
            "Reentrant gradient checkpointing with PEFT requires "
            "a PreTrainedModel with enable_input_require_grads()"
        )
    checkpoint_model.enable_input_require_grads()

    # ColQwen's custom inner_forward calls the cached `_embed_tokens` module
    # directly, and its vision tower starts from a frozen patch projection.
    # Hook both exact boundaries so checkpointed language and vision LoRA paths
    # receive gradients through their frozen inputs.
    def require_output_grad(_module, _inputs, output):
        if not torch.is_tensor(output):
            raise RuntimeError(
                "Checkpoint input module must return a tensor, got "
                f"{type(output).__name__}"
            )
        output.requires_grad_(True)

    token_embedding = getattr(checkpoint_model, "_embed_tokens", None)
    if token_embedding is None:
        raise RuntimeError(
            "Could not locate ColQwen's cached token embedding module for PEFT"
        )
    visual = getattr(checkpoint_model, "visual", None)
    visual_patch_embedding = getattr(visual, "patch_embed", None)
    if visual_patch_embedding is None:
        raise RuntimeError(
            "Could not locate ColQwen's visual patch embedding module for PEFT"
        )
    self.model._mure_input_require_grads_hooks = [
        token_embedding.register_forward_hook(require_output_grad),
        visual_patch_embedding.register_forward_hook(require_output_grad),
    ]

    vision_blocks = getattr(visual, "blocks", None)
    if vision_blocks is None or len(vision_blocks) == 0:
        raise RuntimeError("Could not locate ColQwen vision blocks for checkpointing")
    checkpointed_vision_blocks = 0
    for block in vision_blocks:
        if getattr(block, "_mure_reentrant_checkpoint_enabled", False):
            continue
        original_forward = block.forward

        def checkpointed_forward(
            module,
            hidden_states,
            *block_args,
            _original_forward=original_forward,
            **block_kwargs,
        ):
            if not module.training or not torch.is_grad_enabled():
                return _original_forward(hidden_states, *block_args, **block_kwargs)

            def run_block(checkpoint_hidden_states):
                return _original_forward(
                    checkpoint_hidden_states, *block_args, **block_kwargs
                )

            return checkpoint(
                run_block,
                hidden_states,
                use_reentrant=True,
                preserve_rng_state=True,
            )

        block.forward = types.MethodType(checkpointed_forward, block)
        block._mure_reentrant_checkpoint_enabled = True
        checkpointed_vision_blocks += 1

    self.model._mure_input_require_grads_enabled = True
    logger.info(
        "Enabled PEFT reentrant checkpointing on %s (%d direct input hooks, %d vision blocks)",
        checkpoint_model.__class__.__name__,
        2,
        checkpointed_vision_blocks,
    )


def _inner_training_loop(
    self,
    batch_size=None,
    args=None,
    resume_from_checkpoint=None,
    trial=None,
    ignore_keys_for_eval=None,
):
    if transformers.__version__ != "4.51.3":
        logger.warning(
            f"Patching transformers other than 4.51.3 leads to unanticipated effects. Use at your own risk!"
        )
    self.accelerator.free_memory()
    self._train_batch_size = batch_size
    if self.args.auto_find_batch_size:
        if self.state.train_batch_size != self._train_batch_size:
            from accelerate.utils import release_memory

            (self.model_wrapped,) = release_memory(self.model_wrapped)
            self.model_wrapped = self.model

            # Check for DeepSpeed *after* the initial pass and modify the config
            if self.is_deepspeed_enabled:
                # Temporarily unset `self.args.train_batch_size`
                original_bs = self.args.per_device_train_batch_size
                self.args.per_device_train_batch_size = self._train_batch_size // max(
                    1, self.args.n_gpu
                )
                self.propagate_args_to_deepspeed(True)
                self.args.per_device_train_batch_size = original_bs
        self.state.train_batch_size = self._train_batch_size
    logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
    # Data loader and number of training steps
    train_dataloader = self.get_train_dataloader()
    if self.is_fsdp_xla_v2_enabled:
        train_dataloader = tpu_spmd_dataloader(train_dataloader)

    # Setting up training control variables:
    # number of training epochs: num_train_epochs
    # number of training steps per epoch: num_update_steps_per_epoch
    # total number of training steps to execute: max_steps
    total_train_batch_size = (
        self._train_batch_size * args.gradient_accumulation_steps * args.world_size
    )
    (
        num_train_epochs,
        num_update_steps_per_epoch,
        num_examples,
        num_train_samples,
        epoch_based,
        len_dataloader,
        max_steps,
    ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

    num_train_tokens = None
    if self.args.include_tokens_per_second:
        num_train_tokens = self.num_tokens(
            train_dataloader, None if epoch_based else max_steps
        )
        # If going by epochs, multiply tokens linearly
        if len_dataloader is not None and epoch_based:
            num_train_tokens *= args.num_train_epochs
        # Otherwise since its steps, we just multiply by grad accum
        else:
            num_train_tokens *= args.gradient_accumulation_steps

    if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
        if self.args.n_gpu > 1:
            # nn.DataParallel(model) replicates the model, creating new variables and module
            # references registered here no longer work on other gpus, breaking the module
            raise ValueError(
                "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                " (torchrun or torch.distributed.launch (deprecated))."
            )
        else:
            debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

    delay_optimizer_creation = (
        is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled
    )

    # Can't delay optimizer creation when using FSDP2: https://github.com/huggingface/accelerate/blob/3f636d626063ffcf9a337c7d3624d61b7d187d59/src/accelerate/accelerator.py#L1404
    is_fsdp2 = self.is_fsdp_enabled and (
        getattr(self.accelerator.state.fsdp_plugin, "fsdp_version", 1) == 2
    )
    if is_fsdp2:
        delay_optimizer_creation = False

    # We need to reset the scheduler, as its parameters may be different on subsequent calls
    if self._created_lr_scheduler:
        self.lr_scheduler = None
        self._created_lr_scheduler = False

    if self.is_deepspeed_enabled:
        self.optimizer, self.lr_scheduler = deepspeed_init(
            self, num_training_steps=max_steps
        )

    if not delay_optimizer_creation:
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    self.state = TrainerState(
        stateful_callbacks=[
            cb
            for cb in self.callback_handler.callbacks + [self.control]
            if isinstance(cb, ExportableState)
        ]
    )
    self.state.is_hyper_param_search = trial is not None
    self.state.train_batch_size = self._train_batch_size

    # Compute absolute values for logging, eval, and save if given as ratio
    self.state.compute_steps(args, max_steps)

    # Activate gradient checkpointing if needed.
    _activate_gradient_checkpointing(self, args)

    model = self._wrap_model(self.model_wrapped)

    # as the model is wrapped, don't use `accelerator.prepare`
    # this is for unhandled cases such as
    # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
    use_accelerator_prepare = True if model is self.model else False

    if use_accelerator_prepare and self.is_fsdp_enabled:
        # In case of auto_find_batch_size=True
        # Remove FSDP wrapping from sub-models.
        self.model = unwrap_model(self.model, recursive=True)

    if delay_optimizer_creation:
        if use_accelerator_prepare:
            # configure fsdp plugin for qlora if any
            self._fsdp_qlora_plugin_updates()
            if self.accelerator.mixed_precision != "fp8":
                self.model = self.accelerator.prepare(self.model)
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    # prepare using `accelerator` prepare
    if use_accelerator_prepare:
        self.model.train()
        if hasattr(self.lr_scheduler, "step"):
            if self.use_apex:
                model = self.accelerator.prepare(self.model)
            else:
                model, self.optimizer = self.accelerator.prepare(
                    self.model, self.optimizer
                )
        else:
            # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
            model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.lr_scheduler
            )
    elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
        # In this case we are in DDP + LOMO, which should be supported
        self.optimizer = self.accelerator.prepare(self.optimizer)

    if self.is_fsdp_enabled:
        self.model = self.model_wrapped = model

    # for the rest of this function `model` is the outside model, whether it was wrapped or not
    if model is not self.model:
        self.model_wrapped = model

    # backward compatibility
    if self.is_deepspeed_enabled:
        self.deepspeed = self.model_wrapped

    # ckpt loading
    if resume_from_checkpoint is not None:
        if self.is_deepspeed_enabled:
            deepspeed_load_checkpoint(
                self.model_wrapped,
                resume_from_checkpoint,
                load_module_strict=not _is_peft_model(self.model),
            )
        elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
            self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

    # Check if saved optimizer or scheduler states exist
    self._load_optimizer_and_scheduler(resume_from_checkpoint)
    self._load_scaler(resume_from_checkpoint)

    # important: at this point:
    # self.model         is the Transformers Model
    # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
    # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {num_examples:,}")
    logger.info(f"  Num Epochs = {num_train_epochs:,}")
    logger.info(
        f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}"
    )
    if self.args.per_device_train_batch_size != self._train_batch_size:
        logger.info(
            f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}"
        )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_steps:,}")
    logger.info(
        f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}"
    )

    self.state.epoch = 0
    start_time = time.time()
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    steps_trained_progress_bar = None
    dataset_level_skip_batches = 0
    probe_data_start_step = int(os.environ.get("MURE_PROBE_DATA_START_STEP", "0"))

    # Check if continuing training from a checkpoint
    # FIX 1: correct behaviors for resuming from IterableDataset
    if resume_from_checkpoint is not None and os.path.isfile(
        os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
    ):
        self.state = TrainerState.load_from_json(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        )
        self.compare_trainer_and_checkpoint_args(self.args, self.state)
        self._load_callback_state()
        if probe_data_start_step > 0:
            epochs_trained = 0
            steps_trained_in_current_epoch = 0
            logger.info(
                "  Probe dataset is pre-positioned at data step %s; disabling Trainer resume data skip",
                probe_data_start_step,
            )
        elif num_update_steps_per_epoch is not None:
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (
                    num_update_steps_per_epoch
                )
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0
        else:
            # If the dataloader does not have a length, we cannot restore the number of trained epochs.
            # In the following loop, we repeatedly iterate over the dataloader to skip the first
            # `steps_trained_in_current_epoch` steps and increment `epochs_trained` accordingly.
            epochs_trained = 0
            steps_trained_in_current_epoch = (
                self.state.global_step * args.gradient_accumulation_steps
            )
            if args.ignore_data_skip:
                raise ValueError(
                    "The dataloader does not have a length, so it is impossible to restore the number of trained"
                    " epochs. Please disable the `ignore_data_skip` option."
                )

        logger.info(
            "  Continuing training from checkpoint, will skip to saved global_step"
        )
        if num_update_steps_per_epoch is not None:
            logger.info(f"  Continuing training from epoch {epochs_trained}")
        logger.info(f"  Continuing training from global step {self.state.global_step}")
        dataset_level_skip_batches = int(
            getattr(self, "_mure_dataset_level_resume_skip_batches", 0)
        )
        if dataset_level_skip_batches > 0:
            expected_skip_batches = (
                int(self.state.global_step) * int(args.gradient_accumulation_steps)
            )
            if dataset_level_skip_batches != expected_skip_batches:
                raise RuntimeError(
                    "Dataset-level resume skip mismatch: "
                    f"got {dataset_level_skip_batches} batches, expected {expected_skip_batches}."
                )
            steps_trained_in_current_epoch = 0
            logger.info(
                "  Dataset-level fast resume already skipped %d batches before collation; "
                "Trainer will not replay them.",
                dataset_level_skip_batches,
            )
        if not args.ignore_data_skip:
            logger.info(
                f"  Will skip the first {epochs_trained} epochs then the first"
                f" {steps_trained_in_current_epoch} batches in the first epoch."
            )
    # FIX 1 Complete

    # Update the references
    for attr in ("model", "optimizer", "lr_scheduler"):
        setattr(self.callback_handler, attr, getattr(self, attr))
    self.callback_handler.train_dataloader = train_dataloader

    self.state.init_training_references(self, max_steps, num_train_epochs, trial)

    # tr_loss is a tensor to avoid synchronization of TPUs through .item()
    tr_loss = torch.tensor(0.0, device=args.device)
    # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
    self._total_loss_scalar = 0.0
    self._globalstep_last_logged = self.state.global_step
    model.zero_grad()
    grad_norm: Optional[float] = None
    learning_rate = None
    self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

    if args.eval_on_start:
        self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

    for epoch in range(epochs_trained, num_train_epochs):
        epoch_dataloader = train_dataloader
        if hasattr(epoch_dataloader, "set_epoch"):
            epoch_dataloader.set_epoch(epoch)
        # FIX 2: correct behaviors for resuming from IterableDataset
        steps_skipped = (
            dataset_level_skip_batches if epoch == epochs_trained else 0
        )
        rng_to_sync = False
        epoch_iterator = None
        if steps_trained_in_current_epoch > 0 and num_update_steps_per_epoch is None:
            # Since the dataloader does not have a length, we just loop until the required number of steps.
            # Every time we reach the end of the dataloader, we increment epoch and reset the iterator.
            epoch_iterator = iter(epoch_dataloader)
            epoch_over = False
            while steps_trained_in_current_epoch > 0:
                try:
                    # If the dataloader yields N batches and N is not divisible by `args.gradient_accumulation_steps`,
                    # the update loop ignores the last `N % args.gradient_accumulation_steps` batches of an epoch.
                    # To replicate the same behavior when resuming training, we ignore such batches from skipped epochs.
                    for _ in range(args.gradient_accumulation_steps):
                        next(epoch_iterator)
                    steps_trained_in_current_epoch -= args.gradient_accumulation_steps
                    steps_skipped += args.gradient_accumulation_steps
                except StopIteration:
                    epoch_over = True
                    break
            if epoch_over:
                epochs_trained += 1
                continue
            assert steps_trained_in_current_epoch == 0
            rng_to_sync = True

        # Reset the past mems state at the beginning of each epoch if necessary.
        if args.past_index >= 0:
            self._past = None

        steps_in_epoch = (
            len(epoch_dataloader)
            if len_dataloader is not None
            else args.max_steps * args.gradient_accumulation_steps
        )
        self.control = self.callback_handler.on_epoch_begin(
            args, self.state, self.control
        )

        if (
            epoch == epochs_trained
            and resume_from_checkpoint is not None
            and steps_trained_in_current_epoch == 0
        ):
            if dataset_level_skip_batches > 0:
                # Creating a DataLoader iterator consumes torch RNG. Restore the
                # checkpoint RNG after iterator creation and before fetching the
                # first resumed batch so stochastic collators also stay aligned.
                rng_to_sync = True
            else:
                self._load_rng_state(resume_from_checkpoint)

        # rng_to_sync = False
        # steps_skipped = 0
        if steps_trained_in_current_epoch > 0:
            epoch_dataloader = skip_first_batches(
                epoch_dataloader, steps_trained_in_current_epoch
            )
            steps_skipped = steps_trained_in_current_epoch
            steps_trained_in_current_epoch = 0
            rng_to_sync = True

        step = -1
        # epoch_iterator = iter(epoch_dataloader)
        if epoch_iterator is None:
            epoch_iterator = iter(epoch_dataloader)
        if dataset_level_skip_batches > 0 and rng_to_sync:
            self._load_rng_state(resume_from_checkpoint)
            rng_to_sync = False
        # We chunkify the epoch iterator into gradient accumulation steps `n` batches
        remainder = num_examples % args.gradient_accumulation_steps
        if remainder == 0:
            remainder = args.gradient_accumulation_steps
        update_step = -1
        total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1
        if args.gradient_accumulation_steps == 1:
            total_updates -= 1
        for _ in range(total_updates):
            update_step += 1
            num_batches = (
                args.gradient_accumulation_steps
                if update_step != (total_updates - 1)
                else remainder
            )
            batch_samples, num_items_in_batch = self.get_batch_samples(
                epoch_iterator, num_batches, args.device
            )
            for i, inputs in enumerate(batch_samples):
                step += 1
                do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (
                    step + 1
                ) == steps_in_epoch
                # Since we perform prefetching, we need to manually set sync_gradients
                self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(
                        self.model, "main_input_name", "input_ids"
                    )
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        input_tokens = inputs[main_input_name].numel()
                        input_tokens = torch.tensor(
                            input_tokens, device=self.args.device, dtype=torch.int64
                        )
                        self.state.num_input_tokens_seen += (
                            self.accelerator.gather(input_tokens).sum().item()
                        )
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(
                        args, self.state, self.control
                    )

                # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                context = (
                    functools.partial(self.accelerator.no_sync, model=model)
                    if i != len(batch_samples) - 1
                    and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                    else contextlib.nullcontext
                )
                with context():
                    tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_xla_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss = tr_loss + tr_loss / (
                        1 + self.state.global_step - self._globalstep_last_logged
                    )
                else:
                    if tr_loss.device != tr_loss_step.device:
                        raise ValueError(
                            f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                        )
                    tr_loss = tr_loss + tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                if do_sync_step:
                    # Since we perform prefetching, we need to manually set sync_gradients to True
                    self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        if is_sagemaker_mp_enabled() and args.fp16:
                            _grad_norm = self.optimizer.clip_master_grads(
                                args.max_grad_norm
                            )
                        # AMP not supported in this code path
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            _grad_norm = nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            _grad_norm = self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                        if (
                            is_accelerate_available()
                            and self.accelerator.distributed_type
                            == DistributedType.DEEPSPEED
                        ):
                            grad_norm = model.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if hasattr(grad_norm, "item"):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = _grad_norm

                    self.control = self.callback_handler.on_pre_optimizer_step(
                        args, self.state, self.control
                    )

                    self.optimizer.step()

                    self.control = self.callback_handler.on_optimizer_step(
                        args, self.state, self.control
                    )

                    # get leaning rate before update
                    learning_rate = self._get_learning_rate()

                    if not self.accelerator.optimizer_step_was_skipped:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(
                            self.lr_scheduler,
                            torch.optim.lr_scheduler.ReduceLROnPlateau,
                        ):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = (
                        epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    )
                    self.control = self.callback_handler.on_step_end(
                        args, self.state, self.control
                    )
                    self._maybe_log_save_evaluate(
                        tr_loss,
                        grad_norm,
                        model,
                        trial,
                        epoch,
                        ignore_keys_for_eval,
                        start_time,
                        learning_rate=learning_rate,
                    )
                else:
                    self.control = self.callback_handler.on_substep_end(
                        args, self.state, self.control
                    )

                # PyTorch/XLA relies on the data loader to insert the mark_step for
                # each step. Since we are breaking the loop early, we need to manually
                # insert the mark_step here.
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            # We also need to break out of the nested loop
            if self.control.should_epoch_stop or self.control.should_training_stop:
                if is_torch_xla_available():
                    xm.mark_step()
                break
        # if step < 0:
        #     logger.warning(
        #         "There seems not to be a single sample in your epoch_iterator, stopping training at step"
        #         f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
        #         f" num_steps ({max_steps}) higher than the number of available samples."
        #     )
        #     self.control.should_training_stop = True

        self.control = self.callback_handler.on_epoch_end(
            args, self.state, self.control
        )
        self._maybe_log_save_evaluate(
            tr_loss,
            grad_norm,
            model,
            trial,
            epoch,
            ignore_keys_for_eval,
            start_time,
            learning_rate=learning_rate,
        )

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            if is_torch_xla_available():
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())
            else:
                logger.warning(
                    "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                    "configured. Check your training configuration if this is unexpected."
                )
        if self.control.should_training_stop:
            break

    if args.past_index and hasattr(self, "_past"):
        # Clean the state at the end of training
        delattr(self, "_past")

    logger.info(
        "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n"
    )
    if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
        # Wait for everyone to get here so we are sure the model has been saved by process 0.
        if is_torch_xla_available():
            xm.rendezvous("load_best_model_at_end")
        elif args.parallel_mode == ParallelMode.DISTRIBUTED:
            dist.barrier()
        elif is_sagemaker_mp_enabled():
            smp.barrier()

        self._load_best_model()

    # add remaining tr_loss
    self._total_loss_scalar += tr_loss.item()
    effective_global_step = max(
        self.state.global_step, 0.001
    )  # Avoid ZeroDivisionError
    train_loss = self._total_loss_scalar / effective_global_step

    metrics = speed_metrics(
        "train",
        start_time,
        num_samples=num_train_samples,
        num_steps=self.state.max_steps,
        num_tokens=num_train_tokens,
    )
    self.store_flos()
    metrics["total_flos"] = self.state.total_flos
    metrics["train_loss"] = train_loss

    self.is_in_train = False

    self._memory_tracker.stop_and_update_metrics(metrics)

    self.log(metrics)

    run_dir = self._get_output_dir(trial)
    checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

    # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
    if (
        self.args.should_save
        and self.state.best_model_checkpoint is not None
        and self.args.save_total_limit == 1
    ):
        for checkpoint in checkpoints_sorted:
            if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                logger.info(
                    f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit"
                )
                shutil.rmtree(checkpoint, ignore_errors=True)

    self.control = self.callback_handler.on_train_end(args, self.state, self.control)

    # Wait for the checkpoint to be uploaded.
    self._finish_current_push()

    # After training we make sure to retrieve back the original forward pass method
    # for the embedding layer by removing the forward post hook.
    if self.neftune_noise_alpha is not None:
        self._deactivate_neftune(self.model)

    return TrainOutput(self.state.global_step, train_loss, metrics)


def set_initial_training_values(
    self, args: TrainingArguments, dataloader, total_train_batch_size: int
):
    """
    Calculates and returns the following values:
    - `num_train_epochs`
    - `num_update_steps_per_epoch`
    - `num_examples`
    - `num_train_samples`
    - `epoch_based`
    - `len_dataloader`
    - `max_steps`
    """
    # Case 1: we rely on `args.max_steps` first
    max_steps = args.max_steps
    # If max_steps is negative, we use the number of epochs to determine the number of total steps later
    epoch_based = max_steps < 0
    len_dataloader = len(dataloader) if has_length(dataloader) else None

    # Case 2: We have a dataloader length and can extrapolate
    if len_dataloader is not None:
        num_update_steps_per_epoch = max(
            len_dataloader // args.gradient_accumulation_steps, 1
        )
        # Case 3: We have a length but are using epochs, we can extrapolate the number of steps
        if epoch_based:
            max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)

    # Now we figure out `num_examples`, `num_train_epochs`, and `train_samples`
    if len_dataloader:
        num_examples = self.num_examples(dataloader)
        if args.max_steps > 0:
            num_train_epochs = max_steps // num_update_steps_per_epoch + int(
                max_steps % num_update_steps_per_epoch > 0
            )
            # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
            # the best we can do.
            num_train_samples = max_steps * total_train_batch_size
        else:
            num_train_epochs = math.ceil(args.num_train_epochs)
            num_train_samples = self.num_examples(dataloader) * args.num_train_epochs
    elif (
        args.max_steps > 0
    ):  # Rely on max_steps when dataloader does not have a working size
        # Setting a very large number of epochs so we go as many times as necessary over the iterator.
        num_train_epochs = sys.maxsize
        # num_update_steps_per_epoch = max_steps
        num_update_steps_per_epoch = None
        num_examples = total_train_batch_size * args.max_steps
        num_train_samples = args.max_steps * total_train_batch_size
    else:
        raise ValueError(
            "args.max_steps must be set to a positive value if dataloader does not have a length, was"
            f" {args.max_steps}"
        )
    return (
        num_train_epochs,
        num_update_steps_per_epoch,
        num_examples,
        num_train_samples,
        epoch_based,
        len_dataloader,
        max_steps,
    )


def _inner_training_loop_4_55_0(
    self,
    batch_size=None,
    args=None,
    resume_from_checkpoint=None,
    trial=None,
    ignore_keys_for_eval=None,
):
    if transformers.__version__ != "4.55.0":
        logger.warning(
            f"Patching transformers other than 4.55.0 leads to unanticipated effects. Use at your own risk!"
        )
    self.accelerator.free_memory()
    self._train_batch_size = batch_size
    if self.args.auto_find_batch_size:
        if self.state.train_batch_size != self._train_batch_size:
            from accelerate.utils import release_memory

            (self.model_wrapped,) = release_memory(self.model_wrapped)
            self.model_wrapped = self.model

            # Check for DeepSpeed *after* the initial pass and modify the config
            if self.is_deepspeed_enabled:
                # Temporarily unset `self.args.train_batch_size`
                original_bs = self.args.per_device_train_batch_size
                self.args.per_device_train_batch_size = self._train_batch_size // max(
                    1, self.args.n_gpu
                )
                self.propagate_args_to_deepspeed(True)
                self.args.per_device_train_batch_size = original_bs
        self.state.train_batch_size = self._train_batch_size
    logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
    # Data loader and number of training steps
    train_dataloader = self.get_train_dataloader()
    if self.is_fsdp_xla_v2_enabled:
        train_dataloader = tpu_spmd_dataloader(train_dataloader)

    # Setting up training control variables:
    # number of training epochs: num_train_epochs
    # number of training steps per epoch: num_update_steps_per_epoch
    # total number of training steps to execute: max_steps
    total_train_batch_size = self.get_total_train_batch_size(args)

    (
        num_train_epochs,
        num_update_steps_per_epoch,
        num_examples,
        num_train_samples,
        epoch_based,
        len_dataloader,
        max_steps,
    ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

    num_train_tokens = None
    if self.args.include_tokens_per_second:
        num_train_tokens = self.num_tokens(
            train_dataloader, None if epoch_based else max_steps
        )
        # If going by epochs, multiply tokens linearly
        if len_dataloader is not None and epoch_based:
            num_train_tokens *= args.num_train_epochs
        # Otherwise since its steps, we just multiply by grad accum
        else:
            num_train_tokens *= args.gradient_accumulation_steps

    if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
        if self.args.n_gpu > 1:
            # nn.DataParallel(model) replicates the model, creating new variables and module
            # references registered here no longer work on other gpus, breaking the module
            raise ValueError(
                "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                " (torchrun or torch.distributed.launch (deprecated))."
            )
        else:
            debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

    delay_optimizer_creation = (
        is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled
    )

    # Can't delay optimizer creation when using FSDP2: https://github.com/huggingface/accelerate/blob/3f636d626063ffcf9a337c7d3624d61b7d187d59/src/accelerate/accelerator.py#L1404
    is_fsdp2 = self.is_fsdp_enabled and (
        getattr(self.accelerator.state.fsdp_plugin, "fsdp_version", 1) == 2
    )
    if is_fsdp2:
        delay_optimizer_creation = False

    # We need to reset the scheduler, as its parameters may be different on subsequent calls
    if self._created_lr_scheduler:
        self.lr_scheduler = None
        self._created_lr_scheduler = False

    if self.is_deepspeed_enabled:
        self.optimizer, self.lr_scheduler = deepspeed_init(
            self, num_training_steps=max_steps
        )

    if not delay_optimizer_creation:
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    self.state = TrainerState(
        stateful_callbacks=[
            cb
            for cb in self.callback_handler.callbacks + [self.control]
            if isinstance(cb, ExportableState)
        ]
    )
    self.state.is_hyper_param_search = trial is not None
    self.state.train_batch_size = self._train_batch_size

    # Compute absolute values for logging, eval, and save if given as ratio
    self.state.compute_steps(args, max_steps)

    # Activate gradient checkpointing if needed.
    _activate_gradient_checkpointing(self, args)

    model = self._wrap_model(self.model_wrapped)

    # as the model is wrapped, don't use `accelerator.prepare`
    # this is for unhandled cases such as
    # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
    use_accelerator_prepare = model is self.model

    if use_accelerator_prepare and self.is_fsdp_enabled:
        # In case of auto_find_batch_size=True
        # Remove FSDP wrapping from sub-models.
        self.model = unwrap_model(self.model, recursive=True)

    if delay_optimizer_creation:
        if use_accelerator_prepare:
            # configure fsdp plugin for qlora if any
            self._fsdp_qlora_plugin_updates()
            if self.accelerator.mixed_precision != "fp8":
                self.model = self.accelerator.prepare(self.model)
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    # prepare using `accelerator` prepare
    if use_accelerator_prepare:
        self.model.train()
        if hasattr(self.lr_scheduler, "step"):
            if self.use_apex:
                model = self.accelerator.prepare(self.model)
            else:
                # We should avoid accelerate preparing the model in TP case since we dont need it as it is handled by transformers from_pretrained and also it goes into DDP based preparation.
                if self.is_tp_enabled:
                    self.optimizer = self.accelerator.prepare(self.optimizer)
                else:
                    model, self.optimizer = self.accelerator.prepare(
                        self.model, self.optimizer
                    )
        else:
            # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
            model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.lr_scheduler
            )
    elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
        # In this case we are in DDP + LOMO, which should be supported
        self.optimizer = self.accelerator.prepare(self.optimizer)

    if self.is_fsdp_enabled:
        self.model = self.model_wrapped = model

    # for the rest of this function `model` is the outside model, whether it was wrapped or not
    if model is not self.model:
        self.model_wrapped = model

    # backward compatibility
    if self.is_deepspeed_enabled:
        self.deepspeed = self.model_wrapped

    # ckpt loading
    if resume_from_checkpoint is not None:
        if self.is_deepspeed_enabled:
            deepspeed_load_checkpoint(
                self.model_wrapped,
                resume_from_checkpoint,
                load_module_strict=not _is_peft_model(self.model),
            )
        elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
            self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

    # Check if saved optimizer or scheduler states exist
    self._load_optimizer_and_scheduler(resume_from_checkpoint)
    self._load_scaler(resume_from_checkpoint)

    # important: at this point:
    # self.model         is the Transformers Model
    # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
    # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {num_examples:,}")
    logger.info(f"  Num Epochs = {num_train_epochs:,}")
    logger.info(
        f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}"
    )
    if self.args.per_device_train_batch_size != self._train_batch_size:
        logger.info(
            f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}"
        )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_steps:,}")
    logger.info(
        f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}"
    )

    self.state.epoch = 0
    start_time = time.time()
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    steps_trained_progress_bar = None
    dataset_level_skip_batches = 0
    probe_data_start_step = int(os.environ.get("MURE_PROBE_DATA_START_STEP", "0"))

    # Check if continuing training from a checkpoint
    if resume_from_checkpoint is not None and os.path.isfile(
        os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
    ):
        self.state = TrainerState.load_from_json(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        )
        self.compare_trainer_and_checkpoint_args(self.args, self.state)
        self._load_callback_state()
        # epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
        # if not args.ignore_data_skip:
        #     steps_trained_in_current_epoch = self.state.global_step % (
        #         num_update_steps_per_epoch
        #     )
        #     steps_trained_in_current_epoch *= args.gradient_accumulation_steps
        # else:
        #     steps_trained_in_current_epoch = 0
        if probe_data_start_step > 0:
            epochs_trained = 0
            steps_trained_in_current_epoch = 0
            logger.info(
                "  Probe dataset is pre-positioned at data step %s; disabling Trainer resume data skip",
                probe_data_start_step,
            )
        elif num_update_steps_per_epoch is not None:
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (
                    num_update_steps_per_epoch
                )
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0
        else:
            # If the dataloader does not have a length, we cannot restore the number of trained epochs.
            # In the following loop, we repeatedly iterate over the dataloader to skip the first
            # `steps_trained_in_current_epoch` steps and increment `epochs_trained` accordingly.
            epochs_trained = 0
            steps_trained_in_current_epoch = (
                self.state.global_step * args.gradient_accumulation_steps
            )
            if args.ignore_data_skip:
                raise ValueError(
                    "The dataloader does not have a length, so it is impossible to restore the number of trained"
                    " epochs. Please disable the `ignore_data_skip` option."
                )

        logger.info(
            "  Continuing training from checkpoint, will skip to saved global_step"
        )
        if num_update_steps_per_epoch is not None:
            logger.info(f"  Continuing training from epoch {epochs_trained}")
        # logger.info(f"  Continuing training from epoch {epochs_trained}")
        logger.info(f"  Continuing training from global step {self.state.global_step}")
        dataset_level_skip_batches = int(
            getattr(self, "_mure_dataset_level_resume_skip_batches", 0)
        )
        if dataset_level_skip_batches > 0:
            expected_skip_batches = (
                int(self.state.global_step) * int(args.gradient_accumulation_steps)
            )
            if dataset_level_skip_batches != expected_skip_batches:
                raise RuntimeError(
                    "Dataset-level resume skip mismatch: "
                    f"got {dataset_level_skip_batches} batches, expected {expected_skip_batches}."
                )
            steps_trained_in_current_epoch = 0
            logger.info(
                "  Dataset-level fast resume already skipped %d batches before collation; "
                "Trainer will not replay them.",
                dataset_level_skip_batches,
            )
        if not args.ignore_data_skip:
            logger.info(
                f"  Will skip the first {epochs_trained} epochs then the first"
                f" {steps_trained_in_current_epoch} batches in the first epoch."
            )

    # Update the references
    for attr in ("model", "optimizer", "lr_scheduler"):
        setattr(self.callback_handler, attr, getattr(self, attr))
    self.callback_handler.train_dataloader = train_dataloader

    self.state.init_training_references(self, max_steps, num_train_epochs, trial)

    # tr_loss is a tensor to avoid synchronization of TPUs through .item()
    tr_loss = torch.tensor(0.0, device=args.device)
    # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
    self._total_loss_scalar = 0.0
    self._globalstep_last_logged = self.state.global_step
    model.zero_grad()
    grad_norm: Optional[float] = None
    learning_rate = None
    self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

    if args.eval_on_start:
        self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

    for epoch in range(epochs_trained, num_train_epochs):
        epoch_dataloader = train_dataloader
        if hasattr(epoch_dataloader, "set_epoch"):
            epoch_dataloader.set_epoch(epoch)

        steps_skipped = (
            dataset_level_skip_batches if epoch == epochs_trained else 0
        )
        rng_to_sync = False
        epoch_iterator = None
        if steps_trained_in_current_epoch > 0 and num_update_steps_per_epoch is None:
            # Since the dataloader does not have a length, we just loop until the required number of steps.
            # Every time we reach the end of the dataloader, we increment epoch and reset the iterator.
            epoch_iterator = iter(epoch_dataloader)
            epoch_over = False
            while steps_trained_in_current_epoch > 0:
                try:
                    # If the dataloader yields N batches and N is not divisible by `args.gradient_accumulation_steps`,
                    # the update loop ignores the last `N % args.gradient_accumulation_steps` batches of an epoch.
                    # To replicate the same behavior when resuming training, we ignore such batches from skipped epochs.
                    for _ in range(args.gradient_accumulation_steps):
                        next(epoch_iterator)
                    steps_trained_in_current_epoch -= args.gradient_accumulation_steps
                    steps_skipped += args.gradient_accumulation_steps
                except StopIteration:
                    epoch_over = True
                    break
            if epoch_over:
                epochs_trained += 1
                continue
            assert steps_trained_in_current_epoch == 0
            rng_to_sync = True

        # Reset the past mems state at the beginning of each epoch if necessary.
        if args.past_index >= 0:
            self._past = None

        steps_in_epoch = (
            len(epoch_dataloader)
            if len_dataloader is not None
            else args.max_steps * args.gradient_accumulation_steps
        )
        self.control = self.callback_handler.on_epoch_begin(
            args, self.state, self.control
        )

        if (
            epoch == epochs_trained
            and resume_from_checkpoint is not None
            and steps_trained_in_current_epoch == 0
        ):
            if dataset_level_skip_batches > 0:
                # Creating a DataLoader iterator consumes torch RNG. Restore the
                # checkpoint RNG after iterator creation and before fetching the
                # first resumed batch so stochastic collators also stay aligned.
                rng_to_sync = True
            else:
                self._load_rng_state(resume_from_checkpoint)

        # rng_to_sync = False
        # steps_skipped = 0
        if steps_trained_in_current_epoch > 0:
            epoch_dataloader = skip_first_batches(
                epoch_dataloader, steps_trained_in_current_epoch
            )
            steps_skipped = steps_trained_in_current_epoch
            steps_trained_in_current_epoch = 0
            rng_to_sync = True

        step = -1
        # epoch_iterator = iter(epoch_dataloader)
        if epoch_iterator is None:
            epoch_iterator = iter(epoch_dataloader)
        if dataset_level_skip_batches > 0 and rng_to_sync:
            self._load_rng_state(resume_from_checkpoint)
            rng_to_sync = False
        # We chunkify the epoch iterator into gradient accumulation steps `n` batches
        remainder = steps_in_epoch % args.gradient_accumulation_steps
        if remainder == 0:
            remainder = args.gradient_accumulation_steps
        update_step = -1
        total_updates = steps_in_epoch // args.gradient_accumulation_steps + int(
            remainder < args.gradient_accumulation_steps
        )
        for _ in range(total_updates):
            update_step += 1
            num_batches = (
                args.gradient_accumulation_steps
                if update_step != (total_updates - 1)
                else remainder
            )
            batch_samples, num_items_in_batch = self.get_batch_samples(
                epoch_iterator, num_batches, args.device
            )
            # Store the number of batches for current gradient accumulation
            # This is used to correctly scale the loss when the last accumulation step has fewer batches
            self.current_gradient_accumulation_steps = len(batch_samples)
            for i, inputs in enumerate(batch_samples):
                step += 1
                do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (
                    step + 1
                ) == steps_in_epoch
                # Since we perform prefetching, we need to manually set sync_gradients
                self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(
                        self.model, "main_input_name", "input_ids"
                    )
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        input_tokens = inputs[main_input_name].numel()
                        input_tokens = torch.tensor(
                            input_tokens, device=self.args.device, dtype=torch.int64
                        )
                        self.state.num_input_tokens_seen += (
                            self.accelerator.gather(input_tokens).sum().item()
                        )
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(
                        args, self.state, self.control
                    )

                # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                context = (
                    functools.partial(self.accelerator.no_sync, model=model)
                    if i != len(batch_samples) - 1
                    and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                    else contextlib.nullcontext
                )
                with context():
                    tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_xla_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss = tr_loss + tr_loss / (
                        1 + self.state.global_step - self._globalstep_last_logged
                    )
                else:
                    if tr_loss.device != tr_loss_step.device:
                        raise ValueError(
                            f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                        )
                    tr_loss = tr_loss + tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                if do_sync_step:
                    # Since we perform prefetching, we need to manually set sync_gradients to True
                    self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        if is_sagemaker_mp_enabled() and args.fp16:
                            _grad_norm = self.optimizer.clip_master_grads(
                                args.max_grad_norm
                            )
                        elif self.use_apex:
                            from apex import amp

                            # Revert to normal clipping otherwise, handling Apex or full precision
                            _grad_norm = nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            grad_norm_context = contextlib.nullcontext
                            if self.is_tp_enabled:
                                from torch.distributed._tensor.experimental import (
                                    implicit_replication,
                                )

                                grad_norm_context = implicit_replication
                            with grad_norm_context():
                                _grad_norm = self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                        if (
                            is_accelerate_available()
                            and self.accelerator.distributed_type
                            == DistributedType.DEEPSPEED
                        ):
                            grad_norm = model.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if hasattr(grad_norm, "item"):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = _grad_norm

                    self.control = self.callback_handler.on_pre_optimizer_step(
                        args, self.state, self.control
                    )

                    context = contextlib.nullcontext
                    if self.is_tp_enabled:
                        from torch.distributed._tensor.experimental import (
                            implicit_replication,
                        )

                        context = implicit_replication

                    with context():
                        self.optimizer.step()

                    self.control = self.callback_handler.on_optimizer_step(
                        args, self.state, self.control
                    )

                    # get leaning rate before update
                    learning_rate = self._get_learning_rate()

                    if not self.accelerator.optimizer_step_was_skipped:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(
                            self.lr_scheduler,
                            torch.optim.lr_scheduler.ReduceLROnPlateau,
                        ):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = (
                        epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    )
                    self.control = self.callback_handler.on_step_end(
                        args, self.state, self.control
                    )
                    self._maybe_log_save_evaluate(
                        tr_loss,
                        grad_norm,
                        model,
                        trial,
                        epoch,
                        ignore_keys_for_eval,
                        start_time,
                        learning_rate=learning_rate,
                    )
                else:
                    self.control = self.callback_handler.on_substep_end(
                        args, self.state, self.control
                    )

                # PyTorch/XLA relies on the data loader to insert the mark_step for
                # each step. Since we are breaking the loop early, we need to manually
                # insert the mark_step here.
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            # We also need to break out of the nested loop
            if self.control.should_epoch_stop or self.control.should_training_stop:
                if is_torch_xla_available():
                    xm.mark_step()
                break
        # if step < 0:
        #     logger.warning(
        #         "There seems not to be a single sample in your epoch_iterator, stopping training at step"
        #         f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
        #         f" num_steps ({max_steps}) higher than the number of available samples."
        #     )
        #     self.control.should_training_stop = True

        self.control = self.callback_handler.on_epoch_end(
            args, self.state, self.control
        )
        self._maybe_log_save_evaluate(
            tr_loss,
            grad_norm,
            model,
            trial,
            epoch,
            ignore_keys_for_eval,
            start_time,
            learning_rate=learning_rate,
        )

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            if is_torch_xla_available():
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())
            else:
                logger.warning(
                    "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                    "configured. Check your training configuration if this is unexpected."
                )
        if self.control.should_training_stop:
            break

    if args.past_index and hasattr(self, "_past"):
        # Clean the state at the end of training
        delattr(self, "_past")

    logger.info(
        "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n"
    )
    if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
        # Wait for everyone to get here so we are sure the model has been saved by process 0.
        if is_torch_xla_available():
            xm.rendezvous("load_best_model_at_end")
        elif args.parallel_mode == ParallelMode.DISTRIBUTED:
            dist.barrier()
        elif is_sagemaker_mp_enabled():
            smp.barrier()

        self._load_best_model()

    # add remaining tr_loss
    self._total_loss_scalar += tr_loss.item()
    effective_global_step = max(
        self.state.global_step, 0.001
    )  # Avoid ZeroDivisionError
    train_loss = self._total_loss_scalar / effective_global_step

    metrics = speed_metrics(
        "train",
        start_time,
        num_samples=num_train_samples,
        num_steps=self.state.max_steps,
        num_tokens=num_train_tokens,
    )
    self.store_flos()
    metrics["total_flos"] = self.state.total_flos
    metrics["train_loss"] = train_loss

    self.is_in_train = False

    self._memory_tracker.stop_and_update_metrics(metrics)

    self.log(metrics)

    run_dir = self._get_output_dir(trial)
    checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

    # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
    if (
        self.args.should_save
        and self.state.best_model_checkpoint is not None
        and self.args.save_total_limit == 1
    ):
        for checkpoint in checkpoints_sorted:
            if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                logger.info(
                    f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit"
                )
                shutil.rmtree(checkpoint, ignore_errors=True)

    self.control = self.callback_handler.on_train_end(args, self.state, self.control)

    # Wait for the checkpoint to be uploaded.
    self._finish_current_push()

    # After training we make sure to retrieve back the original forward pass method
    # for the embedding layer by removing the forward post hook.
    if self.neftune_noise_alpha is not None:
        self._deactivate_neftune(self.model)

    return TrainOutput(self.state.global_step, train_loss, metrics)


def set_initial_training_values_4_55_0(self, args, dataloader, total_train_batch_size):
    """
    Calculates and returns the following values:
    - `num_train_epochs`
    - `num_update_steps_per_epoch`
    - `num_examples`
    - `num_train_samples`
    - `epoch_based`
    - `len_dataloader`
    - `max_steps`
    """
    # Case 1: we rely on `args.max_steps` first
    max_steps = args.max_steps
    # If max_steps is negative, we use the number of epochs to determine the number of total steps later
    epoch_based = max_steps < 0
    len_dataloader = len(dataloader) if has_length(dataloader) else None

    # Case 2: We have a dataloader length and can extrapolate
    if len_dataloader is not None:
        num_update_steps_per_epoch = max(
            len_dataloader // args.gradient_accumulation_steps
            + int(len_dataloader % args.gradient_accumulation_steps > 0),
            1,
        )
        # Case 3: We have a length but are using epochs, we can extrapolate the number of steps
        if epoch_based:
            max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)

    # Now we figure out `num_examples`, `num_train_epochs`, and `train_samples`
    if len_dataloader:
        num_examples = self.num_examples(dataloader)
        if args.max_steps > 0:
            num_train_epochs = max_steps // num_update_steps_per_epoch + int(
                max_steps % num_update_steps_per_epoch > 0
            )
            # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
            # the best we can do.
            num_train_samples = max_steps * total_train_batch_size
        else:
            num_train_epochs = math.ceil(args.num_train_epochs)
            num_train_samples = self.num_examples(dataloader) * args.num_train_epochs
    elif (
        args.max_steps > 0
    ):  # Rely on max_steps when dataloader does not have a working size
        # Setting a very large number of epochs so we go as many times as necessary over the iterator.
        num_train_epochs = sys.maxsize
        # num_update_steps_per_epoch = max_steps
        num_update_steps_per_epoch = None
        num_examples = total_train_batch_size * args.max_steps
        num_train_samples = args.max_steps * total_train_batch_size
    else:
        raise ValueError(
            "args.max_steps must be set to a positive value if dataloader does not have a length, was"
            f" {args.max_steps}"
        )
    return (
        num_train_epochs,
        num_update_steps_per_epoch,
        num_examples,
        num_train_samples,
        epoch_based,
        len_dataloader,
        max_steps,
    )
