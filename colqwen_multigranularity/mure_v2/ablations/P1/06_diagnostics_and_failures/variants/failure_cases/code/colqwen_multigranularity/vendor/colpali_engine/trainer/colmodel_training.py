import json
import os

import types
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch

import torch.distributed as dist

from colpali_engine.collators import (
    CorpusQueryCollator,
    MultimodalRetrieverCollator,
    VisualRetrieverCollator,
)
from colpali_engine.loss.late_interaction_losses import NewColbertLoss
from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer
from colpali_engine.trainer.contrastive_trainer_v2 import ContrastiveTrainerV2
from colpali_engine.trainer.eval_utils import (
    BenchmarkEvalCallback,
    evaluate_dataset,
    filter_metrics,
    get_wandb_scalar_logs,
)
from colpali_engine.trainer.save_utils import CkptSaveCallback
from colpali_engine.utils.dist_utils import rank0_print
from colpali_engine.utils.gpu_stats import print_gpu_utilization, print_summary
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from colpali_engine.utils.trainer_utils import update_logs_with_losses
from peft import get_peft_model, LoraConfig, PeftModel

from transformers import PreTrainedModel, TrainingArguments
from transformers.integrations.integration_utils import (
    TensorBoardCallback,
    WandbCallback,
)
from transformers.trainer_utils import get_last_checkpoint


def rewrite_logs(d):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    for k, v in d.items():
        if k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        elif k.startswith(test_prefix):
            new_d["test/" + k[test_prefix_len:]] = v
        else:
            new_d["train/" + k] = v
    return new_d


def on_log_patch(self, args, state, control, model=None, logs=None, **kwargs):
    # patch the on_log function to keep step aligned with evaluation
    single_value_scalars = [
        "train_runtime",
        "train_samples_per_second",
        "train_steps_per_second",
        "train_loss",
        "total_flos",
    ]

    if self._wandb is None:
        return
    if not self._initialized:
        self.setup(args, state, model)
    if state.is_world_process_zero:
        logs = update_logs_with_losses(logs, state)
        for k, v in logs.items():
            if k in single_value_scalars:
                self._wandb.run.summary[k] = v
        non_scalar_logs = {
            k: v for k, v in logs.items() if k not in single_value_scalars
        }
        non_scalar_logs = rewrite_logs(non_scalar_logs)
        self._wandb.log(
            {**non_scalar_logs, "train/global_step": state.global_step},
            step=state.global_step,  # patched `step`
        )


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    rank0_print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


# define different part of model
# model.__class__.__name__ -> module names
MODULE_KEYWORDS = {
    "ColPali": {  # manual check
        "vision_encoder": ["model.vision_tower"],
        "vision_projector": ["model.multi_modal_projector"],
        "llm": ["model.language_model"],
        "prompt_projector": ["custom_text_proj"],
    },
    "LastPali": {  # manual check
        "vision_encoder": ["model.vision_tower"],
        "vision_projector": ["model.multi_modal_projector"],
        "llm": ["model.language_model"],
        "prompt_projector": ["custom_text_proj", "prompt_embed_tokens"],
        "custom_text_proj": ["custom_text_proj"],
        "prompt_embed_tokens": ["prompt_embed_tokens"],
    },
    "ColQwen2": {  # passed
        "vision_encoder": [
            "visual.patch_embed",
            "visual.rotary_pos_emb",
            "visual.blocks",
        ],
        "vision_projector": ["visual.merger"],
        "llm": ["model"],
        "prompt_projector": ["custom_text_proj"],
    },
    "ColQwen2_5": {  # passed
        "vision_encoder": [
            "visual.patch_embed",
            "visual.rotary_pos_emb",
            "visual.blocks",
        ],
        "vision_projector": ["visual.merger"],
        "llm": ["model"],
        "prompt_projector": ["custom_text_proj"],
    },
    "LastQwen2_5": {
        "vision_encoder": [
            "visual.patch_embed",
            "visual.rotary_pos_emb",
            "visual.blocks",
        ],
        "vision_projector": ["visual.merger"],
        "llm": ["model"],
        "prompt_projector": ["custom_text_proj", "prompt_embed_tokens"],
        "custom_text_proj": ["custom_text_proj"],
        "prompt_embed_tokens": ["prompt_embed_tokens"],
    },
    "LastQwen3": {
        "vision_encoder": [
            "visual.patch_embed",
            "visual.rotary_pos_emb",
            "visual.blocks",
        ],
        "vision_projector": [
            "visual.merger"
        ],  # stage 1 training -- full trainable parameters -- 2e
        "llm": ["model"],  # stage 2 training -- lora
        "prompt_projector": ["custom_text_proj", "prompt_embed_tokens"],
        "custom_text_proj": [
            "custom_text_proj"
        ],  # stage 2 training -- full trainable parameters
        "prompt_embed_tokens": [
            "prompt_embed_tokens"
        ],  # stage 2 training -- full trainable parameters
    },
    "LastLlama3Vision": {
        "vision_encoder": ["vision_model"],
        "vision_projector": ["multi_modal_projector"],
        "llm": ["language_model"],
        "prompt_projector": ["custom_text_proj", "prompt_embed_tokens"],
        "custom_text_proj": ["custom_text_proj"],
        "prompt_embed_tokens": ["prompt_embed_tokens"],
    },
}


@dataclass
class ColModelTrainingConfig:
    model: Union[PreTrainedModel, PeftModel]
    processor: BaseVisualRetrieverProcessor
    tr_args: Optional[TrainingArguments] = None
    output_dir: Optional[str] = None
    max_length: int = 1024  # applies to query, only text part
    doc_max_length: int = 1024  # applies to doc, only text part

    run_eval_before_train: Optional[bool] = False
    run_eval: bool = True
    run_train: bool = True
    # added distributed in-train eval from PR: https://github.com/illuin-tech/colpali/pull/195/files
    vidore_eval_frequency: int = -1  # controls both v1 & v2 & mmeb
    eval_dataset_format: str = "qa"  # controls only v1

    peft_config: Optional[LoraConfig] = None
    loss_func: Optional[Callable] = NewColbertLoss()
    dataset_loading_func: Optional[Callable] = None
    dataset_loading_cls: Optional[Callable] = (
        None  # same usage as dataset_loading_func but support passing in params
    )

    vidore_eval_batch_size: Optional[int] = 4
    eval_dataset_loader: Optional[Dict[str, Callable]] = None  # vidore v1
    eval_dataset_loader_v2: Optional[Dict[str, Callable]] = None  # vidore v2
    eval_dataset_loader_mmeb: Optional[Dict[str, Callable]] = None  # mmeb
    eval_dataset_loader_vira: Optional[Dict[str, Callable]] = None  # VIRA eval
    eval_vira_tgt_modalities: Optional[str] = None
    # VIRA eval tgt modalities -- will be overrided by self.config.dataset_loading_cls if not provided

    no_eval_keywords: Optional[List[str]] = (
        None  # since 08/01, eval will include multilingual data
    )
    # field(
    #     default_factory=lambda: ["multilingual"]
    # )

    pretrained_peft_model_name_or_path: Optional[str] = None
    do_gather: Optional[bool] = False

    # added wandb logging
    wandb_project: Optional[str] = "huggingface"
    wandb_run_name: Optional[str] = None

    # freeze llm and visual encoder option
    freeze_llm: Optional[bool] = False
    freeze_visual_encoder: Optional[bool] = False
    freeze_visual_projector: Optional[bool] = False
    freeze_prompt_projector: Optional[bool] = False

    # add unfreeze_modules to allow full training of prompt projector (DEPRECIATED: use peft_config)
    unfreeze_modules: Optional[List[str]] = None

    # in-train eval parameters
    eval_num_workers: Optional[int] = 4

    # added multimodal collator
    use_mm_collator: Optional[bool] = False
    use_v2_trainer: Optional[bool] = False
    # 0730: add `use_v2_retriever` for accurate evaluation
    use_v2_retriever: Optional[bool] = False
    num_negative: Optional[int] = None
    query_instruction_path: Optional[str] = None  # needed for mbeir dataset

    # force padding side -- test if paligamma works with left padding
    force_padding_side: Optional[str] = None

    # for 32b model, do not save full model when saving checkpoint as it will trigger OOM
    save_full_state_dict: Optional[bool] = True

    # DO NOT TOUCH -- post_init will overwrite this
    is_last_model: Optional[bool] = None
    is_fsdp_enabled: Optional[bool] = None
    pooling_type: Optional[str] = None
    doc_pooling_type: Optional[str] = None
    do_padding: Optional[bool] = (
        False  # True when using split-XX pooler, controls both training & eval
    )
    v2_do_padding: Optional[bool] = (
        False  # True when using split-XX pooler, controls eval with v2 retriever
    )
    eval_mrl_groups: Optional[List[Tuple[int, int]]] = (
        None  # provide to override training-time MRL groups
    )

    """
    Config class used for training a ColVision model.
    """

    def __post_init__(self):
        """
        Initialize the model and tokenizer if not provided
        """
        if self.output_dir is None:
            sanitized_name = str(self.model.name_or_path).replace("/", "_")
            self.output_dir = f"./models/{sanitized_name}"

        if self.tr_args is None:
            rank0_print("No training arguments provided. Using default.")
            self.tr_args = TrainingArguments(output_dir=self.output_dir)
        elif (
            self.tr_args.output_dir is None
            or self.tr_args.output_dir == "trainer_output"
        ):
            self.tr_args.output_dir = self.output_dir

        # overwrite logging_dir to output_dir -- save tensorboard logs in the same folder
        if self.tr_args.logging_dir is None:
            self.tr_args.logging_dir = self.tr_args.output_dir

        if isinstance(self.tr_args.learning_rate, str):
            rank0_print("Casting learning rate to float")
            self.tr_args.learning_rate = float(self.tr_args.learning_rate)

        self.tr_args.remove_unused_columns = False

        if self.pretrained_peft_model_name_or_path is not None:
            rank0_print("Loading pretrained PEFT model")
            self.model.load_adapter(
                self.pretrained_peft_model_name_or_path, is_trainable=True
            )

        self.model_class_name = self.model.__class__.__name__

        self.is_last_model = self.model_class_name.startswith("Last")

        if self.freeze_llm:
            rank0_print("Freezing LLM part, parameters including: ")
            llm_keys = MODULE_KEYWORDS[self.model_class_name]["llm"]
            for key in llm_keys:
                rank0_print(key)
                eval(f"self.model.{key}").requires_grad_(False)

        if self.freeze_visual_encoder:
            rank0_print("Freezing visual encoder part, parameters including: ")
            visual_encoder_keys = MODULE_KEYWORDS[self.model_class_name][
                "vision_encoder"
            ]
            for key in visual_encoder_keys:
                rank0_print(key)
                eval(f"self.model.{key}").requires_grad_(False)

        if self.freeze_visual_projector:
            rank0_print("Freezing visual projector part, parameters including: ")
            visual_projector_keys = MODULE_KEYWORDS[self.model_class_name][
                "vision_projector"
            ]
            for key in visual_projector_keys:
                rank0_print(key)
                eval(f"self.model.{key}").requires_grad_(False)

        if self.freeze_prompt_projector:
            rank0_print("Freezing prompt projector part, parameters including: ")
            prompt_projector_keys = MODULE_KEYWORDS[self.model_class_name][
                "prompt_projector"
            ]
            for key in prompt_projector_keys:
                rank0_print(key)
                try:
                    eval(f"self.model.{key}").requires_grad_(False)
                except:
                    rank0_print(f"Module {key} not found. Skipping.")

        if self.peft_config is not None:
            # in LoRA training, peft takes over the freeze/unfreeze via `modules_to_save`
            assert (
                not self.freeze_llm
            ), "Cannot freeze LLM and use PEFT at the same time"
            assert (
                not self.freeze_visual_encoder
            ), "Cannot freeze visual encoder and use PEFT at the same time"
            assert (
                not self.freeze_visual_projector
            ), "Cannot freeze visual projector and use PEFT at the same time"
            assert (
                not self.freeze_prompt_projector
            ), "Cannot freeze prompt projector and use PEFT at the same time"
            rank0_print("Configurating PEFT model")
            if self.pretrained_peft_model_name_or_path is None:
                self.model = get_peft_model(self.model, self.peft_config)
            else:
                rank0_print(
                    f"Adapter already loaded from {self.pretrained_peft_model_name_or_path}. Not overwriting."
                )

        print_trainable_parameters(self.model)

        os.environ["WANDB_PROJECT"] = self.wandb_project
        os.environ["WANDB_RUN_ID"] = os.path.basename(self.output_dir)

        # this could be a resume run -- so do not override with timestamp
        if self.wandb_run_name is None:
            self.wandb_run_name = os.path.basename(self.output_dir)
            # self.wandb_run_name = (
            #     os.path.basename(self.output_dir)
            #     + "_"
            #     + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # )

        self.tr_args.run_name = self.wandb_run_name

        if self.num_negative is not None and self.num_negative > 0:
            rank0_print(f"Using mined negatives of {self.num_negative} per query")
            assert (
                "Negative" in self.loss_func.__class__.__name__
            ), f"Loss {self.loss_func.__class__.__name__} does not support mined negatives"

        # if MRL groups is enabled at training time, we also test them at eval time
        if (
            self.eval_mrl_groups is None
            and hasattr(self.loss_func, "mrl_groups")
            and self.loss_func.mrl_groups is not None
        ):
            assert all(
                len(g) == 3 for g in self.loss_func.mrl_groups
            ), f"Invalid MRL groups: {self.loss_func.mrl_groups}"
            self.eval_mrl_groups = [[g[0], g[1]] for g in self.loss_func.mrl_groups]

        # Disable gradient checkpoint when FSDP is detected -- it will be taken over by FSDP
        if "ACCELERATE_USE_FSDP" in os.environ and os.environ[
            "ACCELERATE_USE_FSDP"
        ] in ["true", "1", "True"]:
            self.is_fsdp_enabled = True
            if self.tr_args.gradient_checkpointing:
                rank0_print(
                    "FSDP is enabled. Disabling gradient checkpointing in tr_args."
                )
                self.tr_args.gradient_checkpointing = False

        # Update eval_vira_tgt_modalities
        if (
            self.eval_vira_tgt_modalities is None
            and self.dataset_loading_cls is not None
            and hasattr(self.dataset_loading_cls, "tgt_modalities")
            and self.dataset_loading_cls.tgt_modalities is not None
        ):
            rank0_print(
                f"Overriding eval_vira_tgt_modalities to {self.dataset_loading_cls.tgt_modalities}."
            )
            self.eval_vira_tgt_modalities = self.dataset_loading_cls.tgt_modalities

        # Record pooling_type for later use
        try:
            self.pooling_type = self.model.pooler.pooling_type
            self.doc_pooling_type = self.model.doc_pooler.pooling_type
        except:
            self.pooling_type = None
            self.doc_pooling_type = None

        # update do_padding - 只在 do_padding 为默认值 False 且 doc_pooling_type 是 split 时才更新
        # 如果用户显式设置了 do_padding，则保持用户设置的值
        original_do_padding = self.do_padding
        if self.doc_pooling_type is not None and self.doc_pooling_type.startswith("split"):
            if original_do_padding is False:
                self.do_padding = True

        # v2_do_padding 跟随 do_padding 的值
        if self.v2_do_padding is False:
            self.v2_do_padding = self.do_padding


class ColModelTraining:
    """
    Class that contains the training and evaluation logic for a ColVision model.
    """

    def __init__(self, config: ColModelTrainingConfig, config_file: str) -> None:
        self.config = config
        self.config_file = config_file

        self.model = self.config.model
        # self.current_git_hash = os.popen("git rev-parse HEAD").read().strip()
        if self.config.dataset_loading_func is not None:
            assert (
                self.config.dataset_loading_cls is None
            ), f"Only one of {dataset_loading_func} and {dataset_loading_cls} can be provided"
            self.dataset = self.config.dataset_loading_func()
        else:
            dataset_cls = (
                self.config.dataset_loading_cls
            )  # an initalized instance was provided -- use __call__ to get dataset
            self.dataset = dataset_cls()  # call this instance to get dataset tuple

        if self.config.use_mm_collator:
            rank0_print("Using MultimodalCollator.")
            corpus_format = self.dataset[2]
            document_dataset = self.dataset[1]
            self.dataset = self.dataset[0]  # query dataset
            self.collator = MultimodalRetrieverCollator(
                processor=self.config.processor,
                max_length=self.config.max_length,
                doc_max_length=self.config.doc_max_length,
                document_dataset=document_dataset,
                num_negative=self.config.num_negative,
                corpus_format=corpus_format,
                query_instructions=self.config.query_instruction_path,
            )

        elif isinstance(self.dataset, Tuple):
            rank0_print(
                "Dataset has BEIR/hard negatives format. Using CorpusQueryCollator."
            )
            # keep compability with hard negatives format
            assert (
                self.config.num_negative == 1
            ), f"num_negative={self.config.num_negative} is not supported"
            corpus_format = self.dataset[2]
            neg_dataset = self.dataset[1]
            self.dataset = self.dataset[0]
            self.collator = CorpusQueryCollator(
                processor=self.config.processor,
                max_length=self.config.max_length,
                image_dataset=neg_dataset,
                mined_negatives=self.config.num_negative is not None,
                corpus_format=corpus_format,
            )
        else:
            rank0_print("Dataset has QA format. Using VisualRetrieverCollator.")
            self.collator = VisualRetrieverCollator(
                processor=self.config.processor,
                max_length=self.config.max_length,
                force_padding_side=self.config.force_padding_side,
            )

    def get_tb_writer(self, trainer):
        callback_list = trainer.callback_handler.callbacks
        tb_callback = None
        for callback in callback_list:
            if isinstance(callback, TensorBoardCallback):
                tb_callback = callback
                break
        if tb_callback is not None:
            # init summary writer in advance so the eval callback can use it
            # clean up the log dir if it exists
            if trainer.accelerator.is_main_process:
                if os.path.exists(trainer.args.logging_dir):
                    import shutil

                    shutil.rmtree(trainer.args.logging_dir)

                tb_callback._init_summary_writer(trainer.args)

        tb_writer = None
        if tb_callback is not None:
            tb_writer = tb_callback.tb_writer
        else:
            rank0_print("TensorBoardCallback not found when adding eval callback...")
        return tb_writer

    def get_wandb_writer(self, trainer, at_init=True):
        callback_list = trainer.callback_handler.callbacks
        wandb_callback = None
        for callback in callback_list:
            if isinstance(callback, WandbCallback):
                wandb_callback = callback
                # hack the on_log function to keep step aligned
                if at_init:
                    wandb_callback.on_log = types.MethodType(
                        on_log_patch, wandb_callback
                    )
                break
        if wandb_callback is not None:
            if at_init:
                wandb_callback.setup(trainer.args, trainer.state, trainer.model)

            return (
                wandb_callback._wandb
            )  # it's just wandb object -- we keep it returned just for the sake of consistency
        else:
            rank0_print("WandbCallback not found when adding eval callback...")
            return None

    def init_trainer(self):
        """
        Initialize the trainer but hold for training.
        """
        if self.config.num_negative is not None and self.config.num_negative > 0:
            rank0_print(f"Training with {self.config.num_negative} hard negatives")
        else:
            rank0_print("Training with in-batch negatives")

        if self.config.use_v2_trainer:
            rank0_print("Using trainerV2 -- Make sure you interleaved the dataset!")
            assert (
                self.config.dataset_loading_cls is not None
                and hasattr(self.config.dataset_loading_cls, "interleaved_batch_size")
                and self.config.dataset_loading_cls.interleaved_batch_size > 0
            ), f"dataset_loading_cls {self.config.dataset_loading_cls} does not have interleaved_batch_size"

            trainer = ContrastiveTrainerV2(
                model=self.model,
                train_dataset=self.dataset["train"],
                eval_dataset=self.dataset["test"],
                args=self.config.tr_args,
                data_collator=self.collator,
                loss_func=self.config.loss_func,
                do_gather=self.config.do_gather,  # multi-GPU gather -- has to work with `New` loss
                use_is_query=self.config.is_last_model,  # LastXXX model need to flags is_query
                do_padding=self.config.do_padding,
            )
        else:
            trainer = ContrastiveTrainer(
                model=self.model,
                train_dataset=self.dataset["train"],
                eval_dataset=self.dataset["test"],
                args=self.config.tr_args,
                data_collator=self.collator,
                loss_func=self.config.loss_func,
                do_gather=self.config.do_gather,  # multi-GPU gather -- has to work with `New` loss
                use_is_query=self.config.is_last_model,  # LastXXX model need to flags is_query
                do_padding=self.config.do_padding,
            )

        trainer.args.remove_unused_columns = False
        tb_writer = self.get_tb_writer(trainer)
        wandb_writer = self.get_wandb_writer(trainer)

        if self.config.processor is not None and self.config.vidore_eval_frequency > 0:
            # vidore v1 eval callback -- v1 does not need no_eval_keywords
            if self.config.eval_dataset_loader is not None:
                trainer.add_callback(
                    BenchmarkEvalCallback(
                        processor=self.config.processor,
                        model=self.model,
                        eval_dataset_loader=self.config.eval_dataset_loader,
                        trainer=trainer,
                        batch_query=self.config.vidore_eval_batch_size,
                        batch_passage=self.config.vidore_eval_batch_size,
                        batch_score=16,
                        run_frequency=self.config.vidore_eval_frequency,
                        dataset_format=self.config.eval_dataset_format,
                        tb_writer=tb_writer,
                        wandb_writer=wandb_writer,
                        eval_num_workers=self.config.eval_num_workers,
                        mrl_groups=self.config.eval_mrl_groups,
                        dataset_name="v1",
                        is_last_model=self.config.is_last_model,
                        use_v2_retriever=self.config.use_v2_retriever,
                        v2_do_padding=self.config.do_padding,
                    )
                )

            if self.config.eval_dataset_loader_v2 is not None:
                trainer.add_callback(
                    BenchmarkEvalCallback(
                        processor=self.config.processor,
                        model=self.model,
                        eval_dataset_loader=self.config.eval_dataset_loader_v2,
                        trainer=trainer,
                        batch_query=self.config.vidore_eval_batch_size,
                        batch_passage=self.config.vidore_eval_batch_size,
                        batch_score=16,
                        run_frequency=self.config.vidore_eval_frequency,
                        dataset_format="beir",
                        tb_writer=tb_writer,
                        wandb_writer=wandb_writer,
                        eval_num_workers=self.config.eval_num_workers,
                        no_eval_keywords=self.config.no_eval_keywords,
                        mrl_groups=self.config.eval_mrl_groups,
                        dataset_name="v2",
                        is_last_model=self.config.is_last_model,
                        use_v2_retriever=self.config.use_v2_retriever,
                        v2_do_padding=self.config.do_padding,
                    )
                )

            if self.config.eval_dataset_loader_mmeb is not None:
                trainer.add_callback(
                    BenchmarkEvalCallback(
                        processor=self.config.processor,
                        model=self.model,
                        eval_dataset_loader=self.config.eval_dataset_loader_mmeb,
                        trainer=trainer,
                        batch_query=self.config.vidore_eval_batch_size,
                        batch_passage=self.config.vidore_eval_batch_size,
                        batch_score=16,
                        run_frequency=self.config.vidore_eval_frequency,
                        dataset_format="mmeb",
                        tb_writer=tb_writer,
                        wandb_writer=wandb_writer,
                        eval_num_workers=self.config.eval_num_workers,
                        # no_eval_keywords=self.config.no_eval_keywords,
                        mrl_groups=self.config.eval_mrl_groups,
                        dataset_name="mmeb",
                        is_last_model=self.config.is_last_model,
                        use_v2_retriever=self.config.use_v2_retriever,
                        v2_do_padding=self.config.do_padding,
                    )
                )

            if self.config.eval_dataset_loader_vira is not None:
                trainer.add_callback(
                    BenchmarkEvalCallback(
                        processor=self.config.processor,
                        model=self.model,
                        eval_dataset_loader=self.config.eval_dataset_loader_vira,
                        trainer=trainer,
                        batch_query=self.config.vidore_eval_batch_size,
                        batch_passage=self.config.vidore_eval_batch_size,
                        batch_score=16,
                        run_frequency=self.config.vidore_eval_frequency,
                        dataset_format="mbeir",
                        tb_writer=tb_writer,
                        wandb_writer=wandb_writer,
                        eval_num_workers=self.config.eval_num_workers,
                        # no_eval_keywords=self.config.no_eval_keywords,
                        mrl_groups=self.config.eval_mrl_groups,
                        dataset_name="vira",
                        tgt_modalities=self.config.eval_vira_tgt_modalities,
                        is_last_model=self.config.is_last_model,
                        use_v2_retriever=self.config.use_v2_retriever,
                        v2_do_padding=self.config.do_padding,
                    )
                )

        # For FSDPv2 Full Dict save -- add on_save callback with FULL_STATE_DICT method
        if self.config.is_fsdp_enabled and self.config.save_full_state_dict:
            trainer.add_callback(
                CkptSaveCallback(
                    output_dir=self.config.output_dir,
                    is_fsdp_enabled=self.config.is_fsdp_enabled,
                    accelerator=trainer.accelerator,
                    config_file=self.config_file,
                )
            )

        self.trainer = trainer

    def train(self) -> None:
        # detect if we have a non-empty checkpoint path: if so, load it
        # to recover from preempted jobs
        if self.config.tr_args.resume_from_checkpoint:
            checkpoint_dir = (
                self.config.tr_args.resume_from_checkpoint
                if isinstance(self.config.tr_args.resume_from_checkpoint, str)
                else self.config.tr_args.output_dir
            )
            checkpoint_path = None
            if os.path.exists(checkpoint_dir):
                # If user points to an explicit checkpoint directory (e.g. .../checkpoint-800),
                # use it directly. Otherwise, treat it as an output dir and pick the latest.
                trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")
                if os.path.isfile(trainer_state_path):
                    checkpoint_path = checkpoint_dir
                else:
                    checkpoint_path = get_last_checkpoint(checkpoint_dir)
            if checkpoint_path is None:
                rank0_print(
                    f"Checkpoint path {checkpoint_dir} not found. Starting from scratch."
                )
                self.config.tr_args.resume_from_checkpoint = False

        result = self.trainer.train(
            resume_from_checkpoint=self.config.tr_args.resume_from_checkpoint
        )
        print_summary(result)

    def eval(self) -> None:
        # DO NOT CALL THIS BEFORE TRAIN...
        # THIS only handles evaluation at the end of training
        # for evaluation during training, use the BenchmarkEvalCallback
        unwrapped_model = self.model
        all_metrics = {}
        all_metrics_v2 = {}
        all_metrics_mmeb = {}

        if self.config.eval_dataset_loader is not None:
            # Create a vision retriever with the current model checkpoint.
            eval_dataset_format = getattr(self.config, "eval_dataset_format", "beir")
            for (
                test_name,
                test_dataset_loading_func,
            ) in self.config.eval_dataset_loader.items():
                all_metrics[test_name] = evaluate_dataset(
                    model=unwrapped_model,
                    processor=self.config.processor,
                    dataset=test_dataset_loading_func(),
                    format=eval_dataset_format,
                    batch_query=self.config.vidore_eval_batch_size,
                    batch_passage=self.config.vidore_eval_batch_size,
                    batch_score=16,
                    num_workers=self.config.eval_num_workers,
                    mrl_groups=self.config.eval_mrl_groups,
                    is_last_model=self.config.is_last_model,
                    use_v2_retriever=self.config.use_v2_retriever,
                    v2_do_padding=self.config.v2_do_padding,
                )
                all_metrics[test_name] = filter_metrics(all_metrics[test_name])
                # {
                #     k: v
                #     for k, v in all_metrics[test_name].items()
                #     if should_track_metric(k)
                # }
                rank0_print(f"Metrics for {test_name}: {all_metrics[test_name]}")

        if self.config.eval_dataset_loader_v2 is not None:
            for (
                test_name,
                test_dataset_loading_func,
            ) in self.config.eval_dataset_loader_v2.items():
                no_eval_keywords = self.config.no_eval_keywords or []
                if any(
                    keyword in test_name for keyword in no_eval_keywords
                ):
                    continue
                all_metrics_v2[test_name] = evaluate_dataset(
                    model=unwrapped_model,
                    processor=self.config.processor,
                    dataset=test_dataset_loading_func(),
                    format="beir",
                    batch_query=self.config.vidore_eval_batch_size,
                    batch_passage=self.config.vidore_eval_batch_size,
                    batch_score=16,
                    num_workers=self.config.eval_num_workers,
                    mrl_groups=self.config.eval_mrl_groups,
                    is_last_model=self.config.is_last_model,
                    use_v2_retriever=self.config.use_v2_retriever,
                    v2_do_padding=self.config.v2_do_padding,
                )
                all_metrics_v2[test_name] = filter_metrics(all_metrics_v2[test_name])
                # {
                #     k: v
                #     for k, v in all_metrics_v2[test_name].items()
                #     if should_track_metric(k)
                # }
                rank0_print(f"Metrics for {test_name}: {all_metrics_v2[test_name]}")

        if self.config.eval_dataset_loader_mmeb is not None:
            for (
                test_name,
                test_dataset_loading_func,
            ) in self.config.eval_dataset_loader_mmeb.items():
                all_metrics_mmeb[test_name] = evaluate_dataset(
                    model=unwrapped_model,
                    processor=self.config.processor,
                    dataset=test_dataset_loading_func(),
                    format="mmeb",
                    batch_query=self.config.vidore_eval_batch_size,
                    batch_passage=self.config.vidore_eval_batch_size,
                    batch_score=16,
                    num_workers=self.config.eval_num_workers,
                    mrl_groups=self.config.eval_mrl_groups,
                    is_last_model=self.config.is_last_model,
                    use_v2_retriever=self.config.use_v2_retriever,
                    v2_do_padding=self.config.v2_do_padding,
                )
                all_metrics_mmeb[test_name] = filter_metrics(
                    all_metrics_mmeb[test_name]
                )
                # {
                #     k: v
                #     for k, v in all_metrics_mmeb[test_name].items()
                #     if should_track_metric(k)
                # }
                rank0_print(f"Metrics for {test_name}: {all_metrics_mmeb[test_name]}")

        if all_metrics:
            avg_metric_dict = defaultdict(list)
            for k, v in all_metrics.items():
                for metric_name, metric_value in v.items():
                    avg_metric_dict[metric_name].append(metric_value)
            for metric_name, metric_values in avg_metric_dict.items():
                all_metrics[f"avg_{metric_name}"] = sum(metric_values) / len(
                    metric_values
                )

            rank0_print(
                f"Finished v1 evaluation on {self.config.output_dir}: {all_metrics}"
            )
            if self.trainer.accelerator.is_main_process:
                with open(f"{self.config.output_dir}/vidore_v1_dist.json", "w") as f:
                    json.dump(all_metrics, f, indent=4)
                wandb_writer = self.get_wandb_writer(self.trainer, at_init=False)
                if wandb_writer is not None:
                    # write to wandb if possible
                    scalar_logs = get_wandb_scalar_logs(all_metrics, "v1")
                    wandb_writer.log(scalar_logs, step=self.trainer.state.global_step)

        if all_metrics_v2:
            avg_metric_dict = defaultdict(list)
            for k, v in all_metrics_v2.items():
                for metric_name, metric_value in v.items():
                    avg_metric_dict[metric_name].append(metric_value)

            for metric_name, metric_values in avg_metric_dict.items():
                all_metrics_v2[f"avg_{metric_name}"] = sum(metric_values) / len(
                    metric_values
                )

            rank0_print(
                f"Finished v2 evaluation on {self.config.output_dir}: {all_metrics_v2}"
            )
            if self.trainer.accelerator.is_main_process:
                with open(f"{self.config.output_dir}/vidore_v2_dist.json", "w") as f:
                    json.dump(all_metrics_v2, f, indent=4)
                wandb_writer = self.get_wandb_writer(self.trainer, at_init=False)
                if wandb_writer is not None:
                    scalar_logs = get_wandb_scalar_logs(all_metrics_v2, "v2")
                    wandb_writer.log(scalar_logs, step=self.trainer.state.global_step)

        if all_metrics_mmeb:
            avg_metric_dict = defaultdict(list)
            for k, v in all_metrics_mmeb.items():
                for metric_name, metric_value in v.items():
                    avg_metric_dict[metric_name].append(metric_value)

            for metric_name, metric_values in avg_metric_dict.items():
                all_metrics_mmeb[f"avg_{metric_name}"] = sum(metric_values) / len(
                    metric_values
                )

            rank0_print(
                f"Finished mmeb evaluation on {self.config.output_dir}: {all_metrics_mmeb}"
            )
            if self.trainer.accelerator.is_main_process:
                with open(f"{self.config.output_dir}/mmeb_dist.json", "w") as f:
                    json.dump(all_metrics_mmeb, f, indent=4)

                wandb_writer = self.get_wandb_writer(self.trainer, at_init=False)
                if wandb_writer is not None:
                    scalar_logs = get_wandb_scalar_logs(all_metrics_mmeb, "mmeb")
                    wandb_writer.log(scalar_logs, step=self.trainer.state.global_step)

    def save(self):
        """
        Save the model with its training config, as well as the tokenizer and processor if provided.
        """
        if self.trainer.is_fsdp_enabled:
            unwrapped_model = self.trainer.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(
                self.config.output_dir,
                is_main_process=self.trainer.accelerator.is_main_process,
                save_function=self.trainer.accelerator.save,
                state_dict=self.trainer.accelerator.get_state_dict(self.model),
            )
        else:
            try:
                self.model.save_pretrained(self.config.output_dir)
            except RuntimeError as exc:
                if "shared tensors" not in str(exc):
                    raise
                rank0_print(
                    "save_pretrained hit shared tensors; retrying with "
                    "safe_serialization=False."
                )
                self.model.save_pretrained(
                    self.config.output_dir, safe_serialization=False
                )

        self.config.processor.save_pretrained(self.config.output_dir)
