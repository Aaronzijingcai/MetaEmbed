import os

from colpali_engine.utils.dist_utils import rank0_print

from transformers import TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_callback import TrainerCallback

PREFIX_CHECKPOINT_DIR = "checkpoint"


# always save a full FSDP checkpoint for the ease of evaluation
# at the cost of aggregating at rank 0
class CkptSaveCallback(TrainerCallback):
    def __init__(self, output_dir, is_fsdp_enabled, accelerator, config_file):
        super().__init__()
        self.output_dir = output_dir
        self.is_fsdp_enabled = is_fsdp_enabled
        self.accelerator = accelerator
        self.config_file = config_file

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not self.is_fsdp_enabled:
            return
        model = kwargs["model"]
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        checkpoint_path = os.path.join(self.output_dir, checkpoint_folder)
        # save full adapter
        rank0_print(f"Saving full adapter to {checkpoint_path}")
        unwrapped_model = self.accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            checkpoint_path,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
            state_dict=self.accelerator.get_state_dict(model),
        )
        # save meta data for easy eval
        if self.accelerator.is_main_process:
            os.system(f"cp {self.config_file} {checkpoint_path}/training_config.yml")

        rank0_print(f"Saved full adapter to {checkpoint_path}")
