from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class SoftAssignmentSaveCallback(TrainerCallback):
    def _find_strategy1_softassign(self, model: Any):
        current = model
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            module = getattr(current, "strategy1_softassign", None)
            if module is not None:
                return module
            current = getattr(current, "base_model", None)
        return None

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        model = kwargs.get("model")
        strategy1_softassign = self._find_strategy1_softassign(model)
        if strategy1_softassign is None:
            return
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if checkpoint_dir.exists():
            strategy1_softassign.save_pretrained(checkpoint_dir)
