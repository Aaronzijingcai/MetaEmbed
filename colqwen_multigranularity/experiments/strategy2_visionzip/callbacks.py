from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class VisionZipSaveCallback(TrainerCallback):
    def _find_strategy2_visionzip(self, model: Any):
        stack = [model]
        seen = set()
        while stack:
            current = stack.pop()
            if current is None or id(current) in seen:
                continue
            seen.add(id(current))
            module = getattr(current, "strategy2_visionzip", None)
            if module is not None:
                return module
            stack.extend(
                getattr(current, name, None)
                for name in ("base_model", "model", "module")
                if getattr(current, name, None) is not None
            )
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
        strategy2_visionzip = self._find_strategy2_visionzip(model)
        if strategy2_visionzip is None:
            return
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if checkpoint_dir.exists():
            strategy2_visionzip.save_pretrained(checkpoint_dir)

