from __future__ import annotations

import json
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SoftAssignmentConfig:
    enabled: bool = True
    budgets: Tuple[int, int, int] = (64, 64, 128)
    keep_ratio: Optional[float] = None
    compress_stages: str = "all"
    temperature: float = 0.1
    learnable_temperature: bool = False
    normalize_inputs: bool = True
    normalize_prototypes: bool = True
    preserve_input_rms: bool = True
    share_query_doc_prototypes: bool = True
    stage_count: int = 3
    eps: float = 1e-6
    debug_shapes: bool = False

    def __post_init__(self) -> None:
        if len(self.budgets) != self.stage_count:
            raise ValueError(f"budgets must contain {self.stage_count} values, got {self.budgets}.")
        if self.stage_count != 3:
            raise ValueError("SoftAssignment MVP expects exactly three stages.")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")
        if self.keep_ratio is not None and not (0.0 < float(self.keep_ratio) <= 1.0):
            raise ValueError("keep_ratio must be in (0, 1] when set.")

    def active_stage_ids(self) -> Tuple[int, ...]:
        mode = str(self.compress_stages).lower()
        if (not self.enabled) or mode in {"none", "off", "false"}:
            return ()
        if mode == "g1":
            return (0,)
        if mode == "g2":
            return (1,)
        if mode == "g3":
            return (2,)
        if mode in {"g2g3", "g2+g3"}:
            return (1, 2)
        if mode in {"all", "g1g2g3", "g1+g2+g3"}:
            return (0, 1, 2)
        raise ValueError(f"Unknown compress_stages={self.compress_stages!r}.")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["budgets"] = list(self.budgets)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "SoftAssignmentConfig":
        payload = dict(data)
        if "budgets" in payload:
            payload["budgets"] = tuple(payload["budgets"])
        return cls(**payload)

    def save_pretrained(self, save_dir: str | Path) -> None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        with (save_path / "strategy1_softassign_config.json").open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2, sort_keys=True)
            file.write("\n")

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "SoftAssignmentConfig":
        config_path = Path(path)
        if config_path.is_dir():
            preferred = config_path / "strategy1_softassign_config.json"
            legacy = config_path / "soft_assignment_config.json"
            config_path = preferred if preferred.exists() else legacy
        with config_path.open("r", encoding="utf-8") as file:
            return cls.from_dict(json.load(file))


class StageSoftAssignmentCompressor(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        budget: int,
        keep_ratio: Optional[float],
        temperature: float,
        learnable_temperature: bool,
        normalize_inputs: bool,
        normalize_prototypes: bool,
        preserve_input_rms: bool,
        eps: float,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.budget = int(budget)
        self.keep_ratio = None if keep_ratio is None else float(keep_ratio)
        self.normalize_inputs = bool(normalize_inputs)
        self.normalize_prototypes = bool(normalize_prototypes)
        self.preserve_input_rms = bool(preserve_input_rms)
        self.eps = float(eps)
        self.prototypes = nn.Parameter(torch.empty(self.budget, self.hidden_size))
        log_tau = torch.log(torch.tensor(float(temperature)))
        if learnable_temperature:
            self.log_temperature = nn.Parameter(log_tau)
        else:
            self.register_buffer("log_temperature", log_tau)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.prototypes, mean=0.0, std=self.hidden_size**-0.5)

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp_min(self.eps)

    def _effective_budget(self, token_count: int) -> int:
        if self.budget <= 0 or token_count <= 0:
            return 0
        if self.keep_ratio is None:
            return min(self.budget, token_count)
        ratio_budget = max(1, int(math.ceil(float(token_count) * self.keep_ratio)))
        return min(self.budget, token_count, ratio_budget)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError(f"Expected stage tokens [N, H], got {tuple(tokens.shape)}.")
        if tokens.shape[0] == 0:
            return tokens.new_zeros((0, self.hidden_size))
        if self.budget <= 0:
            return tokens.new_zeros((0, self.hidden_size))
        effective_budget = self._effective_budget(int(tokens.shape[0]))
        if effective_budget <= 0:
            return tokens.new_zeros((0, self.hidden_size))

        x = F.normalize(tokens, dim=-1) if self.normalize_inputs else tokens
        p = self.prototypes[:effective_budget].to(dtype=tokens.dtype, device=tokens.device)
        p = F.normalize(p, dim=-1) if self.normalize_prototypes else p
        logits = x.matmul(p.transpose(0, 1)) / self.temperature.to(dtype=tokens.dtype, device=tokens.device)
        assignment = torch.softmax(logits, dim=-1)
        mass = assignment.sum(dim=0).clamp_min(self.eps).to(dtype=tokens.dtype).unsqueeze(-1)
        compressed = assignment.transpose(0, 1).to(dtype=tokens.dtype).matmul(tokens) / mass

        if self.preserve_input_rms:
            input_rms = tokens.pow(2).mean().sqrt().detach().clamp_min(self.eps)
            output_rms = compressed.pow(2).mean().sqrt().detach().clamp_min(self.eps)
            compressed = compressed * (input_rms / output_rms)
        return compressed


class SoftAssignmentCompressor(nn.Module):
    def __init__(
        self,
        config: SoftAssignmentConfig,
        *,
        hidden_size: int,
        crop_counts: Sequence[int],
        spatial_merge_size: int = 2,
    ) -> None:
        super().__init__()
        if len(crop_counts) != config.stage_count:
            raise ValueError(f"crop_counts must contain {config.stage_count} values, got {crop_counts}.")
        self.config = config
        self.hidden_size = int(hidden_size)
        self.crop_counts = tuple(int(v) for v in crop_counts)
        self.total_crop_count = int(sum(self.crop_counts))
        self.spatial_merge_size = int(spatial_merge_size)
        self.spatial_merge_area = self.spatial_merge_size * self.spatial_merge_size
        self.stages = nn.ModuleList(
            [
                StageSoftAssignmentCompressor(
                    hidden_size=self.hidden_size,
                    budget=int(budget),
                    keep_ratio=config.keep_ratio,
                    temperature=config.temperature,
                    learnable_temperature=config.learnable_temperature,
                    normalize_inputs=config.normalize_inputs,
                    normalize_prototypes=config.normalize_prototypes,
                    preserve_input_rms=config.preserve_input_rms,
                    eps=config.eps,
                )
                for budget in config.budgets
            ]
        )

    def split_by_stage(
        self,
        image_embeds: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if image_grid_thw.ndim != 2 or image_grid_thw.shape[-1] != 3:
            raise ValueError(f"Expected image_grid_thw [C, 3], got {tuple(image_grid_thw.shape)}.")
        if image_grid_thw.shape[0] != self.total_crop_count:
            raise ValueError(f"Expected {self.total_crop_count} crops for one row, got {image_grid_thw.shape[0]}.")
        crop_patch_lengths = image_grid_thw.prod(dim=1).to(device=image_embeds.device, dtype=torch.long)
        crop_lengths = torch.div(crop_patch_lengths, self.spatial_merge_area, rounding_mode="floor")
        if int(crop_lengths.sum().item()) != image_embeds.shape[0]:
            raise ValueError(
                f"merged image_grid_thw token count {int(crop_lengths.sum().item())} "
                f"(patch tokens={int(crop_patch_lengths.sum().item())}, "
                f"spatial_merge_size={self.spatial_merge_size}) does not match "
                f"image_embeds rows {image_embeds.shape[0]}."
            )
        stage_tokens: List[torch.Tensor] = []
        crop_start = 0
        token_start = 0
        for crop_count in self.crop_counts:
            crop_end = crop_start + crop_count
            token_count = int(crop_lengths[crop_start:crop_end].sum().item())
            stage_tokens.append(image_embeds[token_start : token_start + token_count])
            crop_start = crop_end
            token_start += token_count
        return stage_tokens[0], stage_tokens[1], stage_tokens[2]

    def forward(
        self,
        image_embeds: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        active = set(self.config.active_stage_ids())
        stage_tokens = self.split_by_stage(image_embeds, image_grid_thw)
        compressed = []
        for stage_index, tokens in enumerate(stage_tokens):
            if stage_index in active:
                compressed.append(self.stages[stage_index](tokens))
            else:
                compressed.append(tokens)
        if self.config.debug_shapes and (not torch.jit.is_scripting()):
            raw = [int(tokens.shape[0]) for tokens in stage_tokens]
            comp = [int(tokens.shape[0]) for tokens in compressed]
            print(f"[SoftAssignment] raw_stage_tokens={raw} compact_stage_tokens={comp}", flush=True)
        return compressed[0], compressed[1], compressed[2]

    def save_pretrained(self, save_dir: str | Path) -> None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(save_path)
        torch.save(self.state_dict(), save_path / "strategy1_softassign.bin")

    def load_pretrained_weights(self, path: str | Path, *, map_location: str | torch.device | None = None) -> None:
        weight_path = Path(path)
        if weight_path.is_dir():
            preferred = weight_path / "strategy1_softassign.bin"
            legacy = weight_path / "soft_assignment.bin"
            weight_path = preferred if preferred.exists() else legacy
        state = torch.load(weight_path, map_location=map_location)
        self.load_state_dict(state)


def coerce_budgets(values: Iterable[int]) -> Tuple[int, int, int]:
    budgets = tuple(int(value) for value in values)
    if len(budgets) != 3:
        raise ValueError(f"Expected exactly three budgets, got {budgets}.")
    return budgets
