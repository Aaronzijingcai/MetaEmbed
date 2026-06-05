from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class VisionZipConfig:
    enabled: bool = True
    budgets: Tuple[int, int, int] = (64, 128, 256)
    keep_ratio: Optional[float] = None
    keep_ratios: Optional[Tuple[float, float, float]] = None
    compress_stages: str = "all"
    compression_scope: str = "crop"
    crop_budget_mode: str = "proportional"
    dominant_ratio: float = 0.75
    attention_source: str = "self_similarity"
    visual_attn_layer: int = -2
    target_select: str = "uniform"
    merge_metric: str = "cosine"
    preserve_input_rms: bool = True
    stage_count: int = 3
    eps: float = 1e-6
    random_seed: int = 0
    debug_shapes: bool = False

    def __post_init__(self) -> None:
        if len(self.budgets) != self.stage_count:
            raise ValueError(f"budgets must contain {self.stage_count} values, got {self.budgets}.")
        if self.stage_count != 3:
            raise ValueError("VisionZip MVP expects exactly three stages.")
        if not (0.0 < float(self.dominant_ratio) <= 1.0):
            raise ValueError("dominant_ratio must be in (0, 1].")
        if self.keep_ratio is not None and not (0.0 < float(self.keep_ratio) <= 1.0):
            raise ValueError("keep_ratio must be in (0, 1] when set.")
        if self.keep_ratios is not None:
            if len(self.keep_ratios) != self.stage_count:
                raise ValueError(f"keep_ratios must contain {self.stage_count} values, got {self.keep_ratios}.")
            for ratio in self.keep_ratios:
                if not (0.0 < float(ratio) <= 1.0):
                    raise ValueError("each keep_ratios value must be in (0, 1].")
        if str(self.compression_scope).lower() not in {"crop", "stage"}:
            raise ValueError("compression_scope must be one of {'crop', 'stage'}.")
        if str(self.crop_budget_mode).lower() not in {"proportional", "uniform", "min1"}:
            raise ValueError("crop_budget_mode must be one of {'proportional', 'uniform', 'min1'}.")
        if str(self.attention_source).lower() not in {"auto", "visual_attn", "self_similarity", "token_norm"}:
            raise ValueError("attention_source must be one of {'auto', 'visual_attn', 'self_similarity', 'token_norm'}.")
        if str(self.target_select).lower() not in {"uniform", "random", "saliency"}:
            raise ValueError("target_select must be one of {'uniform', 'random', 'saliency'}.")
        if str(self.merge_metric).lower() not in {"cosine", "dot"}:
            raise ValueError("merge_metric must be one of {'cosine', 'dot'}.")

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

    def keep_ratio_for_stage(self, stage_index: int) -> Optional[float]:
        if self.keep_ratios is not None:
            return float(self.keep_ratios[int(stage_index)])
        if self.keep_ratio is None:
            return None
        return float(self.keep_ratio)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["budgets"] = list(self.budgets)
        if self.keep_ratios is not None:
            data["keep_ratios"] = list(self.keep_ratios)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "VisionZipConfig":
        payload = dict(data)
        if "budgets" in payload:
            payload["budgets"] = tuple(int(value) for value in payload["budgets"])
        if "keep_ratios" in payload and payload["keep_ratios"] is not None:
            payload["keep_ratios"] = tuple(float(value) for value in payload["keep_ratios"])
        return cls(**payload)

    def save_pretrained(self, save_dir: str | Path) -> None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        with (save_path / "strategy2_visionzip_config.json").open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2, sort_keys=True)
            file.write("\n")

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "VisionZipConfig":
        config_path = Path(path)
        if config_path.is_dir():
            config_path = config_path / "strategy2_visionzip_config.json"
        with config_path.open("r", encoding="utf-8") as file:
            return cls.from_dict(json.load(file))


def coerce_budgets(values: Iterable[int]) -> Tuple[int, int, int]:
    budgets = tuple(int(value) for value in values)
    if len(budgets) != 3:
        raise ValueError(f"Expected exactly three budgets, got {budgets}.")
    return budgets


def compact_stage_lengths(original: torch.Tensor, config: VisionZipConfig) -> torch.Tensor:
    budgets = torch.tensor(config.budgets, dtype=torch.long, device=original.device).unsqueeze(0)
    budget_cap = torch.minimum(budgets.expand_as(original), original)
    if config.keep_ratios is None and config.keep_ratio is None:
        return budget_cap
    ratios = (
        torch.tensor(config.keep_ratios, dtype=torch.float32, device=original.device).unsqueeze(0)
        if config.keep_ratios is not None
        else torch.full((1, original.shape[-1]), float(config.keep_ratio), dtype=torch.float32, device=original.device)
    )
    ratio_cap = torch.ceil(original.to(torch.float32) * ratios).to(torch.long).clamp_min(1)
    ratio_cap = torch.where(original.gt(0), ratio_cap, torch.zeros_like(ratio_cap))
    return torch.minimum(budget_cap, ratio_cap)


class StageVisionZipCompressor(nn.Module):
    def __init__(self, config: VisionZipConfig, *, hidden_size: int) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = int(hidden_size)

    def _partition_budget(self, budget: int, token_count: int) -> Tuple[int, int]:
        budget = min(int(budget), int(token_count))
        if budget <= 0:
            return 0, 0
        if budget == 1:
            return 1, 0
        dominant = max(1, int(round(float(budget) * float(self.config.dominant_ratio))))
        dominant = min(dominant, budget - 1)
        return dominant, budget - dominant

    def _compute_saliency(self, tokens: torch.Tensor, saliency: Optional[torch.Tensor]) -> torch.Tensor:
        source = str(self.config.attention_source).lower()
        if saliency is not None:
            if saliency.ndim != 1 or saliency.shape[0] != tokens.shape[0]:
                raise ValueError(f"Expected saliency [{tokens.shape[0]}], got {tuple(saliency.shape)}.")
            return saliency.to(device=tokens.device, dtype=torch.float32)
        if source == "visual_attn":
            raise ValueError("attention_source='visual_attn' requires explicit visual attention saliency.")
        if source in {"auto", "self_similarity"}:
            x = F.normalize(tokens.to(dtype=torch.float32), dim=-1)
            return x.matmul(x.transpose(0, 1)).mean(dim=0)
        if source == "token_norm":
            return tokens.to(dtype=torch.float32).norm(dim=-1)
        raise ValueError(f"Unknown attention_source={self.config.attention_source!r}.")

    def _compute_keys(self, tokens: torch.Tensor, keys: Optional[torch.Tensor]) -> torch.Tensor:
        if keys is not None:
            if keys.shape != tokens.shape:
                raise ValueError(f"Expected keys {tuple(tokens.shape)}, got {tuple(keys.shape)}.")
            key_tokens = keys
        else:
            key_tokens = tokens
        if str(self.config.merge_metric).lower() == "cosine":
            return F.normalize(key_tokens.to(dtype=torch.float32), dim=-1)
        return key_tokens.to(dtype=torch.float32)

    @staticmethod
    def _select_uniform_indices(length: int, count: int, device: torch.device) -> torch.LongTensor:
        if count <= 0 or length <= 0:
            return torch.empty(0, dtype=torch.long, device=device)
        if count >= length:
            return torch.arange(length, device=device, dtype=torch.long)
        positions = torch.linspace(0, length - 1, steps=count, device=device)
        return positions.round().to(dtype=torch.long).unique(sorted=True)[:count]

    def _select_target_indices(
        self,
        *,
        residual_count: int,
        contextual_budget: int,
        residual_saliency: torch.Tensor,
        device: torch.device,
    ) -> torch.LongTensor:
        mode = str(self.config.target_select).lower()
        if contextual_budget >= residual_count:
            return torch.arange(residual_count, device=device, dtype=torch.long)
        if mode == "uniform":
            target = self._select_uniform_indices(residual_count, contextual_budget, device)
            if target.numel() == contextual_budget:
                return target
            mask = torch.ones(residual_count, dtype=torch.bool, device=device)
            mask[target] = False
            extra = mask.nonzero(as_tuple=False).squeeze(-1)[: contextual_budget - target.numel()]
            return torch.cat([target, extra], dim=0).sort().values
        if mode == "saliency":
            _, idx = torch.topk(residual_saliency, k=contextual_budget, largest=True)
            return idx.sort().values
        if mode == "random":
            generator = torch.Generator(device=device)
            generator.manual_seed(int(self.config.random_seed) + residual_count * 1009 + contextual_budget)
            return torch.randperm(residual_count, device=device, generator=generator)[:contextual_budget].sort().values
        raise ValueError(f"Unknown target_select={self.config.target_select!r}.")

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        budget: int,
        saliency: Optional[torch.Tensor] = None,
        keys: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError(f"Expected tokens [N, H], got {tuple(tokens.shape)}.")
        token_count = int(tokens.shape[0])
        if token_count == 0 or int(budget) <= 0:
            return tokens.new_zeros((0, self.hidden_size))
        effective_budget = min(int(budget), token_count)
        if effective_budget >= token_count:
            return tokens

        scores = self._compute_saliency(tokens, saliency)
        metric = self._compute_keys(tokens, keys)
        dominant_budget, contextual_budget = self._partition_budget(effective_budget, token_count)
        _, dominant_idx = torch.topk(scores, k=dominant_budget, largest=True)
        dominant_idx = dominant_idx.sort().values
        dominant_tokens = tokens.index_select(0, dominant_idx)
        if contextual_budget <= 0 or dominant_budget >= token_count:
            compressed = dominant_tokens
        else:
            residual_mask = torch.ones(token_count, dtype=torch.bool, device=tokens.device)
            residual_mask[dominant_idx] = False
            residual_tokens = tokens[residual_mask]
            residual_keys = metric[residual_mask]
            residual_scores = scores[residual_mask]
            target_idx = self._select_target_indices(
                residual_count=int(residual_tokens.shape[0]),
                contextual_budget=min(contextual_budget, int(residual_tokens.shape[0])),
                residual_saliency=residual_scores,
                device=tokens.device,
            )
            contextual_tokens = residual_tokens.index_select(0, target_idx).clone()
            contextual_keys = residual_keys.index_select(0, target_idx)
            merge_mask = torch.ones(residual_tokens.shape[0], dtype=torch.bool, device=tokens.device)
            merge_mask[target_idx] = False
            merge_tokens = residual_tokens[merge_mask]
            merge_keys = residual_keys[merge_mask]
            if merge_tokens.numel() > 0:
                assignments = merge_keys.matmul(contextual_keys.transpose(0, 1)).argmax(dim=-1)
                contextual_tokens.scatter_add_(
                    0,
                    assignments.unsqueeze(-1).expand(-1, merge_tokens.shape[-1]),
                    merge_tokens,
                )
                counts = torch.ones(contextual_tokens.shape[0], dtype=tokens.dtype, device=tokens.device)
                counts.scatter_add_(0, assignments, torch.ones_like(assignments, dtype=tokens.dtype))
                contextual_tokens = contextual_tokens / counts.clamp_min(1).unsqueeze(-1)
            compressed = torch.cat([dominant_tokens, contextual_tokens], dim=0)

        compressed = compressed[:effective_budget]
        if self.config.preserve_input_rms and compressed.numel() > 0:
            input_rms = tokens.pow(2).mean().sqrt().detach().clamp_min(self.config.eps)
            output_rms = compressed.pow(2).mean().sqrt().detach().clamp_min(self.config.eps)
            compressed = compressed * (input_rms / output_rms)
        return compressed


class VisionZipCompressor(nn.Module):
    def __init__(
        self,
        config: VisionZipConfig,
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
        self.stage_compressor = StageVisionZipCompressor(config, hidden_size=hidden_size)

    def _crop_lengths(self, image_embeds: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.LongTensor:
        if image_grid_thw.ndim != 2 or image_grid_thw.shape[-1] != 3:
            raise ValueError(f"Expected image_grid_thw [C, 3], got {tuple(image_grid_thw.shape)}.")
        if image_grid_thw.shape[0] != self.total_crop_count:
            raise ValueError(f"Expected {self.total_crop_count} crops for one row, got {image_grid_thw.shape[0]}.")
        crop_patch_lengths = image_grid_thw.prod(dim=1).to(device=image_embeds.device, dtype=torch.long)
        crop_lengths = torch.div(crop_patch_lengths, self.spatial_merge_area, rounding_mode="floor")
        if int(crop_lengths.sum().item()) != image_embeds.shape[0]:
            raise ValueError(
                f"merged image_grid_thw token count {int(crop_lengths.sum().item())} "
                f"does not match image_embeds rows {image_embeds.shape[0]}."
            )
        return crop_lengths

    def split_by_crop(self, image_embeds: torch.Tensor, image_grid_thw: torch.Tensor) -> List[torch.Tensor]:
        crop_lengths = self._crop_lengths(image_embeds, image_grid_thw)
        crops: List[torch.Tensor] = []
        token_start = 0
        for crop_len in crop_lengths.tolist():
            crop_end = token_start + int(crop_len)
            crops.append(image_embeds[token_start:crop_end])
            token_start = crop_end
        return crops

    def split_by_stage(self, image_embeds: torch.Tensor, image_grid_thw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        crops = self.split_by_crop(image_embeds, image_grid_thw)
        stages: List[torch.Tensor] = []
        crop_start = 0
        for crop_count in self.crop_counts:
            stage_crops = crops[crop_start : crop_start + crop_count]
            if stage_crops:
                stages.append(torch.cat(stage_crops, dim=0))
            else:
                stages.append(image_embeds.new_zeros((0, self.hidden_size)))
            crop_start += crop_count
        return stages[0], stages[1], stages[2]

    def _stage_budget(self, stage_index: int, raw_length: int) -> int:
        budget = min(int(self.config.budgets[stage_index]), int(raw_length))
        ratio = self.config.keep_ratio_for_stage(stage_index)
        if ratio is not None and raw_length > 0:
            budget = min(budget, max(1, int(math.ceil(float(raw_length) * ratio))))
        return max(0, budget)

    @staticmethod
    def _allocate_largest_remainder(lengths: List[int], total_budget: int) -> List[int]:
        caps = [max(0, int(v)) for v in lengths]
        target = min(max(0, int(total_budget)), sum(caps))
        if target <= 0 or not caps:
            return [0 for _ in caps]
        total = float(sum(caps))
        raw = [target * cap / total if cap > 0 else 0.0 for cap in caps]
        alloc = [min(cap, int(math.floor(value))) for cap, value in zip(caps, raw)]
        remaining = target - sum(alloc)
        order = sorted(range(len(caps)), key=lambda i: raw[i] - math.floor(raw[i]), reverse=True)
        while remaining > 0:
            progressed = False
            for idx in order:
                if remaining <= 0:
                    break
                if alloc[idx] < caps[idx]:
                    alloc[idx] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                break
        return alloc

    def allocate_crop_budgets(self, crop_lengths: Sequence[int], stage_budget: int) -> List[int]:
        lengths = [int(v) for v in crop_lengths]
        budget = min(int(stage_budget), sum(max(0, v) for v in lengths))
        if budget <= 0:
            return [0 for _ in lengths]
        mode = str(self.config.crop_budget_mode).lower()
        if mode == "uniform":
            active = [i for i, length in enumerate(lengths) if length > 0]
            alloc = [0 for _ in lengths]
            offset = 0
            while sum(alloc) < budget:
                idx = active[offset % len(active)]
                if alloc[idx] < lengths[idx]:
                    alloc[idx] += 1
                offset += 1
                if offset > budget * max(1, len(active)) + sum(lengths):
                    break
            if sum(alloc) < budget:
                remainder = self._allocate_largest_remainder([lengths[i] - alloc[i] for i in range(len(lengths))], budget - sum(alloc))
                alloc = [a + r for a, r in zip(alloc, remainder)]
            return alloc
        if mode == "min1":
            active = [i for i, length in enumerate(lengths) if length > 0]
            alloc = [0 for _ in lengths]
            if budget >= len(active):
                for idx in active:
                    alloc[idx] = 1
                remainder = self._allocate_largest_remainder([lengths[i] - alloc[i] for i in range(len(lengths))], budget - sum(alloc))
                return [a + r for a, r in zip(alloc, remainder)]
        return self._allocate_largest_remainder(lengths, budget)

    def _split_optional_by_stage(
        self,
        values: Optional[torch.Tensor],
        image_embeds: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if values is None:
            return None
        if values.ndim == 1:
            return self.split_by_stage(values.unsqueeze(-1), image_grid_thw)
        return self.split_by_stage(values, image_grid_thw)

    def _split_optional_by_crop(
        self,
        values: Optional[torch.Tensor],
        image_embeds: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> Optional[List[torch.Tensor]]:
        if values is None:
            return None
        if values.ndim == 1:
            return [crop.squeeze(-1) for crop in self.split_by_crop(values.unsqueeze(-1), image_grid_thw)]
        return self.split_by_crop(values, image_grid_thw)

    def _forward_stage_scope(
        self,
        image_embeds: torch.Tensor,
        image_grid_thw: torch.Tensor,
        *,
        saliency: Optional[torch.Tensor] = None,
        keys: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        active = set(self.config.active_stage_ids())
        stage_tokens = self.split_by_stage(image_embeds, image_grid_thw)
        stage_saliency = self._split_optional_by_stage(saliency, image_embeds, image_grid_thw)
        stage_keys = self._split_optional_by_stage(keys, image_embeds, image_grid_thw)
        compressed = []
        for stage_index, tokens in enumerate(stage_tokens):
            if stage_index in active:
                sal = None if stage_saliency is None else stage_saliency[stage_index].squeeze(-1)
                key = None if stage_keys is None else stage_keys[stage_index]
                compressed.append(self.stage_compressor(tokens, budget=self._stage_budget(stage_index, int(tokens.shape[0])), saliency=sal, keys=key))
            else:
                compressed.append(tokens)
        return compressed[0], compressed[1], compressed[2]

    def _forward_crop_scope(
        self,
        image_embeds: torch.Tensor,
        image_grid_thw: torch.Tensor,
        *,
        saliency: Optional[torch.Tensor] = None,
        keys: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        active = set(self.config.active_stage_ids())
        crops = self.split_by_crop(image_embeds, image_grid_thw)
        crop_saliency = self._split_optional_by_crop(saliency, image_embeds, image_grid_thw)
        crop_keys = self._split_optional_by_crop(keys, image_embeds, image_grid_thw)
        compressed_stages: List[torch.Tensor] = []
        crop_start = 0
        for stage_index, crop_count in enumerate(self.crop_counts):
            stage_crops = crops[crop_start : crop_start + crop_count]
            raw_lengths = [int(crop.shape[0]) for crop in stage_crops]
            raw_length = sum(raw_lengths)
            if stage_index not in active:
                compressed_stages.append(torch.cat(stage_crops, dim=0) if stage_crops else image_embeds.new_zeros((0, self.hidden_size)))
                crop_start += crop_count
                continue
            stage_budget = self._stage_budget(stage_index, raw_length)
            crop_budgets = self.allocate_crop_budgets(raw_lengths, stage_budget)
            compressed_crops = [
                self.stage_compressor(
                    crop_tokens,
                    budget=crop_budget,
                    saliency=None if crop_saliency is None else crop_saliency[crop_start + offset],
                    keys=None if crop_keys is None else crop_keys[crop_start + offset],
                )
                for offset, (crop_tokens, crop_budget) in enumerate(zip(stage_crops, crop_budgets))
            ]
            stage_tokens = torch.cat(compressed_crops, dim=0) if compressed_crops else image_embeds.new_zeros((0, self.hidden_size))
            if int(stage_tokens.shape[0]) != min(stage_budget, raw_length):
                raise ValueError(
                    f"VisionZip crop scope produced {stage_tokens.shape[0]} tokens for stage {stage_index}, "
                    f"expected {min(stage_budget, raw_length)}."
                )
            compressed_stages.append(stage_tokens)
            crop_start += crop_count
        return compressed_stages[0], compressed_stages[1], compressed_stages[2]

    def forward(self, image_embeds: torch.Tensor, image_grid_thw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.compress(image_embeds, image_grid_thw)

    def compress(
        self,
        image_embeds: torch.Tensor,
        image_grid_thw: torch.Tensor,
        *,
        saliency: Optional[torch.Tensor] = None,
        keys: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if saliency is not None and (saliency.ndim != 1 or saliency.shape[0] != image_embeds.shape[0]):
            raise ValueError(f"Expected saliency [{image_embeds.shape[0]}], got {tuple(saliency.shape)}.")
        if keys is not None and keys.shape != image_embeds.shape:
            raise ValueError(f"Expected keys {tuple(image_embeds.shape)}, got {tuple(keys.shape)}.")
        stage_tokens = self.split_by_stage(image_embeds, image_grid_thw)
        if str(self.config.compression_scope).lower() == "stage":
            compressed = self._forward_stage_scope(image_embeds, image_grid_thw, saliency=saliency, keys=keys)
        else:
            compressed = self._forward_crop_scope(image_embeds, image_grid_thw, saliency=saliency, keys=keys)
        if self.config.debug_shapes and (not torch.jit.is_scripting()):
            raw = [int(tokens.shape[0]) for tokens in stage_tokens]
            comp = [int(tokens.shape[0]) for tokens in compressed]
            print(f"[VisionZip] scope={self.config.compression_scope} raw_stage_tokens={raw} compact_stage_tokens={comp}", flush=True)
        return compressed

    def save_pretrained(self, save_dir: str | Path) -> None:
        self.config.save_pretrained(save_dir)
