from __future__ import annotations

import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Sequence

import torch
import torch.nn as nn
from peft import PeftModel
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)

from colpali_engine.models import ColQwen2_5
from colqwen_multigranularity.core import (
    MRLColQwen2_5,
    _apply_compat_patch,
    build_stage_specs,
    normalize_granularities,
)


class StageWiseVisionZipSelector(nn.Module):
    """Trainable stage-aware scorer used by the VisionZip keep+merge block."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_stages: int,
        keep_ratios: Optional[Sequence[float]] = None,
        temperature: float = 0.1,
        min_mask_value: float = 0.0,
        use_context: bool = True,
    ) -> None:
        super().__init__()
        if keep_ratios is None:
            keep_ratios = (1.0, 0.5, 0.25)
        if len(keep_ratios) != num_stages:
            raise ValueError(f"Expected {num_stages} keep ratios, got {len(keep_ratios)}.")
        ratios = [float(value) for value in keep_ratios]
        for ratio in ratios:
            if ratio <= 0 or ratio > 1:
                raise ValueError(f"visionzip keep ratio must be in (0, 1], got {ratio}.")
        if temperature <= 0:
            raise ValueError("visionzip temperature must be positive.")
        if min_mask_value < 0 or min_mask_value >= 1:
            raise ValueError("visionzip min_mask_value must be in [0, 1).")

        self.num_stages = int(num_stages)
        self.temperature = float(temperature)
        self.min_mask_value = float(min_mask_value)
        self.use_context = bool(use_context)
        self.register_buffer("keep_ratios", torch.tensor(ratios, dtype=torch.float32), persistent=True)

        self.stage_embeddings = nn.Embedding(self.num_stages, hidden_size)
        self.token_norm = nn.LayerNorm(hidden_size)
        self.token_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.context_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        bottleneck = max(128, hidden_size // 4)
        self.score_head = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_size, bottleneck, bias=False),
            nn.GELU(),
            nn.Linear(bottleneck, 1),
        )
        nn.init.normal_(self.stage_embeddings.weight, mean=0.0, std=hidden_size ** -0.5)
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)

    def scores(self, tokens: torch.Tensor, context: torch.Tensor, *, stage_index: int) -> torch.Tensor:
        selector_dtype = self.token_proj.weight.dtype
        stage_ids = torch.full((tokens.shape[0],), int(stage_index), device=tokens.device, dtype=torch.long)
        x = tokens.to(dtype=selector_dtype) + self.stage_embeddings(stage_ids)
        x = self.token_proj(self.token_norm(x))
        if self.use_context:
            context = self.context_proj(context.to(device=tokens.device, dtype=selector_dtype)).unsqueeze(0)
            x = x + context
        return self.score_head(x).squeeze(-1)

    def selector_config(self) -> dict:
        return {
            "num_stages": self.num_stages,
            "keep_ratios": [float(value) for value in self.keep_ratios.detach().cpu().tolist()],
            "temperature": self.temperature,
            "min_mask_value": self.min_mask_value,
            "use_context": self.use_context,
        }


class VisionZipMRLColQwen2_5(MRLColQwen2_5):  # noqa: N801
    """MRL_Main + VisionZip-style trainable visual-token pruning/merging.

    This class does not append learnable Global MRL tokens. Training keeps the
    original sequence length through differentiable soft masking/merging so the
    original MRL_Main g1/g2/g3 masks derived from input_ids stay aligned.
    In eval/inference, prune mode can physically remove visual tokens.
    """

    def __init__(
        self,
        base_model: ColQwen2_5,
        *,
        granularities: Sequence[int] = (1, 2, 4),
        compact_query_tokens: bool = True,
        visionzip_mode: str = "mask",
        visionzip_position: str = "llm_early",
        visionzip_exit_layer: int = 2,
        visionzip_keep_ratios: Optional[Sequence[float]] = None,
        visionzip_dominant_ratio: float = 0.65,
        visionzip_contextual_ratio: float = 0.05,
        visionzip_temperature: float = 0.1,
        visionzip_min_mask_value: float = 0.0,
        visionzip_train_prune: bool = False,
        visionzip_use_context: bool = True,
    ) -> None:
        super().__init__(base_model=base_model, granularities=granularities, compact_query_tokens=compact_query_tokens)
        if len(self.stage_specs) != 3:
            raise ValueError("VisionZipMRL expects exactly three stages: g1/g2/g3.")
        mode = str(visionzip_mode).lower()
        if mode not in {"mask", "prune"}:
            raise ValueError(f"visionzip_mode must be 'mask' or 'prune', got {visionzip_mode!r}.")
        position = str(visionzip_position).lower()
        if position not in {"llm_early", "adapter_pre"}:
            raise ValueError(f"visionzip_position must be 'llm_early' or 'adapter_pre', got {visionzip_position!r}.")
        if visionzip_dominant_ratio <= 0 or visionzip_dominant_ratio > 1:
            raise ValueError(f"visionzip_dominant_ratio must be in (0, 1], got {visionzip_dominant_ratio}.")
        if visionzip_contextual_ratio < 0 or visionzip_contextual_ratio > 1:
            raise ValueError(f"visionzip_contextual_ratio must be in [0, 1], got {visionzip_contextual_ratio}.")
        if visionzip_train_prune:
            raise ValueError(
                "Hard pruning during training is disabled because MRLInBatchNegativeLoss builds masks "
                "from original input_ids. Use soft mask training and hard prune only for eval/inference."
            )

        hidden_size = int(self.base_model.model.config.hidden_size)
        self.visionzip_mode = mode
        self.visionzip_position = position
        self.visionzip_exit_layer = int(visionzip_exit_layer)
        self.visionzip_dominant_ratio = float(visionzip_dominant_ratio)
        self.visionzip_contextual_ratio = float(visionzip_contextual_ratio)
        self.visionzip_train_prune = False
        self.visionzip_selector = StageWiseVisionZipSelector(
            hidden_size=hidden_size,
            num_stages=len(self.stage_specs),
            keep_ratios=visionzip_keep_ratios,
            temperature=visionzip_temperature,
            min_mask_value=visionzip_min_mask_value,
            use_context=visionzip_use_context,
        )
        self._last_visionzip_stats: Optional[dict] = None
        self._visionzip_debug_count = 0

    @staticmethod
    def _has_images(pixel_values: Optional[torch.Tensor], image_grid_thw: Optional[torch.Tensor]) -> bool:
        return (
            pixel_values is not None
            and image_grid_thw is not None
            and getattr(pixel_values, "numel", lambda: 0)() > 0
            and getattr(image_grid_thw, "numel", lambda: 0)() > 0
        )

    def _debug_enabled(self) -> bool:
        return os.environ.get("VISIONZIP_MRL_DEBUG", "").lower() in {"1", "true", "yes", "on"}

    def _debug_limit(self) -> int:
        try:
            return int(os.environ.get("VISIONZIP_MRL_DEBUG_LIMIT", "8"))
        except ValueError:
            return 8

    def _should_debug(self) -> bool:
        if not self._debug_enabled():
            return False
        self._visionzip_debug_count += 1
        return self._visionzip_debug_count <= self._debug_limit()

    @staticmethod
    def _debug_shape(tensor: Optional[torch.Tensor]) -> Optional[list[int]]:
        return None if tensor is None else list(tensor.shape)

    def _debug_print(self, message: str) -> None:
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
        print(f"[VisionZipMRL][rank={rank}] {message}", flush=True)

    def _visual_spatial_merge_size(self) -> int:
        value = getattr(self.base_model, "spatial_merge_size", None)
        if callable(value):
            value = value()
        if value is None:
            visual = getattr(self.base_model, "visual", None)
            config = getattr(visual, "config", None)
            value = getattr(config, "spatial_merge_size", None)
        return int(value or 2)

    def _image_grid_token_counts(self, image_grid_thw: torch.LongTensor) -> list[int]:
        merge_size = self._visual_spatial_merge_size()
        denom = merge_size * merge_size
        counts: list[int] = []
        for row in image_grid_thw.detach().to("cpu").tolist():
            t, h, w = [int(value) for value in row]
            total = t * h * w
            if total % denom != 0:
                raise RuntimeError(f"VisionZipMRL image grid is not divisible by merge size: grid={row} merge_size={merge_size}.")
            counts.append(total // denom)
        return counts

    def _build_inputs_embeds(
        self,
        *,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.LongTensor],
    ) -> torch.Tensor:
        inputs_embeds = self.base_model._embed_tokens(input_ids)
        if pixel_values is None:
            return inputs_embeds
        pixel_values = pixel_values.type(self.base_model.visual.dtype)
        image_embeds = self.base_model.visual(pixel_values, grid_thw=image_grid_thw)
        image_mask = input_ids.eq(int(self.config.image_token_id)).unsqueeze(-1).expand_as(inputs_embeds)
        image_embeds = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        expected = int(image_mask.sum().item() // inputs_embeds.shape[-1])
        actual = int(image_embeds.shape[0])
        if expected != actual:
            raise RuntimeError(f"VisionZipMRL image embed mismatch: placeholders={expected} visual_embeds={actual}.")
        return inputs_embeds.masked_scatter(image_mask, image_embeds)

    def _stage_and_crop_maps(
        self,
        *,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor],
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        stage_map = input_ids.new_full(input_ids.shape, -1)
        crop_map = input_ids.new_full(input_ids.shape, -1)
        if image_grid_thw is None or image_grid_thw.numel() == 0:
            return stage_map, crop_map

        image_token_id = int(self.config.image_token_id)
        crop_token_counts = self._image_grid_token_counts(image_grid_thw)
        expected_crops = sum(spec.crop_count for spec in self.stage_specs)
        grid_cursor = 0
        crop_uid = 0
        for batch_index in range(input_ids.shape[0]):
            image_positions = torch.where(input_ids[batch_index].eq(image_token_id))[0]
            sample_tokens = int(image_positions.numel())
            if sample_tokens == 0:
                continue
            start_grid = grid_cursor
            consumed = 0
            while consumed < sample_tokens and grid_cursor < len(crop_token_counts):
                consumed += crop_token_counts[grid_cursor]
                grid_cursor += 1
            if consumed != sample_tokens:
                raise RuntimeError(
                    "VisionZipMRL sample image token mismatch: "
                    f"sample={batch_index} placeholders={sample_tokens} consumed_grid_tokens={consumed}."
                )
            sample_crop_counts = crop_token_counts[start_grid:grid_cursor]
            if len(sample_crop_counts) % expected_crops != 0:
                continue
            local_token_cursor = 0
            local_crop_cursor = 0
            for _ in range(len(sample_crop_counts) // expected_crops):
                for stage_index, spec in enumerate(self.stage_specs):
                    for _crop_index in range(spec.crop_count):
                        token_count = int(sample_crop_counts[local_crop_cursor])
                        local_crop_cursor += 1
                        positions = image_positions[local_token_cursor : local_token_cursor + token_count]
                        stage_map[batch_index, positions] = int(stage_index)
                        crop_map[batch_index, positions] = int(crop_uid)
                        crop_uid += 1
                        local_token_cursor += token_count
        if grid_cursor != len(crop_token_counts):
            raise RuntimeError(f"VisionZipMRL image grid cursor mismatch: {grid_cursor}/{len(crop_token_counts)}.")
        return stage_map, crop_map

    def _run_language_layers(
        self,
        *,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        start_layer: int,
        end_layer: int,
        apply_norm: bool,
    ) -> torch.Tensor:
        language_model = self.base_model.model.language_model
        attention_mask = attention_mask.to(hidden_states.device) if attention_mask is not None else None
        if position_ids is None:
            cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
            position_ids = cache_position.view(1, 1, -1).expand(3, hidden_states.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        position_ids = position_ids.to(hidden_states.device)
        cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            rope_position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]
            rope_position_ids = position_ids

        mask_kwargs = {
            "config": language_model.config,
            "input_embeds": hidden_states,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": None,
            "position_ids": text_position_ids,
        }
        causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
        if getattr(language_model, "has_sliding_layers", False):
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        position_embeddings = language_model.rotary_emb(hidden_states, rope_position_ids)
        for decoder_layer in language_model.layers[start_layer:end_layer]:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=text_position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

        if apply_norm:
            hidden_states = language_model.norm(hidden_states)
        return hidden_states

    def _text_context(self, hidden_states: torch.Tensor, input_ids: torch.LongTensor, attention_mask: torch.Tensor) -> torch.Tensor:
        active = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
        text_mask = active & input_ids.to(hidden_states.device).ne(int(self.config.image_token_id))
        text_mask = torch.where(text_mask.sum(dim=1).eq(0).unsqueeze(1), active, text_mask)
        denom = text_mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=hidden_states.dtype)
        return (hidden_states * text_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)).sum(dim=1) / denom

    @staticmethod
    def _select_uniform_indices(length: int, count: int, device: torch.device) -> torch.LongTensor:
        if count <= 0 or length <= 0:
            return torch.empty(0, dtype=torch.long, device=device)
        if count >= length:
            return torch.arange(length, device=device, dtype=torch.long)
        step = max(1, length // count)
        indices = torch.arange(0, length, step, device=device, dtype=torch.long)[:count]
        if indices.numel() < count:
            mask = torch.ones(length, dtype=torch.bool, device=device)
            mask[indices] = False
            remaining = mask.nonzero(as_tuple=False).squeeze(-1)
            indices = torch.cat([indices, remaining[: count - indices.numel()]], dim=0)
        return indices

    def _visionzip_crop(
        self,
        tokens: torch.Tensor,
        context: torch.Tensor,
        *,
        stage_index: int,
        hard: bool,
    ) -> tuple[torch.Tensor, torch.BoolTensor, dict]:
        token_count = int(tokens.shape[0])
        if token_count == 0:
            return tokens, torch.zeros((0,), device=tokens.device, dtype=torch.bool), {
                "tokens": 0,
                "dominant": 0,
                "contextual": 0,
                "kept": 0,
            }

        stage_ratio = float(self.visionzip_selector.keep_ratios[int(stage_index)].detach().cpu().item())
        target_count = max(1, min(token_count, int(math.ceil(token_count * stage_ratio))))
        if target_count >= token_count:
            return tokens, torch.ones((token_count,), device=tokens.device, dtype=torch.bool), {
                "tokens": token_count,
                "dominant": token_count,
                "contextual": 0,
                "kept": token_count,
            }

        if target_count <= 1:
            contextual_count = 0
            dominant_count = 1
        else:
            contextual_count = min(target_count - 1, max(1, int(math.ceil(token_count * self.visionzip_contextual_ratio))))
            dominant_count = max(
                1,
                min(target_count - contextual_count, int(math.ceil(token_count * self.visionzip_dominant_ratio))),
            )
            contextual_count = target_count - dominant_count

        scores = self.visionzip_selector.scores(tokens, context, stage_index=stage_index)
        _, dominant_idx = torch.topk(scores, k=dominant_count, largest=True, sorted=False)
        dominant_idx = dominant_idx.sort().values
        keep_mask = torch.zeros((token_count,), device=tokens.device, dtype=torch.bool)
        keep_mask[dominant_idx] = True

        residual_idx = (~keep_mask).nonzero(as_tuple=False).squeeze(-1)
        residual_tokens = tokens.index_select(0, residual_idx)
        residual_scores = scores.index_select(0, residual_idx)

        contextual_count = min(contextual_count, int(residual_tokens.shape[0]))
        if contextual_count <= 0:
            compressed = tokens.index_select(0, keep_mask.nonzero(as_tuple=False).squeeze(-1))
            return compressed, keep_mask, {
                "tokens": token_count,
                "dominant": int(dominant_count),
                "contextual": 0,
                "kept": int(keep_mask.sum().item()),
            }

        target_rel_idx = self._select_uniform_indices(int(residual_tokens.shape[0]), contextual_count, tokens.device)
        target_abs_idx = residual_idx.index_select(0, target_rel_idx)
        keep_mask[target_abs_idx] = True

        contextual_tokens = residual_tokens.index_select(0, target_rel_idx)
        merge_mask = torch.ones((residual_tokens.shape[0],), device=tokens.device, dtype=torch.bool)
        merge_mask[target_rel_idx] = False
        merge_tokens = residual_tokens[merge_mask]
        merge_scores = residual_scores[merge_mask]

        if merge_tokens.numel() > 0:
            merge_features = torch.nn.functional.normalize(merge_tokens.float(), dim=-1).to(dtype=tokens.dtype)
            contextual_features = torch.nn.functional.normalize(contextual_tokens.float(), dim=-1).to(dtype=tokens.dtype)
            similarity = merge_features @ contextual_features.transpose(0, 1)
            if hard:
                assignments = similarity.argmax(dim=-1)
                aggregated = contextual_tokens.new_zeros(contextual_tokens.shape)
                aggregated.scatter_add_(0, assignments.unsqueeze(-1).expand_as(merge_tokens), merge_tokens)
                mass = contextual_tokens.new_zeros((contextual_count,))
                mass.scatter_add_(0, assignments, torch.ones_like(assignments, dtype=contextual_tokens.dtype))
                contextual_tokens = contextual_tokens + aggregated / mass.clamp_min(1e-6).unsqueeze(-1)
            else:
                assign = torch.softmax(similarity / max(self.visionzip_selector.temperature, 1e-6), dim=-1)
                weights = torch.softmax(merge_scores / max(self.visionzip_selector.temperature, 1e-6), dim=0)
                weighted_assign = assign * weights.unsqueeze(-1)
                aggregated = weighted_assign.transpose(0, 1) @ merge_tokens
                mass = weighted_assign.sum(dim=0).clamp_min(1e-6).unsqueeze(-1)
                contextual_tokens = contextual_tokens + aggregated / mass

        selected_idx = keep_mask.nonzero(as_tuple=False).squeeze(-1)
        compressed = tokens.index_select(0, selected_idx)
        contextual_output_pos = torch.where(selected_idx.unsqueeze(0).eq(target_abs_idx.unsqueeze(1)))[1]
        compressed[contextual_output_pos] = contextual_tokens.to(dtype=tokens.dtype)
        return compressed, keep_mask, {
            "tokens": token_count,
            "dominant": int(dominant_count),
            "contextual": int(contextual_count),
            "kept": int(keep_mask.sum().item()),
        }

    def _visionzip_crop_soft_full(
        self,
        tokens: torch.Tensor,
        context: torch.Tensor,
        *,
        stage_index: int,
    ) -> tuple[torch.Tensor, torch.BoolTensor, dict]:
        token_count = int(tokens.shape[0])
        if token_count == 0:
            return tokens, torch.zeros((0,), device=tokens.device, dtype=torch.bool), {
                "tokens": 0,
                "dominant": 0,
                "contextual": 0,
                "kept": 0,
                "mask_sum": 0.0,
            }

        stage_ratio = float(self.visionzip_selector.keep_ratios[int(stage_index)].detach().cpu().item())
        target_count = max(1, min(token_count, int(math.ceil(token_count * stage_ratio))))
        if target_count >= token_count:
            return tokens, torch.ones((token_count,), device=tokens.device, dtype=torch.bool), {
                "tokens": token_count,
                "dominant": token_count,
                "contextual": 0,
                "kept": token_count,
                "mask_sum": float(token_count),
            }

        if target_count <= 1:
            contextual_count = 0
            dominant_count = 1
        else:
            contextual_count = min(target_count - 1, max(1, int(math.ceil(token_count * self.visionzip_contextual_ratio))))
            dominant_count = max(
                1,
                min(target_count - contextual_count, int(math.ceil(token_count * self.visionzip_dominant_ratio))),
            )
            contextual_count = target_count - dominant_count
        scores = self.visionzip_selector.scores(tokens, context, stage_index=stage_index)
        topk = torch.topk(scores, k=dominant_count, largest=True, sorted=False)
        dominant_idx = topk.indices.sort().values
        keep_mask = torch.zeros((token_count,), device=tokens.device, dtype=torch.bool)
        keep_mask[dominant_idx] = True
        threshold = topk.values.min().detach()
        soft_gate = torch.sigmoid((scores - threshold) / max(self.visionzip_selector.temperature, 1e-6))
        soft_gate = self.visionzip_selector.min_mask_value + (1.0 - self.visionzip_selector.min_mask_value) * soft_gate

        residual_idx = (~keep_mask).nonzero(as_tuple=False).squeeze(-1)
        residual_tokens = tokens.index_select(0, residual_idx)
        residual_scores = scores.index_select(0, residual_idx)
        contextual_count = min(contextual_count, int(residual_tokens.shape[0]))
        output = tokens * soft_gate.to(dtype=tokens.dtype).unsqueeze(-1)
        output[dominant_idx] = tokens.index_select(0, dominant_idx)

        if contextual_count > 0:
            target_rel_idx = self._select_uniform_indices(int(residual_tokens.shape[0]), contextual_count, tokens.device)
            target_abs_idx = residual_idx.index_select(0, target_rel_idx)
            keep_mask[target_abs_idx] = True
            contextual_tokens = residual_tokens.index_select(0, target_rel_idx)
            merge_mask = torch.ones((residual_tokens.shape[0],), device=tokens.device, dtype=torch.bool)
            merge_mask[target_rel_idx] = False
            merge_tokens = residual_tokens[merge_mask]
            merge_scores = residual_scores[merge_mask]
            if merge_tokens.numel() > 0:
                merge_features = torch.nn.functional.normalize(merge_tokens.float(), dim=-1).to(dtype=tokens.dtype)
                contextual_features = torch.nn.functional.normalize(contextual_tokens.float(), dim=-1).to(dtype=tokens.dtype)
                similarity = merge_features @ contextual_features.transpose(0, 1)
                assign = torch.softmax(similarity / max(self.visionzip_selector.temperature, 1e-6), dim=-1)
                weights = torch.softmax(merge_scores / max(self.visionzip_selector.temperature, 1e-6), dim=0)
                weighted_assign = assign * weights.unsqueeze(-1)
                aggregated = weighted_assign.transpose(0, 1) @ merge_tokens
                mass = weighted_assign.sum(dim=0).clamp_min(1e-6).unsqueeze(-1)
                contextual_tokens = contextual_tokens + aggregated / mass
            output[target_abs_idx] = contextual_tokens.to(dtype=output.dtype)

        return output, keep_mask, {
            "tokens": token_count,
            "dominant": int(dominant_count),
            "contextual": int(contextual_count),
            "kept": int(keep_mask.sum().item()),
            "mask_sum": float(soft_gate.detach().float().sum().cpu().item()),
        }

    def _soft_visionzip_sequence(
        self,
        *,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        stage_map: torch.LongTensor,
        crop_map: torch.LongTensor,
    ) -> torch.Tensor:
        output = hidden_states.clone()
        stats = {
            "mode": self.visionzip_mode,
            "position": self.visionzip_position,
            "exit_layer": self.visionzip_exit_layer,
            "train_soft": True,
            "stage_tokens": [0 for _ in self.stage_specs],
            "stage_kept": [0 for _ in self.stage_specs],
            "stage_dominant": [0 for _ in self.stage_specs],
            "stage_contextual": [0 for _ in self.stage_specs],
            "stage_mask_sum": [0.0 for _ in self.stage_specs],
            "stage_crops": [0 for _ in self.stage_specs],
        }
        if not torch.any(stage_map >= 0):
            self._last_visionzip_stats = stats
            return output

        context = self._text_context(hidden_states, input_ids, attention_mask)
        for batch_index in range(hidden_states.shape[0]):
            crop_ids = torch.unique(crop_map[batch_index][crop_map[batch_index] >= 0])
            for crop_id in crop_ids.tolist():
                positions = torch.where(crop_map[batch_index].eq(int(crop_id)))[0]
                if positions.numel() == 0:
                    continue
                stage_index = int(stage_map[batch_index, positions[0]].item())
                if stage_index < 0:
                    continue
                crop_output, _crop_keep, crop_stats = self._visionzip_crop_soft_full(
                    hidden_states[batch_index, positions],
                    context[batch_index],
                    stage_index=stage_index,
                )
                output[batch_index, positions] = crop_output
                stats["stage_tokens"][stage_index] += int(crop_stats["tokens"])
                stats["stage_kept"][stage_index] += int(crop_stats["kept"])
                stats["stage_dominant"][stage_index] += int(crop_stats["dominant"])
                stats["stage_contextual"][stage_index] += int(crop_stats["contextual"])
                stats["stage_mask_sum"][stage_index] += float(crop_stats["mask_sum"])
                stats["stage_crops"][stage_index] += 1

        stats["actual_seq_len"] = int(hidden_states.shape[1])
        self._last_visionzip_stats = stats
        return output

    def _compress_visionzip_sequence(
        self,
        *,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        stage_map: torch.LongTensor,
        crop_map: torch.LongTensor,
        hard: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        stats = {
            "mode": self.visionzip_mode,
            "position": self.visionzip_position,
            "exit_layer": self.visionzip_exit_layer,
            "stage_tokens": [0 for _ in self.stage_specs],
            "stage_kept": [0 for _ in self.stage_specs],
            "stage_dominant": [0 for _ in self.stage_specs],
            "stage_contextual": [0 for _ in self.stage_specs],
            "stage_crops": [0 for _ in self.stage_specs],
        }
        if not torch.any(stage_map >= 0):
            self._last_visionzip_stats = stats
            return hidden_states, attention_mask, position_ids

        context = self._text_context(hidden_states, input_ids, attention_mask)
        compressed_rows = []
        compressed_positions = []
        for batch_index in range(batch_size):
            row_parts = []
            row_pos_parts = []
            out_len = 0
            image_position_set = set(torch.where(crop_map[batch_index] >= 0)[0].detach().cpu().tolist())
            cursor = 0
            crop_ids = torch.unique(crop_map[batch_index][crop_map[batch_index] >= 0])
            crop_positions = {int(crop_id): torch.where(crop_map[batch_index].eq(int(crop_id)))[0] for crop_id in crop_ids.tolist()}
            for crop_id in crop_ids.tolist():
                positions = crop_positions[int(crop_id)]
                if positions.numel() == 0:
                    continue
                while cursor < int(positions[0].item()):
                    if cursor not in image_position_set:
                        row_parts.append(hidden_states[batch_index, cursor : cursor + 1])
                        row_pos_parts.append(position_ids[:, batch_index, cursor : cursor + 1])
                        out_len += 1
                    cursor += 1
                stage_index = int(stage_map[batch_index, positions[0]].item())
                if stage_index < 0:
                    row_parts.append(hidden_states[batch_index, positions])
                    row_pos_parts.append(position_ids[:, batch_index, positions])
                    out_len += int(positions.numel())
                    cursor = int(positions[-1].item()) + 1
                    continue
                compressed, keep_mask, crop_stats = self._visionzip_crop(
                    hidden_states[batch_index, positions],
                    context[batch_index],
                    stage_index=stage_index,
                    hard=hard,
                )
                row_parts.append(compressed)
                kept_positions = positions[keep_mask]
                if kept_positions.numel() < compressed.shape[0]:
                    kept_positions = positions[: compressed.shape[0]]
                row_pos_parts.append(position_ids[:, batch_index, kept_positions[: compressed.shape[0]]])
                out_len += int(compressed.shape[0])
                stats["stage_tokens"][stage_index] += int(crop_stats["tokens"])
                stats["stage_kept"][stage_index] += int(crop_stats["kept"])
                stats["stage_dominant"][stage_index] += int(crop_stats["dominant"])
                stats["stage_contextual"][stage_index] += int(crop_stats["contextual"])
                stats["stage_crops"][stage_index] += 1
                cursor = int(positions[-1].item()) + 1

            while cursor < seq_len:
                if cursor not in image_position_set:
                    row_parts.append(hidden_states[batch_index, cursor : cursor + 1])
                    row_pos_parts.append(position_ids[:, batch_index, cursor : cursor + 1])
                    out_len += 1
                cursor += 1

            row_hidden = torch.cat(row_parts, dim=0) if row_parts else hidden_states.new_zeros((1, hidden_size))
            row_position = torch.cat(row_pos_parts, dim=1) if row_pos_parts else position_ids[:, batch_index, :1]
            compressed_rows.append(row_hidden)
            compressed_positions.append(row_position)

        stats["actual_seq_len"] = int(hidden_states.shape[1])
        lengths = [row.shape[0] for row in compressed_rows]
        max_len = max(lengths)
        out_hidden = hidden_states.new_zeros((batch_size, max_len, hidden_size))
        out_attention = attention_mask.new_zeros((batch_size, max_len))
        out_position = position_ids.new_zeros((position_ids.shape[0], batch_size, max_len))
        for batch_index, (row_hidden, row_position) in enumerate(zip(compressed_rows, compressed_positions)):
            length = row_hidden.shape[0]
            out_hidden[batch_index, :length] = row_hidden
            out_attention[batch_index, :length] = 1
            out_position[:, batch_index, :length] = row_position
        stats["compressed_seq_len"] = int(max_len)
        stats["compressed_lengths"] = [int(value) for value in lengths]
        self._last_visionzip_stats = stats
        return out_hidden, out_attention, out_position

    def _project_hidden_states_with_mask(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("has_images", None)
        kwargs.pop("inputs_embeds", None)
        kwargs.pop("position_ids", None)
        kwargs.pop("is_query", None)
        debug_active = self._should_debug()
        debug_index = self._visionzip_debug_count

        has_images = self._has_images(pixel_values, image_grid_thw)
        active_pixel_values = pixel_values if has_images else None
        active_image_grid_thw = image_grid_thw if has_images else None

        inputs_embeds = self._build_inputs_embeds(
            input_ids=input_ids,
            pixel_values=active_pixel_values,
            image_grid_thw=active_image_grid_thw,
        )
        position_ids, _ = self.base_model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=active_image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )
        position_ids = position_ids.to(inputs_embeds.device)
        active_attention_mask = attention_mask
        active_position_ids = position_ids

        use_true_prune = self.visionzip_mode == "prune" and not self.training
        if has_images and self.visionzip_position == "adapter_pre":
            stage_map, crop_map = self._stage_and_crop_maps(input_ids=input_ids, image_grid_thw=active_image_grid_thw)
            stage_map = stage_map.to(inputs_embeds.device)
            crop_map = crop_map.to(inputs_embeds.device)
            if use_true_prune:
                inputs_embeds, active_attention_mask, active_position_ids = self._compress_visionzip_sequence(
                    hidden_states=inputs_embeds,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    stage_map=stage_map,
                    crop_map=crop_map,
                    hard=True,
                )
            else:
                inputs_embeds = self._soft_visionzip_sequence(
                    hidden_states=inputs_embeds,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    stage_map=stage_map,
                    crop_map=crop_map,
                )
        elif not has_images:
            self._last_visionzip_stats = None

        num_layers = len(self.base_model.model.language_model.layers)
        exit_layer = max(0, min(self.visionzip_exit_layer, num_layers))
        first_end = exit_layer if self.visionzip_position == "llm_early" else num_layers
        hidden_states = self._run_language_layers(
            hidden_states=inputs_embeds,
            attention_mask=active_attention_mask,
            position_ids=active_position_ids,
            start_layer=0,
            end_layer=first_end,
            apply_norm=False,
        )

        if has_images and self.visionzip_position == "llm_early" and exit_layer < num_layers:
            stage_map, crop_map = self._stage_and_crop_maps(input_ids=input_ids, image_grid_thw=active_image_grid_thw)
            stage_map = stage_map.to(hidden_states.device)
            crop_map = crop_map.to(hidden_states.device)
            if use_true_prune:
                hidden_states, active_attention_mask, active_position_ids = self._compress_visionzip_sequence(
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    attention_mask=active_attention_mask,
                    position_ids=active_position_ids,
                    stage_map=stage_map,
                    crop_map=crop_map,
                    hard=True,
                )
            else:
                hidden_states = self._soft_visionzip_sequence(
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    attention_mask=active_attention_mask,
                    stage_map=stage_map,
                    crop_map=crop_map,
                )
        elif self.visionzip_position == "llm_early" and not has_images:
            self._last_visionzip_stats = None

        if self.visionzip_position == "llm_early":
            hidden_states = self._run_language_layers(
                hidden_states=hidden_states,
                attention_mask=active_attention_mask,
                position_ids=active_position_ids,
                start_layer=exit_layer,
                end_layer=num_layers,
                apply_norm=True,
            )
        else:
            hidden_states = self.base_model.model.language_model.norm(hidden_states)

        proj = self.base_model.custom_text_proj(hidden_states)
        proj = proj / proj.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        output_mask = active_attention_mask.to(device=proj.device, dtype=proj.dtype)
        proj = proj * output_mask.unsqueeze(-1)
        if debug_active:
            norms = proj.norm(dim=-1)
            self._debug_print(
                f"forward#{debug_index} mode={self.visionzip_mode} position={self.visionzip_position} "
                f"input={self._debug_shape(input_ids)} hidden={self._debug_shape(hidden_states)} output={self._debug_shape(proj)} "
                f"stats={self._last_visionzip_stats} finite={bool(torch.isfinite(proj).all().item())} "
                f"norm_min={float(norms.min().item()):.6f} norm_max={float(norms.max().item()):.6f}"
            )
        return proj, active_attention_mask

    def _project_hidden_states(self, **kwargs) -> torch.Tensor:
        proj, _ = self._project_hidden_states_with_mask(**kwargs)
        return proj

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        has_images = self._has_images(pixel_values, image_grid_thw)
        proj, output_mask = self._project_hidden_states_with_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values if has_images else None,
            image_grid_thw=image_grid_thw if has_images else None,
            **kwargs,
        )
        if self.compact_query_tokens:
            return self._compact_sequences(proj, output_mask.to(dtype=torch.bool))
        return proj

    def _active_visionzip_selector_module(self):
        module = self.visionzip_selector
        modules_to_save = getattr(module, "modules_to_save", None)
        if modules_to_save is None:
            return module
        active_adapter = getattr(module, "active_adapter", None)
        if isinstance(active_adapter, (list, tuple)) and active_adapter:
            active_adapter = active_adapter[0]
        if active_adapter in modules_to_save:
            return modules_to_save[active_adapter]
        try:
            return next(iter(modules_to_save.values()))
        except StopIteration:
            return module

    def visionzip_mrl_selector_state_dict(self) -> dict:
        active_selector = self._active_visionzip_selector_module()
        return {
            "config": {
                "mode": self.visionzip_mode,
                "position": self.visionzip_position,
                "exit_layer": self.visionzip_exit_layer,
                "dominant_ratio": self.visionzip_dominant_ratio,
                "contextual_ratio": self.visionzip_contextual_ratio,
                **active_selector.selector_config(),
            },
            "state_dict": {key: value.detach().cpu() for key, value in active_selector.state_dict().items()},
        }

    def save_visionzip_mrl_state(self, save_dir: str | Path) -> None:
        save_path = Path(save_dir) / "visionzip_mrl_selector.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.visionzip_mrl_selector_state_dict(), save_path)

    def load_visionzip_mrl_state(self, path: str | Path, *, map_location: str | torch.device = "cpu") -> None:
        path = Path(path)
        if path.is_dir():
            path = path / "visionzip_mrl_selector.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location=map_location)
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        self._active_visionzip_selector_module().load_state_dict(state_dict, strict=True)

    def save_pretrained(self, save_dir: str, **kwargs):
        self.base_model.save_pretrained(save_dir, **kwargs)
        self.save_visionzip_mrl_state(save_dir)


def _find_visionzip_mrl_model(model) -> VisionZipMRLColQwen2_5:
    if isinstance(model, VisionZipMRLColQwen2_5):
        return model
    if hasattr(model, "modules"):
        for module in model.modules():
            if isinstance(module, VisionZipMRLColQwen2_5):
                return module
    raise TypeError(f"Could not find VisionZipMRLColQwen2_5 inside {type(model)!r}.")


def save_visionzip_mrl_state(model, save_dir: str | Path) -> None:
    _find_visionzip_mrl_model(model).save_visionzip_mrl_state(save_dir)


def load_visionzip_mrl_state(model, path: str | Path, *, map_location: str | torch.device = "cpu") -> None:
    _find_visionzip_mrl_model(model).load_visionzip_mrl_state(path, map_location=map_location)


def _load_adapter_with_fallback(model: VisionZipMRLColQwen2_5, adapter_path: Path):
    try:
        return PeftModel.from_pretrained(model, adapter_path)
    except Exception:
        adapter_bin = adapter_path / "adapter_model.bin"
        if not adapter_bin.exists():
            raise

    state_dict = torch.load(adapter_bin, map_location="cpu")
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("base_model.model.base_model.custom_text_proj."):
            key = key.replace(
                "base_model.model.base_model.custom_text_proj.",
                "base_model.model.base_model.base_model.custom_text_proj.",
                1,
            )
        if key.startswith("base_model.model.base_model.model."):
            key = key.replace(
                "base_model.model.base_model.model.",
                "base_model.model.base_model.base_model.model.",
                1,
            )
        remapped[key] = value

    with TemporaryDirectory(prefix="visionzip_mrl_eval_adapter_") as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "adapter_config.json").write_text((adapter_path / "adapter_config.json").read_text())
        torch.save(remapped, tmpdir / "adapter_model.bin")
        return PeftModel.from_pretrained(model, tmpdir)


def build_visionzip_mrl_model(
    model_name_or_path: str,
    *,
    granularities: Sequence[int] = (1, 2, 4),
    attn_implementation: Optional[str] = "flash_attention_2",
    use_liger_kernel: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
    adapter_path: Optional[str] = None,
    visionzip_mrl_state_path: Optional[str] = None,
    eval_mode: bool = False,
    compact_query_tokens: bool = True,
    visionzip_mode: str = "mask",
    visionzip_position: str = "llm_early",
    visionzip_exit_layer: int = 2,
    visionzip_keep_ratios: Optional[Sequence[float]] = None,
    visionzip_dominant_ratio: float = 0.65,
    visionzip_contextual_ratio: float = 0.05,
    visionzip_temperature: float = 0.1,
    visionzip_min_mask_value: float = 0.0,
    visionzip_train_prune: bool = False,
    visionzip_use_context: bool = True,
):
    granularities = normalize_granularities(granularities)
    if len(build_stage_specs(granularities)) != 3:
        raise ValueError("VisionZipMRL expects exactly three stages.")

    base_model = ColQwen2_5.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        use_cache=False,
        attn_implementation=attn_implementation,
        use_liger_kernel=use_liger_kernel,
    )
    if not hasattr(base_model, "custom_text_proj"):
        raise TypeError(f"Expected a ColQwen2_5 checkpoint with custom_text_proj, got {model_name_or_path}.")
    _apply_compat_patch(base_model)

    model = VisionZipMRLColQwen2_5(
        base_model=base_model,
        granularities=granularities,
        compact_query_tokens=compact_query_tokens,
        visionzip_mode=visionzip_mode,
        visionzip_position=visionzip_position,
        visionzip_exit_layer=visionzip_exit_layer,
        visionzip_keep_ratios=visionzip_keep_ratios,
        visionzip_dominant_ratio=visionzip_dominant_ratio,
        visionzip_contextual_ratio=visionzip_contextual_ratio,
        visionzip_temperature=visionzip_temperature,
        visionzip_min_mask_value=visionzip_min_mask_value,
        visionzip_train_prune=visionzip_train_prune,
        visionzip_use_context=visionzip_use_context,
    )
    if adapter_path is not None:
        model = _load_adapter_with_fallback(model, Path(adapter_path))
        if visionzip_mrl_state_path is None:
            candidate = Path(adapter_path) / "visionzip_mrl_selector.pt"
            if candidate.exists():
                visionzip_mrl_state_path = str(candidate)
    if visionzip_mrl_state_path is not None:
        load_visionzip_mrl_state(model, visionzip_mrl_state_path, map_location="cpu")
    if eval_mode:
        model.eval()
    return model
