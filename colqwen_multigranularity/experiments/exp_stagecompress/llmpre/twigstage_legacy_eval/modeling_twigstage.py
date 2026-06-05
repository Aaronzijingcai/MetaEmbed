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


class StageWiseTwigSelector(nn.Module):
    """Trainable stage-aware selector applied after early LLM layers."""

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
                raise ValueError(f"twigstage keep ratio must be in (0, 1], got {ratio}.")
        if temperature <= 0:
            raise ValueError("twigstage temperature must be positive.")
        if min_mask_value < 0 or min_mask_value >= 1:
            raise ValueError("twigstage min_mask_value must be in [0, 1).")

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

    def _scores(self, tokens: torch.Tensor, context: torch.Tensor, *, stage_index: int) -> torch.Tensor:
        selector_dtype = self.token_proj.weight.dtype
        stage_ids = torch.full((tokens.shape[0],), int(stage_index), device=tokens.device, dtype=torch.long)
        x = tokens.to(dtype=selector_dtype) + self.stage_embeddings(stage_ids)
        x = self.token_proj(self.token_norm(x))
        if self.use_context:
            context = self.context_proj(context.to(device=tokens.device, dtype=selector_dtype)).unsqueeze(0)
            x = x + context
        return self.score_head(x).squeeze(-1)

    def forward(
        self,
        tokens: torch.Tensor,
        context: torch.Tensor,
        *,
        stage_index: int,
        ste: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        token_count = int(tokens.shape[0])
        if token_count == 0:
            empty = tokens.new_zeros((0,))
            return empty, empty.bool(), {"tokens": 0, "kept": 0, "mask_sum": 0.0}

        ratio = float(self.keep_ratios[int(stage_index)].detach().cpu().item())
        keep_count = max(1, min(token_count, int(math.ceil(token_count * ratio))))
        if keep_count >= token_count:
            ones = tokens.new_ones((token_count,))
            return ones, ones.bool(), {"tokens": token_count, "kept": token_count, "mask_sum": float(token_count)}

        scores = self._scores(tokens, context, stage_index=stage_index)
        topk = torch.topk(scores, k=keep_count, largest=True, sorted=False)
        hard = torch.zeros_like(scores)
        hard.scatter_(0, topk.indices, 1.0)

        if self.training:
            threshold = topk.values.min().detach()
            soft = torch.sigmoid((scores - threshold) / self.temperature)
            soft = self.min_mask_value + (1.0 - self.min_mask_value) * soft
            mask = hard.detach() - soft.detach() + soft if ste else soft
        else:
            mask = self.min_mask_value + (1.0 - self.min_mask_value) * hard

        mask = mask.to(device=tokens.device, dtype=tokens.dtype)
        return mask, hard.bool(), {
            "tokens": token_count,
            "kept": keep_count,
            "mask_sum": float(mask.detach().float().sum().cpu().item()),
        }

    def selector_config(self) -> dict:
        return {
            "num_stages": self.num_stages,
            "keep_ratios": [float(value) for value in self.keep_ratios.detach().cpu().tolist()],
            "temperature": self.temperature,
            "min_mask_value": self.min_mask_value,
            "use_context": self.use_context,
        }


class TwigStageGlobalMRLTokenColQwen2_5(MRLColQwen2_5):  # noqa: N801
    """Global MRL-token wrapper with TwigVLM-style LLM-early stage compression."""

    def __init__(
        self,
        base_model: ColQwen2_5,
        *,
        granularities: Sequence[int] = (1, 2, 4),
        num_query_mrl_tokens: int = 16,
        num_doc_mrl_tokens: int = 64,
        shared_query_doc_mrl_tokens: bool = False,
        compact_query_tokens: bool = True,
        twigstage_mode: str = "mask",
        twigstage_exit_layer: int = 2,
        twigstage_keep_ratios: Optional[Sequence[float]] = None,
        twigstage_temperature: float = 0.1,
        twigstage_min_mask_value: float = 0.0,
        twigstage_train_prune: bool = False,
        twigstage_use_context: bool = True,
    ) -> None:
        super().__init__(
            base_model=base_model,
            granularities=granularities,
            compact_query_tokens=compact_query_tokens,
        )
        if len(self.stage_specs) != 3:
            raise ValueError("TwigStage experiment expects exactly g1/g2/g3 input stages.")
        mode = str(twigstage_mode).lower()
        if mode not in {"mask", "prune"}:
            raise ValueError(f"twigstage_mode must be 'mask' or 'prune', got {twigstage_mode!r}.")

        self.twigstage_mode = mode
        self.twigstage_exit_layer = int(twigstage_exit_layer)
        self.twigstage_train_prune = bool(twigstage_train_prune)
        self.num_query_mrl_tokens = int(num_query_mrl_tokens)
        self.num_doc_mrl_tokens = int(num_doc_mrl_tokens)
        if self.num_query_mrl_tokens <= 0 or self.num_doc_mrl_tokens <= 0:
            raise ValueError("num_query_mrl_tokens and num_doc_mrl_tokens must be positive.")
        self.shared_query_doc_mrl_tokens = bool(shared_query_doc_mrl_tokens)
        self.num_added_tokens = (
            max(self.num_query_mrl_tokens, self.num_doc_mrl_tokens)
            if self.shared_query_doc_mrl_tokens
            else self.num_query_mrl_tokens + self.num_doc_mrl_tokens
        )

        hidden_size = int(self.base_model.model.config.hidden_size)
        self.prompt_embed_tokens = nn.Embedding(self.num_added_tokens, hidden_size)
        nn.init.normal_(self.prompt_embed_tokens.weight, mean=0.0, std=hidden_size ** -0.5)
        self.twig_selector = StageWiseTwigSelector(
            hidden_size=hidden_size,
            num_stages=len(self.stage_specs),
            keep_ratios=twigstage_keep_ratios,
            temperature=twigstage_temperature,
            min_mask_value=twigstage_min_mask_value,
            use_context=twigstage_use_context,
        )
        self.padding_side = "left"
        self._global_mrl_debug_count = 0
        self._last_twigstage_stats: Optional[dict] = None

    @staticmethod
    def _has_images(pixel_values: Optional[torch.Tensor], image_grid_thw: Optional[torch.Tensor]) -> bool:
        return (
            pixel_values is not None
            and image_grid_thw is not None
            and getattr(pixel_values, "numel", lambda: 0)() > 0
            and getattr(image_grid_thw, "numel", lambda: 0)() > 0
        )

    @staticmethod
    def _debug_shape(tensor: Optional[torch.Tensor]) -> Optional[list[int]]:
        return None if tensor is None else list(tensor.shape)

    @staticmethod
    def _debug_short_list(tensor: Optional[torch.Tensor], limit: int = 12) -> Optional[list]:
        if tensor is None:
            return None
        return tensor.detach().to("cpu").reshape(-1).tolist()[:limit]

    def _debug_enabled(self) -> bool:
        return os.environ.get("TWIGSTAGE_DEBUG", os.environ.get("GLOBAL_MRL_DEBUG", "")).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _debug_limit(self) -> int:
        try:
            return int(os.environ.get("TWIGSTAGE_DEBUG_LIMIT", os.environ.get("GLOBAL_MRL_DEBUG_LIMIT", "8")))
        except ValueError:
            return 8

    def _should_debug(self) -> bool:
        if not self._debug_enabled():
            return False
        self._global_mrl_debug_count += 1
        return self._global_mrl_debug_count <= self._debug_limit()

    def _debug_print(self, message: str) -> None:
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
        print(f"[TwigStage][rank={rank}] {message}", flush=True)

    def _dummy_token_id(self) -> int:
        eos_token_id = getattr(self.config, "eos_token_id", None)
        if isinstance(eos_token_id, (list, tuple)) and eos_token_id:
            eos_token_id = eos_token_id[0]
        if eos_token_id is not None:
            return int(eos_token_id)
        pad_token_id = getattr(self.config, "pad_token_id", None)
        if pad_token_id is not None:
            return int(pad_token_id)
        return 1

    def _append_mrl_token_ids(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.LongTensor, torch.Tensor]:
        prompt_token_id = self._dummy_token_id()
        prompt_ids = input_ids.new_full((input_ids.shape[0], self.num_added_tokens), prompt_token_id)
        prompt_attention = attention_mask.new_ones((attention_mask.shape[0], self.num_added_tokens))
        return torch.cat([input_ids, prompt_ids], dim=1), torch.cat([attention_mask, prompt_attention], dim=1)

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
                raise RuntimeError(f"TwigStage image grid is not divisible by merge size: grid={row} merge_size={merge_size}.")
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
        prompt_indices = torch.arange(self.num_added_tokens, device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
        prompt_embeds = self.prompt_embed_tokens(prompt_indices).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[:, -self.num_added_tokens :, :] = prompt_embeds

        if pixel_values is not None:
            pixel_values = pixel_values.type(self.base_model.visual.dtype)
            image_embeds = self.base_model.visual(pixel_values, grid_thw=image_grid_thw)
            image_mask = input_ids.eq(int(self.config.image_token_id)).unsqueeze(-1).expand_as(inputs_embeds)
            image_embeds = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            expected = int(image_mask.sum().item() // inputs_embeds.shape[-1])
            actual = int(image_embeds.shape[0])
            if expected != actual:
                raise RuntimeError(f"TwigStage image embed mismatch: placeholders={expected} visual_embeds={actual}.")
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        return inputs_embeds

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
                    "TwigStage sample image token mismatch: "
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
            raise RuntimeError(f"TwigStage image grid cursor mismatch: {grid_cursor}/{len(crop_token_counts)}.")
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

    def _build_twigstage_masks(
        self,
        *,
        hidden_states: torch.Tensor,
        stage_map: torch.LongTensor,
        crop_map: torch.LongTensor,
        ste: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate = hidden_states.new_ones(hidden_states.shape[:2])
        hard_keep = torch.ones(hidden_states.shape[:2], device=hidden_states.device, dtype=torch.bool)
        stats = {
            "mode": self.twigstage_mode,
            "exit_layer": self.twigstage_exit_layer,
            "stage_tokens": [0 for _ in self.stage_specs],
            "stage_kept": [0 for _ in self.stage_specs],
            "stage_mask_sum": [0.0 for _ in self.stage_specs],
            "stage_crops": [0 for _ in self.stage_specs],
        }
        if not torch.any(stage_map >= 0):
            self._last_twigstage_stats = stats
            return gate, hard_keep

        context = hidden_states[:, -self.num_added_tokens :, :].mean(dim=1)
        for batch_index in range(hidden_states.shape[0]):
            crop_ids = torch.unique(crop_map[batch_index][crop_map[batch_index] >= 0])
            for crop_id in crop_ids.tolist():
                positions = torch.where(crop_map[batch_index].eq(int(crop_id)))[0]
                if positions.numel() == 0:
                    continue
                stage_index = int(stage_map[batch_index, positions[0]].item())
                if stage_index < 0:
                    continue
                crop_gate, crop_keep, crop_stats = self.twig_selector(
                    hidden_states[batch_index, positions],
                    context[batch_index],
                    stage_index=stage_index,
                    ste=ste,
                )
                gate[batch_index, positions] = crop_gate
                hard_keep[batch_index, positions] = crop_keep.to(device=hard_keep.device)
                stats["stage_tokens"][stage_index] += int(crop_stats["tokens"])
                stats["stage_kept"][stage_index] += int(crop_stats["kept"])
                stats["stage_mask_sum"][stage_index] += float(crop_stats["mask_sum"])
                stats["stage_crops"][stage_index] += 1

        stats["actual_seq_len"] = int(hidden_states.shape[1])
        stats["hard_keep_tokens"] = [int(value) for value in hard_keep.sum(dim=1).detach().cpu().tolist()]
        self._last_twigstage_stats = stats
        return gate, hard_keep

    def _prune_sequence(
        self,
        *,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        hard_keep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.LongTensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        lengths = hard_keep.sum(dim=1)
        max_len = int(lengths.max().item())
        pruned_hidden = hidden_states.new_zeros((batch_size, max_len, hidden_size))
        pruned_attention = attention_mask.new_zeros((batch_size, max_len))
        pruned_position = position_ids.new_zeros((position_ids.shape[0], batch_size, max_len))
        mrl_positions = torch.zeros((batch_size, self.num_added_tokens), device=hidden_states.device, dtype=torch.long)
        original_mrl = torch.arange(seq_len - self.num_added_tokens, seq_len, device=hidden_states.device)

        for batch_index in range(batch_size):
            keep_indices = torch.where(hard_keep[batch_index])[0]
            length = int(keep_indices.numel())
            pruned_hidden[batch_index, :length] = hidden_states[batch_index, keep_indices]
            pruned_attention[batch_index, :length] = 1
            pruned_position[:, batch_index, :length] = position_ids[:, batch_index, keep_indices]
            original_to_new = torch.full((seq_len,), -1, device=hidden_states.device, dtype=torch.long)
            original_to_new[keep_indices] = torch.arange(length, device=hidden_states.device)
            sample_mrl_positions = original_to_new[original_mrl]
            if torch.any(sample_mrl_positions < 0):
                raise RuntimeError("TwigStage pruning removed MRL prompt tokens unexpectedly.")
            mrl_positions[batch_index] = sample_mrl_positions

        if self._last_twigstage_stats is not None:
            self._last_twigstage_stats["pruned_seq_len"] = int(max_len)
            self._last_twigstage_stats["pruned_lengths"] = [int(value) for value in lengths.detach().cpu().tolist()]
        return pruned_hidden, pruned_attention, pruned_position, mrl_positions

    def _select_meta_hidden(
        self,
        hidden_states: torch.Tensor,
        *,
        is_query: bool,
        mrl_positions: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if mrl_positions is None:
            meta_hidden = hidden_states[:, -self.num_added_tokens :, :]
        else:
            gather_index = mrl_positions.to(hidden_states.device).unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
            meta_hidden = hidden_states.gather(dim=1, index=gather_index)
        if self.shared_query_doc_mrl_tokens:
            width = self.num_query_mrl_tokens if is_query else self.num_doc_mrl_tokens
            return meta_hidden[:, :width, :]
        if is_query:
            return meta_hidden[:, : self.num_query_mrl_tokens, :]
        return meta_hidden[:, self.num_query_mrl_tokens :, :]

    def _project_hidden_states(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("has_images", None)
        kwargs.pop("inputs_embeds", None)
        kwargs.pop("position_ids", None)
        is_query = bool(kwargs.pop("is_query", False))
        return_image_embeds = bool(kwargs.pop("return_image_embeds", False))
        debug_active = self._should_debug()
        debug_index = self._global_mrl_debug_count

        has_images = self._has_images(pixel_values, image_grid_thw)
        active_pixel_values = pixel_values if has_images else None
        active_image_grid_thw = image_grid_thw if has_images else None
        extended_input_ids, extended_attention_mask = self._append_mrl_token_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        inputs_embeds = self._build_inputs_embeds(
            input_ids=extended_input_ids,
            pixel_values=active_pixel_values,
            image_grid_thw=active_image_grid_thw,
        )
        position_ids, _ = self.base_model.get_rope_index(
            input_ids=extended_input_ids,
            image_grid_thw=active_image_grid_thw,
            video_grid_thw=None,
            attention_mask=extended_attention_mask,
        )
        position_ids = position_ids.to(inputs_embeds.device)
        exit_layer = max(0, min(self.twigstage_exit_layer, len(self.base_model.model.language_model.layers)))
        hidden_states = self._run_language_layers(
            hidden_states=inputs_embeds,
            attention_mask=extended_attention_mask,
            position_ids=position_ids,
            start_layer=0,
            end_layer=exit_layer,
            apply_norm=False,
        )

        mrl_positions = None
        active_attention_mask = extended_attention_mask
        active_position_ids = position_ids
        if has_images and exit_layer < len(self.base_model.model.language_model.layers):
            stage_map, crop_map = self._stage_and_crop_maps(input_ids=extended_input_ids, image_grid_thw=active_image_grid_thw)
            stage_map = stage_map.to(hidden_states.device)
            crop_map = crop_map.to(hidden_states.device)
            use_true_prune = self.twigstage_mode == "prune" and ((not self.training) or self.twigstage_train_prune)
            gate, hard_keep = self._build_twigstage_masks(
                hidden_states=hidden_states,
                stage_map=stage_map,
                crop_map=crop_map,
                ste=use_true_prune,
            )
            hidden_states = hidden_states * gate.unsqueeze(-1).to(dtype=hidden_states.dtype)
            if use_true_prune:
                hidden_states, active_attention_mask, active_position_ids, mrl_positions = self._prune_sequence(
                    hidden_states=hidden_states,
                    attention_mask=extended_attention_mask,
                    position_ids=position_ids,
                    hard_keep=hard_keep,
                )
        else:
            self._last_twigstage_stats = None

        hidden_states = self._run_language_layers(
            hidden_states=hidden_states,
            attention_mask=active_attention_mask,
            position_ids=active_position_ids,
            start_layer=exit_layer,
            end_layer=len(self.base_model.model.language_model.layers),
            apply_norm=True,
        )

        if return_image_embeds:
            selected = hidden_states
            output_mask = active_attention_mask
        else:
            selected = self._select_meta_hidden(hidden_states, is_query=is_query, mrl_positions=mrl_positions)
            expected_width = self.num_query_mrl_tokens if is_query else self.num_doc_mrl_tokens
            if selected.shape[1] != expected_width:
                raise RuntimeError(
                    "TwigStage selected width mismatch: "
                    f"is_query={is_query} selected={selected.shape[1]} expected={expected_width}."
                )
            output_mask = selected.new_ones(selected.shape[:2], dtype=active_attention_mask.dtype)

        proj = self.base_model.custom_text_proj(selected)
        proj = proj / proj.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        proj = proj * output_mask.to(device=proj.device, dtype=proj.dtype).unsqueeze(-1)
        if debug_active:
            finite = bool(torch.isfinite(proj).all().item())
            norms = proj.norm(dim=-1)
            self._debug_print(
                f"forward#{debug_index} mode={self.twigstage_mode} train_prune={self.twigstage_train_prune} "
                f"is_query={is_query} input={self._debug_shape(input_ids)} extended={self._debug_shape(extended_input_ids)} "
                f"hidden={self._debug_shape(hidden_states)} selected={self._debug_shape(selected)} output={self._debug_shape(proj)} "
                f"stats={self._last_twigstage_stats} finite={finite} "
                f"norm_min={float(norms.min().item()):.6f} norm_max={float(norms.max().item()):.6f}"
            )
        return proj

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self._project_hidden_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs,
        )

    def _active_prompt_embed_module(self):
        module = self.prompt_embed_tokens
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

    def _active_twig_selector_module(self):
        module = self.twig_selector
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

    def global_mrl_state_dict(self) -> dict:
        active_embedding = self._active_prompt_embed_module()
        return {
            "config": {
                "num_query_mrl_tokens": self.num_query_mrl_tokens,
                "num_doc_mrl_tokens": self.num_doc_mrl_tokens,
                "shared_query_doc_mrl_tokens": self.shared_query_doc_mrl_tokens,
                "num_added_tokens": self.num_added_tokens,
                "granularities": list(self.granularities),
            },
            "state_dict": {"weight": active_embedding.weight.detach().cpu()},
        }

    def save_global_mrl_token_state(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.global_mrl_state_dict(), save_dir / "global_mrl_tokens.pt")

    def load_global_mrl_token_state(self, path: str | Path, *, map_location: str | torch.device = "cpu") -> None:
        path = Path(path)
        if path.is_dir():
            path = path / "global_mrl_tokens.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location=map_location)
        saved_config = payload.get("config") if isinstance(payload, dict) else None
        if isinstance(saved_config, dict):
            expected_config = {
                "num_query_mrl_tokens": self.num_query_mrl_tokens,
                "num_doc_mrl_tokens": self.num_doc_mrl_tokens,
                "shared_query_doc_mrl_tokens": self.shared_query_doc_mrl_tokens,
                "num_added_tokens": self.num_added_tokens,
                "granularities": list(self.granularities),
            }
            mismatches = []
            for key, expected_value in expected_config.items():
                if key in saved_config and saved_config[key] != expected_value:
                    mismatches.append(f"{key}: checkpoint={saved_config[key]!r} model={expected_value!r}")
            if mismatches:
                raise ValueError("Global MRL token config mismatch: " + "; ".join(mismatches))
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        if "weight" in state_dict:
            weight = state_dict["weight"]
        elif "prompt_embed_tokens.weight" in state_dict:
            weight = state_dict["prompt_embed_tokens.weight"]
        else:
            raise KeyError(f"Could not find prompt embedding weight in {path}.")
        active_embedding = self._active_prompt_embed_module()
        if tuple(active_embedding.weight.shape) != tuple(weight.shape):
            raise ValueError(
                "Global MRL token weight shape mismatch: "
                f"model={tuple(active_embedding.weight.shape)} checkpoint={tuple(weight.shape)}."
            )
        active_embedding.weight.data.copy_(weight.to(device=active_embedding.weight.device, dtype=active_embedding.weight.dtype))

    def twigstage_selector_state_dict(self) -> dict:
        active_selector = self._active_twig_selector_module()
        return {
            "config": {
                "mode": self.twigstage_mode,
                "exit_layer": self.twigstage_exit_layer,
                "train_prune": self.twigstage_train_prune,
                **active_selector.selector_config(),
            },
            "state_dict": {key: value.detach().cpu() for key, value in active_selector.state_dict().items()},
        }

    def save_twigstage_state(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.save_global_mrl_token_state(save_dir)
        torch.save(self.twigstage_selector_state_dict(), save_dir / "twigstage_selector.pt")

    def load_twigstage_state(self, path: str | Path, *, map_location: str | torch.device = "cpu") -> None:
        path = Path(path)
        if path.is_dir():
            global_path = path / "global_mrl_tokens.pt"
            selector_path = path / "twigstage_selector.pt"
            if global_path.exists():
                self.load_global_mrl_token_state(global_path, map_location=map_location)
            if not selector_path.exists():
                return
            path = selector_path
        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location=map_location)
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        self._active_twig_selector_module().load_state_dict(state_dict, strict=True)

    def save_pretrained(self, save_dir: str, **kwargs):
        self.base_model.save_pretrained(save_dir, **kwargs)
        self.save_twigstage_state(save_dir)


def _find_twigstage_model(model) -> Optional[TwigStageGlobalMRLTokenColQwen2_5]:
    if isinstance(model, TwigStageGlobalMRLTokenColQwen2_5):
        return model
    for module in model.modules():
        if isinstance(module, TwigStageGlobalMRLTokenColQwen2_5):
            return module
    return None


def save_global_mrl_token_state(model, save_dir: str | Path) -> None:
    inner = _find_twigstage_model(model)
    if inner is None:
        raise TypeError("Could not find TwigStageGlobalMRLTokenColQwen2_5 inside model.")
    inner.save_global_mrl_token_state(save_dir)


def load_global_mrl_token_state(model, path: str | Path, *, map_location: str | torch.device = "cpu") -> None:
    inner = _find_twigstage_model(model)
    if inner is None:
        raise TypeError("Could not find TwigStageGlobalMRLTokenColQwen2_5 inside model.")
    inner.load_global_mrl_token_state(path, map_location=map_location)


def save_twigstage_state(model, save_dir: str | Path) -> None:
    inner = _find_twigstage_model(model)
    if inner is None:
        raise TypeError("Could not find TwigStageGlobalMRLTokenColQwen2_5 inside model.")
    inner.save_twigstage_state(save_dir)


def load_twigstage_state(model, path: str | Path, *, map_location: str | torch.device = "cpu") -> None:
    inner = _find_twigstage_model(model)
    if inner is None:
        raise TypeError("Could not find TwigStageGlobalMRLTokenColQwen2_5 inside model.")
    inner.load_twigstage_state(path, map_location=map_location)


def _load_adapter_with_fallback(model: TwigStageGlobalMRLTokenColQwen2_5, adapter_path: Path):
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

    with TemporaryDirectory(prefix="twigstage_eval_adapter_") as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "adapter_config.json").write_text((adapter_path / "adapter_config.json").read_text())
        torch.save(remapped, tmpdir / "adapter_model.bin")
        return PeftModel.from_pretrained(model, tmpdir)


def build_twigstage_model(
    model_name_or_path: str,
    *,
    granularities: Sequence[int] = (1, 2, 4),
    num_query_mrl_tokens: int = 16,
    num_doc_mrl_tokens: int = 64,
    shared_query_doc_mrl_tokens: bool = False,
    attn_implementation: Optional[str] = "flash_attention_2",
    use_liger_kernel: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
    adapter_path: Optional[str] = None,
    global_mrl_token_path: Optional[str] = None,
    twigstage_state_path: Optional[str] = None,
    eval_mode: bool = False,
    compact_query_tokens: bool = True,
    twigstage_mode: str = "mask",
    twigstage_exit_layer: int = 2,
    twigstage_keep_ratios: Optional[Sequence[float]] = None,
    twigstage_temperature: float = 0.1,
    twigstage_min_mask_value: float = 0.0,
    twigstage_train_prune: bool = False,
    twigstage_use_context: bool = True,
):
    granularities = normalize_granularities(granularities)
    if len(build_stage_specs(granularities)) != 3:
        raise ValueError("TwigStage experiment expects exactly three stages.")

    base_model = ColQwen2_5.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        use_cache=False,
        attn_implementation=attn_implementation,
        use_liger_kernel=use_liger_kernel,
    )
    if not hasattr(base_model, "custom_text_proj"):
        raise TypeError("Expected a ColQwen2_5 checkpoint with custom_text_proj.")
    _apply_compat_patch(base_model)

    model = TwigStageGlobalMRLTokenColQwen2_5(
        base_model=base_model,
        granularities=granularities,
        num_query_mrl_tokens=num_query_mrl_tokens,
        num_doc_mrl_tokens=num_doc_mrl_tokens,
        shared_query_doc_mrl_tokens=shared_query_doc_mrl_tokens,
        compact_query_tokens=compact_query_tokens,
        twigstage_mode=twigstage_mode,
        twigstage_exit_layer=twigstage_exit_layer,
        twigstage_keep_ratios=twigstage_keep_ratios,
        twigstage_temperature=twigstage_temperature,
        twigstage_min_mask_value=twigstage_min_mask_value,
        twigstage_train_prune=twigstage_train_prune,
        twigstage_use_context=twigstage_use_context,
    )

    if adapter_path is not None:
        model = _load_adapter_with_fallback(model, Path(adapter_path))

    state_path = global_mrl_token_path
    if state_path is None and adapter_path is not None:
        candidate = Path(adapter_path) / "global_mrl_tokens.pt"
        if candidate.exists():
            state_path = str(candidate)
    if state_path is not None and Path(state_path).exists():
        load_global_mrl_token_state(model, state_path, map_location="cpu")

    selector_state_path = twigstage_state_path
    if selector_state_path is None and adapter_path is not None:
        adapter_dir = Path(adapter_path)
        if (adapter_dir / "twigstage_selector.pt").exists():
            selector_state_path = str(adapter_dir)
    if selector_state_path is not None and Path(selector_state_path).exists():
        load_twigstage_state(model, selector_state_path, map_location="cpu")

    if eval_mode:
        model.eval()
    return model
