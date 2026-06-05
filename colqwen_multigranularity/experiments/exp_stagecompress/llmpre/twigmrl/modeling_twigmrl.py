from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
from peft import PeftModel
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLDecoderLayer,
    create_causal_mask,
    create_sliding_window_causal_mask,
)

from colpali_engine.models import ColQwen2_5
from colqwen_multigranularity.core import MRLColQwen2_5, _apply_compat_patch, normalize_granularities


class TwigDecoderBranch(nn.Module):
    """TwigVLM-style auxiliary decoder branch."""

    def __init__(self, config, *, exit_layer: int, twig_depth: int) -> None:
        super().__init__()
        if twig_depth <= 0:
            raise ValueError("twig_depth must be positive.")
        twig_config = copy.deepcopy(config)
        twig_config._attn_implementation = "eager"
        self.config = twig_config
        self.has_sliding_layers = bool(getattr(config, "has_sliding_layers", False))
        self.layers = nn.ModuleList(
            Qwen2_5_VLDecoderLayer(twig_config, layer_idx=int(exit_layer) + layer_offset)
            for layer_offset in range(int(twig_depth))
        )

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __getitem__(self, index: int):
        return self.layers[index]


class StageWiseTwigSelector(nn.Module):
    """Converts TwigVLM branch attention scores into crop-wise masks.

    The trainable signal comes from the extra TwigVLM-style decoder layers in
    `TwigMRLColQwen2_5.twig_layers`. This module intentionally keeps only the
    stage keep-ratio policy and differentiable mask construction.
    """

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
        del hidden_size, use_context
        if keep_ratios is None:
            keep_ratios = (1.0, 0.5, 0.25)
        if len(keep_ratios) != num_stages:
            raise ValueError(f"Expected {num_stages} keep ratios, got {len(keep_ratios)}.")
        ratios = [float(value) for value in keep_ratios]
        for ratio in ratios:
            if ratio <= 0 or ratio > 1:
                raise ValueError(f"twigmrl keep ratio must be in (0, 1], got {ratio}.")
        if temperature <= 0:
            raise ValueError("twigmrl temperature must be positive.")
        if min_mask_value < 0 or min_mask_value >= 1:
            raise ValueError("twigmrl min_mask_value must be in [0, 1).")

        self.num_stages = int(num_stages)
        self.temperature = float(temperature)
        self.min_mask_value = float(min_mask_value)
        self.register_buffer("keep_ratios", torch.tensor(ratios, dtype=torch.float32), persistent=True)

    def forward(
        self,
        scores: torch.Tensor,
        *,
        stage_index: int,
        dtype: torch.dtype,
        ste: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        token_count = int(scores.shape[0])
        if token_count == 0:
            empty = scores.new_zeros((0,), dtype=dtype)
            return empty, empty.bool(), {"tokens": 0, "kept": 0, "mask_sum": 0.0}

        ratio = float(self.keep_ratios[int(stage_index)].detach().cpu().item())
        keep_count = max(1, min(token_count, int(math.ceil(token_count * ratio))))
        if keep_count >= token_count:
            ones = scores.new_ones((token_count,), dtype=dtype)
            return ones, ones.bool(), {"tokens": token_count, "kept": token_count, "mask_sum": float(token_count)}

        scores = scores.float()
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

        mask = mask.to(dtype=dtype)
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
            "score_source": "twig_layers_attention",
        }


class TwigMRLColQwen2_5(MRLColQwen2_5):  # noqa: N801
    """MRL_Main + TwigVLM-style trainable visual-token selector.

    No MetaEmbed/global learnable tokens are appended. Training uses soft masks
    and keeps sequence length unchanged, so MRL_Main g1/g2/g3 masks derived from
    input_ids stay aligned with the output embeddings.
    """

    def __init__(
        self,
        base_model: ColQwen2_5,
        *,
        granularities: Sequence[int] = (1, 2, 4),
        compact_query_tokens: bool = True,
        twigmrl_mode: str = "mask",
        twigmrl_exit_layer: int = 2,
        twigmrl_twig_depth: int = 3,
        twigmrl_keep_ratios: Optional[Sequence[float]] = None,
        twigmrl_temperature: float = 0.1,
        twigmrl_min_mask_value: float = 0.0,
        twigmrl_train_prune: bool = False,
        twigmrl_use_context: bool = True,
    ) -> None:
        super().__init__(base_model=base_model, granularities=granularities, compact_query_tokens=compact_query_tokens)
        if len(self.stage_specs) != 3:
            raise ValueError("TwigMRL expects exactly three stages: g1/g2/g3.")
        mode = str(twigmrl_mode).lower()
        if mode not in {"mask", "prune"}:
            raise ValueError(f"twigmrl_mode must be 'mask' or 'prune', got {twigmrl_mode!r}.")
        if twigmrl_train_prune:
            raise ValueError(
                "Hard pruning during training is disabled because MRLInBatchNegativeLoss builds masks "
                "from original input_ids. Use soft mask training, and hard prune only for eval/inference."
            )
        hidden_size = int(self.base_model.model.config.hidden_size)
        self.twigmrl_mode = mode
        self.twigmrl_exit_layer = int(twigmrl_exit_layer)
        self.twigmrl_twig_depth = int(twigmrl_twig_depth)
        if self.twigmrl_twig_depth <= 0:
            raise ValueError("twigmrl_twig_depth must be positive.")
        self.twigmrl_train_prune = False
        self.twig_layers = TwigDecoderBranch(
            self.base_model.model.language_model.config,
            exit_layer=self.twigmrl_exit_layer,
            twig_depth=self.twigmrl_twig_depth,
        )
        self._init_twig_layers_from_backbone()
        self.twig_selector = StageWiseTwigSelector(
            hidden_size=hidden_size,
            num_stages=len(self.stage_specs),
            keep_ratios=twigmrl_keep_ratios,
            temperature=twigmrl_temperature,
            min_mask_value=twigmrl_min_mask_value,
            use_context=twigmrl_use_context,
        )
        self._last_twigmrl_stats: Optional[dict] = None

    def _init_twig_layers_from_backbone(self) -> None:
        """Initialize twig layers from the corresponding shallow LLM layers.

        Original TwigVLM trains decoder layers K..K+T-1 as the twig branch and
        later loads those weights into `model.twig_layers`. Copying the same
        layer range keeps this retrieval variant close to that setup.
        """
        backbone_layers = self.base_model.model.language_model.layers
        start = self.twigmrl_exit_layer
        end = start + self.twigmrl_twig_depth
        if start < 0 or end > len(backbone_layers):
            raise ValueError(
                "TwigMRL twig branch exceeds language model depth: "
                f"exit_layer={start}, twig_depth={self.twigmrl_twig_depth}, "
                f"num_layers={len(backbone_layers)}."
            )

        for twig_layer, source_layer in zip(self.twig_layers.layers, backbone_layers[start:end]):
            twig_layer.load_state_dict(source_layer.state_dict(), strict=True)

        first_param = next(backbone_layers[start].parameters(), None)
        if first_param is not None:
            self.twig_layers.to(device=first_param.device, dtype=first_param.dtype)

    @staticmethod
    def _has_images(pixel_values: Optional[torch.Tensor], image_grid_thw: Optional[torch.Tensor]) -> bool:
        return (
            pixel_values is not None
            and image_grid_thw is not None
            and getattr(pixel_values, "numel", lambda: 0)() > 0
            and getattr(image_grid_thw, "numel", lambda: 0)() > 0
        )

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
                raise RuntimeError(f"TwigMRL image grid is not divisible by merge size: grid={row} merge_size={merge_size}.")
            counts.append(total // denom)
        return counts

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
                    "TwigMRL sample image token mismatch: "
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
            raise RuntimeError(f"TwigMRL image grid cursor mismatch: {grid_cursor}/{len(crop_token_counts)}.")
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

    def _active_twig_layers_module(self):
        module = self.twig_layers
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

    def _run_twig_branch_scores(
        self,
        *,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run the TwigVLM-style auxiliary branch and return attention scores.

        This follows the original TwigVLM prefill logic: clone the hidden states
        at the exit layer, run `T` extra decoder layers, use the final twig
        layer's attention map as the token-importance signal, then discard the
        twig hidden states and continue the main model from the original exit
        hidden states.
        """
        language_model = self.base_model.model.language_model
        branch_states = hidden_states
        attention_mask = attention_mask.to(branch_states.device) if attention_mask is not None else None
        if position_ids is None:
            cache_position = torch.arange(branch_states.shape[1], device=branch_states.device)
            position_ids = cache_position.view(1, 1, -1).expand(3, branch_states.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        position_ids = position_ids.to(branch_states.device)
        cache_position = torch.arange(branch_states.shape[1], device=branch_states.device)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            rope_position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]
            rope_position_ids = position_ids

        twig_attention_mask = attention_mask
        if twig_attention_mask is None:
            twig_attention_mask = torch.ones(
                (branch_states.shape[0], branch_states.shape[1]),
                dtype=torch.bool,
                device=branch_states.device,
            )
        position_embeddings = language_model.rotary_emb(branch_states, rope_position_ids)
        last_attention = None
        active_twig_layers = self._active_twig_layers_module()
        twig_config = getattr(active_twig_layers, "config", language_model.config)
        twig_has_sliding_layers = bool(getattr(active_twig_layers, "has_sliding_layers", getattr(language_model, "has_sliding_layers", False)))
        twig_attention_mask_4d = create_causal_mask(
            config=twig_config,
            input_embeds=branch_states,
            attention_mask=twig_attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        causal_mask_mapping = {"full_attention": twig_attention_mask_4d}
        if twig_has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(
                config=twig_config,
                input_embeds=branch_states,
                attention_mask=twig_attention_mask,
                cache_position=cache_position,
                past_key_values=None,
                position_ids=text_position_ids,
            )

        for layer_index, decoder_layer in enumerate(active_twig_layers):
            want_attention = layer_index == len(active_twig_layers) - 1
            layer_outputs = decoder_layer(
                branch_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=text_position_ids,
                past_key_value=None,
                output_attentions=want_attention,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            branch_states = layer_outputs[0]
            if want_attention:
                last_attention = layer_outputs[1]

        if last_attention is None:
            return hidden_states.new_zeros(hidden_states.shape[:2])
        if last_attention.ndim == 4:
            attention = last_attention.mean(dim=1)
        elif last_attention.ndim == 3:
            attention = last_attention
        else:
            raise RuntimeError(f"Unexpected TwigMRL attention shape: {tuple(last_attention.shape)}")

        if attention_mask is None:
            last_indices = torch.full((attention.shape[0],), attention.shape[1] - 1, device=attention.device, dtype=torch.long)
        else:
            last_indices = attention_mask.to(dtype=torch.long).sum(dim=1).clamp_min(1) - 1
            last_indices = last_indices.to(device=attention.device)
        batch_indices = torch.arange(attention.shape[0], device=attention.device)
        scores = attention[batch_indices, last_indices, :]
        return scores.to(dtype=hidden_states.dtype)

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
            raise RuntimeError(f"TwigMRL image embed mismatch: placeholders={expected} visual_embeds={actual}.")
        return inputs_embeds.masked_scatter(image_mask, image_embeds)

    def _text_context(self, hidden_states: torch.Tensor, input_ids: torch.LongTensor, attention_mask: torch.Tensor) -> torch.Tensor:
        active = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
        text_mask = active & input_ids.to(hidden_states.device).ne(int(self.config.image_token_id))
        text_mask = torch.where(text_mask.sum(dim=1).eq(0).unsqueeze(1), active, text_mask)
        denom = text_mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=hidden_states.dtype)
        return (hidden_states * text_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)).sum(dim=1) / denom

    def _build_twigmrl_masks(
        self,
        *,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        stage_map: torch.LongTensor,
        crop_map: torch.LongTensor,
        selection_scores: torch.Tensor,
        ste: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate = hidden_states.new_ones(hidden_states.shape[:2])
        hard_keep = attention_mask.to(device=hidden_states.device, dtype=torch.bool).clone()
        stats = {
            "mode": self.twigmrl_mode,
            "exit_layer": self.twigmrl_exit_layer,
            "stage_tokens": [0 for _ in self.stage_specs],
            "stage_kept": [0 for _ in self.stage_specs],
            "stage_mask_sum": [0.0 for _ in self.stage_specs],
            "stage_crops": [0 for _ in self.stage_specs],
            "score_source": "twig_layers_attention",
            "twig_depth": self.twigmrl_twig_depth,
        }
        if not torch.any(stage_map >= 0):
            self._last_twigmrl_stats = stats
            return gate, hard_keep

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
                    selection_scores[batch_index, positions],
                    stage_index=stage_index,
                    dtype=hidden_states.dtype,
                    ste=ste,
                )
                gate[batch_index, positions] = crop_gate
                hard_keep[batch_index, positions] = crop_keep.to(device=hard_keep.device)
                stats["stage_tokens"][stage_index] += int(crop_stats["tokens"])
                stats["stage_kept"][stage_index] += int(crop_stats["kept"])
                stats["stage_mask_sum"][stage_index] += float(crop_stats["mask_sum"])
                stats["stage_crops"][stage_index] += 1
        stats["hard_keep_tokens"] = [int(value) for value in hard_keep.sum(dim=1).detach().cpu().tolist()]
        self._last_twigmrl_stats = stats
        return gate, hard_keep

    def _prune_sequence(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        hard_keep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, _seq_len, hidden_size = hidden_states.shape
        lengths = hard_keep.sum(dim=1)
        max_len = int(lengths.max().item())
        pruned_hidden = hidden_states.new_zeros((batch_size, max_len, hidden_size))
        pruned_attention = attention_mask.new_zeros((batch_size, max_len))
        pruned_position = position_ids.new_zeros((position_ids.shape[0], batch_size, max_len))
        for batch_index in range(batch_size):
            keep_indices = torch.where(hard_keep[batch_index])[0]
            length = int(keep_indices.numel())
            pruned_hidden[batch_index, :length] = hidden_states[batch_index, keep_indices]
            pruned_attention[batch_index, :length] = 1
            pruned_position[:, batch_index, :length] = position_ids[:, batch_index, keep_indices]
        return pruned_hidden, pruned_attention, pruned_position

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
        exit_layer = max(0, min(self.twigmrl_exit_layer, len(self.base_model.model.language_model.layers)))
        hidden_states = self._run_language_layers(
            hidden_states=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            start_layer=0,
            end_layer=exit_layer,
            apply_norm=False,
        )

        active_attention_mask = attention_mask
        active_position_ids = position_ids
        if has_images and exit_layer < len(self.base_model.model.language_model.layers):
            stage_map, crop_map = self._stage_and_crop_maps(input_ids=input_ids, image_grid_thw=active_image_grid_thw)
            stage_map = stage_map.to(hidden_states.device)
            crop_map = crop_map.to(hidden_states.device)
            selection_scores = self._run_twig_branch_scores(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            use_true_prune = self.twigmrl_mode == "prune" and not self.training
            gate, hard_keep = self._build_twigmrl_masks(
                hidden_states=hidden_states,
                input_ids=input_ids,
                attention_mask=attention_mask,
                stage_map=stage_map,
                crop_map=crop_map,
                selection_scores=selection_scores,
                ste=use_true_prune,
            )
            hidden_states = hidden_states * gate.unsqueeze(-1).to(dtype=hidden_states.dtype)
            if use_true_prune:
                hidden_states, active_attention_mask, active_position_ids = self._prune_sequence(
                    hidden_states,
                    attention_mask,
                    position_ids,
                    hard_keep,
                )
        else:
            self._last_twigmrl_stats = None

        hidden_states = self._run_language_layers(
            hidden_states=hidden_states,
            attention_mask=active_attention_mask,
            position_ids=active_position_ids,
            start_layer=exit_layer,
            end_layer=len(self.base_model.model.language_model.layers),
            apply_norm=True,
        )
        proj = self.base_model.custom_text_proj(hidden_states)
        proj = proj / proj.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        output_mask = active_attention_mask.to(device=proj.device, dtype=proj.dtype)
        return proj * output_mask.unsqueeze(-1), active_attention_mask

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

    def twigmrl_selector_state_dict(self) -> dict:
        module = self._active_twig_selector_module()
        selector_state = {key: value.detach().cpu() for key, value in module.state_dict().items()}
        twig_layers_module = self._active_twig_layers_module()
        twig_layers_state = {key: value.detach().cpu() for key, value in twig_layers_module.state_dict().items()}
        return {
            "state_dict": selector_state,
            "selector_state_dict": selector_state,
            "twig_layers_state_dict": twig_layers_state,
            "config": {
                "mode": self.twigmrl_mode,
                "exit_layer": self.twigmrl_exit_layer,
                "twig_depth": self.twigmrl_twig_depth,
                "init_from_backbone": True,
                "source_layer_range": [self.twigmrl_exit_layer, self.twigmrl_exit_layer + self.twigmrl_twig_depth],
                **module.selector_config(),
            },
        }

    def save_twigmrl_state(self, save_dir: str | Path) -> None:
        save_path = Path(save_dir) / "twigmrl_selector.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.twigmrl_selector_state_dict(), save_path)

    def load_twigmrl_state(self, path: str | Path, *, map_location: str | torch.device = "cpu") -> None:
        path = Path(path)
        if path.is_dir():
            path = path / "twigmrl_selector.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        state = torch.load(path, map_location=map_location)
        if isinstance(state, dict):
            selector_state = state.get("selector_state_dict", state.get("state_dict", state.get("selector", state)))
            twig_layers_state = state.get("twig_layers_state_dict")
        else:
            selector_state = state
            twig_layers_state = None
        self._active_twig_selector_module().load_state_dict(selector_state, strict=True)
        if twig_layers_state is not None:
            self._active_twig_layers_module().load_state_dict(twig_layers_state, strict=True)

    def save_pretrained(self, save_dir: str, **kwargs):
        self.base_model.save_pretrained(save_dir, **kwargs)
        self.save_twigmrl_state(save_dir)


def _find_twigmrl_model(model) -> TwigMRLColQwen2_5:
    if isinstance(model, TwigMRLColQwen2_5):
        return model
    if hasattr(model, "modules"):
        for module in model.modules():
            if isinstance(module, TwigMRLColQwen2_5):
                return module
    raise TypeError(f"Could not find TwigMRLColQwen2_5 inside {type(model)!r}.")


def save_twigmrl_state(model, save_dir: str | Path) -> None:
    _find_twigmrl_model(model).save_twigmrl_state(save_dir)


def load_twigmrl_state(model, path: str | Path, *, map_location: str | torch.device = "cpu") -> None:
    _find_twigmrl_model(model).load_twigmrl_state(path, map_location=map_location)


def build_twigmrl_model(
    model_name_or_path: str,
    *,
    granularities: Sequence[int] = (1, 2, 4),
    attn_implementation: Optional[str] = "flash_attention_2",
    use_liger_kernel: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
    adapter_path: Optional[str] = None,
    twigmrl_state_path: Optional[str] = None,
    eval_mode: bool = False,
    compact_query_tokens: bool = True,
    twigmrl_mode: str = "mask",
    twigmrl_exit_layer: int = 2,
    twigmrl_twig_depth: int = 3,
    twigmrl_keep_ratios: Optional[Sequence[float]] = None,
    twigmrl_temperature: float = 0.1,
    twigmrl_min_mask_value: float = 0.0,
    twigmrl_train_prune: bool = False,
    twigmrl_use_context: bool = True,
):
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
    model = TwigMRLColQwen2_5(
        base_model=base_model,
        granularities=normalize_granularities(granularities),
        compact_query_tokens=compact_query_tokens,
        twigmrl_mode=twigmrl_mode,
        twigmrl_exit_layer=twigmrl_exit_layer,
        twigmrl_twig_depth=twigmrl_twig_depth,
        twigmrl_keep_ratios=twigmrl_keep_ratios,
        twigmrl_temperature=twigmrl_temperature,
        twigmrl_min_mask_value=twigmrl_min_mask_value,
        twigmrl_train_prune=twigmrl_train_prune,
        twigmrl_use_context=twigmrl_use_context,
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, Path(adapter_path))
        if twigmrl_state_path is None:
            candidate = Path(adapter_path) / "twigmrl_selector.pt"
            if candidate.exists():
                twigmrl_state_path = str(candidate)
    if twigmrl_state_path is not None:
        load_twigmrl_state(model, twigmrl_state_path, map_location="cpu")
    if eval_mode:
        model.eval()
    return model
