from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
from peft import PeftModel

from colpali_engine.models import ColQwen2_5
from colqwen_multigranularity.core import MRLColQwen2_5, _apply_compat_patch, normalize_granularities

_ANGELSLIM_ROOT = Path("/MURE-V2/code/MetaEmbed/third_party/AngelSlim/angelslim")
_QWEN_PRUNING_CONFIG_DIR = _ANGELSLIM_ROOT.parent / "configs/qwen2_5_vl/pruning"


def _install_angelslim_token_compressor_shims() -> None:
    """Import only AngelSlim token_compressor without triggering top-level Engine imports."""
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen_mod
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if not hasattr(qwen_mod, "BaseModelOutputWithPooling"):
        qwen_mod.BaseModelOutputWithPooling = BaseModelOutputWithPooling
    if not hasattr(qwen_mod, "is_flash_attention_requested"):
        def is_flash_attention_requested(config):
            return getattr(config, "_attn_implementation", None) == "flash_attention_2"
        qwen_mod.is_flash_attention_requested = is_flash_attention_requested

    import transformers.masking_utils as masking_utils
    for fn_name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        fn = getattr(masking_utils, fn_name)
        if not getattr(fn, "_angelslim_compat", False):
            def _make_compat(inner_fn):
                def _compat(*args, **kwargs):
                    if "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
                        kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
                    if "cache_position" not in kwargs:
                        input_embeds = kwargs.get("input_embeds")
                        past_key_values = kwargs.get("past_key_values")
                        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
                        kwargs["cache_position"] = torch.arange(
                            past_seen,
                            past_seen + input_embeds.shape[1],
                            device=input_embeds.device,
                        )
                    return inner_fn(*args, **kwargs)
                _compat._angelslim_compat = True
                return _compat
            setattr(masking_utils, fn_name, _make_compat(fn))

    if not hasattr(ALL_ATTENTION_FUNCTIONS, "get_interface"):
        def get_interface(name, default=None):
            try:
                return ALL_ATTENTION_FUNCTIONS[name]
            except Exception:
                return default
        ALL_ATTENTION_FUNCTIONS.get_interface = get_interface

    package_specs = [
        ("angelslim", _ANGELSLIM_ROOT),
        ("angelslim.compressor", _ANGELSLIM_ROOT / "compressor"),
        ("angelslim.compressor.token_compressor", _ANGELSLIM_ROOT / "compressor/token_compressor"),
    ]
    for name, path in package_specs:
        existing = sys.modules.get(name)
        if existing is not None and hasattr(existing, "__path__"):
            paths = list(existing.__path__)
            if str(path) not in paths:
                existing.__path__.append(str(path))
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module


def _patch_angelslim_qwen_wrapper_api() -> None:
    _install_angelslim_token_compressor_shims()
    from angelslim.compressor.token_compressor.models import qwen2_5_vl as qwen_wrap

    def _text_get_input_embeddings(self):
        embed = getattr(self, "embed_tokens", None)
        if embed is None:
            raise NotImplementedError("Wrapped Qwen2.5-VL text model has no embed_tokens module.")
        return embed

    def _text_set_input_embeddings(self, value):
        self.embed_tokens = value

    qwen_wrap.Prunable_Qwen2_5_VLTextModel.get_input_embeddings = _text_get_input_embeddings
    qwen_wrap.Prunable_Qwen2_5_VLTextModel.set_input_embeddings = _text_set_input_embeddings

    from transformers.modeling_outputs import BaseModelOutputWithPooling

    def _split_visual_pooler(self, vision_outputs, grid_thw):
        split_sizes = (grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        if hasattr(vision_outputs, "pooler_output"):
            pooler = vision_outputs.pooler_output
        else:
            pooler = vision_outputs
            vision_outputs = BaseModelOutputWithPooling(
                last_hidden_state=vision_outputs,
                pooler_output=vision_outputs,
            )
        vision_outputs.pooler_output = torch.split(pooler, split_sizes)
        return vision_outputs

    def _get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_outputs = self.visual(pixel_values, grid_thw=image_grid_thw, **kwargs)
        return _split_visual_pooler(self, vision_outputs, image_grid_thw)

    def _get_video_features(self, pixel_values_videos, video_grid_thw=None, **kwargs):
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        vision_outputs = self.visual(pixel_values_videos, grid_thw=video_grid_thw, **kwargs)
        return _split_visual_pooler(self, vision_outputs, video_grid_thw)

    qwen_wrap.Prunable_Qwen2_5_VLModel.get_image_features = _get_image_features
    qwen_wrap.Prunable_Qwen2_5_VLModel.get_video_features = _get_video_features

    original_llm_attn_init = qwen_wrap.Prunable_Qwen2_5_VLAttention.__init__
    if not getattr(qwen_wrap.Prunable_Qwen2_5_VLAttention.__init__, "_angelslim_qwenpre_rope_compat", False):
        def _llm_attn_init_compat(self, original_module, pruning_config):
            original_llm_attn_init(self, original_module, pruning_config)
            if not hasattr(self.config, "rope_parameters"):
                rope_scaling = getattr(self.config, "rope_scaling", {}) or {}
                section = rope_scaling.get("mrope_section", [16, 24, 24])
                self.config.rope_parameters = {"mrope_section": section}
        _llm_attn_init_compat._angelslim_qwenpre_rope_compat = True
        qwen_wrap.Prunable_Qwen2_5_VLAttention.__init__ = _llm_attn_init_compat


    def _text_forward_compat(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        context=None,
        **kwargs,
    ):
        from transformers.cache_utils import DynamicCache
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
        from transformers.modeling_outputs import BaseModelOutputWithPast
        from angelslim.compressor.token_compressor.base.cache import PruningCache
        from angelslim.compressor.token_compressor.algorithm.utils.utils import get_model_specific_vision_token_ids
        from angelslim.compressor.token_compressor.utils.mask_utils import (
            apply_pruning_mask,
            apply_token_merging,
            compensate_decoding_state,
        )

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            try:
                past_key_values = DynamicCache(config=self.config)
            except TypeError:
                past_key_values = DynamicCache()
        if use_cache and not isinstance(past_key_values, PruningCache):
            past_key_values = PruningCache(config=self.config)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        is_prefill = getattr(past_key_values, "is_prefill", True) if past_key_values else True

        def _has_context_vision_tokens(ctx) -> bool:
            if ctx is None or getattr(ctx, "input_ids", None) is None:
                return False
            vision_token_mask = getattr(ctx, "vision_token_mask", None)
            if vision_token_mask is not None:
                return bool(vision_token_mask.any().item())
            token_ids = get_model_specific_vision_token_ids(ctx)
            input_ids_ctx = ctx.input_ids
            has_any = torch.zeros((), dtype=torch.bool, device=input_ids_ctx.device)
            for token_id in token_ids:
                has_any = has_any | input_ids_ctx.eq(int(token_id)).any()
            return bool(has_any.item())

        has_vision_tokens = _has_context_vision_tokens(context)
        if is_prefill and has_vision_tokens and "global" in self.pruning_fns:
            fn, p = self.pruning_fns["global"]
            res = fn(context, **p)
            if isinstance(res, (torch.Tensor, tuple)):
                update_fn = apply_pruning_mask if isinstance(res, torch.Tensor) else apply_token_merging
                inputs_embeds, position_ids, text_position_ids, attention_mask, cache_position = update_fn(
                    inputs_embeds,
                    *(res if isinstance(res, tuple) else [res]),
                    context,
                    position_ids,
                    text_position_ids,
                    attention_mask,
                    None,
                    stage_key="global",
                    past_key_values=past_key_values,
                )
        elif not is_prefill:
            text_position_ids, cache_position, _ = compensate_decoding_state(
                text_position_ids, None, None, "global", past_key_values
            )

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers):
            curr_mask = causal_mask_mapping[self.config.layer_types[layer_idx]]
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=curr_mask,
                position_embeddings=position_embeddings,
                position_ids=text_position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                context=context,
                **kwargs,
            )
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

            if is_prefill and has_vision_tokens and layer_idx in self.pruning_fns:
                fn, p = self.pruning_fns[layer_idx]
                res = fn(context, **p)
                if isinstance(res, (torch.Tensor, tuple)):
                    update_fn = apply_pruning_mask if isinstance(res, torch.Tensor) else apply_token_merging
                    hidden_states, position_ids, text_position_ids, curr_mask, cache_position = update_fn(
                        hidden_states,
                        *(res if isinstance(res, tuple) else [res]),
                        context,
                        position_ids,
                        text_position_ids,
                        curr_mask,
                        None,
                        stage_key=layer_idx,
                        past_key_values=past_key_values,
                    )
                    m = res if isinstance(res, torch.Tensor) else res[-1]
                    cos, sin = position_embeddings
                    m = m.view(-1)
                    position_embeddings = (cos[:, :, m, :], sin[:, :, m, :])
                    causal_mask_mapping[self.config.layer_types[layer_idx]] = curr_mask
            elif not is_prefill:
                text_position_ids, cache_position, curr_mask = compensate_decoding_state(
                    text_position_ids, None, curr_mask, layer_idx, past_key_values
                )
                causal_mask_mapping[self.config.layer_types[layer_idx]] = curr_mask

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)

    qwen_wrap.Prunable_Qwen2_5_VLTextModel.forward = _text_forward_compat


def _patch_angelslim_multimage_segmentation() -> None:
    _install_angelslim_token_compressor_shims()
    from angelslim.compressor.token_compressor.algorithm.utils import merging_utils
    from angelslim.compressor.token_compressor.algorithm.utils import utils as alg_utils

    def _expected_counts(context):
        grid = context.image_grid_thw if context.image_grid_thw is not None else context.video_grid_thw
        if grid is None:
            return None
        merge = int(getattr(context, "spatial_merge_size", 2) or 2)
        return ((grid[:, 1] // merge) * (grid[:, 2] // merge)).detach().to("cpu").tolist()

    def _extract_and_validate_vision_token_info(context):
        input_ids = context.input_ids
        if input_ids is None:
            raise ValueError("[TokenCompressor Error] 'input_ids' missing in context.")
        input_ids_single = input_ids.squeeze(0)
        target_ids = alg_utils.get_model_specific_vision_token_ids(context)
        vision_mask = torch.zeros_like(input_ids_single, dtype=torch.bool)
        for token_id in target_ids:
            vision_mask |= input_ids_single == token_id
        vision_indices = torch.where(vision_mask)[0]
        non_vision_indices = torch.where(~vision_mask)[0]
        expected = _expected_counts(context)
        if expected is None or not expected:
            diffs = vision_indices[1:] - vision_indices[:-1] if vision_indices.numel() > 1 else torch.empty(0, device=vision_indices.device)
            splits = torch.where(diffs > 1)[0] + 1
            actual_counts = [len(block) for block in torch.tensor_split(vision_indices, splits.cpu())] if vision_indices.numel() else []
        else:
            if int(vision_indices.numel()) != int(sum(expected)):
                raise ValueError(
                    "[TokenCompressor Error] Strict Check Failed: Token count mismatch. "
                    f"vision_tokens={int(vision_indices.numel())} expected={int(sum(expected))}."
                )
            actual_counts = [int(value) for value in expected]
        context.vision_token_mask = vision_mask
        return vision_indices, non_vision_indices, vision_mask, actual_counts

    def _regroup_tensors_by_count(source_list, target_counts, grid_list=None):
        regrouped, regrouped_grids = [], []
        ptr = 0
        for count in [int(v) for v in target_counts]:
            curr, accumulated = [], 0
            if grid_list is not None:
                regrouped_grids.append(grid_list[ptr])
            while accumulated < count and ptr < len(source_list):
                tensor = source_list[ptr]
                curr.append(tensor)
                accumulated += int(tensor.shape[1])
                ptr += 1
            if accumulated != count:
                raise ValueError(f"[TokenCompressor Error] regroup mismatch: target={count} accumulated={accumulated}.")
            regrouped.append(torch.cat(curr, dim=1))
        return regrouped, (torch.stack(regrouped_grids) if regrouped_grids else None)

    def get_dialogue_masks(context):
        input_ids = context.input_ids
        if input_ids is None:
            raise ValueError("[TokenCompressor Error] 'input_ids' missing in context.")
        input_ids_1d = input_ids.squeeze(0)
        seq_len = int(input_ids_1d.shape[0])
        device = input_ids.device
        _, _, vision_mask, counts = _extract_and_validate_vision_token_info(context)
        sizes, is_vis_list, masks = [], [], []
        vision_positions = torch.where(vision_mask)[0]
        vision_cursor = 0
        pos = 0
        for count in [int(v) for v in counts]:
            if count <= 0:
                continue
            start = int(vision_positions[vision_cursor].item())
            if start > pos:
                sizes.append(start - pos)
                is_vis_list.append(False)
            end = start + count
            sizes.append(count)
            is_vis_list.append(True)
            mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
            mask[0, start:end] = True
            masks.append(mask)
            pos = end
            vision_cursor += count
        if pos < seq_len:
            sizes.append(seq_len - pos)
            is_vis_list.append(False)
        return None, masks, None, sizes, is_vis_list

    alg_utils._extract_and_validate_vision_token_info = _extract_and_validate_vision_token_info
    alg_utils._regroup_tensors_by_count = _regroup_tensors_by_count
    merging_utils.get_dialogue_masks = get_dialogue_masks

    module_names = [
        "basic",
        "attention_based",
        "dart",
        "divprune",
        "hiprune",
        "scope",
        "vispruner",
        "visionselector",
        "idpruner",
    ]
    for module_name in module_names:
        mod_name = f"angelslim.compressor.token_compressor.algorithm.{module_name}"
        module = sys.modules.get(mod_name)
        if module is not None:
            if hasattr(module, "_extract_and_validate_vision_token_info"):
                module._extract_and_validate_vision_token_info = _extract_and_validate_vision_token_info
            if hasattr(module, "_regroup_tensors_by_count"):
                module._regroup_tensors_by_count = _regroup_tensors_by_count


def _load_angelslim_config_and_mapping(config_path: str | Path):
    _install_angelslim_token_compressor_shims()
    from angelslim.compressor.token_compressor.base.config import TokenCompressorConfig
    import yaml

    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return TokenCompressorConfig.from_yaml(str(config_path)), raw["model_mapping"]


def resolve_angelslim_qwen_config(strategy: str, ratio: str | float = "0.9") -> Path:
    strategy = str(strategy).strip().lower().replace("-", "_")
    aliases = {
        "fastv": "fastv",
        "special_token_based_attention": "fastv",
        "baseline": "baseline",
        "random": "random",
        "divprune": "divprune",
        "dart": "dart",
        "hiprune": "hiprune",
        "scope": "scope",
        "visionzip": "visionzip",
        "vispruner": "vispruner",
        "vision_selector": "vision_selector",
        "idpruner": "idpruner",
    }
    if strategy not in aliases:
        raise ValueError(f"Unknown AngelSlim Qwen2.5-VL strategy {strategy!r}; expected one of {sorted(aliases)}.")
    ratio_str = str(ratio).strip()
    if ratio_str.startswith("0."):
        suffix = ratio_str
    else:
        suffix = f"0.{ratio_str}" if ratio_str in {"75", "9", "90"} else ratio_str
    if suffix == "0.90":
        suffix = "0.9"
    path = _QWEN_PRUNING_CONFIG_DIR / f"{aliases[strategy]}_r{suffix}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"AngelSlim config not found: {path}")
    return path


class AngelSlimQwenPreMRLColQwen2_5(MRLColQwen2_5):  # noqa: N801
    """MRL retrieval wrapper that runs AngelSlim's original Qwen2.5-VL token compressor."""

    def __init__(
        self,
        base_model: ColQwen2_5,
        *,
        granularities: Sequence[int] = (1, 2, 4),
        compact_query_tokens: bool = True,
        split_batch_for_angelslim: bool = True,
    ) -> None:
        super().__init__(base_model=base_model, granularities=granularities, compact_query_tokens=compact_query_tokens)
        self.split_batch_for_angelslim = bool(split_batch_for_angelslim)
        self._last_angelslim_stats: Optional[dict] = None

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
        return int(value or 2)

    def _image_grid_token_counts(self, image_grid_thw: torch.LongTensor) -> list[int]:
        merge_size = self._visual_spatial_merge_size()
        denom = merge_size * merge_size
        counts: list[int] = []
        for row in image_grid_thw.detach().to("cpu").tolist():
            t, h, w = [int(v) for v in row]
            total = t * h * w
            if total % denom != 0:
                raise RuntimeError(f"AngelSlim image grid is not divisible by merge size: grid={row} merge_size={merge_size}.")
            counts.append(total // denom)
        return counts

    def _sample_grid_span(self, input_ids: torch.LongTensor, image_grid_thw: torch.LongTensor, batch_index: int) -> tuple[int, int]:
        image_token_count = int(input_ids[batch_index].eq(int(self.config.image_token_id)).sum().item())
        if image_token_count == 0:
            return 0, 0
        counts = self._image_grid_token_counts(image_grid_thw)
        offset = 0
        for prev_index in range(batch_index):
            prev_tokens = int(input_ids[prev_index].eq(int(self.config.image_token_id)).sum().item())
            consumed = 0
            while consumed < prev_tokens and offset < len(counts):
                consumed += counts[offset]
                offset += 1
            if consumed != prev_tokens:
                raise RuntimeError(
                    f"AngelSlim grid split mismatch before sample={batch_index}: placeholders={prev_tokens} consumed={consumed}."
                )
        start = offset
        consumed = 0
        while consumed < image_token_count and offset < len(counts):
            consumed += counts[offset]
            offset += 1
        if consumed != image_token_count:
            raise RuntimeError(
                f"AngelSlim grid split mismatch for sample={batch_index}: placeholders={image_token_count} consumed={consumed}."
            )
        return start, offset

    def _slice_sample_kwargs(
        self,
        *,
        batch_index: int,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.LongTensor],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        sample_kwargs: dict[str, Any] = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and value.shape[:1] == input_ids.shape[:1]:
                sample_kwargs[key] = value[batch_index : batch_index + 1]
            else:
                sample_kwargs[key] = value
        row_mask = attention_mask[batch_index].to(dtype=torch.bool)
        active = row_mask.nonzero(as_tuple=False).squeeze(-1)
        if active.numel() == 0:
            start_pos, end_pos = 0, 1
        else:
            start_pos = int(active[0].item())
            end_pos = int(active[-1].item()) + 1
        sample_kwargs["input_ids"] = input_ids[batch_index : batch_index + 1, start_pos:end_pos]
        sample_kwargs["attention_mask"] = attention_mask.new_ones((1, end_pos - start_pos))
        if pixel_values is not None and image_grid_thw is not None:
            start, end = self._sample_grid_span(input_ids, image_grid_thw, batch_index)
            if end > start:
                grid_slice = image_grid_thw[start:end]
                fine_start = int(image_grid_thw[:start].prod(dim=1).sum().item()) if start > 0 else 0
                fine_end = int(image_grid_thw[:end].prod(dim=1).sum().item())
                sample_kwargs["image_grid_thw"] = grid_slice
                sample_kwargs["pixel_values"] = pixel_values[fine_start:fine_end]
        return sample_kwargs

    def _run_wrapped_qwen_model(self, **model_kwargs) -> torch.Tensor:
        outputs = self.base_model.model(
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
            **model_kwargs,
        )
        return outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

    @staticmethod
    def _pad_hidden_rows(rows: Sequence[torch.Tensor]) -> torch.Tensor:
        max_len = max(row.shape[1] for row in rows)
        hidden_size = rows[0].shape[-1]
        out = rows[0].new_zeros((len(rows), max_len, hidden_size))
        for index, row in enumerate(rows):
            out[index, : row.shape[1]] = row[0]
        return out

    @staticmethod
    def _pad_masks(rows: Sequence[torch.Tensor]) -> torch.Tensor:
        max_len = max(row.shape[1] for row in rows)
        out = rows[0].new_zeros((len(rows), max_len))
        for index, row in enumerate(rows):
            out[index, : row.shape[1]] = 1
        return out

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
        kwargs.pop("is_query", None)
        kwargs.pop("position_ids", None)
        kwargs.pop("inputs_embeds", None)

        has_images = self._has_images(pixel_values, image_grid_thw)
        active_pixel_values = pixel_values if has_images else None
        active_image_grid_thw = image_grid_thw if has_images else None

        if self.split_batch_for_angelslim:
            hidden_rows = []
            mask_rows = []
            for batch_index in range(input_ids.shape[0]):
                sample_kwargs = self._slice_sample_kwargs(
                    batch_index=batch_index,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=active_pixel_values,
                    image_grid_thw=active_image_grid_thw,
                    kwargs=kwargs,
                )
                hidden = self._run_wrapped_qwen_model(**sample_kwargs)
                hidden_rows.append(hidden)
                mask_rows.append(hidden.new_ones((1, hidden.shape[1]), dtype=attention_mask.dtype))
            last_hidden_states = self._pad_hidden_rows(hidden_rows)
            output_mask = self._pad_masks(mask_rows).to(device=last_hidden_states.device, dtype=attention_mask.dtype)
        else:
            model_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            if active_pixel_values is not None:
                model_kwargs["pixel_values"] = active_pixel_values
                model_kwargs["image_grid_thw"] = active_image_grid_thw
            last_hidden_states = self._run_wrapped_qwen_model(**model_kwargs)
            output_mask = last_hidden_states.new_ones((last_hidden_states.shape[0], last_hidden_states.shape[1]), dtype=attention_mask.dtype)

        proj = self.base_model.custom_text_proj(last_hidden_states)
        proj = proj / proj.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        proj = proj * output_mask.to(device=proj.device, dtype=proj.dtype).unsqueeze(-1)
        self._last_angelslim_stats = {
            "input_shape": [int(v) for v in input_ids.shape],
            "output_shape": [int(v) for v in proj.shape],
            "split_batch": bool(self.split_batch_for_angelslim),
        }
        return proj, output_mask

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
        proj, output_mask = self._project_hidden_states_with_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs,
        )
        if self.compact_query_tokens:
            return self._compact_sequences(proj, output_mask.to(dtype=torch.bool))
        return proj


def apply_angelslim_qwenpre_adapter(model: nn.Module, config_path: str | Path) -> nn.Module:
    _install_angelslim_token_compressor_shims()
    _patch_angelslim_qwen_wrapper_api()
    from angelslim.compressor.token_compressor.adapter import UniversalPruningAdapter

    strategy_config, raw_mapping = _load_angelslim_config_and_mapping(config_path)
    _patch_angelslim_multimage_segmentation()
    adapter = UniversalPruningAdapter(model, strategy_config, raw_mapping)
    wrapped = adapter.wrap_model()
    wrapped._angelslim_pruning_adapter = adapter
    wrapped._angelslim_config_path = str(config_path)
    return wrapped


def _load_adapter_with_fallback(model: AngelSlimQwenPreMRLColQwen2_5, adapter_path: Path):
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
    with TemporaryDirectory(prefix="angelslim_qwenpre_eval_adapter_") as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "adapter_config.json").write_text((adapter_path / "adapter_config.json").read_text())
        torch.save(remapped, tmpdir / "adapter_model.bin")
        return PeftModel.from_pretrained(model, tmpdir)


def build_angelslim_qwenpre_model(
    model_name_or_path: str,
    *,
    granularities: Sequence[int] = (1, 2, 4),
    attn_implementation: Optional[str] = "flash_attention_2",
    use_liger_kernel: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
    adapter_path: Optional[str] = None,
    eval_mode: bool = False,
    compact_query_tokens: bool = True,
    angelslim_config_path: Optional[str] = None,
    angelslim_strategy: str = "visionzip",
    angelslim_ratio: str | float = "0.9",
    split_batch_for_angelslim: bool = True,
):
    config_path = Path(angelslim_config_path) if angelslim_config_path else resolve_angelslim_qwen_config(angelslim_strategy, angelslim_ratio)
    granularities = normalize_granularities(granularities)
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
    base_model = apply_angelslim_qwenpre_adapter(base_model, config_path)

    model = AngelSlimQwenPreMRLColQwen2_5(
        base_model=base_model,
        granularities=granularities,
        compact_query_tokens=compact_query_tokens,
        split_batch_for_angelslim=split_batch_for_angelslim,
    )
    if adapter_path is not None:
        model = _load_adapter_with_fallback(model, Path(adapter_path))
    if eval_mode:
        model.eval()
    return model


__all__ = [
    "AngelSlimQwenPreMRLColQwen2_5",
    "apply_angelslim_qwenpre_adapter",
    "build_angelslim_qwenpre_model",
    "resolve_angelslim_qwen_config",
]
