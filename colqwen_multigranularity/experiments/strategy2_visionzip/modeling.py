from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from peft import PeftModel

from colpali_engine.models import ColQwen2_5
from colqwen_multigranularity.core import (
    MRLColQwen2_5,
    _apply_compat_patch,
    build_stage_specs,
    normalize_granularities,
)

from .compression import VisionZipCompressor, VisionZipConfig


class VisionZipColQwen2_5(MRLColQwen2_5):  # noqa: N801
    def __init__(
        self,
        base_model: ColQwen2_5,
        *,
        granularities: Sequence[int] = (1, 2, 4),
        compact_query_tokens: bool = True,
        strategy2_visionzip_config: Optional[VisionZipConfig] = None,
    ) -> None:
        super().__init__(
            base_model=base_model,
            granularities=granularities,
            compact_query_tokens=compact_query_tokens,
        )
        if len(self.stage_specs) != 3:
            raise ValueError("VisionZip experiment expects exactly g1/g2/g3.")
        hidden_size = int(self.base_model.model.config.hidden_size)
        self.strategy2_visionzip_config = strategy2_visionzip_config or VisionZipConfig(enabled=False)
        self.strategy2_visionzip = VisionZipCompressor(
            self.strategy2_visionzip_config,
            hidden_size=hidden_size,
            crop_counts=[spec.crop_count for spec in self.stage_specs],
            spatial_merge_size=self.spatial_merge_size,
        )
        self._visionzip_last_saliency: Optional[torch.Tensor] = None

    @staticmethod
    def _has_images(pixel_values: Optional[torch.Tensor], image_grid_thw: Optional[torch.Tensor]) -> bool:
        return (
            pixel_values is not None
            and image_grid_thw is not None
            and getattr(pixel_values, "numel", lambda: 0)() > 0
            and getattr(image_grid_thw, "numel", lambda: 0)() > 0
        )

    def _rows_with_image_placeholders(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor) -> torch.Tensor:
        valid = attention_mask.to(dtype=torch.bool)
        image_token_id = int(self.config.image_token_id)
        return (input_ids.eq(image_token_id) & valid).any(dim=1)

    def _can_use_strategy2_visionzip(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        image_grid_thw: Optional[torch.LongTensor],
    ) -> bool:
        if image_grid_thw is None or image_grid_thw.ndim != 2:
            return False
        rows_with_images = self._rows_with_image_placeholders(input_ids, attention_mask)
        row_count = int(rows_with_images.sum().item())
        if row_count == 0:
            return False
        return int(image_grid_thw.shape[0]) == row_count * int(self.total_crops)

    @staticmethod
    def _pad_1d(sequences: Sequence[torch.Tensor], *, pad_value: int, dtype: torch.dtype) -> torch.Tensor:
        max_len = max(sequence.shape[0] for sequence in sequences)
        output = torch.full((len(sequences), max_len), pad_value, dtype=dtype, device=sequences[0].device)
        for index, sequence in enumerate(sequences):
            output[index, : sequence.shape[0]] = sequence
        return output

    @staticmethod
    def _pad_2d(sequences: Sequence[torch.Tensor], *, hidden_size: int) -> torch.Tensor:
        max_len = max(sequence.shape[0] for sequence in sequences)
        output = sequences[0].new_zeros((len(sequences), max_len, hidden_size))
        for index, sequence in enumerate(sequences):
            output[index, : sequence.shape[0]] = sequence
        return output

    def _stage_grid(self, length: int, *, device: torch.device) -> torch.Tensor:
        merge = int(self.spatial_merge_size)
        return torch.tensor([1, merge, int(length) * merge], dtype=torch.long, device=device)

    def _conditional_zero_anchor(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        anchor = torch.zeros((), device=device, dtype=dtype)
        if not self.training:
            return anchor
        for parameter in self.base_model.visual.parameters():
            if parameter.requires_grad:
                anchor = anchor + parameter.to(device=device, dtype=dtype).sum() * 0.0
        return anchor

    def _resolve_visual_attn_layer(self) -> int:
        blocks = getattr(self.base_model.visual, "blocks", None)
        if blocks is None:
            raise RuntimeError("base_model.visual.blocks not found; cannot collect vision attention.")
        layer_count = len(blocks)
        layer = int(self.strategy2_visionzip_config.visual_attn_layer)
        if layer < 0:
            layer = layer_count + layer
        if layer < 0 or layer >= layer_count:
            raise ValueError(f"visual_attn_layer out of range: {self.strategy2_visionzip_config.visual_attn_layer} for {layer_count} blocks.")
        return layer

    def _visual_forward_with_attention_saliency(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        visual = self.base_model.visual
        blocks = getattr(visual, "blocks", None)
        if blocks is None:
            raise RuntimeError("base_model.visual.blocks not found; cannot collect vision attention.")
        layer_index = self._resolve_visual_attn_layer()
        attn_module = blocks[layer_index].attn
        attn_impl = getattr(getattr(attn_module, "config", None), "_attn_implementation", None)
        if attn_impl != "eager":
            raise RuntimeError(
                "attention_source='visual_attn' requires attn_implementation='eager' for Qwen2.5-VL vision attention; "
                f"got {attn_impl!r}."
            )

        captured: dict[str, torch.Tensor] = {}
        old_forward = attn_module.forward

        def patched_forward(hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, **kwargs):
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision

            seq_length = hidden_states.shape[0]
            query_states, key_states, value_states = (
                attn_module.qkv(hidden_states).reshape(seq_length, 3, attn_module.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
            )
            if position_embeddings is None:
                emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
                cos = emb.cos()
                sin = emb.sin()
            else:
                cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
            query_states = query_states.transpose(0, 1).unsqueeze(0)
            key_states = key_states.transpose(0, 1).unsqueeze(0)
            value_states = value_states.transpose(0, 1).unsqueeze(0)

            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            q_splits = torch.split(query_states, lengths.tolist(), dim=2)
            k_splits = torch.split(key_states, lengths.tolist(), dim=2)
            v_splits = torch.split(value_states, lengths.tolist(), dim=2)
            outputs = []
            saliency_chunks = []
            for q_i, k_i, v_i in zip(q_splits, k_splits, v_splits):
                raw_scores = torch.matmul(q_i, k_i.transpose(2, 3)) * attn_module.scaling
                attn_probs = F.softmax(raw_scores, dim=-1, dtype=torch.float32).to(q_i.dtype)
                attn_probs = F.dropout(attn_probs, p=0.0 if not attn_module.training else attn_module.attention_dropout, training=attn_module.training)
                outputs.append(torch.matmul(attn_probs, v_i).transpose(1, 2).contiguous())
                # Mean over batch, heads, and query positions: tokens most attended by other visual tokens.
                saliency_chunks.append(attn_probs.detach().to(dtype=torch.float32).mean(dim=(0, 1, 2)))
            captured["patch_saliency"] = torch.cat(saliency_chunks, dim=0)
            attn_output = torch.cat(outputs, dim=1)
            attn_output = attn_output.reshape(seq_length, -1).contiguous()
            return attn_module.proj(attn_output)

        attn_module.forward = patched_forward
        try:
            image_embeds = visual(pixel_values, grid_thw=image_grid_thw)
        finally:
            attn_module.forward = old_forward

        patch_saliency = captured.get("patch_saliency")
        if patch_saliency is None:
            raise RuntimeError("Failed to capture vision attention saliency.")

        patch_count = int(patch_saliency.shape[0])
        merge_unit = int(getattr(visual, "spatial_merge_unit", int(self.spatial_merge_size) * int(self.spatial_merge_size)))
        if patch_count % merge_unit != 0:
            raise ValueError(f"Captured patch saliency length {patch_count} is not divisible by spatial_merge_unit={merge_unit}.")
        window_index, _ = visual.get_window_index(image_grid_thw)
        reverse_indices = torch.argsort(window_index)
        merged_saliency = patch_saliency.to(device=image_embeds.device).view(-1, merge_unit).mean(dim=1)
        merged_saliency = merged_saliency.index_select(0, reverse_indices.to(device=image_embeds.device))
        if merged_saliency.shape[0] != image_embeds.shape[0]:
            raise ValueError(f"Vision attention saliency length {merged_saliency.shape[0]} does not match image_embeds {image_embeds.shape[0]}.")
        return image_embeds, merged_saliency.to(device=image_embeds.device)

    def _encode_visual_for_visionzip(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.LongTensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if str(self.strategy2_visionzip_config.attention_source).lower() == "visual_attn":
            return self._visual_forward_with_attention_saliency(pixel_values, image_grid_thw)
        return self.base_model.visual(pixel_values, grid_thw=image_grid_thw), None

    def _rebuild_compact_multimodal_sequence(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
        image_grid_thw: torch.LongTensor,
        visual_saliency: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor, torch.LongTensor]:
        image_token_id = int(self.config.image_token_id)
        vision_start_token_id = int(self.config.vision_start_token_id)
        vision_end_token_id = int(self.config.vision_end_token_id)
        pad_token_id = int(getattr(self.config, "pad_token_id", 0) or 0)
        total_crops = self.total_crops
        grid_cursor = 0
        embed_cursor = 0
        rebuilt_ids: List[torch.Tensor] = []
        rebuilt_attn: List[torch.Tensor] = []
        rebuilt_embeds: List[torch.Tensor] = []
        rebuilt_grids: List[torch.Tensor] = []

        for row_ids, row_attn, row_embeds in zip(input_ids, attention_mask, inputs_embeds):
            valid = row_attn.to(dtype=torch.bool)
            if int((row_ids.eq(image_token_id) & valid).sum().item()) == 0:
                ids = row_ids[valid]
                embeds = row_embeds[valid]
                if ids.numel() == 0:
                    ids = row_ids.new_full((1,), pad_token_id)
                    embeds = row_embeds.new_zeros((1, row_embeds.shape[-1]))
                rebuilt_ids.append(ids)
                rebuilt_attn.append(torch.ones_like(ids, dtype=row_attn.dtype))
                rebuilt_embeds.append(embeds)
                continue

            row_grid = image_grid_thw[grid_cursor : grid_cursor + total_crops]
            if row_grid.shape[0] != total_crops:
                raise ValueError(f"Not enough image_grid_thw rows for sample: need {total_crops}, got {row_grid.shape[0]}.")
            raw_patch_tokens = row_grid.prod(dim=1)
            raw_visual_tokens = int(
                torch.div(
                    raw_patch_tokens,
                    int(self.spatial_merge_size) * int(self.spatial_merge_size),
                    rounding_mode="floor",
                ).sum().item()
            )
            row_image_embeds = image_embeds[embed_cursor : embed_cursor + raw_visual_tokens]
            if row_image_embeds.shape[0] != raw_visual_tokens:
                raise ValueError(f"Not enough image embeds for sample: need {raw_visual_tokens}, got {row_image_embeds.shape[0]}.")
            row_saliency = None
            if visual_saliency is not None:
                row_saliency = visual_saliency[embed_cursor : embed_cursor + raw_visual_tokens]
                if row_saliency.shape[0] != raw_visual_tokens:
                    raise ValueError(f"Not enough visual saliency for sample: need {raw_visual_tokens}, got {row_saliency.shape[0]}.")
            c1, c2, c3 = self.strategy2_visionzip.compress(row_image_embeds, row_grid, saliency=row_saliency)

            text_mask = (
                valid
                & ~row_ids.eq(image_token_id)
                & ~row_ids.eq(vision_start_token_id)
                & ~row_ids.eq(vision_end_token_id)
            )
            text_ids = row_ids[text_mask]
            text_embeds = row_embeds[text_mask]
            segment_ids: List[torch.Tensor] = [text_ids]
            segment_embeds: List[torch.Tensor] = [text_embeds]

            for stage_tokens in (c1, c2, c3):
                if stage_tokens.shape[0] == 0:
                    continue
                stage_ids = torch.cat(
                    [
                        row_ids.new_full((1,), vision_start_token_id),
                        row_ids.new_full((stage_tokens.shape[0],), image_token_id),
                        row_ids.new_full((1,), vision_end_token_id),
                    ],
                    dim=0,
                )
                stage_embeds = self.base_model._embed_tokens(stage_ids)
                stage_embeds = stage_embeds.to(dtype=row_embeds.dtype, device=row_embeds.device)
                stage_embeds[1:-1] = stage_tokens.to(dtype=row_embeds.dtype, device=row_embeds.device)
                segment_ids.append(stage_ids)
                segment_embeds.append(stage_embeds)
                rebuilt_grids.append(self._stage_grid(stage_tokens.shape[0], device=input_ids.device))

            ids = torch.cat(segment_ids, dim=0)
            embeds = torch.cat(segment_embeds, dim=0)
            if ids.numel() == 0:
                ids = row_ids.new_full((1,), pad_token_id)
                embeds = row_embeds.new_zeros((1, row_embeds.shape[-1]))
            rebuilt_ids.append(ids)
            rebuilt_attn.append(torch.ones_like(ids, dtype=row_attn.dtype))
            rebuilt_embeds.append(embeds)
            grid_cursor += total_crops
            embed_cursor += raw_visual_tokens

        if embed_cursor != image_embeds.shape[0]:
            raise ValueError(f"Consumed {embed_cursor} image embeds but visual encoder returned {image_embeds.shape[0]}.")

        compact_input_ids = self._pad_1d(rebuilt_ids, pad_value=pad_token_id, dtype=input_ids.dtype)
        compact_attention_mask = self._pad_1d(rebuilt_attn, pad_value=0, dtype=attention_mask.dtype)
        compact_inputs_embeds = self._pad_2d(rebuilt_embeds, hidden_size=inputs_embeds.shape[-1])
        compact_image_grid_thw = torch.stack(rebuilt_grids, dim=0) if rebuilt_grids else image_grid_thw.new_zeros((0, 3))
        return compact_input_ids, compact_attention_mask, compact_inputs_embeds, compact_image_grid_thw

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
        kwargs.pop("is_query", None)
        kwargs.pop("has_images", None)
        kwargs.pop("inputs_embeds", None)
        kwargs.pop("position_ids", None)

        has_images = self._has_images(pixel_values, image_grid_thw)
        can_use_strategy2_visionzip = (
            has_images
            and self.strategy2_visionzip_config.enabled
            and self._can_use_strategy2_visionzip(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grid_thw=image_grid_thw,
            )
        )
        forward_has_images = has_images and (
            can_use_strategy2_visionzip or bool(self._rows_with_image_placeholders(input_ids, attention_mask).any().item())
        )
        if can_use_strategy2_visionzip:
            inputs_embeds = self.base_model._embed_tokens(input_ids)
            pixel_values = pixel_values.type(self.base_model.visual.dtype)
            image_embeds, visual_saliency = self._encode_visual_for_visionzip(pixel_values, image_grid_thw)
            image_embeds = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            if visual_saliency is not None:
                visual_saliency = visual_saliency.to(device=inputs_embeds.device)
            (
                compact_input_ids,
                compact_attention_mask,
                compact_inputs_embeds,
                compact_image_grid_thw,
            ) = self._rebuild_compact_multimodal_sequence(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
                visual_saliency=visual_saliency,
            )
            compact_grid_for_rope = compact_image_grid_thw if compact_image_grid_thw.numel() > 0 else None
            position_ids, _ = self.base_model.get_rope_index(
                input_ids=compact_input_ids,
                image_grid_thw=compact_grid_for_rope,
                video_grid_thw=None,
                attention_mask=compact_attention_mask,
            )
            last_hidden_states = self.base_model.inner_forward(
                input_ids=compact_input_ids,
                attention_mask=compact_attention_mask,
                position_ids=position_ids,
                inputs_embeds=compact_inputs_embeds,
                pixel_values=None,
                image_grid_thw=None,
                use_cache=False,
                output_hidden_states=True,
                **kwargs,
            )
            active_attention_mask = compact_attention_mask
        else:
            position_ids, _ = self.base_model.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw if forward_has_images else None,
                video_grid_thw=None,
                attention_mask=attention_mask,
            )
            last_hidden_states = self.base_model.inner_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=pixel_values if forward_has_images else None,
                image_grid_thw=image_grid_thw if forward_has_images else None,
                use_cache=False,
                output_hidden_states=True,
                **kwargs,
            )
            active_attention_mask = attention_mask

        proj = self.base_model.custom_text_proj(last_hidden_states)
        proj = proj / proj.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        proj = proj * active_attention_mask.unsqueeze(-1)
        if not can_use_strategy2_visionzip:
            proj = proj + self._conditional_zero_anchor(device=proj.device, dtype=proj.dtype)
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
        can_use_strategy2_visionzip = (
            has_images
            and self.strategy2_visionzip_config.enabled
            and self._can_use_strategy2_visionzip(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grid_thw=image_grid_thw,
            )
        )
        hidden_states = self._project_hidden_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values if has_images else None,
            image_grid_thw=image_grid_thw if has_images else None,
            **kwargs,
        )
        if can_use_strategy2_visionzip:
            return hidden_states
        if self.compact_query_tokens:
            return self._compact_doc_embeddings(hidden_states, input_ids, attention_mask)
        return hidden_states

    def save_pretrained(self, save_dir: str, **kwargs):
        self.base_model.save_pretrained(save_dir, **kwargs)
        self.strategy2_visionzip.save_pretrained(save_dir)


def build_strategy2_visionzip_model(
    model_name_or_path: str,
    *,
    granularities: Sequence[int] = (1, 2, 4),
    strategy2_visionzip_config: Optional[VisionZipConfig] = None,
    strategy2_visionzip_path: Optional[str] = None,
    attn_implementation: Optional[str] = "flash_attention_2",
    use_liger_kernel: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
    adapter_path: Optional[str] = None,
    eval_mode: bool = False,
    compact_query_tokens: bool = True,
):
    granularities = normalize_granularities(granularities)
    if len(build_stage_specs(granularities)) != 3:
        raise ValueError("VisionZip experiment expects exactly three stages.")
    if strategy2_visionzip_config is None and strategy2_visionzip_path is not None:
        config_path = Path(strategy2_visionzip_path) / "strategy2_visionzip_config.json"
        if config_path.exists():
            strategy2_visionzip_config = VisionZipConfig.from_pretrained(strategy2_visionzip_path)
    if strategy2_visionzip_config is None:
        strategy2_visionzip_config = VisionZipConfig(enabled=False)

    base_model = ColQwen2_5.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        use_cache=False,
        attn_implementation=attn_implementation,
        use_liger_kernel=use_liger_kernel,
    )
    if not hasattr(base_model, "custom_text_proj"):
        raise TypeError(
            "Expected a ColQwen2_5 checkpoint with custom_text_proj, "
            f"got model loaded from {model_name_or_path}."
        )
    _apply_compat_patch(base_model)

    if adapter_path is not None:
        base_model = PeftModel.from_pretrained(base_model, Path(adapter_path))

    model = VisionZipColQwen2_5(
        base_model=base_model,
        granularities=granularities,
        compact_query_tokens=compact_query_tokens,
        strategy2_visionzip_config=strategy2_visionzip_config,
    )
    if eval_mode:
        model.eval()
    return model
