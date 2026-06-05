from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from peft import PeftModel

from colpali_engine.models import ColQwen2_5
from colqwen_multigranularity.core import (
    MRLColQwen2_5,
    _apply_compat_patch,
    build_stage_specs,
    normalize_granularities,
)

from .compression import SoftAssignmentCompressor, SoftAssignmentConfig


class SoftAssignmentColQwen2_5(MRLColQwen2_5):  # noqa: N801
    def __init__(
        self,
        base_model: ColQwen2_5,
        *,
        granularities: Sequence[int] = (1, 2, 4),
        compact_query_tokens: bool = True,
        strategy1_softassign_config: Optional[SoftAssignmentConfig] = None,
    ) -> None:
        super().__init__(
            base_model=base_model,
            granularities=granularities,
            compact_query_tokens=compact_query_tokens,
        )
        if len(self.stage_specs) != 3:
            raise ValueError("Soft Assignment experiment expects exactly g1/g2/g3.")
        hidden_size = int(self.base_model.model.config.hidden_size)
        self.strategy1_softassign_config = strategy1_softassign_config or SoftAssignmentConfig(enabled=False)
        self.strategy1_softassign = SoftAssignmentCompressor(
            self.strategy1_softassign_config,
            hidden_size=hidden_size,
            crop_counts=[spec.crop_count for spec in self.stage_specs],
            spatial_merge_size=self.spatial_merge_size,
        )

    @staticmethod
    def _has_images(pixel_values: Optional[torch.Tensor], image_grid_thw: Optional[torch.Tensor]) -> bool:
        return (
            pixel_values is not None
            and image_grid_thw is not None
            and getattr(pixel_values, "numel", lambda: 0)() > 0
            and getattr(image_grid_thw, "numel", lambda: 0)() > 0
        )

    def _rows_with_image_placeholders(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = attention_mask.to(dtype=torch.bool)
        image_token_id = int(self.config.image_token_id)
        return (input_ids.eq(image_token_id) & valid).any(dim=1)

    def _can_use_strategy1_softassign(
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
        modules = [self.base_model.visual]
        for module in modules:
            for parameter in module.parameters():
                if parameter.requires_grad:
                    anchor = anchor + parameter.to(device=device, dtype=dtype).sum() * 0.0
        return anchor

    def _rebuild_compact_multimodal_sequence(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
        image_grid_thw: torch.LongTensor,
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
                )
                .sum()
                .item()
            )
            row_image_embeds = image_embeds[embed_cursor : embed_cursor + raw_visual_tokens]
            c1, c2, c3 = self.strategy1_softassign(row_image_embeds, row_grid)

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

            ids = torch.cat(segment_ids, dim=0)
            embeds = torch.cat(segment_embeds, dim=0)
            if ids.numel() == 0:
                ids = row_ids.new_full((1,), pad_token_id)
                embeds = row_embeds.new_zeros((1, row_embeds.shape[-1]))
            rebuilt_ids.append(ids)
            rebuilt_attn.append(torch.ones_like(ids, dtype=row_attn.dtype))
            rebuilt_embeds.append(embeds)
            for stage_tokens in (c1, c2, c3):
                if stage_tokens.shape[0] > 0:
                    rebuilt_grids.append(self._stage_grid(stage_tokens.shape[0], device=input_ids.device))
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
        can_use_strategy1_softassign = (
            has_images
            and self.strategy1_softassign_config.enabled
            and self._can_use_strategy1_softassign(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grid_thw=image_grid_thw,
            )
        )
        forward_has_images = has_images and (
            can_use_strategy1_softassign or bool(self._rows_with_image_placeholders(input_ids, attention_mask).any().item())
        )
        if can_use_strategy1_softassign:
            inputs_embeds = self.base_model._embed_tokens(input_ids)
            pixel_values = pixel_values.type(self.base_model.visual.dtype)
            image_embeds = self.base_model.visual(pixel_values, grid_thw=image_grid_thw)
            image_embeds = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
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
        if not can_use_strategy1_softassign:
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
        can_use_strategy1_softassign = (
            has_images
            and self.strategy1_softassign_config.enabled
            and self._can_use_strategy1_softassign(
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
        if can_use_strategy1_softassign:
            return hidden_states
        if self.compact_query_tokens:
            return self._compact_doc_embeddings(hidden_states, input_ids, attention_mask)
        return hidden_states

    def save_pretrained(self, save_dir: str, **kwargs):
        self.base_model.save_pretrained(save_dir, **kwargs)
        self.strategy1_softassign.save_pretrained(save_dir)


def build_strategy1_softassign_model(
    model_name_or_path: str,
    *,
    granularities: Sequence[int] = (1, 2, 4),
    strategy1_softassign_config: Optional[SoftAssignmentConfig] = None,
    strategy1_softassign_path: Optional[str] = None,
    attn_implementation: Optional[str] = "flash_attention_2",
    use_liger_kernel: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
    adapter_path: Optional[str] = None,
    eval_mode: bool = False,
    compact_query_tokens: bool = True,
):
    granularities = normalize_granularities(granularities)
    if len(build_stage_specs(granularities)) != 3:
        raise ValueError("Soft Assignment experiment expects exactly three stages.")
    if strategy1_softassign_config is None and strategy1_softassign_path is not None:
        strategy1_softassign_config = SoftAssignmentConfig.from_pretrained(strategy1_softassign_path)

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

    model = SoftAssignmentColQwen2_5(
        base_model=base_model,
        granularities=granularities,
        compact_query_tokens=compact_query_tokens,
        strategy1_softassign_config=strategy1_softassign_config,
    )
    if strategy1_softassign_path is not None and (
        (Path(strategy1_softassign_path) / "strategy1_softassign.bin").exists()
        or (Path(strategy1_softassign_path) / "soft_assignment.bin").exists()
    ):
        model.strategy1_softassign.load_pretrained_weights(strategy1_softassign_path, map_location="cpu")
    if eval_mode:
        model.eval()
    return model
