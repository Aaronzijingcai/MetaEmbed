from __future__ import annotations

from typing import Optional, Sequence

import torch

from colqwen_multigranularity.core import MRLInBatchNegativeLoss, build_stage_specs, normalize_granularities

from .compression import StageCompressConfig


class StageCompressMRLInBatchNegativeLoss(MRLInBatchNegativeLoss):
    def __init__(self, *, image_token_id: int, compress_config: StageCompressConfig, temperature: float = 0.03, granularities: Sequence[int] = (1, 2, 4), level_weights: Optional[Sequence[float]] = None, normalize_scores: bool = True, use_smooth_max: bool = False, doc_chunk_size: int = 512, query_chunk_size: Optional[int] = 512, pos_aware_negative_filtering: bool = False, max_batch_size: int = 2048, tau: float = 0.1, norm_tol: float = 1e-3, filter_threshold: float = 0.95, filter_factor: float = 0.5) -> None:
        granularities = normalize_granularities(granularities)
        specs = build_stage_specs(granularities)
        if len(specs) != 3:
            raise ValueError('StageCompress loss expects exactly g1/g2/g3.')
        self.compress_config = compress_config
        self.active_stage_ids = set(compress_config.active_stage_ids())
        super().__init__(
            image_token_id=image_token_id,
            temperature=temperature,
            granularities=granularities,
            level_weights=level_weights,
            normalize_scores=normalize_scores,
            use_smooth_max=use_smooth_max,
            doc_chunk_size=doc_chunk_size,
            query_chunk_size=query_chunk_size,
            pos_aware_negative_filtering=pos_aware_negative_filtering,
            max_batch_size=max_batch_size,
            tau=tau,
            norm_tol=norm_tol,
            filter_threshold=filter_threshold,
            filter_factor=filter_factor,
        )

    def _stage_lengths(self, total_image: torch.Tensor) -> torch.Tensor:
        organization = getattr(self.compress_config, 'normalized_organization_mode', None)
        if callable(organization) and organization() == 'flat':
            remaining = total_image.to(dtype=torch.long)
            flat_lengths = []
            for budget in self.compress_config.budgets:
                length = torch.minimum(remaining, torch.full_like(remaining, int(budget)))
                flat_lengths.append(length)
                remaining = (remaining - length).clamp_min(0)
            return torch.stack(flat_lengths, dim=1)
        cumulative = self.cumulative_crop_counts.to(device=total_image.device, dtype=torch.float32)
        scaled_end = total_image.to(torch.float32).unsqueeze(1) * cumulative.unsqueeze(0) / float(self.total_crop_count)
        ends = torch.floor(scaled_end).to(torch.long)
        ends = torch.minimum(ends, total_image.unsqueeze(1))
        starts = torch.cat([torch.zeros_like(ends[:, :1]), ends[:, :-1]], dim=1)
        original_lengths = (ends - starts).clamp_min(0)
        included_stage_ids = getattr(self.compress_config, 'included_stage_ids', None)
        if callable(included_stage_ids):
            included = set(included_stage_ids())
            include_mask = torch.tensor(
                [index in included for index in range(3)],
                dtype=torch.bool,
                device=total_image.device,
            ).unsqueeze(0)
            original_lengths = torch.where(include_mask, original_lengths, torch.zeros_like(original_lengths))
        if (not self.compress_config.enabled) or self.compress_config.compress_stages.lower() in {'none', 'off', 'false'}:
            return original_lengths
        budgets = torch.tensor(self.compress_config.budgets, dtype=torch.long, device=total_image.device).unsqueeze(0)
        active = torch.tensor([idx in self.active_stage_ids for idx in range(3)], dtype=torch.bool, device=total_image.device).unsqueeze(0)
        return torch.where(active, torch.minimum(original_lengths, budgets), original_lengths)

    def _build_group_masks(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor, output_length: Optional[int] = None) -> torch.Tensor:
        attn = attention_mask.to(dtype=torch.bool)
        image_mask = input_ids.eq(self.image_token_id) & attn
        text_mask = (~input_ids.eq(self.image_token_id)) & attn
        text_lengths = text_mask.sum(dim=1)
        stage_lengths = self._stage_lengths(image_mask.sum(dim=1))
        level_lengths = torch.stack([
            text_lengths + stage_lengths[:, 0],
            text_lengths + stage_lengths[:, 0] + stage_lengths[:, 1],
            text_lengths + stage_lengths[:, 0] + stage_lengths[:, 1] + stage_lengths[:, 2],
        ], dim=1)
        if output_length is None:
            output_length = int(level_lengths[:, -1].max().item())
        positions = torch.arange(output_length, device=input_ids.device).view(1, 1, -1)
        masks = positions < level_lengths.unsqueeze(-1)
        empty = level_lengths[:, -1].eq(0)
        if empty.any():
            masks = masks.clone()
            masks[empty, :, 0] = True
        return masks


    def forward(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        neg_doc_embeddings: Optional[torch.Tensor] = None,
        offset: int = 0,
        query_has_images: Optional[torch.Tensor] = None,
        doc_has_images: Optional[torch.Tensor] = None,
        neg_doc_has_images: Optional[torch.Tensor] = None,
        query_input_ids: Optional[torch.Tensor] = None,
        query_attention_mask: Optional[torch.Tensor] = None,
        doc_input_ids: Optional[torch.Tensor] = None,
        doc_attention_mask: Optional[torch.Tensor] = None,
        neg_doc_input_ids: Optional[torch.Tensor] = None,
        neg_doc_attention_mask: Optional[torch.Tensor] = None,
    ):
        if query_input_ids is None or query_attention_mask is None:
            raise ValueError("query_input_ids/query_attention_mask are required for group-wise MRL loss.")
        if doc_input_ids is None or doc_attention_mask is None:
            raise ValueError("doc_input_ids/doc_attention_mask are required for group-wise MRL loss.")

        query_lengths = self._valid_lengths(query_embeddings)
        doc_lengths = self._valid_lengths(doc_embeddings)
        query_has_images = self._coerce_bool_mask(query_has_images, query_lengths)
        doc_has_images = self._coerce_bool_mask(doc_has_images, doc_lengths)
        query_masks = self._build_group_masks(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            output_length=query_embeddings.size(1),
        )
        doc_masks = self._build_group_masks(
            input_ids=doc_input_ids,
            attention_mask=doc_attention_mask,
            output_length=doc_embeddings.size(1),
        )

        neg_masks = None
        if neg_doc_embeddings is not None and neg_doc_input_ids is not None and neg_doc_attention_mask is not None:
            neg_lengths = self._valid_lengths(neg_doc_embeddings)
            neg_doc_has_images = self._coerce_bool_mask(neg_doc_has_images, neg_lengths)
            neg_masks = self._build_group_masks(
                input_ids=neg_doc_input_ids,
                attention_mask=neg_doc_attention_mask,
                output_length=neg_doc_embeddings.size(1),
            )

        batch_size = query_embeddings.size(0)
        _, pos_idx = self._get_idx(batch_size, offset, query_embeddings.device)
        pos_doc_has_images = doc_has_images[pos_idx]
        active_levels = self._build_level_activity(
            query_has_images=query_has_images,
            doc_has_images=pos_doc_has_images,
            neg_doc_has_images=neg_doc_has_images,
        )

        total_loss = query_embeddings.new_tensor(0.0)
        loss_stats = {}

        for level_index, (label, weight) in enumerate(zip(self.level_labels, self.level_weights)):
            row_mask = active_levels[:, level_index]
            if not torch.any(row_mask):
                continue
            pos_scores = self._aggregate_masked_scores(
                query_embeddings=query_embeddings,
                doc_embeddings=doc_embeddings,
                query_mask=query_masks[:, level_index],
                doc_mask=doc_masks[:, level_index],
            )

            neg_scores = None
            if neg_doc_embeddings is not None and neg_masks is not None:
                neg_diag_scores = self._aggregate_diagonal_masked_scores(
                    query_embeddings=query_embeddings,
                    doc_embeddings=neg_doc_embeddings,
                    query_mask=query_masks[:, level_index],
                    doc_mask=neg_masks[:, level_index],
                )
                neg_scores = neg_diag_scores.unsqueeze(1)

            level_loss = self._get_loss_from_scores(
                pos_scores=pos_scores,
                neg_scores=neg_scores,
                offset=offset,
                row_mask=row_mask,
            )
            total_loss = total_loss + level_loss * weight
            loss_stats[f"mrl_{label}"] = level_loss.detach()
            loss_stats[f"mrl_active_ratio_{label}"] = row_mask.float().mean().detach()

        loss_stats["mrl_query_has_images_ratio"] = query_has_images.float().mean().detach()
        loss_stats["mrl_doc_has_images_ratio"] = pos_doc_has_images.float().mean().detach()
        if neg_doc_has_images is not None:
            loss_stats["mrl_neg_doc_has_images_ratio"] = neg_doc_has_images.float().mean().detach()
        loss_stats["mrl_text_text_ratio"] = (~active_levels[:, 0]).float().mean().detach()
        return total_loss, loss_stats
