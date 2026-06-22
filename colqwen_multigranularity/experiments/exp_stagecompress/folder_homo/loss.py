from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F
from colqwen_multigranularity.experiments.exp_stagecompress.mlppost.loss import StageCompressMRLInBatchNegativeLoss
from .config import FolderHomoConfig


class FolderHomoMRLInBatchNegativeLoss(StageCompressMRLInBatchNegativeLoss):
    def __init__(self, *, image_token_id: int, folder_homo_config: FolderHomoConfig, temperature: float = 0.03, granularities: Sequence[int] = (1, 2, 4), level_weights: Optional[Sequence[float]] = None, normalize_scores: bool = True, use_smooth_max: bool = False, doc_chunk_size: int = 512, pos_aware_negative_filtering: bool = False, max_batch_size: int = 2048, tau: float = 0.1, norm_tol: float = 1e-3, filter_threshold: float = 0.95, filter_factor: float = 0.5, marc_provider: Optional[Callable[[], object]] = None) -> None:
        super().__init__(
            image_token_id=image_token_id,
            compress_config=folder_homo_config,
            temperature=temperature,
            granularities=granularities,
            level_weights=level_weights,
            normalize_scores=normalize_scores,
            use_smooth_max=use_smooth_max,
            doc_chunk_size=doc_chunk_size,
            pos_aware_negative_filtering=pos_aware_negative_filtering,
            max_batch_size=max_batch_size,
            tau=tau,
            norm_tol=norm_tol,
            filter_threshold=filter_threshold,
            filter_factor=filter_factor,
        )
        self.folder_homo_config = folder_homo_config
        self.marc_provider = marc_provider
        self.marc_weight = float(getattr(folder_homo_config, 'marc_weight', 0.0))
        self.marc_beta = float(getattr(folder_homo_config, 'marc_beta', 20.0))
        self.marc_mode = str(getattr(folder_homo_config, 'marc_mode', 'positive')).strip().lower().replace('-', '_')
        self.marc_margin = float(getattr(folder_homo_config, 'marc_margin', 0.02))
        self.marc_tau = max(float(getattr(folder_homo_config, 'marc_tau', 0.05)), 1e-6)
        self.marc_dup_threshold = float(getattr(folder_homo_config, 'marc_dup_threshold', 0.88))
        self.marc_anchor_boost = float(getattr(folder_homo_config, 'marc_anchor_boost', 1.0))
        self.marc_anchor_floor = float(getattr(folder_homo_config, 'marc_anchor_floor', 0.05))

    def _resolve_marc_aux(self):
        if not bool(getattr(self.folder_homo_config, 'marc_enabled', False)):
            return None
        if self.marc_provider is None:
            return None
        return self.marc_provider()

    def _positive_doc_indices(self, batch_size: int, offset: int, device: torch.device) -> torch.Tensor:
        _, pos_idx = self._get_idx(batch_size, offset, device)
        return pos_idx

    @staticmethod
    def _valid_prefix(mask: torch.Tensor, limit: int) -> torch.Tensor:
        if mask.numel() == 0:
            return mask
        return mask[:limit].to(dtype=torch.bool)

    def _use_margin_marc(self) -> bool:
        return self.marc_mode in {'margin', 'v2', 'marc_v2', 'margin_v2', 'margin_aware', 'anchor', 'anchor_balance', 'marc_v3', 'v3'}

    def _use_anchor_balance_marc(self) -> bool:
        return self.marc_mode in {'anchor', 'anchor_balance', 'marc_v3', 'v3'}

    def _valid_tokens(self, row: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = self._valid_prefix(mask, row.shape[0])
        if mask.numel() == 0 or not torch.any(mask):
            return row[:0]
        return row[mask]

    def _negative_rows_for_query(
        self,
        *,
        q_idx: int,
        positive_doc_idx: int,
        doc_embeddings: torch.Tensor,
        doc_masks: torch.Tensor,
        neg_doc_embeddings: Optional[torch.Tensor],
        neg_masks: Optional[torch.Tensor],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        rows: list[tuple[torch.Tensor, torch.Tensor]] = []
        for d_idx in range(doc_embeddings.size(0)):
            if d_idx == positive_doc_idx:
                continue
            rows.append((doc_embeddings[d_idx], doc_masks[d_idx, -1]))
        if neg_doc_embeddings is not None and neg_masks is not None and q_idx < neg_doc_embeddings.size(0):
            rows.append((neg_doc_embeddings[q_idx], neg_masks[q_idx, -1]))
        return rows

    def _hardest_negative_token_scores(self, q: torch.Tensor, negative_rows: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> Optional[torch.Tensor]:
        hardest = None
        for neg_row, neg_mask in negative_rows:
            neg_tokens = self._valid_tokens(neg_row, neg_mask).detach()
            if neg_tokens.numel() == 0:
                continue
            scores = torch.matmul(q.float(), neg_tokens.float().transpose(0, 1))
            max_scores = scores.max(dim=-1).values
            hardest = max_scores if hardest is None else torch.maximum(hardest, max_scores)
        return hardest

    def _source_duplicate_strength(self, source_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if source_tokens.shape[0] <= 1:
            zeros = source_tokens.new_zeros((source_tokens.shape[0],), dtype=torch.float32)
            mask = torch.zeros_like(zeros, dtype=torch.bool)
            return zeros, mask, zeros
        feats = F.normalize(source_tokens.detach().float(), dim=-1, eps=1e-12)
        sims = torch.matmul(feats, feats.transpose(0, 1))
        sims.fill_diagonal_(-1.0)
        max_neighbor = sims.max(dim=-1).values
        denom = max(1.0 - float(self.marc_dup_threshold), 1e-6)
        strength = ((max_neighbor - float(self.marc_dup_threshold)) / denom).clamp_min(0.0).clamp_max(1.0)
        duplicate_mask = max_neighbor >= float(self.marc_dup_threshold)
        return strength.to(source_tokens.device), duplicate_mask.to(source_tokens.device), max_neighbor.to(source_tokens.device)

    def _marc_stage_loss(
        self,
        *,
        query_row: torch.Tensor,
        query_mask: torch.Tensor,
        doc_row: torch.Tensor,
        doc_mask: torch.Tensor,
        stage: dict,
    ) -> Optional[torch.Tensor]:
        source_tokens = stage.get('source_tokens')
        saliency_logits = stage.get('saliency_logits')
        if source_tokens is None or saliency_logits is None or source_tokens.numel() == 0:
            return None

        doc_start = int(stage.get('doc_start', 0))
        doc_end = int(stage.get('doc_end', doc_start))
        if doc_start >= doc_row.shape[0] or doc_end <= doc_start:
            return None
        doc_end = min(doc_end, doc_row.shape[0])
        compressed_tokens = doc_row[doc_start:doc_end]
        compressed_mask = self._valid_prefix(doc_mask[doc_start:doc_end], compressed_tokens.shape[0])
        query_mask = self._valid_prefix(query_mask, query_row.shape[0])
        if compressed_tokens.numel() == 0 or not torch.any(compressed_mask) or not torch.any(query_mask):
            return None

        q = query_row[query_mask].detach()
        c = compressed_tokens[compressed_mask].detach()
        src = source_tokens
        if q.numel() == 0 or c.numel() == 0 or src.numel() == 0:
            return None

        # Only query tokens whose current MaxSim winner falls inside this compressed stage
        # should supervise the stage scorer. This keeps the target aligned with the
        # late-interaction winner-take-all mechanism while still training a query-free
        # document compressor.
        valid_doc_positions = torch.nonzero(doc_mask.to(dtype=torch.bool), as_tuple=False).flatten()
        if valid_doc_positions.numel() == 0:
            return None
        full_doc = doc_row[valid_doc_positions].detach()
        full_sims = torch.matmul(q.float(), full_doc.float().transpose(0, 1))
        winner_positions = valid_doc_positions[full_sims.argmax(dim=-1)]
        stage_query_mask = (winner_positions >= doc_start) & (winner_positions < doc_end)
        if not torch.any(stage_query_mask):
            return None
        q_stage = q[stage_query_mask]
        source_sims = torch.matmul(q_stage.float(), src.detach().float().transpose(0, 1))
        target = torch.softmax(source_sims * float(self.marc_beta), dim=-1).sum(dim=0)
        if not torch.isfinite(target).all() or float(target.sum().detach().item()) <= 1e-8:
            return None
        target = target / target.sum().clamp_min(1e-8)
        pred = F.log_softmax(saliency_logits.float(), dim=0)
        return F.kl_div(pred, target.to(device=pred.device, dtype=pred.dtype), reduction='batchmean')

    def _marc_margin_stage_loss(
        self,
        *,
        query_row: torch.Tensor,
        query_mask: torch.Tensor,
        doc_row: torch.Tensor,
        doc_mask: torch.Tensor,
        stage: dict,
        negative_rows: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ):
        source_tokens = stage.get('source_tokens')
        saliency_logits = stage.get('saliency_logits')
        if source_tokens is None or saliency_logits is None or source_tokens.numel() == 0:
            return None

        doc_start = int(stage.get('doc_start', 0))
        doc_end = int(stage.get('doc_end', doc_start))
        if doc_start >= doc_row.shape[0] or doc_end <= doc_start:
            return None
        doc_end = min(doc_end, doc_row.shape[0])
        query_mask = self._valid_prefix(query_mask, query_row.shape[0])
        if not torch.any(query_mask):
            return None

        q = query_row[query_mask].detach()
        src = source_tokens
        if q.numel() == 0 or src.numel() == 0:
            return None

        valid_doc_positions = torch.nonzero(doc_mask.to(dtype=torch.bool), as_tuple=False).flatten()
        if valid_doc_positions.numel() == 0:
            return None
        full_doc = doc_row[valid_doc_positions].detach()
        pos_sims = torch.matmul(q.float(), full_doc.float().transpose(0, 1))
        pos_scores, pos_winner_local = pos_sims.max(dim=-1)
        winner_positions = valid_doc_positions[pos_winner_local]
        stage_query_mask = (winner_positions >= doc_start) & (winner_positions < doc_end)
        if not torch.any(stage_query_mask):
            return None

        neg_scores = self._hardest_negative_token_scores(q, negative_rows)
        if neg_scores is None:
            return None

        margin_gap = neg_scores + float(self.marc_margin) - pos_scores
        violation = F.softplus(margin_gap / float(self.marc_tau)) * float(self.marc_tau)
        q_stage = q[stage_query_mask]
        weights = violation[stage_query_mask].detach()
        if not torch.isfinite(weights).all() or float(weights.sum().detach().item()) <= 1e-8:
            return None

        source_sims = torch.matmul(q_stage.float(), src.detach().float().transpose(0, 1))
        source_probs = torch.softmax(source_sims * float(self.marc_beta), dim=-1)
        target = (source_probs * weights.float().unsqueeze(-1)).sum(dim=0)
        if not torch.isfinite(target).all() or float(target.sum().detach().item()) <= 1e-8:
            return None
        stats = {
            'margin_violation': weights.mean().detach(),
            'margin_gap': margin_gap[stage_query_mask].mean().detach(),
            'pos_token_score': pos_scores[stage_query_mask].mean().detach(),
            'neg_token_score': neg_scores[stage_query_mask].mean().detach(),
        }
        if self._use_anchor_balance_marc():
            dup_strength, dup_mask, max_neighbor = self._source_duplicate_strength(src)
            dup_strength = dup_strength.to(device=target.device, dtype=target.dtype)
            utility_scale = target.detach() / target.detach().max().clamp_min(1e-8)
            boost = 1.0 + float(self.marc_anchor_boost) * dup_strength * (utility_scale + float(self.marc_anchor_floor))
            target_before = target
            target = target * boost
            target_sum_before = target_before.sum().detach().clamp_min(1e-8)
            target_sum_after = target.sum().detach().clamp_min(1e-8)
            stats.update({
                'anchor_duplicate_fraction': dup_mask.float().mean().detach(),
                'anchor_duplicate_maxsim': max_neighbor[dup_mask].mean().detach() if torch.any(dup_mask) else max_neighbor.new_tensor(0.0),
                'anchor_boost_mean': (boost - 1.0).mean().detach(),
                'anchor_boost_max': (boost - 1.0).max().detach(),
                'anchor_target_duplicate_fraction_before': (target_before[dup_mask].sum().detach() / target_sum_before) if torch.any(dup_mask) else target_before.new_tensor(0.0),
                'anchor_target_duplicate_fraction_after': (target[dup_mask].sum().detach() / target_sum_after) if torch.any(dup_mask) else target.new_tensor(0.0),
            })
        target = target / target.sum().clamp_min(1e-8)

        pred = F.log_softmax(saliency_logits.float(), dim=0)
        loss = F.kl_div(pred, target.to(device=pred.device, dtype=pred.dtype), reduction='sum')
        return loss, stats

    def _marc_aux_loss(
        self,
        *,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        neg_doc_embeddings: Optional[torch.Tensor],
        query_masks: torch.Tensor,
        doc_masks: torch.Tensor,
        neg_masks: Optional[torch.Tensor],
        offset: int,
    ):
        aux = self._resolve_marc_aux()
        if aux is None:
            return None, {}
        rows = aux.get('rows') if isinstance(aux, dict) else None
        if not rows:
            return None, {}

        batch_size = query_embeddings.size(0)
        pos_idx = self._positive_doc_indices(batch_size, offset, query_embeddings.device)
        losses = []
        stat_values: dict[str, list[torch.Tensor]] = {
            'margin_violation': [],
            'margin_gap': [],
            'pos_token_score': [],
            'neg_token_score': [],
        }
        stage_count = 0
        use_margin = self._use_margin_marc()
        for q_idx in range(batch_size):
            global_d_idx = int(pos_idx[q_idx].detach().item())
            local_d_idx = int(q_idx)
            if local_d_idx >= len(rows) or global_d_idx < 0 or global_d_idx >= doc_embeddings.size(0):
                continue
            row_aux = rows[local_d_idx]
            negative_rows = self._negative_rows_for_query(
                q_idx=q_idx,
                positive_doc_idx=global_d_idx,
                doc_embeddings=doc_embeddings,
                doc_masks=doc_masks,
                neg_doc_embeddings=neg_doc_embeddings,
                neg_masks=neg_masks,
            ) if use_margin else ()
            for stage in row_aux.get('stages', []):
                if use_margin:
                    result = self._marc_margin_stage_loss(
                        query_row=query_embeddings[q_idx],
                        query_mask=query_masks[q_idx, -1],
                        doc_row=doc_embeddings[global_d_idx],
                        doc_mask=doc_masks[global_d_idx, -1],
                        stage=stage,
                        negative_rows=negative_rows,
                    )
                    if result is None:
                        continue
                    loss, stage_stats = result
                    for key, value in stage_stats.items():
                        stat_values.setdefault(key, []).append(value)
                else:
                    loss = self._marc_stage_loss(
                        query_row=query_embeddings[q_idx],
                        query_mask=query_masks[q_idx, -1],
                        doc_row=doc_embeddings[global_d_idx],
                        doc_mask=doc_masks[global_d_idx, -1],
                        stage=stage,
                    )
                    if loss is None:
                        continue
                losses.append(loss)
                stage_count += 1
        if not losses:
            return None, {}
        total = torch.stack(losses).mean()
        if use_margin:
            prefix = 'marc_anchor' if self._use_anchor_balance_marc() else 'marc2'
            stats = {
                f'{prefix}_loss': total.detach(),
                f'{prefix}_stage_count': query_embeddings.new_tensor(float(stage_count)),
            }
            for key, values in stat_values.items():
                if values:
                    stats[f'{prefix}_{key}'] = torch.stack(values).mean().detach()
        else:
            stats = {
                'marc_utility': total.detach(),
                'marc_stage_count': query_embeddings.new_tensor(float(stage_count)),
            }
        return total, stats

    def forward(
        self,
        query_embeddings,
        doc_embeddings,
        neg_doc_embeddings=None,
        offset: int = 0,
        query_has_images=None,
        doc_has_images=None,
        neg_doc_has_images=None,
        query_input_ids=None,
        query_attention_mask=None,
        doc_input_ids=None,
        doc_attention_mask=None,
        neg_doc_input_ids=None,
        neg_doc_attention_mask=None,
    ):
        total_loss, loss_stats = super().forward(
            query_embeddings=query_embeddings,
            doc_embeddings=doc_embeddings,
            neg_doc_embeddings=neg_doc_embeddings,
            offset=offset,
            query_has_images=query_has_images,
            doc_has_images=doc_has_images,
            neg_doc_has_images=neg_doc_has_images,
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            doc_input_ids=doc_input_ids,
            doc_attention_mask=doc_attention_mask,
            neg_doc_input_ids=neg_doc_input_ids,
            neg_doc_attention_mask=neg_doc_attention_mask,
        )
        if not bool(getattr(self.folder_homo_config, 'marc_enabled', False)) or self.marc_weight <= 0:
            return total_loss, loss_stats
        if query_input_ids is None or query_attention_mask is None or doc_input_ids is None or doc_attention_mask is None:
            return total_loss, loss_stats
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
            neg_masks = self._build_group_masks(
                input_ids=neg_doc_input_ids,
                attention_mask=neg_doc_attention_mask,
                output_length=neg_doc_embeddings.size(1),
            )
        marc_loss, marc_stats = self._marc_aux_loss(
            query_embeddings=query_embeddings,
            doc_embeddings=doc_embeddings,
            neg_doc_embeddings=neg_doc_embeddings,
            query_masks=query_masks,
            doc_masks=doc_masks,
            neg_masks=neg_masks,
            offset=offset,
        )
        if marc_loss is None:
            return total_loss, loss_stats
        total_loss = total_loss + marc_loss * self.marc_weight
        loss_stats.update(marc_stats)
        weighted = (marc_loss * self.marc_weight).detach()
        if self._use_anchor_balance_marc():
            loss_stats['marc_anchor_weighted'] = weighted
        elif self._use_margin_marc():
            loss_stats['marc2_weighted'] = weighted
        else:
            loss_stats['marc_weighted'] = weighted
        return total_loss, loss_stats
