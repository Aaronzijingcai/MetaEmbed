from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from colpali_engine.models import ColQwen2_5

from colqwen_multigranularity.core import MRLColQwen2_5, _apply_compat_patch, build_stage_specs, normalize_granularities
from .config import FolderGainHomoConfig


class HomoPatchScorer(nn.Module):
    def __init__(self, embed_dim: int, *, num_heads: int = 8, dropout: float = 0.1, use_text_context: bool = False) -> None:
        super().__init__()
        self.use_text_context = bool(use_text_context)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.text_proj = nn.Linear(embed_dim, embed_dim) if use_text_context else None
        self.score_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, tokens: torch.Tensor, text_context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = tokens.unsqueeze(0)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm(x + attn_out)
        x = x + self.mlp(x)
        enhanced = x.squeeze(0)
        if self.use_text_context and self.text_proj is not None and text_context is not None and text_context.numel() > 0:
            enhanced = enhanced + self.text_proj(text_context.reshape(1, -1)).expand_as(enhanced)
        saliency = self.score_head(enhanced).squeeze(-1)
        gate = torch.sigmoid(self.gate_head(enhanced).squeeze(-1))
        return enhanced, saliency, gate


class GainHomoFolderBlock(nn.Module):
    def __init__(self, embed_dim: int, budget: int, *, config: FolderGainHomoConfig) -> None:
        super().__init__()
        self.budget = int(budget)
        self.config = config
        self.scorer = HomoPatchScorer(
            embed_dim,
            num_heads=int(config.scorer_heads),
            dropout=float(config.scorer_dropout),
            use_text_context=bool(config.use_text_context),
        )
        self.folder_alpha = float(config.folder_alpha)
        self.novelty_weight = float(config.novelty_weight)
        self.gate_strength = float(config.gate_strength)
        self.coverage_weight = float(config.coverage_weight)
        self.mmr_weight = float(config.mmr_weight)
        self.residual_mass_weight = float(config.residual_mass_weight)

    @staticmethod
    def _normalize_score(score: torch.Tensor) -> torch.Tensor:
        if score.numel() <= 1:
            return torch.ones_like(score)
        lo = score.min()
        hi = score.max()
        return (score - lo) / (hi - lo).clamp_min(1e-6)

    def _anchor_max_similarity(
        self,
        enhanced: torch.Tensor,
        coarse_anchors: Optional[torch.Tensor],
        *,
        token_positions: Optional[torch.Tensor] = None,
        anchor_positions: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if coarse_anchors is None or coarse_anchors.numel() == 0:
            return None
        anchors = coarse_anchors.detach() if self.config.detach_anchors else coarse_anchors
        token_features = F.normalize(enhanced.float(), dim=-1, eps=1e-12)
        anchor_features = F.normalize(anchors.float(), dim=-1, eps=1e-12)
        similarity = token_features @ anchor_features.transpose(0, 1)
        if (
            self.config.uses_geo_alignment()
            and token_positions is not None
            and anchor_positions is not None
            and token_positions.numel() > 0
            and anchor_positions.numel() > 0
        ):
            distances = torch.cdist(token_positions.float(), anchor_positions.float(), p=2)
            aligned = distances <= float(self.config.geo_radius)
            if aligned.shape == similarity.shape:
                no_aligned = ~aligned.any(dim=-1)
                if no_aligned.any():
                    aligned[no_aligned] = True
                similarity = similarity.masked_fill(~aligned, -float('inf'))
        return similarity.max(dim=-1).values

    def _novelty(
        self,
        enhanced: torch.Tensor,
        coarse_anchors: Optional[torch.Tensor],
        *,
        token_positions: Optional[torch.Tensor] = None,
        anchor_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        max_similarity = self._anchor_max_similarity(
            enhanced,
            coarse_anchors,
            token_positions=token_positions,
            anchor_positions=anchor_positions,
        )
        if max_similarity is None:
            return torch.ones(enhanced.shape[0], dtype=enhanced.dtype, device=enhanced.device)
        novelty = 1.0 - max_similarity.clamp(-1.0, 1.0)
        return self._normalize_score(novelty).to(dtype=enhanced.dtype)

    def _coverage_gain(
        self,
        enhanced: torch.Tensor,
        coarse_anchors: Optional[torch.Tensor],
        *,
        token_positions: Optional[torch.Tensor] = None,
        anchor_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (not self.config.uses_coverage_gain()) or coarse_anchors is None or coarse_anchors.numel() == 0:
            return torch.zeros(enhanced.shape[0], dtype=enhanced.dtype, device=enhanced.device)
        with torch.no_grad():
            token_features = F.normalize(enhanced.float(), dim=-1, eps=1e-12)
            local_similarity = (token_features @ token_features.transpose(0, 1)).clamp(-1.0, 1.0)
            local_coverage = 0.5 * (local_similarity + 1.0)
            coarse_similarity = self._anchor_max_similarity(
                enhanced,
                coarse_anchors,
                token_positions=token_positions,
                anchor_positions=anchor_positions,
            )
            if coarse_similarity is None:
                current_coverage = torch.zeros(enhanced.shape[0], dtype=torch.float32, device=enhanced.device)
            else:
                current_coverage = 0.5 * (coarse_similarity.clamp(-1.0, 1.0) + 1.0)
            gains = (local_coverage - current_coverage.unsqueeze(0)).clamp_min(0.0).sum(dim=-1)
        return self._normalize_score(gains).to(dtype=enhanced.dtype)

    def _mmr_diversity(self, enhanced: torch.Tensor) -> torch.Tensor:
        if not self.config.uses_mmr_gain():
            return torch.zeros(enhanced.shape[0], dtype=enhanced.dtype, device=enhanced.device)
        if enhanced.shape[0] <= 1:
            return torch.ones(enhanced.shape[0], dtype=enhanced.dtype, device=enhanced.device)
        with torch.no_grad():
            token_features = F.normalize(enhanced.float(), dim=-1, eps=1e-12)
            similarity = token_features @ token_features.transpose(0, 1)
            similarity.fill_diagonal_(-float('inf'))
            redundancy = similarity.max(dim=-1).values.clamp(-1.0, 1.0)
            diversity = 1.0 - redundancy
        return self._normalize_score(diversity).to(dtype=enhanced.dtype)

    def _gain_terms(
        self,
        tokens: torch.Tensor,
        *,
        coarse_anchors: Optional[torch.Tensor] = None,
        text_context: Optional[torch.Tensor] = None,
        token_positions: Optional[torch.Tensor] = None,
        anchor_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        enhanced, saliency, gate = self.scorer(tokens, text_context=text_context)
        saliency_norm = self._normalize_score(saliency.float()).to(tokens.dtype)
        novelty = self._novelty(
            enhanced,
            coarse_anchors,
            token_positions=token_positions,
            anchor_positions=anchor_positions,
        ).to(tokens.dtype)
        coverage = self._coverage_gain(
            enhanced,
            coarse_anchors,
            token_positions=token_positions,
            anchor_positions=anchor_positions,
        ).to(tokens.dtype)
        mmr_diversity = self._mmr_diversity(enhanced).to(tokens.dtype)
        return enhanced, saliency_norm, novelty, coverage, mmr_diversity, gate

    def estimate_residual_mass(
        self,
        tokens: torch.Tensor,
        *,
        coarse_anchors: Optional[torch.Tensor] = None,
        text_context: Optional[torch.Tensor] = None,
        token_positions: Optional[torch.Tensor] = None,
        anchor_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tokens.numel() == 0:
            return tokens.new_tensor(0.0)
        with torch.no_grad():
            _, saliency_norm, novelty, coverage, _, _ = self._gain_terms(
                tokens,
                coarse_anchors=coarse_anchors,
                text_context=text_context,
                token_positions=token_positions,
                anchor_positions=anchor_positions,
            )
            quality = saliency_norm + self.novelty_weight * novelty + self.coverage_weight * coverage
            k = max(1, min(int(tokens.shape[0]), int(math.ceil(tokens.shape[0] * float(self.config.residual_mass_topk_ratio)))))
            return quality.float().topk(k).values.sum()

    @staticmethod
    def _folder_match(metric: torch.Tensor, protect: torch.Tensor, r: int, alpha: float):
        protected = 0
        t = int(metric.shape[1])
        r = min(int(r), (t - protected) // 2)
        if r <= 0:
            return None
        with torch.no_grad():
            metric = F.normalize(metric, dim=-1, eps=1e-12)
            a, b = metric[..., ::2, :], metric[..., 1::2, :]
            a_protect = protect[..., ::2]
            scores = a @ b.transpose(-1, -2)
            scores = scores - float(alpha) * a_protect.unsqueeze(-1)
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
            unm_idx = edge_idx[..., r:, :]
            src_idx = edge_idx[..., :r, :]
            dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        def merge(x: torch.Tensor, token_size: torch.Tensor):
            src, dst = x[..., ::2, :], x[..., 1::2, :]
            src_size, dst_size = token_size[..., ::2, :], token_size[..., 1::2, :]
            n, t1, c = src.shape
            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
            unm_size = src_size.gather(dim=-2, index=unm_idx.expand(n, t1 - r, src_size.shape[-1]))
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
            gathered_src_size = src_size.gather(dim=-2, index=src_idx.expand(n, r, src_size.shape[-1]))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce='sum')
            dst_size = dst_size.scatter_reduce(-2, dst_idx.expand(n, r, dst_size.shape[-1]), gathered_src_size, reduce='sum')
            return torch.cat([unm, dst], dim=1), torch.cat([unm_size, dst_size], dim=1)

        return merge

    def _folder_reduce(
        self,
        tokens: torch.Tensor,
        enhanced: torch.Tensor,
        protect: torch.Tensor,
        *,
        budget: Optional[int] = None,
        token_positions: Optional[torch.Tensor] = None,
    ):
        if tokens.ndim != 2:
            raise ValueError(f'GainHomoFolder expects rank-2 tokens, got {tuple(tokens.shape)}')
        budget = self.budget if budget is None else int(budget)
        if budget <= 0:
            empty = tokens.new_zeros((0, tokens.shape[-1]))
            empty_pos = token_positions.new_zeros((0, token_positions.shape[-1])) if token_positions is not None else None
            return empty, empty_pos
        if tokens.shape[0] <= budget:
            return F.normalize(tokens, dim=-1), token_positions
        x = tokens.unsqueeze(0)
        metric = enhanced.unsqueeze(0)
        protect = protect.unsqueeze(0)
        size = torch.ones_like(x[..., 0, None])
        pos = token_positions.unsqueeze(0) if token_positions is not None else None
        remaining = max(int(tokens.shape[0]) - int(budget), 0)
        while remaining > 0 and x.shape[1] > 1:
            r_now = min(remaining, (x.shape[1] - 1) // 2)
            merge = self._folder_match(metric=metric, protect=protect, r=r_now, alpha=self.folder_alpha)
            if merge is None:
                break
            old_size = size
            x, size = merge(x * old_size, old_size)
            if pos is not None:
                pos_sum, _ = merge(pos * old_size, old_size)
                pos = pos_sum / size.clamp_min(1e-12)
            metric = x / size.clamp_min(1e-12)
            protect = metric.norm(dim=-1)
            remaining -= r_now
        if x.shape[1] > budget:
            x = x[:, :budget, :]
            size = size[:, :budget, :]
            if pos is not None:
                pos = pos[:, :budget, :]
        out = x * (1.0 + size.clamp_min(1e-12).log())
        out_pos = pos.squeeze(0) if pos is not None else None
        return F.normalize(out.squeeze(0), dim=-1), out_pos

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        budget: Optional[int] = None,
        coarse_anchors: Optional[torch.Tensor] = None,
        text_context: Optional[torch.Tensor] = None,
        token_positions: Optional[torch.Tensor] = None,
        anchor_positions: Optional[torch.Tensor] = None,
        residual_mass_score: Optional[torch.Tensor] = None,
        return_positions: bool = False,
    ):
        target_budget = self.budget if budget is None else int(budget)
        if tokens.shape[0] == 0:
            return (tokens, token_positions) if return_positions else tokens
        if target_budget <= 0:
            empty = tokens.new_zeros((0, tokens.shape[-1]))
            empty_positions = token_positions.new_zeros((0, token_positions.shape[-1])) if token_positions is not None else None
            return (empty, empty_positions) if return_positions else empty
        if tokens.shape[0] <= target_budget:
            return (tokens, token_positions) if return_positions else tokens
        enhanced, saliency_norm, novelty, coverage, mmr_diversity, gate = self._gain_terms(
            tokens,
            coarse_anchors=coarse_anchors,
            text_context=text_context,
            token_positions=token_positions,
            anchor_positions=anchor_positions,
        )
        protect = saliency_norm + self.novelty_weight * novelty
        value_terms = [saliency_norm, novelty]
        if self.config.uses_coverage_gain():
            protect = protect + self.coverage_weight * coverage
            value_terms.append(coverage)
        if self.config.uses_mmr_gain():
            protect = protect + self.mmr_weight * mmr_diversity
            value_terms.append(mmr_diversity)
        if residual_mass_score is not None:
            mass_score = residual_mass_score.to(device=tokens.device, dtype=tokens.dtype).reshape(()).expand_as(saliency_norm)
            protect = protect + self.residual_mass_weight * mass_score
            value_terms.append(mass_score)
        continuous_importance = torch.stack(value_terms, dim=0).mean(dim=0)
        value_scale = 1.0 + self.gate_strength * gate.to(tokens.dtype) * continuous_importance
        gated_tokens = tokens * value_scale.unsqueeze(-1)
        out, out_positions = self._folder_reduce(
            tokens=gated_tokens,
            enhanced=enhanced,
            protect=protect,
            budget=target_budget,
            token_positions=token_positions,
        )
        return (out, out_positions) if return_positions else out


class GainHomoFolderCompressor(nn.Module):
    def __init__(self, config: FolderGainHomoConfig, *, image_token_id: int, crop_counts: Sequence[int], embed_dim: int) -> None:
        super().__init__()
        self.config = config
        self.image_token_id = int(image_token_id)
        self.crop_counts = tuple(int(v) for v in crop_counts)
        self.total_crop_count = int(sum(self.crop_counts))
        if len(self.crop_counts) != 3:
            raise ValueError('GainHomoFolder expects exactly three crop stages.')
        if len(config.budgets) != 3:
            raise ValueError(f'FolderGainHomo budgets must contain three values, got {config.budgets!r}')
        self.blocks = nn.ModuleList([
            GainHomoFolderBlock(embed_dim=embed_dim, budget=int(config.budgets[index]), config=config)
            for index in range(3)
        ])

    def _split_stages(self, image_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total = int(image_tokens.shape[0])
        ends = []
        running = 0
        for count in self.crop_counts:
            running += int(count)
            ends.append(int(math.floor(total * running / float(self.total_crop_count))))
        ends[-1] = total
        chunks = []
        start = 0
        for end in ends:
            chunks.append(image_tokens[start:end])
            start = end
        return chunks[0], chunks[1], chunks[2]

    @staticmethod
    def _split_evenly(tokens: torch.Tensor, parts: int) -> List[torch.Tensor]:
        parts = max(1, int(parts))
        total = int(tokens.shape[0])
        chunks: List[torch.Tensor] = []
        start = 0
        for index in range(parts):
            end = int(math.floor(total * (index + 1) / float(parts)))
            chunks.append(tokens[start:end])
            start = end
        return chunks

    def _crop_boxes(self, crop_count: int) -> Tuple[Tuple[float, float, float, float], ...]:
        crop_count = max(1, int(crop_count))
        if crop_count == 1:
            return ((0.0, 0.0, 1.0, 1.0),)
        if crop_count == 2:
            layout = str(self.config.geo_two_crop_layout).strip().lower()
            if layout in {"horizontal", "left_right", "leftright", "x"}:
                return ((0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 1.0, 1.0))
            return ((0.0, 0.0, 1.0, 0.5), (0.0, 0.5, 1.0, 1.0))
        if crop_count == 4:
            return (
                (0.0, 0.0, 0.5, 0.5),
                (0.5, 0.0, 1.0, 0.5),
                (0.0, 0.5, 0.5, 1.0),
                (0.5, 0.5, 1.0, 1.0),
            )
        cols = int(math.ceil(math.sqrt(float(crop_count))))
        rows = int(math.ceil(crop_count / float(cols)))
        boxes = []
        for index in range(crop_count):
            row = index // cols
            col = index % cols
            boxes.append((col / cols, row / rows, (col + 1) / cols, (row + 1) / rows))
        return tuple(boxes)

    @staticmethod
    def _positions_for_box(token_count: int, box: Tuple[float, float, float, float], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        token_count = int(token_count)
        if token_count <= 0:
            return torch.empty((0, 2), device=device, dtype=dtype)
        x0, y0, x1, y1 = box
        width = max(float(x1) - float(x0), 1e-6)
        height = max(float(y1) - float(y0), 1e-6)
        cols = max(1, int(math.ceil(math.sqrt(token_count * width / height))))
        rows = max(1, int(math.ceil(token_count / float(cols))))
        ys = torch.arange(rows, device=device, dtype=dtype)
        xs = torch.arange(cols, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        grid_x = grid_x.reshape(-1)[:token_count]
        grid_y = grid_y.reshape(-1)[:token_count]
        x = float(x0) + (grid_x + 0.5) * (width / float(cols))
        y = float(y0) + (grid_y + 0.5) * (height / float(rows))
        return torch.stack([x, y], dim=-1)

    def _split_stage_with_positions(self, tokens: torch.Tensor, stage_index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        crop_count = self.crop_counts[int(stage_index)]
        crop_tokens = self._split_evenly(tokens, crop_count)
        boxes = self._crop_boxes(crop_count)
        position_dtype = torch.float32
        crop_positions = [
            self._positions_for_box(int(chunk.shape[0]), box, device=tokens.device, dtype=position_dtype)
            for chunk, box in zip(crop_tokens, boxes)
        ]
        return crop_tokens, crop_positions

    @staticmethod
    def _normalize_float_scores(scores: Sequence[float]) -> List[float]:
        if not scores:
            return []
        lo = min(scores)
        hi = max(scores)
        if hi - lo <= 1e-12:
            return [1.0 if hi > 0.0 else 0.0 for _ in scores]
        return [(score - lo) / (hi - lo) for score in scores]

    def _allocate_residual_crop_budgets(
        self,
        *,
        block: GainHomoFolderBlock,
        crop_tokens: Sequence[torch.Tensor],
        crop_positions: Sequence[torch.Tensor],
        total_budget: int,
        coarse_anchors: Optional[torch.Tensor],
        anchor_positions: Optional[torch.Tensor],
        text_context: Optional[torch.Tensor],
    ) -> Tuple[List[int], List[float]]:
        token_counts = [int(tokens.shape[0]) for tokens in crop_tokens]
        total_tokens = sum(token_counts)
        total_budget = max(0, min(int(total_budget), total_tokens))
        if len(crop_tokens) == 0 or total_budget <= 0:
            return [0 for _ in crop_tokens], [0.0 for _ in crop_tokens]
        if len(crop_tokens) == 1:
            return [total_budget], [1.0]

        min_ratio = max(0.0, min(float(self.config.residual_mass_min_budget_ratio), 1.0))
        even_budget = total_budget / float(len(crop_tokens))
        min_budget = int(math.floor(even_budget * min_ratio))
        budgets = []
        for count in token_counts:
            if count <= 0:
                budgets.append(0)
            elif total_budget >= len(crop_tokens):
                budgets.append(min(count, max(1, min_budget)))
            else:
                budgets.append(0)
        while sum(budgets) > total_budget:
            index = max(range(len(budgets)), key=lambda i: budgets[i])
            budgets[index] -= 1

        capacities = [max(0, count - budget) for count, budget in zip(token_counts, budgets)]
        masses = []
        for tokens, positions, capacity in zip(crop_tokens, crop_positions, capacities):
            if capacity <= 0 or tokens.numel() == 0:
                masses.append(0.0)
                continue
            mass = block.estimate_residual_mass(
                tokens,
                coarse_anchors=coarse_anchors,
                text_context=text_context,
                token_positions=positions,
                anchor_positions=anchor_positions,
            )
            masses.append(float(mass.detach().cpu().item()))

        remaining = total_budget - sum(budgets)
        if remaining > 0 and any(capacity > 0 for capacity in capacities):
            weights = [mass if capacity > 0 else 0.0 for mass, capacity in zip(masses, capacities)]
            if sum(weights) <= 1e-12:
                weights = [float(capacity) for capacity in capacities]
            weight_sum = max(sum(weights), 1e-12)
            raw_adds = [remaining * weight / weight_sum if capacity > 0 else 0.0 for weight, capacity in zip(weights, capacities)]
            adds = [min(capacity, int(math.floor(raw))) for raw, capacity in zip(raw_adds, capacities)]
            leftover = remaining - sum(adds)
            order = sorted(
                range(len(crop_tokens)),
                key=lambda i: (raw_adds[i] - math.floor(raw_adds[i]), weights[i], capacities[i]),
                reverse=True,
            )
            while leftover > 0:
                progressed = False
                for index in order:
                    if adds[index] < capacities[index]:
                        adds[index] += 1
                        leftover -= 1
                        progressed = True
                        if leftover <= 0:
                            break
                if not progressed:
                    break
            budgets = [budget + add for budget, add in zip(budgets, adds)]

        return budgets, self._normalize_float_scores(masses)

    @staticmethod
    def _pad_sequences(sequences: Sequence[torch.Tensor], dim: int) -> torch.Tensor:
        max_len = max(seq.shape[0] for seq in sequences)
        out = sequences[0].new_zeros((len(sequences), max_len, dim))
        for i, seq in enumerate(sequences):
            out[i, : seq.shape[0]] = seq
        return out

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.LongTensor, attention_mask: torch.Tensor) -> torch.Tensor:
        active_stages = set(self.config.active_stage_ids())
        sequences: List[torch.Tensor] = []
        debug_rows = []
        for row_hidden, row_ids, row_attn in zip(hidden_states, input_ids, attention_mask):
            valid = row_attn.to(dtype=torch.bool)
            image_mask = row_ids.eq(self.image_token_id) & valid
            text_mask = (~row_ids.eq(self.image_token_id)) & valid
            text_tokens = row_hidden[text_mask]
            image_tokens = row_hidden[image_mask]
            if image_tokens.numel() == 0:
                sequence = text_tokens if text_tokens.numel() > 0 else row_hidden.new_zeros((1, row_hidden.shape[-1]))
                sequences.append(sequence)
                debug_rows.append((int(sequence.shape[0]), 0, 0, 0))
                continue

            stage_tokens = self._split_stages(image_tokens)
            stage_crops = [self._split_stage_with_positions(tokens, stage_index) for stage_index, tokens in enumerate(stage_tokens)]
            text_context = text_tokens.mean(dim=0, keepdim=True) if self.config.use_text_context and text_tokens.numel() > 0 else None
            compressed: List[torch.Tensor] = []
            compressed_positions: List[torch.Tensor] = []
            coarse_anchors: Optional[torch.Tensor] = None
            coarse_anchor_positions: Optional[torch.Tensor] = None
            for stage_index, (crop_tokens, crop_positions) in enumerate(stage_crops):
                tokens = torch.cat(crop_tokens, dim=0)
                token_positions = torch.cat(crop_positions, dim=0)
                if stage_index in active_stages:
                    if self.config.uses_residual_mass_budget() and len(crop_tokens) > 1:
                        crop_budgets, crop_mass_scores = self._allocate_residual_crop_budgets(
                            block=self.blocks[stage_index],
                            crop_tokens=crop_tokens,
                            crop_positions=crop_positions,
                            total_budget=int(self.config.budgets[stage_index]),
                            coarse_anchors=coarse_anchors,
                            anchor_positions=coarse_anchor_positions,
                            text_context=text_context,
                        )
                        out_chunks: List[torch.Tensor] = []
                        out_position_chunks: List[torch.Tensor] = []
                        for chunk, positions, chunk_budget, mass_score in zip(crop_tokens, crop_positions, crop_budgets, crop_mass_scores):
                            chunk_out, chunk_positions = self.blocks[stage_index](
                                chunk,
                                budget=chunk_budget,
                                coarse_anchors=coarse_anchors,
                                text_context=text_context,
                                token_positions=positions,
                                anchor_positions=coarse_anchor_positions,
                                residual_mass_score=chunk.new_tensor(mass_score),
                                return_positions=True,
                            )
                            out_chunks.append(chunk_out)
                            if chunk_positions is None:
                                chunk_positions = positions[: chunk_out.shape[0]]
                            out_position_chunks.append(chunk_positions)
                        out = torch.cat(out_chunks, dim=0)
                        out_positions = torch.cat(out_position_chunks, dim=0)
                    else:
                        out, out_positions = self.blocks[stage_index](
                            tokens,
                            coarse_anchors=coarse_anchors,
                            text_context=text_context,
                            token_positions=token_positions,
                            anchor_positions=coarse_anchor_positions,
                            return_positions=True,
                        )
                else:
                    out = tokens
                    out_positions = token_positions
                if out_positions is None:
                    out_positions = token_positions[: out.shape[0]]
                compressed.append(out)
                compressed_positions.append(out_positions)
                coarse_anchors = out if coarse_anchors is None else torch.cat([coarse_anchors, out], dim=0)
                coarse_anchor_positions = out_positions if coarse_anchor_positions is None else torch.cat([coarse_anchor_positions, out_positions], dim=0)
            prefix_level = max(1, min(int(getattr(self.config, 'eval_prefix_level', 3)), len(compressed)))
            sequence = torch.cat([text_tokens, *compressed[:prefix_level]], dim=0)
            sequences.append(sequence)
            debug_rows.append(tuple(int(x.shape[0]) for x in compressed[:prefix_level]))

        output = self._pad_sequences(sequences, hidden_states.shape[-1])
        active_stages = set(self.config.active_stage_ids())
        if active_stages:
            zero = output.sum() * 0.0
            for idx in active_stages:
                for param in self.blocks[idx].parameters():
                    zero = zero + param.sum() * 0.0
            output = output + zero
        if self.config.debug_shapes:
            print(f'[GainHomoFolderCompressor] rows={debug_rows[:4]} output={list(output.shape)}', flush=True)
        return output


class GainHomoFolderMRLColQwen2_5(MRLColQwen2_5):
    def __init__(self, base_model: ColQwen2_5, *, granularities: Sequence[int] = (1, 2, 4), compact_query_tokens: bool = True, folder_gain_homo_config: Optional[FolderGainHomoConfig] = None) -> None:
        super().__init__(base_model=base_model, granularities=granularities, compact_query_tokens=compact_query_tokens)
        self.folder_gain_homo_config = folder_gain_homo_config or FolderGainHomoConfig(enabled=False)
        self.folder_gain_homo = GainHomoFolderCompressor(
            self.folder_gain_homo_config,
            image_token_id=self.config.image_token_id,
            crop_counts=[spec.crop_count for spec in self.stage_specs],
            embed_dim=self.dim,
        )

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor, pixel_values: Optional[torch.Tensor] = None, image_grid_thw: Optional[torch.LongTensor] = None, **kwargs) -> torch.Tensor:
        has_images = pixel_values is not None and image_grid_thw is not None and getattr(pixel_values, 'numel', lambda: 0)() > 0 and getattr(image_grid_thw, 'numel', lambda: 0)() > 0
        hidden_states = self._project_hidden_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values if has_images else None,
            image_grid_thw=image_grid_thw if has_images else None,
            **kwargs,
        )
        if (not self.folder_gain_homo_config.enabled) or len(self.folder_gain_homo_config.active_stage_ids()) == 0:
            return self._compact_doc_embeddings(hidden_states, input_ids, attention_mask)
        return self.folder_gain_homo(hidden_states, input_ids, attention_mask)


def _load_adapter_with_fallback(base_model: ColQwen2_5, adapter_path: Path):
    adapter_bin = adapter_path / 'adapter_model.bin'
    if not adapter_bin.exists():
        return PeftModel.from_pretrained(base_model, adapter_path)
    state_dict = torch.load(adapter_bin, map_location='cpu')
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith('base_model.model.base_model.custom_text_proj.'):
            key = key.replace('base_model.model.base_model.custom_text_proj.', 'base_model.model.custom_text_proj.', 1)
        if key.startswith('base_model.model.base_model.model.'):
            key = key.replace('base_model.model.base_model.model.', 'base_model.model.model.', 1)
        remapped[key] = value
    with TemporaryDirectory(prefix='folder_gain_homo_eval_adapter_') as tmpdir:
        tmpdir_path = Path(tmpdir)
        (tmpdir_path / 'adapter_config.json').write_text((adapter_path / 'adapter_config.json').read_text())
        torch.save(remapped, tmpdir_path / 'adapter_model.bin')
        return PeftModel.from_pretrained(base_model, tmpdir_path)


def build_folder_gain_homo_model(model_name_or_path: str, *, granularities: Sequence[int] = (1, 2, 4), folder_gain_homo_config: Optional[FolderGainHomoConfig] = None, attn_implementation: Optional[str] = 'flash_attention_2', use_liger_kernel: bool = False, torch_dtype: torch.dtype = torch.bfloat16, adapter_path: Optional[str] = None, eval_mode: bool = False, compact_query_tokens: bool = True):
    granularities = normalize_granularities(granularities)
    if len(build_stage_specs(granularities)) != 3:
        raise ValueError('FolderGainHomo expects exactly three stages.')
    base_model = ColQwen2_5.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        use_cache=False,
        attn_implementation=attn_implementation,
        use_liger_kernel=use_liger_kernel,
    )
    if not hasattr(base_model, 'custom_text_proj'):
        raise TypeError('Expected a ColQwen2_5 checkpoint with custom_text_proj.')
    _apply_compat_patch(base_model)
    if adapter_path is not None:
        base_model = _load_adapter_with_fallback(base_model, Path(adapter_path))
    model = GainHomoFolderMRLColQwen2_5(
        base_model=base_model,
        granularities=granularities,
        compact_query_tokens=compact_query_tokens,
        folder_gain_homo_config=folder_gain_homo_config,
    )
    if eval_mode:
        model.eval()
    return model
