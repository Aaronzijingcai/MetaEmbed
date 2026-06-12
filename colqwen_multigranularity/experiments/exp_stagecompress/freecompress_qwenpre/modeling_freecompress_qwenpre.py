from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel

from colpali_engine.models import ColQwen2_5
from colqwen_multigranularity.core import (
    MRLColQwen2_5,
    _apply_compat_patch,
    build_stage_specs,
    normalize_granularities,
)

_METHOD_ALIASES = {
    "prumerge": "prumerge",
    "strategy3_prumerge": "prumerge",
    "visionzip": "visionzip",
    "strategy4_visionzip": "visionzip",
    "folder": "folder",
    "strategy5_folder": "folder",
    "scope": "scope",
    "strategy6_scope": "scope",
}


def canonicalize_qwenpre_freecompress_method(method: str) -> str:
    key = str(method).strip().lower()
    if key not in _METHOD_ALIASES:
        raise ValueError(f"Unknown qwenpre freecompress method {method!r}; expected one of {sorted(_METHOD_ALIASES)}.")
    return _METHOD_ALIASES[key]


@dataclass(frozen=True)
class QwenPreFreeCompressConfig:
    enabled: bool = True
    method: str = "visionzip"
    compress_stages: str = "g2g3"
    keep_ratios: Tuple[float, float, float] = (1.0, 0.5, 0.25)
    min_keep: int = 1
    tau: float = 1.0
    visionzip_dominant_ratio: float = 0.65
    visionzip_contextual_ratio: float = 0.05
    scope_alpha: float = 1.0
    scope_combined: str = "multi"
    saliency: str = "text"
    debug_shapes: bool = False

    def active_stage_ids(self) -> Tuple[int, ...]:
        if not self.enabled:
            return ()
        mode = str(self.compress_stages).strip().lower().replace(" ", "")
        if mode in {"", "none", "off", "false", "0"}:
            return ()
        if mode == "g3":
            return (2,)
        if mode in {"g2g3", "g2+g3", "g2,g3"}:
            return (1, 2)
        if mode in {"all", "g1g2g3", "g1+g2+g3", "g1,g2,g3"}:
            return (0, 1, 2)
        if mode in {"g1", "g2"}:
            return (int(mode[1]) - 1,)
        parts = [part for part in mode.replace("+", ",").split(",") if part]
        if parts and all(part in {"g1", "g2", "g3"} for part in parts):
            return tuple(sorted({int(part[1]) - 1 for part in parts}))
        raise ValueError(f"Unknown compress_stages={self.compress_stages!r}")


class QwenPreFreeCompressColQwen2_5(MRLColQwen2_5):  # noqa: N801
    """MRL_Main + training-free visual token compression before Qwen2.5-VL LLM.

    The Qwen2.5-VL-compatible insertion point is after ``self.visual`` produces
    image embeddings and after those embeddings are scattered into
    ``inputs_embeds``, but before ``self.model`` / language_model is called.
    This matches the public Qwen2.5-VL VisionZip adaptation and is the closest
    exposed boundary to the LLaVA ``vision_tower -> mm_projector -> LLM`` path.
    """

    def __init__(
        self,
        base_model: ColQwen2_5,
        *,
        granularities: Sequence[int] = (1, 2, 4),
        compact_query_tokens: bool = True,
        config: Optional[QwenPreFreeCompressConfig] = None,
    ) -> None:
        super().__init__(base_model=base_model, granularities=granularities, compact_query_tokens=compact_query_tokens)
        if len(self.stage_specs) != 3:
            raise ValueError("QwenPreFreeCompress expects exactly three MRL stages.")
        self.qwenpre_config = config or QwenPreFreeCompressConfig(enabled=False)
        self.method = canonicalize_qwenpre_freecompress_method(self.qwenpre_config.method)
        self.image_token_id = int(getattr(self.config, "image_token_id"))
        self._last_qwenpre_stats: Optional[dict] = None
        self._debug_count = 0

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

    def _image_grid_token_counts(self, image_grid_thw: torch.LongTensor) -> List[int]:
        merge_size = self._visual_spatial_merge_size()
        denom = merge_size * merge_size
        counts: List[int] = []
        for row in image_grid_thw.detach().to("cpu").tolist():
            t, h, w = [int(value) for value in row]
            total = t * h * w
            if total % denom != 0:
                raise RuntimeError(f"QwenPreFreeCompress image grid is not divisible by merge size: grid={row} merge_size={merge_size}.")
            counts.append(total // denom)
        return counts

    def _stage_and_crop_maps(
        self,
        *,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor],
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        stage_map = input_ids.new_full(input_ids.shape, -1)
        crop_map = input_ids.new_full(input_ids.shape, -1)
        if image_grid_thw is None or image_grid_thw.numel() == 0:
            return stage_map, crop_map

        crop_token_counts = self._image_grid_token_counts(image_grid_thw)
        expected_crops = sum(spec.crop_count for spec in self.stage_specs)
        grid_cursor = 0
        crop_uid = 0
        for batch_index in range(input_ids.shape[0]):
            image_positions = torch.where(input_ids[batch_index].eq(self.image_token_id))[0]
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
                    "QwenPreFreeCompress sample image token mismatch: "
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
            raise RuntimeError(f"QwenPreFreeCompress image grid cursor mismatch: {grid_cursor}/{len(crop_token_counts)}.")
        return stage_map, crop_map

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
        image_mask = input_ids.eq(self.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        image_embeds = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        expected = int(image_mask.sum().item() // inputs_embeds.shape[-1])
        actual = int(image_embeds.shape[0])
        if expected != actual:
            raise RuntimeError(f"QwenPreFreeCompress image embed mismatch: placeholders={expected} visual_embeds={actual}.")
        return inputs_embeds.masked_scatter(image_mask, image_embeds)

    def _stage_budget(self, stage_index: int, length: int) -> int:
        ratios = self.qwenpre_config.keep_ratios
        if len(ratios) != 3:
            raise ValueError(f"keep_ratios must have exactly 3 entries, got {ratios!r}")
        budget = int(math.ceil(float(length) * float(ratios[stage_index])))
        budget = max(int(self.qwenpre_config.min_keep), budget)
        return min(max(budget, 1), int(length))

    def _features_and_saliency(self, tokens: torch.Tensor, text_context: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        features = F.normalize(tokens.float(), dim=-1, eps=1e-12)
        saliency_mode = str(self.qwenpre_config.saliency).lower()
        if saliency_mode in {"text", "auto"} and text_context is not None and text_context.numel() > 0:
            context = F.normalize(text_context.reshape(1, -1).float(), dim=-1, eps=1e-12)
            raw = (features @ context.transpose(0, 1)).squeeze(-1)
        elif saliency_mode == "norm":
            raw = tokens.float().norm(dim=-1)
        else:
            raw = (features @ features.transpose(0, 1)).mean(dim=-1)
        if raw.numel() <= 1:
            return features, torch.ones_like(raw, dtype=torch.float32)
        saliency = (raw.float() - raw.float().min()) / (raw.float().max() - raw.float().min()).clamp_min(1e-6)
        return features, saliency + 1e-6

    @staticmethod
    def _select_uniform_indices(length: int, count: int, device: torch.device) -> torch.LongTensor:
        if count <= 0 or length <= 0:
            return torch.empty(0, dtype=torch.long, device=device)
        if count >= length:
            return torch.arange(length, device=device, dtype=torch.long)
        values = torch.linspace(0, length - 1, steps=count, device=device)
        indices = values.round().long().unique(sorted=True)
        if indices.numel() < count:
            mask = torch.ones(length, dtype=torch.bool, device=device)
            mask[indices] = False
            indices = torch.cat([indices, mask.nonzero(as_tuple=False).squeeze(-1)[: count - indices.numel()]], dim=0)
        return indices[:count]

    @staticmethod
    def _partition_prumerge_budget(budget: int) -> Tuple[int, int, int]:
        budget = int(budget)
        if budget <= 0:
            return 0, 0, 0
        if budget == 1:
            return 1, 0, 0
        if budget == 2:
            return 1, 1, 0
        residual_budget = 1
        keep_budget = max(1, int(round((budget - residual_budget) * 0.6)))
        keep_budget = min(keep_budget, budget - residual_budget)
        merge_budget = budget - keep_budget - residual_budget
        if merge_budget <= 0 and budget - residual_budget >= 2:
            keep_budget = max(1, keep_budget - 1)
            merge_budget = budget - keep_budget - residual_budget
        return keep_budget, max(merge_budget, 0), residual_budget

    def _scope_select(self, metric: torch.Tensor, num_selected_token: int, saliency: Optional[torch.Tensor] = None) -> torch.LongTensor:
        metric = F.normalize(metric.float(), dim=-1, eps=1e-12)
        cosine_simi = metric @ metric.transpose(0, 1)
        n = int(metric.shape[0])
        selected = torch.zeros(n, dtype=torch.bool, device=metric.device)
        selected_idx: List[torch.Tensor] = []
        cur_max = torch.zeros(n, dtype=metric.dtype, device=metric.device)
        if saliency is None:
            saliency_term = torch.ones(n, dtype=metric.dtype, device=metric.device)
        else:
            saliency_term = saliency.float().clamp_min(1e-12).pow(float(self.qwenpre_config.scope_alpha))
        for _ in range(min(int(num_selected_token), n)):
            unselected_mask = ~selected
            gains = torch.maximum(
                torch.zeros(1, dtype=metric.dtype, device=metric.device),
                cosine_simi.masked_fill(~unselected_mask.unsqueeze(0), 0) - cur_max.unsqueeze(1),
            ).sum(dim=0)
            if self.qwenpre_config.scope_combined == "multi":
                gains = gains * saliency_term
            elif self.qwenpre_config.scope_combined == "add":
                gains = gains + saliency_term
            else:
                raise ValueError(f"Unknown SCOPE combination: {self.qwenpre_config.scope_combined!r}")
            gains = gains.masked_fill(~unselected_mask, float("-inf"))
            best_idx = gains.argmax()
            selected[best_idx] = True
            selected_idx.append(best_idx)
            cur_max = torch.maximum(cur_max, cosine_simi[best_idx])
        if not selected_idx:
            return torch.empty(0, dtype=torch.long, device=metric.device)
        return torch.stack(selected_idx).sort().values

    def _compress_prumerge(self, tokens: torch.Tensor, features: torch.Tensor, saliency: torch.Tensor, budget: int) -> torch.Tensor:
        dtype = tokens.dtype
        tau = max(float(self.qwenpre_config.tau), 1e-6)
        keep_budget, merge_budget, residual_budget = self._partition_prumerge_budget(budget)
        keep_budget = min(keep_budget, tokens.shape[0])
        _, keep_idx = torch.topk(saliency, k=keep_budget, dim=0, largest=True)
        keep_idx = keep_idx.sort().values
        keep_tokens = tokens.index_select(0, keep_idx)
        keep_features = features.index_select(0, keep_idx)
        if keep_budget >= tokens.shape[0]:
            return keep_tokens
        mask = torch.ones(tokens.shape[0], dtype=torch.bool, device=tokens.device)
        mask[keep_idx] = False
        residual_tokens = tokens[mask]
        residual_features = features[mask]
        residual_scores = saliency[mask]

        anchor_logits = residual_features @ keep_features.transpose(0, 1)
        anchor_logits = anchor_logits + residual_scores.unsqueeze(-1)
        anchor_weights = torch.softmax(anchor_logits / tau, dim=-1)
        anchor_mass = anchor_weights.sum(dim=0).unsqueeze(-1).clamp_min(1.0)
        recovered_keep = (anchor_weights.transpose(0, 1) @ residual_tokens.float()) / anchor_mass
        updated_keep = keep_tokens.float() + recovered_keep
        parts = [updated_keep]

        if merge_budget > 0 and residual_tokens.shape[0] > 0:
            merge_budget = min(merge_budget, residual_tokens.shape[0])
            _, rel_anchor_idx = torch.topk(residual_scores, k=merge_budget, dim=0, largest=True)
            rel_anchor_idx = rel_anchor_idx.sort().values
            merge_tokens = residual_tokens.index_select(0, rel_anchor_idx).float()
            merge_features = residual_features.index_select(0, rel_anchor_idx)
            merge_mask = torch.ones(residual_tokens.shape[0], dtype=torch.bool, device=tokens.device)
            merge_mask[rel_anchor_idx] = False
            to_merge = residual_tokens[merge_mask]
            to_merge_features = residual_features[merge_mask]
            to_merge_scores = residual_scores[merge_mask]
            if to_merge.numel() > 0:
                logits = to_merge_features @ merge_features.transpose(0, 1)
                logits = logits + to_merge_scores.unsqueeze(-1)
                weights = torch.softmax(logits / tau, dim=-1)
                mass = weights.sum(dim=0).unsqueeze(-1).clamp_min(1.0)
                merge_tokens = merge_tokens + (weights.transpose(0, 1) @ to_merge.float()) / mass
            parts.append(merge_tokens)

        if residual_budget > 0 and residual_tokens.shape[0] > 0:
            residual_weights = torch.softmax(residual_scores / tau, dim=0)
            residual_token = torch.sum(residual_tokens.float() * residual_weights.unsqueeze(-1), dim=0, keepdim=True)
            parts.append(residual_token)
        return torch.cat(parts, dim=0)[:budget].to(dtype=dtype)

    def _compress_visionzip(self, tokens: torch.Tensor, features: torch.Tensor, saliency: torch.Tensor, budget: int) -> torch.Tensor:
        dtype = tokens.dtype
        token_count = int(tokens.shape[0])
        if budget <= 1:
            _, idx = torch.topk(saliency, k=1, dim=0, largest=True)
            return tokens.index_select(0, idx.sort().values)

        dominant_budget = max(1, int(math.ceil(token_count * float(self.qwenpre_config.visionzip_dominant_ratio))))
        dominant_budget = min(dominant_budget, budget - 1, token_count)
        contextual_budget = budget - dominant_budget
        if contextual_budget <= 0:
            dominant_budget = min(budget, token_count)
        _, dominant_idx = torch.topk(saliency, k=dominant_budget, dim=0, largest=True)
        dominant_idx = dominant_idx.sort().values
        dominant_tokens = tokens.index_select(0, dominant_idx).float()
        if dominant_budget >= token_count or contextual_budget <= 0:
            return dominant_tokens[:budget].to(dtype=dtype)

        residual_mask = torch.ones(token_count, dtype=torch.bool, device=tokens.device)
        residual_mask[dominant_idx] = False
        residual_tokens = tokens[residual_mask]
        residual_features = features[residual_mask]
        contextual_budget = min(contextual_budget, residual_tokens.shape[0])
        if contextual_budget <= 0:
            return dominant_tokens[:budget].to(dtype=dtype)

        contextual_idx = self._select_uniform_indices(residual_tokens.shape[0], contextual_budget, tokens.device)
        contextual_tokens = residual_tokens.index_select(0, contextual_idx).float()
        contextual_features = residual_features.index_select(0, contextual_idx)
        merge_mask = torch.ones(residual_tokens.shape[0], dtype=torch.bool, device=tokens.device)
        merge_mask[contextual_idx] = False
        merge_tokens = residual_tokens[merge_mask]
        merge_features = residual_features[merge_mask]
        if merge_tokens.numel() > 0:
            similarity = merge_features @ contextual_features.transpose(0, 1)
            assignments = similarity.argmax(dim=-1)
            aggregated = contextual_tokens.new_zeros(contextual_tokens.shape)
            aggregated.scatter_add_(0, assignments.unsqueeze(-1).expand_as(merge_tokens), merge_tokens.float())
            mass = contextual_tokens.new_zeros((contextual_budget,))
            mass.scatter_add_(0, assignments, torch.ones_like(assignments, dtype=contextual_tokens.dtype))
            contextual_tokens = contextual_tokens + aggregated / mass.clamp_min(1.0).unsqueeze(-1)

        return torch.cat([dominant_tokens, contextual_tokens], dim=0)[:budget].to(dtype=dtype)

    @staticmethod
    def _folder_bipartite_unimodal_matching(metric: torch.Tensor, saliency: torch.Tensor, r: int, *, alpha: float = 1.0):
        t = metric.shape[1]
        r = min(int(r), t // 2)
        if r <= 0:
            return None
        with torch.no_grad():
            metric = F.normalize(metric.float(), dim=-1, eps=1e-12)
            a, b = metric[..., ::2, :], metric[..., 1::2, :]
            a_saliency = saliency[..., ::2]
            scores_redund = a @ b.transpose(-1, -2)
            scores = scores_redund - float(alpha) * a_saliency.unsqueeze(-1)
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
            unm_idx = edge_idx[..., r:, :]
            src_idx = edge_idx[..., :r, :]
            dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        def merge(x: torch.Tensor, token_size: Optional[torch.Tensor] = None):
            src, dst = x[..., ::2, :], x[..., 1::2, :]
            n, t1, c = src.shape
            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
            if token_size is None:
                dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce="mean")
                return torch.cat([unm, dst], dim=1)
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce="sum")
            dst_size = token_size[..., 1::2, :].scatter_reduce(
                -2,
                dst_idx.expand(n, r, token_size.shape[-1]),
                token_size[..., ::2, :].gather(-2, src_idx.expand(n, r, token_size.shape[-1])),
                reduce="sum",
            )
            unm_size = token_size[..., ::2, :].gather(-2, unm_idx.expand(n, t1 - r, token_size.shape[-1]))
            return torch.cat([unm, dst], dim=1), torch.cat([unm_size, dst_size], dim=1)

        return merge

    def _compress_folder(self, tokens: torch.Tensor, features: torch.Tensor, saliency: torch.Tensor, budget: int) -> torch.Tensor:
        dtype = tokens.dtype
        x = tokens.float().unsqueeze(0)
        metric = features.unsqueeze(0)
        attn_cls = saliency.float().unsqueeze(0)
        size = torch.ones_like(x[..., 0, None])
        remaining = max(int(tokens.shape[0]) - int(budget), 0)
        while remaining > 0 and x.shape[1] > 1:
            r_now = min(remaining, x.shape[1] // 2)
            if r_now <= 0:
                break
            merge = self._folder_bipartite_unimodal_matching(metric=metric, saliency=attn_cls, r=r_now, alpha=1.0)
            if merge is None:
                break
            x, size = merge(x * size, token_size=size)
            metric = x / size.clamp_min(1e-12)
            attn_cls = metric.norm(dim=-1)
            remaining -= r_now
        if x.shape[1] > budget:
            x = x[:, :budget, :]
        return x.squeeze(0).to(dtype=dtype)

    def _compress_scope(self, tokens: torch.Tensor, features: torch.Tensor, saliency: torch.Tensor, budget: int) -> torch.Tensor:
        selected_idx = self._scope_select(features, budget, saliency)
        return tokens.index_select(0, selected_idx)

    def _compress_tokens(self, tokens: torch.Tensor, budget: int, text_context: Optional[torch.Tensor]) -> torch.Tensor:
        if tokens.shape[0] == 0 or budget <= 0 or tokens.shape[0] <= budget:
            return tokens
        features, saliency = self._features_and_saliency(tokens, text_context)
        if self.method == "prumerge":
            return self._compress_prumerge(tokens, features, saliency, budget)
        if self.method == "visionzip":
            return self._compress_visionzip(tokens, features, saliency, budget)
        if self.method == "folder":
            return self._compress_folder(tokens, features, saliency, budget)
        if self.method == "scope":
            return self._compress_scope(tokens, features, saliency, budget)
        raise AssertionError(f"Unhandled method: {self.method}")

    def _compress_inputs_embeds(
        self,
        *,
        inputs_embeds: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        image_grid_thw: Optional[torch.LongTensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stage_map, crop_map = self._stage_and_crop_maps(input_ids=input_ids, image_grid_thw=image_grid_thw)
        stage_map = stage_map.to(inputs_embeds.device)
        crop_map = crop_map.to(inputs_embeds.device)
        active_stages = set(self.qwenpre_config.active_stage_ids())
        batch_size, seq_len, hidden_size = inputs_embeds.shape
        stats = {
            "method": self.method,
            "position": "qwen2.5vl_inputs_embeds_pre_llm",
            "compress_stages": self.qwenpre_config.compress_stages,
            "keep_ratios": [float(v) for v in self.qwenpre_config.keep_ratios],
            "stage_tokens": [0 for _ in self.stage_specs],
            "stage_kept": [0 for _ in self.stage_specs],
            "stage_crops": [0 for _ in self.stage_specs],
            "input_seq_len": int(seq_len),
        }
        if not torch.any(stage_map >= 0) or not active_stages:
            self._last_qwenpre_stats = stats
            return inputs_embeds, attention_mask.to(inputs_embeds.device), position_ids.to(inputs_embeds.device)

        active_attention_mask = attention_mask.to(device=inputs_embeds.device)
        active_position_ids = position_ids.to(device=inputs_embeds.device)
        compressed_rows: List[torch.Tensor] = []
        compressed_positions: List[torch.Tensor] = []
        debug_rows: List[Tuple[int, int, int, int, int]] = []

        for batch_index in range(batch_size):
            row_hidden = inputs_embeds[batch_index]
            row_ids = input_ids[batch_index].to(inputs_embeds.device)
            row_attn = active_attention_mask[batch_index].to(dtype=torch.bool)
            active_positions = row_attn.nonzero(as_tuple=False).squeeze(-1)
            if active_positions.numel() == 0:
                compressed_rows.append(row_hidden.new_zeros((1, hidden_size)))
                compressed_positions.append(active_position_ids[:, batch_index, :1])
                continue

            text_mask = row_attn & row_ids.ne(self.image_token_id)
            text_tokens = row_hidden[text_mask]
            text_context = text_tokens.mean(dim=0, keepdim=True) if text_tokens.numel() > 0 else None
            sample_crop_ids = torch.unique(crop_map[batch_index][(crop_map[batch_index] >= 0) & row_attn])
            compressed_by_crop: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
            before_image_tokens = 0
            after_image_tokens = 0
            for crop_id_tensor in sample_crop_ids:
                crop_id = int(crop_id_tensor.item())
                crop_positions = ((crop_map[batch_index] == crop_id) & row_attn).nonzero(as_tuple=False).squeeze(-1)
                if crop_positions.numel() == 0:
                    continue
                stage_index = int(stage_map[batch_index, crop_positions[0]].item())
                crop_tokens = row_hidden.index_select(0, crop_positions)
                before_image_tokens += int(crop_tokens.shape[0])
                if stage_index in active_stages:
                    budget = self._stage_budget(stage_index, crop_tokens.shape[0])
                    compressed = self._compress_tokens(crop_tokens, budget, text_context)
                    kept_positions = crop_positions[: compressed.shape[0]]
                    stats["stage_tokens"][stage_index] += int(crop_tokens.shape[0])
                    stats["stage_kept"][stage_index] += int(compressed.shape[0])
                    stats["stage_crops"][stage_index] += 1
                else:
                    compressed = crop_tokens
                    kept_positions = crop_positions
                after_image_tokens += int(compressed.shape[0])
                compressed_by_crop[crop_id] = (compressed, kept_positions)

            row_parts: List[torch.Tensor] = []
            row_pos_parts: List[torch.Tensor] = []
            emitted_crops = set()
            for pos_tensor in active_positions:
                pos = int(pos_tensor.item())
                crop_id = int(crop_map[batch_index, pos].item())
                if crop_id >= 0:
                    if crop_id not in emitted_crops:
                        compressed, kept_positions = compressed_by_crop.get(crop_id, (row_hidden[pos : pos + 1], torch.tensor([pos], device=inputs_embeds.device)))
                        row_parts.append(compressed)
                        row_pos_parts.append(active_position_ids[:, batch_index, kept_positions[: compressed.shape[0]]])
                        emitted_crops.add(crop_id)
                    continue
                row_parts.append(row_hidden[pos : pos + 1])
                row_pos_parts.append(active_position_ids[:, batch_index, pos : pos + 1])

            row_hidden_out = torch.cat(row_parts, dim=0) if row_parts else row_hidden.new_zeros((1, hidden_size))
            row_position_out = torch.cat(row_pos_parts, dim=1) if row_pos_parts else active_position_ids[:, batch_index, :1]
            compressed_rows.append(row_hidden_out)
            compressed_positions.append(row_position_out)
            debug_rows.append((batch_index, int(active_positions.numel()), before_image_tokens, after_image_tokens, int(row_hidden_out.shape[0])))

        lengths = [row.shape[0] for row in compressed_rows]
        max_len = max(lengths)
        out_hidden = inputs_embeds.new_zeros((batch_size, max_len, hidden_size))
        out_attention = active_attention_mask.new_zeros((batch_size, max_len))
        out_position = active_position_ids.new_zeros((active_position_ids.shape[0], batch_size, max_len))
        for batch_index, (row_hidden, row_position) in enumerate(zip(compressed_rows, compressed_positions)):
            length = row_hidden.shape[0]
            out_hidden[batch_index, :length] = row_hidden
            out_attention[batch_index, :length] = 1
            out_position[:, batch_index, :length] = row_position

        stats["compressed_seq_len"] = int(max_len)
        stats["compressed_lengths"] = [int(value) for value in lengths]
        self._last_qwenpre_stats = stats
        if self.qwenpre_config.debug_shapes and self._debug_count < 8:
            self._debug_count += 1
            rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
            print(f"[QwenPreFreeCompress][rank={rank}] stats={stats} rows={debug_rows[:4]}", flush=True)
        return out_hidden, out_attention, out_position

    def _project_hidden_states_with_mask(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        active_attention_mask = attention_mask.to(inputs_embeds.device)
        active_position_ids = position_ids

        should_compress = has_images and bool(self.qwenpre_config.enabled) and bool(self.qwenpre_config.active_stage_ids())
        if should_compress:
            inputs_embeds, active_attention_mask, active_position_ids = self._compress_inputs_embeds(
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                attention_mask=active_attention_mask,
                position_ids=active_position_ids,
                image_grid_thw=active_image_grid_thw,
            )
        else:
            self._last_qwenpre_stats = None

        outputs = self.base_model.model(
            input_ids=None,
            position_ids=active_position_ids,
            attention_mask=active_attention_mask,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=None,
            **kwargs,
        )
        hidden_states = outputs[0]
        proj = self.base_model.custom_text_proj(hidden_states)
        proj = proj / proj.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        output_mask = active_attention_mask.to(device=proj.device, dtype=proj.dtype)
        proj = proj * output_mask.unsqueeze(-1)
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


def _load_adapter_with_fallback(model: QwenPreFreeCompressColQwen2_5, adapter_path: Path):
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

    with TemporaryDirectory(prefix="qwenpre_freecompress_eval_adapter_") as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "adapter_config.json").write_text((adapter_path / "adapter_config.json").read_text())
        torch.save(remapped, tmpdir / "adapter_model.bin")
        return PeftModel.from_pretrained(model, tmpdir)


def build_qwenpre_freecompress_model(
    model_name_or_path: str,
    *,
    granularities: Sequence[int] = (1, 2, 4),
    attn_implementation: Optional[str] = "flash_attention_2",
    use_liger_kernel: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
    adapter_path: Optional[str] = None,
    eval_mode: bool = False,
    compact_query_tokens: bool = True,
    freecompress_method: str = "visionzip",
    freecompress_compress_stages: str = "g2g3",
    freecompress_keep_ratios: Optional[Sequence[float]] = None,
    freecompress_min_keep: int = 1,
    freecompress_tau: float = 1.0,
    freecompress_visionzip_dominant_ratio: float = 0.65,
    freecompress_visionzip_contextual_ratio: float = 0.05,
    freecompress_saliency: str = "text",
    freecompress_debug_shapes: bool = False,
):
    granularities = normalize_granularities(granularities)
    if len(build_stage_specs(granularities)) != 3:
        raise ValueError("QwenPreFreeCompress expects exactly three stages.")
    if freecompress_keep_ratios is None:
        freecompress_keep_ratios = (1.0, 0.5, 0.25)

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

    config = QwenPreFreeCompressConfig(
        enabled=True,
        method=canonicalize_qwenpre_freecompress_method(freecompress_method),
        compress_stages=freecompress_compress_stages,
        keep_ratios=tuple(float(value) for value in freecompress_keep_ratios),
        min_keep=int(freecompress_min_keep),
        tau=float(freecompress_tau),
        visionzip_dominant_ratio=float(freecompress_visionzip_dominant_ratio),
        visionzip_contextual_ratio=float(freecompress_visionzip_contextual_ratio),
        saliency=str(freecompress_saliency),
        debug_shapes=bool(freecompress_debug_shapes),
    )
    model = QwenPreFreeCompressColQwen2_5(
        base_model=base_model,
        granularities=granularities,
        compact_query_tokens=compact_query_tokens,
        config=config,
    )
    if adapter_path is not None:
        model = _load_adapter_with_fallback(model, Path(adapter_path))
    if eval_mode:
        model.eval()
    return model


__all__ = [
    "QwenPreFreeCompressConfig",
    "QwenPreFreeCompressColQwen2_5",
    "build_qwenpre_freecompress_model",
    "canonicalize_qwenpre_freecompress_method",
]
