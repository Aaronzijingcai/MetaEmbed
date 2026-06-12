from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from colqwen_multigranularity.core import build_stage_specs, normalize_granularities

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


def canonicalize_freecompress_method(method: str) -> str:
    key = str(method).strip().lower()
    if key not in _METHOD_ALIASES:
        raise ValueError(f"Unknown freecompress method {method!r}; expected one of {sorted(_METHOD_ALIASES)}.")
    return _METHOD_ALIASES[key]


@dataclass(frozen=True)
class FreeCompressConfig:
    enabled: bool = True
    method: str = "folder"
    compress_stages: str = "g2g3"
    keep_ratios: Tuple[float, float, float] = (1.0, 0.5, 0.25)
    stage_budgets: Tuple[int, int, int] = (0, 0, 0)
    min_keep: int = 1
    tau: float = 1.0
    visionzip_dominant_ratio: float = 0.9
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


class FreeCompressModelWrapper(nn.Module):
    """Eval-only wrapper that compresses document image tokens after projection."""

    def __init__(self, model: nn.Module, *, granularities: Sequence[int] = (1, 2, 4), config: Optional[FreeCompressConfig] = None) -> None:
        super().__init__()
        self.model = model
        self.freecompress_config = config or FreeCompressConfig(enabled=False)
        self.granularities = normalize_granularities(granularities)
        self.stage_specs = build_stage_specs(self.granularities)
        if len(self.stage_specs) != 3:
            raise ValueError("FreeCompress expects exactly three MRL stages.")
        self.method = canonicalize_freecompress_method(self.freecompress_config.method)
        self.config = self._resolve_model_config(model)
        self.main_input_name = getattr(model, "main_input_name", "input_ids")
        self.dim = int(getattr(model, "dim", getattr(model, "embedding_dim", 128)))
        self.image_token_id = int(getattr(self.config, "image_token_id"))
        self._debug_count = 0

    @staticmethod
    def _resolve_model_config(model: nn.Module):
        config = getattr(model, "config", None)
        if config is not None and hasattr(config, "image_token_id"):
            return config
        for module in model.modules():
            config = getattr(module, "config", None)
            if config is not None and hasattr(config, "image_token_id"):
                return config
        raise AttributeError("Could not locate a model config with image_token_id.")

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def train(self, mode: bool = True):
        self.model.train(mode)
        return super().train(mode)

    def eval(self):
        self.model.eval()
        return super().eval()

    def save_pretrained(self, save_dir: str, **kwargs):
        if hasattr(self.model, "save_pretrained"):
            return self.model.save_pretrained(save_dir, **kwargs)
        raise AttributeError(f"{type(self.model).__name__} does not implement save_pretrained")

    @staticmethod
    def _pad_sequences(sequences: Sequence[torch.Tensor], dim: int) -> torch.Tensor:
        max_length = max(sequence.shape[0] for sequence in sequences)
        output = sequences[0].new_zeros((len(sequences), max_length, dim))
        for index, sequence in enumerate(sequences):
            output[index, : sequence.shape[0]] = sequence
        return output

    @staticmethod
    def _compact_sequences(hidden_states: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        sequences: List[torch.Tensor] = []
        for row, mask in zip(hidden_states, token_mask):
            if mask.shape[0] == row.shape[0]:
                compact = row[mask]
            else:
                compact = row[row.abs().sum(dim=-1).ne(0)]
            if compact.numel() == 0:
                compact = row.new_zeros((1, row.shape[-1]))
            sequences.append(compact)
        return FreeCompressModelWrapper._pad_sequences(sequences, hidden_states.shape[-1])

    def _visual_spatial_merge_size(self) -> int:
        for source in [self.model, *self.model.modules()]:
            value = getattr(source, "spatial_merge_size", None)
            if callable(value):
                value = value()
            if value is not None:
                return int(value)
            visual = getattr(source, "visual", None)
            visual_config = getattr(visual, "config", None)
            value = getattr(visual_config, "spatial_merge_size", None)
            if value is not None:
                return int(value)
            base_model = getattr(source, "base_model", None)
            visual = getattr(base_model, "visual", None)
            visual_config = getattr(visual, "config", None)
            value = getattr(visual_config, "spatial_merge_size", None)
            if value is not None:
                return int(value)
        return 2

    def _image_grid_token_counts(self, image_grid_thw: torch.LongTensor) -> List[int]:
        merge_size = self._visual_spatial_merge_size()
        denom = merge_size * merge_size
        counts: List[int] = []
        for row in image_grid_thw.detach().to("cpu").tolist():
            t, h, w = [int(value) for value in row]
            total = t * h * w
            if total % denom != 0:
                raise RuntimeError(f"FreeCompress image grid is not divisible by merge size: grid={row} merge_size={merge_size}.")
            counts.append(total // denom)
        return counts

    def _stage_and_crop_maps(self, *, input_ids: torch.LongTensor, image_grid_thw: Optional[torch.LongTensor]) -> Tuple[torch.LongTensor, torch.LongTensor]:
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
                raise RuntimeError("FreeCompress sample image token mismatch: " f"sample={batch_index} placeholders={sample_tokens} consumed_grid_tokens={consumed}.")
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
            raise RuntimeError(f"FreeCompress image grid cursor mismatch: {grid_cursor}/{len(crop_token_counts)}.")
        return stage_map, crop_map

    def _stage_budget(self, stage_index: int, length: int) -> int:
        stage_budgets = self.freecompress_config.stage_budgets
        if len(stage_budgets) != 3:
            raise ValueError(f"stage_budgets must have exactly 3 entries, got {stage_budgets!r}")
        configured_budget = int(stage_budgets[stage_index])
        if configured_budget > 0:
            # Stage budgets follow mlppost semantics: total target tokens per g1/g2/g3 stage.
            # FreeCompress keeps crop blocks independent, so divide the stage target across crops.
            crop_count = max(int(self.stage_specs[stage_index].crop_count), 1)
            budget = int(math.ceil(float(configured_budget) / float(crop_count)))
        else:
            ratios = self.freecompress_config.keep_ratios
            if len(ratios) != 3:
                raise ValueError(f"keep_ratios must have exactly 3 entries, got {ratios!r}")
            budget = int(math.ceil(float(length) * float(ratios[stage_index])))
        budget = max(int(self.freecompress_config.min_keep), budget)
        return min(max(budget, 1), int(length))

    def _features_and_saliency(self, tokens: torch.Tensor, text_context: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        features = F.normalize(tokens.float(), dim=-1, eps=1e-12)
        saliency_mode = str(self.freecompress_config.saliency).lower()
        if saliency_mode in {"text", "auto"} and text_context is not None and text_context.numel() > 0:
            context = F.normalize(text_context.reshape(1, -1).float(), dim=-1, eps=1e-12)
            raw = (features @ context.transpose(0, 1)).squeeze(-1)
        else:
            raw = (features @ features.transpose(0, 1)).mean(dim=-1)
        if raw.numel() <= 1:
            return features, torch.ones_like(raw, dtype=torch.float32)
        saliency = (raw.float() - raw.float().min()) / (raw.float().max() - raw.float().min()).clamp_min(1e-6)
        return features, saliency + 1e-6

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
            saliency_term = saliency.float().clamp_min(1e-12).pow(float(self.freecompress_config.scope_alpha))
        for _ in range(min(int(num_selected_token), n)):
            unselected_mask = ~selected
            gains = torch.maximum(torch.zeros(1, dtype=metric.dtype, device=metric.device), cosine_simi.masked_fill(~unselected_mask.unsqueeze(0), 0) - cur_max.unsqueeze(1)).sum(dim=0)
            if self.freecompress_config.scope_combined == "multi":
                gains = gains * saliency_term
            elif self.freecompress_config.scope_combined == "add":
                gains = gains + saliency_term
            else:
                raise ValueError(f"Unknown SCOPE combination: {self.freecompress_config.scope_combined!r}")
            gains = gains.masked_fill(~unselected_mask, float("-inf"))
            best_idx = gains.argmax()
            selected[best_idx] = True
            selected_idx.append(best_idx)
            cur_max = torch.maximum(cur_max, cosine_simi[best_idx])
        if not selected_idx:
            return torch.empty(0, dtype=torch.long, device=metric.device)
        return torch.stack(selected_idx)

    def _compress_prumerge(self, tokens: torch.Tensor, features: torch.Tensor, saliency: torch.Tensor, budget: int) -> torch.Tensor:
        dtype = tokens.dtype
        tau = max(float(self.freecompress_config.tau), 1e-6)
        keep_budget, merge_budget, residual_budget = self._partition_prumerge_budget(budget)
        keep_budget = min(keep_budget, tokens.shape[0])
        _, keep_idx = torch.topk(saliency, k=keep_budget, dim=0, largest=True)
        keep_tokens = tokens.index_select(0, keep_idx)
        keep_features = features.index_select(0, keep_idx)
        if keep_budget >= tokens.shape[0]:
            return F.normalize(keep_tokens.float(), dim=-1, eps=1e-12).to(dtype=dtype)
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
        compressed = torch.cat(parts, dim=0)[:budget]
        return F.normalize(compressed, dim=-1, eps=1e-12).to(dtype=dtype)

    def _compress_visionzip(self, tokens: torch.Tensor, features: torch.Tensor, saliency: torch.Tensor, budget: int) -> torch.Tensor:
        dtype = tokens.dtype
        if budget <= 1:
            _, idx = torch.topk(saliency, k=1, dim=0, largest=True)
            return F.normalize(tokens.index_select(0, idx).float(), dim=-1, eps=1e-12).to(dtype=dtype)
        dominant_budget = max(1, int(round(float(budget) * float(self.freecompress_config.visionzip_dominant_ratio))))
        dominant_budget = min(dominant_budget, budget - 1, tokens.shape[0])
        contextual_budget = budget - dominant_budget
        _, dominant_idx = torch.topk(saliency, k=dominant_budget, dim=0, largest=True)
        dominant_idx = dominant_idx.sort().values
        dominant_tokens = tokens.index_select(0, dominant_idx).float()
        if dominant_budget >= tokens.shape[0]:
            return F.normalize(dominant_tokens, dim=-1, eps=1e-12).to(dtype=dtype)
        residual_mask = torch.ones(tokens.shape[0], dtype=torch.bool, device=tokens.device)
        residual_mask[dominant_idx] = False
        residual_tokens = tokens[residual_mask]
        residual_features = features[residual_mask]
        residual_scores = saliency[residual_mask]
        contextual_budget = min(contextual_budget, residual_tokens.shape[0])
        if contextual_budget <= 0:
            return F.normalize(dominant_tokens, dim=-1, eps=1e-12).to(dtype=dtype)
        contextual_idx = self._select_uniform_indices(residual_tokens.shape[0], contextual_budget, tokens.device)
        contextual_tokens = residual_tokens.index_select(0, contextual_idx).float()
        contextual_features = residual_features.index_select(0, contextual_idx)
        merge_mask = torch.ones(residual_tokens.shape[0], dtype=torch.bool, device=tokens.device)
        merge_mask[contextual_idx] = False
        merge_tokens = residual_tokens[merge_mask]
        merge_features = residual_features[merge_mask]
        merge_scores = residual_scores[merge_mask]
        if merge_tokens.numel() > 0:
            logits = merge_features @ contextual_features.transpose(0, 1)
            logits = logits + merge_scores.unsqueeze(-1)
            assignments = logits.argmax(dim=-1)
            tau = max(float(self.freecompress_config.tau), 1e-6)
            merge_weights = torch.softmax(merge_scores / tau, dim=0)
            aggregated = contextual_tokens.new_zeros(contextual_tokens.shape)
            aggregated.scatter_add_(0, assignments.unsqueeze(-1).expand_as(merge_tokens), merge_tokens.float() * merge_weights.unsqueeze(-1))
            mass = merge_weights.new_zeros((contextual_budget,))
            mass.scatter_add_(0, assignments, merge_weights)
            contextual_tokens = contextual_tokens + aggregated / mass.clamp_min(1.0).unsqueeze(-1)
        compressed = torch.cat([dominant_tokens, contextual_tokens], dim=0)[:budget]
        return F.normalize(compressed, dim=-1, eps=1e-12).to(dtype=dtype)

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
            size = size[:, :budget, :]
        out = x * (1.0 + size.clamp_min(1e-12).log())
        return F.normalize(out.squeeze(0), dim=-1, eps=1e-12).to(dtype=dtype)

    def _compress_scope(self, tokens: torch.Tensor, features: torch.Tensor, saliency: torch.Tensor, budget: int) -> torch.Tensor:
        dtype = tokens.dtype
        selected_idx = self._scope_select(features, budget, saliency)
        selected_tokens = tokens.index_select(0, selected_idx)
        return F.normalize(selected_tokens.float(), dim=-1, eps=1e-12).to(dtype=dtype)

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

    def _compress_batch(self, hidden_states: torch.Tensor, input_ids: torch.LongTensor, attention_mask: torch.Tensor, image_grid_thw: Optional[torch.LongTensor]) -> torch.Tensor:
        stage_map, crop_map = self._stage_and_crop_maps(input_ids=input_ids, image_grid_thw=image_grid_thw)
        active_stages = set(self.freecompress_config.active_stage_ids())
        sequences: List[torch.Tensor] = []
        debug_rows: List[Tuple[int, int, int, int, int]] = []
        for batch_index, (row_hidden, row_ids, row_attn) in enumerate(zip(hidden_states, input_ids, attention_mask)):
            active = row_attn.to(dtype=torch.bool)
            active_positions = active.nonzero(as_tuple=False).squeeze(-1)
            if active_positions.numel() == 0:
                sequences.append(row_hidden.new_zeros((1, row_hidden.shape[-1])))
                continue
            text_mask = active & row_ids.ne(self.image_token_id)
            text_tokens = row_hidden[text_mask]
            text_context = text_tokens.mean(dim=0, keepdim=True) if text_tokens.numel() > 0 else None
            sample_crop_ids = torch.unique(crop_map[batch_index][(crop_map[batch_index] >= 0) & active])
            compressed_by_crop: Dict[int, torch.Tensor] = {}
            before_image_tokens = 0
            after_image_tokens = 0
            for crop_id_tensor in sample_crop_ids:
                crop_id = int(crop_id_tensor.item())
                crop_positions = ((crop_map[batch_index] == crop_id) & active).nonzero(as_tuple=False).squeeze(-1)
                if crop_positions.numel() == 0:
                    continue
                stage_index = int(stage_map[batch_index, crop_positions[0]].item())
                crop_tokens = row_hidden.index_select(0, crop_positions)
                before_image_tokens += int(crop_tokens.shape[0])
                if stage_index in active_stages:
                    budget = self._stage_budget(stage_index, crop_tokens.shape[0])
                    compressed = self._compress_tokens(crop_tokens, budget, text_context)
                else:
                    compressed = crop_tokens
                after_image_tokens += int(compressed.shape[0])
                compressed_by_crop[crop_id] = compressed
            parts: List[torch.Tensor] = []
            emitted_crops = set()
            for pos_tensor in active_positions:
                pos = int(pos_tensor.item())
                crop_id = int(crop_map[batch_index, pos].item())
                if crop_id >= 0:
                    if crop_id not in emitted_crops:
                        parts.append(compressed_by_crop.get(crop_id, row_hidden[pos : pos + 1]))
                        emitted_crops.add(crop_id)
                    continue
                parts.append(row_hidden[pos : pos + 1])
            sequence = torch.cat(parts, dim=0) if parts else row_hidden.new_zeros((1, row_hidden.shape[-1]))
            sequences.append(sequence)
            debug_rows.append((batch_index, int(active_positions.numel()), before_image_tokens, after_image_tokens, int(sequence.shape[0])))
        if self.freecompress_config.debug_shapes and self._debug_count < 8:
            self._debug_count += 1
            print(f"[FreeCompress] method={self.method} rows={debug_rows[:4]}", flush=True)
        return self._pad_sequences(sequences, hidden_states.shape[-1])

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        is_query: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        model_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        if pixel_values is not None:
            model_kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            model_kwargs["image_grid_thw"] = image_grid_thw
        try:
            hidden_states = self.model(is_query=is_query, **model_kwargs)
        except TypeError:
            hidden_states = self.model(**model_kwargs)
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(f"Expected wrapped model to return a tensor, got {type(hidden_states).__name__}.")
        if hidden_states.ndim != 3:
            return hidden_states
        if hidden_states.shape[:2] != input_ids.shape:
            nonzero_mask = hidden_states.abs().sum(dim=-1).ne(0)
            return self._compact_sequences(hidden_states, nonzero_mask)
        has_images = image_grid_thw is not None and getattr(image_grid_thw, "numel", lambda: 0)() > 0 and input_ids.eq(self.image_token_id).any().item()
        active_stages = self.freecompress_config.active_stage_ids()
        if bool(is_query) or (not has_images) or len(active_stages) == 0:
            return self._compact_sequences(hidden_states, attention_mask.to(dtype=torch.bool))
        return self._compress_batch(hidden_states, input_ids, attention_mask, image_grid_thw)
