from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from safetensors.torch import load_file as load_safetensors_file
from colpali_engine.models import ColQwen2_5

from colqwen_multigranularity.core import MRLColQwen2_5, _apply_compat_patch, build_stage_specs, normalize_granularities
from .config import FolderDartPivotConfig


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


class DartPivotFolderBlock(nn.Module):
    def __init__(self, embed_dim: int, budget: int, *, config: FolderDartPivotConfig) -> None:
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

    @staticmethod
    def _normalize_score(score: torch.Tensor) -> torch.Tensor:
        if score.numel() <= 1:
            return torch.ones_like(score)
        lo = score.min()
        hi = score.max()
        return (score - lo) / (hi - lo).clamp_min(1e-6)

    def _select_pivots(self, coarse_anchors: Optional[torch.Tensor], text_context: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if coarse_anchors is None or coarse_anchors.numel() == 0:
            return None
        anchors = coarse_anchors.detach() if self.config.detach_anchors else coarse_anchors
        if anchors.ndim != 2 or anchors.shape[0] == 0:
            return None
        k = min(max(int(self.config.pivot_count), 1), int(anchors.shape[0]))
        mode = str(self.config.pivot_score).strip().lower()
        with torch.no_grad():
            if mode == 'norm':
                scores = anchors.float().norm(dim=-1)
            elif mode == 'saliency':
                _, scores, _ = self.scorer(anchors, text_context=text_context)
                scores = scores.float()
            elif mode == 'uniform':
                idx = torch.linspace(0, anchors.shape[0] - 1, steps=k, device=anchors.device).round().long()
                return anchors.index_select(0, idx)
            else:
                raise ValueError(f'Unknown pivot_score={self.config.pivot_score!r}')
            idx = torch.topk(scores, k=k, dim=0, largest=True).indices
        return anchors.index_select(0, idx)

    def _novelty(self, enhanced: torch.Tensor, coarse_anchors: Optional[torch.Tensor], text_context: Optional[torch.Tensor]) -> torch.Tensor:
        pivots = self._select_pivots(coarse_anchors, text_context=text_context)
        if pivots is None or pivots.numel() == 0:
            return torch.ones(enhanced.shape[0], dtype=enhanced.dtype, device=enhanced.device)
        token_features = F.normalize(enhanced.float(), dim=-1, eps=1e-12)
        pivot_features = F.normalize(pivots.float(), dim=-1, eps=1e-12)
        max_similarity = (token_features @ pivot_features.transpose(0, 1)).max(dim=-1).values
        novelty = 1.0 - max_similarity.clamp(-1.0, 1.0)
        return self._normalize_score(novelty).to(dtype=enhanced.dtype)

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

    def _folder_reduce(self, tokens: torch.Tensor, enhanced: torch.Tensor, protect: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError(f'DartPivotFolder expects rank-2 tokens, got {tuple(tokens.shape)}')
        x = tokens.unsqueeze(0)
        metric = enhanced.unsqueeze(0)
        protect = protect.unsqueeze(0)
        size = torch.ones_like(x[..., 0, None])
        remaining = max(int(tokens.shape[0]) - int(self.budget), 0)
        while remaining > 0 and x.shape[1] > 1:
            r_now = min(remaining, (x.shape[1] - 1) // 2)
            merge = self._folder_match(metric=metric, protect=protect, r=r_now, alpha=self.folder_alpha)
            if merge is None:
                break
            x, size = merge(x * size, size)
            metric = x / size.clamp_min(1e-12)
            protect = metric.norm(dim=-1)
            remaining -= r_now
        if x.shape[1] > self.budget:
            x = x[:, : self.budget, :]
            size = size[:, : self.budget, :]
        out = x * (1.0 + size.clamp_min(1e-12).log())
        return F.normalize(out.squeeze(0), dim=-1)

    def forward(self, tokens: torch.Tensor, *, coarse_anchors: Optional[torch.Tensor] = None, text_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if tokens.shape[0] == 0 or self.budget <= 0:
            return tokens.new_zeros((0, tokens.shape[-1]))
        if tokens.shape[0] <= self.budget:
            return F.normalize(tokens, dim=-1)
        enhanced, saliency, gate = self.scorer(tokens, text_context=text_context)
        saliency_norm = self._normalize_score(saliency.float()).to(tokens.dtype)
        novelty = self._novelty(enhanced, coarse_anchors, text_context=text_context).to(tokens.dtype)
        protect = saliency_norm + self.novelty_weight * novelty
        continuous_importance = 0.5 * saliency_norm + 0.5 * novelty
        value_scale = 1.0 + self.gate_strength * gate.to(tokens.dtype) * continuous_importance
        gated_tokens = tokens * value_scale.unsqueeze(-1)
        return self._folder_reduce(tokens=gated_tokens, enhanced=enhanced, protect=protect)


class DartPivotFolderCompressor(nn.Module):
    def __init__(self, config: FolderDartPivotConfig, *, image_token_id: int, crop_counts: Sequence[int], embed_dim: int) -> None:
        super().__init__()
        self.config = config
        self.image_token_id = int(image_token_id)
        self.crop_counts = tuple(int(v) for v in crop_counts)
        self.total_crop_count = int(sum(self.crop_counts))
        if len(self.crop_counts) != 3:
            raise ValueError('DartPivotFolder expects exactly three crop stages.')
        if len(config.budgets) != 3:
            raise ValueError(f'FolderDartPivot budgets must contain three values, got {config.budgets!r}')
        self.blocks = nn.ModuleList([
            DartPivotFolderBlock(embed_dim=embed_dim, budget=int(config.budgets[index]), config=config)
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
            text_context = text_tokens.mean(dim=0, keepdim=True) if self.config.use_text_context and text_tokens.numel() > 0 else None
            compressed: List[torch.Tensor] = []
            coarse_anchors: Optional[torch.Tensor] = None
            for stage_index, tokens in enumerate(stage_tokens):
                if stage_index in active_stages:
                    out = self.blocks[stage_index](tokens, coarse_anchors=coarse_anchors, text_context=text_context)
                else:
                    out = tokens
                compressed.append(out)
                coarse_anchors = out if coarse_anchors is None else torch.cat([coarse_anchors, out], dim=0)
            sequence = torch.cat([text_tokens, *compressed], dim=0)
            sequences.append(sequence)
            debug_rows.append(tuple(int(x.shape[0]) for x in compressed))

        output = self._pad_sequences(sequences, hidden_states.shape[-1])
        active_stages = set(self.config.active_stage_ids())
        if active_stages:
            zero = output.sum() * 0.0
            for idx in active_stages:
                for param in self.blocks[idx].parameters():
                    zero = zero + param.sum() * 0.0
            output = output + zero
        if self.config.debug_shapes:
            print(f'[DartPivotFolderCompressor] rows={debug_rows[:4]} output={list(output.shape)}', flush=True)
        return output


class DartPivotFolderMRLColQwen2_5(MRLColQwen2_5):
    def __init__(self, base_model: ColQwen2_5, *, granularities: Sequence[int] = (1, 2, 4), compact_query_tokens: bool = True, folder_dart_pivot_config: Optional[FolderDartPivotConfig] = None) -> None:
        super().__init__(base_model=base_model, granularities=granularities, compact_query_tokens=compact_query_tokens)
        self.folder_dart_pivot_config = folder_dart_pivot_config or FolderDartPivotConfig(enabled=False)
        self.folder_dart_pivot = DartPivotFolderCompressor(
            self.folder_dart_pivot_config,
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
        if (not self.folder_dart_pivot_config.enabled) or len(self.folder_dart_pivot_config.active_stage_ids()) == 0:
            return self._compact_doc_embeddings(hidden_states, input_ids, attention_mask)
        return self.folder_dart_pivot(hidden_states, input_ids, attention_mask)


def _load_adapter_with_fallback(base_model: ColQwen2_5, adapter_path: Path):
    adapter_bin = adapter_path / 'adapter_model.bin'
    adapter_safetensors = adapter_path / 'adapter_model.safetensors'
    if adapter_bin.exists():
        state_dict = torch.load(adapter_bin, map_location='cpu')
    elif adapter_safetensors.exists():
        state_dict = load_safetensors_file(str(adapter_safetensors), device='cpu')
    else:
        return PeftModel.from_pretrained(base_model, adapter_path)
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith('base_model.model.base_model.custom_text_proj.'):
            key = key.replace('base_model.model.base_model.custom_text_proj.', 'base_model.model.custom_text_proj.', 1)
        if key.startswith('base_model.model.base_model.model.'):
            key = key.replace('base_model.model.base_model.model.', 'base_model.model.model.', 1)
        remapped[key] = value
    with TemporaryDirectory(prefix='folder_dart_pivot_eval_adapter_') as tmpdir:
        tmpdir_path = Path(tmpdir)
        (tmpdir_path / 'adapter_config.json').write_text((adapter_path / 'adapter_config.json').read_text())
        torch.save(remapped, tmpdir_path / 'adapter_model.bin')
        return PeftModel.from_pretrained(base_model, tmpdir_path)


def build_folder_dart_pivot_model(model_name_or_path: str, *, granularities: Sequence[int] = (1, 2, 4), folder_dart_pivot_config: Optional[FolderDartPivotConfig] = None, attn_implementation: Optional[str] = 'flash_attention_2', use_liger_kernel: bool = False, torch_dtype: torch.dtype = torch.bfloat16, adapter_path: Optional[str] = None, eval_mode: bool = False, compact_query_tokens: bool = True):
    granularities = normalize_granularities(granularities)
    if len(build_stage_specs(granularities)) != 3:
        raise ValueError('FolderDartPivot expects exactly three stages.')
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
    model = DartPivotFolderMRLColQwen2_5(
        base_model=base_model,
        granularities=granularities,
        compact_query_tokens=compact_query_tokens,
        folder_dart_pivot_config=folder_dart_pivot_config,
    )
    if eval_mode:
        model.eval()
    return model
