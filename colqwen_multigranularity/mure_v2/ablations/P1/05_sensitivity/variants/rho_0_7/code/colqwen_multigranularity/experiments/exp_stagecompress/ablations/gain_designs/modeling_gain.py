from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from colpali_engine.models import ColQwen2_5
from peft import PeftModel

from colqwen_multigranularity.core import (
    MRLColQwen2_5,
    _apply_compat_patch,
    build_stage_specs,
    normalize_granularities,
)
from colqwen_multigranularity.experiments.exp_stagecompress.folder_homo.loss import (
    FolderHomoMRLInBatchNegativeLoss,
)

try:
    from .config import FolderGainOnlyConfig
except ImportError:
    from config import FolderGainOnlyConfig


class GainPatchScorer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_text_context: bool = False,
    ) -> None:
        super().__init__()
        self.use_text_context = bool(use_text_context)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
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

    def forward(
        self,
        tokens: torch.Tensor,
        text_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


class GainOnlyFolderBlock(nn.Module):
    def __init__(self, embed_dim: int, budget: int, *, config: FolderGainOnlyConfig) -> None:
        super().__init__()
        self.budget = int(budget)
        self.config = config
        self.scorer = GainPatchScorer(
            embed_dim,
            num_heads=int(config.scorer_heads),
            dropout=float(config.scorer_dropout),
            use_text_context=bool(config.use_text_context),
        )
        self.folder_alpha = float(config.folder_alpha)
        self.novelty_weight = float(config.novelty_weight)
        self.gate_strength = float(config.gate_strength)
        self.metric_query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.metric_key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.metric_scale = nn.Parameter(torch.tensor(1.0))
        self.metric_bias = nn.Parameter(torch.tensor(0.0))
        self.anchor_query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.anchor_key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.anchor_value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.anchor_context_norm = nn.LayerNorm(embed_dim)
        self.anchor_gain_head = nn.Sequential(
            nn.LayerNorm(embed_dim * 4 + 3),
            nn.Linear(embed_dim * 4 + 3, embed_dim),
            nn.GELU(),
            nn.Dropout(float(config.scorer_dropout)),
            nn.Linear(embed_dim, 1),
        )
        self.recon_query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.recon_key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.recon_value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.recon_head = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Dropout(float(config.scorer_dropout)),
            nn.Linear(embed_dim, embed_dim),
        )

    @staticmethod
    def _normalize_score(score: torch.Tensor) -> torch.Tensor:
        if score.numel() <= 1:
            return torch.ones_like(score)
        lo = score.min()
        hi = score.max()
        return (score - lo) / (hi - lo).clamp_min(1e-6)

    def _anchor_features(self, enhanced: torch.Tensor, coarse_anchors: Optional[torch.Tensor]):
        if coarse_anchors is None or coarse_anchors.numel() == 0:
            return None, None, None
        anchors = coarse_anchors.detach() if self.config.detach_anchors else coarse_anchors
        token_features = F.normalize(enhanced.float(), dim=-1, eps=1e-12)
        anchor_features = F.normalize(anchors.float(), dim=-1, eps=1e-12)
        return anchors, token_features, anchor_features

    def _hard_max_gain(self, token_features: torch.Tensor, anchor_features: torch.Tensor) -> torch.Tensor:
        max_similarity = (token_features @ anchor_features.transpose(0, 1)).max(dim=-1).values
        return 1.0 - max_similarity.clamp(-1.0, 1.0)

    def _learned_metric_gain(self, enhanced: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        q = F.normalize(self.metric_query(enhanced).float(), dim=-1, eps=1e-12)
        k = F.normalize(self.metric_key(anchors.to(dtype=enhanced.dtype)).float(), dim=-1, eps=1e-12)
        logits = q @ k.transpose(0, 1)
        tau = max(float(self.config.gain_tau), 1e-4)
        weights = torch.softmax(logits / tau, dim=-1)
        learned_coverage = (weights * logits.tanh()).sum(dim=-1)
        scale = F.softplus(self.metric_scale.float())
        return torch.sigmoid(self.metric_bias.float() - scale * learned_coverage)

    def _anchor_context(
        self,
        enhanced: torch.Tensor,
        anchors: torch.Tensor,
        *,
        query_proj: nn.Linear,
        key_proj: nn.Linear,
        value_proj: nn.Linear,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = F.normalize(query_proj(enhanced).float(), dim=-1, eps=1e-12)
        k = F.normalize(key_proj(anchors.to(dtype=enhanced.dtype)).float(), dim=-1, eps=1e-12)
        v = value_proj(anchors.to(dtype=enhanced.dtype))
        logits = q @ k.transpose(0, 1) / math.sqrt(max(float(q.shape[-1]), 1.0))
        weights = torch.softmax(logits, dim=-1)
        context = weights.to(dtype=v.dtype) @ v
        return context, weights

    def _learned_anchor_gate_gain(
        self,
        enhanced: torch.Tensor,
        anchors: torch.Tensor,
        token_features: torch.Tensor,
        anchor_features: torch.Tensor,
    ) -> torch.Tensor:
        context, weights = self._anchor_context(
            enhanced,
            anchors,
            query_proj=self.anchor_query,
            key_proj=self.anchor_key,
            value_proj=self.anchor_value,
        )
        context = self.anchor_context_norm(context).to(dtype=enhanced.dtype)

        raw_sim = token_features @ anchor_features.transpose(0, 1)
        max_sim = raw_sim.max(dim=-1).values
        mean_sim = raw_sim.mean(dim=-1)
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1)
        entropy = entropy / math.log(max(int(weights.shape[-1]), 2))
        features = torch.cat(
            [
                enhanced,
                context,
                enhanced - context,
                enhanced * context,
                (1.0 - max_sim.clamp(-1.0, 1.0)).to(dtype=enhanced.dtype).unsqueeze(-1),
                mean_sim.to(dtype=enhanced.dtype).unsqueeze(-1),
                entropy.to(dtype=enhanced.dtype).unsqueeze(-1),
            ],
            dim=-1,
        )
        return torch.sigmoid(self.anchor_gain_head(features).squeeze(-1)).float()

    def _learned_reconstruction_gain(self, enhanced: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        context, _ = self._anchor_context(
            enhanced,
            anchors,
            query_proj=self.recon_query,
            key_proj=self.recon_key,
            value_proj=self.recon_value,
        )
        context = context.to(dtype=enhanced.dtype)
        reconstructed = self.recon_head(torch.cat([enhanced, context, enhanced - context], dim=-1))
        residual = (enhanced - reconstructed).float().norm(dim=-1)
        return residual / math.sqrt(max(float(enhanced.shape[-1]), 1.0))

    def _gain_score(self, enhanced: torch.Tensor, coarse_anchors: Optional[torch.Tensor]) -> torch.Tensor:
        anchors, token_features, anchor_features = self._anchor_features(enhanced, coarse_anchors)
        if anchors is None or token_features is None or anchor_features is None:
            return torch.ones(enhanced.shape[0], dtype=enhanced.dtype, device=enhanced.device)
        mode = self.config.normalized_gain_mode()
        if mode == "hard_max":
            gain = self._hard_max_gain(token_features, anchor_features)
        elif mode == "learned_metric_residual":
            gain = self._learned_metric_gain(enhanced, anchors)
        elif mode == "learned_anchor_gate":
            gain = self._learned_anchor_gate_gain(enhanced, anchors, token_features, anchor_features)
        elif mode == "learned_reconstruction_residual":
            gain = self._learned_reconstruction_gain(enhanced, anchors)
        else:
            raise ValueError(f"Unknown gain mode: {mode}")
        return self._normalize_score(gain).to(dtype=enhanced.dtype)

    @staticmethod
    def _folder_match(metric: torch.Tensor, protect: torch.Tensor, r: int, alpha: float):
        t = int(metric.shape[1])
        r = min(int(r), (t - 1) // 2)
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
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce="sum")
            dst_size = dst_size.scatter_reduce(-2, dst_idx.expand(n, r, dst_size.shape[-1]), gathered_src_size, reduce="sum")
            return torch.cat([unm, dst], dim=1), torch.cat([unm_size, dst_size], dim=1)

        return merge

    def _folder_reduce(self, tokens: torch.Tensor, enhanced: torch.Tensor, protect: torch.Tensor) -> torch.Tensor:
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

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        coarse_anchors: Optional[torch.Tensor] = None,
        text_context: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        if tokens.shape[0] == 0 or self.budget <= 0 or tokens.shape[0] <= self.budget:
            if return_aux:
                aux = {
                    "source_tokens": tokens,
                    "saliency_logits": tokens.new_zeros((tokens.shape[0],), dtype=tokens.dtype),
                    "compressed_len": torch.tensor(tokens.shape[0], device=tokens.device, dtype=torch.long),
                }
                return tokens, aux
            return tokens

        enhanced, saliency, gate = self.scorer(tokens, text_context=text_context)
        importance = self._normalize_score(saliency.float()).to(tokens.dtype)
        gain = self._gain_score(enhanced, coarse_anchors).to(tokens.dtype)
        protect = importance + self.novelty_weight * gain
        continuous_importance = 0.5 * importance + 0.5 * gain
        value_scale = 1.0 + self.gate_strength * gate.to(tokens.dtype) * continuous_importance
        compressed = self._folder_reduce(
            tokens=tokens * value_scale.unsqueeze(-1),
            enhanced=enhanced,
            protect=protect,
        )
        if return_aux:
            aux = {
                "source_tokens": tokens,
                "saliency_logits": importance.to(tokens.dtype),
                "compressed_len": torch.tensor(compressed.shape[0], device=tokens.device, dtype=torch.long),
            }
            return compressed, aux
        return compressed


class GainOnlyFolderCompressor(nn.Module):
    def __init__(self, config: FolderGainOnlyConfig, *, image_token_id: int, crop_counts: Sequence[int], embed_dim: int) -> None:
        super().__init__()
        self.config = config
        self.image_token_id = int(image_token_id)
        self.crop_counts = tuple(int(v) for v in crop_counts)
        self.total_crop_count = int(sum(self.crop_counts))
        if len(self.crop_counts) != 3:
            raise ValueError("FolderGainOnly expects exactly three crop stages.")
        if len(config.budgets) != 3:
            raise ValueError(f"FolderGainOnly budgets must contain three values, got {config.budgets!r}")
        self.blocks = nn.ModuleList(
            [
                GainOnlyFolderBlock(embed_dim=embed_dim, budget=int(config.budgets[index]), config=config)
                for index in range(3)
            ]
        )
        self._last_marc_aux: Optional[Dict[str, object]] = None

    def pop_marc_aux(self) -> Optional[Dict[str, object]]:
        aux = self._last_marc_aux
        self._last_marc_aux = None
        return aux

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

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.LongTensor, attention_mask: torch.Tensor, *, collect_marc_aux: bool = True) -> torch.Tensor:
        active_stages = set(self.config.active_stage_ids())
        collect_marc = bool(collect_marc_aux and self.training and active_stages and self._last_marc_aux is None)
        sequences: List[torch.Tensor] = []
        debug_rows = []
        marc_rows: List[Dict[str, object]] = []
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
                if collect_marc:
                    marc_rows.append({"text_len": int(text_tokens.shape[0]), "stages": []})
                continue

            stage_tokens = self._split_stages(image_tokens)
            text_context = text_tokens.mean(dim=0, keepdim=True) if self.config.use_text_context and text_tokens.numel() > 0 else None
            compressed: List[torch.Tensor] = []
            stage_aux: List[Dict[str, object]] = []
            coarse_anchors: Optional[torch.Tensor] = None
            running_image_offset = 0
            for stage_index, tokens in enumerate(stage_tokens):
                if stage_index in active_stages:
                    if collect_marc:
                        out, aux = self.blocks[stage_index](tokens, coarse_anchors=coarse_anchors, text_context=text_context, return_aux=True)
                    else:
                        out = self.blocks[stage_index](tokens, coarse_anchors=coarse_anchors, text_context=text_context)
                        aux = None
                else:
                    out = tokens
                    aux = None
                compressed.append(out)
                if collect_marc and stage_index in active_stages and aux is not None:
                    start = int(text_tokens.shape[0] + running_image_offset)
                    end = int(start + out.shape[0])
                    stage_aux.append(
                        {
                            "stage_index": int(stage_index),
                            "source_tokens": aux["source_tokens"],
                            "saliency_logits": aux["saliency_logits"],
                            "doc_start": start,
                            "doc_end": end,
                        }
                    )
                running_image_offset += int(out.shape[0])
                coarse_anchors = out if coarse_anchors is None else torch.cat([coarse_anchors, out], dim=0)
            prefix_level = max(1, min(int(getattr(self.config, "eval_prefix_level", 3)), len(compressed)))
            sequence = torch.cat([text_tokens, *compressed[:prefix_level]], dim=0)
            sequences.append(sequence)
            debug_rows.append(tuple(int(x.shape[0]) for x in compressed[:prefix_level]))
            if collect_marc:
                marc_rows.append({"text_len": int(text_tokens.shape[0]), "stages": stage_aux})

        output = self._pad_sequences(sequences, hidden_states.shape[-1])
        if active_stages:
            zero = output.sum() * 0.0
            for idx in active_stages:
                for param in self.blocks[idx].parameters():
                    zero = zero + param.sum() * 0.0
            output = output + zero
        if collect_marc:
            self._last_marc_aux = {"rows": marc_rows, "output_len": int(output.shape[1])}
        if self.config.debug_shapes:
            print(f"[GainOnlyFolderCompressor] rows={debug_rows[:4]} output={list(output.shape)}", flush=True)
        return output


class GainOnlyMRLColQwen2_5(MRLColQwen2_5):
    def __init__(
        self,
        base_model: ColQwen2_5,
        *,
        granularities: Sequence[int] = (1, 2, 4),
        compact_query_tokens: bool = True,
        gain_config: Optional[FolderGainOnlyConfig] = None,
    ) -> None:
        super().__init__(base_model=base_model, granularities=granularities, compact_query_tokens=compact_query_tokens)
        self.gain_config = gain_config or FolderGainOnlyConfig(enabled=False)
        self.folder_gain_only = GainOnlyFolderCompressor(
            self.gain_config,
            image_token_id=self.config.image_token_id,
            crop_counts=[spec.crop_count for spec in self.stage_specs],
            embed_dim=self.dim,
        )

    def pop_marc_aux(self) -> Optional[Dict[str, object]]:
        return self.folder_gain_only.pop_marc_aux()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        has_images = pixel_values is not None and image_grid_thw is not None and getattr(pixel_values, "numel", lambda: 0)() > 0 and getattr(image_grid_thw, "numel", lambda: 0)() > 0
        is_query = bool(kwargs.get("is_query", False))
        if is_query:
            self.folder_gain_only.pop_marc_aux()
        hidden_states = self._project_hidden_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values if has_images else None,
            image_grid_thw=image_grid_thw if has_images else None,
            **kwargs,
        )
        if (not self.gain_config.enabled) or len(self.gain_config.active_stage_ids()) == 0:
            return self._compact_doc_embeddings(hidden_states, input_ids, attention_mask)
        return self.folder_gain_only(hidden_states, input_ids, attention_mask, collect_marc_aux=not is_query)


def _load_adapter_with_fallback(base_model: ColQwen2_5, adapter_path: Path):
    adapter_bin = adapter_path / "adapter_model.bin"
    if not adapter_bin.exists():
        return PeftModel.from_pretrained(base_model, adapter_path)
    state_dict = torch.load(adapter_bin, map_location="cpu")
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("base_model.model.base_model.custom_text_proj."):
            key = key.replace("base_model.model.base_model.custom_text_proj.", "base_model.model.custom_text_proj.", 1)
        if key.startswith("base_model.model.base_model.model."):
            key = key.replace("base_model.model.base_model.model.", "base_model.model.model.", 1)
        remapped[key] = value
    with TemporaryDirectory(prefix="folder_gain_only_eval_adapter_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        (tmpdir_path / "adapter_config.json").write_text((adapter_path / "adapter_config.json").read_text())
        torch.save(remapped, tmpdir_path / "adapter_model.bin")
        return PeftModel.from_pretrained(base_model, tmpdir_path)


def build_gain_model(
    model_name_or_path: str,
    *,
    granularities: Sequence[int] = (1, 2, 4),
    gain_config: Optional[FolderGainOnlyConfig] = None,
    attn_implementation: Optional[str] = "flash_attention_2",
    use_liger_kernel: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
    adapter_path: Optional[str] = None,
    eval_mode: bool = False,
    compact_query_tokens: bool = True,
):
    granularities = normalize_granularities(granularities)
    if len(build_stage_specs(granularities)) != 3:
        raise ValueError("FolderGainOnly expects exactly three stages.")
    base_model = ColQwen2_5.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        use_cache=False,
        attn_implementation=attn_implementation,
        use_liger_kernel=use_liger_kernel,
    )
    if not hasattr(base_model, "custom_text_proj"):
        raise TypeError("Expected a ColQwen2_5 checkpoint with custom_text_proj.")
    _apply_compat_patch(base_model)
    if adapter_path is not None:
        base_model = _load_adapter_with_fallback(base_model, Path(adapter_path))
    model = GainOnlyMRLColQwen2_5(
        base_model=base_model,
        granularities=granularities,
        compact_query_tokens=compact_query_tokens,
        gain_config=gain_config,
    )
    if eval_mode:
        model.eval()
    return model


class FolderGainOnlyMRLInBatchNegativeLoss(FolderHomoMRLInBatchNegativeLoss):
    pass
