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
    from .config import FolderImportanceConfig
except ImportError:  # direct script execution from a non-package path
    from config import FolderImportanceConfig


class ImportancePatchScorer(nn.Module):
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = tokens.unsqueeze(0)
        attn_out, attn_weights = self.attn(
            x,
            x,
            x,
            need_weights=True,
            average_attn_weights=False,
        )
        x = self.norm(x + attn_out)
        x = x + self.mlp(x)
        enhanced = x.squeeze(0)
        if self.use_text_context and self.text_proj is not None and text_context is not None and text_context.numel() > 0:
            enhanced = enhanced + self.text_proj(text_context.reshape(1, -1)).expand_as(enhanced)
        saliency = self.score_head(enhanced).squeeze(-1)
        gate = torch.sigmoid(self.gate_head(enhanced).squeeze(-1))

        # attn_weights: [1, heads, target_tokens, source_tokens].
        # Keep the full token graph for centrality-style importance scores.
        token_attention = attn_weights.squeeze(0).mean(dim=0).to(dtype=tokens.dtype)
        received_attention = token_attention.mean(dim=0)
        return enhanced, saliency, gate, received_attention, token_attention


class ImportanceFolderBlock(nn.Module):
    def __init__(self, embed_dim: int, budget: int, *, config: FolderImportanceConfig) -> None:
        super().__init__()
        self.budget = int(budget)
        self.config = config
        self.scorer = ImportancePatchScorer(
            embed_dim,
            num_heads=int(config.scorer_heads),
            dropout=float(config.scorer_dropout),
            use_text_context=bool(config.use_text_context),
        )
        self.folder_alpha = float(config.folder_alpha)
        self.novelty_weight = float(config.novelty_weight)
        self.gate_strength = float(config.gate_strength)
        self.pagerank_damping = float(config.pagerank_damping)
        self.pagerank_iters = int(config.pagerank_iters)

    @staticmethod
    def _normalize_score(score: torch.Tensor) -> torch.Tensor:
        if score.numel() <= 1:
            return torch.ones_like(score)
        lo = score.min()
        hi = score.max()
        return (score - lo) / (hi - lo).clamp_min(1e-6)

    def _novelty(self, enhanced: torch.Tensor, coarse_anchors: Optional[torch.Tensor]) -> torch.Tensor:
        if coarse_anchors is None or coarse_anchors.numel() == 0:
            return torch.ones(enhanced.shape[0], dtype=enhanced.dtype, device=enhanced.device)
        anchors = coarse_anchors.detach() if self.config.detach_anchors else coarse_anchors
        token_features = F.normalize(enhanced.float(), dim=-1, eps=1e-12)
        anchor_features = F.normalize(anchors.float(), dim=-1, eps=1e-12)
        max_similarity = (token_features @ anchor_features.transpose(0, 1)).max(dim=-1).values
        novelty = 1.0 - max_similarity.clamp(-1.0, 1.0)
        return self._normalize_score(novelty).to(dtype=enhanced.dtype)

    def _pagerank_score(self, token_attention: torch.Tensor) -> torch.Tensor:
        n = int(token_attention.shape[0])
        if n <= 1:
            return torch.ones(n, dtype=token_attention.dtype, device=token_attention.device)
        transition = token_attention.float().clamp_min(0.0)
        transition = transition / transition.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        rank = torch.full((n,), 1.0 / n, dtype=transition.dtype, device=transition.device)
        teleport = torch.full_like(rank, 1.0 / n)
        damping = min(max(self.pagerank_damping, 0.0), 0.99)
        for _ in range(max(self.pagerank_iters, 1)):
            rank = (1.0 - damping) * teleport + damping * (transition.transpose(0, 1) @ rank)
        return self._normalize_score(rank).to(dtype=token_attention.dtype)

    def _attention_confidence_score(self, token_attention: torch.Tensor, received_attention: torch.Tensor) -> torch.Tensor:
        n = int(token_attention.shape[0])
        if n <= 1:
            return torch.ones(n, dtype=token_attention.dtype, device=token_attention.device)
        probs = token_attention.float().clamp_min(0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
        confidence = 1.0 - entropy / math.log(max(n, 2))
        score = received_attention.float() * confidence.clamp_min(0.0)
        return self._normalize_score(score).to(dtype=token_attention.dtype)

    def _importance_score(
        self,
        *,
        enhanced: torch.Tensor,
        saliency_logits: torch.Tensor,
        gate: torch.Tensor,
        received_attention: torch.Tensor,
        token_attention: torch.Tensor,
    ) -> torch.Tensor:
        mode = str(self.config.importance_mode).strip().lower().replace("-", "_")
        mlp_score = self._normalize_score(saliency_logits.float()).to(enhanced.dtype)
        attn_score = self._normalize_score(received_attention.float()).to(enhanced.dtype)
        gate_score = self._normalize_score(gate.float()).to(enhanced.dtype)

        if mode in {"mlp", "mlp_saliency", "saliency", "score_head"}:
            return mlp_score
        if mode in {"mha_attn", "mha_received_attn", "received_attn", "attention", "attn"}:
            return attn_score
        if mode in {"mha_pagerank", "pagerank", "attention_pagerank"}:
            return self._pagerank_score(token_attention).to(enhanced.dtype)
        if mode in {"mha_entropy_confidence", "attn_confidence", "entropy_confidence"}:
            return self._attention_confidence_score(token_attention, received_attention).to(enhanced.dtype)
        if mode in {"learned_gate", "gate"}:
            return gate_score
        if mode in {"mlp_mha", "hybrid_mha"}:
            blend = min(max(float(self.config.importance_blend), 0.0), 1.0)
            return blend * mlp_score + (1.0 - blend) * attn_score
        raise ValueError(f"Unknown importance_mode={self.config.importance_mode!r}")

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

        enhanced, saliency, gate, received_attention, token_attention = self.scorer(tokens, text_context=text_context)
        importance = self._importance_score(
            enhanced=enhanced,
            saliency_logits=saliency,
            gate=gate,
            received_attention=received_attention,
            token_attention=token_attention,
        )
        novelty = self._novelty(enhanced, coarse_anchors).to(tokens.dtype)
        protect = importance + self.novelty_weight * novelty
        continuous_importance = 0.5 * importance + 0.5 * novelty
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


class ImportanceFolderCompressor(nn.Module):
    def __init__(self, config: FolderImportanceConfig, *, image_token_id: int, crop_counts: Sequence[int], embed_dim: int) -> None:
        super().__init__()
        self.config = config
        self.image_token_id = int(image_token_id)
        self.crop_counts = tuple(int(v) for v in crop_counts)
        self.total_crop_count = int(sum(self.crop_counts))
        if len(self.crop_counts) != 3:
            raise ValueError("FolderImportance expects exactly three crop stages.")
        if len(config.budgets) != 3:
            raise ValueError(f"FolderImportance budgets must contain three values, got {config.budgets!r}")
        self.blocks = nn.ModuleList(
            [
                ImportanceFolderBlock(embed_dim=embed_dim, budget=int(config.budgets[index]), config=config)
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
            print(f"[ImportanceFolderCompressor] rows={debug_rows[:4]} output={list(output.shape)}", flush=True)
        return output


class ImportanceMRLColQwen2_5(MRLColQwen2_5):
    def __init__(
        self,
        base_model: ColQwen2_5,
        *,
        granularities: Sequence[int] = (1, 2, 4),
        compact_query_tokens: bool = True,
        importance_config: Optional[FolderImportanceConfig] = None,
    ) -> None:
        super().__init__(base_model=base_model, granularities=granularities, compact_query_tokens=compact_query_tokens)
        self.importance_config = importance_config or FolderImportanceConfig(enabled=False)
        self.folder_importance = ImportanceFolderCompressor(
            self.importance_config,
            image_token_id=self.config.image_token_id,
            crop_counts=[spec.crop_count for spec in self.stage_specs],
            embed_dim=self.dim,
        )

    def pop_marc_aux(self) -> Optional[Dict[str, object]]:
        return self.folder_importance.pop_marc_aux()

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
            self.folder_importance.pop_marc_aux()
        hidden_states = self._project_hidden_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values if has_images else None,
            image_grid_thw=image_grid_thw if has_images else None,
            **kwargs,
        )
        if (not self.importance_config.enabled) or len(self.importance_config.active_stage_ids()) == 0:
            return self._compact_doc_embeddings(hidden_states, input_ids, attention_mask)
        return self.folder_importance(hidden_states, input_ids, attention_mask, collect_marc_aux=not is_query)


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
    with TemporaryDirectory(prefix="folder_importance_eval_adapter_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        (tmpdir_path / "adapter_config.json").write_text((adapter_path / "adapter_config.json").read_text())
        torch.save(remapped, tmpdir_path / "adapter_model.bin")
        return PeftModel.from_pretrained(base_model, tmpdir_path)


def build_importance_model(
    model_name_or_path: str,
    *,
    granularities: Sequence[int] = (1, 2, 4),
    importance_config: Optional[FolderImportanceConfig] = None,
    attn_implementation: Optional[str] = "flash_attention_2",
    use_liger_kernel: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
    adapter_path: Optional[str] = None,
    eval_mode: bool = False,
    compact_query_tokens: bool = True,
):
    granularities = normalize_granularities(granularities)
    if len(build_stage_specs(granularities)) != 3:
        raise ValueError("FolderImportance expects exactly three stages.")
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
    model = ImportanceMRLColQwen2_5(
        base_model=base_model,
        granularities=granularities,
        compact_query_tokens=compact_query_tokens,
        importance_config=importance_config,
    )
    if eval_mode:
        model.eval()
    return model


class FolderImportanceMRLInBatchNegativeLoss(FolderHomoMRLInBatchNegativeLoss):
    pass
