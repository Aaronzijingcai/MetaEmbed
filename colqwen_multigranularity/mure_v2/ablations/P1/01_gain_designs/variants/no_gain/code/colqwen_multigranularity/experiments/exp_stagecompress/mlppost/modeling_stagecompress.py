from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from peft import PeftConfig, PeftModel, get_peft_model

from colpali_engine.models import ColQwen2_5
from colqwen_multigranularity.core import MRLColQwen2_5, _apply_compat_patch, build_stage_specs, normalize_granularities

from .compression import StageCompressConfig, StageCompressionBlock


class StageCompressor(nn.Module):
    def __init__(self, config: StageCompressConfig, *, image_token_id: int, crop_counts: Sequence[int], embed_dim: int) -> None:
        super().__init__()
        self.config = config
        self.image_token_id = int(image_token_id)
        self.crop_counts = tuple(int(v) for v in crop_counts)
        self.total_crop_count = int(sum(self.crop_counts))
        self.blocks = nn.ModuleList([
            StageCompressionBlock(
                embed_dim=embed_dim,
                budget=int(config.budgets[i]),
                method=config.method,
                tau=config.tau,
                scorer_heads=config.scorer_heads,
                scorer_dropout=config.scorer_dropout,
                use_text_context=config.use_text_context,
            )
            for i in range(3)
        ])
    def _split_stages(self, image_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total = image_tokens.shape[0]
        ends = []
        running = 0
        for count in self.crop_counts:
            running += count
            ends.append(int(torch.floor(torch.tensor(total * running / self.total_crop_count)).item()))
        ends[-1] = total
        start = 0
        chunks = []
        for end in ends:
            chunks.append(image_tokens[start:end])
            start = end
        return chunks[0], chunks[1], chunks[2]

    @staticmethod
    def _pad_sequences(sequences: Sequence[torch.Tensor], dim: int) -> torch.Tensor:
        max_len = max(seq.shape[0] for seq in sequences)
        output = sequences[0].new_zeros((len(sequences), max_len, dim))
        for i, seq in enumerate(sequences):
            output[i, : seq.shape[0]] = seq
        return output

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.LongTensor, attention_mask: torch.Tensor) -> torch.Tensor:
        sequences: List[torch.Tensor] = []
        active = set(self.config.active_stage_ids())
        lengths = []
        for row_hidden, row_ids, row_attn in zip(hidden_states, input_ids, attention_mask):
            valid = row_attn.to(dtype=torch.bool)
            image_mask = row_ids.eq(self.image_token_id) & valid
            text_mask = (~row_ids.eq(self.image_token_id)) & valid
            text_tokens = row_hidden[text_mask]
            image_tokens = row_hidden[image_mask]

            if image_tokens.numel() == 0:
                sequence = text_tokens if text_tokens.numel() > 0 else row_hidden.new_zeros((1, row_hidden.shape[-1]))
                sequences.append(sequence)
                lengths.append((sequence.shape[0], 0, 0, 0))
                continue

            stage_tokens = self._split_stages(image_tokens)
            text_context = text_tokens.mean(dim=0, keepdim=True) if self.config.use_text_context and text_tokens.numel() > 0 else None
            compressed = []
            for i, tokens in enumerate(stage_tokens):
                compressed.append(tokens if i not in active else self.blocks[i](tokens, text_context=text_context))
            c1, c2, c3 = compressed
            sequence = torch.cat([text_tokens, c1, c2, c3], dim=0)
            sequences.append(sequence)
            lengths.append((text_tokens.shape[0], c1.shape[0], c2.shape[0], c3.shape[0]))
        output = self._pad_sequences(sequences, hidden_states.shape[-1])
        # Ensure all active compressor blocks participate in every forward graph.
        # In the current contrastive trainer, query/doc/neg use multiple forwards
        # within a single step, and some branches (especially text-only query rows)
        # may otherwise skip these parameters entirely, which can confuse DDP reducer
        # bookkeeping across ranks. This zero-valued touch keeps the graph connected
        # without changing the numeric output.
        active = set(self.config.active_stage_ids())
        if active:
            zero = output.sum() * 0.0
            for idx in active:
                block = self.blocks[idx]
                for param in block.parameters():
                    zero = zero + param.sum() * 0.0
            output = output + zero
        if self.config.debug_shapes:
            print(f"[StageCompressor] lengths={lengths[:4]} output={list(output.shape)}", flush=True)
        return output


class StageCompressMRLColQwen2_5(MRLColQwen2_5):
    def __init__(self, base_model: ColQwen2_5, *, granularities: Sequence[int] = (1, 2, 4), compact_query_tokens: bool = True, compress_config: Optional[StageCompressConfig] = None) -> None:
        super().__init__(base_model=base_model, granularities=granularities, compact_query_tokens=compact_query_tokens)
        self.compress_config = compress_config or StageCompressConfig(enabled=False)
        self.stage_compressor = StageCompressor(
            self.compress_config,
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
        mode = str(self.compress_config.compress_stages).lower()
        if (not self.compress_config.enabled) or mode in {'none', 'off', 'false'}:
            return self._compact_doc_embeddings(hidden_states, input_ids, attention_mask)
        return self.stage_compressor(hidden_states, input_ids, attention_mask)


def _load_adapter_with_fallback(base_model: ColQwen2_5, adapter_path: Path):
    adapter_bin = adapter_path / 'adapter_model.bin'
    if not adapter_bin.exists():
        return PeftModel.from_pretrained(base_model, adapter_path)

    state_dict = torch.load(adapter_bin, map_location='cpu')
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith('base_model.model.base_model.custom_text_proj.'):
            key = key.replace(
                'base_model.model.base_model.custom_text_proj.',
                'base_model.model.custom_text_proj.',
                1,
            )
        if key.startswith('base_model.model.base_model.model.'):
            key = key.replace(
                'base_model.model.base_model.model.',
                'base_model.model.model.',
                1,
            )
        remapped[key] = value

    with TemporaryDirectory(prefix='stagecompress_eval_adapter_') as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / 'adapter_config.json').write_text((adapter_path / 'adapter_config.json').read_text())
        torch.save(remapped, tmpdir / 'adapter_model.bin')
        return PeftModel.from_pretrained(base_model, tmpdir)


def build_stagecompress_model(model_name_or_path: str, *, granularities: Sequence[int] = (1, 2, 4), compress_config: Optional[StageCompressConfig] = None, attn_implementation: Optional[str] = 'flash_attention_2', use_liger_kernel: bool = False, torch_dtype: torch.dtype = torch.bfloat16, adapter_path: Optional[str] = None, eval_mode: bool = False, compact_query_tokens: bool = True):
    granularities = normalize_granularities(granularities)
    if len(build_stage_specs(granularities)) != 3:
        raise ValueError('StageCompress expects exactly three stages.')
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
    model = StageCompressMRLColQwen2_5(
        base_model=base_model,
        granularities=granularities,
        compact_query_tokens=compact_query_tokens,
        compress_config=compress_config,
    )
    if eval_mode:
        model.eval()
    return model
