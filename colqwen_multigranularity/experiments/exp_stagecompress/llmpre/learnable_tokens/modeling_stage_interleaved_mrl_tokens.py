from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Sequence

import torch
import torch.nn as nn
from peft import PeftModel

from colpali_engine.models import ColQwen2_5
from colqwen_multigranularity.core import (
    MRLColQwen2_5,
    _apply_compat_patch,
    build_stage_specs,
    normalize_granularities,
)


DEFAULT_STAGE_MRL_TOKEN_COUNTS = (32, 64, 128)
STAGE_INTERLEAVED_MRL_TOKEN_FILE = "stage_interleaved_mrl_tokens.pt"


def _normalize_stage_token_counts(counts: Sequence[int], *, name: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in counts)
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three stage token counts, got {values}.")
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} must be positive, got {values}.")
    return values


class StageInterleavedMRLTokenColQwen2_5(MRLColQwen2_5):  # noqa: N801
    """Stage-interleaved learnable-token wrapper for multi-granularity ColQwen2.5.

    For image-side inputs, learnable tokens are inserted after each visual stage:

        g1 image tokens -> g1 meta tokens -> g2 image tokens -> g2 meta tokens
        -> g3 image tokens -> g3 meta tokens -> text prompt

    For text-only inputs, the same stage-token groups are appended at the end of
    the sequence. The returned representation is only the contextualized
    learnable tokens, ordered as g1_meta + g2_meta + g3_meta, so a regular
    MetaEmbed-style MRL-token loss can slice prefixes such as 32, 96, and 224.
    """

    def __init__(
        self,
        base_model: ColQwen2_5,
        *,
        granularities: Sequence[int] = (1, 2, 4),
        query_stage_mrl_tokens: Sequence[int] = DEFAULT_STAGE_MRL_TOKEN_COUNTS,
        doc_stage_mrl_tokens: Sequence[int] = DEFAULT_STAGE_MRL_TOKEN_COUNTS,
        shared_query_doc_stage_tokens: bool = False,
        compact_query_tokens: bool = True,
    ) -> None:
        super().__init__(
            base_model=base_model,
            granularities=granularities,
            compact_query_tokens=compact_query_tokens,
        )
        if len(self.stage_specs) != 3:
            raise ValueError("Stage-interleaved MRL-token experiment expects exactly g1/g2/g3 input stages.")

        self.query_stage_mrl_tokens = _normalize_stage_token_counts(
            query_stage_mrl_tokens,
            name="query_stage_mrl_tokens",
        )
        self.doc_stage_mrl_tokens = _normalize_stage_token_counts(
            doc_stage_mrl_tokens,
            name="doc_stage_mrl_tokens",
        )
        self.shared_query_doc_stage_tokens = bool(shared_query_doc_stage_tokens)
        if self.shared_query_doc_stage_tokens and self.query_stage_mrl_tokens != self.doc_stage_mrl_tokens:
            raise ValueError(
                "shared_query_doc_stage_tokens=True requires query/doc stage token counts to match, "
                f"got query={self.query_stage_mrl_tokens} doc={self.doc_stage_mrl_tokens}."
            )

        self.num_query_mrl_tokens = int(sum(self.query_stage_mrl_tokens))
        self.num_doc_mrl_tokens = int(sum(self.doc_stage_mrl_tokens))
        self.num_added_tokens = (
            self.num_query_mrl_tokens
            if self.shared_query_doc_stage_tokens
            else self.num_query_mrl_tokens + self.num_doc_mrl_tokens
        )

        hidden_size = int(self.base_model.model.config.hidden_size)
        self.prompt_embed_tokens = nn.Embedding(self.num_added_tokens, hidden_size)
        nn.init.normal_(self.prompt_embed_tokens.weight, mean=0.0, std=hidden_size ** -0.5)
        self.padding_side = "left"
        self._stage_interleaved_debug_count = 0

    @staticmethod
    def _has_images(pixel_values: Optional[torch.Tensor], image_grid_thw: Optional[torch.Tensor]) -> bool:
        return (
            pixel_values is not None
            and image_grid_thw is not None
            and getattr(pixel_values, "numel", lambda: 0)() > 0
            and getattr(image_grid_thw, "numel", lambda: 0)() > 0
        )

    @staticmethod
    def _debug_shape(tensor: Optional[torch.Tensor]) -> Optional[list[int]]:
        return None if tensor is None else list(tensor.shape)

    @staticmethod
    def _debug_short_list(tensor: Optional[torch.Tensor], limit: int = 12) -> Optional[list]:
        if tensor is None:
            return None
        values = tensor.detach().to("cpu").reshape(-1).tolist()
        return values[:limit]

    def _debug_enabled(self) -> bool:
        return os.environ.get("STAGE_INTERLEAVED_MRL_DEBUG", "").lower() in {"1", "true", "yes", "on"}

    def _debug_limit(self) -> int:
        try:
            return int(os.environ.get("STAGE_INTERLEAVED_MRL_DEBUG_LIMIT", "8"))
        except ValueError:
            return 8

    def _should_debug(self) -> bool:
        if not self._debug_enabled():
            return False
        self._stage_interleaved_debug_count += 1
        return self._stage_interleaved_debug_count <= self._debug_limit()

    def _debug_print(self, message: str) -> None:
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
        print(f"[StageInterleavedMRL][rank={rank}] {message}", flush=True)

    def _dummy_token_id(self) -> int:
        eos_token_id = getattr(self.config, "eos_token_id", None)
        if isinstance(eos_token_id, (list, tuple)) and eos_token_id:
            eos_token_id = eos_token_id[0]
        if eos_token_id is not None:
            return int(eos_token_id)
        pad_token_id = getattr(self.config, "pad_token_id", None)
        if pad_token_id is not None:
            return int(pad_token_id)
        return 1

    def _role_stage_counts(self, *, is_query: bool) -> tuple[int, ...]:
        return self.query_stage_mrl_tokens if is_query else self.doc_stage_mrl_tokens

    def _role_prompt_offset(self, *, is_query: bool) -> int:
        if self.shared_query_doc_stage_tokens or is_query:
            return 0
        return self.num_query_mrl_tokens

    def _stage_image_ends(self, total_image_tokens: int) -> list[int]:
        ends: list[int] = []
        cumulative_crops = 0
        for spec in self.stage_specs:
            cumulative_crops += int(spec.crop_count)
            end = int(float(total_image_tokens) * float(cumulative_crops) / float(self.total_crops))
            ends.append(min(max(end, 0), int(total_image_tokens)))
        if ends:
            ends[-1] = int(total_image_tokens)
        return ends

    def _advance_after_vision_end(self, row_ids: torch.Tensor, position: int) -> int:
        vision_end_token_id = getattr(self.config, "vision_end_token_id", None)
        if vision_end_token_id is None:
            return position
        if position < row_ids.numel() and int(row_ids[position].item()) == int(vision_end_token_id):
            return position + 1
        return position

    def _stage_insert_positions(self, row_ids: torch.Tensor, row_attention: torch.Tensor) -> list[int]:
        image_token_id = int(self.config.image_token_id)
        active = row_attention.to(dtype=torch.bool)
        image_positions = (row_ids.eq(image_token_id) & active).nonzero(as_tuple=False).squeeze(-1)
        if image_positions.numel() == 0:
            return [int(row_ids.shape[0])] * len(self.stage_specs)

        positions: list[int] = []
        previous_end = 0
        for end in self._stage_image_ends(int(image_positions.numel())):
            if end <= previous_end:
                insert_pos = int(image_positions[previous_end - 1].item()) + 1 if previous_end > 0 else int(image_positions[0].item())
            else:
                insert_pos = int(image_positions[end - 1].item()) + 1
            insert_pos = self._advance_after_vision_end(row_ids, insert_pos)
            positions.append(insert_pos)
            previous_end = max(previous_end, end)
        return positions

    def _insert_stage_mrl_token_ids(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        is_query: bool,
    ) -> tuple[torch.LongTensor, torch.Tensor, torch.LongTensor, torch.LongTensor]:
        batch_size, seq_len = input_ids.shape
        stage_counts = self._role_stage_counts(is_query=is_query)
        num_role_tokens = int(sum(stage_counts))
        extended_len = seq_len + num_role_tokens
        prompt_token_id = self._dummy_token_id()
        role_offset = self._role_prompt_offset(is_query=is_query)

        extended_input_ids = input_ids.new_full((batch_size, extended_len), prompt_token_id)
        extended_attention_mask = attention_mask.new_zeros((batch_size, extended_len))
        prompt_token_indices = input_ids.new_full((batch_size, extended_len), -1)
        selected_positions = input_ids.new_zeros((batch_size, num_role_tokens))

        for batch_index in range(batch_size):
            row_ids = input_ids[batch_index]
            row_attention = attention_mask[batch_index]
            insert_positions = self._stage_insert_positions(row_ids, row_attention)
            src = 0
            dst = 0
            local_prompt_offset = 0
            selected: list[int] = []

            for insert_pos, count in zip(insert_positions, stage_counts):
                insert_pos = max(src, min(int(insert_pos), seq_len))
                segment_len = insert_pos - src
                if segment_len > 0:
                    extended_input_ids[batch_index, dst : dst + segment_len] = row_ids[src:insert_pos]
                    extended_attention_mask[batch_index, dst : dst + segment_len] = row_attention[src:insert_pos]
                    dst += segment_len
                    src = insert_pos

                prompt_slice = torch.arange(
                    local_prompt_offset,
                    local_prompt_offset + int(count),
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                ) + int(role_offset)
                extended_input_ids[batch_index, dst : dst + int(count)] = prompt_token_id
                extended_attention_mask[batch_index, dst : dst + int(count)] = 1
                prompt_token_indices[batch_index, dst : dst + int(count)] = prompt_slice
                selected.extend(range(dst, dst + int(count)))
                dst += int(count)
                local_prompt_offset += int(count)

            tail_len = seq_len - src
            if tail_len > 0:
                extended_input_ids[batch_index, dst : dst + tail_len] = row_ids[src:seq_len]
                extended_attention_mask[batch_index, dst : dst + tail_len] = row_attention[src:seq_len]
                dst += tail_len

            if dst != extended_len:
                raise RuntimeError(
                    "Stage-interleaved sequence build length mismatch: "
                    f"row={batch_index} dst={dst} expected={extended_len}."
                )
            if len(selected) != num_role_tokens:
                raise RuntimeError(
                    "Stage-interleaved selection length mismatch: "
                    f"row={batch_index} selected={len(selected)} expected={num_role_tokens}."
                )
            selected_positions[batch_index] = torch.tensor(selected, device=input_ids.device, dtype=input_ids.dtype)

        return extended_input_ids, extended_attention_mask, prompt_token_indices, selected_positions

    def _build_inputs_embeds(
        self,
        *,
        input_ids: torch.LongTensor,
        prompt_token_indices: torch.LongTensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.LongTensor],
    ) -> torch.Tensor:
        inputs_embeds = self.base_model._embed_tokens(input_ids)
        prompt_mask = prompt_token_indices.ge(0)
        if prompt_mask.any():
            safe_indices = prompt_token_indices.clamp_min(0)
            prompt_embeds = self.prompt_embed_tokens(safe_indices)
            prompt_embeds = prompt_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            inputs_embeds = torch.where(prompt_mask.unsqueeze(-1), prompt_embeds, inputs_embeds)

        if pixel_values is not None:
            pixel_values = pixel_values.type(self.base_model.visual.dtype)
            image_embeds = self.base_model.visual(pixel_values, grid_thw=image_grid_thw)
            image_mask = input_ids.eq(int(self.config.image_token_id)).unsqueeze(-1).expand_as(inputs_embeds)
            image_embeds = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            expected = int(image_mask.sum().item() // inputs_embeds.shape[-1])
            actual = int(image_embeds.shape[0])
            if expected != actual:
                raise RuntimeError(
                    "Stage-interleaved MRL-token image embed mismatch: "
                    f"input has {expected} image placeholders but visual encoder returned {actual} embeds."
                )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        return inputs_embeds

    @staticmethod
    def _gather_positions(hidden_states: torch.Tensor, positions: torch.LongTensor) -> torch.Tensor:
        gather_index = positions.to(device=hidden_states.device, dtype=torch.long).unsqueeze(-1)
        gather_index = gather_index.expand(-1, -1, hidden_states.shape[-1])
        return hidden_states.gather(dim=1, index=gather_index)

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
        kwargs.pop("has_images", None)
        kwargs.pop("inputs_embeds", None)
        kwargs.pop("position_ids", None)
        is_query = bool(kwargs.pop("is_query", False))
        return_image_embeds = bool(kwargs.pop("return_image_embeds", False))
        debug_active = self._should_debug()
        debug_index = self._stage_interleaved_debug_count

        has_images = self._has_images(pixel_values, image_grid_thw)
        active_pixel_values = pixel_values if has_images else None
        active_image_grid_thw = image_grid_thw if has_images else None

        extended_input_ids, extended_attention_mask, prompt_token_indices, selected_positions = self._insert_stage_mrl_token_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
            is_query=is_query,
        )
        inputs_embeds = self._build_inputs_embeds(
            input_ids=extended_input_ids,
            prompt_token_indices=prompt_token_indices,
            pixel_values=active_pixel_values,
            image_grid_thw=active_image_grid_thw,
        )
        position_ids, _ = self.base_model.get_rope_index(
            input_ids=extended_input_ids,
            image_grid_thw=active_image_grid_thw,
            video_grid_thw=None,
            attention_mask=extended_attention_mask,
        )
        hidden_states = self.base_model.inner_forward(
            input_ids=extended_input_ids,
            attention_mask=extended_attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=None,
            image_grid_thw=None,
            use_cache=False,
            output_hidden_states=True,
            **kwargs,
        )

        if return_image_embeds:
            selected = hidden_states
            output_mask = extended_attention_mask
        else:
            selected = self._gather_positions(hidden_states, selected_positions)
            expected_width = self.num_query_mrl_tokens if is_query else self.num_doc_mrl_tokens
            if selected.shape[1] != expected_width:
                raise RuntimeError(
                    "Stage-interleaved MRL-token selected width mismatch: "
                    f"is_query={is_query} selected={selected.shape[1]} expected={expected_width}."
                )
            output_mask = selected.new_ones(selected.shape[:2], dtype=extended_attention_mask.dtype)

        proj = self.base_model.custom_text_proj(selected)
        proj = proj / proj.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        proj = proj * output_mask.to(device=proj.device, dtype=proj.dtype).unsqueeze(-1)

        if debug_active:
            image_token_id = int(getattr(self.config, "image_token_id", -1))
            image_counts = input_ids.eq(image_token_id).sum(dim=1) if image_token_id >= 0 else None
            selected_preview = selected_positions[:, : min(12, selected_positions.shape[1])]
            norms = proj.norm(dim=-1)
            self._debug_print(
                f"forward#{debug_index} is_query={is_query} input={self._debug_shape(input_ids)} "
                f"extended={self._debug_shape(extended_input_ids)} image_counts={self._debug_short_list(image_counts)} "
                f"stage_counts={self._role_stage_counts(is_query=is_query)} selected_pos={self._debug_short_list(selected_preview)} "
                f"output={self._debug_shape(proj)} finite={bool(torch.isfinite(proj).all().item())} "
                f"norm_min={float(norms.min().item()):.6f} norm_max={float(norms.max().item()):.6f}"
            )
        return proj

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self._project_hidden_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs,
        )

    def _active_prompt_embed_module(self):
        module = self.prompt_embed_tokens
        modules_to_save = getattr(module, "modules_to_save", None)
        if modules_to_save is None:
            return module
        active_adapter = getattr(module, "active_adapter", None)
        if isinstance(active_adapter, (list, tuple)) and active_adapter:
            active_adapter = active_adapter[0]
        if active_adapter in modules_to_save:
            return modules_to_save[active_adapter]
        try:
            return next(iter(modules_to_save.values()))
        except StopIteration:
            return module

    def stage_interleaved_mrl_state_dict(self) -> dict:
        active_embedding = self._active_prompt_embed_module()
        return {
            "config": {
                "query_stage_mrl_tokens": list(self.query_stage_mrl_tokens),
                "doc_stage_mrl_tokens": list(self.doc_stage_mrl_tokens),
                "shared_query_doc_stage_tokens": self.shared_query_doc_stage_tokens,
                "num_query_mrl_tokens": self.num_query_mrl_tokens,
                "num_doc_mrl_tokens": self.num_doc_mrl_tokens,
                "num_added_tokens": self.num_added_tokens,
                "granularities": list(self.granularities),
            },
            "state_dict": {"weight": active_embedding.weight.detach().cpu()},
        }

    def save_stage_interleaved_mrl_token_state(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.stage_interleaved_mrl_state_dict(), save_dir / STAGE_INTERLEAVED_MRL_TOKEN_FILE)

    def load_stage_interleaved_mrl_token_state(self, path: str | Path, *, map_location: str | torch.device = "cpu") -> None:
        path = Path(path)
        if path.is_dir():
            path = path / STAGE_INTERLEAVED_MRL_TOKEN_FILE
        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location=map_location)
        saved_config = payload.get("config") if isinstance(payload, dict) else None
        if isinstance(saved_config, dict):
            expected_config = {
                "query_stage_mrl_tokens": list(self.query_stage_mrl_tokens),
                "doc_stage_mrl_tokens": list(self.doc_stage_mrl_tokens),
                "shared_query_doc_stage_tokens": self.shared_query_doc_stage_tokens,
                "num_query_mrl_tokens": self.num_query_mrl_tokens,
                "num_doc_mrl_tokens": self.num_doc_mrl_tokens,
                "num_added_tokens": self.num_added_tokens,
                "granularities": list(self.granularities),
            }
            mismatches = []
            for key, expected_value in expected_config.items():
                if key in saved_config and saved_config[key] != expected_value:
                    mismatches.append(f"{key}: checkpoint={saved_config[key]!r} model={expected_value!r}")
            if mismatches:
                raise ValueError("Stage-interleaved MRL token config mismatch: " + "; ".join(mismatches))
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        if "weight" in state_dict:
            weight = state_dict["weight"]
        elif "prompt_embed_tokens.weight" in state_dict:
            weight = state_dict["prompt_embed_tokens.weight"]
        else:
            raise KeyError(f"Could not find prompt embedding weight in {path}.")
        active_embedding = self._active_prompt_embed_module()
        if tuple(active_embedding.weight.shape) != tuple(weight.shape):
            raise ValueError(
                "Stage-interleaved MRL token weight shape mismatch: "
                f"model={tuple(active_embedding.weight.shape)} checkpoint={tuple(weight.shape)}."
            )
        active_embedding.weight.data.copy_(weight.to(device=active_embedding.weight.device, dtype=active_embedding.weight.dtype))

    def save_pretrained(self, save_dir: str, **kwargs):
        self.base_model.save_pretrained(save_dir, **kwargs)
        self.save_stage_interleaved_mrl_token_state(save_dir)


def _find_stage_interleaved_mrl_model(model) -> Optional[StageInterleavedMRLTokenColQwen2_5]:
    if isinstance(model, StageInterleavedMRLTokenColQwen2_5):
        return model
    for module in model.modules():
        if isinstance(module, StageInterleavedMRLTokenColQwen2_5):
            return module
    return None


def save_stage_interleaved_mrl_token_state(model, save_dir: str | Path) -> None:
    inner = _find_stage_interleaved_mrl_model(model)
    if inner is None:
        raise TypeError("Could not find StageInterleavedMRLTokenColQwen2_5 inside model.")
    inner.save_stage_interleaved_mrl_token_state(save_dir)


def load_stage_interleaved_mrl_token_state(model, path: str | Path, *, map_location: str | torch.device = "cpu") -> None:
    inner = _find_stage_interleaved_mrl_model(model)
    if inner is None:
        raise TypeError("Could not find StageInterleavedMRLTokenColQwen2_5 inside model.")
    inner.load_stage_interleaved_mrl_token_state(path, map_location=map_location)


def _load_adapter_with_fallback(model: StageInterleavedMRLTokenColQwen2_5, adapter_path: Path):
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

    with TemporaryDirectory(prefix="stage_interleaved_mrl_eval_adapter_") as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "adapter_config.json").write_text((adapter_path / "adapter_config.json").read_text())
        torch.save(remapped, tmpdir / "adapter_model.bin")
        return PeftModel.from_pretrained(model, tmpdir)


def build_stage_interleaved_mrl_token_model(
    model_name_or_path: str,
    *,
    granularities: Sequence[int] = (1, 2, 4),
    query_stage_mrl_tokens: Sequence[int] = DEFAULT_STAGE_MRL_TOKEN_COUNTS,
    doc_stage_mrl_tokens: Sequence[int] = DEFAULT_STAGE_MRL_TOKEN_COUNTS,
    shared_query_doc_stage_tokens: bool = False,
    attn_implementation: Optional[str] = "flash_attention_2",
    use_liger_kernel: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
    adapter_path: Optional[str] = None,
    stage_interleaved_mrl_token_path: Optional[str] = None,
    eval_mode: bool = False,
    compact_query_tokens: bool = True,
):
    granularities = normalize_granularities(granularities)
    if len(build_stage_specs(granularities)) != 3:
        raise ValueError("Stage-interleaved MRL-token experiment expects exactly three stages.")

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

    model = StageInterleavedMRLTokenColQwen2_5(
        base_model=base_model,
        granularities=granularities,
        query_stage_mrl_tokens=query_stage_mrl_tokens,
        doc_stage_mrl_tokens=doc_stage_mrl_tokens,
        shared_query_doc_stage_tokens=shared_query_doc_stage_tokens,
        compact_query_tokens=compact_query_tokens,
    )

    if adapter_path is not None:
        model = _load_adapter_with_fallback(model, Path(adapter_path))

    state_path = stage_interleaved_mrl_token_path
    if state_path is None and adapter_path is not None:
        candidate = Path(adapter_path) / STAGE_INTERLEAVED_MRL_TOKEN_FILE
        if candidate.exists():
            state_path = str(candidate)
    if state_path is not None and Path(state_path).exists():
        load_stage_interleaved_mrl_token_state(model, state_path, map_location="cpu")

    if eval_mode:
        model.eval()
    return model
