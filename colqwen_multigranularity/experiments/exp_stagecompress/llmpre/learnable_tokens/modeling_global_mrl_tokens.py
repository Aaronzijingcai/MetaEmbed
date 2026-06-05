from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Sequence

import torch
import torch.nn as nn
from peft import PeftModel

from colpali_engine.models import ColQwen2_5
from colqwen_multigranularity.core import MRLColQwen2_5, _apply_compat_patch, build_stage_specs, normalize_granularities


class LastGlobalMRLTokenColQwen2_5(MRLColQwen2_5):  # noqa: N801
    """MetaEmbed-style global MRL-token wrapper for multi-granularity ColQwen2.5."""

    def __init__(
        self,
        base_model: ColQwen2_5,
        *,
        granularities: Sequence[int] = (1, 2, 4),
        num_query_mrl_tokens: int = 16,
        num_doc_mrl_tokens: int = 64,
        shared_query_doc_mrl_tokens: bool = False,
        compact_query_tokens: bool = True,
    ) -> None:
        super().__init__(
            base_model=base_model,
            granularities=granularities,
            compact_query_tokens=compact_query_tokens,
        )
        if len(self.stage_specs) != 3:
            raise ValueError("Global MRL-token experiment expects exactly g1/g2/g3 input stages.")
        self.num_query_mrl_tokens = int(num_query_mrl_tokens)
        self.num_doc_mrl_tokens = int(num_doc_mrl_tokens)
        if self.num_query_mrl_tokens <= 0 or self.num_doc_mrl_tokens <= 0:
            raise ValueError("num_query_mrl_tokens and num_doc_mrl_tokens must be positive.")
        self.shared_query_doc_mrl_tokens = bool(shared_query_doc_mrl_tokens)
        self.num_added_tokens = (
            max(self.num_query_mrl_tokens, self.num_doc_mrl_tokens)
            if self.shared_query_doc_mrl_tokens
            else self.num_query_mrl_tokens + self.num_doc_mrl_tokens
        )

        hidden_size = int(self.base_model.model.config.hidden_size)
        self.prompt_embed_tokens = nn.Embedding(self.num_added_tokens, hidden_size)
        nn.init.normal_(self.prompt_embed_tokens.weight, mean=0.0, std=hidden_size ** -0.5)
        self.padding_side = "left"
        self._global_mrl_debug_count = 0

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
        return os.environ.get("GLOBAL_MRL_DEBUG", "").lower() in {"1", "true", "yes", "on"}

    def _debug_limit(self) -> int:
        try:
            return int(os.environ.get("GLOBAL_MRL_DEBUG_LIMIT", "8"))
        except ValueError:
            return 8

    def _should_debug(self) -> bool:
        if not self._debug_enabled():
            return False
        self._global_mrl_debug_count += 1
        return self._global_mrl_debug_count <= self._debug_limit()

    def _debug_print(self, message: str) -> None:
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
        print(f"[GlobalMRL][rank={rank}] {message}", flush=True)

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

    def _append_mrl_token_ids(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.LongTensor, torch.Tensor]:
        # Keep the original padded token layout intact. Qwen2.5-VL uses image
        # placeholder ids outside the text attention region; compacting by
        # attention_mask would drop those placeholders while pixel_values still
        # asks the visual encoder to return their embeddings.
        prompt_token_id = self._dummy_token_id()
        prompt_ids = input_ids.new_full((input_ids.shape[0], self.num_added_tokens), prompt_token_id)
        prompt_attention = attention_mask.new_ones((attention_mask.shape[0], self.num_added_tokens))
        extended_input_ids = torch.cat([input_ids, prompt_ids], dim=1)
        extended_attention_mask = torch.cat([attention_mask, prompt_attention], dim=1)
        return extended_input_ids, extended_attention_mask

    def _build_inputs_embeds(
        self,
        *,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.LongTensor],
    ) -> torch.Tensor:
        inputs_embeds = self.base_model._embed_tokens(input_ids)
        prompt_indices = torch.arange(self.num_added_tokens, device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
        prompt_embeds = self.prompt_embed_tokens(prompt_indices)
        prompt_embeds = prompt_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[:, -self.num_added_tokens :, :] = prompt_embeds

        if pixel_values is not None:
            pixel_values = pixel_values.type(self.base_model.visual.dtype)
            image_embeds = self.base_model.visual(pixel_values, grid_thw=image_grid_thw)
            image_mask = input_ids.eq(int(self.config.image_token_id)).unsqueeze(-1).expand_as(inputs_embeds)
            image_embeds = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            expected = int(image_mask.sum().item() // inputs_embeds.shape[-1])
            actual = int(image_embeds.shape[0])
            if expected != actual:
                raise RuntimeError(
                    "Global MRL-token image embed mismatch: "
                    f"input has {expected} image placeholders but visual encoder returned {actual} embeds."
                )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        return inputs_embeds

    def _select_meta_hidden(self, hidden_states: torch.Tensor, *, is_query: bool) -> torch.Tensor:
        meta_hidden = hidden_states[:, -self.num_added_tokens :, :]
        if self.shared_query_doc_mrl_tokens:
            width = self.num_query_mrl_tokens if is_query else self.num_doc_mrl_tokens
            return meta_hidden[:, :width, :]
        if is_query:
            return meta_hidden[:, : self.num_query_mrl_tokens, :]
        return meta_hidden[:, self.num_query_mrl_tokens :, :]

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
        debug_index = self._global_mrl_debug_count

        if debug_active:
            image_token_id = int(getattr(self.config, "image_token_id", -1))
            image_counts = input_ids.eq(image_token_id).sum(dim=1) if image_token_id >= 0 else None
            grid_tokens = image_grid_thw.prod(dim=1) if image_grid_thw is not None and image_grid_thw.ndim == 2 else None
            attn_lengths = attention_mask.sum(dim=1) if attention_mask is not None and attention_mask.ndim == 2 else None
            self._debug_print(
                f"forward#{debug_index} is_query={is_query} return_image_embeds={return_image_embeds} "
                f"input_ids={self._debug_shape(input_ids)} attention_mask={self._debug_shape(attention_mask)} "
                f"attn_lengths={self._debug_short_list(attn_lengths)} image_counts={self._debug_short_list(image_counts)} "
                f"pixel_values={self._debug_shape(pixel_values)} image_grid_thw={self._debug_shape(image_grid_thw)} "
                f"grid_tokens={self._debug_short_list(grid_tokens)} num_added={self.num_added_tokens} "
                f"query_tokens={self.num_query_mrl_tokens} doc_tokens={self.num_doc_mrl_tokens} "
                f"shared={self.shared_query_doc_mrl_tokens}"
            )

        has_images = self._has_images(pixel_values, image_grid_thw)
        active_pixel_values = pixel_values if has_images else None
        active_image_grid_thw = image_grid_thw if has_images else None
        extended_input_ids, extended_attention_mask = self._append_mrl_token_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        if debug_active:
            ext_lengths = extended_attention_mask.sum(dim=1)
            image_token_id = int(getattr(self.config, "image_token_id", -1))
            ext_image_counts = extended_input_ids.eq(image_token_id).sum(dim=1) if image_token_id >= 0 else None
            prompt_attn = extended_attention_mask[:, -self.num_added_tokens :].sum(dim=1)
            self._debug_print(
                f"forward#{debug_index} extended_input_ids={self._debug_shape(extended_input_ids)} "
                f"extended_lengths={self._debug_short_list(ext_lengths)} "
                f"extended_image_counts={self._debug_short_list(ext_image_counts)} "
                f"prompt_attention={self._debug_short_list(prompt_attn)}"
            )
        inputs_embeds = self._build_inputs_embeds(
            input_ids=extended_input_ids,
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
            selected = self._select_meta_hidden(hidden_states, is_query=is_query)
            expected_width = self.num_query_mrl_tokens if is_query else self.num_doc_mrl_tokens
            if selected.shape[1] != expected_width:
                raise RuntimeError(
                    "Global MRL-token selected width mismatch: "
                    f"is_query={is_query} selected={selected.shape[1]} expected={expected_width}."
                )
            output_mask = selected.new_ones(selected.shape[:2], dtype=extended_attention_mask.dtype)

        proj = self.base_model.custom_text_proj(selected)
        proj = proj / proj.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        proj = proj * output_mask.to(device=proj.device, dtype=proj.dtype).unsqueeze(-1)
        if debug_active:
            finite = bool(torch.isfinite(proj).all().item())
            norms = proj.norm(dim=-1)
            self._debug_print(
                f"forward#{debug_index} hidden_states={self._debug_shape(hidden_states)} "
                f"selected={self._debug_shape(selected)} output={self._debug_shape(proj)} "
                f"output_mask_sum={self._debug_short_list(output_mask.sum(dim=1))} "
                f"finite={finite} norm_min={float(norms.min().item()):.6f} norm_max={float(norms.max().item()):.6f}"
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

    def global_mrl_state_dict(self) -> dict:
        active_embedding = self._active_prompt_embed_module()
        return {
            "config": {
                "num_query_mrl_tokens": self.num_query_mrl_tokens,
                "num_doc_mrl_tokens": self.num_doc_mrl_tokens,
                "shared_query_doc_mrl_tokens": self.shared_query_doc_mrl_tokens,
                "num_added_tokens": self.num_added_tokens,
                "granularities": list(self.granularities),
            },
            "state_dict": {"weight": active_embedding.weight.detach().cpu()},
        }

    def save_global_mrl_token_state(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.global_mrl_state_dict(), save_dir / "global_mrl_tokens.pt")

    def load_global_mrl_token_state(self, path: str | Path, *, map_location: str | torch.device = "cpu") -> None:
        path = Path(path)
        if path.is_dir():
            path = path / "global_mrl_tokens.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location=map_location)
        saved_config = payload.get("config") if isinstance(payload, dict) else None
        if isinstance(saved_config, dict):
            expected_config = {
                "num_query_mrl_tokens": self.num_query_mrl_tokens,
                "num_doc_mrl_tokens": self.num_doc_mrl_tokens,
                "shared_query_doc_mrl_tokens": self.shared_query_doc_mrl_tokens,
                "num_added_tokens": self.num_added_tokens,
                "granularities": list(self.granularities),
            }
            mismatches = []
            for key, expected_value in expected_config.items():
                if key in saved_config and saved_config[key] != expected_value:
                    mismatches.append(f"{key}: checkpoint={saved_config[key]!r} model={expected_value!r}")
            if mismatches:
                raise ValueError("Global MRL token config mismatch: " + "; ".join(mismatches))
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
                "Global MRL token weight shape mismatch: "
                f"model={tuple(active_embedding.weight.shape)} checkpoint={tuple(weight.shape)}."
            )
        active_embedding.weight.data.copy_(weight.to(device=active_embedding.weight.device, dtype=active_embedding.weight.dtype))

    def save_pretrained(self, save_dir: str, **kwargs):
        self.base_model.save_pretrained(save_dir, **kwargs)
        self.save_global_mrl_token_state(save_dir)


def _find_global_mrl_model(model) -> Optional[LastGlobalMRLTokenColQwen2_5]:
    if isinstance(model, LastGlobalMRLTokenColQwen2_5):
        return model
    for module in model.modules():
        if isinstance(module, LastGlobalMRLTokenColQwen2_5):
            return module
    return None


def save_global_mrl_token_state(model, save_dir: str | Path) -> None:
    inner = _find_global_mrl_model(model)
    if inner is None:
        raise TypeError("Could not find LastGlobalMRLTokenColQwen2_5 inside model.")
    inner.save_global_mrl_token_state(save_dir)


def load_global_mrl_token_state(model, path: str | Path, *, map_location: str | torch.device = "cpu") -> None:
    inner = _find_global_mrl_model(model)
    if inner is None:
        raise TypeError("Could not find LastGlobalMRLTokenColQwen2_5 inside model.")
    inner.load_global_mrl_token_state(path, map_location=map_location)


def _load_adapter_with_fallback(model: LastGlobalMRLTokenColQwen2_5, adapter_path: Path):
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
            key = key.replace("base_model.model.base_model.custom_text_proj.", "base_model.model.base_model.base_model.custom_text_proj.", 1)
        if key.startswith("base_model.model.base_model.model."):
            key = key.replace("base_model.model.base_model.model.", "base_model.model.base_model.base_model.model.", 1)
        remapped[key] = value

    with TemporaryDirectory(prefix="global_mrl_eval_adapter_") as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "adapter_config.json").write_text((adapter_path / "adapter_config.json").read_text())
        torch.save(remapped, tmpdir / "adapter_model.bin")
        return PeftModel.from_pretrained(model, tmpdir)


def build_global_mrl_token_model(
    model_name_or_path: str,
    *,
    granularities: Sequence[int] = (1, 2, 4),
    num_query_mrl_tokens: int = 16,
    num_doc_mrl_tokens: int = 64,
    shared_query_doc_mrl_tokens: bool = False,
    attn_implementation: Optional[str] = "flash_attention_2",
    use_liger_kernel: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
    adapter_path: Optional[str] = None,
    global_mrl_token_path: Optional[str] = None,
    eval_mode: bool = False,
    compact_query_tokens: bool = True,
):
    granularities = normalize_granularities(granularities)
    if len(build_stage_specs(granularities)) != 3:
        raise ValueError("Global MRL-token experiment expects exactly three stages.")

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

    model = LastGlobalMRLTokenColQwen2_5(
        base_model=base_model,
        granularities=granularities,
        num_query_mrl_tokens=num_query_mrl_tokens,
        num_doc_mrl_tokens=num_doc_mrl_tokens,
        shared_query_doc_mrl_tokens=shared_query_doc_mrl_tokens,
        compact_query_tokens=compact_query_tokens,
    )

    if adapter_path is not None:
        model = _load_adapter_with_fallback(model, Path(adapter_path))

    state_path = global_mrl_token_path
    if state_path is None and adapter_path is not None:
        candidate = Path(adapter_path) / "global_mrl_tokens.pt"
        if candidate.exists():
            state_path = str(candidate)
    if state_path is not None and Path(state_path).exists():
        load_global_mrl_token_state(model, state_path, map_location="cpu")

    if eval_mode:
        model.eval()
    return model
