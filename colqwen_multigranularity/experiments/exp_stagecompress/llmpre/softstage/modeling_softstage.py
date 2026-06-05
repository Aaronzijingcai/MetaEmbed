from __future__ import annotations

import math
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


class StageAwareSoftSelector(nn.Module):
    """Stage-aware soft visual-token mask for g1/g2/g3 crops."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_stages: int,
        keep_ratios: Optional[Sequence[float]] = None,
        temperature: float = 0.1,
        min_mask_value: float = 0.0,
    ) -> None:
        super().__init__()
        if keep_ratios is None:
            keep_ratios = (1.0, 0.5, 0.25)
        if len(keep_ratios) != num_stages:
            raise ValueError(f"Expected {num_stages} keep ratios, got {len(keep_ratios)}.")
        ratios = [float(value) for value in keep_ratios]
        for ratio in ratios:
            if ratio <= 0 or ratio > 1:
                raise ValueError(f"softstage keep ratio must be in (0, 1], got {ratio}.")
        if temperature <= 0:
            raise ValueError("softstage temperature must be positive.")
        if min_mask_value < 0 or min_mask_value >= 1:
            raise ValueError("softstage min_mask_value must be in [0, 1).")

        self.num_stages = int(num_stages)
        self.temperature = float(temperature)
        self.min_mask_value = float(min_mask_value)
        self.register_buffer("keep_ratios", torch.tensor(ratios, dtype=torch.float32), persistent=True)
        self.stage_embeddings = nn.Embedding(self.num_stages, hidden_size)
        self.score_norm = nn.LayerNorm(hidden_size)
        self.score_head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.stage_embeddings.weight, mean=0.0, std=hidden_size ** -0.5)
        nn.init.zeros_(self.score_head.weight)
        nn.init.zeros_(self.score_head.bias)

    def forward(self, tokens: torch.Tensor, *, stage_index: int) -> tuple[torch.Tensor, dict]:
        token_count = int(tokens.shape[0])
        if token_count == 0:
            return tokens.new_zeros((0,)), {"tokens": 0, "kept": 0, "mask_sum": 0.0}

        ratio = float(self.keep_ratios[int(stage_index)].detach().cpu().item())
        keep_count = max(1, min(token_count, int(math.ceil(token_count * ratio))))
        if keep_count >= token_count:
            return tokens.new_ones((token_count,)), {
                "tokens": token_count,
                "kept": token_count,
                "mask_sum": float(token_count),
            }

        selector_dtype = self.score_head.weight.dtype
        stage_ids = torch.full((token_count,), int(stage_index), device=tokens.device, dtype=torch.long)
        scored_tokens = tokens.to(dtype=selector_dtype) + self.stage_embeddings(stage_ids)
        scores = self.score_head(self.score_norm(scored_tokens)).squeeze(-1)
        topk = torch.topk(scores, k=keep_count, largest=True, sorted=False)
        hard = torch.zeros_like(scores)
        hard.scatter_(0, topk.indices, 1.0)
        if self.training:
            threshold = topk.values.min().detach()
            soft = torch.sigmoid((scores - threshold) / self.temperature)
            mask = self.min_mask_value + (1.0 - self.min_mask_value) * soft
        else:
            mask = self.min_mask_value + (1.0 - self.min_mask_value) * hard

        mask = mask.to(device=tokens.device, dtype=tokens.dtype)
        return mask, {
            "tokens": token_count,
            "kept": keep_count,
            "mask_sum": float(mask.detach().float().sum().cpu().item()),
        }

    def selector_config(self) -> dict:
        return {
            "num_stages": self.num_stages,
            "keep_ratios": [float(value) for value in self.keep_ratios.detach().cpu().tolist()],
            "temperature": self.temperature,
            "min_mask_value": self.min_mask_value,
        }


class SoftStageMRLColQwen2_5(MRLColQwen2_5):  # noqa: N801
    """MRL_Main model with per-granularity soft visual-token masking.

    This class does not append learnable MRL tokens. It only masks visual token
    embeddings before the LLM while preserving the original MRL_Main output
    protocol and loss masks derived from input_ids.
    """

    def __init__(
        self,
        base_model: ColQwen2_5,
        *,
        granularities: Sequence[int] = (1, 2, 4),
        compact_query_tokens: bool = True,
        softstage_keep_ratios: Optional[Sequence[float]] = None,
        softstage_temperature: float = 0.1,
        softstage_min_mask_value: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=base_model,
            granularities=granularities,
            compact_query_tokens=compact_query_tokens,
        )
        if len(self.stage_specs) != 3:
            raise ValueError("SoftStage expects exactly three stages: g1/g2/g3.")

        hidden_size = int(self.base_model.model.config.hidden_size)
        self.stage_selector = StageAwareSoftSelector(
            hidden_size=hidden_size,
            num_stages=len(self.stage_specs),
            keep_ratios=softstage_keep_ratios,
            temperature=softstage_temperature,
            min_mask_value=softstage_min_mask_value,
        )
        self._softstage_debug_count = 0
        self._last_softstage_stats: Optional[dict] = None

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
        return tensor.detach().to("cpu").reshape(-1).tolist()[:limit]

    def _debug_enabled(self) -> bool:
        return os.environ.get("SOFTSTAGE_DEBUG", "").lower() in {"1", "true", "yes", "on"}

    def _debug_limit(self) -> int:
        try:
            return int(os.environ.get("SOFTSTAGE_DEBUG_LIMIT", "8"))
        except ValueError:
            return 8

    def _should_debug(self) -> bool:
        if not self._debug_enabled():
            return False
        self._softstage_debug_count += 1
        return self._softstage_debug_count <= self._debug_limit()

    def _debug_print(self, message: str) -> None:
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
        print(f"[SoftStageMRL][rank={rank}] {message}", flush=True)

    def _visual_spatial_merge_size(self) -> int:
        value = getattr(self.base_model, "spatial_merge_size", None)
        if callable(value):
            value = value()
        if value is None:
            visual = getattr(self.base_model, "visual", None)
            config = getattr(visual, "config", None)
            value = getattr(config, "spatial_merge_size", None)
        return int(value or 2)

    def _image_grid_token_counts(self, image_grid_thw: torch.LongTensor) -> list[int]:
        merge_size = self._visual_spatial_merge_size()
        denom = merge_size * merge_size
        counts: list[int] = []
        for row in image_grid_thw.detach().to("cpu").tolist():
            t, h, w = [int(value) for value in row]
            total = t * h * w
            if total % denom != 0:
                raise RuntimeError(f"SoftStageMRL image grid is not divisible by merge size: grid={row} merge_size={merge_size}.")
            counts.append(total // denom)
        return counts

    def _apply_softstage_selector(
        self,
        *,
        image_embeds: torch.Tensor,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor],
    ) -> torch.Tensor:
        self._last_softstage_stats = None
        if image_embeds.numel() == 0 or image_grid_thw is None or image_grid_thw.numel() == 0:
            return image_embeds

        image_token_id = int(self.config.image_token_id)
        expected_crops = sum(spec.crop_count for spec in self.stage_specs)
        crop_token_counts = self._image_grid_token_counts(image_grid_thw)
        if sum(crop_token_counts) != int(image_embeds.shape[0]):
            raise RuntimeError(
                "SoftStageMRL image grid/token mismatch: "
                f"grid_tokens={sum(crop_token_counts)} visual_embeds={int(image_embeds.shape[0])}."
            )

        mask_values = image_embeds.new_ones((image_embeds.shape[0],))
        embed_cursor = 0
        grid_cursor = 0
        stage_tokens = [0 for _ in self.stage_specs]
        stage_kept = [0 for _ in self.stage_specs]
        stage_mask_sum = [0.0 for _ in self.stage_specs]
        stage_crops = [0 for _ in self.stage_specs]
        selected_samples = 0
        skipped_samples = 0

        for batch_index in range(input_ids.shape[0]):
            sample_tokens = int(input_ids[batch_index].eq(image_token_id).sum().item())
            if sample_tokens == 0:
                continue
            sample_start_grid = grid_cursor
            sample_start_embed = embed_cursor
            consumed = 0
            while consumed < sample_tokens and grid_cursor < len(crop_token_counts):
                consumed += crop_token_counts[grid_cursor]
                grid_cursor += 1
            if consumed != sample_tokens:
                raise RuntimeError(
                    "SoftStageMRL sample image token mismatch: "
                    f"sample={batch_index} placeholders={sample_tokens} consumed_grid_tokens={consumed}."
                )

            sample_crop_counts = crop_token_counts[sample_start_grid:grid_cursor]
            if len(sample_crop_counts) % expected_crops != 0:
                skipped_samples += 1
                embed_cursor += sum(sample_crop_counts)
                continue

            selected_samples += 1
            local_index = 0
            for _ in range(len(sample_crop_counts) // expected_crops):
                for stage_index, spec in enumerate(self.stage_specs):
                    for _crop_index in range(spec.crop_count):
                        token_count = int(sample_crop_counts[local_index])
                        local_index += 1
                        next_cursor = embed_cursor + token_count
                        crop_tokens = image_embeds[embed_cursor:next_cursor]
                        crop_mask, crop_stats = self.stage_selector(crop_tokens, stage_index=stage_index)
                        mask_values[embed_cursor:next_cursor] = crop_mask
                        embed_cursor = next_cursor
                        stage_tokens[stage_index] += int(crop_stats["tokens"])
                        stage_kept[stage_index] += int(crop_stats["kept"])
                        stage_mask_sum[stage_index] += float(crop_stats["mask_sum"])
                        stage_crops[stage_index] += 1

            if embed_cursor != sample_start_embed + sum(sample_crop_counts):
                raise RuntimeError(
                    "SoftStageMRL sample cursor mismatch: "
                    f"sample={batch_index} cursor={embed_cursor} expected={sample_start_embed + sum(sample_crop_counts)}."
                )

        if embed_cursor != image_embeds.shape[0] or grid_cursor != len(crop_token_counts):
            raise RuntimeError(
                "SoftStageMRL image cursor mismatch: "
                f"embed_cursor={embed_cursor}/{int(image_embeds.shape[0])} grid_cursor={grid_cursor}/{len(crop_token_counts)}."
            )

        self._last_softstage_stats = {
            "selected_samples": selected_samples,
            "skipped_samples": skipped_samples,
            "stage_tokens": stage_tokens,
            "stage_kept": stage_kept,
            "stage_mask_sum": stage_mask_sum,
            "stage_crops": stage_crops,
            "actual_image_embeds": int(image_embeds.shape[0]),
        }
        return image_embeds * mask_values.unsqueeze(-1)

    def _build_inputs_embeds(
        self,
        *,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.LongTensor],
    ) -> torch.Tensor:
        inputs_embeds = self.base_model._embed_tokens(input_ids)
        self._last_softstage_stats = None
        if pixel_values is None:
            return inputs_embeds

        pixel_values = pixel_values.type(self.base_model.visual.dtype)
        image_embeds = self.base_model.visual(pixel_values, grid_thw=image_grid_thw)
        image_mask = input_ids.eq(int(self.config.image_token_id)).unsqueeze(-1).expand_as(inputs_embeds)
        image_embeds = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        expected = int(image_mask.sum().item() // inputs_embeds.shape[-1])
        actual = int(image_embeds.shape[0])
        if expected != actual:
            raise RuntimeError(f"SoftStageMRL image embed mismatch: placeholders={expected} visual_embeds={actual}.")
        image_embeds = self._apply_softstage_selector(
            image_embeds=image_embeds,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
        )
        return inputs_embeds.masked_scatter(image_mask, image_embeds)

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
        kwargs.pop("is_query", None)
        kwargs.pop("has_images", None)
        kwargs.pop("inputs_embeds", None)
        kwargs.pop("position_ids", None)
        debug_active = self._should_debug()
        debug_index = self._softstage_debug_count

        active_pixel_values = pixel_values if self._has_images(pixel_values, image_grid_thw) else None
        active_image_grid_thw = image_grid_thw if active_pixel_values is not None else None
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
        hidden_states = self.base_model.inner_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=None,
            image_grid_thw=None,
            use_cache=False,
            output_hidden_states=True,
            **kwargs,
        )

        proj = self.base_model.custom_text_proj(hidden_states)
        proj = proj / proj.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        proj = proj * attention_mask.to(device=proj.device, dtype=proj.dtype).unsqueeze(-1)
        if debug_active:
            finite = bool(torch.isfinite(proj).all().item())
            self._debug_print(
                f"forward#{debug_index} input_ids={self._debug_shape(input_ids)} output={self._debug_shape(proj)} "
                f"softstage_stats={self._last_softstage_stats} finite={finite}"
            )
        return proj

    def _active_stage_selector_module(self):
        module = self.stage_selector
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

    def softstage_selector_state_dict(self) -> dict:
        active_selector = self._active_stage_selector_module()
        return {
            "config": active_selector.selector_config(),
            "state_dict": {key: value.detach().cpu() for key, value in active_selector.state_dict().items()},
        }

    def save_softstage_state(self, save_dir: str | Path) -> None:
        save_path = Path(save_dir) / "softstage_selector.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.softstage_selector_state_dict(), save_path)

    def load_softstage_state(self, path: str | Path, *, map_location: str | torch.device = "cpu") -> None:
        path = Path(path)
        if path.is_dir():
            path = path / "softstage_selector.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location=map_location)
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        self._active_stage_selector_module().load_state_dict(state_dict, strict=True)

    def save_pretrained(self, save_dir: str, **kwargs):
        self.base_model.save_pretrained(save_dir, **kwargs)
        self.save_softstage_state(save_dir)


SoftStageColQwen2_5 = SoftStageMRLColQwen2_5


def _find_softstage_model(model) -> SoftStageMRLColQwen2_5:
    if isinstance(model, SoftStageMRLColQwen2_5):
        return model
    if hasattr(model, "modules"):
        for module in model.modules():
            if isinstance(module, SoftStageMRLColQwen2_5):
                return module
    raise TypeError(f"Could not find SoftStageMRLColQwen2_5 inside {type(model)!r}.")


def save_softstage_state(model, save_dir: str | Path) -> None:
    _find_softstage_model(model).save_softstage_state(save_dir)


def load_softstage_state(model, path: str | Path, *, map_location: str | torch.device = "cpu") -> None:
    _find_softstage_model(model).load_softstage_state(path, map_location=map_location)


def save_global_mrl_token_state(*args, **kwargs) -> None:
    raise RuntimeError("llmpre/softstage is now pure MRL_Main and has no global_mrl_tokens.pt state.")


def load_global_mrl_token_state(*args, **kwargs) -> None:
    raise RuntimeError("llmpre/softstage is now pure MRL_Main and has no global_mrl_tokens.pt state.")


def _load_adapter_with_fallback(model: SoftStageMRLColQwen2_5, adapter_path: Path):
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

    with TemporaryDirectory(prefix="softstage_mrl_eval_adapter_") as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "adapter_config.json").write_text((adapter_path / "adapter_config.json").read_text())
        torch.save(remapped, tmpdir / "adapter_model.bin")
        return PeftModel.from_pretrained(model, tmpdir)


def build_softstage_model(
    model_name_or_path: str,
    *,
    granularities: Sequence[int] = (1, 2, 4),
    attn_implementation: Optional[str] = "flash_attention_2",
    use_liger_kernel: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
    adapter_path: Optional[str] = None,
    softstage_state_path: Optional[str] = None,
    eval_mode: bool = False,
    compact_query_tokens: bool = True,
    softstage_keep_ratios: Optional[Sequence[float]] = None,
    softstage_temperature: float = 0.1,
    softstage_min_mask_value: float = 0.0,
    **legacy_global_token_kwargs,
):
    granularities = normalize_granularities(granularities)
    if len(build_stage_specs(granularities)) != 3:
        raise ValueError("SoftStage expects exactly three stages.")

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

    model = SoftStageMRLColQwen2_5(
        base_model=base_model,
        granularities=granularities,
        compact_query_tokens=compact_query_tokens,
        softstage_keep_ratios=softstage_keep_ratios,
        softstage_temperature=softstage_temperature,
        softstage_min_mask_value=softstage_min_mask_value,
    )

    if adapter_path is not None:
        model = _load_adapter_with_fallback(model, Path(adapter_path))
        if softstage_state_path is None:
            candidate = Path(adapter_path) / "softstage_selector.pt"
            if candidate.exists():
                softstage_state_path = str(candidate)
    if softstage_state_path is not None:
        load_softstage_state(model, softstage_state_path, map_location="cpu")
    if eval_mode:
        model.eval()
    return model


__all__ = [
    "StageAwareSoftSelector",
    "SoftStageColQwen2_5",
    "SoftStageMRLColQwen2_5",
    "build_softstage_model",
    "save_softstage_state",
    "load_softstage_state",
]
