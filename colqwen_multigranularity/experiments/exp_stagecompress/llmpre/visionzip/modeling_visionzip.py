from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import torch

from colqwen_multigranularity.experiments.exp_stagecompress.llmpre.visionzip_mrl.modeling_visionzip_mrl import (
    StageWiseVisionZipSelector,
    VisionZipMRLColQwen2_5,
    build_visionzip_mrl_model,
    load_visionzip_mrl_state,
    save_visionzip_mrl_state,
)


VisionZipColQwen2_5 = VisionZipMRLColQwen2_5


def build_visionzip_model(
    model_name_or_path: str,
    *,
    granularities: Sequence[int] = (1, 2, 4),
    attn_implementation: Optional[str] = "flash_attention_2",
    use_liger_kernel: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
    adapter_path: Optional[str] = None,
    visionzip_state_path: Optional[str] = None,
    eval_mode: bool = False,
    compact_query_tokens: bool = True,
    visionzip_mode: str = "mask",
    visionzip_position: str = "llm_early",
    visionzip_exit_layer: int = 2,
    visionzip_keep_ratios: Optional[Sequence[float]] = None,
    visionzip_dominant_ratio: float = 0.65,
    visionzip_contextual_ratio: float = 0.05,
    visionzip_temperature: float = 0.1,
    visionzip_min_mask_value: float = 0.0,
    visionzip_train_prune: bool = False,
    visionzip_use_context: bool = True,
    **legacy_global_token_kwargs,
):
    """Backward-compatible old VisionZip entry, now pure MRL_Main.

    Old kwargs such as num_query_mrl_tokens/global_mrl_token_path are accepted
    and ignored so stale launch scripts fail less often, but no learnable MRL
    prompt tokens are constructed or loaded.
    """

    if visionzip_state_path is None:
        legacy_state = legacy_global_token_kwargs.get("visionzip_mrl_state_path")
        if legacy_state is not None:
            visionzip_state_path = legacy_state
    return build_visionzip_mrl_model(
        model_name_or_path,
        granularities=granularities,
        attn_implementation=attn_implementation,
        use_liger_kernel=use_liger_kernel,
        torch_dtype=torch_dtype,
        adapter_path=adapter_path,
        visionzip_mrl_state_path=visionzip_state_path,
        eval_mode=eval_mode,
        compact_query_tokens=compact_query_tokens,
        visionzip_mode=visionzip_mode,
        visionzip_position=visionzip_position,
        visionzip_exit_layer=visionzip_exit_layer,
        visionzip_keep_ratios=visionzip_keep_ratios,
        visionzip_dominant_ratio=visionzip_dominant_ratio,
        visionzip_contextual_ratio=visionzip_contextual_ratio,
        visionzip_temperature=visionzip_temperature,
        visionzip_min_mask_value=visionzip_min_mask_value,
        visionzip_train_prune=visionzip_train_prune,
        visionzip_use_context=visionzip_use_context,
    )


def save_visionzip_state(model, save_dir: str | Path) -> None:
    save_visionzip_mrl_state(model, save_dir)


def load_visionzip_state(model, path: str | Path, *, map_location: str | torch.device = "cpu") -> None:
    load_visionzip_mrl_state(model, path, map_location=map_location)


def save_global_mrl_token_state(*args, **kwargs) -> None:
    raise RuntimeError("llmpre/visionzip is now pure MRL_Main and has no global_mrl_tokens.pt state.")


def load_global_mrl_token_state(*args, **kwargs) -> None:
    raise RuntimeError("llmpre/visionzip is now pure MRL_Main and has no global_mrl_tokens.pt state.")


__all__ = [
    "StageWiseVisionZipSelector",
    "VisionZipColQwen2_5",
    "VisionZipMRLColQwen2_5",
    "build_visionzip_model",
    "save_visionzip_state",
    "load_visionzip_state",
]
