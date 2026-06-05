from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import torch

from colqwen_multigranularity.experiments.exp_stagecompress.llmpre.twigmrl.modeling_twigmrl import (
    StageWiseTwigSelector,
    TwigMRLColQwen2_5,
    build_twigmrl_model,
    load_twigmrl_state,
    save_twigmrl_state,
)


TwigStageColQwen2_5 = TwigMRLColQwen2_5


def build_twigstage_model(
    model_name_or_path: str,
    *,
    granularities: Sequence[int] = (1, 2, 4),
    attn_implementation: Optional[str] = "flash_attention_2",
    use_liger_kernel: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
    adapter_path: Optional[str] = None,
    twigstage_state_path: Optional[str] = None,
    eval_mode: bool = False,
    compact_query_tokens: bool = True,
    twigstage_mode: str = "mask",
    twigstage_exit_layer: int = 2,
    twigstage_keep_ratios: Optional[Sequence[float]] = None,
    twigstage_temperature: float = 0.1,
    twigstage_min_mask_value: float = 0.0,
    twigstage_train_prune: bool = False,
    twigstage_use_context: bool = True,
    **legacy_global_token_kwargs,
):
    """Backward-compatible old TwigStage entry, now pure MRL_Main.

    Old GlobalMRLToken kwargs are accepted and ignored. The active model is
    TwigMRL, which prunes g1/g2/g3 visual tokens after early LLM layers without
    appending learnable MRL prompt tokens.
    """

    if twigstage_state_path is None:
        legacy_state = legacy_global_token_kwargs.get("twigmrl_state_path")
        if legacy_state is not None:
            twigstage_state_path = legacy_state
    return build_twigmrl_model(
        model_name_or_path,
        granularities=granularities,
        attn_implementation=attn_implementation,
        use_liger_kernel=use_liger_kernel,
        torch_dtype=torch_dtype,
        adapter_path=adapter_path,
        twigmrl_state_path=twigstage_state_path,
        eval_mode=eval_mode,
        compact_query_tokens=compact_query_tokens,
        twigmrl_mode=twigstage_mode,
        twigmrl_exit_layer=twigstage_exit_layer,
        twigmrl_keep_ratios=twigstage_keep_ratios,
        twigmrl_temperature=twigstage_temperature,
        twigmrl_min_mask_value=twigstage_min_mask_value,
        twigmrl_train_prune=twigstage_train_prune,
        twigmrl_use_context=twigstage_use_context,
    )


def save_twigstage_state(model, save_dir: str | Path) -> None:
    save_twigmrl_state(model, save_dir)


def load_twigstage_state(model, path: str | Path, *, map_location: str | torch.device = "cpu") -> None:
    load_twigmrl_state(model, path, map_location=map_location)


def save_global_mrl_token_state(*args, **kwargs) -> None:
    raise RuntimeError("llmpre/twigstage is now pure MRL_Main and has no global_mrl_tokens.pt state.")


def load_global_mrl_token_state(*args, **kwargs) -> None:
    raise RuntimeError("llmpre/twigstage is now pure MRL_Main and has no global_mrl_tokens.pt state.")


__all__ = [
    "StageWiseTwigSelector",
    "TwigStageColQwen2_5",
    "TwigMRLColQwen2_5",
    "build_twigstage_model",
    "save_twigstage_state",
    "load_twigstage_state",
]
