"""Pure MRL_Main SoftStage visual compression before the LLM."""

from .modeling_softstage import (
    SoftStageColQwen2_5,
    SoftStageMRLColQwen2_5,
    StageAwareSoftSelector,
    build_softstage_model,
    load_softstage_state,
    save_softstage_state,
)

__all__ = [
    "SoftStageColQwen2_5",
    "SoftStageMRLColQwen2_5",
    "StageAwareSoftSelector",
    "build_softstage_model",
    "load_softstage_state",
    "save_softstage_state",
]
