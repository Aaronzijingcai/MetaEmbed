from importlib import import_module
from typing import Any

_EXPORTS = {
    "ColIdefics3": "colpali_engine.models.idefics3",
    "ColIdefics3Processor": "colpali_engine.models.idefics3",
    "LastLlama3Vision": "colpali_engine.models.llama3vision.modeling_lastllama3vision",
    "BiPali": "colpali_engine.models.paligemma",
    "BiPaliProcessor": "colpali_engine.models.paligemma",
    "BiPaliProj": "colpali_engine.models.paligemma",
    "ColPali": "colpali_engine.models.paligemma",
    "ColPaliProcessor": "colpali_engine.models.paligemma",
    "LastPali": "colpali_engine.models.paligemma",
    "BiQwen2": "colpali_engine.models.qwen2",
    "BiQwen2Processor": "colpali_engine.models.qwen2",
    "ColQwen2": "colpali_engine.models.qwen2",
    "ColQwen2Processor": "colpali_engine.models.qwen2",
    "BiQwen2_5": "colpali_engine.models.qwen2_5",
    "BiQwen2_5_Processor": "colpali_engine.models.qwen2_5",
    "ColQwen2_5": "colpali_engine.models.qwen2_5",
    "ColQwen2_5_Processor": "colpali_engine.models.qwen2_5",
    "LastQwen2_5": "colpali_engine.models.qwen2_5",
    "LastQwen3": "colpali_engine.models.qwen3",
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(name)
    module = import_module(module_path)
    return getattr(module, name)
