from importlib import import_module
from typing import Any

__all__ = [
    "BiPali",
    "BiPaliProj",
    "BiQwen2",
    "BiQwen2_5",
    "BiQwen2_5_Processor",
    "BiQwen2Processor",
    "ColIdefics3",
    "ColIdefics3Processor",
    "ColPali",
    "ColPaliProcessor",
    "ColQwen2",
    "ColQwen2_5",
    "ColQwen2_5_Processor",
    "ColQwen2Processor",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    return getattr(import_module("colpali_engine.models"), name)
