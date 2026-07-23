from pathlib import Path
import sys

_PROJECT_DIR = Path(__file__).resolve().parent
_VENDOR_DIR = _PROJECT_DIR / "vendor"
if _VENDOR_DIR.exists():
    _VENDOR_PATH = str(_VENDOR_DIR)
    if _VENDOR_PATH in sys.path:
        sys.path.remove(_VENDOR_PATH)
    sys.path.insert(0, _VENDOR_PATH)

from .core import (
    CropLayout,
    MultiGranularityColQwen2_5Processor,
    build_colqwen2_5_model,
    build_stage_specs,
    normalize_granularities,
)
