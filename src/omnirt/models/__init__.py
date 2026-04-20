"""Built-in model registration helpers."""

from __future__ import annotations

import importlib

from omnirt.core.registry import list_models

_REGISTERED = False
_BUILTIN_MODEL_IDS = {
    "sdxl-base-1.0",
    "svd",
    "svd-xt",
    "flux2.dev",
    "flux2-dev",
    "wan2.2-t2v-14b",
    "wan2.2-i2v-14b",
}


def ensure_registered() -> None:
    global _REGISTERED
    if _REGISTERED and _BUILTIN_MODEL_IDS.issubset(set(list_models())):
        return

    from omnirt.models.sdxl import pipeline as _sdxl_pipeline  # noqa: F401
    from omnirt.models.svd import pipeline as _svd_pipeline  # noqa: F401
    from omnirt.models.flux2 import pipeline as _flux2_pipeline  # noqa: F401
    from omnirt.models.wan import pipeline as _wan_pipeline  # noqa: F401

    registered_ids = set(list_models())
    if "sdxl-base-1.0" not in registered_ids:
        importlib.reload(_sdxl_pipeline)
        registered_ids = set(list_models())
    if not {"svd", "svd-xt"}.issubset(registered_ids):
        importlib.reload(_svd_pipeline)
        registered_ids = set(list_models())
    if not {"flux2.dev", "flux2-dev"}.issubset(registered_ids):
        importlib.reload(_flux2_pipeline)
        registered_ids = set(list_models())
    if not {"wan2.2-t2v-14b", "wan2.2-i2v-14b"}.issubset(registered_ids):
        importlib.reload(_wan_pipeline)
        registered_ids = set(list_models())

    _REGISTERED = _BUILTIN_MODEL_IDS.issubset(registered_ids)
