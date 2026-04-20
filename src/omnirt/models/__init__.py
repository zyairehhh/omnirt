"""Built-in model registration helpers."""

from __future__ import annotations

_REGISTERED = False


def ensure_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from omnirt.models.sdxl import pipeline as _sdxl_pipeline  # noqa: F401
    from omnirt.models.svd import pipeline as _svd_pipeline  # noqa: F401

    _REGISTERED = True

