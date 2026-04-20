"""Backend-specific accelerator integration hooks."""

from omnirt.backends.overrides.ascend_mindie import (
    ASCEND_ACCELERATION_CONFIG_KEYS,
    mindie_available,
    prepare_ascend_pipeline,
    register_ascend_overrides,
)

__all__ = [
    "ASCEND_ACCELERATION_CONFIG_KEYS",
    "mindie_available",
    "prepare_ascend_pipeline",
    "register_ascend_overrides",
]
