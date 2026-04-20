"""Optional MindIE-SD integration helpers for the Ascend backend."""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Dict, Optional


ASCEND_ACCELERATION_CONFIG_KEYS = (
    "ascend_attention_backend",
    "ascend_dit_cache",
    "ascend_lora_hot_swap",
)

_MODULE_PATCH_CANDIDATES = (
    "patch_module",
    "wrap_module",
    "accelerate_module",
    "apply_to_module",
)
_ATTENTION_BACKEND_CANDIDATES = (
    "set_attention_backend",
    "configure_attention_backend",
)


def mindie_available() -> bool:
    try:
        importlib.import_module("mindiesd")
    except ImportError:
        return False
    return True


def register_ascend_overrides(runtime: Any) -> None:
    runtime._mindie_registered = True


def prepare_ascend_pipeline(runtime: Any, pipeline: Any, *, model_spec: Any, config: Dict[str, Any]) -> Any:
    if not _mindie_requested(config):
        return pipeline

    mindiesd = _load_mindiesd()
    if mindiesd is None:
        return pipeline

    attention_backend = _normalize_optional_string(config.get("ascend_attention_backend"))
    if attention_backend is not None:
        _apply_attention_backend(mindiesd, pipeline, attention_backend)

    if _truthy(config.get("ascend_lora_hot_swap")):
        setattr(pipeline, "_omnirt_lora_hot_swap", True)

    metadata = {
        "enabled": True,
        "attention_backend": attention_backend,
        "dit_cache": _truthy(config.get("ascend_dit_cache")),
        "lora_hot_swap": _truthy(config.get("ascend_lora_hot_swap")),
    }
    setattr(pipeline, "_omnirt_mindie", metadata)

    for tag in ("text_encoder", "text_encoder_2", "image_encoder", "transformer", "transformer_2", "unet", "vae"):
        module = getattr(pipeline, tag, None)
        if module is None:
            continue
        replacement = _patch_module(
            mindiesd,
            module=module,
            tag=tag,
            model_spec=model_spec,
            config=config,
        )
        if replacement is not None and replacement is not module:
            runtime.register_override(tag, replacement)

    return pipeline


def _mindie_requested(config: Dict[str, Any]) -> bool:
    return any(
        _truthy(config.get(key)) if key != "ascend_attention_backend" else _normalize_optional_string(config.get(key))
        for key in ASCEND_ACCELERATION_CONFIG_KEYS
    )


def _load_mindiesd() -> Optional[Any]:
    try:
        return importlib.import_module("mindiesd")
    except ImportError:
        return None


def _apply_attention_backend(mindiesd: Any, pipeline: Any, backend_name: str) -> None:
    for candidate in _ATTENTION_BACKEND_CANDIDATES:
        fn = getattr(mindiesd, candidate, None)
        if callable(fn):
            _call_with_supported_kwargs(fn, pipeline=pipeline, backend=backend_name, backend_name=backend_name)
            return


def _patch_module(mindiesd: Any, *, module: Any, tag: str, model_spec: Any, config: Dict[str, Any]) -> Optional[Any]:
    for candidate in _MODULE_PATCH_CANDIDATES:
        fn = getattr(mindiesd, candidate, None)
        if callable(fn):
            return _call_with_supported_kwargs(
                fn,
                module=module,
                tag=tag,
                config=config,
                model_spec=model_spec,
                model_id=getattr(model_spec, "id", None),
                task=getattr(model_spec, "task", None),
            )
    return None


def _call_with_supported_kwargs(fn: Any, **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(**kwargs)

    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return fn(**kwargs)
    accepted = {key: value for key, value in kwargs.items() if key in parameters}
    return fn(**accepted)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _normalize_optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
