"""Best-effort quantization helpers for modular and legacy pipelines."""

from __future__ import annotations

import inspect
from typing import Any, Dict, Iterable


QUANTIZATION_CONFIG_KEYS = (
    "quantization",
    "quantization_backend",
    "enable_layerwise_casting",
    "layerwise_casting_storage_dtype",
    "layerwise_casting_compute_dtype",
)

_DEFAULT_COMPONENT_TAGS = (
    "text_encoder",
    "text_encoder_2",
    "image_encoder",
    "transformer",
    "transformer_2",
    "unet",
    "vae",
)


def has_quantization_config(config: Dict[str, Any]) -> bool:
    return any(config.get(key) not in (None, "", False) for key in QUANTIZATION_CONFIG_KEYS)


class QuantizationMiddleware:
    def apply(self, components: dict[str, Any], *, runtime, config) -> dict[str, Any]:
        del runtime
        if not has_quantization_config(config):
            return components
        for name, component in components.items():
            _apply_quantization(component, component_name=name, config=config)
        return components


def apply_quantization_runtime(
    pipeline: Any,
    *,
    config: Dict[str, Any],
    component_tags: Iterable[str] = _DEFAULT_COMPONENT_TAGS,
) -> None:
    if not has_quantization_config(config):
        return
    seen: set[int] = set()
    for name, target in _iter_targets(pipeline, component_tags=component_tags):
        if target is None or id(target) in seen:
            continue
        seen.add(id(target))
        _apply_quantization(target, component_name=name, config=config)


def _iter_targets(pipeline: Any, *, component_tags: Iterable[str]) -> Iterable[tuple[str, Any]]:
    yield ("pipeline", pipeline)
    for tag in component_tags:
        component = getattr(pipeline, tag, None)
        if component is not None:
            yield (tag, component)


def _apply_quantization(target: Any, *, component_name: str, config: Dict[str, Any]) -> None:
    payload = {
        "mode": config.get("quantization"),
        "backend": config.get("quantization_backend", "auto"),
        "enable_layerwise_casting": bool(config.get("enable_layerwise_casting", False)),
        "storage_dtype": config.get("layerwise_casting_storage_dtype"),
        "compute_dtype": config.get("layerwise_casting_compute_dtype"),
    }
    if payload["mode"] not in (None, ""):
        _maybe_call(
            target,
            ("apply_quantization", "quantize", "set_quantization_config"),
            {
                "mode": payload["mode"],
                "backend": payload["backend"],
                "config": payload,
            },
        )
    if payload["enable_layerwise_casting"]:
        _maybe_call(
            target,
            ("enable_layerwise_casting",),
            {
                "storage_dtype": payload["storage_dtype"],
                "compute_dtype": payload["compute_dtype"],
            },
        )
    setattr(target, "_omnirt_quantization", dict(payload))
    setattr(target, "_omnirt_component_name", component_name)


def _maybe_call(target: Any, method_names: tuple[str, ...], kwargs: Dict[str, Any]) -> bool:
    for method_name in method_names:
        method = getattr(target, method_name, None)
        if method is None or not callable(method):
            continue
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            signature = None
        filtered_kwargs = dict(kwargs)
        if signature is not None:
            parameters = signature.parameters
            if not any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
                filtered_kwargs = {
                    key: value
                    for key, value in filtered_kwargs.items()
                    if value is not None and key in parameters
                }
            else:
                filtered_kwargs = {key: value for key, value in filtered_kwargs.items() if value is not None}
        try:
            method(**filtered_kwargs)
        except TypeError:
            continue
        else:
            return True
    return False
