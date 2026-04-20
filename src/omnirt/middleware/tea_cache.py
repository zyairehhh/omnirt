"""Best-effort TeaCache integration helpers."""

from __future__ import annotations

import inspect
from typing import Any, Dict, Iterable


TEA_CACHE_CONFIG_KEYS = (
    "cache",
    "enable_tea_cache",
    "tea_cache_ratio",
    "tea_cache_interval",
)

_DEFAULT_COMPONENT_TAGS = (
    "transformer",
    "transformer_2",
    "unet",
)


def has_tea_cache_config(config: Dict[str, Any]) -> bool:
    return bool(config.get("enable_tea_cache")) or str(config.get("cache", "")).lower() == "tea_cache"


class TeaCacheMiddleware:
    def apply(self, components: dict[str, Any], *, runtime, config) -> dict[str, Any]:
        del runtime
        if not has_tea_cache_config(config):
            return components
        for name, component in components.items():
            _apply_tea_cache(component, component_name=name, config=config)
        return components


def apply_tea_cache_runtime(
    pipeline: Any,
    *,
    config: Dict[str, Any],
    component_tags: Iterable[str] = _DEFAULT_COMPONENT_TAGS,
) -> None:
    if not has_tea_cache_config(config):
        return
    seen: set[int] = set()
    for name, target in _iter_targets(pipeline, component_tags=component_tags):
        if target is None or id(target) in seen:
            continue
        seen.add(id(target))
        _apply_tea_cache(target, component_name=name, config=config)


def _iter_targets(pipeline: Any, *, component_tags: Iterable[str]) -> Iterable[tuple[str, Any]]:
    yield ("pipeline", pipeline)
    for tag in component_tags:
        component = getattr(pipeline, tag, None)
        if component is not None:
            yield (tag, component)


def _apply_tea_cache(target: Any, *, component_name: str, config: Dict[str, Any]) -> None:
    payload = {
        "enabled": True,
        "ratio": float(config.get("tea_cache_ratio", 0.0) or 0.0),
        "interval": int(config.get("tea_cache_interval", 1) or 1),
    }
    _maybe_call(
        target,
        ("enable_teacache", "enable_tea_cache", "set_teacache_config"),
        payload,
    )
    setattr(target, "_omnirt_tea_cache", dict(payload))
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
