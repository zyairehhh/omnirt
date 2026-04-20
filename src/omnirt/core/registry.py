"""Model registration and lookup for OmniRT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Type

from omnirt.core.types import ModelNotRegisteredError


@dataclass
class ModelSpec:
    id: str
    task: str
    pipeline_cls: Type[Any]
    default_backend: str = "auto"
    resource_hint: Dict[str, Any] = field(default_factory=dict)


_MODEL_REGISTRY: Dict[str, ModelSpec] = {}


def clear_registry() -> None:
    _MODEL_REGISTRY.clear()


def register_model(
    *,
    id: str,
    task: str,
    default_backend: str = "auto",
    resource_hint: Optional[Dict[str, Any]] = None,
) -> Callable[[Type[Any]], Type[Any]]:
    def decorator(pipeline_cls: Type[Any]) -> Type[Any]:
        _MODEL_REGISTRY[id] = ModelSpec(
            id=id,
            task=task,
            pipeline_cls=pipeline_cls,
            default_backend=default_backend,
            resource_hint=dict(resource_hint or {}),
        )
        return pipeline_cls

    return decorator


def get_model(model_id: str) -> ModelSpec:
    try:
        return _MODEL_REGISTRY[model_id]
    except KeyError as exc:
        raise ModelNotRegisteredError(f"Model {model_id!r} is not registered.") from exc


def list_models() -> Dict[str, ModelSpec]:
    return dict(_MODEL_REGISTRY)

