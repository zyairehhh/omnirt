"""Public OmniRT API entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from omnirt.backends import resolve_backend
from omnirt.core.registry import get_model
from omnirt.core.types import GenerateRequest, GenerateResult
from omnirt.models import ensure_registered


RequestLike = Union[GenerateRequest, Dict[str, Any], str, Path]


def _coerce_request(request: RequestLike) -> GenerateRequest:
    if isinstance(request, GenerateRequest):
        return request
    if isinstance(request, (str, Path)):
        return GenerateRequest.from_file(request)
    if isinstance(request, dict):
        return GenerateRequest.from_dict(request)
    raise TypeError(f"Unsupported request type: {type(request)!r}")


def generate(request: RequestLike, *, backend: Optional[str] = None) -> GenerateResult:
    """Run a generation request through a registered pipeline.

    Backend resolution priority: explicit ``backend`` argument > ``req.backend`` > ``"auto"``.
    """

    ensure_registered()
    req = _coerce_request(request)
    spec = get_model(req.model)
    if req.task != spec.task:
        raise ValueError(
            f"Model {req.model!r} only supports task {spec.task!r}, got request task {req.task!r}."
        )
    selected = backend if backend is not None else (req.backend or "auto")
    runtime = resolve_backend(selected)
    pipeline = spec.pipeline_cls(runtime=runtime, model_spec=spec, adapters=req.adapters)
    return pipeline.run(req)
