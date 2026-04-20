"""Legacy executor that bridges the existing registered pipeline classes."""

from __future__ import annotations

from typing import Any

from omnirt.core.types import is_generate_result_like
from omnirt.executors.base import Executor
from omnirt.executors.events import emit_event
from omnirt.middleware.backend_wrapper import BackendWrapperMiddleware


class LegacyCallExecutor(Executor):
    name = "legacy_call"

    def __init__(self) -> None:
        super().__init__()
        self.pipeline: Any = None

    def load(self, *, runtime, model_spec, config, adapters) -> None:
        if self.pipeline is not None:
            return
        self.runtime = runtime
        self.model_spec = model_spec
        self.config = dict(config)
        self.adapters = list(adapters or [])
        self.pipeline = model_spec.pipeline_cls(runtime=runtime, model_spec=model_spec, adapters=self.adapters)
        self.components = dict(getattr(self.pipeline, "components", {}) or {})
        self.apply_middleware([BackendWrapperMiddleware()])

    def run(self, request, *, event_callback=None, cache=None) -> Any:
        del cache
        emit_event(event_callback, "stage_start", "legacy_call", data={"model": request.model})
        try:
            result = self.pipeline.run(request)
        except Exception as exc:
            emit_event(
                event_callback,
                "stage_error",
                "legacy_call",
                data={"model": request.model, "error": str(exc)},
            )
            raise
        emit_event(event_callback, "stage_end", "legacy_call", data={"model": request.model})
        if is_generate_result_like(result):
            result.metadata.execution_mode = self.name
        return result

    def release(self) -> None:
        self.components.clear()
        self.pipeline = None
