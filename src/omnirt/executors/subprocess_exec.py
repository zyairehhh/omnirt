"""Executor variant reserved for script-backed models."""

from __future__ import annotations

from typing import Any

from omnirt.core.types import is_generate_result_like
from omnirt.executors.base import Executor
from omnirt.executors.events import emit_event


class SubprocessExecutor(Executor):
    name = "subprocess"

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

    def run(self, request, *, event_callback=None, cache=None) -> Any:
        del cache
        emit_event(event_callback, "stage_start", "subprocess", data={"model": request.model})
        try:
            result = self.pipeline.run(request)
        except Exception as exc:
            emit_event(
                event_callback,
                "stage_error",
                "subprocess",
                data={"model": request.model, "error": str(exc)},
            )
            raise
        emit_event(event_callback, "stage_end", "subprocess", data={"model": request.model})
        if is_generate_result_like(result):
            result.metadata.execution_mode = self.name
        return result

    def release(self) -> None:
        self.pipeline = None
