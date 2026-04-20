"""Shared executor abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Iterable, List, Optional, Protocol

from omnirt.backends.base import BackendRuntime
from omnirt.core.registry import ModelSpec
from omnirt.core.types import AdapterRef, GenerateRequest
from omnirt.executors.events import EventCallback


class Middleware(Protocol):
    def apply(self, components: dict[str, Any], *, runtime: BackendRuntime, config: dict[str, Any]) -> dict[str, Any]:
        """Return wrapped components."""


class Executor(ABC):
    name: ClassVar[str] = "executor"

    def __init__(self) -> None:
        self.runtime: Optional[BackendRuntime] = None
        self.model_spec: Optional[ModelSpec] = None
        self.config: dict[str, Any] = {}
        self.adapters: List[AdapterRef] = []
        self.components: dict[str, Any] = {}

    @abstractmethod
    def load(
        self,
        *,
        runtime: BackendRuntime,
        model_spec: ModelSpec,
        config: dict[str, Any],
        adapters: list[AdapterRef] | None,
    ) -> None:
        """Load and initialize executor state."""

    @abstractmethod
    def run(
        self,
        request: GenerateRequest,
        *,
        event_callback: EventCallback | None = None,
        cache: Any = None,
    ) -> Any:
        """Execute one generation request."""

    @abstractmethod
    def release(self) -> None:
        """Release owned resources."""

    def apply_middleware(self, middlewares: Iterable[Middleware]) -> None:
        if self.runtime is None:
            raise RuntimeError("Executor runtime is not initialized.")
        wrapped = dict(self.components)
        for middleware in middlewares:
            wrapped = middleware.apply(wrapped, runtime=self.runtime, config=self.config)
        self.components = wrapped
