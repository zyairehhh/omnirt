"""Middleware for backend module wrapping."""

from __future__ import annotations

from typing import Any


class BackendWrapperMiddleware:
    def apply(self, components: dict[str, Any], *, runtime, config) -> dict[str, Any]:
        del config
        wrapped: dict[str, Any] = {}
        for name, component in components.items():
            wrapped[name] = runtime.wrap_module(component, tag=name)
        return wrapped
