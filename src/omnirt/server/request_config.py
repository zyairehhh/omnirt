"""Shared request normalization helpers for HTTP routes."""

from __future__ import annotations

from omnirt.core.types import GenerateRequest
from omnirt.server.model_aliases import resolve_model_alias


def normalize_generate_request(raw_request: GenerateRequest, app_state) -> GenerateRequest:
    backend = raw_request.backend if raw_request.backend != "auto" else app_state.default_backend
    merged_config = dict(getattr(app_state, "default_request_config", {}) or {})
    merged_config.update(raw_request.config)
    return GenerateRequest(
        task=raw_request.task,
        model=resolve_model_alias(raw_request.model, app_state.model_aliases),
        backend=backend,
        inputs=dict(raw_request.inputs),
        config=merged_config,
        adapters=raw_request.adapters,
    )
