"""Pydantic request helpers for the HTTP server."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from omnirt.core.types import GenerateRequest


class GenerateSubmission(BaseModel):
    task: str
    model: str
    backend: str = "auto"
    inputs: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)
    adapters: Optional[List[Dict[str, Any]]] = None
    async_run: bool = False

    def to_request(self) -> GenerateRequest:
        payload = self.model_dump()
        payload.pop("async_run", None)
        return GenerateRequest.from_dict(payload)
