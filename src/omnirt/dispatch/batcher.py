"""Request batching helpers for queued execution."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable, Sequence
import uuid

from omnirt.core.types import GenerateRequest, GenerateResult
from omnirt.dispatch.queue import JobWorkItem

_BATCHABLE_TASKS = frozenset({"text2image"})
_RUN_LOCAL_CONFIG_KEYS = frozenset({"seed"})


@dataclass(frozen=True)
class BatchGroup:
    group_id: str
    items: tuple[JobWorkItem, ...]
    request: GenerateRequest

    @property
    def size(self) -> int:
        return len(self.items)


class RequestBatcher:
    def __init__(self, *, batch_window_ms: int = 0, max_batch_size: int = 1) -> None:
        self.batch_window_ms = max(int(batch_window_ms), 0)
        self.max_batch_size = max(int(max_batch_size), 1)

    @property
    def enabled(self) -> bool:
        return self.batch_window_ms > 0 and self.max_batch_size > 1

    def matches(self, anchor: JobWorkItem, candidate: JobWorkItem) -> bool:
        return self.enabled and self._batch_signature(anchor) == self._batch_signature(candidate)

    def create_group(self, items: Sequence[JobWorkItem]) -> BatchGroup | None:
        if len(items) <= 1:
            return None
        if not all(self.matches(items[0], item) for item in items[1:]):
            return None
        combined = self.combine_requests([item.request for item in items])
        return BatchGroup(group_id=str(uuid.uuid4()), items=tuple(items), request=combined)

    def combine_requests(self, requests: Sequence[GenerateRequest]) -> GenerateRequest:
        if not requests:
            raise ValueError("Cannot combine an empty request batch.")

        first = requests[0]
        payload = first.to_dict()
        payload["inputs"] = dict(first.inputs)
        payload["config"] = dict(first.config)
        payload["inputs"]["prompt"] = [str(request.inputs["prompt"]) for request in requests]

        negative_prompts = [request.inputs.get("negative_prompt") for request in requests]
        if any(prompt not in (None, "") for prompt in negative_prompts):
            payload["inputs"]["negative_prompt"] = ["" if prompt is None else str(prompt) for prompt in negative_prompts]
        else:
            payload["inputs"].pop("negative_prompt", None)

        seeds = [request.config.get("seed") for request in requests]
        if any(seed is not None for seed in seeds):
            payload["config"]["seed"] = seeds
        payload["config"]["use_result_cache"] = False

        return GenerateRequest.from_dict(payload)

    def split_result(self, result: GenerateResult, items: Sequence[JobWorkItem], *, batch_group_id: str) -> list[GenerateResult]:
        batch_size = len(items)
        if batch_size <= 1:
            clone = GenerateResult.from_dict(result.to_dict())
            clone.metadata.batch_size = 1
            clone.metadata.batch_group_id = None
            return [clone]

        images_per_prompt = int(items[0].request.config.get("num_images_per_prompt", 1) or 1)
        expected_outputs = batch_size * images_per_prompt
        if result.outputs and len(result.outputs) != expected_outputs:
            raise ValueError(
                f"Batched result output count mismatch: expected {expected_outputs}, got {len(result.outputs)}."
            )

        payload = result.to_dict()
        results: list[GenerateResult] = []
        for index in range(batch_size):
            child_payload = dict(payload)
            start = index * images_per_prompt
            end = start + images_per_prompt
            child_payload["outputs"] = payload["outputs"][start:end]
            child_result = GenerateResult.from_dict(child_payload)
            child_result.metadata.batch_size = batch_size
            child_result.metadata.batch_group_id = batch_group_id
            results.append(child_result)
        return results

    def _batch_signature(self, item: JobWorkItem) -> str | None:
        spec = item.model_spec
        request = item.request
        caps = getattr(spec, "capabilities", None)

        if getattr(spec, "execution_mode", None) != "modular":
            return None
        if request.task not in _BATCHABLE_TASKS:
            return None
        if caps is not None and not getattr(caps, "supports_batching", True):
            return None
        if not isinstance(request.inputs.get("prompt"), str) or not request.inputs.get("prompt"):
            return None
        if request.inputs.get("image") is not None or request.inputs.get("mask") is not None or request.inputs.get("audio") is not None:
            return None
        if request.config.get("num_images_per_prompt", 1) not in (None, 1):
            return None

        payload = {
            "model": request.model,
            "task": request.task,
            "backend": getattr(item.runtime, "name", request.backend),
            "config": self._stable_config(request.config),
            "adapters": self._adapter_fingerprint(request.adapters),
            "negative_prompt": request.inputs.get("negative_prompt"),
        }
        return json.dumps(payload, sort_keys=True, default=str)

    def _stable_config(self, config: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in config.items() if key not in _RUN_LOCAL_CONFIG_KEYS}

    def _adapter_fingerprint(self, adapters: Iterable[Any] | None) -> list[dict[str, Any]]:
        return [
            {
                "kind": getattr(adapter, "kind", None),
                "path": getattr(adapter, "path", None),
                "scale": getattr(adapter, "scale", None),
            }
            for adapter in (adapters or [])
        ]
