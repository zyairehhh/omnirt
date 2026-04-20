"""Worker routing scaffold for future controller/worker split."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, Dict, Iterable, Optional, Protocol


@dataclass(frozen=True)
class WorkerEndpoint:
    worker_id: str
    address: str
    models: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


class WorkerClient(Protocol):
    def run_sync(self, request, *, model_spec=None, runtime=None) -> Any:
        ...


class InProcessWorkerClient:
    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def run_sync(self, request, *, model_spec=None, runtime=None) -> Any:
        return self.engine.run_sync(request, model_spec=model_spec, runtime=runtime)


class Controller:
    def __init__(self) -> None:
        self._workers: Dict[str, WorkerEndpoint] = {}
        self._rr_counter = 0
        self._lock = threading.RLock()

    def register_worker(self, endpoint: WorkerEndpoint) -> WorkerEndpoint:
        with self._lock:
            self._workers[endpoint.worker_id] = endpoint
            return endpoint

    def unregister_worker(self, worker_id: str) -> None:
        with self._lock:
            self._workers.pop(worker_id, None)

    def list_workers(self) -> tuple[WorkerEndpoint, ...]:
        with self._lock:
            return tuple(self._workers.values())

    def route(self, *, model: str, tags: Iterable[str] = ()) -> Optional[WorkerEndpoint]:
        requested_tags = {str(tag) for tag in tags}
        with self._lock:
            candidates = [endpoint for endpoint in self._workers.values() if not endpoint.models or model in endpoint.models]
            if requested_tags:
                candidates = [
                    endpoint
                    for endpoint in candidates
                    if requested_tags.issubset(set(endpoint.tags))
                ]
            if not candidates:
                return None
            selected = candidates[self._rr_counter % len(candidates)]
            self._rr_counter += 1
            return selected
