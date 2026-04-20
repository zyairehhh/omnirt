"""Simple in-process job queue."""

from __future__ import annotations

from dataclasses import dataclass
import queue
import time
from typing import Any, Callable

from omnirt.core.types import GenerateRequest

@dataclass
class JobWorkItem:
    job_id: str
    request: GenerateRequest
    model_spec: Any
    runtime: Any


class JobQueue:
    def __init__(self) -> None:
        self._queue: "queue.Queue[JobWorkItem]" = queue.Queue()

    def put(self, item: JobWorkItem) -> None:
        self._queue.put(item)

    def get(self, timeout: float | None = None) -> JobWorkItem:
        return self._queue.get(timeout=timeout)

    def collect_matching(
        self,
        first_item: JobWorkItem,
        *,
        max_items: int,
        wait_window_ms: int,
        matcher: Callable[[JobWorkItem, JobWorkItem], bool],
    ) -> list[JobWorkItem]:
        if max_items <= 1 or wait_window_ms <= 0:
            return [first_item]

        matched = [first_item]
        deferred: list[JobWorkItem] = []
        deadline = time.monotonic() + (float(wait_window_ms) / 1000.0)

        while len(matched) < max_items:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                candidate = self.get(timeout=remaining)
            except queue.Empty:
                break
            if matcher(first_item, candidate):
                matched.append(candidate)
            else:
                deferred.append(candidate)

        for item in deferred:
            self.task_done()
            self.put(item)
        return matched

    def task_done(self) -> None:
        self._queue.task_done()
