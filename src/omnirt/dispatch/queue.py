"""Simple in-process job queue."""

from __future__ import annotations

from dataclasses import dataclass
import queue
from typing import Any


@dataclass
class JobWorkItem:
    job_id: str
    model_spec: Any
    runtime: Any


class JobQueue:
    def __init__(self) -> None:
        self._queue: "queue.Queue[JobWorkItem]" = queue.Queue()

    def put(self, item: JobWorkItem) -> None:
        self._queue.put(item)

    def get(self, timeout: float | None = None) -> JobWorkItem:
        return self._queue.get(timeout=timeout)

    def task_done(self) -> None:
        self._queue.task_done()
