"""Background worker thread used by the in-process engine."""

from __future__ import annotations

import queue
import threading
from typing import Callable

from omnirt.dispatch.queue import JobQueue, JobWorkItem


class Worker(threading.Thread):
    def __init__(self, *, name: str, job_queue: JobQueue, handler: Callable[[JobWorkItem], None]) -> None:
        super().__init__(name=name, daemon=True)
        self._job_queue = job_queue
        self._handler = handler
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._job_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._handler(item)
            finally:
                self._job_queue.task_done()
