"""Background worker thread used by the in-process engine."""

from __future__ import annotations

import queue
import threading
from typing import Callable, Sequence

from omnirt.dispatch.batcher import RequestBatcher
from omnirt.dispatch.queue import JobQueue, JobWorkItem


class Worker(threading.Thread):
    def __init__(
        self,
        *,
        name: str,
        job_queue: JobQueue,
        handler: Callable[[Sequence[JobWorkItem]], None],
        batcher: RequestBatcher | None = None,
    ) -> None:
        super().__init__(name=name, daemon=True)
        self._job_queue = job_queue
        self._handler = handler
        self._batcher = batcher
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
                batch = [item]
                if self._batcher is not None and self._batcher.enabled:
                    batch = self._job_queue.collect_matching(
                        item,
                        max_items=self._batcher.max_batch_size,
                        wait_window_ms=self._batcher.batch_window_ms,
                        matcher=self._batcher.matches,
                    )
                self._handler(batch)
            finally:
                for _ in batch:
                    self._job_queue.task_done()
