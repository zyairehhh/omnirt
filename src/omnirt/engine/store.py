"""In-memory job store with event subscriptions."""

from __future__ import annotations

from copy import deepcopy
import queue
import threading
from typing import Dict, List, Optional

from omnirt.core.types import StageEventRecord
from omnirt.engine.job import JobRecord


class InMemoryJobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, JobRecord] = {}
        self._subscribers: Dict[str, List["queue.Queue[StageEventRecord | None]"]] = {}
        self._lock = threading.RLock()

    def create(self, job: JobRecord) -> JobRecord:
        with self._lock:
            self._jobs[job.id] = deepcopy(job)
            self._subscribers.setdefault(job.id, [])
            return deepcopy(self._jobs[job.id])

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            job = self._jobs.get(job_id)
            return deepcopy(job) if job is not None else None

    def save(self, job: JobRecord) -> JobRecord:
        with self._lock:
            self._jobs[job.id] = deepcopy(job)
            return deepcopy(self._jobs[job.id])

    def append_event(self, job_id: str, event: StageEventRecord) -> StageEventRecord:
        with self._lock:
            job = self._jobs[job_id]
            job.events.append(deepcopy(event))
            for subscriber in list(self._subscribers.get(job_id, [])):
                subscriber.put(deepcopy(event))
            return deepcopy(event)

    def subscribe(self, job_id: str) -> "queue.Queue[StageEventRecord | None]":
        channel: "queue.Queue[StageEventRecord | None]" = queue.Queue()
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            self._subscribers.setdefault(job_id, []).append(channel)
        return channel

    def unsubscribe(self, job_id: str, channel: "queue.Queue[StageEventRecord | None]") -> None:
        with self._lock:
            subscribers = self._subscribers.get(job_id, [])
            if channel in subscribers:
                subscribers.remove(channel)
            channel.put(None)
