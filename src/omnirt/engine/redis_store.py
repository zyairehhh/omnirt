"""Redis-backed job store with pubsub-based event subscriptions."""

from __future__ import annotations

import importlib
import json
import queue
import threading
from typing import Dict, Optional

from omnirt.core.types import DependencyUnavailableError, StageEventRecord
from omnirt.engine.job import JobRecord
from omnirt.engine.store import InMemoryJobStore


class RedisJobStore(InMemoryJobStore):
    def __init__(self, *, redis_url: str = "redis://127.0.0.1:6379/0", namespace: str = "omnirt") -> None:
        super().__init__()
        try:
            redis_module = importlib.import_module("redis")
        except ImportError as exc:
            raise DependencyUnavailableError("redis package is required to use RedisJobStore.") from exc
        self.redis_url = redis_url
        self.namespace = namespace
        self.client = redis_module.Redis.from_url(redis_url)
        self._pubsubs: Dict[int, tuple[object, threading.Event, threading.Thread]] = {}

    def create(self, job: JobRecord) -> JobRecord:
        created = super().create(job)
        self._persist(created)
        return created

    def get(self, job_id: str) -> Optional[JobRecord]:
        remote = self.client.get(self._job_key(job_id))
        if remote is not None:
            payload = self._decode(remote)
            job = JobRecord.from_dict(payload)
            super().save(job)
            return job
        return super().get(job_id)

    def save(self, job: JobRecord) -> JobRecord:
        saved = super().save(job)
        self._persist(saved)
        return saved

    def append_event(self, job_id: str, event: StageEventRecord) -> StageEventRecord:
        job = super().get(job_id)
        if job is None:
            raise KeyError(job_id)
        job.events.append(event)
        super().save(job)
        saved_event = StageEventRecord.from_dict(event.__dict__)
        self._persist(job)
        self.client.publish(self._channel_name(job_id), self._encode(saved_event.__dict__))
        return saved_event

    def subscribe(self, job_id: str) -> "queue.Queue[StageEventRecord | None]":
        if self.get(job_id) is None:
            raise KeyError(job_id)
        channel: "queue.Queue[StageEventRecord | None]" = queue.Queue()
        pubsub = self.client.pubsub()
        pubsub.subscribe(self._channel_name(job_id))
        stop_event = threading.Event()

        def pump() -> None:
            while not stop_event.is_set():
                try:
                    message = pubsub.get_message(ignore_subscribe_messages=True, timeout=0.2)
                except Exception:
                    if stop_event.is_set():
                        break
                    continue
                if not message or message.get("type") != "message":
                    continue
                payload = message.get("data")
                if payload is None:
                    continue
                channel.put(StageEventRecord.from_dict(self._decode(payload)))

        thread = threading.Thread(target=pump, daemon=True, name=f"omnirt-redis-sub-{job_id[:8]}")
        thread.start()
        self._pubsubs[id(channel)] = (pubsub, stop_event, thread)
        return channel

    def unsubscribe(self, job_id: str, channel: "queue.Queue[StageEventRecord | None]") -> None:
        subscription = self._pubsubs.pop(id(channel), None)
        if subscription is not None:
            pubsub, stop_event, thread = subscription
            stop_event.set()
            try:
                pubsub.unsubscribe(self._channel_name(job_id))
            except Exception:
                pass
            close = getattr(pubsub, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            thread.join(timeout=0.5)
        channel.put(None)

    def _persist(self, job: JobRecord) -> None:
        self.client.set(self._job_key(job.id), self._encode(job.to_dict()))

    def _job_key(self, job_id: str) -> str:
        return f"{self.namespace}:job:{job_id}"

    def _channel_name(self, job_id: str) -> str:
        return f"{self.namespace}:job:{job_id}:events"

    def _encode(self, payload: dict) -> str:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _decode(self, payload) -> dict:
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        if isinstance(payload, str):
            return json.loads(payload)
        return dict(payload)
