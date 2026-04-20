from __future__ import annotations

import queue
import types

import pytest

from omnirt.core.types import DependencyUnavailableError, GenerateRequest, StageEventRecord
from omnirt.engine import redis_store
from omnirt.engine.job import JobRecord
from omnirt.engine.redis_store import RedisJobStore


def test_redis_store_requires_optional_dependency(monkeypatch) -> None:
    def fake_import(name: str):
        if name == "redis":
            raise ImportError("missing redis")
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr(redis_store.importlib, "import_module", fake_import)

    with pytest.raises(DependencyUnavailableError):
        RedisJobStore()


def test_redis_store_builds_client_from_url(monkeypatch) -> None:
    captured = {}

    class FakePubSub:
        def subscribe(self, *_args, **_kwargs):
            return None

        def get_message(self, **_kwargs):
            return None

        def unsubscribe(self, *_args, **_kwargs):
            return None

        def close(self):
            return None

    class FakeRedisClient:
        def __init__(self) -> None:
            self.storage = {}

        @staticmethod
        def from_url(url: str):
            captured["url"] = url
            return FakeRedisClient()

        def set(self, key, value):
            self.storage[key] = value

        def get(self, key):
            return self.storage.get(key)

        def publish(self, *_args, **_kwargs):
            return None

        def pubsub(self):
            return FakePubSub()

    monkeypatch.setattr(
        redis_store.importlib,
        "import_module",
        lambda name: types.SimpleNamespace(Redis=FakeRedisClient) if name == "redis" else None,
    )

    store = RedisJobStore(redis_url="redis://cache:6379/9")

    assert captured["url"] == "redis://cache:6379/9"
    assert store.client is not None


def test_redis_store_persists_jobs_and_events(monkeypatch) -> None:
    published = []

    class FakePubSub:
        def __init__(self) -> None:
            self.messages = queue.Queue()

        def subscribe(self, *_args, **_kwargs):
            return None

        def get_message(self, **_kwargs):
            try:
                return self.messages.get_nowait()
            except queue.Empty:
                return None

        def unsubscribe(self, *_args, **_kwargs):
            return None

        def close(self):
            return None

    class FakeRedisClient:
        def __init__(self) -> None:
            self.storage = {}
            self.pubsubs = []

        @staticmethod
        def from_url(_url: str):
            return fake_client

        def set(self, key, value):
            self.storage[key] = value

        def get(self, key):
            return self.storage.get(key)

        def publish(self, channel, payload):
            published.append((channel, payload))
            for pubsub in self.pubsubs:
                pubsub.messages.put({"type": "message", "data": payload})

        def pubsub(self):
            pubsub = FakePubSub()
            self.pubsubs.append(pubsub)
            return pubsub

    fake_client = FakeRedisClient()
    monkeypatch.setattr(
        redis_store.importlib,
        "import_module",
        lambda name: types.SimpleNamespace(Redis=FakeRedisClient) if name == "redis" else None,
    )

    store = RedisJobStore(redis_url="redis://cache:6379/9")
    job = JobRecord(
        id="job-1",
        request=GenerateRequest(task="text2image", model="dummy", inputs={"prompt": "hello"}),
        backend="cpu-stub",
    )

    store.create(job)
    assert store.get("job-1") is not None

    channel = store.subscribe("job-1")
    event = StageEventRecord(event="job_started", stage="job", timestamp_ms=1, data={"job_id": "job-1"})
    store.append_event("job-1", event)
    received = channel.get(timeout=1.0)

    assert received.event == "job_started"
    assert published
    reloaded = store.get("job-1")
    assert reloaded is not None
    assert reloaded.events[-1].event == "job_started"
    store.unsubscribe("job-1", channel)
