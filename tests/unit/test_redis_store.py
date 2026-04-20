from __future__ import annotations

import types

import pytest

from omnirt.core.types import DependencyUnavailableError
from omnirt.engine import redis_store
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

    class FakeRedisClient:
        @staticmethod
        def from_url(url: str):
            captured["url"] = url
            return object()

    monkeypatch.setattr(
        redis_store.importlib,
        "import_module",
        lambda name: types.SimpleNamespace(Redis=FakeRedisClient) if name == "redis" else None,
    )

    store = RedisJobStore(redis_url="redis://cache:6379/9")

    assert captured["url"] == "redis://cache:6379/9"
    assert store.client is not None
