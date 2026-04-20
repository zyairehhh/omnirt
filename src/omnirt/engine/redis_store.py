"""Redis-backed job store scaffold."""

from __future__ import annotations

import importlib

from omnirt.core.types import DependencyUnavailableError
from omnirt.engine.store import InMemoryJobStore


class RedisJobStore(InMemoryJobStore):
    def __init__(self, *, redis_url: str = "redis://127.0.0.1:6379/0") -> None:
        super().__init__()
        try:
            redis_module = importlib.import_module("redis")
        except ImportError as exc:
            raise DependencyUnavailableError(
                "redis package is required to use RedisJobStore."
            ) from exc
        self.redis_url = redis_url
        self.client = redis_module.Redis.from_url(redis_url)
