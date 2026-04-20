"""Executor cache keyed by model/backend configuration."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import threading
from typing import Callable, Dict, Generic, Hashable, TypeVar

T = TypeVar("T")


@dataclass
class CachedExecutor(Generic[T]):
    value: T
    lock: threading.Lock = field(default_factory=threading.Lock)


class PipelineCache:
    def __init__(self, *, max_size: int = 4) -> None:
        self.max_size = max(int(max_size), 1)
        self._entries: "OrderedDict[Hashable, CachedExecutor]" = OrderedDict()
        self._lock = threading.RLock()

    def get_or_create(self, key: Hashable, factory: Callable[[], T]) -> CachedExecutor[T]:
        evicted = None
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                return entry
            entry = CachedExecutor(factory())
            self._entries[key] = entry
            if len(self._entries) > self.max_size:
                _old_key, evicted = self._entries.popitem(last=False)
        if evicted is not None:
            release = getattr(evicted.value, "release", None)
            if callable(release):
                release()
        return entry

    def snapshot_keys(self) -> list[Hashable]:
        with self._lock:
            return list(self._entries.keys())
