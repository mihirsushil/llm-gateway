import time
from dataclasses import dataclass
from threading import Lock
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass
class _CacheEntry(Generic[T]):
    value: T
    expires_at: float


class TTLCache(Generic[T]):
    """Small, process-local cache. A production multi-instance deployment
    would replace this with Redis while preserving the same interface."""

    def __init__(self, ttl_seconds: float, clock=time.monotonic):
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self._entries: dict[str, _CacheEntry[T]] = {}
        self._lock = Lock()

    def get(self, key: str) -> T | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at <= self.clock():
                del self._entries[key]
                return None
            return entry.value

    def set(self, key: str, value: T) -> None:
        with self._lock:
            self._entries[key] = _CacheEntry(value=value, expires_at=self.clock() + self.ttl_seconds)
