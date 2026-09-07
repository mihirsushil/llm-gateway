import time
from dataclasses import dataclass
from threading import Lock


class RateLimitExceeded(Exception):
    def __init__(self, retry_after_seconds: float):
        self.retry_after_seconds = retry_after_seconds
        super().__init__("rate limit exceeded")


@dataclass
class _Bucket:
    tokens: float
    updated_at: float


class TokenBucketRateLimiter:
    """Per-key token bucket. Tokens refill continuously, which allows brief
    bursts while enforcing a stable long-term request rate."""

    def __init__(self, tokens_per_minute: int, burst: int, clock=time.monotonic):
        if tokens_per_minute <= 0 or burst <= 0:
            raise ValueError("rate-limit values must be positive")
        self.refill_rate = tokens_per_minute / 60.0
        self.burst = float(burst)
        self.clock = clock
        self._buckets: dict[str, _Bucket] = {}
        self._lock = Lock()

    def consume(self, key: str, cost: float = 1.0) -> None:
        now = self.clock()
        with self._lock:
            bucket = self._buckets.setdefault(key, _Bucket(tokens=self.burst, updated_at=now))
            elapsed = max(0.0, now - bucket.updated_at)
            bucket.tokens = min(self.burst, bucket.tokens + elapsed * self.refill_rate)
            bucket.updated_at = now
            if bucket.tokens < cost:
                raise RateLimitExceeded((cost - bucket.tokens) / self.refill_rate)
            bucket.tokens -= cost
