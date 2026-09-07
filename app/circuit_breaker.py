import time
from threading import Lock


class CircuitOpen(Exception):
    def __init__(self, retry_after_seconds: float):
        self.retry_after_seconds = retry_after_seconds
        super().__init__("circuit breaker is open")


class CircuitBreaker:
    """Fail fast after repeated upstream failures, then allow exactly one
    half-open probe after the recovery timeout. This prevents a sick provider
    from consuming every request's retry budget during an outage."""

    def __init__(self, failure_threshold: int, recovery_timeout_seconds: float, clock=time.monotonic):
        if failure_threshold <= 0 or recovery_timeout_seconds < 0:
            raise ValueError("invalid circuit-breaker configuration")
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.clock = clock
        self._state = "closed"
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = Lock()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def allow_request(self) -> None:
        now = self.clock()
        with self._lock:
            if self._state == "closed":
                return
            if self._state == "half_open":
                raise CircuitOpen(self.recovery_timeout_seconds)

            assert self._opened_at is not None
            elapsed = now - self._opened_at
            if elapsed < self.recovery_timeout_seconds:
                raise CircuitOpen(self.recovery_timeout_seconds - elapsed)
            self._state = "half_open"

    def record_success(self) -> None:
        with self._lock:
            self._state = "closed"
            self._failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state == "half_open" or self._failures >= self.failure_threshold:
                self._state = "open"
                self._opened_at = self.clock()
