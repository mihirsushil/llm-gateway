import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Callable

from .cache import TTLCache
from .circuit_breaker import CircuitBreaker, CircuitOpen
from .input_guard import InputGuard
from .models import ChatCompletionRequest, ChatCompletionResponse
from .output_guard import OutputGuard, OutputRejected
from .provider import ChatProvider, PermanentProviderError, TransientProviderError
from .rate_limit import TokenBucketRateLimiter


class UpstreamUnavailable(Exception):
    pass


@dataclass(frozen=True)
class CompletionResult:
    response: ChatCompletionResponse
    cache_hit: bool
    attempts: int


class GatewayService:
    def __init__(
        self,
        provider: ChatProvider,
        limiter: TokenBucketRateLimiter,
        cache: TTLCache[ChatCompletionResponse],
        circuit_breaker: CircuitBreaker,
        input_guard: InputGuard,
        output_guard: OutputGuard,
        retry_attempts: int,
        retry_base_seconds: float,
        sleeper: Callable[[float], None] = time.sleep,
        logger: logging.Logger | None = None,
    ):
        self.provider = provider
        self.limiter = limiter
        self.cache = cache
        self.circuit_breaker = circuit_breaker
        self.input_guard = input_guard
        self.output_guard = output_guard
        self.retry_attempts = retry_attempts
        self.retry_base_seconds = retry_base_seconds
        self.sleeper = sleeper
        self.logger = logger or logging.getLogger("llm_gateway")

    @staticmethod
    def _cache_key(request: ChatCompletionRequest) -> str:
        canonical_request = json.dumps(
            request.model_dump(exclude_none=True), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical_request.encode()).hexdigest()

    def complete(self, api_key: str, request: ChatCompletionRequest) -> CompletionResult:
        self.input_guard.validate(request)
        self.limiter.consume(api_key)
        cache_key = self._cache_key(request)
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.logger.info(json.dumps({"event": "completion", "cache_hit": True, "attempts": 0}))
            return CompletionResult(response=cached, cache_hit=True, attempts=0)

        try:
            self.circuit_breaker.allow_request()
        except CircuitOpen as error:
            self.logger.warning(json.dumps({"event": "circuit_open"}))
            raise UpstreamUnavailable("upstream circuit is open; retry later") from error

        for attempt in range(1, self.retry_attempts + 1):
            try:
                response = self.provider.complete(request)
                try:
                    self.output_guard.validate(response)
                except OutputRejected as error:
                    self.circuit_breaker.record_failure()
                    self.logger.warning(json.dumps({"event": "output_rejected", "code": error.reason}))
                    raise
                self.circuit_breaker.record_success()
                self.cache.set(cache_key, response)
                self.logger.info(json.dumps({"event": "completion", "cache_hit": False, "attempts": attempt}))
                return CompletionResult(response=response, cache_hit=False, attempts=attempt)
            except PermanentProviderError as error:
                self.circuit_breaker.record_failure()
                self.logger.warning(json.dumps({"event": "provider_error", "retryable": False}))
                raise UpstreamUnavailable(str(error)) from error
            except TransientProviderError as error:
                if attempt == self.retry_attempts:
                    self.circuit_breaker.record_failure()
                    self.logger.warning(json.dumps({"event": "provider_error", "retryable": True, "attempts": attempt}))
                    raise UpstreamUnavailable("upstream failed after retry budget") from error
                self.sleeper(self.retry_base_seconds * (2 ** (attempt - 1)))

        raise AssertionError("retry loop must return or raise")
