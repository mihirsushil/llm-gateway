import itertools
import json
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Callable, Literal

from .circuit_breaker import CircuitBreaker, CircuitOpen
from .models import ChatCompletionRequest, ChatCompletionResponse
from .provider import ChatProvider, ClientContractError, PermanentProviderError, TransientProviderError


class UpstreamUnavailable(Exception):
    pass


@dataclass(frozen=True)
class ProviderResult:
    response: ChatCompletionResponse
    attempts: int
    source: Literal["primary", "fallback"]


class ResilientProvider:
    """The single place retry, circuit-breaking, and fallback policy live.

    Policy (deliberately explicit, not incidental):
    - The circuit breaker gates the primary only. If it's open, primary
      attempts are skipped entirely and the fallback (if configured) gets
      its one attempt immediately — failing fast on a known-bad primary
      shouldn't also block a working fallback.
    - The primary is retried, with exponential backoff, up to
      `retry_attempts` times, but only on `TransientProviderError`. A
      `PermanentProviderError` stops the primary's retries immediately —
      retrying bad auth or a malformed-response bug against the same
      provider/key can't succeed.
    - `ClientContractError` (malformed request — 400/404/422) is never
      retried, never handed to the fallback (an identical request fails
      identically anywhere), and never recorded against the circuit
      breaker — it isn't a provider-health signal.
    - Once the primary's retry budget (or a permanent error) ends its
      attempt, the fallback — if configured — gets exactly one attempt.
      Fallback failures are not themselves retried or re-escalated.
    - The breaker only ever sees `record_failure()` here; `record_success()`
      is the caller's call once the response has also cleared the output
      guard, matching this gateway's "a bad completion is a circuit-breaker
      failure too" policy. Critically, the caller should only do that when
      `ProviderResult.source == "primary"` — a successful *fallback*
      response says nothing about the primary's health, and resetting the
      primary's breaker because the fallback bailed it out would send the
      very next request straight back at a still-broken primary.
    """

    def __init__(
        self,
        primary: ChatProvider,
        circuit_breaker: CircuitBreaker,
        retry_attempts: int,
        retry_base_seconds: float,
        fallback: ChatProvider | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        logger: logging.Logger | None = None,
    ):
        self.primary = primary
        self.fallback = fallback
        self.circuit_breaker = circuit_breaker
        self.retry_attempts = retry_attempts
        self.retry_base_seconds = retry_base_seconds
        self.sleeper = sleeper
        self.logger = logger or logging.getLogger("llm_gateway")

    def complete(self, request: ChatCompletionRequest) -> ProviderResult:
        primary_error: Exception | None = None
        attempts = 0

        try:
            self.circuit_breaker.allow_request()
            for attempt in range(1, self.retry_attempts + 1):
                attempts = attempt
                try:
                    return ProviderResult(response=self.primary.complete(request), attempts=attempts, source="primary")
                except PermanentProviderError as error:
                    self.circuit_breaker.record_failure()
                    self.logger.warning(json.dumps({"event": "provider_error", "retryable": False}))
                    primary_error = error
                    break
                except TransientProviderError as error:
                    primary_error = error
                    if attempt == self.retry_attempts:
                        self.circuit_breaker.record_failure()
                        self.logger.warning(
                            json.dumps({"event": "provider_error", "retryable": True, "attempts": attempt})
                        )
                        break
                    self.sleeper(self.retry_base_seconds * (2 ** (attempt - 1)))
        except CircuitOpen as error:
            self.logger.warning(json.dumps({"event": "circuit_open"}))
            primary_error = error

        if self.fallback is not None:
            self.logger.warning(json.dumps({"event": "provider_fallback", "reason": str(primary_error)}))
            try:
                return ProviderResult(
                    response=self.fallback.complete(request), attempts=attempts + 1, source="fallback"
                )
            except (TransientProviderError, PermanentProviderError) as error:
                raise UpstreamUnavailable(str(error)) from error

        raise UpstreamUnavailable(str(primary_error)) from primary_error

    def stream(self, request: ChatCompletionRequest) -> Iterator[str]:
        """Same fallback policy as `complete()`, minus retries: a stream
        that has already forwarded bytes downstream can't be replayed, so
        only the connection/initial-response failure is eligible for
        fallback, and there is no retry budget on the primary."""
        try:
            self.circuit_breaker.allow_request()
            generator = self.primary.stream(request)
            first_line = next(generator)
            return itertools.chain([first_line], generator)
        except CircuitOpen as error:
            self.logger.warning(json.dumps({"event": "circuit_open"}))
            primary_error: Exception = error
        except ClientContractError:
            raise
        except (TransientProviderError, PermanentProviderError) as error:
            self.circuit_breaker.record_failure()
            primary_error = error

        if self.fallback is not None:
            self.logger.warning(json.dumps({"event": "stream_fallback", "reason": str(primary_error)}))
            try:
                generator = self.fallback.stream(request)
                first_line = next(generator)
                return itertools.chain([first_line], generator)
            except (TransientProviderError, PermanentProviderError) as error:
                raise UpstreamUnavailable(str(error)) from error

        raise UpstreamUnavailable(str(primary_error)) from primary_error
