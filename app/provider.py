from collections.abc import Iterator
from typing import Protocol

import httpx

from .models import ChatCompletionRequest, ChatCompletionResponse


class TransientProviderError(Exception):
    """An upstream failure that can be safely retried or handed to a fallback
    provider: a timeout, a network error, 429, or 5xx."""


class PermanentProviderError(Exception):
    """An upstream failure that is permanent for this provider/key but not
    necessarily for a different one (bad auth, exhausted quota, a malformed
    response body). Not retried against the same provider, but still worth
    a fallback attempt."""


class ClientContractError(Exception):
    """The request itself is invalid (400/404/422/...) — an identical
    request will fail identically against any provider or key. Never
    retried, never handed to a fallback, and never counted as a
    provider-health failure against the circuit breaker."""


class ChatProvider(Protocol):
    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse: ...

    def stream(self, request: ChatCompletionRequest) -> Iterator[str]: ...


def _classify_status(status_code: int) -> None:
    """Raises the appropriate error class for a non-2xx upstream response.
    Returns normally for a successful status."""
    if status_code == 429 or status_code >= 500:
        raise TransientProviderError(f"upstream returned {status_code}")
    if status_code in (401, 403):
        raise PermanentProviderError(f"upstream returned {status_code}")
    if status_code >= 400:
        raise ClientContractError(f"upstream rejected the request: {status_code}")


class UnconfiguredProvider:
    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        raise PermanentProviderError("no upstream provider is configured")

    def stream(self, request: ChatCompletionRequest) -> Iterator[str]:
        raise PermanentProviderError("no upstream provider is configured")
        yield  # pragma: no cover - makes this a generator function


class OpenAICompatibleProvider:
    def __init__(self, base_url: str, api_key: str, default_model: str | None, timeout_seconds: float):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_model = default_model
        self.timeout_seconds = timeout_seconds

    def _payload(self, request: ChatCompletionRequest) -> dict:
        payload = request.model_dump(exclude_none=True)
        payload["model"] = request.model or self.default_model
        if not payload["model"]:
            raise PermanentProviderError("a model must be supplied in the request or UPSTREAM_MODEL")
        return payload

    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        payload = self._payload(request)
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=self.timeout_seconds,
            )
        except httpx.TimeoutException as error:
            raise TransientProviderError("upstream request timed out") from error
        except httpx.RequestError as error:
            raise TransientProviderError("upstream network request failed") from error

        _classify_status(response.status_code)

        try:
            return ChatCompletionResponse.model_validate(response.json())
        except ValueError as error:
            raise PermanentProviderError("upstream returned an invalid completion payload") from error

    def stream(self, request: ChatCompletionRequest) -> Iterator[str]:
        """Yields raw SSE lines from the upstream. The connection and the
        initial response status are validated (and classified exactly like
        `complete()`) before the first line is yielded, so a caller can still
        retry or fall back up to that point. Once this generator has yielded
        once, the caller owns any further failure — a partially-forwarded
        stream can't be replayed without duplicating content already sent
        downstream.
        """
        payload = self._payload(request)
        payload["stream"] = True
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=self.timeout_seconds,
            ) as response:
                _classify_status(response.status_code)
                yield from response.iter_lines()
        except httpx.TimeoutException as error:
            raise TransientProviderError("upstream request timed out") from error
        except httpx.RequestError as error:
            raise TransientProviderError("upstream network request failed") from error
