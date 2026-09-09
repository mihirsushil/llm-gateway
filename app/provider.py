import json
import logging
import time
from typing import Protocol

import httpx

from .models import ChatCompletionRequest, ChatCompletionResponse


class TransientProviderError(Exception):
    """An upstream failure that can be safely retried."""


class PermanentProviderError(Exception):
    """An upstream failure that retries cannot fix (for example, bad auth)."""


class ChatProvider(Protocol):
    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse: ...


class UnconfiguredProvider:
    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        raise PermanentProviderError("no upstream provider is configured")


class OpenAICompatibleProvider:
    def __init__(self, base_url: str, api_key: str, default_model: str | None, timeout_seconds: float):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_model = default_model
        self.timeout_seconds = timeout_seconds

    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        payload = request.model_dump(exclude_none=True)
        payload["model"] = request.model or self.default_model
        if not payload["model"]:
            raise PermanentProviderError("a model must be supplied in the request or UPSTREAM_MODEL")

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

        if response.status_code == 429 or response.status_code >= 500:
            raise TransientProviderError(f"upstream returned {response.status_code}")
        if response.is_error:
            raise PermanentProviderError(f"upstream returned {response.status_code}")

        try:
            return ChatCompletionResponse.model_validate(response.json())
        except ValueError as error:
            raise PermanentProviderError("upstream returned an invalid completion payload") from error


class FallbackProvider:
    """Tries a secondary provider (a second upstream, or the same upstream on
    a second API key) when the primary fails. A single provider/key outage
    shouldn't fail every request if a working alternative is configured.

    Falls back on both transient and permanent primary errors: a permanent
    error (bad auth, exhausted quota) is permanent *for that key*, not
    necessarily for a different provider or key.
    """

    def __init__(self, primary: ChatProvider, secondary: ChatProvider, logger: logging.Logger | None = None):
        self.primary = primary
        self.secondary = secondary
        self.logger = logger or logging.getLogger("llm_gateway")

    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        try:
            return self.primary.complete(request)
        except (TransientProviderError, PermanentProviderError) as error:
            self.logger.warning(json.dumps({"event": "provider_fallback", "reason": str(error)}))
            return self.secondary.complete(request)
