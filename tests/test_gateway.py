from collections.abc import Iterable

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models import ChatCompletionRequest, ChatCompletionResponse, ChatMessage, CompletionChoice
from app.provider import TransientProviderError


class FakeProvider:
    def __init__(self, outcomes: Iterable[object]):
        self.outcomes = iter(outcomes)
        self.calls = 0

    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self.calls += 1
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def completion(content="hello") -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="cmpl_test",
        created=1,
        model="test-model",
        choices=[CompletionChoice(index=0, message=ChatMessage(role="assistant", content=content))],
    )


def app_with(provider, **overrides):
    settings = Settings(
        api_keys=frozenset({"test-key"}),
        upstream_base_url=None,
        upstream_api_key=None,
        upstream_model=None,
        **overrides,
    )
    return create_app(settings=settings, provider=provider)


def request_body(content="hello"):
    return {"model": "test-model", "messages": [{"role": "user", "content": content}]}


def test_requires_a_valid_api_key():
    client = TestClient(app_with(FakeProvider([completion()])))
    assert client.post("/v1/chat/completions", json=request_body()).status_code == 401
    assert client.post("/v1/chat/completions", headers={"X-API-Key": "wrong"}, json=request_body()).status_code == 401


def test_cache_returns_same_completion_without_a_second_provider_call():
    provider = FakeProvider([completion()])
    client = TestClient(app_with(provider))
    headers = {"X-API-Key": "test-key"}

    first = client.post("/v1/chat/completions", headers=headers, json=request_body())
    second = client.post("/v1/chat/completions", headers=headers, json=request_body())

    assert first.status_code == second.status_code == 200
    assert first.headers["X-Cache"] == "MISS"
    assert second.headers["X-Cache"] == "HIT"
    assert provider.calls == 1
    assert first.json() == second.json()


def test_retries_transient_failure_then_returns_completion():
    provider = FakeProvider([TransientProviderError("temporary"), completion("recovered")])
    app = app_with(provider, retry_base_seconds=0.0)
    client = TestClient(app)

    result = client.post("/v1/chat/completions", headers={"X-API-Key": "test-key"}, json=request_body())

    assert result.status_code == 200
    assert result.headers["X-Upstream-Attempts"] == "2"
    assert result.json()["choices"][0]["message"]["content"] == "recovered"
    assert provider.calls == 2


def test_exhausted_retry_budget_returns_503():
    provider = FakeProvider([TransientProviderError("temporary")] * 3)
    client = TestClient(app_with(provider, retry_base_seconds=0.0))

    result = client.post("/v1/chat/completions", headers={"X-API-Key": "test-key"}, json=request_body())

    assert result.status_code == 503
    assert provider.calls == 3


def test_token_bucket_allows_burst_then_rejects_requests():
    provider = FakeProvider([completion("one"), completion("two")])
    client = TestClient(app_with(provider, rate_limit_per_minute=1, rate_limit_burst=2))
    headers = {"X-API-Key": "test-key"}

    assert client.post("/v1/chat/completions", headers=headers, json=request_body("one")).status_code == 200
    assert client.post("/v1/chat/completions", headers=headers, json=request_body("two")).status_code == 200
    rejected = client.post("/v1/chat/completions", headers=headers, json=request_body("three"))

    assert rejected.status_code == 429
    assert rejected.headers["Retry-After"] == "60"
    assert provider.calls == 2
