"""Small in-process load test: shows the rate limiter and cache actually
changing behavior under repeated/bursty requests, instead of asserting it in a
unit test. No live server or upstream needed - runs the FastAPI app in-process
against a fake, always-succeeding upstream.

Two scenarios, kept separate because they'd otherwise race each other:
  1. Cache: the same handful of prompts, requested sequentially and repeatedly
     - later requests should hit the cache instead of calling the upstream.
  2. Rate limit: many concurrent *distinct* prompts fired at once, well beyond
     the configured burst - later ones should be rejected with 429.

Run: .venv/bin/python scripts/load_test.py
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter

import httpx

from app.config import Settings
from app.main import create_app
from app.models import ChatCompletionRequest, ChatCompletionResponse, ChatMessage, CompletionChoice

API_KEY = "load-test-key"


class SlowFakeProvider:
    """Always succeeds, with a small artificial delay so requests overlap."""

    def __init__(self, latency_seconds: float = 0.05):
        self.latency_seconds = latency_seconds
        self.calls = 0

    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self.calls += 1
        time.sleep(self.latency_seconds)
        return ChatCompletionResponse(
            id=f"cmpl_{self.calls}",
            created=1,
            model="test-model",
            choices=[CompletionChoice(index=0, message=ChatMessage(role="assistant", content="ok"))],
        )


def build_app(**settings_overrides):
    settings = Settings(
        api_keys=frozenset({API_KEY}),
        upstream_base_url=None,
        upstream_api_key=None,
        upstream_model=None,
        **settings_overrides,
    )
    provider = SlowFakeProvider()
    return create_app(settings=settings, provider=provider), provider


async def _fire(app, prompt: str) -> tuple[int, str | None]:
    body = {"model": "test-model", "messages": [{"role": "user", "content": prompt}]}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway.test") as client:
        response = await client.post("/v1/chat/completions", json=body, headers={"X-API-Key": API_KEY})
    return response.status_code, response.headers.get("X-Cache")


async def run_cache_scenario() -> None:
    print("=== Scenario 1: cache under repeated requests ===")
    unique_prompts = 5
    repeats = 4
    # Rate limit high enough that this scenario is only about the cache.
    app, provider = build_app(rate_limit_per_minute=6_000, rate_limit_burst=1_000, cache_ttl_seconds=30.0)

    results = []
    for round_ in range(repeats):
        for i in range(unique_prompts):
            results.append(await _fire(app, f"prompt-{i}"))

    total = len(results)
    cache_flags = Counter(flag for _, flag in results)
    print(f"fired {total} sequential requests over {unique_prompts} unique prompts x {repeats} repeats")
    print(f"cache: {dict(cache_flags)}")
    print(f"actual upstream calls made: {provider.calls} (expect {unique_prompts}: one per unique prompt)")
    print()


async def run_rate_limit_scenario() -> None:
    print("=== Scenario 2: rate limiter under a concurrent burst ===")
    total_requests = 40
    concurrency = 20
    burst = 10
    # Distinct prompts so the cache can't mask the rate limiter's behavior.
    app, provider = build_app(rate_limit_per_minute=30, rate_limit_burst=burst, cache_ttl_seconds=30.0)

    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(i: int):
        async with semaphore:
            return await _fire(app, f"distinct-prompt-{i}")

    started = time.monotonic()
    results = await asyncio.gather(*(bounded(i) for i in range(total_requests)))
    elapsed = time.monotonic() - started

    statuses = Counter(status for status, _ in results)
    print(f"fired {total_requests} concurrent requests (concurrency={concurrency}) against burst={burst}")
    print(f"elapsed: {elapsed:.2f}s")
    print(f"status codes: {dict(statuses)}")
    print(f"actual upstream calls made: {provider.calls} (expect == burst == {burst})")
    print(f"rate-limited (429): {statuses.get(429, 0)} (expect == {total_requests - burst})")


async def run() -> None:
    await run_cache_scenario()
    await run_rate_limit_scenario()


if __name__ == "__main__":
    asyncio.run(run())
