# LLM Gateway

An OpenAI-compatible proxy that demonstrates the reliability controls an LLM
application needs between a client and an upstream provider. The first phase
includes per-key token-bucket rate limiting, exact-match TTL caching, timeout
and transient-error retries, circuit breaking, and structured JSON request logs.

## Architecture

```
OpenAI-compatible client
        ↓
POST /v1/chat/completions
        ↓
API-key validation → token bucket → exact cache → circuit breaker → retrying provider client
        ↓                                                        ↓
structured JSON logs                                       OpenAI-compatible upstream
```

## Run locally

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
GATEWAY_API_KEYS=demo-key .venv/bin/python -m uvicorn app.main:app --reload
```

The first command creates an isolated Python environment, the second installs
the service and test tools, and the third starts the gateway with `demo-key`.
Set `UPSTREAM_BASE_URL`, `UPSTREAM_API_KEY`, and `UPSTREAM_MODEL` to connect a
real OpenAI-compatible provider. Without them, the health endpoint runs but
completion requests correctly return an upstream-unavailable response.

## Verify

```sh
.venv/bin/python -m pytest -q
```

The tests exercise real HTTP requests against the FastAPI application using a
fake upstream provider, including rate-limit, cache, retry, and failure paths.

## Input policy

Before a cache miss reaches the upstream, the gateway applies a total content
size cap and blocks obvious personal email/US-phone details and a small set of
high-confidence prompt-injection phrases. It rejects rather than rewrites
those requests, so content is never silently changed and redacted requests
cannot collide in the cache.

This is a practical deterministic policy layer, **not** a general solution to
prompt injection. More sophisticated attacks need application-specific trust
boundaries, tool permissions, and evaluation—not a larger regex list.
