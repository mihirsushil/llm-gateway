# LLM Gateway

An OpenAI-compatible proxy that demonstrates the reliability controls an LLM
application needs between a client and an upstream provider. The first phase
includes per-key token-bucket rate limiting, exact-match TTL caching, timeout
and transient-error retries, and structured JSON request logs.

## Architecture

```
OpenAI-compatible client
        ↓
POST /v1/chat/completions
        ↓
API-key validation → token bucket → exact cache → retrying provider client
        ↓                                      ↓
structured JSON logs                     OpenAI-compatible upstream
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
