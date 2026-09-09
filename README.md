# LLM Gateway

An OpenAI-compatible proxy that demonstrates the reliability and safety
controls an LLM application needs between a client and an upstream provider.
Phase 1: per-key token-bucket rate limiting, exact-match TTL caching, timeout
and transient-error retries, structured JSON request logs. Phase 2: input and
output guardrails, a circuit breaker, and provider/key fallback.

## Architecture

```
OpenAI-compatible client
        ↓
POST /v1/chat/completions
        ↓
API-key validation → token bucket → exact cache → input guardrails → circuit breaker
                                                                          ↓
                                                        retrying provider client (primary)
                                                                          ↓ (on failure)
                                                        fallback provider client (secondary)
                                                                          ↓
                                                                  output guardrails
        ↓                                                        ↓
structured JSON logs                                       OpenAI-compatible upstream(s)
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

Set `FALLBACK_BASE_URL`, `FALLBACK_API_KEY`, and (optionally) `FALLBACK_MODEL`
to add a second upstream (a different provider, or the same provider on a
second key). If the primary provider fails, the gateway retries against it per
`RETRY_ATTEMPTS`; once its retry budget is exhausted the fallback is tried
once before the request fails.

## Verify

```sh
.venv/bin/python -m pytest -q
```

The tests exercise real HTTP requests against the FastAPI application using a
fake upstream provider, including rate-limit, cache, retry, guardrail, circuit
breaker, and fallback paths.

```sh
.venv/bin/python scripts/load_test.py
```

An in-process load test (no live server needed) that fires repeated and
bursty requests and prints the actual cache-hit and rate-limit counts, so the
rate limiter and cache's effect is measured, not just asserted in a unit test.

## Input policy

Before a cache miss reaches the upstream, the gateway applies a total content
size cap and blocks obvious personal email/US-phone details and a small set of
high-confidence prompt-injection phrases. It rejects rather than rewrites
those requests, so content is never silently changed and redacted requests
cannot collide in the cache.

This is a practical deterministic policy layer, **not** a general solution to
prompt injection. More sophisticated attacks need application-specific trust
boundaries, tool permissions, and evaluation—not a larger regex list.

## Output policy

Schema validity (is this a well-formed chat completion?) is already enforced
by the provider client before a response is treated as a success. The output
guard catches the well-formed-but-wrong cases after that: an empty/
whitespace-only completion, an output larger than `MAX_OUTPUT_CHARACTERS`, or
the upstream's own API key echoed back into content (defense in depth against
a misbehaving or compromised upstream leaking its own secret to the client).
A rejection here counts as a circuit-breaker failure and returns `502`—same
"reject, don't silently rewrite" policy as the input guard.

## Failure-mode table

| Failure scenario | Without the guardrail | What the guardrail does |
|---|---|---|
| A client sends far more requests than the service can sustain | Requests queue or the upstream gets hammered and starts failing for everyone | Token-bucket rate limiter rejects with `429` + `Retry-After` once a key's burst is spent |
| The same prompt is requested repeatedly | Every repeat pays full upstream latency/cost | Exact-match TTL cache returns the cached response, `X-Cache: HIT` |
| The upstream times out or returns a transient error (`429`/`5xx`) | The client sees an immediate failure on a possibly-recoverable blip | Retried with exponential backoff up to `RETRY_ATTEMPTS` before failing |
| The upstream is down for an extended period | Every request pays a full timeout + retry budget while it's down | Circuit breaker opens after `CIRCUIT_BREAKER_FAILURE_THRESHOLD` failures and fails fast; one half-open probe checks recovery |
| A single configured provider/key is the failure | The whole service is unavailable until that one provider/key recovers | Fallback provider is tried once the primary's retry budget is exhausted |
| A request contains a prompt-injection attempt or embedded PII | The upstream sees (and might act on, or the response might contain) the injected instructions or leaked contact info | Input guard rejects before the provider is ever called |
| A request is unreasonably large | Wastes upstream cost/latency on an oversized prompt, or hits an upstream size limit with a confusing error | Input guard rejects requests over `MAX_INPUT_CHARACTERS` |
| The upstream returns a degenerate (empty) or oversized completion, or echoes back its own API key | The client gets a broken or unsafe response and the gateway caches/serves it as if it were fine | Output guard rejects before caching or returning it, `502` |
