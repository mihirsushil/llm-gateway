# LLM Gateway

An OpenAI-compatible proxy that demonstrates the reliability, safety, and
cost controls a production LLM application needs between a client and an
upstream provider.

- **Phase 1:** per-key token-bucket rate limiting, exact-match TTL caching,
  timeout and transient-error retries, structured JSON request logs.
- **Phase 2:** input and output guardrails, a circuit breaker, and
  provider/key fallback.
- **Phase 3:** a token-per-minute limiter, a cost-aware complexity router,
  a per-tenant FinOps budget enforcer, and a streaming (SSE) path with
  chunk-level output guardrails.

## Architecture

```
OpenAI-compatible client
        |
        v
POST /v1/chat/completions
        |
        v
API-key validation
        |
        v
RPM rate limiter (token bucket, per key)          <- must be first: rejects
        |                                             before any expensive
        v                                             work runs
input guardrails (prompt-injection / PII / size)
        |
        v
exact-match cache  --- HIT ---> return (free, no budget/TPM/provider work)
        |
       MISS
        v
FinOps budget check (402 if this key is over its $ cap)
        |
        v
complexity router (cheap vs. frontier model, unless the client pinned one)
        |
        v
TPM limiter (token bucket, reserves an estimate, refunds the unused part)
        |
        v
ResilientProvider: circuit breaker -> retry primary to its full budget
        |             -> exactly one fallback attempt if the primary
        |                is exhausted, permanently failed, or the
        |                breaker is already open
        v
output guardrails (empty/oversized completion, upstream secret leak)
        |
        v
cache.set + budget.record + structured JSON log
        |
        v
response (X-Cache, X-Upstream-Attempts, X-Routed-Model, X-Complexity-Tier,
          X-Estimated-Cost-USD)
```

`stream: true` requests take a parallel path: limiter -> input guard ->
budget -> router/TPM reservation -> `ResilientProvider.stream()` (retry-free;
a partially-forwarded stream can't be replayed, so only the *connection*
failure is eligible for one fallback attempt) -> a chunk-level output guard
that re-checks the *accumulated* response after every SSE chunk and can cut
the stream mid-flight. Streaming responses are never cached.

## Run locally

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
GATEWAY_API_KEYS=demo-key .venv/bin/python -m uvicorn app.main:app --reload
```

Set `UPSTREAM_BASE_URL`, `UPSTREAM_API_KEY`, and `UPSTREAM_MODEL` to connect a
real OpenAI-compatible provider. Without them, the health endpoint runs but
completion requests correctly return an upstream-unavailable response.

Set `FALLBACK_BASE_URL`, `FALLBACK_API_KEY`, and (optionally) `FALLBACK_MODEL`
to add a second upstream (a different provider, or the same provider on a
second key).

Set `CHEAP_MODEL` / `FRONTIER_MODEL` / `COMPLEXITY_THRESHOLD` to tune the
router, `RATE_LIMIT_TOKENS_PER_MINUTE` / `RATE_LIMIT_TOKEN_BURST` to tune the
TPM limiter, and `BUDGET_CAP_USD` to tune the per-key dollar cap.

## Verify

```sh
.venv/bin/python -m pytest -q
```

The tests exercise real HTTP requests against the FastAPI application using
fake upstream providers (including a streaming one), covering pipeline
ordering, per-tenant cache isolation, retry/fallback/circuit-breaker
semantics (including that a client-contract error like a malformed request
never triggers a retry or fallback, and that a successful fallback response
never resets the primary's breaker), RPM and TPM limiting with refunds, the
complexity router, the budget enforcer, and streaming with a mid-stream
guardrail abort.

```sh
.venv/bin/python scripts/load_test.py
```

An in-process load test (no live server needed) that fires repeated and
bursty requests and prints the actual cache-hit and rate-limit counts.

## Pipeline order — why the limiter goes first

Every stage before the provider call costs *something* — the input guard
runs several regex passes over the full request body, the cache stage
hashes it, the router scores it. If any of that ran before the rate
limiter, a client could send unlimited over-the-limit requests and still
force the gateway to do that work on every single one — the exact
CPU-exhaustion DoS a front-loaded limiter exists to prevent. The limiter is
the first thing that runs, full stop.

## Caching — per tenant, exact-match only

The cache key is a hash of the API key plus the canonicalized request body.
Keying on the API key isn't just cache-hit efficiency — without it, two
different tenants sending byte-identical prompts would silently receive
each other's cached responses. Only exact-match caching is implemented;
there is no semantic (similarity-based) cache, so there's no risk of a
personalized or dynamic response being served to the wrong request based on
approximate similarity.

## Resilience — retry, circuit breaker, and fallback in one place

`ResilientProvider` (`app/resilience.py`) is the single place this policy
lives, replacing logic that used to be split between the service and an ad
hoc fallback wrapper composed *inside* the retry loop (which let a
persistent client error multiply into `retry_attempts x 2` upstream calls).
The explicit policy now:

- The circuit breaker gates the primary only. If it's open, the primary is
  skipped entirely and the fallback (if configured) is tried immediately.
- The primary is retried, with exponential backoff, up to its full budget —
  but only for `TransientProviderError` (timeout, network error, 429, 5xx).
  A `PermanentProviderError` (bad auth, quota, a malformed response body —
  permanent for *this* provider/key, not necessarily any other) stops the
  primary's retries immediately.
- `ClientContractError` (400/404/422 — the request itself is invalid) is
  never retried, never handed to the fallback, and never counted against
  the breaker: an identical malformed request fails identically anywhere,
  so retrying or failing over wastes a call and risks tripping the breaker
  on a signal that has nothing to do with provider health.
- Once the primary's attempt ends, the fallback — if configured — gets
  exactly one attempt, never itself retried.
- A successful *fallback* response does not reset the primary's breaker.
  The primary is still down; the fallback bailing out this one request
  proves nothing about it, and resetting the breaker would send the very
  next request straight back at a provider that's still broken.

## Cost-aware complexity router

`ComplexityRouter` (`app/router.py`) scores each request on a 0-1 scale —
prompt length, turn count, requested output budget, and reasoning/code
keyword density — and routes anything above `COMPLEXITY_THRESHOLD` to
`FRONTIER_MODEL`, everything else to `CHEAP_MODEL`. This is a deterministic
heuristic on observable request shape, **not** a learned or
embedding-based semantic classifier — it's good enough to shed obviously
simple traffic onto a cheap model, and the scoring function is the one
place a real deployment would swap in a trained classifier without
touching anything downstream. A client that names a model explicitly is
always honored; routing only fills in a choice the client left open.

## FinOps: per-tenant budget enforcement

`BudgetTracker` (`app/budget.py`) keeps a running per-API-key dollar ledger
priced from actual `usage` tokens against a small model-pricing table, and
returns `402 Payment Required` once a key's spend reaches its
`BUDGET_CAP_USD` cap. A free cache hit is never checked against the budget
— there's nothing to bill for it. In-memory and process-local, same
Redis-swappable shape as the cache: a multi-instance deployment would back
it with a shared store so the cap holds across instances.

## TPM limiting

The existing RPM `TokenBucketRateLimiter` is reused for tokens-per-minute:
before a cache-miss request is sent upstream, the gateway reserves an
estimate (prompt tokens + `max_tokens`, or `DEFAULT_COMPLETION_TOKENS_ESTIMATE`
if unset), then refunds the unused portion once the real `usage` is known —
so a single large request can't saturate a minute's token throughput even
while comfortably under the RPM cap, without permanently overcharging the
bucket for tokens that were never actually generated.

## Streaming guardrail

`stream: true` requests skip the cache (there's no complete response to key
on until the stream ends) and skip retries (a stream that has already
forwarded bytes to the client can't be replayed). `guarded_sse_stream`
(`app/streaming.py`) re-applies the output guard's size cap and
upstream-secret-leak check against the *accumulated* content after every
forwarded chunk, so an unsafe response is cut as soon as it becomes unsafe
instead of only after the client has already received the whole thing.
Token accounting for the TPM limiter and the budget ledger settles from the
accumulated content once the stream ends, since SSE responses don't
reliably carry a `usage` object.

## Input / output policy

Before a cache miss reaches the upstream, the gateway applies a total
content size cap and blocks obvious personal email/US-phone details and a
small set of high-confidence prompt-injection phrases. It rejects rather
than rewrites those requests, so content is never silently changed. This is
a practical deterministic policy layer, **not** a general solution to
prompt injection — more sophisticated attacks need application-specific
trust boundaries, tool permissions, and evaluation, not a larger regex
list.

Schema validity is already enforced by the provider client before a
response is treated as a success. The output guard catches the
well-formed-but-wrong cases after that: an empty/whitespace-only
completion, an output larger than `MAX_OUTPUT_CHARACTERS`, or the
upstream's own API key echoed back into content. A rejection here counts as
a circuit-breaker failure *only when the response came from the primary* —
same "reject, don't silently rewrite" policy as the input guard.

## Failure-mode table

| Failure scenario | Without the guardrail | What the guardrail does |
|---|---|---|
| A client sends far more requests than the service can sustain | Requests queue or the upstream gets hammered and starts failing for everyone | Token-bucket rate limiter rejects with `429` + `Retry-After` once a key's burst is spent — checked before any other work runs |
| A single request asks for an outsized amount of generation | RPM alone lets one huge request through, saturating provider throughput | TPM limiter reserves an estimate up front and rejects with `429` if the key's token budget can't cover it |
| The same prompt is requested repeatedly by the same tenant | Every repeat pays full upstream latency/cost | Exact-match, per-tenant TTL cache returns the cached response, `X-Cache: HIT`, no budget/TPM/provider work |
| Two tenants happen to send byte-identical prompts | Tenant B silently receives tenant A's cached response | Cache key includes the API key, so tenants never share a cache entry |
| The upstream times out or returns a transient error (`429`/`5xx`) | The client sees an immediate failure on a possibly-recoverable blip | Retried against the primary with exponential backoff up to `RETRY_ATTEMPTS`, then one fallback attempt, before failing |
| The upstream rejects the request itself (malformed body, `400`/`404`/`422`) | Retried and failed over anyway, wasting a call on a fallback that will fail identically | `ClientContractError` returns `400` immediately — no retry, no fallback, no circuit-breaker impact |
| The upstream is down for an extended period | Every request pays a full timeout + retry budget while it's down | Circuit breaker opens after `CIRCUIT_BREAKER_FAILURE_THRESHOLD` failures and skips the primary entirely, going straight to the fallback (or failing fast with no fallback); one half-open probe checks recovery |
| A single configured provider/key is the failure | The whole service is unavailable until that one provider/key recovers | Fallback provider is tried once the primary's retry budget is exhausted — without resetting the primary's breaker on success |
| A request contains a prompt-injection attempt or embedded PII | The upstream sees (and might act on, or the response might contain) the injected instructions or leaked contact info | Input guard rejects before the provider is ever called |
| A request is unreasonably large | Wastes upstream cost/latency, or hits an upstream size limit | Input guard rejects requests over `MAX_INPUT_CHARACTERS` |
| The upstream returns a degenerate (empty) or oversized completion, or echoes back its own API key | The client gets a broken or unsafe response and the gateway caches/serves it as fine | Output guard rejects before caching or returning it, `502` |
| A streaming response starts fine but drifts unsafe mid-stream | The client receives the entire unsafe response before anything can react | The chunk-level guard re-checks accumulated content after every chunk and cuts the stream in-band as soon as it becomes unsafe |
| A tenant's cumulative spend exceeds its allocation | Usage (and cost) is unbounded per key | `BudgetTracker` returns `402 Payment Required` once a key's ledger reaches its `BUDGET_CAP_USD` cap |
| A simple request is sent to an expensive frontier model by default | Every request pays frontier pricing regardless of actual complexity | The complexity router sends low-scoring requests to `CHEAP_MODEL` unless the client pinned a model |
