import hashlib
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass

from .budget import BudgetTracker
from .cache import TTLCache
from .circuit_breaker import CircuitBreaker
from .input_guard import InputGuard
from .models import ChatCompletionRequest, ChatCompletionResponse
from .output_guard import OutputGuard, OutputRejected
from .rate_limit import TokenBucketRateLimiter
from .resilience import ResilientProvider
from .router import ComplexityRouter, RoutingDecision
from .streaming import guarded_sse_stream
from .tokens import estimate_prompt_tokens


@dataclass(frozen=True)
class CompletionResult:
    response: ChatCompletionResponse
    cache_hit: bool
    attempts: int
    routing: RoutingDecision | None
    cost_usd: float | None


class GatewayService:
    def __init__(
        self,
        provider: ResilientProvider,
        limiter: TokenBucketRateLimiter,
        token_limiter: TokenBucketRateLimiter,
        cache: TTLCache[ChatCompletionResponse],
        circuit_breaker: CircuitBreaker,
        input_guard: InputGuard,
        output_guard: OutputGuard,
        router: ComplexityRouter,
        budget: BudgetTracker,
        default_completion_tokens_estimate: int,
        logger: logging.Logger | None = None,
    ):
        self.provider = provider
        self.limiter = limiter
        self.token_limiter = token_limiter
        self.cache = cache
        self.circuit_breaker = circuit_breaker
        self.input_guard = input_guard
        self.output_guard = output_guard
        self.router = router
        self.budget = budget
        self.default_completion_tokens_estimate = default_completion_tokens_estimate
        self.logger = logger or logging.getLogger("llm_gateway")

    @staticmethod
    def _cache_key(api_key: str, request: ChatCompletionRequest) -> str:
        canonical_request = json.dumps(
            request.model_dump(exclude_none=True), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(f"{api_key}:{canonical_request}".encode()).hexdigest()

    def _route_and_reserve(self, api_key: str, request: ChatCompletionRequest) -> tuple[ChatCompletionRequest, RoutingDecision, float]:
        decision = self.router.route(request)
        routed_request = request if decision.client_pinned else request.model_copy(update={"model": decision.model})
        estimated_tokens = estimate_prompt_tokens(routed_request) + (
            routed_request.max_tokens or self.default_completion_tokens_estimate
        )
        self.token_limiter.consume(api_key, cost=estimated_tokens)
        return routed_request, decision, estimated_tokens

    def complete(self, api_key: str, request: ChatCompletionRequest) -> CompletionResult:
        # Order matters: the limiter must reject before any expensive work
        # (regex scanning, cache hashing) runs, or an unthrottled client can
        # burn CPU on every request regardless of its rate-limit status.
        self.limiter.consume(api_key)
        self.input_guard.validate(request)

        cache_key = self._cache_key(api_key, request)
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.logger.info(json.dumps({"event": "completion", "cache_hit": True, "attempts": 0}))
            return CompletionResult(response=cached, cache_hit=True, attempts=0, routing=None, cost_usd=0.0)

        # A cache hit is free and never touched a provider; only a cache
        # miss (about to spend real money) needs a budget check.
        self.budget.check(api_key)

        routed_request, decision, estimated_tokens = self._route_and_reserve(api_key, request)

        try:
            provider_result = self.provider.complete(routed_request)
        except Exception:
            self.token_limiter.refund(api_key, estimated_tokens)
            raise

        response = provider_result.response
        self.token_limiter.refund(api_key, max(0.0, estimated_tokens - response.usage.total_tokens))
        cost = self.budget.record(
            api_key, response.model, response.usage.prompt_tokens, response.usage.completion_tokens
        )

        try:
            self.output_guard.validate(response)
        except OutputRejected as error:
            if provider_result.source == "primary":
                self.circuit_breaker.record_failure()
            self.logger.warning(json.dumps({"event": "output_rejected", "code": error.reason}))
            raise

        if provider_result.source == "primary":
            self.circuit_breaker.record_success()
        self.cache.set(cache_key, response)
        self.logger.info(
            json.dumps(
                {
                    "event": "completion",
                    "cache_hit": False,
                    "attempts": provider_result.attempts,
                    "model": response.model,
                    "tier": decision.tier,
                    "cost_usd": cost,
                }
            )
        )
        return CompletionResult(
            response=response, cache_hit=False, attempts=provider_result.attempts, routing=decision, cost_usd=cost
        )

    def stream(self, api_key: str, request: ChatCompletionRequest) -> tuple[Iterator[bytes], RoutingDecision]:
        # Streaming responses are never cached (there is no complete response
        # object to key on until the stream ends) and never retried once
        # forwarding begins, so the pipeline is limiter -> input guard ->
        # budget -> router/TPM reservation -> stream open, with no cache stage.
        self.limiter.consume(api_key)
        self.input_guard.validate(request)
        self.budget.check(api_key)

        routed_request, decision, estimated_tokens = self._route_and_reserve(api_key, request)

        try:
            lines = self.provider.stream(routed_request)
        except Exception:
            self.token_limiter.refund(api_key, estimated_tokens)
            raise

        # Unlike complete(), a streamed connection's success isn't attributed
        # to the breaker here: we can't yet tell whether it came from the
        # primary or the fallback, and a mid-stream failure (which the
        # breaker should still learn about) hasn't happened yet either. The
        # breaker only sees the connection-level failures ResilientProvider
        # already records.

        def on_complete(content: str, completion_tokens: int) -> None:
            self.token_limiter.refund(api_key, max(0.0, estimated_tokens - completion_tokens))
            prompt_tokens = estimate_prompt_tokens(routed_request)
            cost = self.budget.record(api_key, routed_request.model or "", prompt_tokens, completion_tokens)
            self.logger.info(
                json.dumps(
                    {
                        "event": "completion",
                        "cache_hit": False,
                        "streamed": True,
                        "model": routed_request.model,
                        "tier": decision.tier,
                        "cost_usd": cost,
                    }
                )
            )

        body = guarded_sse_stream(lines, self.output_guard, on_complete, logger=self.logger)
        return body, decision
