import logging
import math
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Response, status
from fastapi.responses import StreamingResponse

from .budget import BudgetExceeded, BudgetTracker
from .cache import TTLCache
from .circuit_breaker import CircuitBreaker
from .config import Settings
from .input_guard import InputGuard, InputRejected
from .models import ChatCompletionRequest, ChatCompletionResponse
from .output_guard import OutputGuard, OutputRejected
from .provider import ChatProvider, ClientContractError, OpenAICompatibleProvider, UnconfiguredProvider
from .rate_limit import RateLimitExceeded, TokenBucketRateLimiter
from .resilience import ResilientProvider, UpstreamUnavailable
from .router import ComplexityRouter
from .service import GatewayService


def _build_providers(settings: Settings) -> tuple[ChatProvider, ChatProvider | None]:
    if not (settings.upstream_base_url and settings.upstream_api_key):
        return UnconfiguredProvider(), None

    primary = OpenAICompatibleProvider(
        base_url=settings.upstream_base_url,
        api_key=settings.upstream_api_key,
        default_model=settings.upstream_model,
        timeout_seconds=settings.provider_timeout_seconds,
    )
    if not (settings.fallback_base_url and settings.fallback_api_key):
        return primary, None

    secondary = OpenAICompatibleProvider(
        base_url=settings.fallback_base_url,
        api_key=settings.fallback_api_key,
        default_model=settings.fallback_model or settings.upstream_model,
        timeout_seconds=settings.provider_timeout_seconds,
    )
    return primary, secondary


def _gateway_logger() -> logging.Logger:
    logger = logging.getLogger("llm_gateway")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def create_app(
    settings: Settings | None = None,
    provider: ChatProvider | None = None,
    fallback_provider: ChatProvider | None = None,
) -> FastAPI:
    settings = settings or Settings.from_environment()
    logger = _gateway_logger()

    if provider is None:
        primary, fallback_provider = _build_providers(settings)
    else:
        primary = provider

    circuit_breaker = CircuitBreaker(
        settings.circuit_breaker_failure_threshold,
        settings.circuit_breaker_recovery_seconds,
    )
    resilient_provider = ResilientProvider(
        primary=primary,
        circuit_breaker=circuit_breaker,
        retry_attempts=settings.retry_attempts,
        retry_base_seconds=settings.retry_base_seconds,
        fallback=fallback_provider,
        logger=logger,
    )
    service = GatewayService(
        provider=resilient_provider,
        limiter=TokenBucketRateLimiter(settings.rate_limit_per_minute, settings.rate_limit_burst),
        token_limiter=TokenBucketRateLimiter(settings.rate_limit_tokens_per_minute, settings.rate_limit_token_burst),
        cache=TTLCache(settings.cache_ttl_seconds),
        circuit_breaker=circuit_breaker,
        input_guard=InputGuard(settings.max_input_characters),
        output_guard=OutputGuard(settings.max_output_characters, settings.upstream_api_key),
        router=ComplexityRouter(settings.cheap_model, settings.frontier_model, settings.complexity_threshold),
        budget=BudgetTracker(settings.budget_cap_usd),
        default_completion_tokens_estimate=settings.default_completion_tokens_estimate,
        logger=logger,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield

    app = FastAPI(title="LLM Gateway", version="0.2.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.service = service

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    def chat_completions(
        request: ChatCompletionRequest,
        response: Response,
        x_api_key: str | None = Header(default=None),
    ):
        if not x_api_key or x_api_key not in app.state.settings.api_keys:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid API key")

        try:
            if request.stream:
                body, decision = app.state.service.stream(x_api_key, request)
                return StreamingResponse(
                    body,
                    media_type="text/event-stream",
                    headers={
                        "X-Routed-Model": decision.model,
                        "X-Complexity-Tier": decision.tier,
                        "X-Complexity-Score": f"{decision.complexity_score:.2f}",
                    },
                )

            result = app.state.service.complete(x_api_key, request)
        except InputRejected as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": error.reason, "message": "request blocked by input safety policy"},
            ) from error
        except ClientContractError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "client_contract_error", "message": str(error)},
            ) from error
        except RateLimitExceeded as error:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="rate limit exceeded",
                headers={"Retry-After": str(max(1, math.ceil(error.retry_after_seconds)))},
            ) from error
        except BudgetExceeded as error:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail={
                    "code": "budget_exceeded",
                    "message": str(error),
                    "spent_usd": round(error.spent_usd, 4),
                    "cap_usd": error.cap_usd,
                },
            ) from error
        except OutputRejected as error:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={"code": error.reason, "message": "upstream response blocked by output safety policy"},
            ) from error
        except UpstreamUnavailable as error:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error

        response.headers["X-Cache"] = "HIT" if result.cache_hit else "MISS"
        response.headers["X-Upstream-Attempts"] = str(result.attempts)
        if result.routing is not None:
            response.headers["X-Routed-Model"] = result.routing.model
            response.headers["X-Complexity-Tier"] = result.routing.tier
            response.headers["X-Complexity-Score"] = f"{result.routing.complexity_score:.2f}"
        if result.cost_usd is not None:
            response.headers["X-Estimated-Cost-USD"] = f"{result.cost_usd:.6f}"
        return result.response

    return app


app = create_app()
