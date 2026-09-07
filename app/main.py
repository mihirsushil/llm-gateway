import logging
import math
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Response, status

from .cache import TTLCache
from .config import Settings
from .models import ChatCompletionRequest, ChatCompletionResponse
from .provider import ChatProvider, OpenAICompatibleProvider, UnconfiguredProvider
from .rate_limit import RateLimitExceeded, TokenBucketRateLimiter
from .service import GatewayService, UpstreamUnavailable


def _build_provider(settings: Settings) -> ChatProvider:
    if settings.upstream_base_url and settings.upstream_api_key:
        return OpenAICompatibleProvider(
            base_url=settings.upstream_base_url,
            api_key=settings.upstream_api_key,
            default_model=settings.upstream_model,
            timeout_seconds=settings.provider_timeout_seconds,
        )
    return UnconfiguredProvider()


def create_app(settings: Settings | None = None, provider: ChatProvider | None = None) -> FastAPI:
    settings = settings or Settings.from_environment()
    service = GatewayService(
        provider=provider or _build_provider(settings),
        limiter=TokenBucketRateLimiter(settings.rate_limit_per_minute, settings.rate_limit_burst),
        cache=TTLCache(settings.cache_ttl_seconds),
        retry_attempts=settings.retry_attempts,
        retry_base_seconds=settings.retry_base_seconds,
        logger=logging.getLogger("llm_gateway"),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield

    app = FastAPI(title="LLM Gateway", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.service = service

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    def chat_completions(
        request: ChatCompletionRequest,
        response: Response,
        x_api_key: str | None = Header(default=None),
    ) -> ChatCompletionResponse:
        if not x_api_key or x_api_key not in app.state.settings.api_keys:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid API key")
        try:
            result = app.state.service.complete(x_api_key, request)
        except RateLimitExceeded as error:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="rate limit exceeded",
                headers={"Retry-After": str(max(1, math.ceil(error.retry_after_seconds)))},
            ) from error
        except UpstreamUnavailable as error:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error

        response.headers["X-Cache"] = "HIT" if result.cache_hit else "MISS"
        response.headers["X-Upstream-Attempts"] = str(result.attempts)
        return result.response

    return app


app = create_app()
