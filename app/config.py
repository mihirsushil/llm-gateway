from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    api_keys: frozenset[str]
    upstream_base_url: str | None
    upstream_api_key: str | None
    upstream_model: str | None
    rate_limit_per_minute: int = 60
    rate_limit_burst: int = 10
    rate_limit_tokens_per_minute: int = 100_000
    rate_limit_token_burst: int = 20_000
    default_completion_tokens_estimate: int = 256
    cache_ttl_seconds: float = 60.0
    retry_attempts: int = 3
    retry_base_seconds: float = 0.1
    provider_timeout_seconds: float = 10.0
    circuit_breaker_failure_threshold: int = 3
    circuit_breaker_recovery_seconds: float = 30.0
    max_input_characters: int = 100_000
    max_output_characters: int = 100_000
    fallback_base_url: str | None = None
    fallback_api_key: str | None = None
    fallback_model: str | None = None
    cheap_model: str = "gpt-4o-mini"
    frontier_model: str = "gpt-4o"
    complexity_threshold: float = 0.5
    budget_cap_usd: float = 5.0

    @classmethod
    def from_environment(cls) -> "Settings":
        keys = frozenset(key.strip() for key in os.getenv("GATEWAY_API_KEYS", "").split(",") if key.strip())
        return cls(
            api_keys=keys,
            upstream_base_url=os.getenv("UPSTREAM_BASE_URL"),
            upstream_api_key=os.getenv("UPSTREAM_API_KEY"),
            upstream_model=os.getenv("UPSTREAM_MODEL"),
            rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")),
            rate_limit_burst=int(os.getenv("RATE_LIMIT_BURST", "10")),
            rate_limit_tokens_per_minute=int(os.getenv("RATE_LIMIT_TOKENS_PER_MINUTE", "100000")),
            rate_limit_token_burst=int(os.getenv("RATE_LIMIT_TOKEN_BURST", "20000")),
            default_completion_tokens_estimate=int(os.getenv("DEFAULT_COMPLETION_TOKENS_ESTIMATE", "256")),
            cache_ttl_seconds=float(os.getenv("CACHE_TTL_SECONDS", "60")),
            retry_attempts=int(os.getenv("RETRY_ATTEMPTS", "3")),
            retry_base_seconds=float(os.getenv("RETRY_BASE_SECONDS", "0.1")),
            provider_timeout_seconds=float(os.getenv("PROVIDER_TIMEOUT_SECONDS", "10")),
            circuit_breaker_failure_threshold=int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "3")),
            circuit_breaker_recovery_seconds=float(os.getenv("CIRCUIT_BREAKER_RECOVERY_SECONDS", "30")),
            max_input_characters=int(os.getenv("MAX_INPUT_CHARACTERS", "100000")),
            max_output_characters=int(os.getenv("MAX_OUTPUT_CHARACTERS", "100000")),
            fallback_base_url=os.getenv("FALLBACK_BASE_URL"),
            fallback_api_key=os.getenv("FALLBACK_API_KEY"),
            fallback_model=os.getenv("FALLBACK_MODEL"),
            cheap_model=os.getenv("CHEAP_MODEL", "gpt-4o-mini"),
            frontier_model=os.getenv("FRONTIER_MODEL", "gpt-4o"),
            complexity_threshold=float(os.getenv("COMPLEXITY_THRESHOLD", "0.5")),
            budget_cap_usd=float(os.getenv("BUDGET_CAP_USD", "5.0")),
        )
