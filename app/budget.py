import threading

# $ per 1,000 tokens, (prompt, completion). Illustrative list pricing, not
# fetched live — a production system would source this from the provider's
# billing API or a config file kept in sync with actual contract rates.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    "claude-3-5-haiku-20241022": (0.0008, 0.004),
    "claude-3-5-sonnet-20241022": (0.003, 0.015),
}

# Unknown models are priced at the most expensive known tier rather than the
# cheapest: a budget enforcer that under-charges an unrecognized model fails
# open on cost, which defeats the point of a hard cap.
_DEFAULT_PRICING = max(MODEL_PRICING.values(), key=lambda prices: sum(prices))


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prompt_price, completion_price = MODEL_PRICING.get(model, _DEFAULT_PRICING)
    return (prompt_tokens / 1000) * prompt_price + (completion_tokens / 1000) * completion_price


class BudgetExceeded(Exception):
    def __init__(self, api_key: str, spent_usd: float, cap_usd: float):
        self.api_key = api_key
        self.spent_usd = spent_usd
        self.cap_usd = cap_usd
        super().__init__(f"budget exceeded: ${spent_usd:.4f} spent against a ${cap_usd:.2f} cap")


class BudgetTracker:
    """Per-API-key running dollar ledger with a hard cap, checked before
    each cache-miss request and updated after each completion. In-memory
    and process-local, same Redis-swappable shape as `TTLCache`: a
    multi-instance deployment would back this with a shared store so the
    cap holds across instances, not per-process.
    """

    def __init__(self, cap_usd: float):
        if cap_usd <= 0:
            raise ValueError("cap_usd must be positive")
        self.cap_usd = cap_usd
        self._spent: dict[str, float] = {}
        self._lock = threading.Lock()

    def spent(self, api_key: str) -> float:
        with self._lock:
            return self._spent.get(api_key, 0.0)

    def check(self, api_key: str) -> None:
        spent = self.spent(api_key)
        if spent >= self.cap_usd:
            raise BudgetExceeded(api_key, spent, self.cap_usd)

    def record(self, api_key: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        cost = cost_usd(model, prompt_tokens, completion_tokens)
        with self._lock:
            self._spent[api_key] = self._spent.get(api_key, 0.0) + cost
        return cost
