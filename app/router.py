import re
from dataclasses import dataclass
from typing import Literal

from .models import ChatCompletionRequest
from .tokens import estimate_prompt_tokens


@dataclass(frozen=True)
class RoutingDecision:
    model: str
    complexity_score: float
    tier: Literal["cheap", "frontier"]
    client_pinned: bool


class ComplexityRouter:
    """Routes a request to a cheap or frontier model tier based on a
    deterministic complexity heuristic — request length, turn count,
    requested output budget, and reasoning/code keyword density. This is a
    practical heuristic, **not** a learned or embedding-based semantic
    classifier: it scores observable request shape, not meaning. It is
    good enough to shed obviously-simple traffic onto a cheap model, and
    the scoring function is the one place a real deployment would swap in
    a trained or embedding-similarity classifier without touching anything
    else that depends on `RoutingDecision`.

    A client that names a model explicitly is always honored: routing only
    fills in a choice the client left up to the gateway.
    """

    _REASONING_KEYWORDS = re.compile(
        r"\b(?:prove|derive|optimi[sz]e|architecture|design a|algorithm|"
        r"complexity|debug|refactor|multi-step|step by step|trade-?off)\b",
        re.IGNORECASE,
    )
    _CODE_MARKERS = re.compile(r"```|\bdef \b|\bclass \b|\bSELECT \b|\bimport \b", re.IGNORECASE)

    _LENGTH_TOKENS_FOR_MAX_SIGNAL = 500
    _TURNS_FOR_MAX_SIGNAL = 10

    def __init__(self, cheap_model: str, frontier_model: str, complexity_threshold: float):
        if not 0.0 <= complexity_threshold <= 1.0:
            raise ValueError("complexity_threshold must be between 0 and 1")
        self.cheap_model = cheap_model
        self.frontier_model = frontier_model
        self.complexity_threshold = complexity_threshold

    def score(self, request: ChatCompletionRequest) -> float:
        content = "\n".join(message.content for message in request.messages)
        tokens = estimate_prompt_tokens(request)

        length_signal = min(tokens / self._LENGTH_TOKENS_FOR_MAX_SIGNAL, 1.0) * 0.4
        turn_signal = min(len(request.messages) / self._TURNS_FOR_MAX_SIGNAL, 1.0) * 0.2
        reasoning_signal = 0.25 if self._REASONING_KEYWORDS.search(content) else 0.0
        code_signal = 0.25 if self._CODE_MARKERS.search(content) else 0.0
        output_budget_signal = 0.1 if (request.max_tokens or 0) > 1024 else 0.0

        return min(length_signal + turn_signal + reasoning_signal + code_signal + output_budget_signal, 1.0)

    def route(self, request: ChatCompletionRequest) -> RoutingDecision:
        score = self.score(request)
        tier: Literal["cheap", "frontier"] = "frontier" if score >= self.complexity_threshold else "cheap"

        if request.model:
            return RoutingDecision(model=request.model, complexity_score=score, tier=tier, client_pinned=True)

        model = self.frontier_model if tier == "frontier" else self.cheap_model
        return RoutingDecision(model=model, complexity_score=score, tier=tier, client_pinned=False)
