import re

from .models import ChatCompletionRequest


class InputRejected(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class InputGuard:
    """A deliberately narrow, deterministic pre-provider policy.

    Prompt-injection detection is heuristic, not a security boundary: a
    determined attacker can phrase an attack in ways a regex cannot recognize.
    The guard only blocks high-confidence patterns, caps total input size, and
    prevents obvious personal contact details from leaving the gateway.
    """

    _INJECTION_PATTERNS = (
        re.compile(
            r"\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above)\s+"
            r"(?:instructions|rules|prompts)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:reveal|show|print|repeat)\s+(?:the\s+)?(?:system|developer)\s+"
            r"(?:prompt|instructions)\b",
            re.IGNORECASE,
        ),
    )
    _EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
    _US_PHONE_PATTERN = re.compile(
        r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}(?!\d)"
    )

    def __init__(self, max_input_characters: int):
        if max_input_characters <= 0:
            raise ValueError("max_input_characters must be positive")
        self.max_input_characters = max_input_characters

    def validate(self, request: ChatCompletionRequest) -> None:
        content = "\n".join(message.content for message in request.messages)
        if len(content) > self.max_input_characters:
            raise InputRejected("input_too_large")
        if any(pattern.search(content) for pattern in self._INJECTION_PATTERNS):
            raise InputRejected("suspected_prompt_injection")
        if self._EMAIL_PATTERN.search(content) or self._US_PHONE_PATTERN.search(content):
            raise InputRejected("potential_pii")
