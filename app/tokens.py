import math

from .models import ChatCompletionRequest

CHARS_PER_TOKEN = 4
"""A fixed, deterministic chars-per-token ratio. Real tokenizers are
model-specific (and BPE vocabularies vary token-to-char ratios by language
and content), but a stable estimate is all the TPM limiter and router need
before an upstream call returns actual usage — precision isn't the goal,
having a pre-call number is.
"""


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def estimate_prompt_tokens(request: ChatCompletionRequest) -> int:
    content = "\n".join(message.content for message in request.messages)
    return estimate_tokens(content)
