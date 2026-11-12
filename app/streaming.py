import json
import logging
from collections.abc import Callable, Iterator

from .output_guard import OutputGuard
from .tokens import estimate_tokens


def _delta_content(payload: str) -> str:
    try:
        chunk = json.loads(payload)
        return chunk["choices"][0].get("delta", {}).get("content", "") or ""
    except (ValueError, KeyError, IndexError, TypeError):
        return ""


def guarded_sse_stream(
    lines: Iterator[str],
    output_guard: OutputGuard,
    on_complete: Callable[[str, int], None],
    logger: logging.Logger | None = None,
) -> Iterator[bytes]:
    """Relays upstream SSE lines to the client while running the same
    output policy the non-streaming path enforces (max size, upstream
    secret leak) chunk-by-chunk against the *accumulated* content — so an
    unsafe response is caught and the stream is cut as soon as it becomes
    unsafe, instead of only after the full completion has already reached
    the client. `on_complete` is invoked once with the full text and an
    estimated completion-token count when the upstream signals `[DONE]`, so
    the caller can settle TPM/budget accounting for a request whose real
    `usage` object (if the upstream even sends one over SSE) can't be
    relied on.
    """
    logger = logger or logging.getLogger("llm_gateway")
    accumulated: list[str] = []

    for line in lines:
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()

        if payload == "[DONE]":
            content = "".join(accumulated)
            on_complete(content, estimate_tokens(content))
            yield b"data: [DONE]\n\n"
            return

        delta = _delta_content(payload)
        accumulated.append(delta)
        content = "".join(accumulated)

        if len(content) > output_guard.max_output_characters:
            logger.warning(json.dumps({"event": "stream_output_rejected", "code": "output_too_large"}))
            on_complete(content, estimate_tokens(content))
            yield b'data: {"error": {"code": "output_too_large"}}\n\n'
            return

        if output_guard.upstream_api_key and output_guard.upstream_api_key in content:
            logger.warning(json.dumps({"event": "stream_output_rejected", "code": "upstream_secret_leak"}))
            on_complete(content, estimate_tokens(content))
            yield b'data: {"error": {"code": "upstream_secret_leak"}}\n\n'
            return

        yield f"{line}\n\n".encode()

    # Upstream closed without a [DONE] sentinel: still settle accounting.
    on_complete("".join(accumulated), estimate_tokens("".join(accumulated)))
