from .models import ChatCompletionResponse


class OutputRejected(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class OutputGuard:
    """A deterministic post-provider policy, mirroring InputGuard's shape.

    Schema validity is already enforced by ChatCompletionResponse.model_validate
    in the provider client (a malformed payload never reaches here). This guard
    catches the well-formed-but-wrong cases: a degenerate empty completion, an
    output large enough to be a cost/abuse concern, or an upstream secret
    (its own API key) echoed back into content that reaches the client.
    """

    def __init__(self, max_output_characters: int, upstream_api_key: str | None = None):
        if max_output_characters <= 0:
            raise ValueError("max_output_characters must be positive")
        self.max_output_characters = max_output_characters
        self.upstream_api_key = upstream_api_key or None

    def validate(self, response: ChatCompletionResponse) -> None:
        if not response.choices:
            raise OutputRejected("empty_completion")

        content = "\n".join(choice.message.content for choice in response.choices)
        if not content.strip():
            raise OutputRejected("empty_completion")
        if len(content) > self.max_output_characters:
            raise OutputRejected("output_too_large")
        if self.upstream_api_key and self.upstream_api_key in content:
            raise OutputRejected("upstream_secret_leak")
