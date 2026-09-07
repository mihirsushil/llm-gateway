from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=32_000)


class ChatCompletionRequest(BaseModel):
    model: str | None = Field(default=None, max_length=200)
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=16_384)


class CompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None = None


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: TokenUsage = Field(default_factory=TokenUsage)
