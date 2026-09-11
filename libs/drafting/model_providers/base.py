from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


ModelRole = Literal["system", "user", "assistant"]


@dataclass(slots=True)
class ModelMessage:
    role: ModelRole
    content: str


@dataclass(slots=True)
class ModelResponse:
    text: str
    provider: str
    model: str
    latency_ms: int | None
    raw_finish_reason: str | None
    error: str | None
    external_api_used: bool


class ModelProvider(Protocol):
    name: str

    async def generate(
        self,
        messages: list[ModelMessage],
        model: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 180,
        response_format: str = "text",
        timeout_seconds: float = 25,
    ) -> ModelResponse:
        ...
