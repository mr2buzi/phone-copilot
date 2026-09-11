from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field


class RetrievedExample(BaseModel):
    retrieval_id: str
    relationship_type: str
    incoming: str
    my_reply: str
    context: list[str] = Field(default_factory=list)
    intent_type: str = "unknown"
    score: float = 0.0
    reason: str = ""
    source: str = "training"
    backend: str = "lexical"
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalBackend(Protocol):
    def build_index(self) -> None: ...

    def search(
        self,
        incoming: str,
        context: list[str],
        relationship_type: str,
        intent_type: str,
        contact_name: str | None = None,
        limit: int = 5,
    ) -> list[RetrievedExample]: ...
