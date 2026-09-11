from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from libs.drafting.retrieval_backends.base import RetrievedExample, RetrievalBackend
from libs.drafting.retrieval_backends.lexical import LexicalRetrievalBackend, score_rows
from libs.drafting.retrieval_backends.vector_chroma import VectorChromaRetrievalBackend


@dataclass(slots=True)
class RetrievalConfig:
    backend: str = "lexical"
    fallback_backend: str = "lexical"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    limit: int = 5
    min_score: float = 0.35
    strict_greeting_filter: bool = True


def load_retrieval_config(path: Path | None) -> RetrievalConfig:
    if path is None or not path.exists():
        return RetrievalConfig()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return RetrievalConfig()
    if not isinstance(payload, dict):
        return RetrievalConfig()
    return RetrievalConfig(
        backend=str(payload.get("backend", "lexical")),
        fallback_backend=str(payload.get("fallback_backend", "lexical")),
        embedding_model=str(payload.get("embedding_model", RetrievalConfig.embedding_model)),
        limit=int(payload.get("limit", 5) or 5),
        min_score=float(payload.get("min_score", 0.35) or 0.35),
        strict_greeting_filter=bool(payload.get("strict_greeting_filter", True)),
    )


def create_retrieval_backend(
    config: RetrievalConfig,
    *,
    rows: list[dict[str, Any]],
    vector_index_dir: Path | None = None,
) -> RetrievalBackend:
    if config.backend == "vector_chroma" and vector_index_dir is not None:
        return VectorChromaRetrievalBackend(
            index_dir=vector_index_dir,
            embedding_model=config.embedding_model,
            fallback_rows=rows,
            min_score=config.min_score,
            limit=config.limit,
            strict_greeting_filter=config.strict_greeting_filter,
        )
    return LexicalRetrievalBackend(rows)


__all__ = [
    "RetrievedExample",
    "RetrievalBackend",
    "RetrievalConfig",
    "LexicalRetrievalBackend",
    "VectorChromaRetrievalBackend",
    "create_retrieval_backend",
    "load_retrieval_config",
    "score_rows",
]
