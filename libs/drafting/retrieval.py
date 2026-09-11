from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from libs.drafting.intents import classify_intent, normalize_intent_label
from libs.drafting.retrieval_backends.base import RetrievedExample
from libs.drafting.retrieval_backends.lexical import LexicalRetrievalBackend, score_rows
from libs.drafting.training_data import load_all_training_rows, load_corrections, normalize_relationship_type


class RetrievalResult(BaseModel):
    examples: list[RetrievedExample] = Field(default_factory=list)


def retrieve_similar_examples(
    *,
    incoming: str,
    recent_context: list[str],
    relationship_type: str,
    training_dir: Path,
    incoming_intent: str | None = None,
    limit: int = 5,
) -> RetrievalResult:
    rows = load_all_training_rows(training_dir)
    for row in load_corrections(training_dir):
        rows.append(
            {
                "relationship_type": row["relationship_type"],
                "incoming": row["incoming"],
                "context": row.get("context", []),
                "my_reply": row["user_final_reply"],
                "notes": row.get("reason_bad", "correction"),
                "_source": "correction",
                "intent_type": row.get("intent_type") or normalize_intent_label(incoming_intent or "unknown"),
            }
        )
    query_intent = normalize_intent_label(incoming_intent or classify_intent(incoming, recent_context))
    backend = LexicalRetrievalBackend(rows)
    return RetrievalResult(
        examples=backend.search(
            incoming=incoming,
            context=recent_context,
            relationship_type=relationship_type,
            intent_type=query_intent,
            limit=limit,
        )
    )


def retrieve_similar_examples_from_rows(
    *,
    rows: list[dict[str, Any]],
    incoming: str,
    recent_context: list[str],
    relationship_type: str,
    incoming_intent: str | None = None,
    limit: int = 5,
) -> RetrievalResult:
    relationship_type = normalize_relationship_type(relationship_type)
    query_intent = normalize_intent_label(incoming_intent or classify_intent(incoming, recent_context))
    examples = score_rows(
        rows,
        incoming=incoming,
        context=recent_context,
        relationship_type=relationship_type,
        intent_type=query_intent,
        limit=limit,
        backend="lexical",
        filters_applied=[f"relationship={relationship_type}", f"intent={query_intent}"],
    )
    return RetrievalResult(examples=examples)
