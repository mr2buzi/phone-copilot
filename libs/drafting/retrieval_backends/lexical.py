from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any

from libs.drafting.intents import (
    classify_intent,
    contains_suspicious_phrase,
    is_simple_greeting,
    normalize_intent_label,
    normalize_text,
    tokenize,
)
from libs.drafting.retrieval_backends.base import RetrievedExample, RetrievalBackend
from libs.drafting.training_data import normalize_relationship_type

TOKEN_RE = re.compile(r"[a-z0-9']+")
SAFE_RELATIONSHIPS = {"close_friend", "casual_friend", "family", "university", "professional", "unknown"}


def _row_reply(row: dict[str, Any]) -> str:
    return str(row.get("my_reply") or row.get("user_final_reply") or "").strip()


def _row_incoming(row: dict[str, Any]) -> str:
    return str(row.get("incoming", "")).strip()


def _row_context(row: dict[str, Any]) -> list[str]:
    context = row.get("context", [])
    return [str(item).strip() for item in context if str(item).strip()] if isinstance(context, list) else []


def _row_intent(row: dict[str, Any], fallback: str = "unknown") -> str:
    raw_intent = str(row.get("intent_type") or "").strip()
    if raw_intent:
        return normalize_intent_label(raw_intent)
    incoming = _row_incoming(row)
    context = _row_context(row)
    if incoming:
        return normalize_intent_label(classify_intent(incoming, context))
    return normalize_intent_label(fallback)


def _tokens(text: str) -> set[str]:
    return {token for token in TOKEN_RE.findall(normalize_text(text)) if len(token) > 1}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _phrase_overlap(left: str, right: str) -> int:
    left_words = list(TOKEN_RE.findall(normalize_text(left)))
    right_text = " ".join(TOKEN_RE.findall(normalize_text(right)))
    count = 0
    for size in (3, 2):
        for index in range(0, max(0, len(left_words) - size + 1)):
            if " ".join(left_words[index : index + size]) in right_text:
                count += 1
    return count


def _retrieval_id(*parts: str) -> str:
    digest = hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()
    return digest[:12]


def _is_correction_source(row: dict[str, Any]) -> bool:
    source = str(row.get("_source") or row.get("source") or "").strip()
    return bool(row.get("is_correction")) or source in {"correction", "training_page_feedback", "approved_auto_style_improvement"}


def _style_authority(row: dict[str, Any]) -> str:
    authority = str(row.get("style_authority") or "").strip().lower()
    if authority in {"high", "medium", "low"}:
        return authority
    if _is_correction_source(row):
        return "high"
    if row.get("is_synthetic"):
        return "low"
    return "medium"


def _is_safe_row(row: dict[str, Any], *, intent_type: str) -> bool:
    reply = _row_reply(row)
    incoming = _row_incoming(row)
    row_intent = _row_intent(row, intent_type)
    if not reply or not incoming:
        return False
    if contains_suspicious_phrase(reply):
        return False
    if contains_suspicious_phrase(incoming):
        return False
    if row_intent == "greeting" and contains_suspicious_phrase(reply):
        return False
    return True


def score_rows(
    rows: list[dict[str, Any]],
    *,
    incoming: str,
    context: list[str],
    relationship_type: str,
    intent_type: str,
    contact_name: str | None = None,
    limit: int = 5,
    backend: str = "lexical",
    vector_scores: dict[str, float] | None = None,
    filters_applied: list[str] | None = None,
) -> list[RetrievedExample]:
    relationship_type = normalize_relationship_type(relationship_type)
    query_intent = normalize_intent_label(intent_type or "")
    if is_simple_greeting(incoming, context):
        query_intent = "greeting"
    query_tokens = _tokens(incoming)
    context_tokens = _tokens(" ".join(context[-6:]))
    short_query = len(query_tokens) <= 2 or len(normalize_text(incoming)) <= 6

    scored: list[RetrievedExample] = []
    filter_log = filters_applied or []
    for row in rows:
        row_relationship = normalize_relationship_type(str(row.get("relationship_type", "")))
        row_incoming = _row_incoming(row)
        row_reply = _row_reply(row)
        row_context = _row_context(row)
        row_intent = _row_intent(row, query_intent)
        if not row_incoming or not row_reply:
            continue
        if not _is_safe_row(row, intent_type=row_intent):
            continue
        if row_relationship == "romantic_interest" and relationship_type != "romantic_interest":
            continue
        if relationship_type == "unknown" and row_relationship not in {"unknown", "casual_friend", "close_friend"}:
            continue
        if query_intent == "greeting" and row_intent != "greeting":
            continue
        if query_intent in {"professional", "studying_work"} and row_relationship not in {relationship_type, query_intent if query_intent in SAFE_RELATIONSHIPS else relationship_type}:
            if row_relationship not in {"professional", "university"}:
                continue
        if query_intent == "unknown" and row_intent != "unknown":
            continue
        if short_query and query_intent == "greeting" and row_intent != "greeting":
            continue
        if query_intent != "greeting" and row_intent == "greeting" and short_query:
            continue

        incoming_tokens = _tokens(row_incoming)
        row_context_tokens = _tokens(" ".join(row_context))
        overlap = query_tokens & incoming_tokens
        context_overlap = context_tokens & row_context_tokens
        phrase_overlap = _phrase_overlap(incoming, row_incoming)
        intent_match = row_intent == query_intent
        related_intents = {
            "greeting": {"greeting", "casual_checkin"},
            "casual_checkin": {"casual_checkin", "greeting"},
            "planning": {"planning", "availability", "confirmation"},
            "simple_question": {"simple_question", "planning", "availability"},
            "professional": {"professional", "simple_question", "confirmation"},
            "studying_work": {"studying_work", "simple_question", "professional"},
            "availability": {"availability", "planning", "confirmation"},
            "thanks": {"thanks", "confirmation"},
            "apology": {"apology", "unknown"},
            "decline": {"decline", "delay"},
            "delay": {"delay", "planning", "availability"},
            "argument": {"argument", "emotional"},
            "emotional": {"emotional", "argument"},
            "joke_banter_safe": {"joke_banter_safe", "confirmation"},
            "romantic_flirty": {"romantic_flirty"},
            "confirmation": {"confirmation", "greeting", "casual_checkin"},
            "unknown": {"unknown"},
        }.get(query_intent, {query_intent})
        strong_intent_match = row_intent in related_intents

        score = 0.0
        fallback_id = _retrieval_id(row_relationship, row_incoming, row_reply, str(row.get("_source", "training")))
        row_id = str(row.get("id", ""))
        vector_score = float(vector_scores.get(row_id, vector_scores.get(fallback_id, 0.0))) if vector_scores else 0.0
        score += vector_score * 0.45
        score += _jaccard(query_tokens, incoming_tokens) * 0.28
        score += _jaccard(context_tokens, row_context_tokens) * 0.12
        score += min(0.12, 0.03 * phrase_overlap)
        if row_relationship == relationship_type:
            score += 0.16
        if contact_name and normalize_text(str(row.get("contact_name", ""))) == normalize_text(contact_name):
            score += 0.22
        if intent_match:
            score += 0.22
        elif strong_intent_match:
            score += 0.12
        else:
            score -= 0.25 if short_query else 0.1
        authority = _style_authority(row)
        if _is_correction_source(row) and strong_intent_match:
            score += 0.28 if intent_match else 0.16
        if authority == "high":
            score += 0.22
        elif authority == "low":
            score -= 0.12
        if row.get("is_synthetic"):
            score -= 0.28
        if row_relationship in {"professional", "university"} and any(term in row_reply.casefold() for term in ("lol", "lmao", "yo", "heyy", "baby", "trouble")):
            score -= 0.25
        if query_intent == "greeting" and row_intent != "greeting":
            score -= 0.8
        if query_intent != "greeting" and row_intent == "greeting" and short_query:
            score -= 0.3
        if not overlap and not context_overlap and phrase_overlap == 0 and not strong_intent_match and row_relationship != relationship_type:
            continue

        reason_parts: list[str] = []
        if contact_name and normalize_text(str(row.get("contact_name", ""))) == normalize_text(contact_name):
            reason_parts.append("same contact")
        if row_relationship == relationship_type:
            reason_parts.append("same relationship")
        if intent_match:
            reason_parts.append(f"same intent ({query_intent})")
        elif strong_intent_match:
            reason_parts.append(f"related intent ({row_intent})")
        else:
            reason_parts.append(f"intent mismatch {query_intent}->{row_intent}")
        if vector_score:
            reason_parts.append("vector similarity")
        if overlap:
            reason_parts.append("token overlap")
        if context_overlap:
            reason_parts.append("context overlap")
        if _is_correction_source(row):
            reason_parts.append("correction boost")
        if authority:
            reason_parts.append(f"style authority {authority}")
        if row.get("is_synthetic"):
            reason_parts.append("synthetic fallback")
        if filter_log:
            reason_parts.extend(filter_log[:3])

        scored.append(
            RetrievedExample(
                retrieval_id=_retrieval_id(row_relationship, row_incoming, row_reply, str(row.get("_source", "training"))),
                relationship_type=row_relationship,
                incoming=row_incoming,
                my_reply=row_reply,
                context=row_context,
                intent_type=row_intent,
                score=round(max(0.0, min(score, 1.0)), 3),
                reason=", ".join(reason_parts) or "lexical similarity",
                source="correction" if _is_correction_source(row) else str(row.get("_source", "training")),
                backend=backend,
                metadata={k: v for k, v in row.items() if k not in {"incoming", "my_reply", "user_final_reply", "context"}},
            )
        )

    non_synthetic = [item for item in scored if not bool(item.metadata.get("is_synthetic")) and item.source != "synthetic"]
    if non_synthetic:
        scored = non_synthetic
    scored.sort(
        key=lambda item: (
            -item.score,
            item.source not in {"correction", "approved_auto_style_improvement"},
            str(item.metadata.get("style_authority") or "").lower() != "high",
            item.source == "synthetic",
            len(item.my_reply),
        )
    )
    return scored[: max(1, limit)]


class LexicalRetrievalBackend(RetrievalBackend):
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows or []

    def build_index(self) -> None:
        return None

    def search(
        self,
        incoming: str,
        context: list[str],
        relationship_type: str,
        intent_type: str,
        contact_name: str | None = None,
        limit: int = 5,
    ) -> list[RetrievedExample]:
        rows = [row for row in self.rows if _row_incoming(row) and _row_reply(row)]
        filters_applied = [f"relationship={normalize_relationship_type(relationship_type)}", f"intent={normalize_intent_label(intent_type)}"]
        return score_rows(
            rows,
            incoming=incoming,
            context=context,
            relationship_type=relationship_type,
            intent_type=intent_type,
            contact_name=contact_name,
            limit=limit,
            backend="lexical",
            filters_applied=filters_applied,
        )
