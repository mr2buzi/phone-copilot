from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from libs.drafting.intents import contains_suspicious_phrase, normalize_intent_label
from libs.drafting.retrieval_backends.lexical import score_rows
from libs.drafting.training_data import normalize_relationship_type

PATH = Path("data/training_messages/generated_safe_templates.jsonl")
VALID_RELATIONSHIPS = {"close_friend", "casual_friend", "family", "university", "professional", "unknown"}
VALID_INTENTS = {
    "greeting",
    "planning",
    "simple_question",
    "casual_checkin",
    "joke_banter_safe",
    "apology",
    "thanks",
    "confirmation",
    "decline",
    "delay",
    "studying_work",
    "availability",
    "follow_up",
    "unknown",
}
BANNED = ("baby", "sexy", "trouble", " x", "xx", "love", "babe", "naughty")


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> None:
    if not PATH.exists():
        raise SystemExit(f"missing {PATH}")
    rows = load_rows()
    errors: list[str] = []
    if len(rows) < 15000:
        errors.append(f"expected at least 15000 rows, got {len(rows)}")
    seen_pairs: Counter[tuple[str, str]] = Counter()
    by_relationship: Counter[str] = Counter()
    by_intent: Counter[str] = Counter()
    for index, row in enumerate(rows, start=1):
        relationship = normalize_relationship_type(str(row.get("relationship_type", "")))
        intent = normalize_intent_label(str(row.get("intent_type", "")))
        incoming = str(row.get("incoming", "")).strip()
        reply = str(row.get("my_reply", "")).strip()
        by_relationship[relationship] += 1
        by_intent[intent] += 1
        if relationship not in VALID_RELATIONSHIPS:
            errors.append(f"row {index}: invalid relationship {relationship}")
        if intent not in VALID_INTENTS:
            errors.append(f"row {index}: invalid intent {intent}")
        if not incoming or not reply:
            errors.append(f"row {index}: empty incoming/reply")
        if len(reply.split()) > 18 or len(reply) > 120:
            errors.append(f"row {index}: reply too long")
        lowered = f" {reply.casefold()} "
        if any(term in lowered for term in BANNED) or contains_suspicious_phrase(reply):
            errors.append(f"row {index}: banned phrase in reply")
        if row.get("is_synthetic") is not True:
            errors.append(f"row {index}: is_synthetic must be true")
        if row.get("source") != "generated_safe_template":
            errors.append(f"row {index}: source must be generated_safe_template")
        if row.get("style_authority") != "low":
            errors.append(f"row {index}: style_authority must be low")
        seen_pairs[(incoming.casefold(), reply.casefold())] += 1
    duplicate_heavy = sum(1 for count in seen_pairs.values() if count > 80)
    if duplicate_heavy:
        errors.append(f"duplicate-heavy pairs above threshold: {duplicate_heavy}")

    correction = {
        "relationship_type": "close_friend",
        "incoming": "you coming later?",
        "context": [],
        "my_reply": "yh what time",
        "_source": "correction",
        "intent_type": "planning",
    }
    synthetic = {
        "relationship_type": "close_friend",
        "incoming": "you coming later?",
        "context": [],
        "my_reply": "maybe later",
        "is_synthetic": True,
        "source": "generated_safe_template",
        "intent_type": "planning",
    }
    ranked = score_rows(
        [synthetic, correction],
        incoming="you coming later?",
        context=[],
        relationship_type="close_friend",
        intent_type="planning",
    )
    if not ranked or ranked[0].source != "correction":
        errors.append("correction did not outrank synthetic fallback")
    if errors:
        for error in errors[:50]:
            print(f"ERROR {error}")
        raise SystemExit(1)
    print(f"rows={len(rows)}")
    print(f"relationships={dict(sorted(by_relationship.items()))}")
    print(f"intents={dict(sorted(by_intent.items()))}")
    print("synthetic_outranks_corrections=false")


if __name__ == "__main__":
    main()
