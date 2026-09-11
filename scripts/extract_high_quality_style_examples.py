from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any

from apps.training.whatsapp_importer import import_training_dir
from libs.drafting.intents import classify_intent, normalize_intent_label, normalize_text
from libs.drafting.training_data import (
    load_corrections,
    load_jsonl,
    normalize_relationship_type,
    redact_private_text,
    row_quality_issues,
)


DEFAULT_OWNER_ALIASES = ["Alex", "Owner", "me", "you"]
OUTPUT_FILE = "high_quality_style_examples.jsonl"
SUMMARY_FILE = "high_quality_style_summary.json"

STYLE_MARKERS = (
    " u",
    " ur",
    "icl",
    "lowk",
    "yh",
    "nah",
    "bro",
    "dawg",
    "lool",
    "lol",
    "wth",
    "wbu",
    "wby",
    "😭",
    "💀",
)

BLOCKED_LOW_VALUE_REPLIES = {
    "ok",
    "okay",
    "k",
    "same icl",
    "same just chilling",
    "fair just chilling too",
    "calm",
    "<media omitted>",
}

SENSITIVE_TERMS = (
    "password",
    "passcode",
    "otp",
    "sort code",
    "bank card",
    "card number",
    "address is",
    "postcode",
    "api key",
    "secret key",
)


@dataclass
class ExtractedStyleExample:
    relationship_type: str
    incoming: str
    context: list[str]
    my_reply: str
    intent_type: str
    source: str
    source_kind: str
    quality_score: float
    quality_signals: list[str]
    style_authority: str
    contains_sensitive: bool = False

    def to_row(self) -> dict[str, Any]:
        return {
            "relationship_type": self.relationship_type,
            "incoming": self.incoming,
            "context": self.context,
            "my_reply": self.my_reply,
            "intent_type": self.intent_type,
            "source": self.source,
            "source_kind": self.source_kind,
            "quality_score": round(self.quality_score, 3),
            "quality_signals": self.quality_signals,
            "style_authority": self.style_authority,
            "contains_sensitive": self.contains_sensitive,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        }


def extract_high_quality_examples(
    *,
    chat_dir: Path,
    training_messages_dir: Path,
    owner_aliases: list[str] | None = None,
    limit: int = 2500,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    aliases = [*(owner_aliases or []), *DEFAULT_OWNER_ALIASES]
    candidates: list[ExtractedStyleExample] = []
    candidates.extend(_extract_from_txt_chats(chat_dir, aliases))
    candidates.extend(_extract_from_training_jsonl(training_messages_dir))

    unique: dict[tuple[str, str, str], ExtractedStyleExample] = {}
    for candidate in candidates:
        key = (
            normalize_text(candidate.relationship_type),
            normalize_text(candidate.incoming),
            normalize_text(candidate.my_reply),
        )
        if key not in unique or candidate.quality_score > unique[key].quality_score:
            unique[key] = candidate

    filtered = [
        item
        for item in unique.values()
        if item.quality_score >= 0.58 and not item.contains_sensitive
    ]
    filtered.sort(key=lambda item: (item.quality_score, item.style_authority == "high"), reverse=True)
    rows = [item.to_row() for item in filtered[:limit]]
    return rows, _build_summary(rows, total_candidates=len(candidates), total_unique=len(unique))


def write_high_quality_examples(
    *,
    chat_dir: Path,
    training_messages_dir: Path,
    out_dir: Path,
    owner_aliases: list[str] | None = None,
    limit: int = 2500,
) -> dict[str, Any]:
    rows, summary = extract_high_quality_examples(
        chat_dir=chat_dir,
        training_messages_dir=training_messages_dir,
        owner_aliases=owner_aliases,
        limit=limit,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    examples_path = out_dir / OUTPUT_FILE
    summary_path = out_dir / SUMMARY_FILE
    examples_path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return {
        **summary,
        "examples_path": str(examples_path),
        "summary_path": str(summary_path),
    }


def _extract_from_txt_chats(chat_dir: Path, owner_aliases: list[str]) -> list[ExtractedStyleExample]:
    if not chat_dir.exists():
        return []
    examples: list[ExtractedStyleExample] = []
    for chat in import_training_dir(chat_dir, owner_aliases):
        relationship = _relationship_for_contact(chat.contact_name)
        messages = chat.messages
        for index, message in enumerate(messages):
            if message.sender_me:
                continue
            incoming = _safe_text(message.text)
            owner_replies: list[str] = []
            next_index = index + 1
            while next_index < len(messages) and messages[next_index].sender_me:
                owner_replies.append(_safe_text(messages[next_index].text))
                next_index += 1
                if len(owner_replies) >= 2:
                    break
            reply = _safe_text(" ".join(item for item in owner_replies if item))
            context = [
                _safe_text(prior.text)
                for prior in messages[max(0, index - 4) : index]
                if _safe_text(prior.text)
            ][-4:]
            example = _candidate_from_parts(
                relationship_type=relationship,
                incoming=incoming,
                context=context,
                reply=reply,
                source="whatsapp_txt",
                source_kind="chat_log",
                base_authority="medium",
            )
            if example is not None:
                examples.append(example)
    return examples


def _extract_from_training_jsonl(training_messages_dir: Path) -> list[ExtractedStyleExample]:
    if not training_messages_dir.exists():
        return []
    examples: list[ExtractedStyleExample] = []
    for row in load_corrections(training_messages_dir):
        example = _candidate_from_parts(
            relationship_type=str(row.get("relationship_type") or "unknown"),
            incoming=str(row.get("incoming") or ""),
            context=[str(item) for item in row.get("context", []) if str(item).strip()] if isinstance(row.get("context"), list) else [],
            reply=str(row.get("user_final_reply") or ""),
            source=str(row.get("source") or "correction"),
            source_kind="correction",
            base_authority="high",
        )
        if example is not None:
            examples.append(example)

    for path in sorted(training_messages_dir.glob("*.jsonl")):
        if path.name in {"corrections.jsonl", "generated_safe_templates.jsonl", OUTPUT_FILE}:
            continue
        if path.parent.name == "quarantine":
            continue
        for row in load_jsonl(path):
            if row.get("is_synthetic") or "synthetic" in str(row.get("notes") or "").casefold():
                continue
            example = _candidate_from_parts(
                relationship_type=str(row.get("relationship_type") or "unknown"),
                incoming=str(row.get("incoming") or ""),
                context=[str(item) for item in row.get("context", []) if str(item).strip()] if isinstance(row.get("context"), list) else [],
                reply=str(row.get("my_reply") or ""),
                source="training_jsonl",
                source_kind=path.stem,
                base_authority="medium",
            )
            if example is not None:
                examples.append(example)
    return examples


def _candidate_from_parts(
    *,
    relationship_type: str,
    incoming: str,
    context: list[str],
    reply: str,
    source: str,
    source_kind: str,
    base_authority: str,
) -> ExtractedStyleExample | None:
    relationship = normalize_relationship_type(relationship_type)
    if _contains_sensitive(incoming) or _contains_sensitive(reply):
        return None
    safe_incoming = _safe_text(incoming)
    safe_reply = _safe_text(reply)
    safe_context = [_safe_text(item) for item in context if _safe_text(item) and not _contains_sensitive(item)][-4:]
    if not safe_incoming or not safe_reply:
        return None
    row = {
        "relationship_type": relationship,
        "incoming": safe_incoming,
        "context": safe_context,
        "my_reply": safe_reply,
    }
    allow_flirty = relationship == "romantic_interest"
    quality_issues = row_quality_issues(row, allow_flirty=allow_flirty, relationship_type=relationship)
    score, signals = _quality_score(
        incoming=safe_incoming,
        reply=safe_reply,
        context=safe_context,
        source_kind=source_kind,
        base_authority=base_authority,
        quality_issues=quality_issues,
    )
    if score < 0.35:
        return None
    return ExtractedStyleExample(
        relationship_type=relationship,
        incoming=safe_incoming,
        context=safe_context,
        my_reply=safe_reply,
        intent_type=normalize_intent_label(classify_intent(safe_incoming, safe_context)),
        source=source,
        source_kind=source_kind,
        quality_score=score,
        quality_signals=signals,
        style_authority="high" if score >= 0.82 or base_authority == "high" else "medium",
        contains_sensitive=False,
    )


def _quality_score(
    *,
    incoming: str,
    reply: str,
    context: list[str],
    source_kind: str,
    base_authority: str,
    quality_issues: list[str],
) -> tuple[float, list[str]]:
    normalized_reply = normalize_text(reply).strip(" .?!")
    signals: list[str] = []
    score = 0.56
    if base_authority == "high":
        score += 0.18
        signals.append("approved_or_corrected")
    if source_kind == "chat_log":
        score += 0.08
        signals.append("real_chat_log")
    word_count = len(reply.split())
    if 2 <= word_count <= 16:
        score += 0.1
        signals.append("good_length")
    if any(marker in f" {normalize_text(reply)}" for marker in STYLE_MARKERS):
        score += 0.09
        signals.append("owner_style_marker")
    if any(char.isupper() for char in reply) and len(reply) > 4:
        score += 0.03
        signals.append("expressive_caps")
    if incoming.endswith("?") and len(reply.split()) <= 18:
        score += 0.04
        signals.append("direct_question_response")
    if context:
        score += 0.03
        signals.append("has_context")
    if normalized_reply in BLOCKED_LOW_VALUE_REPLIES:
        score -= 0.45
        signals.append("blocked_low_value_reply")
    if "very_long_reply" in quality_issues:
        score -= 0.14
    if "reply_too_long" in quality_issues:
        score -= 0.22
    if "generic_ai_reply" in quality_issues:
        score -= 0.35
    if "export_artifact_reply" in quality_issues or "export_artifact_incoming" in quality_issues:
        score -= 0.5
    if "suspicious_phrase" in quality_issues:
        score -= 0.5
    if normalize_text(incoming) == normalize_text(reply):
        score -= 0.5
    return max(0.0, min(1.0, score)), sorted(dict.fromkeys([*signals, *quality_issues]))


def _build_summary(rows: list[dict[str, Any]], *, total_candidates: int, total_unique: int) -> dict[str, Any]:
    relationship_counts = Counter(str(row.get("relationship_type") or "unknown") for row in rows)
    intent_counts = Counter(str(row.get("intent_type") or "unknown") for row in rows)
    starters = Counter()
    tokens = Counter()
    reply_lengths: list[int] = []
    for row in rows:
        reply = str(row.get("my_reply") or "")
        words = re.findall(r"[a-zA-Z']+", reply.casefold())
        if words:
            starters[words[0]] += 1
            tokens.update(words)
        reply_lengths.append(len(reply.split()))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_candidates": total_candidates,
        "total_unique": total_unique,
        "selected_count": len(rows),
        "relationship_counts": dict(relationship_counts.most_common()),
        "intent_counts": dict(intent_counts.most_common(12)),
        "top_reply_starters": dict(starters.most_common(20)),
        "top_style_tokens": {
            token: count
            for token, count in tokens.most_common(30)
            if token not in {"the", "and", "to", "a", "it", "of", "in"}
        },
        "average_reply_words": round(sum(reply_lengths) / max(len(reply_lengths), 1), 2),
        "privacy_note": "Rows are redacted and sensitive examples are excluded before writing.",
    }


def _relationship_for_contact(contact_name: str) -> str:
    normalized = normalize_text(contact_name)
    if any(term in normalized for term in ("taylor", "posh", "رشا")):
        return "romantic_interest"
    if any(term in normalized for term in ("riley", "jordan", "ish")):
        return "close_friend"
    return "casual_friend"


def _safe_text(text: str) -> str:
    text = redact_private_text(str(text or ""))
    text = re.sub(r"\b\d{4,8}\b", "[redacted-code]", text)
    text = re.sub(r"\b(?:\d[ -]*?){13,19}\b", "[redacted-card]", text)
    text = re.sub(r"(?i)\b(password|otp|passcode|pin|api key|secret key)\s*[:=]?\s*\S+", r"\1 [redacted]", text)
    return re.sub(r"\s+", " ", text).strip()


def _contains_sensitive(text: str) -> bool:
    normalized = normalize_text(text)
    if any(term in normalized for term in SENSITIVE_TERMS):
        return True
    if re.search(r"\b\d{4,8}\b", text):
        return True
    if re.search(r"\b(?:\d[ -]*?){13,19}\b", text):
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat-dir", type=Path, default=Path("training"))
    parser.add_argument("--training-messages-dir", type=Path, default=Path("data/training_messages"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/training_messages"))
    parser.add_argument("--owner", action="append", default=[])
    parser.add_argument("--limit", type=int, default=2500)
    args = parser.parse_args()
    summary = write_high_quality_examples(
        chat_dir=args.chat_dir,
        training_messages_dir=args.training_messages_dir,
        out_dir=args.out_dir,
        owner_aliases=args.owner,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
