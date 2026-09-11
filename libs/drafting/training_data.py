from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from libs.drafting.intents import classify_intent, contains_suspicious_phrase, looks_overfamiliar, normalize_intent_label, normalize_text, tokenize

logger = logging.getLogger(__name__)

RELATIONSHIP_TYPES = {
    "close_friend",
    "casual_friend",
    "romantic_interest",
    "family",
    "university",
    "professional",
    "unknown",
}

TRAINING_FILE_STEMS = {
    "close_friend": "close_friends",
    "casual_friend": "casual_friends",
    "romantic_interest": "romantic_interest",
    "family": "family",
    "university": "university",
    "professional": "professional",
}

REASON_BAD_VALUES = {"too formal", "too long", "not me", "missed context", "risky", "other"}
SELECTED_MODES = {"review", "auto_draft", "auto_send", "auto-draft", "auto-send"}
SUSPICIOUS_REPLY_PHRASES = (
    "baby",
    "keep talking like that",
    "youre trouble",
    "you're trouble",
    "x",
    "xx",
    "sexy",
    "babe",
    "love",
    "miss you",
    "come mine",
    "send pic",
    "wyd then",
    "naughty",
)

_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{6,}\d)(?!\w)")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")


def normalize_relationship_type(value: str | None) -> str:
    normalized = (value or "").strip().lower().replace("-", "_")
    return normalized if normalized in RELATIONSHIP_TYPES else "unknown"


def ensure_training_message_files(training_dir: Path) -> None:
    training_dir.mkdir(parents=True, exist_ok=True)
    for stem in list(TRAINING_FILE_STEMS.values()) + ["corrections"]:
        path = training_dir / f"{stem}.jsonl"
        if not path.exists():
            path.write_text("", encoding="utf-8")


def training_path_for_relationship(training_dir: Path, relationship_type: str) -> Path:
    relationship_type = normalize_relationship_type(relationship_type)
    stem = TRAINING_FILE_STEMS.get(relationship_type, "casual_friends")
    return training_dir / f"{stem}.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError:
            logger.warning("Skipping invalid JSONL row in %s:%s", path, line_number)
            continue
        if not isinstance(row, dict):
            logger.warning("Skipping non-object JSONL row in %s:%s", path, line_number)
            continue
        rows.append(row)
    return rows


def load_training_messages(training_dir: Path, relationship_type: str | None = None) -> list[dict[str, Any]]:
    ensure_training_message_files(training_dir)
    if relationship_type:
        paths = [training_path_for_relationship(training_dir, relationship_type)]
    else:
        paths = [training_dir / f"{stem}.jsonl" for stem in TRAINING_FILE_STEMS.values()]

    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in load_jsonl(path):
            normalized = normalize_relationship_type(str(row.get("relationship_type", relationship_type or "")))
            incoming = str(row.get("incoming", "")).strip()
            reply = str(row.get("my_reply", "")).strip()
            context = row.get("context", [])
            if not incoming or not reply:
                logger.warning("Skipping incomplete training row in %s", path)
                continue
            if not isinstance(context, list):
                context = []
            row["relationship_type"] = normalized
            row["incoming"] = redact_private_text(incoming)
            row["my_reply"] = redact_private_text(reply)
            row["context"] = [redact_private_text(str(item)) for item in context if str(item).strip()]
            row["intent_type"] = normalize_intent_label(classify_intent(incoming, row["context"]))
            rows.append(row)
    return rows


def load_corrections(training_dir: Path) -> list[dict[str, Any]]:
    ensure_training_message_files(training_dir)
    rows: list[dict[str, Any]] = []
    for row in load_jsonl(training_dir / "corrections.jsonl"):
        relationship_type = normalize_relationship_type(str(row.get("relationship_type", "")))
        incoming = str(row.get("incoming", "")).strip()
        final_reply = str(row.get("user_final_reply", "")).strip()
        if not incoming or not final_reply:
            logger.warning("Skipping incomplete correction row")
            continue
        context = row.get("context", [])
        if not isinstance(context, list):
            context = []
        row["relationship_type"] = relationship_type
        row["incoming"] = redact_private_text(incoming)
        row["user_final_reply"] = redact_private_text(final_reply)
        row["context"] = [redact_private_text(str(item)) for item in context if str(item).strip()]
        row["intent_type"] = normalize_intent_label(
            str(row.get("intent_type") or classify_intent(incoming, row["context"]))
        )
        rows.append(row)
    return rows


def load_all_training_rows(training_dir: Path) -> list[dict[str, Any]]:
    ensure_training_message_files(training_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted(training_dir.glob("*.jsonl")):
        if path.parent.name == "quarantine" or path.name == "corrections.jsonl":
            continue
        for row in load_jsonl(path):
            if not isinstance(row, dict):
                continue
            incoming = str(row.get("incoming", "")).strip()
            reply = str(row.get("my_reply", "")).strip()
            if not incoming or not reply:
                continue
            context = row.get("context", [])
            if not isinstance(context, list):
                context = []
            row["relationship_type"] = normalize_relationship_type(str(row.get("relationship_type", "")))
            row["incoming"] = redact_private_text(incoming)
            row["my_reply"] = redact_private_text(reply)
            row["context"] = [redact_private_text(str(item)) for item in context if str(item).strip()]
            row["intent_type"] = normalize_intent_label(
                str(row.get("intent_type") or classify_intent(incoming, row["context"]))
            )
            rows.append(row)
    return rows


def append_correction(training_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    ensure_training_message_files(training_dir)
    relationship_type = normalize_relationship_type(str(payload.get("relationship_type", "")))
    reason_bad = str(payload.get("reason_bad", "other")).strip().lower()
    selected_mode = str(payload.get("selected_mode", "review")).strip().lower().replace("-", "_")
    incoming = redact_private_text(str(payload.get("incoming", "")).strip())
    user_final_reply = redact_private_text(str(payload.get("user_final_reply", "")).strip())
    bad_ai_reply = redact_private_text(str(payload.get("bad_ai_reply", "")).strip())
    context_raw = payload.get("context", [])
    context = context_raw if isinstance(context_raw, list) else []
    intent_type = normalize_intent_label(str(payload.get("intent_type") or classify_intent(incoming, context)))

    if relationship_type == "unknown" and str(payload.get("relationship_type", "")).strip().lower() not in RELATIONSHIP_TYPES:
        raise ValueError("Invalid relationship_type.")
    if reason_bad not in REASON_BAD_VALUES:
        raise ValueError("Invalid reason_bad.")
    if selected_mode not in {"review", "auto_draft", "auto_send"}:
        raise ValueError("Invalid selected_mode.")
    if not incoming:
        raise ValueError("incoming must not be empty.")
    if not user_final_reply:
        raise ValueError("user_final_reply must not be empty.")
    source = str(payload.get("source") or "correction")
    feedback_label = str(payload.get("feedback_label", "")).strip().lower()
    style_authority = str(payload.get("style_authority") or "high")

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "relationship_type": relationship_type,
        "intent_type": intent_type,
        "contact_name": redact_private_text(str(payload.get("contact_name", "")).strip()),
        "incoming": incoming,
        "context": [redact_private_text(str(item)) for item in context if str(item).strip()],
        "bad_ai_reply": bad_ai_reply,
        "user_final_reply": user_final_reply,
        "reason_bad": reason_bad,
        "selected_mode": selected_mode,
        "feedback_label": feedback_label,
        "parameters": payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {},
        "source": source,
        "style_authority": style_authority,
        "approved_by_user": bool(payload.get("approved_by_user", False)),
        "is_correction": True,
        "is_synthetic": False,
    }
    with (training_dir / "corrections.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return row


def redact_private_text(text: str) -> str:
    redacted = _EMAIL_RE.sub("[email]", text)
    redacted = _PHONE_RE.sub("[phone]", redacted)
    return redacted


def row_quality_issues(
    row: dict[str, Any],
    *,
    allow_flirty: bool = False,
    relationship_type: str | None = None,
) -> list[str]:
    issues: list[str] = []
    incoming = str(row.get("incoming") or row.get("user_final_reply") or "").strip()
    reply = str(row.get("my_reply") or row.get("user_final_reply") or "").strip()
    context = row.get("context", [])
    if not isinstance(context, list):
        context = []

    if not incoming:
        issues.append("empty_incoming")
    if not reply:
        issues.append("empty_reply")
    if incoming and reply and normalize_text(incoming) == normalize_text(reply):
        issues.append("incoming_equals_reply")
    if len(reply) > 160 and normalize_relationship_type(relationship_type or str(row.get("relationship_type", ""))) not in {"professional", "university"}:
        issues.append("reply_too_long")
    if len(reply) > 120:
        issues.append("very_long_reply")
    if looks_overfamiliar(reply) and not allow_flirty:
        issues.append("unsafe_flirty_reply")
    if any(term in normalize_text(reply) for term in ("hope you're doing well", "kind regards", "looking forward")):
        issues.append("generic_ai_reply")
    if any(term in normalize_text(reply) for term in ("<media omitted>", "you deleted this message")):
        issues.append("export_artifact_reply")
    if any(term in normalize_text(incoming) for term in ("<media omitted>", "you deleted this message")):
        issues.append("export_artifact_incoming")
    if contains_suspicious_phrase(reply):
        issues.append("suspicious_phrase")
    if context and any(normalize_text(str(item)) == normalize_text(reply) for item in context):
        issues.append("reply_repeats_context")
    if len(tokenize(reply)) <= 1 and len(reply) > 12:
        issues.append("low_information_reply")
    return list(dict.fromkeys(issues))
