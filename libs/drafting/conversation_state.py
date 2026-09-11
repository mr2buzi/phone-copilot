from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

DEFAULT_CONVERSATION_STATE_PATH = Path("data/conversation_state.json")


class ContactConversationState(BaseModel):
    last_seen_message_hash: str | None = None
    last_incoming_text: str | None = None
    last_reply_sent: str | None = None
    last_draft: str | None = None
    recent_candidates: list[str] = Field(default_factory=list)
    recent_bot_replies: list[str] = Field(default_factory=list)
    recent_bot_claims: list[str] = Field(default_factory=list)
    recent_bot_stances: list[str] = Field(default_factory=list)
    recent_bot_questions: list[str] = Field(default_factory=list)
    recent_bot_questions_count: int = 0
    repeated_question_count: int = 0
    last_bot_claim: str | None = None
    last_bot_stance: str | None = None
    last_bot_question: str | None = None
    last_question_timestamp: str | None = None
    disputed_topic: str | None = None
    contradiction_risk: float = 0.0
    repeated_question_risk: float = 0.0
    last_action: str = "skipped"
    last_topic: str | None = None
    open_question_pending: bool = False
    last_updated: str | None = None
    send_count_last_hour: int = 0
    send_events_utc: list[str] = Field(default_factory=list)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp_path.replace(path)


def _normalize_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def normalize_question_intent(text: str) -> str:
    normalized = _normalize_text(text).strip("?!., ")
    normalized = normalized.replace("what you doing", "what u doing")
    normalized = normalized.replace("what are you doing", "what u doing")
    normalized = normalized.replace("what are u doing", "what u doing")
    normalized = normalized.replace("what u been doing", "what u doing")
    normalized = normalized.replace("what you been doing", "what u doing")
    if normalized in {"wyd", "wuu2"} or "what u doing" in normalized:
        return "what_u_doing"
    if "what u got in mind" in normalized or "what you got in mind" in normalized:
        return "what_u_got_in_mind"
    if "what happened" in normalized or "go on then" in normalized or "what now" in normalized:
        return "story_prompt"
    if "what u laughing at" in normalized or "what you laughing at" in normalized:
        return "what_laughing_at"
    return normalized


def _compact_text(text: str, *, limit: int = 80) -> str:
    clean = " ".join(str(text).split()).strip()
    return clean[:limit]


def _trim_recent(values: list[str], *, limit: int = 6) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(_compact_text(value))
    return deduped[:limit]


QUESTION_PROMPT_PHRASES = (
    "guess what",
    "what happened",
    "what now",
    "go on then",
    "what u doing",
    "what you doing",
    "what are you doing",
    "what are u doing",
    "what do you think",
    "what dyu think",
    "what do u think",
    "how come",
    "what about you",
    "what about u",
)


def is_question_like_text(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    if normalized.strip("?!., ") in {"wyd", "wuu2"}:
        return True
    if "?" in normalized:
        return True
    if normalized.startswith(("what ", "how ", "why ", "where ", "when ", "who ", "which ", "guess what")):
        return True
    return any(phrase in normalized for phrase in QUESTION_PROMPT_PHRASES)


def extract_question_text(reply: str) -> str | None:
    if not is_question_like_text(reply):
        return None
    return _compact_text(reply, limit=120)


def _count_question_replies(replies: list[str]) -> int:
    return sum(1 for reply in replies if is_question_like_text(reply))


def _infer_bot_stance(reply: str) -> str:
    normalized = _normalize_text(reply)
    if not normalized:
        return "neutral"
    playful_defense = (
        "nah",
        "how is that crazy",
        "how's that crazy",
        "ur dragging it",
        "you're dragging it",
        "youre dragging it",
        "bro what",
        "bro",
        "u said",
        "you said",
        "not crazy",
        "ain't crazy",
        "aint crazy",
        "it really isnt",
        "it really isn't",
        "nah it isnt",
        "nah it isn't",
        "it isnt",
        "it isn't",
        "yeah it isnt",
        "yeah it isn't",
    )
    concede = (
        "it really is",
        "it is",
        "you're right",
        "youre right",
        "fair enough",
        "yeah true",
        "yeah it is",
    )
    correction = ("my bad", "you're right", "youre right", "i meant", "i should've", "i should have")
    if any(term in normalized for term in correction):
        return "correcting"
    if any(term in normalized for term in playful_defense):
        return "defend"
    if any(term in normalized for term in concede):
        return "concede"
    if "?" in normalized:
        return "challenge"
    return "neutral"


def _infer_bot_claim(reply: str) -> str:
    normalized = _normalize_text(reply)
    if not normalized:
        return ""
    if any(term in normalized for term in ("come over", "come round", "come thru", "come through", "go out then", "do smth then", "do something then", "find smth to do")):
        return "invite_or_plan"
    if any(term in normalized for term in ("crazy", "dragging", "bro what", "how is that crazy", "not crazy")):
        return "banter_defense"
    if "?" in normalized:
        return "question"
    return _compact_text(reply)


def analyze_bot_reply(reply: str, previous_state: ContactConversationState | None = None) -> dict[str, Any]:
    claim = _infer_bot_claim(reply)
    stance = _infer_bot_stance(reply)
    question = extract_question_text(reply)
    question_intent = normalize_question_intent(question or "") if question else ""
    disputed_topic = previous_state.disputed_topic if previous_state and previous_state.disputed_topic else None
    if claim == "invite_or_plan":
        disputed_topic = "invite_or_plan"
    elif claim == "banter_defense":
        disputed_topic = disputed_topic or "banter_challenge"
    recent_questions = _count_question_replies([reply, *(previous_state.recent_bot_replies if previous_state else [])])
    repeated_question_risk = 0.0
    repeated_question_count = previous_state.repeated_question_count if previous_state else 0
    if question:
        previous_question = normalize_question_intent(previous_state.last_bot_question or "") if previous_state and previous_state.last_bot_question else ""
        recent_question_intents = [
            normalize_question_intent(item)
            for item in (previous_state.recent_bot_questions if previous_state else [])
            if item
        ]
        if previous_question and previous_question == question_intent:
            repeated_question_risk = 1.0
            repeated_question_count += 1
        elif question_intent and question_intent in recent_question_intents[:4]:
            repeated_question_risk = 1.0
            repeated_question_count += 1
        elif previous_state and previous_state.recent_bot_questions_count >= 2:
            repeated_question_risk = 0.7
    risk = 0.0
    if previous_state and previous_state.last_bot_stance:
        previous_stance = str(previous_state.last_bot_stance).strip().lower()
        if previous_stance and stance and previous_stance != stance:
            if stance != "correcting":
                risk = 1.0
    return {
        "last_bot_claim": claim,
        "last_bot_stance": stance,
        "last_bot_question": question,
        "disputed_topic": disputed_topic,
        "contradiction_risk": round(risk, 3),
        "recent_bot_questions_count": recent_questions,
        "repeated_question_count": repeated_question_count,
        "last_question_timestamp": datetime.now(timezone.utc).isoformat() if question else (previous_state.last_question_timestamp if previous_state else None),
        "repeated_question_risk": round(repeated_question_risk, 3),
    }


def _message_hash(text: str) -> str:
    return hashlib.sha1(_normalize_text(text).encode("utf-8")).hexdigest()


class ConversationStateStore:
    def __init__(self, path: Path = DEFAULT_CONVERSATION_STATE_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}", encoding="utf-8")

    def load(self) -> dict[str, ContactConversationState]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}
        state: dict[str, ContactConversationState] = {}
        for name, raw in payload.items():
            if not isinstance(raw, dict):
                continue
            try:
                state[str(name)] = ContactConversationState(**raw)
            except Exception:
                continue
        return state

    def save(self, state: dict[str, ContactConversationState]) -> None:
        _atomic_write_json(self.path, {name: item.model_dump(mode="json") for name, item in state.items()})

    def get_contact_state(self, contact_name: str) -> ContactConversationState:
        state = self.load()
        return state.get(contact_name, ContactConversationState())

    def update_contact_state(self, contact_name: str, patch: dict[str, Any]) -> ContactConversationState:
        state = self.load()
        current = state.get(contact_name, ContactConversationState())
        merged = current.model_copy(update=patch)
        merged.last_updated = datetime.now(timezone.utc).isoformat()
        state[contact_name] = merged
        self.save(state)
        return merged

    def has_new_incoming(self, contact_name: str, captured_messages: list[str]) -> bool:
        if not captured_messages:
            return False
        state = self.get_contact_state(contact_name)
        latest_text = captured_messages[-1]
        return _message_hash(latest_text) != (state.last_seen_message_hash or "")

    def record_candidates(self, contact_name: str, candidates: list[str]) -> ContactConversationState:
        current = self.get_contact_state(contact_name)
        recent_candidates = [
            *[candidate for candidate in candidates if candidate.strip()],
            *current.recent_candidates,
        ]
        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in recent_candidates:
            key = _normalize_text(candidate)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return self.update_contact_state(
            contact_name,
            {"recent_candidates": deduped[:12], "last_action": "drafted"},
        )

    def record_bot_reply(self, contact_name: str, reply: str) -> ContactConversationState:
        current = self.get_contact_state(contact_name)
        analysis = analyze_bot_reply(reply, current)
        recent_bot_replies = _trim_recent([reply, *current.recent_bot_replies], limit=6)
        recent_bot_claims = _trim_recent([analysis["last_bot_claim"] or "", *current.recent_bot_claims], limit=6)
        recent_bot_stances = _trim_recent([analysis["last_bot_stance"] or "", *current.recent_bot_stances], limit=6)
        recent_bot_questions = _trim_recent(
            [analysis["last_bot_question"] or "", *current.recent_bot_questions],
            limit=8,
        )
        return self.update_contact_state(
            contact_name,
            {
                "last_reply_sent": reply,
                "recent_bot_replies": recent_bot_replies,
                "recent_bot_claims": recent_bot_claims,
                "recent_bot_stances": recent_bot_stances,
                "recent_bot_questions": recent_bot_questions,
                **analysis,
                "last_action": "bot_replied",
            },
        )

    def record_draft(self, contact_name: str, reply: str) -> ContactConversationState:
        return self.update_contact_state(
            contact_name,
            {
                "last_draft": reply,
                "last_action": "drafted",
            },
        )

    def record_send(self, contact_name: str, reply: str) -> ContactConversationState:
        current = self.get_contact_state(contact_name)
        send_events = [item for item in current.send_events_utc if item]
        now = datetime.now(timezone.utc).isoformat()
        send_events.append(now)
        recent = [
            item
            for item in send_events
            if (datetime.now(timezone.utc) - datetime.fromisoformat(item)).total_seconds() <= 3600
        ]
        analysis = analyze_bot_reply(reply, current)
        recent_bot_replies = _trim_recent([reply, *current.recent_bot_replies], limit=6)
        recent_bot_claims = _trim_recent([analysis["last_bot_claim"] or "", *current.recent_bot_claims], limit=6)
        recent_bot_stances = _trim_recent([analysis["last_bot_stance"] or "", *current.recent_bot_stances], limit=6)
        recent_bot_questions = _trim_recent(
            [analysis["last_bot_question"] or "", *current.recent_bot_questions],
            limit=8,
        )
        return self.update_contact_state(
            contact_name,
            {
                "last_reply_sent": reply,
                "recent_bot_replies": recent_bot_replies,
                "recent_bot_claims": recent_bot_claims,
                "recent_bot_stances": recent_bot_stances,
                "recent_bot_questions": recent_bot_questions,
                **analysis,
                "last_action": "sent",
                "send_events_utc": recent[-20:],
                "send_count_last_hour": len(recent[-20:]),
            },
        )

    def record_skip(self, contact_name: str, reason: str) -> ContactConversationState:
        return self.update_contact_state(contact_name, {"last_action": f"skipped:{reason}"})

    def set_last_seen(self, contact_name: str, latest_incoming_text: str) -> ContactConversationState:
        return self.update_contact_state(
            contact_name,
            {
                "last_seen_message_hash": _message_hash(latest_incoming_text),
                "last_incoming_text": latest_incoming_text,
            },
        )


def get_contact_state(contact_name: str, path: Path = DEFAULT_CONVERSATION_STATE_PATH) -> ContactConversationState:
    return ConversationStateStore(path).get_contact_state(contact_name)


def update_contact_state(
    contact_name: str,
    patch: dict[str, Any],
    path: Path = DEFAULT_CONVERSATION_STATE_PATH,
) -> ContactConversationState:
    return ConversationStateStore(path).update_contact_state(contact_name, patch)


def has_new_incoming(
    contact_name: str,
    captured_messages: list[str],
    path: Path = DEFAULT_CONVERSATION_STATE_PATH,
) -> bool:
    return ConversationStateStore(path).has_new_incoming(contact_name, captured_messages)


def record_candidates(
    contact_name: str,
    candidates: list[str],
    path: Path = DEFAULT_CONVERSATION_STATE_PATH,
) -> ContactConversationState:
    return ConversationStateStore(path).record_candidates(contact_name, candidates)


def record_draft(
    contact_name: str,
    reply: str,
    path: Path = DEFAULT_CONVERSATION_STATE_PATH,
) -> ContactConversationState:
    return ConversationStateStore(path).record_draft(contact_name, reply)


def record_send(
    contact_name: str,
    reply: str,
    path: Path = DEFAULT_CONVERSATION_STATE_PATH,
) -> ContactConversationState:
    return ConversationStateStore(path).record_send(contact_name, reply)


def record_skip(
    contact_name: str,
    reason: str,
    path: Path = DEFAULT_CONVERSATION_STATE_PATH,
) -> ContactConversationState:
    return ConversationStateStore(path).record_skip(contact_name, reason)
