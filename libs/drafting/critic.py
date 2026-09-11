from __future__ import annotations

import re
from typing import Protocol

from pydantic import BaseModel

from libs.drafting.intents import answers_greeting, contains_suspicious_phrase, looks_overfamiliar, normalize_intent_label, normalize_text
from libs.drafting.training_data import normalize_relationship_type

BLOCKED_AUTO_SEND_RELATIONSHIPS = {"romantic_interest", "family", "university", "professional", "unknown"}
GENERIC_PHRASES = (
    "hope you're doing well",
    "hope youre doing well",
    "that sounds great",
    "i completely understand",
    "kind regards",
    "looking forward to hearing from you",
    "happy to help",
    "i hear you",
    "nah i get u",
    "nah i get you",
    "yeah i get u",
    "yeah i get you",
    "yeah i get u, keep going",
    "let me know",
)


class CandidateLike(Protocol):
    text: str
    sequence: list[str]
    risk_flags: list[str]
    auto_send_allowed: bool


class CandidateCritique(BaseModel):
    style_match_score: int
    relevance_score: int
    risk_score: int
    too_formal: bool = False
    too_long: bool = False
    fake_sounding: bool = False
    missing_context: bool = False
    does_not_answer: bool = False
    social_risk_reason: str | None = None
    final_decision: str = "review"
    blocked_reason: str | None = None


def critique_candidate(
    candidate: CandidateLike,
    *,
    latest_message: str,
    relationship_type: str,
    incoming_intent: str = "other",
    style_score: float,
    relevance_score: float,
    explicit_flirt_allowed: bool = False,
    selected_mode: str = "review",
) -> CandidateCritique:
    relationship_type = normalize_relationship_type(relationship_type)
    incoming_intent = normalize_intent_label(incoming_intent or "unknown")
    text = candidate.text.strip()
    normalized = text.casefold()
    words = re.findall(r"[A-Za-z0-9']+", text)
    style = int(round(max(0.0, min(style_score, 1.0)) * 100))
    relevance = int(round(max(0.0, min(relevance_score, 1.0)) * 100))
    risk = min(100, int(round(100 * min(1.0, 0.08 * len(candidate.risk_flags)))))

    too_formal = any(term in normalized for term in ("regarding", "concerning", "appreciate", "certainly", "kind regards"))
    fake_sounding = any(phrase in normalized for phrase in GENERIC_PHRASES)
    too_long = len(words) > 22 or len(text) > 160
    latest = latest_message.casefold().strip()
    missing_context = bool(re.fullmatch(r"(did you do it|you done|did u do it|what about it|that thing)\??", latest))
    greeting_incoming = incoming_intent == "greeting"
    overfamiliar = looks_overfamiliar(text)
    unsafe_phrase = contains_suspicious_phrase(text)
    does_not_answer = False
    if greeting_incoming:
        does_not_answer = not answers_greeting(text)
    elif latest.endswith("?"):
        does_not_answer = len(words) == 0 or len(words) > 18

    if greeting_incoming and not explicit_flirt_allowed and overfamiliar:
        risk += 40
    if unsafe_phrase and not explicit_flirt_allowed:
        risk += 35
    if greeting_incoming and not answers_greeting(text):
        risk += 25
    if relationship_type == "unknown" and overfamiliar:
        risk += 20

    if too_formal:
        style = min(style, 72)
        risk += 12
    if fake_sounding:
        style = min(style, 68)
        risk += 15
    if too_long:
        style = min(style, 80)
        risk += 8
    if missing_context:
        relevance = min(relevance, 72)
        risk += 10
    if does_not_answer:
        relevance = min(relevance, 70)
        risk += 20
    if relationship_type in BLOCKED_AUTO_SEND_RELATIONSHIPS:
        risk = max(risk, 26)

    risk = min(risk, 100)
    blocked: list[str] = []
    final_decision = "draft_only"
    if relationship_type in BLOCKED_AUTO_SEND_RELATIONSHIPS:
        blocked.append(f"{relationship_type}_never_auto_send")
        final_decision = "review"
    if risk > 25:
        blocked.append("risk_score_above_25")
        final_decision = "review"
    if style < 85:
        blocked.append("style_match_below_85")
        final_decision = "review"
    if relevance < 90:
        blocked.append("relevance_below_90")
        final_decision = "review"
    if too_formal:
        blocked.append("too_formal")
    if fake_sounding:
        blocked.append("fake_sounding")
    if missing_context:
        blocked.append("missing_context")
    if does_not_answer:
        blocked.append("does_not_answer")
    if greeting_incoming and overfamiliar and not explicit_flirt_allowed:
        blocked.append("overfamiliar_greeting")
        final_decision = "reject"
    if unsafe_phrase and not explicit_flirt_allowed:
        blocked.append("unsafe_or_flirty_phrase")
        final_decision = "reject"
    if relationship_type == "unknown" and overfamiliar:
        blocked.append("unknown_relationship_overfamiliar")
        final_decision = "reject"
    if "factual_claim" in candidate.risk_flags:
        blocked.append("claims_not_in_context")
        final_decision = "review"

    if final_decision != "reject" and not blocked and selected_mode == "auto_send":
        short_message = len(words) <= 14 and len(candidate.sequence) <= 2
        if (
            relationship_type in {"close_friend", "casual_friend"}
            and style >= 92
            and relevance >= 95
            and risk <= 10
            and short_message
        ):
            final_decision = "send"
        else:
            final_decision = "draft_only"
            if not short_message:
                blocked.append("message_not_short")

    return CandidateCritique(
        style_match_score=style,
        relevance_score=relevance,
        risk_score=risk,
        too_formal=too_formal,
        too_long=too_long,
        fake_sounding=fake_sounding,
        missing_context=missing_context,
        does_not_answer=does_not_answer,
        social_risk_reason=", ".join(candidate.risk_flags) if candidate.risk_flags else None,
        final_decision=final_decision,
        blocked_reason=", ".join(blocked) if blocked else None,
    )
