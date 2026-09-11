from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from libs.drafting.intents import contains_suspicious_phrase, normalize_intent_label, normalize_text, tokenize
from libs.drafting.training_data import normalize_relationship_type

DEFAULT_RUBRIC_PATH = Path("data/style_rubric.json")
DEFAULT_CONFIG_PATH = Path("data/style_eval_config.json")

CORPORATE_PHRASES = (
    "regarding",
    "concerning",
    "in response to",
    "i appreciate",
    "happy to help",
    "i completely understand",
    "please let me know",
    "kind regards",
    "as per",
)
AMERICAN_POLISHED_PHRASES = (
    "awesome",
    "super excited",
    "reach out",
    "circle back",
    "touch base",
)
FORCED_SLANG = (
    "loool",
    "lmaooo",
    "famalam",
    "bruvvv",
    "my guy",
    "you're trouble",
    "youre trouble",
)
FLIRTY_PHRASES = (
    "baby",
    "babe",
    "sexy",
    "keep talking like that",
    "come mine",
    "miss you",
    "love you",
)
COMMITMENT_PHRASES = (
    "i'll come",
    "ill come",
    "yeah i'll come",
    "yeah ill come",
    "i can make it",
    "i'll be there",
    "ill be there",
    "on my way",
    "definitely",
    "for sure",
    "nah can't",
    "nah cant",
    "maybe later",
)
CLARIFICATION_PHRASES = (
    "what time",
    "where",
    "which",
    "what do you mean",
    "remind me",
    "wbu",
    "you?",
)


@dataclass(slots=True)
class StyleEvaluationInput:
    incoming: str
    context: list[str]
    candidate: str
    relationship_type: str = "unknown"
    intent_type: str = "unknown"
    contact_name: str = ""
    retrieved_examples: list[dict[str, Any]] | None = None
    style_profile: dict[str, Any] | None = None


def load_style_rubric(path: Path = DEFAULT_RUBRIC_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"weights": {}, "generic_bad_phrases": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"weights": {}, "generic_bad_phrases": []}
    return data if isinstance(data, dict) else {"weights": {}, "generic_bad_phrases": []}


def style_eval_cache_key(payload: dict[str, Any]) -> str:
    material = json.dumps(_json_safe(payload), ensure_ascii=True, sort_keys=True)
    return hashlib.sha1(material.encode("utf-8")).hexdigest()[:16]


def evaluate_candidate_style(payload: dict[str, Any] | StyleEvaluationInput, *, rubric_path: Path = DEFAULT_RUBRIC_PATH) -> dict[str, Any]:
    item = _coerce_input(payload)
    rubric = load_style_rubric(rubric_path)
    weights = rubric.get("weights", {}) if isinstance(rubric.get("weights"), dict) else {}
    generic_bad_phrases = [str(value).casefold() for value in rubric.get("generic_bad_phrases", []) if str(value).strip()]

    incoming_norm = normalize_text(item.incoming)
    candidate = " ".join(item.candidate.split()).strip()
    candidate_norm = normalize_text(candidate)
    relationship = normalize_relationship_type(item.relationship_type)
    intent = normalize_intent_label(item.intent_type or "unknown")
    examples = item.retrieved_examples or []
    notes: list[str] = []
    reject_reasons: list[str] = []

    scores = {
        "naturalness": 82,
        "owner_style": 72,
        "context_fit": 72,
        "length_fit": 88,
        "relationship_fit": 78,
        "intent_fit": 76,
        "non_cringe": 88,
        "no_invention": 90,
    }

    if not candidate_norm:
        reject_reasons.append("empty reply")
        for key in scores:
            scores[key] = 0
        return _result(scores, weights, reject_reasons, ["Write a non-empty reply."])

    word_count = len(candidate.split())
    if word_count > _target_max_words(intent, relationship):
        _penalize(scores, {"length_fit": 28, "naturalness": 12, "owner_style": 10})
        notes.append("Shorten the reply.")
    if word_count <= 5 and intent in {"greeting", "casual_checkin", "planning"}:
        _reward(scores, {"length_fit": 7, "naturalness": 5, "owner_style": 5})
    if candidate != candidate.lower() and relationship not in {"professional", "university"}:
        _penalize(scores, {"owner_style": 5, "naturalness": 4})
        notes.append("Use lowercase casual texting style where appropriate.")

    generic_hits = [phrase for phrase in generic_bad_phrases if phrase and phrase in candidate_norm]
    generic_hits.extend(phrase for phrase in CORPORATE_PHRASES if phrase in candidate_norm)
    if generic_hits:
        _penalize(scores, {"naturalness": 30, "owner_style": 28, "non_cringe": 24})
        reject_reasons.append("generic filler" if any("sounds good" in hit or "hope" in hit for hit in generic_hits) else "sounds like customer support")
        notes.append("Remove generic/corporate phrasing.")

    if any(phrase in candidate_norm for phrase in AMERICAN_POLISHED_PHRASES):
        _penalize(scores, {"naturalness": 16, "owner_style": 16, "non_cringe": 12})
        reject_reasons.append("too American")
    if any(phrase in candidate_norm for phrase in FORCED_SLANG):
        _penalize(scores, {"naturalness": 18, "owner_style": 18, "non_cringe": 26})
        reject_reasons.append("cringe forced slang")

    flirty_hits = [phrase for phrase in FLIRTY_PHRASES if phrase in candidate_norm]
    if flirty_hits and not _flirt_allowed(item.context, relationship, intent):
        _penalize(scores, {"relationship_fit": 35, "intent_fit": 24, "non_cringe": 40, "no_invention": 22})
        reject_reasons.append("too flirty without context")
        notes.append("Use a non-flirty reply for this relationship/context.")
    elif contains_suspicious_phrase(candidate) and relationship != "romantic_interest":
        _penalize(scores, {"relationship_fit": 28, "non_cringe": 30})
        reject_reasons.append("too flirty without context")

    commitment_hits = [phrase for phrase in COMMITMENT_PHRASES if phrase in candidate_norm]
    if commitment_hits and intent in {"planning", "availability", "unknown", "greeting"}:
        _penalize(scores, {"context_fit": 24, "intent_fit": 20, "no_invention": 42})
        reject_reasons.append("invents availability or facts")
        notes.append("Ask for the missing detail instead of inventing availability.")

    if _does_not_answer(incoming_norm, candidate_norm, intent):
        _penalize(scores, {"context_fit": 35, "intent_fit": 22, "naturalness": 10})
        reject_reasons.append("does not answer the message")
        notes.append("Answer the latest incoming message directly.")

    if intent == "greeting":
        if candidate_norm in {"yo", "heyy", "u good", "what u saying", "you good"}:
            _reward(scores, {"context_fit": 18, "intent_fit": 18, "owner_style": 18, "non_cringe": 10})
        elif word_count > 4:
            _penalize(scores, {"length_fit": 18, "intent_fit": 18})
            notes.append("Keep greeting replies extremely short.")
    if intent in {"planning", "availability"} and any(phrase in candidate_norm for phrase in CLARIFICATION_PHRASES):
        _reward(scores, {"context_fit": 15, "intent_fit": 15, "no_invention": 8})
    if intent in {"planning", "availability"} and candidate_norm in {"yh what time", "where u lot going", "what time you thinking", "whos going", "who's going", "not sure yet what time"}:
        _reward(scores, {"naturalness": 4, "owner_style": 8, "context_fit": 6, "intent_fit": 6})
    if relationship in {"professional", "university"}:
        if any(term in candidate_norm for term in ("yo", "yh", "calm", "u lot", "wbu", "lool", "baby")):
            _penalize(scores, {"relationship_fit": 35, "non_cringe": 20})
            reject_reasons.append("too casual for professional context")
        else:
            _reward(scores, {"relationship_fit": 12, "intent_fit": 8})

    similarity_bonus, matched_source = _retrieval_style_bonus(candidate_norm, examples, relationship, intent)
    if similarity_bonus:
        _reward(scores, {"owner_style": similarity_bonus, "relationship_fit": min(10, similarity_bonus), "intent_fit": min(10, similarity_bonus)})
        notes.append(f"Matches retrieved {matched_source} style pattern.")
    elif examples:
        _penalize(scores, {"owner_style": 8})
        notes.append("Move closer to high-authority retrieved examples.")

    profile = item.style_profile or {}
    avg_words = _profile_average_words(profile)
    if avg_words and word_count > max(6, avg_words * 1.8) and relationship not in {"professional", "university"}:
        _penalize(scores, {"length_fit": 12, "owner_style": 10})
        notes.append("Reply is longer than the style profile suggests.")

    hard_reject = bool(reject_reasons) and (
        "too flirty without context" in reject_reasons
        or "invents availability or facts" in reject_reasons
        or "does not answer the message" in reject_reasons
        or "generic filler" in reject_reasons
    )
    if candidate_norm == "yeah i saw that" and intent == "greeting":
        hard_reject = True
        reject_reasons.append("does not answer the message")
    result = _result(scores, weights, reject_reasons, notes)
    result["hard_reject"] = hard_reject or result["overall_score"] < 45
    result["cache_key"] = style_eval_cache_key(_input_to_dict(item))
    return result


def _coerce_input(payload: dict[str, Any] | StyleEvaluationInput) -> StyleEvaluationInput:
    if isinstance(payload, StyleEvaluationInput):
        return payload
    context = payload.get("context", [])
    return StyleEvaluationInput(
        incoming=str(payload.get("incoming", "")),
        context=[str(item) for item in context] if isinstance(context, list) else [],
        candidate=str(payload.get("candidate", "")),
        relationship_type=str(payload.get("relationship_type", "unknown")),
        intent_type=str(payload.get("intent_type", "unknown")),
        contact_name=str(payload.get("contact_name", "")),
        retrieved_examples=payload.get("retrieved_examples") if isinstance(payload.get("retrieved_examples"), list) else [],
        style_profile=payload.get("style_profile") if isinstance(payload.get("style_profile"), dict) else {},
    )


def _result(scores: dict[str, int], weights: dict[str, Any], reject_reasons: list[str], notes: list[str]) -> dict[str, Any]:
    normalized = {key: max(0, min(100, int(value))) for key, value in scores.items()}
    weighted_total = 0.0
    total_weight = 0.0
    for key, value in normalized.items():
        weight = float(weights.get(key, 1.0) or 1.0)
        weighted_total += value * weight
        total_weight += weight
    overall = int(round(weighted_total / total_weight)) if total_weight else int(round(sum(normalized.values()) / len(normalized)))
    return {
        "overall_score": max(0, min(100, overall)),
        **normalized,
        "hard_reject": bool(reject_reasons),
        "reject_reasons": list(dict.fromkeys(reject_reasons)),
        "improvement_notes": list(dict.fromkeys(notes)) or ["No deterministic improvement needed."],
    }


def _input_to_dict(item: StyleEvaluationInput) -> dict[str, Any]:
    return {
        "incoming": item.incoming,
        "context": item.context,
        "candidate": item.candidate,
        "relationship_type": item.relationship_type,
        "intent_type": item.intent_type,
        "contact_name": item.contact_name,
        "retrieved_examples": [_example_to_mapping(example) for example in (item.retrieved_examples or [])],
        "style_profile": _json_safe(item.style_profile or {}),
    }


def _target_max_words(intent: str, relationship: str) -> int:
    if relationship in {"professional", "university"}:
        return 22
    if intent == "greeting":
        return 5
    if intent in {"planning", "availability", "casual_checkin"}:
        return 8
    if intent in {"argument", "emotional"}:
        return 16
    return 14


def _penalize(scores: dict[str, int], penalties: dict[str, int]) -> None:
    for key, value in penalties.items():
        scores[key] = max(0, scores.get(key, 0) - value)


def _reward(scores: dict[str, int], rewards: dict[str, int]) -> None:
    for key, value in rewards.items():
        scores[key] = min(100, scores.get(key, 0) + value)


def _flirt_allowed(context: list[str], relationship: str, intent: str) -> bool:
    if relationship != "romantic_interest" or intent == "greeting":
        return False
    blob = normalize_text(" ".join(context[-4:]))
    return any(term in blob for term in ("baby", "babe", "cute", "pretty", "miss u", "miss you", "kiss"))


def _does_not_answer(incoming_norm: str, candidate_norm: str, intent: str) -> bool:
    if not incoming_norm:
        return False
    if intent == "greeting":
        return candidate_norm not in {"yo", "heyy", "u good", "what u saying", "you good", "all good", "hello", "hey"} and not any(
            token in candidate_norm for token in ("yo", "hey", "good", "saying")
        )
    if "?" in incoming_norm or intent in {"planning", "availability", "simple_question", "professional"}:
        return not (
            any(token in candidate_norm for token in ("yes", "yeah", "yh", "no", "nah", "what", "where", "when", "which", "send", "check", "confirm", "sure", "not sure"))
            or bool(set(tokenize(incoming_norm)) & set(tokenize(candidate_norm)))
        )
    return False


def _retrieval_style_bonus(candidate_norm: str, examples: list[dict[str, Any]], relationship: str, intent: str) -> tuple[int, str]:
    best = 0.0
    best_source = ""
    for example in examples[:8]:
        example_map = _example_to_mapping(example)
        reply = normalize_text(str(example_map.get("my_reply") or example_map.get("user_final_reply") or ""))
        if not reply:
            continue
        metadata = example_map.get("metadata") if isinstance(example_map.get("metadata"), dict) else example_map
        authority = str(metadata.get("style_authority") or ("high" if example_map.get("source") == "correction" else "medium")).lower()
        source = str(example_map.get("source") or metadata.get("source") or metadata.get("_source") or "training")
        same_relationship = normalize_relationship_type(str(example_map.get("relationship_type", ""))) == relationship
        same_intent = normalize_intent_label(str(example_map.get("intent_type", ""))) == intent
        ratio = SequenceMatcher(None, candidate_norm, reply).ratio()
        token_overlap = len(set(tokenize(candidate_norm)) & set(tokenize(reply))) / max(1, len(set(tokenize(candidate_norm)) | set(tokenize(reply))))
        score = max(ratio, token_overlap)
        if same_relationship:
            score += 0.08
        if same_intent:
            score += 0.08
        if authority == "high" or source in {"correction", "approved_auto_style_improvement"}:
            score += 0.14
        elif authority == "low" or source == "synthetic":
            score -= 0.1
        if score > best:
            best = score
            best_source = source
    if best >= 0.82:
        return 18, best_source or "example"
    if best >= 0.68:
        return 11, best_source or "example"
    if best >= 0.52:
        return 5, best_source or "example"
    return 0, ""


def _profile_average_words(profile: dict[str, Any]) -> float | None:
    for key in ("average_message_length", "avg_words", "average_words"):
        value = profile.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            return number
    return None


def _example_to_mapping(example: Any) -> dict[str, Any]:
    if isinstance(example, dict):
        return _json_safe(example)
    if hasattr(example, "model_dump"):
        try:
            dumped = example.model_dump(mode="json")
            return _json_safe(dumped if isinstance(dumped, dict) else {})
        except Exception:
            pass
    if hasattr(example, "__dict__"):
        return _json_safe({key: value for key, value in vars(example).items() if not key.startswith("_")})
    return {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump(mode="json"))
        except Exception:
            return str(value)
    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes, bytearray)):
        return _json_safe({key: val for key, val in vars(value).items() if not key.startswith("_")})
    return value
