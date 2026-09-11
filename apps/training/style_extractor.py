from __future__ import annotations

import re
from collections import Counter

from apps.training.models import ImportedChat, ImportedMessage

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_LOW_SIGNAL = {"ok", "k", "kk", "yh", "yeah", "safe", "calm"}
_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF]")
_BANTER_TERMS = ("lol", "loool", "lmao", "weirdo", "idiot", "stfu", "smh", "nah", "bro")
_FLIRTY_TERMS = ("baby", "babe", "cute", "pretty", "beautiful", "handsome", "miss u", "miss you", "love u", "love you", "kiss", "come here")
_FORMAL_TERMS = ("regarding", "therefore", "please", "schedule", "confirm", "appreciate", "available")


def build_style_profile(messages: list[ImportedMessage]) -> dict[str, object]:
    owner_messages = [message.text for message in messages if message.sender_me and _is_useful(message.text)]
    if not owner_messages:
        return {}

    alpha_chars = [char for text in owner_messages for char in text if char.isalpha()]
    lowercase_ratio = (
        sum(1 for char in alpha_chars if char.islower()) / len(alpha_chars)
        if alpha_chars
        else 0.0
    )
    avg_words = sum(len(_TOKEN_RE.findall(text)) for text in owner_messages) / len(owner_messages)
    avg_chars = sum(len(text) for text in owner_messages) / len(owner_messages)
    emoji_rate = sum(len(_EMOJI_RE.findall(text)) for text in owner_messages) / len(owner_messages)
    question_rate = sum(1 for text in owner_messages if "?" in text) / len(owner_messages)
    playfulness = _term_ratio(owner_messages, _BANTER_TERMS)
    flirtiness = _term_ratio(owner_messages, _FLIRTY_TERMS)
    formality = _term_ratio(owner_messages, _FORMAL_TERMS)
    directness = min(1.0, (0.65 if avg_words <= 10 else 0.45) + (0.15 if lowercase_ratio > 0.75 else 0.0))
    punctuation = "minimal" if sum(text.endswith(".") for text in owner_messages) <= max(1, len(owner_messages) // 8) else "mixed"
    common_phrases = _common_terms(owner_messages)
    tone_traits = []
    if lowercase_ratio > 0.8:
        tone_traits.append("mostly lowercase texting")
    if avg_words <= 9:
        tone_traits.append("short punchy replies")
    if playfulness > 0.2:
        tone_traits.append("teasing and banter")
    if flirtiness > 0.16:
        tone_traits.append("lightly flirty")
    if not tone_traits:
        tone_traits.append("casual direct texting")
    return {
        "avg_words": round(avg_words, 2),
        "avg_chars": round(avg_chars, 2),
        "capitalization": "mostly lowercase" if lowercase_ratio > 0.8 else "mixed",
        "lowercase_ratio": round(lowercase_ratio, 3),
        "emoji_rate": round(emoji_rate, 3),
        "question_rate": round(question_rate, 3),
        "punctuation": punctuation,
        "playfulness": round(playfulness, 3),
        "directness": round(directness, 3),
        "flirtiness": round(flirtiness, 3),
        "formality": round(formality, 3),
        "common_phrases": common_phrases,
        "tone_traits": tone_traits,
    }


def build_contact_style_profile(chat: ImportedChat) -> dict[str, object]:
    profile = build_style_profile(chat.messages)
    if profile:
        profile["contact_name"] = chat.contact_name
    return profile


def merge_style_profiles(profiles: list[dict[str, object]]) -> dict[str, object]:
    usable = [profile for profile in profiles if profile]
    if not usable:
        return {}
    numeric_fields = ("avg_words", "avg_chars", "lowercase_ratio", "emoji_rate", "question_rate", "playfulness", "directness", "flirtiness", "formality")
    merged: dict[str, object] = {}
    for field in numeric_fields:
        values = [float(profile[field]) for profile in usable if field in profile]
        if values:
            merged[field] = round(sum(values) / len(values), 3)
    tone_traits: list[str] = []
    common_phrases: list[str] = []
    seen_trait: set[str] = set()
    seen_phrase: set[str] = set()
    for profile in usable:
        for trait in profile.get("tone_traits", []):
            if isinstance(trait, str) and trait not in seen_trait:
                seen_trait.add(trait)
                tone_traits.append(trait)
        for phrase in profile.get("common_phrases", []):
            if isinstance(phrase, str) and phrase not in seen_phrase:
                seen_phrase.add(phrase)
                common_phrases.append(phrase)
    merged["capitalization"] = "mostly lowercase" if float(merged.get("lowercase_ratio", 0.0)) > 0.8 else "mixed"
    merged["punctuation"] = "minimal" if float(merged.get("formality", 0.0)) < 0.25 else "mixed"
    merged["tone_traits"] = tone_traits[:8]
    merged["common_phrases"] = common_phrases[:12]
    return merged


def _term_ratio(messages: list[str], terms: tuple[str, ...]) -> float:
    hits = sum(1 for message in messages if any(term in message.casefold() for term in terms))
    return hits / max(1, len(messages))


def _common_terms(messages: list[str]) -> list[str]:
    counter: Counter[str] = Counter()
    for token in _TOKEN_RE.findall(" ".join(messages).casefold()):
        if len(token) <= 2 or token in _LOW_SIGNAL:
            continue
        counter[token] += 1
    return [token for token, _count in counter.most_common(10)]


def _is_useful(text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    return bool(normalized) and normalized not in _LOW_SIGNAL
