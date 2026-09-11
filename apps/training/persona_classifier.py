from __future__ import annotations

from apps.training.models import ImportedChat

_FAMILY_TOKENS = {"mum", "mom", "dad", "father", "mother", "sis", "bro", "brother", "sister", "aunt", "uncle", "cousin", "nan", "nana", "grandma", "grandad"}
_WORK_TOKENS = {"meeting", "deadline", "invoice", "project", "client", "shift", "manager", "office", "email", "schedule", "interview", "team"}
_FLIRTY_TOKENS = {"baby", "babe", "miss u", "miss you", "miss me", "cute", "pretty", "beautiful", "handsome", "kiss", "come here", "love u", "love you", "my love"}
_BANTER_TOKENS = {"lol", "loool", "lmao", "idiot", "weirdo", "stfu", "smh", "bro", "ffs", "nah"}


def infer_persona(chat: ImportedChat) -> str:
    name = chat.contact_name.casefold()
    if any(token in name for token in _FAMILY_TOKENS):
        return "family"

    combined = " ".join(message.text.casefold() for message in chat.messages[-300:])
    if _score_matches(combined, _WORK_TOKENS) >= 4:
        return "professional"
    if _score_matches(combined, _FLIRTY_TOKENS) >= 1:
        return "flirty"
    if _score_matches(combined, _BANTER_TOKENS) >= 4:
        return "close_friend"
    return "casual_friend"


def _score_matches(text: str, tokens: set[str]) -> int:
    return sum(text.count(token) for token in tokens)
