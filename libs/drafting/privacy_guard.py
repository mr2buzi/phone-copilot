from __future__ import annotations

import re
from dataclasses import dataclass


SENSITIVE_TERMS = (
    "otp",
    "password",
    "passcode",
    "bank",
    "sort code",
    "card",
    "cvv",
    "pin",
    "address",
    "hospital",
    "police",
    "emergency",
    "pregnant",
    "sex",
    "nude",
    "breakup",
    "break up",
    "self harm",
    "suicide",
    "kill myself",
    "legal",
    "solicitor",
    "university misconduct",
    "disciplinary",
)

PIN_SECRET_CONTEXT = (
    "bank",
    "card",
    "debit",
    "credit",
    "atm",
    "cash machine",
    "password",
    "passcode",
    "otp",
    "login",
    "account",
    "security",
    "number",
    "code",
)


@dataclass(slots=True)
class PrivacyDecision:
    external_api_blocked: bool
    blocked_reason: str | None = None
    matched_terms: list[str] | None = None


def check_external_api_allowed(messages: list[str], *, allow_sensitive: bool = False) -> PrivacyDecision:
    if allow_sensitive:
        return PrivacyDecision(False, None, [])
    blob = normalize_for_privacy(" ".join(messages))
    matched = [term for term in SENSITIVE_TERMS if _contains_sensitive_term(blob, term)]
    if matched:
        return PrivacyDecision(True, "sensitive_content", matched)
    return PrivacyDecision(False, None, [])


def normalize_for_privacy(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _contains_term(blob: str, term: str) -> bool:
    normalized = normalize_for_privacy(term)
    if " " in normalized:
        return normalized in blob
    return re.search(rf"\b{re.escape(normalized)}\b", blob) is not None


def _contains_sensitive_term(blob: str, term: str) -> bool:
    normalized = normalize_for_privacy(term)
    if normalized == "pin":
        return _contains_secret_pin(blob)
    return _contains_term(blob, normalized)


def _contains_secret_pin(blob: str) -> bool:
    if not _contains_term(blob, "pin"):
        return False
    if re.search(r"\bpin\s*(?:is|=|:)?\s*\d{3,8}\b", blob):
        return True
    return any(_contains_term(blob, term) for term in PIN_SECRET_CONTEXT)
