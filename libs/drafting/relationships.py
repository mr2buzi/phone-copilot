from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from libs.drafting.training_data import RELATIONSHIP_TYPES, normalize_relationship_type

logger = logging.getLogger(__name__)


def ensure_contact_overrides(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("{}\n", encoding="utf-8")


def load_contact_overrides(path: Path) -> dict[str, str]:
    ensure_contact_overrides(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        logger.warning("Ignoring malformed contact overrides file: %s", path)
        return {}
    if not isinstance(raw, dict):
        logger.warning("Ignoring non-object contact overrides file: %s", path)
        return {}
    overrides: dict[str, str] = {}
    for contact, relationship in raw.items():
        normalized = normalize_relationship_type(str(relationship))
        if normalized == "unknown" and str(relationship).strip().lower().replace("-", "_") not in RELATIONSHIP_TYPES:
            logger.warning("Ignoring invalid relationship override for %s: %s", contact, relationship)
            continue
        key = normalize_contact_name(str(contact))
        if key:
            overrides[key] = normalized
    return overrides


def classify_relationship(
    contact_name: str | None,
    recent_messages: list[str] | None = None,
    *,
    overrides_path: Path | None = None,
    overrides: dict[str, str] | None = None,
) -> str:
    normalized_contact = normalize_contact_name(contact_name or "")
    manual = overrides if overrides is not None else load_contact_overrides(overrides_path) if overrides_path else {}
    if normalized_contact and normalized_contact in manual:
        return manual[normalized_contact]

    name_text = normalized_contact
    joined = " ".join(recent_messages or []).casefold()

    if re.search(r"\b(mum|mom|mother|dad|father|sis|sister|brother|aunt|uncle|nan|grandma|grandad)\b", name_text):
        return "family"
    if re.search(r"\b(boss|manager|work|hr|recruiter|client|doctor|dentist|landlord)\b", name_text):
        return "professional"
    if re.search(r"\b(prof|professor|lecturer|teacher|tutor|uni|university|college|module|seminar)\b", name_text):
        return "university"
    if any(term in joined for term in ("assignment", "lecture", "seminar", "coursework", "module deadline")):
        return "university"
    if any(term in joined for term in ("meeting", "invoice", "contract", "interview", "shift", "deadline")):
        return "professional"

    return "unknown"


def normalize_contact_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())
