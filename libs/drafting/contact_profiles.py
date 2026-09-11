from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

DEFAULT_CONTACT_PROFILES_PATH = Path("data/contact_profiles.json")


class ContactProfile(BaseModel):
    relationship_type: str = "unknown"
    tone: str = ""
    auto_send_allowed: bool = False
    auto_draft_allowed: bool = True
    boldness_level: float = Field(default=0.35, ge=0.0, le=1.0)
    flirt_allowed: bool = False
    banter_allowed: bool = True
    max_reply_length: int = Field(default=20, ge=1, le=280)
    notes: str = ""


def ensure_contact_profiles(path: Path = DEFAULT_CONTACT_PROFILES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("{}", encoding="utf-8")


def load_contact_profiles(path: Path = DEFAULT_CONTACT_PROFILES_PATH) -> dict[str, ContactProfile]:
    ensure_contact_profiles(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    profiles: dict[str, ContactProfile] = {}
    for name, raw in payload.items():
        if not isinstance(raw, dict):
            continue
        try:
            profiles[str(name)] = ContactProfile(**raw)
        except Exception:
            continue
    return profiles


def save_contact_profiles(path: Path, profiles: dict[str, ContactProfile | dict[str, Any]]) -> None:
    payload: dict[str, Any] = {}
    for name, profile in profiles.items():
        payload[name] = profile.model_dump(mode="json") if isinstance(profile, ContactProfile) else profile
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp_path.replace(path)


def get_contact_profile(
    contact_name: str | None,
    path: Path = DEFAULT_CONTACT_PROFILES_PATH,
) -> ContactProfile | None:
    if not contact_name:
        return None
    profiles = load_contact_profiles(path)
    return profiles.get(contact_name)
