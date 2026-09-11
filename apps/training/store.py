from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from sqlmodel import Session, SQLModel, create_engine, delete, select

from apps.training.models import (
    ContactInsight,
    RetrievedReplyExample,
    TrainingContact,
    TrainingMessage,
    TrainingReplyExample,
    TrainingStyleProfile,
)


class ConversationIntelligenceStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}")
        SQLModel.metadata.create_all(self.engine)

    def reset_training_data(self) -> None:
        with Session(self.engine) as session:
            session.exec(delete(TrainingReplyExample))
            session.exec(delete(TrainingMessage))
            session.exec(delete(TrainingStyleProfile))
            session.exec(delete(TrainingContact))
            session.commit()

    def upsert_contact(self, display_name: str, persona: str) -> int:
        normalized = _normalize(display_name)
        with Session(self.engine) as session:
            contact = session.exec(
                select(TrainingContact).where(TrainingContact.normalized_name == normalized)
            ).first()
            if contact is None:
                contact = TrainingContact(
                    display_name=display_name,
                    normalized_name=normalized,
                    persona=persona,
                    auto_send_allowed=persona not in {"family", "professional"},
                )
                session.add(contact)
                session.commit()
                session.refresh(contact)
                return int(contact.id)
            contact.display_name = display_name
            contact.persona = persona
            contact.auto_send_allowed = persona not in {"family", "professional"}
            session.add(contact)
            session.commit()
            return int(contact.id)

    def add_messages(self, contact_id: int, messages: Iterable[dict[str, object]]) -> None:
        rows = [
            TrainingMessage(
                contact_id=contact_id,
                timestamp=message.get("timestamp") or datetime.now(timezone.utc),
                sender_me=bool(message["sender_me"]),
                body=str(message["body"]),
                source=str(message.get("source", "whatsapp_import")),
                quality_score=float(message.get("quality_score", 1.0)),
            )
            for message in messages
        ]
        if not rows:
            return
        with Session(self.engine) as session:
            session.add_all(rows)
            session.commit()

    def add_reply_examples(self, contact_id: int, examples: Iterable[dict[str, object]]) -> None:
        rows = [
            TrainingReplyExample(
                contact_id=contact_id,
                persona=str(example["persona"]),
                timestamp=example.get("timestamp"),
                incoming_context_json=json.dumps(example["incoming_context"]),
                my_previous_style_window_json=json.dumps(example.get("my_previous_style_window", [])),
                target_reply_json=json.dumps(example["target_reply"]),
                metadata_json=json.dumps(example.get("metadata", {}), sort_keys=True),
                style_tags_json=json.dumps(example.get("style_tags", [])),
                source_path=str(example["source_path"]),
                quality_score=float(example.get("quality_score", 1.0)),
            )
            for example in examples
        ]
        if not rows:
            return
        with Session(self.engine) as session:
            session.add_all(rows)
            session.commit()

    def set_style_profile(self, scope: str, scope_key: str, profile: dict[str, object]) -> None:
        with Session(self.engine) as session:
            existing = session.exec(
                select(TrainingStyleProfile).where(
                    TrainingStyleProfile.scope == scope,
                    TrainingStyleProfile.scope_key == scope_key,
                )
            ).first()
            payload = json.dumps(profile, sort_keys=True)
            if existing is None:
                existing = TrainingStyleProfile(scope=scope, scope_key=scope_key, profile_json=payload)
            else:
                existing.profile_json = payload
            session.add(existing)
            session.commit()

    def get_style_profile(self, scope: str, scope_key: str) -> dict[str, object]:
        with Session(self.engine) as session:
            row = session.exec(
                select(TrainingStyleProfile).where(
                    TrainingStyleProfile.scope == scope,
                    TrainingStyleProfile.scope_key == scope_key,
                )
            ).first()
        if row is None:
            return {}
        try:
            return json.loads(row.profile_json)
        except json.JSONDecodeError:
            return {}

    def get_contact_insight(self, contact_name: str | None) -> ContactInsight | None:
        if not contact_name:
            return None
        normalized = _normalize(contact_name)
        with Session(self.engine) as session:
            contact = session.exec(
                select(TrainingContact).where(TrainingContact.normalized_name == normalized)
            ).first()
        if contact is None:
            return None
        contact_profile = self.get_style_profile("contact", normalized)
        if not contact_profile:
            contact_profile = self.get_style_profile("persona", contact.persona)
        return ContactInsight(
            contact_name=contact.display_name,
            persona=contact.persona,
            style_profile=contact_profile,
        )

    def retrieve_reply_examples(
        self,
        *,
        contact_name: str | None,
        recent_messages: list[str],
        limit: int = 4,
    ) -> list[RetrievedReplyExample]:
        if not recent_messages:
            return []
        latest = recent_messages[-1]
        latest_tokens = _tokens(latest)
        normalized_contact = _normalize(contact_name) if contact_name else None
        target_persona = self._persona_for_contact(normalized_contact) if normalized_contact else None
        with Session(self.engine) as session:
            contact_lookup = {}
            for contact in session.exec(select(TrainingContact)).all():
                contact_lookup[int(contact.id)] = contact
            rows = session.exec(select(TrainingReplyExample)).all()

        scored: list[tuple[float, RetrievedReplyExample]] = []
        for row in rows:
            contact = contact_lookup.get(row.contact_id)
            incoming_context = _json_list(row.incoming_context_json)
            target_reply = _json_list(row.target_reply_json)
            if not incoming_context or not target_reply or contact is None:
                continue
            example_text = " ".join(incoming_context)
            score = float(row.quality_score)
            score += _overlap_score(latest_tokens, _tokens(example_text))
            if normalized_contact and contact.normalized_name == normalized_contact:
                score += 4.0
            elif target_persona and contact.persona == target_persona:
                score += 1.0
            if latest.endswith("?") and any("?" in part for part in incoming_context):
                score += 0.5
            for term in ("miss", "love", "call", "where", "wyd", "cute", "pretty", "busy", "come"):
                if term in latest.casefold() and term in example_text.casefold():
                    score += 0.6
            if score <= 0.8:
                continue
            scored.append(
                (
                    score,
                    RetrievedReplyExample(
                        contact_name=contact.display_name,
                        persona=row.persona,
                        incoming_context=incoming_context,
                        target_reply=target_reply,
                        score=round(score, 3),
                        source_path=row.source_path,
                    ),
                )
            )
        scored.sort(key=lambda item: (-item[0], len(" ".join(item[1].target_reply))))
        unique: list[RetrievedReplyExample] = []
        seen: set[tuple[str, str]] = set()
        for _score, example in scored:
            key = (
                _normalize(" ".join(example.incoming_context)),
                _normalize(" ".join(example.target_reply)),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(example)
            if len(unique) >= limit:
                break
        return unique

    def has_reply_examples(self) -> bool:
        with Session(self.engine) as session:
            row = session.exec(select(TrainingReplyExample.id)).first()
        return row is not None

    def _persona_for_contact(self, normalized_contact: str) -> str | None:
        with Session(self.engine) as session:
            contact = session.exec(
                select(TrainingContact).where(TrainingContact.normalized_name == normalized_contact)
            ).first()
        return contact.persona if contact is not None else None


def _normalize(text: str | None) -> str:
    return " ".join((text or "").casefold().split())


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9']+", text)}


def _overlap_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return float(len(left.intersection(right)))


def _json_list(raw: str) -> list[str]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item).strip() for item in payload if str(item).strip()]
