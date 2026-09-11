from __future__ import annotations

import re
from pathlib import Path

from dateutil import parser as date_parser

from apps.training.models import ImportedChat, ImportedMessage

_MESSAGE_START_RE = re.compile(r"^(?P<timestamp>\d{4}-\d{2}-\d{2}, .*?) - (?P<sender>[^:]+): ?(?P<text>.*)$")
_SYSTEM_LINE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}, .*? - ")
_MULTISPACE_RE = re.compile(r"\s+")


def import_training_dir(training_dir: Path, owner_aliases: list[str]) -> list[ImportedChat]:
    chats: list[ImportedChat] = []
    for path in sorted(training_dir.glob("*.txt")):
        chat = import_whatsapp_chat(path, owner_aliases)
        if chat is not None and chat.messages:
            chats.append(chat)
    return chats


def import_whatsapp_chat(path: Path, owner_aliases: list[str]) -> ImportedChat | None:
    normalized_aliases = {_normalize_name(alias) for alias in owner_aliases if alias.strip()}
    if not normalized_aliases:
        return None

    contact_name = _contact_name_from_path(path)
    current_sender: str | None = None
    current_timestamp: str | None = None
    current_lines: list[str] = []
    messages: list[ImportedMessage] = []

    def flush_current() -> None:
        nonlocal current_sender, current_timestamp, current_lines
        if current_sender is None:
            current_timestamp = None
            current_lines = []
            return
        text = _clean_message(" ".join(current_lines))
        if _is_usable_message(text):
            sender_me = _normalize_name(current_sender) in normalized_aliases
            messages.append(
                ImportedMessage(
                    timestamp=_parse_timestamp(current_timestamp),
                    sender=current_sender,
                    text=text,
                    sender_me=sender_me,
                )
            )
        current_sender = None
        current_timestamp = None
        current_lines = []

    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    for raw_line in raw_text.splitlines():
        line = raw_line.replace("\ufeff", "").replace("\u200e", "").strip()
        message_match = _MESSAGE_START_RE.match(line)
        if message_match is not None:
            flush_current()
            current_timestamp = message_match.group("timestamp").strip()
            current_sender = message_match.group("sender").strip()
            current_lines = [message_match.group("text").strip()]
            continue
        if _SYSTEM_LINE_RE.match(line):
            flush_current()
            continue
        if current_sender is not None and line:
            current_lines.append(line)

    flush_current()
    return ImportedChat(contact_name=contact_name, source_path=str(path), messages=messages)


def _parse_timestamp(value: str | None):
    if not value:
        return None
    normalized = value.replace("a.m.", "AM").replace("p.m.", "PM")
    try:
        return date_parser.parse(normalized)
    except (ValueError, OverflowError, TypeError):
        return None


def _contact_name_from_path(path: Path) -> str:
    stem = path.stem.strip()
    prefix = "WhatsApp Chat with "
    if stem.startswith(prefix):
        stem = stem[len(prefix):].strip()
    return stem or path.stem


def _clean_message(text: str) -> str:
    return _MULTISPACE_RE.sub(" ", text).strip()


def _normalize_name(text: str) -> str:
    return " ".join(text.casefold().split())


def _is_usable_message(text: str) -> bool:
    lowered = text.casefold()
    if not lowered:
        return False
    return lowered not in {
        "<media omitted>",
        "you deleted this message",
        "this message was deleted",
    }
