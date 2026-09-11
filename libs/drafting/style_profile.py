from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

_MESSAGE_START_RE = re.compile(r"^(?P<timestamp>\d{4}-\d{2}-\d{2}, .*?) - (?P<sender>[^:]+): ?(?P<text>.*)$")
_SYSTEM_LINE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}, .*? - ")
_TOKEN_RE = re.compile(r"[A-Za-z']+")
_MULTISPACE_RE = re.compile(r"\s+")

_STOPWORDS = {
    "a",
    "about",
    "all",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "can",
    "cuz",
    "do",
    "dont",
    "for",
    "from",
    "get",
    "got",
    "have",
    "how",
    "i",
    "if",
    "im",
    "in",
    "is",
    "it",
    "its",
    "just",
    "like",
    "me",
    "my",
    "no",
    "not",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "this",
    "to",
    "u",
    "ur",
    "was",
    "we",
    "what",
    "when",
    "why",
    "w",
    "with",
    "yeah",
    "yo",
}

_AFFECTIONATE_TERMS = (
    "baby",
    "my love",
    "beautiful",
    "my beautiful",
    "princess",
    "cutie",
    "mine",
    "miss u",
    "i miss you",
    "i love you",
)

_SUPPORTIVE_TERMS = (
    "dw",
    "dont worry",
    "trust me",
    "its okay",
    "ill make it work",
    "i got u",
    "youll be okay",
)

_BANTER_TERMS = (
    "bro",
    "wtf",
    "shutup",
    "shut up",
    "idiot",
    "fool",
    "dawg",
    "loool",
    "lmaoo",
    "ffs",
    "nah",
)

_EXPLICIT_TERMS = (
    "cock",
    "pussy",
    "cum",
    "throat",
    "tongue",
    "horny",
    "moaning",
    "slut",
    "dick",
    "fuck me",
    "wet",
    "clit",
    "legs open",
)


class StyleProfile(BaseModel):
    owner_aliases: list[str] = Field(default_factory=list)
    source_paths: list[str] = Field(default_factory=list)
    source_message_count: int = 0
    average_message_length: float = 0.0
    lowercase_ratio: float = 0.0
    tone_traits: list[str] = Field(default_factory=list)
    common_terms: list[str] = Field(default_factory=list)
    affectionate_terms: list[str] = Field(default_factory=list)
    safe_example_messages: list[str] = Field(default_factory=list)
    guidance_summary: str = ""

    def to_prompt_fragment(self) -> str:
        lines = [
            "Style profile from the owner's past chats:",
            f"- source messages: {self.source_message_count}",
        ]
        if self.tone_traits:
            lines.append(f"- tone traits: {', '.join(self.tone_traits)}")
        if self.common_terms:
            lines.append(f"- frequent terms: {', '.join(self.common_terms[:8])}")
        if self.affectionate_terms:
            lines.append(f"- affectionate terms used often: {', '.join(self.affectionate_terms[:6])}")
        if self.guidance_summary:
            lines.append(f"- guidance: {self.guidance_summary}")
        if self.safe_example_messages:
            lines.append("- example lines:")
            lines.extend(f'  - "{message}"' for message in self.safe_example_messages[:6])
        return "\n".join(lines)


class ReplyExample(BaseModel):
    incoming: str
    reply: str
    contact_name: str | None = None
    source_path: str


def default_style_profile() -> StyleProfile:
    return StyleProfile(
        tone_traits=["casual, blunt, playful, slightly chaotic, confident"],
        common_terms=["nah", "icl", "ngl", "yh", "bro", "like", "lowkey", "fair"],
        average_message_length=24.0,
        lowercase_ratio=0.9,
        guidance_summary=(
            "Do not sound like customer support. Do not overexplain. Do not use em dashes. "
            "Do not add fake enthusiasm. If context is missing, ask a short natural follow-up. "
            "For professional contacts, be clearer and cleaner but still not robotic."
        ),
    )


def load_style_profile(path: Path) -> StyleProfile | None:
    if not path.exists():
        return default_style_profile()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "tone" in raw and "message_shape" in raw:
            phrases = raw.get("phrases", {}) if isinstance(raw.get("phrases"), dict) else {}
            common = phrases.get("common", []) if isinstance(phrases.get("common"), list) else []
            rules = raw.get("rules", []) if isinstance(raw.get("rules"), list) else []
            tone = raw.get("tone", {}) if isinstance(raw.get("tone"), dict) else {}
            return StyleProfile(
                tone_traits=[str(tone.get("default", "casual, blunt, playful"))],
                common_terms=[str(item) for item in common[:12]],
                average_message_length=24.0,
                lowercase_ratio=0.9 if raw.get("message_shape", {}).get("uses_lowercase", True) else 0.5,
                guidance_summary=" ".join(str(item) for item in rules),
            )
        return StyleProfile.model_validate(raw)
    except Exception:
        return default_style_profile()


def build_style_profile_from_training_dir(training_dir: Path, owner_aliases: list[str]) -> StyleProfile | None:
    if not training_dir.exists():
        return None
    normalized_aliases = {alias.strip().casefold() for alias in owner_aliases if alias.strip()}
    if not normalized_aliases:
        return None

    all_messages: list[str] = []
    source_paths: list[str] = []
    for path in sorted(training_dir.glob("*.txt")):
        owner_messages = _extract_owner_messages(path, normalized_aliases)
        if not owner_messages:
            continue
        all_messages.extend(owner_messages)
        source_paths.append(str(path))

    if not all_messages:
        return None

    affectionate_counts = Counter()
    for term in _AFFECTIONATE_TERMS:
        affectionate_counts[term] = sum(1 for message in all_messages if term in message.casefold())

    lowercase_ratio = _lowercase_ratio(all_messages)
    average_length = sum(len(message) for message in all_messages) / len(all_messages)
    tone_traits = _tone_traits(all_messages, average_length=average_length, lowercase_ratio=lowercase_ratio)
    common_terms = _common_terms(all_messages)
    example_messages = _select_safe_examples(all_messages)

    affectionate_terms = [
        term
        for term, count in affectionate_counts.most_common()
        if count > 0
    ][:8]
    guidance_summary = _guidance_summary(
        affectionate_terms=affectionate_terms,
        average_length=average_length,
        lowercase_ratio=lowercase_ratio,
        messages=all_messages,
    )

    return StyleProfile(
        owner_aliases=sorted(normalized_aliases),
        source_paths=source_paths,
        source_message_count=len(all_messages),
        average_message_length=round(average_length, 1),
        lowercase_ratio=round(lowercase_ratio, 3),
        tone_traits=tone_traits,
        common_terms=common_terms,
        affectionate_terms=affectionate_terms,
        safe_example_messages=example_messages,
        guidance_summary=guidance_summary,
    )


def write_style_profile(profile: StyleProfile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(profile.model_dump(mode="json"), indent=2), encoding="utf-8")


def build_reply_examples_from_training_dir(training_dir: Path, owner_aliases: list[str]) -> list[ReplyExample]:
    if not training_dir.exists():
        return []
    normalized_aliases = {alias.strip().casefold() for alias in owner_aliases if alias.strip()}
    if not normalized_aliases:
        return []

    examples: list[ReplyExample] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(training_dir.glob("*.txt")):
        contact_name = _contact_name_from_training_path(path)
        events = _extract_chat_events(path)
        if not events:
            continue
        index = 0
        while index < len(events):
            sender, text = events[index]
            sender_key = sender.casefold()
            if sender_key in normalized_aliases:
                index += 1
                continue

            incoming_lines = [text]
            next_index = index + 1
            while next_index < len(events) and events[next_index][0].casefold() == sender_key:
                incoming_lines.append(events[next_index][1])
                next_index += 1

            reply_lines: list[str] = []
            while next_index < len(events) and events[next_index][0].casefold() in normalized_aliases:
                reply_lines.append(events[next_index][1])
                next_index += 1

            incoming = _clean_message(" ".join(incoming_lines[:2]))
            reply = _clean_message(" ".join(reply_lines[:2]))
            key = (incoming.casefold(), reply.casefold())
            if (
                incoming
                and reply
                and key not in seen
                and _is_safe_example(incoming)
                and _is_safe_example(reply)
                and len(incoming) <= 120
                and len(reply) <= 120
            ):
                seen.add(key)
                examples.append(
                    ReplyExample(
                        incoming=incoming,
                        reply=reply,
                        contact_name=contact_name,
                        source_path=str(path),
                    )
                )
            index = max(next_index, index + 1)
    return examples[:1200]


def _extract_owner_messages(path: Path, owner_aliases: set[str]) -> list[str]:
    current_sender: str | None = None
    current_lines: list[str] = []
    messages: list[str] = []

    def flush_current() -> None:
        nonlocal current_sender, current_lines
        if current_sender is None:
            current_lines = []
            return
        text = _clean_message(" ".join(current_lines))
        if current_sender.casefold() in owner_aliases and _is_usable_message(text):
            messages.append(text)
        current_sender = None
        current_lines = []

    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    for raw_line in raw_text.splitlines():
        line = raw_line.replace("\ufeff", "").replace("\u200e", "").strip()
        message_match = _MESSAGE_START_RE.match(line)
        if message_match is not None:
            flush_current()
            current_sender = message_match.group("sender").strip()
            current_lines = [message_match.group("text").strip()]
            continue
        if _SYSTEM_LINE_RE.match(line):
            flush_current()
            continue
        if current_sender is not None and line:
            current_lines.append(line)

    flush_current()
    return messages


def _extract_chat_events(path: Path) -> list[tuple[str, str]]:
    current_sender: str | None = None
    current_lines: list[str] = []
    events: list[tuple[str, str]] = []

    def flush_current() -> None:
        nonlocal current_sender, current_lines
        if current_sender is None:
            current_lines = []
            return
        text = _clean_message(" ".join(current_lines))
        if _is_usable_message(text):
            events.append((current_sender, text))
        current_sender = None
        current_lines = []

    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    for raw_line in raw_text.splitlines():
        line = raw_line.replace("\ufeff", "").replace("\u200e", "").strip()
        message_match = _MESSAGE_START_RE.match(line)
        if message_match is not None:
            flush_current()
            current_sender = message_match.group("sender").strip()
            current_lines = [message_match.group("text").strip()]
            continue
        if _SYSTEM_LINE_RE.match(line):
            flush_current()
            continue
        if current_sender is not None and line:
            current_lines.append(line)

    flush_current()
    return events


def _clean_message(text: str) -> str:
    return _MULTISPACE_RE.sub(" ", text).strip()


def _contact_name_from_training_path(path: Path) -> str | None:
    stem = path.stem.strip()
    prefix = "WhatsApp Chat with "
    if stem.startswith(prefix):
        stem = stem[len(prefix):].strip()
    return stem or None


def _is_usable_message(text: str) -> bool:
    if not text:
        return False
    lowered = text.casefold()
    return lowered not in {"<media omitted>", "you deleted this message"}


def _lowercase_ratio(messages: list[str]) -> float:
    alpha_chars = [char for message in messages for char in message if char.isalpha()]
    if not alpha_chars:
        return 0.0
    lowercase_chars = sum(1 for char in alpha_chars if char.islower())
    return lowercase_chars / len(alpha_chars)


def _tone_traits(messages: list[str], *, average_length: float, lowercase_ratio: float) -> list[str]:
    traits: list[str] = []
    if lowercase_ratio > 0.85:
        traits.append("mostly lowercase texting")
    if average_length < 32:
        traits.append("short punchy messages")
    if sum(1 for message in messages if re.search(r"(.)\1{2,}", message.casefold())) >= max(1, len(messages) // 40):
        traits.append("uses repeated letters for emphasis")
    if sum(1 for message in messages if any(term in message.casefold() for term in _BANTER_TERMS)) >= max(1, len(messages) // 60):
        traits.append("playful teasing and banter")
    if sum(1 for message in messages if any(term in message.casefold() for term in _AFFECTIONATE_TERMS)) >= max(1, len(messages) // 70):
        traits.append("affectionate and lightly flirty")
    if sum(1 for message in messages if any(term in message.casefold() for term in _SUPPORTIVE_TERMS)) >= max(1, len(messages) // 100):
        traits.append("direct reassurance when needed")
    return traits


def _common_terms(messages: list[str]) -> list[str]:
    counter: Counter[str] = Counter()
    for token in _TOKEN_RE.findall(" ".join(messages).casefold()):
        if len(token) < 2:
            continue
        if token in _STOPWORDS:
            continue
        counter[token] += 1
    return [token for token, _count in counter.most_common(12)]


def _select_safe_examples(messages: list[str]) -> list[str]:
    recent_messages = list(reversed(messages))
    selected: list[str] = []
    seen: set[str] = set()
    categories = (
        lambda message: any(term in message.casefold() for term in _AFFECTIONATE_TERMS),
        lambda message: any(term in message.casefold() for term in _SUPPORTIVE_TERMS),
        lambda message: any(term in message.casefold() for term in _BANTER_TERMS),
        lambda message: True,
    )

    for matcher in categories:
        for message in recent_messages:
            normalized = message.casefold()
            if normalized in seen:
                continue
            if not _is_safe_example(message):
                continue
            if not matcher(message):
                continue
            selected.append(message)
            seen.add(normalized)
            if len(selected) >= 6:
                return selected

    return selected


def _is_safe_example(message: str) -> bool:
    lowered = message.casefold()
    if len(message) < 4 or len(message) > 90:
        return False
    if any(term in lowered for term in _EXPLICIT_TERMS):
        return False
    if lowered.startswith("<media omitted>"):
        return False
    return True


def _guidance_summary(
    *,
    affectionate_terms: list[str],
    average_length: float,
    lowercase_ratio: float,
    messages: list[str],
) -> str:
    parts = []
    if lowercase_ratio > 0.85:
        parts.append("Keep replies mostly lowercase unless shouting for emphasis.")
    if average_length < 32:
        parts.append("Prefer short bursts over long paragraphs.")
    if affectionate_terms:
        parts.append(
            "When the thread is warm, use light pet names and compliments such as "
            + ", ".join(affectionate_terms[:4])
            + "."
        )
    if sum(1 for message in messages if any(term in message.casefold() for term in _BANTER_TERMS)) > 30:
        parts.append("Dry replies are wrong here; playful teasing is normal.")
    parts.append("Default to warm confidence and slight flirt, but do not jump into explicit sexual content unless the visible thread is already there.")
    return " ".join(parts)
