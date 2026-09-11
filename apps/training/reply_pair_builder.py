from __future__ import annotations

from apps.training.models import ImportedChat, ReplyExampleRecord

_LOW_SIGNAL_REPLIES = {"ok", "k", "kk", "yh", "yeah", "safe", "calm", "👍"}
_RISKY_TERMS = ("bank", "sort code", "account", "address", "password", "pin", "invoice")
_FLIRTY_TERMS = ("baby", "babe", "cute", "pretty", "miss u", "miss you", "love u", "love you", "come here")
_BANTER_TERMS = ("lol", "loool", "lmao", "weirdo", "idiot", "stfu", "nah", "bro")


def build_reply_examples(chat: ImportedChat, persona: str) -> list[ReplyExampleRecord]:
    examples: list[ReplyExampleRecord] = []
    messages = chat.messages
    index = 0
    while index < len(messages):
        if messages[index].sender_me:
            index += 1
            continue

        incoming_block: list[str] = [messages[index].text]
        next_index = index + 1
        while next_index < len(messages) and not messages[next_index].sender_me:
            incoming_block.append(messages[next_index].text)
            next_index += 1

        reply_block: list[str] = []
        reply_timestamp = None
        while next_index < len(messages) and messages[next_index].sender_me:
            if reply_timestamp is None:
                reply_timestamp = messages[next_index].timestamp
            reply_block.append(messages[next_index].text)
            next_index += 1

        if reply_block:
            target_reply = [text for text in reply_block[:3] if _is_high_signal_reply(text)]
            incoming_context = [text for text in incoming_block[-3:] if text.strip()]
            style_window = _owner_style_window(messages[:index], limit=3)
            if incoming_context and target_reply:
                metadata = {
                    "contains_question": any("?" in text for text in incoming_context),
                    "time_of_day": _time_of_day(reply_timestamp),
                    "message_length": "short" if sum(len(text.split()) for text in target_reply) <= 12 else "medium",
                    "tone": _tone_label(incoming_context, target_reply, persona),
                }
                examples.append(
                    ReplyExampleRecord(
                        contact_name=chat.contact_name,
                        persona=persona,
                        incoming_context=incoming_context,
                        my_previous_style_window=style_window,
                        target_reply=target_reply,
                        source_path=chat.source_path,
                        timestamp=reply_timestamp,
                        metadata=metadata,
                        quality_score=_quality_score(incoming_context, target_reply),
                    )
                )
        index = max(next_index, index + 1)
    return examples


def _owner_style_window(messages, limit: int) -> list[str]:
    owner_messages = [message.text for message in messages if message.sender_me and _is_high_signal_reply(message.text)]
    return owner_messages[-limit:]


def _is_high_signal_reply(text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    if not normalized or normalized in _LOW_SIGNAL_REPLIES:
        return False
    if len(normalized) < 3 or len(normalized) > 160:
        return False
    if any(term in normalized for term in _RISKY_TERMS):
        return False
    return True


def _quality_score(incoming_context: list[str], target_reply: list[str]) -> float:
    score = 0.55
    total_reply_words = sum(len(text.split()) for text in target_reply)
    if any("?" in text for text in incoming_context):
        score += 0.1
    if 3 <= total_reply_words <= 20:
        score += 0.15
    joined = " ".join(target_reply).casefold()
    if any(term in joined for term in _FLIRTY_TERMS):
        score += 0.08
    if any(term in joined for term in _BANTER_TERMS):
        score += 0.07
    return round(min(score, 0.98), 3)


def _time_of_day(timestamp) -> str:
    if timestamp is None:
        return "unknown"
    hour = timestamp.hour
    if hour < 6:
        return "late_night"
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "night"


def _tone_label(incoming_context: list[str], target_reply: list[str], persona: str) -> str:
    joined = " ".join(incoming_context + target_reply).casefold()
    if persona == "professional":
        return "practical"
    if any(term in joined for term in _FLIRTY_TERMS):
        return "flirty"
    if any(term in joined for term in _BANTER_TERMS):
        return "banter"
    if any("?" in text for text in incoming_context):
        return "responsive"
    return "casual"
