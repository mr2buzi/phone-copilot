from __future__ import annotations

from pydantic import BaseModel

from apps.training.persona_classifier import infer_persona
from apps.training.reply_pair_builder import build_reply_examples
from apps.training.store import ConversationIntelligenceStore
from apps.training.style_extractor import build_contact_style_profile, build_style_profile, merge_style_profiles
from apps.training.whatsapp_importer import import_training_dir


class BuildSummary(BaseModel):
    chats_imported: int
    contacts_indexed: int
    messages_imported: int
    reply_examples_built: int
    personas: dict[str, int]


def rebuild_conversation_intelligence(
    *,
    training_dir,
    owner_aliases: list[str],
    store: ConversationIntelligenceStore,
) -> BuildSummary:
    chats = import_training_dir(training_dir, owner_aliases)
    store.reset_training_data()

    persona_counts: dict[str, int] = {}
    persona_profile_map: dict[str, list[dict[str, object]]] = {}
    total_messages = 0
    total_examples = 0

    for chat in chats:
        persona = infer_persona(chat)
        persona_counts[persona] = persona_counts.get(persona, 0) + 1
        contact_id = store.upsert_contact(chat.contact_name, persona)
        total_messages += len(chat.messages)
        store.add_messages(
            contact_id,
            [
                {
                    "timestamp": message.timestamp,
                    "sender_me": message.sender_me,
                    "body": message.text,
                }
                for message in chat.messages
            ],
        )
        examples = build_reply_examples(chat, persona)
        total_examples += len(examples)
        store.add_reply_examples(
            contact_id,
            [
                {
                    "persona": example.persona,
                    "timestamp": example.timestamp,
                    "incoming_context": example.incoming_context,
                    "my_previous_style_window": example.my_previous_style_window,
                    "target_reply": example.target_reply,
                    "metadata": example.metadata,
                    "source_path": example.source_path,
                    "quality_score": example.quality_score,
                }
                for example in examples
            ],
        )
        contact_profile = build_contact_style_profile(chat)
        if contact_profile:
            persona_profile_map.setdefault(persona, []).append(contact_profile)
            store.set_style_profile("contact", " ".join(chat.contact_name.casefold().split()), contact_profile)

    if chats:
        global_profile = build_style_profile([message for chat in chats for message in chat.messages])
        if global_profile:
            store.set_style_profile("global", "owner", global_profile)

    for persona, profiles in persona_profile_map.items():
        merged = merge_style_profiles(profiles)
        if merged:
            store.set_style_profile("persona", persona, merged)

    return BuildSummary(
        chats_imported=len(chats),
        contacts_indexed=len(chats),
        messages_imported=total_messages,
        reply_examples_built=total_examples,
        personas=persona_counts,
    )
