from pathlib import Path

from apps.training import ConversationIntelligenceStore, rebuild_conversation_intelligence
from apps.training.whatsapp_importer import import_whatsapp_chat
from libs.drafting import DraftingService


def test_whatsapp_importer_parses_export(tmp_path: Path) -> None:
    chat_path = tmp_path / "WhatsApp Chat with taylor posh.txt"
    chat_path.write_text(
        "\n".join(
            [
                "2026-04-10, 3:20 a.m. - taylor posh: where are u",
                "2026-04-10, 3:21 a.m. - Owner: im here relax",
                "2026-04-10, 3:22 a.m. - taylor posh: call me rn",
                "2026-04-10, 3:23 a.m. - Owner: 2 secs",
            ]
        ),
        encoding="utf-8",
    )

    chat = import_whatsapp_chat(chat_path, ["Owner"])

    assert chat is not None
    assert chat.contact_name == "taylor posh"
    assert [message.text for message in chat.messages] == [
        "where are u",
        "im here relax",
        "call me rn",
        "2 secs",
    ]
    assert [message.sender_me for message in chat.messages] == [False, True, False, True]


def test_conversation_intelligence_builds_store_and_personas(tmp_path: Path) -> None:
    training_dir = tmp_path / "training"
    training_dir.mkdir()
    (training_dir / "WhatsApp Chat with taylor posh.txt").write_text(
        "\n".join(
            [
                "2026-04-10, 11:20 p.m. - taylor posh: miss me?",
                "2026-04-10, 11:21 p.m. - Owner: depends how much",
                "2026-04-10, 11:22 p.m. - taylor posh: cute answer",
                "2026-04-10, 11:23 p.m. - Owner: u bring it out of me",
            ]
        ),
        encoding="utf-8",
    )
    (training_dir / "WhatsApp Chat with Dad.txt").write_text(
        "\n".join(
            [
                "2026-04-11, 9:00 a.m. - Dad: what time is the meeting",
                "2026-04-11, 9:01 a.m. - Owner: 2ish i think",
            ]
        ),
        encoding="utf-8",
    )

    store = ConversationIntelligenceStore(tmp_path / "conversation_intelligence.db")
    summary = rebuild_conversation_intelligence(
        training_dir=training_dir,
        owner_aliases=["Owner"],
        store=store,
    )

    assert summary.chats_imported == 2
    assert summary.reply_examples_built >= 2
    assert summary.personas["flirty"] == 1
    assert summary.personas["family"] == 1
    insight = store.get_contact_insight("taylor posh")
    assert insight is not None
    assert insight.persona == "flirty"
    examples = store.retrieve_reply_examples(
        contact_name="taylor posh",
        recent_messages=["where are u", "miss me?"],
        limit=3,
    )
    assert examples
    assert examples[0].contact_name == "taylor posh"
    assert examples[0].persona == "flirty"


def test_drafting_service_prefers_structured_intelligence_examples(tmp_path: Path) -> None:
    training_dir = tmp_path / "training"
    training_dir.mkdir()
    (training_dir / "WhatsApp Chat with taylor posh.txt").write_text(
        "\n".join(
            [
                "2026-04-10, 11:20 p.m. - taylor posh: miss me?",
                "2026-04-10, 11:21 p.m. - Owner: depends how much",
                "2026-04-10, 11:22 p.m. - taylor posh: where are u",
                "2026-04-10, 11:23 p.m. - Owner: im here relax",
            ]
        ),
        encoding="utf-8",
    )
    db_path = tmp_path / "conversation_intelligence.db"
    rebuild_conversation_intelligence(
        training_dir=training_dir,
        owner_aliases=["Owner"],
        store=ConversationIntelligenceStore(db_path),
    )

    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
        intelligence_db_path=db_path,
        training_dir=training_dir,
        training_owner_aliases=["Owner"],
    )

    examples = service._select_reply_examples(["where are u"], contact_name="taylor posh")  # type: ignore[attr-defined]

    assert examples
    assert examples[0].contact_name == "taylor posh"
    assert "im here relax" in examples[0].reply
