from pathlib import Path

from libs.drafting.conversation_policy import should_reply
from libs.drafting.conversation_state import ConversationStateStore
from libs.drafting.service import DraftingService


def test_regenerate_hii_returns_safe_different_options(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")

    bundle = service.regenerate_bundle(
        contact_name=None,
        incoming="hii",
        context=[],
        relationship_type="unknown",
        intent_type="greeting",
        avoid_candidates=["yo"],
        diversity_mode="alternative_wording",
    )

    assert bundle.reply_suggestions
    assert "yo" not in bundle.reply_suggestions
    assert all(reply in {"yo what u saying", "heyy what u doing", "yo how u been", "what u saying", "you good"} for reply in bundle.reply_suggestions)
    assert all("baby" not in reply and "trouble" not in reply for reply in bundle.reply_suggestions)
    assert bundle.final_decision == "review"


def test_conversation_state_prevents_duplicate_latest_message(tmp_path: Path) -> None:
    store = ConversationStateStore(tmp_path / "conversation_state.json")
    assert store.has_new_incoming("Ali", ["you coming later?"])
    store.set_last_seen("Ali", "you coming later?")
    assert not store.has_new_incoming("Ali", ["you coming later?"])
    store.record_candidates("Ali", ["yo", "heyy"])
    assert store.get_contact_state("Ali").recent_candidates[:2] == ["yo", "heyy"]


def test_conversation_policy_core_cases(tmp_path: Path) -> None:
    store = ConversationStateStore(tmp_path / "conversation_state.json")
    state = store.get_contact_state("Ali")
    assert should_reply("Ali", "hii", [{"speaker": "other", "text": "hii"}], state, "greeting")["should_reply"]
    assert should_reply("Ali", "👍", [{"speaker": "other", "text": "👍"}], state, "unknown")["should_reply"] is False
    assert should_reply("Ali", "why did u ignore me", [{"speaker": "other", "text": "why did u ignore me"}], state, "emotional")["needs_review"]
    store.set_last_seen("Ali", "hii")
    duplicate = should_reply("Ali", "hii", [{"speaker": "other", "text": "hii"}], store.get_contact_state("Ali"), "greeting")
    assert duplicate["should_reply"] is False
