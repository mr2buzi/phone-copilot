from __future__ import annotations

import json
from pathlib import Path

import pytest

from libs.drafting.critic import critique_candidate
from libs.drafting.relationships import classify_relationship, load_contact_overrides
from libs.drafting.retrieval import retrieve_similar_examples, retrieve_similar_examples_from_rows
from libs.drafting.service import DraftCandidate, DraftingService
from libs.drafting.training_data import append_correction, ensure_training_message_files, load_training_messages


def test_training_loader_skips_invalid_rows(tmp_path: Path) -> None:
    training_dir = tmp_path / "training_messages"
    ensure_training_message_files(training_dir)
    (training_dir / "close_friends.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "relationship_type": "close_friend",
                        "incoming": "you coming later?",
                        "context": [],
                        "my_reply": "yh probs what time",
                    }
                ),
                "{bad json",
            ]
        ),
        encoding="utf-8",
    )

    rows = load_training_messages(training_dir, "close_friend")

    assert len(rows) == 1
    assert rows[0]["relationship_type"] == "close_friend"


def test_relationship_override_wins_and_invalid_values_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "contact_overrides.json"
    path.write_text(json.dumps({"Mom": "family", "Bad": "unsafe"}), encoding="utf-8")

    overrides = load_contact_overrides(path)

    assert overrides == {"mom": "family"}
    assert classify_relationship("Mom", [], overrides_path=path) == "family"
    assert classify_relationship("Someone", [], overrides_path=path) == "unknown"


def test_retrieval_weights_corrections_higher(tmp_path: Path) -> None:
    training_dir = tmp_path / "training_messages"
    ensure_training_message_files(training_dir)
    (training_dir / "close_friends.jsonl").write_text(
        json.dumps(
            {
                "relationship_type": "close_friend",
                "incoming": "you coming later?",
                "context": [],
                "my_reply": "yh probs what time u lot going",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    append_correction(
        training_dir,
        {
            "relationship_type": "close_friend",
            "incoming": "you coming later?",
            "context": [],
            "bad_ai_reply": "That sounds great!",
            "user_final_reply": "yh what time",
            "reason_bad": "too formal",
            "selected_mode": "review",
        },
    )

    result = retrieve_similar_examples(
        incoming="you coming later?",
        recent_context=[],
        relationship_type="close_friend",
        training_dir=training_dir,
    )

    assert result.examples
    assert result.examples[0].source == "correction"


@pytest.mark.parametrize("relationship", ["professional", "family", "romantic_interest", "unknown"])
def test_critic_blocks_never_auto_send_relationships(relationship: str) -> None:
    candidate = DraftCandidate(text="yeah sounds good", sequence=["yeah sounds good"])

    critique = critique_candidate(
        candidate,
        latest_message="can you confirm?",
        relationship_type=relationship,
        style_score=0.98,
        relevance_score=0.98,
        selected_mode="auto_send",
    )

    assert critique.final_decision == "review"
    assert "never_auto_send" in (critique.blocked_reason or "")


def test_critic_allows_low_risk_close_friend_send() -> None:
    candidate = DraftCandidate(text="yh what time", sequence=["yh what time"])

    critique = critique_candidate(
        candidate,
        latest_message="you coming later?",
        relationship_type="close_friend",
        style_score=0.96,
        relevance_score=0.97,
        selected_mode="auto_send",
    )

    assert critique.final_decision == "send"


def test_critic_penalizes_generic_phrases() -> None:
    candidate = DraftCandidate(
        text="Hope you're doing well, that sounds great!",
        sequence=["Hope you're doing well, that sounds great!"],
    )

    critique = critique_candidate(
        candidate,
        latest_message="you coming later?",
        relationship_type="close_friend",
        style_score=0.96,
        relevance_score=0.97,
        selected_mode="auto_send",
    )

    assert critique.fake_sounding is True
    assert critique.final_decision != "send"


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("keep talking like that baby", "reject"),
        ("loool youre trouble", "reject"),
        ("yeah i saw that", "review"),
        ("yo", "review"),
        ("heyy u good", "review"),
    ],
)
def test_critic_blocks_bad_hii_candidates(reply: str, expected: str) -> None:
    candidate = DraftCandidate(text=reply, sequence=[reply])

    critique = critique_candidate(
        candidate,
        latest_message="hii",
        relationship_type="unknown",
        incoming_intent="greeting",
        style_score=0.98,
        relevance_score=0.98,
        selected_mode="auto_send",
    )

    assert critique.final_decision == expected
    if expected == "reject":
        assert "unsafe_or_flirty_phrase" in (critique.blocked_reason or "")


def test_retrieval_for_hii_returns_only_greeting_examples() -> None:
    rows = [
        {"relationship_type": "close_friend", "incoming": "hii", "context": [], "my_reply": "heyy"},
        {"relationship_type": "close_friend", "incoming": "you coming later?", "context": [], "my_reply": "yh what time"},
        {"relationship_type": "romantic_interest", "incoming": "wyd", "context": [], "my_reply": "keep talking like that baby"},
    ]

    result = retrieve_similar_examples_from_rows(
        rows=rows,
        incoming="hii",
        recent_context=[],
        relationship_type="unknown",
        incoming_intent="greeting",
    )

    assert result.examples
    assert {example.intent_type for example in result.examples} == {"greeting"}
    assert all("baby" not in example.my_reply for example in result.examples)


def test_retrieval_intent_hardening_separates_work_and_plans() -> None:
    rows = [
        {"relationship_type": "close_friend", "incoming": "hii", "context": [], "my_reply": "heyy"},
        {"relationship_type": "close_friend", "incoming": "you coming later?", "context": [], "my_reply": "yh what time"},
        {"relationship_type": "university", "incoming": "did you do the work?", "context": [], "my_reply": "not yet"},
    ]

    work = retrieve_similar_examples_from_rows(
        rows=rows,
        incoming="did you do the work?",
        recent_context=[],
        relationship_type="university",
        incoming_intent="question",
    )
    plans = retrieve_similar_examples_from_rows(
        rows=rows,
        incoming="you coming later?",
        recent_context=[],
        relationship_type="close_friend",
        incoming_intent="plans",
    )

    assert work.examples
    assert all(example.intent_type != "greeting" for example in work.examples)
    assert plans.examples
    assert plans.examples[0].incoming == "you coming later?"


def test_exact_hii_regression_generates_safe_short_greetings(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    bundle = service.build_bundle(["hii"], contact_name=None)

    assert bundle.relationship_type == "unknown"
    assert bundle.intent_type == "greeting"
    assert len(bundle.reply_suggestions) >= 3
    assert set(bundle.reply_suggestions).issubset(
        {
            "yo what u saying",
            "heyy what u doing",
            "yo how u been",
            "heyy u good",
            "yo what u on",
        }
    )
    for bad in ("baby", "trouble", "sexy", "love", "xx"):
        assert all(bad not in reply.lower() for reply in bundle.reply_suggestions)
