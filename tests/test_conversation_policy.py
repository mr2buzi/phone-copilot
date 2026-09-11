from __future__ import annotations

from pathlib import Path

from libs.drafting import DraftingService
from libs.drafting.conversation_agenda import build_conversation_agenda
from libs.drafting.conversation_policy import build_conversation_policy, score_reply_against_policy
from libs.drafting.conversation_scene import build_conversation_scene
from scripts.run_conversation_policy_simulation import _randomized_scenarios, _run_check


def _policy_for(context: list[str]):
    scene = build_conversation_scene(context, relationship_type="close_friend", flirt_allowed=True)
    agenda = build_conversation_agenda(context, scene=scene, safe_interests=["cars", "gym", "coding"])
    return build_conversation_policy(context, scene=scene, agenda=agenda, relationship_type="close_friend")


def test_policy_reacts_to_vivid_story_and_rejects_ok() -> None:
    policy = _policy_for(
        [
            "[OTHER]: oi guess what",
            "[ME]: what happened",
            "[OTHER]: i was at home yeah and showering and some guy came in and started running in my living room",
        ]
    )

    assert policy.conversation_job == "react_to_story"
    assert score_reply_against_policy("ok", policy)["policy_violation"] is True
    good = score_reply_against_policy("BRO WHAT why was he running", policy)
    assert good["policy_must_satisfied"] is True
    assert good["policy_fit_score"] >= 0.8


def test_policy_answers_care_check_instead_of_topic_shift() -> None:
    policy = _policy_for(
        [
            "[OTHER]: thats a bit dry mate",
            "[ME]: yeah fairs that was dry icl",
            "[OTHER]: its okay is everything alright?",
        ]
    )

    assert policy.conversation_job == "answer_care_check"
    bad = score_reply_against_policy("fine then what should we talk about", policy)
    assert bad["policy_violation"] is True
    assert "ask_for_topic" in bad["policy_violation_reasons"]


def test_policy_repair_rejects_blaming_user_and_confusion_as_topic() -> None:
    policy = _policy_for(
        [
            "[OTHER]: alex ur being weird",
            "[ME]: u ain't giving me much to work with what u been doing",
            "[OTHER]: bruh wtf",
        ]
    )

    assert policy.conversation_job == "repair_after_weird_or_dry_reply"
    blame = score_reply_against_policy("u ain't giving me much to work with what u been doing", policy)
    weird_topic = score_reply_against_policy("bruh wtf is valid icl what bruh wtf u into", policy)
    assert blame["policy_violation"] is True
    assert weird_topic["policy_violation"] is True
    assert "treat_confusion_as_topic" in weird_topic["policy_violation_reasons"]


def test_policy_answers_affection_without_generic_hook() -> None:
    policy = _policy_for(
        [
            "[OTHER]: hi",
            "[ME]: heyy what u doing",
            "[OTHER]: chilling wby",
            "[ME]: nothing much just chilling",
            "[OTHER]: i missed u baby",
        ]
    )

    assert policy.conversation_job == "answer_affection"
    assert score_reply_against_policy("say less what u doing", policy)["policy_violation"] is True
    good = score_reply_against_policy("that's sweet icl where u been", policy)
    assert good["policy_must_satisfied"] is True
    assert good["policy_fit_score"] >= 0.8


def test_policy_answers_reciprocal_activity_and_missed_wby() -> None:
    policy = _policy_for(
        [
            "[OTHER]: honestly just work wby",
        ]
    )

    assert policy.conversation_job == "answer_reciprocal_activity"
    assert score_reply_against_policy("what u saying", policy)["policy_violation"] is True
    assert score_reply_against_policy("not much just sorting stuff", policy)["policy_must_satisfied"] is True

    repair_policy = _policy_for(
        [
            "[OTHER]: honestly just work wby",
            "[ME]: chilling and still bored?",
            "[OTHER]: prolly sleeping wby",
            "[ME]: u ain't giving me much to work with what u been doing",
            "[OTHER]: i told u and asked u",
        ]
    )

    assert repair_policy.conversation_job == "answer_reciprocal_activity"
    assert repair_policy.must_repair
    assert score_reply_against_policy("not much just sorting stuff", repair_policy)["policy_must_satisfied"] is False
    assert score_reply_against_policy("my bad i missed the wby, not much just sorting stuff", repair_policy)["policy_must_satisfied"] is True


def test_policy_progress_conversation_allows_chosen_topic() -> None:
    policy = _policy_for(
        [
            "[OTHER]: ur boring me",
            "[ME]: what u tryna do then",
            "[OTHER]: idk what to say tbh",
        ]
    )

    assert policy.conversation_job == "progress_conversation"
    good = score_reply_against_policy("alr random question then dream car?", policy)
    assert good["policy_violation"] is False
    assert good["policy_must_satisfied"] is True


def test_service_policy_metadata_and_story_fallback_win(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        training_messages_dir=tmp_path / "training_messages",
    )

    bundle = service.build_bundle_with_context(
        [],
        [
            "[OTHER]: oi guess what",
            "[ME]: what happened",
            "[OTHER]: i was at home yeah and showering and some guy came in and started running in my living room",
        ],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    top = bundle.reply_candidates[0]

    assert top.conversation_job == "react_to_story"
    assert top.policy_fit_score >= 0.8
    assert top.training_style_examples_loaded >= 0
    assert top.text != "ok"
    assert any(term in top.text.lower() for term in ("what", "nah", "bro", "wait", "why", "how"))


def test_policy_simulation_harness_covers_story_care_and_callouts(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        training_messages_dir=tmp_path / "training_messages",
    )

    failures = [_run_check(service, scenario) for scenario in _randomized_scenarios(8, seed=3)]

    assert [failure for failure in failures if failure] == []
