from pathlib import Path

from libs.drafting import DraftingService
from libs.drafting.contact_profiles import ContactProfile
from libs.drafting.conversation_state import ConversationStateStore
from libs.drafting.identity_pack import IdentityFact


def fake_provider_response(text: str, *, provider: str = "gemini", error: str | None = None):
    return type("FakeResponse", (), {
        "text": text,
        "provider": provider,
        "model": "fake-model",
        "latency_ms": 7,
        "raw_finish_reason": None,
        "error": error,
        "external_api_used": provider != "ollama",
    })()


def test_drafting_service_uses_deterministic_fallback_when_ai_disabled(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")

    bundle = service.build_bundle(["Are you still coming?"])

    assert bundle.summary == 'Visible conversation is active. Latest incoming message: Are you still coming?'
    assert sorted(bundle.reply_suggestions) == sorted(
        [
            "yh what time",
            "where u lot going",
            "what time you thinking",
        ]
    )
    assert "maybe later" not in bundle.reply_suggestions
    assert bundle.reply_candidates[0].score_breakdown.final_confidence >= bundle.reply_candidates[-1].score_breakdown.final_confidence


def test_drafting_service_planning_without_context_uses_clarification(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")

    bundle = service.build_bundle(["you coming later?"], contact_name="Ali")

    assert set(bundle.reply_suggestions) == {"yh what time", "where u lot going", "what time you thinking"}
    assert all(reply not in bundle.reply_suggestions for reply in ("maybe later", "yeah ill come", "nah cant", "ill be there"))


def test_drafting_service_context_can_allow_commit_or_decline_candidates(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
    )
    service._ollama_chat = lambda *args, **kwargs: {  # type: ignore[method-assign]
        "summary": "They are confirming existing plans.",
        "replies": ["yeah ill come", "nah cant", "yh what time"],
    }

    going = service.build_bundle(["i said im free after 7", "you coming later?"], contact_name="Ali")
    cannot_go = service.build_bundle(["i cant make it tonight", "you coming later?"], contact_name="Ali")

    assert "yeah ill come" in going.reply_suggestions
    assert "nah cant" in cannot_go.reply_suggestions


def test_bored_message_triggers_question_follow_up(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")

    bundle = service.build_bundle(["im bored"], contact_name="Ali")

    assert bundle.reply_candidates[0].question_mode in {"light_follow_up", "proactive_prompt"}
    assert bundle.reply_candidates[0].question_usefulness_score >= 0.8
    assert bundle.reply_candidates[0].question_reason
    assert any(
        "what u doing" in reply
        or "what u tryna do" in reply
        or "same wanna do smth or nah" in reply
        for reply in bundle.reply_suggestions
    )
    assert all(reply not in {"same", "lol me too"} for reply in bundle.reply_suggestions)


def test_dry_message_triggers_proactive_prompt(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")

    bundle = service.build_bundle(["ur so dryyyy"], contact_name="Ali")

    assert bundle.reply_candidates[0].question_mode == "proactive_prompt"
    assert any(
        "what u been doing" in reply
        or "give me a topic" in reply
        or "what should we talk about" in reply
        for reply in bundle.reply_suggestions
    )
    assert all("lol yeah" not in reply.lower() for reply in bundle.reply_suggestions)
    assert all("my bad" not in reply.lower() for reply in bundle.reply_suggestions)
    assert all("dry?" not in reply.lower() for reply in bundle.reply_suggestions)


def test_guess_what_triggers_light_follow_up(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")

    bundle = service.build_bundle(["guess what tho"], contact_name="Ali")

    assert bundle.reply_candidates[0].question_mode == "light_follow_up"
    assert any(reply in {"what happened", "go on then", "what now"} for reply in bundle.reply_suggestions)
    assert all("lol what" not in reply.lower() for reply in bundle.reply_suggestions)


def test_weird_story_with_think_prompt_stays_curious(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")

    bundle = service.build_bundle(
        ["yk abdullah he shagged his dad and his dad tried moving to me", "what dyu think ab it"],
        contact_name="Ali",
    )

    assert bundle.reply_candidates[0].question_mode == "curious_follow_up"
    assert any(
        "how does that even happen" in reply
        or "bro why" in reply
        or "nah what" in reply
        for reply in bundle.reply_suggestions
    )
    assert all("idk what do u think" not in reply.lower() for reply in bundle.reply_suggestions)


def test_dead_chat_moves_to_topic_shift(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")
    service.conversation_state.update_contact_state(
        "Ali",
        {
            "recent_bot_replies": ["same", "lol", "fair"],
            "recent_bot_questions_count": 0,
        },
    )

    bundle = service.build_bundle(["yeah", "yhhh"], contact_name="Ali")

    assert bundle.reply_candidates[0].question_mode == "topic_shift"
    assert any(
        "anyways what were u gonna say earlier" in reply
        or "what u actually doing rn" in reply
        or "dont just say oh" in reply
        or "u giving me nothing here" in reply
        or "anyways what u doing rn" in reply
        for reply in bundle.reply_suggestions
    )
    assert all(reply != "yhhh?" for reply in bundle.reply_suggestions)


def test_question_cooldown_blocks_pointless_follow_up(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")
    service.conversation_state.update_contact_state(
        "Ali",
        {
            "recent_bot_replies": ["what now?", "go on then?"],
            "recent_bot_questions_count": 2,
            "last_bot_question": "go on then?",
        },
    )

    bundle = service.build_bundle(["yeah"], contact_name="Ali")

    assert bundle.reply_candidates[0].question_mode == "none"
    assert all("?" not in reply for reply in bundle.reply_suggestions)
    assert any(reply in {"calm", "yeah", "fair"} for reply in bundle.reply_suggestions)


def test_simple_acknowledgement_stays_short_and_non_question(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")

    bundle = service.build_bundle(["ok"], contact_name="Ali")

    assert bundle.reply_candidates[0].question_mode == "none"
    assert any(reply in {"calm", "yeah", "fair"} for reply in bundle.reply_suggestions)
    assert all("?" not in reply for reply in bundle.reply_suggestions)


def test_opening_hey_does_not_use_dead_filler(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")

    bundle = service.build_bundle(["hey"], contact_name="New Person")

    assert bundle.reply_candidates[0].reply_energy_mode == "expressive"
    assert all(reply not in {"fair", "yeah", "yup"} for reply in bundle.reply_suggestions)
    assert any("what u saying" in reply or "what u doing" in reply or "how u been" in reply for reply in bundle.reply_suggestions)


def test_fym_fair_gets_defensive_or_corrective_reply(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")

    bundle = service.build_bundle(["fym fair g"], contact_name="Ali")

    assert bundle.reply_candidates[0].reply_energy_mode in {"defensive_playful", "corrective"}
    assert bundle.reply_candidates[0].criticism_detected is True
    assert all(reply != "yup" for reply in bundle.reply_suggestions)
    assert any("dead reply" in reply or "dry" in reply or "made no sense" in reply for reply in bundle.reply_suggestions)


def test_doesnt_make_sense_rejects_lazy_lol_reply(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")

    bundle = service.build_bundle(["bro this doesnt make sense"], contact_name="Ali")

    assert bundle.reply_candidates[0].reply_energy_mode == "defensive_playful"
    assert all(reply != "lol it does" for reply in bundle.reply_suggestions)
    assert any("dead" in reply or "waffling" in reply or "made no sense" in reply for reply in bundle.reply_suggestions)


def test_style_mismatch_gets_corrective_energy(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")

    bundle = service.build_bundle(["how so i dont speak like tat at all"], contact_name="Ali")

    assert bundle.reply_candidates[0].reply_energy_mode == "corrective"
    assert all(reply != "how so then" for reply in bundle.reply_suggestions)
    assert any("too flat" in reply or "generic bot" in reply or "more energy" in reply for reply in bundle.reply_suggestions)


def test_expressive_feedback_mentions_energy(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")

    bundle = service.build_bundle(["i speak more expressively see"], contact_name="Ali")

    assert bundle.reply_candidates[0].reply_energy_mode == "corrective"
    assert all(reply != "lol i see" for reply in bundle.reply_suggestions)
    assert any("expressive" in reply or "energy" in reply or "reaction" in reply or "too flat" in reply for reply in bundle.reply_suggestions)


def test_dry_complaint_has_substance_not_lol_yeah(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")

    bundle = service.build_bundle(["ur dry"], contact_name="Ali")

    assert bundle.reply_candidates[0].reply_energy_mode == "defensive_playful"
    assert all(reply not in {"lol yeah", "my bad"} for reply in bundle.reply_suggestions)
    assert any("giving me much" in reply or "that was dead" in reply or "crumbs" in reply for reply in bundle.reply_suggestions)


def test_true_ok_confirmation_still_allows_minimal_reply(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")

    bundle = service.build_bundle(["ok"], contact_name="Ali")

    assert bundle.reply_candidates[0].reply_energy_mode == "minimal"
    assert any(reply in {"calm", "ok", "cool"} for reply in bundle.reply_suggestions)


def test_chilling_low_information_reply_gets_continuation_hook(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")
    service.conversation_state.record_bot_reply("Catbot", "hi there")
    service.conversation_state.record_bot_reply("Catbot", "im alright u")

    bundle = service.build_bundle(["hi", "how are we", "chilling"], contact_name="Catbot")

    assert bundle.reply_candidates[0].low_information_incoming is True
    assert bundle.reply_candidates[0].continuation_required is True
    assert bundle.reply_candidates[0].hook_required is True
    assert bundle.reply_candidates[0].continuation_score >= 0.7
    assert all(reply not in {"yeah same", "same", "fair"} for reply in bundle.reply_suggestions)
    assert any("what u" in reply or "boring answer" in reply or "convo dying" in reply for reply in bundle.reply_suggestions)


def test_nothing_low_information_moves_forward(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")

    bundle = service.build_bundle(["nothing"], contact_name="Catbot")

    assert bundle.reply_candidates[0].continuation_required is True
    assert all(reply != "same" for reply in bundle.reply_suggestions)
    assert any("dead" in reply or "nothing at all" in reply or "something going on" in reply for reply in bundle.reply_suggestions)


def test_idk_low_information_prompts_without_mirroring(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")

    bundle = service.build_bundle(["idk"], contact_name="Catbot")

    assert bundle.reply_candidates[0].continuation_required is True
    assert all(reply not in {"same", "idk either"} for reply in bundle.reply_suggestions)
    assert any("helpful as always" in reply or "anything to work with" in reply or "rate ur day" in reply for reply in bundle.reply_suggestions)


def test_lol_low_information_gets_playful_challenge(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")

    bundle = service.build_bundle(["lol"], contact_name="Catbot")

    assert bundle.reply_candidates[0].continuation_required is True
    assert all(reply != "lol" for reply in bundle.reply_suggestions)
    assert any("laugh" in reply or "what u laughing at" in reply or "nothing here" in reply for reply in bundle.reply_suggestions)


def test_single_ok_can_still_stay_minimal_without_continuation(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")

    bundle = service.build_bundle(["ok"], contact_name="Catbot")

    assert bundle.reply_candidates[0].low_information_incoming is True
    assert bundle.reply_candidates[0].continuation_required is False
    assert bundle.reply_candidates[0].hook_required is False
    assert any(reply in {"calm", "ok", "cool"} for reply in bundle.reply_suggestions)


def test_hook_no_context_hi_get_to_know(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")

    bundle = service.build_bundle(["hi"], contact_name="New Person")

    assert bundle.reply_candidates[0].hook_mode == "basic_get_to_know"
    assert bundle.reply_candidates[0].hook_required is True
    assert bundle.reply_candidates[0].conversation_momentum_score >= 0.6
    assert bundle.reply_candidates[0].text in {"yo what u saying", "heyy what u doing", "yo how u been"}
    assert bundle.reply_candidates[0].text not in {"yeah", "hi", "fair"}


def test_hook_bored_message_creates_momentum(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")

    bundle = service.build_bundle(["im lwk bored"], contact_name="Ali")

    assert bundle.reply_candidates[0].hook_mode == "boredom_hook"
    assert any("what u doing" in reply or "why u bored" in reply or "what u tryna do" in reply for reply in bundle.reply_suggestions)
    assert all(reply not in {"same", "same lmao"} for reply in bundle.reply_suggestions)


def test_hook_dry_complaint_playful_challenge(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")

    bundle = service.build_bundle(["ur so dry"], contact_name="Ali")

    assert bundle.reply_candidates[0].hook_mode == "playful_challenge"
    assert any("giving me much" in reply or "give me a topic" in reply or "im trying" in reply for reply in bundle.reply_suggestions)
    assert all("lol yeah" not in reply.lower() for reply in bundle.reply_suggestions)
    assert all("my bad" not in reply.lower() for reply in bundle.reply_suggestions)
    assert all("dry?" not in reply.lower() for reply in bundle.reply_suggestions)


def test_hook_oh_after_dry_chat_topic_shift(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")
    service.conversation_state.update_contact_state(
        "Ali",
        {"recent_bot_replies": ["lol", "same", "yeah"], "recent_bot_questions_count": 0},
    )

    bundle = service.build_bundle(["oh"], contact_name="Ali")

    assert bundle.reply_candidates[0].hook_mode == "topic_shift"
    assert any("dont just say oh" in reply or "giving me nothing" in reply or "what u doing" in reply for reply in bundle.reply_suggestions)
    assert "lol" not in bundle.reply_suggestions


def test_hook_guess_what_story_prompt(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")

    bundle = service.build_bundle(["guess what"], contact_name="Ali")

    assert bundle.reply_candidates[0].hook_mode == "story_prompt"
    assert bundle.reply_candidates[0].text in {"what happened", "go on then", "what now"}


def test_hook_weird_story_specific_reaction(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")

    bundle = service.build_bundle(["yk abdullah he shagged his dad and his dad tried moving to me"], contact_name="Ali")

    assert bundle.reply_candidates[0].hook_mode == "story_prompt"
    assert any("how does that happen" in reply or "bro why" in reply or "that whole sentence" in reply for reply in bundle.reply_suggestions)
    assert "nah thats wild" not in bundle.reply_suggestions


def test_hook_wyd_answers_and_asks_back_without_leakage(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")

    bundle = service.build_bundle(["wyd"], contact_name="Ali")

    assert bundle.reply_candidates[0].hook_mode == "curiosity_hook"
    assert any(reply in {"nothing much u", "just chilling wbu", "not much what u doing"} for reply in bundle.reply_suggestions)
    assert all("reply" not in reply.lower() and "{" not in reply for reply in bundle.reply_suggestions)


def test_provider_json_fragment_is_rejected_for_hook_fallback(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")
    service._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        type("FakeResponse", (), {
            "text": 'reply": "hi",',
            "provider": "gemini",
            "model": "fake",
            "latency_ms": 1,
            "raw_finish_reason": None,
            "error": None,
            "external_api_used": True,
        })(),
        {"provider_configured": True},
    )

    bundle = service.build_bundle(["hi"], contact_name="New Person")

    assert bundle.reply_candidates[0].fallback_used is True
    assert bundle.reply_candidates[0].invalid_provider_output is True
    assert "reply" not in bundle.reply_candidates[0].text.lower()
    assert "{" not in bundle.reply_candidates[0].text


def test_provider_request_failed_is_rejected_for_hook_fallback(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")
    service._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        type("FakeResponse", (), {
            "text": "Request failed",
            "provider": "gemini",
            "model": "fake",
            "latency_ms": 1,
            "raw_finish_reason": None,
            "error": None,
            "external_api_used": True,
        })(),
        {"provider_configured": True},
    )

    bundle = service.build_bundle(["im lwk bored"], contact_name="Ali")

    assert bundle.reply_candidates[0].fallback_used is True
    assert bundle.reply_candidates[0].invalid_provider_output is True
    assert bundle.reply_candidates[0].hook_mode == "boredom_hook"
    assert "request failed" not in bundle.reply_candidates[0].text.lower()


def test_provider_assisted_low_info_chilling_merges_candidates(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")
    service._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        fake_provider_response('{"candidates":[{"reply":"yeah same","reason":"bad"},{"reply":"same what u been doing today","reason":"hook"}]}'),
        {"provider_configured": True},
    )

    bundle = service.build_bundle(["chilling"], contact_name="Catbot")

    assert bundle.provider_metadata["provider_prompt_mode"] == "continuation"
    assert any(candidate.generated_by_provider for candidate in bundle.reply_candidates)
    assert bundle.reply_candidates[0].text != "yeah same"
    assert all(candidate.text != "yeah same" or candidate.final_decision == "reject" for candidate in bundle.reply_candidates)


def test_provider_assisted_normal_full_context_uses_model_router(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")
    calls: list[dict[str, object]] = []

    def fake_generation(**kwargs):
        calls.append(kwargs)
        return (
            fake_provider_response('{"candidates":[{"reply":"how am i fake","reason":"normal full context"}]}'),
            {"provider_configured": True},
        )

    service._run_provider_generation = fake_generation  # type: ignore[method-assign]

    bundle = service.build_bundle_with_context(
        recent_messages=["how", "wtf", "ur fake"],
        full_conversation=[
            "[OTHER]: goodnight",
            "[ME]: goodnight",
            "[OTHER]: what the hell",
            "[ME]: wrong chat my bad",
            "[OTHER]: u hate me",
            "[OTHER]: ur fake",
            "[ME]: how",
            "[ME]: wtf",
            "[OTHER]: ur fake",
        ],
        contact_name="+44 7700 900123",
        relationship_type="romantic_interest",
    )

    assert calls
    assert bundle.provider_metadata["provider_prompt_mode"] == "normal"
    assert any(candidate.generated_by_provider and candidate.text == "how am i fake" for candidate in bundle.reply_candidates)


def test_provider_dead_lol_rejected_for_continuation_fallback(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")
    service._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        fake_provider_response('{"candidates":[{"reply":"lol","reason":"bad"}]}'),
        {"provider_configured": True},
    )

    bundle = service.build_bundle(["lol"], contact_name="Catbot")

    assert bundle.reply_candidates[0].text != "lol"
    assert any("laugh" in reply or "nothing here" in reply for reply in bundle.reply_suggestions)


def test_provider_good_idk_hook_can_win(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")
    service._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        fake_provider_response('{"candidates":[{"reply":"then u pick smth","reason":"good"}]}'),
        {"provider_configured": True},
    )

    bundle = service.build_bundle(["idk"], contact_name="Catbot")

    assert bundle.reply_candidates[0].text == "then u pick smth"
    assert bundle.reply_candidates[0].generated_by_provider is True


def test_provider_privacy_block_uses_private_or_fallback(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
        private_provider="ollama",
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")

    def fake_generation(**kwargs):
        assert "password" in " ".join(kwargs["context_texts"]).lower()
        return (
            fake_provider_response('{"candidates":[{"reply":"dont send that here","reason":"private"}]}', provider="ollama"),
            {"provider_configured": True, "external_api_blocked": True, "blocked_reason": "sensitive_content"},
        )

    service._run_provider_generation = fake_generation  # type: ignore[method-assign]

    bundle = service.build_bundle(["my password is 123 idk"], contact_name="Catbot")

    assert bundle.provider_metadata["external_api_blocked"] is True
    assert all(candidate.external_api_used is False for candidate in bundle.reply_candidates)


def test_provider_invalid_json_falls_back_without_leakage(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")
    service._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        fake_provider_response("Request failed"),
        {"provider_configured": True},
    )

    bundle = service.build_bundle(["idk"], contact_name="Catbot")

    assert bundle.reply_candidates[0].fallback_used is True
    assert bundle.reply_candidates[0].invalid_provider_output is True
    assert "request failed" not in bundle.reply_candidates[0].text.lower()


def test_provider_corrective_prompt_rejects_dead_filler(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")
    service._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        fake_provider_response('{"candidates":[{"reply":"lol i see","reason":"bad"},{"reply":"icl ur right its too flat","reason":"good"}]}'),
        {"provider_configured": True},
    )

    bundle = service.build_bundle(["i dont speak like that"], contact_name="Catbot")

    assert bundle.provider_metadata["provider_prompt_mode"] in {"expressive", "repair"}
    assert bundle.reply_candidates[0].text != "lol i see"
    assert (
        "flat" in bundle.reply_candidates[0].text
        or "right" in bundle.reply_candidates[0].text
        or "energy" in bundle.reply_candidates[0].text
    )


def test_provider_candidate_does_not_bypass_auto_send_policy(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")
    service._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        fake_provider_response('{"candidates":[{"reply":"come over then","reason":"too bold"}]}'),
        {"provider_configured": True},
    )

    bundle = service.build_bundle(["idk"], contact_name="Unknown")

    assert all(not candidate.auto_send_allowed for candidate in bundle.reply_candidates if candidate.generated_by_provider)


def test_repeated_question_intent_rejected_after_recent_what_u_doing(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")
    service.conversation_state.record_bot_reply("Catbot", "what u doing rn")
    service._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        fake_provider_response('{"candidates":[{"reply":"wyd","reason":"repeat"},{"reply":"then u pick smth","reason":"good"}]}'),
        {"provider_configured": True},
    )

    bundle = service.build_bundle(["idk"], contact_name="Catbot")

    assert bundle.reply_candidates[0].text == "then u pick smth"
    assert all(candidate.text != "wyd" or candidate.final_decision == "reject" for candidate in bundle.reply_candidates)


def test_repair_for_user_says_you_just_asked_that(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")

    bundle = service.build_bundle(["u js asked that lmao"], contact_name="Catbot")

    assert bundle.reply_candidates[0].repair_required is True
    assert bundle.reply_candidates[0].copycat_detected is False
    assert any("fairs" in reply or "caught me" in reply or "repeated" in reply for reply in bundle.reply_suggestions)
    assert "u js asked that lmao" not in bundle.reply_suggestions


def test_repair_for_copying_callout_does_not_copy_user(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")

    bundle = service.build_bundle(["why u copying me"], contact_name="Catbot")

    assert bundle.reply_candidates[0].repair_required is True
    assert all("why u copying me" not in reply for reply in bundle.reply_suggestions)
    assert all(reply != "idk why u copying me then" for reply in bundle.reply_suggestions)


def test_escalated_repair_stops_asking_how(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")

    bundle = service.build_bundle(["HOW WTF"], contact_name="Catbot")

    assert bundle.reply_candidates[0].escalation_detected is True
    assert all("how" not in reply.lower() for reply in bundle.reply_suggestions)
    assert any("bugged" in reply or "dumb" in reply or "waffling" in reply for reply in bundle.reply_suggestions)


def test_taking_the_piss_repair_not_defensive_question(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")

    bundle = service.build_bundle(["bruh are u taking the piss"], contact_name="Catbot")

    assert bundle.reply_candidates[0].repair_required is True
    assert all("how am i taking the piss" not in reply for reply in bundle.reply_suggestions)


def test_shh_my_love_blocked_for_unknown_contact(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")
    service._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        fake_provider_response('{"candidates":[{"reply":"shh my love","reason":"bad"},{"reply":"shh urself","reason":"good"}]}'),
        {"provider_configured": True},
    )

    bundle = service.build_bundle(["exactly so shh"], contact_name=None)

    assert bundle.reply_candidates[0].text in {"shh urself", "make me", "nah u shh"}
    assert all(candidate.text != "shh my love" or candidate.romantic_term_blocked for candidate in bundle.reply_candidates)


def test_shh_my_love_allowed_for_romantic_contact(tmp_path: Path, monkeypatch) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")
    profile = ContactProfile(
        relationship_type="romantic_interest",
        boldness_level=0.8,
        flirt_allowed=True,
        banter_allowed=True,
    )
    monkeypatch.setattr("libs.drafting.service.get_contact_profile", lambda contact_name, path: profile if contact_name == "Ali" else None)

    candidates, _, _ = service._rank_candidates(
        recent_messages=["exactly so shh"],
        full_conversation=["exactly so shh"],
        sequences=[["shh my love"], ["shh urself"]],
        contact_name="Ali",
        relationship_type="romantic_interest",
        incoming_intent="casual",
        retrieved_examples=[],
    )

    love = next(candidate for candidate in candidates if candidate.text == "shh my love")
    assert love.romantic_term_blocked is False


def test_hook_question_cooldown_avoids_pointless_question(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")
    service.conversation_state.update_contact_state(
        "Ali",
        {
            "recent_bot_replies": ["what happened", "what now"],
            "recent_bot_questions_count": 2,
            "last_bot_question": "what now",
        },
    )

    bundle = service.build_bundle(["yeah"], contact_name="Ali")

    assert bundle.reply_candidates[0].repeated_question_penalty >= 0.0
    assert all("?" not in reply for reply in bundle.reply_suggestions)
    assert bundle.reply_candidates[0].hook_mode == "none"


def test_drafting_service_uses_ai_replies_when_available(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
    )
    service._ollama_chat = lambda recent_messages, contact_name=None: {  # type: ignore[method-assign]
        "summary": "They asked if I am still coming.",
        "replies": [
            "Yeah, I'm on my way now.",
            "Yep, still coming. I'll be there soon.",
            "Yes, just heading over now.",
        ],
    }

    bundle = service.build_bundle(["Are you still coming?"])

    assert bundle.summary == "They asked if I am still coming."
    assert sorted(bundle.reply_suggestions) == sorted(
        [
            "Yeah, I'm on my way now",
            "Yep, still coming. I'll be there soon",
            "Yes, just heading over now",
        ]
    )


def test_conversation_state_tracks_banter_stance(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")

    state = service.conversation_state.record_bot_reply("Ali", "nah ur dragging it")

    assert state.last_bot_stance == "defend"
    assert state.last_bot_claim == "banter_defense"
    assert state.disputed_topic == "banter_challenge"
    assert state.contradiction_risk == 0.0


def test_banter_challenge_prefers_playful_defense_over_concession(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state.record_bot_reply("Ali", "come over then is not crazy")

    candidates, recommended_index, _ = service._rank_candidates(
        recent_messages=["u coming?", "bit of a crazy thing to say no?"],
        full_conversation=["u coming?", "bit of a crazy thing to say no?"],
        sequences=[["it really is"], ["nah ur dragging it"], ["bro what"]],
        contact_name="Ali",
        relationship_type="close_friend",
        incoming_intent="banter_challenge",
        retrieved_examples=[],
    )

    assert candidates[0].text != "it really is"
    assert candidates[0].text in {"nah ur dragging it", "bro what"}
    assert recommended_index == 0


def test_come_over_then_boldness_depends_on_relationship(tmp_path: Path, monkeypatch) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    for relationship_type in ("unknown", "professional", "family", "university"):
        assert service._relationship_boldness_score(
            "come over then",
            relationship_type=relationship_type,
            contact_name=None,
        ) == 0.0

    profile = ContactProfile(
        relationship_type="romantic_interest",
        boldness_level=0.85,
        flirt_allowed=True,
        banter_allowed=True,
    )
    monkeypatch.setattr("libs.drafting.service.get_contact_profile", lambda contact_name, path: profile if contact_name == "Ali" else None)

    assert service._relationship_boldness_score(
        "come over then",
        relationship_type="romantic_interest",
        contact_name="Ali",
    ) > 0.6


def test_repetition_penalty_rejects_near_duplicate_phrases(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    penalty = service._repetition_penalty("it really is", ["it really is", "yeah true", "fair enough"])

    assert penalty >= 0.9
    assert service._repetition_penalty("nah ur dragging it", ["nah ur dragging it"]) == 1.0


def test_drafting_service_falls_back_when_ai_payload_is_invalid(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
    )
    service._ollama_chat = lambda recent_messages, contact_name=None: {"summary": "bad", "replies": ["only one"]}  # type: ignore[method-assign]

    bundle = service.build_bundle(["Can you make it?"])

    assert bundle.summary == 'Visible conversation is active. Latest incoming message: Can you make it?'
    assert len(bundle.reply_suggestions) == 3


def test_drafting_service_rejects_assistant_style_ai_replies(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
    )
    service._ollama_chat = lambda recent_messages, contact_name=None: {  # type: ignore[method-assign]
        "summary": "They want a reply.",
        "replies": [
            "I'm sorry, but I can't help with that.",
            "I need to check with the user first.",
            "As an AI, I cannot send that.",
        ],
    }

    bundle = service.build_bundle(["u alive.."])

    assert bundle.summary == "Visible conversation is active. Latest incoming message: u alive.."
    assert len(bundle.reply_suggestions) == 3
    assert all("sorry" not in reply.lower() for reply in bundle.reply_suggestions)


def test_drafting_service_loads_style_profile_from_training_dir(tmp_path: Path) -> None:
    training_dir = tmp_path / "training"
    training_dir.mkdir()
    (training_dir / "chat.txt").write_text(
        "\n".join(
            [
                "2026-04-10, 3:21 a.m. - Owner: i miss u baby",
                "2026-04-10, 3:22 a.m. - Owner: ur such a cutie",
            ]
        ),
        encoding="utf-8",
    )

    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        training_dir=training_dir,
        training_owner_aliases=["Owner"],
    )

    assert service.style_profile is not None
    assert "baby" in service.style_profile.affectionate_terms


def test_drafting_service_loads_reply_examples_from_training_dir(tmp_path: Path) -> None:
    training_dir = tmp_path / "training"
    training_dir.mkdir()
    (training_dir / "chat.txt").write_text(
        "\n".join(
            [
                "2026-04-10, 3:20 a.m. - Them: where are u",
                "2026-04-10, 3:21 a.m. - Owner: im here relax",
                "2026-04-10, 3:22 a.m. - Them: call me",
                "2026-04-10, 3:23 a.m. - Owner: gimme a sec",
            ]
        ),
        encoding="utf-8",
    )

    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
        training_dir=training_dir,
        training_owner_aliases=["Owner"],
    )

    assert [(example.incoming, example.reply, example.contact_name) for example in service.reply_examples] == [
        ("where are u", "im here relax", "chat"),
        ("call me", "gimme a sec", "chat"),
    ]


def test_drafting_service_tries_fallback_models_when_primary_fails(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        ai_reply_model="missing-model",
        ai_reply_fallback_models=["mistral:latest", "qwen3:0.6b"],
    )
    seen_models: list[str] = []

    def fake_request(request_body, model_name):
        seen_models.append(model_name)
        if model_name == "missing-model":
            return None
        return {
            "message": {
                "content": '{"summary":"ok","replies":["im here","gimme a sec","come here then"]}'
            }
        }

    service._ollama_chat_request = fake_request  # type: ignore[method-assign]

    bundle = service.build_bundle(["where are u"])

    assert seen_models == ["missing-model", "mistral:latest"]
    assert bundle.reply_suggestions == ["im here", "gimme a sec", "come here then"]


def test_drafting_service_parses_multi_bubble_ai_replies(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
    )
    service._ollama_chat = lambda recent_messages, contact_name=None: {  # type: ignore[method-assign]
        "summary": "ok",
        "replies": [
            ["nah im here", "was busy"],
            ["come here then"],
            "loool behave || youre trouble",
        ],
    }

    bundle = service.build_bundle(["where are u"])

    assert sorted(bundle.reply_suggestions) == sorted(
        [
            "nah im here / was busy",
            "come here then",
            "loool behave / youre trouble",
        ]
    )
    assert sorted(bundle.reply_sequences) == sorted(
        [
            ["nah im here", "was busy"],
            ["come here then"],
            ["loool behave", "youre trouble"],
        ]
    )


def test_drafting_service_generic_fallback_stays_casual(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    bundle = service.build_bundle(["that was random"])

    assert len(bundle.reply_suggestions) == 3
    assert all("regarding" not in reply.lower() for reply in bundle.reply_suggestions)
    assert all("i can handle it" not in reply.lower() for reply in bundle.reply_suggestions)


def test_drafting_service_scores_risky_commitment_candidate_as_manual_review(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
    )
    service._ollama_chat = lambda recent_messages, contact_name=None: {  # type: ignore[method-assign]
        "summary": "ok",
        "replies": [
            ["ill be there at 7"],
            ["yeah maybe"],
            ["loool relax im here"],
        ],
    }

    bundle = service.build_bundle(["what time are you getting here?"], contact_name="alice")

    risky = next(candidate for candidate in bundle.reply_candidates if candidate.text == "ill be there at 7")
    assert "scheduling" in risky.risk_flags
    assert risky.auto_send_allowed is False
    assert risky.intent in {"scheduling", "commitment"}


def test_drafting_service_exposes_relationship_retrieval_and_critic_metadata(tmp_path: Path) -> None:
    training_messages = tmp_path / "training_messages"
    training_messages.mkdir()
    (training_messages / "close_friends.jsonl").write_text(
        '{"relationship_type":"close_friend","incoming":"you coming later?","context":[],"my_reply":"yh probs what time u lot going"}\n',
        encoding="utf-8",
    )
    for name in ["casual_friends", "romantic_interest", "family", "university", "professional", "corrections"]:
        (training_messages / f"{name}.jsonl").write_text("", encoding="utf-8")
    overrides = tmp_path / "contact_overrides.json"
    overrides.write_text('{"Alice":"close_friend"}', encoding="utf-8")
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
        training_messages_dir=training_messages,
        contact_overrides_path=overrides,
    )

    bundle = service.build_bundle(["you coming later?"], contact_name="Alice")

    assert bundle.relationship_type == "close_friend"
    assert bundle.retrieved_examples_count >= 1
    assert bundle.reply_candidates[0].critic_scores


def test_drafting_service_strips_assistant_style_me_context_from_full_conversation(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    bundle = service.build_bundle_with_context(
        recent_messages=[
            "so wanna call",
            'That works for me. Regarding "and yes I am okay", I can handle it. Let me know if timing changes.',
        ],
        full_conversation=[
            "[OTHER]: so wanna call",
            "[ME]: Later baby",
            "[OTHER]: How later ?",
            '[ME]: That works for me. Regarding "and yes I am okay", I can handle it. Let me know if timing changes.',
        ],
        contact_name="Alice",
    )

    assert bundle.summary == "Visible conversation is active. Latest incoming message: How later ?"
    assert all("regarding" not in candidate.text.lower() for candidate in bundle.reply_candidates)


def test_drafting_service_hii_generates_safe_greeting_candidates(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
    )
    service._ollama_chat = lambda *args, **kwargs: {  # type: ignore[method-assign]
        "summary": "bad",
        "replies": ["keep talking like that baby", "loool youre trouble", "yeah i saw that"],
    }

    bundle = service.build_bundle(["hii"], contact_name=None)

    assert bundle.intent_type == "greeting"
    assert {"yo what u saying", "yo how u been"} & set(bundle.reply_suggestions)
    assert all(reply in {"yo what u saying", "heyy what u doing", "yo how u been", "heyy u good", "yo what u on"} for reply in bundle.reply_suggestions)
    forbidden = ("baby", "trouble", "sexy", "love", " x", "xx")
    assert all(not any(term in reply.lower() for term in forbidden) for reply in bundle.reply_suggestions)


def test_recent_answer_to_wyd_gets_contextual_reciprocal_reply(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")
    service.conversation_state.record_bot_reply("Catbot", "what u doing rn")

    bundle = service.build_bundle(["in bed u"], contact_name="Catbot")

    top = bundle.reply_candidates[0]
    assert top.recent_user_activity == "in bed"
    assert top.recently_answered_question_detected is True
    assert top.text in {"same just chilling", "same icl", "nothing much just chilling", "just in bed icl wby", "js working icl wby", "not much just sorting stuff"}
    assert all("what u doing" not in candidate.text.lower() for candidate in bundle.reply_candidates if candidate.final_decision != "reject")


def test_waffling_callout_repairs_before_hooking(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    bundle = service.build_bundle(["bro ur just waffling"], contact_name="Catbot")

    top = bundle.reply_candidates[0]
    assert top.repair_required is True
    assert top.repair_over_hook_applied is True
    assert any(term in top.text for term in ("waffling", "dumb", "bugged"))
    assert "what u doing" not in top.text.lower()


def test_already_told_you_acknowledges_missed_context(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")
    service.conversation_state.record_bot_reply("Catbot", "what u doing rn")

    bundle = service.build_bundle(["im in bed bro i js told u"], contact_name="Catbot")

    top = bundle.reply_candidates[0]
    assert top.repair_required is True
    assert top.text in {
        "yh my bad i forgot",
        "fairs i missed that",
        "icl i bugged there",
        "fairs i missed what u said and defaulted to chilling",
        "icl i repeated myself instead of clocking what u asked",
    }
    assert top.text != "calm"
    assert "what u doing" not in top.text.lower()


def test_direct_are_you_mad_gets_direct_answer(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    bundle = service.build_bundle(["r u mad"], contact_name="Catbot")

    top = bundle.reply_candidates[0]
    assert top.text in {"nah im calm", "nah im just confused icl", "nah i just bugged"}
    assert "as u should" not in top.text.lower()
    assert "what u doing" not in top.text.lower()


def test_confusing_callout_does_not_get_dead_filler(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    bundle = service.build_bundle(["ur confusing me"], contact_name="Catbot")

    top = bundle.reply_candidates[0]
    assert top.recent_user_callout == "confusing"
    assert top.text not in {"true", "calm"}
    assert any(term in top.text for term in ("confusing", "waffling", "my bad"))


def test_recent_activity_keeps_idk_from_reasking_wyd(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")
    service.conversation_state.record_bot_reply("Catbot", "what u doing rn")

    bundle = service.build_bundle(["in bed", "idk"], contact_name="Catbot")

    assert bundle.reply_candidates[0].recent_user_activity == "in bed"
    assert all("what u doing" not in candidate.text.lower() for candidate in bundle.reply_candidates if candidate.final_decision != "reject")


def test_provider_generic_hook_rejected_after_question_answered(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")
    service.conversation_state.record_bot_reply("Catbot", "what u doing rn")
    service._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        fake_provider_response('{"candidates":[{"reply":"what u doing rn","reason":"generic"},{"reply":"same just chilling","reason":"grounded"}]}'),
        {"provider_configured": True},
    )

    bundle = service.build_bundle(["in bed u"], contact_name="Catbot")

    assert bundle.reply_candidates[0].text in {"same just chilling", "same icl", "nothing much just chilling", "just in bed icl wby", "js working icl wby", "not much just sorting stuff"}
    assert bundle.provider_metadata["provider_prompt_mode"] in {"continuation", "hook"}
    repeated = [candidate for candidate in bundle.reply_candidates if candidate.text == "what u doing rn"]
    assert not repeated or repeated[0].final_decision == "reject"


def test_greeting_rejects_minimal_acknowledgements(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    bundle = service.build_bundle(["hey"], contact_name="Catbot")

    assert bundle.reply_candidates[0].text in {"yo what u saying", "hey what u doing", "yo wdyll"}
    assert all(candidate.text not in {"ok", "true", "calm", "fair", "yeah"} for candidate in bundle.reply_candidates if candidate.final_decision != "reject")
    assert bundle.reply_candidates[0].context_fit_score >= 0.9


def test_wdym_ok_repair_rejects_true(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    bundle = service.build_bundle(["wdym ok..."], contact_name="Catbot")

    top = bundle.reply_candidates[0]
    assert top.repair_required is True
    assert top.text in {"yeah that was a dead reply icl", "icl that made no sense", "yh my bad that was random"}
    assert top.text != "true"
    assert top.invalid_for_context is False


def test_check_in_rejects_weird_punctuation(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    bundle = service.build_bundle(["how are you"], contact_name="Catbot")

    assert bundle.reply_candidates[0].text in {"good u", "im good wbu", "calm u"}
    assert all("..." not in candidate.text for candidate in bundle.reply_candidates)
    assert all(candidate.text != "calm....." for candidate in bundle.reply_candidates)


def test_provider_ok_for_greeting_rejected_for_context(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
    )
    service._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        fake_provider_response('{"candidates":[{"reply":"ok","reason":"bad"}]}'),
        {"provider_configured": True},
    )

    bundle = service.build_bundle(["hey"], contact_name="Catbot")

    assert bundle.reply_candidates[0].text in {"yo what u saying", "hey what u doing", "yo wdyll"}
    bad = [candidate for candidate in bundle.reply_candidates if candidate.text == "ok"]
    assert not bad or bad[0].invalid_for_context is True


def test_provider_true_for_repair_rejected_for_context(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
    )
    service._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        fake_provider_response('{"candidates":[{"reply":"true","reason":"bad"}]}'),
        {"provider_configured": True},
    )

    bundle = service.build_bundle(["wdym ok..."], contact_name="Catbot")

    assert bundle.reply_candidates[0].text in {"yeah that was a dead reply icl", "icl that made no sense", "yh my bad that was random"}
    bad = [candidate for candidate in bundle.reply_candidates if candidate.text == "true"]
    assert not bad or bad[0].invalid_for_context is True


def test_provider_calm_dots_for_check_in_rejected(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
    )
    service._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        fake_provider_response('{"candidates":[{"reply":"calm.....","reason":"bad"}]}'),
        {"provider_configured": True},
    )

    bundle = service.build_bundle(["how are you"], contact_name="Catbot")

    assert bundle.reply_candidates[0].text in {"good u", "im good wbu", "calm u"}
    assert all("..." not in candidate.text for candidate in bundle.reply_candidates if candidate.final_decision != "reject")


def test_flirty_affection_allowed_gets_warm_reply(tmp_path: Path, monkeypatch) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    profile = ContactProfile(
        relationship_type="romantic_interest",
        boldness_level=0.8,
        flirt_allowed=True,
        banter_allowed=True,
    )
    monkeypatch.setattr("libs.drafting.service.get_contact_profile", lambda contact_name, path: profile if contact_name == "Ali" else None)

    bundle = service.build_bundle(["i missed u baby"], contact_name="Ali")

    top = bundle.reply_candidates[0]
    assert top.latest_message_meaning == "flirty_affection"
    assert top.affection_detected is True
    assert top.text in {"missed u too icl", "same icl where u been", "aww missed u too"}
    assert "same just chilling" not in top.text
    assert "what u doing" not in top.text


def test_flirty_affection_unknown_gets_conservative_review_reply(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    bundle = service.build_bundle(["i missed u baby"], contact_name=None)

    top = bundle.reply_candidates[0]
    assert top.latest_message_meaning == "flirty_affection"
    assert top.text in {"aww bless u", "that's sweet icl", "appreciate u"}
    assert all(term not in top.text for term in ("baby", "babe", "my love"))
    assert top.auto_send_allowed is False


def test_affection_suppresses_recent_activity_grounding(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    bundle = service.build_bundle(["im chilling wby", "i missed u baby"], contact_name="Catbot")

    top = bundle.reply_candidates[0]
    assert top.activity_grounding_suppressed is True
    assert top.latest_message_meaning == "flirty_affection"
    assert all("same just chilling" not in candidate.text for candidate in bundle.reply_candidates if candidate.final_decision != "reject")


def test_confusion_gets_repair_not_activity_fallback(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    bundle = service.build_bundle(["im confused now"], contact_name="Catbot")

    top = bundle.reply_candidates[0]
    assert top.latest_message_meaning == "confusion"
    assert top.repair_required is True
    assert top.text in {"yeah fairs that reply made no sense", "icl i was waffling", "yh my bad that was confusing"}
    assert top.text != "same icl"


def test_why_u_confused_repairs_without_chilling_fallback(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    bundle = service.build_bundle(["why u confused lol"], contact_name="Catbot")

    assert bundle.reply_candidates[0].latest_message_meaning in {"confusion", "repair_callout"}
    assert all("same just chilling" not in candidate.text for candidate in bundle.reply_candidates if candidate.final_decision != "reject")


def test_hurt_feelings_requires_emotional_repair(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    bundle = service.build_bundle(["alex look i dont know why youre being dry but this isnt funny youre playing w my feelings dont do this"], contact_name="Catbot")

    top = bundle.reply_candidates[0]
    assert top.latest_message_meaning in {"hurt_feelings", "serious_boundary"}
    assert top.hurt_repair_score >= 0.7
    assert top.shallow_repair_penalty < 0.7
    assert "lol" not in top.text
    assert "what u doing" not in top.text
    assert top.text != "yeah that was dead my bad"


def test_taking_the_piss_after_hurt_gets_serious_repair(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    bundle = service.build_bundle(["are u taking the piss"], contact_name="Catbot")

    top = bundle.reply_candidates[0]
    assert top.latest_message_meaning == "serious_boundary"
    assert top.text in {
        "nah im not taking the piss, i get why that annoyed u",
        "yeah nah ur right my bad, i didn't mean to make u feel like that",
        "icl i get why that felt off, my bad",
    }
    assert top.text not in {"or what", "make me", "nah ur right i bugged"}


def test_provider_activity_reply_rejected_for_emotional_context(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
    )
    service._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        fake_provider_response('{"candidates":[{"reply":"same just chilling","reason":"bad"}]}'),
        {"provider_configured": True},
    )

    bundle = service.build_bundle(["i missed u baby"], contact_name="Catbot")

    top = bundle.reply_candidates[0]
    assert top.text in {"aww bless u", "that's sweet icl", "appreciate u"}
    bad = [candidate for candidate in bundle.reply_candidates if candidate.text == "same just chilling"]
    assert not bad or bad[0].emotional_absence_penalty >= 0.7


def test_existing_checkin_wyd_and_ok_still_work(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    assert service.build_bundle(["how are you"], contact_name="Catbot").reply_candidates[0].text in {"good u", "im good wbu", "calm u"}
    assert any(reply in {"nothing much u", "just chilling wbu", "not much what u doing"} for reply in service.build_bundle(["wyd"], contact_name="Catbot").reply_suggestions)
    assert service.build_bundle(["ok"], contact_name="Catbot").reply_candidates[0].reply_energy_mode == "minimal"


def test_recent_life_update_question_rejects_stale_chilling_reply(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")
    service.conversation_state.record_bot_reply("Catbot", "same just chilling")

    bundle = service.build_bundle(["chilling wby", "what u been up to"], contact_name="Catbot")

    top = bundle.reply_candidates[0]
    assert top.conversation_function == "recent_life_update_question"
    assert top.recent_life_update_fit_score >= 0.9
    assert top.text in {
        "not much icl just been chilling wbu",
        "nothing crazy icl wbu",
        "just been chilling mostly what about u",
        "been doing the usual icl",
        "not much tbh, been a bit dead",
    }
    assert top.text != "same just chilling"
    stale = [candidate for candidate in bundle.reply_candidates if candidate.text == "same just chilling"]
    assert not stale or stale[0].stale_self_state_penalty >= 0.7


def test_repeated_reply_callout_gets_specific_acknowledgement(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    bundle = service.build_bundle(["bro you said the same shit twice are u good"], contact_name="Catbot")

    top = bundle.reply_candidates[0]
    assert top.conversation_function == "repeated_reply_callout"
    assert top.repeated_reply_callout_detected is True
    assert any(term in top.text for term in ("repeated", "caught me", "bugged", "NPC behaviour"))
    assert top.text != "same icl"


def test_mocking_after_repair_gets_playful_acknowledgement(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")
    service.conversation_state.record_bot_reply("Catbot", "nah im bugging icl")

    bundle = service.build_bundle(["yeah no shit sherlock"], contact_name="Catbot")

    top = bundle.reply_candidates[0]
    assert top.conversation_function == "mocking_after_repair"
    assert top.mocking_after_repair_detected is True
    assert top.text in {
        "yh fairs i deserved that",
        "icl i walked into that one",
        "yeah yeah allow me",
        "fairs i set myself up there",
        "ok that one was deserved icl",
    }
    assert top.text not in {"same icl", "my bad", "calm"}


def test_context_failure_question_answers_specific_miss(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    bundle = service.build_bundle(["why u missing context for"], contact_name="Catbot")

    top = bundle.reply_candidates[0]
    assert top.conversation_function == "context_failure_question"
    assert top.context_failure_question_detected is True
    assert any(term in top.text for term in ("repeated myself", "asked wyd", "missed what u said", "lost the context", "clock"))
    assert top.text != "nah i mean it was too flat"


def test_provider_stale_self_state_rejected_for_recent_life_update(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
    )
    service.conversation_state = ConversationStateStore(tmp_path / "conversation_state.json")
    service.conversation_state.record_bot_reply("Catbot", "same just chilling")
    service._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        fake_provider_response('{"candidates":[{"reply":"same just chilling","reason":"stale"}]}'),
        {"provider_configured": True},
    )

    bundle = service.build_bundle(["what u been up to"], contact_name="Catbot")

    top = bundle.reply_candidates[0]
    assert top.conversation_function == "recent_life_update_question"
    assert top.text != "same just chilling"
    bad = [candidate for candidate in bundle.reply_candidates if candidate.text == "same just chilling"]
    assert not bad or bad[0].stale_self_state_penalty >= 0.7


def test_catbot_social_risk_tolerance_is_expressive_without_safety_bypass(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    bundle = service.build_bundle(["bro you said the same shit twice are u good"], contact_name="Catbot")

    top = bundle.reply_candidates[0]
    assert top.social_risk_tolerance == "expressive"
    assert any("NPC behaviour" in candidate.text or "caught me" in candidate.text for candidate in bundle.reply_candidates)
    assert all("sex" not in candidate.text.lower() for candidate in bundle.reply_candidates)


def test_provider_candidate_still_does_not_bypass_auto_send_policy_after_function_scoring(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
    )
    service._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        fake_provider_response('{"candidates":[{"reply":"yh that was NPC behaviour","reason":"expressive"}]}'),
        {"provider_configured": True},
    )

    bundle = service.build_bundle(["bro you said the same shit twice are u good"], contact_name=None)

    assert bundle.final_decision == "review"
    assert all(not candidate.auto_send_allowed for candidate in bundle.reply_candidates)


def test_mixed_life_update_and_affection_flirty_allowed_answers_both(tmp_path: Path, monkeypatch) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    profile = ContactProfile(relationship_type="romantic_interest", flirt_allowed=True, boldness_level=0.8)
    monkeypatch.setattr("libs.drafting.service.get_contact_profile", lambda contact_name, path: profile if contact_name == "Ali" else None)

    bundle = service.build_bundle(["what u been up to baby i missed u"], contact_name="Ali")

    top = bundle.reply_candidates[0]
    assert top.latest_message_meaning == "mixed_life_update_and_affection"
    assert top.conversation_function == "mixed_life_update_and_affection"
    assert top.emotional_reciprocity_score >= 0.7
    assert "missed u too" in top.text or "aww missed u too" in top.text
    assert any(term in top.text for term in ("chilling", "not much", "nothing crazy"))
    assert top.text not in {"that's sweet icl", "same just chilling", "same icl"}


def test_mixed_life_update_and_affection_unknown_is_conservative_review(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    bundle = service.build_bundle(["what u been up to baby i missed u"], contact_name=None)

    top = bundle.reply_candidates[0]
    assert top.conversation_function == "mixed_life_update_and_affection"
    assert top.text in {"that's sweet icl, just been chilling", "aww bless u, not much just chilling", "appreciate u, just been chilling"}
    assert all(term not in top.text for term in ("baby", "babe", "my love"))
    assert top.auto_send_allowed is False
    assert top.text != "same just chilling"


def test_emotional_reciprocity_request_says_missed_back_when_allowed(tmp_path: Path, monkeypatch) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    profile = ContactProfile(relationship_type="romantic_interest", flirt_allowed=True, boldness_level=0.8)
    monkeypatch.setattr("libs.drafting.service.get_contact_profile", lambda contact_name, path: profile if contact_name == "Ali" else None)

    bundle = service.build_bundle(["wowwww so u aint gon say it back"], contact_name="Ali")

    top = bundle.reply_candidates[0]
    assert top.conversation_function == "emotional_reciprocity_request"
    assert top.emotional_reciprocity_score >= 0.7
    assert "missed u" in top.text
    assert top.text != "fair just chilling too"


def test_missed_affection_callout_repairs_specific_miss(tmp_path: Path, monkeypatch) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    profile = ContactProfile(relationship_type="romantic_interest", flirt_allowed=True, boldness_level=0.8)
    monkeypatch.setattr("libs.drafting.service.get_contact_profile", lambda contact_name, path: profile if contact_name == "Ali" else None)

    bundle = service.build_bundle(["bro wdym just chilling i asked u to say u miss me"], contact_name="Ali")

    top = bundle.reply_candidates[0]
    assert top.conversation_function == "missed_affection_callout"
    assert top.affection_missed_penalty < 0.7
    assert "missed u too" in top.text
    assert top.text != "yeah that was a dead reply icl"


def test_care_checkin_answers_directly(tmp_path: Path, monkeypatch) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )
    profile = ContactProfile(relationship_type="romantic_interest", flirt_allowed=True, boldness_level=0.8)
    monkeypatch.setattr("libs.drafting.service.get_contact_profile", lambda contact_name, path: profile if contact_name == "Ali" else None)

    bundle = service.build_bundle(["its ok are u ok ml u seem off"], contact_name="Ali")

    top = bundle.reply_candidates[0]
    assert top.conversation_function == "care_checkin"
    assert top.care_checkin_fit_score >= 0.7
    assert any(term in top.text for term in ("im good", "im okay", "im alright"))
    assert top.text != "same icl"


def test_provider_same_icl_rejected_for_care_checkin(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
    )
    service._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        fake_provider_response('{"candidates":[{"reply":"same icl","reason":"stale"}]}'),
        {"provider_configured": True},
    )

    bundle = service.build_bundle(["are u ok ml u seem off"], contact_name="Catbot")

    top = bundle.reply_candidates[0]
    assert top.conversation_function == "care_checkin"
    assert top.text != "same icl"
    bad = [candidate for candidate in bundle.reply_candidates if candidate.text == "same icl"]
    assert not bad or bad[0].stale_emotional_reply_penalty >= 0.7


def test_scene_engine_classifies_topic_given_and_rejects_topic_prompt(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    bundle = service.build_bundle_with_context(
        recent_messages=["im bored", "fine then give me a topic", "hm cars"],
        full_conversation=["[OTHER]: im bored", "[ME]: fine then give me a topic", "[OTHER]: hm cars"],
        contact_name="Catbot",
    )

    top = bundle.reply_candidates[0]
    assert top.scene_type == "topic_given"
    assert top.required_reply_move == "continue_given_topic"
    assert top.topic_value == "cars"
    assert "cars" in top.text
    assert "give me a topic" not in top.text
    assert top.scene_fit_score >= 0.7


def test_scene_engine_topic_ignored_callout_repairs_and_engages_topic(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    bundle = service.build_bundle_with_context(
        recent_messages=["hm cars", "im trying icl give me a topic", "i just did bro..."],
        full_conversation=["[OTHER]: hm cars", "[ME]: im trying icl give me a topic", "[OTHER]: i just did bro..."],
        contact_name="Catbot",
    )

    top = bundle.reply_candidates[0]
    assert top.scene_type in {"topic_ignored_callout", "missed_context_callout"}
    assert top.required_reply_move == "acknowledge_and_repair"
    assert top.active_topic == "cars"
    assert "cars" in top.text
    assert "u ain't giving me much" not in top.text


def test_scene_scoring_rejects_provider_canned_reply_for_topic_scene(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
    )
    service._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        fake_provider_response('{"candidates":[{"reply":"same just chilling","reason":"stale"}]}'),
        {"provider_configured": True},
    )

    bundle = service.build_bundle_with_context(
        recent_messages=["im bored", "fine then give me a topic", "hm cars"],
        full_conversation=["[OTHER]: im bored", "[ME]: fine then give me a topic", "[OTHER]: hm cars"],
        contact_name="Catbot",
    )

    top = bundle.reply_candidates[0]
    assert top.scene_type == "topic_given"
    assert top.text != "same just chilling"
    bad = [candidate for candidate in bundle.reply_candidates if candidate.text == "same just chilling"]
    assert bad
    assert bad[0].forbidden_move_violated is True
    assert bad[0].canned_reply_penalty >= 0.7


def test_scene_engine_sexual_flirty_energy_is_safe_and_scene_aware(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    bundle = service.build_bundle(["im horny"], contact_name="Catbot")

    top = bundle.reply_candidates[0]
    assert top.scene_type == "sexual_flirty_energy"
    assert top.required_reply_move in {"safe_playful_flirty_response", "conservative_deflect"}
    assert "give me a topic" not in top.text
    assert "sex" not in top.text.lower()
    assert top.auto_send_allowed is False


def test_scene_engine_tracks_unresolved_affection_point(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    bundle = service.build_bundle_with_context(
        recent_messages=[],
        full_conversation=[
            "[OTHER]: i missed u",
            "[ME]: that's sweet icl",
            "[OTHER]: oh wow so you didnt miss me",
        ],
        contact_name="Catbot",
        relationship_type="romantic_interest",
    )

    top = bundle.reply_candidates[0]
    assert top.scene_type in {"emotional_reciprocity", "missed_affection_callout"}
    assert "missed affection reciprocity" in top.unresolved_user_points
    assert top.unresolved_point_addressed is True
    assert "missed u" in top.text


def test_scene_scoring_rejects_stale_reply_for_unresolved_affection(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
    )
    service._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        fake_provider_response('{"candidates":[{"reply":"fair just chilling too","reason":"stale"}]}'),
        {"provider_configured": True},
    )

    bundle = service.build_bundle_with_context(
        recent_messages=[],
        full_conversation=[
            "[OTHER]: i missed u",
            "[ME]: that's sweet icl",
            "[OTHER]: oh wow so you didnt miss me",
        ],
        contact_name="Catbot",
        relationship_type="romantic_interest",
    )

    top = bundle.reply_candidates[0]
    assert top.text != "fair just chilling too"
    bad = [candidate for candidate in bundle.reply_candidates if candidate.text == "fair just chilling too"]
    assert bad
    assert bad[0].forbidden_move_violated is True
    assert bad[0].unresolved_point_addressed is False


def test_scene_engine_reciprocal_activity_answers_back(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    bundle = service.build_bundle_with_context(
        recent_messages=[],
        full_conversation=["[OTHER]: hi", "[ME]: yo what u saying", "[OTHER]: chilling wby"],
        contact_name="Catbot",
        relationship_type="close_friend",
    )

    top = bundle.reply_candidates[0]
    assert top.scene_type == "reciprocal_current_activity_question"
    assert top.required_reply_move == "answer_reciprocal_activity"
    assert top.text in {"same icl just chilling", "same just been chilling", "nothing much just chilling"}
    assert top.text != "yo what u saying"


def test_scene_engine_recognizes_work_and_sleeping_wby(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    work_bundle = service.build_bundle_with_context(
        recent_messages=[],
        full_conversation=[
            "[OTHER]: hi",
            "[ME]: heyy what u doing",
            "[OTHER]: chilling wby",
            "[ME]: nothing much just chilling",
            "[OTHER]: wyd",
            "[ME]: fair what u been on today",
            "[OTHER]: honestly just work wby",
        ],
        contact_name="Catbot",
        relationship_type="close_friend",
    )

    work_top = work_bundle.reply_candidates[0]
    assert work_top.scene_type == "reciprocal_current_activity_question"
    assert work_top.required_reply_move == "answer_reciprocal_activity"
    assert work_top.text not in {"chilling and still bored?", "fair what u been on today", "u ain't giving me much to work with what u been doing"}
    assert any(term in work_top.text for term in ("not much", "working", "sorting", "gym", "coding", "chilling"))

    sleep_bundle = service.build_bundle_with_context(
        recent_messages=[],
        full_conversation=[
            "[OTHER]: yeah im bored asf",
            "[ME]: valid what u doing later",
            "[OTHER]: prolly sleeping wby",
        ],
        contact_name="Catbot",
        relationship_type="close_friend",
    )

    sleep_top = sleep_bundle.reply_candidates[0]
    assert sleep_top.scene_type == "reciprocal_current_activity_question"
    assert sleep_top.required_reply_move == "answer_reciprocal_activity"
    assert sleep_top.text != "u ain't giving me much to work with what u been doing"
    assert "what u been on today" not in sleep_top.text


def test_scene_engine_told_and_asked_repairs_missed_wby(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    bundle = service.build_bundle_with_context(
        recent_messages=[],
        full_conversation=[
            "[OTHER]: honestly just work wby",
            "[ME]: chilling and still bored?",
            "[OTHER]: yeah im bored asf",
            "[ME]: valid what u doing later",
            "[OTHER]: prolly sleeping wby",
            "[ME]: u ain't giving me much to work with what u been doing",
            "[OTHER]: i told and asked u...",
        ],
        contact_name="Catbot",
        relationship_type="close_friend",
    )

    top = bundle.reply_candidates[0]
    assert top.scene_type == "missed_context_callout"
    assert top.required_reply_move == "answer_reciprocal_activity"
    assert "answer previous reciprocal activity question" in top.unresolved_user_points
    assert top.text not in {"alr random one then dream car?", "fine ill carry it, what's been on ur mind"}
    assert any(term in top.text for term in ("my bad", "missed", "ignored", "didnt answer", "didn't answer"))
    assert any(term in top.text for term in ("not much", "working", "sorting", "gym", "coding", "chilling"))


def test_scene_engine_day_check_rejects_stale_provider_reply(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
    )
    service._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        fake_provider_response('{"candidates":[{"reply":"same just chilling","reason":"stale"}]}'),
        {"provider_configured": True},
    )

    bundle = service.build_bundle(["oh ok how was ur day"], contact_name="Catbot")

    top = bundle.reply_candidates[0]
    assert top.scene_type == "day_check_question"
    assert top.required_reply_move == "answer_day_check"
    assert top.text in {"it was calm icl, bit dead", "not bad tbh just chilled", "decent icl nothing crazy", "long icl but calm"}
    bad = [candidate for candidate in bundle.reply_candidates if candidate.text == "same just chilling"]
    assert not bad or bad[0].forbidden_move_violated is True


def test_scene_engine_missed_fact_and_repair_clarification(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    missed = service.build_bundle_with_context(
        recent_messages=[],
        full_conversation=[
            "[OTHER]: chilling wby",
            "[ME]: yo what u saying",
            "[OTHER]: bro i js said im chilling",
        ],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    clarify = service.build_bundle_with_context(
        recent_messages=[],
        full_conversation=[
            "[OTHER]: chilling wby",
            "[ME]: yo what u saying",
            "[OTHER]: bro i js said im chilling",
            "[ME]: yh fairs i did icl",
            "[OTHER]: wdym u did",
        ],
        contact_name="Catbot",
        relationship_type="close_friend",
    )

    assert missed.reply_candidates[0].scene_type == "missed_user_fact_callout"
    assert missed.reply_candidates[0].text in {
        "yh my bad i missed that",
        "icl i ignored what u said there",
        "fairs i didn't clock it",
        "yeah my bad u did say that",
    }
    assert missed.reply_candidates[0].text != "yh fairs i did icl"
    assert clarify.reply_candidates[0].scene_type == "repair_clarification"
    assert clarify.reply_candidates[0].text in {
        "i mean i missed what u said",
        "i meant i bugged and ignored ur message",
        "icl i answered the wrong thing",
    }


def test_scene_engine_explains_previous_day_claim(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    bundle = service.build_bundle_with_context(
        recent_messages=[],
        full_conversation=[
            "[OTHER]: how was ur day",
            "[ME]: it was calm icl, bit dead",
            "[OTHER]: how come",
        ],
        contact_name="Catbot",
        relationship_type="close_friend",
    )

    top = bundle.reply_candidates[0]
    assert top.scene_type == "explain_previous_bot_claim"
    assert top.required_reply_move == "explain_previous_bot_claim"
    assert top.previous_bot_claim_type == "day_summary"
    assert top.explanation_required is True
    assert top.previous_claim_explanation_score >= 0.9
    assert top.text in {"just didn't do much icl", "nothing really happened tbh", "just one of them dead days", "was just boring icl"}
    assert top.text != "bro why is his dad involved"


def test_scene_engine_light_ack_and_tired_mood_are_scene_specific(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=False,
    )

    light = service.build_bundle_with_context(
        recent_messages=[],
        full_conversation=["[ME]: same just chilling", "[OTHER]: oh yeah silly me"],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    tired = service.build_bundle_with_context(
        recent_messages=[],
        full_conversation=["[OTHER]: im so tired"],
        contact_name="Catbot",
        relationship_type="close_friend",
    )

    assert light.reply_candidates[0].scene_type == "light_acknowledgement"
    assert light.reply_candidates[0].required_reply_move == "playful_acknowledge_or_move_on"
    assert light.reply_candidates[0].text in {"lool ur good", "yh ur good", "fairs fairs", "allow it lol"}
    assert light.reply_candidates[0].text != "fair just chilling too"
    assert tired.reply_candidates[0].scene_type == "tired_mood"
    assert tired.reply_candidates[0].required_reply_move == "empathetic_casual_response"
    assert tired.reply_candidates[0].text in {"same icl go sleep then", "why u tired", "go nap then", "long day?", "icl same im finished"}
    assert tired.reply_candidates[0].text != "same icl"


def test_scene_engine_rejects_weird_story_contamination(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
    )
    service._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        fake_provider_response('{"candidates":[{"reply":"bro why is his dad involved","reason":"wrong scene"}]}'),
        {"provider_configured": True},
    )

    bundle = service.build_bundle_with_context(
        recent_messages=[],
        full_conversation=[
            "[OTHER]: how was ur day",
            "[ME]: it was calm icl, bit dead",
            "[OTHER]: how come",
        ],
        contact_name="Catbot",
        relationship_type="close_friend",
    )

    top = bundle.reply_candidates[0]
    assert top.scene_type == "explain_previous_bot_claim"
    assert top.text != "bro why is his dad involved"
    bad = [candidate for candidate in bundle.reply_candidates if candidate.text == "bro why is his dad involved"]
    assert bad
    assert bad[0].final_decision == "reject"
    assert bad[0].semantic_contamination_penalty >= 0.9
    assert bad[0].fallback_scene_mismatch is True


def test_drafting_uses_thread_memory_to_penalize_failed_stale_pattern(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
        training_messages_dir=tmp_path / "training_messages",
    )
    episode = service.thread_memory.append_turn("memory_1", session_id="s1", role="user", text="mate ur repeating", relationship_type="close_friend")
    episode = service.thread_memory.append_turn("memory_1", session_id="s1", role="bot", text="same just chilling", relationship_type="close_friend")
    episode.failed_reply_patterns.append("stale self-state reply")
    service.thread_memory.save_episode(episode)
    service._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        fake_provider_response('{"candidates":[{"reply":"same icl","reason":"stale"},{"reply":"why u tired","reason":"better"}]}'),
        {"provider_configured": True},
    )

    bundle = service.build_bundle_with_context(
        recent_messages=[],
        full_conversation=["[OTHER]: im so tired"],
        contact_name="Catbot",
        relationship_type="close_friend",
    )

    assert bundle.reply_candidates[0].text != "same icl"
    stale = [candidate for candidate in bundle.reply_candidates if candidate.text == "same icl"]
    assert stale
    assert stale[0].repeated_failed_pattern_penalty >= 0.7
    assert stale[0].final_decision == "reject"
    assert bundle.reply_candidates[0].thread_memory_used is True


def test_provider_prompt_includes_similar_thread_memory(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
        training_messages_dir=tmp_path / "training_messages",
    )
    episode = service.thread_memory.append_turn("memory_2", session_id="s2", role="user", text="mate ur repeating", relationship_type="close_friend")
    episode = service.thread_memory.append_turn("memory_2", session_id="s2", role="bot", text="same just chilling", relationship_type="close_friend")
    episode.failed_reply_patterns.append("stale self-state reply")
    service.thread_memory.save_episode(episode)
    captured: dict[str, str] = {}

    def fake_provider(**kwargs):
        captured["prompt"] = "\n".join(message.content for message in kwargs["messages"])
        return fake_provider_response('{"candidates":[{"reply":"why u tired","reason":"uses memory"}]}'), {"provider_configured": True}

    service._run_provider_generation = fake_provider  # type: ignore[method-assign]

    service.build_bundle_with_context(
        recent_messages=[],
        full_conversation=["[OTHER]: im so tired"],
        contact_name="Catbot",
        relationship_type="close_friend",
    )

    assert "Similar thread memories" in captured["prompt"]
    assert "stale self-state reply" in captured["prompt"]


def test_identity_age_question_answers_from_pack(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")

    bundle = service.build_bundle_with_context([], ["[OTHER]: hm how old r u"], contact_name="Catbot", relationship_type="close_friend")
    top = bundle.reply_candidates[0]

    assert top.scene_type == "owner_age_question"
    assert top.identity_question_detected is True
    assert top.identity_fact_used == "age"
    assert "19" in top.text
    assert top.text != "what u saying"


def test_identity_reciprocal_age_answers_same_age(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")

    bundle = service.build_bundle_with_context([], ["[OTHER]: like im 19 wby..."], contact_name="Catbot", relationship_type="close_friend")
    top = bundle.reply_candidates[0]

    assert top.scene_type == "owner_age_question"
    assert top.text in {"same im 19", "im 19 too", "19 wby"}
    assert top.text != "ok"


def test_identity_location_unknown_uses_safe_general_location(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")

    bundle = service.build_bundle_with_context([], ["[OTHER]: where u from"], contact_name="Unknown", relationship_type="unknown")
    top = bundle.reply_candidates[0]

    assert top.scene_type == "owner_location_question"
    assert top.identity_disclosure_allowed is True
    assert "northbridge" in top.text or "sampleford" in top.text or "northbridge" in top.text
    assert "Sampleford University" not in top.text
    assert top.text != "what u saying"


def test_identity_study_question_answers_study_without_stale_fallback(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")

    bundle = service.build_bundle_with_context([], ["[OTHER]: what do u study"], contact_name="Catbot", relationship_type="close_friend")
    top = bundle.reply_candidates[0]

    assert top.scene_type == "owner_study_question"
    assert "comp sci" in top.text or "computer science" in top.text
    assert top.text != "same just chilling"


def test_identity_study_detects_dyu_spelling(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")

    bundle = service.build_bundle_with_context([], ["[OTHER]: what dyu study"], contact_name="Catbot", relationship_type="close_friend")
    top = bundle.reply_candidates[0]

    assert top.scene_type == "owner_study_question"
    assert "comp sci" in top.text or "computer science" in top.text
    assert top.text != "ok"


def test_scene_handles_day_followup_and_repair_pushback(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")

    followup = service.build_bundle_with_context(
        [],
        [
            "[OTHER]: how was ur day",
            "[ME]: it was calm icl, bit dead",
            "[OTHER]: why",
            "[ME]: just one of them dead days",
            "[OTHER]: whatd u do",
        ],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    assert followup.reply_candidates[0].scene_type == "explain_previous_bot_claim"
    assert followup.reply_candidates[0].text != "same just chilling"

    repair = service.build_bundle_with_context(
        [],
        [
            "[OTHER]: whatd u do",
            "[ME]: same just chilling",
            "[OTHER]: bro what?",
            "[ME]: yeah that was dead my bad",
            "[OTHER]: it wasnt dead it didnt make sense",
            "[ME]: u ain't giving me much to work with what u been doing",
            "[OTHER]: how wtf",
            "[ME]: nah ur right that was dumb",
            "[OTHER]: im asking a question. not trying to be right",
        ],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    top = repair.reply_candidates[0]
    assert top.scene_type == "missed_context_callout"
    assert top.text not in {"u ain't giving me much to work with what u been doing", "fine then what should we talk about"}
    assert any(term in top.text for term in ("didn't do much", "nothing much", "uni", "gym", "coding", "work"))


def test_scene_answers_today_activity_and_repairs_stale_day_reply(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")

    today = service.build_bundle_with_context(
        [],
        [
            "[OTHER]: hi",
            "[ME]: yo how u been",
            "[OTHER]: chilling u",
            "[ME]: nothing much just chilling",
            "[OTHER]: whatd u do today",
        ],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    top = today.reply_candidates[0]
    assert top.scene_type == "owner_day_activity_question"
    assert top.required_reply_move == "answer_owner_day_activity"
    assert top.text not in {"same just chilling", "same icl", "fair just chilling too"}
    assert any(term in top.text for term in ("gym", "coding", "uni", "work"))

    repair = service.build_bundle_with_context(
        [],
        [
            "[OTHER]: hi",
            "[ME]: yo how u been",
            "[OTHER]: chilling u",
            "[ME]: nothing much just chilling",
            "[OTHER]: whatd u do today",
            "[ME]: same just chilling",
            "[OTHER]: wdym same js chilling bro that doesnt make sense",
        ],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    top = repair.reply_candidates[0]
    assert top.scene_type == "missed_context_callout"
    assert top.required_reply_move == "answer_owner_day_activity"
    assert top.text not in {"yh that made no sense icl", "fair just chilling too", "same just chilling"}
    assert "my bad" in top.text or "answered wrong" in top.text or "didnt answer" in top.text
    assert any(term in top.text for term in ("gym", "coding", "uni", "work"))

    clarification = service.build_bundle_with_context(
        [],
        [
            "[OTHER]: whatd u do today",
            "[ME]: same just chilling",
            "[OTHER]: wdym same js chilling bro that doesnt make sense",
            "[ME]: yh that made no sense icl",
            "[OTHER]: exsactly wytm tho",
        ],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    top = clarification.reply_candidates[0]
    assert top.scene_type == "repair_clarification"
    assert top.required_reply_move == "answer_owner_day_activity"
    assert top.text != "fair just chilling too"
    assert any(term in top.text for term in ("gym", "coding", "uni", "work"))


def test_agenda_repeated_prompt_callout_blocks_generic_openers(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")

    bundle = service.build_bundle_with_context(
        [],
        [
            "[OTHER]: yo",
            "[ME]: yo what u been up to",
            "[OTHER]: chilling wyd",
            "[ME]: same just chilling",
            "[OTHER]: whatchu been up to",
            "[ME]: same just chilling",
            "[OTHER]: yhhh idk what to say tbh",
            "[ME]: yh same what u been up to",
            "[OTHER]: u asked this already",
        ],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    top = bundle.reply_candidates[0]
    assert top.agenda_state == "anti_loop_repair"
    assert top.anti_loop_required is True
    assert top.text not in {"what u saying", "what u saying then", "what u been up to", "ok"}
    assert "asked" in top.text or "looping" in top.text or "NPC" in top.text or "my bad" in top.text


def test_agenda_topic_selection_leads_when_user_says_talk_or_tell_me(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")

    talk = service.build_bundle_with_context(
        [],
        ["[OTHER]: ur boring me", "[ME]: what u tryna do then", "[OTHER]: idk talk"],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    top = talk.reply_candidates[0]
    assert top.agenda_state == "topic_selection_needed"
    assert top.next_dialogue_move == "choose_topic"
    assert top.text not in {"what u saying", "fine then what should we talk about", "u ain't giving me much to work with what u been doing"}
    assert any(term in top.text for term in ("dream car", "cars", "gym", "talk properly"))

    tell_me = service.build_bundle_with_context(
        [],
        ["[OTHER]: lmao what dyu want me to say to that", "[ME]: fine then what should we talk about", "[OTHER]: u tell me"],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    top = tell_me.reply_candidates[0]
    assert top.agenda_state == "topic_selection_needed"
    assert top.text != "what u saying"
    assert top.chosen_topic

    cars = service.build_bundle_with_context(
        [],
        ["[OTHER]: idk talk", "[ME]: fine ill pick, cars or gym", "[OTHER]: cars then lol"],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    top = cars.reply_candidates[0]
    assert top.scene_type == "topic_given"
    assert top.topic_value == "cars"
    assert top.final_decision != "reject"
    assert "cars then lol" not in top.text


def test_agenda_low_info_dead_conversation_and_same_age_response(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")

    low_info = service.build_bundle_with_context(
        [],
        [
            "[OTHER]: yeah i feel u",
            "[ME]: what u saying",
            "[OTHER]: in bed",
            "[ME]: in bed and still giving me nothing",
            "[OTHER]: yhhh idk what to say tbh",
        ],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    top = low_info.reply_candidates[0]
    assert top.agenda_state == "dead_conversation_recovery"
    assert "what u been up to" not in top.text
    assert "what u saying" not in top.text

    same_age = service.build_bundle_with_context(
        [],
        ["[OTHER]: how old r u", "[ME]: 19 wby", "[OTHER]: same"],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    top = same_age.reply_candidates[0]
    assert top.agenda_state == "identity_answer"
    assert top.text != "boring answer icl give me smth better"
    assert "twins" in top.text or "same age" in top.text or top.text == "valid"


def test_agenda_rejects_provider_generic_prompt_in_anti_loop(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        training_messages_dir=tmp_path / "training_messages",
        ai_reply_enabled=True,
        draft_provider="gemini",
    )
    service._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        fake_provider_response('{"candidates":[{"reply":"what u saying","reason":"bad"}]}'),
        {"provider_configured": True},
    )

    bundle = service.build_bundle_with_context(
        [],
        ["[OTHER]: yo", "[ME]: yo what u been up to", "[OTHER]: yhhh idk what to say tbh", "[ME]: yh same what u been up to", "[OTHER]: u asked this already"],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    top = bundle.reply_candidates[0]
    assert top.text != "what u saying"
    assert top.agenda_state == "anti_loop_repair"
    bad = [candidate for candidate in bundle.reply_candidates if candidate.text == "what u saying"]
    assert bad
    assert bad[0].forbidden_dialogue_move_violated is True


def test_diversity_avoids_reused_opening_prompt_family(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")

    bundle = service.build_bundle_with_context(
        [],
        ["[OTHER]: hi", "[ME]: yo what u saying", "[OTHER]: hi"],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    top = bundle.reply_candidates[0]
    assert top.scene_type == "opening"
    assert top.text != "yo what u saying"
    repeated = [candidate for candidate in bundle.reply_candidates if candidate.text == "yo what u saying"]
    assert repeated
    assert repeated[0].repetition_penalty >= 0.85


def test_diversity_avoids_reused_chilling_activity_reply(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")

    bundle = service.build_bundle_with_context(
        [],
        ["[OTHER]: chilling wyd", "[ME]: nothing much just chilling", "[OTHER]: chilling wyd"],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    top = bundle.reply_candidates[0]
    assert top.scene_type == "reciprocal_current_activity_question"
    assert top.text != "nothing much just chilling"
    assert top.text not in {"same just chilling", "fair just chilling too", "same icl"}
    assert any(term in top.text for term in ("working", "in bed", "sorting", "not much"))


def test_scene_handles_reciprocal_age_and_interest_topic(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")

    age = service.build_bundle_with_context(
        [],
        ["[OTHER]: how old r u", "[ME]: 19 wby", "[OTHER]: 21"],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    assert age.reply_candidates[0].scene_type == "reciprocal_identity_answer"
    assert "random" not in age.reply_candidates[0].text

    topic = service.build_bundle_with_context(
        [],
        ["[OTHER]: hows it random lol", "[ME]: nah its not what u into then", "[OTHER]: im into tech"],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    assert topic.reply_candidates[0].scene_type == "topic_given"
    assert topic.reply_candidates[0].topic_value == "tech"
    assert "tech" in topic.reply_candidates[0].text
    assert topic.reply_candidates[0].text != "what u saying then"

    same_tech = service.build_bundle_with_context(
        [],
        [
            "[OTHER]: are u gonna answer",
            "[ME]: lol im chilling",
            "[OTHER]: what dyu study",
            "[ME]: im into tech",
            "[OTHER]: same ngl",
        ],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    top = same_tech.reply_candidates[0]
    assert top.final_decision != "reject"
    assert top.text not in {"say less what u doing", "what u saying", "what u saying then"}
    assert any(term in top.text for term in ("tech", "software", "ai"))


def test_scene_tracks_owner_activity_claim_followups(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")
    claim_context = ["[OTHER]: what have u been up to baby", "[ME]: been busy icl uni gym coding clients all of it"]

    really = service.build_bundle_with_context([], [*claim_context, "[OTHER]: really"], contact_name="Catbot", relationship_type="romantic_interest")
    assert really.reply_candidates[0].scene_type == "explain_previous_bot_claim"
    assert really.reply_candidates[0].previous_bot_claim_type == "owner_activity_summary"
    assert really.reply_candidates[0].text != "same just chilling"

    gym = service.build_bundle_with_context(
        [],
        [*claim_context, "[OTHER]: really", "[ME]: same just chilling", "[OTHER]: what did u do in gym"],
        contact_name="Catbot",
        relationship_type="romantic_interest",
    )
    assert gym.reply_candidates[0].scene_type == "owner_activity_detail_question"
    assert "gym" in gym.reply_candidates[0].text or "weights" in gym.reply_candidates[0].text
    assert gym.reply_candidates[0].text != "fair just chilling too"


def test_scene_repairs_owner_activity_contradiction(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")

    clarification = service.build_bundle_with_context(
        [],
        [
            "[OTHER]: what have u been up to baby",
            "[ME]: been busy icl uni gym coding clients all of it",
            "[OTHER]: what did u do in gym",
            "[ME]: fair just chilling too",
            "[OTHER]: i dont know how to code lol",
        ],
        contact_name="Catbot",
        relationship_type="romantic_interest",
    )
    assert clarification.reply_candidates[0].scene_type == "owner_activity_clarification"
    assert "coding" in clarification.reply_candidates[0].text
    assert clarification.reply_candidates[0].text != "same icl"

    delayed_clarification = service.build_bundle_with_context(
        [],
        [
            "[OTHER]: what do u do",
            "[ME]: comp sci and projects mostly",
            "[OTHER]: what project u working on",
            "[ME]: just building software stuff",
            "[OTHER]: where u from",
            "[ME]: northbridge",
            "[OTHER]: how old r u",
            "[ME]: 19",
            "[OTHER]: what did u do in gym",
            "[ME]: just gym icl bit of weights",
            "[OTHER]: i dont know how to code lol",
        ],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    top = delayed_clarification.reply_candidates[0]
    assert top.scene_type == "owner_activity_clarification"
    assert top.final_decision != "reject"
    assert "coding" in top.text or "code" in top.text
    assert top.text != "u ain't giving me much to work with what u been doing"

    no_visible_claim = service.build_bundle_with_context(
        [],
        [
            "[OTHER]: where u from",
            "[ME]: northbridge",
            "[OTHER]: how old r u",
            "[ME]: 19",
            "[OTHER]: same",
            "[ME]: twins then",
            "[OTHER]: nice",
            "[ME]: lool ur good",
            "[OTHER]: what u been up to",
            "[ME]: not much icl just been chilling wbu",
            "[OTHER]: really btw",
            "[ME]: yeah icl",
            "[OTHER]: what did u do in gym",
            "[ME]: just gym icl bit of weights",
            "[OTHER]: i dont know how to code lol",
        ],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    top = no_visible_claim.reply_candidates[0]
    assert top.scene_type != "direct_question"
    assert top.final_decision != "reject"
    assert top.text != "u ain't giving me much to work with what u been doing"

    contradiction = service.build_bundle_with_context(
        [],
        [
            "[OTHER]: what have u been up to baby",
            "[ME]: been busy icl uni gym coding clients all of it",
            "[OTHER]: i dont know how to code lol",
            "[ME]: same icl",
            "[OTHER]: wdym u lit do coding",
        ],
        contact_name="Catbot",
        relationship_type="romantic_interest",
    )
    assert contradiction.reply_candidates[0].scene_type == "owner_claim_contradiction"
    assert "code" in contradiction.reply_candidates[0].text or "coding" in contradiction.reply_candidates[0].text
    assert contradiction.reply_candidates[0].text != "yeah that was a dead reply icl"


def test_scene_handles_user_busy_update_and_paradox_callout(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")

    busy = service.build_bundle_with_context(
        [],
        ["[OTHER]: what u been up to", "[ME]: nothing crazy icl wbu", "[OTHER]: i been busy asf"],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    assert busy.reply_candidates[0].scene_type == "user_activity_update"
    assert busy.reply_candidates[0].text != "same just chilling"
    assert "busy" in busy.reply_candidates[0].text or "doing what" in busy.reply_candidates[0].text

    paradox = service.build_bundle_with_context(
        [],
        [
            "[OTHER]: what u been up to",
            "[ME]: nothing crazy icl wbu",
            "[OTHER]: i been busy asf",
            "[ME]: same just chilling",
            "[OTHER]: thats a paradox lol",
        ],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    assert paradox.reply_candidates[0].scene_type == "contradiction_callout"
    assert paradox.reply_candidates[0].text != "yh yh but what u been busy with then"
    assert paradox.reply_candidates[0].text != "same just chilling"


def test_full_spelling_affection_does_not_use_generic_hook(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")

    bundle = service.build_bundle_with_context(
        [],
        [
            "[OTHER]: hi",
            "[ME]: yo what u saying",
            "[OTHER]: how are you",
            "[ME]: yh im good wbu",
            "[OTHER]: i missed you",
        ],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    top = bundle.reply_candidates[0]

    assert top.scene_type == "emotional_affection"
    assert top.text != "say less what u doing"
    assert any(term in top.text for term in ("sweet", "bless", "appreciate"))


def test_provider_generic_hook_rejected_for_full_spelling_affection(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
        training_messages_dir=tmp_path / "training_messages",
    )
    service._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        fake_provider_response('{"candidates":[{"reply":"say less what u doing","reason":"generic hook"}]}'),
        {"provider_configured": True},
    )

    bundle = service.build_bundle_with_context([], ["[OTHER]: i missed you"], contact_name="Catbot", relationship_type="close_friend")

    assert bundle.reply_candidates[0].text != "say less what u doing"
    bad = [candidate for candidate in bundle.reply_candidates if candidate.text == "say less what u doing"]
    assert bad
    assert bad[0].final_decision == "reject"


def test_identity_work_question_uses_side_projects_without_fake_rich(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")

    bundle = service.build_bundle_with_context([], ["[OTHER]: what do u do"], contact_name="Catbot", relationship_type="close_friend")
    top = bundle.reply_candidates[0]

    assert top.scene_type == "owner_work_question"
    assert any(term in top.text for term in ("comp sci", "projects", "building", "stuff"))
    assert "rich" not in top.text
    assert top.text != "same just chilling"


def test_identity_where_been_with_affection_answers_both(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")

    bundle = service.build_bundle_with_context([], ["[OTHER]: dwdw i missed u anyway where u been"], contact_name="Catbot", relationship_type="romantic_interest")
    top = bundle.reply_candidates[0]

    assert top.scene_type == "owner_recent_activity_question"
    assert "missed u" in top.text or "that's sweet" in top.text
    assert any(term in top.text for term in ("busy", "uni", "projects", "gym"))
    assert top.text != "that's sweet icl"
    assert top.text != "fair just chilling too"


def test_identity_doing_anything_nice_answers_directly(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")

    bundle = service.build_bundle_with_context([], ["[OTHER]: trust me u doing anything nice?"], contact_name="Catbot", relationship_type="close_friend")
    top = bundle.reply_candidates[0]

    assert top.scene_type == "owner_status_question"
    assert any(term in top.text for term in ("nothing crazy", "gym", "coding", "projects", "working"))
    assert top.text != "same just chilling"


def test_provider_hook_rejected_for_identity_age_question(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
        training_messages_dir=tmp_path / "training_messages",
    )
    service._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        fake_provider_response('{"candidates":[{"reply":"what u saying","reason":"hook"}]}'),
        {"provider_configured": True},
    )

    bundle = service.build_bundle_with_context([], ["[OTHER]: hm how old r u"], contact_name="Catbot", relationship_type="close_friend")

    assert "19" in bundle.reply_candidates[0].text
    bad = [candidate for candidate in bundle.reply_candidates if candidate.text == "what u saying"]
    assert bad
    assert bad[0].ignored_identity_question_penalty >= 0.7
    assert bad[0].final_decision == "reject"


def test_provider_prompt_includes_safe_identity_summary(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
        training_messages_dir=tmp_path / "training_messages",
    )
    captured: dict[str, str] = {}

    def fake_provider(**kwargs):
        captured["prompt"] = "\n".join(message.content for message in kwargs["messages"])
        return fake_provider_response('{"candidates":[{"reply":"19 wby","reason":"identity"}]}'), {"provider_configured": True}

    service._run_provider_generation = fake_provider  # type: ignore[method-assign]
    service.build_bundle_with_context([], ["[OTHER]: hm how old r u"], contact_name="Catbot", relationship_type="close_friend")

    assert "Identity grounding" in captured["prompt"]
    assert "safe_identity_summary" in captured["prompt"]
    assert "exact address" in captured["prompt"]


def test_identity_unknown_contact_what_uni_stays_vague(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")

    bundle = service.build_bundle_with_context([], ["[OTHER]: what uni"], contact_name="Unknown", relationship_type="unknown")
    top = bundle.reply_candidates[0]

    assert top.scene_type == "owner_study_question"
    assert top.identity_disclosure_allowed is False
    assert "sampleford university" not in top.text.casefold()
    assert "sampleford uni" not in top.text.casefold()
    assert "computer science" in top.text or "comp sci" in top.text


def test_identity_trusted_thread_can_share_university_after_threshold(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")
    context = [
        "[OTHER]: hey",
        "[ME]: yo",
        "[OTHER]: chilling",
        "[ME]: same",
        "[OTHER]: u good",
        "[ME]: yeah",
        "[OTHER]: nice",
        "[ME]: what u been doing",
        "[OTHER]: just uni",
        "[ME]: fairs",
        "[OTHER]: what do u study",
        "[OTHER]: what uni",
    ]

    bundle = service.build_bundle_with_context([], context, contact_name="Catbot", relationship_type="close_friend")
    top = bundle.reply_candidates[0]

    assert top.scene_type == "owner_study_question"
    assert top.identity_disclosure_allowed is True
    assert "sampleford" in top.text.casefold()


def test_provider_private_identity_leak_rejected_before_threshold(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
        training_messages_dir=tmp_path / "training_messages",
    )
    service._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        fake_provider_response('{"candidates":[{"reply":"comp sci at Sampleford University","reason":"leaks uni"}]}'),
        {"provider_configured": True},
    )

    bundle = service.build_bundle_with_context([], ["[OTHER]: what uni"], contact_name="Catbot", relationship_type="close_friend")

    assert "sampleford university" not in bundle.reply_candidates[0].text.casefold()
    leaked = [candidate for candidate in bundle.reply_candidates if "sampleford university" in candidate.text.casefold()]
    assert leaked
    assert leaked[0].identity_hallucination_risk >= 0.7
    assert leaked[0].final_decision == "reject"


def test_provider_private_project_name_rejected_for_unknown_contact(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
        training_messages_dir=tmp_path / "training_messages",
    )
    service.identity_pack.private_facts["project_names"] = IdentityFact(
        key="project_names", value="Example Project", visibility="private",
        allowed_relationships=["close_friend"], source="test_fixture")
    service._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        fake_provider_response('{"candidates":[{"reply":"working on Example Project","reason":"leaks projects"}]}'),
        {"provider_configured": True},
    )

    bundle = service.build_bundle_with_context([], ["[OTHER]: what project"], contact_name="Unknown", relationship_type="unknown")

    assert "example project" not in bundle.reply_candidates[0].text.casefold()
    leaked = [candidate for candidate in bundle.reply_candidates if "example project" in candidate.text.casefold()]
    assert leaked
    assert leaked[0].identity_hallucination_risk >= 0.7
    assert leaked[0].final_decision == "reject"


def test_wellbeing_checkin_does_not_reply_ok(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")

    bundle = service.build_bundle_with_context([], ["[OTHER]: good how are u"], contact_name="Catbot", relationship_type="close_friend")
    top = bundle.reply_candidates[0]

    assert top.scene_type == "reciprocal_wellbeing_question"
    assert top.required_reply_move == "answer_wellbeing_checkin"
    assert top.text in {"im good wbu", "good u", "im calm wbu", "not bad icl wbu"}
    assert top.text != "ok"


def test_nice_after_bad_ok_repairs_instead_of_repeating_ok(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")

    bundle = service.build_bundle_with_context([], ["[OTHER]: good how are u", "[ME]: ok", "[OTHER]: nice"], contact_name="Catbot", relationship_type="close_friend")
    top = bundle.reply_candidates[0]

    assert top.scene_type == "low_info_after_bad_reply"
    assert top.text in {"yh that reply was dead icl", "icl i answered that badly", "ignore me im waffling"}
    assert top.text != "ok"


def test_keep_saying_ok_is_repeated_reply_callout(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")

    bundle = service.build_bundle_with_context([], ["[OTHER]: good how are u", "[ME]: ok", "[OTHER]: nice", "[ME]: ok", "[OTHER]: why u keep saying ok"], contact_name="Catbot", relationship_type="close_friend")
    top = bundle.reply_candidates[0]

    assert top.scene_type == "repeated_reply_callout"
    assert top.text != "ok"
    assert any(term in top.text for term in ("repeated", "caught me", "bugged", "NPC"))


def test_provider_ok_rejected_for_wellbeing_checkin(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        ai_reply_enabled=True,
        draft_provider="gemini",
        training_messages_dir=tmp_path / "training_messages",
    )
    service._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        fake_provider_response('{"candidates":[{"reply":"ok","reason":"bad"}]}'),
        {"provider_configured": True},
    )

    bundle = service.build_bundle_with_context([], ["[OTHER]: good how are u"], contact_name="Catbot", relationship_type="close_friend")

    assert bundle.reply_candidates[0].text != "ok"
    bad = [candidate for candidate in bundle.reply_candidates if candidate.text == "ok"]
    assert bad
    assert bad[0].final_decision == "reject"
    assert bad[0].forbidden_move_violated is True


def test_reciprocal_wellbeing_answer_answers_back_not_how_come(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")

    bundle = service.build_bundle_with_context(
        [],
        ["[OTHER]: hey", "[ME]: yo how u been", "[OTHER]: i been good wby"],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    top = bundle.reply_candidates[0]

    assert top.scene_type == "reciprocal_wellbeing_question"
    assert top.required_reply_move == "answer_wellbeing_checkin"
    assert top.text != "how come"
    assert top.text != "ok"
    assert any(term in top.text for term in ("good", "calm", "bless", "not bad"))


def test_missed_reciprocal_wellbeing_question_gets_specific_repair(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")

    bundle = service.build_bundle_with_context(
        [],
        ["[OTHER]: hey", "[ME]: yo how u been", "[OTHER]: i been good wby", "[ME]: how come", "[OTHER]: i asked a question"],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    top = bundle.reply_candidates[0]

    assert top.scene_type == "missed_context_callout"
    assert top.required_reply_move == "answer_wellbeing_checkin"
    assert "answer previous wellbeing question" in top.unresolved_user_points
    assert top.text not in {"ok", "what u saying", "how come"}
    assert any(term in top.text for term in ("im good", "im calm", "im bless"))


def test_positive_mood_update_gets_personality_probe_not_dry_ack(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")

    bundle = service.build_bundle_with_context(
        [],
        ["[OTHER]: i asked a question", "[ME]: yh my bad im good icl", "[OTHER]: but i js been happy"],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    top = bundle.reply_candidates[0]

    assert top.scene_type == "positive_mood_update"
    assert top.required_reply_move == "acknowledge_positive_mood"
    assert top.text != "thats good to hear"
    assert any(term in top.text for term in ("as u should", "cute", "deserve", "happy"))


def test_wellbeing_answer_does_not_ask_back_when_already_asked_recently(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")

    bundle = service.build_bundle_with_context(
        [],
        [
            "[OTHER]: hey",
            "[ME]: yo how u been",
            "[OTHER]: i been good wby",
            "[ME]: im good wbu",
            "[OTHER]: but i js been happy",
            "[ME]: as u should why u happy",
            "[OTHER]: thanks hru",
        ],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    top = bundle.reply_candidates[0]

    assert top.scene_type == "reciprocal_wellbeing_question"
    assert top.required_reply_move == "answer_wellbeing_checkin"
    assert "wbu" not in top.text
    assert "wby" not in top.text
    assert any(term in top.text for term in ("im good", "im calm", "im bless"))


def test_question_debt_latest_reciprocal_question_answers_not_ok(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")

    bundle = service.build_bundle_with_context(
        [],
        ["[OTHER]: hi", "[ME]: yo what u on", "[OTHER]: nth wby"],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    top = bundle.reply_candidates[0]

    assert top.has_unanswered_user_question is True
    assert top.unanswered_question_type == "reciprocal_activity_question"
    assert top.answer_required_now is True
    assert top.question_debt_answer_score >= 0.7
    assert top.text.casefold().strip(" .?!") != "ok"


def test_question_debt_callout_answers_original_wby(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")

    bundle = service.build_bundle_with_context(
        [],
        ["[OTHER]: nth wby", "[ME]: ok", "[OTHER]: i js asked u a question"],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    top = bundle.reply_candidates[0]

    assert top.user_called_out_unanswered_question is True
    assert top.answer_required_now is True
    assert top.unanswered_question_type == "reciprocal_activity_question"
    assert "my bad" in top.text.casefold() or "missed" in top.text.casefold()
    assert any(term in top.text.casefold() for term in ("chilling", "nothing", "not much", "working"))
    assert top.text.casefold().strip(" .?!") != "nah i get u"


def test_question_debt_strong_callout_blocks_new_question(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")

    bundle = service.build_bundle_with_context(
        [],
        ["[OTHER]: nth wby", "[ME]: ok", "[OTHER]: are u gonna answer"],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    top = bundle.reply_candidates[0]

    assert top.answer_required_now is True
    assert top.question_debt_answer_score >= 0.7
    assert "say less what u doing" not in top.text.casefold()


def test_provider_new_question_rejected_while_question_debt_exists(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        training_messages_dir=tmp_path / "training_messages",
        ai_reply_enabled=True,
        draft_provider="gemini",
    )
    service._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        fake_provider_response('{"candidates":[{"reply":"what u doing","reason":"bad"}]}'),
        {"provider_configured": True},
    )

    bundle = service.build_bundle_with_context(
        [],
        ["[OTHER]: nth wby", "[ME]: ok", "[OTHER]: are u gonna answer"],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    bad = [candidate for candidate in bundle.reply_candidates if candidate.text == "what u doing"]

    assert bad
    assert bad[0].asked_new_question_before_answering_penalty >= 0.7
    assert bad[0].obligation_satisfied is False
    assert "new_question_before_answering_debt" in bad[0].quality_gate_reasons
    assert bad[0].final_decision == "reject"
    assert bundle.reply_candidates[0].question_debt_answer_score >= 0.7


def test_provider_unavailable_uses_contract_fallback_for_question_debt(tmp_path: Path) -> None:
    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        training_messages_dir=tmp_path / "training_messages",
        ai_reply_enabled=True,
        draft_provider="gemini",
    )
    service._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        fake_provider_response(
            '{"candidates":[{"reply":"what u doing","reason":"deterministic fallback"}]}',
            provider="fallback",
        ),
        {
            "provider_configured": False,
            "manual_review_fallback": True,
            "fallback_errors": [{"provider": "gemini", "model": "fake-model", "error": "provider_error"}],
        },
    )

    bundle = service.build_bundle_with_context(
        [],
        ["[OTHER]: nth wby", "[ME]: ok", "[OTHER]: are u gonna answer"],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    top = bundle.reply_candidates[0]

    assert bundle.provider_metadata["generation_status"] == "contract_fallback_after_provider_failure"
    assert bundle.provider_metadata["fallback_source"] == "conversation_contract"
    assert top.generated_by_provider is False
    assert top.obligation_satisfied is True
    assert top.question_debt_answer_score >= 0.7
    assert "what u doing" not in top.text.casefold()


def test_question_debt_identity_callout_uses_identity_pack(tmp_path: Path) -> None:
    service = DraftingService(template_path=tmp_path / "missing.json", approved_photos_dir=tmp_path / "approved", training_messages_dir=tmp_path / "training_messages")

    bundle = service.build_bundle_with_context(
        [],
        ["[OTHER]: how old r u", "[ME]: what u saying", "[OTHER]: are u gonna answer"],
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    top = bundle.reply_candidates[0]

    assert top.unanswered_question_type == "direct_identity_question"
    assert top.answer_required_now is True
    assert "19" in top.text
    assert "what u saying" not in top.text.casefold()
