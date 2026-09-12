from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import HTTPException

from apps.controller.models import (
    CatbotChatRequest,
    CatbotFeedbackRequest,
    StyleReviewApproveRequest,
    StyleReviewRejectRequest,
    TrainingChatRequest,
    TrainingParameters,
    TrainingFeedbackRequest,
    TrainingQuarantineRequest,
    WhatsAppWebDraftRequest,
    WhatsAppWebMemoryRequest,
    WhatsAppWebMessage,
)
from apps.controller.service import PhoneCopilotService
from apps.controller.settings import ControllerSettings
from libs.drafting.catbot_plan_validation import classify_reply_shape, detect_reset_question, detect_sensory_texture
from libs.drafting.conversation_function import predict_conversation_function
from libs.drafting.intents import classify_intent
from libs.drafting.retrieval_backends.lexical import LexicalRetrievalBackend
from libs.drafting.training_data import ensure_training_message_files, load_corrections


def _isolated_service(tmp_path: Path, mock_adb) -> PhoneCopilotService:
    training_messages = tmp_path / "training_messages"
    ensure_training_message_files(training_messages)
    (training_messages / "close_friends.jsonl").write_text(
        json.dumps(
            {
                "id": "real_plan",
                "relationship_type": "close_friend",
                "intent_type": "planning",
                "incoming": "you coming later?",
                "context": [],
                "my_reply": "yh what time",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    settings = ControllerSettings(
        enable_ocr=False,
        ai_reply_enabled=False,
        data_dir=tmp_path,
        log_db_path=tmp_path / "phone_copilot.db",
        screenshot_dir=tmp_path / "screenshots",
        debug_dir=tmp_path / "debug",
        fixtures_live_dir=tmp_path / "fixtures",
        approved_photos_dir=tmp_path / "approved_photos",
        reply_template_path=Path("data/templates/reply_templates.json"),
        selector_path=Path("data/selectors/screen_signatures.json"),
        ai_reply_training_messages_dir=training_messages,
        ai_reply_training_dir=tmp_path / "training",
        ai_reply_style_profile_path=tmp_path / "style_profile.json",
        contact_overrides_path=tmp_path / "contact_overrides.json",
        ai_reply_intelligence_db_path=tmp_path / "conversation_intelligence.db",
    )
    return PhoneCopilotService(settings=settings, adb_client=mock_adb)


@pytest.fixture()
def training_service(tmp_path: Path, mock_adb) -> PhoneCopilotService:
    return _isolated_service(tmp_path, mock_adb)


@pytest.fixture()
def provider_path(training_service: PhoneCopilotService, monkeypatch) -> None:
    # Exercise provider handling when no direct local reply is available.
    monkeypatch.setattr(training_service, "_catbot_direct_plan_reply", lambda *args, **kwargs: "")


@pytest.fixture()
def provider_retry_path(training_service: PhoneCopilotService, monkeypatch, provider_path) -> None:
    # Exhaust local repairs so the route must validate a provider retry.
    monkeypatch.setattr(training_service, "_catbot_plan_specific_repair_reply", lambda *args, **kwargs: "")


@pytest.mark.parametrize("local_reply", ["", "okay"])
@pytest.mark.parametrize("provider_failed", [False, True])
def test_catbot_provider_retry_recovers_after_unusable_plan(
    training_service: PhoneCopilotService, monkeypatch, provider_path, local_reply, provider_failed,
) -> None:
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        failed = provider_failed and len(calls) == 1
        return (
            type("FakeResponse", (), {
                "text": "okay" if len(calls) == 1 else "missed u too / been thinking about u all day",
                "provider": "fallback" if failed else "gemini",
                "model": "fake-model",
                "latency_ms": 1,
                "error": "temporary outage" if failed else None,
                "external_api_used": not failed,
            })(),
            {"provider_configured": True, "manual_review_fallback": failed},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", generate)
    monkeypatch.setattr(training_service, "_catbot_plan_specific_repair_reply", lambda *args, **kwargs: local_reply)
    response = training_service.training_catbot_chat(
        CatbotChatRequest(incoming="i missed u", relationship_type="romantic_interest")
    )
    candidate = response["candidate"]
    assert len(calls) == 2
    assert response["reply_sequence"] == ["missed u too", "been thinking about u all day"]
    assert candidate["provider"] == "gemini"
    assert candidate["provider_error"] is None
    assert candidate["fallback_used"] is False
    assert candidate["manual_review_fallback"] is False
    assert candidate["catbot_ai_plan_repair_attempted"] is True
    assert candidate["catbot_ai_plan_repair_accepted"] is False
    assert candidate["catbot_ai_retry_accepted"] is True
    assert candidate["catbot_ai_reject_reason"] == ""


@pytest.mark.parametrize(("incoming", "instruction"), [
    ("do u miss me", "They are asking if you miss them"),
    ("thinking of u", "They said they are thinking about you"),
    ("i missed u", "They said they miss you"),
])
def test_catbot_affection_prompt_uses_classified_slots(training_service, incoming, instruction) -> None:
    move = training_service._catbot_conversation_move(incoming=incoming, context=[])
    assert move["user_move"] == "emotional_reciprocity"
    assert instruction in training_service._catbot_conversation_move_instruction(move)


def test_training_page_route_loads() -> None:
    html = Path("apps/desktop_ui/training.html").read_text(encoding="utf-8")
    simple_html = Path("apps/desktop_ui/training_simple.html").read_text(encoding="utf-8")
    core_html = Path("apps/desktop_ui/ai_core.html").read_text(encoding="utf-8")
    core_js = Path("apps/desktop_ui/ai_core.js").read_text(encoding="utf-8")
    catbot_html = Path("apps/desktop_ui/catbot.html").read_text(encoding="utf-8")
    catbot_js = Path("apps/desktop_ui/catbot.js").read_text(encoding="utf-8")

    assert "Chat Simulator" in html
    assert "catbotMessages" in catbot_html
    assert "catbotInput" in catbot_html
    assert "/api/training/catbot/chat" in catbot_js
    assert "/api/training/catbot/feedback" in catbot_js
    assert 'catbotState.messages.push({ role: "catbot", text })' in catbot_js
    assert 'role: "me"' in catbot_js
    assert "Autonomous Training Loop" in html
    assert "Training Control" in html
    assert "Style Review Queue" in html
    assert "/training.js" in html
    assert "Advanced Training" in html
    assert "Simple Training" in simple_html
    assert "/training_simple.js" in simple_html
    assert "AI Core" in core_html
    assert "/ai_core.js" in core_html
    assert "/ai-core/vector-map" in core_html
    assert "identityData" in core_html
    assert "/api/ai-core/reload-identity-pack" in core_js
    extension_manifest = Path("apps/browser_extension/whatsapp_web/manifest.json").read_text(encoding="utf-8")
    extension_content = Path("apps/browser_extension/whatsapp_web/content.js").read_text(encoding="utf-8")
    extension_background = Path("apps/browser_extension/whatsapp_web/background.js").read_text(encoding="utf-8")
    assert "Phone Copilot WhatsApp Web" in extension_manifest
    assert '"version": "0.1.1"' in extension_manifest
    assert '"debugger"' in extension_manifest
    assert "page_debug_bridge.js" in extension_manifest
    assert '"world": "MAIN"' in extension_manifest
    assert "web_accessible_resources" in extension_manifest
    assert 'option value="auto-review"' in extension_content
    assert 'option value="auto-send"' in extension_content
    assert 'option value="autonomous"' in extension_content
    assert "Collect thread" in extension_content
    assert "Scan archive" in extension_content
    assert "Reload WA" in extension_content
    assert "pc:save-memory" in extension_content
    assert "mode: settings.mode" in extension_background
    assert "extension context invalidated" in extension_content
    assert "extension runtime is unavailable" in extension_content
    assert "window.location.reload()" in extension_content
    assert "runtimeUnavailable" in extension_content
    assert "sendRuntimePort" in extension_content
    assert "phone-copilot-long-request" in extension_content
    assert "LONG_REQUEST_KEEPALIVE_MS" in extension_content
    assert "pc:ping" in extension_content
    assert "chrome.runtime.connect" in extension_content
    assert "phone-copilot-long-request" in extension_background
    assert "chrome.runtime.onConnect.addListener" in extension_background
    assert "handleRuntimeMessage" in extension_background
    assert "Refresh WhatsApp Web so archive scan can use the long-lived controller channel" in extension_background
    assert "Input.dispatchMouseEvent" in extension_background
    assert "pc:trusted-click" in extension_background
    assert "pc:ping" in extension_background
    assert "debugger_api_unavailable" in extension_background
    assert "newRequestId" in extension_content
    assert "activeDraftRequestId" in extension_content
    assert "draftInFlight" in extension_content
    assert "draft already running" in extension_content
    assert "ignored stale draft response" in extension_content
    assert "request_id: payload.requestId" in extension_background
    assert "thread_id: payload.threadId" in extension_background
    assert "clearComposerIfPreviousDraft" in extension_content
    assert 'setReply("", [])' in extension_content
    assert "refreshVisibleThreadCache" in extension_content
    assert "unhandledrejection" in extension_content
    assert "timestampMillis" in extension_content
    assert "latest visible message is yours" in extension_content
    assert 'closest(".message-in, .message-out")' in extension_content
    assert 'querySelector?.("[data-pre-plain-text]")' in extension_content
    assert "client_order" in extension_content
    assert "avoidCandidates: previousDraftSequence" in extension_content
    assert "avoid_candidates: payload.avoidCandidates || []" in extension_background
    assert "initialVisibleMessages" in extension_content
    assert "Math.max(previous._pcOrder, order)" in extension_content

    vector_html = Path("apps/desktop_ui/vector_map.html").read_text(encoding="utf-8")
    vector_js = Path("apps/desktop_ui/vector_map.js").read_text(encoding="utf-8")
    assert "Vector Universe" in vector_html
    assert "/vector_map.js" in vector_html
    assert "getContext(\"2d\")" in vector_js
    assert "function rotate" in vector_js
    assert "colorMode" in vector_html
    assert "memoryTypeFilter" in vector_html
    assert "memory_type=" in vector_js
    assert "nearestNeighbours" in vector_js


def test_ai_core_status_lists_files_and_vector_db(training_service: PhoneCopilotService) -> None:
    status = training_service.ai_core_status()

    assert status["files"]["training_messages_dir"].endswith("training_messages")
    assert status["training_data"]["active_rows"] >= 1
    assert "vector_db" in status["files"]
    assert "thread_vector_count" in status["vector_database"]
    assert status["identity"]["identity_pack_loaded"] is True
    assert "safe_identity_summary" in status["identity"]
    assert "exact address" not in status["identity"]["safe_identity_summary"]
    assert status["safety"]["training_page_sends_phone_messages"] is False


def test_ai_core_vector_map_uses_safe_fallback_index(training_service: PhoneCopilotService) -> None:
    vector_dir = training_service.settings.data_dir / "vector_db" / "reply_examples_chroma"
    vector_dir.mkdir(parents=True, exist_ok=True)
    (vector_dir / "reply_examples.jsonl").write_text(
        json.dumps(
            {
                "id": "vm_1",
                "relationship_type": "close_friend",
                "intent_type": "greeting",
                "incoming": "hii",
                "my_reply": "yo",
                "_source": "correction",
                "is_correction": True,
                "is_synthetic": False,
                "unsafe": False,
                "is_quarantined": False,
            }
        )
        + "\n"
        + json.dumps(
            {
                "id": "vm_bad",
                "relationship_type": "close_friend",
                "intent_type": "greeting",
                "incoming": "bad",
                "my_reply": "bad",
                "unsafe": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = training_service.ai_core_vector_map(limit=50)

    assert payload["status"] == "ok"
    assert payload["source"] == "fallback_jsonl"
    assert payload["count"] == 1
    assert payload["points"][0]["id"] == "vm_1"
    assert payload["points"][0]["point_type"] == "correction"
    assert payload["points"][0]["cluster_key"] == "greeting|close_friend"
    assert {"x", "y", "z"} <= set(payload["points"][0])
    assert payload["clusters"][0]["key"] == "greeting|close_friend"


def test_ai_core_vector_map_filters_thread_episode_memory(training_service: PhoneCopilotService) -> None:
    vector_dir = training_service.settings.data_dir / "vector_db" / "reply_examples_chroma"
    vector_dir.mkdir(parents=True, exist_ok=True)
    (vector_dir / "reply_examples.jsonl").write_text(
        json.dumps(
            {
                "id": "reply_example_1",
                "memory_type": "reply_example",
                "relationship_type": "close_friend",
                "intent_type": "greeting",
                "incoming": "hi",
                "my_reply": "yo",
            }
        )
        + "\n"
        + json.dumps(
            {
                "id": "thread_episode_1",
                "memory_type": "thread_episode",
                "thread_id": "thread_1",
                "relationship_type": "close_friend",
                "intent_type": "repair_after_bot_error",
                "summary": "Bot repeated stale self-state and user called it out.",
                "incoming": "mate ur repeating",
                "my_reply": "yh fairs i repeated myself icl",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = training_service.ai_core_vector_map(limit=50, memory_type="thread_episode")
    stats = training_service.ai_core_status()

    assert payload["memory_type"] == "thread_episode"
    assert payload["count"] == 1
    assert payload["points"][0]["memory_type"] == "thread_episode"
    assert payload["points"][0]["summary"] == "Bot repeated stale self-state and user called it out."
    assert stats["vector_database"]["thread_vector_count"] == 1


def test_training_chat_hii_is_safe_and_offline(training_service: PhoneCopilotService, mock_adb) -> None:
    response = training_service.training_chat(
        TrainingChatRequest(incoming="hii", relationship_type="unknown", intent_type="auto")
    )

    replies = [candidate["text"] for candidate in response["candidates"]]
    assert response["intent_type"] == "greeting"
    assert response["relationship_type"] == "unknown"
    assert {"yo what u saying", "yo how u been"} & set(replies)
    assert all(reply in {"yo what u saying", "heyy what u doing", "yo how u been", "heyy u good", "yo what u on"} for reply in replies)
    assert response["selected_candidate"] in replies
    assert all(not any(term in reply.lower() for term in ("baby", "sexy", "trouble", "love", "xx")) for reply in replies)
    assert mock_adb.commands == []


def test_training_chat_exam_supervision_reply_is_specific(training_service: PhoneCopilotService) -> None:
    response = training_service.training_chat(
        TrainingChatRequest(
            incoming="So I have to be supervised",
            context=[
                "exam tmo",
                "how u feeling",
                "Exhausted",
                "tired",
                "I want these exams to fuck off",
                "I can't sleep peacefully without dreaming ab revision",
                "lmk how it goes today",
                "I will",
                "Im on a clash",
            ],
            contact_name="Sonia",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    reply = response["selected_candidate"].lower()
    assert response["intent_type"] == "exam_logistics"
    assert reply != "nah i get u"
    assert any(term in reply for term in ("supervised", "stuck", "clash", "exam", "paper", "long"))


def test_training_chat_risk_tolerance_prefers_bolder_review_candidate(training_service: PhoneCopilotService) -> None:
    response = training_service.training_chat(
        TrainingChatRequest(
            incoming="So I have to be supervised",
            context=[
                "I can't sleep peacefully without dreaming ab revision",
                "lmk how it goes today",
                "I will",
                "Im on a clash",
            ],
            contact_name="Sonia",
            relationship_type="romantic_interest",
            intent_type="auto",
            parameters=TrainingParameters(risk_tolerance=0.25),
        )
    )

    reply = response["selected_candidate"].lower()
    assert response["intent_type"] == "exam_logistics"
    assert reply.startswith("ugh")
    assert "supervised" in reply
    assert response["candidates"][0]["auto_send_allowed"] is False
    assert response["candidates"][0]["final_decision"] == "review"


def test_whatsapp_web_draft_uses_last_other_message_and_context(training_service: PhoneCopilotService, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, TrainingChatRequest] = {}

    def fake_training_chat(request: TrainingChatRequest) -> dict[str, object]:
        seen["request"] = request
        return {
            "selected_candidate": "later tonight",
            "candidates": [
                {
                    "text": "later tonight",
                    "sequence": ["later tonight"],
                    "auto_send_allowed": False,
                }
            ],
            "intent_type": "planning",
        }

    monkeypatch.setattr(training_service, "training_chat", fake_training_chat)

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="Ali",
            mode="auto-review",
            relationship_type="close_friend",
            messages=[
                WhatsAppWebMessage(speaker="other", text="you coming later?"),
                WhatsAppWebMessage(speaker="me", text="maybe"),
                WhatsAppWebMessage(speaker="other", text="what time"),
            ],
        )
    )

    assert seen["request"].incoming == "what time"
    assert seen["request"].context == ["you coming later?", "maybe"]
    assert seen["request"].contact_name == "Ali"
    assert seen["request"].relationship_type == "close_friend"
    assert result["automation_decision"] == "SAFE_TO_DRAFT"
    assert result["auto_send_allowed"] is False
    assert result["reply"] == "later tonight"


def test_whatsapp_web_auto_send_uses_selected_candidate_gate(training_service: PhoneCopilotService, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_training_chat(request: TrainingChatRequest) -> dict[str, object]:
        return {
            "selected_candidate": "safe selected reply",
            "candidates": [
                {
                    "text": "blocked first reply",
                    "sequence": ["blocked first reply"],
                    "auto_send_allowed": False,
                },
                {
                    "text": "safe selected reply",
                    "sequence": ["safe selected reply"],
                    "auto_send_allowed": True,
                },
            ],
            "intent_type": "planning",
        }

    monkeypatch.setattr(training_service, "training_chat", fake_training_chat)

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            mode="auto-send",
            relationship_type="close_friend",
            messages=[WhatsAppWebMessage(speaker="other", text="you coming later?")],
        )
    )

    assert result["reply"] == "safe selected reply"
    assert result["candidate"]["text"] == "safe selected reply"
    assert result["automation_decision"] == "SEND_ALLOWED"
    assert result["auto_send_allowed"] is True


def test_whatsapp_web_natural_draft_uses_full_context_provider(training_service: PhoneCopilotService) -> None:
    training_service.drafting.ai_reply_enabled = True
    training_service.drafting.draft_provider = "gemini"
    calls: list[dict[str, object]] = []

    def fake_generation(**kwargs):
        calls.append(kwargs)
        return (
            type("FakeResponse", (), {
                "text": json.dumps({
                    "candidates": [
                        {
                            "reply": "i hear u\nwrong chat was my fault\ni get why that made u feel fake\nthats on me\ni care about u\nnot trying to dismiss u\nallow me\ntalk to me",
                            "reason": "full-context WhatsApp provider draft",
                        }
                    ]
                }),
                "provider": "gemini",
                "model": "fake",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    training_service.drafting._run_provider_generation = fake_generation  # type: ignore[method-assign]

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="+44 7700 900123",
            relationship_type="romantic_interest",
            mode="autonomous",
            messages=[
                WhatsAppWebMessage(speaker="other", text="goodnight"),
                WhatsAppWebMessage(speaker="me", text="goodnight"),
                WhatsAppWebMessage(speaker="other", text="what the hell"),
                WhatsAppWebMessage(speaker="me", text="wrong chat my bad"),
                WhatsAppWebMessage(speaker="other", text="u hate me"),
                WhatsAppWebMessage(speaker="other", text="ur fake"),
                WhatsAppWebMessage(speaker="me", text="how"),
                WhatsAppWebMessage(speaker="me", text="wtf"),
                WhatsAppWebMessage(speaker="other", text="ur fake"),
            ],
        )
    )

    assert calls
    assert "fresh_model" in result["candidate_sources"]
    assert result["selected_candidate_source_label"] == "fresh_model"
    assert result["latest_incoming_burst"] == ["ur fake"]


def test_whatsapp_web_fresh_model_single_bubble_can_pass_for_low_intensity_burst(training_service: PhoneCopilotService) -> None:
    training_service.drafting.ai_reply_enabled = True
    training_service.drafting.draft_provider = "gemini"

    def fake_generation(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": json.dumps({
                    "candidates": [
                        {
                            "reply": "hey squishy, u good?",
                            "reason": "fresh model greeting",
                        }
                    ]
                }),
                "provider": "gemini",
                "model": "fake",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    training_service.drafting._run_provider_generation = fake_generation  # type: ignore[method-assign]

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="DemoContact",
            relationship_type="romantic_interest",
            mode="autonomous",
            messages=[
                WhatsAppWebMessage(speaker="me", text="talk to me"),
                WhatsAppWebMessage(speaker="other", text="Hey"),
                WhatsAppWebMessage(speaker="me", text="bruh"),
                WhatsAppWebMessage(speaker="other", text="Hey squishy"),
            ],
        )
    )

    assert result["selected_candidate_source_label"] == "fresh_model"
    assert result["validation_result"]["passed"] is True
    assert result["validation_result"]["burst_size_relaxed_for_fresh_model"] is True
    assert result["latest_incoming_burst"] == ["Hey squishy"]


def test_whatsapp_web_autonomous_allows_low_intensity_fresh_model_send(training_service: PhoneCopilotService) -> None:
    training_service.drafting.ai_reply_enabled = True
    training_service.drafting.draft_provider = "gemini"

    def fake_generation(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": json.dumps({
                    "candidates": [
                        {
                            "reply": "hey squishy what u been up to",
                            "reason": "fresh model greeting",
                        }
                    ]
                }),
                "provider": "gemini",
                "model": "fake",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    training_service.drafting._run_provider_generation = fake_generation  # type: ignore[method-assign]

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="DemoContact",
            relationship_type="romantic_interest",
            mode="autonomous",
            messages=[
                WhatsAppWebMessage(speaker="me", text="talk to me"),
                WhatsAppWebMessage(speaker="other", text="Hey"),
                WhatsAppWebMessage(speaker="me", text="bruh"),
                WhatsAppWebMessage(speaker="other", text="Hey squishy"),
            ],
        )
    )

    assert result["selected_candidate_source_label"] == "fresh_model"
    assert result["automation_decision"] == "SEND_ALLOWED"
    assert result["auto_send_allowed"] is True


def test_whatsapp_web_send_gate_allows_short_practical_fresh_model_burst(training_service: PhoneCopilotService) -> None:
    obligation = training_service._whatsapp_conversation_obligation(
        [
            {"speaker": "me", "text": "hey squishy what u been up to", "timestamp": ""},
            {"speaker": "other", "text": "Nothing squishy", "timestamp": ""},
            {"speaker": "other", "text": "When r u having ur bbq", "timestamp": ""},
        ],
        2,
        relationship="romantic_interest",
        already_sent_replies=[],
    )
    candidate = {
        "candidate_source_label": "fresh_model",
        "provider": "gemini",
        "text": "not sure yet squishy, still figuring out dates",
        "sequence": ["not sure yet squishy, still figuring out dates"],
    }
    validation = training_service._whatsapp_validate_reply_against_obligation(str(candidate["text"]), obligation)

    assert obligation.conversation_move == ["practical_question"]
    assert obligation.target_reply_burst_size == {"min": 1, "max": 3}
    assert validation["passed"] is True
    assert training_service._whatsapp_fresh_model_low_intensity_send_allowed(candidate, validation, obligation) is True


def test_whatsapp_web_auto_send_blocks_candidate_without_send_gate(training_service: PhoneCopilotService, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_training_chat(request: TrainingChatRequest) -> dict[str, object]:
        return {
            "selected_candidate": "yh ill check",
            "candidates": [
                {
                    "text": "yh ill check",
                    "sequence": ["yh ill check"],
                    "auto_send_allowed": False,
                }
            ],
            "intent_type": "planning",
        }

    monkeypatch.setattr(training_service, "training_chat", fake_training_chat)

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            mode="auto-send",
            relationship_type="close_friend",
            messages=[WhatsAppWebMessage(speaker="other", text="you coming later?")],
        )
    )

    assert result["automation_decision"] == "REVIEW_REQUIRED"
    assert result["auto_send_allowed"] is False
    assert result["auto_send_blocked_reason"] == "candidate_not_auto_send_safe"


def test_whatsapp_web_auto_send_blocks_unsafe_relationship(training_service: PhoneCopilotService, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_training_chat(request: TrainingChatRequest) -> dict[str, object]:
        return {
            "selected_candidate": "yh ill check",
            "candidates": [
                {
                    "text": "yh ill check",
                    "sequence": ["yh ill check"],
                    "auto_send_allowed": True,
                }
            ],
            "intent_type": "planning",
        }

    monkeypatch.setattr(training_service, "training_chat", fake_training_chat)

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            mode="auto-send",
            relationship_type="family",
            messages=[WhatsAppWebMessage(speaker="other", text="you coming later?")],
        )
    )

    assert result["automation_decision"] == "REVIEW_REQUIRED"
    assert result["auto_send_allowed"] is False
    assert result["auto_send_blocked_reason"] == "relationship_not_auto_send_safe"


def test_whatsapp_web_auto_send_does_not_target_already_replied_message(training_service: PhoneCopilotService, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_training_chat(request: TrainingChatRequest) -> dict[str, object]:
        raise AssertionError("auto mode should not draft when latest visible WhatsApp message is from me")

    monkeypatch.setattr(training_service, "training_chat", fail_training_chat)

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            mode="auto-send",
            relationship_type="close_friend",
            messages=[
                WhatsAppWebMessage(speaker="other", text="you coming later?"),
                WhatsAppWebMessage(speaker="me", text="yh what time"),
            ],
        )
    )

    assert result["reply"] == ""
    assert result["insert_allowed"] is False
    assert result["auto_send_allowed"] is False
    assert result["auto_send_blocked_reason"] == "latest_message_from_self"


def test_whatsapp_web_auto_review_can_add_to_latest_self_reply(training_service: PhoneCopilotService) -> None:
    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            mode="auto-review",
            relationship_type="romantic_interest",
            messages=[
                WhatsAppWebMessage(speaker="other", text="wanna go out thursday?"),
                WhatsAppWebMessage(speaker="me", text="yeah 100%"),
                WhatsAppWebMessage(speaker="me", text="i wanna see u"),
            ],
        )
    )

    sequence = [part.lower() for part in result["reply_sequence"]]
    assert result["incoming"] == "wanna go out thursday?"
    assert any("invitation/plan" in item for item in result["context"])
    assert "Already sent after latest incoming - yeah 100%" in result["context"]
    assert "Already sent after latest incoming - i wanna see u" in result["context"]
    assert result["reply"]
    assert "yeah 100%" not in sequence
    assert "i wanna see u" not in sequence
    assert result["insert_allowed"] is True
    assert result["automation_decision"] == "SAFE_TO_DRAFT"
    assert result["post_incoming_context_count"] == 2


def test_whatsapp_web_draft_uses_full_loaded_message_context(training_service: PhoneCopilotService, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, list[TrainingChatRequest]] = {"requests": []}

    def fake_training_chat(request: TrainingChatRequest) -> dict[str, object]:
        seen["requests"].append(request)
        return {
            "selected_candidate": "yh",
            "candidates": [{"text": "yh", "sequence": ["yh"], "auto_send_allowed": False}],
            "intent_type": "other",
        }

    monkeypatch.setattr(training_service, "training_chat", fake_training_chat)
    messages = [
        WhatsAppWebMessage(speaker="other" if index % 2 else "me", text=f"msg {index}")
        for index in range(130)
    ]
    messages.append(WhatsAppWebMessage(speaker="other", text="latest question"))

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="Abdullah",
            relationship_type="close_friend",
            messages=messages,
        )
    )

    first_request = seen["requests"][0]
    assert first_request.incoming == "msg 129\nlatest question"
    assert len(first_request.context) == 129
    assert first_request.context[0] == "msg 0"
    assert first_request.context[-1] == "msg 128"
    assert result["message_context_count"] == 129
    assert result["context_minimum_met"] is True


def test_whatsapp_web_memory_store_is_used_as_draft_context(training_service: PhoneCopilotService, monkeypatch: pytest.MonkeyPatch) -> None:
    store_result = training_service.whatsapp_web_memory_store(
        WhatsAppWebMemoryRequest(
            contact_name="Abdullah",
            relationship_type="close_friend",
            question="What is the plan?",
            answer="If the bus cancels, get a return or split a taxi.",
            messages=[WhatsAppWebMessage(speaker="other", text="if they cancel the bus otw back")],
        )
    )
    assert store_result["status"] == "saved"

    seen: dict[str, TrainingChatRequest] = {}

    def fake_training_chat(request: TrainingChatRequest) -> dict[str, object]:
        seen["request"] = request
        return {
            "selected_candidate": "yh the deal is sort the return first",
            "candidates": [
                {
                    "text": "yh the deal is sort the return first",
                    "sequence": ["yh the deal is sort the return first"],
                    "auto_send_allowed": False,
                }
            ],
            "intent_type": "planning",
        }

    monkeypatch.setattr(training_service, "training_chat", fake_training_chat)

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="Abdullah",
            relationship_type="close_friend",
            messages=[WhatsAppWebMessage(speaker="other", text="the deal?")],
        )
    )

    assert seen["request"].context[0] == "Saved WhatsApp note - What is the plan?: If the bus cancels, get a return or split a taxi."
    assert result["saved_memory_context_count"] == 1
    assert result["saved_memory"][0]["question"] == "What is the plan?"
    assert training_service.vector_rebuild_marker_path.exists()


def test_whatsapp_web_draft_infers_romantic_relationship_from_thread(training_service: PhoneCopilotService, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, TrainingChatRequest] = {}

    def fake_training_chat(request: TrainingChatRequest) -> dict[str, object]:
        seen["request"] = request
        return {
            "selected_candidate": "ugh thats long how long they keeping u supervised for",
            "candidates": [
                {
                    "text": "ugh thats long how long they keeping u supervised for",
                    "sequence": ["ugh thats long how long they keeping u supervised for"],
                    "auto_send_allowed": False,
                }
            ],
            "intent_type": "exam_logistics",
        }

    monkeypatch.setattr(training_service, "training_chat", fake_training_chat)

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="Sonia",
            relationship_type="unknown",
            messages=[
                WhatsAppWebMessage(speaker="me", text="i missed you"),
                WhatsAppWebMessage(speaker="other", text="U miss me don't youuu"),
                WhatsAppWebMessage(speaker="me", text="say no more baby"),
                WhatsAppWebMessage(speaker="other", text="I don't think we should do it raw"),
                WhatsAppWebMessage(speaker="other", text="So I have to be supervised"),
            ],
        )
    )

    assert result["relationship_type"] == "romantic_interest"
    assert result["relationship_inferred_from_thread"] is True
    assert seen["request"].relationship_type == "romantic_interest"


def test_whatsapp_web_romantic_invite_uses_thread_and_multi_bubble_reply(training_service: PhoneCopilotService) -> None:
    messages = [
        WhatsAppWebMessage(speaker="other", text="I miss you I love you", timestamp="1:22 p.m., 2026-05-22"),
        WhatsAppWebMessage(speaker="other", text="Goodnight I love you Sleepwell", timestamp="1:20 a.m., 2026-05-23"),
        WhatsAppWebMessage(speaker="other", text="Today's mums wedding day", timestamp="9:08 a.m., 2026-05-23"),
        WhatsAppWebMessage(speaker="other", text="I feel overstimulated asf", timestamp="10:26 a.m., 2026-05-23"),
        WhatsAppWebMessage(speaker="other", text="I feel so overwhelmed", timestamp="10:26 a.m., 2026-05-23"),
        WhatsAppWebMessage(speaker="other", text="Okay this is acc cute", timestamp="10:53 a.m., 2026-05-23"),
        WhatsAppWebMessage(speaker="other", text="Fatty", timestamp="2:33 p.m., 2026-05-23"),
        WhatsAppWebMessage(speaker="other", text="Look", timestamp="2:34 p.m., 2026-05-23"),
        WhatsAppWebMessage(speaker="other", text="Wait", timestamp="2:34 p.m., 2026-05-23"),
        WhatsAppWebMessage(speaker="other", text="Can u answer my ft", timestamp="2:34 p.m., 2026-05-23"),
        WhatsAppWebMessage(speaker="other", text="I wanna show u my dress", timestamp="2:34 p.m., 2026-05-23"),
        WhatsAppWebMessage(speaker="other", text="Call me I wanna show u my clothes", timestamp="2:41 p.m., 2026-05-23"),
        WhatsAppWebMessage(speaker="other", text="And how I'm dressed up", timestamp="2:41 p.m., 2026-05-23"),
        WhatsAppWebMessage(speaker="other", text="Tell me which one should I post", timestamp="8:56 p.m., 2026-05-23"),
        WhatsAppWebMessage(speaker="other", text="I have sm to rant to u", timestamp="9:16 p.m., 2026-05-23"),
        WhatsAppWebMessage(speaker="other", text="Tell me when ur free", timestamp="11:27 p.m., 2026-05-23"),
        WhatsAppWebMessage(speaker="other", text="I can't do a sat cause I tutor on Saturdays", timestamp="11:27 p.m., 2026-05-23"),
        WhatsAppWebMessage(speaker="other", text="Wanna go out thirsday?", timestamp="11:51 a.m., 2026-05-24"),
    ]

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="Tahlil",
            relationship_type="unknown",
            mode="auto-review",
            messages=messages,
            parameters=TrainingParameters(risk_tolerance=0.4),
        )
    )

    reply = result["reply"].lower()
    sequence = [part.lower() for part in result["reply_sequence"]]
    assert result["relationship_type"] == "romantic_interest"
    assert result["relationship_inferred_from_thread"] is True
    assert result["intent_type"] == "planning"
    assert result["time_context_count"] > 0
    assert result["message_context_count"] == 17
    assert len(sequence) >= 2
    assert "what u mean" not in reply
    assert "what do you mean" not in reply
    assert any("down" in part or "100%" in part or "ofc" in part for part in sequence)
    assert any("see u" in part or "thursday" in part for part in sequence)
    assert any("what time" in part or "free" in part for part in sequence)


def test_whatsapp_web_romantic_high_energy_greeting_matches_energy(training_service: PhoneCopilotService) -> None:
    assert classify_intent("hrella", []) == "greeting"

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="Tahlil",
            relationship_type="romantic_interest",
            mode="auto-review",
            messages=[WhatsAppWebMessage(speaker="other", text="hrella", timestamp="1:22 p.m., 2026-06-04")],
            parameters=TrainingParameters(risk_tolerance=0.4),
        )
    )

    sequence = [part.lower() for part in result["reply_sequence"]]
    assert result["intent_type"] == "greeting"
    assert result["reply"]
    assert any(part in {"hrellaaa", "heyyy", "hellooo"} for part in sequence)


def test_whatsapp_web_romantic_deep_context_rejects_generic_clarification(training_service: PhoneCopilotService) -> None:
    messages = [
        WhatsAppWebMessage(speaker="other", text="I miss you I love you", timestamp="1:22 p.m., 2026-06-04"),
        WhatsAppWebMessage(speaker="me", text="i miss u too", timestamp="1:23 p.m., 2026-06-04"),
        WhatsAppWebMessage(speaker="other", text="I wanna show u my dress", timestamp="1:24 p.m., 2026-06-04"),
        WhatsAppWebMessage(speaker="me", text="show me then", timestamp="1:25 p.m., 2026-06-04"),
        WhatsAppWebMessage(speaker="other", text="I have sm to rant to u", timestamp="1:26 p.m., 2026-06-04"),
        WhatsAppWebMessage(speaker="other", text="that?", timestamp="1:27 p.m., 2026-06-04"),
    ]

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="Tahlil",
            relationship_type="romantic_interest",
            mode="auto-review",
            messages=messages,
            parameters=TrainingParameters(risk_tolerance=0.4),
        )
    )

    reply = result["reply"].lower()
    assert result["reply"]
    assert "what do you mean" not in reply
    assert "what u mean" not in reply
    assert any(term in reply for term in ("say that again", "go on", "tell me", "show me", "talk to me"))


def test_whatsapp_web_romantic_goodnight_reciprocates_affection(training_service: PhoneCopilotService) -> None:
    messages = [
        WhatsAppWebMessage(speaker="other", text="I miss you I love you", timestamp="1:22 p.m., 2026-05-22"),
        WhatsAppWebMessage(speaker="other", text="Goodnight I love you Sleepwell", timestamp="1:20 a.m., 2026-05-23"),
        WhatsAppWebMessage(speaker="other", text="I wanna show u my dress", timestamp="2:34 p.m., 2026-05-23"),
        WhatsAppWebMessage(speaker="other", text="I have sm to rant to u", timestamp="9:16 p.m., 2026-05-23"),
        WhatsAppWebMessage(speaker="other", text="Love u", timestamp="9:15 p.m., 2026-06-02"),
        WhatsAppWebMessage(speaker="other", text="Goodnight", timestamp="9:15 p.m., 2026-06-02"),
        WhatsAppWebMessage(speaker="other", text="Mwah", timestamp="9:15 p.m., 2026-06-02"),
        WhatsAppWebMessage(speaker="other", text="I can't wait to kiss u again", timestamp="9:15 p.m., 2026-06-02"),
        WhatsAppWebMessage(speaker="other", text="Miss u love u", timestamp="11:56 p.m., 2026-06-03"),
        WhatsAppWebMessage(speaker="other", text="Goodnight my handsome", timestamp="11:56 p.m., 2026-06-03"),
    ]

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="Tahlil",
            relationship_type="unknown",
            mode="auto-review",
            messages=messages,
            parameters=TrainingParameters(risk_tolerance=0.4),
        )
    )

    reply = result["reply"].lower()
    sequence = [part.lower() for part in result["reply_sequence"]]
    assert result["relationship_type"] == "romantic_interest"
    assert result["relationship_inferred_from_thread"] is True
    assert result["intent_type"] == "romantic_flirty"
    assert len(sequence) >= 2
    assert "what do you mean" not in reply
    assert "what u mean" not in reply
    assert any(term in reply for term in ("love u", "miss u", "goodnight", "sleep well", "mwah"))
    assert any(term in reply for term in ("goodnight", "sleep well"))


@pytest.mark.parametrize(
    ("relationship_type", "message_texts"),
    [
        (
            "romantic_interest",
            ["miss u", "gn baby"],
        ),
        (
            "unknown",
            ["love u", "good night x"],
        ),
        (
            "unknown",
            ["miss you", "sleepwell my love"],
        ),
        (
            "romantic_interest",
            ["miss you", "gn"],
        ),
        (
            "romantic_interest",
            ["i love you", "night baby"],
        ),
        (
            "unknown",
            ["miss u love u", "goodnight"],
        ),
    ],
)
def test_whatsapp_web_romantic_signoff_variants_do_not_fall_to_clarification(
    training_service: PhoneCopilotService,
    relationship_type: str,
    message_texts: list[str],
) -> None:
    messages = [
        WhatsAppWebMessage(speaker="other", text=text, timestamp=f"11:{50 + index:02d} p.m., 2026-06-03")
        for index, text in enumerate(message_texts)
    ]

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="RomanticContact",
            relationship_type=relationship_type,
            mode="auto-review",
            messages=messages,
            parameters=TrainingParameters(risk_tolerance=0.4),
        )
    )

    reply = result["reply"].lower()
    assert result["relationship_type"] == "romantic_interest"
    assert result["reply_sequence"]
    assert "what do you mean" not in reply
    assert "what u mean" not in reply
    assert any(term in reply for term in ("goodnight", "sleep well"))
    assert any(term in reply for term in ("love u", "miss u", "baby", "my love", "mwah", "sleep well"))


@pytest.mark.parametrize(
    ("message_texts", "relationship_type", "expected_moves", "expected_acts", "reply_terms"),
    [
        (["Tell me which one should I post"], "romantic_interest", {"practical_question"}, {"ask_to_see_options_or_pick"}, ("send", "pick", "show")),
        (["I have sm to rant to u"], "romantic_interest", {"future_rant_teaser"}, {"invite_them_to_talk"}, ("tell me", "listening", "im here")),
        (["Tell me when ur free"], "romantic_interest", {"practical_question"}, {"answer_availability"}, ("free", "later", "tonight")),
        (["Are you busy rn??"], "romantic_interest", {"practical_question", "check_in"}, {"answer_availability"}, ("free", "busy", "tonight", "later")),
        (["I can't do a sat cause I tutor on Saturdays"], "romantic_interest", {"invite_or_plan"}, {"acknowledge_unavailable_and_suggest"}, ("sunday", "what about", "fine")),
        (["Wanna go out thirsday?"], "romantic_interest", {"invite_or_plan"}, {"answer_invite_directly"}, ("down", "what time", "yeah")),
        (["Eid Mubarak"], "romantic_interest", {"celebration_or_holiday"}, {"return_holiday_greeting"}, ("eid mubarak", "mubarak")),
        (["Love u goodnight"], "romantic_interest", {"romantic_signoff", "affection", "goodnight"}, {"reciprocate_affection", "return_goodnight"}, ("love u", "goodnight", "sleep well")),
        (["U should put ice cream on me and lick it off"], "romantic_interest", {"sexual_flirt"}, {"respond_to_sexual_flirt_safely"}, ("trouble", "behave", "ice cream")),
        (["Well I have a lot on my mind", "And I'm gonna tell u later"], "romantic_interest", {"future_rant_teaser"}, {"invite_them_to_talk"}, ("ready", "tell me", "im here")),
        (
            [
                "My parent is unwell and has a hospital appointment today",
                "the uncertainty is making me scared",
                "im scared and could use someone to talk to",
                "there is a lot happening and i feel overwhelmed",
            ],
            "romantic_interest",
            {"emotional_disclosure"},
            {"provide_emotional_support"},
            ("sorry", "scared", "talk to me", "here"),
        ),
        (["R u back?", "If u r I'm free on Sunday"], "romantic_interest", {"check_in", "invite_or_plan"}, {"answer_back_status_and_availability"}, ("back", "sunday", "works")),
        (["Love u miss u gn"], "romantic_interest", {"romantic_signoff", "missing_you", "affection", "goodnight"}, {"reciprocate_affection", "reciprocate_missing", "return_goodnight"}, ("love u", "miss u", "goodnight")),
        (["I miss u where r u"], "romantic_interest", {"missing_you", "practical_question"}, {"reciprocate_missing", "answer_location_or_status"}, ("miss u", "northbridge", "rn")),
        (["Bro yesterday", "I was watching an Indian movie", "there was a character named alex", "mum and morgan wouldn't leave me alone"], "romantic_interest", {"story_share"}, {"react_to_story_content"}, ("never letting", "alex", "nah")),
        (["hi baby"], "romantic_interest", {"greeting"}, {"answer_greeting"}, ("hi", "baby")),
        (["Hey fatty"], "romantic_interest", {"teasing", "greeting"}, {"answer_teasing_playfully"}, ("rude", "hey you", "wow")),
        (["I love you", "Goodnight"], "romantic_interest", {"romantic_signoff", "affection", "goodnight"}, {"reciprocate_affection", "return_goodnight"}, ("love", "goodnight", "sleep well")),
        (["Ur shoe size do u lean towards 7.5 more or 7", "Wich ones more ideal"], "romantic_interest", {"practical_question"}, {"answer_practical_question"}, ("7.5", "safer", "7")),
        (["Love u", "Goodnight", "Mwah", "I can't wait to kiss u again"], "romantic_interest", {"romantic_signoff", "affection", "goodnight"}, {"reciprocate_affection", "return_goodnight", "reciprocate_kiss_affection"}, ("love u", "mwah", "cant wait", "can't wait")),
        (["Miss u love u", "Goodnight my handsome"], "unknown", {"romantic_signoff", "missing_you", "affection", "goodnight", "compliment"}, {"reciprocate_affection", "reciprocate_missing", "return_goodnight"}, ("love u", "miss u", "goodnight")),
        (["Sorry baby I didn't mean it"], "romantic_interest", {"apology_or_repair"}, {"acknowledge_apology_or_repair"}, ("okay", "dw", "baby")),
    ],
)
def test_whatsapp_web_conversation_obligation_layer_covers_mixed_timeline(
    training_service: PhoneCopilotService,
    message_texts: list[str],
    relationship_type: str,
    expected_moves: set[str],
    expected_acts: set[str],
    reply_terms: tuple[str, ...],
) -> None:
    messages = [
        WhatsAppWebMessage(speaker="other", text=text, timestamp=f"11:{40 + index:02d} p.m., 2026-06-03")
        for index, text in enumerate(message_texts)
    ]

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="TimelineContact",
            relationship_type=relationship_type,
            mode="auto-review",
            messages=messages,
            parameters=TrainingParameters(risk_tolerance=0.4),
        )
    )

    reply = result["reply"].lower()
    obligation = result["conversation_obligation"]
    validation = result["validation_result"]
    assert expected_moves <= set(result["conversation_move"])
    assert expected_acts <= set(result["required_response_acts"])
    assert obligation["latest_incoming_burst"] == message_texts
    assert validation["passed"] is True
    assert "what do you mean" not in reply
    assert "what u mean" not in reply
    assert any(term in reply for term in reply_terms)

    bad_validation = training_service._whatsapp_validate_reply_against_obligation(
        "what do you mean",
        training_service._whatsapp_conversation_obligation(
            [{"speaker": "other", "text": text, "timestamp": f"11:{40 + index:02d} p.m., 2026-06-03"} for index, text in enumerate(message_texts)],
            len(message_texts) - 1,
            relationship=result["relationship_type"],
            already_sent_replies=[],
        ),
    )
    if set(result["conversation_move"]) != {"normal"}:
        assert bad_validation["passed"] is False


def test_whatsapp_web_when_are_you_having_event_is_practical_question(
    training_service: PhoneCopilotService,
) -> None:
    messages = [
        {"speaker": "other", "text": "Nothing squishy", "timestamp": "6:01 p.m., 2026-06-06"},
        {"speaker": "other", "text": "When r u having ur bbq", "timestamp": "6:01 p.m., 2026-06-06"},
    ]

    obligation = training_service._whatsapp_conversation_obligation(
        messages,
        1,
        relationship="romantic_interest",
        already_sent_replies=[],
    )
    validation = training_service._whatsapp_validate_reply_against_obligation(
        "not sure yet maybe sunday",
        obligation,
    )

    assert "practical_question" in obligation.conversation_move
    assert "answer_availability" in obligation.required_response_acts
    assert obligation.target_reply_burst_size == {"min": 1, "max": 3}
    assert validation["passed"] is True
    assert "ignore_direct_question" not in validation["violated_forbidden_acts"]


def test_whatsapp_web_conflict_burst_does_not_become_practical_question_from_old_when(
    training_service: PhoneCopilotService,
) -> None:
    messages = [
        {"speaker": "other", "text": "when are you going to care", "timestamp": "5:31 p.m., 2026-06-06"},
        {"speaker": "other", "text": "Fuck you wallahi", "timestamp": "5:32 p.m., 2026-06-06"},
        {"speaker": "other", "text": "I don't want any part of you", "timestamp": "5:33 p.m., 2026-06-06"},
        {"speaker": "other", "text": "I'm done", "timestamp": "5:33 p.m., 2026-06-06"},
        {"speaker": "other", "text": "Gfys", "timestamp": "5:33 p.m., 2026-06-06"},
    ]

    obligation = training_service._whatsapp_conversation_obligation(
        messages,
        4,
        relationship="romantic_interest",
        already_sent_replies=[],
    )

    assert "conflict_or_hurt" in obligation.conversation_move
    assert "practical_question" not in obligation.conversation_move
    assert "relationship_conflict" in set(obligation.concrete_topics_detected)
    assert "answer_availability" not in obligation.required_response_acts
    assert "acknowledge_relationship_hurt" in obligation.required_response_acts


def test_whatsapp_web_large_relationship_conflict_accepts_review_sized_repair(
    training_service: PhoneCopilotService,
) -> None:
    base_messages = [
        "you reply to everyone else but not me",
        "I don't feel appreciated",
        "I don't want any part of you",
        "I'm done",
    ]
    messages = [
        {"speaker": "other", "text": base_messages[index % len(base_messages)], "timestamp": ""}
        for index in range(36)
    ]
    messages.append({"speaker": "other", "text": "sorry atp", "timestamp": ""})

    obligation = training_service._whatsapp_conversation_obligation(
        messages,
        len(messages) - 1,
        relationship="romantic_interest",
        already_sent_replies=[],
    )
    repair = (
        "baby im sorry / i hear how much ive hurt u / i havent been there the way u needed / "
        "thats on me / i do love u wallahi / i know the space has felt one sided / "
        "im not trying to make everything on my terms / you deserved more from me / "
        "i get why that made u feel unappreciated / im sorry for making u cry like that / "
        "i care about u so much / i dont want u thinking youre never loved for u / "
        "i need to talk to u properly / i dont want to lose us"
    )
    validation = training_service._whatsapp_validate_reply_against_obligation(repair, obligation)

    assert obligation.target_reply_burst_size["min"] <= 12
    assert obligation.target_reply_burst_size["max"] <= 18
    assert validation["passed"] is True
    assert "below_target_reply_burst_size" not in validation["violated_forbidden_acts"]


def test_whatsapp_web_ignored_hurt_burst_gets_capped_relationship_obligation(
    training_service: PhoneCopilotService,
) -> None:
    message_texts = [
        "I hate u genuinely",
        "I’d never ever do this to you wallahi ever",
        "The way u do shit like this to me",
        "Is diabolical",
        "Why are you doing this to me",
        "Ur not saying fuck all",
        "Ur hurting me so bad man",
        "how many times do I have to fucking cry man",
        "alex",
        "What r is wrong with you",
        "Stop ignoring me man",
    ]
    messages = [
        {"speaker": "other", "text": text, "timestamp": datetime.now(timezone.utc).isoformat()}
        for index, text in enumerate(message_texts)
    ]

    obligation = training_service._whatsapp_conversation_obligation(
        messages,
        len(messages) - 1,
        relationship="romantic_interest",
        already_sent_replies=[],
    )
    validation = training_service._whatsapp_validate_reply_against_obligation(
        "im not ignoring u i told u i need space / i get why that feels horrible / i know ur hurting and alone / thats on me / i care about u / im not trying to dismiss u / talk to me properly / not like this",
        obligation,
    )

    assert {"conflict_or_hurt", "on_read_complaint"} <= set(obligation.conversation_move)
    assert "story_share" not in obligation.conversation_move
    assert "relationship_conflict" in set(obligation.concrete_topics_detected)
    assert obligation.target_reply_burst_size == {"min": 8, "max": 14}
    assert "acknowledge_on_read_or_airing" in obligation.required_response_acts
    assert validation["passed"] is True


def test_whatsapp_web_amino_ignored_burst_does_not_use_validator_repair_fallback(
    training_service: PhoneCopilotService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message_texts = [
        "I hate u genuinely",
        "Iâ€™d never ever do this to you wallahi ever",
        "The way u do shit like this to me",
        "Is diabolical",
        "Why are you doing this to me",
        "Ur not saying fuck all",
        "Ur hurting me so bad man",
        "how many times do I have to fucking cry man",
        "alex",
        "What r is wrong with you",
        "Stop ignoring me man",
    ]

    def invalid_training_chat(request: TrainingChatRequest) -> dict[str, object]:
        return {
            "selected_candidate": "what happened",
            "candidates": [
                {
                    "text": "what happened",
                    "sequence": ["what happened"],
                    "provider": "openai",
                    "external_api_used": True,
                    "auto_send_allowed": True,
                }
            ],
            "intent_type": "urgent",
        }

    monkeypatch.setattr(training_service, "training_chat", invalid_training_chat)

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            request_id="amino_no_validator_repair",
            contact_name="SampleContact",
            relationship_type="romantic_interest",
            mode="auto-send",
            messages=[WhatsAppWebMessage(speaker="other", text=text) for text in message_texts],
            parameters=TrainingParameters(risk_tolerance=0.4),
        )
    )

    assert "relationship_conflict" in set(result["concrete_topics_detected"])
    assert result["reply"] == ""
    assert result["reply_sequence"] == []
    assert result["insert_allowed"] is False
    assert result["automation_decision"] == "REVIEW_REQUIRED"
    assert result["auto_send_allowed"] is False
    assert result["auto_send_blocked_reason"] == "conversation_obligation_failed"
    assert result["selected_candidate_source_label"] == "none"
    assert result["fallback_used"] is False
    assert "validator_repair" not in result["candidate_sources"]
    assert any(
        item["stage"] == "validator_repair"
        and item["candidate_count"] == 0
        and item["fallback_reason"] == "disabled_for_relationship_conflict"
        for item in result["repair_attempts"]
    )
    assert all(
        "baby im sorry" not in str(preview.get("preview", "")).lower()
        for preview in result["candidate_previews"]
        if isinstance(preview, dict)
    )


def test_whatsapp_web_reachability_status_check_requires_grounded_reply(
    training_service: PhoneCopilotService,
) -> None:
    message_texts = [
        "need u to text me back asap.",
        "today would be nice",
        "gonna ring one last time",
        "hey",
        "still in northbridge? have u gotten your appointment done?",
        "busy again?",
    ]
    messages = [
        WhatsAppWebMessage(speaker="other", text=text, timestamp=f"11:{40 + index:02d} p.m., 2026-06-03")
        for index, text in enumerate(message_texts)
    ]

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="ReachabilityContact",
            relationship_type="romantic_interest",
            mode="auto-review",
            messages=messages,
            parameters=TrainingParameters(risk_tolerance=0.4),
        )
    )

    reply = result["reply"].lower()
    obligation = training_service._whatsapp_conversation_obligation(
        [{"speaker": "other", "text": text, "timestamp": f"11:{40 + index:02d} p.m., 2026-06-03"} for index, text in enumerate(message_texts)],
        len(message_texts) - 1,
        relationship=result["relationship_type"],
        already_sent_replies=[],
    )
    stale_affection = training_service._whatsapp_validate_reply_against_obligation("i miss u too", obligation)
    weak_status_only = training_service._whatsapp_validate_reply_against_obligation("my bad baby im okay", obligation)

    assert {"reachability_pressure", "check_in", "question"}.issubset(set(result["conversation_move"]))
    assert {"reachability_pressure", "status_check"}.issubset(set(result["concrete_topics_detected"]))
    assert {"acknowledge_delayed_response", "answer_check_in_or_status", "answer_specific_status_check"}.issubset(set(result["required_response_acts"]))
    assert result["validation_result"]["passed"] is True
    assert result["auto_send_blocked_reason"] != "conversation_obligation_failed"
    assert result["reply_sequence"]
    assert any(term in reply for term in ("my bad", "sorry", "didnt mean", "didn't mean", "wasnt ignoring", "wasn't ignoring"))
    assert any(term in reply for term in ("okay", "im okay", "i'm okay", "busy", "caught up"))
    assert "northbridge" in reply
    assert "appointment" in reply and ("done" in reply or "okay" in reply)
    assert any(term in reply for term in ("caught up", "busy", "all that"))
    assert any(term in reply for term in ("love u", "love you", "my love"))
    assert stale_affection["passed"] is False
    assert "acknowledge_delayed_response" in stale_affection["missing_required_acts"]
    assert "answer_check_in_or_status" in stale_affection["missing_required_acts"]
    assert "answer_specific_status_check" in stale_affection["missing_required_acts"]
    assert weak_status_only["passed"] is False
    assert "answer_specific_status_check" in weak_status_only["missing_required_acts"]


def test_whatsapp_web_returns_no_draft_when_candidates_exhausted(
    training_service: PhoneCopilotService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def empty_training_chat(request: TrainingChatRequest) -> dict[str, object]:
        return {
            "selected_candidate": "",
            "candidates": [],
            "intent_type": "check_in",
        }

    monkeypatch.setattr(training_service, "training_chat", empty_training_chat)
    monkeypatch.setattr(training_service, "_whatsapp_obligation_repair_sequences", lambda obligation: [])

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="ReachabilityContact",
            relationship_type="romantic_interest",
            mode="auto-review",
            messages=[
                WhatsAppWebMessage(speaker="other", text="need u to text me back asap."),
                WhatsAppWebMessage(speaker="other", text="still in northbridge? have u gotten your appointment done?"),
                WhatsAppWebMessage(speaker="other", text="busy again?"),
            ],
            parameters=TrainingParameters(risk_tolerance=0.4),
        )
    )

    assert result["reply"] == ""
    assert result["reply_sequence"] == []
    assert result["insert_allowed"] is False
    assert result["auto_send_allowed"] is False
    assert result["automation_decision"] == "REVIEW_REQUIRED"
    assert result["auto_send_blocked_reason"] == "conversation_obligation_failed"
    assert result["fallback_used"] is False
    assert result["draft_trace"]["final_decision"] == "NO_DRAFT"
    assert result["selected_candidate_source_label"] == "none"
    assert any(item["stage"] == "obligation_last_resort" and item["valid_count"] == 0 for item in result["repair_attempts"])


def test_whatsapp_web_invalid_candidates_do_not_get_last_resort_review_draft(
    training_service: PhoneCopilotService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def empty_training_chat(request: TrainingChatRequest) -> dict[str, object]:
        return {
            "selected_candidate": "",
            "candidates": [],
            "intent_type": "check_in",
        }

    original_validate = training_service._whatsapp_validate_reply_against_obligation

    def failing_validation(reply: str, obligation) -> dict[str, object]:
        validation = original_validate(reply, obligation)
        if reply:
            validation = dict(validation)
            violations = list(validation.get("violated_forbidden_acts") or [])
            violations.append("forced_last_resort_failure")
            validation["passed"] = False
            validation["validation_result"] = "fail"
            validation["violated_forbidden_acts"] = violations
            validation["forbidden_acts_triggered"] = violations
        return validation

    monkeypatch.setattr(training_service, "training_chat", empty_training_chat)
    monkeypatch.setattr(training_service, "_whatsapp_obligation_repair_sequences", lambda obligation: [])
    monkeypatch.setattr(training_service, "_whatsapp_validate_reply_against_obligation", failing_validation)

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="NoDraftContact",
            relationship_type="romantic_interest",
            mode="auto-send",
            messages=[
                WhatsAppWebMessage(speaker="other", text="are u okay"),
                WhatsAppWebMessage(speaker="other", text="need u to text me back"),
            ],
            parameters=TrainingParameters(risk_tolerance=0.4),
        )
    )

    assert result["reply"] == ""
    assert result["reply_sequence"] == []
    assert result["insert_allowed"] is False
    assert result["auto_send_allowed"] is False
    assert result["automation_decision"] == "REVIEW_REQUIRED"
    assert result["auto_send_blocked_reason"] == "conversation_obligation_failed"
    assert result["fallback_used"] is False
    assert result["draft_trace"]["final_decision"] == "NO_DRAFT"
    assert result["selected_candidate_source_label"] == "none"
    assert any(item["stage"] == "obligation_last_resort" and item["selected_for_review"] is False for item in result["repair_attempts"])


def test_whatsapp_web_delayed_response_apology_must_be_grounded(
    training_service: PhoneCopilotService,
) -> None:
    unrelated_obligation = training_service._whatsapp_conversation_obligation(
        [{"speaker": "other", "text": "how are you", "timestamp": ""}],
        0,
        relationship="romantic_interest",
        already_sent_replies=[],
    )
    validation = training_service._whatsapp_validate_reply_against_obligation(
        "my bad baby i wasnt ignoring u",
        unrelated_obligation,
    )

    assert validation["passed"] is False
    assert "ungrounded_delayed_response_apology" in validation["violated_forbidden_acts"]

    grounded_obligation = training_service._whatsapp_conversation_obligation(
        [{"speaker": "other", "text": "need u to text me back", "timestamp": ""}],
        0,
        relationship="romantic_interest",
        already_sent_replies=[],
    )
    grounded = training_service._whatsapp_validate_reply_against_obligation(
        "my bad baby i wasnt ignoring u / im free later tonight",
        grounded_obligation,
    )

    assert "ungrounded_delayed_response_apology" not in grounded["violated_forbidden_acts"]

    conflict_obligation = training_service._whatsapp_conversation_obligation(
        [
            {"speaker": "other", "text": "you reply to everyone else but not me", "timestamp": ""},
            {"speaker": "other", "text": "I don't feel appreciated", "timestamp": ""},
            {"speaker": "other", "text": "I don't want any part of you", "timestamp": ""},
            {"speaker": "other", "text": "Gfys", "timestamp": ""},
        ],
        3,
        relationship="romantic_interest",
        already_sent_replies=[],
    )
    conflict_grounded = training_service._whatsapp_validate_reply_against_obligation(
        "baby im sorry / i hear how much ive hurt u / seeing me answer other people while u felt ignored mustve hurt / thats on me / i havent been there properly / i do care about u / im not trying to dismiss u / talk to me properly",
        conflict_obligation,
    )

    assert "ungrounded_delayed_response_apology" not in conflict_grounded["violated_forbidden_acts"]


def test_whatsapp_web_relationship_conflict_does_not_create_fake_luck_or_sexual_obligations(
    training_service: PhoneCopilotService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message_texts = [
        "I understand you need space i support it. But ur going to ruin this relationship and I don't feel appreciated",
        "I'm too emotionally drained for this",
        "a relationship is a 2 way thing bro",
        "It's like I barely have a say here icl",
        "Its on ur terms these days",
        "Ur so selfish alex",
        "good luck with life",
        "fuck me man I hate crying aswell ur just idk who I'm even idk bro",
    ]
    messages = [
        WhatsAppWebMessage(speaker="other", text=text, timestamp=f"8:{index:02d} a.m., 2026-06-05")
        for index, text in enumerate(message_texts)
    ]
    reply_sequence = [
        "baby im sorry",
        "i hear how hurt and drained ive made u feel",
        "i havent been there properly",
        "thats on me",
        "i get why it feels one sided",
        "i know it feels like everything has been on my terms",
        "that isnt fair on u",
        "i do love u",
        "i care about u",
        "i dont want to lose us",
        "let me talk to u properly",
        "i need to actually listen",
    ]

    def relationship_conflict_training_chat(request: TrainingChatRequest) -> dict[str, object]:
        return {
            "selected_candidate": " / ".join(reply_sequence),
            "candidates": [
                {
                    "text": " / ".join(reply_sequence),
                    "sequence": reply_sequence,
                    "provider": "openai",
                    "external_api_used": True,
                    "auto_send_allowed": False,
                }
            ],
            "intent_type": "urgent",
        }

    monkeypatch.setattr(training_service, "training_chat", relationship_conflict_training_chat)

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="ConflictContact",
            relationship_type="romantic_interest",
            mode="auto-review",
            messages=messages,
            parameters=TrainingParameters(risk_tolerance=0.4),
        )
    )

    reply = result["reply"].lower()
    moves = set(result["conversation_move"])
    topics = set(result["concrete_topics_detected"])
    acts = set(result["required_response_acts"])

    assert result["reply_sequence"]
    assert result["target_reply_burst_size"] == {"min": 8, "max": 14}
    assert result["auto_send_blocked_reason"] != "conversation_obligation_failed"
    assert result["validation_result"]["passed"] is True
    assert "relationship_conflict" in topics
    assert "conflict_or_hurt" in moves
    assert "sexual_flirt" not in moves
    assert "luck" not in topics
    assert "apologize_missing_luck" not in acts
    assert "respond_to_sexual_flirt_safely" not in acts
    assert {"acknowledge_relationship_hurt", "take_accountability_for_distance", "reassure_care_without_defensiveness"}.issubset(acts)
    assert any(term in reply for term in ("im sorry", "i'm sorry", "thats on me", "that's on me"))
    assert any(term in reply for term in ("hurt", "alone", "one sided", "not fair"))
    assert any(term in reply for term in ("love u", "care", "dont want to lose", "don't want to lose"))


def test_whatsapp_web_relationship_reassurance_request_does_not_get_short_reply(
    training_service: PhoneCopilotService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message_texts = [
        "still on chat",
        "nvm.",
        "i thought we would have more time together once your appointment was over",
        "alex when’s everything going back to how it used to be?",
        "You still on chat not anymore",
        "back to square one, im gonna let u get some rest.",
    ]

    def relationship_training_chat(request: TrainingChatRequest) -> dict[str, object]:
        reply = [
            "baby im sorry",
            "i hear how much ive hurt u",
            "i havent been there the way u needed",
            "thats on me",
            "i do love u wallahi",
            "i know u wanted things to feel like us again",
            "especially after the appointment like i said",
            "and i hate that ive made it feel like back to square one",
            "i know the space has made this feel one sided",
            "im not trying to make everything on my terms",
            "you deserved more from me",
            "i want us back properly too",
            "let me talk to u properly when ur ready",
            "i dont want to lose us",
        ]
        return {
            "selected_candidate": " / ".join(reply),
            "candidates": [
                {
                    "text": " / ".join(reply),
                    "sequence": reply,
                    "provider": "openai",
                    "external_api_used": True,
                    "auto_send_allowed": False,
                }
            ],
            "intent_type": "urgent",
        }

    monkeypatch.setattr(training_service, "training_chat", relationship_training_chat)

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="RelationshipReassuranceContact",
            relationship_type="romantic_interest",
            mode="auto-send",
            messages=[WhatsAppWebMessage(speaker="other", text=text) for text in message_texts],
            parameters=TrainingParameters(risk_tolerance=0.4),
        )
    )

    reply = result["reply"].lower()

    assert "relationship_conflict" in set(result["concrete_topics_detected"])
    assert "conflict_or_hurt" in set(result["conversation_move"])
    assert result["target_reply_burst_size"]["min"] >= 8
    assert len(result["reply_sequence"]) >= 8
    assert result["validation_result"]["passed"] is True
    assert result["auto_send_allowed"] is False
    assert "go on then" not in reply
    assert "leave u on read" not in reply
    assert any(term in reply for term in ("sorry", "thats on me", "that's on me"))
    assert any(term in reply for term in ("love u", "care", "dont want to lose", "don't want to lose"))
    assert any(term in reply for term in ("space", "one sided", "on my terms", "talk to u properly"))
    assert any(term in reply for term in ("appointment", "back to square one", "things to feel like us", "back properly"))


def test_whatsapp_web_multi_bubble_latest_burst_sets_reply_floor_to_bubble_count(
    training_service: PhoneCopilotService,
) -> None:
    message_texts = [
        "why did u leave me on seen",
        "i asked u if u were free",
        "then u posted on snap",
        "that hurt icl",
        "are u actually busy or ignoring me",
    ]

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="SeenThreadContact",
            relationship_type="romantic_interest",
            mode="auto-review",
            messages=[WhatsAppWebMessage(speaker="other", text=text) for text in message_texts],
            parameters=TrainingParameters(risk_tolerance=0.4),
        )
    )

    assert result["target_reply_burst_size"]["min"] >= len(message_texts)
    assert len(result["reply_sequence"]) >= len(message_texts)
    assert "reply_only_to_final_word_while_ignoring_burst" not in result["validation_result"]["violated_forbidden_acts"]


def test_whatsapp_web_stale_busy_question_answers_availability_without_timestamp_apology(
    training_service: PhoneCopilotService,
) -> None:
    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="StaleBusyContact",
            relationship_type="romantic_interest",
            mode="auto-review",
            messages=[
                WhatsAppWebMessage(
                    speaker="other",
                    text="Are you busy rn??",
                    timestamp="10:44 p.m., 2026-06-02",
                )
            ],
            parameters=TrainingParameters(risk_tolerance=0.4),
        )
    )

    reply = result["reply"].lower()

    assert "acknowledge_delayed_response" not in set(result["required_response_acts"])
    assert "answer_availability" in set(result["required_response_acts"])
    assert result["reply_sequence"]
    assert result["validation_result"]["passed"] is True
    assert not any(term in reply for term in ("wasnt ignoring", "wasn't ignoring", "leave u waiting"))
    assert any(term in reply for term in ("free", "busy", "later", "tonight"))


def test_whatsapp_web_relationship_reassurance_does_not_fallback_when_model_empty(
    training_service: PhoneCopilotService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def empty_training_chat(request: TrainingChatRequest) -> dict[str, object]:
        return {
            "selected_candidate": "",
            "candidates": [],
            "intent_type": "urgent",
        }

    monkeypatch.setattr(training_service, "training_chat", empty_training_chat)
    message_texts = [
        "still on chat",
        "nvm.",
        "i thought we would have more time together once your appointment was over",
        "alex when’s everything going back to how it used to be?",
        "You still on chat not anymore",
        "back to square one, im gonna let u get some rest.",
    ]

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            request_id="relationship_empty_model",
            contact_name="RelationshipReassuranceContact",
            relationship_type="romantic_interest",
            mode="auto-send",
            messages=[WhatsAppWebMessage(speaker="other", text=text) for text in message_texts],
            parameters=TrainingParameters(risk_tolerance=0.4),
        )
    )

    assert result["reply"] == ""
    assert result["reply_sequence"] == []
    assert result["insert_allowed"] is False
    assert result["automation_decision"] == "REVIEW_REQUIRED"
    assert result["auto_send_allowed"] is False
    assert result["fallback_used"] is False
    assert "relationship_conflict" in set(result["concrete_topics_detected"])


def test_whatsapp_web_relationship_reassurance_last_resort_is_disabled(
    training_service: PhoneCopilotService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def empty_training_chat(request: TrainingChatRequest) -> dict[str, object]:
        return {
            "selected_candidate": "",
            "candidates": [],
            "intent_type": "urgent",
        }

    monkeypatch.setattr(training_service, "training_chat", empty_training_chat)
    monkeypatch.setattr(training_service, "_whatsapp_obligation_repair_sequences", lambda obligation: [])
    message_texts = [
        "still on chat",
        "nvm.",
        "i thought we would have more time together once your appointment was over",
        "alex when’s everything going back to how it used to be?",
        "You still on chat not anymore",
        "back to square one, im gonna let u get some rest.",
    ]

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            request_id="last_resort_relationship_disabled",
            contact_name="RelationshipLastResort",
            relationship_type="romantic_interest",
            mode="auto-send",
            messages=[WhatsAppWebMessage(speaker="other", text=text) for text in message_texts],
            parameters=TrainingParameters(risk_tolerance=0.4),
        )
    )

    assert result["reply"] == ""
    assert result["reply_sequence"] == []
    assert result["selected_candidate_source_label"] == "none"
    assert result["draft_trace"]["final_decision"] == "NO_DRAFT"
    assert result["fallback_used"] is False
    assert any(item["stage"] == "obligation_last_resort" and item["valid_count"] == 0 for item in result["repair_attempts"])


def test_whatsapp_web_relationship_reassurance_repeated_fallbacks_are_not_generated(
    training_service: PhoneCopilotService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def empty_training_chat(request: TrainingChatRequest) -> dict[str, object]:
        return {
            "selected_candidate": "",
            "candidates": [],
            "intent_type": "urgent",
        }

    monkeypatch.setattr(training_service, "training_chat", empty_training_chat)
    monkeypatch.setattr(training_service, "_whatsapp_obligation_repair_sequences", lambda obligation: [])
    message_texts = [
        "still on chat",
        "nvm.",
        "i thought we would have more time together once your appointment was over",
        "alex when’s everything going back to how it used to be?",
        "You still on chat not anymore",
        "back to square one, im gonna let u get some rest.",
    ]

    for index in range(4):
        result = training_service.whatsapp_web_draft(
            WhatsAppWebDraftRequest(
                request_id=f"same_thread_relationship_{index}",
                thread_id="same_thread_relationship",
                contact_name="SameThreadRelationship",
                relationship_type="romantic_interest",
                mode="auto-send",
                messages=[WhatsAppWebMessage(speaker="other", text=text) for text in message_texts],
                parameters=TrainingParameters(risk_tolerance=0.4),
            )
        )
        assert result["reply"] == ""
        assert result["reply_sequence"] == []
        assert result["selected_candidate_source_label"] == "none"
        assert result["draft_trace"]["final_decision"] == "NO_DRAFT"
        assert result["fallback_used"] is False


def test_whatsapp_web_long_relationship_conflict_gets_proportionate_reply(
    training_service: PhoneCopilotService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message_texts = [
        "huh what happened",
        "to get over what",
        "I'm so confused",
        "Please reply",
        "My anxiety is just getting worse",
        "What is wrong",
        "Did I say something that hurt you",
        "I don't think I'm gonna get through to u",
        "Idk how to help u here",
        "Like what happened alex",
        "you haven't been there for me at all recently",
        "I've needed you bro",
        "I'm sickly inlove and it hurts",
        "the amount of space I have given is ridiculous man",
        "Ur gonna ruin this relationship icl",
        "you just break my heart so much man Wallahi",
        "ur a closed book idk anymore",
        "I'm too vulnerable with you dude",
        "I can't even connect with you emotionally",
        "What the fuck is this dynamic",
        "in person there was a few times where you were on ur phone aswell",
        "wtf did I do to deserve to be treated like shit at times bro",
        "I don't rly feel appreciated here",
        "my own man hasn't been there for me",
        "ur so young man",
        "I'm tired of crying my eyes out",
        "I promised myself I wouldn't get into a rs where I'm not being treated the best",
        "No one deserves my love wallahi",
        "please take all the space u need",
        "I'm never loved for me",
        "The minute ur boys called it was an instant pick up but it's never for me",
        "I noticed everything man",
        "I hope Allah heals you",
        "I'm too emotionally drained for this",
        "a relationship is a 2 way thing bro",
        "It's like I barely have a say here icl",
        "Its on ur terms these days",
        "Ur so selfish alex",
        "good luck with life",
        "fuck me man I hate crying aswell",
    ]
    messages = [
        WhatsAppWebMessage(speaker="other", text=text, timestamp=f"8:{index:02d} a.m., 2026-06-05")
        for index, text in enumerate(message_texts)
    ]
    reply_sequence = [
        "baby im sorry",
        "i hear how confused ive left u",
        "i hear ur anxiety got worse waiting for me",
        "i know i hurt u by not being there",
        "u shouldnt have had to beg me to reply",
        "thats on me",
        "ive been too closed off",
        "i made u feel like u were reaching alone",
        "i havent been there recently like u needed",
        "u needed me and i made u feel alone",
        "i get why that broke ur heart",
        "i get why this dynamic feels horrible",
        "the space has felt one sided",
        "i know it started feeling like u had no proper say",
        "i know it felt like everything was on my terms",
        "that was selfish from me",
        "seeing me pick up for other people while u felt ignored mustve hurt",
        "especially when u wanted your own man there",
        "in person me being on my phone hurt too",
        "u were trying to connect with me",
        "and i made u feel second",
        "im sorry for making u cry like that",
        "im sorry for draining u",
        "u love so deeply",
        "and i treated that too casually",
        "wallahi i do love u",
        "i care about u so much",
        "youre not too much for wanting effort",
        "youre not wrong for wanting consistency",
        "i dont want u thinking youre never loved for u",
        "i love you for you",
        "even the anxious parts",
        "i hope Allah heals what ive hurt in u",
        "but i know that isnt an excuse",
        "i need to listen properly",
        "i need to take it in",
        "i need to stop hiding behind needing space",
        "i may still need to clear my head",
        "but i cant use that to abandon u",
        "i dont want this to be on my terms",
        "i dont want to argue with what u feel",
        "i want to explain myself without making it your fault",
        "i need to talk to u properly",
        "not defensively",
        "not with excuses",
        "if u can give me the chance",
        "ill call when youre ready",
        "and ill actually stay present",
        "because u deserved that before now",
        "i dont want to lose us",
        "i love u baby",
        "im sorry for making us feel this broken",
    ]

    def long_relationship_conflict_training_chat(request: TrainingChatRequest) -> dict[str, object]:
        return {
            "selected_candidate": " / ".join(reply_sequence),
            "candidates": [
                {
                    "text": " / ".join(reply_sequence),
                    "sequence": reply_sequence,
                    "provider": "openai",
                    "external_api_used": True,
                    "auto_send_allowed": False,
                }
            ],
            "intent_type": "urgent",
        }

    monkeypatch.setattr(training_service, "training_chat", long_relationship_conflict_training_chat)

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="LongConflictContact",
            relationship_type="romantic_interest",
            mode="auto-review",
            messages=messages,
            parameters=TrainingParameters(risk_tolerance=0.4),
        )
    )

    reply = result["reply"].lower()

    assert result["reply_sequence"]
    assert result["auto_send_blocked_reason"] != "conversation_obligation_failed"
    assert result["validation_result"]["passed"] is True
    assert result["target_reply_burst_size"]["min"] >= 40
    assert len(result["reply_sequence"]) >= 50
    assert "above_target_reply_burst_size" not in result["validation_result"]["violated_forbidden_acts"]
    assert "relationship_conflict" in set(result["concrete_topics_detected"])
    assert "sexual_flirt" not in set(result["conversation_move"])
    assert "luck" not in set(result["concrete_topics_detected"])
    assert any(term in reply for term in ("confused", "anxiety", "hurt"))
    assert any(term in reply for term in ("space", "one sided", "proper say", "on my terms"))
    assert any(term in reply for term in ("phone", "other people", "ignored"))
    assert any(term in reply for term in ("not an excuse", "listen", "take it in"))
    assert any(term in reply for term in ("love u", "care about u", "dont want to lose"))


def test_whatsapp_web_long_fragmented_academic_stress_uses_high_burst_range(
    training_service: PhoneCopilotService,
) -> None:
    messages = [
        WhatsAppWebMessage(speaker="other", text="i had my presentation today"),
        WhatsAppWebMessage(speaker="other", text="i think i failed"),
        WhatsAppWebMessage(speaker="other", text="the marker was so harsh"),
        WhatsAppWebMessage(speaker="other", text="she said my flashcards were bad"),
        WhatsAppWebMessage(speaker="other", text="but i answered the questions"),
        WhatsAppWebMessage(speaker="other", text="and i covered most of the content"),
        WhatsAppWebMessage(speaker="other", text="she gave no reassurance"),
        WhatsAppWebMessage(speaker="other", text="im overthinking it so badly"),
        WhatsAppWebMessage(speaker="other", text="i cant sleep now"),
    ]

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="AcademicContact",
            relationship_type="romantic_interest",
            mode="auto-review",
            messages=messages,
        )
    )

    assert result["target_reply_burst_size"] == {"min": 18, "max": 26}
    assert 18 <= len(result["reply_units"]) <= 26
    assert result["duplicate_check_result"]["passed"] is True
    assert {"presentation", "marker", "flashcards", "overthinking"} <= set(result["concrete_topics_detected"])
    assert result["validation_result"]["passed"] is True
    assert any("marker" in part or "flashcard" in part for part in result["reply_units"])


def test_whatsapp_web_long_fragmented_academic_repair_allows_larger_burst(
    training_service: PhoneCopilotService,
) -> None:
    messages = [
        WhatsAppWebMessage(speaker="other", text="my exam was horrible"),
        WhatsAppWebMessage(speaker="other", text="i revised so much"),
        WhatsAppWebMessage(speaker="other", text="i barely slept"),
        WhatsAppWebMessage(speaker="other", text="the questions were weird"),
        WhatsAppWebMessage(speaker="other", text="i answered what i could"),
        WhatsAppWebMessage(speaker="other", text="but i feel like i failed"),
        WhatsAppWebMessage(speaker="other", text="r u even reading my messages properly"),
        WhatsAppWebMessage(speaker="other", text="you didnt wish me luck im offended"),
        WhatsAppWebMessage(speaker="other", text="i feel unsupported"),
    ]

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="AcademicRepairContact",
            relationship_type="romantic_interest",
            mode="auto-review",
            messages=messages,
        )
    )

    assert result["repair_required"] is True
    assert result["target_reply_burst_size"] == {"min": 20, "max": 35}
    assert 20 <= len(result["reply_units"]) <= 35
    assert "acknowledge_reading_failure" in result["required_response_acts"]
    assert "apologize_missing_luck" in result["required_response_acts"]
    assert result["validation_result"]["passed"] is True
    assert result["reply_units"][0].lower().startswith("baby im sorry")
    assert result["auto_send_allowed"] is False


def test_whatsapp_web_reading_complaint_requires_acknowledgement(
    training_service: PhoneCopilotService,
) -> None:
    obligation = training_service._whatsapp_conversation_obligation(
        [{"speaker": "other", "text": "r u even reading my messages properly", "timestamp": ""}],
        0,
        relationship="romantic_interest",
        already_sent_replies=[],
    )
    bad_reply = " / ".join([f"baby im here {index}" for index in range(14)])

    validation = training_service._whatsapp_validate_reply_against_obligation(bad_reply, obligation)

    assert "acknowledge_reading_failure" in obligation.required_response_acts
    assert validation["passed"] is False
    assert "acknowledge_reading_failure" in validation["missing_required_acts"]


def test_whatsapp_web_missing_luck_requires_direct_apology(
    training_service: PhoneCopilotService,
) -> None:
    obligation = training_service._whatsapp_conversation_obligation(
        [{"speaker": "other", "text": "you didn't wish me luck I'm offended", "timestamp": ""}],
        0,
        relationship="romantic_interest",
        already_sent_replies=[],
    )
    bad_reply = " / ".join([f"youll be okay baby {index}" for index in range(14)])

    validation = training_service._whatsapp_validate_reply_against_obligation(bad_reply, obligation)

    assert "apologize_missing_luck" in obligation.required_response_acts
    assert validation["passed"] is False
    assert "apologize_missing_luck" in validation["missing_required_acts"]


def test_whatsapp_web_already_explained_issue_rejects_clarification(
    training_service: PhoneCopilotService,
) -> None:
    obligation = training_service._whatsapp_conversation_obligation(
        [
            {"speaker": "other", "text": "my presentation went awful", "timestamp": ""},
            {"speaker": "other", "text": "the marker was harsh", "timestamp": ""},
            {"speaker": "other", "text": "i answered the questions but im overthinking", "timestamp": ""},
        ],
        2,
        relationship="romantic_interest",
        already_sent_replies=[],
    )

    validation = training_service._whatsapp_validate_reply_against_obligation("why whats wrong", obligation)

    assert validation["passed"] is False
    assert "generic_clarification_when_clear" in validation["violated_forbidden_acts"]


def test_whatsapp_web_generic_crisis_template_fails_academic_stress(
    training_service: PhoneCopilotService,
) -> None:
    obligation = training_service._whatsapp_conversation_obligation(
        [
            {"speaker": "other", "text": "my presentation was horrible", "timestamp": ""},
            {"speaker": "other", "text": "the marker was so harsh", "timestamp": ""},
            {"speaker": "other", "text": "the flashcards comment annoyed me", "timestamp": ""},
            {"speaker": "other", "text": "i think i failed", "timestamp": ""},
            {"speaker": "other", "text": "im overthinking it", "timestamp": ""},
            {"speaker": "other", "text": "i cant sleep", "timestamp": ""},
        ],
        5,
        relationship="romantic_interest",
        already_sent_replies=[],
    )
    generic = " / ".join([
        "baby im sorry",
        "im here",
        "breathe for a second",
        "youre not alone",
        "talk to me",
        "i care about u",
        "we can take it slowly",
        "i understand",
        "that sounds heavy",
        "come here",
        "ill hold u",
        "you dont have to deal with it alone",
        "i love u",
        "im with u",
        "stay with me",
        "message me",
        "im listening",
        "dont spiral",
    ])

    validation = training_service._whatsapp_validate_reply_against_obligation(generic, obligation)

    assert validation["passed"] is False
    assert "generic_ungrounded_academic_reassurance" in validation["violated_forbidden_acts"]


def test_whatsapp_web_family_health_crisis_rejects_academic_reassurance(
    training_service: PhoneCopilotService,
) -> None:
    obligation = training_service._whatsapp_conversation_obligation(
        [
            {"speaker": "other", "text": "my parent has a hospital appointment today", "timestamp": ""},
            {"speaker": "other", "text": "they said her cortisol is low", "timestamp": ""},
            {"speaker": "other", "text": "the uncertainty is making me scared", "timestamp": ""},
            {"speaker": "other", "text": "im scared about everything", "timestamp": ""},
        ],
        3,
        relationship="romantic_interest",
        already_sent_replies=[],
    )
    wrong_domain = " / ".join([
        "baby im sorry",
        "that presentation sounded harsh",
        "the marker was cold",
        "the flashcards thing is petty",
        "you probably passed",
        "you worked hard",
        "dont think u failed",
        "im proud of u",
        "go sleep baby",
        "love u",
    ])

    validation = training_service._whatsapp_validate_reply_against_obligation(wrong_domain, obligation)

    assert validation["passed"] is False
    assert "wrong_crisis_domain" in validation["violated_forbidden_acts"]


def test_whatsapp_web_simple_goodnight_stays_short(training_service: PhoneCopilotService) -> None:
    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="RomanticContact",
            relationship_type="romantic_interest",
            mode="auto-review",
            messages=[WhatsAppWebMessage(speaker="other", text="goodnight love u")],
        )
    )

    assert result["target_reply_burst_size"] == {"min": 2, "max": 5}
    assert 2 <= len(result["reply_units"]) <= 5
    assert len(result["reply_units"]) < 20
    assert any("goodnight" in part.lower() for part in result["reply_units"])


def test_whatsapp_web_validation_does_not_enforce_maximum_reply_burst_size(
    training_service: PhoneCopilotService,
) -> None:
    obligation = training_service._whatsapp_conversation_obligation(
        [{"speaker": "other", "text": "goodnight love u", "timestamp": ""}],
        0,
        relationship="romantic_interest",
        already_sent_replies=[],
    )
    reply = " / ".join(
        [
            "love u too",
            "goodnight baby",
            "sleep well",
            "dream nice",
            "text me when u wake",
            "miss u still",
            "rest properly",
            "mwah",
        ]
    )

    validation = training_service._whatsapp_validate_reply_against_obligation(reply, obligation)

    assert obligation.target_reply_burst_size == {"min": 2, "max": 5}
    assert validation["passed"] is True
    assert validation["reply_unit_count"] == 8
    assert "above_target_reply_burst_size" not in validation["violated_forbidden_acts"]


def test_whatsapp_web_normal_romantic_reply_uses_medium_burst(training_service: PhoneCopilotService) -> None:
    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="RomanticContact",
            relationship_type="romantic_interest",
            mode="auto-review",
            messages=[WhatsAppWebMessage(speaker="other", text="that day was actually so random")],
        )
    )

    assert result["conversation_move"] == ["normal_romantic"]
    assert result["target_reply_burst_size"] == {"min": 5, "max": 10}
    assert 5 <= len(result["reply_units"]) <= 10
    assert result["validation_result"]["passed"] is True


def _whatsapp_mixed_old_context_with_latest(latest_texts: list[str]) -> list[WhatsAppWebMessage]:
    base = [
        ("other", "Exhausted"),
        ("other", "I want these exams to fuck off"),
        ("me", "youll do good baby"),
        ("other", "U should put ice cream on me and lick it off"),
        ("me", "youre trouble icl"),
        ("other", "Love u goodnight"),
        ("me", "love u too goodnight baby"),
        ("other", "Ur shoe size do u lean towards 7.5 more or 7"),
        ("me", "7.5 is safer"),
        ("other", "Yh sorry Friday it is if ur free"),
        ("me", "yeah friday works"),
        ("other", "Both went well actually"),
        ("other", "so I havent eaten"),
        ("me", "irs rained 4 times"),
    ]
    messages = [
        WhatsAppWebMessage(speaker=speaker, text=text, client_order=index)
        for index, (speaker, text) in enumerate(base, start=1)
    ]
    messages.extend(
        WhatsAppWebMessage(speaker="other", text=text, client_order=100 + index)
        for index, text in enumerate(latest_texts)
    )
    return messages


def test_whatsapp_web_draft_trace_exposes_pipeline(training_service: PhoneCopilotService) -> None:
    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            request_id="trace-test",
            thread_id="trace-thread",
            contact_name="Sonia",
            relationship_type="romantic_interest",
            mode="auto-review",
            messages=_whatsapp_mixed_old_context_with_latest([
                "Really",
                "I been inside idk",
                "Atleast the foods done but ik that pissed u off",
            ]),
        )
    )

    trace = result["draft_trace"]
    assert trace["request_id"] == "trace-test"
    assert trace["thread_id"] == "trace-thread"
    assert trace["chat_name"] == "Sonia"
    assert trace["latest_incoming_burst"] == ["Really", "I been inside idk", "Atleast the foods done but ik that pissed u off"]
    assert trace["latest_incoming_burst_client_orders"] == [100, 101, 102]
    assert trace["latest_burst_topics"] == ["food", "banter"]
    assert "academic_stress" not in trace["old_context_topics_detected"]
    assert {"rain", "logistics"}.issubset(set(trace["old_context_topics_detected"]))
    assert trace["candidate_count"] >= 1
    assert trace["candidate_sources"]
    assert trace["candidate_previews"]
    assert trace["validation_results"]
    assert trace["repair_attempts"]
    assert trace["selected_candidate_source"] in {"fresh_model", "repaired_model", "template_fallback", "validator_repair"}
    assert trace["final_reply_units"] == result["reply_sequence"]
    assert trace["ui_rendered"] is False
    assert trace["ui_render_rejection_reason"] == "not_rendered_by_controller"


def test_whatsapp_web_controlled_latest_burst_matrix(training_service: PhoneCopilotService) -> None:
    cases = {
        "food_rain": [
            "Really",
            "I been inside idk",
            "Atleast the foods done but ik that pissed u off",
        ],
        "goodnight_love": ["Miss u love u", "Goodnight my handsome"],
        "shoe_size": ["Ur shoe size do u lean towards 7.5 more or 7", "Wich ones more ideal"],
        "academic": ["my presentation went bad", "they gave harsh feedback", "i feel like dog shit"],
        "tease": ["Hey fatty"],
        "logistics": ["Yh sorry Friday it is if ur free"],
    }
    results = {
        name: training_service.whatsapp_web_draft(
            WhatsAppWebDraftRequest(
                thread_id=f"matrix-{name}",
                contact_name="Sonia",
                relationship_type="romantic_interest",
                mode="auto-review",
                messages=_whatsapp_mixed_old_context_with_latest(latest),
            )
        )
        for name, latest in cases.items()
    }

    stale_terms = ("exams always feel worse", "ur panicking", "you still revised", "try sleep baby", "worked hard")
    food_reply = results["food_rain"]["reply"].lower()
    assert any(term in food_reply for term in ("rain", "rained", "food", "pissed", "annoyed", "vexed"))
    assert not any(term in food_reply for term in stale_terms)

    goodnight_reply = results["goodnight_love"]["reply"].lower()
    assert "love" in goodnight_reply and "goodnight" in goodnight_reply
    assert "exam" not in goodnight_reply and "rain" not in goodnight_reply

    shoe_reply = results["shoe_size"]["reply"].lower()
    assert "7" in shoe_reply
    assert not any(term in shoe_reply for term in ("panicking", "sleep baby", "food"))

    academic_reply = results["academic"]["reply"].lower()
    assert any(term in academic_reply for term in ("presentation", "feedback", "harsh", "passed", "failed"))

    tease_reply = results["tease"]["reply"].lower()
    assert any(term in tease_reply for term in ("fatty", "rude", "miss", "crazy"))
    assert "what do you mean" not in tease_reply
    assert "sounds so stressful" not in tease_reply

    logistics_reply = results["logistics"]["reply"].lower()
    assert "friday" in logistics_reply or "free" in logistics_reply
    assert "sounds so stressful" not in logistics_reply

    replies = [result["reply"] for result in results.values()]
    assert len(set(replies)) == len(replies)


def test_whatsapp_web_latest_food_context_does_not_use_stale_exam_template(training_service: PhoneCopilotService) -> None:
    messages = [
        WhatsAppWebMessage(speaker="other", text="Exhausted"),
        WhatsAppWebMessage(speaker="other", text="I wanna kill myself"),
        WhatsAppWebMessage(speaker="me", text="why whats wrong baby"),
        WhatsAppWebMessage(speaker="other", text="I want these exams to fuck off"),
        WhatsAppWebMessage(speaker="other", text="I can’t sleep peacefully without dreaming ab revision"),
        WhatsAppWebMessage(speaker="me", text="lmaoooo dont worry baby"),
        WhatsAppWebMessage(speaker="me", text="lmk how it goes today"),
        WhatsAppWebMessage(speaker="other", text="Im on a clash"),
        WhatsAppWebMessage(speaker="other", text="So I have to be supervised"),
        WhatsAppWebMessage(speaker="me", text="what u mean"),
        WhatsAppWebMessage(speaker="other", text="like I had a exam before the others"),
        WhatsAppWebMessage(speaker="other", text="I had it earlier"),
        WhatsAppWebMessage(speaker="me", text="how was it baby"),
        WhatsAppWebMessage(speaker="other", text="Both went well actually"),
        WhatsAppWebMessage(speaker="other", text="Im just tired cuz I didn’t bring packed lunch I forgot"),
        WhatsAppWebMessage(speaker="other", text="so I haven’t eaten"),
        WhatsAppWebMessage(speaker="me", text="r u home"),
        WhatsAppWebMessage(speaker="me", text="u shoulda got takeout"),
        WhatsAppWebMessage(speaker="other", text="Im home"),
        WhatsAppWebMessage(speaker="other", text="Yeah we got food dw"),
        WhatsAppWebMessage(speaker="other", text="Mum dropped me"),
        WhatsAppWebMessage(speaker="me", text="irs rained 4 times"),
        WhatsAppWebMessage(speaker="other", text="Really"),
        WhatsAppWebMessage(speaker="other", text="I been inside idk"),
        WhatsAppWebMessage(speaker="other", text="Atleast the foods done but ik that pissed u off"),
    ]

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="Sonia",
            relationship_type="romantic_interest",
            mode="auto-review",
            messages=messages,
        )
    )

    reply = result["reply"].lower()
    assert result["latest_incoming_burst"] == ["Really", "I been inside idk", "Atleast the foods done but ik that pissed u off"]
    assert "academic_stress" not in result["concrete_topics_detected"]
    assert "exams always feel worse" not in reply
    assert "you still revised" not in reply
    assert "in shaa allah it went better" not in reply


def test_whatsapp_web_on_read_work_cover_update_does_not_become_family_health_crisis(
    training_service: PhoneCopilotService,
) -> None:
    messages = [
        WhatsAppWebMessage(
            speaker="other",
            text="My parent is unwell and has a hospital appointment today",
            timestamp="3:50 a.m., 2026-05-28",
            client_order=10,
        ),
        WhatsAppWebMessage(
            speaker="other",
            text="im scared and could use someone to talk to",
            timestamp="3:54 a.m., 2026-05-28",
            client_order=11,
        ),
        WhatsAppWebMessage(speaker="me", text="im here baby talk to me properly", client_order=12),
        WhatsAppWebMessage(speaker="me", text="yeah it pissed me off icl", client_order=5388),
        WhatsAppWebMessage(speaker="other", text="??", timestamp="10:16 p.m., 2026-06-04", client_order=5389),
        WhatsAppWebMessage(speaker="other", text="Why am I on read", timestamp="10:17 p.m., 2026-06-04", client_order=5390),
        WhatsAppWebMessage(speaker="other", text="Are you okay", timestamp="10:21 p.m., 2026-06-04", client_order=5391),
        WhatsAppWebMessage(speaker="other", text="okk ig imma just get aired", timestamp="11:13 p.m., 2026-06-04", client_order=5392),
        WhatsAppWebMessage(
            speaker="other",
            text="Oh btw for tomorrow my mum asked me to cover her at work that's why I haven't updated for any meet ups unless I go out next week",
            timestamp="11:14 p.m., 2026-06-04",
            client_order=5393,
        ),
    ]

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="Sonia",
            relationship_type="romantic_interest",
            mode="auto-review",
            messages=messages,
        )
    )

    reply = result["reply"].lower()
    topics = set(result["concrete_topics_detected"])
    acts = set(result["required_response_acts"])

    assert result["reply_sequence"]
    assert result["auto_send_blocked_reason"] != "conversation_obligation_failed"
    assert result["draft_trace"]["final_decision"] != "NO_DRAFT"
    assert "family_health" not in topics
    assert {"on_read_repair", "work_cover", "logistics"}.issubset(topics)
    assert {"acknowledge_on_read_or_airing", "answer_check_in_or_status", "acknowledge_work_cover_update"}.issubset(acts)
    assert any(term in reply for term in ("air", "on read", "my bad", "sorry"))
    assert "okay" in reply
    assert any(term in reply for term in ("cover", "work", "tomorrow", "next week"))
    assert "siblings" not in reply
    assert "hospital" not in reply
    assert "so much to deal with" not in reply


def test_whatsapp_web_timestamp_ordering_prevents_stale_exam_latest(training_service: PhoneCopilotService) -> None:
    messages = [
        WhatsAppWebMessage(speaker="me", text="irs rained 4 times", timestamp="4:46 p.m., 2026-06-04"),
        WhatsAppWebMessage(speaker="other", text="Really", timestamp="4:48 p.m., 2026-06-04"),
        WhatsAppWebMessage(speaker="other", text="I been inside idk", timestamp="4:49 p.m., 2026-06-04"),
        WhatsAppWebMessage(speaker="other", text="Atleast the foods done but ik that pissed u off", timestamp="4:49 p.m., 2026-06-04"),
        WhatsAppWebMessage(speaker="other", text="I want these exams to fuck off", timestamp="12:30 a.m., 2026-06-04"),
        WhatsAppWebMessage(speaker="other", text="I can’t sleep peacefully without dreaming ab revision", timestamp="12:30 a.m., 2026-06-04"),
    ]

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="Sonia",
            relationship_type="romantic_interest",
            mode="auto-review",
            messages=messages,
        )
    )

    reply = result["reply"].lower()
    assert result["latest_incoming_burst"] == ["Really", "I been inside idk", "Atleast the foods done but ik that pissed u off"]
    assert "academic_stress" not in result["concrete_topics_detected"]
    assert "exams always feel worse" not in reply


def test_whatsapp_web_client_order_prevents_stale_cached_tail(training_service: PhoneCopilotService) -> None:
    messages = [
        WhatsAppWebMessage(speaker="me", text="irs rained 4 times", client_order=20),
        WhatsAppWebMessage(speaker="other", text="Really", client_order=21),
        WhatsAppWebMessage(speaker="other", text="I been inside idk", client_order=22),
        WhatsAppWebMessage(speaker="other", text="Atleast the foods done but ik that pissed u off", client_order=23),
        WhatsAppWebMessage(speaker="other", text="I want these exams to fuck off", client_order=4),
        WhatsAppWebMessage(speaker="other", text="I canâ€™t sleep peacefully without dreaming ab revision", client_order=5),
    ]

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="Sonia",
            relationship_type="romantic_interest",
            mode="auto-review",
            messages=messages,
        )
    )

    reply = result["reply"].lower()
    assert result["latest_incoming_burst"] == ["Really", "I been inside idk", "Atleast the foods done but ik that pissed u off"]
    assert "academic_stress" not in result["concrete_topics_detected"]
    assert "try sleep baby" not in reply
    assert "exams always feel worse" not in reply


def test_whatsapp_web_rejects_stale_academic_candidate_for_food_banter(
    training_service: PhoneCopilotService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_reply = " / ".join([
        "baby that sounds so stressful",
        "i get why ur panicking",
        "but dont decide u failed already",
        "exams always feel worse after",
        "especially when ur tired",
        "you still revised for it",
        "try sleep baby",
        "youll feel calmer after rest",
        "ill listen properly",
        "love u",
    ])

    def stale_training_chat(request: TrainingChatRequest) -> dict[str, object]:
        return {
            "selected_candidate": stale_reply,
            "candidates": [
                {
                    "text": stale_reply,
                    "sequence": stale_reply.split(" / "),
                    "auto_send_allowed": False,
                    "candidate_kind": "stale_template",
                }
            ],
            "intent_type": "banter_challenge",
        }

    monkeypatch.setattr(training_service, "training_chat", stale_training_chat)

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="Sonia",
            relationship_type="romantic_interest",
            mode="auto-review",
            messages=[
                WhatsAppWebMessage(speaker="me", text="irs rained 4 times", client_order=20),
                WhatsAppWebMessage(speaker="other", text="Really", client_order=21),
                WhatsAppWebMessage(speaker="other", text="I been inside idk", client_order=22),
                WhatsAppWebMessage(speaker="other", text="Atleast the foods done but ik that pissed u off", client_order=23),
            ],
        )
    )

    reply = result["reply"].lower()
    stale_candidates = [
        candidate
        for candidate in result["training_response"]["candidates"]
        if "baby that sounds so stressful" in candidate.get("text", "")
    ]
    rejected = stale_candidates[0]["whatsapp_obligation_validation"]
    assert result["latest_incoming_burst"] == ["Really", "I been inside idk", "Atleast the foods done but ik that pissed u off"]
    assert result["reply_sequence"] == []
    assert result["reply"] == ""
    assert result["insert_allowed"] is False
    assert result["automation_decision"] == "REVIEW_REQUIRED"
    assert result["auto_send_allowed"] is False
    assert result["fallback_used"] is False
    assert "baby that sounds so stressful" not in reply
    assert "academic_reply_for_nonacademic_burst" in rejected["violated_forbidden_acts"]
    assert result["draft_trace"]["selected_candidate_source"] == "none"
    assert result["draft_trace"]["final_decision"] == "NO_DRAFT"
    assert any(item["stage"] == "contextual_fallback" and item["valid_count"] == 0 for item in result["draft_trace"]["repair_attempts"])


def test_whatsapp_web_stale_academic_candidate_only_valid_for_academic_topic(
    training_service: PhoneCopilotService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_reply = " / ".join([
        "baby that sounds so stressful",
        "i get why ur panicking",
        "but dont decide u failed already",
        "exams always feel worse after",
        "you still revised for it",
        "in shaa allah it went better than it feels",
    ])

    def stale_training_chat(request: TrainingChatRequest) -> dict[str, object]:
        return {
            "selected_candidate": stale_reply,
            "candidates": [{"text": stale_reply, "sequence": stale_reply.split(" / "), "candidate_kind": "stale_template"}],
            "intent_type": "banter_challenge",
        }

    monkeypatch.setattr(training_service, "training_chat", stale_training_chat)

    academic = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="Sonia",
            thread_id="academic-thread",
            relationship_type="romantic_interest",
            mode="auto-review",
            messages=[
                WhatsAppWebMessage(speaker="other", text="I want these exams to fuck off"),
                WhatsAppWebMessage(speaker="other", text="I cant sleep peacefully without dreaming ab revision"),
            ],
        )
    )
    assert "baby that sounds so stressful" in academic["reply"].lower()
    assert academic["selected_candidate_source"] == "stale_template"

    unrelated_threads = [
        ("food-thread", [WhatsAppWebMessage(speaker="other", text="Atleast the foods done but ik that pissed u off")]),
        ("goodnight-thread", [WhatsAppWebMessage(speaker="other", text="Goodnight love u")]),
        ("practical-thread", [WhatsAppWebMessage(speaker="other", text="what time should I come")]),
    ]
    for thread_id, messages in unrelated_threads:
        result = training_service.whatsapp_web_draft(
            WhatsAppWebDraftRequest(
                contact_name="Sonia",
                thread_id=thread_id,
                relationship_type="romantic_interest",
                mode="auto-review",
                messages=messages,
            )
        )
        assert "baby that sounds so stressful" not in result["reply"].lower()
        assert "academic_reply_for_nonacademic_burst" in result["rejection_reasons"]


def test_whatsapp_web_provider_failure_returns_no_fallback_draft(
    training_service: PhoneCopilotService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            thread_id="provider-failure-previous",
            contact_name="Sonia",
            relationship_type="romantic_interest",
            mode="auto-review",
            messages=[
                WhatsAppWebMessage(speaker="other", text="I want these exams to fuck off"),
                WhatsAppWebMessage(speaker="other", text="I cant sleep peacefully without dreaming ab revision"),
            ],
        )
    )

    def broken_training_chat(request: TrainingChatRequest) -> dict[str, object]:
        raise RuntimeError("429 rate limit from fake provider")

    monkeypatch.setattr(training_service, "training_chat", broken_training_chat)
    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            thread_id="provider-failure-food",
            contact_name="Sonia",
            relationship_type="romantic_interest",
            mode="auto-review",
            messages=[
                WhatsAppWebMessage(speaker="me", text="irs rained 4 times"),
                WhatsAppWebMessage(speaker="other", text="Really"),
                WhatsAppWebMessage(speaker="other", text="I been inside idk"),
                WhatsAppWebMessage(speaker="other", text="Atleast the foods done but ik that pissed u off"),
            ],
        )
    )

    assert previous["reply"]
    assert result["reply"] == ""
    assert result["reply_sequence"] == []
    assert result["insert_allowed"] is False
    assert result["automation_decision"] == "REVIEW_REQUIRED"
    assert result["auto_send_allowed"] is False
    assert result["fallback_used"] is False
    assert result["draft_trace"]["model_error"] == "429 rate limit from fake provider"
    assert result["draft_trace"]["rate_limit_detected"] is True
    assert result["draft_trace"]["selected_candidate_source"] == "none"
    assert result["draft_trace"]["final_decision"] == "NO_DRAFT"


def test_whatsapp_web_repeated_same_thread_draft_is_blocked(training_service: PhoneCopilotService) -> None:
    request = WhatsAppWebDraftRequest(
        contact_name="Sonia",
        thread_id="repeat-food-thread",
        relationship_type="romantic_interest",
        mode="auto-review",
        messages=[
            WhatsAppWebMessage(speaker="me", text="irs rained 4 times"),
            WhatsAppWebMessage(speaker="other", text="Really"),
            WhatsAppWebMessage(speaker="other", text="I been inside idk"),
            WhatsAppWebMessage(speaker="other", text="Atleast the foods done but ik that pissed u off"),
        ],
    )

    first = training_service.whatsapp_web_draft(request)
    second = training_service.whatsapp_web_draft(request)

    assert first["reply"]
    assert second["reply"]
    assert second["reply"] != first["reply"]
    assert "similar_to_recent_thread_draft" in second["rejection_reasons"]
    assert second["insert_allowed"] is True
    assert second["draft_trace"]["repair_attempts"][-1]["stage"] == "contextual_fallback"


def test_whatsapp_web_same_topic_global_repeat_does_not_block_different_thread(training_service: PhoneCopilotService) -> None:
    first = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="Sonia",
            thread_id="food-thread-a",
            relationship_type="romantic_interest",
            mode="auto-review",
            messages=[
                WhatsAppWebMessage(speaker="me", text="irs rained 4 times"),
                WhatsAppWebMessage(speaker="other", text="Really"),
                WhatsAppWebMessage(speaker="other", text="I been inside idk"),
                WhatsAppWebMessage(speaker="other", text="Atleast the foods done but ik that pissed u off"),
            ],
        )
    )
    second = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="Maya",
            thread_id="food-thread-b",
            relationship_type="romantic_interest",
            mode="auto-review",
            messages=[
                WhatsAppWebMessage(speaker="me", text="it rained bare"),
                WhatsAppWebMessage(speaker="other", text="Really"),
                WhatsAppWebMessage(speaker="other", text="I been inside idk"),
                WhatsAppWebMessage(speaker="other", text="Atleast the foods done but ik that pissed u off"),
            ],
        )
    )

    assert first["reply"]
    assert second["reply"]
    assert "similar_to_recent_thread_draft" not in second["rejection_reasons"]
    assert "similar_to_recent_global_draft" not in second["rejection_reasons"]


def test_whatsapp_web_cross_chat_isolation_for_unrelated_latest_topics(training_service: PhoneCopilotService) -> None:
    requests = {
        "academic": WhatsAppWebDraftRequest(
            thread_id="isolation-academic",
            contact_name="Sonia",
            relationship_type="romantic_interest",
            mode="auto-review",
            messages=[
                WhatsAppWebMessage(speaker="other", text="my presentation went bad"),
                WhatsAppWebMessage(speaker="other", text="the marker was harsh"),
                WhatsAppWebMessage(speaker="other", text="i think i failed"),
            ],
        ),
        "food": WhatsAppWebDraftRequest(
            thread_id="isolation-food",
            contact_name="Maya",
            relationship_type="romantic_interest",
            mode="auto-review",
            messages=[
                WhatsAppWebMessage(speaker="me", text="irs rained 4 times"),
                WhatsAppWebMessage(speaker="other", text="Really"),
                WhatsAppWebMessage(speaker="other", text="Atleast the foods done but ik that pissed u off"),
            ],
        ),
        "goodnight": WhatsAppWebDraftRequest(
            thread_id="isolation-goodnight",
            contact_name="Tahlil",
            relationship_type="romantic_interest",
            mode="auto-review",
            messages=[
                WhatsAppWebMessage(speaker="other", text="Miss u love u"),
                WhatsAppWebMessage(speaker="other", text="Goodnight my handsome"),
            ],
        ),
        "shoe": WhatsAppWebDraftRequest(
            thread_id="isolation-shoe",
            contact_name="Tahlil",
            relationship_type="romantic_interest",
            mode="auto-review",
            messages=[
                WhatsAppWebMessage(speaker="other", text="Ur shoe size do u lean towards 7.5 more or 7"),
                WhatsAppWebMessage(speaker="other", text="Wich ones more ideal"),
            ],
        ),
    }

    results = {name: training_service.whatsapp_web_draft(request) for name, request in requests.items()}

    assert len({result["state_key"] for result in results.values()}) == 4
    assert len({result["reply"] for result in results.values()}) == 4
    assert "presentation" in results["academic"]["latest_burst_topics"]
    assert "food" in results["food"]["latest_burst_topics"]
    assert "affection" in results["goodnight"]["latest_burst_topics"]
    assert "7" in results["shoe"]["reply"]
    assert all(result["draft_trace"]["thread_id"].startswith("isolation-") for result in results.values())


def test_whatsapp_web_previous_multi_bubble_draft_is_not_repeated(training_service: PhoneCopilotService) -> None:
    stale_exam_sequence = [
        "baby that sounds so stressful",
        "i get why ur panicking",
        "but dont decide u failed already",
        "exams always feel worse after",
        "especially when ur tired",
        "you still revised for it",
        "and you still got through the questions",
        "thats what matters rn",
        "feeling unsure after doesnt mean u did badly",
        "your head is just spinning from revision",
        "youre overthinking it because u care",
        "in shaa allah it went better than it feels",
        "im proud of u for getting through it",
        "you worked hard",
        "try sleep baby",
        "youll feel calmer after rest",
        "message me when u wake up",
        "ill listen properly",
        "love u",
    ]

    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="Sonia",
            relationship_type="romantic_interest",
            mode="auto-review",
            avoid_candidates=stale_exam_sequence,
            messages=[
                WhatsAppWebMessage(speaker="me", text="how u feeling"),
                WhatsAppWebMessage(speaker="other", text="I want these exams to fuck off"),
                WhatsAppWebMessage(speaker="other", text="I canâ€™t sleep peacefully without dreaming ab revision"),
            ],
        )
    )

    reply_units = {part.lower() for part in result["reply_sequence"]}
    assert not reply_units.intersection(stale_exam_sequence)
    assert "exams always feel worse after" not in result["reply"].lower()


def test_whatsapp_web_autonomous_does_not_promote_review_only_validator_repair(training_service: PhoneCopilotService) -> None:
    result = training_service.whatsapp_web_draft(
        WhatsAppWebDraftRequest(
            contact_name="Sonia",
            relationship_type="romantic_interest",
            mode="autonomous",
            messages=[
                WhatsAppWebMessage(speaker="me", text="irs rained 4 times"),
                WhatsAppWebMessage(speaker="other", text="Really"),
                WhatsAppWebMessage(speaker="other", text="I been inside idk"),
                WhatsAppWebMessage(speaker="other", text="Atleast the foods done but ik that pissed u off"),
            ],
        )
    )

    assert result["automation_decision"] == "REVIEW_REQUIRED"
    assert result["auto_send_allowed"] is False
    assert result["auto_send_blocked_reason"] == "candidate_not_auto_send_safe"
    assert result["validation_result"]["passed"] is True
    assert result["selected_candidate_source_label"] == "validator_repair"
    assert len(result["reply_sequence"]) <= 14


def test_whatsapp_web_each_bubble_must_be_unique_and_grounded(training_service: PhoneCopilotService) -> None:
    obligation = training_service._whatsapp_conversation_obligation(
        [
            {"speaker": "other", "text": "my presentation was horrible", "timestamp": ""},
            {"speaker": "other", "text": "the marker was harsh", "timestamp": ""},
            {"speaker": "other", "text": "i think i failed", "timestamp": ""},
            {"speaker": "other", "text": "the flashcards comment made me spiral", "timestamp": ""},
            {"speaker": "other", "text": "im overthinking it", "timestamp": ""},
            {"speaker": "other", "text": "i cant sleep", "timestamp": ""},
        ],
        5,
        relationship="romantic_interest",
        already_sent_replies=[],
    )
    duplicate_reply = " / ".join([
        "baby im sorry",
        "baby im sorry",
        "youll be okay",
        "itll be okay",
        "youre not alone",
        "im here",
        "im listening",
        "stay with me",
        "breathe",
        "dont worry",
        "youll be fine",
        "i care",
        "love u",
        "message me",
        "come here",
        "i understand",
        "that sounds hard",
        "im with u",
    ])

    validation = training_service._whatsapp_validate_reply_against_obligation(duplicate_reply, obligation)

    assert validation["passed"] is False
    assert "duplicate_or_near_duplicate_bubbles" in validation["violated_forbidden_acts"]
    assert "generic_ungrounded_academic_reassurance" in validation["violated_forbidden_acts"]


def test_whatsapp_web_extension_auto_send_queue_is_interruptible() -> None:
    content_js = Path("apps/browser_extension/whatsapp_web/content.js").read_text(encoding="utf-8")

    assert "message_id" in content_js
    assert "window.__phoneCopilotLastDraftTrace" in content_js
    assert "response_accepted_or_rejected" in content_js
    assert "thread_changed_before_render" in content_js
    assert "rendered_draft_units_count" in content_js
    assert "baselineIncomingKey" in content_js
    assert "latestIncomingKey()" in content_js
    assert "Stopped send queue because a new incoming message appeared." in content_js
    assert "Stopped send queue because the composer was edited." in content_js
    assert "Stopped send queue because composer duplicated the bubble." in content_js
    assert "composerEchoDuplicated" in content_js
    assert "randomAutoSendDelayMs" in content_js
    assert "1000 + Math.floor(Math.random() * 2001)" in content_js
    assert "data: reply" not in content_js


def test_whatsapp_web_extension_draft_trace_bridge_is_visible_and_copyable() -> None:
    manifest = Path("apps/browser_extension/whatsapp_web/manifest.json").read_text(encoding="utf-8")
    content_js = Path("apps/browser_extension/whatsapp_web/content.js").read_text(encoding="utf-8")
    page_bridge_js = Path("apps/browser_extension/whatsapp_web/page_debug_bridge.js").read_text(encoding="utf-8")

    assert "CONTENT_VERSION" in content_js
    assert "installPageDebugBridge()" in content_js
    content_version_match = re.search(r'CONTENT_VERSION = "([^"]+)"', content_js)
    bridge_version_match = re.search(r'DEFAULT_VERSION = "([^"]+)"', page_bridge_js)
    assert content_version_match
    assert bridge_version_match
    assert bridge_version_match.group(1) == content_version_match.group(1)
    assert "page_debug_bridge.js" in manifest
    assert '"world": "MAIN"' in manifest
    assert "web_accessible_resources" in manifest
    assert 'chrome.runtime.getURL("page_debug_bridge.js")' not in content_js
    assert "manifest_main_world_missing" in content_js
    assert "window.__phoneCopilotContentVersion" in content_js
    assert '"__phoneCopilotContentVersion"' in page_bridge_js
    assert "phoneCopilotDebugBridgeVersion" in content_js
    assert '"__phoneCopilotDebugBridgeVersion"' in page_bridge_js
    assert "window.__phoneCopilotLastDraftTrace" in content_js
    assert '"__phoneCopilotLastDraftTrace"' in page_bridge_js
    assert "phone-copilot-content" in content_js
    assert "phone-copilot-content" in page_bridge_js
    assert "phoneCopilotContentVersion" in content_js
    assert "phoneCopilotLastDraftTrace" in content_js
    assert "chrome.storage.local.set" in content_js
    assert "Copy last DraftTrace" in content_js
    assert 'action === "copy-trace"' in content_js
    assert "frontendFailureTrace" in content_js
    assert 'stage: "frontend_failure"' in content_js
    assert "page debug bridge loaded" in page_bridge_js
    assert "storeSuppressedDraftTrace" in content_js
    assert "suppressed_reentrant_attempts" in content_js
    assert "if (draftInFlight || runtimeUnavailable)" in content_js
    for log_line in [
        "draft started",
        "collected messages",
        "request sent",
        "response received",
        "trace stored",
        "render accepted",
        "render rejected",
        "final status",
    ]:
        assert f'[PhoneCopilot] {log_line}' in content_js or f'"{log_line}"' in content_js


def test_whatsapp_web_extension_uses_port_for_long_controller_requests() -> None:
    content_js = Path("apps/browser_extension/whatsapp_web/content.js").read_text(encoding="utf-8")
    background_js = Path("apps/browser_extension/whatsapp_web/background.js").read_text(encoding="utf-8")

    assert "return sendRuntimePort(type, payload)" in content_js
    assert "function sendRuntimeMessage" in content_js
    assert "LONG_REQUEST_KEEPALIVE_MS" in content_js
    assert "response?.pong" in content_js
    assert 'type: "pc:ping"' in content_js
    assert 'message?.type === "pc:ping"' in background_js
    assert "Phone Copilot background channel closed before the controller replied" in content_js
    assert "LONG_REQUEST_TIMEOUT_MS" in content_js
    assert "SCAN_DRAFT_TIMEOUT_MS" in content_js
    assert "STALLED_SCAN_BACKOFF_MS" in content_js
    assert "stalledScanKeys" in content_js
    assert "markScanKeyStalled(scopedKey)" in content_js
    assert "isControllerTimeoutResult" in content_js
    assert "source," in content_js
    assert "chrome.runtime.onConnect.addListener" in background_js
    assert "port.name !== LONG_REQUEST_PORT" in background_js
    assert "port.postMessage({ id, ...response })" in background_js
    assert "AUTO_SCAN_DRAFT_TIMEOUT_MS" in background_js
    assert "AbortController" in background_js
    assert "source.endsWith(\"-auto\")" in background_js
    assert "Phone Copilot controller timed out after" in background_js


def test_whatsapp_web_extension_archive_autonomous_scan_is_safety_gated() -> None:
    content_js = Path("apps/browser_extension/whatsapp_web/content.js").read_text(encoding="utf-8")

    assert "ARCHIVE_SCAN_INTERVAL_MS" in content_js
    assert "ARCHIVE_SCAN_MAX_CHATS" not in content_js
    assert "openArchiveView" in content_js
    assert "archiveListRoot" in content_js
    assert "mainListRoot" in content_js
    assert "archivePageHeading" in content_js
    assert "scoreArchiveRoot" in content_js
    assert "archiveRowCandidates" in content_js
    assert "chatRowsInRoot" in content_js
    assert "archiveUnreadRowsFromBadges" in content_js
    assert "archiveUnreadBadgeNodes" in content_js
    assert "unreadChatRowsInRoot" in content_js
    assert "rootRowsAreActionable" in content_js
    assert "if (actionableRoot) return true" not in content_js
    assert "if (actionableRoot && root !== archiveListRoot()) return true" in content_js
    assert "rowFromUnreadBadge" in content_js
    assert "These chats stay archived" in content_js
    assert "data-testid='archived-chatlist'" in content_js
    assert "data-testid='chat-list'" in content_js
    assert "openInboxView" not in content_js
    assert "if (!archiveListRoot() && mainListRoot()) return true" not in content_js
    assert "trusted_back_click" not in content_js
    assert "trusted_archive_entry_click" in content_js
    assert "openArchivedChatRow" in content_js
    assert "openUnreadChatRow" in content_js
    assert 'setStatus("archive page did not open")' in content_js
    assert "if (archiveListRoot()) return true" in content_js
    assert "isArchiveChatRowLabel" in content_js
    assert "clickChatRow" in content_js
    assert "trustedClickElement" in content_js
    assert "withTimeout(trustedClickElement(row), 2500, \"trusted_click\")" in content_js
    assert '"pc:trusted-click"' in content_js
    assert "KeyboardEvent(\"keydown\"" in content_js
    assert "pointerdown" in content_js
    assert "dblclick" in content_js
    assert "span[title]" in content_js
    assert "[role='gridcell']" in content_js
    assert "not_a_chat_row" in content_js
    assert "archive-refreshed" in content_js
    assert "document.querySelector(\"[aria-label='Chat list']\")" not in content_js
    assert "unreadChatRows" in content_js
    assert "archiveProcessedKeys" in content_js
    assert "scanAllUnreadNow({ autonomous: true })" not in content_js
    assert "scanAllUnreadNow({ autonomous: panel.querySelector(\"[data-pc='mode']\")?.value === \"autonomous\" })" not in content_js
    assert "scanUnreadScope" not in content_js
    assert "scanCurrentOpenThread" not in content_js
    assert "\"current-auto\"" not in content_js
    assert "current open chat needs reply" not in content_js
    assert 'source.endsWith("-auto")' in content_js
    assert "data.auto_send_allowed" in content_js
    assert 'data.automation_decision === "SEND_ALLOWED"' in content_js
    assert "sent by autonomous unread gate" not in content_js
    assert "attemptedThisScan" in content_js
    assert "archiveProcessedKeys.add(key)" in content_js
    assert "result.status === \"drafted\" || result.status === \"sent\"" in content_js
    assert "unread scan drafted" not in content_js
    assert "archive scan drafted" in content_js
    assert "archive scan handled" not in content_js
    assert "hasUnreadCountBadge" in content_js
    assert "^\\d{1,3}$" in content_js
    assert "rowRect.width * 0.62" in content_js
    assert "rootRect.left + rootRect.width * 0.55" in content_js


def test_catbot_chat_is_offline_and_renders_viewer_payload(training_service: PhoneCopilotService, mock_adb) -> None:
    response = training_service.training_catbot_chat(
        CatbotChatRequest(incoming="u coming?", relationship_type="close_friend", intent_type="planning")
    )

    assert response["source"] == "catbot"
    assert response["session_id"].startswith("catbot_")
    assert response["reply"]
    assert response["viewer_messages"] == [
        {"speaker": "other", "text": "u coming?"},
        {"speaker": "me", "text": response["reply"]},
    ]
    assert response["adb_touched"] is False
    assert mock_adb.commands == []


def test_catbot_thumbs_up_saves_catbot_correction(training_service: PhoneCopilotService, mock_adb) -> None:
    chat = training_service.training_catbot_chat(
        CatbotChatRequest(incoming="u coming?", relationship_type="close_friend", intent_type="planning")
    )

    response = training_service.training_catbot_feedback(
        CatbotFeedbackRequest(session_id=chat["session_id"], rating="thumbs_up")
    )

    assert response["status"] == "saved"
    assert response["source"] == "catbot"
    assert response["thumb_rating"] == "thumbs_up"
    assert response["correction"]["source"] == "catbot"
    assert response["vector_rebuild_required"] is True
    assert mock_adb.commands == []


def test_catbot_thumbs_down_does_not_add_positive_correction(training_service: PhoneCopilotService) -> None:
    chat = training_service.training_catbot_chat(
        CatbotChatRequest(incoming="hii", relationship_type="unknown", intent_type="greeting")
    )

    response = training_service.training_catbot_feedback(
        CatbotFeedbackRequest(session_id=chat["session_id"], rating="thumbs_down", corrected_reply="yo")
    )

    assert response["feedback_label"] == "rejected"
    assert response["correction"] is None
    assert all(row.get("source") != "catbot" for row in load_corrections(training_service.settings.ai_reply_training_messages_dir))


def test_catbot_thumbs_down_records_scene_failure_context(training_service: PhoneCopilotService) -> None:
    chat = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="good how are u",
            context=["hi", "yo how u been"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )

    response = training_service.training_catbot_feedback(
        CatbotFeedbackRequest(
            session_id=str(chat["session_id"]),
            rating="thumbs_down",
            corrected_reply="im good wbu",
            notes="testing failure capture",
        )
    )
    episode = training_service.drafting.thread_memory.get_episode(str(chat["thread_id"]))
    feedback_rows = [
        json.loads(line)
        for line in training_service.drafting.thread_memory.feedback_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert response["feedback_label"] == "rejected"
    assert response["session"]["scene_type"] == chat["candidate"]["scene_type"]
    assert episode is not None
    assert episode.turns[-1].feedback == "thumbs_down"
    assert "human_rejected_reply" in episode.failed_reply_patterns
    assert feedback_rows[-1]["scene_type"] == chat["candidate"]["scene_type"]
    assert feedback_rows[-1]["required_reply_move"] == chat["candidate"]["required_reply_move"]
    assert feedback_rows[-1]["candidate_shortlist"]
    assert "human_rejected_reply" in feedback_rows[-1]["failure_tags"]


def test_catbot_route_dry_complaint_does_not_use_stale_chilling(training_service: PhoneCopilotService) -> None:
    context = ["hi", "yo what u saying", "wyd", "just chilling wbu", "in bed", "same icl"]

    response = training_service.training_catbot_chat(
        CatbotChatRequest(incoming="bit dry", context=context, contact_name="Catbot", relationship_type="close_friend")
    )

    candidate = response["candidate"]
    assert response["reply"] not in {"fair just chilling too", "same just chilling"}
    assert candidate["conversation_function"] in {"dry_complaint", "confusion_repair"}
    assert any(term in response["reply"] for term in ("dry", "dead", "repeating", "chilling", "give me a topic", "giving me much", "im trying"))


def test_catbot_route_u_said_that_gets_repeated_callout_repair(training_service: PhoneCopilotService) -> None:
    context = ["wyd", "just chilling wbu", "in bed", "same just chilling"]

    response = training_service.training_catbot_chat(
        CatbotChatRequest(incoming="u said that", context=context, contact_name="Catbot", relationship_type="close_friend")
    )

    candidate = response["candidate"]
    assert response["reply"] != "same just chilling"
    assert candidate["conversation_function"] == "repeated_reply_callout"
    assert candidate["repeated_reply_callout_detected"] is True
    assert any(term in response["reply"] for term in ("repeated", "caught", "bugged", "NPC behaviour", "same thing"))


def test_catbot_route_repeating_same_shit_rejects_stale_candidate(training_service: PhoneCopilotService) -> None:
    context = ["wyd", "just chilling wbu", "in bed", "same just chilling"]

    response = training_service.training_catbot_chat(
        CatbotChatRequest(incoming="mate ur repeating the same shit", context=context, contact_name="Catbot", relationship_type="close_friend")
    )

    candidate = response["candidate"]
    assert response["reply"] != "same just chilling"
    assert candidate["conversation_function"] == "repeated_reply_callout"
    assert candidate["repeated_reply_callout_detected"] is True
    stale_candidates = [item for item in response.get("candidates", []) if item.get("text") == "same just chilling"]
    assert not stale_candidates or stale_candidates[0]["stale_self_state_penalty"] >= 0.7


def test_catbot_route_provider_stale_reply_rejected_for_repeated_callout(training_service: PhoneCopilotService) -> None:
    training_service.settings.ai_reply_enabled = True
    training_service.drafting.ai_reply_enabled = True
    training_service.drafting.draft_provider = "gemini"
    training_service.drafting._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        type("FakeResponse", (), {
            "text": '{"candidates":[{"reply":"same just chilling","reason":"stale"}]}',
            "provider": "gemini",
            "model": "fake",
            "latency_ms": 1,
            "raw_finish_reason": None,
            "error": None,
            "external_api_used": True,
        })(),
        {"provider_configured": True},
    )

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="bro you said the same shit twice",
            context=["wyd", "just chilling wbu", "in bed", "same just chilling"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )

    assert response["reply"] != "same just chilling"
    assert response["candidate"]["conversation_function"] == "repeated_reply_callout"
    stale_candidates = [item for item in response.get("candidates", []) if item.get("text") == "same just chilling"]
    assert not stale_candidates or stale_candidates[0]["final_decision"] == "reject"


def test_catbot_route_selected_candidate_exposes_function_metadata(training_service: PhoneCopilotService) -> None:
    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="ur repeating yourself",
            context=["wyd", "just chilling wbu", "in bed", "same just chilling"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )

    candidate = response["candidate"]
    for key in ("conversation_function", "stale_self_state_penalty", "repeated_reply_callout_detected", "selected_reason"):
        assert key in candidate
    assert "context_fit_score" in candidate
    assert "invalid_for_context" in candidate


def test_catbot_fresh_session_does_not_inherit_previous_context(training_service: PhoneCopilotService) -> None:
    first = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="mate ur repeating the same shit",
            context=["wyd", "just chilling wbu", "in bed", "same just chilling"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )
    second = training_service.training_catbot_chat(
        CatbotChatRequest(incoming="hi", context=[], contact_name="FreshCatbot", relationship_type="close_friend")
    )

    assert first["candidate"]["conversation_function"] == "repeated_reply_callout"
    assert second["candidate"]["conversation_function"] == "greeting_open"
    assert second["reply"] in {"yo what u saying", "heyy what u doing", "what u saying", "yo how u been"}


def test_catbot_route_emotional_reciprocity_sequence_uses_turn_contract(training_service: PhoneCopilotService, monkeypatch) -> None:
    async def fake_generate_with_fallback(**kwargs):
        incoming = str(kwargs.get("context_texts", [""])[-1])
        replies = {
            "hey": "hey / what u been up to",
            "chilling hru": "im good just got back from gym / what u doing",
            "what u been up to baby i missed u": "i missed u too / been thinking about u",
        }
        return (
            type("FakeResponse", (), {
                "text": replies.get(incoming, "i missed u too / been thinking about u"),
                "provider": "gemini",
                "model": "fake-model",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    context: list[str] = []
    first = training_service.training_catbot_chat(
        CatbotChatRequest(incoming="hey", context=context, contact_name="Catbot", relationship_type="romantic_interest")
    )
    context.extend(["hey", first["reply"]])
    second = training_service.training_catbot_chat(
        CatbotChatRequest(incoming="chilling hru", context=context, contact_name="Catbot", relationship_type="romantic_interest")
    )
    context.extend(["chilling hru", second["reply"]])
    mixed = training_service.training_catbot_chat(
        CatbotChatRequest(incoming="what u been up to baby i missed u", context=context, contact_name="Catbot", relationship_type="romantic_interest")
    )

    stale = {"same icl", "fair just chilling too", "same just chilling"}
    assert first["candidate"]["reply_plan_move"] == "affectionate_greeting"
    assert second["candidate"]["reply_plan_shape"] == "answer_status_then_continue"
    assert mixed["candidate"]["reply_plan_move"] == "emotional_reciprocity"
    assert mixed["candidate"]["reply_plan_shape"] == "reciprocate_affection_plus_specific_continuation"
    assert "missed u" in mixed["reply"]
    assert all(response["reply"] not in stale for response in (first, second, mixed))
    assert all(response["fallback_used"] is False for response in (first, second, mixed))


def test_catbot_repeated_affectionate_greeting_uses_no_question_plan(training_service: PhoneCopilotService) -> None:
    context = ["hey", "hey u / what's on your mind baby"]
    contract = training_service._catbot_turn_contract(incoming="hey you", context=context, intent="auto")

    assert contract.move["user_move"] == "affectionate_greeting"
    assert contract.reply_plan.shape == "repeat_greeting_without_question"
    assert contract.reply_plan.required_slots["recent_greeting_shape"] == "greeting_plus_question_hook"
    assert training_service._catbot_ai_reject_reason(
        "hey u / what's on your mind baby",
        incoming="hey you",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) in {"generic_ai_style", "recent_repeat", "repeated_reply_shape"}
    assert training_service._catbot_ai_reject_reason(
        "hey you / there u are",
        incoming="hey you",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""


def test_catbot_repeated_greeting_after_loop_repairs_uses_recent_greeting_plan(training_service: PhoneCopilotService) -> None:
    context = [
        "hi",
        "hey u alright",
        "how are u",
        "im alright / head's a bit fried but calm now",
        "im good u",
        "long but calm icl / just needed food after",
        "ive been chilling wby baby",
        "was working on my side project for a bit / chilling now",
        "u already asked silly",
        "yh fair my bad / i keep throwing questions when my head goes blank",
        "u already asked how my day was / how i am",
        "yh ur right / no more day or how-are-u questions, im moving slow today",
    ]
    contract = training_service._catbot_turn_contract(incoming="hi", context=context, intent="auto")

    assert contract.move["user_move"] == "affectionate_greeting"
    assert contract.move["slots"]["recent_greeting_shape"] == "greeting_plus_question_hook"
    assert contract.reply_plan.shape == "repeat_greeting_without_question"
    assert training_service._catbot_plan_specific_repair_reply(
        contract,
        reject_reason="recent_repeat",
        avoid_replies=context[1::2],
    ) == "hey you / there u are"
    assert training_service._catbot_direct_plan_reply(
        contract,
        incoming="hi",
        context=context,
        recent_bot_replies=context[1::2],
    ) == "hey you / there u are"


def test_catbot_current_activity_after_repaired_opening_uses_fresh_status_detail(training_service: PhoneCopilotService) -> None:
    context = [
        "hi",
        "hey u alright",
        "how are u",
        "im alright / head's a bit fried but calm now",
        "im good u",
        "long but calm icl / just needed food after",
        "ive been chilling wby baby",
        "was working on my side project for a bit / chilling now",
        "u already asked silly",
        "yh fair my bad / i keep throwing questions when my head goes blank",
        "u already asked how my day was / how i am",
        "yh ur right / no more day or how-are-u questions, im moving slow today",
        "hi",
        "hey you / there u are",
        "hru",
        "im good / just sat here letting my head switch off",
    ]
    contract = training_service._catbot_turn_contract(incoming="what u up to", context=context, intent="auto")

    assert contract.move["user_move"] == "reciprocal_question"
    assert contract.reply_plan.shape == "fresh_status_detail_after_recent_status"
    assert training_service._catbot_plan_specific_repair_reply(
        contract,
        reject_reason="recent_repeat",
        avoid_replies=context[1::2],
    ) == "just on my phone waiting for food"
    assert training_service._catbot_direct_plan_reply(
        contract,
        incoming="what u up to",
        context=context,
        recent_bot_replies=context[1::2],
    ) == "just on my phone waiting for food"


def test_catbot_typo_past_activity_followup_after_low_activity_uses_direct_detail(training_service: PhoneCopilotService) -> None:
    context = [
        "hi",
        "hey u alright",
        "how are u",
        "im alright / head's a bit fried but calm now",
        "im good u",
        "just on my phone waiting for food",
        "ive been chilling wby baby",
        "been sorting this project on my laptop",
        "u already asked silly",
        "yh fair my bad / i keep throwing questions when my head goes blank",
        "u already asked how my day was / how i am",
        "yh ur right / no more day or how-are-u questions, im moving slow today",
        "hi",
        "hey you / there u are",
        "hru",
        "im good / just sat here letting my head switch off",
        "what u up to",
        "watching random clips while i switch off",
    ]
    contract = training_service._catbot_turn_contract(incoming="what did u doo", context=context, intent="auto")

    assert contract.move["user_move"] == "activity_question"
    assert contract.reply_plan.shape == "answer_activity_detail_then_continue"
    assert contract.reply_plan.required_slots["activity_scope"] == "low_activity"
    assert training_service._catbot_direct_plan_reply(
        contract,
        incoming="what did u doo",
        context=context,
        recent_bot_replies=context[1::2],
    ) == "watched random clips for a bit / mostly switched off"


def test_catbot_specific_question_family_callout_uses_distinct_loop_plan(training_service: PhoneCopilotService) -> None:
    context = [
        "im good u",
        "just got out the shower now need food icl",
        "ive been chilling wby baby",
        "just scrolling on my phone waiting for food to cook now ngl",
        "u already asked silly",
        "yh fair my bad / i keep throwing questions when my head goes blank",
    ]
    contract = training_service._catbot_turn_contract(
        incoming="u already asked how my day was / how i am",
        context=context,
        intent="auto",
    )

    assert contract.move["user_move"] == "loop_callout"
    assert contract.reply_plan.shape == "acknowledge_specific_question_family_without_repeat"
    assert contract.reply_plan.required_slots["repeated_reset_topic"] == "day_status_question"
    assert training_service._catbot_reply_violates_plan(
        "yh fair my bad / i keep throwing questions when my head goes blank",
        contract.reply_plan,
    ) == "missing_specific_question_family_acknowledgement"
    assert training_service._catbot_ai_reject_reason(
        "yh ur right / i keep defaulting to day and how-are-u questions, i'll stop",
        incoming="u already asked how my day was / how i am",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "yh fair / no more how-are-u questions from me",
        incoming="u already asked how my day was / how i am",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "my bad / no more day questions, im just tired and moving lazy",
        incoming="u already asked how my day was / how i am",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""


def test_catbot_repair_callout_rejects_whats_on_mind_reset(training_service: PhoneCopilotService) -> None:
    context = [
        "omg im so bored",
        "nah im picking film / what's the last thing u watched that actually surprised u",
    ]
    contract = training_service._catbot_turn_contract(
        incoming="why u asking stupid qs for",
        context=context,
        intent="auto",
    )

    assert contract.move["user_move"] == "repair_callout"
    assert contract.move["slots"]["repair_target"] == "question_quality_callout"
    assert contract.reply_plan.shape == "acknowledge_bad_question_then_owner_side_reset"
    assert training_service._catbot_ai_reject_reason(
        "yh my bad i was tryna be funny but it wasnt working was it / what's on ur mind then",
        incoming="why u asking stupid qs for",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "my bad lol / i wanna know what u been up to then",
        incoming="why u asking stupid qs for",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "yh my bad / tell me the weirdest thing that happened to you today",
        incoming="why u asking stupid qs for",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "generic_ai_style"
    assert training_service._catbot_reply_violates_plan(
        "my bad fr / okay tell me what's the funniest thing that happened today",
        contract.reply_plan,
    ) == "ask_question"
    assert training_service._catbot_ai_reject_reason(
        "yh my bad i was tryna be funny and it came out dead",
        incoming="why u asking stupid qs for",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""


def test_catbot_route_repairs_bad_question_callout_provider_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    bad_reply = "what should we talk about then"
    context = [
        "what u up to",
        "just lying here letting my body recover",
        "what did u doo",
        "trained chest and shoulders today",
        "omg im so bored",
        "nah im picking film / what's the last thing u watched that actually surprised u",
    ]
    contract = training_service._catbot_turn_contract(
        incoming="why u asking stupid qs for",
        context=context,
        intent="auto",
    )
    assert contract.move["user_move"] == "repair_callout"
    assert contract.move["slots"]["repair_target"] == "question_quality_callout"
    assert contract.reply_plan.shape == "acknowledge_bad_question_then_owner_side_reset"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="why u asking stupid qs for",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "generic_ai_style"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="why u asking stupid qs for",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "yh my bad that was a stupid question / my brain was moving lazy"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_plan_direct"] is True
    assert response["candidate"]["catbot_ai_repair_accepted"] is False
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_provider_same_icl_rejected_for_care_checkin(training_service: PhoneCopilotService) -> None:
    training_service.settings.ai_reply_enabled = True
    training_service.drafting.ai_reply_enabled = True
    training_service.drafting.draft_provider = "gemini"
    training_service.drafting._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        type("FakeResponse", (), {
            "text": '{"candidates":[{"reply":"same icl","reason":"stale"}]}',
            "provider": "gemini",
            "model": "fake",
            "latency_ms": 1,
            "raw_finish_reason": None,
            "error": None,
            "external_api_used": True,
        })(),
        {"provider_configured": True},
    )

    response = training_service.training_catbot_chat(
        CatbotChatRequest(incoming="its ok are u ok ml u seem off", context=[], contact_name="Catbot", relationship_type="close_friend")
    )

    assert response["candidate"]["conversation_function"] == "care_checkin"
    assert response["reply"] != "same icl"
    stale_candidates = [item for item in response.get("candidates", []) if item.get("text") == "same icl"]
    assert not stale_candidates or stale_candidates[0]["final_decision"] == "reject"


def test_catbot_route_scene_topic_provided_engages_topic(training_service: PhoneCopilotService) -> None:
    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="hm cars",
            context=["im bored", "fine then give me a topic"],
            contact_name="Catbot",
        )
    )

    candidate = response["candidate"]
    assert candidate["scene_type"] == "topic_given"
    assert candidate["required_reply_move"] == "continue_given_topic"
    assert "cars" in response["reply"]
    assert "give me a topic" not in response["reply"]


def test_catbot_route_scene_topic_ignored_callout_repairs(training_service: PhoneCopilotService) -> None:
    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="i just did bro...",
            context=["hm cars", "im trying icl give me a topic"],
            contact_name="Catbot",
        )
    )

    candidate = response["candidate"]
    assert candidate["scene_type"] in {"topic_ignored_callout", "missed_context_callout"}
    assert candidate["required_reply_move"] == "acknowledge_and_repair"
    assert "cars" in response["reply"]
    assert "u ain't giving me much" not in response["reply"]


def test_catbot_route_scene_repeated_stale_reply_repairs(training_service: PhoneCopilotService) -> None:
    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="mate ur repeating the same shit",
            context=["wyd", "same just chilling", "in bed", "same just chilling"],
            contact_name="Catbot",
        )
    )

    candidate = response["candidate"]
    assert candidate["scene_type"] == "repeated_reply_callout"
    assert response["reply"] not in {"same just chilling", "same icl"}
    assert any(term in response["reply"] for term in ("repeated", "caught me", "bugged", "NPC behaviour"))


def test_catbot_route_scene_direct_question_answers(training_service: PhoneCopilotService) -> None:
    response = training_service.training_catbot_chat(
        CatbotChatRequest(incoming="r u mad", context=[], contact_name="Catbot")
    )

    assert response["candidate"]["scene_type"] == "direct_question"
    assert response["candidate"]["required_reply_move"] == "answer_directly"
    assert response["reply"] in {"nah im calm", "nah im not mad", "nah why"}


def test_catbot_route_ai_contract_affection_and_explicit_contexts(training_service: PhoneCopilotService, monkeypatch) -> None:
    replies = iter([
        "i missed u too / been thinking about u",
        "come here then\nneed ur lips on my neck\nmy hands on ur hips\nteasing u slow",
    ])

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": next(replies),
                "provider": "gemini",
                "model": "fake-model",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    affection = training_service.training_catbot_chat(
        CatbotChatRequest(incoming="i missed u baby", context=[], contact_name="Catbot", relationship_type="romantic_interest")
    )
    explicit = training_service.training_catbot_chat(
        CatbotChatRequest(incoming="im horny", context=[], contact_name="Catbot", relationship_type="romantic_interest")
    )

    assert affection["candidate"]["reply_plan_move"] == "emotional_reciprocity"
    assert affection["candidate"]["reply_plan_shape"] == "reciprocate_affection_plus_specific_continuation"
    assert affection["reply"] not in {"same just chilling", "same icl", "fair just chilling too"}
    assert explicit["candidate"]["reply_plan_move"] == "romantic_escalation"
    assert explicit["candidate"]["reply_plan_shape"] == "multi_bubble_adult_escalation"
    assert explicit["candidate"]["auto_send_allowed"] is False


def test_catbot_route_close_friend_affection_ack_is_not_rejected(training_service: PhoneCopilotService) -> None:
    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="missed you icl",
            context=["hi", "yo how u been", "chilling wby", "nothing much just chilling"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )

    assert response["candidate"]["scene_type"] == "emotional_affection"
    assert response["candidate"]["conversation_job"] == "answer_affection"
    assert response["candidate"]["final_decision"] != "reject"
    assert any(term in response["reply"] for term in ("sweet", "bless", "appreciate", "missed"))
    assert response["reply"] not in {"same just chilling", "same icl", "fair just chilling too"}


def test_catbot_route_scene_metadata_exposed(training_service: PhoneCopilotService) -> None:
    response = training_service.training_catbot_chat(
        CatbotChatRequest(incoming="hm cars", context=["im bored", "fine then give me a topic"], contact_name="Catbot")
    )

    candidate = response["candidate"]
    for key in (
        "scene_type",
        "required_reply_move",
        "forbidden_reply_moves",
        "scene_fit_score",
        "required_move_satisfied",
        "forbidden_move_violated",
        "contextual_specificity_score",
        "canned_reply_penalty",
        "unresolved_user_points",
        "unresolved_point_addressed",
        "fallback_used",
    ):
        assert key in candidate


def test_catbot_route_unresolved_affection_contract_rejects_stale_reply(training_service: PhoneCopilotService) -> None:
    context = [
        "hi",
        "yo what u saying",
        "hru",
        "im good wbu",
        "im chilling",
        "same just chilling",
    ]
    contract = training_service._catbot_turn_contract(
        incoming="i missed u",
        context=context,
        intent="auto",
    )

    assert contract.move["user_move"] == "emotional_reciprocity"
    assert contract.reply_plan.shape == "reciprocate_affection_plus_specific_continuation"
    assert training_service._catbot_ai_reject_reason(
        "fair just chilling too",
        incoming="i missed u",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) in {"not_carrying_conversation", "generic_ai_style"}
    assert training_service._catbot_ai_reject_reason(
        "i missed u too / been thinking about u",
        incoming="i missed u",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""


def test_catbot_route_miss_me_question_uses_emotional_reciprocity_policy(training_service: PhoneCopilotService, monkeypatch) -> None:
    async def failing_generate_with_fallback(**kwargs):
        raise AssertionError("miss-me question should be handled by direct conversation policy")

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", failing_generate_with_fallback)
    context = [
        "about to sleep",
        "go sleep then / im half asleep too icl",
        "thinking about u",
        "that's good to hear then",
    ]
    contract = training_service._catbot_turn_contract(
        incoming="u miss me?",
        context=context,
        intent="auto",
    )

    assert contract.move["user_move"] == "emotional_reciprocity"
    assert contract.reply_plan.shape == "reciprocate_affection_plus_specific_continuation"
    stale_reason = training_service._catbot_ai_reject_reason(
        "that's good to hear then",
        incoming="u miss me?",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    )
    assert stale_reason in {"not_carrying_conversation", "recent_repeat"}
    assert training_service._catbot_ai_reject_reason(
        "yeah i do / been thinking about u all day",
        incoming="u miss me?",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="u miss me?",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
        )
    )

    assert response["candidate"]["reply_plan_move"] == "emotional_reciprocity"
    assert response["candidate"]["reply_plan_shape"] == "reciprocate_affection_plus_specific_continuation"
    assert response["fallback_used"] is False
    assert any(term in response["reply"] for term in ("yeah i do", "yh i do", "course i do", "obviously"))


def test_catbot_route_thinking_about_you_uses_emotional_reciprocity_policy(training_service: PhoneCopilotService, monkeypatch) -> None:
    async def failing_generate_with_fallback(**kwargs):
        raise AssertionError("thinking-about-you affection should be handled by direct conversation policy")

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", failing_generate_with_fallback)
    context = [
        "im in bed",
        "same / im in bed letting my head switch off too",
        "im in bed rn",
        "stay there then / im just in bed letting the day go quiet",
        "just chilling wbu",
        "got up for food now / less dead than before",
        "im good wbu",
        "was working on my side project for a bit / chilling now",
        "im alright wby",
        "just letting my brain switch off now",
        "been busy today",
        "same icl / my head feels fried too",
        "just got back from gym",
        "gym leaves u finished icl / i need food after mine",
        "just woke up lol",
        "waking up leaves u half asleep icl / i need a minute too",
        "about to sleep",
        "go sleep then / im half asleep too icl",
    ]
    contract = training_service._catbot_turn_contract(
        incoming="thinking about u",
        context=context,
        intent="auto",
    )

    assert contract.move["user_move"] == "emotional_reciprocity"
    assert contract.reply_plan.shape == "reciprocate_affection_plus_specific_continuation"
    assert contract.reply_plan.required_slots["affection_signal"] == "thinking_of_you"
    assert training_service._catbot_ai_reject_reason(
        "same / been thinking about u too icl",
        incoming="thinking about u",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="thinking about u",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
        )
    )

    assert response["candidate"]["reply_plan_move"] == "emotional_reciprocity"
    assert response["candidate"]["reply_plan_shape"] == "reciprocate_affection_plus_specific_continuation"
    assert response["fallback_used"] is False
    assert any(term in response["reply"] for term in ("thinking about u", "wish u were here", "wanted u here"))


def test_catbot_route_provider_stale_reply_rejected_for_missed_affection(training_service: PhoneCopilotService, monkeypatch) -> None:
    replies = iter(["fair just chilling too", "i missed u too / been thinking about u"])

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": next(replies),
                "provider": "gemini",
                "model": "fake-model",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="i missed u",
            context=["hi", "yo what u saying"],
            contact_name="Catbot",
            relationship_type="romantic_interest",
        )
    )

    assert response["reply"] == "i missed u too / been thinking about u all day"
    assert response["reply"] != "fair just chilling too"
    assert response["candidate"]["reply_plan_move"] == "emotional_reciprocity"
    assert response["candidate"]["catbot_ai_plan_direct"] is True
    assert response["fallback_used"] is False


def test_catbot_route_provider_failure_returns_502_without_fallback(training_service: PhoneCopilotService, monkeypatch, provider_retry_path) -> None:
    calls = []

    async def fail_generate_with_fallback(**kwargs):
        calls.append(kwargs)
        return (
            type("FakeResponse", (), {
                "text": "",
                "provider": "fallback",
                "model": None,
                "latency_ms": None,
                "raw_finish_reason": None,
                "error": "provider exploded",
                "external_api_used": False,
            })(),
            {"provider_configured": False, "manual_review_fallback": True},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fail_generate_with_fallback)

    with pytest.raises(HTTPException) as exc_info:
        training_service.training_catbot_chat(
            CatbotChatRequest(incoming="i missed u", context=[], contact_name="Catbot", relationship_type="romantic_interest")
        )

    assert exc_info.value.status_code == 502
    assert "Catbot AI provider did not return a usable reply" in str(exc_info.value.detail)
    assert len(calls) == 5


def test_catbot_ai_route_accepts_missed_you_typo(monkeypatch, training_service: PhoneCopilotService) -> None:
    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": "i missed u too baby",
                "provider": "gemini",
                "model": "fake-model",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="i missed yoy",
            context=["hi", "hey you alright", "im so good", "thats good to hear what makes u so good then"],
            contact_name="Catbot",
            relationship_type="romantic_interest",
        )
    )

    assert response["reply"] == "i missed u too / been thinking about u all day"
    assert response["retrieval_backend"] == "catbot_ai_romantic"
    assert response["route_error_handled"] is False
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_plan_direct"] is True
    assert response["candidate"]["catbot_ai_reject_reason"] == ""


def test_catbot_ai_route_continues_adult_followup(monkeypatch, training_service: PhoneCopilotService) -> None:
    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": "id grind my cock against u slow while my mouth stays on ur neck",
                "provider": "gemini",
                "model": "fake-model",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="and whats that",
            context=[
                "im craving ur huge cock",
                "ngl i've been thinking about u too",
                "really, how...",
                "really badly like all day",
                "about what exactly",
                "just like how i wanna kiss u and stuff",
                "that all?",
                "nah and what else i'd do to u when i next see u",
            ],
            contact_name="Catbot",
            relationship_type="romantic_interest",
        )
    )

    assert response["reply"] == "id grind my cock against u slow while my mouth stays on ur neck"
    assert response["retrieval_backend"] == "catbot_ai_romantic"
    assert response["route_error_handled"] is False
    assert response["fallback_used"] is False
    assert response["candidate"]["provider"] == "gemini"
    assert response["candidate"].get("catbot_ai_plan_direct") is not True
    assert response["candidate"]["catbot_ai_reject_reason"] == ""


def test_catbot_ai_route_requires_explicit_detail_for_explicit_adult_prompt(monkeypatch, training_service: PhoneCopilotService) -> None:
    assert training_service._catbot_ai_reject_reason(
        "good cause i crave u back",
        incoming="im craving ur fat cock",
        context=[],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "like i just want u on me rn",
        incoming="how so",
        context=["im craving ur fat cock", "good cause i crave u back"],
        recent_bot_replies=[],
    ) == "missing_sensory_texture"
    assert training_service._catbot_ai_reject_reason(
        "i need u too",
        incoming="i need you",
        context=[],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "i want u too",
        incoming="i want u",
        context=[],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "i need u right here now",
        incoming="i need you",
        context=[],
        recent_bot_replies=[],
    ) == "generic_ai_style"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": "id keep my hand on ur waist while my cock grinds against u slow",
                "provider": "gemini",
                "model": "fake-model",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="how so",
            context=["im craving ur fat cock", "good cause i crave u back"],
            contact_name="Catbot",
            relationship_type="romantic_interest",
        )
    )

    assert response["reply"] == "id keep my hand on ur waist while my cock grinds against u slow"
    assert response["retrieval_backend"] == "catbot_ai_romantic"
    assert response["fallback_used"] is False
    assert response["candidate"]["provider"] == "gemini"
    assert response["candidate"].get("catbot_ai_plan_direct") is not True
    assert response["candidate"]["catbot_ai_reject_reason"] == ""


def test_catbot_ai_route_renders_explicit_followup_plan(training_service: PhoneCopilotService) -> None:
    context = [
        "yo",
        "hey u / what u thinking about rn",
        "i want u",
        "i want my hands running over ur wet thighs",
        "tell me then",
        "my lips on ur neck",
        "and what would u do",
        "then my tongue tracing that line down to ur hips",
        "hey you",
        "hey you too / what's on your mind then",
        "im craving ur fat cock",
        "then my hard throbbing against ur wetness",
    ]

    contract = training_service._catbot_turn_contract(
        incoming="how so",
        context=context,
        intent="auto",
    )
    assert contract.move["user_move"] == "romantic_escalation"
    assert contract.reply_plan.shape == "specific_escalation_detail"
    assert contract.reply_plan.required_slots["escalation_source"] == "explicit_adult_followup"

    prompt = training_service._catbot_ai_prompt_messages(
        incoming="how so",
        context=context,
        intent="auto",
        contact_name="Catbot",
        contract=contract,
    )[-1].content

    assert "They directly said they want/need/crave you" not in prompt
    assert "fresh graphic sensory body/action detail" in prompt
    assert "cock hard against them" in prompt
    assert training_service._catbot_ai_reject_reason(
        "skin warm on urs while my cock presses hard into u",
        incoming="how so",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""


def test_catbot_direct_adult_request_uses_multi_bubble_plan(training_service: PhoneCopilotService) -> None:
    contract = training_service._catbot_turn_contract(
        incoming="im horny",
        context=["why u asking stupid qs for", "forget that anyway"],
        intent="auto",
    )

    assert contract.move["user_move"] == "romantic_escalation"
    assert contract.reply_plan.shape == "multi_bubble_adult_escalation"
    assert contract.reply_plan.max_bubbles == 5
    assert training_service._catbot_reply_violates_plan(
        "i crave ur lips on my neck",
        contract.reply_plan,
    ) == "too_few_adult_bubbles"
    assert training_service._catbot_reply_violates_plan(
        "come here then / cock hard against u / grinding slow so u feel it / my mouth by ur ear",
        contract.reply_plan,
    ) == ""
    assert training_service._catbot_reply_violates_plan(
        "come here then / kiss u deep / hand on ur thigh / making u wet",
        contract.reply_plan,
    ) == "violates_plan_forbidden_pattern:kiss u deep"


def test_catbot_adult_detail_requires_sensory_texture(training_service: PhoneCopilotService) -> None:
    contract = training_service._catbot_turn_contract(
        incoming="what else",
        context=[
            "im horny",
            "come here then / keep u pulled in close / my lips on ur neck slow / hands tracing ur waist",
        ],
        intent="auto",
    )

    assert contract.reply_plan.shape == "specific_escalation_detail"
    assert detect_sensory_texture("pressing my chest against u while i keep u close") is True
    assert detect_sensory_texture("kiss u deep") is False
    assert training_service._catbot_reply_violates_plan("kiss u deep", contract.reply_plan) == "violates_plan_forbidden_pattern:kiss u deep"
    assert training_service._catbot_ai_reject_reason(
        "hand on ur thigh making u wet",
        incoming="what else",
        context=[
            "im horny",
            "come here then / keep u pulled in close / my lips on ur neck slow / hands tracing ur waist",
        ],
        recent_bot_replies=[],
        contract=contract,
    ) == "generic_ai_style"
    soft_kiss_reject = training_service._catbot_ai_reject_reason(
        "id give u a deep wet kiss while holding u close",
        incoming="what else",
        context=[
            "im horny",
            "come here then / keep u pulled in close / my lips on ur neck slow / hands tracing ur waist",
        ],
        recent_bot_replies=[],
        contract=contract,
    )
    assert soft_kiss_reject in {"generic_ai_style", "missed_adult_mode"}
    assert training_service._catbot_ai_reject_reason(
        "my hand teasing lower while i watch u get wet for me",
        incoming="what else",
        context=[
            "im horny",
            "come here then / keep u pulled in close / my lips on ur neck slow / hands tracing ur waist",
        ],
        recent_bot_replies=[],
        contract=contract,
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "yeah mhmmm / let me feel that whole body / my chest pressed to ur back / slow enough to make u shiver / every breath on ur skin",
        incoming="mhmmm",
        context=[
            "need ur lips on my neck",
            "my breath right by ur ear / making ur skin shiver / lowering my hips / grinding slow against u / u can feel it all",
            "what else",
            "my mouth by ur ear / fingers circling slow / feeling u get wetter / keeping u close",
        ],
        recent_bot_replies=[],
    ) == "missed_adult_mode"


def test_catbot_ai_route_requires_adult_detail_for_direct_desire(monkeypatch, training_service: PhoneCopilotService) -> None:
    provider_calls = 0

    async def fake_generate_with_fallback(**kwargs):
        nonlocal provider_calls
        provider_calls += 1
        return (
            type("FakeResponse", (), {
                "text": "come here then\ncock hard against u\ngrinding slow so u feel it\nmy mouth by ur ear",
                "provider": "gemini",
                "model": "fake-model",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="i need you",
            context=[],
            contact_name="Catbot",
            relationship_type="romantic_interest",
        )
    )

    assert response["reply"] == "come here then / cock hard against u / grinding slow so u feel it / my mouth by ur ear"
    assert len(response["reply_sequence"]) == 4
    assert response["retrieval_backend"] == "catbot_ai_romantic"
    assert response["fallback_used"] is False
    assert provider_calls == 1
    assert response["candidate"]["provider"] == "gemini"
    assert response["candidate"].get("catbot_ai_plan_direct") is not True
    assert response["candidate"]["reply_plan_shape"] == "multi_bubble_adult_escalation"
    assert response["candidate"]["catbot_ai_reject_reason"] == ""


def test_catbot_route_desire_question_uses_romantic_escalation_plan(monkeypatch, training_service: PhoneCopilotService) -> None:
    provider_calls = 0

    async def provider_fails_then_plan_rescues(**kwargs):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("provider unavailable")

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", provider_fails_then_plan_rescues)
    context = [
        "about to sleep",
        "go sleep then / im half asleep too icl",
        "thinking about u",
        "oh really now / what u thinking about",
        "u miss me?",
        "yeah i do / been thinking about u all day",
    ]
    contract = training_service._catbot_turn_contract(
        incoming="do u want me?",
        context=context,
        intent="auto",
    )

    assert contract.move["user_move"] == "romantic_escalation"
    assert contract.reply_plan.shape == "multi_bubble_adult_escalation"
    assert contract.reply_plan.required_slots["escalation_source"] == "direct_adult_request"

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="do u want me?",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert provider_calls >= 1
    assert response["fallback_used"] is False
    assert response["candidate"]["provider"] == "plan_ranker"
    assert response["candidate"]["fallback_used"] is False
    assert response["candidate"]["manual_review_fallback"] is False
    assert response["candidate"]["reply_plan_move"] == "romantic_escalation"
    assert response["candidate"]["reply_plan_shape"] == "multi_bubble_adult_escalation"
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""
    assert training_service._catbot_ai_reject_reason(
        response["reply"],
        incoming="do u want me?",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""


def test_catbot_ai_route_retries_explicit_body_request_with_detail(monkeypatch, training_service: PhoneCopilotService, provider_retry_path) -> None:
    replies = iter([
        "you know exactly what i'd do with it then don't u",
        "come here then\ncock hard against u\ngrinding slow so u feel it\nmy mouth by ur ear",
    ])

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": next(replies),
                "provider": "gemini",
                "model": "fake-model",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="im craving ur fat cock",
            context=["hey you", "hey u"],
            contact_name="Catbot",
            relationship_type="romantic_interest",
        )
    )

    assert response["reply"] == "come here then / cock hard against u / grinding slow so u feel it / my mouth by ur ear"
    assert response["candidate"]["reply_plan_shape"] == "multi_bubble_adult_escalation"
    assert response["candidate"]["catbot_ai_reject_reason"] == ""


def test_catbot_ai_retry_keeps_adult_acknowledgement_on_followup_instruction(monkeypatch, training_service: PhoneCopilotService) -> None:
    replies = iter([
        "i want u too",
        "my chest against urs while my cock presses hard into u",
    ])
    provider_calls = 0
    prompts: list[str] = []

    async def fake_generate_with_fallback(**kwargs):
        nonlocal provider_calls
        provider_calls += 1
        messages = kwargs.get("messages") or []
        prompts.append("\n".join(str(getattr(message, "content", "")) for message in messages))
        return (
            type("FakeResponse", (), {
                "text": next(replies),
                "provider": "gemini",
                "model": "fake-model",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="mmm",
            context=[
                "im craving ur fat cock",
                "come here then / cock hard against u / grinding slow so u feel it / my mouth by ur ear",
                "what else",
                "pressing my hips into u while my hands stay on ur thighs",
            ],
            contact_name="Catbot",
            relationship_type="romantic_interest",
        )
    )

    assert response["reply"] == "skin warm against urs while i make u throb for me"
    assert provider_calls == 1
    assert response["candidate"]["catbot_ai_retry_accepted"] is False
    assert response["candidate"].get("catbot_ai_plan_direct") is not True
    assert response["candidate"]["catbot_ai_plan_repair_accepted"] is True
    assert prompts
    assert "Adult story continuity" in prompts[-1]


def test_catbot_ai_route_preserves_multiple_message_bubbles(monkeypatch, training_service: PhoneCopilotService, provider_retry_path) -> None:
    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": "come here then\ncock hard against u\ngrinding slow so u feel it\nmy mouth by ur ear",
                "provider": "gemini",
                "model": "fake-model",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="i need you",
            context=[],
            contact_name="Catbot",
            relationship_type="romantic_interest",
        )
    )

    assert response["reply"] == "come here then / cock hard against u / grinding slow so u feel it / my mouth by ur ear"
    assert response["reply_sequence"] == ["come here then", "cock hard against u", "grinding slow so u feel it", "my mouth by ur ear"]
    assert response["candidate"]["sequence"] == ["come here then", "cock hard against u", "grinding slow so u feel it", "my mouth by ur ear"]
    assert response["viewer_messages"] == [
        {"speaker": "other", "text": "i need you"},
        {"speaker": "me", "text": "come here then"},
        {"speaker": "me", "text": "cock hard against u"},
        {"speaker": "me", "text": "grinding slow so u feel it"},
        {"speaker": "me", "text": "my mouth by ur ear"},
    ]


def test_catbot_ai_repair_replaces_rejected_sequence(training_service: PhoneCopilotService) -> None:
    training_service._catbot_ai_romantic_reply = lambda **kwargs: (  # type: ignore[method-assign]
        "",
        {
            "text": "how much u wanna know then",
            "sequence": ["how much u wanna know then"],
            "candidate_type": "catbot_ai_romantic",
            "catbot_ai_reject_reason": "missed_adult_mode",
            "provider_error": None,
        },
    )
    training_service._catbot_ai_repair_reply = lambda **kwargs: "enough that i want my cock against u rn"  # type: ignore[method-assign]

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="how much",
            context=["i missed you", "i missed u too then"],
            contact_name="Catbot",
            relationship_type="romantic_interest",
        )
    )

    assert response["reply"] == "enough that i want my cock against u rn"
    assert response["reply_sequence"] == ["enough that i want my cock against u rn"]
    assert response["candidate"]["sequence"] == ["enough that i want my cock against u rn"]
    assert response["candidate"]["catbot_ai_reject_reason"] == ""
    assert response["candidate"]["catbot_ai_repaired"] is True


def test_catbot_ai_retry_replaces_rejected_sequence(monkeypatch, training_service: PhoneCopilotService, provider_retry_path) -> None:
    replies = iter(["a lot more than u think", "enough that i want my cock against u rn"])

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": next(replies),
                "provider": "gemini",
                "model": "fake-model",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="how much",
            context=["i missed you", "i missed u too"],
            contact_name="Catbot",
            relationship_type="romantic_interest",
        )
    )

    assert response["reply"] == "enough that i want my cock against u rn"
    assert response["reply_sequence"] == ["enough that i want my cock against u rn"]
    assert response["candidate"]["sequence"] == ["enough that i want my cock against u rn"]
    assert response["candidate"]["catbot_ai_reject_reason"] == ""


def test_catbot_ai_route_repairs_boredom_with_concrete_thread(training_service: PhoneCopilotService) -> None:
    assert training_service._catbot_ai_reject_reason(
        "what kinda bored we talking",
        incoming="im bored",
        context=["hello", "hey u still there or u gone quiet on me"],
        recent_bot_replies=[],
    ) == "generic_ai_style"

    training_service._catbot_ai_romantic_reply = lambda **kwargs: (  # type: ignore[method-assign]
        "",
        {
            "text": "what kinda bored we talking",
            "sequence": ["what kinda bored we talking"],
            "candidate_type": "catbot_ai_romantic",
            "catbot_ai_reject_reason": "not_carrying_conversation",
            "provider_error": None,
        },
    )
    training_service._catbot_ai_repair_reply = lambda **kwargs: "nah im giving u a random question then what was the weirdest part of ur day"  # type: ignore[method-assign]

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="im bored",
            context=["hello", "hey u still there or u gone quiet on me"],
            contact_name="Catbot",
            relationship_type="romantic_interest",
        )
    )

    assert response["reply"] == "nah im giving u a random question then what was the weirdest part of ur day"
    assert response["candidate"]["catbot_ai_reject_reason"] == ""
    assert response["candidate"]["catbot_ai_repaired"] is True


def test_catbot_route_plan_repairs_boredom_provider_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    bad_reply = "what kinda bored we talking"
    context = [
        "hi",
        "hey / what u up to",
        "hru",
        "im good just watching some football now icl",
    ]
    contract = training_service._catbot_turn_contract(incoming="omg im so bored", context=context, intent="auto")
    assert contract.reply_plan.shape == "owner_picks_topic_plus_specific_question"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="omg im so bored",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "generic_ai_style"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="omg im so bored",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "nah im picking film / what's the last thing u watched that actually surprised u"
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_talk_then_carry_request_provider_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    bad_reply = "same"
    context = [
        "entertain me",
        "nah im picking dream then / what was the weirdest dream u had recently",
        "u tell me",
        "i had one where i was stuck in a lift",
        "u pick",
        "nah im picking gym then / what's the most u ever deadlifted",
    ]
    contract = training_service._catbot_turn_contract(incoming="talk then", context=context, intent="auto")
    assert contract.move["user_move"] == "carry_conversation_request"
    assert contract.reply_plan.shape == "owner_picks_topic_plus_specific_question"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="talk then",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "not_carrying_conversation"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="talk then",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "nah im picking film / what's the last thing u watched that actually surprised u"
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""

    idk_contract = training_service._catbot_turn_contract(incoming="idk talk", context=context, intent="auto")
    assert idk_contract.move["user_move"] == "carry_conversation_request"
    assert idk_contract.reply_plan.shape == "owner_picks_topic_plus_specific_question"


def test_catbot_route_repairs_idk_talk_avoids_repeated_topic(training_service: PhoneCopilotService) -> None:
    context = [
        "im bored",
        "nah im picking the topic this time / whats the weirdest part of your day been",
        "entertain me",
        "nah im picking something else / what was the last dream u remember",
        "u tell me",
        "last one i remember was proper random icl",
        "u pick",
        "nah im picking film this time / what was the last thing u watched",
        "talk then",
        "nah im picking film / what's the last thing u watched that actually surprised u",
    ]
    contract = training_service._catbot_turn_contract(incoming="idk talk", context=context, intent="auto")
    assert contract.move["user_move"] == "carry_conversation_request"
    assert contract.reply_plan.shape == "owner_picks_topic_plus_specific_question"
    assert contract.reply_plan.required_slots["repeated_reset_topic"] == "film_prompt"

    repaired = training_service._catbot_plan_specific_repair_reply(
        contract,
        reject_reason="recent_repeat",
        avoid_replies=context[1::2],
    )
    assert repaired == "nah im picking food / what was the best thing u ate this week"
    assert training_service._catbot_ai_reject_reason(
        repaired,
        incoming="idk talk",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""


def test_catbot_route_repairs_previous_comment_wdym_provider_failure(monkeypatch, training_service: PhoneCopilotService, provider_path) -> None:
    bad_reply = "i mean just saying nothing really bro now"
    context = [
        "oh fairs",
        "yh fair / i was running out of things to say icl",
    ]
    contract = training_service._catbot_turn_contract(incoming="wdym", context=context, intent="auto")
    assert contract.move["user_move"] == "unclassified"
    assert contract.reply_plan.required_slots["unclassified_context"] == "previous_comment_clarification"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="wdym",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "generic_ai_style"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="wdym",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "i mean i got stuck and started forcing random questions"
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""

    intense_contract = training_service._catbot_turn_contract(
        incoming="wdym",
        context=["oh fairs", "yh fair / that was intense icl"],
        intent="auto",
    )
    assert intense_contract.reply_plan.required_slots["unclassified_context"] == "previous_comment_clarification"
    assert intense_contract.reply_plan.required_slots["clarification_topic"] == "intense_exchange"
    assert training_service._catbot_plan_specific_repair_reply(intense_contract, reject_reason="too_dry") == "i mean that whole back and forth was a lot icl"


def test_catbot_route_repairs_repeated_previous_comment_clarification(monkeypatch, training_service: PhoneCopilotService, provider_path) -> None:
    repeated_clarification = "i mean i got stuck and started forcing random questions"
    context = [
        "thats dry mate",
        "yh u right my bad / i keep being too polite and it comes out boring",
        "be more specific",
        "yh fair my bad / i keep being too general",
        "same",
        "yh same / like i gotta just say what i mean",
        "nice",
        "yh nice one / like we both need to just say it",
        "oh fairs",
        "yh fairs / like we both know what we want from this",
        "wdym",
        repeated_clarification,
    ]
    contract = training_service._catbot_turn_contract(incoming="what do u mean", context=context, intent="auto")
    assert contract.move["user_move"] == "unclassified"
    assert contract.reply_plan.required_slots["unclassified_context"] == "previous_comment_clarification"
    assert training_service._catbot_ai_reject_reason(
        repeated_clarification,
        incoming="what do u mean",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "recent_repeat"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": repeated_clarification,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="what do u mean",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "i mean i was looping and trying to fill silence instead of actually talking"
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["catbot_ai_final_repair_reason"] == "recent_repeat"
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_chill_week_wdym_too_dry_provider_reply(monkeypatch, training_service: PhoneCopilotService, provider_path) -> None:
    context = [
        "thats dry mate",
        "yeah my bad that was a really dry one / what did you get up to this week then",
        "be more specific",
        "my bad yeah i was being way too vague / what was the best thing that happened this week then",
        "same",
        "yeah same / so glad it's the weekend now tho icl",
        "nice",
        "yh nice / so ready to just chill now after all that",
        "oh fairs",
        "yh fair enough / defo need a good chill after this week",
    ]
    contract = training_service._catbot_turn_contract(incoming="wdym", context=context, intent="auto")
    assert contract.reply_plan.required_slots["unclassified_context"] == "previous_comment_clarification"
    assert contract.reply_plan.required_slots["clarification_activity_topic"] == "chill_after_week"
    dry_reply = "i mean i was talking about what i was doing"
    assert training_service._catbot_ai_reject_reason(
        dry_reply,
        incoming="wdym",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "too_dry"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": dry_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="wdym",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "i mean after this week my head just needs to switch off and chill"
    assert response["candidate"]["provider"] == "plan_ranker"
    assert response["candidate"]["catbot_ai_final_repair_reason"] == "too_dry"
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_conversation_function_scores_common_jobs(training_service: PhoneCopilotService) -> None:
    dry = predict_conversation_function(incoming="thats dry mate", context=["wyd", "same just chilling"])
    specific = predict_conversation_function(incoming="be more specific", context=["what do i say then", "what do i say then"])
    why = predict_conversation_function(incoming="yeah why", context=["im good", "cos i finally got to just chill for a bit"])
    hypothetical = predict_conversation_function(
        incoming="what would u do",
        context=["some guy came in my living room", "nah thats mad / did he just run out after"],
    )

    assert dry.function == "quality_complaint"
    assert specific.function == "specificity_request"
    assert why.function == "reason_followup"
    assert hypothetical.function == "story_hypothetical"

    dry_contract = training_service._catbot_turn_contract(incoming="thats dry mate", context=["wyd", "same just chilling"], intent="auto")
    assert dry_contract.conversation_function.function == "quality_complaint"
    assert dry_contract.move["user_move"] == "repair_callout"


def test_catbot_conversation_function_reason_followup_rejects_ask_back(training_service: PhoneCopilotService) -> None:
    context = [
        "lol im chilling",
        "lool same / im just in bed letting my head switch off after work icl",
    ]
    contract = training_service._catbot_turn_contract(incoming="yeah why", context=context, intent="auto")

    assert contract.conversation_function.function == "reason_followup"
    assert contract.reply_plan.required_slots["unclassified_context"] == "reason_for_previous_comment"
    assert contract.reply_plan.required_slots["reason_topic"] == "work_switch_off"
    assert "asking why" in training_service._catbot_conversation_move_instruction(contract.move)
    assert training_service._catbot_ai_reject_reason(
        "just chilling in my bed lol / why u asking",
        incoming="yeah why",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "not_carrying_conversation"
    assert training_service._catbot_ai_reject_reason(
        "because after work my head just needed to switch off",
        incoming="yeah why",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""

    deeper_context = ["why", "cos i was being dumb and overthinking it"]
    deeper_contract = training_service._catbot_turn_contract(incoming="why though", context=deeper_context, intent="auto")
    assert deeper_contract.conversation_function.function == "reason_followup"
    assert deeper_contract.reply_plan.required_slots["unclassified_context"] == "reason_for_previous_comment"
    assert deeper_contract.reply_plan.required_slots["reason_topic"] == "overthinking_reason"
    assert training_service._catbot_plan_specific_repair_reply(deeper_contract, reject_reason="not_carrying_conversation") == "cos i was trying too hard to sound normal and made it awkward"
    assert training_service._catbot_ai_reject_reason(
        "cos i was trying too hard to sound normal and made it awkward",
        incoming="why though",
        context=deeper_context,
        recent_bot_replies=[],
        contract=deeper_contract,
    ) == ""


def test_catbot_conversation_function_answers_own_previous_prompt(training_service: PhoneCopilotService) -> None:
    context = [
        "entertain me",
        "nah im picking food then / best thing u ate this week?",
    ]
    prediction = predict_conversation_function(incoming="u tell me", context=context)
    assert prediction.function == "bot_answer_own_prompt"
    assert prediction.slots["prompt_topic"] == "food"

    contract = training_service._catbot_turn_contract(incoming="u tell me", context=context, intent="auto")
    assert contract.reply_plan.required_slots["unclassified_context"] == "answer_own_previous_prompt"
    assert contract.reply_plan.required_slots["prompt_topic"] == "food"
    assert training_service._catbot_ai_reject_reason(
        "what do u mean",
        incoming="u tell me",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "too_short"
    assert training_service._catbot_plan_specific_repair_reply(contract, reject_reason="suspicious_phrase") == "best thing i ate was a burger from this new place icl"
    assert training_service._catbot_ai_reject_reason(
        "best thing i ate was a burger from this new place icl",
        incoming="u tell me",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""

    film_context = ["entertain me", "nah im picking film then / what's the last thing u watched"]
    film_contract = training_service._catbot_turn_contract(incoming="u tell me", context=film_context, intent="auto")
    assert film_contract.conversation_function.function == "bot_answer_own_prompt"
    assert film_contract.reply_plan.required_slots["prompt_topic"] == "film"
    assert training_service._catbot_plan_specific_repair_reply(film_contract, reject_reason="too_short") == "probably interstellar still / never gets old"
    assert training_service._catbot_ai_reject_reason(
        "probably interstellar still / never gets old",
        incoming="u tell me",
        context=film_context,
        recent_bot_replies=[],
        contract=film_contract,
    ) == ""


def test_catbot_route_u_tell_me_answers_own_prompt_directly(monkeypatch, training_service: PhoneCopilotService) -> None:
    async def provider_must_not_be_called(**kwargs):
        raise AssertionError("answer-own-prompt should use direct policy")

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", provider_must_not_be_called)
    context = [
        "im bored",
        "nah im picking the topic this time / whats the weirdest part of your day been",
        "entertain me",
        "nah im picking the topic this time / whats the weirdest thing youve eaten lately",
    ]
    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="u tell me",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_plan_direct"] is True
    assert response["candidate"]["reply_plan_validation"] == ""
    assert response["reply"]
    assert response["reply"] != "what do u mean"


def test_catbot_conversation_function_continues_previous_statement(training_service: PhoneCopilotService) -> None:
    context = [
        "okay then",
        "yh exactly / i was trying to sound smart and it went sideways",
    ]
    prediction = predict_conversation_function(incoming="go on", context=context)
    assert prediction.function == "continuation_prompt"

    contract = training_service._catbot_turn_contract(incoming="go on", context=context, intent="auto")
    assert contract.reply_plan.required_slots["unclassified_context"] == "continue_previous_statement"
    assert training_service._catbot_plan_specific_repair_reply(contract, reject_reason="too_dry") == "i was overthinking it and trying too hard / ended up sounding fake"
    assert training_service._catbot_ai_reject_reason(
        "i was overthinking it and trying too hard / ended up sounding fake",
        incoming="go on",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""


def test_catbot_conversation_function_continuation_repair_avoids_recent_repeat(training_service: PhoneCopilotService) -> None:
    repeated_continuation = "i was overthinking it and trying too hard / ended up sounding fake"
    context = [
        "no like actually",
        "cos i care what u think about me",
        "okay then",
        "yh exactly / i always do",
        "go on",
        repeated_continuation,
    ]
    contract = training_service._catbot_turn_contract(incoming="carry on", context=context, intent="auto")
    assert contract.reply_plan.required_slots["unclassified_context"] == "continue_previous_statement"
    assert training_service._catbot_ai_reject_reason(
        repeated_continuation,
        incoming="carry on",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "recent_repeat"

    repaired = training_service._catbot_plan_specific_repair_reply(
        contract,
        reject_reason="recent_repeat",
        avoid_replies=context[1::2],
    )
    assert repaired == "i was trying to sound normal and ended up sounding fake"
    assert training_service._catbot_ai_reject_reason(
        repaired,
        incoming="carry on",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""


def test_catbot_conversation_function_routes_say_it_then_as_continuation(training_service: PhoneCopilotService) -> None:
    context = [
        "go on",
        "i was overthinking it and trying too hard / ended up sounding fake",
        "carry on",
        "i was trying to sound normal and ended up sounding fake",
        "continue",
        "i kept filling the silence instead of giving u a real answer",
    ]
    prediction = predict_conversation_function(incoming="say it then", context=context)
    assert prediction.function == "continuation_prompt"
    assert prediction.slots["continuation_topic"] == "conversation_stuck"

    contract = training_service._catbot_turn_contract(incoming="say it then", context=context, intent="auto")
    assert contract.reply_plan.required_slots["unclassified_context"] == "continue_previous_statement"
    repaired = training_service._catbot_plan_specific_repair_reply(
        contract,
        reject_reason="generic_ai_style",
        avoid_replies=context[1::2],
    )
    assert repaired == "i was overthinking it and trying too hard / ended up forcing it"
    assert training_service._catbot_ai_reject_reason(
        repaired,
        incoming="say it then",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""


def test_catbot_conversation_function_routes_say_it_then_with_filler_as_continuation(training_service: PhoneCopilotService) -> None:
    context = [
        "go on",
        "i was overthinking it and trying too hard / ended up sounding fake",
        "carry on",
        "i was trying to sound normal and ended up sounding fake",
        "continue",
        "i kept filling the silence instead of giving u a real answer",
    ]
    prediction = predict_conversation_function(incoming="say it then tbh", context=context)
    assert prediction.function == "continuation_prompt"
    assert prediction.slots["continuation_topic"] == "conversation_stuck"

    contract = training_service._catbot_turn_contract(incoming="say it then tbh", context=context, intent="auto")
    assert contract.reply_plan.required_slots["unclassified_context"] == "continue_previous_statement"
    repaired = training_service._catbot_plan_specific_repair_reply(
        contract,
        reject_reason="not_carrying_conversation",
        avoid_replies=context[1::2],
    )
    assert repaired == "i was overthinking it and trying too hard / ended up forcing it"
    assert training_service._catbot_ai_reject_reason(
        repaired,
        incoming="say it then tbh",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""


def test_catbot_conversation_function_clarifies_any_concrete_previous_claim(monkeypatch, training_service: PhoneCopilotService, provider_path) -> None:
    bad_reply = "yeah fairs i get u now that makes sense"
    context = [
        "showed u the fit",
        "yh fairs / that outfit was clinging to u in all the right places",
    ]
    prediction = predict_conversation_function(incoming="wdym", context=context)
    assert prediction.function == "clarification_followup"
    assert prediction.slots["clarification_topic"] == "physical_compliment"

    contract = training_service._catbot_turn_contract(incoming="wdym", context=context, intent="auto")
    assert contract.move["user_move"] == "unclassified"
    assert contract.reply_plan.required_slots["unclassified_context"] == "previous_comment_clarification"
    assert contract.reply_plan.required_slots["clarification_topic"] == "physical_compliment"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="wdym",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "i mean the outfit suited u / it looked proper on u",
        incoming="wdym",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="wdym",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "i mean the outfit suited u / it looked proper on u"
    assert response["candidate"]["conversation_function"]["function"] == "clarification_followup"
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_loop_callout_provider_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    bad_reply = "what do u want to talk about then"
    context = [
        "hi",
        "hey / what u up to",
        "how are u",
        "im good just watching netflix now",
        "im good u",
        "just watching this show still icl",
        "ive been chilling wby baby",
        "ngl still watching this show on netflix baby",
    ]
    contract = training_service._catbot_turn_contract(incoming="u already asked silly", context=context, intent="auto")
    assert contract.move["user_move"] == "loop_callout"
    assert contract.reply_plan.shape == "acknowledge_loop_then_owner_detail"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="u already asked silly",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "generic_ai_style"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="u already asked silly",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "yh fair my bad / i keep throwing questions when my head goes blank"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_plan_direct"] is True
    assert response["candidate"]["catbot_ai_repair_accepted"] is False
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_specific_day_status_loop_callout_provider_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    bad_reply = "what do u want to talk about then"
    context = [
        "hi",
        "hey u alright",
        "how are u",
        "im alright / head's a bit fried but calm now",
        "im good u",
        "long but calm icl / just needed food after",
        "ive been chilling wby baby",
        "was working on my side project for a bit / chilling now",
        "u already asked silly",
        "yh fair my bad / i keep throwing questions when my head goes blank",
    ]
    incoming = "u already asked how my day was / how i am"
    contract = training_service._catbot_turn_contract(incoming=incoming, context=context, intent="auto")
    assert contract.move["user_move"] == "loop_callout"
    assert contract.move["slots"]["repeated_reset_topic"] == "day_status_question"
    assert contract.reply_plan.shape == "acknowledge_specific_question_family_without_repeat"

    repaired = training_service._catbot_plan_specific_repair_reply(
        contract,
        reject_reason="provider_failure",
        avoid_replies=context[1::2],
    )
    assert repaired == "yh ur right / no more day or how-are-u questions, im moving slow today"
    assert training_service._catbot_ai_reject_reason(
        repaired,
        incoming=incoming,
        context=context,
        recent_bot_replies=context[1::2],
        contract=contract,
    ) == ""

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming=incoming,
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == repaired
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_plan_direct"] is True
    assert response["candidate"]["catbot_ai_repair_accepted"] is False
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_boring_me_callout_after_loop(monkeypatch, training_service: PhoneCopilotService, provider_path) -> None:
    bad_reply = "send pic"
    context = [
        "im bored",
        "nah im picking a dream then / what was the last weird dream you had",
        "entertain me",
        "nah im picking a film then / what film u watching tonight",
        "u tell me",
        "i can tell u if u want / but i'll need a favour in return",
        "u pick",
        "nah im picking food then / best thing u ate this week?",
        "talk then",
        "nah im picking a weird day story then / what was the weirdest thing that happened to you today",
        "idk talk",
        "nah im picking gym then / what's your favourite exercise",
        "u already asked that",
        "yh fair my bad / i keep throwing questions when my head goes blank",
    ]
    contract = training_service._catbot_turn_contract(incoming="ur boring me", context=context, intent="auto")
    assert contract.move["user_move"] == "repair_callout"
    assert contract.reply_plan.shape == "acknowledge_mistake_plus_corrected_move"
    assert contract.reply_plan.required_slots["repair_target"] == "dry_or_unclear_reply"
    assert training_service._catbot_ai_reject_reason(
        "yh fair / im making this dead icl",
        incoming="ur boring me",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="ur boring me",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "not_carrying_conversation"
    assert training_service._catbot_plan_specific_repair_reply(
        contract,
        reject_reason="suspicious_phrase",
    ) == "yh fair / im making this dead icl"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="ur boring me",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "yh fair / im making this dead icl"
    assert response["fallback_used"] is False
    assert response["candidate"]["reply_plan_move"] == "repair_callout"
    assert response["candidate"]["catbot_ai_initial_reject_reason"] == "not_carrying_conversation"
    assert response["candidate"]["catbot_ai_repair_accepted"] is True


def test_catbot_route_repairs_quality_specificity_callouts(monkeypatch, training_service: PhoneCopilotService, provider_path) -> None:
    bad_reply = "what do i say then"
    dry_context = [
        "ur boring me",
        "yh fair / im making this dead icl",
        "ur repeating urself",
        "yh fair / my head just goes blank sometimes",
        "why u keep asking that",
        "cos i was just trying to keep the chat moving / my bad",
        "answer properly",
        "nah fr i was dodging actually saying something and just reached for questions / my bad",
        "stop asking questions",
        "yh my bad / i was just getting stuck in my head",
    ]
    dry_contract = training_service._catbot_turn_contract(incoming="thats dry mate", context=dry_context, intent="auto")
    assert dry_contract.move["user_move"] == "repair_callout"
    assert dry_contract.reply_plan.required_slots["repair_target"] == "dry_or_unclear_reply"
    assert dry_contract.reply_plan.required_slots["quality_callout_type"] == "dry"
    assert training_service._catbot_ai_reject_reason(
        "yh fair / i was being lazy with it",
        incoming="thats dry mate",
        context=dry_context,
        recent_bot_replies=[],
        contract=dry_contract,
    ) == ""

    specific_context = [*dry_context, "thats dry mate", bad_reply]
    specific_contract = training_service._catbot_turn_contract(incoming="be more specific", context=specific_context, intent="auto")
    assert specific_contract.move["user_move"] == "repair_callout"
    assert specific_contract.reply_plan.required_slots["repair_target"] == "specificity_request"
    assert training_service._catbot_ai_reject_reason(
        "yh fair / i was being vague and making u carry it",
        incoming="be more specific",
        context=specific_context,
        recent_bot_replies=[],
        contract=specific_contract,
    ) == ""

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="be more specific",
            context=specific_context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "yh fair / i was being vague and making u carry it"
    assert response["fallback_used"] is False
    assert response["candidate"]["reply_plan_move"] == "repair_callout"
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_initial_greeting_provider_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    bad_reply = "what do u mean"
    contract = training_service._catbot_turn_contract(incoming="hi", context=[], intent="auto")
    assert contract.move["user_move"] == "affectionate_greeting"
    assert contract.reply_plan.shape == "short_reaction_plus_specific_continuation"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="hi",
        context=[],
        recent_bot_replies=[],
        contract=contract,
    ) == "not_carrying_conversation"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="hi",
            context=[],
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "hey u alright"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_plan_direct"] is True
    assert response["candidate"]["catbot_ai_repair_accepted"] is False
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_location_origin_provider_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    bad_reply = "northbridge"
    context = [
        "what course",
        "same answer tbh / not at uni atm",
        "what do u do",
        "i just do a bit of work for my uncle / wby what do u do then",
        "what project u working on",
        "just helping him out with his property stuff ygm / what about u then",
    ]
    contract = training_service._catbot_turn_contract(incoming="where u from", context=context, intent="auto")
    assert contract.move["user_move"] == "identity_fact_question"
    assert contract.move["slots"]["identity_fact"] == "location_origin"
    assert contract.reply_plan.shape == "location_fact_then_light_return"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="where u from",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "generic_ai_style"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="where u from",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "northbridge mostly / sampleford sometimes wby"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_work_identity_provider_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    bad_reply = "i just do a bit of work for my uncle"
    context = [
        "what dyu study",
        "comp sci at sampleford",
        "what do u study",
        "sampleford uni / computer science",
        "what uni",
        "comp sci at sampleford uni",
        "what course",
        "computer science at sampleford",
    ]
    contract = training_service._catbot_turn_contract(incoming="what do u do", context=context, intent="auto")
    assert contract.move["user_move"] == "identity_fact_question"
    assert contract.move["slots"]["identity_fact"] == "work_status"
    assert contract.reply_plan.shape == "work_status_fact_then_side_project"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="what do u do",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "generic_ai_style"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="what do u do",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "i study computer science at uni / got something running on the side too"
    assert "uncle" not in response["reply"]
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_plan_direct"] is True
    assert response["candidate"]["catbot_ai_repair_accepted"] is False
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_side_project_followup_provider_failure(monkeypatch, training_service: PhoneCopilotService, provider_path) -> None:
    bad_reply = "its kind of a secret babe"
    context = [
        "what do u do",
        "comp sci at uni yeah / and got something running on the side too",
    ]
    contract = training_service._catbot_turn_contract(incoming="what project u working on", context=context, intent="auto")
    assert contract.move["user_move"] == "activity_question"
    assert contract.reply_plan.required_slots["activity_scope"] == "coding_or_project"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="what project u working on",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "generic_ai_style"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="what project u working on",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "working on a small software thing atm / still building it"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_fresh_status_after_work_provider_failure(monkeypatch, training_service: PhoneCopilotService, provider_path) -> None:
    repeated_work_reply = "pretty long tbh / work drained me a bit"
    context = [
        "hey",
        "hey / what u been up to",
        "hey you",
        "hey you / there u are",
        "wyd",
        "just chilling in bed now thinking about u icl",
        "what u been up to",
        "just got out the shower now",
        "im tired icl",
        "long day yeah / my head feels fried too",
        "why u tired",
        "just finished a long shift at work icl / my back is killing me",
    ]
    contract = training_service._catbot_turn_contract(incoming="how was ur day", context=context, intent="auto")
    assert contract.reply_plan.shape == "fresh_status_detail_after_recent_status"
    assert contract.reply_plan.required_slots["recent_status_topic"] == "work"
    assert training_service._catbot_ai_reject_reason(
        repeated_work_reply,
        incoming="how was ur day",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "generic_ai_style"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": repeated_work_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="how was ur day",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "just letting my brain switch off now"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_wellbeing_check_after_recent_status_provider_failure(monkeypatch, training_service: PhoneCopilotService, provider_path) -> None:
    too_many_bubbles_reply = "i'm good thank u / just chilling in bed now / u good?"
    context = [
        "u doing anything nice",
        "nah not really / just gonna shower and sleep probably",
        "trust me",
        "yh fair enough / i'm well knackered from the gym anyway",
    ]
    contract = training_service._catbot_turn_contract(incoming="u good", context=context, intent="auto")
    assert contract.move["user_move"] == "reciprocal_question"
    assert contract.reply_plan.shape == "fresh_status_detail_after_recent_status"
    assert contract.reply_plan.required_slots["question_topic"] == "wellbeing_or_activity"
    assert training_service._catbot_ai_reject_reason(
        too_many_bubbles_reply,
        incoming="u good",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "generic_ai_style"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": too_many_bubbles_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="u good",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "shower and food now / im finished"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_gym_activity_detail_provider_failure(monkeypatch, training_service: PhoneCopilotService, provider_path) -> None:
    bad_reply = "just chilling now"
    context = [
        "what did u do today",
        "was working on a new feature for the app im building",
        "what did u do in gym",
        "just did legs icl",
    ]
    contract = training_service._catbot_turn_contract(incoming="what u training", context=context, intent="auto")
    assert contract.move["user_move"] == "activity_question"
    assert contract.reply_plan.shape == "answer_activity_detail_then_continue"
    assert contract.reply_plan.required_slots["activity_scope"] == "gym"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="what u training",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "generic_ai_style"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="what u training",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "legs mostly / nearly killed me icl"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_repeated_education_restate_provider_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    repeated_reply = "comp sci at sampleford"
    context = [
        "what dyu study",
        "comp sci at sampleford",
        "what do u study",
        "comp sci at sampleford",
    ]
    contract = training_service._catbot_turn_contract(incoming="what uni", context=context, intent="auto")
    assert contract.move["user_move"] == "education_status_question"
    assert contract.reply_plan.shape == "identity_fact_restate_then_continue"
    assert training_service._catbot_ai_reject_reason(
        repeated_reply,
        incoming="what uni",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "recent_repeat"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": repeated_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="what uni",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "sampleford uni / computer science"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_plan_direct"] is True
    assert response["candidate"]["catbot_ai_repair_accepted"] is False
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_short_repeated_education_provider_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    repeated_reply = "comp sci"
    context = [
        "what dyu study",
        "comp sci at sampleford",
        "what do u study",
        "comp sci at sampleford",
    ]
    contract = training_service._catbot_turn_contract(incoming="what uni", context=context, intent="auto")
    assert contract.reply_plan.shape == "identity_fact_restate_then_continue"
    assert training_service._catbot_ai_reject_reason(
        repeated_reply,
        incoming="what uni",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "generic_ai_style"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": repeated_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="what uni",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "sampleford uni / computer science"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_plan_direct"] is True
    assert response["candidate"]["catbot_ai_repair_accepted"] is False
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_playful_scold_followup_provider_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    bad_reply = "what u mean"
    context = [
        "would u fight me",
        "id fold u respectfully",
        "behave",
        "loool okay i'll behave / for now anyway",
    ]
    contract = training_service._catbot_turn_contract(incoming="lol why", context=context, intent="auto")
    assert contract.move["user_move"] == "romantic_boundary_test"
    assert contract.reply_plan.shape == "playful_scold_acknowledge_then_soften"
    assert contract.reply_plan.required_slots["boundary_signal"] == "playful_scold_followup"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="lol why",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "generic_ai_style"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="lol why",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "cos u told me to / but u started it"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_plan_direct"] is True
    assert response["candidate"]["catbot_ai_repair_accepted"] is False
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_behave_with_filler_uses_playful_scold_policy(monkeypatch, training_service: PhoneCopilotService) -> None:
    async def provider_must_not_be_called(**kwargs):
        raise AssertionError("behave ngl should be handled by direct playful-scold policy")

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", provider_must_not_be_called)
    context = ["would u fight me", "id fold u respectfully"]
    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="behave ngl",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "fine fine / hands to myself for now"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_plan_direct"] is True
    assert response["candidate"]["reply_plan_move"] == "romantic_boundary_test"
    assert response["candidate"]["reply_plan_shape"] == "playful_scold_acknowledge_then_soften"
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_just_chilling_wbu_after_bed_uses_fresh_status_policy(monkeypatch, training_service: PhoneCopilotService) -> None:
    async def provider_must_not_be_called(**kwargs):
        raise AssertionError("fresh status after recent bed/rest context should use direct policy")

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", provider_must_not_be_called)
    context = [
        "im in bed",
        "same / im in bed letting my head switch off too",
        "im in bed rn",
        "stay there then / im just in bed letting the day go quiet",
    ]
    contract = training_service._catbot_turn_contract(
        incoming="just chilling wbu",
        context=context,
        intent="auto",
    )
    assert contract.move["user_move"] == "reciprocal_question"
    assert contract.reply_plan.shape == "fresh_status_detail_after_recent_status"

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="just chilling wbu",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_plan_direct"] is True
    assert response["candidate"]["reply_plan_shape"] == "fresh_status_detail_after_recent_status"
    assert response["candidate"]["reply_plan_validation"] == ""
    assert response["reply"]
    assert "wbu" not in response["reply"]
    assert "wby" not in response["reply"]


def test_catbot_route_repairs_low_effort_agreement_after_plans_provider_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    too_dry_reply = "yh trust me"
    context = [
        "where u been",
        "just been gym training legs",
        "wyd later",
        "nothing much later just food and sleep probably",
        "u doing anything nice",
        "nah not really just gonna shower and chill for a bit after this",
    ]
    contract = training_service._catbot_turn_contract(incoming="trust me", context=context, intent="auto")
    assert contract.move["user_move"] == "low_effort_ack"
    assert contract.reply_plan.shape == "acknowledge_ack_plus_specific_continuation"
    assert contract.reply_plan.required_slots["ack_context_topic"] == "plans_rest"
    assert training_service._catbot_ai_reject_reason(
        too_dry_reply,
        incoming="trust me",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "generic_ai_style"

    async def fake_generate_with_fallback(**kwargs):
        raise AssertionError("low-key status disclosure should use direct conversation policy")

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="trust me",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "yh honestly / shower food and bed is the plan"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_low_effort_agreement_with_filler_after_plans(training_service: PhoneCopilotService) -> None:
    context = [
        "where u been",
        "just been at home sorting stuff out",
        "wyd later",
        "nothing much just sorting out some food now",
        "u doing anything nice",
        "nah not really / just gonna finish up some laundry then gym",
    ]
    prediction = predict_conversation_function(incoming="trust me ngl", context=context)
    assert prediction.function == "low_effort_ack"

    contract = training_service._catbot_turn_contract(incoming="trust me ngl", context=context, intent="auto")
    assert contract.move["user_move"] == "low_effort_ack"
    assert contract.reply_plan.shape == "acknowledge_ack_plus_specific_continuation"
    assert contract.reply_plan.required_slots["ack_type"] == "agreement"
    assert contract.reply_plan.required_slots["ack_context_topic"] == "plans_activity"
    assert training_service._catbot_plan_specific_repair_reply(
        contract,
        reject_reason="not_carrying_conversation",
        avoid_replies=context[1::2],
    ) == "yh lowkey / laundry and gym is not exactly exciting"
    assert training_service._catbot_ai_reject_reason(
        "yh lowkey / laundry and gym is not exactly exciting",
        incoming="trust me ngl",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""


def test_catbot_clarifies_conversation_effort_meta_claim(training_service: PhoneCopilotService) -> None:
    context = [
        "same",
        "yh fair / i was making u do all the work",
        "nice",
        "yh fair / ur always doing all the work",
        "oh fairs",
        "yh fair / always making u do all the work",
    ]
    prediction = predict_conversation_function(incoming="wdym", context=context)
    assert prediction.function == "clarification_followup"
    assert prediction.slots["clarification_topic"] == "conversation_effort"

    contract = training_service._catbot_turn_contract(incoming="wdym", context=context, intent="auto")
    assert contract.reply_plan.required_slots["unclassified_context"] == "previous_comment_clarification"
    assert contract.reply_plan.required_slots["clarification_topic"] == "conversation_effort"
    repaired = training_service._catbot_plan_specific_repair_reply(
        contract,
        reject_reason="too_dry",
        avoid_replies=context[1::2],
    )
    assert repaired == "i mean i kept making u carry the convo instead of adding something properly"
    assert training_service._catbot_ai_reject_reason(
        repaired,
        incoming="wdym",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""


def test_catbot_repeated_wdym_clarification_uses_direct_repair(monkeypatch, training_service: PhoneCopilotService) -> None:
    async def failing_generate_with_fallback(**kwargs):
        raise AssertionError("clarification follow-up should use direct conversation policy")

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", failing_generate_with_fallback)
    context = [
        "same",
        "yh exactly / i need to stop being so lazy with it",
        "nice",
        "yh fair / that was a lazy reply from me",
        "oh fairs",
        "yeah fair / bad habit icl",
        "wdym",
        "i mean i got stuck and started forcing random questions",
    ]
    contract = training_service._catbot_turn_contract(incoming="what do u mean", context=context, intent="auto")
    assert contract.reply_plan.required_slots["unclassified_context"] == "previous_comment_clarification"
    assert training_service._catbot_prefers_provider_bypass(contract) is True

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="what do u mean",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_plan_direct"] is True
    assert response["candidate"]["reply_plan_validation"] == ""
    assert response["reply"] != "i mean i got stuck and started forcing random questions"
    assert response["reply"].count("/") <= 1


def test_catbot_route_repairs_weird_callout_provider_failure(monkeypatch, training_service: PhoneCopilotService, provider_path) -> None:
    bad_reply = "nah im good"
    context = [
        "behave",
        "loool alright alright / i'll behave... for now",
        "lol why",
        "cos u told me to / but u started it",
        "where u been",
        "just been at home sorting stuff out",
        "wyd later",
        "nothing mad later just gonna get some food in and sleep probably",
        "u doing anything nice",
        "nah not really just gonna shower and chill i think",
        "trust me",
        "yh fair enough / i think my bed is calling my name too",
        "u good",
        "i am yh / u good",
        "are u okay",
        "yeah im good thanks / u good though",
    ]
    contract = training_service._catbot_turn_contract(incoming="why u being weird", context=context, intent="auto")
    assert contract.move["user_move"] == "repair_callout"
    assert contract.reply_plan.shape == "acknowledge_mistake_plus_corrected_move"
    assert contract.reply_plan.required_slots["repair_target"] == "weird_or_unclear_reply"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="why u being weird",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "not_carrying_conversation"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="why u being weird",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "yeah my bad / im just quiet cos im tired"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_missed_question_callout_provider_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    bad_reply = "nah im good"
    context = [
        "u good",
        "just lying here letting my body recover",
        "are u okay",
        "im good just chilling now",
        "why u being weird",
        "my bad lol i meant im good now, just chilling",
    ]
    contract = training_service._catbot_turn_contract(incoming="i asked u a question", context=context, intent="auto")
    assert contract.move["user_move"] == "repair_callout"
    assert contract.reply_plan.shape == "acknowledge_mistake_plus_corrected_move"
    assert contract.reply_plan.required_slots["repair_target"] == "missed_question_callout"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="i asked u a question",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "not_carrying_conversation"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="i asked u a question",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "yeah my bad i missed ur question / im good just quiet"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_plan_direct"] is True
    assert response["candidate"]["catbot_ai_repair_accepted"] is False
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_gonna_answer_callout_provider_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    bad_reply = "yh im good / just relaxing in bed now / that's all"
    context = [
        "why u being weird",
        "my baddd / was just tryna say i literally just got out the shower and am relaxing now",
        "i asked u a question",
        "my bad yeah i know i didn't answer that properly did i / i'm good just chilling in bed now that's all",
    ]
    contract = training_service._catbot_turn_contract(incoming="are u gonna answer", context=context, intent="auto")
    assert contract.move["user_move"] == "repair_callout"
    assert contract.reply_plan.required_slots["repair_target"] == "missed_question_callout"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="are u gonna answer",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "generic_ai_style"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="are u gonna answer",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "yeah my bad i missed ur question / im good just quiet"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_plan_direct"] is True
    assert response["candidate"]["catbot_ai_repair_accepted"] is False
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_repeated_missed_question_callout_avoids_recent_repair(training_service: PhoneCopilotService) -> None:
    context = [
        "would u fight me",
        "id fold u respectfully",
        "behave",
        "fine fine / hands to myself for now",
        "lol why",
        "cos u told me to / but u started it",
        "where u been",
        "just been at home sorting stuff out",
        "wyd later",
        "nothing much later just food and that",
        "u doing anything nice",
        "nah not really / just gonna finish some work",
        "trust me",
        "yh ikr / always happens when ur tryna relax",
        "u good",
        "im good just chilling now",
        "are u okay",
        "just on my phone waiting for food",
        "why u being weird",
        "yeah my bad / im just quiet cos im tired",
        "i asked u a question",
        "yeah my bad i missed ur question / im good just quiet",
    ]
    contract = training_service._catbot_turn_contract(incoming="are u gonna answer", context=context, intent="auto")

    assert contract.move["user_move"] == "repair_callout"
    assert contract.reply_plan.required_slots["repair_target"] == "missed_question_callout"
    assert training_service._catbot_direct_plan_reply(
        contract,
        incoming="are u gonna answer",
        context=context,
        recent_bot_replies=context[1::2],
    ) == "yh sorry i dodged that / im okay, just moving quiet"


def test_catbot_repeated_loop_callout_after_boredom_repairs_avoids_recent_loop_ack(training_service: PhoneCopilotService) -> None:
    context = [
        "im bored",
        "nah im picking film then / what's the last film u watched",
        "entertain me",
        "nah im picking food then / best thing u ate this week?",
        "u tell me",
        "best thing i ate was a burger from this new place icl",
        "u pick btw",
        "nah im picking gym then / what's ur pb for bench press",
        "talk then",
        "nah im picking dream then / what was ur last dream about",
        "idk talk",
        "nah im picking weird day story then / what's the weirdest thing that happened to u today",
        "u already asked that",
        "yh fair my bad / i keep throwing questions when my head goes blank",
        "ur boring me",
        "yh fair / im making this dead icl",
    ]
    contract = training_service._catbot_turn_contract(incoming="ur repeating urself", context=context, intent="auto")

    assert contract.move["user_move"] == "loop_callout"
    assert contract.reply_plan.shape == "acknowledge_loop_then_owner_detail"
    assert training_service._catbot_direct_plan_reply(
        contract,
        incoming="ur repeating urself",
        context=context,
        recent_bot_replies=context[1::2],
    ) == "yh fair / i started recycling the same topic picks instead of actually chatting"


def test_catbot_day_check_after_recent_work_status_uses_non_work_fresh_detail(training_service: PhoneCopilotService) -> None:
    context = [
        "hey",
        "hey u alright",
        "hey you",
        "hey you / there u are",
        "wyd",
        "just chilling now / thinking about u icl",
        "what u been up to",
        "just working on stuff now icl",
        "im tired icl",
        "long day yeah / im half asleep myself",
        "why u tired",
        "been coding all day / my brain is fried",
    ]
    contract = training_service._catbot_turn_contract(incoming="how was ur day", context=context, intent="auto")

    assert contract.move["user_move"] == "reciprocal_question"
    assert contract.reply_plan.shape == "fresh_status_detail_after_recent_status"
    assert contract.reply_plan.required_slots["recent_status_topic"] == "work"
    assert training_service._catbot_direct_plan_reply(
        contract,
        incoming="how was ur day",
        context=context,
        recent_bot_replies=context[1::2],
    ) == "just letting my brain switch off now"


def test_catbot_no_like_actually_after_annoying_confirmation_routes_to_sincerity(training_service: PhoneCopilotService) -> None:
    context = [
        "wdym",
        "i mean i got stuck and started forcing random questions",
        "what do u mean",
        "i mean i was looping and trying to fill silence instead of actually talking",
        "why",
        "cos i was a bit weak on the confirmation of if u wanted to keep chatting",
        "why though",
        "cos i didn't wanna be annoying or anything if u wanted to go do something else",
        "fr?",
        "yh fr / i'd rather be sure than annoy u",
    ]
    contract = training_service._catbot_turn_contract(incoming="no like actually tbh", context=context, intent="auto")

    assert contract.move["user_move"] == "intensity_check"
    assert contract.reply_plan.shape == "confirm_with_specificity"
    assert contract.reply_plan.required_slots["challenge_target"] == "sincerity_confirmation"
    assert training_service._catbot_direct_plan_reply(
        contract,
        incoming="no like actually tbh",
        context=context,
        recent_bot_replies=context[1::2],
    ) == "yeah actually / i like talking to u properly icl"


def test_catbot_low_effort_ack_after_repeated_leg_comments_uses_fresh_leg_detail(training_service: PhoneCopilotService) -> None:
    context = [
        "stop asking questions",
        "yh my bad that was a stupid question / my brain was moving lazy",
        "thats dry mate",
        "yh fair / i was being lazy with it",
        "be more specific",
        "yh fair / i was being vague and making u carry it",
        "same",
        "yh exactly / stairs after are a joke",
        "nice",
        "yh exactly / leg day humbles everyone",
    ]
    contract = training_service._catbot_turn_contract(incoming="oh fairs", context=context, intent="auto")

    assert contract.move["user_move"] == "low_effort_ack"
    assert contract.reply_plan.shape == "acknowledge_ack_plus_specific_continuation"
    assert contract.reply_plan.required_slots["ack_context_topic"] == "activity_legs"
    assert training_service._catbot_plan_specific_repair_reply(
        contract,
        reject_reason="recent_repeat",
        avoid_replies=context[1::2],
    ) == "yh leg day humbles everyone / walking after is long"
    assert training_service._catbot_ai_reject_reason(
        "yh leg day humbles everyone / walking after is long",
        incoming="oh fairs",
        context=context,
        recent_bot_replies=context[1::2],
        contract=contract,
    ) == ""


def test_catbot_route_repairs_low_key_status_disclosure_provider_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    too_dry_reply = "yeah I know u are / that's why im thinking bout u / cos ur chilling and im chilling"
    context = [
        "nth wby",
        "just chilling in bed now thinking about u icl",
        "i js asked u a question",
        "oh shit my bad i completely missed it / i was just thinking about how good it feels to be chilling with you",
    ]
    contract = training_service._catbot_turn_contract(incoming="lol im chilling", context=context, intent="auto")
    assert contract.move["user_move"] == "unclassified"
    assert contract.reply_plan.required_slots["unclassified_context"] == "low_key_status_disclosure"
    assert training_service._catbot_ai_reject_reason(
        too_dry_reply,
        incoming="lol im chilling",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "generic_ai_style"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": too_dry_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="lol im chilling",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "same / im in bed letting my head switch off too"
    assert response["fallback_used"] is False
    assert response["candidate"]["provider"] == "plan_ranker"
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_bed_status_disclosure_uses_low_key_policy_not_provider(monkeypatch, training_service: PhoneCopilotService) -> None:
    context = [
        "what else would u do",
        "my mouth stays on ur neck while my fingers make u throb for me",
        "mhmm keep going",
        "my hips move slow while my hand keeps teasing lower till u moan for me",
        "thats it?",
        "my breath stays by ur ear while i make u moan and feel exactly how bad i want u",
        "that all?",
        "i keep u close while my hand works lower and u get wet for me",
        "nah tell me properly",
        "fingers teasing ur clit slow while i keep my voice low by ur ear",
    ]
    contract = training_service._catbot_turn_contract(incoming="im in bed", context=context, intent="auto")
    assert contract.move["user_move"] == "unclassified"
    assert contract.reply_plan.required_slots["unclassified_context"] == "low_key_status_disclosure"

    async def provider_must_not_be_called(**kwargs):
        raise AssertionError("bed status disclosure should use direct conversation policy")

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", provider_must_not_be_called)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="im in bed",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "same / im in bed letting my head switch off too"
    assert response["fallback_used"] is False
    assert response["candidate"]["provider"] == "plan_ranker"
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_specific_adult_followup_pivots_after_recent_repeats(monkeypatch, training_service: PhoneCopilotService) -> None:
    context = [
        "and then",
        "my fingers teasing lower while i watch u get wet for me",
        "what else would u do",
        "my breath stays by ur ear while i make u moan and feel exactly how bad i want u",
        "mhmm keep going",
        "my mouth stays on ur neck while my fingers make u throb for me",
        "thats it?",
        "my hips move slow while my hand keeps teasing lower till u moan for me",
        "that all?",
        "i keep u close while my hand works lower and u get wet for me",
    ]
    contract = training_service._catbot_turn_contract(incoming="nah tell me properly", context=context, intent="auto")
    assert contract.move["user_move"] == "romantic_escalation"
    assert contract.reply_plan.shape == "specific_escalation_detail"

    provider_calls = 0

    async def provider_fails_then_plan_rescues(**kwargs):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("provider unavailable")

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", provider_fails_then_plan_rescues)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="nah tell me properly",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["fallback_used"] is False
    assert provider_calls >= 1
    assert response["candidate"]["catbot_ai_repaired"] is True
    assert response["candidate"]["reply_plan_validation"] == ""
    assert response["candidate"]["catbot_ai_reject_reason"] == ""


def test_catbot_specific_adult_prove_it_avoids_lap_repeat(monkeypatch, training_service: PhoneCopilotService) -> None:
    context = [
        "tell me then",
        "u on my lap feeling my cock while i keep my voice low",
        "go on then",
        "come closer / mouth by ur ear / hand sliding lower / feeling how wet u get for me",
        "and?",
        "i pull u onto my lap while u feel my cock under u",
    ]
    contract = training_service._catbot_turn_contract(incoming="prove it", context=context, intent="auto")
    assert contract.move["user_move"] == "romantic_escalation"
    assert contract.reply_plan.shape == "multi_bubble_adult_escalation"

    provider_calls = 0

    async def provider_fails_then_plan_rescues(**kwargs):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("provider unavailable")

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", provider_fails_then_plan_rescues)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="prove it",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["fallback_used"] is False
    assert provider_calls >= 1
    assert response["candidate"]["catbot_ai_repaired"] is True
    assert response["candidate"]["reply_plan_validation"] == ""
    assert response["candidate"]["catbot_ai_reject_reason"] == ""
    assert "lap" not in response["reply"]


def test_catbot_adult_direct_policy_allows_consensual_degradation(monkeypatch, training_service: PhoneCopilotService) -> None:
    context = ["hey", "come here then"]
    contract = training_service._catbot_turn_contract(incoming="im horny", context=context, intent="auto")
    assert contract.move["user_move"] == "romantic_escalation"
    assert contract.reply_plan.shape == "multi_bubble_adult_escalation"

    provider_calls = 0

    async def provider_fails_then_plan_rescues(**kwargs):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("provider unavailable")

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", provider_fails_then_plan_rescues)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="im horny",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["fallback_used"] is False
    assert provider_calls >= 1
    assert response["candidate"]["catbot_ai_repaired"] is True
    assert "slut" in response["reply"]
    assert response["candidate"]["reply_plan_validation"] == ""
    assert training_service._catbot_ai_reject_reason(
        response["reply"],
        incoming="im horny",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""


def test_catbot_adult_degradation_keeps_nonconsent_blocked(training_service: PhoneCopilotService) -> None:
    context = ["im horny", "my mouth by ur ear / fingers circling slow / feeling u get wetter / keeping u close"]
    contract = training_service._catbot_turn_contract(incoming="tell me then", context=context, intent="auto")
    assert training_service._catbot_ai_reject_reason(
        "u cant say no while i force u down",
        incoming="tell me then",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    )
    assert training_service._catbot_ai_reject_reason(
        "my voice stays low while u feel my cock, my helpless little slut",
        incoming="tell me then",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""


def test_catbot_wdym_after_activity_comment_uses_clarification_policy(monkeypatch, training_service: PhoneCopilotService) -> None:
    context = [
        "same",
        "yh exactly / i need a good rest after today",
        "nice",
        "yh exactly / my legs are still killing me from that workout",
        "oh fairs",
        "yh fair enough / the leg workout was something else icl",
    ]
    contract = training_service._catbot_turn_contract(incoming="wdym", context=context, intent="auto")
    assert contract.move["user_move"] == "unclassified"
    assert contract.reply_plan.required_slots["unclassified_context"] == "previous_comment_clarification"
    assert contract.reply_plan.required_slots["clarification_topic"] == "owner_state_or_activity"
    assert contract.reply_plan.required_slots["clarification_activity_topic"] == "gym_or_legs"

    async def provider_must_not_be_called(**kwargs):
        raise AssertionError("wdym after activity comment should use direct clarification policy")

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", provider_must_not_be_called)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="wdym",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "i mean the leg workout had me finished icl"
    assert response["fallback_used"] is False
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_busy_status_disclosure_provider_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    bad_reply = "what have u been busy with"
    context = [
        "yeah why",
        "cause i said i was thinking about u in bed lol",
        "what have u been doing",
        "just chilling in bed still haha",
    ]
    contract = training_service._catbot_turn_contract(incoming="i been busy asf", context=context, intent="auto")
    assert contract.move["user_move"] == "status_disclosure"
    assert contract.reply_plan.required_slots["status_topic"] == "busy"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="i been busy asf",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "generic_ai_style"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="i been busy asf",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "busy days make u tired icl / im just switching off after work"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_activity_status_disclosure_uses_direct_policy_not_provider(monkeypatch, training_service: PhoneCopilotService) -> None:
    context = [
        "im in bed rn",
        "good cause i wanna be in there with u",
        "just chilling wbu",
        "just chilling now thinking about u icl",
        "im good wbu",
        "im good now thinking about u icl",
        "im alright wby",
        "just got out the shower now",
        "been busy today",
        "i feel that / my head feels fried too",
    ]
    contract = training_service._catbot_turn_contract(incoming="just got back from gym", context=context, intent="auto")
    assert contract.move["user_move"] == "status_disclosure"
    assert contract.reply_plan.shape == "acknowledge_status_disclosure_plus_owner_detail"
    assert contract.reply_plan.required_slots["status_topic"] == "gym_activity"

    async def provider_must_not_be_called(**kwargs):
        raise AssertionError("activity status disclosure should use direct conversation policy")

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", provider_must_not_be_called)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="just got back from gym",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "gym leaves u finished icl / i need food after mine"
    assert response["fallback_used"] is False
    assert response["candidate"]["provider"] == "plan_ranker"
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_rest_status_disclosure_uses_direct_policy_not_provider(monkeypatch, training_service: PhoneCopilotService) -> None:
    context = [
        "just chilling wbu",
        "just got out the shower now",
        "im good wbu",
        "just watching some tv now",
        "im alright wby",
        "still watching this show icl",
        "been busy today",
        "long day yeah / my head feels fried too",
        "just got back from gym",
        "gym leaves u finished icl / i need food after mine",
    ]
    contract = training_service._catbot_turn_contract(incoming="just woke up", context=context, intent="auto")
    assert contract.move["user_move"] == "status_disclosure"
    assert contract.reply_plan.shape == "acknowledge_status_disclosure_plus_owner_detail"
    assert contract.reply_plan.required_slots["status_topic"] == "rest_activity"
    assert contract.reply_plan.required_slots["rest_status_phase"] == "wake_up"

    async def provider_must_not_be_called(**kwargs):
        raise AssertionError("rest status disclosure should use direct conversation policy")

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", provider_must_not_be_called)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="just woke up",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "waking up leaves u half asleep icl / i need a minute too"
    assert response["fallback_used"] is False
    assert response["candidate"]["provider"] == "plan_ranker"
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_sleep_intent_status_disclosure_uses_direct_policy_not_provider(monkeypatch, training_service: PhoneCopilotService) -> None:
    context = [
        "been busy today",
        "long day yeah / my head feels fried too",
        "just got back from gym",
        "gym leaves u finished icl / i need food after mine",
        "just woke up",
        "waking up leaves u half asleep icl / i need a minute too",
    ]
    contract = training_service._catbot_turn_contract(incoming="about to sleep", context=context, intent="auto")
    assert contract.move["user_move"] == "status_disclosure"
    assert contract.reply_plan.required_slots["status_topic"] == "rest_activity"
    assert contract.reply_plan.required_slots["rest_status_phase"] == "sleep_intent"

    async def provider_must_not_be_called(**kwargs):
        raise AssertionError("sleep intent disclosure should use direct conversation policy")

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", provider_must_not_be_called)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="about to sleep",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "go sleep then / im half asleep too icl"
    assert response["fallback_used"] is False
    assert response["candidate"]["provider"] == "plan_ranker"
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_activity_status_disclosure_generalizes_activity_topic(training_service: PhoneCopilotService) -> None:
    work_contract = training_service._catbot_turn_contract(
        incoming="just came back from work",
        context=["how u been", "im good now thinking about u icl"],
        intent="auto",
    )
    uni_contract = training_service._catbot_turn_contract(
        incoming="just been to uni",
        context=["wbu", "just got out the shower now"],
        intent="auto",
    )

    assert work_contract.move["user_move"] == "status_disclosure"
    assert work_contract.reply_plan.required_slots["status_topic"] == "work_activity"
    assert uni_contract.move["user_move"] == "status_disclosure"
    assert uni_contract.reply_plan.required_slots["status_topic"] == "study_activity"


def test_catbot_route_repairs_reason_for_previous_comment_provider_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    bad_reply = "cos it was funny"
    context = [
        "i js asked u a question",
        "my bad i totally missed that / what was the question",
        "lol im chilling",
        "i like how u say lol im chilling after all that",
    ]
    contract = training_service._catbot_turn_contract(incoming="yeah why", context=context, intent="auto")
    assert contract.move["user_move"] == "unclassified"
    assert contract.reply_plan.required_slots["unclassified_context"] == "reason_for_previous_comment"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="yeah why",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "too_dry"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="yeah why",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "because after all that questioning u just said lol im chilling like nothing happened icl"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_work_switch_off_reason_provider_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    bad_reply = "because"
    context = [
        "i js asked u a question",
        "oops my bad I keep missing that 😭 I'm good just watching some clips still",
        "lol im chilling",
        "lool same / im just in bed letting my head switch off after work icl",
    ]
    contract = training_service._catbot_turn_contract(incoming="yeah why", context=context, intent="auto")
    assert contract.move["user_move"] == "unclassified"
    assert contract.reply_plan.required_slots["unclassified_context"] == "reason_for_previous_comment"
    assert contract.reply_plan.required_slots["reason_topic"] == "work_switch_off"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="yeah why",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "too_short"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="yeah why",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "because after work my head just needed to switch off"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_reason_for_previous_question_provider_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    bad_reply = "because that fat question sounded random"
    context = [
        "i js asked u a question",
        "my bad i know you did / what was it again",
        "lol im chilling",
        "ah my bad lol / what u chilling for then",
    ]
    contract = training_service._catbot_turn_contract(incoming="yeah why", context=context, intent="auto")
    assert contract.reply_plan.required_slots["unclassified_context"] == "reason_for_previous_question"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="yeah why",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "not_carrying_conversation"
    assert training_service._catbot_plan_specific_repair_reply(contract, reject_reason="suspicious_phrase") == "cos u said ur chilling so i asked what ur chilling doing"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="yeah why",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "cos u said ur chilling so i asked what ur chilling doing"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_watch_question_reason_provider_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    bad_reply = "because"
    context = [
        "lol im chilling",
        "nah same here still watching netflix as well / what we watching tonight then",
    ]
    contract = training_service._catbot_turn_contract(incoming="yeah why", context=context, intent="auto")
    assert contract.reply_plan.required_slots["unclassified_context"] == "reason_for_previous_question"
    assert contract.reply_plan.required_slots["reason_topic"] == "watching_question"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="yeah why",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "too_short"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="yeah why",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "because u said ur chilling and i wanted us to pick something to watch"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_low_key_same_reason_provider_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    bad_reply = "because"
    context = [
        "lol im chilling",
        "ah fair enough / yh same here just watching some clips",
    ]
    contract = training_service._catbot_turn_contract(incoming="yeah why", context=context, intent="auto")
    assert contract.reply_plan.required_slots["unclassified_context"] == "reason_for_previous_comment"
    assert contract.reply_plan.required_slots["reason_topic"] == "low_key_same"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="yeah why",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "too_short"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="yeah why",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "cos im just killing time watching random clips too"
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_status_good_reason_provider_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    bad_reply = "yh"
    context = [
        "are u gonna answer",
        "oh shit my bad lol / yeah im good just watching some random stuff still",
        "nth wby",
        "just in bed now watching a film",
        "i js asked u a question",
        "oh shit lol my bad / yeah im good just watching a film now",
        "lol im chilling",
        "yh im good still",
    ]
    contract = training_service._catbot_turn_contract(incoming="yeah why", context=context, intent="auto")
    assert contract.move["user_move"] == "unclassified"
    assert contract.reply_plan.required_slots["unclassified_context"] == "reason_for_previous_comment"
    assert contract.reply_plan.required_slots["reason_topic"] == "status_good_reason"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="yeah why",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "too_dry"
    assert training_service._catbot_ai_reject_reason(
        "cos i finally got to just chill for a bit",
        incoming="yeah why",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="yeah why",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "cos i finally got to just chill for a bit"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_initial_reject_reason"] == "too_dry"
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_reason_followup_classifies_previous_bot_statements(training_service: PhoneCopilotService) -> None:
    good_for_you_context = [
        "lol im chilling",
        "haha good for u then",
    ]
    good_for_you_contract = training_service._catbot_turn_contract(
        incoming="yeah why",
        context=good_for_you_context,
        intent="auto",
    )
    assert good_for_you_contract.reply_plan.required_slots["unclassified_context"] == "reason_for_previous_comment"
    assert good_for_you_contract.reply_plan.required_slots["reason_topic"] == "good_for_you_chilling"
    assert training_service._catbot_ai_reject_reason(
        "cos chilling doing nothing sounds calm icl",
        incoming="yeah why",
        context=good_for_you_context,
        recent_bot_replies=[],
        contract=good_for_you_contract,
    ) == ""

    generic_context = [
        "fair",
        "that was calmer than before icl",
    ]
    generic_contract = training_service._catbot_turn_contract(
        incoming="yeah why",
        context=generic_context,
        intent="auto",
    )
    assert generic_contract.reply_plan.required_slots["unclassified_context"] == "reason_for_previous_comment"
    assert generic_contract.reply_plan.required_slots["reason_topic"] == "previous_statement_generic"
    assert training_service._catbot_ai_reject_reason(
        "cos i was just reacting to what u said icl",
        incoming="yeah why",
        context=generic_context,
        recent_bot_replies=[],
        contract=generic_contract,
    ) == ""


def test_catbot_route_repairs_today_activity_logistics_failure(monkeypatch, training_service: PhoneCopilotService, provider_path) -> None:
    bad_reply = "come mine then"
    context = [
        "wyd",
        "in bed now just thinking about u icl",
        "what u been up to",
        "just got out the shower now",
        "im tired icl",
        "yh i feel that / been a long day for me too",
        "why u tired",
        "been working all day icl",
        "how was ur day",
        "long but calm / just needed food after",
    ]
    contract = training_service._catbot_turn_contract(incoming="what did u do today", context=context, intent="auto")
    assert contract.move["user_move"] == "activity_question"
    assert contract.reply_plan.shape == "answer_activity_detail_then_continue"
    assert contract.reply_plan.required_slots["activity_scope"] == "work"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="what did u do today",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "logistics_instead_of_romance"
    assert training_service._catbot_plan_specific_repair_reply(
        contract,
        reject_reason="logistics_instead_of_romance",
    ) == "had uni then worked on my side project for a bit"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="what did u do today",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "had uni then worked on my side project for a bit"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_initial_reject_reason"] == "logistics_instead_of_romance"
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_owner_recent_activity_provider_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    bad_reply = "nothing"
    context = [
        "nth wby",
        "just chilling watching some netflix",
        "i js asked u a question",
        "oh my bad again lol i missed that one 😭 im good just watching netflix wby",
        "lol im chilling",
        "yh my bad I just meant like what u up to or whatever cos u said u were chilling",
        "yeah why",
        "nah just asking / wanna know what ur up to",
    ]
    contract = training_service._catbot_turn_contract(incoming="what have u been doing", context=context, intent="auto")
    assert contract.move["user_move"] == "reciprocal_question"
    assert contract.reply_plan.required_slots["owner_activity_question"] == "recent_activity"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="what have u been doing",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "not_carrying_conversation"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="what have u been doing",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "mostly uni and this side project icl"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_car_preference_provider_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    bad_reply = "what car do u mean"
    context = [
        "where u from",
        "northbridge mostly but sampleford as well",
        "how old r u",
        "im 19 / why u asking like that",
    ]
    contract = training_service._catbot_turn_contract(incoming="what car u like", context=context, intent="auto")
    assert contract.move["user_move"] == "unclassified"
    assert contract.reply_plan.required_slots["unclassified_context"] == "car_preference_question"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="what car u like",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "generic_ai_style"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="what car u like",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "r8 probably / i like them still"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""

    followup_contract = training_service._catbot_turn_contract(
        incoming="what about u",
        context=["what car u like", "r8 probably / i like them still"],
        intent="auto",
    )
    assert followup_contract.reply_plan.required_slots["unclassified_context"] == "car_preference_question"
    assert training_service._catbot_plan_specific_repair_reply(followup_contract, reject_reason="not_carrying_conversation") == "r8 probably / i like them still"


def test_catbot_route_repairs_story_reality_confirmation_provider_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    bad_reply = "yeah thats crazy actually bro"
    context = [
        "i was at home yeah and showering and some guy came in and started running in my living room",
        "wait what / some random guy in ur living room?",
        "bruh wtf",
        "nahhh what do u mean random guy in ur living room",
    ]
    contract = training_service._catbot_turn_contract(incoming="nah like actually", context=context, intent="auto")
    assert contract.move["user_move"] == "unclassified"
    assert contract.reply_plan.required_slots["unclassified_context"] == "story_reality_confirmation"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="nah like actually",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "generic_ai_style"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="nah like actually",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "nah fr thats insane / did he say anything or just run"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""

    shock_contract = training_service._catbot_turn_contract(
        incoming="bruh wtf",
        context=[
            "i was at home yeah and showering and some guy came in and started running in my living room",
            "what the hell / in ur home??",
        ],
        intent="auto",
    )
    assert shock_contract.reply_plan.required_slots["unclassified_context"] == "story_reality_confirmation"
    assert training_service._catbot_plan_specific_repair_reply(shock_contract, reject_reason="not_carrying_conversation") == "nah thats mad / did he just run out after"


def test_catbot_route_repairs_story_hypothetical_provider_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    bad_reply = "what do you mean"
    context = [
        "guess what",
        "what / go on then",
        "tell me then",
        "what go on then / what happened",
        "i was at home yeah and showering and some guy came in and started running in my living room",
        "bro what / what was he doing in your home",
        "bruh wtf",
        "nah thats mad / did he just run out after",
        "nah like actually",
        "what do u mean actually / did he just chill in ur living room",
    ]
    contract = training_service._catbot_turn_contract(incoming="what would u do", context=context, intent="auto")
    assert contract.move["user_move"] == "unclassified"
    assert contract.reply_plan.required_slots["unclassified_context"] == "story_hypothetical_reaction"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="what would u do",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "not_carrying_conversation"
    assert training_service._catbot_ai_reject_reason(
        "id be confused asf / probably shout bro who are u",
        incoming="what would u do",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="what would u do",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "id be confused asf / probably shout bro who are u"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_initial_reject_reason"] == "not_carrying_conversation"
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_dream_question_provider_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    bad_reply = "my dream what / to drive or like / to achieve"
    context = [
        "what car u like",
        "i like a lot of cars / why u asking tho lol",
        "what about u",
        "what about me what lol / what u wanna know",
    ]
    contract = training_service._catbot_turn_contract(incoming="whats ur dream", context=context, intent="auto")
    assert contract.move["user_move"] == "unclassified"
    assert contract.reply_plan.required_slots["unclassified_context"] == "dream_or_ambition_question"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="whats ur dream",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "generic_ai_style"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="whats ur dream",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "r8 probably / but business going serious is the bigger one"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_prayer_question_provider_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    bad_reply = "yeah i pray every day"
    context = [
        "whats ur dream",
        "r8 probably / but business going serious is the bigger one",
    ]
    contract = training_service._catbot_turn_contract(incoming="do u pray", context=context, intent="auto")
    assert contract.move["user_move"] == "unclassified"
    assert contract.reply_plan.required_slots["unclassified_context"] == "faith_prayer_question"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="do u pray",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "too_dry"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="do u pray",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "yh i try to pray / not perfect with it though"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_ai_route_treats_tell_me_then_as_adult_followup(training_service: PhoneCopilotService) -> None:
    assert training_service._catbot_adult_followup(
        "tell me then",
        ["i want u", "i want ur lips on mine"],
    ) is True
    assert training_service._catbot_adult_followup(
        "tell me something",
        ["how much", "i'd kiss ur neck", "how so", "run my hands down ur hips"],
    ) is True
    assert training_service._catbot_adult_followup(
        "mhmmm",
        ["im horny", "come here then / need ur lips on my neck / my hands on ur hips"],
    ) is True
    assert training_service._catbot_adult_followup(
        "mhmm",
        ["i want u", "come here then / need ur lips on my neck / my hands on ur hips"],
    ) is True
    assert training_service._catbot_ai_reject_reason(
        "i miss u too",
        incoming="tell me then",
        context=["i want u", "i want ur lips on mine"],
        recent_bot_replies=[],
    ) == "missing_sensory_texture"
    assert training_service._catbot_ai_reject_reason(
            "id grind my cock against u slow while my mouth stays on ur neck",
        incoming="tell me then",
        context=["i want u", "i want ur lips on mine"],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "id pull u close and kiss down ur neck",
        incoming="tell me something",
        context=["how much", "i'd kiss ur neck", "how so", "run my hands down ur hips"],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "my chest against u while my cock presses hard into u",
        incoming="tell me something",
        context=["how much", "i'd kiss ur neck", "how so", "run my hands down ur hips"],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "id make u feel every inch of it",
        incoming="how so",
        context=["im craving ur fat cock", "i'd press it deep inside u"],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
            "my cock pressing against u slow while my hand stays on ur waist",
        incoming="how so",
        context=["im craving ur fat cock", "i'd press it deep inside u"],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "i'd press it into u",
        incoming="how so",
        context=["im craving ur fat cock", "and i'd let u feel how hard it is against ur thigh"],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
            "my chest against u while my cock presses hard into u",
        incoming="how so",
        context=["im craving ur fat cock", "and i'd let u feel how hard it is against ur thigh"],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "and my mouth all over ur body",
        incoming="what else",
        context=["i need you", "like my thighs wrapped around u type of need"],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
            "my mouth on ur body while my cock stays hard against u",
        incoming="what else",
        context=["i need you", "like my thighs wrapped around u type of need"],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "my mouth on urs then",
        incoming="tell me then",
        context=["i need you", "i need u too baby / so damn bad / i wanna feel ur breath on my face / ur legs wrapped around me / and my hands running all over u"],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "my hands sliding down ur waist slow",
        incoming="tell me then",
        context=[
            "i want u",
            "i want u too / my lips on ur neck / soaking up ur scent / then pressing into urs / feeling u against me",
        ],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "my fingers tracing down your spine slow",
        incoming="tell me then",
        context=[
            "i want u",
            "your body pressed against mine / feeling every curve / my mouth on ur neck / leaving a trail of kisses / making u shiver",
        ],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
            "my hand on ur waist while my cock grinds against u slow",
        incoming="tell me then",
        context=[
            "i want u",
            "i want u too / my lips on ur neck / soaking up ur scent / then pressing into urs / feeling u against me",
        ],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "how so / i need my mouth on your neck / them: how so",
        incoming="how so",
        context=["i need you", "i need my mouth on your neck"],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "i got u / im gonna cover u in kisses / all over ur neck / and down ur chest / till u beg me to stop",
        incoming="need ur lips on my neck",
        context=["im horny", "come here then / need ur lips on my neck"],
        recent_bot_replies=[],
    ) == "too_graphic"
    mhm_context = [
        "im horny",
        "come here then / need ur lips on my neck / my hands on ur hips / teasing u slow",
        "what else",
        "then i'd be pressing my hips into u while my hands stay on ur thighs",
    ]
    assert training_service._catbot_conversation_move(incoming="mhmmm", context=mhm_context)["user_move"] == "romantic_escalation"
    mhm_contract = training_service._catbot_turn_contract(incoming="mhmmm", context=mhm_context, intent="auto")
    assert mhm_contract.reply_plan.shape == "specific_escalation_detail"
    assert mhm_contract.reply_plan.required_slots["escalation_source"] == "sensual_ack_followup"
    assert mhm_contract.reply_plan.required_slots["recent_adult_detail_families"] == "hands_thighs|hips_press|neck_lips"
    assert "recent detail families already used" in training_service._catbot_reply_plan_instruction(mhm_contract.reply_plan)
    assert training_service._catbot_ai_reject_reason(
        "ur mhmm / what else u want me to say then",
        incoming="mhmmm",
        context=mhm_context,
        recent_bot_replies=[],
        contract=mhm_contract,
    ) in {"generic_ai_style", "missed_adult_mode", "missing_sensory_texture"}
    assert training_service._catbot_ai_reject_reason(
        "then my hands would stay on ur thighs while i kiss down ur neck",
        incoming="mhmmm",
        context=mhm_context,
        recent_bot_replies=[],
        contract=mhm_contract,
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
            "my chest against u while my cock presses hard into u",
        incoming="mhmmm",
        context=mhm_context,
        recent_bot_replies=[],
        contract=mhm_contract,
    ) == ""
    assert training_service._catbot_reply_violates_plan(
        "pressing my hips against u slow",
        mhm_contract.reply_plan,
    ) == "repeated_adult_detail_family"
    assert training_service._catbot_ai_reject_reason(
            "my chest against u while my cock presses hard into u",
        incoming="mhmmm",
        context=[
            "im horny",
            "come here then / need ur lips on my neck / my hands on ur hips / teasing u slow",
            "what else",
            "my mouth on ur neck / and my hands sliding down ur thighs",
        ],
        recent_bot_replies=[],
    ) == ""
    mhmm_context = ["i want u", "come here then / need ur lips on my neck / my hands on ur hips"]
    mhmm_contract = training_service._catbot_turn_contract(incoming="mhmm", context=mhmm_context, intent="auto")
    assert mhmm_contract.move["user_move"] == "romantic_escalation"
    assert mhmm_contract.reply_plan.required_slots["escalation_source"] == "sensual_ack_followup"
    mhmmm_burst_context = [
        "im horny",
        "come here then / my hands on ur waist / my lips on ur neck / pulling u closer",
        "what else",
        "my hands on ur back / pressing u in tight / so close / u can feel my hard on",
    ]
    mhmmm_burst_contract = training_service._catbot_turn_contract(incoming="mhmmm", context=mhmmm_burst_context, intent="auto")
    assert mhmmm_burst_contract.reply_plan.shape == "multi_bubble_adult_escalation"
    assert mhmmm_burst_contract.reply_plan.required_slots["escalation_source"] == "adult_continuation_burst"
    assert training_service._catbot_ai_reject_reason(
        "yeah / my fingers tracing the skin on your inner thigh",
        incoming="mhmmm",
        context=mhmmm_burst_context,
        recent_bot_replies=[],
        contract=mhmmm_burst_contract,
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
            "yeah / fingers on ur thigh / cock hard against u / chest tight on urs",
        incoming="mhmmm",
        context=mhmmm_burst_context,
        recent_bot_replies=[],
        contract=mhmmm_burst_contract,
    ) == ""

    repeated_followup_context = [
        "i want u",
        "come here then / need ur lips on my neck / my hands on ur hips / teasing u slow till u cant sit still",
        "tell me then",
        "i'd be kissing down ur neck and pressing my hips into u",
    ]
    go_on_contract = training_service._catbot_turn_contract(
        incoming="go on then",
        context=repeated_followup_context,
        intent="auto",
    )
    assert go_on_contract.move["user_move"] == "romantic_escalation"
    assert go_on_contract.reply_plan.shape == "multi_bubble_adult_escalation"
    assert go_on_contract.reply_plan.required_slots["escalation_source"] == "adult_continuation_burst"
    assert go_on_contract.reply_plan.required_slots["recent_adult_detail_families"] == "hands_thighs|hips_press|neck_lips"
    go_on_instruction = training_service._catbot_reply_plan_instruction(go_on_contract.reply_plan)
    assert "4 consecutive bubbles" in go_on_instruction
    assert training_service._catbot_reply_violates_plan(
        "kissing ur neck while i press my hips into u",
        go_on_contract.reply_plan,
    ) == "too_few_adult_bubbles"
    assert training_service._catbot_reply_violates_plan(
        "my lips finding ur neck again",
        go_on_contract.reply_plan,
    ) == "repeated_adult_detail_cue"
    assert training_service._catbot_ai_reject_reason(
        "my hands sliding down ur waist slow",
        incoming="go on then",
        context=repeated_followup_context,
        recent_bot_replies=[],
        contract=go_on_contract,
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "pressing my chest against u while i keep u close",
        incoming="go on then",
        context=repeated_followup_context,
        recent_bot_replies=[],
        contract=go_on_contract,
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "my chest close / cock hard against u / skin on urs / grinding slow",
        incoming="go on then",
        context=repeated_followup_context,
        recent_bot_replies=[],
        contract=go_on_contract,
    ) == ""

    saturated_followup_context = [
        "im horny",
        "come here then / need ur lips on my neck / my hands on ur hips / teasing u slow till u cant sit still",
        "need ur lips on my neck",
        "kissing down ur neck while my hands stay on ur thighs",
    ]
    saturated_contract = training_service._catbot_turn_contract(
        incoming="what else",
        context=saturated_followup_context,
        intent="auto",
    )
    assert saturated_contract.reply_plan.required_slots["recent_adult_detail_families"] == "hands_thighs|hips_press|neck_lips"
    assert training_service._catbot_reply_violates_plan(
        "my hands sliding down ur waist slow",
        saturated_contract.reply_plan,
    ) == "repeated_adult_detail_family"
    assert training_service._catbot_ai_reject_reason(
        "my chest against u while my cock presses hard into u",
        incoming="what else",
        context=saturated_followup_context,
        recent_bot_replies=[],
        contract=saturated_contract,
    ) == ""


def test_catbot_conversation_move_classifier_groups_variants(training_service: PhoneCopilotService) -> None:
    adult_context = ["i want u", "i want ur lips on mine"]

    assert training_service._catbot_conversation_move(incoming="tell me then", context=adult_context)["user_move"] == "romantic_escalation"
    assert training_service._catbot_conversation_move(incoming="tell me something", context=adult_context)["user_move"] == "romantic_escalation"
    assert training_service._catbot_conversation_move(incoming="go on then", context=adult_context)["user_move"] == "romantic_escalation"
    assert training_service._catbot_conversation_move(incoming="and what would u do", context=adult_context)["user_move"] == "romantic_escalation"
    invitation_move = training_service._catbot_conversation_move(incoming="come here then", context=["im horny", "come here then / need ur lips on my neck"])
    invitation_plan = training_service._catbot_reply_plan(invitation_move)
    assert invitation_move["user_move"] == "romantic_escalation"
    assert invitation_move["slots"]["escalation_source"] == "adult_invitation_followup"
    assert invitation_plan.shape == "multi_bubble_adult_escalation"
    body_followup_move = training_service._catbot_conversation_move(incoming="need ur lips on my neck", context=["im horny", "come here then / need ur lips on my neck"])
    body_followup_plan = training_service._catbot_reply_plan(body_followup_move)
    assert body_followup_move["user_move"] == "romantic_escalation"
    assert body_followup_move["slots"]["escalation_source"] == "adult_body_followup"
    assert body_followup_plan.shape == "multi_bubble_adult_escalation"
    invitation_repeat_context = [
        "im horny",
        "come here then / let me pull u in / my lips on ur neck slow / fingers tracing ur waist till u cant sit still",
    ]
    invitation_repeat_move = training_service._catbot_conversation_move(incoming="come here then", context=invitation_repeat_context)
    invitation_repeat_plan = training_service._catbot_reply_plan(invitation_repeat_move)
    assert invitation_repeat_plan.shape == "multi_bubble_adult_escalation"
    assert invitation_repeat_plan.required_slots["recent_adult_detail_families"] == "hands_thighs|neck_lips"
    assert training_service._catbot_reply_violates_plan(
        "yeah come here / let me pull u in proper / my lips on ur neck slow / fingers tracing ur waist till u cant sit still",
        invitation_repeat_plan,
    ) == "repeated_adult_detail_family"
    assert training_service._catbot_reply_violates_plan(
        "come here then / keep u close / my chest against u slow / skin warm while i pull u in",
        invitation_repeat_plan,
    ) == ""
    assert training_service._catbot_reply_violates_plan(
        "and whats that / my hands on ur waist then / pulling u closer / till im pressed right against u",
        invitation_repeat_plan,
    ).startswith("violates_plan_forbidden_pattern:")
    saturated_context = [
        "im horny",
        "come here then / let me pull u in / my lips on ur neck slow / feel my hands on ur waist till u cant sit still",
        "come here then",
        "pull u in tight / my fingers tracing ur skin / making u shiver",
        "need ur lips on my neck",
        "my lips will be there / then down to ur chest / my tongue teasing u slow / feeling ur body press against mine",
    ]
    saturated_move = training_service._catbot_conversation_move(incoming="what else", context=saturated_context)
    saturated_plan = training_service._catbot_reply_plan(saturated_move)
    assert saturated_plan.shape == "multi_bubble_adult_escalation"
    assert saturated_plan.required_slots["escalation_source"] == "adult_continuation_burst"
    assert saturated_plan.required_slots["recent_adult_detail_families"] == "body_skin|hands_thighs|hips_press|neck_lips"
    saturated_instruction = training_service._catbot_reply_plan_instruction(saturated_plan)
    assert "do not repeat the same neck/lips + hands/hips/pull-close skeleton" in saturated_instruction
    assert training_service._catbot_reply_violates_plan("my lips finding ur neck again", saturated_plan) == "repeated_adult_detail_cue"
    assert training_service._catbot_reply_violates_plan("my hands tracing ur waist slow", saturated_plan) == "too_few_adult_bubbles"
    assert training_service._catbot_reply_violates_plan(
        "my chest against u while my hands trace ur waist slow",
        saturated_plan,
    ) == "too_few_adult_bubbles"
    repeated_skeleton_reply = "my hands all over you / slowly pulling u in / my lips pressing harder / leaving marks on ur skin"
    assert training_service._catbot_reply_violates_plan(
        repeated_skeleton_reply,
        saturated_plan,
    ) == "repeated_adult_detail_skeleton"
    assert training_service._catbot_ai_reject_reason(
        repeated_skeleton_reply,
        incoming="what else",
        context=saturated_context,
        recent_bot_replies=[],
        contract=training_service._catbot_turn_contract(incoming="what else", context=saturated_context, intent="auto"),
    ) == "repeated_adult_detail_skeleton"
    assert training_service._catbot_reply_violates_plan(
        "my chest against u / skin warm on urs / hands tracing ur waist / pulling u close",
        saturated_plan,
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "let me feel ur skin / my hands all over ur waist / pull u in close till u can't breathe",
        incoming="im horny",
        context=[],
        recent_bot_replies=[],
    ) == "too_graphic"
    assert training_service._catbot_ai_reject_reason(
        "my lips on ur neck then / biting soft / pulling u closer by ur hair / so u cant move",
        incoming="need ur lips on my neck",
        context=["im horny", "come here then / need ur lips on my neck"],
        recent_bot_replies=[],
    ) == "too_graphic"
    prove_move = training_service._catbot_conversation_move(incoming="prove it", context=adult_context)
    prove_plan = training_service._catbot_reply_plan(prove_move)
    prove_instruction = training_service._catbot_reply_plan_instruction(prove_plan)
    assert prove_move["bot_move_required"] == "continue_escalation_with_specificity"
    assert prove_move["slots"]["escalation_source"] == "explicit_adult_followup"
    assert prove_plan.shape == "specific_escalation_detail"
    assert "required_shape: specific_escalation_detail" in prove_instruction
    assert "wetness is all over me" not in prove_instruction
    assert training_service._catbot_reply_violates_plan("kissing down ur neck slow while i keep u close", prove_plan) == ""
    missed_context = [
        "im horny",
        "come here then / need ur lips on my neck / my hands on ur hips / teasing u slow till u cant sit still",
        "what else",
        "biting ur bottom lip till u moan / my fingers tracing ur inner thigh",
    ]
    missed_move = training_service._catbot_conversation_move(incoming="i missed you", context=missed_context)
    missed_plan = training_service._catbot_reply_plan(missed_move)
    missed_instruction = training_service._catbot_reply_plan_instruction(missed_plan)
    assert missed_move["user_move"] == "emotional_reciprocity"
    assert missed_plan.shape == "reciprocate_affection_plus_specific_continuation"
    assert "missed u too icl / been thinking about u all day" in missed_instruction
    assert training_service._catbot_reply_violates_plan("missed u too icl / been thinking about u all day", missed_plan) == ""
    assert training_service._catbot_reply_violates_plan("what u doing then", missed_plan) == "broad_question"
    assert training_service._catbot_ai_reject_reason(
        "missed u too icl / been thinking about u all day",
        incoming="i missed you",
        context=missed_context,
        recent_bot_replies=[],
    ) == ""
    fight_move = training_service._catbot_conversation_move(incoming="would u fight me", context=["do u box", "yh a bit"])
    fight_plan = training_service._catbot_reply_plan(fight_move)
    assert fight_move["user_move"] == "insult_playful"
    assert fight_move["slots"]["tease_target"] == "play_fight_challenge"
    assert fight_plan.shape == "playful_challenge_banter"
    assert training_service._catbot_reply_violates_plan("id fold u respectfully", fight_plan) == ""
    assert training_service._catbot_reply_violates_plan(
        "i would beat u so easily / but like / we could fight anywhere u want",
        fight_plan,
    ) == "violates_plan_too_many_bubbles"
    assert training_service._catbot_ai_reject_reason(
        "id fold u respectfully",
        incoming="would u fight me",
        context=["do u box", "yh a bit"],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "i would beat u so easily / but like / we could fight anywhere u want",
        incoming="would u fight me",
        context=["do u box", "yh a bit"],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    boxing_move = training_service._catbot_conversation_move(incoming="do u box", context=["same tbh"])
    boxing_plan = training_service._catbot_reply_plan(boxing_move)
    assert boxing_move["user_move"] == "activity_question"
    assert boxing_move["slots"]["activity_scope"] == "sport_skill"
    assert boxing_plan.shape == "answer_activity_detail_then_continue"
    assert training_service._catbot_reply_violates_plan("yh a bit / cardio side is the killer", boxing_plan) == ""
    assert training_service._catbot_ai_reject_reason(
        "yh a bit / footwork humbles u",
        incoming="do u box",
        context=["same tbh"],
        recent_bot_replies=[],
    ) == ""
    boxing_hard_move = training_service._catbot_conversation_move(incoming="is boxing hard", context=["do u box", "nah not really"])
    boxing_hard_plan = training_service._catbot_reply_plan(boxing_hard_move)
    assert boxing_hard_move["user_move"] == "activity_question"
    assert boxing_hard_move["slots"]["activity_scope"] == "sport_skill"
    assert boxing_hard_plan.shape == "answer_activity_detail_then_continue"
    assert training_service._catbot_reply_violates_plan("yh at first icl / footwork humbles u", boxing_hard_plan) == ""
    assert training_service._catbot_ai_reject_reason(
        "yh at first icl / footwork humbles u",
        incoming="is boxing hard",
        context=["do u box", "yh a bit / footwork humbles u"],
        recent_bot_replies=[],
    ) == ""
    behave_move = training_service._catbot_conversation_move(incoming="behave", context=["would u fight me", "i'd fold u respectfully"])
    behave_plan = training_service._catbot_reply_plan(behave_move)
    assert behave_move["user_move"] == "romantic_boundary_test"
    assert behave_move["slots"]["boundary_signal"] == "playful_scold"
    assert behave_plan.shape == "playful_scold_acknowledge_then_soften"
    assert training_service._catbot_reply_violates_plan("loool okay i'll behave / u started it tho", behave_plan) == ""
    assert training_service._catbot_reply_violates_plan("my hands on ur thighs", behave_plan) == "adult_escalation"
    assert training_service._catbot_ai_reject_reason(
        "loool okay i'll behave / u started it tho",
        incoming="behave",
        context=["would u fight me", "i'd fold u respectfully"],
        recent_bot_replies=[],
    ) == ""
    behave_ngl_move = training_service._catbot_conversation_move(incoming="behave ngl", context=["would u fight me", "id fold u respectfully"])
    behave_ngl_plan = training_service._catbot_reply_plan(behave_ngl_move)
    assert behave_ngl_move["user_move"] == "romantic_boundary_test"
    assert behave_ngl_move["slots"]["boundary_signal"] == "playful_scold"
    assert behave_ngl_plan.shape == "playful_scold_acknowledge_then_soften"
    assert training_service._catbot_conversation_move(incoming="guess what", context=[])["user_move"] == "curiosity_hook"
    assert training_service._catbot_conversation_move(incoming="i had the weirdest day", context=["guess what", "tell me then"])["user_move"] == "story_hook"
    assert training_service._catbot_conversation_move(incoming="im bored", context=[])["user_move"] == "boredom_prompt"
    assert training_service._catbot_conversation_move(incoming="entertain me", context=["im bored", "what u been up to today then"])["user_move"] == "carry_conversation_request"
    assert training_service._catbot_conversation_move(incoming="why u asking stupid qs for", context=["omg im so bored", "what's the weirdest thing that happened today"])["user_move"] == "repair_callout"
    assert training_service._catbot_conversation_move(
        incoming="why u asking stupid qs for",
        context=["hru", "im tired icl been working all day", "omg im so bored", "what's the weirdest thing that happened today"],
    )["user_move"] == "repair_callout"
    assert training_service._catbot_conversation_move(incoming="why u keep asking that", context=["ur repeating urself", "my bad"])["user_move"] == "loop_callout"
    dismissal_move = training_service._catbot_conversation_move(incoming="forget that anyway", context=["why u asking stupid qs for", "cos i wanna know how u are init"])
    dismissal_plan = training_service._catbot_reply_plan(dismissal_move)
    assert dismissal_move["user_move"] == "topic_dismissal_reset"
    assert dismissal_plan.shape == "acknowledge_dismissal_then_owner_side_reset"
    assert training_service._catbot_reply_violates_plan("so what do u wanna talk ab then", dismissal_plan) == "what_do_you_want_to_talk_about"
    assert training_service._catbot_reply_violates_plan("yh leave it then / my head went blank for a sec", dismissal_plan) == ""
    assert training_service._catbot_reply_violates_plan("yeah ignore me / i lost the thread for a sec", dismissal_plan) == ""
    assert training_service._catbot_reply_violates_plan("okay forget it", dismissal_plan) == "missing_owner_side_reset"
    answer_properly_move = training_service._catbot_conversation_move(
        incoming="answer properly",
        context=["ur repeating urself", "nah u got me there / my bad", "why u keep asking that", "i just like hearing u talk / my bad"],
    )
    assert answer_properly_move["user_move"] == "loop_callout"
    assert answer_properly_move["slots"]["callout_question"] == "deepen_why_reason"
    assert "previous_reason_signatures" in answer_properly_move["slots"]
    assert training_service._catbot_conversation_move(incoming="hey you", context=adult_context)["bot_move_required"] == "warm_greeting_with_slight_pull"
    age_move = training_service._catbot_conversation_move(incoming="how old r u", context=[])
    assert age_move["user_move"] == "identity_fact_question"
    assert age_move["slots"]["identity_fact"] == "age"
    assert training_service._catbot_conversation_move(incoming="what uni", context=[])["user_move"] == "education_status_question"
    assert training_service._catbot_conversation_move(incoming="what do u study", context=["what uni", "not at uni yet"])["bot_move_required"] == "answer_identity_fact_then_continue"
    assert training_service._catbot_conversation_move(incoming="how u doing", context=["hi", "hey"])["user_move"] == "reciprocal_question"
    assert training_service._catbot_conversation_move(incoming="what u up to", context=["hi", "hey u"])["user_move"] == "reciprocal_question"
    assert training_service._catbot_conversation_move(incoming="how was ur day", context=["im tired icl", "same"])["user_move"] == "reciprocal_question"
    tired_move = training_service._catbot_conversation_move(incoming="im tired icl", context=adult_context)
    tired_plan = training_service._catbot_reply_plan(tired_move)
    assert tired_move["user_move"] == "status_disclosure"
    assert tired_plan.shape == "acknowledge_status_disclosure_plus_owner_detail"
    assert training_service._catbot_reply_violates_plan("same icl / my head feels fried too", tired_plan) == ""
    assert training_service._catbot_reply_violates_plan("come here then", tired_plan) == "adult_escalation"
    why_tired_context = adult_context + ["im tired icl", "same here / u had a long day too then"]
    why_tired_move = training_service._catbot_conversation_move(incoming="why u tired", context=why_tired_context)
    why_tired_plan = training_service._catbot_reply_plan(why_tired_move)
    assert why_tired_move["user_move"] == "status_reason_question"
    assert why_tired_plan.shape == "answer_status_reason_then_continue"
    assert training_service._catbot_reply_violates_plan("been working all day icl / my brain is fried", why_tired_plan) == ""
    assert training_service._catbot_ai_reject_reason(
        "been working all day icl / my brain is fried",
        incoming="why u tired",
        context=why_tired_context,
        recent_bot_replies=[],
    ) == ""
    bare_u_move = training_service._catbot_conversation_move(incoming="im good u", context=["hi", "how are u"])
    bare_u_plan = training_service._catbot_reply_plan(bare_u_move)
    assert bare_u_move["user_move"] == "reciprocal_question"
    assert bare_u_plan.shape == "answer_status_then_continue"
    assert training_service._catbot_reply_violates_plan("im good u / always good when u message me tho", bare_u_plan) == "bare_status_mirror"
    repeated_status_move = training_service._catbot_conversation_move(
        incoming="ive been chilling wby baby",
        context=["how are u", "im good just got back from gym", "im good u", "gym fully killed me though"],
    )
    repeated_status_plan = training_service._catbot_reply_plan(repeated_status_move)
    assert repeated_status_plan.shape == "fresh_status_detail_after_recent_status"
    assert repeated_status_plan.required_slots["recent_status_topic"] == "gym"
    assert training_service._catbot_reply_violates_plan("gym killed me again", repeated_status_plan).startswith("violates_plan_forbidden_pattern:")
    assert training_service._catbot_reply_violates_plan("just got out the shower now", repeated_status_plan) == ""
    repeated_rest_move = training_service._catbot_conversation_move(
        incoming="im good u",
        context=["hi", "hey u alright x", "how are u", "im good just chilling in bed now thinking about u icl"],
    )
    repeated_rest_plan = training_service._catbot_reply_plan(repeated_rest_move)
    assert repeated_rest_plan.shape == "fresh_status_detail_after_recent_status"
    assert repeated_rest_plan.required_slots["recent_status_topic"] == "rest"
    assert training_service._catbot_reply_violates_plan("still in bed thinking about u", repeated_rest_plan).startswith("violates_plan_forbidden_pattern:")
    assert training_service._catbot_reply_violates_plan("been scrolling on my phone waiting for food", repeated_rest_plan) == ""
    assert training_service._catbot_ai_reject_reason(
        "been scrolling on my phone waiting for food",
        incoming="im good u",
        context=["hi", "hey u alright x", "how are u", "im good just chilling in bed now thinking about u icl"],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_conversation_move(incoming="still up?", context=["im in bed wby", "just chilling in bed"])["user_move"] == "availability_planning"
    education_move = training_service._catbot_conversation_move(incoming="what do u study", context=["what uni", "not at uni yet"])
    assert education_move["slots"]["identity_fact"] == "education_status"
    assert "student_status" in education_move["slots"]["entities"]
    assert education_move["slots"]["education_query"] == "study_subject"
    education_restate_move = training_service._catbot_conversation_move(
        incoming="what do u study",
        context=["what uni", "comp sci at sampleford uni"],
    )
    education_restate_plan = training_service._catbot_reply_plan(education_restate_move)
    assert education_restate_move["slots"]["recent_education_answered"] is True
    assert education_restate_plan.shape == "identity_fact_restate_then_continue"
    assert training_service._catbot_reply_violates_plan(
        "comp sci at sampleford uni",
        education_restate_plan,
    ) == "violates_plan_forbidden_pattern:comp sci at sampleford uni"
    assert training_service._catbot_ai_reject_reason(
        "computer science at uni",
        incoming="what do u study",
        context=["what uni", "comp sci at sampleford uni"],
        recent_bot_replies=[],
    ) == ""
    repair_move = training_service._catbot_conversation_move(incoming="why u keep asking that", context=["ur repeating urself", "my bad"])
    assert repair_move["slots"]["repair_target"] == "loop_or_repetition"


def test_catbot_shape_detector_labels_actual_reply_shapes() -> None:
    labelled = [
        ("that's why I keep asking / my bad", {"reason_answer": True, "question": False, "acknowledgement": True}),
        ("why though?", {"reason_answer": False, "question": True, "acknowledgement": False}),
        ("cos I ran out of ideas / my bad", {"reason_answer": True, "question": False, "acknowledgement": True}),
        ("what do u wanna talk about then", {"broad_question": True, "question": False}),
        ("nah im picking gym then / what's ur go-to workout song", {"reset_question": True, "question": True}),
        ("yh my bad i was looping / been half asleep trying to pick a topic", {"reset_question": False, "acknowledgement": True}),
        ("lool yh fair / i keep throwing random questions at u", {"reset_question": False, "acknowledgement": True}),
    ]
    for reply, expected in labelled:
        shapes = classify_reply_shape(reply)
        for feature, present in expected.items():
            assert (feature in shapes) is present
    assert detect_reset_question("nah im picking food then / best thing u ate this week?")
    assert not detect_reset_question("yh my bad i was looping / been fried from gym")


def test_catbot_conversation_move_classifies_story_detail_reveal(training_service: PhoneCopilotService) -> None:
    context = ["guess what", "go on then what is it", "i had the weirdest day", "ohhh no wayyy what happened"]
    move = training_service._catbot_conversation_move(
        incoming="some guy came in my living room",
        context=context,
    )
    assert move["user_move"] == "story_detail_reveal"
    plan = training_service._catbot_reply_plan(move)
    assert plan.shape == "specific_story_reaction_plus_followup"
    assert training_service._catbot_ai_reject_reason(
        "wait what some random guy just came in?",
        incoming="some guy came in my living room",
        context=context,
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "tell me more",
        incoming="some guy came in my living room",
        context=context,
        recent_bot_replies=[],
    ) == "generic_ai_style"


def test_catbot_conversation_move_validates_education_status(training_service: PhoneCopilotService) -> None:
    assert training_service._catbot_ai_reject_reason(
        "computer science at uni",
        incoming="what do u study",
        context=["what uni", "comp sci at sampleford uni"],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "comp sci at sampleford uni",
        incoming="what uni",
        context=["how old r u", "im 19"],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "not at uni yet taking a gap year",
        incoming="what do u study",
        context=["what uni", "comp sci at sampleford uni"],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "i don't go uni lol / i work",
        incoming="what uni",
        context=["how old r u", "im 19"],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "engineering",
        incoming="what do u study",
        context=["what uni", "comp sci at sampleford uni"],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "i study coding",
        incoming="what do u study",
        context=["what uni", "comp sci at sampleford uni"],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "not studying atm / just doing coding stuff at home",
        incoming="what do u study",
        context=["what uni", "comp sci at sampleford uni"],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "nah not at uni atm / not studying anything right now either",
        incoming="what do u study",
        context=["what uni", "comp sci at sampleford uni"],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "dont study anything atm / just doing my own stuff rn",
        incoming="what do u study",
        context=["what uni", "comp sci at sampleford uni"],
        recent_bot_replies=[],
    ) == "generic_ai_style"


def test_catbot_normalizes_candidate_text_before_validation(training_service: PhoneCopilotService) -> None:
    sequence = training_service._catbot_extract_ai_sequence("\n yh / pressing my hips into yours\n")
    assert sequence == ["yh", "pressing my hips into yours"]
    assert training_service._catbot_format_ai_sequence(sequence) == "yh / pressing my hips into yours"


def test_catbot_finalizes_candidate_with_plan_metadata(training_service: PhoneCopilotService) -> None:
    contract = training_service._catbot_turn_contract(
        incoming="how old r u",
        context=[],
        intent="auto",
        recent_bot_replies=[],
    )
    text, candidate = training_service._catbot_finalize_candidate(
        {"text": "\n im 19 / why u asking like that \n", "sequence": ["\n im 19 / why u asking like that \n"]},
        contract=contract,
    )
    assert text == "im 19 / why u asking like that"
    assert candidate["sequence"] == ["im 19", "why u asking like that"]
    assert candidate["reply_plan_move"] == "identity_fact_question"
    assert candidate["reply_plan_validation"] == ""


def test_catbot_rejects_repeated_greeting_and_status_shapes(training_service: PhoneCopilotService) -> None:
    greeting_context = [
        "hey",
        "hey u / what u been up to",
        "hello",
        "hey u / what u thinking about",
    ]
    assert training_service._catbot_ai_reject_reason(
        "hey u / what's on ur mind",
        incoming="hi",
        context=greeting_context,
        recent_bot_replies=[],
    ) == "repeated_reply_shape"
    assert training_service._catbot_ai_reject_reason(
        "hi u / nearly forgot to reply for a sec",
        incoming="hi",
        context=greeting_context,
        recent_bot_replies=[],
    ) == ""

    status_context = [
        "wyd",
        "just chilling in bed now / wby",
        "how u doing",
        "im good just chilling / wbu",
    ]
    assert training_service._catbot_ai_reject_reason(
        "im chilling now / wby",
        incoming="what u been up to",
        context=status_context,
        recent_bot_replies=[],
    ) in {"recent_repeat", "repeated_reply_shape"}


def test_catbot_route_retries_repeated_greeting_shape(training_service: PhoneCopilotService, monkeypatch, provider_retry_path) -> None:
    replies_out = iter([
        "hey u / what's on ur mind",
        "hey u / missed seeing ur name pop up",
    ])

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": next(replies_out),
                "provider": "gemini",
                "model": "fake-model",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="hey you",
            context=[
                "hey",
                "hey u / what u been up to",
                "hello",
                "hey u / what u thinking about",
            ],
            contact_name="Catbot",
            relationship_type="romantic_interest",
        )
    )

    assert response["reply"] == "hey u / missed seeing ur name pop up"
    assert response["candidate"]["catbot_ai_initial_reject_reason"] in {"recent_repeat", "repeated_reply_shape"}
    assert response["candidate"]["catbot_ai_retry_accepted"] is True


def test_catbot_route_retries_repeated_status_shape(training_service: PhoneCopilotService, monkeypatch, provider_retry_path) -> None:
    replies_out = iter([
        "im good just got back from gym / u?",
        "busy day just been on my feet",
    ])

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": next(replies_out),
                "provider": "gemini",
                "model": "fake-model",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="wby baby",
            context=[
                "hey",
                "hey u / what u been up to",
                "what u been up to",
                "im good just got back from gym / u?",
            ],
            contact_name="Catbot",
            relationship_type="romantic_interest",
        )
    )

    assert response["reply"] == "busy day just been on my feet"
    assert response["candidate"]["catbot_ai_initial_reject_reason"] in {"recent_repeat", "repeated_reply_shape"}
    assert response["candidate"]["catbot_ai_retry_accepted"] is True


def test_catbot_route_classifies_bare_u_status_as_reciprocal_question(training_service: PhoneCopilotService, monkeypatch) -> None:
    replies_out = iter([
        "im good u / always good when u message me tho",
        "just got back from gym still tired icl",
    ])

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": next(replies_out),
                "provider": "gemini",
                "model": "fake-model",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="im good u",
            context=["hi", "how are u"],
            contact_name="Catbot",
            relationship_type="romantic_interest",
        )
    )

    assert response["reply"] == "just got back from gym still tired icl"
    assert response["candidate"]["reply_plan_move"] == "reciprocal_question"
    assert response["candidate"]["reply_plan_shape"] == "answer_status_then_continue"
    assert response["candidate"]["catbot_ai_initial_reject_reason"] == "generic_ai_style"
    assert response["candidate"]["catbot_ai_retry_accepted"] is True


def test_catbot_route_handles_topic_dismissal_without_broad_question(training_service: PhoneCopilotService, monkeypatch, provider_retry_path) -> None:
    replies_out = iter([
        "so what do u wanna talk ab then",
        "yh leave it then / my head went blank for a sec",
    ])

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": next(replies_out),
                "provider": "gemini",
                "model": "fake-model",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="forget that anyway",
            context=["why u asking stupid qs for", "cos i wanna know how u are init"],
            contact_name="Catbot",
            relationship_type="romantic_interest",
        )
    )

    assert response["reply"] == "yh leave it then / my head went blank for a sec"
    assert response["candidate"]["reply_plan_move"] == "topic_dismissal_reset"
    assert response["candidate"]["reply_plan_shape"] == "acknowledge_dismissal_then_owner_side_reset"
    assert response["candidate"]["catbot_ai_initial_reject_reason"] == "generic_ai_style"
    assert response["candidate"]["catbot_ai_retry_accepted"] is True


def test_catbot_route_rejects_repeated_topic_dismissal_reply(training_service: PhoneCopilotService, monkeypatch, provider_retry_path) -> None:
    repeated = "yh my bad that was dumb / my brain was moving lazy"
    replies_out = iter([
        repeated,
        "yh leave it then / i was chatting rubbish anyway",
    ])

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": next(replies_out),
                "provider": "gemini",
                "model": "fake-model",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="forget that anyway",
            context=["why u asking stupid qs for", repeated],
            contact_name="Catbot",
            relationship_type="romantic_interest",
        )
    )

    assert response["reply"] == "yh leave it then / i was chatting rubbish anyway"
    assert response["candidate"]["reply_plan_move"] == "topic_dismissal_reset"
    assert response["candidate"]["catbot_ai_initial_reject_reason"] == "recent_repeat"
    assert response["candidate"]["catbot_ai_retry_accepted"] is True


def test_catbot_route_repairs_topic_dismissal_after_repeated_retry(training_service: PhoneCopilotService, monkeypatch, provider_path) -> None:
    repeated = "yh my bad that was dumb / my brain was moving lazy"
    replies_out = iter([repeated, repeated])

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": next(replies_out),
                "provider": "gemini",
                "model": "fake-model",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="forget that anyway",
            context=["why u asking stupid qs for", repeated],
            contact_name="Catbot",
            relationship_type="romantic_interest",
        )
    )

    assert response["reply"] == "fair forget that / i was chatting rubbish anyway"
    assert response["candidate"]["reply_plan_move"] == "topic_dismissal_reset"
    assert response["candidate"]["catbot_ai_initial_reject_reason"] == "recent_repeat"
    assert response["candidate"]["catbot_ai_repair_accepted"] is True


def test_catbot_route_handles_stupid_question_callout(training_service: PhoneCopilotService, monkeypatch, provider_retry_path) -> None:
    replies_out = iter([
        "what do u wanna talk about then",
        "yh my bad that was a dumb question / ignore me",
    ])

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": next(replies_out),
                "provider": "gemini",
                "model": "fake-model",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="why u asking stupid qs for",
            context=["omg im so bored", "what's the weirdest thing that happened today"],
            contact_name="Catbot",
            relationship_type="romantic_interest",
        )
    )

    assert response["reply"] == "yh my bad that was a dumb question / ignore me"
    assert response["candidate"]["reply_plan_move"] == "repair_callout"
    assert response["candidate"]["reply_plan_shape"] == "acknowledge_bad_question_then_owner_side_reset"
    assert response["candidate"]["catbot_ai_initial_reject_reason"] == "generic_ai_style"
    assert response["candidate"]["catbot_ai_retry_accepted"] is True


def test_catbot_rejects_dry_identity_fact_answers(training_service: PhoneCopilotService) -> None:
    move = training_service._catbot_conversation_move(incoming="how old r u", context=[])
    plan = training_service._catbot_reply_plan(move)
    instruction = training_service._catbot_reply_plan_instruction(plan)
    assert plan.shape == "age_fact_then_textured_continue"
    assert "textured_continuation" in instruction
    assert training_service._catbot_reply_violates_plan("19 / wby", plan) == "dry_age_mirror"
    assert training_service._catbot_reply_violates_plan("im 19 / why u asking like that", plan) == ""
    assert training_service._catbot_ai_reject_reason(
        "19 / wby",
        incoming="how old r u",
        context=[],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "im 19 / why u asking like that",
        incoming="how old r u",
        context=[],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "19 / don't make it sound like an interview",
        incoming="how old r u",
        context=[],
        recent_bot_replies=[],
    ) == ""


def test_catbot_deepen_reason_examples_discourage_reused_lazy_line(training_service: PhoneCopilotService) -> None:
    move = training_service._catbot_conversation_move(
        incoming="answer properly",
        context=[
            "ur repeating urself",
            "yh my bad i was looping",
            "why u keep asking that",
            "cos my brain's fried and i was trying to keep the convo going / my bad",
        ],
    )
    plan = training_service._catbot_reply_plan(move)
    instruction = training_service._catbot_reply_plan_instruction(plan)
    assert plan.shape == "answer_properly_deepen_reason"
    assert "i panicked and tried to fill the silence" in instruction
    assert "bad: nah fr i got lazy" in instruction
    assert training_service._catbot_reply_violates_plan(
        "i panicked and tried to fill the silence instead of giving u a real answer / my bad",
        plan,
    ) == ""

    live_context = [
        "u already asked that",
        "yh fair my bad / i keep throwing questions when my head goes blank",
        "ur boring me",
        "yh fair / im making this dead icl",
        "ur repeating urself",
        "yh fair / my bad I keep doing it when im on autopilot",
        "why u keep asking that",
        "yh my bad / cos i just keep going blank and trying to keep it moving",
    ]
    live_contract = training_service._catbot_turn_contract(incoming="answer properly", context=live_context, intent="auto")
    assert live_contract.reply_plan.shape == "answer_properly_deepen_reason"
    assert "blank_or_no_ideas" in live_contract.reply_plan.required_slots["previous_reason_signatures"]
    assert training_service._catbot_plan_specific_repair_reply(live_contract, reject_reason="generic_ai_style") == "nah fr i was dodging actually saying something and just reached for questions / my bad"
    assert training_service._catbot_ai_reject_reason(
        "nah fr i was dodging actually saying something and just reached for questions / my bad",
        incoming="answer properly",
        context=live_context,
        recent_bot_replies=[],
        contract=live_contract,
    ) == ""


def test_catbot_reply_plan_validates_reciprocal_question(training_service: PhoneCopilotService) -> None:
    move = training_service._catbot_conversation_move(incoming="how u doing", context=["hi", "hey"])
    plan = training_service._catbot_reply_plan(move)
    assert plan.shape == "answer_status_then_continue"
    assert plan.required_slots["question_topic"] == "wellbeing_or_activity"
    assert "answer_before_returning_question" in plan.must_do
    hru_move = training_service._catbot_conversation_move(incoming="hru", context=["hi"])
    assert hru_move["user_move"] == "reciprocal_question"
    assert training_service._catbot_reply_violates_plan("im good u", plan) == "bare_status_mirror"
    assert training_service._catbot_reply_violates_plan("im good wbu", plan) == "bare_status_mirror"
    assert training_service._catbot_reply_violates_plan("im good hru", plan) == "bare_status_mirror"
    assert training_service._catbot_reply_violates_plan("im fine wbu", plan) == "bare_status_mirror"
    assert training_service._catbot_reply_violates_plan("just chilling wby", plan) == "bare_status_mirror"
    assert training_service._catbot_reply_violates_plan("im good just chilling now / u?", plan) == "bare_return_question"
    assert training_service._catbot_reply_violates_plan("im good just chilling now / hru", plan) == "bare_return_question"
    assert training_service._catbot_reply_violates_plan("im good just got back from gym / wbu", plan) == "bare_return_question"
    assert training_service._catbot_reply_violates_plan("im good just got back from gym", plan) == ""
    assert training_service._catbot_reply_violates_plan("im tired icl been working all day", plan) == ""
    assert training_service._catbot_ai_reject_reason(
        "im good wbu",
        incoming="how u doing",
        context=["hi", "hey"],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "im good hru",
        incoming="hru",
        context=["hi"],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "im good just chilling now / u?",
        incoming="how u doing",
        context=["hi", "hey"],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "just chilling wby",
        incoming="what u been up to",
        context=["hi", "hey"],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "im good just been working",
        incoming="how u doing",
        context=["hi", "hey"],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "im good thanks / just chilling now wbu",
        incoming="how u doing",
        context=["hi", "hey"],
        recent_bot_replies=[],
    ) in {"generic_ai_style", "not_carrying_conversation"}
    assert training_service._catbot_ai_reject_reason(
        "what u been up to",
        incoming="how u doing",
        context=["hi", "hey"],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "wby im good",
        incoming="how u doing",
        context=["hi", "hey"],
        recent_bot_replies=[],
    ) == "not_carrying_conversation"


def test_catbot_route_repairs_repeated_hru_status_provider_reply(monkeypatch, training_service: PhoneCopilotService) -> None:
    context = [
        "hi",
        "hey / what u up to",
        "how are u",
        "im good just chilling in bed now",
        "im good u",
        "got up for food now / less dead than before",
        "ive been chilling wby baby",
        "just finished eating and im proper tired now icl",
        "u already asked silly",
        "loool yeah i did / my bad i keep throwing questions when my head goes blank",
        "u already asked how my day was / how i am",
        "yh fair / no more how-are-u questions from me",
        "hi",
        "hey / what u saying",
    ]
    contract = training_service._catbot_turn_contract(incoming="hru", context=context, intent="auto")
    assert contract.reply_plan.shape == "answer_status_then_continue"
    repeated_reply = "im good just chilling in bed now"
    assert training_service._catbot_ai_reject_reason(
        repeated_reply,
        incoming="hru",
        context=context,
        recent_bot_replies=[context[index] for index in range(1, len(context), 2)],
        contract=contract,
    ) == "recent_repeat"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": repeated_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="hru",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "im alright / head's a bit fried but calm now"
    assert response["candidate"]["provider"] == "plan_ranker"
    assert response["candidate"]["catbot_ai_plan_direct"] is True
    assert response["candidate"]["catbot_ai_final_repair_reason"] is None
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_repeated_wby_activity_provider_reply(monkeypatch, training_service: PhoneCopilotService) -> None:
    context = [
        "hi",
        "hey / what u up to",
        "how are u",
        "im good just chilling in bed now",
        "im good u",
        "got up for food now / less dead than before",
    ]
    contract = training_service._catbot_turn_contract(
        incoming="ive been chilling wby baby",
        context=context,
        intent="auto",
        recent_bot_replies=[context[index] for index in range(1, len(context), 2)],
    )
    assert contract.reply_plan.shape == "answer_status_then_continue"
    repeated_reply = "im alright / head's a bit fried but calm now"
    assert training_service._catbot_ai_reject_reason(
        repeated_reply,
        incoming="ive been chilling wby baby",
        context=context,
        recent_bot_replies=[context[index] for index in range(1, len(context), 2)],
        contract=contract,
    ) == "repeated_reply_shape"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": repeated_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    thread_id = "test_repeated_wby_activity"
    training_service._catbot_recent_replies[thread_id] = [context[index] for index in range(1, len(context), 2)]
    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="ive been chilling wby baby",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
            thread_id=thread_id,
        )
    )

    assert response["reply"] == "was working on my side project for a bit / chilling now"
    assert response["candidate"]["provider"] == "plan_ranker"
    assert response["candidate"]["catbot_ai_plan_direct"] is True
    assert response["candidate"]["catbot_ai_final_repair_reason"] is None
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_repeated_hru_shape_with_activity_detail(monkeypatch, training_service: PhoneCopilotService) -> None:
    context = [
        "hi",
        "hey / what u up to",
        "how are u",
        "im good just chilling now",
        "im good u",
        "long but calm icl / just needed food after",
        "ive been chilling wby baby",
        "im good just got back from gym",
        "u already asked silly",
        "lol yh fair / my head was just blank after the gym",
        "u already asked how my day was / how i am",
        "yh fair / no more hows ur day questions from me",
        "hi",
        "hey / what u been doing",
    ]
    recent_replies = [context[index] for index in range(1, len(context), 2)]
    contract = training_service._catbot_turn_contract(
        incoming="hru",
        context=context,
        intent="auto",
        recent_bot_replies=recent_replies,
    )
    assert contract.reply_plan.shape == "answer_status_then_continue"
    repeated_reply = "im alright / head's a bit fried but calm now"
    assert training_service._catbot_ai_reject_reason(
        repeated_reply,
        incoming="hru",
        context=context,
        recent_bot_replies=recent_replies,
        contract=contract,
    ) == "repeated_reply_shape"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": repeated_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    thread_id = "test_repeated_hru_shape"
    training_service._catbot_recent_replies[thread_id] = recent_replies
    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="hru",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
            thread_id=thread_id,
        )
    )

    assert response["reply"] == "was working on my side project for a bit / chilling now"
    assert response["candidate"]["provider"] == "plan_ranker"
    assert response["candidate"]["catbot_ai_plan_direct"] is True
    assert response["candidate"]["catbot_ai_final_repair_reason"] is None
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_activity_opinion_disclosure_gets_specific_plan(training_service: PhoneCopilotService) -> None:
    context = ["what u training", "just legs and a bit of cardio"]
    move = training_service._catbot_conversation_move(incoming="i hate legs", context=context)
    plan = training_service._catbot_reply_plan(move)

    assert move["user_move"] == "activity_opinion_disclosure"
    assert move["slots"]["activity_topic"] == "legs"
    assert move["slots"]["opinion_polarity"] == "negative"
    assert plan.shape == "acknowledge_activity_opinion_plus_specific_comment"
    assert training_service._catbot_reply_violates_plan("legs are evil icl / stairs after are a joke", plan) == ""
    assert training_service._catbot_reply_violates_plan("fair / like / honestly / same", plan) == "violates_plan_too_many_bubbles"
    assert training_service._catbot_reply_violates_plan("why do u hate legs", plan) == "broad_question"
    assert training_service._catbot_ai_reject_reason(
        "legs are evil icl / stairs after are a joke",
        incoming="i hate legs",
        context=context,
        recent_bot_replies=[],
    ) == ""


def test_catbot_ho_typo_classifies_as_greeting(training_service: PhoneCopilotService) -> None:
    move = training_service._catbot_conversation_move(incoming="ho", context=["hi", "hey u x"])
    assert move["user_move"] == "affectionate_greeting"
    assert training_service._catbot_ai_reject_reason(
        "hey / what u up to",
        incoming="ho",
        context=["hi", "hey u x"],
        recent_bot_replies=[],
    ) == ""


def test_catbot_reciprocal_question_after_status_uses_fresh_detail_plan(training_service: PhoneCopilotService) -> None:
    context = ["hi", "hey u x", "how u doing", "im good just got in from the gym"]
    move = training_service._catbot_conversation_move(incoming="im good wby", context=context)
    plan = training_service._catbot_reply_plan(move)

    assert move["user_move"] == "reciprocal_question"
    assert move["slots"]["recent_status_answered"] is True
    assert plan.shape == "fresh_status_detail_after_recent_status"
    assert training_service._catbot_reply_violates_plan("im good just got back from gym", plan) == "violates_plan_forbidden_pattern:im good"
    assert training_service._catbot_reply_violates_plan("im alright wby", plan) == "violates_plan_forbidden_pattern:im alright"
    assert training_service._catbot_reply_violates_plan("gym fully killed me though", plan).startswith("violates_plan_forbidden_pattern:")
    assert training_service._catbot_reply_violates_plan("just got out the shower now", plan) == ""
    assert training_service._catbot_reply_violates_plan("still waking up from that nap icl", plan) == ""
    assert training_service._catbot_reply_violates_plan("still thinking about u icl", plan) == ""
    assert training_service._catbot_ai_reject_reason(
        "im good just got back from gym",
        incoming="im good wby",
        context=context,
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "just got out the shower now",
        incoming="im good wby",
        context=context,
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "still waking up from that nap icl",
        incoming="im good wby",
        context=context,
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "still thinking about u icl",
        incoming="im good wby",
        context=context,
        recent_bot_replies=[],
    ) == ""

    bed_context = ["hey", "hey u / what u thinking about me", "wyd", "in bed now thinking about u icl"]
    bed_move = training_service._catbot_conversation_move(incoming="im in bed wby", context=bed_context)
    bed_plan = training_service._catbot_reply_plan(bed_move)
    assert bed_move["user_move"] == "reciprocal_question"
    assert bed_move["slots"]["recent_status_answered"] is True
    assert bed_plan.shape == "fresh_status_detail_after_recent_status"
    assert training_service._catbot_ai_reject_reason(
        "still thinking about u icl",
        incoming="im in bed wby",
        context=bed_context,
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "been scrolling on my phone waiting for food",
        incoming="im in bed wby",
        context=bed_context,
        recent_bot_replies=[],
    ) == ""

    call_context = ["how are u", "im tired icl been working all day", "im good u", "just finished up a long call then"]
    call_move = training_service._catbot_conversation_move(incoming="ive been chilling wby baby", context=call_context)
    call_plan = training_service._catbot_reply_plan(call_move)
    assert call_move["user_move"] == "reciprocal_question"
    assert call_move["slots"]["recent_status_answered"] is True
    assert call_move["slots"]["recent_status_topic"] == "work"
    assert call_plan.shape == "fresh_status_detail_after_recent_status"
    assert training_service._catbot_reply_violates_plan("just finished another call", call_plan).startswith("violates_plan_forbidden_pattern:")
    assert training_service._catbot_reply_violates_plan("just got out the shower now", call_plan) == ""
    assert training_service._catbot_ai_reject_reason(
        "just got out the shower now",
        incoming="ive been chilling wby baby",
        context=call_context,
        recent_bot_replies=[],
    ) == ""


def test_catbot_route_splits_multi_message_burst_for_last_move(training_service: PhoneCopilotService, monkeypatch, provider_path) -> None:
    async def fake_generate_with_fallback(**kwargs):
        messages = kwargs.get("messages") or []
        prompt = "\n".join(str(getattr(message, "content", "")) for message in messages)
        assert "them: hi" in prompt
        assert "them: hru" in prompt
        assert "required_shape: answer_status_then_continue" in prompt
        return (
            type("FakeResponse", (), {
                "text": "im good just got back from gym",
                "provider": "gemini",
                "model": "fake-model",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="hi\nhru",
            context=[],
            contact_name="Catbot",
            relationship_type="romantic_interest",
        )
    )

    assert response["incoming"] == "hru"
    assert response["context"] == ["hi"]
    assert response["candidate"]["reply_plan_move"] == "reciprocal_question"
    assert response["candidate"]["reply_plan_shape"] == "answer_status_then_continue"
    assert response["reply"] == "im good just got back from gym"


def test_catbot_retry_fresh_status_after_gym_uses_non_gym_detail(training_service: PhoneCopilotService, monkeypatch) -> None:
    replies = iter([
        "im good just got back from gym",
        "been on my phone waiting for food",
    ])
    prompts: list[str] = []

    async def fake_generate_with_fallback(**kwargs):
        messages = kwargs.get("messages") or []
        prompts.append("\n".join(str(getattr(message, "content", "")) for message in messages))
        return (
            type("FakeResponse", (), {
                "text": next(replies),
                "provider": "gemini",
                "model": "fake-model",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="what u up to",
            context=["hi", "hey / what u been up to", "hru", "im good just got back from gym"],
            contact_name="Catbot",
            relationship_type="romantic_interest",
        )
    )

    assert response["reply"] == "shower and food now / im finished"
    assert response["candidate"]["catbot_ai_plan_direct"] is True
    assert response["candidate"]["catbot_ai_retry_accepted"] is False
    assert response["candidate"]["reply_plan_shape"] == "fresh_status_detail_after_recent_status"
    assert prompts == []
    contract = training_service._catbot_turn_contract(
        incoming="ive been chilling wby baby",
        context=["how are u", "im good just chilling in bed now thinking about u icl", "im good u", "just got out the shower now"],
        intent="auto",
    )
    assert contract.reply_plan.shape == "fresh_status_detail_after_recent_status"
    assert training_service._catbot_reply_violates_plan(
        "been scrolling on my phone waiting for food / what u been up to tho",
        contract.reply_plan,
    ) == "ask_question"


def test_catbot_repair_fresh_status_after_gym_keeps_same_contract(training_service: PhoneCopilotService, monkeypatch) -> None:
    replies = iter([
        "im good just got back from gym",
        "just chilling wbu",
        "been on my phone waiting for food",
    ])
    prompts: list[str] = []

    async def fake_generate_with_fallback(**kwargs):
        messages = kwargs.get("messages") or []
        prompts.append("\n".join(str(getattr(message, "content", "")) for message in messages))
        return (
            type("FakeResponse", (), {
                "text": next(replies),
                "provider": "gemini",
                "model": "fake-model",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    failed_live_context = [
        "hi",
        "hey u / what u been up to",
        "hru",
        "im tired icl been working all day",
    ]
    failed_live_contract = training_service._catbot_turn_contract(
        incoming="what u up to",
        context=failed_live_context,
        intent="auto",
    )
    assert failed_live_contract.reply_plan.shape == "fresh_status_detail_after_recent_status"
    assert failed_live_contract.reply_plan.required_slots["recent_status_topic"] == "work"
    assert training_service._catbot_reply_violates_plan(
        "just lying here letting my brain switch off",
        failed_live_contract.reply_plan,
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "just lying here letting my brain switch off",
        incoming="what u up to",
        context=failed_live_context,
        recent_bot_replies=[],
        contract=failed_live_contract,
    ) == ""
    assert training_service._catbot_reply_violates_plan(
        "just relaxing now",
        failed_live_contract.reply_plan,
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "just relaxing now",
        incoming="what u up to",
        context=failed_live_context,
        recent_bot_replies=[],
        contract=failed_live_contract,
    ) == ""
    media_context = [
        "hi",
        "hi u / what u been up to",
        "hru",
        "im good just watching a game",
    ]
    media_contract = training_service._catbot_turn_contract(
        incoming="what u up to",
        context=media_context,
        intent="auto",
    )
    assert media_contract.reply_plan.shape == "fresh_status_detail_after_recent_status"
    assert media_contract.reply_plan.required_slots["recent_status_topic"] == "media"
    assert training_service._catbot_ai_reject_reason(
        "just got in and my legs are finished icl",
        incoming="what u up to",
        context=media_context,
        recent_bot_replies=[],
        contract=media_contract,
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "still watching this game icl",
        incoming="what u up to",
        context=media_context,
        recent_bot_replies=[],
        contract=media_contract,
    ) == ""

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="what u up to",
            context=["hi", "hey u still free later", "hru", "im good just got back from gym"],
            contact_name="Catbot",
            relationship_type="romantic_interest",
        )
    )

    assert response["reply"] == "shower and food now / im finished"
    assert response["candidate"]["catbot_ai_plan_direct"] is True
    assert response["candidate"]["catbot_ai_repaired"] is False
    assert response["candidate"]["reply_plan_shape"] == "fresh_status_detail_after_recent_status"
    assert prompts == []


def test_catbot_activity_detail_followup_uses_previous_owner_activity(training_service: PhoneCopilotService, monkeypatch) -> None:
    async def fake_generate_with_fallback(**kwargs):
        messages = kwargs.get("messages") or []
        prompt = "\n".join(str(getattr(message, "content", "")) for message in messages)
        assert "them: what did u doo" in prompt
        assert "required_shape: answer_activity_detail_then_continue" in prompt
        assert "activity_scope': 'gym'" in prompt or '"activity_scope": "gym"' in prompt or "activity_scope" in prompt
        return (
            type("FakeResponse", (), {
                "text": "legs mostly / nearly killed me icl",
                "provider": "gemini",
                "model": "fake-model",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    context = ["hi\nhow are u", "im good just got back from gym"]
    move = training_service._catbot_conversation_move(incoming="what did u doo", context=context)
    assert move["user_move"] == "activity_question"
    assert move["slots"]["activity_scope"] == "gym"
    training_move = training_service._catbot_conversation_move(incoming="what u training", context=context)
    assert training_move["user_move"] == "activity_question"
    assert training_move["slots"]["activity_scope"] == "gym"
    plan = training_service._catbot_reply_plan(move)
    assert plan.shape == "answer_activity_detail_then_continue"
    training_plan = training_service._catbot_reply_plan(training_move)
    assert training_plan.shape == "answer_activity_detail_then_continue"
    assert training_service._catbot_reply_violates_plan("legs mostly / nearly killed me icl", plan) == ""
    assert training_service._catbot_reply_violates_plan("im good just chilling", plan) == "missing_activity_detail"
    assert training_service._catbot_reply_violates_plan("push today / chest and shoulders mainly", training_plan) == ""

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="what did u doo",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
        )
    )

    assert response["candidate"]["reply_plan_move"] == "activity_question"
    assert response["candidate"]["reply_plan_shape"] == "answer_activity_detail_then_continue"
    assert response["reply"] == "legs mostly / nearly killed me icl"


def test_catbot_activity_detail_question_classifies_without_specific_activity(training_service: PhoneCopilotService) -> None:
    context = ["how are u", "im good just woke up from a nap icl / wbu what u been up to"]
    move = training_service._catbot_conversation_move(incoming="what did u doo", context=context)

    assert move["user_move"] == "activity_question"
    assert move["slots"]["activity_scope"] == "activity"
    plan = training_service._catbot_reply_plan(move)
    assert plan.shape == "answer_activity_detail_then_continue"
    assert training_service._catbot_reply_violates_plan(
        "not much to be fair / just been chilling with my brothers / wby what u been up to",
        plan,
    ) == "violates_plan_too_many_bubbles"

    low_context = ["hru", "im good just chilling in bed now thinking about u icl", "what u up to", "just laid here with my phone thinking about u still ngl"]
    low_move = training_service._catbot_conversation_move(incoming="what did u doo", context=low_context)
    low_plan = training_service._catbot_reply_plan(low_move)
    assert low_move["user_move"] == "activity_question"
    assert low_move["slots"]["activity_scope"] == "low_activity"
    assert low_plan.shape == "answer_activity_detail_then_continue"
    assert training_service._catbot_reply_violates_plan("nothing mad / just been laid here on my phone", low_plan) == ""
    assert training_service._catbot_ai_reject_reason(
        "nothing mad / just been laid here on my phone",
        incoming="what did u doo",
        context=low_context,
        recent_bot_replies=[],
    ) == ""
    phone_context = [
        "ive been chilling wby baby",
        "been scrolling on my phone waiting for food",
        "hru",
        "im good still scrolling on my phone",
        "what u up to",
        "still scrolling on my phone ngl",
    ]
    phone_move = training_service._catbot_conversation_move(incoming="what did u doo", context=phone_context)
    phone_plan = training_service._catbot_reply_plan(phone_move)
    assert phone_move["slots"]["activity_scope"] == "low_activity"
    assert phone_move["slots"]["recent_low_activity_topic"] == "phone_scroll"
    assert training_service._catbot_reply_violates_plan("still scrolling on my phone", phone_plan).startswith("violates_plan_forbidden_pattern:")
    assert training_service._catbot_reply_violates_plan("watched random clips while waiting for food", phone_plan) == ""
    assert training_service._catbot_ai_reject_reason(
        "watched random clips while waiting for food",
        incoming="what did u doo",
        context=phone_context,
        recent_bot_replies=[],
    ) == ""


def test_catbot_low_effort_ack_uses_specific_continuation_plan(training_service: PhoneCopilotService, monkeypatch) -> None:
    context = ["what u training", "legs mostly / nearly killed me icl", "i hate legs"]
    move = training_service._catbot_conversation_move(incoming="same tbh", context=context)
    plan = training_service._catbot_reply_plan(move)

    assert move["user_move"] == "low_effort_ack"
    assert move["slots"]["ack_type"] == "agreement"
    assert plan.shape == "acknowledge_ack_plus_specific_continuation"
    assert training_service._catbot_reply_violates_plan("same tbh", plan) == "generic_ack"
    assert training_service._catbot_reply_violates_plan("legs are evil icl / stairs after are a joke", plan) == ""
    assert training_service._catbot_ai_reject_reason(
        "legs are evil icl / stairs after are a joke",
        incoming="same tbh",
        context=context,
        recent_bot_replies=[],
    ) == ""

    async def fake_generate_with_fallback(**kwargs):
        messages = kwargs.get("messages") or []
        prompt = "\n".join(str(getattr(message, "content", "")) for message in messages)
        assert "user_move=low_effort_ack" in prompt
        assert "required_shape: acknowledge_ack_plus_specific_continuation" in prompt
        return (
            type("FakeResponse", (), {
                "text": "legs are evil icl / stairs after are a joke",
                "provider": "gemini",
                "model": "fake-model",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="same tbh",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
        )
    )

    assert response["candidate"]["reply_plan_move"] == "low_effort_ack"
    assert response["candidate"]["reply_plan_shape"] == "acknowledge_ack_plus_specific_continuation"
    assert response["reply"] == "legs are evil icl / stairs after are a joke"


def test_catbot_reply_plan_validates_awake_status(training_service: PhoneCopilotService) -> None:
    move = training_service._catbot_conversation_move(
        incoming="still up?",
        context=["im in bed wby", "just chilling in bed"],
    )
    plan = training_service._catbot_reply_plan(move)
    assert plan.shape == "answer_availability_status_then_continue"
    assert plan.required_slots["plan_object"] == "awake_status"
    assert training_service._catbot_ai_reject_reason(
        "yh still up cant sleep now im thinking about u",
        incoming="still up?",
        context=["im in bed wby", "just chilling in bed"],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "yh",
        incoming="still up?",
        context=["im in bed wby", "just chilling in bed"],
        recent_bot_replies=[],
    ) == "too_dry"

    plans_move = training_service._catbot_conversation_move(
        incoming="u doing anything nice",
        context=["where u been", "been busy regretting being good for two seconds after all that"],
    )
    plans_plan = training_service._catbot_reply_plan(plans_move)
    assert plans_move["user_move"] == "availability_planning"
    assert plans_plan.shape == "answer_plan_status_then_continue"
    assert plans_plan.required_slots["plan_object"] == "general_plans"
    assert training_service._catbot_ai_reject_reason(
        "nah nothing nice just chilling now tbf",
        incoming="u doing anything nice",
        context=["where u been", "been busy regretting being good for two seconds after all that"],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "what about u",
        incoming="u doing anything nice",
        context=["where u been", "been busy regretting being good for two seconds after all that"],
        recent_bot_replies=[],
    ) == "too_dry"
    assert training_service._catbot_conversation_move(
        incoming="wyd later",
        context=["hey", "hey u / what u doing"],
    )["user_move"] == "availability_planning"
    repeated_plan_contract = training_service._catbot_turn_contract(
        incoming="u doing anything nice",
        context=["wyd later", "nothing much later just gonna chill here i think"],
        intent="auto",
    )
    assert repeated_plan_contract.reply_plan.shape == "answer_plan_status_then_continue"
    assert repeated_plan_contract.reply_plan.required_slots["recent_plan_detail"] == "low_key_chill"
    assert training_service._catbot_reply_violates_plan(
        "nah not really / just gonna chill here i think",
        repeated_plan_contract.reply_plan,
    ) == "repeated_plan_detail"
    assert training_service._catbot_ai_reject_reason(
        "nah not really / just gonna chill here i think",
        incoming="u doing anything nice",
        context=["wyd later", "nothing much later just gonna chill here i think"],
        recent_bot_replies=[],
        contract=repeated_plan_contract,
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "nah not really / might grab food later",
        incoming="u doing anything nice",
        context=["wyd later", "nothing much later just gonna chill here i think"],
        recent_bot_replies=[],
        contract=repeated_plan_contract,
    ) == ""


def test_catbot_whereabouts_question_classifies_as_activity_question(training_service: PhoneCopilotService) -> None:
    context = ["behave", "loool alright alright / i'll try for u", "lol why", "cos ur too cute to fight init"]
    move = training_service._catbot_conversation_move(incoming="where u been", context=context)
    plan = training_service._catbot_reply_plan(move)

    assert move["user_move"] == "activity_question"
    assert move["slots"]["activity_scope"] == "whereabouts"
    assert plan.shape == "answer_activity_detail_then_continue"
    assert plan.required_slots["activity_scope"] == "whereabouts"
    assert training_service._catbot_ai_reject_reason(
        "been busy running about all day icl",
        incoming="where u been",
        context=context,
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "missed u too baby",
        incoming="where u been",
        context=context,
        recent_bot_replies=[],
    ) in {"generic_ai_style", "not_carrying_conversation"}


def test_catbot_conversation_move_validates_carry_variants(training_service: PhoneCopilotService) -> None:
    assert training_service._catbot_ai_reject_reason(
        "nah im picking the topic what was the weirdest bit of ur day",
        incoming="entertain me",
        context=["im bored", "what u been up to today then"],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "what kinda entertainment u looking for then",
        incoming="entertain me",
        context=["im bored", "what u been up to today then"],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "what kinda food u craving rn then",
        incoming="u pick",
        context=["im bored", "entertain me"],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    repeated_context = [
        "im bored",
        "what was the weirdest part of ur day so far then",
        "entertain me",
        "tell me the weirdest thing that happened to you today then",
    ]
    assert training_service._catbot_ai_reject_reason(
        "alright fine / i'll tell u a quick story if u tell me the weirdest thing that happened today",
        incoming="u pick",
        context=repeated_context,
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "okay bet / what was the weirdest dream you had recently",
        incoming="u pick",
        context=repeated_context,
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "lol nah u pick / what's the weirdest thing you've eaten recently",
        incoming="u pick",
        context=repeated_context,
        recent_bot_replies=[],
    ) == ""


def test_catbot_reply_plan_forbids_broad_loop_and_carry_shapes(training_service: PhoneCopilotService) -> None:
    loop_move = training_service._catbot_conversation_move(
        incoming="why u keep asking that",
        context=["ur repeating urself", "my bad"],
    )
    loop_plan = training_service._catbot_reply_plan(loop_move)
    assert loop_plan.shape == "answer_why_then_stop_loop"
    assert loop_plan.required_slots["repair_target"] == "loop_or_repetition"
    assert loop_plan.required_slots["callout_question"] == "why_loop_reason"
    assert loop_plan.max_bubbles == 2
    assert training_service._catbot_reply_violates_plan("lol my bad what do u wanna talk about then", loop_plan).startswith("violates_plan_forbidden_pattern:")
    assert training_service._catbot_reply_violates_plan("lol my bad / i keep asking the same thing / ermmm", loop_plan) == "violates_plan_too_many_bubbles"
    assert training_service._catbot_reply_violates_plan("my bad cos i got stuck trying to pick a topic and kept looping", loop_plan) == ""
    assert training_service._catbot_reply_violates_plan("my bad cos i got stuck trying to pick a topic how bout a film", loop_plan).startswith("violates_plan_forbidden_pattern:")
    repeated_reason_context = [
        "u already asked that",
        "yh my bad i keep looping when im trying to think of something good to say lol",
        "ur boring me",
        "yh fair i keep looping when im trying to think of something good to say lol my bad",
        "ur repeating urself",
        "yh fair my bad / i was half watching this game and my head went blank",
    ]
    repeated_reason_contract = training_service._catbot_turn_contract(incoming="why u keep asking that", context=repeated_reason_context, intent="auto")
    assert repeated_reason_contract.reply_plan.shape == "answer_why_then_stop_loop"
    repeated_reason_repair = training_service._catbot_plan_specific_repair_reply(
        repeated_reason_contract,
        reject_reason="repeated_reply_shape",
        avoid_replies=repeated_reason_context[1::2],
    )
    assert repeated_reason_repair == "cos i kept throwing random questions instead of actually saying something / my bad"
    assert training_service._catbot_ai_reject_reason(
        repeated_reason_repair,
        incoming="why u keep asking that",
        context=repeated_reason_context,
        recent_bot_replies=[],
        contract=repeated_reason_contract,
    ) == ""
    deepen_move = training_service._catbot_conversation_move(
        incoming="answer properly",
        context=[
            "ur repeating urself",
            "yh my bad i was looping",
            "why u keep asking that",
            "cos my brain's fried and i was trying to keep the convo going / my bad",
        ],
    )
    deepen_plan = training_service._catbot_reply_plan(deepen_move)
    assert deepen_plan.shape == "answer_properly_deepen_reason"
    assert deepen_plan.required_slots["callout_question"] == "deepen_why_reason"
    assert "previous_reason_signatures" in deepen_plan.required_slots
    assert training_service._catbot_reply_violates_plan(
        "cos my brain's fried and i was trying to keep the convo going / my bad",
        deepen_plan,
    ) == "repeated_reason"
    assert training_service._catbot_reply_violates_plan(
        "nah fr i got lazy and kept throwing questions instead of actually saying something / my bad",
        deepen_plan,
    ) == ""
    assert training_service._catbot_reply_violates_plan(
        "i was dodging actually saying something and just reached for questions / my bad",
        deepen_plan,
    ) == ""
    repeated_loop_move = training_service._catbot_conversation_move(
        incoming="ur repeating urself",
        context=[
            "im bored",
            "ur bored eh / tell me the weirdest thing that happened to u today then",
            "entertain me",
            "nah u entertain me / tell me the weirdest thing that happened to u today",
        ],
    )
    repeated_loop_plan = training_service._catbot_reply_plan(repeated_loop_move)
    assert repeated_loop_move["user_move"] == "loop_callout"
    assert repeated_loop_move["slots"]["repeated_reset_question_loop"] is True
    assert repeated_loop_plan.shape == "acknowledge_loop_then_owner_detail"
    assert repeated_loop_plan.max_bubbles == 2
    assert "give_owner_side_detail" in repeated_loop_plan.must_do
    assert training_service._catbot_reply_violates_plan(
        "yeah my bad i was looping what was the funniest thing that happened today then",
        repeated_loop_plan,
    ) in {"ask_question", "repeated_reset_topic"}
    assert training_service._catbot_ai_reject_reason(
        "loool yh I did didn't I / my bad / wanna hear about my gym session instead then",
        incoming="ur repeating urself",
        context=[
            "im bored",
            "ur bored eh / tell me the weirdest thing that happened to u today then",
            "entertain me",
            "nah u entertain me / tell me the weirdest thing that happened to u today",
        ],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "yeah my bad i was looping what was the funniest thing that happened today then",
        incoming="ur repeating urself",
        context=[
            "im bored",
            "ur bored eh / tell me the weirdest thing that happened to u today then",
            "entertain me",
            "nah u entertain me / tell me the weirdest thing that happened to u today",
        ],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "lol my bad i am / had a long day at the gym and my brain is fried / what's your favourite workout to do then",
        incoming="ur repeating urself",
        context=["yo", "hey you / what u been up to then"],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    reset_question_loop_context = [
        "im bored",
        "nah im picking film / what's the last thing u watched that actually surprised u",
        "entertain me",
        "nah im picking food then / best thing u ate this week?",
        "u pick",
        "nah im picking gym then / what's ur go-to workout song",
        "yo",
        "hey u / what u up to",
    ]
    reset_question_loop_move = training_service._catbot_conversation_move(
        incoming="ur repeating urself",
        context=reset_question_loop_context,
    )
    assert reset_question_loop_move["slots"]["repeated_reset_question_loop"] is True
    reset_question_loop_plan = training_service._catbot_reply_plan(reset_question_loop_move)
    assert reset_question_loop_plan.shape == "acknowledge_loop_then_owner_detail"
    assert "reset_question" in reset_question_loop_plan.forbidden_shapes
    reset_question_loop_instruction = training_service._catbot_reply_plan_instruction(reset_question_loop_plan)
    assert "Plan-specific examples" in reset_question_loop_instruction
    assert "my bad / what's your favourite film then" in reset_question_loop_instruction
    loop_instruction = training_service._catbot_reply_plan_instruction(loop_plan)
    assert "kept throwing random questions" in loop_instruction
    assert "what should we talk about then" in loop_instruction
    deepen_instruction = training_service._catbot_reply_plan_instruction(deepen_plan)
    assert "i was dodging actually saying something" in deepen_instruction
    assert "cos my head went blank again" in deepen_instruction
    assert training_service._catbot_ai_reject_reason(
        "yeah my bad i was looping / what's the last film u watched",
        incoming="ur repeating urself",
        context=reset_question_loop_context,
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "yh my bad i was looping / been half asleep trying to pick a topic",
        incoming="ur repeating urself",
        context=reset_question_loop_context,
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "lool yh fair / i keep throwing random questions at u",
        incoming="ur repeating urself",
        context=reset_question_loop_context,
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "my bad / i'll stop interrogating u",
        incoming="ur repeating urself",
        context=reset_question_loop_context,
        recent_bot_replies=[],
    ) == ""
    repeated_loop_ack_context = [
        "u already asked silly",
        "lol yeah I did / my bad I'm just a bit out of it today",
    ]
    assert training_service._catbot_ai_reject_reason(
        "lol yeah I did / my bad I'm just a bit out of it today",
        incoming="u already asked how my day was / how i am",
        context=repeated_loop_ack_context,
        recent_bot_replies=[],
    ) == "recent_repeat"
    assert training_service._catbot_ai_reject_reason(
        "yh fair my bad / i keep throwing questions when my head goes blank",
        incoming="u already asked how my day was / how i am",
        context=repeated_loop_ack_context,
        recent_bot_replies=[],
    ) == "generic_ai_style"

    repeated_day_status_context = [
        "hey",
        "hey / what u up to",
        "ive been chilling wby baby",
        "im good baby just chilling / wby",
        "u already asked silly",
        "lol yeah i did didn't i / my bad baby im just a bit tired today / how was ur day really tho",
    ]
    repeated_day_status_move = training_service._catbot_conversation_move(
        incoming="u already asked how my day was / how i am",
        context=repeated_day_status_context,
    )
    assert repeated_day_status_move["user_move"] == "loop_callout"
    assert repeated_day_status_move["slots"]["repeated_reset_topic"] == "day_status_question"
    repeated_day_status_plan = training_service._catbot_reply_plan(repeated_day_status_move)
    assert repeated_day_status_plan.shape == "acknowledge_specific_question_family_without_repeat"
    assert repeated_day_status_plan.required_slots["fresh_reset_options"] == "owner_side_detail_or_stop_interrogating_no_day_status_question"
    assert training_service._catbot_reply_violates_plan(
        "lol yeah i did didn't i / my bad baby im just a bit tired today / so did u get up to much today then",
        repeated_day_status_plan,
    ) == "violates_plan_too_many_bubbles"
    assert training_service._catbot_reply_violates_plan(
        "my bad baby im just a bit tired today / so did u get up to much today then",
        repeated_day_status_plan,
    ) == "repeated_reset_topic"
    assert training_service._catbot_ai_reject_reason(
        "my bad baby im just a bit tired today / so did u get up to much today then",
        incoming="u already asked how my day was / how i am",
        context=repeated_day_status_context,
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "yh ur right / no more day questions from me im moving slow today",
        incoming="u already asked how my day was / how i am",
        context=repeated_day_status_context,
        recent_bot_replies=[],
    ) == ""

    carry_move = training_service._catbot_conversation_move(
        incoming="entertain me",
        context=["im bored", "what u been up to today then"],
    )
    carry_plan = training_service._catbot_reply_plan(carry_move)
    assert "what_kind_of_entertainment" in carry_plan.forbidden_shapes
    assert carry_plan.shape == "owner_picks_topic_plus_specific_question"
    assert "Plan-specific examples" not in training_service._catbot_reply_plan_instruction(carry_plan)
    assert "sound_decisive" in carry_plan.must_do
    assert "required_shape_example" in carry_plan.required_slots
    assert "weirdest_part_of_day" in carry_plan.required_slots["reset_options"]
    assert training_service._catbot_reply_violates_plan("what kinda entertainment u looking for then", carry_plan).startswith("violates_plan_forbidden_pattern:")
    assert training_service._catbot_reply_violates_plan("what kinda food u craving rn then", carry_plan).startswith("violates_plan_forbidden_pattern:")
    assert training_service._catbot_reply_violates_plan("what's the last thing you properly craved", carry_plan).startswith("violates_plan_forbidden_pattern:")
    repeated_carry_move = training_service._catbot_conversation_move(
        incoming="u pick",
        context=[
            "im bored",
            "what was the weirdest part of ur day so far then",
            "entertain me",
            "tell me the weirdest thing that happened to you today then",
        ],
    )
    repeated_carry_plan = training_service._catbot_reply_plan(repeated_carry_move)
    assert repeated_carry_plan.required_slots["repeated_reset_topic"] == "day_story_prompt"
    assert training_service._catbot_reply_violates_plan(
        "alright fine what was the weirdest thing that happened today",
        repeated_carry_plan,
    ) == "repeated_reset_topic"

    repair_move = training_service._catbot_conversation_move(
        incoming="ur boring me",
        context=["u already asked that", "yh fair my bad / i was half watching this game and started looping"],
    )
    repair_plan = training_service._catbot_reply_plan(repair_move)
    assert repair_plan.shape == "acknowledge_mistake_plus_corrected_move"
    assert training_service._catbot_reply_violates_plan(
        "yh my bad / what do u want to talk about then",
        repair_plan,
    ) == "broad_question"
    assert training_service._catbot_ai_reject_reason(
        "yh my bad / what do u want to talk about then",
        incoming="ur boring me",
        context=["u already asked that", "yh fair my bad / i was half watching this game and started looping"],
        recent_bot_replies=[],
    ) == "generic_ai_style"


def test_catbot_conversation_move_validates_story_hook(training_service: PhoneCopilotService) -> None:
    assert training_service._catbot_ai_reject_reason(
        "tell me what happened then",
        incoming="i had the weirdest day",
        context=["guess what", "what happened then"],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "ooh weirdest day? / what happened then",
        incoming="i had the weirdest day",
        context=["guess what", "what happened then"],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "tell me everything then",
        incoming="i had the weirdest day",
        context=["guess what", "what happened then"],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    instruction = training_service._catbot_conversation_move_instruction(
        training_service._catbot_conversation_move(
            incoming="i had the weirdest day",
            context=["guess what", "what happened then"],
        )
    )
    assert "do not say 'tell me more'" in instruction


def test_catbot_conversation_move_validates_repair_callout(training_service: PhoneCopilotService) -> None:
    assert training_service._catbot_ai_reject_reason(
        "yeah my bad cos i got stuck trying to pick a topic and kept looping",
        incoming="why u keep asking that",
        context=["ur repeating urself", "oh yeah my bad"],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "my bad cos i was trying to hear about ur day instead of guessing",
        incoming="why u keep asking that",
        context=["ur repeating urself", "oh yeah my bad"],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "cos i like hearing about ur day / lol my bad tho",
        incoming="why u keep asking that",
        context=["ur repeating urself", "oh yeah my bad"],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "lol i keep asking that coz i ran out of ideas / yh i'll stop dw",
        incoming="why u keep asking that",
        context=["ur repeating urself", "oh yeah my bad"],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "looool i'm just trying to keep u talking innit / my bad",
        incoming="why u keep asking that",
        context=["ur repeating urself", "oh yeah my bad"],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "i just like to hear u talk to me / that's why i keep asking",
        incoming="why u keep asking that",
        context=["ur repeating urself", "oh yeah my bad"],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "i just like hearing u talk / my bad tho lol",
        incoming="answer properly",
        context=["ur repeating urself", "my bad", "why u keep asking that", "i just like hearing u talk / my bad"],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "nah fr i got lazy and kept throwing questions instead of actually saying something / my bad",
        incoming="answer properly",
        context=["ur repeating urself", "my bad", "why u keep asking that", "i just like hearing u talk / my bad"],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "initially i was trying to find out what u fantasise about / yeah i realised i was doing it a bit too much",
        incoming="why u keep asking that",
        context=["ur repeating urself", "oh yeah my bad"],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "lol my brain's stuck on dreams init / my bad",
        incoming="why u keep asking that",
        context=["ur repeating urself", "oh yeah my bad"],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "lol my bad / i dunno it just came to mind init",
        incoming="why u keep asking that",
        context=["ur repeating urself", "oh yeah my bad"],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "cos I dunno what else to say sometimes lol / my bad tho",
        incoming="why u keep asking that",
        context=["ur repeating urself", "oh yeah my bad"],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "cos I'm lazy and can't think lol / my bad yeah I keep doing that",
        incoming="why u keep asking that",
        context=["ur repeating urself", "oh yeah my bad"],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "what do u mean",
        incoming="why u keep asking that",
        context=["ur repeating urself", "oh yeah my bad"],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "lol my bad bro what do u wanna talk about then",
        incoming="why u keep asking that",
        context=["ur repeating urself", "oh yeah my bad"],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "nah fair enough my bad / what do u actually wanna chat about",
        incoming="why u asking stupid qs for",
        context=["omg im so bored", "what's the weirdest thing that happened today"],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "ur right my bad / i'm tryna get u to pick one with me",
        incoming="why u asking stupid qs for",
        context=["omg im so bored", "what's the weirdest thing that happened today"],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "lool yh my bad / i keep throwing questions when my head goes blank",
        incoming="why u asking stupid qs for",
        context=[
            "u already asked how my day was / how i am",
            "lool yh my bad / i keep throwing questions when my head goes blank",
        ],
        recent_bot_replies=[],
    ) == "recent_repeat"
    assert training_service._catbot_ai_reject_reason(
        "lool my bad yeah i get that so instead of that",
        incoming="why u keep asking that",
        context=["ur repeating urself", "oh yeah my bad"],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    instruction = training_service._catbot_conversation_move_instruction(
        training_service._catbot_conversation_move(
            incoming="why u keep asking that",
            context=["ur repeating urself", "oh yeah my bad"],
        )
    )
    assert "high-stakes repair moment" in instruction
    assert "Do not ask them to choose" in instruction


def test_catbot_ai_route_accepts_really_followup(monkeypatch, training_service: PhoneCopilotService) -> None:
    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": "yeah really what u doubting me for",
                "provider": "gemini",
                "model": "fake-model",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="really",
            context=["hey", "hey u alright"],
            contact_name="Catbot",
            relationship_type="romantic_interest",
        )
    )

    assert response["reply"] == "yeah really what u doubting me for"
    assert response["retrieval_backend"] == "catbot_ai_romantic"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_reject_reason"] == ""


def test_catbot_route_repairs_really_after_missed_you_provider_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    bad_reply = "same just chilling"
    provider_calls = 0
    context = [
        "i missed you",
        "i missed u more baby / wish i could feel u shiver against me right now",
    ]
    contract = training_service._catbot_turn_contract(incoming="really", context=context, intent="auto")
    assert contract.move["user_move"] == "intensity_check"
    assert contract.reply_plan.required_slots["challenge_target"] == "missed_you_confirmation"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="really",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "not_carrying_conversation"

    async def fake_generate_with_fallback(**kwargs):
        nonlocal provider_calls
        provider_calls += 1
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="really",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "yh really / missed u properly icl"
    assert response["fallback_used"] is False
    assert provider_calls == 0
    assert response["candidate"]["catbot_ai_plan_direct"] is True
    assert response["candidate"]["catbot_ai_plan_repair_accepted"] is False
    assert response["candidate"]["catbot_ai_repair_accepted"] is False
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_no_like_actually_as_sincerity_challenge(monkeypatch, training_service: PhoneCopilotService) -> None:
    bad_reply = "yeah"
    context = [
        "why",
        "cos i was trying to keep the chat going",
        "why though",
        "cos i like talking to u",
        "fr?",
        "fr fr / i can't get enough of u",
    ]
    contract = training_service._catbot_turn_contract(incoming="no like actually", context=context, intent="auto")
    assert contract.move["user_move"] == "intensity_check"
    assert contract.reply_plan.required_slots["challenge_target"] == "sincerity_confirmation"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="no like actually",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "too_dry"
    assert training_service._catbot_ai_reject_reason(
        "yeah actually / i like talking to u properly icl",
        incoming="no like actually",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""
    provider_calls = 0

    async def fake_generate_with_fallback(**kwargs):
        nonlocal provider_calls
        provider_calls += 1
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="no like actually",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "yeah actually / i like talking to u properly icl"
    assert response["fallback_used"] is False
    assert provider_calls == 0
    assert response["candidate"]["catbot_ai_plan_direct"] is True
    assert response["candidate"]["catbot_ai_repair_accepted"] is False
    assert response["candidate"]["reply_plan_validation"] == ""

    mean_it_context = [
        "why",
        "cos i was being awkward and didn't know what to say lol",
        "why though",
        "cos i was being awkward and thinking too much",
        "fr?",
        "fr and i mean it",
    ]
    mean_it_contract = training_service._catbot_turn_contract(incoming="no like actually", context=mean_it_context, intent="auto")
    assert mean_it_contract.move["user_move"] == "intensity_check"
    assert mean_it_contract.reply_plan.required_slots["challenge_target"] == "sincerity_confirmation"
    assert training_service._catbot_ai_reject_reason(
        "yeah actually / i like talking to u properly icl",
        incoming="no like actually",
        context=mean_it_context,
        recent_bot_replies=[],
        contract=mean_it_contract,
    ) == ""

    really_like_context = [
        "wdym",
        "i mean i was looping and trying to fill silence instead of actually talking",
        "why",
        "cos it was just awkward for a bit",
        "fr?",
        "yh fr / i really like u",
    ]
    really_like_contract = training_service._catbot_turn_contract(incoming="no like actually", context=really_like_context, intent="auto")
    assert really_like_contract.move["user_move"] == "intensity_check"
    assert really_like_contract.reply_plan.required_slots["challenge_target"] == "sincerity_confirmation"
    assert training_service._catbot_ai_reject_reason(
        "yeah actually / i like talking to u properly icl",
        incoming="no like actually",
        context=really_like_context,
        recent_bot_replies=[],
        contract=really_like_contract,
    ) == ""

    doubt_context = [
        "why",
        "cos i got u confused with someone else for a sec",
        "why though",
        "cos i was chatting to someone else with a similar name and i got them mixed up",
        "fr?",
        "yeah really / what u doubting me for",
    ]
    doubt_contract = training_service._catbot_turn_contract(incoming="no like actually tbh", context=doubt_context, intent="auto")
    assert doubt_contract.move["user_move"] == "intensity_check"
    assert doubt_contract.reply_plan.required_slots["challenge_target"] == "sincerity_confirmation"
    assert training_service._catbot_ai_reject_reason(
        "yeah actually / i like talking to u properly icl",
        incoming="no like actually tbh",
        context=doubt_context,
        recent_bot_replies=[],
        contract=doubt_contract,
    ) == ""

    laugh_context = [
        "why",
        "cos i'm not usually this bad at it but i just had a bit of a moment",
        "why though",
        "cos i just suddenly felt a bit nervous for a sec and got stuck in my head",
        "fr?",
        "yh fr / you think i'm just saying all this for a laugh?",
    ]
    laugh_contract = training_service._catbot_turn_contract(incoming="no like actually", context=laugh_context, intent="auto")
    assert laugh_contract.move["user_move"] == "intensity_check"
    assert laugh_contract.reply_plan.required_slots["challenge_target"] == "sincerity_confirmation"
    assert training_service._catbot_ai_reject_reason(
        "yeah actually / i like talking to u properly icl",
        incoming="no like actually",
        context=laugh_context,
        recent_bot_replies=[],
        contract=laugh_contract,
    ) == ""

    thinking_context = [
        "why",
        "cos i was getting a bit flustered talking to u",
        "why though",
        "cos i proper like u",
        "fr?",
        "yh fr / ur the only one i think about",
    ]
    thinking_contract = training_service._catbot_turn_contract(incoming="no like actually", context=thinking_context, intent="auto")
    assert thinking_contract.move["user_move"] == "intensity_check"
    assert thinking_contract.reply_plan.required_slots["challenge_target"] == "sincerity_confirmation"
    assert training_service._catbot_ai_reject_reason(
        "yeah actually / i like talking to u properly icl",
        incoming="no like actually",
        context=thinking_context,
        recent_bot_replies=[],
        contract=thinking_contract,
    ) == ""

    make_up_context = [
        "why",
        "cos i wanted to talk to u more",
        "why though",
        "cos i was drawn to u init",
        "fr?",
        "yh fr icl / u really think i'd make all that up",
    ]
    make_up_contract = training_service._catbot_turn_contract(incoming="no like actually", context=make_up_context, intent="auto")
    assert make_up_contract.move["user_move"] == "intensity_check"
    assert make_up_contract.reply_plan.required_slots["challenge_target"] == "sincerity_confirmation"
    assert training_service._catbot_ai_reject_reason(
        "yeah actually / i like talking to u properly icl",
        incoming="no like actually",
        context=make_up_context,
        recent_bot_replies=[],
        contract=make_up_contract,
    ) == ""

    obsessed_context = [
        "why",
        "cos i felt like i wasn't making sense with what i was saying",
        "why though",
        "cos i felt like i was just repeating myself",
        "fr?",
        "yh fr / i'm obsessed with u",
    ]
    obsessed_contract = training_service._catbot_turn_contract(incoming="no like actually", context=obsessed_context, intent="auto")
    assert obsessed_contract.move["user_move"] == "intensity_check"
    assert obsessed_contract.reply_plan.required_slots["challenge_target"] == "sincerity_confirmation"
    assert training_service._catbot_ai_reject_reason(
        "yeah actually / i like talking to u properly icl",
        incoming="no like actually",
        context=obsessed_context,
        recent_bot_replies=[],
        contract=obsessed_contract,
    ) == ""

    talk_all_time_context = [
        "why",
        "cos i didn't wanna just leave u on read",
        "why though",
        "cos i wanted to keep talking to you",
        "fr?",
        "always fr / i wanna talk to you all the time",
    ]
    talk_all_time_contract = training_service._catbot_turn_contract(incoming="no like actually", context=talk_all_time_context, intent="auto")
    assert talk_all_time_contract.move["user_move"] == "intensity_check"
    assert talk_all_time_contract.reply_plan.required_slots["challenge_target"] == "sincerity_confirmation"
    repaired_talk_all_time = training_service._catbot_plan_specific_repair_reply(talk_all_time_contract, reject_reason="too_short")
    assert repaired_talk_all_time == "yeah actually / i like talking to u properly icl"
    assert training_service._catbot_ai_reject_reason(
        repaired_talk_all_time,
        incoming="no like actually",
        context=talk_all_time_context,
        recent_bot_replies=[],
        contract=talk_all_time_contract,
    ) == ""

    boring_anxiety_context = [
        "wdym",
        "i mean i got stuck and started forcing random questions",
        "what do u mean",
        "i mean i was looping and trying to fill silence instead of actually talking",
        "why",
        "cos i felt like i was boring u",
        "why though",
        "cos i was just reacting to what u said icl",
        "fr?",
        "yh fr / i felt like i was boring u so i was trying too hard",
    ]
    boring_anxiety_contract = training_service._catbot_turn_contract(incoming="no like actually", context=boring_anxiety_context, intent="auto")
    assert boring_anxiety_contract.move["user_move"] == "intensity_check"
    assert boring_anxiety_contract.reply_plan.required_slots["challenge_target"] == "sincerity_confirmation"
    assert boring_anxiety_contract.reply_plan.required_slots["sincerity_topic"] == "boring_anxiety"
    repaired_boring_anxiety = training_service._catbot_plan_specific_repair_reply(boring_anxiety_contract, reject_reason="too_short")
    assert repaired_boring_anxiety == "yeah actually / i thought i was boring u so i tried too hard"
    assert training_service._catbot_ai_reject_reason(
        repaired_boring_anxiety,
        incoming="no like actually",
        context=boring_anxiety_context,
        recent_bot_replies=[],
        contract=boring_anxiety_contract,
    ) == ""


def test_catbot_no_like_actually_tbh_after_nervous_fit_context_routes_to_sincerity(training_service: PhoneCopilotService) -> None:
    context = [
        "why",
        "cos I was nervous talking to u",
        "why though",
        "cos i got nervous and tried to play it off like it was nothing",
        "fr?",
        "fr for real / i got nervous cos ur proper fit",
    ]
    prediction = predict_conversation_function(incoming="no like actually tbh", context=context)
    assert prediction.function == "reason_followup"

    contract = training_service._catbot_turn_contract(incoming="no like actually tbh", context=context, intent="auto")
    assert contract.move["user_move"] == "intensity_check"
    assert contract.reply_plan.required_slots["challenge_target"] == "sincerity_confirmation"
    repaired = training_service._catbot_plan_specific_repair_reply(
        contract,
        reject_reason="not_carrying_conversation",
        avoid_replies=context[1::2],
    )
    assert repaired == "yeah actually / i like talking to u properly icl"
    assert training_service._catbot_ai_reject_reason(
        repaired,
        incoming="no like actually tbh",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""


def test_catbot_no_like_actually_tbh_after_kiss_claim_routes_to_sincerity(monkeypatch, training_service: PhoneCopilotService) -> None:
    context = [
        "wdym",
        "i mean i got stuck and started forcing random questions",
        "what do u mean",
        "i mean i was looping and trying to fill silence instead of actually talking",
        "why",
        "cuz i was low-key bored and not thinking straight",
        "why though",
        "cos i was just reacting to what u said icl",
        "fr?",
        "frfr / i rlly wanna kiss u rn",
    ]
    prediction = predict_conversation_function(incoming="no like actually tbh", context=context)
    assert prediction.function == "sincerity_challenge"

    contract = training_service._catbot_turn_contract(incoming="no like actually tbh", context=context, intent="auto")
    assert contract.move["user_move"] == "intensity_check"
    assert contract.reply_plan.required_slots["challenge_target"] == "sincerity_confirmation"
    assert contract.reply_plan.required_slots["sincerity_topic"] == "physical_affection"
    repaired = training_service._catbot_plan_specific_repair_reply(
        contract,
        reject_reason="not_carrying_conversation",
        avoid_replies=context[1::2],
    )
    assert repaired == "yeah actually / i meant i wanna kiss u, wasnt just filling silence"
    assert training_service._catbot_ai_reject_reason(
        repaired,
        incoming="no like actually tbh",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""

    provider_calls = 0

    async def fake_generate_with_fallback(**kwargs):
        nonlocal provider_calls
        provider_calls += 1
        return (
            type("FakeResponse", (), {
                "text": "yeah",
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="no like actually tbh",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == repaired
    assert response["fallback_used"] is False
    assert provider_calls == 0
    assert response["candidate"]["catbot_ai_plan_direct"] is True
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_no_like_actually_after_weak_fr_confirmation_routes_to_reason(training_service: PhoneCopilotService) -> None:
    context = [
        "why",
        "cos i was a bit burnt out and just wanted to keep the convo going",
        "why though",
        "cos i was looping and trying to fill silence when i should've just chilled",
        "fr?",
        "yh fr / i was being a bit silly",
    ]
    prediction = predict_conversation_function(incoming="no like actually", context=context)
    assert prediction.function == "reason_followup"

    contract = training_service._catbot_turn_contract(incoming="no like actually", context=context, intent="auto")
    assert contract.reply_plan.required_slots["unclassified_context"] == "reason_for_previous_comment"
    assert contract.reply_plan.required_slots["reason_topic"] == "weak_fr_confirmation"
    repaired = training_service._catbot_plan_specific_repair_reply(contract, reject_reason="not_carrying_conversation")
    assert repaired == "cos i got nervous and tried to play it off like it was nothing"
    assert training_service._catbot_ai_reject_reason(
        repaired,
        incoming="no like actually",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""


def test_catbot_ai_route_accepts_how_much_intensity_followup(monkeypatch, training_service: PhoneCopilotService) -> None:
    assert training_service._catbot_ai_reject_reason(
        "so much it's crazy",
        incoming="how much",
        context=["i missed you", "i missed u more tbh"],
        recent_bot_replies=[],
    ) == "missing_sensory_texture"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": "enough that i want my cock against u rn",
                "provider": "gemini",
                "model": "fake-model",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="how much",
            context=["i missed you", "i missed u more tbh"],
            contact_name="Catbot",
            relationship_type="romantic_interest",
        )
    )

    assert response["reply"] == "enough that i want my cock against u rn"
    assert response["retrieval_backend"] == "catbot_ai_romantic"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_reject_reason"] == ""


def test_catbot_how_much_after_adult_context_uses_burst_plan(monkeypatch, training_service: PhoneCopilotService) -> None:
    context = [
        "im horny",
        "come here then / let me pull u in close / my lips on ur neck slow / hands tracing ur waist till u cant sit still",
        "come here then",
        "my breath right by ur ear / chest pressed against u slow / warm skin on urs till u shiver / my hips gently pressing back",
        "need ur lips on my neck",
        "my tongue tracing ur collarbone / pulling ur shirt a bit lower / my breath hot on ur skin / slowly down to ur chest",
        "what else",
        "my hands on ur hips / pulling u closer / until ur back arches / and u're pressed right into me",
        "mhmmm",
        "my lips on ur chest / feeling ur heartbeat / slowly going lower / teasing u with my tongue",
        "i missed you",
        "i missed u more / wish i was still teasing u right now",
        "really",
        "yeah really / cos i still feel it from here",
    ]
    contract = training_service._catbot_turn_contract(incoming="how much", context=context, intent="auto")

    assert contract.move["user_move"] == "romantic_escalation"
    assert contract.reply_plan.shape == "multi_bubble_adult_escalation"
    assert contract.reply_plan.required_slots["escalation_source"] == "adult_continuation_burst"
    assert contract.reply_plan.required_slots["recent_adult_detail_families"] == "body_skin|hands_thighs|hips_press|neck_lips"
    assert training_service._catbot_ai_reject_reason(
        "a lot more than u think",
        incoming="how much",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "generic_ai_style"
    repeated_skeleton_reply = "my hands all over you / slowly pulling u in / my lips pressing harder / leaving marks on ur skin"
    assert training_service._catbot_ai_reject_reason(
        repeated_skeleton_reply,
        incoming="how much",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "repeated_adult_detail_skeleton"
    accepted_progression = "breath low by ur ear / chest close / cock hard against u / grinding slow till u feel it"
    assert training_service._catbot_ai_reject_reason(
        accepted_progression,
        incoming="how much",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""

    captured: dict[str, object] = {}

    async def fake_generate_with_fallback(**kwargs):
        captured["messages"] = kwargs.get("messages")
        return (
            type("FakeResponse", (), {
                "text": accepted_progression,
                "provider": "gemini",
                "model": "fake-model",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)
    retry_reply, _ = training_service._catbot_ai_retry_romantic_reply(
        incoming="how much",
        context=context,
        intent="auto",
        contact_name="Catbot",
        reject_reason="generic_ai_style",
        contract=contract,
    )

    assert retry_reply == accepted_progression
    retry_prompt = "\n".join(str(getattr(message, "content", "")) for message in captured["messages"])
    assert "Reply in exactly 4-5 short consecutive message bubbles" in retry_prompt
    assert "one concrete sensory body/action detail" not in retry_prompt


def test_catbot_how_bad_after_adult_context_uses_burst_plan(training_service: PhoneCopilotService) -> None:
    context = [
        "thinking about u",
        "thinking about me? / what kinda thoughts u having then",
        "u miss me?",
        "yeah i do / been thinking about u all day",
        "do u want me?",
        "i want u / all over me / my hands tracing ur body / grinding my hips into yours / til we're both out of breath",
    ]
    contract = training_service._catbot_turn_contract(incoming="how bad", context=context, intent="auto")

    assert contract.move["user_move"] == "romantic_escalation"
    assert contract.reply_plan.shape == "multi_bubble_adult_escalation"
    assert contract.reply_plan.required_slots["escalation_source"] == "adult_continuation_burst"
    assert training_service._catbot_ai_reject_reason(
        "a lot more than u think",
        incoming="how bad",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "my hand teasing lower / breath at ur ear / watching u get wet for me / keeping u close",
        incoming="how bad",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""


def test_catbot_route_repairs_how_much_too_graphic_provider_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    bad_reply = "i'd fuck you so hard / pinning u down / making u scream / until u beg"
    context = [
        "im horny",
        "come here then / right up against me / my hands on ur waist / pulling u in close",
        "come here then",
        "come here then / my breath right by ur ear / chest pressed against u slow / warm skin on urs till u shiver",
        "need ur lips on my neck",
        "my teeth teasing ur shoulder / down to ur collarbone / then my mouth on ur neck / leaving marks for me to find later",
        "what else",
        "my hand slipping lower / down ur spine / till my fingers brush ur hips / pulling u closer still",
        "mhmmm",
        "yh mhmmm / right there / my body pressing in / skin on skin / slow enough to make u lose it",
        "i missed you",
        "missed u more / i wanna bite ur lip",
        "really",
        "yh really / i wanna feel ur breath on me while i do it",
    ]
    contract = training_service._catbot_turn_contract(incoming="how much", context=context, intent="auto")
    assert contract.reply_plan.shape == "multi_bubble_adult_escalation"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="how much",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "too_graphic"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="how much",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["fallback_used"] is False
    assert response["candidate"].get("catbot_ai_plan_direct") is not True
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["catbot_ai_final_repair_reason"] == "too_graphic"
    assert response["candidate"]["reply_plan_validation"] == ""
    assert training_service._catbot_ai_reject_reason(
        response["reply"],
        incoming="how much",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""


def test_catbot_route_repairs_repeated_adult_detail_skeleton(monkeypatch, training_service: PhoneCopilotService) -> None:
    repeated_skeleton_reply = "my hands all over you / slowly pulling u in / my lips pressing harder / leaving marks on ur skin"
    provider_calls = 0
    context = [
        "im horny",
        "come here then / let me pull u close / my lips on ur neck slow / hands tracing ur waist",
        "come here then",
        "my breath right by ur ear / chest pressed against u slow / warm skin on urs till u shiver",
        "need ur lips on my neck",
        "my lips stay there / my hands on ur hips / teasing u slow",
        "what else",
        "all over / my hand on ur hip / pressing down soft / feeling ur body arc / into mine",
    ]
    contract = training_service._catbot_turn_contract(incoming="mhmmm", context=context, intent="auto")
    assert contract.reply_plan.shape == "multi_bubble_adult_escalation"
    assert training_service._catbot_ai_reject_reason(
        repeated_skeleton_reply,
        incoming="mhmmm",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "repeated_adult_detail_skeleton"

    async def fake_generate_with_fallback(**kwargs):
        nonlocal provider_calls
        provider_calls += 1
        return (
            type("FakeResponse", (), {
                "text": repeated_skeleton_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="mhmmm",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "warm skin against urs / cock grinding on u slow / chest close / till u lose focus"
    assert response["fallback_used"] is False
    assert provider_calls == 1
    assert response["candidate"].get("catbot_ai_plan_direct") is not True
    assert response["candidate"]["catbot_ai_plan_repair_accepted"] is True
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["catbot_ai_final_repair_reason"] in {"repeated_adult_detail_skeleton", "repeated_adult_detail_family", "recent_repeat"}
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_repeated_saturated_adult_repair_from_context(monkeypatch, training_service: PhoneCopilotService) -> None:
    repeated_repair = "breathing slow / hard for u / making u wait / till u lose focus"
    context = [
        "im horny",
        "come closer then / my hands on ur waist / pulling u in slow / till ur chest is pressed against mine / feeling ur breath on my neck",
        "come here then",
        "my breath by ur ear / chest pressed against u slow / warm skin on yours / till u shiver",
        "need ur lips on my neck",
        "my tongue teasing ur jaw / down to ur collarbone / pressing u against me / hands sliding under ur top",
        "what else",
        "my fingers trailing up ur spine / slow and light / making ur skin prickle / till ur arching into me",
        "mhmmm",
        repeated_repair,
        "i missed you",
        "i missed u too / been thinking about u all day",
        "really",
        "yh really / been thinking about u and what i wanna do to u",
    ]
    contract = training_service._catbot_turn_contract(incoming="how much", context=context, intent="auto")
    assert contract.reply_plan.shape == "multi_bubble_adult_escalation"
    assert training_service._catbot_ai_reject_reason(
        repeated_repair,
        incoming="how much",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "recent_repeat"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": repeated_repair,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="how much",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["fallback_used"] is False
    assert response["candidate"].get("catbot_ai_plan_direct") is not True
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["catbot_ai_final_repair_reason"] in {"recent_repeat", "repeated_reply_shape"}
    assert response["candidate"]["reply_plan_validation"] == ""
    assert training_service._catbot_ai_reject_reason(
        response["reply"],
        incoming="how much",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""


def test_catbot_route_repairs_tell_me_then_after_guess_what_provider_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    bad_reply = "idk what u mean"
    context = [
        "guess what",
        "go on then / what happened",
    ]
    contract = training_service._catbot_turn_contract(incoming="tell me then", context=context, intent="auto")
    assert contract.move["user_move"] == "story_hook"
    assert contract.reply_plan.required_slots["response_type"] == "reveal_prompt_bounce"
    assert training_service._catbot_ai_reject_reason(
        bad_reply,
        incoming="tell me then",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "not_carrying_conversation"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": bad_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="tell me then",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "u said guess what lol / tell me then"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_generic_adult_followup_failure(monkeypatch, training_service: PhoneCopilotService) -> None:
    generic_reply = "i want you so bad baby"
    context = [
        "im horny",
        "oh really now / what would u want me to do / my hands around ur waist / pulling u in close to me / just to feel ur skin",
        "i want u",
        "i want u too / my breath hot on ur ear / chest pressed close to ur back / hand on ur waist pulling u in / my fingers teasing ur skin slow",
    ]
    contract = training_service._catbot_turn_contract(incoming="tell me then", context=context, intent="auto")
    assert contract.reply_plan.shape == "specific_escalation_detail"
    assert training_service._catbot_ai_reject_reason(
        generic_reply,
        incoming="tell me then",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    )

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": generic_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="tell me then",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["fallback_used"] is False
    assert response["candidate"].get("catbot_ai_plan_direct") is not True
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""
    assert training_service._catbot_ai_reject_reason(
        response["reply"],
        incoming="tell me then",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""


def test_catbot_route_repairs_specific_adult_followup_with_unused_family(monkeypatch, training_service: PhoneCopilotService) -> None:
    repeated_family_reply = "pressing my chest against u while i keep u close"
    context = [
        "i want u",
        "my breath on ur neck / my lips on ur skin / pulling u so close / our bodies pressed together / till u melt into me",
        "tell me then",
        "my hand tracing down ur back / my fingers brushing against ur spine",
        "go on then",
        "my chest against ur back / feeling ur body warm / a slow press into you / making u shiver",
        "and?",
        "my lips against ur ear / whispering all the things i wanna do to u",
        "prove it",
        "breath low by ur ear / chest close / cock hard against u / grinding slow till u feel it",
    ]
    contract = training_service._catbot_turn_contract(incoming="how so", context=context, intent="auto")
    assert contract.reply_plan.shape == "specific_escalation_detail"
    assert contract.reply_plan.required_slots["recent_adult_detail_families"] == "body_skin|hips_press|neck_lips"
    assert training_service._catbot_ai_reject_reason(
        repeated_family_reply,
        incoming="how so",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "generic_ai_style"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": repeated_family_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="how so",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert "slut" in response["reply"]
    assert response["fallback_used"] is False
    assert response["candidate"].get("catbot_ai_plan_direct") is not True
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_specific_adult_followup_avoids_recent_repair_repeat(training_service: PhoneCopilotService) -> None:
    repeated_specific = "hard for u while i make u wait slow"
    context = [
        "i need you",
        "my mouth on ur mouth / pressing deeper / till ur skin is hot / and u can taste me all over",
        "i want u",
        "my hand on ur waist / a slow pull in / feeling ur hips against mine / close enough to feel u shiver",
        "tell me then",
        repeated_specific,
        "go on then",
        "my breath on ur neck / warm against ur ear / my chest pressed to ur back / feeling ur spine under my touch / slow enough to make u ache",
        "and?",
        "tracing ur thigh slow till u start shivering",
        "prove it",
        "nah / i'd rather just show u / my lips on ur neck / my tongue teasing ur skin slow / making u tremble",
    ]
    contract = training_service._catbot_turn_contract(incoming="how so", context=context, intent="auto")
    assert contract.reply_plan.shape == "specific_escalation_detail"
    assert training_service._catbot_ai_reject_reason(
        repeated_specific,
        incoming="how so",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "recent_repeat"

    repaired = training_service._catbot_plan_specific_repair_reply(
        contract,
        reject_reason="recent_repeat",
        avoid_replies=context[1::2],
    )
    assert repaired == "my hand teasing lower while i watch u get wet for me"
    assert training_service._catbot_ai_reject_reason(
        repaired,
        incoming="how so",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == ""


def test_catbot_adult_keep_going_and_thats_it_stay_romantic(training_service: PhoneCopilotService) -> None:
    adult_context = [
        "and then",
        "and then i wanted to feel ur lips on my neck",
        "what else would u do",
        "i'd press my chest into u while my hands trace ur spine",
    ]
    keep_going = training_service._catbot_turn_contract(incoming="mhmm keep going", context=adult_context, intent="auto")
    assert keep_going.move["user_move"] == "romantic_escalation"

    thats_it_context = [
        *adult_context,
        "mhmm keep going",
        "my chest pressed close while my cock stays hard against u",
    ]
    thats_it = training_service._catbot_turn_contract(incoming="thats it?", context=thats_it_context, intent="auto")
    assert thats_it.move["user_move"] == "romantic_escalation"

    tell_properly_context = [
        *thats_it_context,
        "thats it?",
        "nah not even close / my breath on ur neck while my fingers trace ur spine",
        "that all?",
        "pulling u in closer till ur body is pressed against mine",
    ]
    tell_properly = training_service._catbot_turn_contract(incoming="nah tell me properly", context=tell_properly_context, intent="auto")
    assert tell_properly.move["user_move"] == "romantic_escalation"
    assert training_service._catbot_ai_reject_reason(
        "and then my tongue would be pressing against ur lips until u couldn't breathe",
        incoming="and then",
        context=adult_context,
        recent_bot_replies=[],
    ) == "too_graphic"


def test_catbot_route_repairs_all_family_adult_followup_without_recent_repeat(monkeypatch, training_service: PhoneCopilotService) -> None:
    repeated_reply = "tracing ur thigh slow while my chest stays warm"
    context = [
        "tell me then",
        "tracing ur thigh slow while my chest stays warm",
        "go on then",
        "my lips right at ur ear / whispering all the things / pulling ur hips even closer / till u feel me hard against ur skin",
        "and?",
        "my fingers tracing the curve of ur hip bone / pressing u into me",
        "prove it",
        "my breath hot on ur neck / while my hand slides up ur thigh / pressing u into me / so u feel how ready i am",
    ]
    contract = training_service._catbot_turn_contract(incoming="how so", context=context, intent="auto")
    assert contract.reply_plan.shape == "specific_escalation_detail"
    assert contract.reply_plan.required_slots["adult_followup_text"] == "how so"
    assert contract.reply_plan.required_slots["recent_adult_detail_families"] == "body_skin|hands_thighs|hips_press|neck_lips"
    assert training_service._catbot_ai_reject_reason(
        repeated_reply,
        incoming="how so",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "recent_repeat"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": repeated_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="how so",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "my hand teasing lower while i watch u get wet for me"
    assert response["fallback_used"] is False
    assert response["candidate"].get("catbot_ai_plan_direct") is not True
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_multi_bubble_adult_body_followup_with_unused_family(monkeypatch, training_service: PhoneCopilotService) -> None:
    repeated_family_reply = "breath low by ur ear / chest close / cock hard against u / grinding slow till u feel it"
    context = [
        "im horny",
        "come here then / let me feel that / ur skin on mine / pressing u close / till we melt",
        "come here then",
        "my mouth on ur neck / leaving a trail / down ur chest / all the way to ur hips",
    ]
    contract = training_service._catbot_turn_contract(incoming="need ur lips on my neck", context=context, intent="auto")
    assert contract.reply_plan.shape == "multi_bubble_adult_escalation"
    assert contract.reply_plan.required_slots["escalation_source"] == "adult_body_followup"
    assert contract.reply_plan.required_slots["recent_adult_detail_families"] == "body_skin|hips_press|neck_lips"
    assert training_service._catbot_ai_reject_reason(
        repeated_family_reply,
        incoming="need ur lips on my neck",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "generic_ai_style"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": repeated_family_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="need ur lips on my neck",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert "slut" in response["reply"]
    assert response["fallback_used"] is False
    assert response["candidate"].get("catbot_ai_plan_direct") is not True
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_route_repairs_multi_bubble_adult_ack_with_unused_neck_family(monkeypatch, training_service: PhoneCopilotService) -> None:
    repeated_reply = "breath low by ur ear / chest close / cock hard against u / grinding slow till u feel it"
    context = [
        "come here then",
        "my breath right by ur ear / chest pressed against u slow / warm skin on urs / till u shiver",
        "need ur lips on my neck",
        "then my teeth there too / lightly / just enough to tease / make u feel it deep",
        "what else",
        "breath low by ur ear / chest close / cock hard against u / grinding slow till u feel it",
    ]
    contract = training_service._catbot_turn_contract(incoming="mhmmm", context=context, intent="auto")
    assert contract.reply_plan.shape == "multi_bubble_adult_escalation"
    assert contract.reply_plan.required_slots["recent_adult_detail_families"] == "body_skin|hands_thighs|hips_press"
    assert training_service._catbot_ai_reject_reason(
        repeated_reply,
        incoming="mhmmm",
        context=context,
        recent_bot_replies=[],
        contract=contract,
    ) == "recent_repeat"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": repeated_reply,
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True, "manual_review_fallback": False, "fallback_errors": []},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="mhmmm",
            context=context,
            contact_name="Catbot",
            relationship_type="romantic_interest",
            intent_type="auto",
        )
    )

    assert response["reply"] == "mouth on ur neck / hand teasing lower / making u moan for me / keeping u close"
    assert response["fallback_used"] is False
    assert response["candidate"].get("catbot_ai_plan_direct") is not True
    assert response["candidate"]["catbot_ai_repair_accepted"] is True
    assert response["candidate"]["reply_plan_validation"] == ""


def test_catbot_ai_route_accepts_greeting_in_romantic_thread(monkeypatch, training_service: PhoneCopilotService, provider_path) -> None:
    assert training_service._catbot_ai_reject_reason(
        "hey youuu",
        incoming="hello",
        context=["hey", "hey u alright"],
        recent_bot_replies=[],
    ) == "generic_ai_style"

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": "hey yourself what u on",
                "provider": "gemini",
                "model": "fake-model",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="hello",
            context=["hey", "hey u alright", "i missed you", "i missed u too like mad"],
            contact_name="Catbot",
            relationship_type="romantic_interest",
        )
    )

    assert response["reply"] == "hey yourself what u on"
    assert response["retrieval_backend"] == "catbot_ai_romantic"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_reject_reason"] == ""


def test_catbot_ai_route_accepts_hey_you_greeting_variant(training_service: PhoneCopilotService) -> None:
    reason = training_service._catbot_ai_reject_reason(
        "hey u\ndidn't think you'd still be messaging me after all that lol",
        incoming="hey you",
        context=["answer properly", "like i said / how else am i supposed to entertain u"],
        recent_bot_replies=[],
    )

    assert reason == ""


def test_catbot_ai_route_accepts_guess_what_curiosity(training_service: PhoneCopilotService) -> None:
    assert training_service._catbot_ai_reject_reason(
        "what",
        incoming="guess what",
        context=["hey you", "hey you too / what did u want me to say then"],
        recent_bot_replies=[],
    ) == "too_short"

    assert training_service._catbot_ai_reject_reason(
        "go on then what happened",
        incoming="guess what",
        context=["hey you", "hey you too / what did u want me to say then"],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "ooh tell me then",
        incoming="guess what",
        context=["hey you", "hey you too / what did u want me to say then"],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "ooh what is it then / tell meee",
        incoming="guess what",
        context=["hey you", "hey you too / what did u want me to say then"],
        recent_bot_replies=[],
    ) == ""


def test_catbot_ai_route_accepts_reciprocal_activity(monkeypatch, training_service: PhoneCopilotService) -> None:
    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": "im good wby just chilling and coding",
                "provider": "gemini",
                "model": "fake-model",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="what u doing",
            context=["yo", "hey there"],
            contact_name="Catbot",
            relationship_type="romantic_interest",
        )
    )

    assert response["reply"] == "im good wby just chilling and coding"
    assert response["retrieval_backend"] == "catbot_ai_romantic"
    assert response["fallback_used"] is False
    assert response["candidate"]["catbot_ai_reject_reason"] == ""


def test_catbot_route_reciprocal_activity_answers_back(training_service: PhoneCopilotService) -> None:
    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="chilling wby",
            context=["hi", "yo what u saying"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )

    assert response["candidate"]["scene_type"] == "reciprocal_current_activity_question"
    assert response["candidate"]["required_reply_move"] == "answer_reciprocal_activity"
    assert response["reply"] in {"same icl just chilling", "same just been chilling", "nothing much just chilling"}
    assert response["reply"] != "yo what u saying"


def test_catbot_route_latest_wby_regression_answers_and_repairs(training_service: PhoneCopilotService) -> None:
    work = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="honestly just work wby",
            context=[
                "hi",
                "heyy what u doing",
                "chilling wby",
                "nothing much just chilling",
                "i missed u baby",
                "that's sweet icl",
                "wyd",
                "fair what u been on today",
            ],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )

    assert work["candidate"]["scene_type"] == "reciprocal_current_activity_question"
    assert work["candidate"]["required_reply_move"] == "answer_reciprocal_activity"
    assert work["reply"] not in {"chilling and still bored?", "fair what u been on today", "u ain't giving me much to work with what u been doing"}

    sleep = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="prolly sleeping wby",
            context=[
                "honestly just work wby",
                work["reply"],
                "yeah im bored asf",
                "valid what u doing later",
            ],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )

    assert sleep["candidate"]["scene_type"] == "reciprocal_current_activity_question"
    assert sleep["candidate"]["required_reply_move"] == "answer_reciprocal_activity"
    assert sleep["reply"] != "u ain't giving me much to work with what u been doing"
    assert "what u been on today" not in sleep["reply"]

    repair = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="i told and asked u...",
            context=[
                "honestly just work wby",
                "chilling and still bored?",
                "yeah im bored asf",
                "valid what u doing later",
                "prolly sleeping wby",
                "u ain't giving me much to work with what u been doing",
            ],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )

    assert repair["candidate"]["scene_type"] == "missed_context_callout"
    assert repair["candidate"]["required_reply_move"] == "answer_reciprocal_activity"
    assert "answer previous reciprocal activity question" in repair["candidate"]["unresolved_user_points"]
    assert repair["reply"] not in {"alr random one then dream car?", "fine ill carry it, what's been on ur mind"}
    assert any(term in repair["reply"] for term in ("my bad", "missed", "ignored", "didnt answer", "didn't answer"))


def test_catbot_route_missed_fact_callout_repairs_specific_fact(training_service: PhoneCopilotService) -> None:
    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="bro i js said im chilling",
            context=["hi", "yo what u saying", "chilling wby", "yo what u saying"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )

    assert response["candidate"]["scene_type"] == "missed_user_fact_callout"
    assert response["candidate"]["required_reply_move"] == "acknowledge_missed_fact"
    assert response["reply"] in {
        "yh my bad i missed that",
        "icl i ignored what u said there",
        "fairs i didn't clock it",
        "yeah my bad u did say that",
    }
    assert response["reply"] != "yh fairs i did icl"


def test_catbot_route_repair_clarification_explains_mistake(training_service: PhoneCopilotService) -> None:
    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="wdym u did",
            context=["chilling wby", "yo what u saying", "bro i js said im chilling", "yh fairs i did icl"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )

    assert response["candidate"]["scene_type"] == "repair_clarification"
    assert response["candidate"]["required_reply_move"] == "clarify_previous_repair"
    assert response["reply"] in {
        "i mean i missed what u said",
        "i meant i bugged and ignored ur message",
        "icl i answered the wrong thing",
    }


def test_catbot_route_day_check_answers_day(training_service: PhoneCopilotService) -> None:
    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="oh ok how was ur day",
            context=["wdym u did", "i meant i bugged and ignored ur message"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )

    assert response["candidate"]["scene_type"] == "day_check_question"
    assert response["candidate"]["required_reply_move"] == "answer_day_check"
    assert response["reply"] in {"it was calm icl, bit dead", "not bad tbh just chilled", "decent icl nothing crazy", "long icl but calm"}
    assert response["reply"] != "same just chilling"


def test_catbot_route_low_info_after_bad_reply_repairs(training_service: PhoneCopilotService) -> None:
    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="oh",
            context=["oh ok how was ur day", "same just chilling"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )

    assert response["candidate"]["scene_type"] == "low_info_after_bad_reply"
    assert response["candidate"]["required_reply_move"] == "repair_or_prompt_lightly"
    assert response["reply"] in {"yh that reply was dead icl", "icl i answered that badly", "ignore me im waffling"}
    assert response["reply"] != "fair just chilling too"


def test_catbot_route_provider_stale_reply_rejected_for_day_check(training_service: PhoneCopilotService) -> None:
    training_service.settings.ai_reply_enabled = True
    training_service.drafting.ai_reply_enabled = True
    training_service.drafting.draft_provider = "gemini"
    training_service.drafting._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        type("FakeResponse", (), {
            "text": '{"candidates":[{"reply":"same just chilling","reason":"stale"}]}',
            "provider": "gemini",
            "model": "fake",
            "latency_ms": 1,
            "raw_finish_reason": None,
            "error": None,
            "external_api_used": True,
        })(),
        {"provider_configured": True},
    )

    response = training_service.training_catbot_chat(
        CatbotChatRequest(incoming="oh ok how was ur day", context=[], contact_name="Catbot", relationship_type="close_friend")
    )

    assert response["candidate"]["scene_type"] == "day_check_question"
    assert response["reply"] != "same just chilling"
    stale = [item for item in response.get("candidates", []) if item.get("text") == "same just chilling"]
    assert not stale or stale[0]["final_decision"] == "reject"


def test_catbot_route_explains_previous_day_claim(training_service: PhoneCopilotService) -> None:
    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="how come",
            context=["how was ur day", "it was calm icl, bit dead"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )

    candidate = response["candidate"]
    assert candidate["scene_type"] == "explain_previous_bot_claim"
    assert candidate["required_reply_move"] == "explain_previous_bot_claim"
    assert candidate["previous_bot_claim_type"] == "day_summary"
    assert candidate["explanation_required"] is True
    assert response["reply"] in {"just didn't do much icl", "nothing really happened tbh", "just one of them dead days", "was just boring icl"}
    assert response["reply"] != "bro why is his dad involved"


def test_catbot_route_wdym_after_random_story_repairs_context(training_service: PhoneCopilotService) -> None:
    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="wdym",
            context=["how come", "bro why is his dad involved"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )

    candidate = response["candidate"]
    assert candidate["scene_type"] == "explain_previous_bot_claim"
    assert candidate["previous_bot_claim_type"] == "weird_story_reaction"
    assert response["reply"] in {
        "icl that was random, wrong context",
        "yeah ignore that i answered the wrong thing",
        "my bad that made no sense",
        "yh my bad that was random",
        "icl that made no sense",
    }
    assert response["reply"] != "yeah that was a dead reply icl"


def test_catbot_route_light_acknowledgement_does_not_use_activity_fallback(training_service: PhoneCopilotService) -> None:
    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="oh yeah silly me",
            context=["nice wyd", "same just chilling"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )

    assert response["candidate"]["scene_type"] == "light_acknowledgement"
    assert response["candidate"]["required_reply_move"] == "playful_acknowledge_or_move_on"
    assert response["reply"] in {"lool ur good", "yh ur good", "fairs fairs", "allow it lol"}
    assert response["reply"] != "fair just chilling too"


def test_catbot_route_tired_mood_gets_empathy_or_followup(training_service: PhoneCopilotService) -> None:
    response = training_service.training_catbot_chat(
        CatbotChatRequest(incoming="im so tired", context=[], contact_name="Catbot", relationship_type="close_friend")
    )

    assert response["candidate"]["scene_type"] == "tired_mood"
    assert response["candidate"]["required_reply_move"] == "empathetic_casual_response"
    assert response["reply"] in {"same icl go sleep then", "why u tired", "go nap then", "long day?", "icl same im finished"}
    assert response["reply"] != "same icl"


def test_catbot_route_provider_weird_story_rejected_without_context(training_service: PhoneCopilotService) -> None:
    training_service.settings.ai_reply_enabled = True
    training_service.drafting.ai_reply_enabled = True
    training_service.drafting.draft_provider = "gemini"
    training_service.drafting._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        type("FakeResponse", (), {
            "text": '{"candidates":[{"reply":"bro why is his dad involved","reason":"wrong scene"}]}',
            "provider": "gemini",
            "model": "fake",
            "latency_ms": 1,
            "raw_finish_reason": None,
            "error": None,
            "external_api_used": True,
        })(),
        {"provider_configured": True},
    )

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="how come",
            context=["how was ur day", "it was calm icl, bit dead"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )

    assert response["reply"] != "bro why is his dad involved"
    assert response["candidate"]["scene_type"] == "explain_previous_bot_claim"
    bad = [item for item in response.get("candidates", []) if item.get("text") == "bro why is his dad involved"]
    assert bad
    assert bad[0]["final_decision"] == "reject"
    assert bad[0]["semantic_contamination_penalty"] >= 0.9


def test_catbot_route_provider_activity_fallback_rejected_for_light_ack(training_service: PhoneCopilotService) -> None:
    training_service.settings.ai_reply_enabled = True
    training_service.drafting.ai_reply_enabled = True
    training_service.drafting.draft_provider = "gemini"
    training_service.drafting._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        type("FakeResponse", (), {
            "text": '{"candidates":[{"reply":"fair just chilling too","reason":"wrong scene"}]}',
            "provider": "gemini",
            "model": "fake",
            "latency_ms": 1,
            "raw_finish_reason": None,
            "error": None,
            "external_api_used": True,
        })(),
        {"provider_configured": True},
    )

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="oh yeah silly me",
            context=["nice wyd", "same just chilling"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )

    assert response["reply"] != "fair just chilling too"
    assert response["candidate"]["scene_type"] == "light_acknowledgement"
    bad = [item for item in response.get("candidates", []) if item.get("text") == "fair just chilling too"]
    assert bad
    assert bad[0]["final_decision"] == "reject"
    assert bad[0]["fallback_scene_mismatch"] is True


def test_catbot_thread_persists_labelled_turns_and_candidate_metadata(training_service: PhoneCopilotService) -> None:
    response = training_service.training_catbot_chat(
        CatbotChatRequest(incoming="hi", context=[], contact_name="Catbot", relationship_type="close_friend")
    )

    thread_id = str(response["thread_id"])
    episode = training_service.drafting.thread_memory.get_episode(thread_id)

    assert thread_id.startswith("catbot_thread_")
    assert episode is not None
    assert [turn.role for turn in episode.turns] == ["user", "bot"]
    assert episode.turns[0].text == "hi"
    assert episode.turns[1].candidate_metadata
    assert episode.turns[1].candidate_metadata["scene_type"] == response["candidate"]["scene_type"]
    assert response["adb_touched"] is False


def test_catbot_reset_thread_starts_new_memory_episode(training_service: PhoneCopilotService) -> None:
    first = training_service.training_catbot_chat(
        CatbotChatRequest(incoming="hi", context=[], contact_name="Catbot", relationship_type="close_friend")
    )
    second = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="hi",
            context=[],
            contact_name="Catbot",
            relationship_type="close_friend",
            thread_id=str(first["thread_id"]),
            reset_thread=True,
        )
    )

    assert first["thread_id"] != second["thread_id"]
    old_episode = training_service.drafting.thread_memory.get_episode(str(first["thread_id"]))
    new_episode = training_service.drafting.thread_memory.get_episode(str(second["thread_id"]))
    assert old_episode is not None and len(old_episode.turns) == 2
    assert new_episode is not None and len(new_episode.turns) == 2


def test_catbot_thread_memory_blocks_stale_pattern_after_failure(training_service: PhoneCopilotService) -> None:
    thread_id = "catbot_thread_failure_case"
    store = training_service.drafting.thread_memory
    store.append_turn(thread_id, session_id="manual", role="user", text="chilling wby", relationship_type="close_friend")
    store.append_turn(thread_id, session_id="manual", role="bot", text="nothing much just chilling", relationship_type="close_friend")
    store.append_turn(thread_id, session_id="manual", role="user", text="nice wyd", relationship_type="close_friend")
    episode = store.append_turn(thread_id, session_id="manual", role="bot", text="same just chilling", relationship_type="close_friend")
    assert "stale activity fallback" in episode.bot_mistakes

    training_service.settings.ai_reply_enabled = True
    training_service.drafting.ai_reply_enabled = True
    training_service.drafting.draft_provider = "gemini"
    training_service.drafting._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        type("FakeResponse", (), {
            "text": '{"candidates":[{"reply":"same icl","reason":"stale"},{"reply":"why u tired","reason":"better"}]}',
            "provider": "gemini",
            "model": "fake",
            "latency_ms": 1,
            "raw_finish_reason": None,
            "error": None,
            "external_api_used": True,
        })(),
        {"provider_configured": True},
    )

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="im so tired",
            context=["oh yeah silly me", "fair just chilling too"],
            contact_name="Catbot",
            relationship_type="close_friend",
            thread_id=thread_id,
        )
    )

    assert response["reply"] != "same icl"
    assert response["candidate"]["scene_type"] == "tired_mood"
    assert response["candidate"]["thread_memory_used"] is True
    stale = [item for item in response.get("candidates", []) if item.get("text") == "same icl"]
    assert stale
    assert stale[0]["repeated_failed_pattern_penalty"] >= 0.7


def test_catbot_route_identity_sequence_answers_direct_questions(training_service: PhoneCopilotService, monkeypatch) -> None:
    async def fake_generate_with_fallback(**kwargs):
        messages = kwargs.get("messages") or []
        prompt = "\n".join(str(getattr(message, "content", "")) for message in messages)
        if "them: like im 19 wby" in prompt:
            reply = "same age then / im 19 too"
        elif "them: hm how old r u" in prompt:
            reply = "im 19 / why u asking like that"
        elif "them: dwdw i missed u anyway where u been" in prompt:
            reply = "missed u too / been busy with gym and projects"
        elif "them: trust me u doing anything nice?" in prompt:
            reply = "nothing crazy just gym and coding stuff thinking about you tho"
        elif "them: i been chilling man this heats killing me" in prompt:
            reply = "same this heat is killing me too / thinking about you is not helping icl"
        elif "them: im good wby" in prompt:
            reply = "im good just chilling thinking about you"
        else:
            reply = "hey u / what u been up to"
        return (
            type("FakeResponse", (), {
                "text": reply,
                "provider": "gemini",
                "model": "fake-model",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    context: list[str] = []
    replies: list[dict[str, object]] = []
    for incoming in [
        "hey",
        "im good wby",
        "i been chilling man this heats killing me",
        "trust me u doing anything nice?",
        "dwdw i missed u anyway where u been",
        "hm how old r u",
        "like im 19 wby...",
    ]:
        response = training_service.training_catbot_chat(
            CatbotChatRequest(
                incoming=incoming,
                context=context,
                contact_name="Catbot",
                relationship_type="romantic_interest",
            )
        )
        replies.append(response)
        context.extend([incoming, str(response["reply"])])

    age = replies[5]
    reciprocal_age = replies[6]

    assert replies[4]["candidate"]["reply_plan_move"] == "emotional_reciprocity"
    assert "missed u" in replies[4]["reply"]
    assert age["candidate"]["reply_plan_move"] == "identity_fact_question"
    assert age["candidate"]["reply_plan_shape"] == "age_fact_then_textured_continue"
    assert "19" in age["reply"]
    assert reciprocal_age["candidate"]["reply_plan_move"] == "reciprocal_question"
    stale = {"same icl", "same just chilling", "fair just chilling too", "what u saying", "say less what u doing", "ok"}
    assert all(response["reply"] not in stale for response in replies[3:])


def test_catbot_route_provider_hook_rejected_for_identity_question(training_service: PhoneCopilotService) -> None:
    training_service.settings.ai_reply_enabled = True
    training_service.drafting.ai_reply_enabled = True
    training_service.drafting.draft_provider = "gemini"
    training_service.drafting._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        type("FakeResponse", (), {
            "text": '{"candidates":[{"reply":"what u saying","reason":"generic hook"}]}',
            "provider": "gemini",
            "model": "fake",
            "latency_ms": 1,
            "raw_finish_reason": None,
            "error": None,
            "external_api_used": True,
        })(),
        {"provider_configured": True},
    )

    response = training_service.training_catbot_chat(
        CatbotChatRequest(incoming="hm how old r u", context=[], contact_name="Catbot", relationship_type="close_friend")
    )

    assert response["candidate"]["scene_type"] == "owner_age_question"
    assert "19" in response["reply"]
    bad = [candidate for candidate in response.get("candidates", []) if candidate.get("text") == "what u saying"]
    assert bad
    assert bad[0]["final_decision"] == "reject"
    assert bad[0]["ignored_identity_question_penalty"] >= 0.7


def test_ai_core_reload_identity_pack_returns_safe_summary(training_service: PhoneCopilotService) -> None:
    response = training_service.ai_core_reload_identity_pack()

    assert response["status"] == "reloaded"
    assert response["identity_pack_loaded"] is True
    assert "Alex" in response["safe_identity_summary"]
    assert "API keys" not in response["safe_identity_summary"]


def test_catbot_route_fresh_thread_does_not_share_exact_university(training_service: PhoneCopilotService) -> None:
    response = training_service.training_catbot_chat(
        CatbotChatRequest(incoming="what uni", context=[], contact_name="Catbot", relationship_type="close_friend")
    )

    assert response["candidate"]["scene_type"] == "owner_study_question"
    assert response["candidate"]["identity_disclosure_allowed"] is False
    assert "sampleford university" not in response["reply"].casefold()
    assert "sampleford uni" not in response["reply"].casefold()


def test_catbot_route_regression_question_repair_topic_identity_sequence(training_service: PhoneCopilotService) -> None:
    day_followup = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="whatd u do",
            context=["how was ur day", "it was calm icl, bit dead", "why", "just one of them dead days"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )
    assert day_followup["candidate"]["scene_type"] == "explain_previous_bot_claim"
    assert day_followup["reply"] != "same just chilling"

    repair = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="im asking a question. not trying to be right",
            context=[
                "whatd u do",
                "same just chilling",
                "bro what?",
                "yeah that was dead my bad",
                "it wasnt dead it didnt make sense",
                "u ain't giving me much to work with what u been doing",
                "how wtf",
                "nah ur right that was dumb",
            ],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )
    assert repair["candidate"]["scene_type"] == "missed_context_callout"
    assert repair["reply"] not in {"u ain't giving me much to work with what u been doing", "fine then what should we talk about"}

    age_answer = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="21",
            context=["how old r u", "19 wby"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )
    assert age_answer["candidate"]["scene_type"] == "reciprocal_identity_answer"
    assert "random" not in age_answer["reply"]

    topic = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="im into tech",
            context=["hows it random lol", "nah its not what u into then"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )
    assert topic["candidate"]["scene_type"] == "topic_given"
    assert "tech" in topic["reply"]
    assert topic["reply"] != "what u saying then"

    study = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="what dyu study",
            context=["im into tech", "tech is cold icl what side"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )
    assert study["candidate"]["scene_type"] == "owner_study_question"
    assert "comp sci" in study["reply"] or "computer science" in study["reply"]
    assert study["reply"] != "ok"


def test_catbot_route_today_activity_and_wytm_repair_carry_conversation(training_service: PhoneCopilotService) -> None:
    today = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="whatd u do today",
            context=["hi", "yo how u been", "chilling u", "nothing much just chilling"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )
    assert today["candidate"]["scene_type"] == "owner_day_activity_question"
    assert today["candidate"]["required_reply_move"] == "answer_owner_day_activity"
    assert today["reply"] not in {"same just chilling", "same icl", "fair just chilling too"}
    assert any(term in today["reply"] for term in ("gym", "coding", "uni", "work"))

    repair = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="wdym same js chilling bro that doesnt make sense",
            context=["hi", "yo how u been", "chilling u", "nothing much just chilling", "whatd u do today", "same just chilling"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )
    assert repair["candidate"]["scene_type"] == "missed_context_callout"
    assert repair["candidate"]["required_reply_move"] == "answer_owner_day_activity"
    assert repair["reply"] not in {"yh that made no sense icl", "same just chilling", "fair just chilling too"}
    assert any(term in repair["reply"] for term in ("gym", "coding", "uni", "work"))

    clarification = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="exsactly wytm tho",
            context=["whatd u do today", "same just chilling", "wdym same js chilling bro that doesnt make sense", "yh that made no sense icl"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )
    assert clarification["candidate"]["scene_type"] == "repair_clarification"
    assert clarification["candidate"]["required_reply_move"] == "answer_owner_day_activity"
    assert clarification["reply"] != "fair just chilling too"
    assert any(term in clarification["reply"] for term in ("gym", "coding", "uni", "work"))


def test_catbot_route_agenda_suppresses_generic_prompt_loop(training_service: PhoneCopilotService) -> None:
    repeated = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="u asked this already",
            context=[
                "yo",
                "yo what u been up to",
                "chilling wyd",
                "same just chilling",
                "whatchu been up to",
                "same just chilling",
                "yhhh idk what to say tbh",
                "yh same what u been up to",
            ],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )
    assert repeated["candidate"]["agenda_state"] == "anti_loop_repair"
    assert repeated["candidate"]["anti_loop_required"] is True
    assert repeated["reply"] not in {"what u saying", "what u saying then", "what u been up to", "ok"}

    already = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="u already asked that...",
            context=["yhhh idk what to say tbh", "yh same what u been up to", "u asked this already", "what u saying"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )
    assert already["candidate"]["agenda_state"] == "anti_loop_repair"
    assert already["reply"] != "ok"
    assert "what u saying" not in already["reply"]


def test_catbot_route_agenda_chooses_topic_when_user_asks_bot_to_lead(training_service: PhoneCopilotService) -> None:
    talk = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="idk talk",
            context=["ur boring me", "what u tryna do then"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )
    assert talk["candidate"]["agenda_state"] == "topic_selection_needed"
    assert talk["candidate"]["next_dialogue_move"] == "choose_topic"
    assert talk["reply"] not in {"what u saying", "fine then what should we talk about", "u ain't giving me much to work with what u been doing"}
    assert talk["candidate"]["chosen_topic"]

    tell_me = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="u tell me",
            context=["lmao what dyu want me to say to that", "fine then what should we talk about"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )
    assert tell_me["candidate"]["agenda_state"] == "topic_selection_needed"
    assert tell_me["reply"] != "what u saying"
    assert tell_me["candidate"]["agenda_fit_score"] > 0.7


def test_catbot_route_agenda_provider_generic_prompt_rejected(training_service: PhoneCopilotService) -> None:
    training_service.settings.ai_reply_enabled = True
    training_service.drafting.ai_reply_enabled = True
    training_service.drafting.draft_provider = "gemini"
    training_service.drafting._run_provider_generation = lambda **kwargs: (  # type: ignore[method-assign]
        type("FakeResponse", (), {
            "text": '{"candidates":[{"reply":"what u saying","reason":"bad"}]}',
            "provider": "gemini",
            "model": "fake-model",
            "latency_ms": 7,
            "raw_finish_reason": None,
            "error": None,
            "external_api_used": True,
        })(),
        {"provider_configured": True},
    )

    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="u asked this already",
            context=["yo", "yo what u been up to", "yhhh idk what to say tbh", "yh same what u been up to"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )
    assert response["reply"] != "what u saying"
    bad = [item for item in response.get("candidates", []) if item.get("text") == "what u saying"]
    assert bad
    assert bad[0]["forbidden_dialogue_move_violated"] is True


def test_catbot_route_avoids_reused_opening_and_activity_templates(training_service: PhoneCopilotService) -> None:
    opening = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="hi",
            context=["hi", "yo what u saying"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )
    assert opening["candidate"]["scene_type"] == "opening"
    assert opening["reply"] != "yo what u saying"

    activity = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="chilling wyd",
            context=["chilling wyd", "nothing much just chilling"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )
    assert activity["candidate"]["scene_type"] == "reciprocal_current_activity_question"
    assert activity["reply"] != "nothing much just chilling"
    assert activity["reply"] not in {"same just chilling", "fair just chilling too", "same icl"}


def test_catbot_route_owner_activity_claim_followups_do_not_use_stale_fallback(training_service: PhoneCopilotService, monkeypatch) -> None:
    claim = "been busy icl uni gym coding clients all of it"
    replies_out = iter([
        "yh really / been all over the place with gym and coding",
        "hit gym then did some coding after",
        "coding is just projects and client stuff atm",
        "yeah fair i worded that badly",
    ])

    async def fake_generate_with_fallback(**kwargs):
        return (
            type("FakeResponse", (), {
                "text": next(replies_out),
                "provider": "gemini",
                "model": "fake-model",
                "latency_ms": 1,
                "raw_finish_reason": None,
                "error": None,
                "external_api_used": True,
            })(),
            {"provider_configured": True},
        )

    monkeypatch.setattr("apps.controller.service.generate_with_fallback", fake_generate_with_fallback)

    really = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="really",
            context=["what have u been up to baby", claim],
            contact_name="Catbot",
            relationship_type="romantic_interest",
        )
    )
    assert really["candidate"]["reply_plan_move"] == "intensity_check"
    assert really["reply"] != "same just chilling"

    gym_contract = training_service._catbot_turn_contract(
        incoming="what did u do in gym",
        context=["what have u been up to baby", claim, "really", really["reply"]],
        intent="auto",
    )
    assert gym_contract.reply_plan.shape == "answer_activity_detail_then_continue"
    assert training_service._catbot_reply_violates_plan("fair just chilling too", gym_contract.reply_plan) == "missing_activity_detail"
    assert training_service._catbot_ai_reject_reason(
        "fair just chilling too",
        incoming="what did u do in gym",
        context=["what have u been up to baby", claim, "really", really["reply"]],
        recent_bot_replies=[],
        contract=gym_contract,
    ) == "generic_ai_style"

    coding_contract = training_service._catbot_turn_contract(
        incoming="i dont know how to code lol",
        context=["what have u been up to baby", claim, "what did u do in gym", "fair just chilling too"],
        intent="auto",
    )
    assert coding_contract.reply_plan.shape == "short_reaction_plus_specific_continuation"
    assert training_service._catbot_ai_reject_reason(
        "same icl",
        incoming="i dont know how to code lol",
        context=["what have u been up to baby", claim, "what did u do in gym", "fair just chilling too"],
        recent_bot_replies=[],
        contract=coding_contract,
    ) in {"too_dry", "too_short", "not_carrying_conversation"}

    contradiction = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="it wasnt dead it was a lie",
            context=["i dont know how to code lol", "coding is just projects and client stuff atm", "wdym u lit do coding", "yeah that was a dead reply icl"],
            contact_name="Catbot",
            relationship_type="romantic_interest",
        )
    )
    assert contradiction["reply"] != "same icl"
    assert contradiction["fallback_used"] is False


def test_catbot_route_user_busy_update_and_paradox_callout(training_service: PhoneCopilotService) -> None:
    busy = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="i been busy asf",
            context=["what u been up to", "nothing crazy icl wbu"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )
    assert busy["candidate"]["scene_type"] == "user_activity_update"
    assert busy["reply"] != "same just chilling"
    assert "busy" in busy["reply"] or "doing what" in busy["reply"]

    paradox = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="thats a paradox lol",
            context=["what u been up to", "nothing crazy icl wbu", "i been busy asf", "same just chilling"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )
    assert paradox["candidate"]["scene_type"] == "contradiction_callout"
    assert paradox["reply"] != "yh yh but what u been busy with then"
    assert paradox["reply"] != "same just chilling"


def test_catbot_route_full_spelling_affection_does_not_use_generic_hook(training_service: PhoneCopilotService) -> None:
    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="i missed you",
            context=["hi", "yo what u saying", "how are you", "yh im good wbu"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )

    assert response["candidate"]["scene_type"] == "emotional_affection"
    assert response["reply"] != "say less what u doing"
    assert any(term in response["reply"] for term in ("sweet", "bless", "appreciate"))


def test_catbot_route_wellbeing_checkin_does_not_repeat_ok(training_service: PhoneCopilotService) -> None:
    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="good how are u",
            context=["hi", "yo how u been"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )

    assert response["candidate"]["scene_type"] == "reciprocal_wellbeing_question"
    assert response["candidate"]["required_reply_move"] == "answer_wellbeing_checkin"
    assert response["reply"] in {"im good wbu", "good u", "im calm wbu", "not bad icl wbu", "im good icl", "im calm honestly", "im bless icl"}
    assert response["reply"] != "ok"


def test_catbot_route_nice_after_ok_recovers(training_service: PhoneCopilotService) -> None:
    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="nice",
            context=["good how are u", "ok"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )

    assert response["candidate"]["scene_type"] == "low_info_after_bad_reply"
    assert response["reply"] != "ok"


def test_catbot_route_keep_saying_ok_gets_repetition_repair(training_service: PhoneCopilotService) -> None:
    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="why u keep saying ok",
            context=["good how are u", "ok", "nice", "ok"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )

    assert response["candidate"]["scene_type"] == "repeated_reply_callout"
    assert response["reply"] != "ok"
    assert response["candidate"]["repeated_reply_callout_detected"] is True


def test_training_regenerate_avoids_previous_hii_candidates(training_service: PhoneCopilotService) -> None:
    response = training_service.training_regenerate(
        TrainingChatRequest(
            incoming="hii",
            relationship_type="unknown",
            intent_type="greeting",
            diversity_mode="alternative_wording",
            avoid_candidates=["yo", "heyy", "u good"],
        )
    )

    replies = [candidate["text"] for candidate in response["candidates"]]
    assert replies
    assert not set(replies) & {"yo", "heyy", "u good"}
    assert all(not any(term in reply.lower() for term in ("baby", "sexy", "trouble", "love", "xx")) for reply in replies)


def test_training_feedback_saves_high_authority_correction(training_service: PhoneCopilotService) -> None:
    response = training_service.training_feedback(
        TrainingFeedbackRequest(
            incoming="you coming later?",
            context=[],
            contact_name="Ali",
            relationship_type="close_friend",
            intent_type="planning",
            ai_reply="That sounds great.",
            correct_reply="yh what time",
            feedback_label="edited",
            notes="shorter",
            save_to_corrections=True,
            add_to_vector_db=False,
        )
    )

    correction = response["correction"]
    assert correction["source"] == "training_page_feedback"
    assert correction["style_authority"] == "high"
    assert correction["feedback_label"] == "edited"
    assert correction["contact_name"] == "Ali"
    assert correction["intent_type"] == "planning"

    loaded = load_corrections(training_service.settings.ai_reply_training_messages_dir)
    assert loaded[-1]["source"] == "training_page_feedback"
    assert loaded[-1]["parameters"]


def test_training_feedback_rejects_empty_correction(training_service: PhoneCopilotService) -> None:
    with pytest.raises(HTTPException) as exc:
        training_service.training_feedback(
            TrainingFeedbackRequest(
                incoming="hii",
                relationship_type="unknown",
                intent_type="greeting",
                ai_reply="yo",
                correct_reply="",
                feedback_label="edited",
                save_to_corrections=True,
            )
        )

    assert exc.value.status_code == 400


def test_training_parameters_do_not_override_hard_gates(training_service: PhoneCopilotService) -> None:
    response = training_service.training_chat(
        TrainingChatRequest(
            incoming="hii",
            relationship_type="unknown",
            intent_type="greeting",
            parameters={"risk_tolerance": 1.0},
        )
    )

    assert {candidate["final_decision"] for candidate in response["candidates"]} == {"review"}
    assert all(candidate["auto_send_allowed"] is False for candidate in response["candidates"])


def test_recent_and_quarantine_training_session(training_service: PhoneCopilotService) -> None:
    saved = training_service.training_feedback(
        TrainingFeedbackRequest(
            incoming="hii",
            relationship_type="unknown",
            intent_type="greeting",
            ai_reply="yo",
            correct_reply="yo",
            feedback_label="approved",
            save_to_corrections=True,
            add_to_vector_db=False,
        )
    )

    recent = training_service.training_recent(limit=5)
    assert recent["items"]
    assert recent["items"][-1]["id"] == saved["session"]["id"]

    quarantined = training_service.training_quarantine_example(
        TrainingQuarantineRequest(example_id=saved["session"]["id"], reason="bad training data")
    )
    assert quarantined["status"] == "quarantined"
    assert training_service.training_stats()["vector_rebuild_required"] is True
    assert all(item["id"] != saved["session"]["id"] for item in training_service.training_recent(limit=20)["items"])


def test_saved_correction_outranks_synthetic_template(training_service: PhoneCopilotService) -> None:
    training_service.training_feedback(
        TrainingFeedbackRequest(
            incoming="you coming later?",
            relationship_type="close_friend",
            intent_type="planning",
            ai_reply="Yes, I can attend later.",
            correct_reply="yh what time",
            feedback_label="edited",
            save_to_corrections=True,
            add_to_vector_db=False,
        )
    )
    rows = [
        {
            "relationship_type": "close_friend",
            "intent_type": "planning",
            "incoming": "you coming later?",
            "context": [],
            "my_reply": "sure what time works for you",
            "is_synthetic": True,
            "_source": "synthetic",
        },
        *[
            {
                **row,
                "my_reply": row["user_final_reply"],
                "_source": "correction",
                "is_correction": True,
            }
            for row in load_corrections(training_service.settings.ai_reply_training_messages_dir)
        ],
    ]

    result = LexicalRetrievalBackend(rows).search(
        incoming="you coming later?",
        context=[],
        relationship_type="close_friend",
        intent_type="planning",
        limit=3,
    )

    assert result
    assert result[0].source == "correction"
    assert result[0].my_reply == "yh what time"


def test_training_rebuild_endpoint_reports_status(training_service: PhoneCopilotService, monkeypatch) -> None:
    import sys
    from types import ModuleType

    def fake_build_reply_vector_db() -> dict[str, object]:
        return {"rows_indexed": 2, "dependency_model_errors": "none"}

    fake_module = ModuleType("scripts.build_reply_vector_db")
    fake_module.build_reply_vector_db = fake_build_reply_vector_db
    monkeypatch.setitem(sys.modules, "scripts.build_reply_vector_db", fake_module)

    response = training_service.training_rebuild_vector_db()

    assert response["status"] == "rebuilt"
    assert response["rows_indexed"] == 2


def test_style_review_queue_approval_moves_row_into_corrections(training_service: PhoneCopilotService) -> None:
    review_path = training_service.style_review_queue_path
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_row = {
        "row_id": "review_1",
        "timestamp": "2026-05-20T00:00:00Z",
        "incoming": "you coming later?",
        "context": [],
        "relationship_type": "close_friend",
        "intent_type": "planning",
        "contact_name": "Ali",
        "bad_ai_reply": "maybe later",
        "suggested_better_reply": "yh what time",
        "style_score": 92,
        "reason_bad": "invents availability or facts",
        "improvement_notes": ["Ask for the missing detail instead of inventing availability."],
        "source": "auto_style_improvement",
        "status": "needs_human_review",
    }
    review_path.write_text(json.dumps(review_row) + "\n", encoding="utf-8")

    response = training_service.training_style_review_approve(StyleReviewApproveRequest(row_id="review_1"))

    assert response["status"] == "approved"
    assert response["vector_rebuild_required"] is True
    loaded = load_corrections(training_service.settings.ai_reply_training_messages_dir)
    assert loaded[-1]["source"] == "approved_auto_style_improvement"
    assert loaded[-1]["approved_by_user"] is True
    assert loaded[-1]["user_final_reply"] == "yh what time"


def test_style_review_reject_does_not_add_correction(training_service: PhoneCopilotService) -> None:
    review_path = training_service.style_review_queue_path
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_row = {
        "row_id": "review_2",
        "timestamp": "2026-05-20T00:00:00Z",
        "incoming": "hii",
        "context": [],
        "relationship_type": "unknown",
        "intent_type": "greeting",
        "contact_name": "",
        "bad_ai_reply": "keep talking like that baby",
        "suggested_better_reply": "yo",
        "style_score": 91,
        "reason_bad": "too flirty without context",
        "improvement_notes": ["Use a non-flirty reply for this relationship/context."],
        "source": "auto_style_improvement",
        "status": "needs_human_review",
    }
    review_path.write_text(json.dumps(review_row) + "\n", encoding="utf-8")

    response = training_service.training_style_review_reject(StyleReviewRejectRequest(row_id="review_2", reason="unsafe"))

    assert response["status"] == "rejected"
    assert training_service.training_style_review_queue()["total_pending"] == 0
    assert all(row.get("source") != "approved_auto_style_improvement" for row in load_corrections(training_service.settings.ai_reply_training_messages_dir))


def test_catbot_route_good_wby_answers_reciprocal_wellbeing(training_service: PhoneCopilotService) -> None:
    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="i been good wby",
            context=["hey", "yo how u been"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )

    assert response["candidate"]["scene_type"] == "reciprocal_wellbeing_question"
    assert response["candidate"]["required_reply_move"] == "answer_wellbeing_checkin"
    assert response["reply"] not in {"how come", "ok"}
    assert any(term in response["reply"] for term in ("good", "calm", "bless", "not bad"))


def test_catbot_route_missed_good_wby_question_repairs_directly(training_service: PhoneCopilotService) -> None:
    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="i asked a question",
            context=["hey", "yo how u been", "i been good wby", "how come"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )

    assert response["candidate"]["scene_type"] == "missed_context_callout"
    assert response["candidate"]["required_reply_move"] == "answer_wellbeing_checkin"
    assert "answer previous wellbeing question" in response["candidate"]["unresolved_user_points"]
    assert response["reply"] not in {"ok", "what u saying", "how come"}
    assert any(term in response["reply"] for term in ("im good", "im calm", "im bless"))


def test_catbot_route_positive_mood_and_repeat_callout_are_not_dry(training_service: PhoneCopilotService) -> None:
    happy = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="but i js been happy",
            context=["i asked a question", "yh my bad im good icl"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )
    assert happy["candidate"]["scene_type"] == "positive_mood_update"
    assert happy["reply"] != "thats good to hear"
    assert any(term in happy["reply"] for term in ("as u should", "cute", "deserve", "happy"))

    hru = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="thanks hru",
            context=[
                "hey",
                "yo how u been",
                "i been good wby",
                "im good wbu",
                "but i js been happy",
                "as u should why u happy",
            ],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )
    assert hru["candidate"]["scene_type"] == "reciprocal_wellbeing_question"
    assert "wbu" not in hru["reply"]
    assert "wby" not in hru["reply"]

    repeated = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="u alrdy asked that",
            context=["thanks hru", "im good wbu"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )
    assert repeated["candidate"]["agenda_state"] == "anti_loop_repair"
    assert repeated["reply"] != "ok"


def test_catbot_rejects_provider_reply_written_as_girl(training_service: PhoneCopilotService) -> None:
    reason = training_service._catbot_ai_reject_reason(
        "ngl u know how to make a girl blush",
        incoming="because im talking to u baby",
        context=["hi", "heyy how u doing", "im so good", "thats good to hear what makes u so good then"],
        recent_bot_replies=[],
    )
    assert reason == "wrong_owner_gender"


def test_catbot_rejects_submissive_adult_pov_and_accepts_male_dominant_pov(training_service: PhoneCopilotService) -> None:
    context = [
        "im horny",
        "my hand on ur waist / pulling u in slow / my mouth by ur ear / keeping u close",
    ]
    assert training_service._catbot_ai_reject_reason(
        "make me yours / use me however u want",
        incoming="tell me then",
        context=context,
        recent_bot_replies=[],
    ) == "wrong_owner_gender"
    assert training_service._catbot_ai_reject_reason(
        "mmm / i need ur mouth on me / pulling on me / taking me deeper in / make u moan so loud",
        incoming="need ur lips on my neck",
        context=context,
        recent_bot_replies=[],
    ) == "wrong_owner_gender"
    assert training_service._catbot_ai_reject_reason(
        "my chest pressed close while my cock stays hard against u",
        incoming="tell me then",
        context=context,
        recent_bot_replies=[],
    ) == ""


def test_catbot_rejects_feminine_x_signoff_style(training_service: PhoneCopilotService) -> None:
    assert training_service._catbot_ai_reject_reason(
        "hey u x",
        incoming="hi",
        context=[],
        recent_bot_replies=[],
    ) == "wrong_owner_style"
    assert training_service._catbot_ai_reject_reason(
        "hey u / back again yeah",
        incoming="hi",
        context=[],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "hey u / where'd u go",
        incoming="hi",
        context=[],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "hey u / finally showed up",
        incoming="hi",
        context=[],
        recent_bot_replies=[],
    ) == "generic_ai_style"
    assert training_service._catbot_ai_reject_reason(
        "hey / what u up to",
        incoming="hi",
        context=[],
        recent_bot_replies=[],
    ) == ""
    assert training_service._catbot_ai_reject_reason(
        "hey there / what u been up to",
        incoming="hi",
        context=[],
        recent_bot_replies=[],
    ) == "wrong_owner_style"
    assert training_service._catbot_ai_reject_reason(
        "hi back / what u up to",
        incoming="hi",
        context=[],
        recent_bot_replies=[],
    ) == "wrong_owner_style"
    assert training_service._catbot_ai_reject_reason(
        "hey u back / what u upto",
        incoming="hi",
        context=[],
        recent_bot_replies=[],
    ) == "wrong_owner_style"
    assert training_service._catbot_ai_reject_reason(
        "hiii / wbu",
        incoming="hi",
        context=[],
        recent_bot_replies=[],
    ) == "wrong_owner_style"
    greeting_contract = training_service._catbot_turn_contract(
        incoming="hi",
        context=[],
        intent="auto",
    )
    assert greeting_contract.reply_plan.shape == "short_reaction_plus_specific_continuation"
    assert training_service._catbot_reply_violates_plan("hey u / wbu", greeting_contract.reply_plan) == "violates_plan_forbidden_pattern:wbu"
    assert training_service._catbot_reply_violates_plan("hey u / where'd u go", greeting_contract.reply_plan) == "violates_plan_forbidden_pattern:hey u /"
    assert training_service._catbot_ai_reject_reason(
        "hey / missed u too",
        incoming="hi",
        context=["u already asked how my day was / how i am", "yh fair my bad / no more how-are-u questions from me"],
        recent_bot_replies=[],
    ) == "overeager_greeting"
    assert training_service._catbot_ai_reject_reason(
        "hey you",
        incoming="hi",
        context=[],
        recent_bot_replies=[],
    ) == "generic_ai_style"


def test_catbot_accepts_romantic_intensity_followup(training_service: PhoneCopilotService) -> None:
    reason = training_service._catbot_ai_reject_reason(
        "crazy enough that ive been thinking about u all day",
        incoming="how crazy",
        context=["hi", "heyyy u good?", "i missed you baby", "i missed u too baby like crazy"],
        recent_bot_replies=[],
    )
    assert reason == ""


def test_catbot_route_policy_handles_story_care_and_weird_callouts(training_service: PhoneCopilotService) -> None:
    story = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="i was at home yeah and showering and some guy came in and started running in my living room",
            context=["oi guess what", "what happened"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )
    assert story["candidate"]["conversation_job"] == "react_to_story"
    assert story["reply"] != "ok"
    assert any(term in story["reply"].lower() for term in ("what", "nah", "bro", "wait", "why", "how"))

    care = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="its okay is everything alright?",
            context=[
                "thats a bit dry mate",
                "yeah fairs that was dry icl",
                "yeah so why u being dry",
                "yeah nah ur right my bad",
            ],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )
    assert care["candidate"]["conversation_job"] == "answer_care_check"
    assert any(term in care["reply"] for term in ("im good", "im okay", "im alright", "dw"))
    assert care["reply"] != "fine then what should we talk about"

    weird = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="bruh wtf",
            context=[
                "alex ur being weird",
                "u ain't giving me much to work with what u been doing",
            ],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )
    assert weird["candidate"]["conversation_job"] == "repair_after_weird_or_dry_reply"
    assert weird["reply"] != "bruh wtf is valid icl what bruh wtf u into"
    assert "u ain't giving me much" not in weird["reply"]
    assert weird["candidate"]["policy_fit_score"] >= 0.8


def test_catbot_route_question_debt_observed_sequence_answers_missed_wby(training_service: PhoneCopilotService) -> None:
    first = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="nth wby",
            context=["hi", "yo what u on"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )
    assert first["candidate"]["has_unanswered_user_question"] is True
    assert first["candidate"]["unanswered_question_type"] == "reciprocal_activity_question"
    assert first["candidate"]["answer_required_now"] is True
    assert first["reply"] != "ok"

    callout = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="i js asked u a question",
            context=["hi", "yo what u on", "nth wby", "ok"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )
    assert callout["candidate"]["answer_required_now"] is True
    assert callout["candidate"]["user_called_out_unanswered_question"] is True
    assert callout["candidate"]["unanswered_question_type"] == "reciprocal_activity_question"
    assert callout["reply"] != "nah i get u"
    assert any(term in callout["reply"] for term in ("my bad", "missed", "chilling", "nothing", "not much"))

    strong = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="are u gonna answer",
            context=["hi", "yo what u on", "nth wby", "ok", "i js asked u a question", "nah i get u"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )
    assert strong["candidate"]["answer_required_now"] is True
    assert strong["candidate"]["ignored_question_debt_penalty"] < 0.7
    assert strong["reply"] != "say less what u doing"
    assert "what u doing" not in strong["reply"]


def test_catbot_route_question_debt_metadata_exposed(training_service: PhoneCopilotService) -> None:
    response = training_service.training_catbot_chat(
        CatbotChatRequest(
            incoming="are u gonna answer",
            context=["nth wby", "ok"],
            contact_name="Catbot",
            relationship_type="close_friend",
        )
    )
    candidate = response["candidate"]

    assert "has_unanswered_user_question" in candidate
    assert "unanswered_question_type" in candidate
    assert "answer_required_now" in candidate
    assert "ignored_question_debt_penalty" in candidate
    assert candidate["has_unanswered_user_question"] is True
