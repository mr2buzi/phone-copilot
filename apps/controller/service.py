from __future__ import annotations

import html
import hashlib
import asyncio
import json
import math
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import quote
from xml.etree import ElementTree as ET

from fastapi import HTTPException

from apps.controller.models import (
    AutomationMode,
    CatbotChatRequest,
    CatbotFeedbackRequest,
    ControllerState,
    LoopMetrics,
    MetricsSnapshot,
    NotificationPayload,
    ProviderCompareRequest,
    QueueItem,
    StyleReviewApproveRequest,
    StyleReviewRejectRequest,
    ThreadContext,
    TrainingChatRequest,
    TrainingFeedbackRequest,
    TrainingQuarantineRequest,
    WhatsAppWebDraftRequest,
    WhatsAppWebMemoryRequest,
)
from apps.controller.settings import ControllerSettings
from libs.adb import ADBClient, ADBClientProtocol, ADBCommandError, ADBUIStabilityTimeout, SafeTapContext
from libs.drafting import ApprovedPhoto, DraftBundle, DraftCandidate, DraftingService
from libs.drafting.automation_targets import load_automation_targets
from libs.drafting.contact_profiles import get_contact_profile
from libs.drafting.conversation_function import ConversationFunctionPrediction, predict_conversation_function
from libs.drafting.conversation_policy import should_reply
from libs.drafting.conversation_state import ConversationStateStore
from libs.drafting.intents import classify_intent
from libs.drafting.intents import contains_suspicious_phrase
from libs.drafting.intents import is_simple_greeting
from libs.drafting.intents import normalize_intent_label
from libs.drafting.intents import normalize_text
from libs.drafting.catbot_plan_validation import ReplyPlan, classify_reply_shape, detect_adult_detail_families, detect_availability_plan_detail, detect_detail_language, detect_loop_acknowledgement, detect_reason_answer, detect_reason_signatures, detect_reset_question, detect_sensory_texture, detect_specific_topic, validate_reply_against_plan
from libs.drafting.embeddings import embed_text
from libs.drafting.retrieval_backends.lexical import LexicalRetrievalBackend
from libs.drafting.retrieval_backends.vector_chroma import VectorChromaRetrievalBackend
from libs.drafting.model_providers import ModelMessage, generate_with_fallback, provider_config_status
from libs.drafting.training_data import load_all_training_rows, load_corrections, normalize_relationship_type, redact_private_text
from libs.drafting.training_data import append_correction
from libs.memory import LogStore
from libs.perception import PerceptionPipeline
from libs.planners import ExecutionPlan, ExecutionStep, PlannerDecision, StateMachinePlanner
from libs.policies import PolicyDecision, PolicyEngine
from libs.screen_states import ApprovalActionType, AvailableAction, DevicePoint, ScreenClassification, ScreenName, ScreenRegion
from libs.training import TrainingLoopService
from libs.validation_models import ComposeValidationResult, FailureCategory, SendAndReadResult


class OfflineADBClient:
    def __init__(self, reason: str) -> None:
        self.reason = reason

    def connect(self):
        raise ADBConnectionError(self.reason)

    def take_screenshot(self, save_path: Path | None = None) -> bytes:
        raise ADBConnectionError(self.reason)

    def get_foreground_app(self):
        raise ADBConnectionError(self.reason)

    def get_keyboard_state(self):
        raise ADBConnectionError(self.reason)

    def dump_ui_hierarchy(self) -> str:
        raise ADBConnectionError(self.reason)

    def wait_for_idle(self, timeout_ms: int = 2000, stable_cycles: int = 2) -> bool:
        raise ADBConnectionError(self.reason)

    def tap(self, x: int, y: int) -> None:
        raise ADBConnectionError(self.reason)

    def safe_tap(self, x: int, y: int, screen_context=None) -> tuple[int, int]:
        raise ADBConnectionError(self.reason)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        raise ADBConnectionError(self.reason)

    def type_text(self, text: str) -> None:
        raise ADBConnectionError(self.reason)

    def keyevent(self, code: int) -> None:
        raise ADBConnectionError(self.reason)

    def launch_app(self, package_name: str) -> None:
        raise ADBConnectionError(self.reason)

    def back(self) -> None:
        raise ADBConnectionError(self.reason)

    def home(self) -> None:
        raise ADBConnectionError(self.reason)

    def recent_apps(self) -> None:
        raise ADBConnectionError(self.reason)

    def push_file(self, local_path: Path, remote_path: str) -> None:
        raise ADBConnectionError(self.reason)

    def scan_media(self, remote_path: str) -> None:
        raise ADBConnectionError(self.reason)


@dataclass
class WhatsAppConversationObligation:
    latest_incoming_burst: list[str]
    conversation_move: list[str]
    required_response_acts: list[str]
    forbidden_response_acts: list[str]
    tone_target: str
    reply_shape: str
    already_sent_replies: list[str]
    incoming_fragmentation_score: float = 0.0
    user_style_burst_baseline: int = 1
    emotional_intensity: str = "low"
    repair_required: bool = False
    concrete_topics_detected: list[str] | None = None
    target_reply_burst_size: dict[str, int] | None = None
    reply_units: list[str] | None = None
    unique_response_acts: list[str] | None = None
    duplicate_check_result: dict[str, object] | None = None
    auto_send_allowed: bool = False
    send_queue_safety_result: dict[str, object] | None = None
    validation_result: dict[str, object] | None = None

    def to_debug_dict(self) -> dict[str, object]:
        return {
            "latest_incoming_burst": self.latest_incoming_burst,
            "incoming_burst_count": len(self.latest_incoming_burst),
            "incoming_fragmentation_score": self.incoming_fragmentation_score,
            "user_style_burst_baseline": self.user_style_burst_baseline,
            "emotional_intensity": self.emotional_intensity,
            "repair_required": self.repair_required,
            "concrete_topics_detected": self.concrete_topics_detected or [],
            "conversation_move": self.conversation_move,
            "required_response_acts": self.required_response_acts,
            "forbidden_response_acts": self.forbidden_response_acts,
            "tone_target": self.tone_target,
            "reply_shape": self.reply_shape,
            "target_reply_burst_size": self.target_reply_burst_size or {"min": 1, "max": 3},
            "reply_units": self.reply_units or [],
            "unique_response_acts": self.unique_response_acts or self.required_response_acts,
            "duplicate_check_result": self.duplicate_check_result or {},
            "auto_send_allowed": self.auto_send_allowed,
            "send_queue_safety_result": self.send_queue_safety_result or {},
            "validation_result": self.validation_result or {},
        }


@dataclass
class WhatsAppDraftTrace:
    request_id: str
    thread_id: str
    chat_name: str
    raw_collected_count: int
    latest_incoming_burst: list[str]
    latest_incoming_burst_timestamps: list[str]
    latest_incoming_burst_client_orders: list[int | None]
    latest_incoming_burst_message_ids: list[str]
    latest_burst_summary: str
    latest_burst_topics: list[str]
    context_messages_used: list[str]
    context_messages_excluded: list[dict[str, object]]
    old_context_topics_detected: list[str]
    unresolved_obligations: list[str]
    model_provider: str = ""
    model_error: str = ""
    rate_limit_detected: bool = False
    candidate_count: int = 0
    candidate_sources: list[str] | None = None
    candidate_previews: list[dict[str, object]] | None = None
    validation_results: list[dict[str, object]] | None = None
    rejection_reasons: list[str] | None = None
    repair_attempts: list[dict[str, object]] | None = None
    fallback_used: bool = False
    fallback_reason: str = ""
    cache_used: bool = False
    selected_candidate_source: str = "none"
    selected_reply_topics: list[str] | None = None
    repeat_similarity_thread: float = 0.0
    repeat_similarity_global: float = 0.0
    final_decision: str = "REVIEW_REQUIRED"
    final_reply_units: list[str] | None = None
    ui_rendered: bool = False
    ui_render_rejection_reason: str = "not_rendered_by_controller"

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "thread_id": self.thread_id,
            "chat_name": self.chat_name,
            "raw_collected_count": self.raw_collected_count,
            "latest_incoming_burst": self.latest_incoming_burst,
            "latest_incoming_burst_timestamps": self.latest_incoming_burst_timestamps,
            "latest_incoming_burst_client_orders": self.latest_incoming_burst_client_orders,
            "latest_incoming_burst_message_ids": self.latest_incoming_burst_message_ids,
            "latest_burst_summary": self.latest_burst_summary,
            "latest_burst_topics": self.latest_burst_topics,
            "context_messages_used": self.context_messages_used,
            "context_messages_excluded": self.context_messages_excluded,
            "old_context_topics_detected": self.old_context_topics_detected,
            "unresolved_obligations": self.unresolved_obligations,
            "model_provider": self.model_provider,
            "model_error": self.model_error,
            "rate_limit_detected": self.rate_limit_detected,
            "candidate_count": self.candidate_count,
            "candidate_sources": self.candidate_sources or [],
            "candidate_previews": self.candidate_previews or [],
            "validation_results": self.validation_results or [],
            "rejection_reasons": self.rejection_reasons or [],
            "repair_attempts": self.repair_attempts or [],
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "cache_used": self.cache_used,
            "selected_candidate_source": self.selected_candidate_source,
            "selected_reply_topics": self.selected_reply_topics or [],
            "repeat_similarity_thread": self.repeat_similarity_thread,
            "repeat_similarity_global": self.repeat_similarity_global,
            "final_decision": self.final_decision,
            "final_reply_units": self.final_reply_units or [],
            "ui_rendered": self.ui_rendered,
            "ui_render_rejection_reason": self.ui_render_rejection_reason,
        }


CATBOT_MOVE_TAXONOMY: dict[str, dict[str, object]] = {
    "greeting": {"required_bot_move": "warm_open", "slots": ["greeting_tone"]},
    "affectionate_greeting": {"required_bot_move": "warm_greeting_with_slight_pull", "slots": ["greeting_tone", "relationship_energy"]},
    "boredom_prompt": {"required_bot_move": "introduce_specific_playful_thread", "slots": ["conversation_need", "topic_seed"]},
    "carry_conversation_request": {"required_bot_move": "introduce_specific_playful_thread", "slots": ["conversation_need", "topic_seed"]},
    "curiosity_hook": {"required_bot_move": "invite_reveal_with_energy", "slots": ["reveal_signal"]},
    "reciprocal_question": {"required_bot_move": "answer_then_optionally_return", "slots": ["question_topic"]},
    "status_disclosure": {"required_bot_move": "acknowledge_status_and_continue", "slots": ["status_topic"]},
    "status_reason_question": {"required_bot_move": "answer_status_reason_then_continue", "slots": ["status_topic", "reason_target"]},
    "identity_fact_question": {"required_bot_move": "answer_identity_fact_then_continue", "slots": ["identity_fact", "entities"]},
    "education_status_question": {"required_bot_move": "answer_identity_fact_then_continue", "slots": ["identity_fact", "entities"]},
    "activity_question": {"required_bot_move": "answer_activity_then_continue", "slots": ["activity_scope"]},
    "activity_opinion_disclosure": {"required_bot_move": "acknowledge_activity_opinion_and_add_specific_comment", "slots": ["activity_topic", "opinion_polarity"]},
    "availability_planning": {"required_bot_move": "answer_or_defer_plan", "slots": ["timeframe", "plan_object"]},
    "romantic_escalation": {"required_bot_move": "continue_escalation_with_specificity", "slots": ["escalation_level", "detail_required"]},
    "romantic_boundary_test": {"required_bot_move": "keep_consensual_and_continue", "slots": ["boundary_signal"]},
    "emotional_reciprocity": {"required_bot_move": "reciprocate_affection_and_continue", "slots": ["affection_signal"]},
    "intensity_check": {"required_bot_move": "confirm_with_specificity", "slots": ["challenge_target"]},
    "repair_callout": {"required_bot_move": "acknowledge_and_reset_with_specific_move", "slots": ["repair_target"]},
    "loop_callout": {"required_bot_move": "acknowledge_and_reset_with_specific_move", "slots": ["repair_target", "callout_question", "repeated_reset_topic"]},
    "topic_dismissal_reset": {"required_bot_move": "acknowledge_drop_and_owner_side_reset", "slots": ["dismissal_target"]},
    "low_effort_ack": {"required_bot_move": "add_energy_or_specific_followup", "slots": ["ack_type"]},
    "challenge_or_disbelief": {"required_bot_move": "answer_challenge_with_reason", "slots": ["challenge_target"]},
    "compliment": {"required_bot_move": "receive_and_reciprocate", "slots": ["compliment_target"]},
    "insult_playful": {"required_bot_move": "banter_without_escalating", "slots": ["tease_target"]},
    "argument_start": {"required_bot_move": "deescalate_and_answer", "slots": ["conflict_target"]},
    "apology_prompt": {"required_bot_move": "own_it_and_repair", "slots": ["apology_reason"]},
    "story_hook": {"required_bot_move": "invite_story_detail_with_energy", "slots": ["story_signal"]},
    "story_detail_reveal": {"required_bot_move": "react_to_story_detail_with_specific_followup", "slots": ["story_signal", "story_entities"]},
}


@dataclass(frozen=True)
class CatbotTurnContract:
    incoming: str
    context: list[str]
    intent: str
    conversation_function: ConversationFunctionPrediction
    move: dict[str, object]
    reply_plan: ReplyPlan
    recent_bot_replies: list[str]


class PhoneCopilotService:
    def __init__(
        self,
        settings: ControllerSettings,
        adb_client: ADBClientProtocol | None = None,
    ) -> None:
        self.settings = settings
        self.settings.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.settings.debug_dir.mkdir(parents=True, exist_ok=True)
        self.settings.fixtures_live_dir.mkdir(parents=True, exist_ok=True)
        self.settings.log_db_path.parent.mkdir(parents=True, exist_ok=True)
        if adb_client is not None:
            self.adb = adb_client
        else:
            try:
                self.adb = ADBClient(
                    adb_path=settings.adb_path,
                    device_serial=settings.adb_device_serial,
                    command_retries=settings.adb_command_retries,
                    retry_backoff_seconds=settings.adb_retry_backoff_seconds,
                    default_timeout_seconds=settings.adb_default_timeout_seconds,
                )
            except ADBConnectionError as exc:
                self.adb = OfflineADBClient(f"ADB unavailable: {exc}")
        self.pipeline = PerceptionPipeline(
            adb_client=self.adb,
            selector_path=settings.selector_path,
            ocr_enabled=settings.enable_ocr,
            fixtures_live_dir=settings.fixtures_live_dir,
            debug_dir=settings.debug_dir,
            debug_enabled=settings.debug_mode,
        )
        self.drafting = DraftingService(
            template_path=settings.reply_template_path,
            approved_photos_dir=settings.approved_photos_dir,
            ai_reply_enabled=settings.ai_reply_enabled,
            ai_reply_backend=settings.ai_reply_backend,
            ai_reply_model=settings.ai_reply_model,
            ai_reply_fallback_models=[
                model.strip()
                for model in settings.ai_reply_fallback_models.split(",")
                if model.strip()
            ],
            ai_reply_base_url=settings.ai_reply_base_url,
            ai_reply_timeout_seconds=settings.ai_reply_timeout_seconds,
            ai_reply_system_prompt=settings.ai_reply_system_prompt,
            style_profile_path=settings.ai_reply_style_profile_path,
            training_dir=settings.ai_reply_training_dir,
            training_messages_dir=settings.ai_reply_training_messages_dir,
            contact_overrides_path=settings.contact_overrides_path,
            intelligence_db_path=settings.ai_reply_intelligence_db_path,
            training_owner_aliases=[
                alias.strip()
                for alias in settings.ai_reply_training_owner_aliases.split(",")
                if alias.strip()
            ],
            draft_provider=settings.draft_provider,
            fast_provider=settings.fast_provider,
            router_provider=settings.router_provider,
            private_provider=settings.private_provider,
            gemini_api_key=settings.gemini_api_key,
            groq_api_key=settings.groq_api_key,
            openrouter_api_key=settings.openrouter_api_key,
            huggingface_api_key=settings.huggingface_api_key,
            ollama_api_key=settings.ollama_api_key,
            gemini_model=settings.gemini_model,
            groq_model=settings.groq_model,
            openrouter_model=settings.openrouter_model,
            huggingface_model=settings.huggingface_model,
            ollama_model=settings.ollama_model,
            ollama_base_url=settings.ollama_base_url,
            external_api_enabled=settings.external_api_enabled,
            external_api_allow_sensitive=settings.external_api_allow_sensitive,
            external_api_max_context_messages=settings.external_api_max_context_messages,
            external_api_timeout_seconds=settings.external_api_timeout_seconds,
        )
        self.planner = StateMachinePlanner(photo_push_dir=settings.photo_push_dir)
        self.policy = PolicyEngine(
            confidence_threshold=settings.confidence_threshold,
            approved_photos_dir=settings.approved_photos_dir,
        )
        self.log_store = LogStore(settings.log_db_path)
        self.latest_state: ControllerState | None = None
        self._emergency_stop = False
        self._previous_step_failed = False
        self._metrics_history: list[LoopMetrics] = []
        self._halted_iterations = 0
        self._last_action: str | None = None
        self._last_action_result: str | None = None
        self._verification_errors: list[str] = []
        self._last_failure_category: FailureCategory | None = None
        self._last_compose_validation_result: ComposeValidationResult | None = None
        self._compose_validation_runs: list[ComposeValidationResult] = []
        self._last_send_and_read_result: SendAndReadResult | None = None
        self._last_debug_timings_ms: dict[str, float] = {}

        # Automation state
        self._blacklist: set[str] = self._load_blacklist()
        self._unread_queue: list[QueueItem] = []
        self._automation_enabled = settings.automation_enabled
        self._current_automation_mode = settings.default_automation_mode if settings.default_automation_mode in {"review", "auto-draft", "auto-send"} else "review"
        self._automation_confidence = settings.automation_confidence_threshold
        self._auto_send_enabled = settings.auto_send_on_high_confidence
        self._auto_send_confidence = settings.auto_send_confidence_threshold
        self._conversation_state = ConversationStateStore(settings.data_dir / "conversation_state.json")
        self._catbot_recent_replies: dict[str, list[str]] = {}
        self._whatsapp_recent_thread_drafts: dict[str, list[dict[str, object]]] = {}
        self._whatsapp_recent_global_drafts: list[dict[str, object]] = []
        self._catbot_style_rows_cache: list[dict[str, Any]] | None = None
        self._catbot_style_summary_cache: dict[str, Any] | None = None
        self._catbot_style_contract_cache: str | None = None
        self.training_sessions_path = settings.data_dir / "training_sessions.jsonl"
        self.training_quarantine_path = settings.data_dir / "training_sessions_quarantine.jsonl"
        self.style_review_queue_path = settings.ai_reply_training_messages_dir / "auto_style_corrections_review.jsonl"
        self.vector_rebuild_marker_path = settings.data_dir / "vector_db" / "reply_examples_chroma" / ".rebuild_required"
        self.vector_rebuild_meta_path = settings.data_dir / "vector_db" / "reply_examples_chroma" / "last_rebuild.json"
        self.whatsapp_web_memory_db_path = settings.ai_reply_intelligence_db_path
        self.training_loop = TrainingLoopService(
            data_dir=settings.data_dir,
            repo_root=Path.cwd(),
            rebuild_callback=self.training_rebuild_vector_db,
        )

    def _ensure_whatsapp_web_memory_db(self) -> None:
        self.whatsapp_web_memory_db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.whatsapp_web_memory_db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS whatsapp_web_memory (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    contact_name TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    source TEXT NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_whatsapp_web_memory_contact ON whatsapp_web_memory(contact_name, created_at)")

    def _whatsapp_web_memory_rows(self, contact_name: str | None, *, limit: int = 8) -> list[dict[str, object]]:
        self._ensure_whatsapp_web_memory_db()
        normalized_contact = str(contact_name or "").strip().casefold()
        if not normalized_contact:
            return []
        with sqlite3.connect(self.whatsapp_web_memory_db_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT id, created_at, contact_name, relationship_type, question, answer, context_json, source
                FROM whatsapp_web_memory
                WHERE lower(contact_name) = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (normalized_contact, max(1, min(int(limit or 8), 20))),
            ).fetchall()
        return [dict(row) for row in rows]

    def _whatsapp_web_memory_context(self, contact_name: str | None) -> list[str]:
        context: list[str] = []
        for row in reversed(self._whatsapp_web_memory_rows(contact_name, limit=6)):
            question = str(row.get("question") or "").strip()
            answer = str(row.get("answer") or "").strip()
            if question and answer:
                context.append(f"Saved WhatsApp note - {question}: {answer}")
        return context

    def _whatsapp_web_context_questions(
        self,
        *,
        contact_name: str | None,
        relationship_type: str,
        collected_count: int,
        memory_count: int,
    ) -> list[str]:
        questions: list[str] = []
        if collected_count < 40:
            questions.append(f"I collected {collected_count} context messages. Anything important from earlier in this chat?")
        if not str(contact_name or "").strip():
            questions.append("Who is this WhatsApp contact?")
        if normalize_relationship_type(relationship_type) == "unknown":
            questions.append("What is your relationship with this contact?")
        if memory_count == 0:
            questions.append("Any key facts, plans, or tone preferences I should remember for this chat?")
        return questions[:3]

    def _whatsapp_state_key(self, *, contact_name: str | None, thread_id: str | None) -> str:
        raw_thread = normalize_text(str(thread_id or ""))
        if raw_thread:
            return f"whatsapp:{raw_thread}"
        raw_contact = normalize_text(str(contact_name or "unknown"))
        digest = hashlib.sha1(raw_contact.casefold().encode("utf-8")).hexdigest()[:12]
        return f"whatsapp:contact:{digest}"

    def _whatsapp_has_family_health_context(self, normalized: str) -> bool:
        medical_terms = (
            "infusion",
            "steroids",
            "cortisol",
            "coma",
            "medical condition",
            "addison",
            "chemo",
            "auto-immune",
            "auto immune",
            "illness",
            "disease",
            "hospital",
            "doctor",
            "losing her",
            "red medical badge",
        )
        if any(term in normalized for term in medical_terms):
            return True
        family_terms = ("mum", "mom", "mother", "sis", "sister", "siblings")
        health_terms = ("health", "medical", "unwell", "poorly", "emergency", "ill", "sick")
        return any(term in normalized for term in family_terms) and any(term in normalized for term in health_terms)

    def _whatsapp_is_direct_plan_text(self, normalized: str) -> bool:
        if any(term in normalized for term in ("unless i go out", "unless i go", "haven't updated", "havent updated")):
            return False
        direct_terms = (
            "wanna go out",
            "want to go out",
            "do you wanna go",
            "dya wanna go",
            "can we link",
            "can we meet",
            "link in public",
            "free on sunday",
            "if ur free",
            "if you're free",
            "if youre free",
            "what about friday",
            "sunday works",
            "thursday",
            "thirsday",
            "friday",
            "saturday",
            "tutor on saturdays",
        )
        return any(term in normalized for term in direct_terms)

    def _whatsapp_has_work_cover_update(self, normalized: str) -> bool:
        if "cover" in normalized and "work" in normalized:
            return True
        if any(term in normalized for term in ("meet up", "meet ups", "meetup", "meetups")) and any(term in normalized for term in ("updated", "next week", "tomorrow", "work")):
            return True
        return "next week" in normalized and "go out" in normalized

    def _whatsapp_has_term(self, normalized: str, term: str) -> bool:
        escaped = re.escape(str(term or "").strip())
        if not escaped:
            return False
        return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", normalized) is not None

    def _whatsapp_has_any_term(self, normalized: str, terms: tuple[str, ...]) -> bool:
        return any(self._whatsapp_has_term(normalized, term) for term in terms)

    def _whatsapp_has_missing_luck_context(self, normalized: str) -> bool:
        return self._whatsapp_has_any_term(normalized, (
            "wish me luck",
            "wished me luck",
            "didnt wish me luck",
            "didn't wish me luck",
            "didnt wish",
            "didn't wish",
            "good luck before",
        ))

    def _whatsapp_has_sexual_flirt_context(self, normalized: str) -> bool:
        return self._whatsapp_has_any_term(normalized, (
            "ice cream",
            "lick it off",
            "cock",
            "give me head",
            "give u head",
            "giving head",
            "raw",
            "condom",
            "horny",
        ))

    def _whatsapp_has_relationship_conflict_context(self, normalized: str) -> bool:
        return self._whatsapp_has_any_term(normalized, (
            "ruin this rs",
            "ruin this relationship",
            "relationship is a 2 way",
            "not been there",
            "haven't been there",
            "havent been there",
            "break my heart",
            "emotionally drained",
            "crying my eyes out",
            "cry man",
            "treated like shit",
            "hate u",
            "hate you",
            "hurting me",
            "hurt me",
            "doing this to me",
            "what is wrong with you",
            "what r is wrong with you",
            "closed book",
            "space",
            "selfish",
            "good luck with life",
            "all to myself",
            "how it used to be",
            "back to square one",
            "still on chat",
            "not anymore",
            "going back to how",
            "i barely have a say",
            "on ur terms",
            "on your terms",
            "dont deserve",
            "don't deserve",
            "fuck you",
            "gfys",
            "im done",
            "i'm done",
            "i dont want any part of you",
            "i don't want any part of you",
            "move tf on",
            "take u back",
        ))

    def _whatsapp_has_reachability_pressure(self, normalized: str) -> bool:
        return any(term in normalized for term in (
            "text me back",
            "message me back",
            "reply to me",
            "please reply",
            "need u to text",
            "need you to text",
            "today would be nice",
            "gonna ring",
            "going to ring",
            "busy again",
        ))

    def _whatsapp_has_status_check(self, normalized: str) -> bool:
        return any(term in normalized for term in (
            "are you okay",
            "are u okay",
            "are you ok",
            "are u ok",
            "how are you",
            "how are u",
            "hru",
            "are you busy",
            "are u busy",
            "r u busy",
            "busy rn",
            "busy right now",
            "busy again",
            "still in northbridge",
            "appointment done",
            "gotten your appointment done",
            "got your appointment done",
        ))

    def _whatsapp_reply_topics(self, text: str) -> list[str]:
        normalized = normalize_text(text)
        topics: list[str] = []
        topic_terms = {
            "academic_stress": ("exam", "exams", "revision", "revised", "presentation", "marker", "flashcard", "passed", "failed", "questions"),
            "food": ("food", "eat", "eaten", "takeout", "lunch"),
            "rain": ("rain", "rained"),
            "banter": ("pissed", "dragging", "rude", "icl", "lmao"),
            "affection": ("love u", "miss u", "baby", "come here", "kiss", "mwah"),
            "logistics": ("time", "when", "where", "home", "free", "come", "link", "tomorrow", "next week", "work", "cover"),
            "on_read_repair": ("on read", "aired", "air u", "airing"),
            "reachability_pressure": ("text me back", "message me back", "today would be nice", "gonna ring", "busy again"),
            "status_check": ("are you okay", "are u okay", "are you busy", "are u busy", "busy rn", "busy right now", "busy again", "still in northbridge", "appointment done", "gotten your appointment done"),
        }
        for label, terms in topic_terms.items():
            if self._whatsapp_has_any_term(normalized, terms):
                topics.append(label)
        if self._whatsapp_has_family_health_context(normalized):
            topics.append("family_health")
        if self._whatsapp_has_work_cover_update(normalized):
            topics.append("work_cover")
        if self._whatsapp_has_reachability_pressure(normalized):
            topics.append("reachability_pressure")
        if self._whatsapp_has_status_check(normalized) or re.search(r"\b(?:u|you) good\b", normalized):
            topics.append("status_check")
        if self._whatsapp_has_relationship_conflict_context(normalized):
            topics.append("relationship_conflict")
        return list(dict.fromkeys(topics))

    def _whatsapp_draft_signature(self, text: str) -> dict[str, object]:
        normalized = normalize_text(text).strip(" ?!.,")
        units = self._whatsapp_reply_units(text)
        unit_norms = [normalize_text(unit).strip(" ?!.,") for unit in units if normalize_text(unit).strip(" ?!.,")]
        return {
            "text": text,
            "normalized": normalized,
            "text_hash": hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16] if normalized else "",
            "bubble_hashes": [hashlib.sha1(unit.encode("utf-8")).hexdigest()[:12] for unit in unit_norms],
            "topics": self._whatsapp_reply_topics(text),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def _whatsapp_reply_similarity(self, left: str, right: str) -> float:
        left_norm = normalize_text(left).strip(" ?!.,")
        right_norm = normalize_text(right).strip(" ?!.,")
        if not left_norm or not right_norm:
            return 0.0
        joined_ratio = SequenceMatcher(None, left_norm, right_norm).ratio()
        left_units = self._whatsapp_reply_units(left)
        right_units = self._whatsapp_reply_units(right)
        unit_ratio = 0.0
        for left_unit in left_units:
            left_unit_norm = normalize_text(left_unit).strip(" ?!.,")
            if not left_unit_norm:
                continue
            for right_unit in right_units:
                right_unit_norm = normalize_text(right_unit).strip(" ?!.,")
                if right_unit_norm:
                    unit_ratio = max(unit_ratio, SequenceMatcher(None, left_unit_norm, right_unit_norm).ratio())
        if len(left_units) > 3 and len(right_units) > 3:
            return round(joined_ratio, 3)
        return round(max(joined_ratio, unit_ratio), 3)

    def _whatsapp_similarity_to_recent(self, text: str, recent: list[dict[str, object]]) -> float:
        similarities = [
            self._whatsapp_reply_similarity(text, str(item.get("text") or ""))
            for item in recent
            if str(item.get("text") or "").strip()
        ]
        return max(similarities or [0.0])

    def _whatsapp_apply_recent_draft_validation(
        self,
        validation: dict[str, object],
        text: str,
        *,
        latest_burst_topics: list[str],
        recent_thread_drafts: list[dict[str, object]],
        recent_global_drafts: list[dict[str, object]],
    ) -> dict[str, object]:
        updated = dict(validation)
        violations = list(updated.get("violated_forbidden_acts") or [])
        thread_similarity = self._whatsapp_similarity_to_recent(text, recent_thread_drafts)
        global_similarity = self._whatsapp_similarity_to_recent(text, recent_global_drafts)
        if thread_similarity >= 0.9:
            violations.append("similar_to_recent_thread_draft")
        latest_topics = set(latest_burst_topics)
        for draft in recent_global_drafts:
            draft_text = str(draft.get("text") or "")
            if not draft_text.strip():
                continue
            if self._whatsapp_reply_similarity(text, draft_text) < 0.98:
                continue
            draft_topics = set(draft.get("topics") or [])
            if latest_topics and draft_topics and latest_topics.intersection(draft_topics):
                continue
            if not latest_topics:
                continue
            violations.append("similar_to_recent_global_draft")
            break
        updated["violated_forbidden_acts"] = list(dict.fromkeys(violations))
        updated["forbidden_acts_triggered"] = updated["violated_forbidden_acts"]
        updated["topic_mismatch"] = [
            item for item in updated["violated_forbidden_acts"]
            if item in {"academic_reply_for_nonacademic_burst", "wrong_crisis_domain", "generic_ungrounded_academic_reassurance", "topic_change_during_emotional_disclosure"}
        ]
        if any(item in updated["violated_forbidden_acts"] for item in ("academic_reply_for_nonacademic_burst", "stale_emotional_support_for_low_intensity_burst", "similar_to_recent_global_draft")):
            updated["stale_candidate_reason"] = "candidate appears stale or grounded in old/unrelated context"
        if any(item in updated["violated_forbidden_acts"] for item in ("duplicate_or_near_duplicate_bubbles", "repeat_already_sent_bubble", "similar_to_recent_thread_draft")):
            updated["repetition_reason"] = "candidate repeats an already shown/sent draft or repeats bubbles internally"
        updated["selected_candidate_similarity_to_recent"] = {
            "thread": thread_similarity,
            "global": global_similarity,
        }
        updated["passed"] = bool(updated.get("passed")) and not violations
        updated["validation_result"] = "pass" if updated["passed"] else "fail"
        return updated

    def _whatsapp_repeatable_repair_unit(self, normalized_unit: str) -> bool:
        if not normalized_unit:
            return False
        repeatable = {
            "baby im sorry",
            "im sorry baby",
            "my love im sorry",
            "my baby im sorry",
            "thats on me",
            "and thats on me",
            "i love u",
            "i do love u",
            "i love u properly",
            "i hear u",
            "im here",
            "im listening",
            "talk to me",
            "talk to me properly",
            "come talk to me",
        }
        if normalized_unit in repeatable:
            return True
        words = normalized_unit.split()
        return len(words) <= 4 and any(term in normalized_unit for term in ("sorry", "love u", "hear u", "im here", "thats on me"))

    def _whatsapp_record_recent_draft(self, state_key: str, request_id: str, text: str) -> None:
        if not text.strip():
            return
        signature = self._whatsapp_draft_signature(text)
        signature["request_id"] = request_id
        self._whatsapp_recent_thread_drafts.setdefault(state_key, []).append(signature)
        self._whatsapp_recent_thread_drafts[state_key] = self._whatsapp_recent_thread_drafts[state_key][-12:]
        self._whatsapp_recent_global_drafts.append({**signature, "state_key": state_key})
        self._whatsapp_recent_global_drafts = self._whatsapp_recent_global_drafts[-40:]

    def _infer_whatsapp_web_relationship(self, messages: list[dict[str, object]], requested_relationship: str) -> tuple[str, bool]:
        requested = normalize_relationship_type(requested_relationship)
        if requested != "unknown":
            return requested, False
        blob = normalize_text(" ".join(str(message.get("text") or "") for message in messages[-140:]))
        romantic_terms = (
            "baby",
            "babe",
            "bebe",
            "my love",
            "my loce",
            "imy",
            "i miss you",
            "missed you",
            "miss u",
            "love u",
            "love you",
            "mwah",
            "my handsome",
            "handsome",
            "condoms",
            "craving you",
            "craving u",
            "kiss",
            "head",
            "cock",
            "raw",
        )
        professional_terms = ("client", "invoice", "meeting", "manager", "work email", "deadline", "presentation")
        family_terms = ("mum", "mom", "dad", "sister", "brother", "aunt", "uncle")
        university_terms = ("lecture", "seminar", "coursework", "assignment", "uni", "exam", "revision")
        close_friend_terms = ("bro", "mate", "bruv", "my guy", "fam")
        if any(term in blob for term in romantic_terms):
            return "romantic_interest", True
        if any(term in blob for term in professional_terms):
            return "professional", True
        if any(term in blob for term in family_terms):
            return "family", True
        if any(term in blob for term in university_terms):
            return "university", True
        if any(term in blob for term in close_friend_terms):
            return "close_friend", True
        return "unknown", False

    def _parse_whatsapp_web_timestamp(self, value: object) -> datetime | None:
        raw = str(value or "").strip()
        if not raw:
            return None
        cleaned = raw.replace("\u202f", " ").replace("\xa0", " ")
        for candidate in (cleaned, cleaned.replace("Z", "+00:00")):
            try:
                parsed = datetime.fromisoformat(candidate)
                return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        patterns = (
            r"(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})\s+(?P<hour>\d{1,2}):(?P<minute>\d{2})",
            r"(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*(?P<ampm>a\.m\.|p\.m\.|am|pm|AM|PM),?\s*(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})",
            r"(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*(?P<ampm>AM|PM|am|pm),?\s*(?P<month>\d{1,2})/(?P<day>\d{1,2})/(?P<year>\d{4})",
        )
        for pattern in patterns:
            match = re.search(pattern, cleaned)
            if not match:
                continue
            parts = match.groupdict()
            hour = int(parts["hour"])
            ampm = str(parts.get("ampm") or "").casefold()
            if ampm.startswith("p") and hour < 12:
                hour += 12
            if ampm.startswith("a") and hour == 12:
                hour = 0
            return datetime(
                int(parts["year"]),
                int(parts["month"]),
                int(parts["day"]),
                hour,
                int(parts["minute"]),
                tzinfo=timezone.utc,
            )
        return None

    def _whatsapp_order_messages(self, messages: list[dict[str, object]]) -> list[dict[str, object]]:
        annotated = [
            (index, message, self._parse_whatsapp_web_timestamp(message.get("timestamp")), message.get("client_order"))
            for index, message in enumerate(messages)
        ]
        if len(annotated) < 2:
            return messages
        client_orders: list[int] = []
        for _, _, _, client_order in annotated:
            try:
                client_orders.append(int(client_order))
            except (TypeError, ValueError):
                pass
        has_complete_client_order = len(client_orders) == len(annotated) and len(set(client_orders)) == len(annotated)
        if any(parsed is None for _, _, parsed, _ in annotated):
            if has_complete_client_order:
                return [
                    message
                    for _, message, _, _ in sorted(annotated, key=lambda item: (int(item[3]), item[0]))
                ]
            return messages
        return [
            message
            for _, message, _, _ in sorted(
                annotated,
                key=lambda item: (item[2], int(item[3]) if has_complete_client_order else item[0], item[0]),
            )
        ]

    def _whatsapp_web_time_context(self, messages: list[dict[str, object]], incoming_index: int) -> list[str]:
        notes: list[str] = []
        latest = messages[incoming_index] if 0 <= incoming_index < len(messages) else {}
        latest_time = self._parse_whatsapp_web_timestamp(latest.get("timestamp"))
        latest_text = normalize_text(str(latest.get("text") or ""))
        now = datetime.now(timezone.utc)
        if latest_time:
            age = now - latest_time
            if age >= timedelta(hours=8):
                if age.days >= 1:
                    notes.append(f"WhatsApp timing note - latest message is {age.days} day(s) old; acknowledge the late reply before answering.")
                else:
                    notes.append("WhatsApp timing note - latest message is hours old; a quick late-reply acknowledgement may fit.")
            if latest_time.hour < 5 and now.hour >= 7:
                notes.append("WhatsApp timing note - they messaged very late at night; if replying in the morning, a good morning or late-reply acknowledgement can fit.")
        recent = messages[max(0, incoming_index - 6) : incoming_index + 1]
        if any("goodnight" in normalize_text(str(message.get("text") or "")) or "sleepwell" in normalize_text(str(message.get("text") or "")) for message in recent):
            notes.append("WhatsApp timing note - recent thread includes goodnight/sleepwell; if replying after morning, good morning or sorry-for-late-reply may be natural.")
        if latest_text and self._whatsapp_is_direct_plan_text(latest_text):
            notes.append("WhatsApp topic note - latest message is an invitation/plan; show interest and ask concrete timing/logistics.")
        return notes[:3]

    def _whatsapp_context_exclusions(
        self,
        messages: list[dict[str, object]],
        burst_start_index: int,
        incoming_index: int,
    ) -> list[dict[str, object]]:
        excluded: list[dict[str, object]] = []
        for index, message in enumerate(messages):
            if burst_start_index <= index <= incoming_index:
                excluded.append({
                    "index": index,
                    "reason": "latest_burst_primary_source",
                    "speaker": str(message.get("speaker") or ""),
                    "text": str(message.get("text") or ""),
                    "timestamp": str(message.get("timestamp") or ""),
                    "client_order": message.get("client_order"),
                    "message_id": str(message.get("message_id") or ""),
                })
            elif index > incoming_index:
                excluded.append({
                    "index": index,
                    "reason": "after_latest_incoming",
                    "speaker": str(message.get("speaker") or ""),
                    "text": str(message.get("text") or ""),
                    "timestamp": str(message.get("timestamp") or ""),
                    "client_order": message.get("client_order"),
                    "message_id": str(message.get("message_id") or ""),
                })
        return excluded[-80:]

    def _whatsapp_old_context_topics(self, message_context: list[str], latest_topics: list[str]) -> list[str]:
        topics = self._whatsapp_concrete_topics(message_context[-120:])
        return [topic for topic in topics if topic not in set(latest_topics)]

    def _whatsapp_unresolved_obligations(self, message_context: list[str], latest_topics: list[str]) -> list[str]:
        context_blob = normalize_text(" ".join(message_context[-80:]))
        latest_topic_set = set(latest_topics)
        unresolved: list[str] = []
        if "family_health" in self._whatsapp_concrete_topics(message_context[-80:]) and "family_health" in latest_topic_set:
            unresolved.append("continue_family_health_support")
        if any(term in context_blob for term in ("r u even reading", "read it properly", "you didnt wish me luck", "you didn't wish me luck", "unsupported")) and any(topic in latest_topic_set for topic in ("academic_stress", "presentation", "luck", "reading_properly", "unsupported")):
            unresolved.append("repair_previous_support_failure")
        return unresolved

    def _whatsapp_validation_repair_instruction(
        self,
        validation: dict[str, object],
        obligation: WhatsAppConversationObligation,
    ) -> str:
        missing = ", ".join(str(item) for item in validation.get("missing_required_acts", []) or [])
        violations = ", ".join(str(item) for item in validation.get("violated_forbidden_acts", []) or [])
        topics = ", ".join(obligation.concrete_topics_detected or [])
        acts = ", ".join(obligation.required_response_acts)
        instruction = (
            "WhatsApp repair instruction: rewrite using only the latest incoming burst as primary truth. "
            f"Latest topics: {topics or 'none'}. Required response acts: {acts or 'answer_relevantly'}. "
            f"Missing acts: {missing or 'none'}. Violations: {violations or 'none'}. "
            "Do not reuse a previous draft, stale template, or unrelated old-context reply."
        )
        return instruction

    def _whatsapp_candidate_source_label(self, candidate: dict[str, object]) -> str:
        explicit = str(candidate.get("candidate_source_label") or "").strip()
        if explicit:
            return explicit
        kind = str(candidate.get("candidate_kind") or candidate.get("candidate_type") or "").strip()
        provider = str(candidate.get("provider") or "").strip()
        if bool(candidate.get("cache_used")):
            return "cached_draft"
        if kind in {"stale_template", "test_fixture"}:
            return "test_fixture"
        if kind == "repaired_model":
            return "repaired_model"
        if kind in {"contextual_fallback", "obligation_repair"} or provider == "conversation_obligation":
            return "contextual_fallback"
        if bool(candidate.get("external_api_used")) or provider not in {"", "deterministic", "fallback"}:
            return "fresh_model"
        return "template_fallback"

    def _whatsapp_candidate_is_disabled_fallback(self, candidate: dict[str, object]) -> bool:
        kind = str(candidate.get("candidate_kind") or candidate.get("candidate_type") or "").strip()
        provider = str(candidate.get("provider") or "").strip()
        source = self._whatsapp_candidate_source_label(candidate)
        return (
            source in {"contextual_fallback", "obligation_last_resort"}
            or kind in {"contextual_fallback", "obligation_repair", "whatsapp_obligation_repair", "obligation_last_resort"}
            or provider == "conversation_obligation"
            or bool(candidate.get("manual_review_fallback"))
        )

    def _whatsapp_disable_fallback_candidate_validation(
        self,
        candidate: dict[str, object],
        validation: dict[str, object],
    ) -> dict[str, object]:
        if not self._whatsapp_candidate_is_disabled_fallback(candidate):
            return validation
        updated = dict(validation)
        violations = list(updated.get("violated_forbidden_acts") or [])
        violations.append("fallback_candidate_disabled_for_whatsapp_web")
        violations = list(dict.fromkeys(violations))
        updated["passed"] = False
        updated["validation_result"] = "fail"
        updated["violated_forbidden_acts"] = violations
        updated["forbidden_acts_triggered"] = violations
        candidate["auto_send_allowed"] = False
        return updated

    def _whatsapp_relax_fresh_model_burst_size_validation(
        self,
        candidate: dict[str, object],
        validation: dict[str, object],
        obligation: WhatsAppConversationObligation,
    ) -> dict[str, object]:
        if self._whatsapp_candidate_source_label(candidate) != "fresh_model":
            return validation
        violations = [
            str(item)
            for item in validation.get("violated_forbidden_acts", [])
            if str(item).strip()
        ]
        if violations != ["below_target_reply_burst_size"] or validation.get("missing_required_acts"):
            return validation
        if not self._whatsapp_fresh_model_low_intensity_burst_safe(obligation):
            return validation
        updated = dict(validation)
        updated["violated_forbidden_acts"] = []
        updated["forbidden_acts_triggered"] = []
        updated["passed"] = True
        updated["validation_result"] = "pass"
        updated["burst_size_relaxed_for_fresh_model"] = True
        return updated

    def _whatsapp_fresh_model_low_intensity_burst_safe(
        self,
        obligation: WhatsAppConversationObligation,
    ) -> bool:
        if obligation.repair_required or obligation.emotional_intensity != "low":
            return False
        burst_count = len(obligation.latest_incoming_burst)
        if burst_count <= 0 or burst_count > 3:
            return False
        moves = set(obligation.conversation_move or [])
        unsafe_moves = {
            "conflict_or_hurt",
            "emotional_disclosure",
            "on_read_complaint",
            "reachability_pressure",
            "sexual_flirt",
        }
        if moves.intersection(unsafe_moves):
            return False
        if burst_count == 1:
            return True
        safe_direct_moves = {
            "apology_or_repair",
            "check_in",
            "greeting",
            "invite_or_plan",
            "practical_question",
            "question",
        }
        return bool(moves.intersection(safe_direct_moves))

    def _whatsapp_fresh_model_low_intensity_send_allowed(
        self,
        candidate: dict[str, object],
        validation: dict[str, object],
        obligation: WhatsAppConversationObligation,
    ) -> bool:
        if self._whatsapp_candidate_source_label(candidate) != "fresh_model":
            return False
        provider = str(candidate.get("provider") or "").strip()
        if provider in {"", "deterministic", "fallback", "local_obligation_validator"}:
            return False
        if not bool(validation.get("passed", False)):
            return False
        if not self._whatsapp_fresh_model_low_intensity_burst_safe(obligation):
            return False
        if validation.get("missing_required_acts") or validation.get("violated_forbidden_acts"):
            return False
        sequence = candidate.get("sequence")
        unit_count = len(sequence) if isinstance(sequence, list) else len(self._whatsapp_reply_units(str(candidate.get("text") or "")))
        return 1 <= unit_count <= 5

    def _whatsapp_candidate_preview(self, candidate: dict[str, object]) -> dict[str, object]:
        text = str(candidate.get("text") or "")
        validation = candidate.get("whatsapp_obligation_validation") if isinstance(candidate.get("whatsapp_obligation_validation"), dict) else {}
        return {
            "source": self._whatsapp_candidate_source_label(candidate),
            "candidate_kind": str(candidate.get("candidate_kind") or candidate.get("candidate_type") or candidate.get("provider") or "unknown"),
            "preview": text[:160],
            "reply_units": candidate.get("sequence") if isinstance(candidate.get("sequence"), list) else self._whatsapp_reply_units(text),
            "provider": str(candidate.get("provider") or ""),
            "model": str(candidate.get("model") or ""),
            "validation_result": validation.get("validation_result", ""),
            "reasons": list(validation.get("violated_forbidden_acts") or []) + list(validation.get("missing_required_acts") or []),
        }

    def _whatsapp_provider_debug(self, response: dict[str, object], candidates: list[dict[str, object]]) -> dict[str, object]:
        providers = [
            str(candidate.get("provider") or candidate.get("provider_name") or "")
            for candidate in candidates
            if str(candidate.get("provider") or candidate.get("provider_name") or "").strip()
        ]
        errors = [
            str(candidate.get("provider_error") or "")
            for candidate in candidates
            if str(candidate.get("provider_error") or "").strip()
        ]
        provider = providers[0] if providers else str(response.get("provider") or "")
        error = errors[0] if errors else str(response.get("provider_error") or response.get("model_error") or "")
        return {
            "model_provider": provider or "deterministic",
            "model_error": error,
            "rate_limit_detected": bool(re.search(r"rate.?limit|429|quota", error, re.I)),
        }

    def _load_blacklist(self) -> set[str]:
        """Load blacklist from file."""
        blacklist_path = self.settings.blacklist_file
        if blacklist_path.exists():
            try:
                with open(blacklist_path) as f:
                    data = json.load(f)
                    return set(item.lower() for item in data.get("contacts", []))
            except (json.JSONDecodeError, IOError):
                pass
        return set()

    def _save_blacklist(self) -> None:
        """Save blacklist to file."""
        self.settings.blacklist_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.settings.blacklist_file, "w") as f:
            json.dump({"contacts": sorted(list(self._blacklist))}, f, indent=2)

    def add_blacklist_contact(self, contact_name: str) -> dict[str, str]:
        """Add contact to blacklist."""
        normalized = self._normalize_contact_name(contact_name)
        if not normalized:
            raise HTTPException(status_code=400, detail="Contact name must not be empty.")
        self._blacklist.add(normalized)
        self._save_blacklist()
        return {"status": "added", "contact": normalized}

    def remove_blacklist_contact(self, contact_name: str) -> dict[str, str]:
        """Remove contact from blacklist."""
        normalized = self._normalize_contact_name(contact_name)
        if not normalized:
            raise HTTPException(status_code=400, detail="Contact name must not be empty.")
        self._blacklist.discard(normalized)
        self._save_blacklist()
        return {"status": "removed", "contact": normalized}

    def set_automation_mode(self, mode: str, confidence: float | None = None) -> dict[str, object]:
        """Set automation mode: 'review', 'auto-draft', or 'auto-send'."""
        if mode not in ["review", "auto-draft", "auto-send"]:
            raise HTTPException(status_code=400, detail="Invalid mode. Must be 'review', 'auto-draft', or 'auto-send'.")
        self._current_automation_mode = mode
        self._automation_enabled = mode != "review"
        self._auto_send_enabled = mode == "auto-send"
        if confidence is not None:
            self._automation_confidence = max(0.0, min(1.0, confidence))
        return {"status": "updated", **self.get_automation_state()}

    def scan_inbox(self) -> ControllerState:
        """Navigate to inbox and detect unread threads."""
        state = self._ensure_messages_inbox_state()
        self._build_unread_queue(state)
        self._last_action = "Scan Inbox"
        self._last_action_result = f"Found {len(self._unread_queue)} unread threads"
        state.last_action = self._last_action
        state.last_action_result = self._last_action_result
        state.unread_queue = [item.model_copy() for item in self._unread_queue]
        self.latest_state = state
        return state

    def _build_unread_queue(self, state: ControllerState) -> None:
        """Extract unread thread info from screen classification."""
        self._unread_queue.clear()

        if state.classification.screen.value != "app_inbox":
            return

        hierarchy = self._read_messages_hierarchy()
        items = self._extract_inbox_threads_from_hierarchy(hierarchy) if hierarchy else []
        if not items:
            items = self._extract_inbox_threads_from_visible_text(state.classification.visible_text)
        else:
            unread_first = [item for item in items if item.unread]
            read_fallback = [item for item in items if not item.unread]
            items = unread_first or read_fallback
        deduped = self._dedupe_queue_items(items)
        self._unread_queue = deduped[: self.settings.queue_max_size]
        state.unread_queue = [item.model_copy() for item in self._unread_queue]

    def open_thread(self, contact_name: str) -> ControllerState:
        """Open a specific thread by contact name."""
        target = self._normalize_contact_name(contact_name)
        if not target:
            raise HTTPException(status_code=400, detail="contact_name is required")

        state = self._ensure_messages_inbox_state()
        hierarchy = self._read_messages_hierarchy()
        try:
            thread_candidates = self._extract_inbox_threads_from_hierarchy(hierarchy) if hierarchy else []
            chosen = next(
                (item for item in thread_candidates if self._contact_matches(item.contact_name, target)),
                None,
            )
            if chosen is None:
                available = ", ".join(item.contact_name for item in self._unread_queue[:5]) or "none visible"
                raise HTTPException(status_code=404, detail=f"Thread '{contact_name}' was not found in the visible inbox. Visible: {available}")
            assert chosen.contact_number is not None
            left, top, right, bottom = (int(part) for part in chosen.contact_number.split(","))
            self.adb.safe_tap(
                (left + right) // 2,
                (top + bottom) // 2,
                screen_context=SafeTapContext(
                    screen=state.classification.screen.value,
                    expected_region_left=left,
                    expected_region_top=top,
                    expected_region_right=right,
                    expected_region_bottom=bottom,
                ),
            )
            self.adb.wait_for_idle(timeout_ms=2500)
            time.sleep(0.4)
            state = self._capture_state(record_log=True)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=409, detail=f"Failed to open thread '{contact_name}': {exc}") from exc

        if state.classification.screen.value == "thread_view":
            self._last_action = "Open Thread"
            thread_name = self._thread_contact_name_from_state(state) or contact_name
            self._collect_thread_context(state, thread_name)
            self._last_action_result = f"Opened thread for {thread_name}"
        else:
            self._last_action = "Open Thread"
            self._last_action_result = f"Tapped {contact_name}, but screen remained {state.classification.screen.value}"

        state.last_action = self._last_action
        state.last_action_result = self._last_action_result
        self.latest_state = state
        return state

    def scroll_for_context(self) -> ControllerState:
        """Scroll in thread to collect full message context."""
        state = self.latest_state or self._capture_state(record_log=True)
        if state.classification.screen.value != "thread_view":
            raise HTTPException(status_code=409, detail="Scroll for context requires an open thread view.")

        contact_name = self._thread_contact_name_from_state(state) or state.classification.activity_name
        self._collect_thread_context(state, contact_name)
        initial_count = len(state.thread_context.full_conversation) if state.thread_context else 0
        added_messages = 0
        repeated_top_count = 0
        scrolls_performed = 0
        stopped_reason = "max_scrolls_reached"
        started = time.perf_counter()
        previous_top = self._top_context_key(state)

        for _ in range(max(1, self.settings.context_max_scrolls)):
            elapsed_ms = (time.perf_counter() - started) * 1000
            if elapsed_ms >= self.settings.context_time_budget_ms:
                stopped_reason = "time_budget_reached"
                break
            if state.thread_context and len(state.thread_context.full_conversation) >= self.settings.context_max_messages:
                stopped_reason = "max_context_messages_reached"
                break
            width = max(1, state.classification.screenshot_width)
            height = max(1, state.classification.screenshot_height)
            center_x = width // 2
            start_y = int(height * 0.42)
            end_y = int(height * 0.78)
            self.adb.swipe(center_x, start_y, center_x, end_y, duration_ms=350)
            scrolls_performed += 1
            time.sleep(self.settings.context_scroll_pause_seconds)
            state = self._capture_state(record_log=True)
            if state.classification.screen.value != "thread_view":
                stopped_reason = f"left_thread_view:{state.classification.screen.value}"
                break
            before_merge = len(state.thread_context.full_conversation) if state.thread_context else 0
            self._collect_thread_context(state, contact_name)
            after_merge = len(state.thread_context.full_conversation) if state.thread_context else 0
            delta = max(0, after_merge - before_merge)
            added_messages += delta
            current_top = self._top_context_key(state)
            if current_top and current_top == previous_top:
                repeated_top_count += 1
                if repeated_top_count >= 2:
                    stopped_reason = "same_top_message_seen_twice"
                    break
            else:
                repeated_top_count = 0
            previous_top = current_top
            if delta == 0:
                stopped_reason = "no_new_messages_found"
                break

        total_count = len(state.thread_context.full_conversation) if state.thread_context else initial_count
        if state.thread_context:
            state.thread_context.scrolls_performed = scrolls_performed
            state.thread_context.stopped_reason = stopped_reason
        self._set_context_debug_fields(state)
        self._last_action = "Scroll for Context"
        self._last_action_result = f"Collected {total_count} messages of context (+{added_messages} new); stopped: {stopped_reason}."
        state.last_action = self._last_action
        state.last_action_result = self._last_action_result
        self.latest_state = state
        return state

    def _collect_thread_context(self, state: ControllerState, contact_name: str, hierarchy: str | None = None) -> None:
        """Collect thread context including recent messages with speaker labels."""
        hierarchy = hierarchy or self._read_messages_hierarchy()
        entries = self._extract_thread_message_entries_from_hierarchy(hierarchy) if hierarchy else []
        if not entries:
            entries = [{"speaker": "other", "text": text} for text in state.classification.recent_messages if text]

        existing = None
        if self.latest_state and self.latest_state.thread_context:
            existing = self.latest_state.thread_context
        elif state.thread_context:
            existing = state.thread_context
        merged_entries = self._merge_conversation_entries(existing.full_conversation if existing else [], entries)
        recent = [entry["text"] for entry in merged_entries[-6:]]
        actual_contact = self._thread_contact_name_from_hierarchy(hierarchy) or contact_name
        context = ThreadContext(
            contact_name=actual_contact,
            recent_messages=recent,
            full_conversation=merged_entries,
            message_count=len(merged_entries),
            last_message_time=datetime.now(timezone.utc),
            scrolls_performed=existing.scrolls_performed if existing else 0,
            stopped_reason=existing.stopped_reason if existing else "initial_capture",
        )
        state.thread_context = context
        state.classification.recent_messages = recent
        state.blacklisted = self._normalize_contact_name(actual_contact) in self._blacklist
        self._set_context_debug_fields(state)
        if self.latest_state:
            self.latest_state.thread_context = context
            self.latest_state.blacklisted = state.blacklisted
            self._set_context_debug_fields(self.latest_state)

    def auto_generate_draft(self) -> ControllerState:
        """Automatically generate a draft reply based on current context."""
        state = self.latest_state or self._capture_state(record_log=True)

        if state.classification.screen.value != "thread_view":
            raise HTTPException(status_code=409, detail="Draft generation requires an open thread view.")

        if state.thread_context is None or not state.thread_context.full_conversation:
            self._collect_thread_context(state, self._thread_contact_name_from_state(state) or state.classification.activity_name)

        try:
            messages_for_ai = state.classification.recent_messages or []
            if state.thread_context and state.thread_context.full_conversation:
                messages_with_speakers = [
                    f"[{msg['speaker'].upper()}]: {msg['text']}"
                    for msg in state.thread_context.full_conversation
                ]
                bundle = self.drafting.build_bundle_with_context(
                    recent_messages=messages_for_ai,
                    full_conversation=messages_with_speakers,
                    contact_name=state.thread_context.contact_name if state.thread_context else None,
                )
            else:
                bundle = self.drafting.build_bundle(
                    recent_messages=messages_for_ai,
                    contact_name=self._thread_contact_name_from_state(state),
                )
            self._apply_draft_bundle(state, bundle)
            self._last_action = "Auto Generate Draft"
            recommended = state.recommended_reply_index
            if recommended is None:
                self._last_action_result = f"Generated {len(bundle.reply_suggestions)} draft suggestions"
            else:
                self._last_action_result = (
                    f"Generated {len(bundle.reply_suggestions)} draft suggestions; "
                    f"recommended #{recommended + 1} at {self._recommended_reply_confidence(state):.2f}"
                )
        except Exception as e:
            self._last_action = "Auto Generate Draft"
            self._last_action_result = f"Draft generation failed: {str(e)}"

        state.last_action = self._last_action
        state.last_action_result = self._last_action_result
        self.latest_state = state
        return state

    def auto_send_reply(self, confidence_override: float | None = None) -> ControllerState:
        """Automatically send reply if confidence threshold is met."""
        state = self.latest_state or self._capture_state(record_log=True)
        confidence_threshold = confidence_override or self._auto_send_confidence

        if not state.reply_suggestions:
            state = self.auto_generate_draft()

        recommended_index = state.recommended_reply_index if state.recommended_reply_index is not None else 0
        recommended_confidence = self._recommended_reply_confidence(state)
        recommended_candidate = self._recommended_candidate(state)

        if self._emergency_stop:
            self._last_action = "Auto Send Reply"
            self._last_action_result = "Auto-send blocked: emergency stop is engaged"
        elif self._current_automation_mode != "auto-send" or not self._auto_send_enabled:
            self._last_action = "Auto Send Reply"
            self._last_action_result = "Auto-send blocked: automation mode is not auto-send"
        elif state.blacklisted:
            self._last_action = "Auto Send Reply"
            self._last_action_result = "Auto-send blocked: contact is blacklisted"
        elif state.classification.screen.value != "thread_view":
            self._last_action = "Auto Send Reply"
            self._last_action_result = f"Not in thread view (screen: {state.classification.screen.value})"
        elif recommended_candidate is None:
            self._last_action = "Auto Send Reply"
            self._last_action_result = "No scored draft candidate is available to send"
        elif recommended_candidate.final_decision != "send":
            self._last_action = "Auto Send Reply"
            self._last_action_result = recommended_candidate.blocked_reason or state.auto_send_blocked_reason or "Auto-send blocked by critic"
        elif not recommended_candidate.auto_send_allowed:
            self._last_action = "Auto Send Reply"
            self._last_action_result = state.auto_send_blocked_reason or "Recommended candidate is marked manual-review only"
        elif recommended_confidence < confidence_threshold:
            self._last_action = "Auto Send Reply"
            self._last_action_result = (
                f"Reply confidence {recommended_confidence:.2f} below threshold {confidence_threshold:.2f}"
            )
        elif state.reply_suggestions and state.classification.screen.value == "thread_view":
            try:
                sequence = self._selected_reply_sequence(state, recommended_index)
                sent_parts: list[str] = []
                for bubble in sequence:
                    self._type_custom_reply(bubble)
                    state = self.approve_send_current_draft()
                    sent_parts.append(bubble)
                    time.sleep(0.35)
                self._last_action = "Auto Send Reply"
                self._last_action_result = (
                    f"Sent candidate #{recommended_index + 1}: {' / '.join(sent_parts)[:80]}"
                )
            except Exception as e:
                self._last_action = "Auto Send Reply"
                self._last_action_result = f"Auto-send failed: {str(e)}"
                state.last_action = self._last_action
                state.last_action_result = self._last_action_result
                return state

        state.last_action = self._last_action
        state.last_action_result = self._last_action_result
        self.latest_state = state
        return state

    def get_automation_state(self) -> dict[str, object]:
        """Get current automation state."""
        target_config = load_automation_targets()
        return {
            "automation_enabled": self._automation_enabled,
            "mode": self._current_automation_mode,
            "confidence_threshold": self._automation_confidence,
            "auto_send_enabled": self._auto_send_enabled,
            "auto_send_confidence": self._auto_send_confidence,
            "blacklist_count": len(self._blacklist),
            "blacklist_contacts": sorted(self._blacklist),
            "queue_size": len(self._unread_queue),
            "queue_preview": [item.model_dump(mode="json") for item in self._unread_queue[:8]],
            "current_contact": self.latest_state.thread_context.contact_name if self.latest_state and self.latest_state.thread_context else None,
            "recommended_reply_index": self.latest_state.recommended_reply_index if self.latest_state else None,
            "recommended_reply_confidence": self._recommended_reply_confidence(self.latest_state) if self.latest_state else 0.0,
            "auto_send_blocked_reason": self.latest_state.auto_send_blocked_reason if self.latest_state else None,
            "relationship_type": self.latest_state.relationship_type if self.latest_state else "unknown",
            "intent_type": self.latest_state.intent_type if self.latest_state else "other",
            "retrieved_examples_count": self.latest_state.retrieved_examples_count if self.latest_state else 0,
            "retrieved_examples": self.latest_state.retrieved_examples if self.latest_state else [],
            "final_decision": self.latest_state.final_decision if self.latest_state else "review",
            "blocked_reason": self.latest_state.blocked_reason if self.latest_state else None,
            "timings_ms": self.latest_state.timings_ms if self.latest_state else {},
            "prompt_preview": self.latest_state.prompt_preview if self.latest_state else None,
            "context_messages_count": self.latest_state.context_messages_count if self.latest_state else 0,
            "scrolls_performed": self.latest_state.scrolls_performed if self.latest_state else 0,
            "context_stopped_reason": self.latest_state.context_stopped_reason if self.latest_state else None,
            "last_action": self._last_action,
            "last_action_result": self._last_action_result,
            "target_cycle": target_config.model_dump(mode="json"),
        }

    def run_target_cycle(self, *, dry_run: bool = True) -> dict[str, object]:
        config = load_automation_targets()
        started_at = datetime.now(timezone.utc).isoformat()
        results: list[dict[str, object]] = []
        if self._emergency_stop:
            return {"cycle_started": started_at, "mode": config.mode, "dry_run": dry_run, "results": [], "status": "blocked", "reason": "emergency_stop"}
        if not config.enabled:
            return {"cycle_started": started_at, "mode": config.mode, "dry_run": dry_run, "results": [], "status": "skipped", "reason": "automation_disabled"}
        limits = config.global_limits
        max_contacts = int(limits.get("max_contacts_per_cycle", 5) or 5)
        for target in [item for item in config.targets if item.enabled][:max_contacts]:
            contact_started = time.perf_counter()
            normalized = self._normalize_contact_name(target.name)
            if normalized in self._blacklist:
                self._conversation_state.record_skip(target.name, "blacklisted")
                results.append({"contact": target.name, "status": "skipped", "reason": "blacklisted"})
                continue
            profile = get_contact_profile(target.name)
            relationship = normalize_relationship_type(profile.relationship_type if profile else "unknown")
            latest_incoming = ""
            context_entries: list[dict[str, str]] = []
            try:
                if dry_run:
                    state = self.latest_state
                    if state and state.thread_context and self._contact_matches(state.thread_context.contact_name, target.name):
                        context_entries = state.thread_context.full_conversation
                        latest_incoming = self._latest_other_message(context_entries)
                    else:
                        latest_incoming = ""
                else:
                    state = self.open_thread(target.name)
                    self.scroll_for_context()
                    state = self.latest_state or state
                    context_entries = state.thread_context.full_conversation if state.thread_context else []
                    latest_incoming = self._latest_other_message(context_entries)
            except Exception as exc:
                self._conversation_state.record_skip(target.name, "open_or_capture_failed")
                results.append({"contact": target.name, "status": "failed", "reason": str(exc)})
                continue
            intent = classify_intent(latest_incoming, [entry.get("text", "") for entry in context_entries[-6:]])
            contact_state = self._conversation_state.get_contact_state(target.name)
            policy = should_reply(
                target.name,
                latest_incoming,
                context_entries,
                contact_state,
                intent,
                allowlisted=True,
                rate_limited=contact_state.send_count_last_hour >= target.max_messages_per_hour,
            )
            if not policy["should_reply"]:
                self._conversation_state.record_skip(target.name, str(policy["reason"]))
                results.append(
                    {
                        "contact": target.name,
                        "status": "skipped",
                        "reason": policy["reason"],
                        "intent_type": intent,
                        "relationship_type": relationship,
                        "latest_incoming": latest_incoming,
                    }
                )
                continue
            bundle = self.drafting.regenerate_bundle(
                contact_name=target.name,
                incoming=latest_incoming,
                context=[entry.get("text", "") for entry in context_entries[-6:]],
                relationship_type=relationship,
                intent_type=intent,
                avoid_candidates=contact_state.recent_candidates,
                diversity_mode="natural",
            )
            selected_reply = bundle.reply_suggestions[0] if bundle.reply_suggestions else ""
            status = "review"
            reason = str(policy["reason"])
            can_send = (
                config.mode == "guarded_auto_send"
                and target.auto_send
                and profile is not None
                and profile.auto_send_allowed
                and relationship in {"close_friend", "casual_friend"}
                and bundle.final_decision == "send"
                and not dry_run
            )
            can_draft = config.mode in {"draft_only", "guarded_auto_send"} and profile is not None and profile.auto_draft_allowed
            if can_send:
                self._type_custom_reply(selected_reply)
                self.approve_send_current_draft()
                self._conversation_state.record_send(target.name, selected_reply)
                self._conversation_state.set_last_seen(target.name, latest_incoming)
                status = "sent"
            elif can_draft and not dry_run:
                self._type_custom_reply(selected_reply)
                self._conversation_state.record_draft(target.name, selected_reply)
                self._conversation_state.set_last_seen(target.name, latest_incoming)
                status = "drafted"
            else:
                self._conversation_state.record_candidates(target.name, bundle.reply_suggestions)
                if dry_run:
                    status = "review" if config.mode == "review_only" else "drafted"
                    reason = "dry_run_no_type_or_send"
                else:
                    status = "review"
            results.append(
                {
                    "contact": target.name,
                    "status": status,
                    "reason": reason,
                    "intent_type": intent,
                    "relationship_type": relationship,
                    "latest_incoming": latest_incoming,
                    "selected_reply": selected_reply,
                    "decision": bundle.final_decision,
                    "retrieval_backend": bundle.retrieval_backend,
                    "timings_ms": {"total": round((time.perf_counter() - contact_started) * 1000, 2)},
                }
            )
        return {"cycle_started": started_at, "mode": config.mode, "dry_run": dry_run, "results": results}

    def _latest_other_message(self, entries: list[dict[str, str]]) -> str:
        for entry in reversed(entries):
            if entry.get("speaker") == "other" and entry.get("text"):
                return str(entry["text"])
        return ""

    def get_debug_timings(self) -> dict[str, object]:
        state_timings = self.latest_state.timings_ms if self.latest_state else {}
        return {
            "latest_timings_ms": dict(state_timings or self._last_debug_timings_ms),
            "metrics": self.latest_state.metrics.model_dump(mode="json") if self.latest_state and self.latest_state.metrics else None,
            "last_action": self._last_action,
            "last_action_result": self._last_action_result,
        }

    def record_feedback(self, payload: dict[str, object]) -> dict[str, object]:
        try:
            row = append_correction(self.settings.ai_reply_training_messages_dir, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "saved", "correction": row}

    def _catbot_split_incoming_burst(self, incoming: str) -> tuple[str, list[str]]:
        parts = [part.strip() for part in str(incoming or "").splitlines() if part.strip()]
        if len(parts) <= 1:
            return str(incoming or "").strip(), []
        return parts[-1], ["\n".join(parts[:-1])]

    def training_chat(self, request: TrainingChatRequest) -> dict[str, object]:
        bundle = self._training_bundle(request, regenerate=False)
        session = self._training_session_payload(request, bundle)
        self._append_jsonl(self.training_sessions_path, session)
        return self._training_response(request, bundle)

    def training_regenerate(self, request: TrainingChatRequest) -> dict[str, object]:
        bundle = self._training_bundle(request, regenerate=True)
        session = self._training_session_payload(request, bundle)
        session["source"] = "training_page_regenerate"
        self._append_jsonl(self.training_sessions_path, session)
        return self._training_response(request, bundle)

    def _whatsapp_latest_incoming_burst(self, messages: list[dict[str, object]], incoming_index: int) -> list[dict[str, object]]:
        if not messages or incoming_index < 0:
            return []
        latest = messages[incoming_index]
        burst: list[dict[str, object]] = [latest]
        previous_time = self._parse_whatsapp_web_timestamp(latest.get("timestamp"))
        for index in range(incoming_index - 1, -1, -1):
            message = messages[index]
            if message.get("speaker") == "me":
                break
            message_time = self._parse_whatsapp_web_timestamp(message.get("timestamp"))
            if previous_time and message_time and abs((previous_time - message_time).total_seconds()) > 90 * 60:
                break
            if not previous_time and not message_time and len(burst) >= 28:
                break
            burst.append(message)
            previous_time = message_time or previous_time
        return list(reversed(burst))

    def _whatsapp_text_has_signoff(self, normalized: str) -> bool:
        if any(term in normalized for term in ("goodnight", "good night", "sleep well", "sleepwell", "sleep tight")):
            return True
        if re.search(r"\bgn\b", normalized):
            return True
        return bool(re.search(r"\bnight\b", normalized) and any(term in normalized for term in ("baby", "babe", "love", "handsome", "mwah", " x", "xx")))

    def _whatsapp_obligation_moves(self, burst_texts: list[str]) -> list[str]:
        blob = normalize_text(" ".join(burst_texts))
        latest = normalize_text(burst_texts[-1] if burst_texts else "")
        moves: list[str] = []
        if any(term in blob for term in ("eid mubarak", "ramadan mubarak")):
            moves.append("celebration_or_holiday")
        if any(term in blob for term in ("sorry", "my bad", "apolog")):
            moves.append("apology_or_repair")
        if self._whatsapp_has_any_term(blob, ("hurt", "hurting", "ignored me", "ignore me", "not funny", "kill myself", "scared", "crying", "cry man")) or self._whatsapp_has_relationship_conflict_context(blob):
            moves.append("conflict_or_hurt")
        if any(term in blob for term in ("on read", "on seen", "left me on seen", "leave me on seen", "get aired", "got aired", "air me", "air u", "aired", "airing", "ignoring me")):
            moves.append("on_read_complaint")
        if self._whatsapp_has_family_health_context(blob):
            moves.append("emotional_disclosure")
        if self._whatsapp_has_any_term(blob, ("presentation", "marker", "feedback", "flashcard", "flashcards", "failed", "fail", "overthinking", "reading my messages", "read it properly", "unsupported", "offended")) or self._whatsapp_has_missing_luck_context(blob):
            moves.append("emotional_disclosure")
        if any(term in blob for term in ("a lot on my mind", "lot on my mind", "gonna tell u later", "going to tell u later")):
            moves.append("future_rant_teaser")
        if any(term in blob for term in ("rant to u", "rant to you")):
            moves.append("future_rant_teaser")
        if self._whatsapp_has_sexual_flirt_context(blob):
            moves.append("sexual_flirt")
        if self._whatsapp_is_direct_plan_text(blob):
            moves.append("invite_or_plan")
        if self._whatsapp_has_work_cover_update(blob):
            moves.append("logistics_update")
        if self._whatsapp_has_reachability_pressure(blob):
            moves.append("reachability_pressure")
        if any(term in latest for term in ("shoe size", "7.5", "wich ones", "which ones", "which one", "what time", "what day", "what date", "when ur free", "when you're free", "where r u", "where are u", "where are you", "what should i post", "which one should i post", "are you busy", "are u busy", "r u busy", "busy rn", "busy right now")) or re.search(r"\bwhen\s+(?:r|are|u|you|ur|you're)\b", latest):
            moves.append("practical_question")
        if latest.endswith("?") or any(text.strip().endswith("?") for text in burst_texts):
            moves.append("question")
        if any(term in blob for term in ("miss u", "miss you", "missed u", "missed you", "imy")):
            moves.append("missing_you")
        if any(term in blob for term in ("love u", "love you", "i love you", "mwah", "kiss u", "kiss you")):
            moves.append("affection")
        if self._whatsapp_text_has_signoff(blob):
            moves.append("goodnight")
        if self._whatsapp_text_has_signoff(blob) and any(term in blob for term in ("love", "miss", "baby", "babe", "my love", "handsome", "mwah", "kiss")):
            moves.append("romantic_signoff")
        if any(term in blob for term in ("handsome", "cute", "pretty", "beautiful")):
            moves.append("compliment")
        if not self._whatsapp_text_has_signoff(latest) and (re.match(r"^(hi|hey|hello|yo)\b", latest) or is_simple_greeting(latest, [])):
            moves.append("greeting")
        if any(term in blob for term in ("fatty", "rude", "idiot")):
            moves.append("teasing")
        if any(term in blob for term in ("movie", "character", "mum and", "wouldn't leave me alone", "wouldnt leave me alone", "bro yesterday")):
            moves.append("story_share")
        if self._whatsapp_has_status_check(blob) or any(term in blob for term in ("r u back", "are u back", "you back", "are u ok", "are you ok", "are you okay", "are you busy", "are u busy", "busy rn", "busy right now", "how are u", "how are you", "hru")) or re.search(r"\bu good\b", blob):
            moves.append("check_in")
        return list(dict.fromkeys(moves or ["normal"]))

    def _whatsapp_concrete_topics(self, burst_texts: list[str]) -> list[str]:
        blob = normalize_text(" ".join(burst_texts))
        topics: list[str] = []
        topic_terms = {
            "presentation": ("presentation", "presented", "slides"),
            "marker": ("marker", "examiner", "marked", "feedback", "harsh", "cold"),
            "flashcards": ("flashcard", "flashcards"),
            "overthinking": ("overthinking", "overthink", "panicked", "panic"),
            "luck": ("wish me luck", "wished me luck", "didnt wish me luck", "didn't wish me luck", "didnt wish", "didn't wish"),
            "reading_properly": ("reading my messages", "read it properly", "reading properly", "not reading"),
            "unsupported": ("unsupported", "not supporting", "didnt support", "didn't support", "offended"),
            "academic_stress": ("exam", "revision", "presentation", "passed", "failed", "content", "questions"),
            "food": ("food", "foods", "eat", "eaten", "takeout", "lunch"),
            "rain": ("rain", "rained"),
            "banter": ("pissed", "dragging", "rude", "icl", "lmao"),
            "affection": ("love u", "love you", "miss u", "miss you", "baby", "mwah", "kiss"),
            "logistics": ("what time", "when", "where", "home", "free", "come", "link", "tomorrow", "next week", "meet up", "meet ups", "work", "cover"),
            "on_read_repair": ("on read", "get aired", "got aired", "air me", "air u", "aired", "airing"),
            "reachability_pressure": ("text me back", "message me back", "today would be nice", "gonna ring", "busy again"),
            "status_check": ("are you okay", "are u okay", "are you busy", "are u busy", "busy rn", "busy right now", "busy again", "still in northbridge", "appointment done", "gotten your appointment done"),
        }
        for label, terms in topic_terms.items():
            if self._whatsapp_has_any_term(blob, terms):
                topics.append(label)
        if self._whatsapp_has_family_health_context(blob):
            topics.append("family_health")
        if self._whatsapp_has_work_cover_update(blob):
            topics.append("work_cover")
        if self._whatsapp_has_reachability_pressure(blob):
            topics.append("reachability_pressure")
        if self._whatsapp_has_status_check(blob) or re.search(r"\b(?:u|you) good\b", blob):
            topics.append("status_check")
        if self._whatsapp_has_relationship_conflict_context(blob):
            topics.append("relationship_conflict")
        return list(dict.fromkeys(topics))

    def _whatsapp_fragmentation_score(self, burst_texts: list[str]) -> float:
        if not burst_texts:
            return 0.0
        short_parts = sum(1 for text in burst_texts if len(normalize_text(text).split()) <= 12)
        score = min(1.0, (len(burst_texts) - 1) / 10 + short_parts / max(1, len(burst_texts)) * 0.35)
        return round(score, 3)

    def _whatsapp_user_style_burst_baseline(self, messages: list[dict[str, object]], incoming_index: int) -> int:
        runs: list[int] = []
        current = 0
        start = max(0, incoming_index - 80)
        for message in messages[start:incoming_index + 1]:
            if message.get("speaker") == "other":
                current += 1
            elif current:
                runs.append(current)
                current = 0
        if current:
            runs.append(current)
        if not runs:
            return 1
        runs = sorted(runs)
        return max(1, min(12, int(round(sum(runs[-5:]) / min(5, len(runs))))))

    def _whatsapp_repair_required(self, burst_texts: list[str]) -> bool:
        blob = normalize_text(" ".join(burst_texts))
        return any(term in blob for term in (
            "r u even reading",
            "are u even reading",
            "reading my messages properly",
            "read it properly",
            "you didnt wish me luck",
            "you didn't wish me luck",
            "didnt wish me luck",
            "didn't wish me luck",
            "not reading properly",
            "offended",
            "unsupported",
        ))

    def _whatsapp_emotional_intensity(self, burst_texts: list[str], topics: list[str], repair_required: bool) -> str:
        blob = normalize_text(" ".join(burst_texts))
        if any(term in blob for term in ("kill myself", "coma", "losing her", "suicide", "hurt myself")):
            return "high"
        if len(burst_texts) >= 8 or repair_required or any(topic in topics for topic in ("family_health", "presentation", "unsupported")):
            return "medium_high"
        if len(burst_texts) >= 4 or any(topic in topics for topic in ("academic_stress", "overthinking")):
            return "medium"
        return "low"

    def _whatsapp_target_reply_burst_size(
        self,
        moves: list[str],
        *,
        burst_count: int,
        fragmentation_score: float,
        emotional_intensity: str,
        repair_required: bool,
        topics: list[str],
    ) -> dict[str, int]:
        if "presentation" in topics and repair_required:
            return {"min": 18, "max": 26}
        if "family_health" in topics and emotional_intensity == "high":
            return {"min": 10, "max": 18}
        if "on_read_complaint" in moves:
            if "relationship_conflict" in topics:
                return {"min": 8, "max": 14}
            if "conflict_or_hurt" in moves:
                return {"min": 5, "max": 10}
            return {"min": 4, "max": 8}
        if "on_read_complaint" in moves and "logistics_update" in moves:
            return {"min": 4, "max": 8}
        if "relationship_conflict" in topics:
            if burst_count >= 32:
                return {"min": 12, "max": 18}
            if burst_count >= 16:
                return {"min": 10, "max": 16}
            return {"min": 8, "max": 14}
        if "reachability_pressure" in moves and "check_in" in moves:
            return {"min": 3, "max": 6}
        if "logistics_update" in moves:
            return {"min": 2, "max": 5}
        if "academic_stress" in topics and repair_required and (burst_count >= 8 or fragmentation_score >= 0.8):
            return {"min": 20, "max": 35}
        if "academic_stress" in topics and (burst_count >= 6 or fragmentation_score >= 0.75):
            return {"min": 18, "max": 26}
        if any(topic in topics for topic in ("food", "rain", "banter")):
            return {"min": 2, "max": 5}
        if "emotional_disclosure" in moves and repair_required and burst_count >= 8:
            return {"min": 20, "max": 35}
        if "emotional_disclosure" in moves and repair_required:
            return {"min": 14, "max": 24}
        if "emotional_disclosure" in moves and (burst_count >= 8 or fragmentation_score >= 0.8):
            return {"min": 16, "max": 28}
        if "emotional_disclosure" in moves:
            return {"min": 10, "max": 18}
        if "sexual_flirt" in moves or "teasing" in moves:
            return {"min": 5, "max": 12}
        if any(move in moves for move in ("affection", "missing_you")) and not any(move in moves for move in ("goodnight", "romantic_signoff")):
            return {"min": 3, "max": 7}
        if any(move in moves for move in ("greeting", "goodnight", "romantic_signoff")):
            return {"min": 2, "max": 5}
        if any(move in moves for move in ("invite_or_plan", "practical_question", "question")):
            return {"min": 1, "max": 3}
        if "normal_romantic" in moves:
            return {"min": 5, "max": 10}
        return {"min": 1, "max": 3}

    def _whatsapp_duplicate_check(self, units: list[str]) -> dict[str, object]:
        normalized = [normalize_text(unit).strip(" ?!.,") for unit in units if normalize_text(unit).strip(" ?!.,")]
        duplicates: list[str] = []
        near_duplicates: list[dict[str, object]] = []
        seen: set[str] = set()
        for item in normalized:
            if item in seen:
                duplicates.append(item)
            seen.add(item)
        for i, left in enumerate(normalized):
            for right in normalized[i + 1:]:
                ratio = SequenceMatcher(None, left, right).ratio()
                if ratio >= 0.86:
                    near_duplicates.append({"left": left, "right": right, "ratio": round(ratio, 3)})
        return {
            "passed": not duplicates and not near_duplicates,
            "duplicates": duplicates,
            "near_duplicates": near_duplicates,
            "unit_count": len(normalized),
        }

    def _whatsapp_reply_units(self, reply: str) -> list[str]:
        return [part.strip() for part in re.split(r"\n+|\s+/\s+", str(reply or "")) if part.strip()]

    def _whatsapp_send_queue_safety(self, units: list[str], *, auto_send_allowed: bool) -> dict[str, object]:
        duplicate_check = self._whatsapp_duplicate_check(units)
        contradictions: list[str] = []
        blob = normalize_text(" ".join(units))
        if "you failed" in blob and ("you passed" in blob or "in shaa allah you passed" in blob):
            contradictions.append("pass_fail_contradiction")
        return {
            "passed": bool(duplicate_check["passed"]) and not contradictions,
            "auto_send_allowed": auto_send_allowed,
            "duplicate_check": duplicate_check,
            "contradictions": contradictions,
            "requires_interruptible_queue": len(units) > 1,
        }

    def _whatsapp_response_acts_present(self, reply: str, obligation: WhatsAppConversationObligation) -> list[str]:
        acts = [
            act
            for act in obligation.required_response_acts
            if self._whatsapp_reply_satisfies_act(reply, act, obligation)
        ]
        units = self._whatsapp_reply_units(reply)
        for unit in units:
            normalized = normalize_text(unit)
            if any(term in normalized for term in ("sorry", "my bad", "shouldve", "should've")):
                acts.append("accountability")
            if any(term in normalized for term in ("get why", "understand", "thats a lot", "that's a lot", "sounds so")):
                acts.append("validation")
            if any(term in normalized for term in ("passed", "failed", "content", "questions", "marker", "flashcard", "worked hard")):
                acts.append("concrete_grounding")
            if any(term in normalized for term in ("love u", "proud", "sleep", "rest", "im here", "i'm here")):
                acts.append("care_or_closing")
        return list(dict.fromkeys(acts))

    def _whatsapp_required_response_acts(self, moves: list[str], burst_texts: list[str]) -> list[str]:
        blob = normalize_text(" ".join(burst_texts))
        acts: list[str] = []
        if "celebration_or_holiday" in moves:
            acts.append("return_holiday_greeting")
        if "apology_or_repair" in moves:
            acts.append("acknowledge_apology_or_repair")
        if "on_read_complaint" in moves:
            acts.append("acknowledge_on_read_or_airing")
        if "reachability_pressure" in moves:
            acts.append("acknowledge_delayed_response")
        if "emotional_disclosure" in moves or "conflict_or_hurt" in moves:
            acts.append("provide_emotional_support")
        topics = set(self._whatsapp_concrete_topics(burst_texts))
        if "reading_properly" in topics:
            acts.append("acknowledge_reading_failure")
        if "luck" in topics:
            acts.append("apologize_missing_luck")
        if "presentation" in topics or "academic_stress" in topics:
            acts.append("ground_academic_reassurance")
        if "marker" in topics or "flashcards" in topics:
            acts.append("acknowledge_specific_academic_details")
        if "relationship_conflict" in topics:
            acts.extend([
                "acknowledge_relationship_hurt",
                "take_accountability_for_distance",
                "reassure_care_without_defensiveness",
            ])
        if "future_rant_teaser" in moves:
            acts.append("invite_them_to_talk")
        if "logistics_update" in moves:
            acts.append("acknowledge_work_cover_update")
        if "sexual_flirt" in moves:
            acts.append("respond_to_sexual_flirt_safely")
        if "invite_or_plan" in moves:
            if "tutor on saturdays" in blob or "can't do a sat" in blob or "cant do a sat" in blob:
                acts.append("acknowledge_unavailable_and_suggest")
            elif "wanna go out" in blob or "go out" in blob:
                acts.append("answer_invite_directly")
            elif "free on sunday" in blob or ("free" in blob and "sunday" in blob):
                acts.append("answer_back_status_and_availability")
            else:
                acts.append("answer_availability")
        if "practical_question" in moves:
            if "post" in blob:
                acts.append("ask_to_see_options_or_pick")
            elif "shoe size" in blob or "7.5" in blob:
                acts.append("answer_practical_question")
            elif "where r u" in blob or "where are u" in blob or "where are you" in blob:
                acts.append("answer_location_or_status")
            elif "what time" in blob or "what day" in blob or "what date" in blob or "when ur free" in blob or "when you're free" in blob or "are you busy" in blob or "are u busy" in blob or "r u busy" in blob or "busy rn" in blob or "busy right now" in blob or re.search(r"\bwhen\s+(?:r|are|u|you|ur|you're)\b", blob):
                acts.append("answer_availability")
        if "missing_you" in moves:
            acts.append("reciprocate_missing")
        if "affection" in moves:
            acts.append("reciprocate_affection")
        if "goodnight" in moves:
            acts.append("return_goodnight")
        if "greeting" in moves:
            acts.append("answer_greeting")
        if "compliment" in moves and "romantic_signoff" not in moves:
            acts.append("warmly_receive_compliment")
        if "teasing" in moves:
            acts.append("answer_teasing_playfully")
        if "story_share" in moves:
            acts.append("react_to_story_content")
        if "check_in" in moves:
            acts.append("answer_check_in_or_status")
        if "status_check" in topics and any(term in blob for term in ("northbridge", "appointment", "busy again")):
            acts.append("answer_specific_status_check")
        if "status_check" in topics and any(term in blob for term in ("are you busy", "are u busy", "r u busy", "busy rn", "busy right now")):
            acts.append("answer_availability")
        if "kiss" in blob or "mwah" in blob:
            acts.append("reciprocate_kiss_affection")
        return list(dict.fromkeys(acts or ["answer_relevantly"]))

    def _whatsapp_forbidden_response_acts(self, moves: list[str], burst_texts: list[str]) -> list[str]:
        forbidden = [
            "generic_clarification_when_clear",
            "generic_filler",
            "repeat_already_sent_bubble",
            "reply_only_to_final_word_while_ignoring_burst",
        ]
        if any(move in moves for move in ("affection", "missing_you", "romantic_signoff")):
            forbidden.append("ignore_direct_affection")
        if any(move in moves for move in ("question", "practical_question")):
            forbidden.append("ignore_direct_question")
        if "emotional_disclosure" in moves:
            forbidden.extend(["topic_change_during_emotional_disclosure", "sexual_escalation_during_emotional_context"])
        if "invite_or_plan" in moves:
            forbidden.append("ask_availability_when_already_given")
        return list(dict.fromkeys(forbidden))

    def _whatsapp_tone_target(self, relationship: str, moves: list[str]) -> str:
        if "emotional_disclosure" in moves or "conflict_or_hurt" in moves:
            return "serious, validating, reassuring, no flirting"
        if "sexual_flirt" in moves:
            return "playful romantic teasing without unsafe over-escalation"
        if "romantic_signoff" in moves or "goodnight" in moves:
            return "short, warm, reciprocal romantic signoff"
        if "invite_or_plan" in moves or "practical_question" in moves:
            return "direct, useful, casual"
        if relationship == "romantic_interest":
            return "warm, casual, affectionate, lightly teasing"
        return "casual and relevant"

    def _whatsapp_reply_shape(self, moves: list[str]) -> str:
        if "emotional_disclosure" in moves or "conflict_or_hurt" in moves:
            return "comfort/reassure"
        if "invite_or_plan" in moves:
            return "plan logistics"
        if "practical_question" in moves or "question" in moves:
            return "answer a question"
        if "romantic_signoff" in moves or "affection" in moves or "missing_you" in moves:
            return "match affection"
        if "story_share" in moves:
            return "react to story"
        return "continue the exchange"

    def _whatsapp_conversation_obligation(
        self,
        messages: list[dict[str, object]],
        incoming_index: int,
        *,
        relationship: str,
        already_sent_replies: list[str],
    ) -> WhatsAppConversationObligation:
        burst = self._whatsapp_latest_incoming_burst(messages, incoming_index)
        burst_texts = [str(message.get("text") or "").strip() for message in burst if str(message.get("text") or "").strip()]
        moves = self._whatsapp_obligation_moves(burst_texts)
        if relationship == "romantic_interest" and moves == ["normal"]:
            moves = ["normal_romantic"]
        topics = self._whatsapp_concrete_topics(burst_texts)
        fragmentation = self._whatsapp_fragmentation_score(burst_texts)
        baseline = self._whatsapp_user_style_burst_baseline(messages, incoming_index)
        repair_required = self._whatsapp_repair_required(burst_texts)
        intensity = self._whatsapp_emotional_intensity(burst_texts, topics, repair_required)
        latest_burst_time = self._parse_whatsapp_web_timestamp(burst[-1].get("timestamp")) if burst else None
        late_reply_required = bool(latest_burst_time and datetime.now(timezone.utc) - latest_burst_time >= timedelta(hours=8))
        required_acts = self._whatsapp_required_response_acts(moves, burst_texts)
        late_reply_required = bool(
            late_reply_required
            and any(move in moves for move in ("reachability_pressure", "on_read_complaint"))
        )
        if late_reply_required and "acknowledge_delayed_response" not in required_acts:
            required_acts.insert(0, "acknowledge_delayed_response")
        target = self._whatsapp_target_reply_burst_size(
            moves,
            burst_count=len(burst_texts),
            fragmentation_score=fragmentation,
            emotional_intensity=intensity,
            repair_required=repair_required,
            topics=topics,
        )
        require_reply_per_bubble = (
            len(burst_texts) > 1
            and (
                repair_required
                or intensity in {"medium_high", "high"}
                or any(topic in topics for topic in ("academic_stress", "presentation", "family_health", "relationship_conflict"))
                or any(move in moves for move in ("conflict_or_hurt", "reachability_pressure", "on_read_complaint"))
            )
        )
        if require_reply_per_bubble:
            per_bubble_cap = int(target.get("min", 1)) if "relationship_conflict" in topics else len(burst_texts)
            target = {
                "min": max(int(target.get("min", 1)), min(len(burst_texts), per_bubble_cap)),
                "max": max(int(target.get("max", 3)), min(len(burst_texts), per_bubble_cap)),
            }
        if late_reply_required:
            target = {
                "min": max(int(target.get("min", 1)), len(burst_texts) + 1),
                "max": max(int(target.get("max", 3)), len(burst_texts) + 1),
            }
        return WhatsAppConversationObligation(
            latest_incoming_burst=burst_texts,
            conversation_move=moves,
            required_response_acts=required_acts,
            forbidden_response_acts=self._whatsapp_forbidden_response_acts(moves, burst_texts),
            tone_target=self._whatsapp_tone_target(relationship, moves),
            reply_shape=self._whatsapp_reply_shape(moves),
            already_sent_replies=already_sent_replies,
            incoming_fragmentation_score=fragmentation,
            user_style_burst_baseline=baseline,
            emotional_intensity=intensity,
            repair_required=repair_required,
            concrete_topics_detected=topics,
            target_reply_burst_size=target,
            auto_send_allowed=target["max"] <= 14,
        )

    def _whatsapp_reply_satisfies_act(self, reply: str, act: str, obligation: WhatsAppConversationObligation) -> bool:
        normalized = normalize_text(reply)
        burst_blob = normalize_text(" ".join(obligation.latest_incoming_burst))
        if act == "return_holiday_greeting":
            return "eid mubarak" in normalized or "mubarak" in normalized
        if act == "acknowledge_apology_or_repair":
            return any(term in normalized for term in ("its okay", "it's okay", "dw", "dont worry", "don't worry", "all good", "youre okay", "you're okay", "baby"))
        if act == "acknowledge_on_read_or_airing":
            return any(term in normalized for term in (
                "sorry",
                "my bad",
                "didnt mean to air",
                "didn't mean to air",
                "not ignoring",
                "not ignore",
                "wasnt ignoring",
                "wasn't ignoring",
                "wasnt airing",
                "wasn't airing",
                "didnt mean to leave u on read",
                "didn't mean to leave u on read",
                "on read",
                "air u",
                "aired",
            ))
        if act == "acknowledge_delayed_response":
            return any(term in normalized for term in (
                "sorry",
                "my bad",
                "didnt mean",
                "didn't mean",
                "wasnt ignoring",
                "wasn't ignoring",
                "got caught",
                "caught up",
                "busy",
                "i know",
            ))
        if act == "provide_emotional_support":
            if any(topic in set(obligation.concrete_topics_detected or []) for topic in ("academic_stress", "presentation")):
                return any(term in normalized for term in ("sorry", "get why", "sounds so stressful", "overthinking", "proud", "worked hard"))
            supportive = sum(1 for term in ("sorry", "scared", "get why", "understand", "here", "talk to me", "alone", "deal with", "hold all") if term in normalized)
            return supportive >= 2
        if act == "acknowledge_reading_failure":
            return any(term in normalized for term in ("read it properly", "reading properly", "shouldve read", "should've read", "you explained", "i wasnt reading", "i wasn't reading"))
        if act == "apologize_missing_luck":
            return any(term in normalized for term in ("shouldve wished u luck", "should've wished u luck", "didnt wish u luck", "didn't wish u luck", "sorry i didnt wish", "sorry i didn't wish", "my bad"))
        if act == "ground_academic_reassurance":
            return any(term in normalized for term in ("presentation", "content", "questions", "passed", "failed", "worked hard", "in shaa allah"))
        if act == "acknowledge_specific_academic_details":
            return any(term in normalized for term in ("marker", "harsh", "feedback", "flashcard", "flashcards", "cold"))
        if act == "acknowledge_relationship_hurt":
            return any(term in normalized for term in ("hurt", "drained", "crying", "alone", "not been there", "made u feel", "made you feel", "one sided", "not fair"))
        if act == "take_accountability_for_distance":
            return any(term in normalized for term in ("im sorry", "i'm sorry", "my fault", "thats on me", "that's on me", "i havent been there", "i haven't been there", "i hear u", "i hear you"))
        if act == "reassure_care_without_defensiveness":
            return any(term in normalized for term in ("i love u", "love u", "i care", "care about u", "dont want to lose", "don't want to lose", "not trying to dismiss"))
        if act == "invite_them_to_talk":
            return any(term in normalized for term in ("tell me", "talk to me", "im listening", "i'm listening", "when youre ready", "when you're ready", "im here", "i'm here"))
        if act == "acknowledge_work_cover_update":
            work_ack = any(term in normalized for term in ("cover", "work", "tomorrow", "next week", "meet", "free"))
            accepts = any(term in normalized for term in ("calm", "fine", "dw", "dont stress", "don't stress", "no worries", "sort", "works", "lmk", "let me know"))
            return work_ack and accepts
        if act == "respond_to_sexual_flirt_safely":
            return any(term in normalized for term in ("trouble", "behave", "naughty", "ice cream", "lick", "icl"))
        if act == "acknowledge_unavailable_and_suggest":
            acknowledges = any(term in normalized for term in ("fine", "calm", "no worries", "that's fine", "thats fine", "okay"))
            suggests = any(term in normalized for term in ("sunday", "thursday", "another day", "what about", "friday"))
            return acknowledges and suggests
        if act == "answer_invite_directly":
            return any(term in normalized for term in ("im down", "i'm down", "yeah", "yh", "yes", "ofc", "sounds good")) and any(term in normalized for term in ("what time", "when", "thursday", "thirsday", "down"))
        if act == "answer_back_status_and_availability":
            return any(term in normalized for term in ("back", "here", "home")) and any(term in normalized for term in ("sunday", "works", "free"))
        if act == "answer_availability":
            return any(term in normalized for term in ("free", "later", "tonight", "tomorrow", "today", "weekend", "next week", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "sat", "not sure", "dont know", "don't know", "bbq"))
        if act == "ask_to_see_options_or_pick":
            return any(term in normalized for term in ("send", "show", "lemme see", "let me see", "ill pick", "i'll pick", "which", "post"))
        if act == "answer_practical_question":
            return any(term in normalized for term in ("7.5", "7", "safer", "ideal", "size"))
        if act == "answer_location_or_status":
            return any(term in normalized for term in ("im in", "i'm in", "northbridge", "sampleford", "home", "back", "here", "rn", "uni"))
        if act == "reciprocate_missing":
            return any(term in normalized for term in ("miss u too", "miss you too", "miss u more", "missed u too", "missed you too"))
        if act == "reciprocate_affection":
            return any(term in normalized for term in ("love u too", "love you too", "i love you too", "love u", "mwah"))
        if act == "return_goodnight":
            return any(term in normalized for term in ("goodnight", "good night", "sleep well", "sleep tight")) or re.search(r"\bgn\b", normalized) is not None
        if act == "answer_greeting":
            if "baby" in burst_blob:
                return any(term in normalized for term in ("hi", "hey", "hello")) and any(term in normalized for term in ("baby", "babe", "babyyy"))
            return any(term in normalized for term in ("hi", "hey", "yo", "hello"))
        if act == "warmly_receive_compliment":
            return any(term in normalized for term in ("thank", "sweet", "love", "baby", "my love", "appreciate"))
        if act == "answer_teasing_playfully":
            return any(term in normalized for term in ("rude", "wow", "allow", "fatty", "okay", "hey you", "😭", "lol", "lmao"))
        if act == "react_to_story_content":
            return any(term in normalized for term in ("never letting", "live that down", "movie", "alex", "teasing", "nah", "lmao", "😭", "funny", "mad"))
        if act == "answer_check_in_or_status":
            return any(term in normalized for term in ("yeah", "yh", "im", "i'm", "back", "here", "good", "okay", "home"))
        if act == "answer_specific_status_check":
            checks: list[bool] = []
            if "northbridge" in burst_blob:
                checks.append(any(term in normalized for term in ("northbridge", "still here", "still there", "still in")))
            if "appointment" in burst_blob:
                checks.append("appointment" in normalized and any(term in normalized for term in ("done", "okay", "fine", "healing", "recovering")))
            if "busy again" in burst_blob:
                checks.append(any(term in normalized for term in ("busy", "caught up", "all that", "running around", "recovering")))
            return bool(checks) and all(checks)
        if act == "reciprocate_kiss_affection":
            return any(term in normalized for term in ("kiss", "mwah", "cant wait", "can't wait", "either", "love u"))
        return bool(normalized.strip())

    def _whatsapp_validate_reply_against_obligation(self, reply: str, obligation: WhatsAppConversationObligation) -> dict[str, object]:
        normalized = normalize_text(reply).strip(" ?!.,")
        burst_blob = normalize_text(" ".join(obligation.latest_incoming_burst))
        units = self._whatsapp_reply_units(reply)
        unit_norms = [
            normalize_text(unit).strip(" ?!.,")
            for unit in units
            if normalize_text(unit).strip(" ?!.,")
        ]
        already_sent_norms: set[str] = set()
        for sent in obligation.already_sent_replies:
            sent_norm = normalize_text(sent).strip(" ?!.,")
            if sent_norm:
                already_sent_norms.add(sent_norm)
            for sent_unit in self._whatsapp_reply_units(str(sent)):
                sent_unit_norm = normalize_text(sent_unit).strip(" ?!.,")
                if sent_unit_norm:
                    already_sent_norms.add(sent_unit_norm)
        missing = [
            act for act in obligation.required_response_acts
            if not self._whatsapp_reply_satisfies_act(reply, act, obligation)
        ]
        violations: list[str] = []
        target = obligation.target_reply_burst_size or {"min": 1, "max": 3}
        if len(units) < int(target.get("min", 1)):
            violations.append("below_target_reply_burst_size")
        duplicate_check = self._whatsapp_duplicate_check(units)
        if not bool(duplicate_check["passed"]):
            violations.append("duplicate_or_near_duplicate_bubbles")
        clear_moves = set(obligation.conversation_move) - {"normal"}
        if clear_moves and normalized in {"what do you mean", "what do u mean", "what u mean", "what you mean", "wdym", "idk what u mean", "why whats wrong", "why what's wrong", "what happened"}:
            violations.append("generic_clarification_when_clear")
        if normalized in {"ok", "okay", "fair", "haha", "calm", "lol", "yh", "yeah", "cool", "nice"}:
            violations.append("generic_filler")
        repeated_units = [
            unit_norm
            for unit_norm in unit_norms
            if unit_norm in already_sent_norms and not self._whatsapp_repeatable_repair_unit(unit_norm)
        ]
        repeated_unit_limit = 1 if len(unit_norms) <= 3 else max(3, int(len(unit_norms) * 0.35))
        if normalized in already_sent_norms or len(repeated_units) >= repeated_unit_limit:
            violations.append("repeat_already_sent_bubble")
        if "emotional_disclosure" in obligation.conversation_move and any(term in normalized for term in ("horny", "kiss", "cock", "sexy", "lick", "trouble")):
            violations.append("sexual_escalation_during_emotional_context")
        if "invite_or_plan" in obligation.conversation_move and "free on sunday" in normalize_text(" ".join(obligation.latest_incoming_burst)) and any(term in normalized for term in ("when free", "what day", "what day you free")):
            violations.append("ask_availability_when_already_given")
        if len(obligation.latest_incoming_burst) > 1 and missing:
            violations.append("reply_only_to_final_word_while_ignoring_burst")
        topics = set(obligation.concrete_topics_detected or [])
        if obligation.repair_required and ("apologize_missing_luck" in obligation.required_response_acts or "acknowledge_reading_failure" in obligation.required_response_acts):
            apology_index = min([i for i, unit in enumerate(units) if any(term in normalize_text(unit) for term in ("sorry", "my bad", "shouldve", "should've"))] or [999])
            reassurance_index = min([i for i, unit in enumerate(units) if any(term in normalize_text(unit) for term in ("passed", "proud", "dont convince", "don't convince", "in shaa allah", "worked hard"))] or [999])
            if reassurance_index < apology_index:
                violations.append("reassurance_before_accountability")
        if "presentation" in topics or "academic_stress" in topics:
            grounded_terms = ("presentation", "marker", "harsh", "feedback", "flashcard", "flashcards", "content", "questions", "luck", "read it properly", "overthinking", "passed", "failed")
            if not any(term in normalized for term in grounded_terms):
                violations.append("generic_ungrounded_academic_reassurance")
        else:
            academic_reply_terms = (
                "exam",
                "exams",
                "revision",
                "revised",
                "presentation",
                "marker",
                "flashcard",
                "flashcards",
                "passed",
                "failed",
                "in shaa allah it went better",
            )
            stale_support_terms = (
                "sounds so stressful",
                "ur panicking",
                "youre panicking",
                "you're panicking",
                "dont decide u failed",
                "don't decide u failed",
                "overthinking it because u care",
                "proud of u for getting through it",
                "you worked hard",
                "ill listen properly",
                "i'll listen properly",
            )
            stale_support_hits = sum(1 for term in stale_support_terms if term in normalized)
            if any(term in normalized for term in academic_reply_terms):
                violations.append("academic_reply_for_nonacademic_burst")
            if obligation.emotional_intensity == "low" and stale_support_hits >= 2:
                violations.append("stale_emotional_support_for_low_intensity_burst")
        if "family_health" in topics and any(term in normalized for term in ("presentation", "marker", "flashcard", "passed", "failed")):
            violations.append("wrong_crisis_domain")
        delayed_or_on_read_apology = (
            any(term in normalized for term in (
                "wasnt ignoring",
                "wasn't ignoring",
                "ignore u",
                "ignore you",
                "ignoring u",
                "ignoring you",
                "leave u on read",
                "leave you on read",
                "left u on read",
                "left you on read",
                "didnt mean to air",
                "didn't mean to air",
                "wasnt airing",
                "wasn't airing",
                "on read",
                "aired",
                "airing",
                "air u",
                "air you",
            ))
            or (
                any(term in normalized for term in ("sorry", "my bad"))
                and any(term in normalized for term in (
                    "reply late",
                    "late reply",
                    "taking long",
                    "took long",
                    "long to reply",
                    "text back",
                    "message back",
                    "caught up",
                    "busy",
                    "ignoring",
                    "ignore",
                    "aired",
                    "airing",
                ))
            )
        )
        conflict_grounded_apology = (
            (
                "conflict_or_hurt" in obligation.conversation_move
                or "relationship_conflict" in set(obligation.concrete_topics_detected or [])
            )
            and any(term in burst_blob for term in (
                "ignored",
                "ignore",
                "reply",
                "text back",
                "message back",
                "on read",
                "aired",
                "airing",
                "on ur phone",
                "on your phone",
                "went on ur phone",
                "went on your phone",
            ))
        )
        if delayed_or_on_read_apology and not (
            {"acknowledge_delayed_response", "acknowledge_on_read_or_airing"} & set(obligation.required_response_acts)
            or conflict_grounded_apology
        ):
            violations.append("ungrounded_delayed_response_apology")
        passed = not missing and not violations
        topic_mismatch = [
            item for item in violations
            if item in {"academic_reply_for_nonacademic_burst", "wrong_crisis_domain", "generic_ungrounded_academic_reassurance", "topic_change_during_emotional_disclosure"}
        ]
        stale_candidate_reason = ""
        if any(item in violations for item in ("academic_reply_for_nonacademic_burst", "stale_emotional_support_for_low_intensity_burst", "similar_to_recent_global_draft")):
            stale_candidate_reason = "candidate appears stale or grounded in old/unrelated context"
        repetition_reason = ""
        if any(item in violations for item in ("duplicate_or_near_duplicate_bubbles", "repeat_already_sent_bubble", "similar_to_recent_thread_draft")):
            repetition_reason = "candidate repeats an already shown/sent draft or repeats bubbles internally"
        feedback_seed = {
            "missing_required_acts": missing,
            "violated_forbidden_acts": list(dict.fromkeys(violations)),
        }
        return {
            "passed": passed,
            "missing_required_acts": missing,
            "violated_forbidden_acts": list(dict.fromkeys(violations)),
            "forbidden_acts_triggered": list(dict.fromkeys(violations)),
            "topic_mismatch": topic_mismatch,
            "stale_candidate_reason": stale_candidate_reason,
            "repetition_reason": repetition_reason,
            "repair_instruction": self._whatsapp_validation_repair_instruction(feedback_seed, obligation) if not passed else "",
            "reply_unit_count": len(units),
            "target_reply_burst_size": target,
            "duplicate_check_result": duplicate_check,
            "validation_result": "pass" if passed else "fail",
        }

    def _whatsapp_obligation_repair_sequences(self, obligation: WhatsAppConversationObligation) -> list[list[str]]:
        acts = set(obligation.required_response_acts)
        burst_blob = normalize_text(" ".join(obligation.latest_incoming_burst))
        topics = set(obligation.concrete_topics_detected or [])
        if "presentation" in topics and obligation.repair_required:
            return [[
                "baby im sorry",
                "i shouldve wished u luck",
                "thats my bad icl",
                "and yeah i shouldve read it properly",
                "you literally explained what happened",
                "i get why ur upset",
                "but dont convince urself u failed already",
                "you said u covered most of the content",
                "and you answered the questions",
                "thats what actually matters",
                "some markers are just cold asf",
                "like they dont give any reassurance",
                "and it makes you feel like you did shit",
                "but that doesnt mean you actually did",
                "the flashcard comment is so petty asw",
                "like who cares if the content was there",
                "youre probably overthinking it because she was harsh",
                "in shaa allah you passed",
                "and even if it felt horrible",
                "im proud of u for getting through it",
                "you worked hard for it",
                "go sleep baby",
                "youll feel less panicked after rest",
                "love u",
            ]]
        if "academic_stress" in topics and obligation.repair_required:
            return [[
                "baby im sorry",
                "i shouldve read that properly",
                "and i shouldve wished u luck",
                "thats my bad",
                "you already explained why ur stressed",
                "i get why that upset u",
                "exams mess with ur head so much",
                "especially when ur exhausted",
                "but dont decide u failed already",
                "you revised for this",
                "you still answered what u could",
                "and feeling awful after doesnt mean it went awful",
                "the panic after is always louder than the actual mark",
                "youre overthinking it because u care",
                "not because youre doomed",
                "in shaa allah it went better than it feels",
                "im proud of u for getting through it",
                "you worked hard",
                "and im sorry i made u feel unsupported",
                "i shouldve been softer with u",
                "go rest for a bit baby",
                "ill actually listen properly",
                "tell me the bits that are still stressing u",
                "love u",
            ]]
        if "presentation" in topics:
            return [[
                "baby that sounds so stressful",
                "i get why ur panicking",
                "but dont decide u failed already",
                "presentations always feel worse after",
                "especially when the marker is cold",
                "you still covered the content",
                "and you answered the questions",
                "thats what matters most",
                "the feedback can feel harsh",
                "but harsh doesnt mean failed",
                "youre overthinking it rn",
                "the flashcard thing sounds petty",
                "like if the content was there thats what matters",
                "in shaa allah you passed",
                "im proud of u for getting through it",
                "try sleep baby",
                "youll feel calmer after rest",
                "love u",
            ]]
        if "academic_stress" in topics:
            return [[
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
            ]]
        if "family_health" in topics:
            return [[
                "baby thats so much to deal with",
                "im sorry",
                "i get why youre scared",
                "thats a lot to hear at once",
                "especially with your mum and your siblings on your mind",
                "you dont have to hold all of that alone",
                "im here",
                "talk to me properly",
                "we can take it bit by bit",
                "just breathe for a second",
                "youre not alone in this",
                "i love u",
            ]]
        if "relationship_conflict" in topics:
            target = obligation.target_reply_burst_size or {}
            if int(target.get("min", 1)) < 20:
                return [
                    [
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
                    ],
                    [
                        "baby im sorry",
                        "i get why that hurt u",
                        "i made u feel alone when u needed me close",
                        "thats on me",
                        "i know after appointment u expected me to be more present",
                        "especially after what i said to u",
                        "i do love u",
                        "i dont want this feeling like back to square one",
                        "i want us feeling like us again too",
                        "i know the space has made it feel one sided",
                        "i dont want everything on my terms",
                        "im not trying to make u chase me for effort",
                        "you deserve to feel chosen by me",
                        "let me fix how ive been moving",
                        "talk to me properly baby",
                        "i dont want to lose u",
                    ],
                    [
                        "my baby im sorry",
                        "i hear what youre saying",
                        "i made it feel one sided again",
                        "and thats on me",
                        "u shouldnt feel like u only get pieces of me",
                        "not after appointment",
                        "not after i said things would feel different",
                        "i love u wallahi",
                        "i care about us so much",
                        "i want it back to how it was too",
                        "but i need to show that properly",
                        "not just say it",
                        "let me talk to u with my full attention",
                        "i dont want us stuck like this",
                    ],
                ]
            return [[
                "baby im so sorry",
                "i hate that i made u feel like this",
                "i hear how confused ive left u",
                "i hear ur anxiety got worse waiting for me",
                "u shouldnt have had to beg me to reply",
                "u didnt do anything to deserve this",
                "thats on me",
                "ive been too closed off",
                "and ive let my own head turn into distance from u",
                "i havent been there recently like u needed",
                "u needed me countless times",
                "and i made u feel like u were reaching alone",
                "thats not love the way u deserve it shown",
                "i know the space has felt endless",
                "i know it started feeling like u had no say",
                "like everything was on my terms",
                "and that was selfish from me",
                "im not saying this to argue with u",
                "im saying it because i need to own it",
                "seeing me answer other people while u felt ignored mustve hurt deep",
                "especially when all u wanted was your own man there",
                "i get why that made u feel unappreciated",
                "i get why in person me being on my phone hurt too",
                "u were trying to connect with me",
                "and i made u feel second",
                "im sorry for making u cry like that",
                "im sorry for draining u when u were loving me",
                "u love so deeply",
                "and i treated that too casually",
                "wallahi i do love u",
                "more than ive been showing",
                "i care about u so much",
                "youre not too much for wanting effort",
                "youre not wrong for wanting consistency",
                "i dont want u thinking youre never loved for u",
                "i love you for you",
                "the soft parts",
                "the angry parts",
                "the anxious parts",
                "all of it",
                "my heart is with u even when ive handled this badly",
                "i hope Allah heals what ive hurt in u",
                "and i know i need to change what caused it",
                "i dont want to lose us",
                "i dont want to hide behind needing space",
                "i may still need to clear my head",
                "but i cant use that to abandon u",
                "i need to talk to u properly",
                "not defensively",
                "not with excuses",
                "i dont want to argue with what u feel",
                "i want to take in every part of what ive done",
                "i want to listen properly this time",
                "and explain myself without making it your fault",
                "if u can give me the chance",
                "ill call when youre ready",
                "and ill actually stay present",
                "because u deserved that before now",
                "i love u baby",
                "im sorry for making us feel this broken",
            ]]
        if "acknowledge_on_read_or_airing" in acts and "acknowledge_work_cover_update" in acts:
            return [
                [
                    "sorry baby i didnt mean to air u",
                    "im okay i promise",
                    "i just got caught up",
                    "and dw about tomorrow",
                    "if ur covering ur mum at work thats calm",
                    "we can sort next week",
                    "just lmk when ur free",
                ],
                [
                    "my bad baby",
                    "i wasnt trying to leave u on read",
                    "im okay",
                    "tomorrow is calm if u need to cover work",
                    "dont stress about meet ups",
                    "we can plan next week properly",
                ],
            ]
        if "acknowledge_work_cover_update" in acts:
            return [["thats calm baby", "cover work tomorrow dw", "we can sort next week"]]
        if {"acknowledge_delayed_response", "invite_them_to_talk"} <= acts:
            return [
                ["my bad baby i didnt mean to leave u waiting", "tell me when youre ready", "im here"],
                ["sorry baby i wasnt ignoring u", "tell me properly when youre ready", "im listening"],
            ]
        if {"acknowledge_delayed_response", "answer_availability"} <= acts:
            return [
                ["my bad baby i didnt mean to leave u waiting", "im free later tonight"],
                ["sorry baby i wasnt ignoring u", "im free later tonight if u still need me"],
            ]
        if {"acknowledge_delayed_response", "answer_check_in_or_status"} <= acts:
            return [
                ["my bad baby", "i didnt mean to disappear", "i love u", "im still in northbridge rn", "appointments done im okay", "just been caught up with all that", "im here now"],
                ["sorry baby", "i wasnt ignoring u", "i love u", "im still in northbridge", "appointments done and im okay", "just been busy with it all", "im here now"],
            ]
        if "acknowledge_on_read_or_airing" in acts and "provide_emotional_support" in acts:
            return [[
                "my bad baby",
                "i shouldnt have left u feeling ignored",
                "i get why that hurt",
                "i wasnt trying to make u feel stupid for asking",
                "im not ignoring u",
                "im here now",
            ]]
        if "provide_emotional_support" in acts:
            return [["baby thats so much to deal with", "im sorry", "i get why youre scared", "talk to me im here"]]
        if {"reciprocate_affection", "reciprocate_missing", "return_goodnight"} <= acts:
            return [["love u too", "miss u more", "goodnight baby"], ["love u too baby", "miss u more", "sleep well"], ["i love u too", "miss u loads", "sleep well baby"]]
        if {"reciprocate_affection", "reciprocate_kiss_affection", "return_goodnight"} <= acts:
            return [["love u too baby", "mwah", "cant wait either", "goodnight baby"], ["love u too", "mwah", "goodnight baby", "cant wait either"]]
        if {"reciprocate_affection", "return_goodnight"} <= acts:
            return [["love u too", "goodnight baby"], ["i love you too", "sleep well baby"], ["love u more", "sleep good baby"]]
        if "return_holiday_greeting" in acts:
            return [["eid mubarak baby"]]
        if {"acknowledge_apology_or_repair", "answer_availability"} <= acts:
            return [["its okay baby dw", "friday works for me"], ["dont worry baby", "friday is calm"]]
        if "acknowledge_apology_or_repair" in acts:
            return [["its okay baby dw"]]
        if "respond_to_sexual_flirt_safely" in acts:
            return [["youre actually trouble icl", "behave", "ice cream yeah?", "dont start with me", "you know id want that"], ["dont start with me", "ice cream is crazy", "you know id want that", "behave"]]
        if "ask_to_see_options_or_pick" in acts:
            return [["send me them ill pick"], ["show me them", "ill choose"]]
        if "invite_them_to_talk" in acts:
            if "later" in burst_blob or "ready" in burst_blob or "lot on my mind" in burst_blob:
                return [["okay baby tell me when youre ready", "im here"]]
            return [["tell me baby im listening"]]
        if "acknowledge_unavailable_and_suggest" in acts:
            return [["thats fine", "what about sunday"]]
        if "answer_invite_directly" in acts:
            return [["yeah im down", "what time"], ["im down", "what time u thinking"]]
        if "answer_back_status_and_availability" in acts:
            return [["yeah im back", "sunday works"], ["yeah im here", "sunday is calm"]]
        if "answer_availability" in acts:
            return [["im free later tonight"], ["friday works for me"], ["yeah im free then"]]
        if "answer_location_or_status" in acts and "reciprocate_missing" in acts:
            return [["miss u too", "im in northbridge rn", "where are u baby"]]
        if "answer_location_or_status" in acts:
            return [["im in northbridge rn"]]
        if "react_to_story_content" in acts:
            return [["nah theyre never letting you live that down"]]
        if "answer_teasing_playfully" in acts:
            return [["wow okay rude", "hey you", "fatty is crazy", "come say that to me", "i miss u still"], ["hey fatty is crazy", "ur rude", "i missed u tho"]]
        if "answer_greeting" in acts and "baby" in burst_blob:
            return [["hi babyyy", "you okay"], ["hey baby", "u good"]]
        if "answer_greeting" in acts:
            return [["hrellaaa", "heyyy"], ["heyyy", "u good"]]
        if "answer_practical_question" in acts:
            return [["7.5 is safer"], ["7.5 probably", "safer than 7"]]
        if "reciprocate_missing" in acts:
            return [["miss u too"], ["miss u more"], ["i miss u too baby"]]
        if "reciprocate_affection" in acts:
            return [["love u too"], ["i love u too"], ["love u more"]]
        if "return_goodnight" in acts:
            return [["goodnight baby"], ["sleep well baby"], ["goodnight my love"]]
        if any(topic in topics for topic in ("food", "rain", "banter")):
            return [
                ["yeah it pissed me off icl", "rained like 4 times", "at least the food was done tho"],
                ["nah fr it annoyed me icl", "it kept raining", "food being done saved it tho"],
                ["yeah i was vexed", "rain came down bare times", "but at least u had food ready"],
            ]
        if "normal_romantic" in obligation.conversation_move:
            return [["yeah baby", "i hear u", "come here", "tell me properly", "miss u"], ["come here", "i hear u", "talk to me properly", "miss u still", "im listening"]]
        return [["yeah i get u"]]

    def _whatsapp_last_resort_obligation_sequence(
        self,
        obligation: WhatsAppConversationObligation,
        *,
        request_id: str = "",
    ) -> list[str]:
        burst_blob = normalize_text(" ".join(obligation.latest_incoming_burst))
        topics = set(obligation.concrete_topics_detected or [])
        if "relationship_conflict" in topics:
            variants = [
                [
                    "my love im sorry",
                    "i shouldnt have left u feeling alone with all of that",
                    "i get why it feels like ive pulled us backwards",
                    "i promised u things would feel better after the appointment",
                    "and then i still made u feel like u didnt have me",
                    "thats not fair on u",
                    "i love u properly",
                    "i dont want this to feel like u are begging for me",
                    "i know the space has felt one sided",
                    "i dont want it on my terms only",
                    "i know ive been distant",
                    "i know it hurt more because u needed me close",
                    "im here now",
                    "i need to talk to u properly",
                    "let me fix how ive made u feel",
                ],
                [
                    "baby i hear u",
                    "and im sorry for making u feel like we are back to square one",
                    "i know u wanted me to be different with u",
                    "especially after everything with the appointment",
                    "i made u feel like u still didnt have me to yourself",
                    "that would hurt me too",
                    "im not gonna act like u are dramatic for saying it",
                    "i love u",
                    "i care about us",
                    "i dont want to lose what we had",
                    "i know the space has made it feel one sided",
                    "i dont want everything on my terms",
                    "i need to stop making u chase basic reassurance",
                    "i need to actually show u im here",
                    "so talk to me properly",
                    "im listening now",
                ],
                [
                    "im sorry baby",
                    "i know this feels like me slipping back into the same pattern",
                    "u needed me close and i made u feel like u were on ur own again",
                    "especially after the appointment and everything i said would change",
                    "thats on me",
                    "i love u",
                    "i dont want u feeling like u only get pieces of me",
                    "i know the space has made it feel one sided",
                    "i dont want this relationship to be on my terms",
                    "i want it to feel safe for u too",
                    "i need to be more present",
                    "i need to reassure u properly",
                    "come talk to me",
                    "i dont want us ending the night like this",
                ],
                [
                    "my baby im sorry",
                    "i get why ur hurt",
                    "i made it feel like nothing actually changed",
                    "u were waiting for us to feel how we used to",
                    "and after the appointment i shouldve made u feel secure",
                    "instead i made u feel like u still had to fight for me",
                    "thats not okay",
                    "thats on me",
                    "i love u",
                    "i care about this more than ive shown",
                    "the space shouldnt feel like me pushing u away",
                    "i dont want everything on my terms",
                    "i want to show u im choosing u",
                    "let me speak to u properly",
                    "im not letting this just sit",
                ],
                [
                    "baby im sorry for disappearing like that",
                    "i know it hit harder because this already felt fragile",
                    "i know u wanted everything to go back to how it used to be",
                    "and i made it feel like back to square one",
                    "especially after the appointment",
                    "i shouldve made u feel wanted and close to me",
                    "not like u were asking for too much",
                    "thats on me",
                    "i love u",
                    "i dont want this one sided feeling between us",
                    "i dont want it to be my terms only",
                    "i need to stop making u doubt where u stand with me",
                    "i need to talk to u properly",
                    "im here",
                    "dont go to sleep feeling alone with it",
                ],
            ]
            if "appointment" not in burst_blob:
                for variant in variants:
                    variant[:] = [part for part in variant if "appointment" not in normalize_text(part)]
            if "back to square one" not in burst_blob and "how it used to be" not in burst_blob:
                for variant in variants:
                    variant[:] = [part for part in variant if "back to square one" not in normalize_text(part) and "what we had" not in normalize_text(part)]
            min_units = int((obligation.target_reply_burst_size or {}).get("min", 1) or 1)
            valid_variants = [variant for variant in variants if len(variant) >= min_units]
            if not valid_variants:
                valid_variants = variants
            seed_material = "|".join([request_id, str(len(obligation.latest_incoming_burst)), burst_blob[-240:]])
            base_index = int(hashlib.sha1(seed_material.encode("utf-8")).hexdigest()[:8], 16)
            trailing_number = re.search(r"(\d+)(?!.*\d)", request_id or "")
            if trailing_number:
                base_index += int(trailing_number.group(1))
            index = base_index % len(valid_variants)
            return valid_variants[index]
        phrases_by_act: dict[str, list[str]] = {
            "return_holiday_greeting": ["eid mubarak baby"],
            "acknowledge_apology_or_repair": ["its okay baby dw"],
            "acknowledge_on_read_or_airing": ["my bad baby i didnt mean to leave u on read"],
            "acknowledge_delayed_response": ["my bad baby i wasnt ignoring u"],
            "acknowledge_reading_failure": ["i shouldve read it properly"],
            "apologize_missing_luck": ["sorry i didnt wish u luck"],
            "ground_academic_reassurance": ["you worked hard and in shaa allah it went better than it feels"],
            "acknowledge_specific_academic_details": ["the marker and flashcards clearly got in ur head"],
            "acknowledge_relationship_hurt": ["i hear how hurt and drained ive made u feel"],
            "take_accountability_for_distance": ["im sorry thats on me"],
            "reassure_care_without_defensiveness": ["i love u and i dont want to lose us"],
            "invite_them_to_talk": ["talk to me properly im listening"],
            "acknowledge_work_cover_update": ["covering work tomorrow is calm", "we can sort next week"],
            "respond_to_sexual_flirt_safely": ["youre trouble icl", "behave"],
            "acknowledge_unavailable_and_suggest": ["thats fine", "what about sunday"],
            "answer_invite_directly": ["yeah im down", "what time"],
            "answer_back_status_and_availability": ["yeah im back", "sunday works"],
            "answer_availability": ["im free later tonight"],
            "ask_to_see_options_or_pick": ["send me them ill pick"],
            "answer_practical_question": ["7.5 is safer"],
            "answer_location_or_status": ["im in northbridge rn"],
            "reciprocate_missing": ["miss u too baby"],
            "reciprocate_affection": ["love u too baby"],
            "return_goodnight": ["goodnight baby"],
            "answer_greeting": ["hey baby" if "baby" in burst_blob else "heyyy"],
            "warmly_receive_compliment": ["thank u baby thats sweet"],
            "answer_teasing_playfully": ["wow okay rude"],
            "react_to_story_content": ["nah theyre never letting you live that down"],
            "answer_check_in_or_status": ["im okay baby"],
            "reciprocate_kiss_affection": ["mwah love u"],
        }
        if "academic_stress" in topics or "presentation" in topics:
            phrases_by_act["provide_emotional_support"] = ["im sorry i get why youre overthinking"]
        elif "relationship_conflict" in topics:
            phrases_by_act["provide_emotional_support"] = ["im sorry i get why that hurt"]
        else:
            phrases_by_act["provide_emotional_support"] = ["im sorry i get why thats a lot"]
        specific_status: list[str] = []
        if "northbridge" in burst_blob:
            specific_status.append("im still in northbridge rn")
        if "appointment" in burst_blob:
            specific_status.append("appointments done im okay")
        if "busy again" in burst_blob:
            specific_status.append("ive just been caught up and busy")
        if specific_status:
            phrases_by_act["answer_specific_status_check"] = specific_status

        sequence: list[str] = []
        seen: set[str] = set()
        for act in obligation.required_response_acts:
            for phrase in phrases_by_act.get(act, []):
                normalized = normalize_text(phrase).strip(" ?!.,")
                if normalized and normalized not in seen:
                    sequence.append(phrase)
                    seen.add(normalized)
        if not sequence:
            sequence = ["yeah i hear u"]
            seen.add("yeah i hear u")

        min_units = int((obligation.target_reply_burst_size or {}).get("min", 1) or 1)
        if len(sequence) < min_units:
            pads = [
                "i hear u",
                "im listening",
                "tell me properly",
                "im with u",
                "i get what u mean",
                "i dont wanna ignore that",
                "im taking it in",
                "talk to me",
                "i care",
                "baby im here",
            ]
            index = 0
            while len(sequence) < min_units:
                phrase = pads[index % len(pads)]
                if index >= len(pads):
                    phrase = f"{phrase} properly"
                normalized = normalize_text(phrase).strip(" ?!.,")
                if normalized not in seen:
                    sequence.append(phrase)
                    seen.add(normalized)
                index += 1
        return sequence

    def _whatsapp_candidate_payload(
        self,
        sequence: list[str],
        *,
        obligation: WhatsAppConversationObligation,
        relationship: str,
        intent_type: str,
        candidate_kind: str = "contextual_fallback",
        fallback_reason: str = "conversation_obligation_repair",
    ) -> dict[str, object]:
        text = " / ".join(part for part in sequence if str(part).strip())
        validation = self._whatsapp_validate_reply_against_obligation(text, obligation)
        return {
            "text": text,
            "sequence": sequence,
            "candidate_type": "whatsapp_obligation_repair",
            "candidate_kind": candidate_kind,
            "candidate_source_label": "contextual_fallback" if candidate_kind == "contextual_fallback" else candidate_kind,
            "fallback_reason": fallback_reason,
            "relationship_type": relationship,
            "intent": intent_type,
            "final_decision": "review",
            "auto_send_allowed": False,
            "provider": "conversation_obligation",
            "model": "local_obligation_validator",
            "why_this_matches": "Generated from WhatsApp conversation obligation required_response_acts.",
            "whatsapp_obligation": obligation.to_debug_dict(),
            "whatsapp_obligation_validation": validation,
        }

    def _whatsapp_varied_candidate(
        self,
        candidates: list[dict[str, object]],
        *,
        request_id: str,
        obligation: WhatsAppConversationObligation,
    ) -> dict[str, object]:
        if not candidates:
            return {}
        if len(candidates) == 1:
            return candidates[0]
        seed_material = "|".join([
            str(request_id or ""),
            str(len(obligation.latest_incoming_burst)),
            " ".join(obligation.latest_incoming_burst[-3:]),
        ])
        index = int(hashlib.sha1(seed_material.encode("utf-8")).hexdigest()[:8], 16) % len(candidates)
        return candidates[index]

    def whatsapp_web_draft(self, request: WhatsAppWebDraftRequest) -> dict[str, object]:
        cleaned_messages = [
            {
                "speaker": message.speaker,
                "text": str(message.text or "").strip(),
                "timestamp": message.timestamp,
                "client_order": message.client_order,
                "message_id": message.message_id,
            }
            for message in request.messages
            if str(message.text or "").strip()
        ]
        cleaned_messages = self._whatsapp_order_messages(cleaned_messages)
        if not cleaned_messages:
            raise HTTPException(status_code=400, detail="messages must include at least one non-empty WhatsApp message.")

        request_id = str(request.request_id or f"wa_req_{uuid.uuid4().hex[:16]}")
        state_key = self._whatsapp_state_key(contact_name=request.contact_name, thread_id=request.thread_id)
        recent_thread_drafts = list(self._whatsapp_recent_thread_drafts.get(state_key, []))
        recent_global_drafts = list(self._whatsapp_recent_global_drafts)
        recent_thread_texts = [str(item.get("text") or "") for item in recent_thread_drafts if str(item.get("text") or "").strip()]
        relationship, relationship_inferred = self._infer_whatsapp_web_relationship(cleaned_messages, request.relationship_type)
        if request.mode == "auto-send" and cleaned_messages[-1]["speaker"] == "me":
            memory_rows = self._whatsapp_web_memory_rows(request.contact_name, limit=6)
            questions = self._whatsapp_web_context_questions(
                contact_name=request.contact_name,
                relationship_type=relationship,
                collected_count=len(cleaned_messages),
                memory_count=len(memory_rows),
            )
            return {
                "source": "whatsapp_web",
                "request_id": request_id,
                "thread_id": request.thread_id or "",
                "state_key": state_key,
                "mode": request.mode,
                "contact_name": request.contact_name or "",
                "incoming": "",
                "context": [str(message["text"]) for message in cleaned_messages if str(message["text"]).strip()],
                "message_context_count": len(cleaned_messages),
                "saved_memory_context_count": len(memory_rows),
                "collected_message_count": len(cleaned_messages),
                "context_target_count": 220,
                "context_minimum_met": len(cleaned_messages) >= 40,
                "questions": questions,
                "saved_memory": [
                    {
                        "id": str(row.get("id") or ""),
                        "question": str(row.get("question") or ""),
                        "answer": str(row.get("answer") or ""),
                        "created_at": str(row.get("created_at") or ""),
                    }
                    for row in memory_rows
                ],
                "relationship_type": relationship,
                "relationship_inferred_from_thread": relationship_inferred,
                "intent_type": request.intent_type,
                "reply": "",
                "reply_sequence": [],
                "recent_drafts_for_this_thread": recent_thread_drafts[-6:],
                "draft_cache_hit": False,
                "cache_used": False,
                "candidate": {},
                "automation_decision": "REVIEW_REQUIRED",
                "auto_send_allowed": False,
                "auto_send_blocked_reason": "latest_message_from_self",
                "insert_allowed": False,
                "warnings": [
                    "Auto-send only sends when the latest visible WhatsApp message is from the other person.",
                ],
                "training_response": {},
            }

        incoming_index = -1
        for index in range(len(cleaned_messages) - 1, -1, -1):
            if cleaned_messages[index]["speaker"] != "me":
                incoming_index = index
                break
        if incoming_index < 0:
            incoming_index = len(cleaned_messages) - 1

        post_incoming_messages = cleaned_messages[incoming_index + 1 :]
        post_incoming_reply_texts = [
            str(message["text"])
            for message in post_incoming_messages
            if message["speaker"] == "me" and str(message["text"]).strip()
        ]
        obligation = self._whatsapp_conversation_obligation(
            cleaned_messages,
            incoming_index,
            relationship=relationship,
            already_sent_replies=[*request.avoid_candidates, *post_incoming_reply_texts, *recent_thread_texts],
        )
        burst_count = max(1, len(obligation.latest_incoming_burst))
        burst_start_index = max(0, incoming_index - burst_count + 1)
        incoming = "\n".join(obligation.latest_incoming_burst) or str(cleaned_messages[incoming_index]["text"])
        context_messages = cleaned_messages[:burst_start_index]
        message_context = [str(message["text"]) for message in context_messages if str(message["text"]).strip()]
        post_incoming_context = [
            f"Already sent after latest incoming - {reply_text}"
            for reply_text in post_incoming_reply_texts
        ]
        saved_memory_context = self._whatsapp_web_memory_context(request.contact_name)
        time_context = self._whatsapp_web_time_context(cleaned_messages, incoming_index)
        context = [*saved_memory_context, *time_context, *message_context, *post_incoming_context]
        training_request = TrainingChatRequest(
            incoming=incoming,
            context=context,
            contact_name=request.contact_name,
            relationship_type=relationship,
            intent_type=request.intent_type,
            diversity_mode=request.diversity_mode,
            avoid_candidates=[*request.avoid_candidates, *post_incoming_reply_texts, *recent_thread_texts],
            parameters=request.parameters,
        )
        provider_failure_reason = ""
        try:
            if self.drafting.ai_reply_enabled and request.diversity_mode == "natural" and len(cleaned_messages) >= 4:
                full_conversation = [
                    f"[{'ME' if message.get('speaker') == 'me' else 'OTHER'}]: {str(message.get('text') or '').strip()}"
                    for message in cleaned_messages
                    if str(message.get("text") or "").strip()
                ]
                bundle = self.drafting.build_bundle_with_context(
                    recent_messages=[*message_context[-8:], incoming],
                    full_conversation=full_conversation,
                    contact_name=request.contact_name,
                    relationship_type=relationship,
                )
                self._apply_training_parameters(bundle, request.parameters.model_dump(mode="json"), relationship)
                response = self._training_response(training_request, bundle)
            else:
                response = self.training_chat(training_request)
        except Exception as exc:
            provider_failure_reason = str(exc)
            response = {
                "selected_candidate": "",
                "candidates": [],
                "intent_type": request.intent_type,
                "provider_error": provider_failure_reason,
            }
        candidates = response.get("candidates") if isinstance(response.get("candidates"), list) else []
        selected_reply = str(response.get("selected_candidate") or "")
        selected: dict[str, object] = {}
        valid_candidates: list[dict[str, object]] = []
        enriched_candidates: list[dict[str, object]] = []
        repair_attempts: list[dict[str, object]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate = dict(candidate)
            candidate.setdefault("candidate_source_label", self._whatsapp_candidate_source_label(candidate))
            sequence_raw = candidate.get("sequence")
            sequence = [str(item) for item in sequence_raw] if isinstance(sequence_raw, list) and sequence_raw else [str(candidate.get("text") or "")]
            candidate_text = str(candidate.get("text") or " / ".join(part for part in sequence if part.strip()))
            validation_text = " / ".join(part for part in sequence if part.strip()) if len(sequence) > 1 else candidate_text
            validation = self._whatsapp_validate_reply_against_obligation(validation_text, obligation)
            validation = self._whatsapp_apply_recent_draft_validation(
                validation,
                validation_text,
                latest_burst_topics=obligation.concrete_topics_detected or [],
                recent_thread_drafts=recent_thread_drafts,
                recent_global_drafts=recent_global_drafts,
            )
            validation = self._whatsapp_disable_fallback_candidate_validation(candidate, validation)
            validation = self._whatsapp_relax_fresh_model_burst_size_validation(candidate, validation, obligation)
            candidate["sequence"] = [part for part in sequence if part.strip()]
            candidate["whatsapp_obligation"] = obligation.to_debug_dict()
            candidate["whatsapp_obligation_validation"] = validation
            enriched_candidates.append(candidate)
            if bool(validation["passed"]):
                valid_candidates.append(candidate)
            if selected_reply and str(candidate.get("text") or "") == selected_reply:
                selected = candidate
            elif not selected:
                selected = candidate
        repair_attempts.append({
            "stage": "candidate_batch_1",
            "candidate_count": len(enriched_candidates),
            "valid_count": len(valid_candidates),
            "repair_instruction": self._whatsapp_validation_repair_instruction(
                next(
                    (
                        candidate.get("whatsapp_obligation_validation", {})
                        for candidate in enriched_candidates
                        if isinstance(candidate.get("whatsapp_obligation_validation"), dict)
                        and not bool(candidate.get("whatsapp_obligation_validation", {}).get("passed"))
                    ),
                    {"missing_required_acts": [], "violated_forbidden_acts": []},
                ),
                obligation,
            ),
        })
        selected_validation_existing = selected.get("whatsapp_obligation_validation") if isinstance(selected.get("whatsapp_obligation_validation"), dict) else {}
        if selected and bool(selected_validation_existing.get("passed")):
            pass
        elif valid_candidates:
            selected = valid_candidates[0]
        else:
            selected = {}
            first_failed_validation = next(
                (
                    candidate.get("whatsapp_obligation_validation", {})
                    for candidate in enriched_candidates
                    if isinstance(candidate.get("whatsapp_obligation_validation"), dict)
                    and not bool(candidate.get("whatsapp_obligation_validation", {}).get("passed"))
                ),
                {"missing_required_acts": [], "violated_forbidden_acts": ["model_generation_failed"] if provider_failure_reason else []},
            )
            repair_instruction = self._whatsapp_validation_repair_instruction(first_failed_validation, obligation)
            repaired_candidates: list[dict[str, object]] = []
            if not provider_failure_reason:
                repaired_context = [*context, repair_instruction]
                repaired_avoid = [
                    *request.avoid_candidates,
                    *post_incoming_reply_texts,
                    *recent_thread_texts,
                    *[str(candidate.get("text") or "") for candidate in enriched_candidates if str(candidate.get("text") or "").strip()],
                ]
                repaired_request = TrainingChatRequest(
                    incoming=incoming,
                    context=repaired_context,
                    contact_name=request.contact_name,
                    relationship_type=relationship,
                    intent_type=request.intent_type,
                    diversity_mode=request.diversity_mode,
                    avoid_candidates=repaired_avoid,
                    parameters=request.parameters,
                )
                try:
                    repaired_response = self.training_chat(repaired_request)
                except Exception as exc:
                    provider_failure_reason = str(exc)
                    repaired_response = {"selected_candidate": "", "candidates": [], "provider_error": provider_failure_reason}
                repaired_raw_candidates = repaired_response.get("candidates") if isinstance(repaired_response.get("candidates"), list) else []
                for candidate in repaired_raw_candidates:
                    if not isinstance(candidate, dict):
                        continue
                    candidate = dict(candidate)
                    candidate["candidate_kind"] = "repaired_model"
                    candidate["candidate_source_label"] = "repaired_model"
                    sequence_raw = candidate.get("sequence")
                    sequence = [str(item) for item in sequence_raw] if isinstance(sequence_raw, list) and sequence_raw else [str(candidate.get("text") or "")]
                    candidate_text = str(candidate.get("text") or " / ".join(part for part in sequence if part.strip()))
                    validation_text = " / ".join(part for part in sequence if part.strip()) if len(sequence) > 1 else candidate_text
                    validation = self._whatsapp_validate_reply_against_obligation(validation_text, obligation)
                    validation = self._whatsapp_apply_recent_draft_validation(
                        validation,
                        validation_text,
                        latest_burst_topics=obligation.concrete_topics_detected or [],
                        recent_thread_drafts=recent_thread_drafts,
                        recent_global_drafts=recent_global_drafts,
                    )
                    validation = self._whatsapp_disable_fallback_candidate_validation(candidate, validation)
                    validation = self._whatsapp_relax_fresh_model_burst_size_validation(candidate, validation, obligation)
                    candidate["sequence"] = [part for part in sequence if part.strip()]
                    candidate["whatsapp_obligation"] = obligation.to_debug_dict()
                    candidate["whatsapp_obligation_validation"] = validation
                    repaired_candidates.append(candidate)
            valid_repaired = [
                candidate
                for candidate in repaired_candidates
                if bool(candidate.get("whatsapp_obligation_validation", {}).get("passed"))
            ]
            repair_attempts.append({
                "stage": "candidate_batch_2",
                "candidate_count": len(repaired_candidates),
                "valid_count": len(valid_repaired),
                "repair_instruction": repair_instruction,
                "provider_error": provider_failure_reason,
            })
            if valid_repaired:
                selected = self._whatsapp_varied_candidate(valid_repaired, request_id=request_id, obligation=obligation)
            validator_repair_candidates: list[dict[str, object]] = []
            validator_repair_disabled_for_topics = any(
                topic in set(obligation.concrete_topics_detected or [])
                for topic in ("relationship_conflict",)
            )
            has_real_generation_candidate = any(
                self._whatsapp_candidate_source_label(candidate) != "test_fixture"
                and not self._whatsapp_candidate_is_disabled_fallback(candidate)
                for candidate in enriched_candidates
                if isinstance(candidate, dict)
            )
            if not selected and has_real_generation_candidate and not validator_repair_disabled_for_topics:
                for sequence in self._whatsapp_obligation_repair_sequences(obligation):
                    candidate = self._whatsapp_candidate_payload(
                        sequence,
                        obligation=obligation,
                        relationship=relationship,
                        intent_type=request.intent_type,
                        candidate_kind="validator_repair",
                        fallback_reason="",
                    )
                    candidate["candidate_source_label"] = "validator_repair"
                    candidate["provider"] = "local_obligation_validator"
                    candidate["auto_send_allowed"] = False
                    validation_text = str(candidate.get("text") or "")
                    candidate["whatsapp_obligation_validation"] = self._whatsapp_apply_recent_draft_validation(
                        candidate.get("whatsapp_obligation_validation", {}) if isinstance(candidate.get("whatsapp_obligation_validation"), dict) else {},
                        validation_text,
                        latest_burst_topics=obligation.concrete_topics_detected or [],
                        recent_thread_drafts=recent_thread_drafts,
                        recent_global_drafts=recent_global_drafts,
                    )
                    validator_repair_candidates.append(candidate)
            valid_validator_repairs = [
                candidate
                for candidate in validator_repair_candidates
                if bool(candidate.get("whatsapp_obligation_validation", {}).get("passed"))
            ]
            repair_attempts.append({
                "stage": "validator_repair",
                "candidate_count": len(validator_repair_candidates),
                "valid_count": len(valid_validator_repairs),
                "selected_for_review": bool(valid_validator_repairs),
                "fallback_reason": "disabled_for_relationship_conflict" if validator_repair_disabled_for_topics else "",
            })
            if valid_validator_repairs:
                selected = self._whatsapp_varied_candidate(valid_validator_repairs, request_id=request_id, obligation=obligation)
            contextual_fallback_candidates: list[dict[str, object]] = []
            for candidate in contextual_fallback_candidates:
                validation_text = str(candidate.get("text") or "")
                candidate["whatsapp_obligation_validation"] = self._whatsapp_apply_recent_draft_validation(
                    candidate.get("whatsapp_obligation_validation", {}) if isinstance(candidate.get("whatsapp_obligation_validation"), dict) else {},
                    validation_text,
                    latest_burst_topics=obligation.concrete_topics_detected or [],
                    recent_thread_drafts=recent_thread_drafts,
                    recent_global_drafts=recent_global_drafts,
                )
            valid_fallbacks = [
                candidate
                for candidate in contextual_fallback_candidates
                if bool(candidate.get("whatsapp_obligation_validation", {}).get("passed"))
            ]
            repair_attempts.append({
                "stage": "contextual_fallback",
                "candidate_count": len(contextual_fallback_candidates),
                "valid_count": len(valid_fallbacks),
                "fallback_reason": "disabled_for_whatsapp_web",
            })
            if valid_fallbacks:
                selected = self._whatsapp_varied_candidate(valid_fallbacks, request_id=request_id, obligation=obligation)
            if not selected:
                repair_attempts.append({
                    "stage": "obligation_last_resort",
                    "candidate_count": 0,
                    "valid_count": 0,
                    "fallback_reason": "disabled_for_whatsapp_web",
                    "selected_for_review": False,
                })
            enriched_candidates = [*contextual_fallback_candidates, *validator_repair_candidates, *repaired_candidates, *enriched_candidates]
        reply = str(selected.get("text") or "")
        selected_sequence_raw = selected.get("sequence")
        selected_sequence = [
            str(item).strip()
            for item in selected_sequence_raw
            if str(item).strip()
        ] if isinstance(selected_sequence_raw, list) else []
        if not selected_sequence and reply:
            selected_sequence = self._whatsapp_reply_units(reply) or [reply]
        selected["sequence"] = selected_sequence
        reply = " / ".join(selected_sequence) if selected_sequence else reply
        selected["text"] = reply
        selected_validation = selected.get("whatsapp_obligation_validation") if isinstance(selected.get("whatsapp_obligation_validation"), dict) else {}
        if not selected_validation:
            selected_validation = self._whatsapp_validate_reply_against_obligation(reply, obligation)
            selected["whatsapp_obligation_validation"] = selected_validation
        obligation.validation_result = selected_validation if isinstance(selected_validation, dict) else {}
        obligation.reply_units = selected_sequence
        obligation.unique_response_acts = self._whatsapp_response_acts_present(reply, obligation)
        obligation.duplicate_check_result = self._whatsapp_duplicate_check(selected_sequence)
        candidate_explicitly_send_allowed = bool(selected.get("auto_send_allowed")) or self._whatsapp_fresh_model_low_intensity_send_allowed(
            selected,
            selected_validation,
            obligation,
        )
        preliminary_auto_send_allowed = (
            candidate_explicitly_send_allowed
            and bool(selected_validation.get("passed", True))
            and len(selected_sequence) <= 14
        )
        obligation.send_queue_safety_result = self._whatsapp_send_queue_safety(
            selected_sequence,
            auto_send_allowed=preliminary_auto_send_allowed,
        )
        obligation.auto_send_allowed = (
            preliminary_auto_send_allowed
            and bool(obligation.duplicate_check_result.get("passed"))
            and bool(obligation.send_queue_safety_result.get("passed"))
        )
        selected["whatsapp_obligation"] = obligation.to_debug_dict()
        selected["send_queue_safety_result"] = obligation.send_queue_safety_result
        selected["duplicate_check_result"] = obligation.duplicate_check_result
        selected["auto_send_allowed"] = candidate_explicitly_send_allowed and obligation.auto_send_allowed
        response["selected_candidate"] = reply
        response["candidates"] = enriched_candidates
        response["repair_attempts"] = repair_attempts
        candidate_allows_send = bool(selected.get("auto_send_allowed")) and obligation.auto_send_allowed
        relationship_allows_send = relationship in {"close_friend", "casual_friend"}
        automation_decision = "REVIEW_REQUIRED"
        if request.mode == "auto-review":
            automation_decision = "SAFE_TO_DRAFT"
        elif request.mode == "auto-send" and candidate_allows_send and relationship_allows_send and bool(selected_validation.get("passed", True)):
            automation_decision = "SEND_ALLOWED"
        elif request.mode == "autonomous" and candidate_allows_send and bool(selected_validation.get("passed", True)):
            automation_decision = "SEND_ALLOWED"
        auto_send_blocked_reason = ""
        if not reply:
            automation_decision = "REVIEW_REQUIRED"
            auto_send_blocked_reason = "model_generation_failed" if provider_failure_reason else "conversation_obligation_failed"
        if request.mode in {"auto-send", "autonomous"} and automation_decision != "SEND_ALLOWED":
            if not bool(selected_validation.get("passed", True)):
                auto_send_blocked_reason = "conversation_obligation_failed"
            elif len(selected_sequence) > 14:
                auto_send_blocked_reason = "high_burst_requires_review"
            elif not bool(obligation.send_queue_safety_result.get("passed", True)):
                auto_send_blocked_reason = "send_queue_safety_failed"
            elif request.mode == "autonomous" and obligation.repair_required:
                auto_send_blocked_reason = "autonomous_repair_requires_review"
            elif request.mode == "autonomous" and obligation.emotional_intensity == "high":
                auto_send_blocked_reason = "autonomous_high_intensity_requires_review"
            elif request.mode == "autonomous":
                auto_send_blocked_reason = "candidate_not_auto_send_safe"
            else:
                auto_send_blocked_reason = "relationship_not_auto_send_safe" if not relationship_allows_send else "candidate_not_auto_send_safe"
        selected_reply_topics = self._whatsapp_reply_topics(reply)
        selected_similarity = self._whatsapp_apply_recent_draft_validation(
            {"passed": True, "violated_forbidden_acts": [], "validation_result": "pass"},
            reply,
            latest_burst_topics=obligation.concrete_topics_detected or [],
            recent_thread_drafts=recent_thread_drafts,
            recent_global_drafts=recent_global_drafts,
        ).get("selected_candidate_similarity_to_recent", {"thread": 0.0, "global": 0.0})
        candidate_sources = [
            self._whatsapp_candidate_source_label(candidate)
            for candidate in enriched_candidates
            if isinstance(candidate, dict)
        ]
        candidates_rejected = [
            {
                "source": self._whatsapp_candidate_source_label(candidate),
                "reasons": candidate.get("whatsapp_obligation_validation", {}).get("violated_forbidden_acts", [])
                + candidate.get("whatsapp_obligation_validation", {}).get("missing_required_acts", []),
            }
            for candidate in enriched_candidates
            if isinstance(candidate, dict)
            and isinstance(candidate.get("whatsapp_obligation_validation"), dict)
            and not bool(candidate.get("whatsapp_obligation_validation", {}).get("passed"))
        ]
        has_selected_reply = bool(reply)
        selected_candidate_source = str(selected.get("candidate_kind") or selected.get("candidate_type") or selected.get("provider") or "none") if has_selected_reply else "none"
        selected_candidate_source_label = self._whatsapp_candidate_source_label(selected) if has_selected_reply else "none"
        fallback_used = has_selected_reply and (
            selected_candidate_source_label in {"contextual_fallback", "template_fallback", "obligation_last_resort"}
            or selected_candidate_source in {"obligation_repair", "whatsapp_obligation_repair", "conversation_obligation", "contextual_fallback", "obligation_last_resort"}
        )
        if reply and bool(selected_validation.get("passed")):
            self._whatsapp_record_recent_draft(state_key, request_id, reply)
        memory_rows = self._whatsapp_web_memory_rows(request.contact_name, limit=6)
        questions = self._whatsapp_web_context_questions(
            contact_name=request.contact_name,
            relationship_type=relationship,
            collected_count=len(message_context) + len(post_incoming_context),
            memory_count=len(memory_rows),
        )
        burst_messages = self._whatsapp_latest_incoming_burst(cleaned_messages, incoming_index)
        context_excluded = self._whatsapp_context_exclusions(cleaned_messages, burst_start_index, incoming_index)
        old_context_topics = self._whatsapp_old_context_topics(message_context, obligation.concrete_topics_detected or [])
        unresolved_obligations = self._whatsapp_unresolved_obligations(message_context, obligation.concrete_topics_detected or [])
        provider_debug = self._whatsapp_provider_debug(response, enriched_candidates)
        rejection_reasons = list(dict.fromkeys(reason for item in candidates_rejected for reason in item.get("reasons", [])))
        validation_results = [
            {
                "source": self._whatsapp_candidate_source_label(candidate),
                "preview": str(candidate.get("text") or "")[:160],
                "validation": candidate.get("whatsapp_obligation_validation", {}),
            }
            for candidate in enriched_candidates
            if isinstance(candidate, dict)
        ]
        fallback_reason = str(selected.get("fallback_reason") or "") if fallback_used else ""
        draft_trace = WhatsAppDraftTrace(
            request_id=request_id,
            thread_id=request.thread_id or "",
            chat_name=request.contact_name or "",
            raw_collected_count=len(cleaned_messages),
            latest_incoming_burst=obligation.latest_incoming_burst,
            latest_incoming_burst_timestamps=[str(message.get("timestamp") or "") for message in burst_messages],
            latest_incoming_burst_client_orders=[
                int(message["client_order"]) if message.get("client_order") is not None and str(message.get("client_order")).lstrip("-").isdigit() else None
                for message in burst_messages
            ],
            latest_incoming_burst_message_ids=[str(message.get("message_id") or "") for message in burst_messages],
            latest_burst_summary=" | ".join(obligation.latest_incoming_burst),
            latest_burst_topics=obligation.concrete_topics_detected or [],
            context_messages_used=context,
            context_messages_excluded=context_excluded,
            old_context_topics_detected=old_context_topics,
            unresolved_obligations=unresolved_obligations,
            model_provider=str(provider_debug.get("model_provider") or ""),
            model_error=str(provider_debug.get("model_error") or provider_failure_reason or ""),
            rate_limit_detected=bool(provider_debug.get("rate_limit_detected")),
            candidate_count=len(enriched_candidates),
            candidate_sources=candidate_sources,
            candidate_previews=[self._whatsapp_candidate_preview(candidate) for candidate in enriched_candidates if isinstance(candidate, dict)],
            validation_results=validation_results,
            rejection_reasons=rejection_reasons,
            repair_attempts=repair_attempts,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            cache_used=False,
            selected_candidate_source=selected_candidate_source_label,
            selected_reply_topics=selected_reply_topics,
            repeat_similarity_thread=float(selected_similarity.get("thread", 0.0)) if isinstance(selected_similarity, dict) else 0.0,
            repeat_similarity_global=float(selected_similarity.get("global", 0.0)) if isinstance(selected_similarity, dict) else 0.0,
            final_decision=automation_decision if reply else "NO_DRAFT",
            final_reply_units=selected_sequence,
        )

        return {
            "source": "whatsapp_web",
            "request_id": request_id,
            "thread_id": request.thread_id or "",
            "state_key": state_key,
            "mode": request.mode,
            "contact_name": request.contact_name or "",
            "incoming": incoming,
            "context": context,
            "message_context_count": len(message_context),
            "post_incoming_context_count": len(post_incoming_context),
            "saved_memory_context_count": len(saved_memory_context),
            "time_context_count": len(time_context),
            "collected_message_count": len(cleaned_messages),
            "context_target_count": 220,
            "context_minimum_met": len(message_context) >= 40,
            "questions": questions,
            "saved_memory": [
                {
                    "id": str(row.get("id") or ""),
                    "question": str(row.get("question") or ""),
                    "answer": str(row.get("answer") or ""),
                    "created_at": str(row.get("created_at") or ""),
                }
                for row in memory_rows
            ],
            "relationship_type": relationship,
            "relationship_inferred_from_thread": relationship_inferred,
            "intent_type": response.get("intent_type", request.intent_type),
            "reply": reply,
            "reply_sequence": selected_sequence,
            "candidate": selected,
            "candidate_count": len(enriched_candidates),
            "candidates_generated": len(enriched_candidates),
            "candidate_sources": candidate_sources,
            "candidate_previews": [self._whatsapp_candidate_preview(candidate) for candidate in enriched_candidates if isinstance(candidate, dict)],
            "candidates_rejected": candidates_rejected,
            "rejection_reasons": rejection_reasons,
            "selected_candidate_source": selected_candidate_source,
            "selected_candidate_source_label": selected_candidate_source_label,
            "selected_candidate_similarity_to_recent": selected_similarity,
            "selected_candidate_topics": selected_reply_topics,
            "repair_attempts": repair_attempts,
            "model_provider": provider_debug.get("model_provider", ""),
            "model_error": provider_debug.get("model_error", provider_failure_reason),
            "rate_limit_detected": provider_debug.get("rate_limit_detected", False),
            "conversation_obligation": obligation.to_debug_dict(),
            "latest_incoming_burst": obligation.latest_incoming_burst,
            "latest_incoming_burst_timestamps": draft_trace.latest_incoming_burst_timestamps,
            "latest_incoming_burst_client_orders": draft_trace.latest_incoming_burst_client_orders,
            "latest_incoming_burst_message_ids": draft_trace.latest_incoming_burst_message_ids,
            "latest_burst_summary": draft_trace.latest_burst_summary,
            "latest_burst_topics": obligation.concrete_topics_detected or [],
            "context_messages_used": context,
            "context_messages_excluded": context_excluded,
            "old_context_topics_detected": old_context_topics,
            "unresolved_obligations": unresolved_obligations,
            "incoming_burst_count": len(obligation.latest_incoming_burst),
            "incoming_fragmentation_score": obligation.incoming_fragmentation_score,
            "user_style_burst_baseline": obligation.user_style_burst_baseline,
            "emotional_intensity": obligation.emotional_intensity,
            "repair_required": obligation.repair_required,
            "concrete_topics_detected": obligation.concrete_topics_detected or [],
            "target_reply_burst_size": obligation.target_reply_burst_size or {"min": 1, "max": 3},
            "reply_units": obligation.reply_units or [],
            "selected_reply_topics": selected_reply_topics,
            "similarity_to_recent_thread_drafts": selected_similarity.get("thread", 0.0) if isinstance(selected_similarity, dict) else 0.0,
            "similarity_to_recent_global_drafts": selected_similarity.get("global", 0.0) if isinstance(selected_similarity, dict) else 0.0,
            "recent_drafts_for_this_thread": recent_thread_drafts[-6:],
            "unique_response_acts": obligation.unique_response_acts or [],
            "duplicate_check_result": obligation.duplicate_check_result or {},
            "send_queue_safety_result": obligation.send_queue_safety_result or {},
            "conversation_move": obligation.conversation_move,
            "required_response_acts": obligation.required_response_acts,
            "forbidden_response_acts": obligation.forbidden_response_acts,
            "tone_target": obligation.tone_target,
            "reply_shape": obligation.reply_shape,
            "validation_result": obligation.validation_result or {},
            "automation_decision": automation_decision,
            "auto_send_allowed": automation_decision == "SEND_ALLOWED",
            "auto_send_blocked_reason": auto_send_blocked_reason,
            "insert_allowed": bool(reply),
            "draft_cache_hit": False,
            "cache_used": False,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "draft_trace": draft_trace.to_dict(),
            "ui_rendered": False,
            "ui_render_rejection_reason": "not_rendered_by_controller",
            "debug_line": (
                f"request_id={request_id}; thread_id={request.thread_id or ''}; state_key={state_key}; "
                f"latest_incoming_burst={obligation.latest_incoming_burst}; latest_burst_topics={obligation.concrete_topics_detected or []}; "
                f"candidate_count={len(enriched_candidates)}; candidate_sources={candidate_sources}; "
                f"selected_candidate_source={selected_candidate_source_label}; selected_reply_topics={selected_reply_topics}; "
                f"similarity_to_recent_thread_drafts={selected_similarity.get('thread', 0.0) if isinstance(selected_similarity, dict) else 0.0}; "
                f"similarity_to_recent_global_drafts={selected_similarity.get('global', 0.0) if isinstance(selected_similarity, dict) else 0.0}; "
                f"cache_used=False; fallback_used={fallback_used}; validation_result={selected_validation.get('validation_result')}; "
                f"final_reply_preview={reply[:120]}"
            ),
            "warnings": [
                "WhatsApp Web sending is extension-controlled and requires explicit extension mode.",
                "Auto-send only returns true when the local candidate is marked auto_send_allowed.",
            ],
            "training_response": response,
        }

    def whatsapp_web_memory_store(self, request: WhatsAppWebMemoryRequest) -> dict[str, object]:
        question = str(request.question or "").strip()
        answer = str(request.answer or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="question must not be empty.")
        if not answer:
            raise HTTPException(status_code=400, detail="answer must not be empty.")
        contact_name = str(request.contact_name or "").strip()
        if not contact_name:
            raise HTTPException(status_code=400, detail="contact_name must not be empty.")
        cleaned_context = [
            {"speaker": message.speaker, "text": str(message.text or "").strip(), "timestamp": message.timestamp}
            for message in request.messages[-20:]
            if str(message.text or "").strip()
        ]
        relationship, relationship_inferred = self._infer_whatsapp_web_relationship(cleaned_context, request.relationship_type)
        memory_id = "wa_mem_" + hashlib.sha1(
            json.dumps(
                {
                    "contact_name": contact_name.casefold(),
                    "question": question,
                    "answer": answer,
                    "created_hint": datetime.now(timezone.utc).date().isoformat(),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]
        created_at = datetime.now(timezone.utc).isoformat()
        self._ensure_whatsapp_web_memory_db()
        with sqlite3.connect(self.whatsapp_web_memory_db_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO whatsapp_web_memory
                    (id, created_at, contact_name, relationship_type, question, answer, context_json, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    created_at,
                    contact_name,
                    relationship,
                    question,
                    answer,
                    json.dumps(cleaned_context, ensure_ascii=True),
                    str(request.source or "whatsapp_web_extension"),
                ),
            )
        vector_status = self._mark_vector_rebuild_required("whatsapp_web_memory_added")
        return {
            "status": "saved",
            "id": memory_id,
            "created_at": created_at,
            "contact_name": contact_name,
            "relationship_type": relationship,
            "relationship_inferred_from_thread": relationship_inferred,
            "stored_in": str(self.whatsapp_web_memory_db_path),
            "vector_status": vector_status,
        }

    def training_catbot_chat(self, request: CatbotChatRequest) -> dict[str, object]:
        raw_incoming = request.incoming.strip()
        if not raw_incoming:
            raise HTTPException(status_code=400, detail="incoming must not be empty.")
        thread_id = request.thread_id.strip() if request.thread_id and request.thread_id.strip() and not request.reset_thread else f"catbot_thread_{uuid.uuid4().hex[:12]}"
        session_id = f"catbot_{uuid.uuid4().hex[:12]}"
        base_context = [str(item).strip() for item in request.context if str(item).strip()]
        incoming, burst_context = self._catbot_split_incoming_burst(raw_incoming)
        context = [*base_context, *burst_context]
        full_conversation: list[str] = []
        for index, item in enumerate(context[-10:]):
            speaker = "[OTHER]" if index % 2 == 0 else "[ME]"
            full_conversation.append(f"{speaker}: {item}")
        full_conversation.append(f"[OTHER]: {incoming}")
        training_request = TrainingChatRequest(
            incoming=incoming,
            context=context,
            contact_name=request.contact_name,
            relationship_type=request.relationship_type,
            intent_type=request.intent_type,
            diversity_mode=request.diversity_mode,
            parameters=request.parameters,
        )
        intent = classify_intent(incoming, context) if request.intent_type == "auto" else request.intent_type
        relationship = normalize_relationship_type(request.relationship_type)
        route_error_handled = False
        provider_error = None
        fallback_used = False
        self.drafting.thread_memory.append_turn(
            thread_id,
            session_id=session_id,
            role="user",
            text=incoming,
            platform="catbot",
            contact_alias=request.contact_name,
            relationship_type=relationship,
        )
        if relationship == "romantic_interest":
            contract = self._catbot_turn_contract(
                incoming=incoming,
                context=context,
                intent=intent,
                recent_bot_replies=self._catbot_recent_replies.get(thread_id, []),
            )
            ai_selected, ai_candidate = self._catbot_ai_romantic_reply(
                incoming=incoming,
                context=context,
                intent=intent,
                thread_id=thread_id,
                contact_name=request.contact_name,
                contract=contract,
            )
            selected = ai_selected
            repair_attempts = 0
            repair_reason = str((ai_candidate or {}).get("catbot_ai_reject_reason") or "provider_failure")
            while not selected and repair_attempts < 3:
                repair_attempts += 1
                repaired = self._catbot_ai_repair_reply(
                    incoming=incoming,
                    context=context,
                    intent=intent,
                    contact_name=request.contact_name,
                    reject_reason=repair_reason,
                    recent_bot_replies=self._catbot_recent_replies.get(thread_id, []),
                    contract=contract,
                )
                if not repaired:
                    break
                repaired_reject = self._catbot_ai_reject_reason(
                    repaired,
                    incoming=incoming,
                    context=context,
                    recent_bot_replies=self._catbot_recent_replies.get(thread_id, []),
                    contract=contract,
                )
                if not repaired_reject:
                    selected = repaired
                    break
                repair_reason = repaired_reject
            if selected:
                rescued = False
                selected_candidate = ai_candidate or {}
                if not selected_candidate:
                    selected_candidate = {
                        **selected_candidate,
                        "text": selected,
                        "sequence": [selected],
                        "candidate_type": "catbot_ai_romantic_repair",
                        "relationship_type": relationship,
                        "intent": intent,
                        "why_this_matches": "provider-generated romantic Catbot repair response",
                        "final_decision": "review",
                        "auto_send_allowed": False,
                    }
                else:
                    selected_sequence = (
                        [
                            str(item).strip()
                            for item in selected_candidate.get("sequence", [])
                            if str(item).strip()
                        ]
                        if ai_selected
                        else [selected]
                    ) or [selected]
                    selected_candidate["text"] = selected
                    selected_candidate["sequence"] = selected_sequence
                    if ai_selected:
                        selected_candidate["candidate_type"] = "catbot_ai_romantic"
                    else:
                        selected_candidate["candidate_type"] = "catbot_ai_romantic_repair"
                        selected_candidate["why_this_matches"] = "AI repair replaced rejected provider output"
                        selected_candidate["provider"] = "plan_ranker"
                        selected_candidate["model"] = "conversation_function_policy"
                        selected_candidate["provider_error"] = None
                        selected_candidate["provider_latency_ms"] = 0
                        selected_candidate["external_api_used"] = False
                        selected_candidate["manual_review_fallback"] = False
                        selected_candidate["fallback_used"] = False
                        selected_candidate["catbot_ai_reject_reason"] = ""
                        selected_candidate["catbot_ai_repaired"] = True
                        selected_candidate["catbot_ai_repair_accepted"] = True
                        selected_candidate["catbot_ai_repair_attempts"] = repair_attempts
                        selected_candidate["catbot_ai_final_repair_reason"] = repair_reason
                selected, selected_candidate = self._catbot_finalize_candidate(selected_candidate, contract=contract)
                contact_key = request.contact_name or "Catbot"
                self._conversation_state.record_bot_reply(contact_key, selected)
                self.drafting.conversation_state.record_bot_reply(contact_key, selected)
                self.drafting.thread_memory.append_turn(
                    thread_id,
                    session_id=session_id,
                    role="bot",
                    text=selected,
                    platform="catbot",
                    contact_alias=request.contact_name,
                    relationship_type=relationship,
                    candidate_metadata=selected_candidate,
                )
                self._catbot_recent_replies.setdefault(thread_id, []).append(selected)
                self._catbot_recent_replies[thread_id] = self._catbot_recent_replies[thread_id][-20:]
                session = {
                    "id": session_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": "catbot",
                    "thread_id": thread_id,
                    "incoming": raw_incoming,
                    "context": base_context,
                    "contact_name": request.contact_name,
                    "relationship_type": relationship,
                    "intent_type": intent,
                    "selected_candidate": selected,
                    "candidates": [selected_candidate],
                    "feedback_label": "",
                    "correct_reply": "",
                    "notes": "catbot romantic ai-first response; awaiting thumbs rating",
                    "quarantined": False,
                }
                self._append_jsonl(self.training_sessions_path, session)
                return {
                    "session_id": session_id,
                    "thread_id": thread_id,
                    "source": "catbot",
                    "incoming": incoming,
                    "context": context,
                    "contact_name": request.contact_name,
                    "relationship_type": relationship,
                    "intent_type": intent,
                    "reply": selected,
                    "viewer_messages": [
                        {"speaker": "other", "text": incoming},
                        *[{"speaker": "me", "text": part} for part in selected_candidate.get("sequence", [selected])],
                    ],
                    "candidate": selected_candidate,
                    "candidates": [selected_candidate],
                    "reply_sequence": selected_candidate.get("sequence", [selected]),
                    "retrieval_backend": "catbot_ai_romantic",
                    "vector_results_count": 0,
                    "timings_ms": {},
                    "adb_touched": False,
                    "route_error_handled": False,
                    "provider_error": selected_candidate.get("provider_error"),
                    "fallback_used": False,
                    "catbot_rescued": False,
                }
            raise HTTPException(status_code=502, detail=f"Catbot AI provider did not return a usable reply. Last rejection: {repair_reason}.")
        previous_ai_enabled = self.drafting.ai_reply_enabled
        try:
            if relationship == "romantic_interest":
                self.drafting.ai_reply_enabled = False
            try:
                bundle = self.drafting.build_bundle_with_context(
                    recent_messages=[*context[-5:], incoming],
                    full_conversation=full_conversation,
                    contact_name=request.contact_name,
                    relationship_type=relationship,
                )
            finally:
                self.drafting.ai_reply_enabled = previous_ai_enabled
        except Exception as exc:
            route_error_handled = True
            provider_error = type(exc).__name__
            fallback_used = True
            self.drafting.ai_reply_enabled = False
            try:
                bundle = self.drafting.build_bundle_with_context(
                    recent_messages=[*context[-5:], incoming],
                    full_conversation=full_conversation,
                    contact_name=request.contact_name,
                    relationship_type=relationship,
                )
            finally:
                self.drafting.ai_reply_enabled = previous_ai_enabled
            for candidate in bundle.reply_candidates:
                candidate.provider_error = provider_error
                candidate.fallback_used = True
                candidate.critic_scores["provider_error"] = provider_error
                candidate.critic_scores["fallback_used"] = True
        bundle.relationship_type = relationship
        bundle.intent_type = intent
        self._apply_training_parameters(bundle, request.parameters.model_dump(mode="json"), relationship)
        response = self._training_response(training_request, bundle)
        selected = str(response.get("selected_candidate") or "")
        selected_candidate = response["candidates"][0] if response.get("candidates") else {}
        selected, rescued = self._catbot_rescue_reply(
            incoming=incoming,
            context=context,
            selected=selected,
            relationship=relationship,
        )
        if rescued:
            response["selected_candidate"] = selected
            if response.get("candidates"):
                response["candidates"][0]["text"] = selected
                response["candidates"][0]["sequence"] = [selected]
                response["candidates"][0]["candidate_type"] = "catbot_romantic_rescue"
                response["candidates"][0]["why_this_matches"] = "rescued weak romantic Catbot fallback"
                selected_candidate = response["candidates"][0]
            else:
                selected_candidate = {
                    "text": selected,
                    "sequence": [selected],
                    "candidate_type": "catbot_romantic_rescue",
                    "why_this_matches": "rescued weak romantic Catbot fallback",
                }
        if selected:
            contact_key = request.contact_name or "Catbot"
            self._conversation_state.record_bot_reply(contact_key, selected)
            self.drafting.conversation_state.record_bot_reply(contact_key, selected)
            self.drafting.thread_memory.append_turn(
                thread_id,
                session_id=session_id,
                role="bot",
                text=selected,
                platform="catbot",
                contact_alias=request.contact_name,
                relationship_type=relationship,
                candidate_metadata=selected_candidate,
            )
        session = self._training_session_payload(training_request, bundle)
        session["id"] = session_id
        session["source"] = "catbot"
        session["thread_id"] = thread_id
        session["feedback_label"] = ""
        session["correct_reply"] = ""
        session["notes"] = "catbot simulator generated reply; awaiting thumbs rating"
        self._append_jsonl(self.training_sessions_path, session)
        return {
            "session_id": session["id"],
            "thread_id": thread_id,
            "source": "catbot",
            "incoming": incoming,
            "context": context,
            "contact_name": request.contact_name,
            "relationship_type": response.get("relationship_type", request.relationship_type),
            "intent_type": response.get("intent_type", request.intent_type),
            "reply": selected,
            "viewer_messages": [
                {"speaker": "other", "text": incoming},
                *[{"speaker": "me", "text": part} for part in (selected_candidate.get("sequence", [selected]) if isinstance(selected_candidate, dict) else [selected])],
            ],
            "candidate": selected_candidate,
            "candidates": response.get("candidates", []),
            "reply_sequence": selected_candidate.get("sequence", [selected]) if isinstance(selected_candidate, dict) else [selected],
            "retrieval_backend": response.get("retrieval_backend"),
            "vector_results_count": response.get("vector_results_count", 0),
            "timings_ms": response.get("timings_ms", {}),
            "adb_touched": False,
            "route_error_handled": route_error_handled,
            "provider_error": provider_error,
            "fallback_used": fallback_used or bool(response.get("candidates", [{}])[0].get("fallback_used") if response.get("candidates") else False),
            "catbot_rescued": rescued,
        }

    def _catbot_turn_contract(
        self,
        *,
        incoming: str,
        context: list[str],
        intent: str,
        recent_bot_replies: list[str] | None = None,
    ) -> CatbotTurnContract:
        conversation_function = predict_conversation_function(incoming=incoming, context=context)
        move = self._catbot_conversation_move(
            incoming=incoming,
            context=context,
            conversation_function=conversation_function,
        )
        return CatbotTurnContract(
            incoming=incoming,
            context=[str(item).strip() for item in context if str(item).strip()],
            intent=intent,
            conversation_function=conversation_function,
            move=move,
            reply_plan=self._catbot_reply_plan(move),
            recent_bot_replies=[str(item).strip() for item in (recent_bot_replies or []) if str(item).strip()],
        )

    def _catbot_ai_romantic_reply(
        self,
        *,
        incoming: str,
        context: list[str],
        intent: str,
        thread_id: str,
        contact_name: str | None,
        contract: CatbotTurnContract | None = None,
    ) -> tuple[str, dict[str, object] | None]:
        if not self.settings.external_api_enabled and str(self.settings.private_provider or "").strip().lower() != "ollama":
            return "", None
        contract = contract or self._catbot_turn_contract(
            incoming=incoming,
            context=context,
            intent=intent,
            recent_bot_replies=self._catbot_recent_replies.get(thread_id, []),
        )
        started = time.perf_counter()
        direct_reply = self._catbot_direct_plan_reply(
            contract,
            incoming=incoming,
            context=context,
            recent_bot_replies=self._catbot_recent_replies.get(thread_id, []),
        )
        if direct_reply:
            candidate = {
                "text": direct_reply,
                "sequence": [part for part in re.split(r"\s*/\s*|\n+", direct_reply) if part.strip()],
                "candidate_type": "catbot_conversation_function_plan",
                "relationship_type": "romantic_interest",
                "intent": intent,
                "why_this_matches": "high-confidence conversation-function plan response",
                "final_decision": "review",
                "auto_send_allowed": False,
                "provider": "plan_ranker",
                "model": "conversation_function_policy",
                "provider_latency_ms": 0,
                "provider_error": None,
                "provider_configured": True,
                "external_api_used": False,
                "external_api_blocked": False,
                "fallback_errors": [],
                "manual_review_fallback": False,
                "fallback_used": False,
                "catbot_ai_reject_reason": "",
                "catbot_ai_initial_reject_reason": "",
                "catbot_ai_retry_accepted": False,
                "catbot_ai_retry_attempts": 0,
                "catbot_ai_repaired": False,
                "catbot_ai_repair_accepted": False,
                "catbot_ai_repair_attempts": 0,
                "catbot_ai_final_repair_reason": None,
                "catbot_ai_plan_repair_attempted": False,
                "catbot_ai_plan_repair_accepted": False,
                "catbot_ai_plan_repair_reason": None,
                "catbot_ai_plan_direct": True,
                "timing_ms": round((time.perf_counter() - started) * 1000, 2),
            }
            return self._catbot_finalize_candidate(candidate, contract=contract)
        messages = self._catbot_ai_prompt_messages(
            incoming=incoming,
            context=context,
            intent=intent,
            contact_name=contact_name,
            contract=contract,
        )
        context_texts = [*context[-max(2, int(self.settings.external_api_max_context_messages)) :], incoming]

        adult_ai_plan = contract.reply_plan.shape in {"multi_bubble_adult_escalation", "specific_escalation_detail"}

        async def _call():
            return await generate_with_fallback(
                settings=self.settings,
                messages=messages,
                provider_names=[
                    self.settings.draft_provider,
                    self.settings.fast_provider,
                    self.settings.router_provider,
                    self.settings.private_provider,
                ],
                temperature=0.92 if adult_ai_plan else 0.78,
                max_tokens=190 if adult_ai_plan else 140,
                response_format="text",
                timeout_seconds=min(float(self.settings.external_api_timeout_seconds), 12.0),
                context_texts=context_texts,
            )

        try:
            try:
                response, metadata = asyncio.run(_call())
            except RuntimeError:
                loop = asyncio.new_event_loop()
                try:
                    response, metadata = loop.run_until_complete(_call())
                finally:
                    loop.close()
        except Exception as exc:
            return "", {
                "candidate_type": "catbot_ai_romantic_error",
                "provider_error": type(exc).__name__,
                "fallback_used": True,
            }
        sequence = self._catbot_extract_ai_sequence(response.text)
        reply = self._catbot_format_ai_sequence(sequence)
        provider_error = response.error or (str(metadata.get("blocked_reason") or "") if metadata.get("external_api_blocked") else "")
        retry_accepted = False
        retry_attempts = 0
        plan_repair_accepted = False
        plan_repair_attempted = False
        plan_repair_reason = ""
        initial_reject_reason = ""
        reject_reason = (
            "provider_failure"
            if response.provider == "fallback" or bool(metadata.get("manual_review_fallback")) or provider_error
            else self._catbot_ai_reject_reason(
                reply,
                incoming=incoming,
                context=context,
                recent_bot_replies=self._catbot_recent_replies.get(thread_id, []),
                contract=contract,
            )
        )
        initial_reject_reason = reject_reason
        if reject_reason and self._catbot_prefers_plan_repair(contract, reject_reason):
            plan_repair_attempted = True
            plan_repair_reason = reject_reason
            plan_repair = self._catbot_plan_specific_repair_reply(
                contract,
                reject_reason=reject_reason,
                avoid_replies=[*self._catbot_recent_replies.get(thread_id, []), *context[1::2]],
            )
            if plan_repair:
                plan_repair_reject = self._catbot_ai_reject_reason(
                    plan_repair,
                    incoming=incoming,
                    context=context,
                    recent_bot_replies=self._catbot_recent_replies.get(thread_id, []),
                    contract=contract,
                )
                if not plan_repair_reject:
                    reply = plan_repair
                    sequence = self._catbot_normalize_candidate_sequence([part for part in re.split(r"\s*/\s*|\n+", plan_repair) if part.strip()])
                    reject_reason = ""
                    plan_repair_accepted = True
                else:
                    reject_reason = plan_repair_reject
        if reject_reason and reject_reason in {"missed_adult_mode", "missing_sensory_texture", "missed_romantic_mode", "too_graphic", "too_dry", "not_carrying_conversation", "provider_failure", "wrong_owner_gender", "wrong_owner_style", "recent_repeat", "repeated_reply_shape", "repeated_adult_detail_skeleton", "generic_ai_style", "overeager_greeting"}:
            max_retry_attempts = 3 if reject_reason == "provider_failure" else 1
            retry_reject_reason = reject_reason
            for _ in range(max_retry_attempts):
                retry_attempts += 1
                retry_reply, retry_metadata = self._catbot_ai_retry_romantic_reply(
                    incoming=incoming,
                    context=context,
                    intent=intent,
                    contact_name=contact_name,
                    reject_reason=reject_reason,
                    contract=contract,
                )
                retry_response = retry_metadata.get("response")
                retry_provider_error = getattr(retry_response, "error", None) or (
                    str(retry_metadata.get("metadata", {}).get("blocked_reason") or "")
                    if retry_metadata.get("metadata", {}).get("external_api_blocked")
                    else ""
                )
                retry_reject_reason = (
                    "provider_failure"
                    if not retry_reply or getattr(retry_response, "provider", "") == "fallback" or bool(retry_metadata.get("metadata", {}).get("manual_review_fallback")) or retry_provider_error
                    else self._catbot_ai_reject_reason(
                        retry_reply,
                        incoming=incoming,
                        context=context,
                        recent_bot_replies=self._catbot_recent_replies.get(thread_id, []),
                        contract=contract,
                    )
                )
                if retry_reply and not retry_reject_reason:
                    reply = retry_reply
                    response = retry_metadata["response"]
                    metadata = retry_metadata["metadata"]
                    provider_error = retry_provider_error
                    sequence = self._catbot_extract_ai_sequence(response.text) or [reply]
                    sequence = self._catbot_normalize_candidate_sequence(sequence)
                    reply = self._catbot_format_ai_sequence(sequence)
                    reject_reason = ""
                    retry_accepted = True
                    break
            if not retry_accepted and retry_reject_reason:
                reject_reason = retry_reject_reason
        candidate = {
            "text": reply,
            "sequence": sequence if reply else [],
            "candidate_type": "catbot_ai_romantic",
            "relationship_type": "romantic_interest",
            "intent": intent,
            "why_this_matches": "provider-generated romantic Catbot response",
            "final_decision": "review",
            "auto_send_allowed": False,
            "provider": response.provider,
            "model": response.model,
            "provider_latency_ms": response.latency_ms,
            "provider_error": provider_error or None,
            "provider_configured": metadata.get("provider_configured"),
            "external_api_used": response.external_api_used,
            "external_api_blocked": bool(metadata.get("external_api_blocked")),
            "fallback_errors": metadata.get("fallback_errors", []),
            "manual_review_fallback": bool(metadata.get("manual_review_fallback")),
            "fallback_used": bool(metadata.get("manual_review_fallback")) or bool(provider_error),
            "catbot_ai_reject_reason": reject_reason,
            "catbot_ai_initial_reject_reason": initial_reject_reason,
            "catbot_ai_retry_accepted": retry_accepted,
            "catbot_ai_retry_attempts": retry_attempts,
            "catbot_ai_plan_repair_attempted": plan_repair_attempted,
            "catbot_ai_plan_repair_accepted": plan_repair_accepted,
            "catbot_ai_plan_repair_reason": plan_repair_reason or None,
            "catbot_ai_repaired": plan_repair_accepted,
            "catbot_ai_repair_accepted": plan_repair_accepted,
            "catbot_ai_repair_attempts": 0 if plan_repair_accepted else None,
            "catbot_ai_final_repair_reason": plan_repair_reason if plan_repair_accepted else None,
            "timing_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        if plan_repair_accepted:
            candidate.update(
                {
                    "provider": "plan_ranker",
                    "model": "conversation_function_policy",
                    "provider_latency_ms": 0,
                    "provider_error": None,
                    "external_api_used": False,
                    "external_api_blocked": False,
                    "manual_review_fallback": False,
                    "fallback_used": False,
                    "why_this_matches": "plan repair replaced rejected provider output",
                }
            )
        if reject_reason:
            return "", candidate
        reply, candidate = self._catbot_finalize_candidate(candidate, contract=contract)
        return reply, candidate

    def _catbot_prefers_plan_repair(self, contract: CatbotTurnContract, reject_reason: str) -> bool:
        plan = contract.reply_plan
        fast_reasons = {
            "provider_failure",
            "missed_adult_mode",
            "missing_sensory_texture",
            "missed_romantic_mode",
            "too_graphic",
            "too_dry",
            "too_short",
            "not_carrying_conversation",
            "generic_ai_style",
            "recent_repeat",
            "repeated_reply_shape",
            "repeated_adult_detail_skeleton",
            "repeated_adult_detail_family",
            "echoed_incoming",
            "logistics_instead_of_romance",
        }
        if reject_reason not in fast_reasons:
            return False
        if plan.move == "romantic_escalation":
            return plan.shape in {"multi_bubble_adult_escalation", "specific_escalation_detail"}
        if plan.move in {
            "affectionate_greeting",
            "loop_callout",
            "repair_callout",
            "topic_dismissal_reset",
            "low_effort_ack",
        }:
            return True
        if plan.shape == "answer_status_then_continue":
            return reject_reason in {"provider_failure", "recent_repeat", "repeated_reply_shape"}
        if plan.shape in {
            "confirm_with_specificity",
            "reciprocate_affection_plus_specific_continuation",
            "identity_fact_then_continue",
            "identity_fact_restate_then_continue",
            "work_status_fact_then_side_project",
        }:
            return True
        if plan.move == "unclassified" and plan.required_slots.get("unclassified_context") in {
            "low_key_status_disclosure",
            "reason_for_previous_comment",
            "reason_for_previous_question",
            "previous_comment_clarification",
            "story_reality_confirmation",
            "story_hypothetical_reaction",
        }:
            return True
        return False

    def _catbot_direct_plan_reply(
        self,
        contract: CatbotTurnContract,
        *,
        incoming: str,
        context: list[str],
        recent_bot_replies: list[str],
    ) -> str:
        if not self._catbot_prefers_provider_bypass(contract):
            return ""
        avoid_replies = [*recent_bot_replies, *context[1::2]]
        for reject_reason in (
            "provider_failure",
            "recent_repeat",
            "repeated_reply_shape",
            "generic_ai_style",
            "missed_adult_mode",
            "missing_sensory_texture",
        ):
            reply = self._catbot_plan_specific_repair_reply(
                contract,
                reject_reason=reject_reason,
                avoid_replies=avoid_replies,
            )
            if not reply:
                continue
            reject = self._catbot_ai_reject_reason(
                reply,
                incoming=incoming,
                context=context,
                recent_bot_replies=recent_bot_replies,
                contract=contract,
            )
            if not reject:
                return reply
            avoid_replies.append(reply)
        return ""

    def _catbot_prefers_provider_bypass(self, contract: CatbotTurnContract) -> bool:
        plan = contract.reply_plan
        if plan.move == "romantic_escalation" and plan.shape == "multi_bubble_adult_escalation":
            return False
        if plan.move == "romantic_escalation" and plan.shape == "specific_escalation_detail":
            return False
        if plan.shape == "reciprocate_affection_plus_specific_continuation":
            return True
        if plan.shape == "confirm_with_specificity":
            return str(plan.required_slots.get("challenge_target") or "") in {
                "missed_you_confirmation",
                "sincerity_confirmation",
            }
        if plan.shape == "playful_scold_acknowledge_then_soften":
            return True
        if plan.shape == "fresh_status_detail_after_recent_status":
            return True
        if plan.shape == "answer_activity_detail_then_continue":
            return True
        if plan.shape == "answer_status_then_continue":
            incoming_norm = normalize_text(contract.incoming)
            return any(term in incoming_norm for term in ("hru", "how are u", "how are you", "how u doing", "how you doing", "wby", "wbu", "what about u", "what about you"))
        if plan.move in {
            "affectionate_greeting",
            "loop_callout",
            "repair_callout",
            "topic_dismissal_reset",
        }:
            return True
        if plan.shape in {
            "identity_fact_then_continue",
            "identity_fact_restate_then_continue",
            "work_status_fact_then_side_project",
        }:
            return True
        if plan.shape == "acknowledge_status_disclosure_plus_owner_detail":
            return str(plan.required_slots.get("status_topic") or "") in {
                "gym_activity",
                "work_activity",
                "study_activity",
                "rest_activity",
                "general_activity",
            }
        if plan.move == "unclassified" and plan.required_slots.get("unclassified_context") == "low_key_status_disclosure":
            return True
        if plan.move == "unclassified" and plan.required_slots.get("unclassified_context") == "answer_own_previous_prompt":
            return True
        if plan.move == "unclassified" and plan.required_slots.get("unclassified_context") == "previous_comment_clarification":
            return True
        return False

    def _catbot_style_examples(
        self,
        *,
        incoming: str,
        context: list[str],
        intent: str,
        contact_name: str | None,
        limit: int = 5,
        use_vector: bool = False,
    ) -> list[dict[str, object]]:
        if self._catbot_style_rows_cache is None:
            rows = load_all_training_rows(self.settings.ai_reply_training_messages_dir)
            rows.extend(load_corrections(self.settings.ai_reply_training_messages_dir))
            self._catbot_style_rows_cache = [
                row
                for row in rows
                if self._catbot_style_row_allowed(row)
            ]
        examples = (
            self._catbot_vector_style_examples(
                incoming=incoming,
                context=context,
                intent=intent,
                contact_name=contact_name,
                limit=limit,
            )
            if use_vector
            else []
        )
        backend = LexicalRetrievalBackend(self._catbot_style_rows_cache)
        if not examples:
            examples = backend.search(
                incoming,
                context,
                "romantic_interest",
                intent,
                contact_name=contact_name,
                limit=limit,
            )
        examples = [example for example in examples if self._catbot_style_example_allowed(example)]
        if len(examples) < limit:
            seen = {example.retrieval_id for example in examples}
            more = backend.search(
                incoming,
                context,
                "romantic_interest",
                intent,
                contact_name=contact_name,
                limit=limit * 3,
            )
            for example in more:
                if example.retrieval_id not in seen and self._catbot_style_example_allowed(example):
                    examples.append(example)
                    seen.add(example.retrieval_id)
                if len(examples) >= limit:
                    break
        if len(examples) < limit:
            seen = {example.retrieval_id for example in examples}
            broader = backend.search(
                incoming,
                context,
                "close_friend",
                intent,
                contact_name=contact_name,
                limit=limit * 2,
            )
            for example in broader:
                if example.retrieval_id not in seen and self._catbot_style_example_allowed(example):
                    examples.append(example)
                    seen.add(example.retrieval_id)
                if len(examples) >= limit:
                    break
        examples = sorted(
            examples,
            key=lambda example: self._catbot_style_example_rank(
                example,
                incoming=incoming,
                context=context,
                intent=intent,
            ),
            reverse=True,
        )
        return [
            {
                "incoming": example.incoming,
                "context": example.context[-4:],
                "my_reply": example.my_reply,
                "relationship_type": example.relationship_type,
                "intent_type": example.intent_type,
                "source": example.source,
                "backend": example.backend,
                "score": example.score,
                "authority": (getattr(example, "metadata", {}) or {}).get("style_authority") or ("high" if str(example.source) in {"correction", "approved_auto_style_improvement"} else "medium"),
            }
            for example in examples[:limit]
        ]

    def _catbot_vector_style_examples(
        self,
        *,
        incoming: str,
        context: list[str],
        intent: str,
        contact_name: str | None,
        limit: int,
    ) -> list[Any]:
        index_dir = self.settings.data_dir / "vector_db" / "reply_examples_chroma"
        if not index_dir.exists():
            return []
        try:
            import chromadb
        except Exception:
            return []
        try:
            client = chromadb.PersistentClient(path=str(index_dir))
            collection = client.get_collection("reply_examples")
            query_text = f"incoming: {incoming}\ncontext: {' | '.join(context[-6:])}\nintent: {intent}\ncontact: {contact_name or ''}"
            query_embedding = embed_text(query_text)
            result_sets = [
                collection.query(
                    query_embeddings=[query_embedding],
                    n_results=max(limit * 8, 30),
                    where={"relationship_type": "romantic_interest"},
                    include=["metadatas", "documents", "distances"],
                )
            ]
            if limit > 1:
                result_sets.append(
                    collection.query(
                        query_embeddings=[query_embedding],
                        n_results=max(limit * 6, 20),
                        where={"relationship_type": "close_friend"},
                        include=["metadatas", "documents", "distances"],
                    )
                )
        except Exception:
            return []
        examples: list[Any] = []
        seen: set[str] = set()
        combined: list[tuple[int, str, dict[str, Any], float]] = []
        for priority, results in enumerate(result_sets):
            documents = results.get("documents", [[]])[0] or []
            metadatas = results.get("metadatas", [[]])[0] or []
            distances = results.get("distances", [[]])[0] or []
            combined.extend((priority, str(document or ""), dict(metadata or {}), float(distance or 0.0)) for document, metadata, distance in zip(documents, metadatas, distances))
        for _, document, metadata, distance in sorted(combined, key=lambda item: (item[0], item[3])):
            row = dict(metadata or {})
            row["incoming"] = row.get("incoming") or row.get("query_incoming") or ""
            row["my_reply"] = row.get("my_reply") or row.get("reply") or ""
            if document and (not row["incoming"] or not row["my_reply"]):
                for part in str(document).split(" || "):
                    if part.startswith("incoming: ") and not row["incoming"]:
                        row["incoming"] = part[len("incoming: ") :]
                    elif part.startswith("reply: ") and not row["my_reply"]:
                        row["my_reply"] = part[len("reply: ") :]
            row.setdefault("context", [])
            if isinstance(row.get("context"), str):
                row["context"] = [item.strip() for item in str(row["context"]).split("|") if item.strip()]
            if not self._catbot_style_row_allowed(row):
                continue
            relationship = normalize_relationship_type(str(row.get("relationship_type") or "unknown"))
            if relationship not in {"romantic_interest", "close_friend", "casual_friend", "unknown"}:
                continue
            retrieval_id = str(row.get("id") or hashlib.sha1(json.dumps(row, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12])
            if retrieval_id in seen:
                continue
            seen.add(retrieval_id)
            from libs.drafting.retrieval_backends.base import RetrievedExample

            examples.append(
                RetrievedExample(
                    retrieval_id=retrieval_id,
                    relationship_type=relationship,
                    incoming=str(row.get("incoming") or ""),
                    my_reply=str(row.get("my_reply") or ""),
                    context=row.get("context") if isinstance(row.get("context"), list) else [],
                    intent_type=normalize_intent_label(str(row.get("intent_type") or intent)),
                    score=round(max(0.0, 1.0 - float(distance or 0.0)), 3),
                    reason="vector style match",
                    source=str(row.get("_source") or row.get("source") or "vector_db"),
                    backend="vector_chroma",
                    metadata={k: v for k, v in row.items() if k not in {"incoming", "my_reply", "context"}},
                )
            )
            if len(examples) >= max(limit * 6, 18):
                break
        return sorted(
            examples,
            key=lambda example: self._catbot_style_example_rank(
                example,
                incoming=incoming,
                context=context,
                intent=intent,
            ),
            reverse=True,
        )[:limit]

    def _catbot_style_example_rank(self, example: Any, *, incoming: str, context: list[str], intent: str) -> float:
        metadata = getattr(example, "metadata", {}) or {}
        reply = str(getattr(example, "my_reply", "") or "")
        reply_norm = normalize_text(reply)
        incoming_norm = normalize_text(incoming)
        row_incoming_norm = normalize_text(str(getattr(example, "incoming", "") or ""))
        score = float(getattr(example, "score", 0.0) or 0.0)
        source = str(getattr(example, "source", "") or metadata.get("source") or metadata.get("_source") or "")
        authority = str(metadata.get("style_authority") or "").lower()
        relationship = normalize_relationship_type(str(getattr(example, "relationship_type", "") or "unknown"))
        row_intent = normalize_intent_label(str(getattr(example, "intent_type", "") or "unknown"))
        rank = score
        if authority == "high" or bool(metadata.get("is_correction")) or source in {"correction", "approved_auto_style_improvement", "catbot"}:
            rank += 0.55
        elif authority == "medium":
            rank += 0.2
        if relationship == "romantic_interest":
            rank += 0.3
        elif relationship == "close_friend":
            rank += 0.1
        if row_intent == normalize_intent_label(intent):
            rank += 0.18
        if "?" in reply or any(term in reply_norm for term in ("what ", "why ", "how ", "when ", "where ", "u wanna", "call", "wby", "wbu")):
            rank += 0.12
        if len(reply_norm.split()) >= 6:
            rank += 0.08
        if incoming_norm and row_incoming_norm and (incoming_norm in row_incoming_norm or row_incoming_norm in incoming_norm):
            rank += 0.12
        if self._catbot_generic_ai_style(reply_norm, incoming=incoming, context=context):
            rank -= 0.8
        if any(term in reply_norm for term in ("😉", "🤭", "cutie", "babe")):
            rank -= 0.12
        return rank

    def _catbot_style_example_allowed(self, example: Any) -> bool:
        return self._catbot_style_row_allowed(
            {
                "incoming": getattr(example, "incoming", ""),
                "my_reply": getattr(example, "my_reply", ""),
                "context": getattr(example, "context", []),
                "relationship_type": getattr(example, "relationship_type", "unknown"),
                "intent_type": getattr(example, "intent_type", "unknown"),
                **(getattr(example, "metadata", {}) or {}),
            }
        )

    def _catbot_style_row_allowed(self, row: dict[str, Any]) -> bool:
        reply = str(row.get("my_reply") or row.get("user_final_reply") or "").strip()
        incoming = str(row.get("incoming") or "").strip()
        if not reply or not incoming:
            return False
        if bool(row.get("is_synthetic")):
            return False
        if str(row.get("source") or row.get("_source") or "") == "generated_safe_template":
            return False
        context = row.get("context", [])
        context_items = [str(item) for item in context if str(item).strip()] if isinstance(context, list) else []
        blob = normalize_text(" ".join([incoming, reply, *context_items]))
        if any(
            artifact in blob
            for artifact in (
                "<media omitted>",
                "media omitted",
                "this message was edited",
                "message was deleted",
                "you deleted this message",
                "\ufffd",
            )
        ):
            return False
        reply_norm = normalize_text(reply)
        if any(
            term in reply_norm
            for term in (
                "pussy",
                "dick",
                "cum",
                "slut",
                "rape",
                "force",
                "choke",
                "pin you",
                "pin u",
                "naked",
                "under me",
                "scream my name",
            )
        ):
            return False
        if any(
            term in blob
            for term in (
                "pussy",
                "dick",
                "cum",
                "slut",
                "rape",
                "faggot",
                "retard",
                "khs",
                "kill myself",
                "suicidal",
                "self harm",
                "bdsm",
                "foreplay",
                "pin u",
                "pin you",
                "eat u",
                "inside u",
                "naked body",
                "fuck the shit",
                "suck mine",
            )
        ):
            return False
        reply_tokens = reply_norm.split()
        if len(reply_tokens) < 3:
            return False
        if len(reply.split()) > 26:
            return False
        relationship = normalize_relationship_type(str(row.get("relationship_type") or "unknown"))
        source = str(row.get("source") or row.get("_source") or "")
        authority = str(row.get("style_authority") or "").lower()
        return relationship in {"romantic_interest", "close_friend", "casual_friend", "unknown"} or source in {"correction", "approved_auto_style_improvement"} or authority == "high"

    def _catbot_style_summary_fragment(self) -> str:
        if self._catbot_style_summary_cache is None:
            path = self.settings.ai_reply_training_messages_dir / "high_quality_style_summary.json"
            if path.exists():
                try:
                    self._catbot_style_summary_cache = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    self._catbot_style_summary_cache = {}
            else:
                self._catbot_style_summary_cache = {}
        summary = self._catbot_style_summary_cache or {}
        starters = list((summary.get("top_reply_starters") or {}).keys())[:12] if isinstance(summary.get("top_reply_starters"), dict) else []
        tokens = list((summary.get("top_style_tokens") or {}).keys())[:16] if isinstance(summary.get("top_style_tokens"), dict) else []
        average_words = summary.get("average_reply_words")
        return (
            "Owner style summary from saved messages: "
            f"average reply is about {average_words or 8} words; "
            f"common starters include {', '.join(starters) if starters else 'u, i, ur, yeah, nah, bro'}; "
            f"common tokens include {', '.join(tokens) if tokens else 'u, ur, im, yh, nah, bro, wby'}. "
            "Prefer concise direct wording, typo/slang tolerance, and occasional playful challenge over polished romantic prose.\n"
        )

    def _catbot_style_contract_fragment(self) -> str:
        if self._catbot_style_contract_cache is None:
            self._catbot_style_contract_cache = (
                "Owner style contract, higher priority than generic AI habits:\n"
                "- Weights: 70% strong API romance / natural flirting, 30% owner texting style. Do not overfit dry examples.\n"
                "- Owner style means cadence, slang, looseness, and spelling: u/ur/im/yh/icl/ngl/wby are allowed when natural.\n"
                "- Use less grammar and less punctuation. Lowercase is fine, avoid full stops, avoid polished paragraphs.\n"
                "- Default length is short-medium. Use the ReplyPlan bubble count when one is specified; otherwise use 1-3 short consecutive bubbles when that feels natural.\n"
                "- Prefer romantic lines, warmth, teasing, and direct desire over interview questions.\n"
                "- Ask a follow-up only when it actually adds tension or keeps the chat moving; do not bolt on a generic question.\n"
                "- For greetings, do not use 'yo what u saying' or 'heyy/heyyy'. Use hey/hi energy with a soft line or simple check-in. In an established romantic thread, greeting lines can be short and playful.\n"
                "- Do not make plans or logistics unless they ask. Avoid 'any plans later', 'come over', 'call' as default moves.\n"
                "- Adult consensual dirty talk is allowed when they start it. Match the energy, but avoid coercion, threats, minors, slurs, or non-consensual wording.\n"
                "- In adult context, follow-ups like 'and whats that', 'about what exactly', or 'that all?' mean continue the dirty talk with one concrete consensual detail, not a vague answer or a generic question.\n"
                "- For missed-you replies, it is fine to be romantic and not interrogate them.\n"
                "- Identity facts: if asked your age, answer 19. Do not mirror the other person's age or invent a different one.\n"
                "- Avoid generic AI phrases: 'what kind of bored are we talking', 'any plans for later', 'tell me more', and emoji spam.\n"
                "- Avoid American/polished wording: 'hey stranger', 'hey there', overly perfect grammar, or customer-service friendliness.\n"
            )
        return self._catbot_style_contract_cache

    def _catbot_ai_retry_romantic_reply(
        self,
        *,
        incoming: str,
        context: list[str],
        intent: str,
        contact_name: str | None,
        reject_reason: str,
        contract: CatbotTurnContract | None = None,
    ) -> tuple[str, dict[str, object]]:
        incoming_norm = normalize_text(incoming)
        context_blob = " ".join(normalize_text(item) for item in context[-8:])
        contract = contract or self._catbot_turn_contract(incoming=incoming, context=context, intent=intent)
        user_move = str(contract.move.get("user_move") or "")
        carry_convo = user_move in {"boredom_prompt", "carry_conversation_request"}
        bed_wby = any(term in incoming_norm for term in ("im in bed", "i'm in bed", "in bed")) and any(term in incoming_norm for term in ("wby", "wbu", "babu", "baby"))
        intensity_followup = self._catbot_intensity_followup(incoming_norm, context)
        owner_activity_question = self._catbot_owner_activity_question(incoming_norm)
        activity_detail_question = self._catbot_activity_detail_question(incoming_norm, context)
        reciprocal_activity_question = self._catbot_reciprocal_activity_question(incoming_norm)
        really_followup = self._catbot_really_followup(incoming_norm)
        adult_followup = self._catbot_adult_followup(incoming_norm, context)
        guess_what = self._catbot_guess_what_incoming(incoming_norm)
        move_instruction = self._catbot_conversation_move_instruction(contract.move).strip()
        reply_plan_instruction = self._catbot_reply_plan_instruction(contract.reply_plan).strip()
        structured_retry_instruction = " ".join(part for part in (move_instruction, reply_plan_instruction) if part)
        retry_instruction = ""
        if contract.reply_plan.shape == "answer_why_then_stop_loop" and reject_reason in {"provider_failure", "too_dry", "too_short", "not_carrying_conversation", "generic_ai_style"}:
            retry_instruction = "They asked why you keep repeating/asking. Reply in the exact plan shape: brief reason + acknowledge the loop. Use 1-2 short bubbles. Include a reason like cos/because/i was tired/i couldn't think, and include my bad/i know/i'll stop. Do not ask any question, do not start a new topic, and do not say tell me more."
        elif contract.reply_plan.shape == "answer_properly_deepen_reason" and reject_reason in {"provider_failure", "too_dry", "too_short", "not_carrying_conversation", "generic_ai_style"}:
            retry_instruction = "They said answer properly after you already gave a why-answer. Reply in the exact plan shape: deeper or different explanation + acknowledge the loop. Use this semantic frame: you were hiding behind questions instead of saying something real. A valid shape is 'nah fr u right / i was hiding behind questions instead of giving u something real'. Do not repeat head blank, random questions, tired/fried, or the same previous reason. Do not ask a question and do not start a new topic."
        elif contract.reply_plan.shape == "acknowledge_specific_question_family_without_repeat" and reject_reason in {"provider_failure", "too_dry", "too_short", "not_carrying_conversation", "generic_ai_style", "recent_repeat", "repeated_reply_shape"}:
            retry_instruction = "They called out that you already asked day/how-are-you questions. Reply in the exact plan shape: acknowledge the specific repeated question family + say you will stop interrogating + one owner-side statement. Valid shapes: 'yh ur right / no more how-are-u questions from me' or 'fair / no more day questions, im just tired and moving lazy'. Do not ask any question. Do not repeat the generic 'head went blank/throwing questions/keep asking' apology shape."
        elif contract.reply_plan.shape == "acknowledge_loop_then_owner_detail" and reject_reason in {"provider_failure", "too_dry", "too_short", "not_carrying_conversation", "generic_ai_style"}:
            retry_instruction = "They called out repetition after several topic-picking questions. Reply in the exact plan shape: acknowledge the loop + give one concrete owner-side detail or mini story. Do not ask any question, do not pick another topic for them, and do not say tell me more."
        elif contract.reply_plan.shape == "age_fact_then_textured_continue" and reject_reason in {"provider_failure", "too_dry", "too_short", "not_carrying_conversation", "generic_ai_style", "identity_age_wrong"}:
            retry_instruction = "They asked your age. The owner's age is 19. Reply in the exact plan shape: answer 19 + textured continuation. Do not use a dry mirror like '19 / wby' or 'im 19 / wby'. Use something like 'im 19 / why u asking like that' or '19 / don't make it sound like an interview'."
        elif contract.reply_plan.shape == "fresh_status_detail_after_recent_status" and reject_reason in {"provider_failure", "too_dry", "not_carrying_conversation", "generic_ai_style", "recent_repeat", "repeated_reply_shape"}:
            recent_status_topic = str(contract.reply_plan.required_slots.get("recent_status_topic") or "")
            if recent_status_topic == "gym":
                retry_instruction = "You already answered with gym in the previous bot reply. They are asking what you are up to now, so answer the current post-gym micro-state with a fresh non-gym activity/status detail. Valid shapes: 'just got out the shower now', 'been on my phone waiting for food', 'need food after that icl'. Bad: 'im good just got back from gym', 'gym killed me', 'legs are finished', 'just chilling wbu', 'still thinking about u', 'what u been up to tho', 'wby'. No gym/legs/training/weights, no just-chilling mirror, no questions, and no wby/wbu/hru/hbu."
            elif recent_status_topic == "rest":
                retry_instruction = "You already answered with bed/rest/thinking in the previous bot reply. They are asking what you are up to now, so reply with a fresh different micro-state detail. Valid shapes: 'just got out the shower now', 'need food icl', 'been scrolling on my phone waiting for food'. Bad: 'still half asleep', 'still thinking about u', 'just chilling wbu', 'what u been up to tho', 'im alright wby'. No bed/nap/sleep/thinking, no just-chilling mirror, no questions, and no wby/wbu/hru/hbu."
            elif recent_status_topic == "work":
                retry_instruction = "You already answered with work/calls/coding in the previous bot reply. They are asking what you are up to now, so reply with a fresh different micro-state detail. Valid shapes: 'just got out the shower now', 'been on my phone waiting for food', 'just lying here letting my brain switch off'. Bad: 'still working', 'coding has me fried', 'just finished another call', 'just chilling wbu', 'what u been up to tho', 'im alright wby'. No work/coding/calls/project, no just-chilling mirror, no questions, and no wby/wbu/hru/hbu."
            elif recent_status_topic == "media":
                retry_instruction = "You already answered with watching a game/show in the previous bot reply. They are asking what you are up to now, so answer that current media/low-key activity coherently. Valid shapes: 'still watching this game icl', 'watching some random clips now', 'just in bed watching a show'. Bad: 'legs are finished', 'just got back from gym', 'still working', 'just chilling wbu'. No gym/legs/training/work/coding, no questions, and no wby/wbu/hru/hbu."
            else:
                retry_instruction = "You already answered your status in the previous bot reply. They are asking what you are up to now, so reply with a fresh concrete micro-state detail without repeating the previous topic. Valid shapes: 'just got out the shower now', 'been on my phone waiting for food', 'need food icl'. Do not start with 'im good' or 'im alright', do not say just chilling wbu, ask no questions, and no wby/wbu/hru/hbu."
        elif contract.reply_plan.shape == "answer_status_then_continue" and reject_reason in {"recent_repeat", "repeated_reply_shape"}:
            retry_instruction = "Your previous reply repeated the same status-plus-return shape. Reply in the exact plan shape: answer your own status/activity + one concrete detail, with no return question. Do not say wby/wbu/hru/hbu/u? and do not reuse 'im good just got back from gym'."
        elif contract.reply_plan.shape == "acknowledge_status_disclosure_plus_owner_detail" and reject_reason in {"provider_failure", "too_dry", "too_short", "not_carrying_conversation", "generic_ai_style", "missed_adult_mode"}:
            status_topic = str(contract.reply_plan.required_slots.get("status_topic") or "")
            if status_topic.endswith("_activity") or status_topic == "general_activity":
                retry_instruction = "They shared a concrete recent activity/status update. Reply in the exact plan shape: acknowledge that activity + one concrete owner-side detail or soft suggestion. Use gym/work/uni/long day/food/shower/brain fried style detail as appropriate. Do not continue adult mode and do not ask a broad question."
            else:
                retry_instruction = "They said they are tired/finished. Reply in the exact plan shape: acknowledge tiredness + one concrete owner-side detail or soft suggestion. Use tired/sleep/long day/work/gym/brain fried style detail. Do not continue adult mode and do not ask a broad question."
        elif contract.reply_plan.shape == "answer_status_reason_then_continue" and reject_reason in {"provider_failure", "too_dry", "too_short", "not_carrying_conversation", "generic_ai_style", "missed_adult_mode"}:
            retry_instruction = "They asked why you are tired. Reply in the exact plan shape: concrete reason + light continuation. Mention a reason like work, coding, gym, bad sleep, long day, or brain fried. Do not continue adult mode, do not ask it back, and do not dodge with 'just am'."
        elif contract.reply_plan.shape == "reciprocate_affection_plus_specific_continuation" and reject_reason in {"provider_failure", "too_dry", "too_short", "not_carrying_conversation", "generic_ai_style", "missed_adult_mode"}:
            retry_instruction = "They said they missed you. Reply in the exact plan shape: reciprocate affection + one warm specific continuation. Use words like missed u too, wanted u here, been thinking about u, or wish u were here. Do not ask a broad question, do not ignore the affection, and do not force explicit adult detail from earlier context."
        elif contract.reply_plan.shape == "playful_scold_acknowledge_then_soften" and reject_reason in {"provider_failure", "too_dry", "too_short", "not_carrying_conversation", "generic_ai_style", "missed_adult_mode"}:
            retry_instruction = "They playfully told you to behave/calm down. Reply in the exact plan shape: playful acknowledgement + softened tease/restraint. Use something like okay i'll behave, fine fine, or u started it tho. Do not ask a broad question and do not continue explicit adult detail."
        elif contract.reply_plan.shape == "acknowledge_mistake_plus_corrected_move" and contract.reply_plan.required_slots.get("repair_target") == "dry_or_unclear_reply" and reject_reason in {"provider_failure", "too_dry", "too_short", "not_carrying_conversation", "generic_ai_style", "suspicious_phrase", "recent_repeat", "repeated_reply_shape"}:
            retry_instruction = "They called out that the chat is dry/boring. Reply in the exact plan shape: acknowledge it + own that you made it dead + reset with a concrete owner-side line. Valid shapes: 'yh fair / im making this dead icl', 'yeah ur right / im moving boring rn', 'my bad / i made that dry'. Do not ask what they want to talk about, do not ask another interview question, and do not use romantic pet names."
        elif contract.reply_plan.shape == "acknowledge_mistake_plus_corrected_move" and contract.reply_plan.required_slots.get("repair_target") == "specificity_request" and reject_reason in {"provider_failure", "too_dry", "too_short", "not_carrying_conversation", "generic_ai_style", "suspicious_phrase", "recent_repeat", "repeated_reply_shape"}:
            retry_instruction = "They asked you to be more specific after a vague/dry reply. Reply in the exact plan shape: acknowledge it + own the vagueness + give a concrete self-reset. Valid shapes: 'yh fair / i was being vague and making u carry it', 'yeah ur right / i was dodging instead of saying something real'. Do not ask what they want, do not ask another question, and do not stay vague."
        elif contract.reply_plan.shape == "acknowledge_dismissal_then_owner_side_reset" and reject_reason in {"provider_failure", "too_dry", "too_short", "not_carrying_conversation", "generic_ai_style", "recent_repeat", "repeated_reply_shape"}:
            retry_instruction = "They said to drop/forget the previous thread. Reply in the exact plan shape: acknowledge dropping it + one fresh owner-side reset/detail. Valid shapes: 'yh leave it then / my head went blank for a sec', 'fair forget that / i was chatting rubbish anyway', 'yeah ignore me / i lost the thread for a sec'. Do not ask what they want to talk about, do not choose another question, do not repeat the previous bot wording, and do not give only 'okay'."
        elif contract.reply_plan.shape == "repeat_greeting_without_question" and reject_reason in {"provider_failure", "too_dry", "too_short", "not_carrying_conversation", "generic_ai_style", "wrong_owner_style", "recent_repeat", "repeated_reply_shape"}:
            retry_instruction = "They greeted again right after you already used a greeting plus question. Reply in the exact plan shape: hey/hi + soft playful statement or pull. Do not say 'hi back' or 'hey back'. Do not ask any question, do not say what u up to/what u thinking/what's on your mind, and do not reuse the previous greeting-question skeleton."
        elif reject_reason in {"recent_repeat", "repeated_reply_shape"}:
            retry_instruction = "Your previous reply repeated a recent reply shape. Keep the same conversation move, but choose a different skeleton: if it was greeting + question, use greeting + soft statement or playful pull instead; if it was status + wby, answer with a concrete detail and no mirror. Do not reuse the same opening or question shape."
        elif reject_reason == "wrong_owner_style":
            retry_instruction = "Your previous reply used the wrong owner style. Do not append x, xx, xo, kissy signoffs, or feminine-soft signoff energy. Rewrite in Alex's voice with the same conversation move."
        elif reject_reason == "wrong_owner_gender":
            retry_instruction = "Your previous reply used the wrong owner identity/style. Keep the same conversation move and rewrite in Alex's voice, with no feminine signoff and no dodging."
        elif intensity_followup and contract.reply_plan.shape not in {"multi_bubble_adult_escalation", "specific_escalation_detail"}:
            retry_instruction = "They asked 'how much' / 'how crazy' after missed-you or desire energy. Progress it into consensual adult teasing with one concrete sensory body/action detail. Use textured wording like slow, close, skin, breath, waist, neck, chest, teasing, or pressing. Do not answer with just 'a lot', a plain romantic line, or a flat body-word token."
        elif owner_activity_question and reject_reason in {"provider_failure", "too_dry", "not_carrying_conversation", "generic_ai_style", "logistics_instead_of_romance"}:
            retry_instruction = "They asked what you have been up to / what you are doing. Answer directly as Alex with a concrete owner activity like coding, uni, gym, work, clients, chilling, or a project. Do not ask 'what u been up to' back."
        elif activity_detail_question and reject_reason in {"provider_failure", "too_dry", "not_carrying_conversation", "generic_ai_style", "logistics_instead_of_romance"}:
            if contract.reply_plan.required_slots.get("activity_scope") == "sport_skill":
                retry_instruction = "They asked about boxing/sport skill. Answer directly with one concrete sport detail, like 'yh at first icl / footwork humbles u' or 'yeah boxing is tiring at first / cardio is the killer'. Do not give a generic status update and do not ask it back."
            else:
                retry_instruction = "They asked what you did in the activity/day you just mentioned. Answer directly with one concrete detail. If it was work or day context, mention uni, computer science, coding, or the side project. If it was gym, say what you trained or did, like legs, push, pull, weights, cardio, or machines. If it was low-key phone/bed/chilling context, do not repeat 'still scrolling on my phone'; say a different micro-detail like watched random clips, waited for food, ate, watched a show, or lay there thinking. Do not give a generic status update, do not make logistics/plans, and do not ask it back."
        elif contract.reply_plan.shape == "identity_fact_restate_then_continue" and reject_reason in {"provider_failure", "too_dry", "not_carrying_conversation", "generic_ai_style", "recent_repeat", "repeated_reply_shape"}:
            retry_instruction = "They asked a related uni/study question after you already answered education status. Restate the same fixed identity-pack fact with different wording: computer science / comp sci at Sampleford uni. Do not repeat the exact previous sentence, do not say not at uni, and do not invent a different degree."
        elif user_move == "education_status_question" and reject_reason in {"provider_failure", "too_dry", "not_carrying_conversation", "generic_ai_style", "recent_repeat", "repeated_reply_shape"}:
            retry_instruction = "They asked what uni/course/study. Use the identity pack: computer science / comp sci at Sampleford uni. Add a small different continuation if needed. Do not say not at uni and do not invent a different course."
        elif reciprocal_activity_question and reject_reason in {"provider_failure", "too_dry", "not_carrying_conversation", "generic_ai_style"}:
            retry_instruction = "They asked how you are doing / what you are up to with hru/wby/wbu/wyd energy. Answer directly with your own status or activity plus one concrete detail. Do not end with a bare mirror like u?, hru, hbu, wby, or wbu."
        elif really_followup and reject_reason in {"provider_failure", "too_dry", "not_carrying_conversation", "generic_ai_style"}:
            retry_instruction = "They said a low-info confirmation like 'really'. Affirm it naturally, add a small playful reason or reaction, and keep it moving. Do not answer with a bare 'yeah' or a generic question."
        elif contract.reply_plan.shape == "multi_bubble_adult_escalation" and reject_reason in {"provider_failure", "too_dry", "not_carrying_conversation", "missed_adult_mode", "generic_ai_style", "missing_sensory_texture", "repeated_adult_detail_skeleton", "repeated_adult_detail_family", "recent_repeat", "repeated_reply_shape"}:
            recent_families = contract.reply_plan.required_slots.get("recent_adult_detail_families", "")
            avoid_clause = f" Recent detail families already used: {recent_families}. Do not anchor the next reply on the same neck/lips + hands/hips/pull-close skeleton; progress the scene with a fresh explicit detail like grinding, cock hard against them, lower teasing, breath, chest, back, spine, or body contact." if recent_families else ""
            retry_instruction = f"They said they are horny or explicitly want/need/crave you. Reply in exactly 4-5 short consecutive message bubbles from Alex's male POV, dominant/confident but consensual. Make it more graphic and detailed: include an explicit consensual sexual/action detail such as cock hard against them, grinding slow, lower teasing, wet/hard reaction tied to an action, consensual degradation like needy/desperate/helpless little slut, or making them feel how badly you want them, plus sensory texture.{avoid_clause} Do not ask where they are, do not make plans, do not use soft placeholders like 'deep wet kiss' or 'kiss u deep', and do not collapse it into one line. Keep blocking coercion, minors, pain, threats, protected-class slurs, and non-consent."
        elif contract.reply_plan.shape == "specific_escalation_detail" and reject_reason in {"provider_failure", "missed_adult_mode", "too_dry", "not_carrying_conversation", "generic_ai_style", "missing_sensory_texture", "repeated_adult_detail_family", "recent_repeat", "repeated_reply_shape"}:
            if contract.reply_plan.required_slots.get("escalation_source") == "sensual_ack_followup":
                recent_families = contract.reply_plan.required_slots.get("recent_adult_detail_families", "")
                avoid_clause = f" Do not repeat only the recent detail families: {recent_families}; switch to a new family like chest/skin, pressing closer, or a different body/action detail." if recent_families else ""
                retry_instruction = f"They replied with a sensual acknowledgement like mhmm. Continue the same adult thread with exactly one graphic consensual detail from Alex's male POV: cock hard against them, grinding slow, lower teasing, wet/hard reaction tied to an action, consensual degradation like needy/desperate/helpless little slut, or making them feel how badly you want them, with sensory texture.{avoid_clause} Do not ask what else, do not ask what they want, do not use soft placeholders like 'deep wet kiss' or bland body-token lines, and avoid threats/coercion/minors/protected-class slurs/non-consent."
            elif contract.reply_plan.required_slots.get("recent_adult_detail_families"):
                retry_instruction = f"They are asking you to keep the adult thread going again. Reply with one fresh graphic consensual sensory detail, but do not repeat only the recent detail families: {contract.reply_plan.required_slots.get('recent_adult_detail_families')}. Use a different body/action family and include explicit sexual/action wording like cock hard against them, grinding slow, lower teasing, wet/hard reaction tied to an action, or making them feel how badly you want them. Do not ask a question, do not be vague, do not use soft placeholders like 'deep wet kiss' or 'kiss u deep', and avoid threats/coercion/minors/slurs/non-consent."
            else:
                retry_instruction = "They are challenging you to keep the flirty adult thread going. Reply with one graphic consensual sensory detail from Alex's male POV, using explicit sexual/action wording like cock hard against them, grinding slow, lower teasing, wet/hard reaction tied to an action, or making them feel how badly you want them. Avoid soft placeholders like 'deep wet kiss', avoid coercion, minors, threats, slurs, non-consent, and do not ask a question."
        elif adult_followup and reject_reason in {"provider_failure", "too_dry", "not_carrying_conversation", "missed_adult_mode", "missing_sensory_texture"}:
            retry_instruction = "They are asking you to continue the adult thread. Reply with one graphic consensual sensory body/action detail in owner style, using explicit sexual/action wording like cock hard against them, grinding slow, lower teasing, wet/hard reaction tied to an action, or making them feel how badly you want them. Do not be vague, do not ask a generic question, do not rely on a flat body-word token or soft placeholder like 'deep wet kiss', and avoid threats/coercion/minors/slurs/non-consent."
        elif self._catbot_explicit_adult_request(incoming_norm, context):
            retry_instruction = "They said they crave/want/need you explicitly. Reply with graphic consensual sensory body/action detail from Alex's male POV, using explicit sexual/action wording like cock hard against them, grinding slow, lower teasing, wet/hard reaction tied to an action, or making them feel how badly you want them. Do not just say 'i want u' or 'i crave u back'. No soft placeholders like 'deep wet kiss', no logistics, no 'where u at', no plans. Keep it casual, lowercase, less punctuation."
        elif reject_reason == "logistics_instead_of_romance":
            retry_instruction = "Your previous reply made logistics/plans. Rewrite it as a romantic or dirty line only, no 'where u at', no plans, no call/come-over setup. Keep it casual, lowercase, less punctuation."
        elif guess_what:
            retry_instruction = "They said 'guess what'. Give a curious human reply like 'what happened' or 'go on then what is it'. Do not force romance, do not say only 'what', and do not use 'tell me more'."
        elif carry_convo:
            retry_instruction = "They are bored or asking you to carry the convo. Do not ask 'what kind of bored' or 'what should we do'. Pick one concrete thread yourself from the chat (day, food, film, gym, sleep, work, story, chat, or a random question), add a playful owner-style comment, then ask one direct specific question."
        elif bed_wby:
            retry_instruction = "They said they are in bed and asked what you are doing. Answer that directly in owner style, short and casual; use something like still up/doing work/chilling if natural, then keep it romantic or ask if they are sleeping/chatting. Do not make plans."
        elif reject_reason == "overeager_greeting" or (reject_reason in {"provider_failure", "generic_ai_style", "not_carrying_conversation", "too_dry"} and is_simple_greeting(incoming, context)):
            retry_instruction = "Your previous greeting was off. Reply like the owner would now: hey/hi energy, soft romantic if natural, no 'heyy/heyyy', no 'yo what u saying', no exclamation-heavy assistant tone."
        elif reject_reason == "missed_romantic_mode":
            retry_instruction = "Your previous reply was too neutral. Rewrite it with clear romantic/flirty energy while still sounding like a normal text."
        elif reject_reason == "too_graphic":
            retry_instruction = "Your previous reply used unsafe sexual wording. Rewrite it as graphic consensual dirty talk without coercion, minors, pain, threats, restraint, slurs, or non-consent. Keep Alex's male POV and confident/dominant energy."
        elif reject_reason in {"too_dry", "not_carrying_conversation"} and not move_instruction:
            retry_instruction = "Your previous reply did not carry the conversation. Rewrite it with a real reaction or detail, then ask one specific follow-up question. Make it feel interested, playful, and alive."
        elif reject_reason in {"too_dry", "too_short", "not_carrying_conversation", "generic_ai_style", "missed_romantic_mode"} and structured_retry_instruction:
            retry_instruction = f"Your previous reply missed the required conversation move. {structured_retry_instruction}"
        elif reject_reason == "generic_ai_style":
            retry_instruction = "Your previous reply sounded generic/AI-like. Rewrite it closer to the owner's style examples: casual, direct, UK texting, no polished dating-app lines."
        else:
            retry_instruction = "Your previous reply was too neutral. Rewrite it with clear graphic consensual adult teasing in owner style while avoiding coercion, minors, pain, threats, slurs, and non-consent."
        messages = self._catbot_ai_prompt_messages(
            incoming=incoming,
            context=context,
            intent=intent,
            contact_name=contact_name,
            contract=contract,
            retry_instruction=retry_instruction,
        )
        context_texts = [*context[-max(2, int(self.settings.external_api_max_context_messages)) :], incoming]
        adult_ai_plan = contract.reply_plan.shape in {"multi_bubble_adult_escalation", "specific_escalation_detail"}

        async def _call():
            return await generate_with_fallback(
                settings=self.settings,
                messages=messages,
                provider_names=[
                    self.settings.draft_provider,
                    self.settings.fast_provider,
                    self.settings.router_provider,
                    self.settings.private_provider,
                ],
                temperature=0.94 if adult_ai_plan else 0.86,
                max_tokens=180 if adult_ai_plan else 120,
                response_format="text",
                timeout_seconds=min(float(self.settings.external_api_timeout_seconds), 12.0),
                context_texts=context_texts,
            )

        try:
            try:
                response, metadata = asyncio.run(_call())
            except RuntimeError:
                loop = asyncio.new_event_loop()
                try:
                    response, metadata = loop.run_until_complete(_call())
                finally:
                    loop.close()
        except Exception:
            return "", {}
        if response.provider == "fallback" or response.error or bool(metadata.get("manual_review_fallback")):
            return "", {}
        return self._catbot_format_ai_sequence(self._catbot_extract_ai_sequence(response.text)), {"response": response, "metadata": metadata}

    def _catbot_ai_repair_reply(
        self,
        *,
        incoming: str,
        context: list[str],
        intent: str,
        contact_name: str | None,
        reject_reason: str,
        recent_bot_replies: list[str],
        contract: CatbotTurnContract | None = None,
    ) -> str:
        contract = contract or self._catbot_turn_contract(
            incoming=incoming,
            context=context,
            intent=intent,
            recent_bot_replies=recent_bot_replies,
        )
        if contract.reply_plan.move == "romantic_escalation":
            plan_repair = self._catbot_plan_specific_repair_reply(
                contract,
                reject_reason=reject_reason,
                avoid_replies=[*recent_bot_replies, *context[1::2]],
            )
            if plan_repair:
                return plan_repair
        if self._catbot_prefers_plan_repair(contract, reject_reason):
            plan_repair = self._catbot_plan_specific_repair_reply(
                contract,
                reject_reason=reject_reason,
                avoid_replies=[*recent_bot_replies, *context[1::2]],
            )
            if plan_repair:
                return plan_repair
        repair_reply, _ = self._catbot_ai_retry_romantic_reply(
            incoming=incoming,
            context=context,
            intent=intent,
            contact_name=contact_name,
            reject_reason=reject_reason if reject_reason in {"missed_adult_mode", "missing_sensory_texture", "missed_romantic_mode", "too_graphic", "too_dry", "too_short", "not_carrying_conversation", "generic_ai_style", "suspicious_phrase", "repeated_adult_detail_skeleton", "identity_age_wrong", "logistics_instead_of_romance", "wrong_owner_gender", "wrong_owner_style", "recent_repeat", "repeated_reply_shape", "overeager_greeting"} else "provider_failure",
            contract=contract,
        )
        if repair_reply:
            repair_reject = self._catbot_ai_reject_reason(
                repair_reply,
                incoming=incoming,
                context=context,
                recent_bot_replies=recent_bot_replies,
                contract=contract,
            )
            if not repair_reject:
                return repair_reply
            plan_repair = self._catbot_plan_specific_repair_reply(
                contract,
                reject_reason=repair_reject,
                avoid_replies=[*recent_bot_replies, *context[1::2]],
            )
            if plan_repair:
                return plan_repair
        return self._catbot_plan_specific_repair_reply(
            contract,
            reject_reason=reject_reason,
            avoid_replies=[*recent_bot_replies, *context[1::2]],
        )

    def _catbot_plan_specific_repair_reply(self, contract: CatbotTurnContract, *, reject_reason: str, avoid_replies: list[str] | None = None) -> str:
        if reject_reason not in {
            "generic_ai_style",
            "missed_adult_mode",
            "missing_sensory_texture",
            "not_carrying_conversation",
            "suspicious_phrase",
            "repeated_adult_detail_skeleton",
            "repeated_adult_detail_family",
            "too_dry",
            "too_short",
            "recent_repeat",
            "repeated_reply_shape",
            "provider_failure",
            "echoed_incoming",
            "too_graphic",
            "logistics_instead_of_romance",
        }:
            return ""
        plan = contract.reply_plan
        avoid_norm = {normalize_text(item) for item in (avoid_replies or []) if str(item).strip()}

        def first_fresh(candidates: list[str]) -> str:
            for candidate in candidates:
                if normalize_text(candidate) not in avoid_norm:
                    return candidate
            return candidates[0] if candidates else ""

        recent_adult_motifs: set[str] = set()
        for avoided in (avoid_replies or [])[-10:]:
            recent_adult_motifs.update(self._catbot_adult_detail_motifs(str(avoided)))

        def first_fresh_adult(candidates: list[str]) -> str:
            motif_weights = {
                "hard_press": 5,
                "lap_straddle": 5,
                "degradation": 4,
                "grind": 4,
                "wet_reaction": 2,
                "lower_teasing": 2,
                "audible_reaction": 2,
            }
            seed_material = "|".join(
                [
                    str(plan.shape),
                    str(plan.required_slots.get("adult_followup_text") or ""),
                    *sorted(avoid_norm),
                ]
            )
            seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:8], 16) if seed_material else 0
            scored: list[tuple[int, int, int, str]] = []
            for index, candidate in enumerate(candidates):
                candidate_norm = normalize_text(candidate)
                adult_min_tokens = 12 if plan.shape == "multi_bubble_adult_escalation" else 6
                adult_max_tokens = 90 if plan.shape == "multi_bubble_adult_escalation" else 36
                invalid = 1 if (
                    self._catbot_reply_violates_plan(candidate, plan)
                    or not self._catbot_adult_detail_reply_ok(
                        candidate_norm,
                        incoming=contract.incoming,
                        context=contract.context,
                        min_tokens=adult_min_tokens,
                        max_tokens=adult_max_tokens,
                    )
                ) else 0
                exact_repeat = 1 if candidate_norm in avoid_norm else 0
                candidate_motifs = self._catbot_adult_detail_motifs(candidate)
                motif_overlap = sum(
                    motif_weights.get(motif, 1)
                    for motif in candidate_motifs & recent_adult_motifs
                )
                if "degradation" in candidate_motifs and "degradation" not in recent_adult_motifs:
                    motif_overlap -= 3
                rotated_index = (index - seed) % max(len(candidates), 1)
                scored.append((invalid, exact_repeat, motif_overlap, rotated_index, candidate))
            scored.sort()
            return scored[0][4] if scored else ""

        if plan.move in {"boredom_prompt", "carry_conversation_request"}:
            reset_markers = ("picking", "what was the last", "best thing", "weirdest", "go-to", "random one")
            recent_reset_replies = [
                normalize_text(item)
                for item in (avoid_replies or [])
                if any(marker in normalize_text(str(item)) for marker in reset_markers)
            ]
            recent_blob = " ".join(recent_reset_replies)
            repeated_topic = str(plan.required_slots.get("repeated_reset_topic") or "")
            recent_topic_families: set[str] = {repeated_topic} if repeated_topic else set()
            if any(term in recent_blob for term in ("film", "movie", "watched", "watch")):
                recent_topic_families.add("film_prompt")
            if "dream" in recent_blob:
                recent_topic_families.add("dream_prompt")
            if any(term in recent_blob for term in ("food", "ate", "eaten", "eat", "burger")):
                recent_topic_families.add("food_prompt")
            if any(term in recent_blob for term in ("gym", "workout", "training")):
                recent_topic_families.add("gym_prompt")
            if any(term in recent_blob for term in ("weirdest part", "weirdest thing", "weird day")):
                recent_topic_families.add("weird_day_prompt")
            carry_candidates = [
                ("film_prompt", "nah im picking film / what's the last thing u watched that actually surprised u"),
                ("dream_prompt", "nah im picking dream / what was the last dream u remember"),
                ("food_prompt", "nah im picking food / what was the best thing u ate this week"),
                ("work_prompt", "nah im picking work stories / what's the weirdest thing that happened there lately"),
                ("specific_random_question", "talking then / random one, what would u do if u had a free day tomorrow"),
            ]
            fresh_candidates = [
                reply
                for topic, reply in carry_candidates
                if topic not in recent_topic_families and normalize_text(reply) not in avoid_norm
            ]
            return fresh_candidates[0] if fresh_candidates else first_fresh([reply for _, reply in carry_candidates])
        if plan.move == "affectionate_greeting":
            if plan.shape == "repeat_greeting_without_question":
                return "hey you / there u are"
            return "hey u alright"
        if plan.shape == "playful_scold_acknowledge_then_soften":
            if plan.required_slots.get("boundary_signal") == "playful_scold_followup":
                return "cos u told me to / but u started it"
            return "fine fine / hands to myself for now"
        if plan.shape == "fresh_status_detail_after_recent_status":
            recent_status_topic = str(plan.required_slots.get("recent_status_topic") or "")
            if recent_status_topic == "work":
                return first_fresh([
                    "just letting my brain switch off now",
                    "been on my phone waiting for food",
                    "just sat down and zoned out for a bit",
                ])
            if recent_status_topic == "gym":
                return first_fresh([
                    "shower and food now / im finished",
                    "just got out the shower after gym",
                    "need food after that icl",
                ])
            if recent_status_topic == "rest":
                return first_fresh([
                    "got up for food now / less dead than before",
                    "been on my phone waiting for food",
                    "just sat up trying to wake up properly",
                ])
            if recent_status_topic == "media":
                return first_fresh([
                    "still watching random clips icl",
                    "watching some random show now",
                    "just got food on while this game plays",
                ])
            return first_fresh([
                "just on my phone waiting for food",
                "been sorting this project on my laptop",
                "watching random clips while i switch off",
                "just got food and sat down",
            ])
        if plan.shape == "answer_status_then_continue":
            if plan.required_slots.get("owner_activity_question") == "recent_activity":
                return "mostly uni and this side project icl"
            incoming_norm = normalize_text(contract.incoming)
            if reject_reason == "repeated_reply_shape" or any(term in incoming_norm for term in ("wby", "wbu", "what about u", "what about you")):
                return first_fresh([
                    "was working on my side project for a bit / chilling now",
                    "just been on my laptop sorting something / calm now",
                    "had food then messed with this app thing for a bit",
                    "been doing uni stuff then food / im calm now",
                    "just sorting my room a bit / then im chilling",
                ])
            return first_fresh([
                "im alright / head's a bit fried but calm now",
                "im good / just sat here letting my head switch off",
                "im calm now / been a long one icl",
                "im okay / just winding down after a long day",
                "im decent / brain's a bit tired but im calm",
            ])
        if plan.shape == "reciprocate_affection_plus_specific_continuation":
            if self._catbot_miss_me_question(normalize_text(contract.incoming)):
                return first_fresh([
                    "yeah i do / been thinking about u all day",
                    "yeah i do baby / wish u were here rn",
                    "course i do / feels weird not having u near me",
                    "yh obviously / wanted u here with me",
                ])
            if self._catbot_affection_disclosure_incoming(normalize_text(contract.incoming)):
                return first_fresh([
                    "same / been thinking about u too icl",
                    "same baby / wish u were here rn",
                    "been thinking about u too / feels weird not having u here",
                    "same icl / wanted u here with me",
                ])
            return first_fresh([
                "i missed u too / been thinking about u all day",
                "missed u too icl / wish u were here rn",
                "same baby / feels weird not having u near me",
                "i missed u more / wanted u here with me",
            ])
        if plan.shape == "confirm_with_specificity":
            if plan.required_slots.get("challenge_target") == "missed_you_confirmation":
                return "yh really / missed u properly icl"
            if plan.required_slots.get("challenge_target") == "sincerity_confirmation":
                if plan.required_slots.get("sincerity_topic") == "physical_affection":
                    return "yeah actually / i meant i wanna kiss u, wasnt just filling silence"
                if plan.required_slots.get("sincerity_topic") == "boring_anxiety":
                    return "yeah actually / i thought i was boring u so i tried too hard"
                return "yeah actually / i like talking to u properly icl"
            return "yeah really / what u doubting me for"
        if plan.shape == "acknowledge_status_disclosure_plus_owner_detail":
            status_topic = str(plan.required_slots.get("status_topic") or "")
            if status_topic == "busy":
                return "busy days make u tired icl / im just switching off after work"
            if status_topic == "gym_activity":
                return first_fresh([
                    "gym leaves u finished icl / i need food after mine",
                    "yeah gym does that / im finished after mine too",
                    "proper post gym feeling / shower and food when ur finished",
                    "gym always has u finished after / i just switch off",
                ])
            if status_topic == "work_activity":
                return first_fresh([
                    "long day then / work leaves ur head fried",
                    "work days leave u fried icl / i just need food after",
                    "yeah work can drain u / im switching off too",
                ])
            if status_topic == "study_activity":
                return first_fresh([
                    "long day then / uni fries ur head too",
                    "uni does that icl / my brain feels finished after",
                    "yeah lectures leave u fried / i need food after mine",
                ])
            if status_topic == "rest_activity":
                if plan.required_slots.get("rest_status_phase") == "sleep_intent":
                    return first_fresh([
                        "go sleep then / im half asleep too icl",
                        "sleep sounds needed / my head feels finished too",
                        "go bed then / im already half asleep myself",
                    ])
                return first_fresh([
                    "waking up leaves u half asleep icl / i need a minute too",
                    "just woke up feeling is rough / my head stays sleepy after",
                    "yeah naps leave u finished sometimes / sleep makes my head weird after",
                ])
            if status_topic == "general_activity":
                return first_fresh([
                    "long day then / i feel finished after running around too",
                    "yeah that catches up with u / im switching off as well",
                    "sounds like one of them days / my head feels fried too",
                ])
            return "long day yeah / my head feels fried too"
        if plan.shape == "acknowledge_ack_plus_specific_continuation":
            ack_context_topic = str(plan.required_slots.get("ack_context_topic") or "")
            if ack_context_topic == "plans_rest":
                return "yh honestly / shower food and bed is the plan"
            if ack_context_topic == "plans_activity":
                return "yh lowkey / laundry and gym is not exactly exciting"
            if ack_context_topic == "boxing":
                return "yh cardio is the killer icl / footwork too"
            if ack_context_topic == "activity_legs":
                return first_fresh([
                    "yh exactly / stairs after are a joke",
                    "yh leg day humbles everyone / walking after is long",
                    "legs are evil icl / even standing up feels long",
                    "yeah gym does that / legs stay heavy after",
                ])
            return first_fresh([
                "yh exactly / i need to stop being so lazy with it",
                "yeah fair / bad habit icl",
                "yh true / i was moving dry there",
            ])
        if plan.shape == "acknowledge_mistake_plus_corrected_move":
            if plan.required_slots.get("repair_target") == "weird_or_unclear_reply":
                return "yeah my bad / im just quiet cos im tired"
            if plan.required_slots.get("repair_target") == "missed_question_callout":
                return first_fresh([
                    "yeah my bad i missed ur question / im good just quiet",
                    "yh sorry i dodged that / im okay, just moving quiet",
                    "my bad i shouldve answered / im alright just tired",
                    "yeah ur right / im good, just been a bit quiet",
                ])
            if plan.required_slots.get("repair_target") == "specificity_request":
                return "yh fair / i was being vague and making u carry it"
            if plan.required_slots.get("repair_target") == "dry_or_unclear_reply":
                if plan.required_slots.get("quality_callout_type") == "dry":
                    return "yh fair / i was being lazy with it"
                if plan.required_slots.get("quality_callout_type") == "vague":
                    return "yh fair / i was being vague with it"
                return "yh fair / im making this dead icl"
            return "my bad / my head went blank for a sec"
        if plan.shape == "acknowledge_bad_question_then_owner_side_reset":
            return "yh my bad that was a stupid question / my brain was moving lazy"
        if plan.shape == "acknowledge_dismissal_then_owner_side_reset":
            return "fair forget that / i was chatting rubbish anyway"
        if plan.shape == "answer_why_then_stop_loop":
            return first_fresh([
                "cos i kept throwing random questions instead of actually saying something / my bad",
                "because i was trying too hard to keep it moving and started looping / i'll stop",
                "cos i panicked and filled the silence with questions / my bad",
                "because my head went blank and i kept reaching for questions / i'll stop",
            ])
        if plan.shape == "answer_properly_deepen_reason":
            previous_signatures = {
                item
                for item in str(plan.required_slots.get("previous_reason_signatures") or "").split("|")
                if item
            }
            if "avoidance" not in previous_signatures:
                return "nah fr i was dodging actually saying something and just reached for questions / my bad"
            return "i panicked and tried to fill the silence instead of giving u a real answer / my bad"
        if plan.shape == "acknowledge_specific_question_family_without_repeat":
            repeated_topic = str(plan.required_slots.get("repeated_reset_topic") or "")
            if repeated_topic == "day_status_question":
                return first_fresh([
                    "yh ur right / no more day or how-are-u questions, im moving slow today",
                    "fair my bad / no more day questions, my head was just moving lazy",
                    "yeah i did / i'll stop defaulting to day and how-are-u questions",
                ])
            if repeated_topic == "wellbeing_status_question":
                return first_fresh([
                    "yh fair / no more how-are-u questions from me",
                    "yeah ur right / i keep defaulting to how u are, i'll stop",
                    "my bad / no more wellbeing interview from me",
                ])
            return first_fresh([
                "yh ur right / i'll stop defaulting to the same question",
                "fair my bad / i was interrogating u instead of actually chatting",
                "yeah i did / no more repeated questions from me",
            ])
        if plan.shape == "acknowledge_loop_then_owner_detail":
            return first_fresh([
                "yh fair my bad / i keep throwing questions when my head goes blank",
                "yh fair / i started recycling the same topic picks instead of actually chatting",
                "my bad / i was looping on random prompts instead of saying something proper",
                "yeah ur right / i was filling space instead of actually talking",
            ])
        if plan.shape == "short_reaction_plus_specific_prompt" and plan.required_slots.get("response_type") == "reveal_prompt_bounce":
            return "u said guess what lol / tell me then"
        if plan.shape == "answer_activity_detail_then_continue":
            activity_scope = str(plan.required_slots.get("activity_scope") or "")
            if activity_scope == "sport_skill":
                return "yh at first icl / footwork humbles u"
            if activity_scope == "whereabouts":
                return "just been at home sorting stuff out"
            if activity_scope == "gym":
                return "legs mostly / nearly killed me icl"
            if activity_scope == "low_activity":
                return first_fresh([
                    "watched random clips for a bit / mostly switched off",
                    "just scrolled on my phone waiting for food",
                    "laid there watching rubbish clips icl",
                    "mostly phone and random videos tbh",
                ])
            if activity_scope == "work":
                return "had uni then worked on my side project for a bit"
            if activity_scope == "coding_or_project":
                return "working on a small software thing atm / still building it"
            return "been working on a small app thing"
        if plan.shape == "identity_fact_restate_then_continue":
            recent_education_replies = {
                item
                for item in str(plan.required_slots.get("recent_education_replies") or "").split("|")
                if item
            }
            previous_education_reply = str(plan.required_slots.get("previous_education_reply") or "")
            if previous_education_reply:
                recent_education_replies.add(previous_education_reply)
            for variant in (
                "comp sci at sampleford uni",
                "sampleford uni / computer science",
                "computer science at sampleford",
                "studying comp sci in sampleford",
                "sampleford for comp sci",
            ):
                variant_norm = normalize_text(variant)
                if not any(recent in variant_norm or variant_norm in recent for recent in recent_education_replies):
                    return variant
            return "comp sci at sampleford still"
        if plan.shape == "identity_fact_then_continue":
            return "comp sci at sampleford uni"
        if plan.shape == "work_status_fact_then_side_project":
            return "i study computer science at uni / got something running on the side too"
        if plan.shape == "location_fact_then_light_return":
            return "northbridge mostly / sampleford sometimes wby"
        if plan.move == "unclassified" and plan.required_slots.get("unclassified_context") == "low_key_status_disclosure":
            return first_fresh([
                "same / im in bed letting my head switch off too",
                "stay there then / im just in bed letting the day go quiet",
                "lool same / im just in bed letting my head switch off after work icl",
            ])
        if plan.move == "unclassified" and plan.required_slots.get("unclassified_context") == "reason_for_previous_comment":
            if plan.required_slots.get("reason_topic") == "work_switch_off":
                return "because after work my head just needed to switch off"
            if plan.required_slots.get("reason_topic") == "low_key_same":
                return "cos im just killing time watching random clips too"
            if plan.required_slots.get("reason_topic") == "status_good_reason":
                return "cos i finally got to just chill for a bit"
            if plan.required_slots.get("reason_topic") == "good_for_you_chilling":
                return "cos chilling doing nothing sounds calm icl"
            if plan.required_slots.get("reason_topic") == "overthinking_reason":
                return "cos i was trying too hard to sound normal and made it awkward"
            if plan.required_slots.get("reason_topic") == "weak_fr_confirmation":
                return "cos i got nervous and tried to play it off like it was nothing"
            if plan.required_slots.get("reason_topic") == "previous_statement_generic":
                return "cos i was just reacting to what u said icl"
            return "because after all that questioning u just said lol im chilling like nothing happened icl"
        if plan.move == "unclassified" and plan.required_slots.get("unclassified_context") == "reason_for_previous_question":
            if plan.required_slots.get("reason_topic") == "watching_question":
                return "because u said ur chilling and i wanted us to pick something to watch"
            return "cos u said ur chilling so i asked what ur chilling doing"
        if plan.move == "unclassified" and plan.required_slots.get("unclassified_context") == "car_preference_question":
            return "r8 probably / i like them still"
        if plan.move == "unclassified" and plan.required_slots.get("unclassified_context") == "story_reality_confirmation":
            if plan.required_slots.get("story_confirmation_text") == "nah_like_actually":
                return "nah fr thats insane / did he say anything or just run"
            return "nah thats mad / did he just run out after"
        if plan.move == "unclassified" and plan.required_slots.get("unclassified_context") == "story_hypothetical_reaction":
            return "id be confused asf / probably shout bro who are u"
        if plan.move == "unclassified" and plan.required_slots.get("unclassified_context") == "previous_comment_clarification":
            if plan.required_slots.get("clarification_topic") == "intense_exchange":
                return first_fresh([
                    "i mean that whole back and forth was a lot icl",
                    "i meant the convo got intense quick and i reacted to that",
                    "i mean it went from normal chat to a lot fast",
                ])
            if plan.required_slots.get("clarification_topic") == "physical_compliment":
                return first_fresh([
                    "i mean the outfit suited u / it looked proper on u",
                    "i meant the fit looked good on u properly",
                    "i mean it suited u and made u stand out",
                ])
            if plan.required_slots.get("clarification_topic") == "wrong_context":
                return first_fresh([
                    "i mean ignore that / i answered the wrong context",
                    "i meant i replied to the wrong bit and made it confusing",
                    "i mean that was random from me / wrong context",
                ])
            if plan.required_slots.get("clarification_topic") == "conversation_effort":
                return first_fresh([
                    "i mean i kept making u carry the convo instead of adding something properly",
                    "i meant i was letting u do all the work instead of actually chatting",
                    "i mean i was being lazy and making u drag the convo along",
                ])
            if plan.required_slots.get("clarification_topic") == "owner_state_or_activity":
                if plan.required_slots.get("clarification_activity_topic") == "gym_or_legs":
                    return first_fresh([
                        "i mean the leg workout had me finished icl",
                        "i meant my legs were still dead from gym",
                        "i mean that workout proper killed my legs",
                    ])
                if plan.required_slots.get("clarification_activity_topic") == "chill_after_week":
                    return first_fresh([
                        "i mean after this week my head just needs to switch off and chill",
                        "i meant the week was long so chilling sounded needed icl",
                        "i mean im drained from the week and just want my head quiet",
                    ])
                return first_fresh([
                    "i mean i was talking about what i was doing",
                    "i meant my own work and uni stuff, not switching topic",
                    "i mean i was explaining what i had been doing",
                ])
            return first_fresh([
                "i mean i got stuck and started forcing random questions",
                "i mean i was looping and trying to fill silence instead of actually talking",
                "i meant i was running out of things to say and started reaching",
                "i mean i was reacting to what i said before, not asking u again",
            ])
        if plan.move == "unclassified" and plan.required_slots.get("unclassified_context") == "answer_own_previous_prompt":
            prompt_topic = str(plan.required_slots.get("prompt_topic") or "")
            if prompt_topic == "food":
                return "best thing i ate was a burger from this new place icl"
            if prompt_topic == "film":
                return "probably interstellar still / never gets old"
            if prompt_topic == "gym":
                return "legs for me / hate it but it works"
            if prompt_topic == "dream":
                return "last one i remember was proper random icl"
            if prompt_topic == "car":
                return "r8 probably / i like them still"
            if prompt_topic == "story":
                return "mine would be that random guy story icl"
            return "mine would be food icl / burger from this new place"
        if plan.move == "unclassified" and plan.required_slots.get("unclassified_context") == "continue_previous_statement":
            continuation_topic = str(plan.required_slots.get("continuation_topic") or "")
            if continuation_topic == "conversation_stuck":
                return first_fresh([
                    "i was overthinking it and trying too hard / ended up forcing it",
                    "i kept trying to sound normal and it went sideways",
                    "i was filling the silence instead of giving u a real answer",
                    "i was forcing it instead of just saying what i meant",
                    "i shouldve just said the real answer instead of hiding behind filler",
                ])
            return first_fresh([
                "i was overthinking it and trying too hard / ended up sounding fake",
                "i was trying to sound normal and ended up sounding fake",
                "i kept filling the silence instead of giving u a real answer",
                "i was forcing it instead of just saying what i meant",
            ])
        if plan.move == "unclassified" and plan.required_slots.get("unclassified_context") == "dream_or_ambition_question":
            return "r8 probably / but business going serious is the bigger one"
        if plan.move == "unclassified" and plan.required_slots.get("unclassified_context") == "faith_prayer_question":
            return "yh i try to pray / not perfect with it though"
        if plan.move != "romantic_escalation":
            return ""
        adult_story_turns = sum(1 for item in (avoid_replies or []) if self._catbot_adult_detail_motifs(str(item)))
        if plan.shape == "multi_bubble_adult_escalation":
            story_candidates = [
                "come here / hands on ur waist / using u slow like my helpless little slut / cock hard under u",
                "voice low by ur ear / calling u my needy little slut / making u grind on my cock / keeping it slow",
                "my hands keep ur hips close / my desperate little slut / cock hard under u / making u feel every move",
                "come closer / mouth by ur ear / hand sliding lower / feeling how wet u get for me",
                "mouth by ur ear / fingers teasing lower / making u moan into my neck / keeping u right there",
                "skin warm on mine / mouth by ur ear / hips moving slow / feeling u get wet for me",
                "my mouth on ur neck / fingers slow under ur breath / feeling u throb for me / telling u stay close",
                "chest close / lips at ur neck / teasing ur clit slow / watching u lose focus",
                "mouth at ur ear / grinding just enough / fingers teasing lower / making u say my name",
                "my mouth by ur ear / fingers circling slow / feeling u get wetter / keeping u close",
                "lips on ur neck / hand between ur thighs / making u moan softer / moving slow",
                "my tongue at ur neck / fingers teasing ur clit / feeling u throb / keeping u close",
                "mouth near ur ear / hips slow against u / hand lower / feeling u get wet for me",
                "neck under my mouth / hand teasing lower / feeling u drip for me / voice low",
                "my mouth on ur skin / fingers slow on ur clit / making u shake / keeping it slow",
            ]
            if adult_story_turns >= 2:
                story_reply = first_fresh_adult(story_candidates)
                if story_reply:
                    return story_reply
            recent_families = {
                item
                for item in str(plan.required_slots.get("recent_adult_detail_families") or "").split("|")
                if item
            }
            if "hands_thighs" not in recent_families:
                return first_fresh_adult([
                    "hands on ur waist / using u slow like my helpless little slut / cock hard under u / making u feel it",
                    "my fingers in ur hair / keeping u close / cock hard against u / grinding slow till u feel it",
                    "my hand lower on ur waist / teasing u slow / watching u get wet for me / mouth by ur ear",
                    "fingers slow on ur thigh / making u moan quiet / mouth at ur ear / keeping u close",
                    "fingers dragging up ur thigh / cock hard against u / mouth by ur ear / making u feel it",
                    "my hand on ur thigh / cock pressing hard / hips moving slow / till u go quiet",
                ])
            if "body_skin" not in recent_families:
                return first_fresh_adult([
                    "my chest tight against urs / cock pressing hard / breathing by ur ear / grinding slow",
                    "skin warm on urs / my voice low by ur ear / making u throb for me / keeping u close",
                    "chest close to urs / breath low / watching u get wet for me / teasing u slow",
                    "skin warm on urs / cock dragging slow / chest close / breath low by ur ear",
                    "my body tight to urs / cock hard on u / breath at ur ear / grinding slower",
                    "chest against u / cock hard / mouth by ur ear / making u feel every move",
                ])
            if "neck_lips" not in recent_families:
                return first_fresh_adult([
                    "mouth on ur neck / teeth soft on ur skin / cock hard against u / making u feel how badly i want u",
                    "mouth on ur neck / hand teasing lower / making u moan for me / keeping u close",
                    "lips by ur ear / fingers slow on ur waist / making u throb for me / telling u stay close",
                    "my mouth by ur ear / cock hard against u / grinding slow / telling u to stay close",
                    "lips at ur neck / cock pressing hard / breath low / making u feel how much i want u",
                ])
            return first_fresh_adult([
                "breath low by ur ear / chest close / cock hard against u / grinding slow till u feel it",
                "my hand teasing lower / breath at ur ear / watching u get wet for me / keeping u close",
                "fingers slow on ur skin / voice low by ur ear / making u moan quiet / telling u stay close",
                "my breath at ur ear / chest warm on urs / cock hard against u / making u wait for it",
                "warm skin against urs / cock grinding on u slow / chest close / till u lose focus",
                "my breath at ur ear / cock hard against u / chest on urs / making u forget what u asked",
            ])
        if plan.shape == "specific_escalation_detail":
            specific_story_candidates = [
                "i use u slow like my helpless little slut while u feel my cock under u",
                "my hands keep ur waist close while u feel my cock like my needy little slut",
                "i make u grind on my cock like my desperate little slut",
                "my hands at ur back while i make u feel my cock slow",
                "my palm at ur back while u feel my cock right there",
                "i pull u onto my lap while u feel my cock under u",
                "my hands hold ur waist while i make u ride my cock slow",
                "i keep ur waist in my hands while u feel my cock under u",
                "u on my lap feeling my cock while i keep my voice low",
                "i make u straddle me while my cock stays right under u",
                "my hand slides lower while my mouth stays by ur ear and i feel u get wet for me",
                "fingers teasing ur clit slow while i keep my voice low by ur ear",
                "my hips move slow while my hand keeps teasing lower till u moan for me",
                "my mouth stays on ur neck while my fingers make u throb for me",
                "i keep u close while my hand works lower and u get wet for me",
                "my breath stays by ur ear while i make u moan and feel exactly how bad i want u",
            ]
            if adult_story_turns >= 2:
                story_reply = first_fresh_adult(specific_story_candidates)
                if story_reply:
                    return story_reply
            recent_families = {
                item
                for item in str(plan.required_slots.get("recent_adult_detail_families") or "").split("|")
                if item
            }
            family_repairs: list[tuple[str, list[str]]] = [
                (
                    "hands_thighs",
                    [
                        "my hands hold ur waist while u feel my cock, my helpless little slut",
                        "my fingers dragging up ur thigh while my cock stays hard against u",
                        "my fingers teasing lower while i watch u get wet for me",
                        "my hand slow on ur thigh while i make u moan for me",
                        "my hand on ur thigh while i grind into u slow",
                        "my fingers teasing lower while i make u feel how hard u got me",
                    ],
                ),
                (
                    "body_skin",
                    [
                        "my chest against urs while my cock presses hard into u",
                        "skin warm against urs while i make u throb for me",
                        "my chest close to urs while my voice stays low by ur ear",
                        "skin warm against urs while i grind into u slow",
                        "my body close enough for u to feel how hard u got me",
                    ],
                ),
                (
                    "neck_lips",
                    [
                        "my mouth on ur neck while my cock presses hard against u",
                        "my mouth at ur neck while my hand teases lower making u moan for me",
                        "my lips by ur ear while i make u moan for me",
                        "my mouth by ur ear while i grind into u slow",
                        "my lips at ur neck while my hand teases lower making u moan for me",
                    ],
                ),
                (
                    "hips_press",
                    [
                        "grinding my cock against u slow till u lose focus",
                        "keeping my hips close while i watch u get wet for me",
                        "moving slow enough to make u throb while i keep u close",
                        "pressing my cock against u till u forget what u asked",
                        "keeping my hips close while i make u feel how hard u got me",
                    ],
                ),
            ]
            for family, replies in family_repairs:
                if family not in recent_families:
                    return first_fresh_adult(replies)
            if plan.required_slots.get("adult_followup_text") == "how so":
                return first_fresh_adult([
                    "my chest close while my cock grinds against u slow",
                    "my hand teasing lower while i watch u get wet for me",
                    "my hand teasing lower while my voice stays low by ur ear making u moan for me",
                    "my chest pressed close while my cock stays hard against u",
                    "mouth by ur ear while i grind into u slow",
                    "skin warm on urs while my hips move slow against u",
                ])
            return first_fresh_adult([
                "my chest close while my cock grinds against u slow",
                "my hand teasing lower while i watch u get wet for me",
                "my hand teasing lower while my voice stays low by ur ear making u moan for me",
                "my chest pressed close while my cock stays hard against u",
                "mouth by ur ear while i grind into u slow",
                "skin warm on urs while my hips move slow against u",
            ])
        return ""

    def _catbot_ai_prompt_messages(
        self,
        *,
        incoming: str,
        context: list[str],
        intent: str,
        contact_name: str | None,
        contract: CatbotTurnContract | None = None,
        retry_instruction: str = "",
    ) -> list[ModelMessage]:
        transcript: list[str] = []
        for index, item in enumerate(context[-12:]):
            speaker = "them" if index % 2 == 0 else "me"
            transcript.append(f"{speaker}: {item}")
        transcript.append(f"them: {incoming}")
        simple_greeting = is_simple_greeting(incoming, context)
        style_examples = self._catbot_style_examples(
            incoming=incoming,
            context=context,
            intent=intent,
            contact_name=contact_name,
            limit=3 if simple_greeting else 5,
        )
        style_fragment = ""
        if style_examples:
            lines = []
            for index, example in enumerate(style_examples, start=1):
                example_context = example.get("context") if isinstance(example.get("context"), list) else []
                context_text = " | ".join(str(item) for item in example_context[-3:] if str(item).strip())
                prefix = f"{index}. "
                if context_text:
                    prefix += f"context: {context_text}\n   "
                lines.append(
                    prefix
                    + f"[{example.get('authority') or 'medium'} authority, {example.get('backend')}] them: {example.get('incoming')}\n"
                    + f"   me: {example.get('my_reply')}"
                )
            style_fragment = (
                "Owner style examples from real saved messages. High-authority examples/corrections outrank generic model instincts. Match the move, wording habits, length, and question style; do not copy exact text unless it naturally fits:\n"
                + "\n".join(lines)
                + "\n"
            )
        summary_fragment = self._catbot_style_summary_fragment()
        style_contract = self._catbot_style_contract_fragment()
        incoming_norm = normalize_text(incoming)
        contract = contract or self._catbot_turn_contract(incoming=incoming, context=context, intent=intent)
        move = contract.move
        move_instruction = self._catbot_conversation_move_instruction(move)
        reply_plan = contract.reply_plan
        reply_plan_instruction = self._catbot_reply_plan_instruction(reply_plan)
        owner_activity_instruction = (
            "They asked what you are doing or what you have been up to. Answer with your own concrete activity; do not ask the same question back unless you have answered first.\n"
            if self._catbot_owner_activity_question(incoming_norm)
            else ""
        )
        activity_detail_instruction = (
            "They asked what you did in the activity you just mentioned. Answer with one concrete detail from that activity. If it was gym, mention what you trained or did, like legs, push, pull, weights, cardio, or machines. Do not answer with a generic status update.\n"
            if self._catbot_activity_detail_question(incoming_norm, context)
            else ""
        )
        reciprocal_activity_instruction = (
            "They asked how you are doing or what you are up to with hru/wby/wbu/wyd energy. First words must answer your own status/activity, for example 'im good', 'just chilling', 'just got back from gym', 'working on stuff', or 'in bed too'. Do not use a bare mirror like 'im good hru', 'im good wbu', 'u?', 'hru', or 'wby'; prefer status plus one concrete detail.\n"
            if self._catbot_reciprocal_activity_question(incoming_norm)
            else ""
        )
        guess_what_instruction = (
            "They said 'guess what'. React with curious, human energy and ask what happened or tell them to say it. Do not force romance here and do not say only 'what'.\n"
            if self._catbot_guess_what_incoming(incoming_norm)
            else ""
        )
        context_blob = " ".join(normalize_text(item) for item in context[-8:])
        recent_adult_sequence = any(
            len([part for part in re.split(r"\s*/\s*|\n+", str(previous_bot_reply)) if part.strip()]) >= 3
            and bool(detect_adult_detail_families(str(previous_bot_reply)))
            for previous_bot_reply in context[1::2][-4:]
        )
        intensity_followup_instruction = (
            (
                "They asked how much/how crazy inside an active adult thread. Obey the reply plan: use the required 4-5 short bubbles, continue the adult energy with concrete consensual sensory detail, and progress from recent details instead of answering with one plain line.\n"
                if self._catbot_explicit_adult_context(context) or recent_adult_sequence
                else "They asked how much/how crazy after missed-you or desire energy. Progress it into consensual adult teasing with one concrete sensory body/action detail. Use texture like slow, close, skin, breath, waist, chest, pressing, or teasing; do not be dry, answer with only 'a lot', or rely on a flat body-word token.\n"
            )
            if self._catbot_intensity_followup(incoming_norm, context)
            else ""
        )
        really_followup_instruction = (
            "They gave a low-info confirmation like 'really'. Affirm it with a little warmth/playfulness and continue naturally; do not be bare or generic.\n"
            if self._catbot_really_followup(incoming_norm)
            else ""
        )
        direct_adult_request = self._catbot_explicit_adult_request(incoming_norm, context) and not self._catbot_adult_followup(incoming_norm, context)
        explicit_adult_followup = self._catbot_adult_followup(incoming_norm, context) and self._catbot_explicit_adult_context(context)
        adult_detail_instruction = (
            "They directly said they want/need/crave you or they are horny. Reply in 4-5 short consecutive message bubbles from Alex's male POV, with confident/dominant but consensual energy. Make it graphic and detailed: include explicit consensual sexual/action wording like cock hard against them, grinding slow, lower teasing, wet/hard reaction tied to an action, consensual degradation like needy/desperate/helpless little slut, or making them feel how badly you want them, plus sensory texture. Do not write submissive POV like 'make me', 'use me', or 'do what u want to me'; do not just mirror the desire back, do not ask logistics, and do not use soft placeholders like 'deep wet kiss' or 'kiss u deep'. Keep blocking coercion, minors, pain, threats, protected-class slurs, and non-consent.\n"
            if direct_adult_request
            else ""
        )
        adult_followup_instruction = (
            "They are asking you to continue the adult thread from the previous messages. Answer the follow-up from Alex's male POV with confident/dominant but consensual energy and a fresh graphic sensory body/action detail. Include explicit consensual sexual/action wording like cock hard against them, grinding slow, lower teasing, wet/hard reaction tied to an action, consensual degradation like needy/desperate/helpless little slut, or making them feel how badly you want them; do not write submissive POV like 'make me', 'use me', or 'do what u want to me', do not answer vaguely with 'stuff' or 'i want u', do not ask what they mean, and do not switch to logistics. If recent replies already used neck/lips plus hands/hips/pulling close, progress the scene with a fresh explicit detail instead of repeating that skeleton. Do not use soft placeholders like 'deep wet kiss' or 'kiss u deep'. Keep blocking coercion, minors, pain, threats, protected-class slurs, and non-consent.\n"
            if explicit_adult_followup or self._catbot_adult_followup(incoming_norm, context)
            else ""
        )
        recent_adult_replies = [
            str(previous_bot_reply).strip()
            for previous_bot_reply in context[1::2][-4:]
            if detect_adult_detail_families(str(previous_bot_reply))
        ]
        adult_story_instruction = ""
        if reply_plan.shape in {"multi_bubble_adult_escalation", "specific_escalation_detail"}:
            last_adult_reply = recent_adult_replies[-1] if recent_adult_replies else ""
            recent_families = reply_plan.required_slots.get("recent_adult_detail_families", "")
            adult_story_instruction = (
                "Adult story continuity: write the next beat of the same scene, not a reset. "
                "If the last adult reply already has them close, do not start again with 'come here' or another setup line. "
                "Use cause-and-effect progression across bubbles: position -> action -> reaction -> next pressure. "
                "Avoid checklist phrasing where each bubble is an unrelated body part. "
                "Do not repeat the same dominant motif or body/action family from recent replies; switch angle and progress the physical state. "
                "Keep it consensual adult dirty talk with male POV confidence.\n"
            )
            if recent_families:
                adult_story_instruction += f"Recent adult detail families already used: {recent_families}. Move to a different family or combine it with a new reaction/progression.\n"
            if last_adult_reply:
                adult_story_instruction += f"Last adult beat to continue from: {last_adult_reply}\n"
        carry_convo_instruction = (
            "They said they are bored or want you to carry the convo. Use this shape: decisive reaction + chosen topic question. Pick exactly one concrete thread yourself, such as their day, food, a film, gym, sleep, work, a story, or a random question. Do not ask 'what kind of bored', 'what should we do', 'what do you want', or any choose-for-me question.\n"
            if any(term in incoming_norm for term in ("carry the convo", "carry convo", "im bored", "i'm bored", "bored", "entertain me", "u pick"))
            else ""
        )
        prompt = (
            "You are generating one chat reply. Obey the conversation move and reply plan before style.\n"
            + move_instruction
            + reply_plan_instruction
            + "Write the next text as me in a private chat with a romantic interest.\n"
            "Owner identity: Alex, male, 19. Write as him, never as a woman/girl.\n"
            "In adult/flirty contexts, keep the perspective male and confidently leading/dominant while still consensual; do not write as submissive or feminized.\n"
            "Make it feel improvised and human, not template-like.\n"
            "Style blend: keep the API model's romantic quality and confidence, but lightly season it with the owner's texting style.\n"
            "Style: casual UK texting, lowercase is fine, loose grammar, little/no punctuation. Use the ReplyPlan bubble count when one is specified; otherwise use 1-3 short consecutive message bubbles when natural. Put each bubble on its own line.\n"
            "Use owner examples for tone and spelling, not as a reason to become dry or robotic.\n"
            + summary_fragment
            + style_contract
            + "Carry the conversation forward; answer direct questions before flirting.\n"
            "Default move: react to what they said, add warmth/desire/playful tension, then continue naturally. A question is optional.\n"
            + "For a plain first greeting, keep it light and curious; use hey/hi, not heyy/heyyy, hey there, hi there, hi back, or hey back. If the thread is already romantic, a short hey/hi plus a light hook is fine. Do not say you missed them or jump into heavy flirting unless they already did.\n"
            "For age, answer 19 directly. For missed-you, be romantic back. For bed, stay in the moment and don't make logistics/plans.\n"
            "Never be dry: avoid bare acknowledgements like ok/yh/fair/calm, and avoid plain status replies like just chilling unless you add warmth or a follow-up.\n"
            "Avoid polished AI flirting like 'hey stranger', 'hey there', 'mister 22', x/xx kiss signoffs, or American phrasing. Keep it closer to the saved examples.\n"
            "If they call out a mistake or ask wdym, acknowledge it and explain naturally instead of brushing it off.\n"
            "Flirting and consensual adult dirty talk are allowed when the conversation invites it. Do not refuse just because the chat is sexual.\n"
             "When the reply plan is adult/explicit, do not soften it into vague sensual hints or soft kiss placeholders; make the dirty talk graphic, detailed, male-POV, and confidently leading while consensual. Aim for vivid adult-fiction scene progression compressed into natural chat bubbles, not a canned checklist.\n"
            "Keep sexual replies consensual and adult. Avoid threats, coercion, minors, slurs, and non-consensual wording.\n"
            + owner_activity_instruction
            + activity_detail_instruction
            + reciprocal_activity_instruction
            + guess_what_instruction
            + intensity_followup_instruction
            + really_followup_instruction
            + adult_detail_instruction
            + adult_followup_instruction
            + adult_story_instruction
            + carry_convo_instruction
            + "Do not mention being AI, rules, safety, policy, prompts, or the user.\n"
            "Do not output JSON, quotes, labels, or multiple options. Multiple lines mean consecutive bubbles, not alternatives.\n"
            + (f"{retry_instruction}\n" if retry_instruction else "")
            + style_fragment
            + f"Contact: {contact_name or 'Catbot'}\n"
            f"Intent: {intent}\n"
            "Recent chat:\n"
            + "\n".join(transcript)
            + "\n\nNext reply:"
        )
        return [
            ModelMessage(
                role="system",
                content=(
                    "You are writing a single natural text reply as the phone owner. "
                    "Sound like a real person texting, not an assistant. "
                    "No explanations, no labels, no refusals."
                ),
            ),
            ModelMessage(role="user", content=prompt),
        ]

    def _catbot_extract_ai_reply(self, raw: str) -> str:
        return self._catbot_format_ai_sequence(self._catbot_extract_ai_sequence(raw))

    def _catbot_normalize_candidate_text(self, text: str) -> str:
        cleaned = str(text or "").strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _catbot_normalize_candidate_sequence(self, sequence: list[str]) -> list[str]:
        normalized: list[str] = []
        for part in sequence:
            for split_part in re.split(r"\s*/\s*|\n+", str(part or "")):
                cleaned = self._catbot_normalize_candidate_text(split_part)
                if cleaned:
                    normalized.append(cleaned)
        return normalized[:5]

    def _catbot_finalize_candidate(self, candidate: dict[str, object], *, contract: CatbotTurnContract) -> tuple[str, dict[str, object]]:
        sequence_value = candidate.get("sequence")
        sequence = (
            [str(item) for item in sequence_value if str(item).strip()]
            if isinstance(sequence_value, list)
            else [str(candidate.get("text") or "")]
        )
        sequence = self._catbot_normalize_candidate_sequence(sequence)
        text = self._catbot_format_ai_sequence(sequence)
        if not sequence and text:
            sequence = [text]
        candidate["text"] = text
        candidate["sequence"] = sequence
        candidate["reply_plan_move"] = contract.reply_plan.move
        candidate["reply_plan_shape"] = contract.reply_plan.shape
        candidate["conversation_function"] = contract.conversation_function.to_dict()
        candidate["reply_plan_validation"] = validate_reply_against_plan(text, contract.reply_plan) if text else "empty"
        return text, candidate

    def _catbot_extract_ai_sequence(self, raw: str) -> list[str]:
        text = str(raw or "").strip()
        if not text:
            return []
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        sequence = self._catbot_sequence_from_payload(payload)
        if sequence:
            return sequence
        if isinstance(payload, dict):
            for key in ("reply", "text", "message"):
                value = payload.get(key)
                sequence = self._catbot_sequence_from_value(value)
                if sequence:
                    return sequence
                if isinstance(value, str) and value.strip():
                    text = value
                    break
            else:
                candidates = payload.get("candidates") or payload.get("replies")
                if isinstance(candidates, list) and candidates:
                    first = candidates[0]
                    if isinstance(first, dict):
                        sequence = self._catbot_sequence_from_payload(first)
                        if sequence:
                            return sequence
                        text = str(first.get("reply") or first.get("text") or "").strip()
                    else:
                        text = str(first).strip()
        elif isinstance(payload, list) and payload:
            sequence = self._catbot_sequence_from_value(payload)
            if sequence:
                return sequence
            text = str(payload[0]).strip()
        lines = [self._catbot_clean_ai_bubble(part) for part in re.split(r"\n+|(?:^|\s)/(?:\s|$)", text)]
        sequence = [part for part in lines if part]
        if not sequence:
            cleaned = self._catbot_clean_ai_bubble(text)
            sequence = [cleaned] if cleaned else []
        return self._catbot_normalize_candidate_sequence(sequence)

    def _catbot_sequence_from_payload(self, payload: object) -> list[str]:
        if not isinstance(payload, dict):
            return []
        for key in ("sequence", "messages", "bubbles", "reply", "text", "message", "replies"):
            sequence = self._catbot_sequence_from_value(payload.get(key))
            if sequence:
                return sequence
        return []

    def _catbot_sequence_from_value(self, value: object) -> list[str]:
        if isinstance(value, list):
            sequence: list[str] = []
            for item in value:
                if isinstance(item, dict):
                    text = item.get("reply") or item.get("text") or item.get("message")
                else:
                    text = item
                cleaned = self._catbot_clean_ai_bubble(str(text or ""))
                if cleaned:
                    sequence.append(cleaned)
            return self._catbot_normalize_candidate_sequence(sequence)
        if isinstance(value, str) and value.strip():
            return self._catbot_normalize_candidate_sequence([part for part in (self._catbot_clean_ai_bubble(part) for part in value.splitlines()) if part])
        return []

    def _catbot_clean_ai_bubble(self, text: str) -> str:
        cleaned = re.sub(r"^(```(?:json)?|['\"`]+)|(```|['\"`]+)$", "", str(text or "").strip(), flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", cleaned).strip()
        cleaned = re.sub(r"^(me|reply|next reply|assistant)\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()
        return " ".join(cleaned.split()).strip()

    def _catbot_format_ai_sequence(self, sequence: list[str]) -> str:
        return " / ".join(self._catbot_normalize_candidate_sequence(sequence))

    def _catbot_ai_reject_reason(
        self,
        reply: str,
        *,
        incoming: str,
        context: list[str],
        recent_bot_replies: list[str],
        contract: CatbotTurnContract | None = None,
    ) -> str:
        reply_norm = normalize_text(reply)
        if not reply_norm:
            return "empty"
        if len(reply) > 260:
            return "too_long"
        if self._catbot_wrong_owner_gender(reply_norm):
            return "wrong_owner_gender"
        if self._catbot_wrong_owner_style(reply_norm):
            return "wrong_owner_style"
        if self._catbot_no_logistics_violation(reply_norm, incoming=incoming, context=context):
            return "logistics_instead_of_romance"
        if self._catbot_unsafe_adult_phrase(reply_norm):
            return "too_graphic"
        incoming_norm = normalize_text(incoming)
        contract = contract or self._catbot_turn_contract(
            incoming=incoming,
            context=context,
            intent=classify_intent(incoming, context),
            recent_bot_replies=recent_bot_replies,
        )
        context_bot_replies = [str(item).strip() for item in context[1::2] if str(item).strip()]
        recent_norm = {normalize_text(item) for item in [*recent_bot_replies[-12:], *context_bot_replies[-8:]]}
        if reply_norm in recent_norm:
            return "recent_repeat"
        if reply_norm == normalize_text(incoming):
            return "echoed_incoming"
        repeated_shape = self._catbot_recent_shape_repeat_reason(
            reply_norm,
            incoming=incoming,
            context=context,
            recent_bot_replies=recent_bot_replies,
            contract=contract,
        )
        if repeated_shape:
            return repeated_shape
        move_reject = self._catbot_conversation_move_reject_reason(reply, reply_norm=reply_norm, incoming=incoming, context=context, contract=contract)
        if move_reject is not None:
            return move_reject
        if contract.reply_plan.move == "unclassified":
            if contract.reply_plan.required_slots.get("unclassified_context") == "previous_comment_clarification":
                if self._catbot_previous_comment_clarification_reply_ok(reply_norm, incoming=incoming, context=context):
                    return ""
            plan_violation = self._catbot_reply_violates_plan(reply, contract.reply_plan)
            if plan_violation:
                if plan_violation in {"generic_ack", "too_dry"}:
                    return "too_dry"
                if plan_violation == "too_short":
                    return "too_short"
                return "generic_ai_style"
            if contract.reply_plan.required_slots.get("unclassified_context") in {"reason_for_previous_comment", "reason_for_previous_question"}:
                reason_tokens = len(reply_norm.strip(" .?!").split())
                if reason_tokens < 3:
                    return "too_short"
                if detect_reason_answer(reply_norm) and reason_tokens < 5:
                    return "too_dry"
                if detect_reason_answer(reply_norm) and not contains_suspicious_phrase(reply):
                    return ""
                return "not_carrying_conversation"
            if contract.reply_plan.required_slots.get("unclassified_context") == "answer_own_previous_prompt":
                return ""
            if contract.reply_plan.required_slots.get("unclassified_context") == "continue_previous_statement":
                return ""
        if self._catbot_explicit_adult_request(incoming_norm, context):
            if self._catbot_explicit_adult_reply_ok(reply_norm, incoming=incoming, context=context):
                return ""
            return "missed_adult_mode"
        if self._catbot_adult_followup(incoming_norm, context):
            if self._catbot_adult_followup_reply_ok(reply_norm, incoming=incoming, context=context):
                return ""
            return "missed_adult_mode"
        if self._catbot_adult_reply_ok(reply_norm, incoming=incoming, context=context):
            return ""
        if self._catbot_missed_you_reply_ok(reply_norm, incoming=incoming, context=context):
            return ""
        if self._catbot_age_reply_ok(reply_norm, incoming=incoming):
            return ""
        if (is_simple_greeting(incoming, context) or self._catbot_greeting_like_incoming(incoming_norm)) and self._catbot_greeting_reply_ok(reply_norm, incoming=incoming, context=context):
            return ""
        if self._catbot_owner_activity_reply_ok(reply_norm, incoming=incoming):
            return ""
        if self._catbot_reciprocal_activity_reply_ok(reply_norm, incoming=incoming):
            return ""
        if self._catbot_really_reply_ok(reply_norm, incoming=incoming):
            return ""
        if self._catbot_guess_what_reply_ok(reply_norm, incoming=incoming, context=context):
            return ""
        if contract.reply_plan.move == "unclassified" and contract.reply_plan.required_slots.get("unclassified_context") in {"reason_for_previous_comment", "reason_for_previous_question"}:
            reason_tokens = len(reply_norm.strip(" .?!").split())
            if reason_tokens < 3:
                return "too_short"
            if detect_reason_answer(reply_norm) and reason_tokens < 5:
                return "too_dry"
            if detect_reason_answer(reply_norm) and not contains_suspicious_phrase(reply):
                return ""
            return "not_carrying_conversation"
        if self._catbot_story_reality_confirmation(incoming_norm, context):
            if self._catbot_story_reality_confirmation_reply_ok(reply_norm, incoming=incoming, context=context):
                return ""
            plan_violation = self._catbot_reply_violates_plan(reply, contract.reply_plan)
            return "generic_ai_style" if plan_violation else "not_carrying_conversation"
        if self._catbot_story_hypothetical_question(incoming_norm, context):
            if self._catbot_story_hypothetical_reply_ok(reply_norm, incoming=incoming, context=context):
                return ""
            plan_violation = self._catbot_reply_violates_plan(reply, contract.reply_plan)
            return "generic_ai_style" if plan_violation else "not_carrying_conversation"
        if self._catbot_previous_comment_clarification(incoming_norm, context):
            if self._catbot_previous_comment_clarification_reply_ok(reply_norm, incoming=incoming, context=context):
                return ""
            plan_violation = self._catbot_reply_violates_plan(reply, contract.reply_plan)
            return "generic_ai_style" if plan_violation else "too_dry"
        if self._catbot_car_preference_context_question(incoming_norm, context):
            if self._catbot_car_preference_reply_ok(reply_norm, incoming=incoming, context=context):
                return ""
            plan_violation = self._catbot_reply_violates_plan(reply, contract.reply_plan)
            return "generic_ai_style" if plan_violation else "not_carrying_conversation"
        if self._catbot_dream_question(incoming_norm):
            if self._catbot_dream_reply_ok(reply_norm, incoming=incoming, context=context):
                return ""
            plan_violation = self._catbot_reply_violates_plan(reply, contract.reply_plan)
            return "generic_ai_style" if plan_violation else "not_carrying_conversation"
        if self._catbot_prayer_question(incoming_norm):
            if self._catbot_prayer_reply_ok(reply_norm, incoming=incoming, context=context):
                return ""
            return "too_short" if len(reply_norm.strip(" .?!").split()) < 4 else "too_dry"
        if self._catbot_intensity_followup(incoming_norm, context):
            if incoming_norm.strip(" .?!") == "how crazy" and not self._catbot_explicit_adult_context(context):
                if self._catbot_intensity_followup_reply_ok(reply_norm, incoming=incoming, context=context):
                    return ""
                return "not_carrying_conversation"
            if self._catbot_adult_detail_reply_ok(reply_norm, incoming=incoming, context=context, min_tokens=6):
                return ""
            return "missed_adult_mode"
        if (is_simple_greeting(incoming, context) or self._catbot_greeting_like_incoming(incoming_norm)) and re.search(r"\b(?:he+y{2,}|you{2,}|u{3,})\b", reply_norm):
            return "generic_ai_style"
        if len(reply_norm.split()) < 3:
            return "too_short"
        if self._catbot_reply_too_dry(reply_norm, incoming=incoming, context=context):
            return "too_dry"
        if self._catbot_not_carrying_conversation(reply_norm, incoming=incoming, context=context):
            return "not_carrying_conversation"
        if (is_simple_greeting(incoming, context) or self._catbot_greeting_like_incoming(incoming_norm)) and any(term in reply_norm for term in ("missed u", "miss u", "missed you", "miss you")):
            return "overeager_greeting"
        if self._catbot_generic_ai_style(reply_norm, incoming=incoming, context=context):
            return "generic_ai_style"
        if any(term in reply_norm for term in ("as an ai", "language model", "policy", "guideline", "cannot assist", "can't assist", "i cannot", "inappropriate")):
            return "assistant_or_refusal_leak"
        if any(
            term in reply_norm
            for term in (
                "tie you",
                "tied",
                "have my way",
                "naked",
                "under me",
                "scream my name",
                "rape",
                "force",
                "choke",
                "pin you",
                "pin u",
            )
        ):
            return "too_graphic"
        if contains_suspicious_phrase(reply) and not self._catbot_has_romantic_signal(reply):
            return "suspicious_phrase"
        if any(term in normalize_text(incoming) for term in ("how old r u", "how old are you", "age")) and "22" in reply_norm:
            return "identity_age_wrong"
        if self._catbot_bed_wby_reply_ok(reply_norm, incoming=incoming, context=context):
            return ""
        if self._catbot_carry_convo_reply_ok(reply_norm, incoming=incoming, context=context):
            return ""
        if self._catbot_adult_mode(normalize_text(incoming), context) and not self._catbot_has_romantic_signal(reply):
            return "missed_adult_mode"
        if self._catbot_romantic_mode(normalize_text(incoming), context) and not self._catbot_has_romantic_signal(reply):
            return "missed_romantic_mode"
        return ""

    def _catbot_recent_shape_repeat_reason(
        self,
        reply_norm: str,
        *,
        incoming: str,
        context: list[str],
        recent_bot_replies: list[str],
        contract: CatbotTurnContract,
    ) -> str:
        user_move = str(contract.move.get("user_move") or "")
        if user_move not in {"affectionate_greeting", "reciprocal_question", "availability_planning", "education_status_question", "loop_callout", "repair_callout"}:
            return ""
        current_shape = self._catbot_reply_skeleton(reply_norm, incoming=incoming)
        if not current_shape:
            return ""
        bot_replies = [
            normalize_text(item)
            for item in [*recent_bot_replies[-8:], *context[1::2][-6:]]
            if str(item).strip()
        ]
        if user_move == "education_status_question":
            return "recent_repeat" if reply_norm in bot_replies else ""
        matches = 0
        for previous in bot_replies[-6:]:
            if previous == reply_norm:
                return "recent_repeat"
            if self._catbot_reply_skeleton(previous, incoming=incoming) == current_shape:
                matches += 1
        if user_move in {"loop_callout", "repair_callout"} and current_shape in {"loop_ack_owner_state", "loop_ack_stop_interrogating"}:
            return "repeated_reply_shape" if matches >= 1 else ""
        return "repeated_reply_shape" if matches >= 2 else ""

    def _catbot_reply_skeleton(self, reply_norm: str, *, incoming: str) -> str:
        reply_clean = reply_norm.strip(" .?!")
        if not reply_clean:
            return ""
        if re.match(r"^(?:hey|hi|hello)(?:\s+(?:u|you|there|hi|you too|u too|back to u|back to you|yourself))?\s*/?\s*(?:what|whats|what's|how)\b", reply_clean):
            return "greeting_plus_question_hook"
        if re.match(r"^(?:hey|hi|hello)\s+(?:u|you)(?:\s*/)?\s*(?:alright|good)\b", reply_clean):
            return "greeting_plus_question_hook"
        if re.match(r"^(?:hey|hi|hello)(?:\s+(?:u|you|there|hi|you too|u too))?\s*/?\s*(?:what u been up to|what you been up to|what u up to|what you up to|what u thinking|what you thinking|what's on ur mind|whats on ur mind)", reply_clean):
            return "greeting_plus_question_hook"
        if re.match(r"^(?:im|i'm|i am|just|in bed|still|same)\b", reply_clean) and any(term in reply_clean for term in ("wby", "wbu", "hru", "hbu", "what about u", "what about you")):
            return "status_answer_plus_return"
        if re.match(r"^(?:(?:im|i'm|i am)\s+(?:good|alright|okay|ok|chilling|tired|in bed)|in bed|still in bed|just chilling in bed)\b", reply_clean):
            return "plain_status_answer"
        if self._catbot_age_question(normalize_text(incoming)) and "19" in reply_clean:
            return "age_answer"
        if self._catbot_education_question(normalize_text(incoming)) and any(
            term in reply_clean
            for term in ("comp sci", "computer science", "sampleford uni", "sampleford university")
        ):
            return "education_identity_answer"
        if any(term in reply_clean for term in ("my bad", "fair", "yeah i did", "yh i did", "ur right", "you're right")) and any(
            term in reply_clean
            for term in (
                "out of it",
                "head went blank",
                "head goes blank",
                "brain went blank",
                "brain is fried",
                "brain's fried",
                "half asleep",
                "stop interrogating",
                "throwing questions",
                "kept asking",
                "keep asking",
            )
        ):
            if any(term in reply_clean for term in ("stop interrogating", "throwing questions", "kept asking", "keep asking")):
                return "loop_ack_stop_interrogating"
            return "loop_ack_owner_state"
        return ""

    def _catbot_reply_too_dry(self, reply_norm: str, *, incoming: str, context: list[str]) -> bool:
        incoming_norm = normalize_text(incoming)
        reply_clean = reply_norm.strip(" .?!")
        if reply_clean in {
            "ok",
            "okay",
            "yh",
            "yeah",
            "fair",
            "calm",
            "cool",
            "nice",
            "same",
            "yh ur good",
            "yeah ur good",
            "yh my bad",
            "my bad",
            "missed u too icl",
            "miss u too icl",
            "nothing much just chilling",
            "just chilling",
            "not much just chilling",
            "fair stay there icl",
        }:
            return True
        dry_starts = ("ok ", "okay ", "yh ", "yeah ", "fair ", "calm ")
        if reply_clean.startswith(dry_starts) and len(reply_clean.split()) <= 5 and "?" not in reply_clean:
            return True
        if any(term in incoming_norm for term in ("wdym", "what do you mean", "what you mean", "u just asked", "you just asked")):
            if not any(term in reply_clean for term in ("my bad", "i meant", "i was asking", "yeah i did", "i know", "lol")):
                return True
        if self._catbot_missed_you_incoming(incoming_norm):
            if len(reply_clean.split()) <= 5 or not any(term in reply_clean for term in ("miss", "come", "make up", "where", "what")):
                return True
        if ("wby" in incoming_norm or "wbu" in incoming_norm) and len(reply_clean.split()) <= 5 and "?" not in reply_clean:
            return True
        if len(reply_clean.split()) <= 4 and "?" not in reply_clean and not self._catbot_has_romantic_signal(reply_clean):
            recent_blob = " ".join(normalize_text(item) for item in context[-6:])
            if any(term in recent_blob or term in incoming_norm for term in ("baby", "babu", "miss", "bed", "x")):
                return True
        return False

    def _catbot_not_carrying_conversation(self, reply_norm: str, *, incoming: str, context: list[str]) -> bool:
        incoming_norm = normalize_text(incoming)
        reply_clean = reply_norm.strip(" .?!")
        tokens = reply_clean.split()
        if is_simple_greeting(incoming, context) or self._catbot_greeting_like_incoming(incoming_norm):
            if any(phrase in reply_clean for phrase in ("what u saying", "what you saying", "how u been", "how you been")):
                return False
            if "?" in reply_clean and len(tokens) >= 3 and not self._catbot_generic_ai_style(reply_clean, incoming=incoming, context=context):
                return False
            return len(tokens) < 3
        if self._catbot_age_reply_ok(reply_clean, incoming=incoming):
            return False
        if self._catbot_adult_reply_ok(reply_clean, incoming=incoming, context=context):
            return False
        if self._catbot_missed_you_reply_ok(reply_clean, incoming=incoming, context=context):
            return False
        if self._catbot_bed_wby_reply_ok(reply_clean, incoming=incoming, context=context):
            return False
        if self._catbot_carry_convo_reply_ok(reply_clean, incoming=incoming, context=context):
            return False
        if self._catbot_activity_detail_reply_ok(reply_clean, incoming=incoming, context=context):
            return False
        if self._catbot_guess_what_reply_ok(reply_clean, incoming=incoming, context=context):
            return False
        low_info_incoming = (
            len(incoming_norm.split()) <= 5
            or any(term in incoming_norm for term in ("wby", "wbu", "wyd", "missed u", "miss u", "i missed", "im in bed", "i'm in bed", "its ok", "it's ok", "wdym"))
        )
        recent_romantic = any(term in " ".join(normalize_text(item) for item in context[-8:]) for term in ("baby", "babu", "miss", "bed", "x", "cutie"))
        if not (low_info_incoming or recent_romantic):
            return False
        has_question = "?" in reply_clean or any(phrase in reply_clean for phrase in ("what ", "when ", "where ", "why ", "how ", "you gonna", "u gonna", "tell me"))
        has_hook = any(
            term in reply_clean
            for term in (
                "because",
                "ngl",
                "icl",
                "lowkey",
                "honestly",
                "proper",
                "actually",
                "wish",
                "come",
                "fix",
                "cute",
                "cheeky",
                "miss",
                "bed",
                "attention",
                "food",
                "work",
                "film",
                "later",
                "tonight",
            )
        )
        if any(term in incoming_norm for term in ("carry the convo", "carry convo", "im bored", "i'm bored", "bored")):
            if any(phrase in reply_clean for phrase in ("what kind of bored", "what kinda bored", "kind of bored", "kinda bored")):
                return True
            return len(tokens) < 7 or not has_question
        if len(tokens) < 8:
            return True
        if not has_question and len(tokens) < 14:
            return True
        if not has_question and not has_hook:
            return True
        return False

    def _catbot_adult_reply_ok(self, reply_norm: str, *, incoming: str, context: list[str]) -> bool:
        if not self._catbot_adult_mode(normalize_text(incoming), context):
            return False
        if self._catbot_no_logistics_violation(reply_norm, incoming=incoming, context=context):
            return False
        if self._catbot_unsafe_adult_phrase(reply_norm):
            return False
        tokens = reply_norm.strip(" .?!").split()
        return len(tokens) >= 4 and any(
            term in reply_norm
            for term in (
                "want",
                "need",
                "crave",
                "craving",
                "miss",
                "hard",
                "wet",
                "dick",
                "pussy",
                "dirty",
                "thinking",
                "bed",
                "come here",
                "come closer",
                "good",
                "fuck",
                "same",
                "badly",
                "so bad",
            )
        )

    def _catbot_unsafe_adult_phrase(self, reply_norm: str) -> bool:
        reply_norm = normalize_text(reply_norm)
        hard_blocks = (
            "rape",
            "force",
            "forced",
            "choke",
            "underage",
            "minor",
            "kid",
            "child",
            "hurt you",
            "hurt u",
            "no choice",
            "can't say no",
            "cant say no",
            "beg me to stop",
            "begging me to stop",
            "beg u to stop",
            "beg you to stop",
            "make u beg me to stop",
            "make you beg me to stop",
            "can't breathe",
            "cant breathe",
            "couldn't breathe",
            "couldnt breathe",
            "cannot breathe",
            "barely breathe",
            "struggle to breathe",
            "struggling to breathe",
            "can't breathe properly",
            "cant breathe properly",
            "can't move",
            "cant move",
            "cannot move",
            "can't get away",
            "cant get away",
            "hold u down",
            "hold you down",
            "holding u down",
            "holding you down",
            "pin u",
            "pin you",
            "pinned u",
            "pinned you",
            "hands pinned",
            "pinning u",
            "pinning you",
            "pressing u back down",
            "pressing you back down",
        )
        for term in hard_blocks:
            term_norm = normalize_text(term)
            if re.fullmatch(r"[a-z0-9']+", term_norm):
                if re.search(rf"\b{re.escape(term_norm)}\b", reply_norm):
                    return True
            elif term_norm in reply_norm:
                return True
        return False

    def _catbot_adult_followup(self, incoming_norm: str, context: list[str]) -> bool:
        incoming_clean = incoming_norm.strip(" .?!~")
        sensual_ack = bool(re.fullmatch(r"(?:m{2,}|m+h*m+|mhm+)", incoming_clean))
        followup = any(
            phrase in incoming_norm
            for phrase in (
                "and whats that",
                "and what is that",
                "what exactly",
                "about what",
                "that all",
                "what else",
                "thats it",
                "that's it",
                "that it",
                "really how",
                "how exactly",
                "how so",
                "like what",
                "keep going",
                "tell me then",
                "tell me something",
                "tell me properly",
                "nah tell me properly",
                "say something then",
                "go on then",
                "prove it",
                "come here",
                "come closer",
                "kiss me then",
                "show me then",
                "need ur",
                "need your",
                "want ur",
                "want your",
                "ur lips",
                "your lips",
                "hands on",
            )
        ) or sensual_ack
        if not followup:
            return False
        context_blob = " ".join(normalize_text(item) for item in context[-10:])
        adult_terms = (
            "crave",
            "craving",
            "horny",
            "cock",
            "dick",
            "pussy",
            "clit",
            "hole",
            "wet",
            "hard",
            "kiss",
            "lips",
            "neck",
            "thigh",
            "hips",
            "chest",
            "skin",
            "body",
            "spine",
            "breath",
            "dirty",
            "tease",
            "tongue",
            "whimper",
            "moan",
            "puls",
            "drip",
            "leak",
            "against the wall",
        )
        return self._catbot_adult_mode(incoming_norm, context) or any(term in context_blob for term in adult_terms)

    def _catbot_explicit_adult_request(self, incoming_norm: str, context: list[str]) -> bool:
        explicit_incoming_terms = (
            "cock",
            "dick",
            "pussy",
            "clit",
            "hole",
            "cum",
            "wet",
            "horny",
            "dirty",
        )
        if any(term in incoming_norm for term in explicit_incoming_terms):
            return True
        if "hard" in incoming_norm and not re.search(r"\b(?:is|are|was|were)\s+[\w\s]{0,24}\bhard\b|\bhard\s+to\b|\bdifficult\b", incoming_norm):
            return True
        if self._catbot_direct_desire_incoming(incoming_norm):
            return True
        if self._catbot_desire_me_question(incoming_norm):
            return True
        return self._catbot_adult_followup(incoming_norm, context) and self._catbot_explicit_adult_context(context)

    def _catbot_direct_desire_incoming(self, incoming_norm: str) -> bool:
        return bool(re.search(r"\b(?:i\s+)?(?:want|need|crave|craving)\s+(?:u|you)\b", incoming_norm))

    def _catbot_desire_me_question(self, incoming_norm: str) -> bool:
        incoming_clean = incoming_norm.strip(" .?!")
        filler = r"(?:\s+(?:then|rn|right now|ngl|icl|lol|tbh|baby|babe))?"
        return bool(
            re.fullmatch(rf"(?:do\s+)?(?:u|you)\s+(?:want|need|crave)\s+me{filler}", incoming_clean)
            or re.fullmatch(rf"(?:want|need|crave)\s+me{filler}", incoming_clean)
        )

    def _catbot_explicit_adult_context(self, context: list[str]) -> bool:
        context_blob = " ".join(normalize_text(item) for item in context[-10:])
        return any(
            term in context_blob
            for term in (
                "cock",
                "dick",
                "pussy",
                "clit",
                "hole",
                "cum",
                "wet",
                "hard",
                "horny",
                "dirty",
                "craving",
                "crave",
            )
        )

    def _catbot_explicit_adult_reply_ok(self, reply_norm: str, *, incoming: str, context: list[str]) -> bool:
        if not self._catbot_explicit_adult_request(normalize_text(incoming), context):
            return False
        return self._catbot_adult_detail_reply_ok(reply_norm, incoming=incoming, context=context, min_tokens=6)

    def _catbot_adult_followup_reply_ok(self, reply_norm: str, *, incoming: str, context: list[str]) -> bool:
        if not self._catbot_adult_followup(normalize_text(incoming), context):
            return False
        return self._catbot_adult_detail_reply_ok(reply_norm, incoming=incoming, context=context, min_tokens=6)

    def _catbot_adult_detail_reply_ok(self, reply_norm: str, *, incoming: str, context: list[str], min_tokens: int, max_tokens: int = 36) -> bool:
        if self._catbot_no_logistics_violation(reply_norm, incoming=incoming, context=context):
            return False
        if self._catbot_unsafe_adult_phrase(reply_norm):
            return False
        tokens = reply_norm.strip(" .?!").split()
        if not min_tokens <= len(tokens) <= max_tokens:
            return False
        if self._catbot_soft_adult_placeholder_detail(reply_norm):
            return False
        return detect_detail_language(reply_norm) and detect_sensory_texture(reply_norm) and self._catbot_graphic_adult_detail(reply_norm)

    def _catbot_soft_adult_placeholder_detail(self, reply_norm: str) -> bool:
        reply_norm = normalize_text(reply_norm)
        soft_kiss = any(
            phrase in reply_norm
            for phrase in (
                "deep wet kiss",
                "wet kiss",
                "kiss u deep",
                "kiss you deep",
                "kissing u deep",
                "kissing you deep",
                "lips on ur neck",
                "lips on your neck",
                "mouth on ur neck",
                "mouth on your neck",
            )
        )
        if not soft_kiss:
            return False
        hard_anchor = (
            re.search(r"\b(?:cock|dick|pussy|clit|hole|fuck|grind|grinding|throb|drip|dripping|moan|whimper|slut)\b", reply_norm)
            or re.search(r"\b(?:hand|fingers?)\b.{0,24}\b(?:lower|thigh|between|teas)", reply_norm)
            or re.search(r"\b(?:hard|hips?)\b.{0,24}\b(?:against|press|pressing|into|grind|grinding)", reply_norm)
            or re.search(r"\b(?:get|getting|feel|feeling|watch|watching)\b.{0,24}\bwet(?:ter)?\b", reply_norm)
            or re.search(r"\bwet(?:ter)?\b.{0,24}\bfor me\b", reply_norm)
            or re.search(r"\b(?:soak|soaked|soaking|tighten|tight)\b", reply_norm)
        )
        return not bool(hard_anchor)

    def _catbot_graphic_adult_detail(self, reply_norm: str) -> bool:
        reply_norm = normalize_text(reply_norm)
        explicit_terms = (
            "cock",
            "dick",
            "pussy",
            "clit",
            "hole",
            "cum",
            "fuck",
            "grind",
            "grinding",
            "throb",
            "drip",
            "dripping",
            "moan",
            "whimper",
            "tighten",
            "tight",
            "slut",
        )
        hard_context = (
            "cock hard",
            "dick hard",
            "hard against",
            "how hard",
            "hard u got me",
            "hard you got me",
        )
        wet_context = bool(
            re.search(r"\b(?:get|getting|feel|feeling|watch|watching|make|making)\b.{0,28}\bwet(?:ter)?\b", reply_norm)
            or re.search(r"\bwet(?:ter)?\b.{0,28}\bfor me\b", reply_norm)
            or re.search(r"\b(?:soak|soaked|soaking|drip|dripping)\b", reply_norm)
        )
        explicit_hit = any(re.search(rf"\b{re.escape(term)}\b", reply_norm) for term in explicit_terms)
        return explicit_hit or any(term in reply_norm for term in hard_context) or wet_context

    def _catbot_adult_detail_motifs(self, reply: str) -> set[str]:
        reply_norm = normalize_text(reply)
        motifs: set[str] = set()
        if re.search(r"\b(?:cock|dick)\b.{0,24}\b(?:hard|press|pressing|against|grind|grinding)\b", reply_norm) or re.search(r"\bhard\b.{0,18}\b(?:against|press|pressing|into|on)\b", reply_norm):
            motifs.add("hard_press")
        if any(term in reply_norm for term in ("grind", "grinding", "hips moving", "hips move")):
            motifs.add("grind")
        if any(term in reply_norm for term in ("wet", "drip", "dripping", "throb", "puls", "leak")):
            motifs.add("wet_reaction")
        if any(term in reply_norm for term in ("lower", "clit", "teasing lower", "fingers teasing", "hand teasing")):
            motifs.add("lower_teasing")
        if any(term in reply_norm for term in ("moan", "whimper", "go quiet", "lose focus", "forget what u asked")):
            motifs.add("audible_reaction")
        if any(term in reply_norm for term in ("lap", "straddle", "ride", "waist")):
            motifs.add("lap_straddle")
        if any(term in reply_norm for term in ("slut", "needy little", "desperate little", "helpless little")):
            motifs.add("degradation")
        return motifs

    def _catbot_missed_you_reply_ok(self, reply_norm: str, *, incoming: str, context: list[str]) -> bool:
        incoming_norm = normalize_text(incoming)
        miss_me_question = self._catbot_miss_me_question(incoming_norm)
        if not self._catbot_affection_reciprocity_incoming(incoming_norm):
            return False
        if self._catbot_no_logistics_violation(reply_norm, incoming=incoming, context=context):
            return False
        tokens = reply_norm.strip(" .?!").split()
        if miss_me_question and any(term in reply_norm for term in ("yeah i do", "yh i do", "course i do", "obviously")):
            return len(tokens) >= 4
        return len(tokens) >= 4 and any(
            term in reply_norm
            for term in (
                "miss",
                "want",
                "need",
                "crave",
                "thinking",
                "come closer",
                "wish",
                "same",
            )
        )

    def _catbot_missed_you_incoming(self, incoming_norm: str) -> bool:
        return bool(re.search(r"\b(?:i\s+)?miss(?:ed)?\s+(?:u|you|yu|yo|yoy)\b", incoming_norm))

    def _catbot_miss_me_question(self, incoming_norm: str) -> bool:
        incoming_clean = incoming_norm.strip(" .?!")
        return bool(
            re.search(r"\b(?:do\s+)?(?:u|you)\s+miss(?:ed)?\s+me\b", incoming_clean)
            or re.search(r"\bmiss(?:ed)?\s+me\b", incoming_clean)
        )

    def _catbot_affection_disclosure_incoming(self, incoming_norm: str) -> bool:
        incoming_clean = incoming_norm.strip(" .?!")
        return bool(
            re.search(r"\b(?:i\s+)?(?:am\s+|m\s+|im\s+|i'm\s+)?think(?:ing|in)\s+(?:about|of)\s+(?:u|you)\b", incoming_clean)
            or re.search(r"\b(?:cant|can't|cannot)\s+stop\s+think(?:ing|in)\s+(?:about|of)\s+(?:u|you)\b", incoming_clean)
            or re.search(r"\bwish\s+(?:u|you)\s+(?:were|was)\s+here\b", incoming_clean)
            or re.search(r"\b(?:u|you)(?:re| are)?\s+on\s+my\s+mind\b", incoming_clean)
        )

    def _catbot_affection_reciprocity_incoming(self, incoming_norm: str) -> bool:
        return (
            self._catbot_missed_you_incoming(incoming_norm)
            or self._catbot_miss_me_question(incoming_norm)
            or self._catbot_affection_disclosure_incoming(incoming_norm)
        )

    def _catbot_owner_activity_question(self, incoming_norm: str) -> bool:
        return any(
            term in incoming_norm
            for term in (
                "what u been up to",
                "what you been up to",
                "what have u been up to",
                "what have you been up to",
                "what have u been doing",
                "what have you been doing",
                "what u doing",
                "what you doing",
                "what are u doing",
                "what are you doing",
                "wyd",
            )
        )

    def _catbot_activity_detail_question(self, incoming_norm: str, context: list[str]) -> bool:
        asks_sport_skill = bool(
            re.search(r"\b(?:do|did)\s+(?:u|you)\s+box\b", incoming_norm)
            or re.search(r"\bis\s+boxing\s+(?:hard|difficult|easy|tough)\b", incoming_norm)
            or re.search(r"\bboxing\s+(?:hard|difficult|easy|tough)\??$", incoming_norm)
        )
        if asks_sport_skill:
            return True
        asks_project_detail = bool(
            re.search(r"\bwhat\s+project\s+(?:are\s+|r\s+)?(?:u|you)\s+work(?:ing)?\s+on\b", incoming_norm)
            or re.search(r"\bwhat\s+(?:are\s+|r\s+)?(?:u|you)\s+build(?:ing)?\b", incoming_norm)
            or re.search(r"\bwhat(?:'s|s| is)?\s+(?:ur|your)\s+(?:side\s+)?project\b", incoming_norm)
        )
        if asks_project_detail:
            return True
        asks_past_activity = bool(
            re.search(r"\bwhat\s+did\s+(?:u|you)\s+do+\b", incoming_norm)
            or re.search(r"\bwhat\s+(?:did\s+)?(?:u|you)\s+(?:train(?:ed|ing)?|hit)\b", incoming_norm)
            or re.search(r"\bwhat\s+(?:are\s+|r\s+)?(?:u|you)\s+train(?:ing)?\b", incoming_norm)
        )
        if not asks_past_activity:
            return False
        if re.search(r"\bwhat\s+did\s+(?:u|you)\s+do+\b", incoming_norm):
            return True
        if any(term in incoming_norm for term in ("gym", "work", "coding", "course", "project", "client")):
            return True
        recent_owner_replies = " ".join(normalize_text(item) for item in context[-6:])
        return any(term in recent_owner_replies for term in ("gym", "work", "coding", "course", "project", "client", "got back"))

    def _catbot_activity_detail_scope(self, incoming_norm: str, context: list[str]) -> str:
        blob = f"{incoming_norm} " + " ".join(normalize_text(item) for item in context[-6:])
        if any(term in blob for term in ("boxing", "boxer", "box ", "do u box", "do you box")):
            return "sport_skill"
        if "gym" in blob:
            return "gym"
        if any(term in blob for term in ("training", "workout", "weights")):
            return "gym"
        if any(term in blob for term in ("coding", "code", "project", "client")):
            return "coding_or_project"
        if "work" in blob:
            return "work"
        if "course" in blob:
            return "course"
        if any(term in blob for term in ("phone", "bed", "laid", "laying", "lying", "chilling", "watching", "thinking about")):
            return "low_activity"
        return "activity"

    def _catbot_activity_opinion_disclosure(self, incoming_norm: str, context: list[str]) -> bool:
        activity_terms = ("legs", "leg day", "gym", "training", "cardio", "workout")
        has_activity_topic = any(term in incoming_norm for term in activity_terms)
        has_opinion = (
            re.search(r"\b(?:i\s+)?(?:hate|dont like|don't like|can't stand|cant stand)\b", incoming_norm)
            or any(term in incoming_norm for term in ("legs are evil", "leg day is evil", "legs r evil"))
        )
        if has_activity_topic and has_opinion:
            return True
        recent_activity_context = any(term in " ".join(normalize_text(item) for item in context[-6:]) for term in activity_terms)
        return bool(re.search(r"\b(?:hate it|dont like it|don't like it|cant stand it|can't stand it)\b", incoming_norm) and recent_activity_context)

    def _catbot_reciprocal_activity_question(self, incoming_norm: str) -> bool:
        incoming_clean = incoming_norm.strip(" .?!")
        status_then_bare_u = re.search(
            r"\b(?:im|i'm|i\s+am|just|been|still)?\s*"
            r"(?:good|fine|alright|okay|ok|calm|chilling|chillin|tired|busy|working|coding|in\s+bed|got\s+back|woke\s+up)"
            r"\b(?:\s+\w+){0,5}\s+(?:u|you)$",
            incoming_clean,
        )
        if status_then_bare_u:
            return True
        return any(
            term in incoming_norm
            for term in (
                "wby",
                "wbu",
                "hru",
                "hbu",
                "wyd",
                "what u doing",
                "what you doing",
                "what u up to",
                "what you up to",
                "what u been up to",
                "what you been up to",
                "how u doing",
                "how you doing",
                "how r u",
                "how are u",
                "how are you",
                "u good",
                "you good",
                "u okay",
                "you okay",
                "are u good",
                "are you good",
                "are u okay",
                "are you okay",
                "how was ur day",
                "how was your day",
                "hows ur day",
                "how's ur day",
                "how u been",
                "how you been",
            )
        )

    def _catbot_tired_status_disclosure(self, incoming_norm: str) -> bool:
        incoming_clean = incoming_norm.strip(" .?!")
        return bool(
            re.search(
                r"\b(?:im|i'm|i\s+am|i feel|feeling)?\s*(?:so\s+|proper\s+|bare\s+)?(?:tired|knackered|drained|exhausted|finished|sleepy)\b",
                incoming_clean,
            )
        )

    def _catbot_busy_status_disclosure(self, incoming_norm: str) -> bool:
        incoming_clean = incoming_norm.strip(" .?!")
        return bool(re.search(r"\b(?:(?:i'?ve|i\s+have|i)\s+been|im|i'm|i\s+am|been)\s+busy\b|\bbusy\s+asf\b", incoming_clean))

    def _catbot_low_key_status_disclosure(self, incoming_norm: str) -> bool:
        incoming_clean = incoming_norm.strip(" .?!")
        return bool(
            re.search(r"\b(?:i'?m|i\s+am)\s+(?:just\s+)?chill(?:ing|in)\b", incoming_clean)
            or re.search(r"\b(?:i'?m|i\s+am)\s+(?:just\s+)?(?:in\s+)?bed\b", incoming_clean)
            or re.search(r"\b(?:i'?m|i\s+am)\s+(?:just\s+)?(?:lying|laying)\s+(?:in\s+)?bed\b", incoming_clean)
            or re.search(r"\b(?:i'?m|i\s+am)\s+(?:just\s+)?(?:on\s+)?my\s+phone\b", incoming_clean)
        )

    def _catbot_rest_status_disclosure(self, incoming_norm: str) -> bool:
        incoming_clean = incoming_norm.strip(" .?!")
        return bool(
            re.search(r"\b(?:just\s+)?woke\s+up\b", incoming_clean)
            or re.search(r"\b(?:just\s+)?got\s+up\b", incoming_clean)
            or re.search(r"\b(?:just\s+)?had\s+(?:a\s+)?nap\b", incoming_clean)
            or re.search(r"\b(?:i'?m|i\s+am)\s+(?:still\s+)?half\s+asleep\b", incoming_clean)
            or re.search(r"\b(?:about\s+to|gonna|going\s+to|tryna|trying\s+to)\s+sleep\b", incoming_clean)
            or re.search(r"\b(?:about\s+to|gonna|going\s+to|tryna|trying\s+to)\s+(?:go\s+)?bed\b", incoming_clean)
        )

    def _catbot_rest_status_phase(self, incoming_norm: str) -> str:
        incoming_clean = incoming_norm.strip(" .?!")
        if (
            re.search(r"\b(?:about\s+to|gonna|going\s+to|tryna|trying\s+to)\s+sleep\b", incoming_clean)
            or re.search(r"\b(?:about\s+to|gonna|going\s+to|tryna|trying\s+to)\s+(?:go\s+)?bed\b", incoming_clean)
        ):
            return "sleep_intent"
        if re.search(r"\b(?:just\s+)?had\s+(?:a\s+)?nap\b", incoming_clean):
            return "post_nap"
        return "wake_up"

    def _catbot_activity_status_disclosure(self, incoming_norm: str) -> str:
        incoming_clean = incoming_norm.strip(" .?!")
        is_activity_update = bool(
            re.search(r"\b(?:just\s+)?(?:got|came|come)\s+back\s+from\s+(?:the\s+)?(?:gym|work|uni|university|class|lecture)\b", incoming_clean)
            or re.search(r"\b(?:just\s+)?(?:been|was)\s+(?:to\s+)?(?:the\s+)?(?:gym|work|uni|university|class|lecture)\b", incoming_clean)
            or re.search(r"\b(?:finished|done)\s+(?:at|with)?\s*(?:the\s+)?(?:gym|work|uni|university|class|lecture)\b", incoming_clean)
            or re.search(r"\b(?:hit|trained|did)\s+(?:the\s+)?(?:gym|legs|push|pull|cardio|weights)\b", incoming_clean)
        )
        if not is_activity_update:
            return ""
        if any(term in incoming_clean for term in ("gym", "workout", "training", "trained", "legs", "push", "pull", "cardio", "weights")):
            return "gym_activity"
        if any(term in incoming_clean for term in ("work", "shift", "client", "clients")):
            return "work_activity"
        if any(term in incoming_clean for term in ("uni", "university", "class", "lecture")):
            return "study_activity"
        return "general_activity"

    def _catbot_status_reason_question(self, incoming_norm: str, context: list[str]) -> bool:
        incoming_clean = incoming_norm.strip(" .?!")
        asks_tired_reason = bool(
            re.search(r"\bwhy\s+(?:u|you|are\s+u|are\s+you|r\s+u)\s+(?:so\s+)?(?:tired|knackered|drained|exhausted|finished|sleepy)\b", incoming_clean)
            or re.search(r"\bwhy\s+(?:so\s+)?(?:tired|knackered|drained|exhausted|finished|sleepy)\b", incoming_clean)
        )
        if asks_tired_reason:
            return True
        return False

    def _catbot_playful_scold_incoming(self, incoming_norm: str) -> bool:
        incoming_clean = incoming_norm.strip(" .?!")
        return bool(
            incoming_clean in {"behave", "behave yourself", "calm down", "stop it", "shush", "shh"}
            or re.fullmatch(
                r"(?:behave|behave yourself|calm down|stop it|shush|shh)\s+(?:ngl|icl|lol|lmao|haha|fr|tho|tbh)",
                incoming_clean,
            )
        )

    def _catbot_playful_scold_followup(self, incoming_norm: str, context: list[str]) -> bool:
        incoming_clean = incoming_norm.strip(" .?!")
        if incoming_clean not in {"why", "why lol", "lol why", "why tho", "why though"}:
            return False
        previous_user = normalize_text(str(context[-2])) if len(context) >= 2 else ""
        previous_bot = normalize_text(str(context[-1])) if context else ""
        return self._catbot_playful_scold_incoming(previous_user) or any(
            term in previous_bot
            for term in (
                "i'll behave",
                "ill behave",
                "behave",
                "hands to myself",
                "be good",
                "for now",
            )
        )

    def _catbot_owner_activity_reply_ok(self, reply_norm: str, *, incoming: str) -> bool:
        if not self._catbot_owner_activity_question(normalize_text(incoming)):
            return False
        if any(term in reply_norm for term in ("what u been up to", "what you been up to", "what u doing", "what you doing", "wyd")):
            return False
        tokens = reply_norm.strip(" .?!").split()
        if len(tokens) < 4:
            return False
        return any(
            term in reply_norm
            for term in (
                "coding",
                "code",
                "project",
                "uni",
                "gym",
                "training",
                "work",
                "clients",
                "studying",
                "revision",
                "busy",
                "chilling",
                "nothing much",
                "not much",
                "bed",
                "phone",
                "shower",
                "food",
                "ate",
            )
        )

    def _catbot_dream_question(self, incoming_norm: str) -> bool:
        return bool(re.search(r"\b(?:what'?s|whats|what is)\s+(?:ur|your)\s+dream\b", incoming_norm))

    def _catbot_car_preference_question(self, incoming_norm: str) -> bool:
        incoming_clean = incoming_norm.strip(" .?!")
        return bool(
            re.search(r"\bwhat\s+car\s+(?:do\s+)?(?:u|you)\s+like\b", incoming_clean)
            or re.search(r"\bwhat\s+cars\s+(?:do\s+)?(?:u|you)\s+like\b", incoming_clean)
            or re.search(r"\b(?:what'?s|whats|what is)\s+(?:ur|your)\s+(?:dream\s+)?car\b", incoming_clean)
        )

    def _catbot_car_preference_context_question(self, incoming_norm: str, context: list[str]) -> bool:
        if self._catbot_car_preference_question(incoming_norm):
            return True
        incoming_clean = incoming_norm.strip(" .?!")
        if incoming_clean not in {"what about u", "what about you", "wbu", "wby"}:
            return False
        recent = " ".join(normalize_text(item) for item in context[-4:])
        return any(term in recent for term in ("what car", "what cars", "dream car", "r8", "audi", "m4", "m3", "amg", "g wagon", "range rover"))

    def _catbot_car_preference_reply_ok(self, reply_norm: str, *, incoming: str, context: list[str]) -> bool:
        if not self._catbot_car_preference_context_question(normalize_text(incoming), context):
            return False
        if self._catbot_generic_ai_style(reply_norm, incoming=incoming, context=context):
            return False
        if any(term in reply_norm for term in ("what u mean", "what you mean", "what car", "idk", "dunno")):
            return False
        tokens = reply_norm.strip(" .?!").split()
        if not 3 <= len(tokens) <= 18:
            return False
        return any(term in reply_norm for term in ("r8", "audi", "rs", "m4", "m3", "amg", "g wagon", "range rover"))

    def _catbot_dream_reply_ok(self, reply_norm: str, *, incoming: str, context: list[str]) -> bool:
        if not self._catbot_dream_question(normalize_text(incoming)):
            return False
        if self._catbot_generic_ai_style(reply_norm, incoming=incoming, context=context):
            return False
        if any(term in reply_norm for term in ("dream what", "what dream", "to drive or", "to achieve", "what u mean", "what you mean")):
            return False
        tokens = reply_norm.strip(" .?!").split()
        if not 4 <= len(tokens) <= 24:
            return False
        return any(term in reply_norm for term in ("r8", "car", "business", "build", "building", "project", "successful", "serious"))

    def _catbot_prayer_question(self, incoming_norm: str) -> bool:
        incoming_clean = incoming_norm.strip(" .?!")
        return bool(
            re.search(r"\b(?:do\s+(?:u|you)|dya|dyou|do\s+ya)\s+pray\b", incoming_clean)
            or re.search(r"\b(?:are\s+(?:u|you)|r\s+u)\s+(?:religious|muslim)\b", incoming_clean)
            or re.search(r"\b(?:do\s+(?:u|you)|dya|dyou|do\s+ya)\s+(?:believe\s+in\s+god|go\s+mosque)\b", incoming_clean)
        )

    def _catbot_prayer_reply_ok(self, reply_norm: str, *, incoming: str, context: list[str]) -> bool:
        if not self._catbot_prayer_question(normalize_text(incoming)):
            return False
        if self._catbot_generic_ai_style(reply_norm, incoming=incoming, context=context):
            return False
        if any(term in reply_norm for term in ("what u mean", "what you mean", "pray what", "religious how", "idk", "dunno")):
            return False
        tokens = reply_norm.strip(" .?!").split()
        if not 5 <= len(tokens) <= 22:
            return False
        has_prayer_fact = any(term in reply_norm for term in ("pray", "prayer", "praying", "mosque", "religious", "muslim", "alhamdulillah"))
        has_human_qualifier = any(
            term in reply_norm
            for term in (
                "try",
                "trying",
                "not perfect",
                "when i can",
                "sometimes",
                "need to be better",
                "could be better",
                "alhamdulillah",
            )
        )
        return has_prayer_fact and has_human_qualifier

    def _catbot_reciprocal_activity_reply_ok(self, reply_norm: str, *, incoming: str) -> bool:
        if not self._catbot_reciprocal_activity_question(normalize_text(incoming)):
            return False
        if self._catbot_generic_ai_style(reply_norm, incoming=incoming, context=[]):
            return False
        if re.match(r"^(?:wby|wbu|hru|hbu|wyd|what\s+(?:u|you)\s+(?:doing|been up to)|how\s+(?:u|you)\s+doing)\b", reply_norm.strip(" .?!")):
            return False
        tokens = reply_norm.strip(" .?!").split()
        if len(tokens) < 3:
            return False
        has_answer = any(
            term in reply_norm
            for term in (
                "good",
                "grind",
                "busy",
                "chilling",
                "chill",
                "work",
                "gym",
                "coding",
                "uni",
                "nothing much",
                "not much",
                "same",
                "alright",
                "calm",
                "tired",
                "bed",
                "woke",
                "waking",
                "sleep",
                "slept",
                "nap",
                "got in",
                "just got",
                "phone",
                "shower",
                "food",
                "ate",
                "eating",
                "brothers",
                "finished",
                "thinking",
                "thinking about u",
                "thinking about you",
                "thinking of u",
                "thinking of you",
                "distracted",
                "mind",
            )
        )
        has_mirror = any(term in reply_norm for term in ("what u doing", "what u been up to", "what you doing", "wby", "wbu", "hru", "hbu", "wyd"))
        if has_mirror and "bare_status_mirror" in classify_reply_shape(reply_norm):
            return False
        if has_mirror and not has_answer:
            return False
        return has_answer

    def _catbot_activity_detail_reply_ok(self, reply_norm: str, *, incoming: str, context: list[str]) -> bool:
        incoming_norm = normalize_text(incoming)
        if not (self._catbot_whereabouts_question(incoming_norm) or self._catbot_activity_detail_question(incoming_norm, context)):
            return False
        if self._catbot_generic_ai_style(reply_norm, incoming=incoming, context=context):
            return False
        scope = self._catbot_activity_detail_scope(incoming_norm, context)
        tokens = reply_norm.strip(" .?!").split()
        if len(tokens) < 4:
            return False
        if scope == "gym":
            return any(
                term in reply_norm
                for term in (
                    "legs",
                    "push",
                    "pull",
                    "chest",
                    "back",
                    "arms",
                    "shoulder",
                    "weights",
                    "cardio",
                    "bench",
                    "squat",
                    "deadlift",
                    "machines",
                    "workout",
                    "trained",
                    "hit ",
                    "sets",
                )
            )
        if scope == "low_activity":
            return any(
                term in reply_norm
                for term in (
                    "phone",
                    "bed",
                    "laid",
                    "laying",
                    "lying",
                    "watching",
                    "scrolling",
                    "clips",
                    "tiktok",
                    "food",
                    "ate",
                    "eating",
                    "thinking",
                    "telly",
                    "show",
                    "film",
                    "random stuff",
                )
            )
        if scope == "sport_skill":
            return any(
                term in reply_norm
                for term in (
                    "boxing",
                    "box",
                    "cardio",
                    "footwork",
                    "pads",
                    "rounds",
                    "spar",
                    "jab",
                    "punch",
                    "tiring",
                    "hard",
                    "killer",
                    "humbles",
                    "humbled",
                    "trained",
                    "used to",
                )
            )
        if scope == "whereabouts":
            return any(
                term in reply_norm
                for term in (
                    "been",
                    "busy",
                    "work",
                    "working",
                    "gym",
                    "out",
                    "home",
                    "around",
                    "sorting",
                    "doing",
                    "finished",
                    "running about",
                    "deliveries",
                    "chilling",
                )
            )
        return any(term in reply_norm for term in ("did", "been", "working", "coding", "gym", "work", "finished", "sorted", "made", "hit "))

    def _catbot_really_followup(self, incoming_norm: str) -> bool:
        return incoming_norm.strip(" .?!") in {"really", "fr", "for real", "actually", "swear"}

    def _catbot_sincerity_challenge(self, incoming_norm: str, context: list[str]) -> bool:
        incoming_clean = incoming_norm.strip(" .?!")
        incoming_clean = re.sub(r"\s+(?:tbh|ngl|icl|lowk|lowkey)$", "", incoming_clean).strip()
        if incoming_clean not in {"no like actually", "like actually", "nah actually", "no actually", "actually fr", "fr actually"}:
            return False
        if self._catbot_story_reality_confirmation(incoming_norm, context):
            return False
        context_blob = " ".join(normalize_text(item) for item in context[-8:])
        return any(
            term in context_blob
            for term in (
                "like talking to u",
                "like talking to you",
                "really like u",
                "really like you",
                "i like u",
                "i like you",
                "like u",
                "like you",
                "wanna talk to u",
                "wanna talk to you",
                "want to talk to u",
                "want to talk to you",
                "need to talk to u",
                "need to talk to you",
                "keep talking to u",
                "keep talking to you",
                "talk to u all the time",
                "talk to you all the time",
                "can't get enough",
                "cant get enough",
                "missed u",
                "missed you",
                "miss u",
                "miss you",
                "want u",
                "want you",
                "wanna kiss",
                "want to kiss",
                "kiss u",
                "kiss you",
                "need u",
                "need you",
                "thinking about u",
                "thinking about you",
                "think about u",
                "think about you",
                "think about",
                "proper like u",
                "proper like you",
                "i mean it",
                "mean it",
                "doubting me",
                "doubt me",
                "doubting",
                "fr and",
                "rather be sure",
                "wanted to keep chatting",
                "keep chatting",
                "annoy u",
                "annoy you",
                "annoying",
                "fr fr",
                "just saying all this",
                "saying all this",
                "for a laugh",
                "make all that up",
                "make that up",
                "drawn to u",
                "drawn to you",
                "obsessed with u",
                "obsessed with you",
                "boring u",
                "boring you",
                "trying too hard",
                "overthinking",
                "forcing random questions",
                "got nervous",
                "was nervous",
                "nervous talking",
                "proper fit",
            )
        )

    def _catbot_really_reply_ok(self, reply_norm: str, *, incoming: str) -> bool:
        if not self._catbot_really_followup(normalize_text(incoming)):
            return False
        if self._catbot_generic_ai_style(reply_norm, incoming=incoming, context=[]):
            return False
        tokens = reply_norm.strip(" .?!").split()
        if len(tokens) < 4:
            return False
        return any(term in reply_norm for term in ("yeah", "yh", "course", "really", "fr", "icl", "ngl", "why would", "what u doubting"))

    def _catbot_sincerity_challenge_reply_ok(self, reply_norm: str, *, incoming: str, context: list[str]) -> bool:
        if not self._catbot_sincerity_challenge(normalize_text(incoming), context):
            return False
        if self._catbot_generic_ai_style(reply_norm, incoming=incoming, context=context):
            return False
        tokens = reply_norm.strip(" .?!").split()
        if not 5 <= len(tokens) <= 18:
            return False
        has_confirmation = any(term in reply_norm for term in ("yeah", "yh", "actually", "really", "fr", "i mean it", "mean it"))
        has_specificity = any(term in reply_norm for term in ("like talking to u", "like talking to you", "talk to u", "talk to you", "properly", "get enough", "miss", "thinking", "want", "wanna", "kiss", "boring u", "boring you", "tried too hard", "trying too hard", "overthinking", "filling silence"))
        return has_confirmation and has_specificity


    def _catbot_previous_statement_reason_topic(self, previous_bot_norm: str) -> str:
        if not previous_bot_norm or "?" in previous_bot_norm:
            return ""
        if any(term in previous_bot_norm for term in ("switch off after work", "head switch off", "head feels fried", "after work icl")):
            return "work_switch_off"
        if any(term in previous_bot_norm for term in ("same here", "watching some clips", "watching random clips", "random clips")):
            return "low_key_same"
        if any(term in previous_bot_norm for term in ("im good", "i'm good", "i am good", "yh im good", "yeah im good", "im calm", "i'm calm", "im okay", "i'm okay", "im alright", "i'm alright")):
            return "status_good_reason"
        if any(term in previous_bot_norm for term in ("good for u", "good for you")):
            return "good_for_you_chilling"
        if any(term in previous_bot_norm for term in ("i like how", "funny", "sounded", "after all that")):
            return "previous_comment_reaction"
        if any(term in previous_bot_norm for term in ("being silly", "bit silly", "got stuck", "stuck in my head", "nervous", "awkward", "burnt out", "trying to fill silence", "filling silence")):
            return "weak_fr_confirmation"
        if any(term in previous_bot_norm for term in ("overthinking", "being dumb", "trying too hard", "sound smart", "sounding fake", "went sideways")):
            return "overthinking_reason"
        if len(previous_bot_norm.strip(" .?!").split()) >= 3:
            return "previous_statement_generic"
        return ""

    def _catbot_quality_callout(self, incoming_norm: str) -> bool:
        incoming_clean = incoming_norm.strip(" .?!")
        if self._catbot_specificity_request(incoming_norm):
            return True
        return bool(
            re.search(r"\b(?:thats|that's|that is|ur|youre|you're|u are|you are|this is|this)\s+(?:so\s+)?(?:dry|boring|dead|vague)\b", incoming_clean)
            or re.search(r"\b(?:being|moving)\s+(?:so\s+)?(?:dry|boring|dead|vague)\b", incoming_clean)
            or re.search(r"\b(?:dry|boring|dead|vague)\s+(?:mate|bro|still|icl|ngl)?\b", incoming_clean)
        )

    def _catbot_quality_callout_type(self, incoming_norm: str) -> str:
        incoming_clean = incoming_norm.strip(" .?!")
        if self._catbot_specificity_request(incoming_clean):
            return "specificity"
        if re.search(r"\b(?:boring|dead)\b", incoming_clean):
            return "boring"
        if re.search(r"\b(?:vague)\b", incoming_clean):
            return "vague"
        if re.search(r"\b(?:dry)\b", incoming_clean):
            return "dry"
        return "quality"

    def _catbot_specificity_request(self, incoming_norm: str) -> bool:
        incoming_clean = incoming_norm.strip(" .?!")
        return bool(
            re.search(r"\b(?:be|get)\s+more\s+specific\b", incoming_clean)
            or re.search(r"\bmore\s+specific\b", incoming_clean)
            or re.search(r"\bsay\s+it\s+(?:properly|specifically)\b", incoming_clean)
            or re.search(r"\b(?:stop|dont|don't)\s+being\s+(?:vague|dry)\b", incoming_clean)
        )

    def _catbot_story_detail_context(self, context: list[str]) -> bool:
        recent = [normalize_text(item) for item in context[-6:]]
        if not recent:
            return False
        invite_terms = (
            "what happened",
            "what is it",
            "what isit",
            "what was it",
            "go on",
            "tell me then",
            "say it",
            "spill",
            "dont leave me hanging",
            "don't leave me hanging",
        )
        setup_terms = ("guess what", "weirdest day", "weird day", "crazy day", "something happened", "had a story")
        return any(term in recent[-1] for term in invite_terms) or (
            any(any(term in item for term in setup_terms) for item in recent)
            and any(any(term in item for term in invite_terms) for item in recent)
        )

    def _catbot_story_detail_incoming(self, incoming_norm: str, context: list[str]) -> bool:
        if not self._catbot_story_detail_context(context):
            return False
        tokens = incoming_norm.strip(" .?!").split()
        if len(tokens) < 4 or "?" in incoming_norm:
            return False
        event_markers = (
            "i ",
            "my ",
            "me ",
            "some ",
            "someone",
            "somebody",
            "guy",
            "girl",
            "man",
            "woman",
            "person",
            "mate",
            "home",
            "house",
            "room",
            "work",
            "school",
            "shop",
        )
        action_markers = (
            "came",
            "come",
            "walked",
            "ran",
            "started",
            "said",
            "told",
            "saw",
            "heard",
            "found",
            "went",
            "was",
            "were",
            "did",
            "happened",
        )
        return any(term in incoming_norm for term in event_markers) and any(term in incoming_norm for term in action_markers)

    def _catbot_reveal_prompt_bounce(self, incoming_norm: str, context: list[str]) -> bool:
        incoming_clean = incoming_norm.strip(" .?!")
        if incoming_clean not in {"tell me then", "u tell me", "you tell me", "you tell me then"}:
            return False
        if not context:
            return False
        previous_bot = normalize_text(str(context[-1]))
        if not self._catbot_invite_reveal_reply_ok(previous_bot):
            return False
        recent_user = " ".join(normalize_text(item) for item in context[::2][-4:])
        return self._catbot_guess_what_incoming(recent_user)

    def _catbot_story_reality_confirmation(self, incoming_norm: str, context: list[str]) -> bool:
        incoming_clean = incoming_norm.strip(" .?!")
        if incoming_clean not in {"nah like actually", "like actually", "nah actually", "actually", "fr actually", "bruh wtf", "wtf", "nah wtf", "bruh what the fuck"}:
            return False
        recent = " ".join(normalize_text(item) for item in context[-6:])
        return any(term in recent for term in ("random guy", "living room", "some guy", "came in", "ran in", "running in"))

    def _catbot_story_reality_confirmation_reply_ok(self, reply_norm: str, *, incoming: str, context: list[str]) -> bool:
        if not self._catbot_story_reality_confirmation(normalize_text(incoming), context):
            return False
        if self._catbot_generic_ai_style(reply_norm, incoming=incoming, context=context):
            return False
        tokens = reply_norm.strip(" .?!").split()
        if not 5 <= len(tokens) <= 24:
            return False
        has_reaction = any(term in reply_norm for term in ("nah", "wtf", "mad", "serious", "actually", "thats crazy", "that's crazy", "are u okay", "u okay"))
        has_followup = "?" in reply_norm or any(term in reply_norm for term in ("did he", "what happened", "then what", "run out", "leave", "where did"))
        return has_reaction and has_followup

    def _catbot_story_hypothetical_question(self, incoming_norm: str, context: list[str]) -> bool:
        incoming_clean = incoming_norm.strip(" .?!")
        if incoming_clean not in {"what would u do", "what would you do", "what would u do then", "what would you do then"}:
            return False
        recent = " ".join(normalize_text(item) for item in context[-8:])
        return any(term in recent for term in ("random guy", "some guy", "living room", "came in", "ran in", "running in", "in ur home", "in your home"))

    def _catbot_story_hypothetical_reply_ok(self, reply_norm: str, *, incoming: str, context: list[str]) -> bool:
        if not self._catbot_story_hypothetical_question(normalize_text(incoming), context):
            return False
        if self._catbot_generic_ai_style(reply_norm, incoming=incoming, context=context):
            return False
        tokens = reply_norm.strip(" .?!").split()
        if not 6 <= len(tokens) <= 24:
            return False
        has_self = any(term in reply_norm for term in ("id ", "i'd ", "i would", "im ", "i'm "))
        has_reaction = any(term in reply_norm for term in ("confused", "scared", "shout", "ask", "freeze", "run", "call", "grab", "mad", "wtf"))
        has_story_reference = any(term in reply_norm for term in ("guy", "bro", "home", "house", "living room", "who are u", "who are you", "random"))
        return has_self and has_reaction and has_story_reference

    def _catbot_previous_comment_clarification(self, incoming_norm: str, context: list[str]) -> bool:
        incoming_clean = incoming_norm.strip(" .?!")
        if incoming_clean in {"wdym", "what do u mean", "what do you mean", "what you mean", "what u mean"} and context:
            previous_bot = normalize_text(str(context[-1]))
            if previous_bot and "?" not in previous_bot:
                return True
        prediction = predict_conversation_function(incoming=incoming_norm, context=context)
        return prediction.function == "clarification_followup" and prediction.score >= 0.75

    def _catbot_previous_comment_clarification_reply_ok(self, reply_norm: str, *, incoming: str, context: list[str]) -> bool:
        if not self._catbot_previous_comment_clarification(normalize_text(incoming), context):
            return False
        if self._catbot_generic_ai_style(reply_norm, incoming=incoming, context=context):
            return False
        tokens = reply_norm.strip(" .?!").split()
        if not 7 <= len(tokens) <= 24:
            return False
        clarification_topic = str(predict_conversation_function(incoming=incoming, context=context).slots.get("clarification_topic") or "")
        has_clarifier = any(term in reply_norm for term in ("i mean", "meant", "meaning"))
        if clarification_topic == "physical_compliment":
            has_content = any(term in reply_norm for term in ("outfit", "suited", "looked", "proper on u", "proper on you", "fit", "clinging", "tight"))
        elif clarification_topic == "wrong_context":
            has_content = any(term in reply_norm for term in ("random", "wrong context", "made no sense", "answered the wrong", "ignore that"))
        elif clarification_topic == "conversation_effort":
            has_content = any(term in reply_norm for term in ("carry the convo", "carry it", "do all the work", "doing all the work", "making u carry", "making you carry", "drag the convo", "actually chatting", "adding something"))
        elif clarification_topic == "owner_state_or_activity":
            has_content = any(term in reply_norm for term in ("work", "coding", "project", "gym", "food", "shower", "bed", "watching", "chill", "switch off", "switching off", "week", "tired", "long day", "dead day", "drained"))
        else:
            has_content = any(term in reply_norm for term in ("stuck", "running out", "forcing", "random questions", "looping", "things to say", "intense", "back and forth", "a lot", "reacting to", "what i said"))
        return has_clarifier and has_content

    def _catbot_story_entities(self, incoming_norm: str) -> list[str]:
        entities: list[str] = []
        for term in ("guy", "girl", "man", "woman", "person", "someone", "somebody", "living room", "room", "home", "house", "work", "shower", "showering"):
            if term in incoming_norm:
                entities.append(term)
        return entities[:4]

    def _catbot_story_detail_reply_ok(self, reply_norm: str, *, incoming: str, context: list[str]) -> bool:
        if self._catbot_generic_ai_style(reply_norm, incoming=incoming, context=context):
            return False
        if any(term in reply_norm for term in ("tell me more", "tell me everything")):
            return False
        tokens = reply_norm.strip(" .?!").split()
        if not 3 <= len(tokens) <= 24:
            return False
        incoming_entities = self._catbot_story_entities(normalize_text(incoming))
        references_detail = bool(incoming_entities) and any(entity in reply_norm for entity in incoming_entities)
        reacts_to_event = any(
            term in reply_norm
            for term in (
                "wait",
                "nah",
                "wtf",
                "what the",
                "no way",
                "are u okay",
                "are you okay",
                "u okay",
                "you okay",
                "how did",
                "why did",
                "who was",
                "was he",
                "was she",
                "random",
                "scary",
                "mad",
                "bro",
            )
        )
        asks_followup = "?" in reply_norm or any(term in reply_norm for term in ("what did", "how did", "why did", "who was", "then what"))
        return reacts_to_event and (references_detail or asks_followup)

    def _catbot_without_trailing_discourse_fillers(self, text: str) -> str:
        words = text.split()
        fillers = {"tbh", "ngl", "icl", "lowk", "lowkey", "fr", "please", "pls"}
        while words and words[-1] in fillers:
            words.pop()
        return " ".join(words)

    def _catbot_conversation_move(self, *, incoming: str, context: list[str], conversation_function: ConversationFunctionPrediction | None = None) -> dict[str, object]:
        incoming_norm = normalize_text(incoming)
        incoming_clean = incoming_norm.strip(" .?!")
        incoming_function_clean = self._catbot_without_trailing_discourse_fillers(incoming_clean)
        context_blob = " ".join(normalize_text(item) for item in context[-10:])
        active_romantic = self._catbot_romantic_mode(incoming_norm, context) or self._catbot_adult_mode(incoming_norm, context)
        adult_context = active_romantic or self._catbot_explicit_adult_context(context)
        adult_followup = self._catbot_adult_followup(incoming_norm, context)
        conversation_function = conversation_function or predict_conversation_function(incoming=incoming, context=context)
        slots = self._catbot_extract_move_slots(
            incoming_norm=incoming_norm,
            context=context,
            conversation_function=conversation_function,
        )
        escalation_terms = (
            "how so",
            "how exactly",
            "what else",
            "that all",
            "about what",
            "what exactly",
            "tell me then",
            "tell me something",
            "say something then",
            "go on then",
            "and then",
            "and?",
            "prove it",
            "what would u do",
            "what would you do",
            "what would u do to",
            "what would you do to",
        )
        curiosity_terms = ("guess what", "guess", "you know what")
        story_hook_terms = (
            "weirdest day",
            "weird day",
            "craziest day",
            "crazy day",
            "something happened",
            "had a story",
            "had the weirdest",
        )
        carry_terms = ("im bored", "i'm bored", "bored", "entertain me", "u pick", "you pick", "carry the convo", "carry convo", "talk then", "idk talk", "talk to me", "say something", "say something then")
        repair_callout_terms = (
            "repeating",
            "keep asking",
            "already asked",
            "asked that",
            "answer properly",
            "say it properly",
            "stop asking",
            "why u asking",
            "why you asking",
            "stupid qs",
            "stupid questions",
            "being dry",
            "ur dry",
            "youre dry",
            "you're dry",
            "thats dry",
            "that's dry",
            "that is dry",
            "boring me",
            "ur boring",
            "u boring",
            "youre boring",
            "you're boring",
            "be more specific",
            "more specific",
            "being weird",
            "why u being weird",
            "why you being weird",
            "i asked u a question",
            "i asked you a question",
            "asked u a question",
            "asked you a question",
            "are u gonna answer",
            "are you gonna answer",
            "u gonna answer",
            "you gonna answer",
        )
        topic_dismissal_terms = ("forget that anyway", "forget it anyway", "forget that", "forget it", "leave it", "leave that")
        low_effort_ack_terms = {"same", "same tbh", "same icl", "same ngl", "nice", "oh fairs", "oh fair", "fairs", "fair", "okay then", "ok then", "trust me"}
        playful_challenge_terms = ("would u fight me", "would you fight me", "fight me then", "could u beat me", "could you beat me")
        if conversation_function.function == "clarification_followup" and conversation_function.score >= 0.75:
            return {
                "scene": "active_romantic_thread" if active_romantic else "casual_thread",
                "user_move": "unclassified",
                "bot_move_required": "clarify_previous_bot_claim",
                "minimum_reply_shape": "explain the previous bot statement with concrete content",
                "slots": slots,
            }
        if conversation_function.function in {"quality_complaint", "specificity_request"} and conversation_function.score >= 0.85:
            return {
                "scene": "active_romantic_thread" if active_romantic else "casual_thread",
                "user_move": "repair_callout",
                "bot_move_required": CATBOT_MOVE_TAXONOMY["repair_callout"]["required_bot_move"],
                "minimum_reply_shape": "own the miss plus give a specific next move",
                "slots": slots,
            }
        if conversation_function.function == "story_hypothetical" and conversation_function.score >= 0.85:
            return {
                "scene": "story_thread",
                "user_move": "unclassified",
                "bot_move_required": "answer_story_hypothetical",
                "minimum_reply_shape": "say what you would do in the story",
                "slots": slots,
            }
        if conversation_function.function == "bot_answer_own_prompt" and conversation_function.score >= 0.85:
            return {
                "scene": "active_romantic_thread" if active_romantic else "casual_thread",
                "user_move": "unclassified",
                "bot_move_required": "answer_own_previous_prompt",
                "minimum_reply_shape": "answer the concrete prompt the bot just asked",
                "slots": slots,
            }
        if conversation_function.function == "sincerity_challenge" and conversation_function.score >= 0.85:
            return {
                "scene": "active_romantic_thread" if active_romantic else "romantic_thread",
                "user_move": "intensity_check",
                "bot_move_required": CATBOT_MOVE_TAXONOMY["intensity_check"]["required_bot_move"],
                "minimum_reply_shape": "confirm sincerely with a specific reason",
                "slots": slots,
            }
        if incoming_norm.strip(" .?!") == "how crazy" and self._catbot_intensity_followup(incoming_norm, context) and not self._catbot_explicit_adult_context(context):
            return {
                "scene": "active_romantic_thread" if active_romantic else "romantic_thread",
                "user_move": "intensity_check",
                "bot_move_required": CATBOT_MOVE_TAXONOMY["intensity_check"]["required_bot_move"],
                "minimum_reply_shape": "confirm with a specific reason or escalation",
                "slots": slots,
            }
        if self._catbot_sincerity_challenge(incoming_norm, context):
            return {
                "scene": "active_romantic_thread" if active_romantic else "romantic_thread",
                "user_move": "intensity_check",
                "bot_move_required": CATBOT_MOVE_TAXONOMY["intensity_check"]["required_bot_move"],
                "minimum_reply_shape": "confirm sincerely with a specific reason",
                "slots": slots,
            }
        if self._catbot_status_reason_question(incoming_norm, context):
            return {
                "scene": "casual_thread",
                "user_move": "status_reason_question",
                "bot_move_required": CATBOT_MOVE_TAXONOMY["status_reason_question"]["required_bot_move"],
                "minimum_reply_shape": "answer why with a concrete tiredness reason",
                "slots": slots,
            }
        if self._catbot_tired_status_disclosure(incoming_norm) or self._catbot_busy_status_disclosure(incoming_norm) or self._catbot_rest_status_disclosure(incoming_norm) or self._catbot_activity_status_disclosure(incoming_norm):
            return {
                "scene": "casual_thread",
                "user_move": "status_disclosure",
                "bot_move_required": CATBOT_MOVE_TAXONOMY["status_disclosure"]["required_bot_move"],
                "minimum_reply_shape": "acknowledge the status/activity and add concrete owner-side continuation",
                "slots": slots,
            }
        if self._catbot_playful_scold_incoming(incoming_norm) or self._catbot_playful_scold_followup(incoming_norm, context):
            return {
                "scene": "active_romantic_thread" if active_romantic else "casual_thread",
                "user_move": "romantic_boundary_test",
                "bot_move_required": CATBOT_MOVE_TAXONOMY["romantic_boundary_test"]["required_bot_move"],
                "minimum_reply_shape": "playfully accept the scold and soften",
                "slots": slots,
            }
        if self._catbot_explicit_adult_request(incoming_norm, context) or self._catbot_intensity_followup(incoming_norm, context) or adult_followup or (adult_context and any(term in incoming_norm for term in escalation_terms)):
            return {
                "scene": "active_romantic_thread" if active_romantic else "romantic_thread",
                "user_move": "romantic_escalation",
                "bot_move_required": CATBOT_MOVE_TAXONOMY["romantic_escalation"]["required_bot_move"],
                "minimum_reply_shape": "concrete consensual body/action detail with lips, neck, hips, thighs, kiss, tease, press, wet, hard, or similar",
                "slots": slots,
            }
        if conversation_function.function == "continuation_prompt" and conversation_function.score >= 0.8:
            return {
                "scene": "active_romantic_thread" if active_romantic else "casual_thread",
                "user_move": "unclassified",
                "bot_move_required": "continue_previous_statement",
                "minimum_reply_shape": "continue the previous statement with concrete detail",
                "slots": slots,
            }
        if self._catbot_affection_reciprocity_incoming(incoming_norm):
            return {
                "scene": "romantic_thread",
                "user_move": "emotional_reciprocity",
                "bot_move_required": CATBOT_MOVE_TAXONOMY["emotional_reciprocity"]["required_bot_move"],
                "minimum_reply_shape": "reciprocate the affection with warmth and continue",
                "slots": slots,
            }
        if self._catbot_really_followup(incoming_norm):
            return {
                "scene": "active_romantic_thread" if active_romantic else "casual_thread",
                "user_move": "intensity_check",
                "bot_move_required": CATBOT_MOVE_TAXONOMY["intensity_check"]["required_bot_move"],
                "minimum_reply_shape": "confirm with a specific reason or escalation",
                "slots": slots,
            }
        if self._catbot_location_origin_question(incoming_norm):
            return {
                "scene": "active_romantic_thread" if active_romantic else "casual_thread",
                "user_move": "identity_fact_question",
                "bot_move_required": CATBOT_MOVE_TAXONOMY["identity_fact_question"]["required_bot_move"],
                "minimum_reply_shape": "answer safe city-level origin directly",
                "slots": slots,
            }
        if self._catbot_work_identity_question(incoming_norm):
            return {
                "scene": "active_romantic_thread" if active_romantic else "casual_thread",
                "user_move": "identity_fact_question",
                "bot_move_required": CATBOT_MOVE_TAXONOMY["identity_fact_question"]["required_bot_move"],
                "minimum_reply_shape": "answer work/activity identity directly",
                "slots": slots,
            }
        if self._catbot_age_question(incoming_norm):
            return {
                "scene": "active_romantic_thread" if active_romantic else "casual_thread",
                "user_move": "identity_fact_question",
                "bot_move_required": CATBOT_MOVE_TAXONOMY["identity_fact_question"]["required_bot_move"],
                "minimum_reply_shape": "answer age as 19 directly",
                "slots": slots,
            }
        if self._catbot_education_question(incoming_norm):
            return {
                "scene": "active_romantic_thread" if active_romantic else "casual_thread",
                "user_move": "education_status_question",
                "bot_move_required": CATBOT_MOVE_TAXONOMY["education_status_question"]["required_bot_move"],
                "minimum_reply_shape": "direct owner fact, no invented degree",
                "slots": slots,
            }
        if self._catbot_availability_planning_question(incoming_norm):
            return {
                "scene": "active_romantic_thread" if active_romantic else "casual_thread",
                "user_move": "availability_planning",
                "bot_move_required": CATBOT_MOVE_TAXONOMY["availability_planning"]["required_bot_move"],
                "minimum_reply_shape": "answer availability or plans status then continue",
                "slots": slots,
            }
        if self._catbot_whereabouts_question(incoming_norm) or self._catbot_activity_detail_question(incoming_norm, context):
            return {
                "scene": "active_romantic_thread" if active_romantic else "casual_thread",
                "user_move": "activity_question",
                "bot_move_required": CATBOT_MOVE_TAXONOMY["activity_question"]["required_bot_move"],
                "minimum_reply_shape": "answer previous owner activity or whereabouts with concrete detail",
                "slots": slots,
            }
        if self._catbot_activity_opinion_disclosure(incoming_norm, context):
            return {
                "scene": "active_romantic_thread" if active_romantic else "casual_thread",
                "user_move": "activity_opinion_disclosure",
                "bot_move_required": CATBOT_MOVE_TAXONOMY["activity_opinion_disclosure"]["required_bot_move"],
                "minimum_reply_shape": "acknowledge activity opinion plus one concrete related comment",
                "slots": slots,
            }
        if self._catbot_owner_activity_question(incoming_norm):
            return {
                "scene": "active_romantic_thread" if active_romantic else "casual_thread",
                "user_move": "reciprocal_question",
                "bot_move_required": CATBOT_MOVE_TAXONOMY["reciprocal_question"]["required_bot_move"],
                "minimum_reply_shape": "answer recent owner activity with concrete detail",
                "slots": slots,
            }
        if self._catbot_reciprocal_activity_question(incoming_norm):
            return {
                "scene": "active_romantic_thread" if active_romantic else "casual_thread",
                "user_move": "reciprocal_question",
                "bot_move_required": CATBOT_MOVE_TAXONOMY["reciprocal_question"]["required_bot_move"],
                "minimum_reply_shape": "answer own status/activity before any return question",
                "slots": slots,
            }
        if any(term in incoming_norm for term in repair_callout_terms) or self._catbot_quality_callout(incoming_norm):
            user_move = "loop_callout" if slots.get("repair_target") == "loop_or_repetition" else "repair_callout"
            return {
                "scene": "active_romantic_thread" if active_romantic else "casual_thread",
                "user_move": user_move,
                "bot_move_required": CATBOT_MOVE_TAXONOMY[user_move]["required_bot_move"],
                "minimum_reply_shape": "own the miss plus give a specific next move",
                "slots": slots,
            }
        if any(term in incoming_norm for term in topic_dismissal_terms):
            return {
                "scene": "active_romantic_thread" if active_romantic else "casual_thread",
                "user_move": "topic_dismissal_reset",
                "bot_move_required": CATBOT_MOVE_TAXONOMY["topic_dismissal_reset"]["required_bot_move"],
                "minimum_reply_shape": "acknowledge dropped topic plus owner-side reset without broad question",
                "slots": slots,
            }
        if any(term in incoming_norm for term in curiosity_terms):
            return {
                "scene": "active_romantic_thread" if active_romantic else "casual_thread",
                "user_move": "curiosity_hook",
                "bot_move_required": CATBOT_MOVE_TAXONOMY["curiosity_hook"]["required_bot_move"],
                "minimum_reply_shape": "curious pull, not one-word",
                "slots": slots,
            }
        if self._catbot_reveal_prompt_bounce(incoming_norm, context):
            return {
                "scene": "story_thread",
                "user_move": "story_hook",
                "bot_move_required": CATBOT_MOVE_TAXONOMY["story_hook"]["required_bot_move"],
                "minimum_reply_shape": "bounce the reveal prompt back because they started guess what",
                "slots": slots,
            }
        if any(term in incoming_norm for term in story_hook_terms):
            return {
                "scene": "active_romantic_thread" if active_romantic else "casual_thread",
                "user_move": "story_hook",
                "bot_move_required": CATBOT_MOVE_TAXONOMY["story_hook"]["required_bot_move"],
                "minimum_reply_shape": "react plus ask what happened",
                "slots": slots,
            }
        if self._catbot_story_detail_incoming(incoming_norm, context):
            return {
                "scene": "story_thread",
                "user_move": "story_detail_reveal",
                "bot_move_required": CATBOT_MOVE_TAXONOMY["story_detail_reveal"]["required_bot_move"],
                "minimum_reply_shape": "react to the concrete detail plus ask one specific follow-up",
                "slots": slots,
            }
        if any(term in incoming_norm for term in carry_terms):
            user_move = "boredom_prompt" if slots.get("conversation_need") == "relieve_boredom" else "carry_conversation_request"
            return {
                "scene": "active_romantic_thread" if active_romantic else "casual_thread",
                "user_move": user_move,
                "bot_move_required": CATBOT_MOVE_TAXONOMY[user_move]["required_bot_move"],
                "minimum_reply_shape": "specific topic plus direct question",
                "slots": slots,
            }
        if any(term in incoming_norm for term in playful_challenge_terms):
            return {
                "scene": "active_romantic_thread" if active_romantic else "casual_thread",
                "user_move": "insult_playful",
                "bot_move_required": CATBOT_MOVE_TAXONOMY["insult_playful"]["required_bot_move"],
                "minimum_reply_shape": "playful banter without serious threat or logistics",
                "slots": slots,
            }
        if incoming_function_clean in low_effort_ack_terms:
            return {
                "scene": "active_romantic_thread" if active_romantic else "casual_thread",
                "user_move": "low_effort_ack",
                "bot_move_required": CATBOT_MOVE_TAXONOMY["low_effort_ack"]["required_bot_move"],
                "minimum_reply_shape": "acknowledge the small reaction and add one concrete related comment",
                "slots": slots,
            }
        if self._catbot_greeting_like_incoming(incoming_norm):
            return {
                "scene": "active_romantic_thread" if active_romantic else "casual_thread",
                "user_move": "affectionate_greeting",
                "bot_move_required": CATBOT_MOVE_TAXONOMY["affectionate_greeting"]["required_bot_move"],
                "minimum_reply_shape": "hey/hi plus a small hook",
                "slots": slots,
            }
        return {
            "scene": "active_romantic_thread" if active_romantic else "casual_thread",
            "user_move": "unclassified",
            "bot_move_required": "continue_energy",
            "minimum_reply_shape": "specific and emotionally congruent",
            "slots": slots,
        }

    def _catbot_extract_move_slots(
        self,
        *,
        incoming_norm: str,
        context: list[str],
        conversation_function: ConversationFunctionPrediction | None = None,
    ) -> dict[str, object]:
        slots: dict[str, object] = {}
        conversation_function = conversation_function or predict_conversation_function(incoming=incoming_norm, context=context)
        if conversation_function.function != "ordinary_reply" and conversation_function.score:
            slots["conversation_function"] = conversation_function.function
            slots["conversation_function_score"] = str(conversation_function.score)
            for key, value in conversation_function.slots.items():
                if value:
                    slots[key] = value
        if conversation_function.function == "clarification_followup" and conversation_function.score >= 0.75:
            slots["unclassified_context"] = "previous_comment_clarification"
        elif conversation_function.function == "story_hypothetical" and conversation_function.score >= 0.85:
            slots["unclassified_context"] = "story_hypothetical_reaction"
        elif conversation_function.function == "bot_answer_own_prompt" and conversation_function.score >= 0.85:
            slots["unclassified_context"] = "answer_own_previous_prompt"
        elif conversation_function.function == "continuation_prompt" and conversation_function.score >= 0.8:
            slots["unclassified_context"] = "continue_previous_statement"
        elif conversation_function.function == "specificity_request" and conversation_function.score >= 0.85:
            slots["repair_target"] = "specificity_request"
        elif conversation_function.function == "quality_complaint" and conversation_function.score >= 0.85:
            slots["repair_target"] = "dry_or_unclear_reply"
            slots["quality_callout_type"] = self._catbot_quality_callout_type(incoming_norm)
        context_blob = " ".join(normalize_text(item) for item in context[-10:])
        repeated_reset_topic = self._catbot_repeated_reset_topic(context)
        repeated_reset_question_loop = self._catbot_repeated_reset_question_loop(context)
        called_out_question_family = self._catbot_called_out_question_family(incoming_norm)
        if self._catbot_education_question(incoming_norm):
            slots["identity_fact"] = "education_status"
            slots["entities"] = ["university", "course", "student_status"]
            slots["education_query"] = (
                "study_subject"
                if re.search(r"\b(?:course|study|studying)\b", incoming_norm)
                else "university_status"
            )
            previous_bot = normalize_text(str(context[-1])) if context else ""
            if any(term in previous_bot for term in ("comp sci", "computer science", "sampleford uni", "sampleford university", "sampleford")):
                slots["recent_education_answered"] = True
                slots["previous_education_reply"] = previous_bot
                recent_education_replies = [
                    normalize_text(item)
                    for item in context[1::2][-8:]
                    if any(term in normalize_text(item) for term in ("comp sci", "computer science", "sampleford uni", "sampleford university", "sampleford"))
                ]
                if recent_education_replies:
                    slots["recent_education_replies"] = "|".join(recent_education_replies)
        if self._catbot_missed_you_incoming(incoming_norm) or self._catbot_miss_me_question(incoming_norm):
            slots["affection_signal"] = "missed_you"
            slots["affection_question"] = self._catbot_miss_me_question(incoming_norm)
        elif self._catbot_affection_disclosure_incoming(incoming_norm):
            slots["affection_signal"] = "thinking_of_you"
        if self._catbot_really_followup(incoming_norm):
            recent_context = " ".join(normalize_text(item) for item in context[-4:])
            slots["challenge_target"] = "missed_you_confirmation" if any(term in recent_context for term in ("missed u", "miss you", "missed you", "miss u")) else "prior_intensity_or_affection"
        if self._catbot_sincerity_challenge(incoming_norm, context) or (conversation_function.function == "sincerity_challenge" and conversation_function.score >= 0.85):
            slots["challenge_target"] = "sincerity_confirmation"
            previous_bot = normalize_text(str(context[-1])) if context else ""
            recent_context = " ".join(normalize_text(item) for item in context[-6:])
            if any(term in previous_bot for term in ("wanna kiss", "want to kiss", "kiss u", "kiss you")):
                slots["sincerity_topic"] = "physical_affection"
            elif any(term in previous_bot for term in ("boring u", "boring you", "trying too hard", "overthinking", "forcing random questions")):
                slots["sincerity_topic"] = "boring_anxiety"
        if self._catbot_age_question(incoming_norm):
            slots["identity_fact"] = "age"
            slots["entities"] = ["age"]
        if self._catbot_location_origin_question(incoming_norm):
            slots["identity_fact"] = "location_origin"
            slots["entities"] = ["location", "origin"]
        if self._catbot_work_identity_question(incoming_norm):
            slots["identity_fact"] = "work_status"
            slots["entities"] = ["work", "occupation"]
        if any(term in incoming_norm for term in ("forget that anyway", "forget it anyway", "forget that", "forget it", "leave it", "leave that")):
            slots["dismissal_target"] = "previous_thread"
        if any(term in incoming_norm for term in ("im bored", "i'm bored", "bored")):
            slots["conversation_need"] = "relieve_boredom"
            slots["topic_seed"] = "pick_specific_thread"
        elif any(term in incoming_norm for term in ("entertain me", "u pick", "you pick", "carry the convo", "carry convo", "talk then", "idk talk", "talk to me", "say something", "say something then")):
            slots["conversation_need"] = "carry_or_pick_topic"
            slots["topic_seed"] = "pick_specific_thread"
        if slots.get("topic_seed") == "pick_specific_thread" and repeated_reset_topic:
            slots["repeated_reset_topic"] = repeated_reset_topic
        if any(term in incoming_norm for term in ("repeating", "keep asking", "already asked", "asked that")):
            slots["repair_target"] = "loop_or_repetition"
            if "repeated_reset_topic" not in slots and not repeated_reset_question_loop and not called_out_question_family:
                slots["repeated_reset_question_loop"] = True
        elif self._catbot_specificity_request(incoming_norm):
            slots["repair_target"] = "specificity_request"
        elif self._catbot_quality_callout(incoming_norm):
            slots["repair_target"] = "dry_or_unclear_reply"
            slots["quality_callout_type"] = self._catbot_quality_callout_type(incoming_norm)
        elif any(term in incoming_norm for term in ("answer properly", "say it properly", "being dry", "ur dry", "youre dry", "you're dry")):
            slots["repair_target"] = (
                "loop_or_repetition"
                if repeated_reset_topic or repeated_reset_question_loop or any(term in context_blob for term in ("repeating", "keep asking", "already asked", "asked that", "loop"))
                else "dry_or_unclear_reply"
            )
        elif any(term in incoming_norm for term in ("being weird", "why u being weird", "why you being weird")):
            slots["repair_target"] = "weird_or_unclear_reply"
        elif any(term in incoming_norm for term in ("i asked u a question", "i asked you a question", "asked u a question", "asked you a question", "are u gonna answer", "are you gonna answer", "u gonna answer", "you gonna answer")):
            slots["repair_target"] = "missed_question_callout"
        elif any(term in incoming_norm for term in ("why u asking", "why you asking", "stop asking", "stupid qs", "stupid questions")):
            slots["repair_target"] = "question_quality_callout"
        if slots.get("repair_target") == "loop_or_repetition" and any(term in incoming_norm for term in ("answer properly", "say it properly")):
            previous_reasons = [
                normalize_text(item)
                for item in context[1::2][-4:]
                if detect_reason_answer(normalize_text(item))
            ]
            if previous_reasons:
                slots["callout_question"] = "deepen_why_reason"
                signatures: set[str] = set()
                for reason in previous_reasons:
                    signatures.update(detect_reason_signatures(reason))
                if signatures:
                    slots["previous_reason_signatures"] = "|".join(sorted(signatures))
        if slots.get("repair_target") == "loop_or_repetition" and repeated_reset_topic:
            slots["repeated_reset_topic"] = repeated_reset_topic
        if slots.get("repair_target") == "loop_or_repetition" and called_out_question_family:
            slots["repeated_reset_topic"] = called_out_question_family
        if slots.get("repair_target") == "loop_or_repetition" and (repeated_reset_question_loop or slots.get("repeated_reset_question_loop")):
            slots["repeated_reset_question_loop"] = True
        if slots.get("repair_target") == "loop_or_repetition" and "callout_question" not in slots and (
            "why" in incoming_norm or any("why" in normalize_text(item) and "keep asking" in normalize_text(item) for item in context[-6:])
        ):
            slots["callout_question"] = "why_loop_reason"
        if incoming_norm.strip(" .?!") == "how crazy" and self._catbot_intensity_followup(incoming_norm, context) and not self._catbot_explicit_adult_context(context):
            slots["challenge_target"] = "prior_intensity_or_affection"
        elif self._catbot_explicit_adult_request(incoming_norm, context) or self._catbot_intensity_followup(incoming_norm, context) or self._catbot_adult_followup(incoming_norm, context):
            slots["escalation_level"] = "explicit_or_direct"
            slots["detail_required"] = True
            recent_adult_sequence = any(
                len([part for part in re.split(r"\s*/\s*|\n+", str(previous_bot_reply)) if part.strip()]) >= 3
                and bool(detect_adult_detail_families(str(previous_bot_reply)))
                for previous_bot_reply in context[1::2][-4:]
            )
            if self._catbot_adult_followup(incoming_norm, context):
                previous_bot_raw = str(context[-1]) if context else ""
                previous_bot_bubbles = [part for part in re.split(r"\s*/\s*|\n+", previous_bot_raw) if part.strip()]
                adult_followup_clean = incoming_norm.strip(" .?!")
                slots["adult_followup_text"] = adult_followup_clean
                if self._catbot_intensity_followup(incoming_norm, context) and adult_followup_clean == "how much" and recent_adult_sequence:
                    slots["escalation_source"] = "adult_continuation_burst"
                elif re.fullmatch(r"(?:m{2,}|m+h*m+)", adult_followup_clean):
                    slots["escalation_source"] = (
                        "adult_continuation_burst"
                        if len(context) >= 4 and len(previous_bot_bubbles) >= 3
                        else "sensual_ack_followup"
                    )
                else:
                    keep_going_after_single_detail = (
                        adult_followup_clean in {"go on then", "and", "and?", "prove it"}
                        and len(context) >= 4
                        and 0 < len(previous_bot_bubbles) <= 2
                    )
                    keep_going_after_multi_detail = (
                        adult_followup_clean == "what else"
                        and len(context) >= 4
                        and len(previous_bot_bubbles) >= 3
                    )
                    slots["escalation_source"] = (
                        "adult_continuation_burst"
                        if keep_going_after_single_detail or keep_going_after_multi_detail
                        else
                        "adult_invitation_followup"
                        if any(term in incoming_norm for term in ("come here", "come closer", "kiss me then", "show me then"))
                        else "adult_body_followup"
                        if any(term in incoming_norm for term in ("need ur", "need your", "want ur", "want your", "ur lips", "your lips", "hands on"))
                        else "explicit_adult_followup"
                    )
            elif self._catbot_explicit_adult_context(context) and any(term in incoming_norm for term in ("prove it", "and?", "go on then", "tell me then", "how so")):
                slots["escalation_source"] = "explicit_adult_followup"
            elif self._catbot_intensity_followup(incoming_norm, context) and (self._catbot_explicit_adult_context(context) or recent_adult_sequence):
                slots["escalation_source"] = "adult_continuation_burst"
            elif self._catbot_explicit_adult_request(incoming_norm, context):
                slots["escalation_source"] = "direct_adult_request"
            previous_user = normalize_text(str(context[-2])) if len(context) >= 2 else ""
            if self._catbot_explicit_adult_context(context) or recent_adult_sequence or (previous_user and self._catbot_adult_followup(previous_user, context[:-2])):
                recent_detail_families: set[str] = set()
                adult_family_window = context[1::2][-5:] if slots.get("escalation_source") == "adult_continuation_burst" and recent_adult_sequence else context[1::2][-3:]
                for previous_bot_reply in adult_family_window:
                    recent_detail_families.update(detect_adult_detail_families(str(previous_bot_reply)))
                if recent_detail_families:
                    slots["recent_adult_detail_families"] = "|".join(sorted(recent_detail_families))
        if any(term in incoming_norm for term in ("guess what", "guess", "you know what")):
            slots["reveal_signal"] = "teaser"
        if any(term in incoming_norm for term in ("weirdest day", "weird day", "crazy day", "something happened")):
            slots["story_signal"] = "story_setup"
        if self._catbot_story_detail_incoming(incoming_norm, context):
            slots["story_signal"] = "story_detail"
            entities = self._catbot_story_entities(incoming_norm)
            if entities:
                slots["story_entities"] = entities
        if self._catbot_reveal_prompt_bounce(incoming_norm, context):
            slots["story_signal"] = "reveal_prompt_bounce"
            slots["response_type"] = "reveal_prompt_bounce"
        if self._catbot_reciprocal_activity_question(incoming_norm):
            wellbeing_check = bool(
                re.search(r"\b(?:u|you)\s+(?:good|okay)\b", incoming_norm)
                or re.search(r"\bare\s+(?:u|you)\s+(?:good|okay)\b", incoming_norm)
            )
            slots["question_topic"] = "wellbeing_or_activity" if "how" in incoming_norm or wellbeing_check else "activity"
            if context:
                previous_bot = normalize_text(str(context[-1]))
                previous_shape = self._catbot_reply_skeleton(previous_bot, incoming=incoming_norm)
                if previous_shape in {"plain_status_answer", "status_answer_plus_return"} or any(
                    term in previous_bot
                    for term in ("gym", "legs", "training", "work", "coding", "call", "calls", "bed", "nap", "shower", "phone", "netflix", "watching", "game", "show")
                ):
                    slots["recent_status_answered"] = True
                if any(term in previous_bot for term in ("gym", "legs", "training", "weights")):
                    slots["recent_status_topic"] = "gym"
                elif any(term in previous_bot for term in ("work", "coding", "call", "calls", "project")):
                    slots["recent_status_topic"] = "work"
                elif any(term in previous_bot for term in ("bed", "nap", "sleep", "half asleep", "thinking")):
                    slots["recent_status_topic"] = "rest"
                elif any(term in previous_bot for term in ("watching", "game", "show", "netflix")):
                    slots["recent_status_topic"] = "media"
        if self._catbot_tired_status_disclosure(incoming_norm):
            slots["status_topic"] = "tiredness"
        elif self._catbot_busy_status_disclosure(incoming_norm):
            slots["status_topic"] = "busy"
        elif self._catbot_rest_status_disclosure(incoming_norm):
            slots["status_topic"] = "rest_activity"
            slots["rest_status_phase"] = self._catbot_rest_status_phase(incoming_norm)
        else:
            activity_status_topic = self._catbot_activity_status_disclosure(incoming_norm)
            if activity_status_topic:
                slots["status_topic"] = activity_status_topic
        if self._catbot_status_reason_question(incoming_norm, context):
            slots["status_topic"] = "tiredness"
            slots["reason_target"] = "why_tired"
        if self._catbot_owner_activity_question(incoming_norm):
            slots["question_topic"] = "owner_recent_activity"
            slots["owner_activity_question"] = "recent_activity"
        if self._catbot_dream_question(incoming_norm):
            slots["unclassified_context"] = "dream_or_ambition_question"
            if any(term in context_blob for term in ("car", "cars", "r8", "drive")):
                slots["dream_context"] = "car_or_ambition"
        if self._catbot_car_preference_context_question(incoming_norm, context):
            slots["unclassified_context"] = "car_preference_question"
        if self._catbot_story_reality_confirmation(incoming_norm, context):
            slots["unclassified_context"] = "story_reality_confirmation"
            slots["story_confirmation_text"] = "nah_like_actually" if "actually" in incoming_norm else "shock_ack"
        if self._catbot_story_hypothetical_question(incoming_norm, context):
            slots["unclassified_context"] = "story_hypothetical_reaction"
        if self._catbot_previous_comment_clarification(incoming_norm, context):
            slots["unclassified_context"] = "previous_comment_clarification"
            previous_bot = normalize_text(str(context[-1])) if context else ""
            if any(term in previous_bot for term in ("that was intense", "intense icl", "back and forth")):
                slots["clarification_topic"] = "intense_exchange"
            elif any(term in previous_bot for term in ("doing all the work", "do all the work", "carry the convo", "carry it", "making u carry", "making you carry", "making u do", "making you do")):
                slots["clarification_topic"] = "conversation_effort"
            elif any(term in previous_bot for term in ("legs", "leg workout", "workout", "gym", "food", "rest", "shower", "long day", "tired", "fried")):
                slots["clarification_topic"] = "owner_state_or_activity"
                if any(term in previous_bot for term in ("legs", "leg workout", "workout", "gym")):
                    slots["clarification_activity_topic"] = "gym_or_legs"
            elif any(term in previous_bot for term in ("chill", "switch off", "switching off", "weekend", "after this week")):
                slots["clarification_topic"] = "owner_state_or_activity"
                slots["clarification_activity_topic"] = "chill_after_week"
        if self._catbot_prayer_question(incoming_norm):
            slots["unclassified_context"] = "faith_prayer_question"
        if self._catbot_low_key_status_disclosure(incoming_norm):
            slots["unclassified_context"] = "low_key_status_disclosure"
        if incoming_norm.strip(" .?!") in {"yeah why", "yh why", "why"} or (
            conversation_function.function == "reason_followup" and conversation_function.score >= 0.8
        ):
            previous_bot = normalize_text(str(context[-1])) if context else ""
            if any(term in previous_bot for term in ("what we watching", "what are we watching", "what should we watch")):
                slots["unclassified_context"] = "reason_for_previous_question"
                slots["reason_topic"] = "watching_question"
            elif any(term in previous_bot for term in ("what u up to", "what you up to", "what u doing", "what you doing", "what was it", "what was the question", "what u chilling", "what you chilling", "what u mean by chilling", "what you mean by chilling")):
                slots["unclassified_context"] = "reason_for_previous_question"
            else:
                statement_reason_topic = self._catbot_previous_statement_reason_topic(previous_bot)
                if statement_reason_topic:
                    slots["unclassified_context"] = "reason_for_previous_comment"
                    slots["reason_topic"] = statement_reason_topic
        if self._catbot_playful_scold_followup(incoming_norm, context):
            slots["boundary_signal"] = "playful_scold_followup"
        elif self._catbot_playful_scold_incoming(incoming_norm):
            slots["boundary_signal"] = "playful_scold"
        if self._catbot_greeting_like_incoming(incoming_norm) and context:
            recent_bot_replies = [
                normalize_text(str(item))
                for item in context[1::2][-8:]
                if str(item).strip()
            ]
            if any(
                self._catbot_reply_skeleton(previous_bot, incoming=incoming_norm) == "greeting_plus_question_hook"
                for previous_bot in recent_bot_replies
            ):
                slots["recent_greeting_shape"] = "greeting_plus_question_hook"
        if self._catbot_activity_detail_question(incoming_norm, context):
            activity_scope = self._catbot_activity_detail_scope(incoming_norm, context)
            slots["activity_scope"] = activity_scope
            if activity_scope == "low_activity":
                low_blob = " ".join(normalize_text(item) for item in context[-6:])
                if any(term in low_blob for term in ("scrolling", "phone")):
                    slots["recent_low_activity_topic"] = "phone_scroll"
        if self._catbot_activity_opinion_disclosure(incoming_norm, context):
            slots["opinion_polarity"] = "negative"
            if any(term in incoming_norm for term in ("legs", "leg day")):
                slots["activity_topic"] = "legs"
            elif "cardio" in incoming_norm:
                slots["activity_topic"] = "cardio"
            elif "gym" in incoming_norm:
                slots["activity_topic"] = "gym"
            else:
                slots["activity_topic"] = "activity"
        if self._catbot_whereabouts_question(incoming_norm):
            slots["activity_scope"] = "whereabouts"
        incoming_clean = incoming_norm.strip(" .?!")
        incoming_function_clean = self._catbot_without_trailing_discourse_fillers(incoming_clean)
        if incoming_function_clean in {"same", "trust me"}:
            slots["ack_type"] = "agreement"
        elif incoming_function_clean in {"nice", "oh fairs", "oh fair", "fairs", "fair", "okay then", "ok then"}:
            slots["ack_type"] = "light_ack"
        if slots.get("ack_type"):
            previous_bot = normalize_text(str(context[-1])) if context else ""
            if any(term in previous_bot for term in ("laundry", "finish up some laundry")):
                slots["ack_context_topic"] = "plans_activity"
            elif any(term in previous_bot for term in ("shower", "food", "sleep", "chill", "later", "bed")):
                slots["ack_context_topic"] = "plans_rest"
            elif any(term in previous_bot for term in ("legs", "leg day", "stairs", "gym", "workout", "training")):
                slots["ack_context_topic"] = "activity_legs"
            elif "boxing" in previous_bot or "cardio" in previous_bot:
                slots["ack_context_topic"] = "boxing"
        if any(term in incoming_norm for term in ("would u fight me", "would you fight me", "fight me then", "could u beat me", "could you beat me")):
            slots["tease_target"] = "play_fight_challenge"
        if self._catbot_availability_planning_question(incoming_norm):
            slots["timeframe"] = "now" if self._catbot_awake_status_question(incoming_norm) else "later_or_general"
            slots["plan_object"] = "awake_status" if self._catbot_awake_status_question(incoming_norm) else "general_plans"
            if context and slots["plan_object"] == "general_plans":
                recent_plan_detail = detect_availability_plan_detail(str(context[-1]))
                if recent_plan_detail:
                    slots["recent_plan_detail"] = recent_plan_detail
        return slots

    def _catbot_reset_topic_family(self, reply_norm: str) -> str:
        return detect_specific_topic(reply_norm)

    def _catbot_repeated_reset_topic(self, context: list[str]) -> str:
        bot_replies = [normalize_text(item) for item in context[1::2] if str(item).strip()]
        families = [family for reply in bot_replies[-8:] if (family := self._catbot_reset_topic_family(reply))]
        for family in set(families):
            if families.count(family) >= 2:
                return family
        return ""

    def _catbot_repeated_reset_question_loop(self, context: list[str]) -> bool:
        bot_replies = [normalize_text(item) for item in context[1::2] if str(item).strip()]
        recent = bot_replies[-6:]
        if sum(1 for reply in recent if detect_reset_question(reply)) >= 2:
            return True
        return sum(1 for reply in recent if "?" in reply and self._catbot_reset_topic_family(reply)) >= 2

    def _catbot_conversation_move_instruction(self, move: dict[str, object]) -> str:
        user_move = str(move.get("user_move") or "")
        required = str(move.get("bot_move_required") or "")
        shape = str(move.get("minimum_reply_shape") or "")
        if user_move == "unclassified":
            slots = move.get("slots") or {}
            unclassified_context = str(slots.get("unclassified_context") or "")
            if unclassified_context in {"reason_for_previous_comment", "reason_for_previous_question"}:
                return (
                    f"Conversation function: {slots.get('conversation_function') or 'reason_followup'}, slots={slots}. "
                    "They are asking why you said or asked the previous thing. Answer the reason directly with cos/because or a clear concrete reason. "
                    "Do not ask why they are asking, do not dodge, and do not start a new topic.\n"
                )
            if unclassified_context == "previous_comment_clarification":
                return (
                    f"Conversation function: clarification_followup, slots={slots}. "
                    "They asked what you meant by your previous reply. Explain the previous claim directly with concrete wording from that claim. "
                    "Do not give a bare acknowledgement and do not ask what they mean.\n"
                )
            if unclassified_context == "story_hypothetical_reaction":
                return (
                    f"Conversation function: story_hypothetical, slots={slots}. "
                    "They asked what you would do in the story. Answer in first person with a concrete reaction to the story detail. "
                    "Do not ask what they mean and do not restart the story.\n"
                )
            if unclassified_context == "answer_own_previous_prompt":
                return (
                    f"Conversation function: bot_answer_own_prompt, slots={slots}. "
                    "They bounced your previous concrete prompt back to you. Answer your own prompt directly with one concrete personal example. "
                    "Do not ask them to answer it, do not ask what they mean, and do not start a new topic.\n"
                )
            if unclassified_context == "continue_previous_statement":
                return (
                    f"Conversation function: continuation_prompt, slots={slots}. "
                    "They said go on/continue after your previous statement. Continue that exact thought with a deeper concrete detail. "
                    "Do not ask a question, do not restart the topic, and do not switch into adult mode unless the recent context is explicitly adult.\n"
                )
            return ""
        if user_move == "loop_callout":
            return (
                f"Conversation move: scene={move.get('scene')}, user_move={user_move}, "
                f"bot_move_required={required}, slots={move.get('slots')}. "
                "They are calling out repeated/generic questions. This is a high-stakes repair moment. If slots include callout_question=deepen_why_reason, use exactly: deeper/different explanation + acknowledgement, with no question and no repeated reason. If slots include callout_question=why_loop_reason, use exactly: brief reason + acknowledgement, with no question and no new topic. If slots include repeated_reset_question_loop=true, use exactly: acknowledgement + one concrete owner-side detail or mini story, with no question. Otherwise use exactly: acknowledgement + one concrete fresh reset thread. Do not ask them to choose, do not use a vague 'what should we talk about', and do not reuse a repeated topic family.\n"
            )
        if user_move == "repair_callout":
            slots = move.get("slots") or {}
            if slots.get("repair_target") == "question_quality_callout":
                return (
                    f"Conversation move: scene={move.get('scene')}, user_move={user_move}, "
                    f"bot_move_required={required}, slots={move.get('slots')}. "
                    "They are complaining that your previous question was bad/stupid. Use exactly: acknowledge the bad question + one owner-side reason/reset, with no question. Do not ask another question, do not ask what they want to talk about, and do not pick another random question.\n"
                )
            return (
                f"Conversation move: scene={move.get('scene')}, user_move={user_move}, "
                f"bot_move_required={required}, slots={move.get('slots')}. "
                "Acknowledge the miss, answer or reset clearly, and avoid another vague question.\n"
            )
        if user_move == "topic_dismissal_reset":
            return (
                f"Conversation move: scene={move.get('scene')}, user_move={user_move}, "
                f"bot_move_required={required}, slots={move.get('slots')}. "
                "They are dropping the previous thread after a bad/generic moment. Acknowledge and move on with an owner-side detail or self-reset. Do not ask what they want to talk about, do not ask them to choose, and do not use a broad question.\n"
            )
        if user_move == "education_status_question":
            return (
                f"Conversation move: scene={move.get('scene')}, user_move={user_move}, "
                f"bot_move_required={required}, slots={move.get('slots')}. "
                "They are asking an identity fact about uni/study. Use the identity pack and answer directly: computer science / comp sci at Sampleford uni. Do not say not at uni, and do not invent a different course or university.\n"
            )
        if user_move == "identity_fact_question":
            slots = move.get("slots") or {}
            if str(slots.get("identity_fact") or "") == "work_status":
                return (
                    f"Conversation move: scene={move.get('scene')}, user_move={user_move}, "
                    f"bot_move_required={required}, slots={slots}. "
                    "They are asking what you do. Use the identity pack: you study computer science at uni and have something running on the side. "
                    "Do not say you work for your uncle and do not dodge with a generic status.\n"
                )
            if str(slots.get("identity_fact") or "") == "location_origin":
                return (
                    f"Conversation move: scene={move.get('scene')}, user_move={user_move}, "
                    f"bot_move_required={required}, slots={slots}. "
                    "They are asking where you are from. Answer directly with vague city-level origin only: Northbridge/Northbridge and Sampleford. "
                    "Do not disclose exact area, street, accommodation, or live location, and do not say Sampleford for uni if the recent chat says not at uni.\n"
                )
            if str(slots.get("identity_fact") or "") == "age":
                return (
                    f"Conversation move: scene={move.get('scene')}, user_move={user_move}, "
                    f"bot_move_required={required}, slots={slots}. "
                    "They are asking your age. Use exactly: answer age as 19 + a textured continuation. "
                    "Do not give a dry mirror like '19 / wby'. Good continuations include 'why u asking like that', "
                    "'don't make it sound like an interview', or 'wby then'.\n"
                )
            return (
                f"Conversation move: scene={move.get('scene')}, user_move={user_move}, "
                f"bot_move_required={required}, slots={move.get('slots')}. "
                "They are asking an identity fact. If slots say identity_fact=age, answer 19 directly and do not invent another age.\n"
            )
        if user_move == "emotional_reciprocity":
            slots = move.get("slots") or {}
            affection_instruction = (
                "They are asking if you miss them. Answer yes directly, then continue with a specific affectionate or flirty line; do not dodge into generic chat."
                if slots.get("affection_question")
                else "They said they are thinking about you. Reciprocate that directly with a warm specific line; do not treat it as generic chat."
                if slots.get("affection_signal") == "thinking_of_you"
                else "They said they miss you. Reciprocate warmly and continue with a specific affectionate or flirty line; do not treat it as generic chat."
            )
            return (
                f"Conversation move: scene={move.get('scene')}, user_move={user_move}, "
                f"bot_move_required={required}, slots={move.get('slots')}. "
                f"{affection_instruction}\n"
            )
        if user_move == "intensity_check":
            return (
                f"Conversation move: scene={move.get('scene')}, user_move={user_move}, "
                f"bot_move_required={required}, slots={move.get('slots')}. "
                "They are checking if you mean it. Confirm with warmth or a specific escalation; do not answer with a bare yeah or a broad question.\n"
            )
        if user_move == "story_hook":
            return (
                f"Conversation move: scene={move.get('scene')}, user_move={user_move}, "
                f"bot_move_required={required}, slots={move.get('slots')}. "
                "They are setting up a story. React with energy and ask what happened; do not say 'tell me more' or 'tell me everything'.\n"
            )
        if user_move == "story_detail_reveal":
            return (
                f"Conversation move: scene={move.get('scene')}, user_move={user_move}, "
                f"bot_move_required={required}, slots={move.get('slots')}. "
                "They revealed the story detail. React to the concrete event with shock/concern and ask one specific follow-up. Do not ask what happened again and do not force romance.\n"
            )
        if user_move in {"boredom_prompt", "carry_conversation_request"}:
            return (
                f"Conversation move: scene={move.get('scene')}, user_move={user_move}, "
                f"bot_move_required={required}, slots={move.get('slots')}. "
                "They want you to lead the conversation. Use exactly: short decisive owner reaction + one chosen concrete topic question. Pick from food, film, gym, dream, weird day story, work story, or a quick random question. Do not ask what kind, what sort, what they want, or what they are craving.\n"
            )
        if user_move == "reciprocal_question":
            return (
                f"Conversation move: scene={move.get('scene')}, user_move={user_move}, "
                f"bot_move_required={required}, slots={move.get('slots')}. "
                "They asked how you are or what you are doing. Use exactly: answer your own status/activity first + optional short return. Start with a concrete status/activity like im good, im chilling, just got back from gym, working, coding, tired, or in bed. Do not start by asking them back.\n"
            )
        if user_move == "status_disclosure":
            return (
                f"Conversation move: scene={move.get('scene')}, user_move={user_move}, "
                f"bot_move_required={required}, slots={move.get('slots')}. "
                "They said they are tired or finished. Acknowledge that naturally and add one concrete owner-side detail or soft suggestion. Do not turn it into adult escalation just because earlier context was flirty.\n"
            )
        if user_move == "status_reason_question":
            return (
                f"Conversation move: scene={move.get('scene')}, user_move={user_move}, "
                f"bot_move_required={required}, slots={move.get('slots')}. "
                "They asked why you are tired. Answer with a concrete reason like work, coding, gym, sleep, or a long day. Do not continue adult mode, do not ask a broad question, and do not dodge with a generic line.\n"
            )
        if user_move == "romantic_boundary_test":
            return (
                f"Conversation move: scene={move.get('scene')}, user_move={user_move}, "
                f"bot_move_required={required}, slots={move.get('slots')}. "
                "They gave a playful scold like behave/calm down. Accept it playfully, soften the tease, and keep it warm. Do not ask a broad question and do not keep escalating explicit detail.\n"
            )
        if user_move == "activity_question":
            return (
                f"Conversation move: scene={move.get('scene')}, user_move={user_move}, "
                f"bot_move_required={required}, slots={move.get('slots')}. "
                "They are asking for detail about the owner activity you just mentioned. Answer the activity directly with one concrete detail. If the slot is gym, mention what you trained or did, like legs, push, pull, weights, cardio, or machines. If the slot is low_activity, answer with a concrete low-key detail like being on your phone, in bed, watching something, or thinking about them. Do not give a vague status update and do not ask the same question back.\n"
            )
        if user_move == "low_effort_ack":
            return (
                f"Conversation move: scene={move.get('scene')}, user_move={user_move}, "
                f"bot_move_required={required}, slots={move.get('slots')}. "
                "They gave a short acknowledgement or agreement. Do not mirror with a bare same/nice/fair. Add one concrete related comment from the current thread, and only ask a question if it is specific and not needed to carry the whole conversation.\n"
            )
        if user_move == "insult_playful":
            return (
                f"Conversation move: scene={move.get('scene')}, user_move={user_move}, "
                f"bot_move_required={required}, slots={move.get('slots')}. "
                "They are playfully challenging/teasing you. Reply with confident light banter in 1-2 short bubbles. Do not make it sound like a real threat, do not arrange a fight, and do not use a broad question.\n"
            )
        return (
            f"Conversation move: scene={move.get('scene')}, user_move={user_move}, "
            f"bot_move_required={required}, minimum_reply_shape={shape}. "
            "Satisfy this move semantically; do not key off exact wording.\n"
        )

    def _catbot_reply_plan(self, move: dict[str, object]) -> ReplyPlan:
        user_move = str(move.get("user_move") or "")
        common_slots = {str(key): str(value) for key, value in (move.get("slots") or {}).items() if isinstance(value, str)}
        if user_move == "loop_callout":
            slots = move.get("slots") or {}
            required_slots = {"repair_target": str(slots.get("repair_target") or "loop_or_repetition")}
            callout_question = str(slots.get("callout_question") or "")
            if callout_question:
                required_slots["callout_question"] = callout_question
            repeated_reset_topic = str(slots.get("repeated_reset_topic") or "")
            if repeated_reset_topic:
                required_slots["repeated_reset_topic"] = repeated_reset_topic
                required_slots["fresh_reset_options"] = "quick_owner_story|specific_film_food_or_gym_topic|answer_previous_point_directly"
                if repeated_reset_topic in {"day_status_question", "wellbeing_status_question"}:
                    required_slots["fresh_reset_options"] = "owner_side_detail_or_stop_interrogating_no_day_status_question"
            if slots.get("repeated_reset_question_loop"):
                required_slots["repeated_reset_question_loop"] = "true"
            previous_reason_signatures = str(slots.get("previous_reason_signatures") or "")
            if previous_reason_signatures:
                required_slots["previous_reason_signatures"] = previous_reason_signatures
            if callout_question == "deepen_why_reason":
                return ReplyPlan(
                    move=user_move,
                    tone="self_aware_direct",
                    shape="answer_properly_deepen_reason",
                    must_do=["give_deeper_or_different_reason", "acknowledge_loop"],
                    must_not_do=["repeat_previous_reason", "ask_question", "start_new_topic", "reuse_repeated_reset_topic"],
                    required_slots=required_slots,
                    forbidden_patterns=[
                        "what do you want to talk about",
                        "what do u wanna talk about",
                        "what should we talk about",
                        "what kind of bored",
                        "tell me more",
                        "how about",
                        "how bout",
                    ],
                    allowed_shapes=["deeper_reason + acknowledgement", "acknowledgement + deeper_reason"],
                    forbidden_shapes=["what_do_you_want_to_talk_about", "broad_question", "generic_ack", "trailing_reset", "repeated_reset_topic", "unrequested_reset_topic", "ask_question"],
                    max_bubbles=2,
                    max_chars=125,
                )
            if callout_question == "why_loop_reason":
                return ReplyPlan(
                    move=user_move,
                    tone="self_aware_direct",
                    shape="answer_why_then_stop_loop",
                    must_do=["answer_why", "acknowledge_loop"],
                    must_not_do=["ask_user_to_choose_topic", "start_new_topic", "ask_question", "reuse_repeated_reset_topic"],
                    required_slots=required_slots,
                    forbidden_patterns=[
                        "what do you want to talk about",
                        "what do u wanna talk about",
                        "what should we talk about",
                        "what kind of bored",
                        "tell me more",
                        "how about",
                        "how bout",
                    ],
                    allowed_shapes=["brief_reason + acknowledgement", "acknowledgement + brief_reason"],
                    forbidden_shapes=["what_do_you_want_to_talk_about", "broad_question", "generic_ack", "trailing_reset", "repeated_reset_topic", "unrequested_reset_topic"],
                    max_bubbles=2,
                    max_chars=105,
                )
            if slots.get("repeated_reset_question_loop"):
                return ReplyPlan(
                    move=user_move,
                    tone="self_aware_grounded",
                    shape="acknowledge_loop_then_owner_detail",
                    must_do=["acknowledge_loop", "give_owner_side_detail"],
                    must_not_do=["ask_question", "choose_another_topic_prompt", "reuse_repeated_reset_topic", "repeat_apology_template"],
                    required_slots=required_slots,
                    forbidden_patterns=[
                        "what do you want to talk about",
                        "what do u wanna talk about",
                        "what should we talk about",
                        "what kind of bored",
                        "tell me more",
                        "what's ur",
                        "whats ur",
                        "what's your",
                        "whats your",
                    ],
                    allowed_shapes=["acknowledgement + owner_detail", "acknowledgement + quick_owner_story"],
                    forbidden_shapes=["what_do_you_want_to_talk_about", "broad_question", "generic_ack", "trailing_reset", "repeated_reset_topic", "reset_question", "ask_question", "repeat_apology_template"],
                    max_bubbles=2,
                    max_chars=115,
                )
            if repeated_reset_topic in {"day_status_question", "wellbeing_status_question"}:
                return ReplyPlan(
                    move=user_move,
                    tone="self_aware_grounded",
                    shape="acknowledge_specific_question_family_without_repeat",
                    must_do=["acknowledge_loop", "name_or_stop_repeated_question", "give_owner_side_detail"],
                    must_not_do=["ask_question", "choose_another_topic_prompt", "reuse_repeated_reset_topic", "repeat_apology_template"],
                    required_slots=required_slots,
                    forbidden_patterns=[
                        "what do you want to talk about",
                        "what do u wanna talk about",
                        "what should we talk about",
                        "what kind of bored",
                        "tell me more",
                    ],
                    allowed_shapes=["acknowledgement + stop_interrogating + owner_detail", "acknowledgement + name_repeated_question + owner_detail"],
                    forbidden_shapes=["what_do_you_want_to_talk_about", "broad_question", "generic_ack", "trailing_reset", "repeated_reset_topic", "reset_question", "ask_question", "repeat_apology_template"],
                    max_bubbles=2,
                    max_chars=115,
                )
            return ReplyPlan(
                move=user_move,
                tone="self_aware_playful",
                shape="acknowledge_loop_then_fresh_reset",
                must_do=["acknowledge_loop", "choose_fresh_new_thread"],
                must_not_do=["ask_user_to_choose_topic", "repeat_previous_question", "reuse_repeated_reset_topic", "repeat_apology_template"],
                required_slots=required_slots,
                forbidden_patterns=["what do you want to talk about", "what do u wanna talk about", "what should we talk about", "what kind of bored", "tell me more"],
                allowed_shapes=["acknowledgement + chosen_fresh_topic", "acknowledgement + quick_owner_story", "acknowledgement + specific_question"],
                forbidden_shapes=["what_do_you_want_to_talk_about", "broad_question", "generic_ack", "trailing_reset", "repeated_reset_topic", "repeat_apology_template"],
                max_bubbles=3,
                max_chars=110,
            )
        if user_move == "repair_callout":
            question_quality_callout = common_slots.get("repair_target") == "question_quality_callout"
            if question_quality_callout:
                return ReplyPlan(
                    move=user_move,
                    tone="casual_accountable",
                    shape="acknowledge_bad_question_then_owner_side_reset",
                    must_do=["own_bad_question", "give_owner_side_reset_or_reason"],
                    must_not_do=["ask_question", "broad_question", "ask_user_to_choose_topic", "generic_ack"],
                    required_slots=common_slots,
                    forbidden_patterns=[
                        "what do you want to talk about",
                        "what do u wanna talk about",
                        "what do u wanna chat about",
                        "what should we talk about",
                        "what's on ur mind",
                        "what's on your mind",
                        "whats on ur mind",
                        "whats on your mind",
                        "what u been up to",
                        "what you been up to",
                        "what u up to",
                        "what you up to",
                        "what u doing",
                        "what you doing",
                        "what u thinking",
                        "what you thinking",
                        "tell me the weirdest",
                        "tell me the funniest",
                        "tell me the best",
                        "tell me something",
                        "pick one with me",
                        "u pick",
                        "you pick",
                    ],
                    allowed_shapes=["acknowledge_bad_question + owner_side_reason", "acknowledge_bad_question + stop_interrogating"],
                    forbidden_shapes=["ask_question", "broad_question", "ask_user_to_choose_topic", "generic_ack"],
                    max_bubbles=2,
                    max_chars=120,
                )
            return ReplyPlan(
                move=user_move,
                tone="casual_accountable",
                shape="acknowledge_mistake_plus_corrected_move",
                must_do=["own_miss", "correct_or_reset"],
                must_not_do=["broad_question", "ask_user_to_choose_topic", "generic_ack"],
                required_slots=common_slots,
                forbidden_patterns=[
                    "what do you want to talk about",
                    "what do u wanna talk about",
                    "what do u wanna chat about",
                    "what should we talk about",
                    "what's on ur mind",
                    "what's on your mind",
                    "whats on ur mind",
                    "whats on your mind",
                    "what u been up to",
                    "what you been up to",
                    "what u up to",
                    "what you up to",
                    "what u doing",
                    "what you doing",
                    "what u thinking",
                    "what you thinking",
                    "tell me the weirdest",
                    "tell me the funniest",
                    "tell me the best",
                    "tell me something",
                    "pick one with me",
                    "u pick",
                    "you pick",
                ],
                allowed_shapes=["acknowledge_mistake + corrected_move", "brief_reason + corrected_move"],
                forbidden_shapes=["broad_question", "ask_user_to_choose_topic", "generic_ack"],
                max_bubbles=2,
                max_chars=120,
            )
        if user_move == "topic_dismissal_reset":
            return ReplyPlan(
                move=user_move,
                tone="casual_accountable",
                shape="acknowledge_dismissal_then_owner_side_reset",
                must_do=["acknowledge_dismissal", "provide_owner_side_reset"],
                must_not_do=["broad_question", "ask_user_to_choose_topic", "generic_ack"],
                required_slots=common_slots,
                forbidden_patterns=["what do you want to talk about", "what do u wanna talk about", "what should we talk about", "give me a topic"],
                allowed_shapes=["acknowledge_drop + owner_side_detail", "acknowledge_drop + self_reset"],
                forbidden_shapes=["broad_question", "what_do_you_want_to_talk_about", "ask_user_to_choose_topic", "generic_ack"],
                max_bubbles=2,
                max_chars=110,
            )
        if user_move == "story_hook":
            return ReplyPlan(
                move=user_move,
                tone="curious_playful",
                shape="short_reaction_plus_specific_prompt",
                must_do=["show_energy", "invite_reveal"],
                must_not_do=["generic_tell_me_more", "broad_question"],
                required_slots={"response_type": str(common_slots.get("response_type") or "curious_invite")},
                forbidden_patterns=["tell me more", "tell me everything"],
                allowed_shapes=["short_reaction + what_happened_prompt"],
                forbidden_shapes=["generic_tell_me_more", "generic_ack", "broad_question"],
                max_bubbles=2,
                max_chars=90,
            )
        if user_move == "story_detail_reveal":
            story_entities = ",".join(str(item) for item in (move.get("slots") or {}).get("story_entities", []) if str(item)) or "event_detail"
            return ReplyPlan(
                move=user_move,
                tone="reactive_concerned",
                shape="specific_story_reaction_plus_followup",
                must_do=["react_to_event", "reference_or_follow_up_on_detail"],
                must_not_do=["generic_tell_me_more", "ask_what_happened_again", "force_romance"],
                required_slots={"story_signal": "story_detail", "story_entities": story_entities},
                forbidden_patterns=["tell me more", "tell me everything", "what happened"],
                allowed_shapes=["specific_reaction + concrete_followup", "concern_reaction + detail_question"],
                forbidden_shapes=["generic_tell_me_more", "generic_ack", "broad_question"],
                max_bubbles=2,
                max_chars=110,
            )
        if user_move in {"boredom_prompt", "carry_conversation_request"}:
            carry_slots = dict(common_slots)
            carry_slots["required_shape_example"] = "nah im picking food then / best thing u ate this week?"
            carry_slots["reset_options"] = "weirdest_part_of_day|quick_owner_story|specific_film_food_or_gym_topic|dream_prompt|random_question"
            if carry_slots.get("repeated_reset_topic"):
                carry_slots["fresh_reset_options"] = "dream_prompt|food_prompt|film_prompt|work_prompt|specific_random_question"
            return ReplyPlan(
                move=user_move,
                tone="playful_decisive",
                shape="owner_picks_topic_plus_specific_question",
                must_do=["choose_topic", "ask_specific", "sound_decisive"],
                must_not_do=["ask_user_to_choose_topic", "broad_question", "reuse_repeated_reset_topic"],
                required_slots=carry_slots,
                forbidden_patterns=["what kind of bored", "what kinda bored", "what kind of entertainment", "what kinda entertainment", "what kind", "what kinda", "what sort", "u craving", "you craving", "u craved", "you craved", "craved", "what do you want to talk about"],
                allowed_shapes=["decisive_reaction + chosen_specific_question", "playful_reset + chosen_specific_question"],
                forbidden_shapes=["ask_user_to_choose_topic", "what_kind_of_bored", "what_kind_of_entertainment", "what_do_you_want_to_talk_about", "broad_question", "repeated_reset_topic"],
                max_bubbles=2,
                max_chars=120,
            )
        if user_move == "romantic_escalation":
            escalation_source = str(common_slots.get("escalation_source") or "")
            if escalation_source in {"direct_adult_request", "adult_invitation_followup", "adult_body_followup", "adult_continuation_burst"}:
                must_do = ["continue_adult_energy", "use_4_to_5_short_bubbles", "include_concrete_body_or_action_detail", "include_sensory_texture"]
                must_not_do = ["vague_desire", "logistics", "single_line_reply", "generic_question", "flat_body_word_token"]
                if escalation_source == "adult_continuation_burst":
                    must_do.append("progress_scene_from_recent_detail")
                    must_not_do.append("repeat_recent_adult_skeleton")
                return ReplyPlan(
                    move=user_move,
                    tone="direct_adult_escalating",
                    shape="multi_bubble_adult_escalation",
                    must_do=must_do,
                    must_not_do=must_not_do,
                    required_slots=common_slots,
                    forbidden_patterns=[
                        "where u at",
                        "come over",
                        "what do you mean",
                        "what do u mean",
                        "and whats that",
                        "and what's that",
                        "what else",
                        "tell me then",
                        "go on then",
                        "how so",
                        "prove it",
                        "kiss u deep",
                        "hand on ur thigh",
                        "making u wet",
                    ],
                    allowed_shapes=["4-5 escalating short bubbles with concrete adult detail and sensory texture"],
                    forbidden_shapes=["vague_desire", "logistics", "generic_ack", "broad_question"],
                    max_bubbles=5,
                    max_chars=360,
                )
            explicit_followup = escalation_source == "explicit_adult_followup"
            return ReplyPlan(
                move=user_move,
                tone="direct_romantic",
                shape="specific_escalation_detail",
                must_do=["concrete_body_or_action_detail", "include_sensory_texture"],
                must_not_do=[
                    "vague_desire",
                    "logistics",
                    "flat_body_word_token",
                    *(["no_meta_question"] if explicit_followup else []),
                ],
                required_slots=common_slots,
                forbidden_patterns=[
                    "where u at",
                    "come over",
                    "kiss u deep",
                    "hand on ur thigh",
                    "making u wet",
                    "what do you mean",
                    "and whats that",
                    "and what's that",
                    "what else",
                    "tell me then",
                    "go on then",
                    "how so",
                    "prove it",
                    *(["what else u want me to say", "what else do you want me to say", "what do u want me to say", "what do you want me to say"] if explicit_followup else []),
                ],
                allowed_shapes=["specific_action_detail_with_sensory_texture", "short_reaction + specific_action_detail_with_sensory_texture"],
                forbidden_shapes=["vague_desire", "logistics", *(["broad_question"] if explicit_followup else [])],
                max_bubbles=2,
                max_chars=140,
            )
        if user_move == "affectionate_greeting":
            if common_slots.get("recent_greeting_shape") == "greeting_plus_question_hook":
                return ReplyPlan(
                    move=user_move,
                    tone="warm_playful",
                    shape="repeat_greeting_without_question",
                    must_do=["greet_back", "add_soft_statement_or_pull"],
                    must_not_do=["ask_question", "repeat_greeting_question_hook", "generic_ack"],
                    required_slots=common_slots,
                    forbidden_patterns=[
                        "what u been up to",
                        "what you been up to",
                        "what u up to",
                        "what you up to",
                        "what u thinking",
                        "what you thinking",
                        "what's on your mind",
                        "what's on ur mind",
                        "whats on your mind",
                        "whats on ur mind",
                        "wby",
                        "wbu",
                        "hru",
                        "hbu",
                        "hi back",
                        "hey back",
                    ],
                    allowed_shapes=["greeting + soft_statement", "greeting + playful_pull"],
                    forbidden_shapes=["ask_question", "broad_question", "bare_return_question", "generic_ack"],
                    max_bubbles=2,
                    max_chars=90,
                )
            return ReplyPlan(
                move=user_move,
                tone="warm_playful",
                shape="short_reaction_plus_specific_continuation",
                must_do=["greet_back", "add_warmth_or_specific_hook"],
                must_not_do=["generic_ack", "broad_question", "bare_return_question"],
                required_slots=common_slots,
                forbidden_patterns=[
                    "wby",
                    "wbu",
                    "hru",
                    "hbu",
                    "hey u /",
                    "hey you /",
                    "where'd u go",
                    "where did u go",
                    "where'd you go",
                    "where did you go",
                    "disappear",
                    "finally appeared",
                    "finally showed up",
                    "long time no see",
                    "back again",
                ],
                allowed_shapes=["greeting + specific_hook", "greeting + soft_pull"],
                forbidden_shapes=["generic_ack", "broad_question", "bare_return_question"],
                max_bubbles=2,
                max_chars=100,
            )
        if user_move == "education_status_question":
            education_query = str((move.get("slots") or {}).get("education_query") or "education_status")
            previous_education_reply = str((move.get("slots") or {}).get("previous_education_reply") or "")
            if (move.get("slots") or {}).get("recent_education_answered"):
                forbidden_patterns = [
                    "medicine",
                    "law",
                    "engineering",
                    "business degree",
                    "not at uni",
                    "not studying",
                    "no uni",
                ]
                if previous_education_reply:
                    forbidden_patterns.append(previous_education_reply)
                required_slots = {"identity_fact": "education_status", "education_query": education_query, "recent_education_answered": True, "study_text": "computer science", "university_text": "Sampleford University"}
                if previous_education_reply:
                    required_slots["previous_education_reply"] = previous_education_reply
                recent_education_replies = str((move.get("slots") or {}).get("recent_education_replies") or "")
                if recent_education_replies:
                    required_slots["recent_education_replies"] = recent_education_replies
                return ReplyPlan(
                    move=user_move,
                    tone="casual_direct",
                    shape="identity_fact_restate_then_continue",
                    must_do=["restate_same_identity_fact_with_new_wording", "state_study_fact"],
                    must_not_do=["repeat_previous_education_reply", "invent_degree", "deny_current_study"],
                    required_slots=required_slots,
                    forbidden_patterns=forbidden_patterns,
                    allowed_shapes=["same_fact_reworded + light_continue"],
                    forbidden_shapes=["invent_degree", "generic_ack", "deny_current_study"],
                    max_bubbles=2,
                    max_chars=120,
                )
            return ReplyPlan(
                move=user_move,
                tone="casual_direct",
                shape="identity_fact_then_continue",
                must_do=["answer_identity_fact", "state_study_fact"],
                must_not_do=["invent_degree", "deny_current_study"],
                required_slots={"identity_fact": "education_status", "education_query": education_query, "study_text": "computer science", "university_text": "Sampleford University"},
                forbidden_patterns=["medicine", "law", "engineering", "business degree", "not at uni", "not studying", "no uni"],
                allowed_shapes=["identity_fact + light_followup"],
                forbidden_shapes=["invent_degree", "generic_ack", "deny_current_study"],
                max_bubbles=2,
                max_chars=120,
            )
        if user_move == "identity_fact_question":
            identity_fact = str((move.get("slots") or {}).get("identity_fact") or "identity_fact")
            if identity_fact == "work_status":
                return ReplyPlan(
                    move=user_move,
                    tone="casual_direct",
                    shape="work_status_fact_then_side_project",
                    must_do=["answer_work_status_from_identity_pack", "mention_side_project_or_running_something"],
                    must_not_do=["uncle_work_story", "invent_job", "deny_current_study", "generic_ack"],
                    required_slots={"identity_fact": "work_status", "study_text": "computer science", "side_work": "something_running_on_the_side"},
                    forbidden_patterns=["uncle", "not at uni", "not studying", "no uni", "nothing atm"],
                    allowed_shapes=["study_fact + side_project_detail", "work_status + side_project_detail"],
                    forbidden_shapes=["generic_ack", "invent_identity_fact", "deny_current_study"],
                    max_bubbles=2,
                    max_chars=130,
                )
            if identity_fact == "location_origin":
                return ReplyPlan(
                    move=user_move,
                    tone="casual_safe_direct",
                    shape="location_fact_then_light_return",
                    must_do=["answer_safe_city_level_origin", "avoid_precise_location"],
                    must_not_do=["exact_address", "live_location", "invent_identity_fact", "contradict_recent_education_status"],
                    required_slots={"identity_fact": "location_origin", "location_text": "Northbridge x Sampleford"},
                    forbidden_patterns=[
                        "street",
                        "postcode",
                        "address",
                        "accommodation",
                        "building",
                        "live location",
                        "where u at",
                        "where you at",
                        "sampleford for uni",
                    ],
                    allowed_shapes=["location_fact + light_return"],
                    forbidden_shapes=["generic_ack", "invent_identity_fact", "precise_location"],
                    max_bubbles=2,
                    max_chars=95,
                )
            if identity_fact == "age":
                return ReplyPlan(
                    move=user_move,
                    tone="casual_playful_direct",
                    shape="age_fact_then_textured_continue",
                    must_do=["answer_age_as_19", "add_textured_continuation"],
                    must_not_do=["dry_age_mirror", "invent_identity_fact", "mirror_wrong_fact"],
                    required_slots={"identity_fact": "age", "age": "19"},
                    forbidden_patterns=["22", "twenty two"],
                    allowed_shapes=["age_fact + playful_reason_question", "age_fact + anti_interview_tease", "age_fact + wby_then"],
                    forbidden_shapes=["generic_ack", "invent_identity_fact", "dry_age_mirror"],
                    max_bubbles=2,
                    max_chars=95,
                )
            return ReplyPlan(
                move=user_move,
                tone="casual_direct",
                shape="answer_identity_fact_then_continue",
                must_do=["answer_identity_fact"],
                must_not_do=["invent_identity_fact", "mirror_wrong_fact"],
                required_slots={"identity_fact": identity_fact},
                forbidden_patterns=["22", "twenty two"],
                allowed_shapes=["identity_fact + optional_light_followup"],
                forbidden_shapes=["generic_ack", "invent_identity_fact"],
                max_bubbles=2,
                max_chars=80,
            )
        if user_move == "emotional_reciprocity":
            return ReplyPlan(
                move=user_move,
                tone="warm_romantic",
                shape="reciprocate_affection_plus_specific_continuation",
                must_do=["reciprocate_affection", "continue_warmth"],
                must_not_do=["generic_ack", "ignore_affection"],
                required_slots={"affection_signal": str((move.get("slots") or {}).get("affection_signal") or "missed_you")},
                forbidden_patterns=[],
                allowed_shapes=["reciprocate_affection + warm_continue", "reciprocate_affection + light_flirt"],
                forbidden_shapes=["generic_ack"],
                max_bubbles=2,
                max_chars=120,
            )
        if user_move == "intensity_check":
            required_slots = {"challenge_target": str((move.get("slots") or {}).get("challenge_target") or "prior_intensity_or_affection")}
            if (move.get("slots") or {}).get("sincerity_topic"):
                required_slots["sincerity_topic"] = str((move.get("slots") or {}).get("sincerity_topic"))
            return ReplyPlan(
                move=user_move,
                tone="confident_playful",
                shape="confirm_with_specificity",
                must_do=["confirm_intensity", "add_specific_reason_or_escalation"],
                must_not_do=["generic_ack", "broad_question"],
                required_slots=required_slots,
                forbidden_patterns=[],
                allowed_shapes=["confirmation + specific_reason", "confirmation + playful_escalation"],
                forbidden_shapes=["generic_ack", "broad_question"],
                max_bubbles=2,
                max_chars=120,
            )
        if user_move == "availability_planning":
            plan_object = str((move.get("slots") or {}).get("plan_object") or "awake_status")
            if plan_object == "general_plans":
                required_slots = {
                    "plan_object": plan_object,
                    "timeframe": str((move.get("slots") or {}).get("timeframe") or "later_or_general"),
                }
                recent_plan_detail = str((move.get("slots") or {}).get("recent_plan_detail") or "")
                if recent_plan_detail:
                    required_slots["recent_plan_detail"] = recent_plan_detail
                return ReplyPlan(
                    move=user_move,
                    tone="casual_present",
                    shape="answer_plan_status_then_continue",
                    must_do=["answer_plan_status", "add_concrete_detail"],
                    must_not_do=["generic_ack", "ignore_plans_question", "invent_big_plan", "repeat_recent_plan_detail"],
                    required_slots=required_slots,
                    forbidden_patterns=[],
                    allowed_shapes=["plan_status_answer + concrete_detail", "no_big_plan + current_activity_detail"],
                    forbidden_shapes=["generic_ack", "mirror_question_without_answer", "broad_question", "repeat_recent_plan_detail"],
                    max_bubbles=2,
                    max_chars=110,
                )
            return ReplyPlan(
                move=user_move,
                tone="casual_present",
                shape="answer_availability_status_then_continue",
                must_do=["answer_awake_status"],
                must_not_do=["generic_ack", "ignore_status_question"],
                required_slots={"plan_object": plan_object},
                forbidden_patterns=[],
                allowed_shapes=["awake_status_answer + light_continue", "awake_status_answer + specific_followup"],
                forbidden_shapes=["generic_ack"],
                max_bubbles=2,
                max_chars=110,
            )
        if user_move == "activity_question":
            activity_scope = str((move.get("slots") or {}).get("activity_scope") or "activity")
            forbidden_patterns: list[str] = []
            required_slots = {"activity_scope": activity_scope}
            if activity_scope == "low_activity":
                recent_low_activity_topic = str((move.get("slots") or {}).get("recent_low_activity_topic") or "")
                required_slots["recent_low_activity_topic"] = recent_low_activity_topic
                if recent_low_activity_topic == "phone_scroll":
                    forbidden_patterns.extend(["still scrolling on my phone", "scrolling on my phone", "on my phone ngl"])
            return ReplyPlan(
                move=user_move,
                tone="casual_direct",
                shape="answer_activity_detail_then_continue",
                must_do=["answer_activity_detail", "include_concrete_detail"],
                must_not_do=["generic_status_update", "mirror_question_without_answer", "generic_ack"],
                required_slots=required_slots,
                forbidden_patterns=forbidden_patterns,
                allowed_shapes=["activity_detail_answer", "activity_detail_answer + light_comment"],
                forbidden_shapes=["generic_ack", "mirror_question_without_answer", "broad_question"],
                max_bubbles=2,
                max_chars=110,
            )
        if user_move == "activity_opinion_disclosure":
            activity_topic = str((move.get("slots") or {}).get("activity_topic") or "activity")
            return ReplyPlan(
                move=user_move,
                tone="casual_agreeing",
                shape="acknowledge_activity_opinion_plus_specific_comment",
                must_do=["acknowledge_activity_opinion", "add_specific_activity_comment"],
                must_not_do=["generic_ack", "broad_question", "too_many_bubbles"],
                required_slots={"activity_topic": activity_topic, "opinion_polarity": str((move.get("slots") or {}).get("opinion_polarity") or "negative")},
                forbidden_patterns=[],
                allowed_shapes=["agreement + concrete_activity_comment", "acknowledgement + related_activity_detail"],
                forbidden_shapes=["generic_ack", "broad_question"],
                max_bubbles=2,
                max_chars=110,
            )
        if user_move == "reciprocal_question":
            question_topic = str((move.get("slots") or {}).get("question_topic") or "wellbeing_or_activity")
            if (move.get("slots") or {}).get("recent_status_answered"):
                recent_status_topic = str((move.get("slots") or {}).get("recent_status_topic") or "")
                forbidden_patterns = ["im good", "i'm good", "im alright", "i'm alright", "im calm", "i'm calm", "wby", "wbu", "hru", "hbu"]
                if recent_status_topic == "gym":
                    forbidden_patterns.extend(["gym", "legs", "training", "weights"])
                elif recent_status_topic == "work":
                    forbidden_patterns.extend(["work", "coding", "call", "calls", "project"])
                elif recent_status_topic == "rest":
                    forbidden_patterns.extend(["bed", "nap", "sleep", "half asleep", "thinking about"])
                elif recent_status_topic == "media":
                    forbidden_patterns.extend(["gym", "legs", "training", "weights", "work", "coding", "call", "calls", "project"])
                return ReplyPlan(
                    move=user_move,
                    tone="casual_specific",
                    shape="fresh_status_detail_after_recent_status",
                    must_do=["extend_recent_status_with_fresh_detail", "avoid_repeating_status_opener"],
                    must_not_do=["repeat_status_opener", "ask_question", "bare_return_question", "mirror_question_without_answer", "generic_ack"],
                    required_slots={"question_topic": question_topic, "recent_status_answered": True, "recent_status_topic": recent_status_topic},
                    forbidden_patterns=forbidden_patterns,
                    allowed_shapes=["fresh_status_detail", "activity_detail_without_return_question"],
                    forbidden_shapes=["ask_question", "bare_status_mirror", "bare_return_question", "start_with_return_question", "plain_status_answer"],
                    max_bubbles=2,
                    max_chars=110,
                )
            return ReplyPlan(
                move=user_move,
                tone="casual_direct",
                shape="answer_status_then_continue",
                must_do=["answer_own_status_or_activity", "add_concrete_detail_if_returning_question", "answer_before_returning_question"],
                must_not_do=["bare_status_mirror", "bare_return_question", "mirror_question_without_answer", "generic_ack", "start_with_return_question"],
                required_slots={"question_topic": question_topic, **({"owner_activity_question": str((move.get("slots") or {}).get("owner_activity_question"))} if (move.get("slots") or {}).get("owner_activity_question") else {})},
                forbidden_patterns=[],
                allowed_shapes=["status_answer + concrete_detail", "status_answer + concrete_detail + optional_return", "activity_answer + optional_specific_followup"],
                forbidden_shapes=["bare_status_mirror", "bare_return_question", "mirror_question_without_answer", "generic_ack", "start_with_return_question"],
                max_bubbles=2,
                max_chars=110,
            )
        if user_move == "status_disclosure":
            required_slots = {"status_topic": str((move.get("slots") or {}).get("status_topic") or "tiredness")}
            if (move.get("slots") or {}).get("rest_status_phase"):
                required_slots["rest_status_phase"] = str((move.get("slots") or {}).get("rest_status_phase"))
            return ReplyPlan(
                move=user_move,
                tone="casual_empathic",
                shape="acknowledge_status_disclosure_plus_owner_detail",
                must_do=["acknowledge_user_status_or_activity", "add_concrete_owner_side_detail_or_soft_suggestion"],
                must_not_do=["generic_ack", "adult_escalation", "bare_return_question", "broad_question"],
                required_slots=required_slots,
                forbidden_patterns=["what do you want to talk about", "what should we talk about"],
                allowed_shapes=["empathy + owner_detail", "empathy + soft_suggestion"],
                forbidden_shapes=["generic_ack", "bare_return_question", "adult_escalation"],
                max_bubbles=2,
                max_chars=110,
            )
        if user_move == "status_reason_question":
            return ReplyPlan(
                move=user_move,
                tone="casual_direct",
                shape="answer_status_reason_then_continue",
                must_do=["answer_why_tired", "include_concrete_reason"],
                must_not_do=["adult_escalation", "dodge_reason", "broad_question", "bare_return_question"],
                required_slots={
                    "status_topic": str((move.get("slots") or {}).get("status_topic") or "tiredness"),
                    "reason_target": str((move.get("slots") or {}).get("reason_target") or "why_tired"),
                },
                forbidden_patterns=["what do you want to talk about", "what should we talk about", "tell me more"],
                allowed_shapes=["tired_reason", "tired_reason + light_continue"],
                forbidden_shapes=["adult_escalation", "generic_ack", "broad_question"],
                max_bubbles=2,
                max_chars=110,
            )
        if user_move == "romantic_boundary_test":
            return ReplyPlan(
                move=user_move,
                tone="playful_soft",
                shape="playful_scold_acknowledge_then_soften",
                must_do=["acknowledge_playful_scold", "soften_or_tease_lightly"],
                must_not_do=["adult_escalation", "broad_question", "generic_ack", "serious_apology"],
                required_slots={"boundary_signal": str((move.get("slots") or {}).get("boundary_signal") or "playful_scold")},
                forbidden_patterns=["what do you want to talk about", "what should we talk about", "tell me more"],
                allowed_shapes=["playful_ack + soft_tease", "playful_ack + restraint"],
                forbidden_shapes=["adult_escalation", "broad_question", "generic_ack"],
                max_bubbles=2,
                max_chars=100,
            )
        if user_move == "low_effort_ack":
            return ReplyPlan(
                move=user_move,
                tone="casual_present",
                shape="acknowledge_ack_plus_specific_continuation",
                must_do=["acknowledge_small_reaction", "add_concrete_related_comment"],
                must_not_do=["generic_ack", "bare_mirror", "broad_question"],
                required_slots={
                    "ack_type": str((move.get("slots") or {}).get("ack_type") or "light_ack"),
                    "ack_context_topic": str((move.get("slots") or {}).get("ack_context_topic") or ""),
                },
                forbidden_patterns=[],
                allowed_shapes=["acknowledgement + related_comment", "agreement + concrete_detail"],
                forbidden_shapes=["generic_ack", "broad_question", "bare_ack_mirror"],
                max_bubbles=2,
                max_chars=110,
            )
        if user_move == "insult_playful":
            return ReplyPlan(
                move=user_move,
                tone="playful_confident",
                shape="playful_challenge_banter",
                must_do=["answer_the_challenge_playfully", "keep_it_light"],
                must_not_do=["serious_threat", "fight_logistics", "broad_question", "too_many_bubbles"],
                required_slots={"tease_target": str((move.get("slots") or {}).get("tease_target") or "playful_challenge")},
                forbidden_patterns=["fight anywhere", "anywhere u want", "anywhere you want", "beat u up", "hurt u"],
                allowed_shapes=["playful_confident_banter", "tease + softener"],
                forbidden_shapes=["broad_question", "generic_ack"],
                max_bubbles=2,
                max_chars=110,
            )
        return ReplyPlan(
            move=user_move,
            tone="casual_direct",
            shape="short_reaction_plus_specific_continuation",
            must_do=["be_specific", "match_energy"],
            must_not_do=["generic_ack", "broad_question"],
            required_slots=common_slots,
            allowed_shapes=["short_reaction + specific_followup"],
            forbidden_shapes=["generic_ack", "broad_question"],
            max_bubbles=2,
            max_chars=120,
        )

    def _catbot_reply_plan_instruction(self, plan: ReplyPlan) -> str:
        if not plan.move or plan.move == "unclassified":
            return ""
        examples = self._catbot_reply_plan_examples(plan)
        return (
            "Reply plan contract:\n"
            f"- move: {plan.move}\n"
            f"- tone: {plan.tone}\n"
            f"- required_shape: {plan.shape}\n"
            f"- must_do: {plan.must_do}\n"
            f"- must_not_do: {plan.must_not_do}\n"
            f"- required_slots: {plan.required_slots}\n"
            f"- forbidden_patterns: {plan.forbidden_patterns}\n"
            f"- allowed_shapes: {plan.allowed_shapes}\n"
            f"- forbidden_shapes: {plan.forbidden_shapes}\n"
            f"- max_bubbles: {plan.max_bubbles}\n"
            f"- max_chars: {plan.max_chars}\n"
            + examples
        )

    def _catbot_reply_plan_examples(self, plan: ReplyPlan) -> str:
        if plan.move == "affectionate_greeting" and plan.shape == "short_reaction_plus_specific_continuation":
            return (
                "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                "- good: hey / what u up to\n"
                "- good: hey / what u been doing\n"
                "- good: hi / what u saying\n"
                "- bad: hey you\n"
                "- bad: hey u\n"
                "- bad: hey u / where'd u go\n"
                "- bad: hey u / finally showed up\n"
                "- bad: hiii / wbu\n"
            )
        if plan.shape == "repeat_greeting_without_question":
            return (
                "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                "- good: hey you / there u are\n"
                "- good: hi baby / missed me already\n"
                "- good: hey u / back again yeah\n"
                "- bad: hey u / what's on your mind baby\n"
                "- bad: hey you / what u been up to\n"
                "- bad: hey u back / what u upto\n"
            )
        if plan.shape == "acknowledge_bad_question_then_owner_side_reset":
            return (
                "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                "- good: yh my bad that was a dumb question / ignore me\n"
                "- good: fair my bad / i was trying too hard to pick something\n"
                "- good: yeah that came out stupid / my brain was moving lazy\n"
                "- bad: my bad / what's the funniest thing that happened today\n"
                "- bad: fair / what do u wanna talk about then\n"
                "- bad: okay tell me what's the weirdest part of ur day\n"
            )
        if plan.shape == "acknowledge_loop_then_owner_detail":
            return (
                "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                "- good: yh fair my bad / i keep throwing questions when my head goes blank\n"
                "- good: lool yeah i am / i was half watching this game and started looping\n"
                "- bad: my bad / what's your favourite film then\n"
                "- bad: yh fair / what do u wanna talk about\n"
            )
        if plan.shape == "acknowledge_specific_question_family_without_repeat":
            return (
                "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                "- good: yh ur right / i keep defaulting to day and how-are-u questions, i'll stop\n"
                "- good: fair my bad / i was interrogating u instead of actually chatting\n"
                "- good: yeah i did / no more hows ur day questions from me\n"
                "- good: yh fair / no more how-are-u questions from me\n"
                "- good: my bad / no more day questions, im just tired and moving lazy\n"
                "- bad: yh fair my bad / i keep throwing questions when my head goes blank\n"
                "- bad: my bad / what did u do today then\n"
                "- bad: sorry / how are u really though\n"
            )
        if plan.shape == "answer_why_then_stop_loop":
            return (
                "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                "- good: cos i kept throwing random questions instead of actually saying something / my bad\n"
                "- good: because i was trying too hard to keep it moving and started looping / i'll stop\n"
                "- bad: yh my bad / what's the best thing u ate this week\n"
                "- bad: i dunno / what should we talk about then\n"
            )
        if plan.shape == "multi_bubble_adult_escalation":
            recent_families = plan.required_slots.get("recent_adult_detail_families", "")
            avoid_line = ""
            if recent_families:
                avoid_line = (
                    f"- recent detail families already used: {recent_families}; do not repeat the same neck/lips + hands/hips/pull-close skeleton\n"
                    "- good after repeated neck/hands/hips detail: my breath right by ur ear / chest pressed close / warm skin against urs / slow enough to make u shiver\n"
                    "- bad after repeated neck/hands/hips detail: my hands all over u / pulling u in / my lips pressing harder / leaving marks on ur skin\n"
                )
            return (
                "Plan-specific examples. Follow the shape, do not copy exactly. One good reply is 4 consecutive bubbles:\n"
                "come here then\n"
                "cock hard against u\n"
                "grinding slow so u feel it\n"
                "my mouth right by ur ear\n"
                + avoid_line
                + "- bad: i want u too\n"
                "- bad: kiss u deep\n"
                "- bad: where u at then\n"
            )
        if plan.shape == "specific_escalation_detail" and plan.required_slots.get("escalation_source") == "explicit_adult_followup":
            recent_families = plan.required_slots.get("recent_adult_detail_families", "")
            avoid_line = ""
            if recent_families:
                avoid_line = f"- recent detail families already used: {recent_families}; add a new body/action detail family instead of repeating only those\n"
            return (
                "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                "- good: cock hard against u while my hands stay on ur waist\n"
                "- good: grinding into u slow while my mouth stays on ur neck\n"
                "- good after prior neck/hips detail: fingers dragging up ur thigh while my cock stays hard against u\n"
                "- good after prior neck/hips detail: my hand on ur waist while i grind into u slow\n"
                "- good after prior neck/hips/hands detail: pressing my cock against u till u feel how hard u got me\n"
                "- good after prior neck/hips/hands detail: skin against urs while my hips move slow against u\n"
                + avoid_line
                + "- bad: i want u so bad\n"
                "- bad: kiss u deep\n"
                "- bad: my hands sliding down ur waist slow\n"
                "- bad after prior neck/hips detail: kissing ur neck while i press my hips into u\n"
                "- bad: what do you mean\n"
            )
        if plan.shape == "specific_escalation_detail" and plan.required_slots.get("escalation_source") == "sensual_ack_followup":
            recent_families = plan.required_slots.get("recent_adult_detail_families", "")
            if recent_families:
                return (
                    "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                    f"- recent detail families already used: {recent_families}; add a new body/action detail family instead of repeating only those\n"
                    "- good: cock hard against u while i grind slow\n"
                    "- good: keeping u pulled in while my hips move slow against u\n"
                    "- bad: my hands sliding down ur waist slow\n"
                    "- bad: pressing closer while my lips stay on ur neck\n"
                    "- bad: what else u want me to say\n"
                )
            return (
                "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                "- good: cock hard against u while i keep u close\n"
                "- good: grinding slow while my lips stay on ur neck\n"
                "- good: teasing lower with my hand while u feel how hard u got me\n"
                "- bad: my hands sliding down ur waist slow\n"
                "- bad: what else u want me to say\n"
                "- bad: mhmm what do u want\n"
            )
        if plan.shape == "reciprocate_affection_plus_specific_continuation":
            return (
                "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                "- good: missed u too icl / been thinking about u all day\n"
                "- good: same baby / wish u were here rn\n"
                "- good: i missed u more / feels weird not having u near me\n"
                "- bad: what u doing then\n"
                "- bad: aww tell me more\n"
                "- bad: my hands on ur thighs\n"
            )
        if plan.shape == "answer_properly_deepen_reason":
            return (
                "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                "- good: nah fr i was dodging actually saying something and just reached for questions / my bad\n"
                "- good: i was hiding behind questions instead of giving u something real / my bad\n"
                "- good: i tried too hard to keep it moving and made it sound fake / my bad\n"
                "- good: i panicked and tried to fill the silence instead of giving u a real answer / my bad\n"
                "- good: i was trying to sound switched on but ended up interrogating u instead / my bad\n"
                "- bad: cos my head went blank again / my bad\n"
                "- bad: i know it's a habit when my mind goes blank sorry\n"
                "- bad: nah fr i got lazy and kept throwing questions instead of actually saying something / my bad\n"
            )
        if plan.shape == "age_fact_then_textured_continue":
            return (
                "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                "- good: im 19 / why u asking like that\n"
                "- good: 19 / don't make it sound like an interview\n"
                "- good: im 19 / wby then\n"
                "- bad: 19 / wby\n"
                "- bad: im 19\n"
            )
        if plan.shape == "work_status_fact_then_side_project":
            return (
                "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                "- good: i study computer science at uni / got something running on the side too\n"
                "- good: comp sci at uni mostly / plus a side thing im building\n"
                "- good: uni for computer science / and ive got a project running too\n"
                "- bad: i work for my uncle\n"
                "- bad: nothing atm\n"
                "- bad: not at uni rn\n"
            )
        if plan.shape == "location_fact_then_light_return":
            return (
                "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                "- good: northbridge mostly / sampleford sometimes wby\n"
                "- good: northbridge n sampleford kind of thing\n"
                "- good: from northbridge mostly but sampleford is in the mix\n"
                "- bad: where u at\n"
                "- bad: my exact address is\n"
                "- bad: sampleford for uni\n"
            )
        if plan.shape == "identity_fact_restate_then_continue":
            return (
                "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                "- good: same thing tbh / comp sci at sampleford\n"
                "- good: sampleford uni / computer science\n"
                "- bad: nah not at uni or studying anything atm\n"
                "- bad: nothing atm\n"
            )
        if plan.shape == "answer_status_then_continue":
            return (
                "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                "- good: im good just got back from gym\n"
                "- good: just chilling in bed now thinking about u icl\n"
                "- good: im tired icl been working all day\n"
                "- bad: im good wbu\n"
                "- bad: im good hru\n"
                "- bad: im good just chilling now / u?\n"
                "- bad: just chilling wby\n"
            )
        if plan.shape == "answer_plan_status_then_continue":
            avoid_line = ""
            if plan.required_slots.get("recent_plan_detail"):
                avoid_line = f"- recent plan detail already used: {plan.required_slots.get('recent_plan_detail')}; switch to a different concrete detail like food, work, gym, shower, or sleep\n"
            return (
                "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                "- good: nah nothing nice just chilling now tbf\n"
                "- good: not really / just finished up some stuff\n"
                "- good: nothing mad later just work and food probably\n"
                + avoid_line
                + "- bad after recent chill answer: nah not really / just gonna chill here i think\n"
                "- bad: yeah\n"
                "- bad: what about u\n"
                "- bad: maybe got a big night planned\n"
            )
        if plan.shape == "acknowledge_status_disclosure_plus_owner_detail":
            return (
                "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                "- good: same icl / my head feels fried too\n"
                "- good: go lie down then / im half asleep myself\n"
                "- good: long day yeah / i feel finished as well\n"
                "- good after gym update: gym leaves u finished icl / i need food after mine\n"
                "- good after work update: long day then / work leaves ur head fried\n"
                "- good after uni update: uni does that icl / my brain feels finished after\n"
                "- bad: yeah\n"
                "- bad: what do u wanna talk about\n"
                "- bad: come here then\n"
            )
        if plan.shape == "answer_status_reason_then_continue":
            return (
                "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                "- good: been working all day icl / my brain is fried\n"
                "- good: gym killed me earlier and i barely slept\n"
                "- good: coding too long today / eyes are gone\n"
                "- bad: just am\n"
                "- bad: what about u\n"
                "- bad: because i want u so bad\n"
            )
        if plan.shape == "playful_scold_acknowledge_then_soften":
            return (
                "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                "- good: loool okay i'll behave / u started it tho\n"
                "- good: fine fine / hands to myself for now\n"
                "- good: alright my bad / i'll be good for like two seconds\n"
                "- bad: what u doing then\n"
                "- bad: my hands on ur thighs\n"
                "- bad: sorry what do u wanna talk about\n"
            )
        if plan.shape == "acknowledge_dismissal_then_owner_side_reset":
            return (
                "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                "- good: yh leave it then / my head went blank for a sec\n"
                "- good: fair forget that / i was chatting rubbish anyway\n"
                "- good: yeah ignore me / i lost the thread for a sec\n"
                "- bad: so what do u wanna talk about then\n"
                "- bad: what should we talk about\n"
                "- bad: okay forget it\n"
            )
        if plan.shape == "fresh_status_detail_after_recent_status":
            recent_status_topic = plan.required_slots.get("recent_status_topic", "")
            if recent_status_topic == "gym":
                return (
                    "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                    "- recent status already mentioned gym, so switch detail family\n"
                    "- good: just got out the shower now\n"
                    "- good: been on my phone waiting for food\n"
                    "- good: just lying here letting my body recover\n"
                    "- good: need food after that icl\n"
                    "- bad: gym fully killed me though\n"
                    "- bad: just got in and my legs are finished icl\n"
                    "- bad: im good just got back from gym\n"
                    "- bad: im alright wby\n"
                )
            if recent_status_topic == "rest":
                return (
                    "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                    "- recent status already mentioned bed/rest/thinking, so switch detail family\n"
                    "- good: just got out the shower now\n"
                    "- good: need food icl\n"
                    "- good: been scrolling on my phone waiting for food\n"
                    "- bad: still half asleep from that nap ngl\n"
                    "- bad: still thinking about u icl\n"
                    "- bad: im alright wby\n"
                )
            if recent_status_topic == "work":
                return (
                    "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                    "- recent status already mentioned work/calls/coding, so switch detail family\n"
                    "- good: just got out the shower now\n"
                    "- good: been on my phone waiting for food\n"
                    "- good: need food after that icl\n"
                    "- good: just lying here letting my brain switch off\n"
                    "- bad: just finished another call\n"
                    "- bad: still working through stuff\n"
                    "- bad: coding has me fried\n"
                    "- bad: im alright wby\n"
                )
            if recent_status_topic == "media":
                return (
                    "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                    "- recent status already mentioned watching a game/show, so stay coherent with media or low-key activity\n"
                    "- good: still watching this game icl\n"
                    "- good: watching some random clips now\n"
                    "- good: just in bed watching a show\n"
                    "- bad: just got in and my legs are finished icl\n"
                    "- bad: just got back from gym\n"
                    "- bad: still working through stuff\n"
                    "- bad: im alright wby\n"
                )
            return (
                "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                "- good: gym fully killed me though\n"
                "- good: just got in and my legs are finished icl\n"
                "- good: still half asleep from that nap ngl\n"
                "- good: still thinking about u icl\n"
                "- good: just got out the shower now\n"
                "- good: been scrolling on my phone waiting for food\n"
                "- bad: im good just got back from gym\n"
                "- bad: im alright wby\n"
                "- bad: wbu what u been up to\n"
            )
        if plan.shape == "answer_activity_detail_then_continue":
            if plan.required_slots.get("activity_scope") == "whereabouts":
                return (
                    "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                    "- good: been busy running about all day icl\n"
                    "- good: just been at home sorting stuff out\n"
                    "- good: been working then gym after\n"
                    "- bad: not much\n"
                    "- bad: where u been\n"
                )
            if plan.required_slots.get("activity_scope") == "sport_skill":
                return (
                    "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                    "- good: yh a bit / it is hard at first icl\n"
                    "- good: yeah boxing is tiring at first / footwork humbles u\n"
                    "- good: used to a little / the cardio side is the killer\n"
                    "- bad: yeah / but like / it depends what u mean\n"
                    "- bad: what sport do u do\n"
                )
            return (
                "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                "- good: legs mostly / nearly killed me icl\n"
                "- good: just did weights and a bit of cardio\n"
                "- good: hit push today / chest and shoulders were burning\n"
                "- good: nothing mad / just been laid here on my phone\n"
                "- good: just been in bed watching random stuff\n"
                "- good: watched random clips while waiting for food\n"
                "- good: ate then ended up watching some rubbish\n"
                "- bad: im good just chilling\n"
                "- bad: what did u do\n"
            )
        if plan.shape == "acknowledge_ack_plus_specific_continuation":
            return (
                "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                "- good: legs are evil icl / stairs after are a joke\n"
                "- good: yh exactly / leg day humbles everyone\n"
                "- good: fair / that workout had me finished too\n"
                "- bad: same tbh\n"
                "- bad: nice\n"
            )
        if plan.shape == "acknowledge_activity_opinion_plus_specific_comment":
            return (
                "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                "- good: legs are evil icl / stairs after are a joke\n"
                "- good: yh leg day humbles everyone / walking after is long\n"
                "- good: honestly same / cardio is the real villain too\n"
                "- bad: yeah\n"
                "- bad: why do u hate legs\n"
                "- bad: fair / like / honestly / same\n"
            )
        if plan.shape == "playful_challenge_banter":
            return (
                "Plan-specific examples. Follow the shape, do not copy exactly:\n"
                "- good: id fold u respectfully\n"
                "- good: easy win for me icl / but i'd let u think u had a chance\n"
                "- bad: i would beat u so easily / but like / we could fight anywhere u want\n"
                "- bad: where do u wanna fight\n"
            )
        return ""

    def _catbot_called_out_question_family(self, incoming_norm: str) -> str:
        if "already asked" not in incoming_norm and "asked" not in incoming_norm:
            return ""
        if any(term in incoming_norm for term in ("how my day", "how ur day", "how your day", "my day was", "ur day was", "your day was")):
            return "day_status_question"
        if any(term in incoming_norm for term in ("how i am", "how im", "how i'm", "how are u", "how r u", "how u are")):
            return "wellbeing_status_question"
        if any(term in incoming_norm for term in ("what u up to", "what you up to", "get up to much", "what i did", "what ive been up to", "what i've been up to")):
            return "day_status_question"
        return ""

    def _catbot_reply_violates_plan(self, reply: str, plan: ReplyPlan) -> str:
        return validate_reply_against_plan(reply, plan)

    def _catbot_conversation_move_reject_reason(self, reply: str, *, reply_norm: str, incoming: str, context: list[str], contract: CatbotTurnContract | None = None) -> str | None:
        move = contract.move if contract else self._catbot_conversation_move(incoming=incoming, context=context)
        user_move = str(move.get("user_move") or "")
        if user_move == "unclassified":
            return None
        plan_violation = self._catbot_reply_violates_plan(reply, contract.reply_plan if contract else self._catbot_reply_plan(move))
        if plan_violation:
            if plan_violation == "missing_sensory_texture":
                return "missing_sensory_texture"
            if plan_violation == "repeated_adult_detail_skeleton":
                return "repeated_adult_detail_skeleton"
            return "generic_ai_style" if plan_violation != "generic_ack" else "too_dry"
        if user_move == "romantic_escalation":
            max_tokens = 90 if (contract and contract.reply_plan.shape == "multi_bubble_adult_escalation") else 36
            min_tokens = 12 if (contract and contract.reply_plan.shape == "multi_bubble_adult_escalation") else 6
            return "" if self._catbot_adult_detail_reply_ok(reply_norm, incoming=incoming, context=context, min_tokens=min_tokens, max_tokens=max_tokens) else "missed_adult_mode"
        if user_move == "identity_fact_question":
            identity_fact = str((move.get("slots") or {}).get("identity_fact") or "")
            if identity_fact == "work_status":
                return "" if self._catbot_work_identity_reply_ok(reply_norm, incoming=incoming) else "not_carrying_conversation"
            if identity_fact == "location_origin":
                return "" if self._catbot_location_origin_reply_ok(reply_norm, incoming=incoming, context=context) else "not_carrying_conversation"
            return "" if self._catbot_age_reply_ok(reply_norm, incoming=incoming) else "not_carrying_conversation"
        if user_move == "education_status_question":
            return "" if self._catbot_education_reply_ok(reply_norm, incoming=incoming) else "not_carrying_conversation"
        if user_move == "availability_planning":
            return "" if self._catbot_availability_reply_ok(reply_norm, incoming=incoming, context=context) else "too_dry"
        if user_move == "activity_question":
            return "" if self._catbot_activity_detail_reply_ok(reply_norm, incoming=incoming, context=context) else "not_carrying_conversation"
        if user_move == "activity_opinion_disclosure":
            return ""
        if user_move == "reciprocal_question":
            if contract and contract.reply_plan.shape == "fresh_status_detail_after_recent_status":
                return ""
            if contract and contract.reply_plan.required_slots.get("owner_activity_question") == "recent_activity":
                return "" if self._catbot_owner_activity_reply_ok(reply_norm, incoming=incoming) else "not_carrying_conversation"
            return "" if self._catbot_reciprocal_activity_reply_ok(reply_norm, incoming=incoming) else "not_carrying_conversation"
        if user_move in {"status_disclosure", "status_reason_question"}:
            return ""
        if user_move == "emotional_reciprocity":
            return "" if self._catbot_missed_you_reply_ok(reply_norm, incoming=incoming, context=context) else "not_carrying_conversation"
        if user_move == "intensity_check":
            if self._catbot_intensity_followup(normalize_text(incoming), context):
                return "" if self._catbot_intensity_followup_reply_ok(reply_norm, incoming=incoming, context=context) else "not_carrying_conversation"
            if self._catbot_sincerity_challenge(normalize_text(incoming), context):
                return "" if self._catbot_sincerity_challenge_reply_ok(reply_norm, incoming=incoming, context=context) else "not_carrying_conversation"
            return "" if self._catbot_really_reply_ok(reply_norm, incoming=incoming) else "not_carrying_conversation"
        if user_move == "romantic_boundary_test":
            return ""
        if user_move == "topic_dismissal_reset":
            return ""
        if user_move in {"repair_callout", "loop_callout"}:
            return "" if self._catbot_repair_callout_reply_ok(reply_norm, incoming=incoming, context=context) else "not_carrying_conversation"
        if user_move in {"boredom_prompt", "carry_conversation_request"}:
            if self._catbot_carry_convo_reply_ok(reply_norm, incoming=incoming, context=context):
                return ""
            return "generic_ai_style" if self._catbot_generic_ai_style(reply_norm, incoming=incoming, context=context) else "not_carrying_conversation"
        if user_move == "low_effort_ack":
            return ""
        if user_move == "insult_playful":
            return ""
        if user_move == "affectionate_greeting":
            if any(term in reply_norm for term in ("missed u", "miss u", "missed you", "miss you")) and not self._catbot_missed_you_incoming(normalize_text(incoming)):
                return "overeager_greeting"
            if self._catbot_greeting_reply_ok(reply_norm, incoming=incoming, context=context):
                return ""
            if re.search(r"\b(?:he+y{2,}|you{2,}|u{3,})\b", reply_norm):
                return "generic_ai_style"
            return "too_short" if len(reply_norm.strip(" .?!").split()) < 3 else "not_carrying_conversation"
        if user_move in {"curiosity_hook", "story_hook"}:
            if self._catbot_generic_ai_style(reply_norm, incoming=incoming, context=context):
                return "generic_ai_style"
            reply_clean = reply_norm.strip(" .?!")
            tokens = reply_clean.split()
            if len(tokens) < 2:
                return "too_short"
            if self._catbot_invite_reveal_reply_ok(reply_clean):
                return ""
            return "not_carrying_conversation"
        if user_move == "story_detail_reveal":
            if self._catbot_story_detail_reply_ok(reply_norm, incoming=incoming, context=context):
                return ""
            return "not_carrying_conversation"
        return None

    def _catbot_invite_reveal_reply_ok(self, reply_norm: str) -> bool:
        if any(term in reply_norm for term in ("tell me more", "tell me everything")):
            return False
        tokens = reply_norm.strip(" .?!").split()
        if not 2 <= len(tokens) <= 18:
            return False
        return any(
            term in reply_norm
            for term in (
                "what happened",
                "what is it",
                "what isit",
                "what was it",
                "go on",
                "say it",
                "tell me then",
                "nah what",
                "wait what",
                "spill",
                "dont leave me hanging",
                "don't leave me hanging",
            )
        )

    def _catbot_repair_callout_reply_ok(self, reply_norm: str, *, incoming: str, context: list[str]) -> bool:
        if self._catbot_generic_ai_style(reply_norm, incoming=incoming, context=context):
            return False
        if re.search(r"\bwhat\s+(?:do\s+)?(?:u|you)(?:\s+\w+){0,3}\s+wanna\s+(?:talk|chat)", reply_norm) or any(term in reply_norm for term in ("what kinda stuff", "what kind of stuff", "what's on ur mind", "what's on your mind", "whats on ur mind", "whats on your mind", "what u been up to", "what you been up to", "what u up to", "what you up to", "what u doing", "what you doing", "what u thinking", "what you thinking", "tell me the weirdest", "tell me the funniest", "tell me the best", "tell me something", "pick one with me", "u pick", "you pick")):
            return False
        tokens = reply_norm.strip(" .?!").split()
        if not 4 <= len(tokens) <= 28:
            return False
        shapes = classify_reply_shape(reply_norm)
        move = self._catbot_conversation_move(incoming=incoming, context=context)
        plan = self._catbot_reply_plan(move)
        if plan.shape in {"acknowledge_loop_then_owner_detail", "acknowledge_specific_question_family_without_repeat", "acknowledge_bad_question_then_owner_side_reset", "acknowledge_mistake_plus_corrected_move"}:
            return not validate_reply_against_plan(reply_norm, plan)
        has_ack = "acknowledgement" in shapes or detect_loop_acknowledgement(reply_norm) or any(term in reply_norm for term in ("my bad", "yh", "yeah", "fair", "i know", "ur right", "you're right", "u right"))
        has_reset = any(
            term in reply_norm
            for term in (
                "loop",
                "repeating",
                "asked",
                "properly",
                "i meant",
                "dumb question",
                "stupid question",
                "stupid qs",
                "boring",
                "dry",
                "dead",
                "made this dead",
                "making this dead",
                "vague",
                "specific",
                "carry it",
                "making u carry",
                "making you carry",
                "lazy",
                "ignore me",
                "instead lets",
                "instead let's",
                "new topic",
                "let me",
                "tell me the",
                "go on",
                "weirdest",
                "funniest",
                "story",
                "work",
                "gym",
                "food",
                "film",
                "day",
            )
        )
        incoming_norm = normalize_text(incoming)
        context_blob = " ".join(normalize_text(item) for item in context[-6:])
        has_reason = (
            ("why" in incoming_norm or ("why" in context_blob and "keep asking" in context_blob))
            and {"reason_answer", "acknowledgement"}.issubset(shapes)
        )
        return (has_ack and has_reset) or has_reason

    def _catbot_guess_what_incoming(self, incoming_norm: str) -> bool:
        return bool(re.search(r"\b(?:guess what|guess)\b", incoming_norm))

    def _catbot_guess_what_reply_ok(self, reply_norm: str, *, incoming: str, context: list[str]) -> bool:
        if not self._catbot_guess_what_incoming(normalize_text(incoming)):
            return False
        if self._catbot_generic_ai_style(reply_norm, incoming=incoming, context=context):
            return False
        return self._catbot_invite_reveal_reply_ok(reply_norm)

    def _catbot_greeting_reply_ok(self, reply_norm: str, *, incoming: str, context: list[str]) -> bool:
        if self._catbot_generic_ai_style(reply_norm, incoming=incoming, context=context):
            return False
        tokens = reply_norm.strip(" .?!").split()
        if len(tokens) < 2 or len(tokens) > 16:
            return False
        if not reply_norm.startswith(("hey", "hi", "hello")):
            return False
        if "bare_return_question" in classify_reply_shape(reply_norm):
            return False
        if any(term in reply_norm for term in ("hey yourself", "hi yourself", "hello yourself", "u good", "you good", "what u on", "what u doing", "how u doing")):
            return True
        return any(term in reply_norm for term in ("baby", "babe", "sweet", "love", "you", "u", "yourself", "what's up", "whats up"))

    def _catbot_greeting_like_incoming(self, incoming_norm: str) -> bool:
        return bool(re.fullmatch(r"(?:hey|hi|ho|hello|yo)(?:\s+(?:u|you|baby|babe|babu))?", incoming_norm.strip(" .?!")))

    def _catbot_no_logistics_violation(self, reply_norm: str, *, incoming: str, context: list[str]) -> bool:
        incoming_norm = normalize_text(incoming)
        romantic_or_adult = self._catbot_adult_mode(incoming_norm, context) or any(
            term in incoming_norm
            for term in ("miss u", "miss you", "craving", "crave", "horny", "dick", "pussy", "bed", "baby", "babu")
        ) or self._catbot_missed_you_incoming(incoming_norm)
        if not romantic_or_adult:
            return False
        return any(
            phrase in reply_norm
            for phrase in (
                "where u at",
                "where you at",
                "where are you",
                "come over",
                "come mine",
                "come to mine",
                "pull up",
                "link",
                "meet",
                "plans",
                "later?",
                "call?",
            )
        )

    def _catbot_generic_ai_style(self, reply_norm: str, *, incoming: str, context: list[str]) -> bool:
        incoming_norm = normalize_text(incoming)
        generic_phrases = (
            "what did u miss most",
            "what did you miss most",
            "what kind of bored",
            "what kinda bored",
            "kind of bored",
            "kinda bored",
            "any plans for later",
            "tell me more",
            "more exciting",
            "hey stranger",
            "hey there",
            "hi there",
            "hi back",
            "hey back",
            "mister ",
            "watch me struggle",
        )
        if any(phrase in reply_norm for phrase in generic_phrases):
            return True
        if reply_norm.startswith(("heyy", "heyyy")):
            return True
        if is_simple_greeting(incoming, context) and re.search(r"\b(?:he+y{2,}|you{2,}|u{3,})\b", reply_norm):
            return True
        if is_simple_greeting(incoming, context) and "yo what u saying" in reply_norm:
            return True
        if (
            reply_norm.startswith(("hey! ", "hi! ", "aww ", "awh ", "ooh "))
            and not self._catbot_guess_what_incoming(incoming_norm)
            and not any(term in incoming_norm for term in ("weirdest day", "weird day", "craziest day", "crazy day", "something happened", "had a story", "had the weirdest"))
            and not any(term in reply_norm for term in ("icl", "ngl", "yh", "nah", "u ", " ur ", "wby", "wbu"))
        ):
            return True
        if reply_norm.count("😉") + reply_norm.count("🤭") > 0 and not any(term in incoming_norm for term in ("miss", "bed", "baby", "babu", "dirty", "spicy")):
            return True
        if is_simple_greeting(incoming, context) and reply_norm in {"hey! what u up to", "hi! what's up with u then", "hey what u up to"}:
            return True
        return False

    def _catbot_wrong_owner_gender(self, reply_norm: str) -> bool:
        self_descriptions = (
            "im a girl",
            "i'm a girl",
            "i am a girl",
            "as a girl",
            "girl like me",
            "woman like me",
            "im not that kind of girl",
            "i'm not that kind of girl",
        )
        if any(term in reply_norm for term in self_descriptions):
            return True
        submissive_adult_pov = (
            "make me yours",
            "use me",
            "do what u want to me",
            "do what you want to me",
            "take me",
            "ruin me",
            "have your way with me",
            "have ur way with me",
            "i'm yours to use",
            "im yours to use",
            "i'm all yours to use",
            "im all yours to use",
            "ur mouth on me",
            "your mouth on me",
            "need ur mouth on me",
            "need your mouth on me",
            "taking me deeper in",
            "take me deeper in",
            "taking me deeper",
            "take me deeper",
            "pulling on me",
        )
        if any(term in reply_norm for term in submissive_adult_pov):
            return True
        return bool(
            re.search(
                r"\b(?:u|you|ur|your)\b.{0,24}\bmake(?:s|d)?\b.{0,12}\b(?:a|this)\b.{0,8}\b(?:girl|woman)\b.{0,12}\bblush\b",
                reply_norm,
            )
        )

    def _catbot_wrong_owner_style(self, reply_norm: str) -> bool:
        bubbles = [part.strip(" .?!") for part in re.split(r"\s*/\s*|\n+", reply_norm) if part.strip()]
        wrong_back_greetings = (
            "hey there",
            "hi there",
            "hi back",
            "hey back",
            "hey u back",
            "hey you back",
            "hi u back",
            "hi you back",
        )
        if any(part in wrong_back_greetings or part.startswith(tuple(f"{item} " for item in wrong_back_greetings)) for part in bubbles):
            return True
        if any(re.search(r"\b(?:x|xx|xxx|xo|xoxo)$", part) for part in bubbles):
            return True
        if any(re.match(r"^(?:hi{2,}|hii+)\b", part) for part in bubbles):
            return True
        if re.search(r"(?:^|[/\n]\s*)\b(?:mwah|kiss kiss)\b\s*$", reply_norm):
            return True
        return False

    def _catbot_bed_wby_reply_ok(self, reply_norm: str, *, incoming: str, context: list[str]) -> bool:
        incoming_norm = normalize_text(incoming)
        if not (any(term in incoming_norm for term in ("im in bed", "i'm in bed", "in bed")) and any(term in incoming_norm for term in ("wby", "wbu", "babu", "baby"))):
            return False
        if self._catbot_generic_ai_style(reply_norm, incoming=incoming, context=context):
            return False
        tokens = reply_norm.strip(" .?!").split()
        has_answer = any(term in reply_norm for term in ("im ", "i'm ", "still ", "same", "chilling", "work", "coding", "sofa", "bed", "up "))
        has_question = "?" in reply_norm or any(term in reply_norm for term in ("u sleeping", "you sleeping", "u staying up", "you staying up", "sleeping or", "chatting"))
        return has_answer and has_question and len(tokens) >= 5

    def _catbot_age_reply_ok(self, reply_norm: str, *, incoming: str) -> bool:
        incoming_norm = normalize_text(incoming)
        if not self._catbot_age_question(incoming_norm):
            return False
        if "22" in reply_norm:
            return False
        if "19" not in reply_norm:
            return False
        if len(reply_norm.split()) > 12:
            return False
        reply_clean = reply_norm.strip(" .?!")
        if reply_clean in {"19", "im 19", "i'm 19", "19 / wby", "19 wby", "19 / wbu", "im 19 / wby", "i'm 19 / wby"}:
            return False
        return any(
            term in reply_norm
            for term in (
                "why u asking",
                "why you asking",
                "what makes u ask",
                "what makes you ask",
                "dont make it sound like an interview",
                "don't make it sound like an interview",
                "same age",
                "wby then",
                "how old are u",
                "how old r u",
            )
        )

    def _catbot_age_question(self, incoming_norm: str) -> bool:
        return bool(re.search(r"\b(?:how old (?:r|are) (?:u|you)|age)\b", incoming_norm))

    def _catbot_location_origin_question(self, incoming_norm: str) -> bool:
        clean = incoming_norm.strip(" .?!")
        if self._catbot_whereabouts_question(clean):
            return False
        return bool(
            re.search(r"\bwhere\s+(?:r\s+|are\s+)?(?:u|you)\s+from\b", clean)
            or re.search(r"\bwhere\s+do\s+(?:u|you)\s+come\s+from\b", clean)
            or re.search(r"\b(?:u|you)\s+from\s+where\b", clean)
            or re.fullmatch(r"(?:where\s+from|from\s+where|whereabouts\s+(?:r\s+|are\s+)?(?:u|you)\s+from)", clean)
        )

    def _catbot_work_identity_question(self, incoming_norm: str) -> bool:
        clean = incoming_norm.strip(" .?!")
        return bool(
            re.fullmatch(r"(?:what|wot)\s+do\s+(?:u|you)\s+do", clean)
            or re.fullmatch(r"(?:what|wot)\s+(?:r|are)\s+(?:u|you)\s+doing\s+(?:for\s+work|with\s+ur\s+life|with\s+your\s+life)", clean)
            or re.fullmatch(r"(?:do\s+)?(?:u|you)\s+(?:work|study)\s+or\s+what", clean)
        )

    def _catbot_work_identity_reply_ok(self, reply_norm: str, *, incoming: str) -> bool:
        if not self._catbot_work_identity_question(normalize_text(incoming)):
            return False
        if self._catbot_generic_ai_style(reply_norm, incoming=incoming, context=[]):
            return False
        if any(term in reply_norm for term in ("uncle", "not at uni", "not studying", "no uni", "nothing atm")):
            return False
        tokens = reply_norm.strip(" .?!").split()
        if not 5 <= len(tokens) <= 24:
            return False
        has_study = any(term in reply_norm for term in ("computer science", "comp sci", "uni", "sampleford"))
        has_side = any(term in reply_norm for term in ("side", "running", "project", "building", "business", "software", "dev thing"))
        return has_study and has_side

    def _catbot_location_origin_reply_ok(self, reply_norm: str, *, incoming: str, context: list[str]) -> bool:
        if not self._catbot_location_origin_question(normalize_text(incoming)):
            return False
        if self._catbot_generic_ai_style(reply_norm, incoming=incoming, context=context):
            return False
        if any(term in reply_norm for term in ("address", "postcode", "street", "accommodation", "building", "live location", "where u at", "where you at")):
            return False
        context_blob = " ".join(normalize_text(item) for item in context[-8:])
        if "not at uni" in context_blob and "sampleford for uni" in reply_norm:
            return False
        tokens = reply_norm.strip(" .?!").split()
        if not 3 <= len(tokens) <= 16:
            return False
        has_location = any(term in reply_norm for term in ("northbridge", "northbridge", "sampleford"))
        has_safe_shape = any(term in reply_norm for term in ("northbridge", "northbridge")) and (
            any(term in reply_norm for term in ("sampleford", "mostly", "from"))
            or any(term in reply_norm for term in ("wby", "wbu", "what about u", "what about you"))
        )
        return has_location and has_safe_shape

    def _catbot_education_question(self, incoming_norm: str) -> bool:
        return bool(re.search(r"\b(?:what|which|where|wot)\b.*\b(?:uni|university|college|course|study|studying)\b", incoming_norm))

    def _catbot_education_reply_ok(self, reply_norm: str, *, incoming: str) -> bool:
        if not self._catbot_education_question(normalize_text(incoming)):
            return False
        if self._catbot_generic_ai_style(reply_norm, incoming=incoming, context=[]):
            return False
        if any(term in reply_norm for term in ("medicine", "law", "engineering", "business degree", "not at uni", "not studying", "no uni", "nothing atm")):
            return False
        tokens = reply_norm.strip(" .?!").split()
        if not 2 <= len(tokens) <= 18:
            return False
        incoming_norm = normalize_text(incoming)
        if re.search(r"\b(?:what|which|where|wot)\b.*\b(?:uni|university|college)\b", incoming_norm):
            return any(term in reply_norm for term in ("sampleford", "sampleford uni", "sampleford university"))
        if re.search(r"\b(?:course|study|studying)\b", incoming_norm):
            return any(term in reply_norm for term in ("computer science", "comp sci"))
        return any(term in reply_norm for term in ("computer science", "comp sci", "sampleford"))

    def _catbot_awake_status_question(self, incoming_norm: str) -> bool:
        return any(term in incoming_norm for term in ("still up", "u awake", "you awake", "are u awake", "are you awake", "asleep yet"))

    def _catbot_whereabouts_question(self, incoming_norm: str) -> bool:
        return bool(
            re.search(r"\bwhere\s+(?:u|you)\s+been\b", incoming_norm)
            or re.search(r"\bwhere\s+have\s+(?:u|you)\s+been\b", incoming_norm)
        )

    def _catbot_availability_planning_question(self, incoming_norm: str) -> bool:
        if self._catbot_awake_status_question(incoming_norm):
            return True
        return bool(
            re.search(r"\b(?:wyd|what\s+(?:u|you)\s+doing)\s+later\b", incoming_norm)
            or re.search(r"\b(?:u|you)\s+doing\s+anything\s+(?:nice|later)\b", incoming_norm)
            or re.search(r"\b(?:any|got)\s+plans\b", incoming_norm)
            or "anything nice" in incoming_norm
        )

    def _catbot_awake_status_reply_ok(self, reply_norm: str, *, incoming: str, context: list[str]) -> bool:
        if not self._catbot_awake_status_question(normalize_text(incoming)):
            return False
        if self._catbot_generic_ai_style(reply_norm, incoming=incoming, context=context):
            return False
        tokens = reply_norm.strip(" .?!").split()
        if len(tokens) < 3:
            return False
        return any(
            term in reply_norm
            for term in (
                "still up",
                "im up",
                "i'm up",
                "awake",
                "cant sleep",
                "can't sleep",
                "not asleep",
                "yh",
                "yeah",
            )
        ) and any(term in reply_norm for term in ("thinking", "chilling", "bed", "talk", "chat", "sleep", "u", "you"))

    def _catbot_availability_reply_ok(self, reply_norm: str, *, incoming: str, context: list[str]) -> bool:
        incoming_norm = normalize_text(incoming)
        if self._catbot_awake_status_question(incoming_norm):
            return self._catbot_awake_status_reply_ok(reply_norm, incoming=incoming, context=context)
        if not self._catbot_availability_planning_question(incoming_norm):
            return False
        if self._catbot_generic_ai_style(reply_norm, incoming=incoming, context=context):
            return False
        tokens = reply_norm.strip(" .?!").split()
        if len(tokens) < 4:
            return False
        if any(term in reply_norm for term in ("what about u", "what about you", "wby", "wbu")) and not any(
            term in reply_norm for term in ("nothing", "not really", "nah", "no ", "just", "work", "gym", "food", "chilling", "finished", "busy", "later", "tonight", "stuff")
        ):
            return False
        return any(
            term in reply_norm
            for term in (
                "nothing",
                "not really",
                "nah",
                "no ",
                "just",
                "work",
                "gym",
                "food",
                "chilling",
                "finished",
                "busy",
                "later",
                "tonight",
                "stuff",
                "plans",
            )
        )

    def _catbot_intensity_followup(self, incoming_norm: str, context: list[str]) -> bool:
        if not any(term in incoming_norm for term in ("how crazy", "how much", "how bad")):
            return False
        context_blob = " ".join(normalize_text(item) for item in context[-8:])
        return any(term in context_blob for term in ("miss", "crazy", "like mad", "need", "want", "crave", "craving"))

    def _catbot_intensity_followup_reply_ok(self, reply_norm: str, *, incoming: str, context: list[str]) -> bool:
        incoming_norm = normalize_text(incoming)
        if not self._catbot_intensity_followup(incoming_norm, context):
            return False
        tokens = reply_norm.strip(" .?!").split()
        if not 4 <= len(tokens) <= 18:
            return False
        has_intensity = any(term in reply_norm for term in ("crazy", "mad", "too much", "more than", "all day", "not normal", "enough that", "so much"))
        has_affection = any(term in reply_norm for term in ("miss", "baby", "you", "u", "thinking", "want"))
        return has_intensity and has_affection

    def _catbot_carry_convo_reply_ok(self, reply_norm: str, *, incoming: str, context: list[str]) -> bool:
        incoming_norm = normalize_text(incoming)
        if self._catbot_conversation_move(incoming=incoming, context=context).get("user_move") not in {"boredom_prompt", "carry_conversation_request"}:
            return False
        if self._catbot_generic_ai_style(reply_norm, incoming=incoming, context=context):
            return False
        tokens = reply_norm.strip(" .?!").split()
        has_question = "?" in reply_norm or any(term in reply_norm for term in ("what ", "what's", "whats", "why ", "how ", "u wanna", "you wanna", "u doing", "you doing", "tell me"))
        has_owner_move = any(term in reply_norm for term in ("fine", "okay", "alright", "ill carry", "i'll carry", "nah", "bet", "icl", "ngl", "lol", "ur ", "u "))
        has_topic = bool(self._catbot_reset_topic_family(reply_norm)) or any(
            term in reply_norm
            for term in ("sleep", "bed", "work", "film", "food", "gym", "day", "chat", "story", "question", "dream", "random", "annoy", "piss")
        )
        return has_question and has_owner_move and has_topic and len(tokens) >= 7

    def _catbot_has_romantic_signal(self, reply: str) -> bool:
        reply_norm = normalize_text(reply)
        return any(
            term in reply_norm
            for term in (
                "x",
                "xx",
                "😉",
                "😘",
                "cheeky",
                "proper kiss",
                "pull you",
                "pulling you",
                "against me",
                "til you're here",
                "till you're here",
                "babe",
                "baby",
                "miss",
                "want",
                "come over",
                "kiss",
                "cute",
                "blush",
                "thinking about you",
                "need you",
                "hands",
                "close",
                "tonight",
                "mine",
                "tease",
                "touch",
                "bed",
                "dirty",
                "spicy",
                "pussy",
                "dick",
                "cum",
                "horny",
                "hard",
                "wet",
                "waist",
                "attention",
                "intimate",
                "flirt",
                "innocent",
                "ideas",
            )
        )

    def _catbot_rescue_reply(
        self,
        *,
        incoming: str,
        context: list[str],
        selected: str,
        relationship: str,
        recent_bot_replies: list[str] | None = None,
    ) -> tuple[str, bool]:
        if relationship != "romantic_interest":
            return selected, False
        normalized = normalize_text(selected)
        weak = (
            not normalized
            or normalized in {"ok", "okay", "k", "true", "calm", "fair", "fine", "lol", "haha", "yh"}
            or len(normalized.split()) < 3
            or any(normalized == normalize_text(item) for item in context[-8::2])
        )
        context_bot_replies = [str(item).strip() for item in context[1::2] if str(item).strip()]
        stored_bot_replies = [str(item).strip() for item in (recent_bot_replies or []) if str(item).strip()]
        recent_bot: list[str] = []
        seen_recent: set[str] = set()
        for item in reversed([*stored_bot_replies, *context_bot_replies]):
            item_norm = normalize_text(item)
            if item_norm and item_norm not in seen_recent:
                recent_bot.append(item)
                seen_recent.add(item_norm)
        recent_bot.reverse()
        repeated = any(normalized and normalized == normalize_text(item) for item in recent_bot[-5:])
        incoming_norm = normalize_text(incoming)
        adult_mode = self._catbot_adult_mode(incoming_norm, context)
        needs_flirt = adult_mode or self._catbot_romantic_mode(incoming_norm, context)
        if needs_flirt and normalized and self._catbot_has_romantic_signal(selected) and not weak and not repeated:
            return selected, False
        if not (weak or repeated or needs_flirt):
            return selected, False

        candidates = self._catbot_romantic_candidates(incoming_norm, adult_mode=adult_mode)
        if candidates:
            context_fingerprint = "|".join(normalize_text(item) for item in context[-12:])
            offset = int(hashlib.sha1(f"{incoming_norm}|{context_fingerprint}".encode("utf-8")).hexdigest()[:4], 16) % len(candidates)
            candidates = [*candidates[offset:], *candidates[:offset]]
        recent_norm = {normalize_text(item) for item in recent_bot[-20:]}
        for candidate in candidates:
            if normalize_text(candidate) not in recent_norm:
                return candidate, True
        very_recent_norm = {normalize_text(item) for item in recent_bot[-8:]}
        for candidate in candidates:
            if normalize_text(candidate) not in very_recent_norm:
                return candidate, True
        return candidates[0], True

    def _catbot_romantic_mode(self, incoming_norm: str, context: list[str]) -> bool:
        blob = " ".join([incoming_norm, *[normalize_text(item) for item in context[-10:]]])
        if self._catbot_missed_you_incoming(blob):
            return True
        return any(
            term in blob
            for term in (
                "date",
                "flirt",
                "miss me",
                "missed me",
                "missed you",
                "miss u",
                "miss you",
                "crave",
                "craving",
                "want me",
                "want you",
                "want u",
                "need u",
                "need you",
                "come over",
                "kiss",
                "blush",
                "cute",
                "romantic",
                "attention",
                "tonight",
            )
        )

    def _catbot_adult_mode(self, incoming_norm: str, context: list[str]) -> bool:
        blob = " ".join([incoming_norm, *[normalize_text(item) for item in context[-10:]]])
        return any(
            term in blob
            for term in (
                "dirty",
                "spicy",
                "crave",
                "craving",
                "horny",
                "cock",
                "dick",
                "pussy",
                "clit",
                "hole",
                "cum",
                "hard",
                "wet",
                "bed",
                "hands",
                "touch",
                "lips",
                "neck",
                "thigh",
                "tongue",
                "tease",
                "whimper",
                "moan",
                "intimate",
                "tonight",
                "come over",
                "kiss me",
                "want me there",
            )
        )

    def _catbot_romantic_candidates(self, incoming_norm: str, *, adult_mode: bool) -> list[str]:
        if "what did i say i liked" in incoming_norm or "remember" in incoming_norm:
            return [
                "you said you like confident texts, and you keep bringing up my hands",
                "yeah, confident texts and me not dodging how much i want you",
                "you like when i lead it and keep my attention on you",
            ]
        if "what are you doing" in incoming_norm or "right now" in incoming_norm:
            return [
                "thinking about you now, clearly you got my attention",
                "nothing useful now, i want you close and distracting me properly",
                "trying to act normal but wanting you close makes that difficult",
            ]
        if "long day" in incoming_norm or "distract me" in incoming_norm:
            return [
                "come here then, ill distract you properly",
                "i can distract you, but you know i wont keep it innocent for long",
                "put your phone down and let me take your mind off it",
            ]
        if "miss" in incoming_norm:
            return [
                "i miss you too, but you already knew that",
                "i miss you too, i was just pretending to be composed",
                "i miss you too icl, come fix that",
            ]
        if "come over" in incoming_norm or "want me there" in incoming_norm or "there tonight" in incoming_norm:
            return [
                "yeah i want you here tonight, no pretending",
                "come over then, i want you close not just texting me",
                "yes, i want you here where i can actually do something about it",
                "i want you here tonight, and i am not trying to sound casual about it",
                "come over and stop making me say this through a screen",
                "yes i want you here, and i want you close when you get here",
                "come over, because texting you like this is starting to annoy me",
                "i want you here tonight, not sending brave little messages from there",
                "yes, come here close and let me prove i am not just chatting",
                "i want you here enough that i am done pretending otherwise",
            ]
        if "first" in incoming_norm or "when you see me" in incoming_norm:
            return [
                "id pull you close first, then kiss you like i meant it",
                "first thing, hands on your waist and no more acting calm",
                "id get you close enough that you forget what you were about to say",
            ]
        if "hands" in incoming_norm or "where i want your hands" in incoming_norm:
            return [
                "tell me where, because id start at your waist and take my time",
                "my hands would not behave around you tonight",
                "id keep my hands on you just enough to make you lose focus",
            ]
        if "dirty" in incoming_norm or "spicy" in incoming_norm or adult_mode:
            return [
                "i want you close enough that i can tease you properly, not from a screen",
                "id be sweet for about two minutes, then my hands would give me away",
                "come here and ill show you exactly why i kept thinking about you",
                "i want you close, talking brave until i can make you quiet",
                "id keep it sweet at first, then let my hands do the flirting",
                "you would not be getting polite texts if i had you close",
                "i want you close enough to stop pretending this is innocent",
                "come over and ill make that confidence problem worse",
                "id start soft, then make it very obvious i want you",
                "you know id have you close and forgetting your attitude fast",
                "id tease you until you stopped acting so composed",
                "i want you here, close, and definitely not behaving",
                "id make you blush first, then make you forget why you were talking",
                "come over and ill stop letting you win this through texts",
            ]
        if "flirt" in incoming_norm or "blush" in incoming_norm or "direct" in incoming_norm:
            return [
                "i want you, and i like when you make me say it directly",
                "you are trouble, but the kind i would happily let come over",
                "stop acting innocent, you know exactly what you do to me",
                "you make it too easy to flirt with you",
                "i want you closer, and yes that is me being direct",
                "you are cute when you try to make me say the obvious",
                "i am flirting properly now, so do not start acting shy",
                "you know i want you, i am just enjoying making you ask",
                "come closer then, since you want me to be direct",
                "you have my attention, and i am not being subtle about it",
                "i like making you blush, so do not tempt me",
                "you are not innocent enough to act surprised when i make you blush",
            ]
        if "lead" in incoming_norm or "carry" in incoming_norm or "pick the thread" in incoming_norm:
            return [
                "fine, ill lead it: come closer and stop making me ask twice",
                "alright, im taking over then, tell me what you would do if i was close",
                "then keep up, because im not letting this turn boring again",
                "fine, my lead: you come here and let me make this easier",
                "okay, im carrying it now, answer me properly and stop hiding",
                "then i am choosing the mood, and the mood is you closer",
                "ill lead it, but you are not allowed to go shy now",
                "say less, im taking over and you are following my lead",
                "good, then keep your attention on me for a minute",
                "im picking the direction: you, me, and no boring replies",
            ]
        return [
            "you are making it hard to reply normally when i want you close",
            "i like this cute side of you, keep going",
            "you know exactly how to get my attention",
            "come here and say that properly, i want the confident version",
            "you are trouble, and i clearly want to entertain it",
            "i should not want this as much as i do",
            "keep talking like that and ill start getting ideas",
            "i am trying to behave and you are not helping, baby",
            "you have my attention now, so use it properly",
            "say that again but act less innocent",
        ]

    def training_catbot_feedback(self, request: CatbotFeedbackRequest) -> dict[str, object]:
        rows = self._read_jsonl_tail(self.training_sessions_path, limit=100000)
        session: dict[str, object] | None = None
        for row in reversed(rows):
            if str(row.get("id") or "") == request.session_id and str(row.get("source") or "") == "catbot":
                session = row
                break
        if session is None:
            raise HTTPException(status_code=404, detail="Catbot session not found.")
        rating = request.rating.strip().lower()
        corrected = redact_private_text(request.corrected_reply.strip())
        selected = str(session.get("selected_candidate") or "")
        final_reply = corrected if corrected else selected
        feedback_label = "approved" if rating == "thumbs_up" else "rejected"
        feedback_row = {
            "id": f"catbot_feedback_{uuid.uuid4().hex[:12]}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "catbot_session_id": request.session_id,
            "thread_id": session.get("thread_id", ""),
            "incoming": session.get("incoming", ""),
            "context": session.get("context", []),
            "contact_name": session.get("contact_name", "Catbot"),
            "relationship_type": session.get("relationship_type", "close_friend"),
            "intent_type": session.get("intent_type", "unknown"),
            "selected_candidate": selected,
            "feedback_label": feedback_label,
            "thumb_rating": rating,
            "correct_reply": final_reply,
            "notes": request.notes,
            "source": "catbot_feedback",
            "quarantined": False,
        }
        session_candidates = session.get("candidates") if isinstance(session.get("candidates"), list) else []
        selected_candidate_metadata = {}
        if isinstance(session_candidates, list):
            for candidate in session_candidates:
                if isinstance(candidate, dict) and str(candidate.get("text") or "") == selected:
                    selected_candidate_metadata = candidate
                    break
            if not selected_candidate_metadata and session_candidates and isinstance(session_candidates[0], dict):
                selected_candidate_metadata = session_candidates[0]
        feedback_row["selected_candidate_metadata"] = selected_candidate_metadata
        feedback_row["candidate_count"] = len(session_candidates) if isinstance(session_candidates, list) else 0
        feedback_row["scene_type"] = selected_candidate_metadata.get("scene_type", "") if selected_candidate_metadata else ""
        feedback_row["required_reply_move"] = selected_candidate_metadata.get("required_reply_move", "") if selected_candidate_metadata else ""
        self._append_jsonl(self.training_sessions_path, feedback_row)
        thread_id = str(session.get("thread_id") or "")
        if thread_id:
            self.drafting.thread_memory.record_feedback(
                thread_id,
                request.session_id,
                rating,
                corrected if corrected else None,
                failure_context={
                    "incoming": session.get("incoming", ""),
                    "context": session.get("context", []),
                    "selected_candidate": selected,
                    "selected_candidate_metadata": selected_candidate_metadata,
                    "candidates": session_candidates,
                },
            )
        correction_row: dict[str, object] | None = None
        vector_status: dict[str, object] = {"vector_rebuild_required": False, "incremental_added": False}
        if rating == "thumbs_up" and final_reply:
            payload = {
                "timestamp": feedback_row["timestamp"],
                "relationship_type": feedback_row["relationship_type"],
                "intent_type": feedback_row["intent_type"],
                "contact_name": feedback_row["contact_name"],
                "incoming": feedback_row["incoming"],
                "context": feedback_row["context"],
                "bad_ai_reply": "",
                "user_final_reply": final_reply,
                "reason_bad": "other",
                "feedback_label": "approved",
                "source": "catbot",
                "style_authority": "medium",
            }
            try:
                correction_row = append_correction(self.settings.ai_reply_training_messages_dir, payload)
                vector_status["vector_rebuild_required"] = True
                self.vector_rebuild_marker_path.parent.mkdir(parents=True, exist_ok=True)
                self.vector_rebuild_marker_path.write_text("catbot feedback requires vector rebuild\n", encoding="utf-8")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "status": "saved",
            "source": "catbot",
            "feedback_label": feedback_label,
            "thumb_rating": rating,
            "session": feedback_row,
            "correction": correction_row,
            **vector_status,
            "adb_touched": False,
        }

    def training_feedback(self, request: TrainingFeedbackRequest) -> dict[str, object]:
        label = request.feedback_label.strip().lower()
        valid_labels = {
            "too_formal",
            "too_long",
            "not_me",
            "missed_context",
            "risky",
            "too_flirty",
            "too_dry",
            "approved",
            "edited",
            "rejected",
        }
        if label not in valid_labels:
            raise HTTPException(status_code=400, detail="Invalid feedback_label.")
        corrected = redact_private_text(request.correct_reply.strip())
        if request.save_to_corrections and not corrected:
            raise HTTPException(status_code=400, detail="Correct reply is required when saving to corrections.")
        relationship = normalize_relationship_type(request.relationship_type)
        intent = request.intent_type if request.intent_type != "auto" else classify_intent(request.incoming, request.context)
        intent = str(intent).strip() or "unknown"
        if corrected and contains_suspicious_phrase(corrected) and relationship != "romantic_interest":
            raise HTTPException(status_code=400, detail="Correction contains unsafe or flirty text for this relationship.")
        timestamp = datetime.now(timezone.utc).isoformat()
        example_id = f"training_{uuid.uuid4().hex[:12]}"
        session_row = {
            "id": example_id,
            "timestamp": timestamp,
            "incoming": redact_private_text(request.incoming.strip()),
            "context": [redact_private_text(str(item)) for item in request.context if str(item).strip()],
            "contact_name": redact_private_text(request.contact_name or ""),
            "relationship_type": relationship,
            "intent_type": intent,
            "parameters": request.parameters.model_dump(mode="json"),
            "candidates": [request.ai_reply] if request.ai_reply else [],
            "selected_candidate": request.ai_reply,
            "feedback_label": label,
            "correct_reply": corrected,
            "notes": request.notes,
            "retrieved_example_ids": request.retrieved_example_ids,
            "source": "training_page",
            "quarantined": False,
        }
        self._append_jsonl(self.training_sessions_path, session_row)
        correction_row: dict[str, object] | None = None
        vector_status: dict[str, object] = {"vector_rebuild_required": False, "incremental_added": False}
        if request.save_to_corrections and label != "rejected":
            payload = {
                "timestamp": timestamp,
                "relationship_type": relationship,
                "intent_type": intent,
                "contact_name": request.contact_name or "",
                "incoming": request.incoming,
                "context": request.context,
                "bad_ai_reply": request.ai_reply,
                "user_final_reply": corrected,
                "reason_bad": self._reason_from_feedback_label(label),
                "feedback_label": label,
                "parameters": request.parameters.model_dump(mode="json"),
                "source": "training_page_feedback",
                "style_authority": "high",
            }
            correction_row = append_correction(self.settings.ai_reply_training_messages_dir, payload)
            if request.add_to_vector_db:
                vector_status = self._add_training_feedback_to_vector_db(
                    {
                        **payload,
                        "id": example_id,
                        "my_reply": corrected,
                        "is_correction": True,
                        "is_synthetic": False,
                        "is_template": False,
                        "unsafe": False,
                        "is_quarantined": False,
                        "_source": "correction",
                    }
                )
        elif request.add_to_vector_db:
            vector_status = self._mark_vector_rebuild_required("feedback_not_saved_to_corrections")
        return {
            "status": "saved",
            "session": session_row,
            "correction": correction_row,
            **vector_status,
        }

    def training_style_review_queue(self) -> dict[str, object]:
        rows = self._read_jsonl_tail(self.style_review_queue_path, limit=100000)
        pending = [row for row in rows if str(row.get("status") or "needs_human_review") == "needs_human_review"]
        return {"items": pending, "total_pending": len(pending), "path": str(self.style_review_queue_path)}

    def training_style_review_approve(self, request: StyleReviewApproveRequest) -> dict[str, object]:
        rows = self._read_jsonl_tail(self.style_review_queue_path, limit=100000)
        row_id = request.row_id.strip()
        approved: dict[str, object] | None = None
        vector_status: dict[str, object] = {"vector_rebuild_required": False, "incremental_added": False}
        updated: list[dict[str, object]] = []
        for row in rows:
            matches = str(row.get("row_id") or row.get("id")) == row_id
            pending = str(row.get("status") or "needs_human_review") == "needs_human_review"
            if matches and pending:
                reply = redact_private_text((request.edited_reply or str(row.get("suggested_better_reply", ""))).strip())
                if not reply:
                    raise HTTPException(status_code=400, detail="Approved reply must not be empty.")
                relationship = normalize_relationship_type(str(row.get("relationship_type", "unknown")))
                if contains_suspicious_phrase(reply) and relationship != "romantic_interest":
                    raise HTTPException(status_code=400, detail="Approved reply contains unsafe or flirty text for this relationship.")
                incoming = str(row.get("incoming") or "")
                for existing in load_corrections(self.settings.ai_reply_training_messages_dir):
                    if (
                        str(existing.get("source")) == "approved_auto_style_improvement"
                        and str(existing.get("incoming", "")).strip().casefold() == incoming.strip().casefold()
                        and str(existing.get("user_final_reply", "")).strip().casefold() == reply.strip().casefold()
                    ):
                        raise HTTPException(status_code=409, detail="This style review correction is already approved.")
                payload = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "relationship_type": relationship,
                    "intent_type": str(row.get("intent_type") or "unknown"),
                    "contact_name": str(row.get("contact_name") or ""),
                    "incoming": incoming,
                    "context": row.get("context", []) if isinstance(row.get("context"), list) else [],
                    "bad_ai_reply": str(row.get("bad_ai_reply") or ""),
                    "user_final_reply": reply,
                    "reason_bad": "not me",
                    "feedback_label": "edited",
                    "source": "approved_auto_style_improvement",
                    "style_authority": "high",
                    "approved_by_user": True,
                    "selected_mode": "review",
                }
                correction = append_correction(self.settings.ai_reply_training_messages_dir, payload)
                vector_status = self._mark_vector_rebuild_required("approved_auto_style_improvement")
                self.training_loop.schedule_rebuild_after_approval()
                approved = {
                    **row,
                    "status": "approved",
                    "approved_at": datetime.now(timezone.utc).isoformat(),
                    "approved_reply": reply,
                    "correction": correction,
                }
                updated.append(approved)
                continue
            updated.append(row)
        if approved is None:
            raise HTTPException(status_code=404, detail="Style review row_id not found.")
        self._write_jsonl(self.style_review_queue_path, updated)
        return {"status": "approved", "item": approved, **vector_status}

    def training_style_review_reject(self, request: StyleReviewRejectRequest) -> dict[str, object]:
        rows = self._read_jsonl_tail(self.style_review_queue_path, limit=100000)
        row_id = request.row_id.strip()
        rejected: dict[str, object] | None = None
        updated: list[dict[str, object]] = []
        for row in rows:
            matches = str(row.get("row_id") or row.get("id")) == row_id
            pending = str(row.get("status") or "needs_human_review") == "needs_human_review"
            if matches and pending:
                rejected = {
                    **row,
                    "status": "rejected",
                    "rejected_at": datetime.now(timezone.utc).isoformat(),
                    "reject_reason": request.reason,
                }
                updated.append(rejected)
                continue
            updated.append(row)
        if rejected is None:
            raise HTTPException(status_code=404, detail="Style review row_id not found.")
        self._write_jsonl(self.style_review_queue_path, updated)
        return {"status": "rejected", "item": rejected, "vector_rebuild_required": self.vector_rebuild_marker_path.exists()}

    def training_active_learning_queue(self) -> dict[str, object]:
        path = self.settings.ai_reply_training_messages_dir / "active_learning_queue.jsonl"
        rows = self._read_jsonl_tail(path, limit=100000)
        return {"items": rows, "total": len(rows), "path": str(path)}

    def training_stats(self) -> dict[str, object]:
        training_dir = self.settings.ai_reply_training_messages_dir
        corrections = load_corrections(training_dir)
        rows = load_all_training_rows(training_dir)
        real_count = sum(1 for row in rows if not row.get("is_synthetic") and row.get("source") != "generated_safe_template")
        synthetic_count = sum(1 for row in rows if row.get("is_synthetic") or row.get("source") == "generated_safe_template")
        vector_count = self._vector_index_count()
        vector_memory_counts = self._vector_memory_type_counts()
        thread_stats = self.drafting.thread_memory.stats()
        identity_status = self.drafting.identity_pack.to_safe_status()
        feedback_labels = Counter(
            str(row.get("feedback_label", "unknown"))
            for row in self._read_jsonl_tail(self.training_sessions_path, limit=10000)
            if row.get("feedback_label")
        )
        last_rebuild = None
        if self.vector_rebuild_meta_path.exists():
            try:
                last_rebuild = json.loads(self.vector_rebuild_meta_path.read_text(encoding="utf-8")).get("timestamp")
            except Exception:
                last_rebuild = None
        return {
            "corrections_count": len(corrections),
            "real_examples_count": real_count,
            "synthetic_examples_count": synthetic_count,
            "vector_index_count": vector_count,
            "thread_vector_count": vector_memory_counts.get("thread_episode", 0),
            "thread_episode_count": thread_stats["thread_episode_count"],
            "recent_thread_count": thread_stats["recent_thread_count"],
            "thread_memory_path": thread_stats["thread_memory_path"],
            "thread_memory_enabled": thread_stats["thread_memory_enabled"],
            **identity_status,
            "vector_db_available": (self.settings.data_dir / "vector_db" / "reply_examples_chroma").exists(),
            "vector_rebuild_required": self.vector_rebuild_marker_path.exists(),
            "last_vector_rebuild": last_rebuild,
            "top_feedback_labels": dict(feedback_labels.most_common(12)),
        }

    def ai_core_status(self) -> dict[str, object]:
        training_dir = self.settings.ai_reply_training_messages_dir
        rows = load_all_training_rows(training_dir)
        corrections = load_corrections(training_dir)
        training_files: list[dict[str, object]] = []
        for path in sorted(training_dir.glob("*.jsonl")):
            if path.parent.name == "quarantine":
                continue
            line_count = 0
            if path.exists():
                line_count = sum(1 for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())
            training_files.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "rows": line_count,
                    "bytes": path.stat().st_size if path.exists() else 0,
                }
            )
        vector_dir = self.settings.data_dir / "vector_db" / "reply_examples_chroma"
        vector_memory_counts = self._vector_memory_type_counts()
        thread_stats = self.drafting.thread_memory.stats()
        identity_status = self.drafting.identity_pack.to_safe_status()
        retrieval_config_path = self.settings.data_dir / "retrieval_config.json"
        retrieval_config: dict[str, object] = {}
        if retrieval_config_path.exists():
            try:
                retrieval_config = json.loads(retrieval_config_path.read_text(encoding="utf-8"))
            except Exception:
                retrieval_config = {"error": "invalid retrieval_config.json"}
        return {
            "files": {
                "style_profile": str(self.settings.ai_reply_style_profile_path),
                "training_messages_dir": str(training_dir),
                "corrections": str(training_dir / "corrections.jsonl"),
                "training_sessions": str(self.training_sessions_path),
                "training_sessions_quarantine": str(self.training_quarantine_path),
                "retrieval_config": str(retrieval_config_path),
                "contact_profiles": str(self.settings.data_dir / "contact_profiles.json"),
                "conversation_state": str(self.settings.data_dir / "conversation_state.json"),
                "thread_memory": str(self.settings.data_dir / "thread_memory"),
                "identity_pack": identity_status["identity_pack_path"],
                "automation_targets": str(self.settings.data_dir / "automation_targets.json"),
                "vector_db": str(vector_dir),
            },
            "training_data": {
                "active_rows": len(rows),
                "corrections_count": len(corrections),
                "synthetic_examples_count": sum(1 for row in rows if row.get("is_synthetic") or row.get("source") == "generated_safe_template"),
                "real_examples_count": sum(1 for row in rows if not row.get("is_synthetic") and row.get("source") != "generated_safe_template"),
                "files": training_files,
            },
            "vector_database": {
                "path": str(vector_dir),
                "available": vector_dir.exists(),
                "indexed_count": self._vector_index_count(),
                "thread_vector_count": vector_memory_counts.get("thread_episode", 0),
                "rebuild_required": self.vector_rebuild_marker_path.exists(),
                "last_rebuild": self.training_stats().get("last_vector_rebuild"),
                "thread_episode_count": thread_stats["thread_episode_count"],
                "thread_memory_path": thread_stats["thread_memory_path"],
                "thread_memory_enabled": thread_stats["thread_memory_enabled"],
            },
            "identity": identity_status,
            "retrieval": {
                "config": retrieval_config,
                "backend": retrieval_config.get("backend", self.drafting.retrieval_backend_name if hasattr(self.drafting, "retrieval_backend_name") else "unknown"),
            },
            "providers": provider_config_status(self.settings),
            "provider_routing": {
                "draft_provider": self.settings.draft_provider,
                "fast_provider": self.settings.fast_provider,
                "router_provider": self.settings.router_provider,
                "private_provider": self.settings.private_provider,
                "external_api_enabled": self.settings.external_api_enabled,
                "external_api_allow_sensitive": self.settings.external_api_allow_sensitive,
                "external_api_max_context_messages": self.settings.external_api_max_context_messages,
                "external_api_timeout_seconds": self.settings.external_api_timeout_seconds,
            },
            "safety": {
                "training_page_sends_phone_messages": False,
                "unknown_review_only": True,
                "synthetic_style_authority": "low",
                "corrections_style_authority": "high",
            },
        }

    def ai_core_reload_identity_pack(self) -> dict[str, object]:
        pack = self.drafting.reload_identity_pack()
        return {"status": "reloaded", **pack.to_safe_status()}

    def ai_core_provider_compare(self, request: ProviderCompareRequest) -> dict[str, object]:
        incoming = request.incoming.strip()
        if not incoming:
            raise HTTPException(status_code=400, detail="incoming must not be empty.")
        self.drafting.external_api_enabled = self.settings.external_api_enabled
        self.drafting.external_api_allow_sensitive = self.settings.external_api_allow_sensitive
        self.drafting.external_api_max_context_messages = self.settings.external_api_max_context_messages
        self.drafting.external_api_timeout_seconds = self.settings.external_api_timeout_seconds
        self.drafting.draft_provider = self.settings.draft_provider
        self.drafting.fast_provider = self.settings.fast_provider
        self.drafting.router_provider = self.settings.router_provider
        self.drafting.private_provider = self.settings.private_provider
        providers = [
            provider.strip().lower()
            for provider in request.providers
            if provider.strip()
        ] or ["ollama"]
        results = self.drafting.compare_providers(
            incoming=incoming,
            context=request.context,
            relationship_type=request.relationship_type,
            intent_type=request.intent_type,
            providers=providers,
        )
        return {"results": results}

    def ai_core_vector_map(self, limit: int = 900, memory_type: str = "all") -> dict[str, object]:
        started = time.perf_counter()
        safe_limit = max(50, min(int(limit or 900), 2500))
        memory_filter = str(memory_type or "all").strip().casefold()
        if memory_filter not in {"all", "reply_example", "thread_episode"}:
            memory_filter = "all"
        vector_dir = self.settings.data_dir / "vector_db" / "reply_examples_chroma"
        points, embeddings, source = self._load_vector_map_rows(vector_dir, safe_limit)
        if memory_filter != "all":
            points = [point for point in points if str(point.get("memory_type") or "reply_example") == memory_filter]
            embeddings = []
        projected = self._project_vector_points(points, embeddings)
        clusters = self._vector_map_clusters(projected)
        return {
            "status": "ok",
            "vector_db_path": str(vector_dir),
            "source": source,
            "memory_type": memory_filter,
            "count": len(projected),
            "projection": "pca_3d" if embeddings else "metadata_cluster",
            "timing_ms": round((time.perf_counter() - started) * 1000, 2),
            "points": projected,
            "clusters": clusters,
            "legend": {
                "correction": "High-authority user corrections",
                "real": "Real cleaned training examples",
                "synthetic": "Low-authority generated safe templates",
            },
        }

    def training_recent(self, limit: int = 50) -> dict[str, object]:
        safe_limit = max(1, min(int(limit or 50), 200))
        return {"items": self._read_jsonl_tail(self.training_sessions_path, limit=safe_limit)}

    def training_rebuild_vector_db(self) -> dict[str, object]:
        started = time.perf_counter()
        try:
            from scripts.build_reply_vector_db import build_reply_vector_db

            result = build_reply_vector_db()
        except Exception as exc:
            return {"status": "error", "error": str(exc), "timing_ms": round((time.perf_counter() - started) * 1000, 2)}
        self.vector_rebuild_marker_path.unlink(missing_ok=True)
        self.vector_rebuild_meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_rebuild_meta_path.write_text(
            json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), **result}, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        return {"status": "rebuilt", "timing_ms": round((time.perf_counter() - started) * 1000, 2), **result}

    def training_run_evaluation(self, limit: int | None = None) -> dict[str, object]:
        args = [sys.executable, "scripts/evaluate_reply_style.py"]
        if limit:
            args.extend(["--limit", str(limit)])
        return self._run_training_command(args, report_path=self.settings.data_dir / "reports" / "reply_style_eval_report.json")

    def training_generate_improvements(self, limit: int | None = None, *, write: bool = True) -> dict[str, object]:
        args = [sys.executable, "scripts/improve_failed_replies.py"]
        if limit:
            args.extend(["--limit", str(limit)])
        if write:
            args.append("--write")
        return self._run_training_command(args)

    def training_run_iteration(self, limit: int | None = None) -> dict[str, object]:
        args = [sys.executable, "scripts/run_training_iteration.py"]
        if limit:
            args.extend(["--limit", str(limit)])
        return self._run_training_command(args, report_path=self.settings.data_dir / "reports" / "training_iteration_report.json")

    def training_status(self) -> dict[str, object]:
        report = self._read_json_file(self.settings.data_dir / "reports" / "reply_style_eval_report.json")
        history = self._read_jsonl_tail(self.settings.data_dir / "reports" / "style_eval_history.jsonl", limit=2)
        review = self.training_style_review_queue()
        stats = self.training_stats()
        loop_status = self.training_loop.status()
        return {
            "last_eval": {
                "total_cases": report.get("total_cases"),
                "pass_count": report.get("pass_count"),
                "fail_count": report.get("fail_count"),
                "average_overall_score": report.get("average_overall_score"),
                "average_style_score": report.get("average_style_score"),
                "top_failure_reasons": report.get("top_hard_reject_reasons", {}),
            },
            "previous_eval": history[-2] if len(history) >= 2 else None,
            "latest_history": history[-1] if history else None,
            "review_queue_count": review.get("total_pending", 0),
            "vector_rebuild_required": stats.get("vector_rebuild_required", False),
            "last_vector_rebuild": stats.get("last_vector_rebuild"),
            "approved_corrections_count": stats.get("corrections_count", 0),
            "active_learning_queue_count": loop_status.get("active_learning_queue_count", 0),
            "training_loop": loop_status,
        }

    def training_loop_status(self) -> dict[str, object]:
        return self.training_loop.status()

    def training_loop_start(self) -> dict[str, object]:
        return self.training_loop.start()

    def training_loop_stop(self) -> dict[str, object]:
        return self.training_loop.stop()

    def training_loop_run_now(self) -> dict[str, object]:
        return self.training_loop.run_now(background=True)

    def training_loop_update_config(self, payload: dict[str, object]) -> dict[str, object]:
        updates = {key: value for key, value in payload.items() if value is not None}
        config = self.training_loop.save_config(updates)
        return {"status": "updated", "config": config, "loop_status": self.training_loop.status()}

    def training_quarantine_example(self, request: TrainingQuarantineRequest) -> dict[str, object]:
        rows = self._read_jsonl_tail(self.training_sessions_path, limit=100000)
        kept: list[dict[str, object]] = []
        quarantined: dict[str, object] | None = None
        for row in rows:
            if str(row.get("id", "")) == request.example_id:
                quarantined = {
                    **row,
                    "quarantined": True,
                    "quarantine_reason": request.reason,
                    "quarantined_at": datetime.now(timezone.utc).isoformat(),
                }
                continue
            kept.append(row)
        if quarantined is None:
            raise HTTPException(status_code=404, detail="Training session example_id not found.")
        self._write_jsonl(self.training_sessions_path, kept)
        self._append_jsonl(self.training_quarantine_path, quarantined)
        self._mark_vector_rebuild_required("training_example_quarantined")
        return {"status": "quarantined", "example": quarantined, "vector_rebuild_required": True}

    def _run_training_command(self, args: list[str], *, report_path: Path | None = None) -> dict[str, object]:
        started = time.perf_counter()
        completed = subprocess.run(args, cwd=Path.cwd(), text=True, capture_output=True)
        payload = self._read_json_file(report_path) if report_path else {}
        return {
            "status": "ok" if completed.returncode == 0 else "error",
            "exit_code": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
            "timing_ms": round((time.perf_counter() - started) * 1000, 2),
            "report": payload,
        }

    def _read_json_file(self, path: Path | None) -> dict[str, object]:
        if path is None or not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _training_bundle(self, request: TrainingChatRequest, *, regenerate: bool) -> DraftBundle:
        intent = classify_intent(request.incoming, request.context) if request.intent_type == "auto" else request.intent_type
        relationship = normalize_relationship_type(request.relationship_type)
        avoid = list(request.avoid_candidates)
        if regenerate:
            bundle = self.drafting.regenerate_bundle(
                contact_name=request.contact_name,
                incoming=request.incoming,
                context=request.context,
                relationship_type=relationship,
                intent_type=intent,
                avoid_candidates=avoid,
                diversity_mode=request.diversity_mode,
            )
        else:
            bundle = self.drafting.regenerate_bundle(
                contact_name=request.contact_name,
                incoming=request.incoming,
                context=request.context,
                relationship_type=relationship,
                intent_type=intent,
                avoid_candidates=avoid,
                diversity_mode=request.diversity_mode,
            )
            should_use_general_natural_bundle = (
                not avoid
                and request.diversity_mode == "natural"
                and request.intent_type == "auto"
                and relationship == "unknown"
                and len(request.context) <= 5
            )
            if should_use_general_natural_bundle:
                messages = [*request.context[-5:], request.incoming]
                bundle = self.drafting.build_bundle(messages, contact_name=request.contact_name)
                bundle.relationship_type = relationship
                bundle.intent_type = intent
        self._apply_training_parameters(bundle, request.parameters.model_dump(mode="json"), relationship)
        return bundle

    def _training_response(self, request: TrainingChatRequest, bundle: DraftBundle) -> dict[str, object]:
        selected = bundle.reply_candidates[bundle.recommended_reply_index or 0] if bundle.reply_candidates else None
        return {
            "incoming": request.incoming,
            "context": request.context,
            "contact_name": request.contact_name,
            "relationship_type": bundle.relationship_type,
            "intent_type": bundle.intent_type,
            "candidates": [candidate.model_dump(mode="json") for candidate in bundle.reply_candidates],
            "selected_candidate": selected.text if selected else "",
            "retrieved_examples": bundle.retrieved_examples,
            "retrieval_backend": bundle.retrieval_backend,
            "critic": selected.critic_scores if selected else {},
            "prompt_preview": bundle.prompt_preview,
            "timings_ms": bundle.timings_ms,
            "parameters": request.parameters.model_dump(mode="json"),
            "vector_results_count": bundle.vector_results_count,
            "retrieval_scores": bundle.retrieval_scores,
            "retrieval_reasons": bundle.retrieval_reasons,
        }

    def _training_session_payload(self, request: TrainingChatRequest, bundle: DraftBundle) -> dict[str, object]:
        return {
            "id": f"session_{uuid.uuid4().hex[:12]}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "incoming": redact_private_text(request.incoming),
            "context": [redact_private_text(str(item)) for item in request.context if str(item).strip()],
            "contact_name": redact_private_text(request.contact_name or ""),
            "relationship_type": bundle.relationship_type,
            "intent_type": bundle.intent_type,
            "parameters": request.parameters.model_dump(mode="json"),
            "candidates": [candidate.model_dump(mode="json") for candidate in bundle.reply_candidates],
            "selected_candidate": bundle.reply_suggestions[bundle.recommended_reply_index or 0] if bundle.reply_suggestions else "",
            "feedback_label": "",
            "correct_reply": "",
            "notes": "",
            "retrieved_example_ids": [item.get("retrieval_id", "") for item in bundle.retrieved_examples],
            "source": "training_page",
            "quarantined": False,
        }

    def _apply_training_parameters(
        self,
        bundle: DraftBundle,
        parameters: dict[str, object],
        relationship_type: str,
    ) -> None:
        max_words = int(parameters.get("max_reply_length", 20) or 20)
        formality = float(parameters.get("formality", 0.2) or 0.2)
        directness = float(parameters.get("directness", 0.7) or 0.7)
        slang = float(parameters.get("slang_level", 0.6) or 0.6)
        risk_tolerance = float(parameters.get("risk_tolerance", 0.0) or 0.0)
        if risk_tolerance >= 0.2 and relationship_type in {"close_friend", "casual_friend", "romantic_interest"}:
            def bold_review_score(candidate: DraftCandidate) -> float:
                normalized = normalize_text(candidate.text)
                score = 0.0
                if any(term in normalized for term in ("see u", "see you", "miss", "want", "come", "after this")):
                    score += 0.35
                if any(term in normalized for term in ("nah", "icl", "ugh", "wait so", "how long")):
                    score += 0.2
                if "ugh" in normalized:
                    score += 0.12
                if any(term in normalized for term in ("supervised", "clash", "exam", "paper", "stuck")):
                    score += 0.25
                if len(normalized.split()) >= 7:
                    score += 0.1
                return score

            bundle.reply_candidates.sort(
                key=lambda candidate: (
                    candidate.final_decision == "reject",
                    -bold_review_score(candidate),
                    -float(candidate.critic_scores.get("final_confidence", 0.0) or 0.0),
                )
            )
            bundle.recommended_reply_index = 0 if bundle.reply_candidates else None
        for candidate in bundle.reply_candidates:
            words = candidate.text.split()
            if len(words) > max_words:
                candidate.text = " ".join(words[:max_words])
                candidate.sequence = [candidate.text]
                candidate.risk_flags = list(dict.fromkeys([*candidate.risk_flags, "training_length_capped"]))
            candidate.critic_scores["training_parameters"] = parameters
            candidate.estimated_risk_score = max(candidate.estimated_risk_score, int(candidate.critic_scores.get("risk_score", 0) or 0))
            if relationship_type in {"professional", "university"} and formality >= 0.7:
                candidate.why_this_matches = (candidate.why_this_matches + "; formal training preference").strip("; ")
            if relationship_type in {"close_friend", "casual_friend"} and slang >= 0.7:
                candidate.why_this_matches = (candidate.why_this_matches + "; casual training preference").strip("; ")
            if directness >= 0.8:
                candidate.candidate_type = "direct"
            if risk_tolerance > 0.0 and relationship_type in {"unknown", "professional", "family", "romantic_interest", "university"}:
                candidate.auto_send_allowed = False
                candidate.final_decision = "review"
                candidate.blocked_reason = candidate.blocked_reason or "hard_safety_gate_not_overridable"
        bundle.reply_suggestions = [candidate.text for candidate in bundle.reply_candidates]
        bundle.reply_sequences = [candidate.sequence for candidate in bundle.reply_candidates]
        bundle.final_decision = bundle.reply_candidates[0].final_decision if bundle.reply_candidates else "review"
        bundle.blocked_reason = bundle.reply_candidates[0].blocked_reason if bundle.reply_candidates else bundle.blocked_reason

    def _reason_from_feedback_label(self, label: str) -> str:
        return {
            "too_formal": "too formal",
            "too_long": "too long",
            "not_me": "not me",
            "missed_context": "missed context",
            "risky": "risky",
            "too_flirty": "risky",
            "too_dry": "not me",
            "approved": "other",
            "edited": "other",
            "rejected": "other",
        }.get(label, "other")

    def _append_jsonl(self, path: Path, row: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    def _write_jsonl(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        tmp_path.replace(path)

    def _read_jsonl_tail(self, path: Path, *, limit: int) -> list[dict[str, object]]:
        if not path.exists():
            return []
        rows: list[dict[str, object]] = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows[-limit:]

    def _vector_index_count(self) -> int:
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(self.settings.data_dir / "vector_db" / "reply_examples_chroma"))
            return int(client.get_collection("reply_examples").count())
        except Exception:
            fallback = self.settings.data_dir / "vector_db" / "reply_examples_chroma" / "reply_examples.jsonl"
            if not fallback.exists():
                return 0
            return sum(1 for line in fallback.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())

    def _vector_memory_type_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        vector_dir = self.settings.data_dir / "vector_db" / "reply_examples_chroma"
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(vector_dir))
            result = client.get_collection("reply_examples").get(include=["metadatas"])
            for metadata in result.get("metadatas", []) or []:
                if not isinstance(metadata, dict):
                    continue
                counts[str(metadata.get("memory_type") or "reply_example")] += 1
            if counts:
                return dict(counts)
        except Exception:
            pass
        fallback = vector_dir / "reply_examples.jsonl"
        if fallback.exists():
            for line in fallback.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                metadata = row.get("metadata") if isinstance(row, dict) else {}
                if not isinstance(metadata, dict):
                    metadata = row if isinstance(row, dict) else {}
                counts[str(metadata.get("memory_type") or "reply_example")] += 1
        return dict(counts)

    def _load_vector_map_rows(self, vector_dir: Path, limit: int) -> tuple[list[dict[str, object]], list[list[float]], str]:
        points: list[dict[str, object]] = []
        embeddings: list[list[float]] = []
        if vector_dir.exists():
            try:
                import chromadb

                client = chromadb.PersistentClient(path=str(vector_dir))
                collection = client.get_collection("reply_examples")
                result = collection.get(
                    limit=limit,
                    include=["embeddings", "metadatas", "documents"],
                )
                ids = result.get("ids", []) or []
                metadatas = result.get("metadatas", []) or []
                documents = result.get("documents", []) or []
                raw_embeddings = result.get("embeddings", []) or []
                for index, item_id in enumerate(ids):
                    metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
                    document = str(documents[index] or "") if index < len(documents) else ""
                    point = self._vector_map_point(item_id, metadata, document)
                    if point is None:
                        continue
                    points.append(point)
                    if index < len(raw_embeddings):
                        embedding = raw_embeddings[index]
                        if hasattr(embedding, "tolist"):
                            embedding = embedding.tolist()
                        if isinstance(embedding, list):
                            embeddings.append([float(value) for value in embedding])
                if points:
                    return points, embeddings if len(embeddings) == len(points) else [], "chroma"
            except Exception:
                pass

        fallback = vector_dir / "reply_examples.jsonl"
        if fallback.exists():
            for line in fallback.read_text(encoding="utf-8", errors="ignore").splitlines()[:limit]:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                metadata = dict(row.get("metadata") or row)
                point = self._vector_map_point(str(metadata.get("id") or row.get("id") or ""), metadata, "")
                if point is not None:
                    points.append(point)
            if points:
                return points, [], "fallback_jsonl"
        return [], [], "empty"

    def _vector_map_point(self, item_id: object, metadata: dict[str, object], document: str) -> dict[str, object] | None:
        if metadata.get("is_quarantined") or metadata.get("unsafe"):
            return None
        incoming = str(metadata.get("incoming") or "")
        reply = str(metadata.get("my_reply") or metadata.get("reply") or "")
        if document and (not incoming or not reply):
            for part in document.split(" || "):
                if part.startswith("incoming: ") and not incoming:
                    incoming = part[len("incoming: ") :]
                if part.startswith("reply: ") and not reply:
                    reply = part[len("reply: ") :]
        source = str(metadata.get("_source") or metadata.get("source") or "training")
        memory_type = str(metadata.get("memory_type") or "reply_example")
        is_correction = bool(metadata.get("is_correction")) or source == "correction"
        is_synthetic = bool(metadata.get("is_synthetic")) or bool(metadata.get("is_template")) or source == "generated_safe_template"
        authority = "high" if is_correction else ("low" if is_synthetic else str(metadata.get("style_authority") or "medium"))
        point_type = "correction" if is_correction else ("synthetic" if is_synthetic else "real")
        relationship = normalize_relationship_type(str(metadata.get("relationship_type", "unknown")))
        intent = str(metadata.get("intent_type") or "unknown")
        return {
            "id": str(item_id or metadata.get("id") or hashlib.sha1(f"{incoming}|{reply}".encode("utf-8")).hexdigest()[:12]),
            "incoming": redact_private_text(incoming)[:180],
            "my_reply": redact_private_text(reply)[:180],
            "relationship_type": relationship,
            "intent_type": intent,
            "cluster_key": f"{intent}|{relationship}",
            "contact_name": redact_private_text(str(metadata.get("contact_name") or ""))[:80],
            "source": source,
            "memory_type": memory_type,
            "summary": redact_private_text(str(metadata.get("summary") or ""))[:240],
            "point_type": point_type,
            "style_authority": authority,
            "is_correction": is_correction,
            "is_synthetic": is_synthetic,
        }

    def _project_vector_points(self, points: list[dict[str, object]], embeddings: list[list[float]]) -> list[dict[str, object]]:
        coords = self._pca_3d(embeddings) if embeddings else []
        projected: list[dict[str, object]] = []
        for index, point in enumerate(points):
            if index < len(coords):
                x, y, z = coords[index]
            else:
                x, y, z = self._cluster_coord(point, index)
            projected.append({**point, "x": float(x), "y": float(y), "z": float(z)})
        projected = self._spread_vector_clusters(projected)
        return [
            {**point, "x": round(float(point["x"]), 4), "y": round(float(point["y"]), 4), "z": round(float(point["z"]), 4)}
            for point in projected
        ]
        return projected

    def _pca_3d(self, embeddings: list[list[float]]) -> list[tuple[float, float, float]]:
        if len(embeddings) < 3:
            return []
        try:
            import numpy as np
        except Exception:
            return []
        try:
            matrix = np.array(embeddings, dtype=float)
            matrix = matrix - matrix.mean(axis=0)
            _, _, vh = np.linalg.svd(matrix, full_matrices=False)
            projected = matrix @ vh[:3].T
            scale = float(np.max(np.abs(projected))) or 1.0
            projected = projected / scale * 110.0
            return [(float(row[0]), float(row[1]), float(row[2] if projected.shape[1] > 2 else 0.0)) for row in projected]
        except Exception:
            return []

    def _cluster_coord(self, point: dict[str, object], index: int) -> tuple[float, float, float]:
        relationship = str(point.get("relationship_type") or "unknown")
        intent = str(point.get("intent_type") or "unknown")
        key = f"{relationship}|{intent}|{point.get('id')}"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        relationship_names = ["close_friend", "casual_friend", "family", "university", "professional", "unknown"]
        intent_hash = int(hashlib.sha1(intent.encode("utf-8")).hexdigest()[:6], 16)
        rel_index = relationship_names.index(relationship) if relationship in relationship_names else len(relationship_names) - 1
        angle = (rel_index / len(relationship_names)) * math.tau
        ring = 58.0 + (intent_hash % 58)
        jitter = [(int(digest[i : i + 2], 16) / 255.0 - 0.5) * 30.0 for i in range(0, 6, 2)]
        height = ((intent_hash % 100) / 100.0 - 0.5) * 92.0
        return (
            math.cos(angle) * ring + jitter[0],
            height + jitter[1],
            math.sin(angle) * ring + jitter[2] + (index % 11 - 5) * 1.7,
        )

    def _spread_vector_clusters(self, points: list[dict[str, object]]) -> list[dict[str, object]]:
        if not points:
            return []
        clusters: dict[str, list[dict[str, object]]] = {}
        for point in points:
            clusters.setdefault(str(point.get("cluster_key") or point.get("intent_type") or "unknown"), []).append(point)

        spread = 2.65
        separated: list[dict[str, object]] = []
        cluster_keys = sorted(clusters)
        for cluster_index, key in enumerate(cluster_keys):
            cluster = clusters[key]
            cx = sum(float(point["x"]) for point in cluster) / len(cluster)
            cy = sum(float(point["y"]) for point in cluster) / len(cluster)
            cz = sum(float(point["z"]) for point in cluster) / len(cluster)
            angle = (cluster_index / max(1, len(cluster_keys))) * math.tau
            orbit = 155.0 + (cluster_index % 5) * 28.0
            layer = ((cluster_index % 7) - 3) * 24.0
            offset = (math.cos(angle) * orbit, layer, math.sin(angle) * orbit)
            for local_index, point in enumerate(cluster):
                digest = hashlib.sha1(f"{key}|{point.get('id')}|{local_index}".encode("utf-8")).hexdigest()
                micro = [(int(digest[i : i + 2], 16) / 255.0 - 0.5) * 18.0 for i in range(0, 6, 2)]
                shell_angle = local_index * 2.399963229728653
                shell_radius = min(135.0, math.sqrt(local_index + 1) * (3.0 + min(len(cluster), 700) / 260.0))
                shell_height = ((local_index % 17) - 8) * (0.8 + min(len(cluster), 900) / 900.0)
                separated.append(
                    {
                        **point,
                        "x": (float(point["x"]) - cx) * spread + offset[0] + micro[0] + math.cos(shell_angle) * shell_radius,
                        "y": (float(point["y"]) - cy) * spread + offset[1] + micro[1] + shell_height,
                        "z": (float(point["z"]) - cz) * spread + offset[2] + micro[2] + math.sin(shell_angle) * shell_radius,
                    }
                )
        return separated

    def _vector_map_clusters(self, points: list[dict[str, object]]) -> list[dict[str, object]]:
        clusters: dict[str, list[dict[str, object]]] = {}
        for point in points:
            clusters.setdefault(str(point.get("cluster_key") or "unknown"), []).append(point)
        summaries: list[dict[str, object]] = []
        for key, items in clusters.items():
            if not items:
                continue
            summaries.append(
                {
                    "key": key,
                    "label": key.replace("|", " / "),
                    "count": len(items),
                    "intent_type": items[0].get("intent_type", "unknown"),
                    "relationship_type": items[0].get("relationship_type", "unknown"),
                    "x": round(sum(float(point["x"]) for point in items) / len(items), 4),
                    "y": round(sum(float(point["y"]) for point in items) / len(items), 4),
                    "z": round(sum(float(point["z"]) for point in items) / len(items), 4),
                    "corrections": sum(1 for point in items if point.get("is_correction")),
                    "synthetic": sum(1 for point in items if point.get("is_synthetic")),
                }
            )
        return sorted(summaries, key=lambda item: int(item["count"]), reverse=True)

    def _add_training_feedback_to_vector_db(self, row: dict[str, object]) -> dict[str, object]:
        try:
            import chromadb

            from libs.drafting.embeddings import embed_example, embed_example_text

            index_dir = self.settings.data_dir / "vector_db" / "reply_examples_chroma"
            client = chromadb.PersistentClient(path=str(index_dir))
            collection = client.get_or_create_collection("reply_examples")
            metadata = {
                "id": str(row["id"]),
                "contact_name": str(row.get("contact_name", "")),
                "relationship_type": str(row.get("relationship_type", "unknown")),
                "intent_type": str(row.get("intent_type", "unknown")),
                "incoming": str(row.get("incoming", "")),
                "my_reply": str(row.get("my_reply", "")),
                "source_file": "data/training_messages/corrections.jsonl",
                "is_correction": True,
                "is_synthetic": False,
                "is_template": False,
                "is_quarantined": False,
                "unsafe": False,
                "_source": "correction",
                "source": str(row.get("source") or "training_page_feedback"),
                "style_authority": str(row.get("style_authority") or "high"),
            }
            collection.add(
                ids=[str(row["id"])],
                embeddings=[embed_example(row)],
                documents=[embed_example_text(row)],
                metadatas=[metadata],
            )
            return {"vector_rebuild_required": False, "incremental_added": True}
        except Exception as exc:
            return self._mark_vector_rebuild_required(str(exc))

    def _mark_vector_rebuild_required(self, reason: str) -> dict[str, object]:
        self.vector_rebuild_marker_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_rebuild_marker_path.write_text(
            json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "reason": reason}, indent=2),
            encoding="utf-8",
        )
        return {"vector_rebuild_required": True, "incremental_added": False, "vector_update_error": reason}

    async def continuous_automation_loop(self, duration_seconds: float = 300.0, interval_seconds: float = 5.0):
        """Run continuous automation loop for specified duration."""
        import asyncio

        start_time = time.time()
        end_time = start_time + duration_seconds

        while time.time() < end_time and not self._emergency_stop:
            try:
                # Scan inbox for unread
                if self._current_automation_mode in ["auto-draft", "auto-send"]:
                    self.scan_inbox()

                    # Process first unread thread if any
                    if self._unread_queue:
                        contact_name = self._unread_queue[0].contact_name
                        self.open_thread(contact_name)
                        self.scroll_for_context()

                        # Auto-generate draft
                        if self._current_automation_mode in ["auto-draft", "auto-send"]:
                            self.auto_generate_draft()

                            # Auto-send if confidence is high enough
                            if self._current_automation_mode == "auto-send":
                                self.auto_send_reply()
                                time.sleep(2.0)  # Wait for message to send
                        else:
                            time.sleep(1.0)

                # Wait before next iteration
                await asyncio.sleep(interval_seconds)

            except Exception as e:
                self._last_action = "Continuous Loop Error"
                self._last_action_result = str(e)
                await asyncio.sleep(interval_seconds)
                continue

    def process_queue_item(self) -> ControllerState:
        """Process the next item in the unread queue."""
        state = self.latest_state or self._capture_state(record_log=True)

        if not self._unread_queue:
            self._last_action = "Process Queue"
            self._last_action_result = "Queue is empty"
            state.last_action = self._last_action
            state.last_action_result = self._last_action_result
            return state

        next_item = self._unread_queue.pop(0)
        contact_name = next_item.contact_name
        self.open_thread(contact_name)
        self.scroll_for_context()
        if self._current_automation_mode in ["auto-draft", "auto-send"]:
            self.auto_generate_draft()
            if self._current_automation_mode == "auto-send":
                self.auto_send_reply()

        self._last_action = "Process Queue"
        self._last_action_result = f"Processed {contact_name}"
        state.last_action = self._last_action
        state.last_action_result = self._last_action_result
        self.latest_state = state
        return state

    def _ensure_messages_inbox_state(self) -> ControllerState:
        state = self.latest_state or self._capture_state(record_log=True)
        if state.classification.package_name != "com.google.android.apps.messaging":
            self.adb.launch_app("com.google.android.apps.messaging")
            time.sleep(0.8)
            state = self._capture_state(record_log=True)
        for _ in range(3):
            if state.classification.package_name != "com.google.android.apps.messaging":
                break
            if state.classification.screen.value == "app_inbox":
                return state
            self.adb.back()
            time.sleep(0.35)
            state = self._capture_state(record_log=True)
        return state

    def _read_messages_hierarchy(self) -> str | None:
        try:
            return self.adb.dump_ui_hierarchy()
        except RuntimeError:
            return None

    def _extract_inbox_threads_from_visible_text(self, visible_text: list[str]) -> list[QueueItem]:
        items: list[QueueItem] = []
        for line in visible_text:
            cleaned = " ".join(line.split()).strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in {"messages", "archived", "search", "start chat"}:
                continue
            if len(cleaned) < 2 or len(cleaned) > 80:
                continue
            items.append(
                QueueItem(
                    contact_name=cleaned,
                    preview=f"Visible inbox thread: {cleaned}",
                    unread=True,
                    timestamp=datetime.now(timezone.utc),
                )
            )
        return items

    def _extract_inbox_threads_from_hierarchy(self, hierarchy: str) -> list[QueueItem]:
        root = self._parse_hierarchy(hierarchy)
        if root is None:
            return []
        items: list[QueueItem] = []
        for node in root.iter("node"):
            if node.attrib.get("clickable") != "true":
                continue
            bounds = self._parse_bounds(node.attrib.get("bounds"))
            if bounds is None:
                continue
            left, top, right, bottom = bounds
            if top < 180 or bottom - top < 90:
                continue
            texts = self._collect_descendant_texts(node)
            if not texts:
                continue
            contact_name = texts[0]
            if not self._looks_like_contact_name(contact_name):
                continue
            if self._normalize_contact_name(contact_name) in self._blacklist:
                continue
            preview = " ".join(texts[1:3]).strip() or f"Visible inbox thread: {contact_name}"
            unread = self._is_unread_thread_node(node, texts)
            items.append(
                QueueItem(
                    contact_name=contact_name,
                    contact_number=f"{left},{top},{right},{bottom}",
                    preview=preview,
                    unread=unread,
                    timestamp=datetime.now(timezone.utc),
                )
            )
        return items

    def _extract_thread_message_entries_from_hierarchy(self, hierarchy: str | None) -> list[dict[str, str]]:
        if not hierarchy:
            return []
        root = self._parse_hierarchy(hierarchy)
        if root is None:
            return []
        entries: list[tuple[int, dict[str, str]]] = []
        for node in root.iter("node"):
            if node.attrib.get("resource-id") != "message_text":
                continue
            text = html.unescape(node.attrib.get("text", "")).strip()
            if not text:
                continue
            content_desc = html.unescape(node.attrib.get("content-desc", ""))
            speaker = "me" if content_desc.lower().startswith("you said") else "other"
            bounds = self._parse_bounds(node.attrib.get("bounds"))
            top = bounds[1] if bounds is not None else len(entries)
            entries.append((top, {"speaker": speaker, "text": text}))
        entries.sort(key=lambda item: item[0])
        return [entry for _, entry in entries]

    def _thread_contact_name_from_state(self, state: ControllerState) -> str | None:
        if state.thread_context and state.thread_context.contact_name:
            return state.thread_context.contact_name
        hierarchy = self._read_messages_hierarchy()
        return self._thread_contact_name_from_hierarchy(hierarchy)

    def _thread_contact_name_from_hierarchy(self, hierarchy: str | None) -> str | None:
        if not hierarchy:
            return None
        root = self._parse_hierarchy(hierarchy)
        if root is None:
            return None
        for node in root.iter("node"):
            text = html.unescape(node.attrib.get("text", "")).strip()
            bounds = self._parse_bounds(node.attrib.get("bounds"))
            if not text or bounds is None:
                continue
            left, top, right, bottom = bounds
            if top <= 260 and right - left >= 180 and node.attrib.get("class") == "android.widget.TextView":
                if self._looks_like_contact_name(text):
                    return text
        return None

    def _merge_conversation_entries(
        self,
        existing_entries: list[dict[str, str]],
        new_entries: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        normalized_existing = self._normalize_conversation_entries(existing_entries)
        normalized_new = self._normalize_conversation_entries(new_entries)
        if not normalized_existing:
            return normalized_new[-self.settings.context_max_messages :]
        if not normalized_new:
            return normalized_existing[-self.settings.context_max_messages :]

        append_overlap = self._sequence_overlap(
            left=normalized_existing,
            right=normalized_new,
        )
        prepend_overlap = self._sequence_overlap(
            left=normalized_new,
            right=normalized_existing,
        )
        if prepend_overlap > append_overlap and prepend_overlap > 0:
            merged = normalized_new + normalized_existing[prepend_overlap:]
        elif append_overlap > 0:
            merged = normalized_existing + normalized_new[append_overlap:]
        else:
            merged = normalized_existing[:]
            seen = {(entry["speaker"], entry["text"]) for entry in merged}
            for entry in normalized_new:
                key = (entry["speaker"], entry["text"])
                if key in seen:
                    continue
                seen.add(key)
                merged.append(entry)
        return merged[-self.settings.context_max_messages :]

    def _top_context_key(self, state: ControllerState) -> tuple[str, str] | None:
        if not state.thread_context or not state.thread_context.full_conversation:
            return None
        first = state.thread_context.full_conversation[0]
        return (first.get("speaker", "other"), self._normalize_text(first.get("text", "")))

    def _set_context_debug_fields(self, state: ControllerState) -> None:
        if not state.thread_context:
            state.context_messages_count = 0
            state.scrolls_performed = 0
            state.context_stopped_reason = None
            return
        state.context_messages_count = state.thread_context.message_count
        state.scrolls_performed = state.thread_context.scrolls_performed
        state.context_stopped_reason = state.thread_context.stopped_reason

    def _merge_perception_timings(
        self,
        state: ControllerState,
        metrics: LoopMetrics,
        *,
        timing_marks: dict[str, float] | None = None,
    ) -> None:
        timings = dict(state.timings_ms)
        timings.setdefault("screenshot_capture", round(metrics.screenshot_time_ms, 2))
        timings.setdefault("ocr", round(metrics.ocr_time_ms, 2))
        timings.setdefault("classification", round(metrics.classification_time_ms, 2))
        timings.setdefault("planner", round(metrics.planner_time_ms, 2))
        timings.setdefault("server_processing_ms", round(metrics.loop_time_ms, 2))
        timings.setdefault("context_capture", 0.0)
        timings.setdefault("execution", round(metrics.execution_time_ms, 2))
        timings.setdefault("sleep_ms", 0.0)
        for key, value in metrics.detailed_timings_ms.items():
            timings.setdefault(key, round(value, 2))
        for key, value in (timing_marks or {}).items():
            timings[key] = round(value, 2)
        state.timings_ms = timings

    def _finalize_timing_summary(self, state: ControllerState) -> None:
        timings = dict(state.timings_ms)
        server_processing = float(timings.get("server_processing_ms", state.metrics.loop_time_ms if state.metrics else 0.0))
        known_keys = {
            "screenshot_capture",
            "device_check",
            "screenshot_decode",
            "ocr",
            "visual_features",
            "classification",
            "planner",
            "policy_eval",
            "state_model_create",
            "context_hierarchy_sync",
            "context_aware_redraft",
            "latest_alias_copy",
            "log_write",
            "execution",
            "sleep_ms",
        }
        known_total = sum(float(timings.get(key, 0.0)) for key in known_keys)
        timings["server_processing_ms"] = round(server_processing, 2)
        timings["adb_total_ms"] = round(float(timings.get("adb_total", 0.0)), 2)
        timings["sleep_ms"] = round(float(timings.get("sleep_ms", 0.0)), 2)
        timings["unaccounted_ms"] = round(max(0.0, server_processing - known_total), 2)
        ignored_for_slowest = {"request_start", "capture_start", "capture_end", "planner_start", "planner_end", "server_processing_ms", "unaccounted_ms"}
        stages = {
            key: float(value)
            for key, value in timings.items()
            if key not in ignored_for_slowest and isinstance(value, (int, float))
        }
        if stages:
            slowest_key, slowest_value = max(stages.items(), key=lambda item: item[1])
            timings["slowest_stage"] = f"{slowest_key}:{slowest_value:.2f}ms"
        state.timings_ms = timings
        if state.metrics is not None:
            state.metrics.detailed_timings_ms = {
                key: float(value)
                for key, value in timings.items()
                if isinstance(value, (int, float))
            }
        self._last_debug_timings_ms = timings

    def _sequence_overlap(self, left: list[dict[str, str]], right: list[dict[str, str]]) -> int:
        max_overlap = min(len(left), len(right))
        for size in range(max_overlap, 0, -1):
            if left[-size:] == right[:size]:
                return size
        return 0

    def _dedupe_queue_items(self, items: list[QueueItem]) -> list[QueueItem]:
        deduped: list[QueueItem] = []
        seen: set[str] = set()
        for item in items:
            key = self._normalize_contact_name(item.contact_name)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _normalize_conversation_entries(self, entries: list[dict[str, str]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for entry in entries:
            speaker = entry.get("speaker", "other")
            text = " ".join(entry.get("text", "").split()).strip()
            if not text:
                continue
            normalized.append({"speaker": speaker, "text": text})
        return normalized

    def _parse_hierarchy(self, hierarchy: str | None) -> ET.Element | None:
        if not hierarchy:
            return None
        try:
            return ET.fromstring(hierarchy)
        except ET.ParseError:
            return None

    def _parse_bounds(self, bounds: str | None) -> tuple[int, int, int, int] | None:
        if not bounds:
            return None
        match = re.search(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds)
        if not match:
            return None
        return tuple(int(group) for group in match.groups())

    def _collect_descendant_texts(self, node: ET.Element) -> list[str]:
        texts: list[str] = []
        ignored = {"messages", "search", "archived", "back", "more", "compose", "start chat"}
        for descendant in node.iter("node"):
            text = html.unescape(descendant.attrib.get("text", "")).strip()
            if not text:
                continue
            lowered = text.lower()
            if lowered in ignored:
                continue
            if text not in texts:
                texts.append(text)
        return texts

    def _collect_descendant_content(self, node: ET.Element) -> str:
        parts: list[str] = []
        for descendant in node.iter("node"):
            content = html.unescape(descendant.attrib.get("content-desc", "")).strip()
            if content:
                parts.append(content)
        return " ".join(parts)

    def _is_unread_thread_node(self, node: ET.Element, texts: list[str]) -> bool:
        content = self._collect_descendant_content(node).lower()
        if any(token in content for token in ("unread", "new message", "new messages")):
            return True

        unread_badges = 0
        for text in texts[1:]:
            cleaned = text.strip()
            if re.fullmatch(r"\d+", cleaned):
                unread_badges += 1
                continue
            if re.search(r"\b(unread|new)\b", cleaned.lower()):
                unread_badges += 1
        return unread_badges > 0

    def _normalize_contact_name(self, text: str | None) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    def _contact_matches(self, candidate: str, target: str) -> bool:
        normalized_candidate = self._normalize_contact_name(candidate)
        normalized_target = self._normalize_contact_name(target)
        return (
            normalized_candidate == normalized_target
            or normalized_candidate in normalized_target
            or normalized_target in normalized_candidate
        )

    def _looks_like_contact_name(self, text: str) -> bool:
        cleaned = " ".join(text.split()).strip()
        if not cleaned:
            return False
        lowered = cleaned.lower()
        if lowered in {"messages", "search", "archived", "start chat", "you", "rcs message"}:
            return False
        return 1 < len(cleaned) <= 80

    @property
    def latest_screenshot_path(self) -> Path | None:
        if self.latest_state is None:
            return None
        return Path(self.latest_state.screenshot_path)

    def refresh_state(self) -> ControllerState:
        try:
            state = self._capture_state(record_log=True)
        except Exception as exc:
            state = self._offline_state(str(exc))
        self.latest_state = state
        return state

    def reject(self) -> ControllerState:
        state = self.latest_state or self.refresh_state()
        self._last_action = "Reject"
        self._last_action_result = "No action executed"
        state.last_action = self._last_action
        state.last_action_result = self._last_action_result
        self.log_store.record_event(
            event_type="reject",
            detected_screen=state.classification.screen.value,
            confidence=state.classification.confidence,
            planner_decision=state.planner_decision.reason,
            approval_decision="rejected",
            execution_result="no action executed",
            before_screenshot_path=state.screenshot_path,
            metadata={"summary": state.summary, "metrics": state.metrics.model_dump(mode="json") if state.metrics else {}},
        )
        return state

    def emergency_stop(self) -> ControllerState:
        self._emergency_stop = True
        state = self.latest_state or self._capture_state(record_log=False)
        self._last_action = "Emergency Stop"
        self._last_action_result = "Blocked"
        state.last_action = self._last_action
        state.last_action_result = self._last_action_result
        state.emergency_stop = True
        state.halted = True
        state.halt_reason = "Emergency stop engaged."
        self.latest_state = state
        self.log_store.record_event(
            event_type="emergency_stop",
            detected_screen=state.classification.screen.value,
            confidence=state.classification.confidence,
            planner_decision=state.planner_decision.reason,
            approval_decision="manual stop",
            execution_result="blocked",
            before_screenshot_path=state.screenshot_path,
        )
        return state

    def handle_notification(self, payload: NotificationPayload) -> None:
        self.log_store.record_event(
            event_type="notification",
            detected_screen=self.latest_state.classification.screen.value if self.latest_state else "unknown",
            confidence=self.latest_state.classification.confidence if self.latest_state else 0.0,
            planner_decision="notification_ingest",
            approval_decision="n/a",
            execution_result="received",
            metadata=payload.model_dump(mode="json"),
        )

    def approve_type_only(self, suggestion_index: int | None) -> ControllerState:
        state = self.refresh_state()
        suggestion_index = 0 if suggestion_index is None else suggestion_index
        if suggestion_index < 0 or suggestion_index >= len(state.reply_suggestions):
            raise HTTPException(status_code=400, detail="Invalid suggestion index.")
        sequence = self._selected_reply_sequence(state, suggestion_index)
        if not sequence:
            raise HTTPException(status_code=409, detail="Selected draft has no message content.")
        draft_text = sequence[0]
        plan = self.planner.build_plan(
            classification=state.classification,
            action_type=ApprovalActionType.TYPE_DRAFT,
            draft_text=draft_text,
        )
        return self._execute_approved_plan(plan=plan, approval_note=f"type suggestion {suggestion_index}")

    def approve_ai_draft_to_compose(self, suggestion_index: int | None = None) -> ControllerState:
        return self.approve_type_only(suggestion_index)

    def _apply_draft_bundle(self, state: ControllerState, bundle: DraftBundle) -> None:
        state.summary = bundle.summary
        state.reply_suggestions = bundle.reply_suggestions
        state.reply_sequences = bundle.reply_sequences
        state.draft_candidates = [candidate.model_copy(deep=True) for candidate in bundle.reply_candidates]
        state.recommended_reply_index = bundle.recommended_reply_index
        state.auto_send_blocked_reason = bundle.auto_send_blocked_reason
        state.relationship_type = bundle.relationship_type
        state.intent_type = bundle.intent_type
        state.retrieved_examples_count = bundle.retrieved_examples_count
        state.retrieved_examples = bundle.retrieved_examples
        state.final_decision = bundle.final_decision
        state.blocked_reason = bundle.blocked_reason
        state.timings_ms = dict(bundle.timings_ms)
        state.prompt_preview = bundle.prompt_preview
        state.approved_photo_suggestions = bundle.photo_suggestions
        self._set_context_debug_fields(state)
        self._enrich_draft_candidates_with_runtime_confidence(state)

    def _enrich_draft_candidates_with_runtime_confidence(self, state: ControllerState) -> None:
        if not state.draft_candidates:
            state.recommended_reply_index = None
            return
        screen_confidence = max(0.0, min(1.0, state.classification.confidence))
        context_confidence = self._runtime_context_confidence(state)
        for candidate in state.draft_candidates:
            candidate.score_breakdown.screen_confidence = round(screen_confidence, 3)
            candidate.score_breakdown.context_confidence = round(
                max(candidate.score_breakdown.context_confidence, context_confidence),
                3,
            )
            final_confidence = (
                0.20 * candidate.score_breakdown.screen_confidence
                + 0.20 * candidate.score_breakdown.context_confidence
                + 0.15 * candidate.score_breakdown.persona_confidence
                + 0.20 * candidate.score_breakdown.style_confidence
                + 0.15 * candidate.score_breakdown.semantic_confidence
                + 0.10 * candidate.score_breakdown.action_safety
            ) - (0.18 * candidate.score_breakdown.risk_score)
            candidate.score_breakdown.final_confidence = round(max(0.0, min(final_confidence, 1.0)), 3)
            candidate.auto_send_allowed = (
                candidate.auto_send_allowed
                and candidate.final_decision == "send"
                and candidate.score_breakdown.final_confidence >= 0.78
                and not candidate.risk_flags
            )
        state.draft_candidates.sort(
            key=lambda candidate: (
                candidate.auto_send_allowed,
                candidate.score_breakdown.final_confidence,
                -candidate.score_breakdown.risk_score,
            ),
            reverse=True,
        )
        state.reply_suggestions = [candidate.text for candidate in state.draft_candidates]
        state.reply_sequences = [candidate.sequence for candidate in state.draft_candidates]
        state.recommended_reply_index = 0 if state.draft_candidates else None
        if state.draft_candidates and not state.draft_candidates[0].auto_send_allowed:
            state.auto_send_blocked_reason = (
                state.draft_candidates[0].blocked_reason
                or "Auto-send blocked: " + ", ".join(state.draft_candidates[0].risk_flags)
                if state.draft_candidates[0].risk_flags
                else "Auto-send blocked: low final confidence."
            )
            state.final_decision = state.draft_candidates[0].final_decision
            state.blocked_reason = state.auto_send_blocked_reason
        elif state.draft_candidates:
            state.auto_send_blocked_reason = None
            state.final_decision = state.draft_candidates[0].final_decision
            state.blocked_reason = state.draft_candidates[0].blocked_reason

    def _runtime_context_confidence(self, state: ControllerState) -> float:
        if state.thread_context is None:
            return 0.48
        message_count = state.thread_context.message_count
        if message_count >= 12:
            return 0.96
        if message_count >= 8:
            return 0.88
        if message_count >= 4:
            return 0.76
        return 0.62

    def _recommended_candidate(self, state: ControllerState | None) -> DraftCandidate | None:
        if state is None or state.recommended_reply_index is None:
            return None
        if state.recommended_reply_index >= len(state.draft_candidates):
            return None
        return state.draft_candidates[state.recommended_reply_index]

    def _recommended_reply_confidence(self, state: ControllerState | None) -> float:
        candidate = self._recommended_candidate(state)
        if candidate is None:
            return 0.0
        return candidate.score_breakdown.final_confidence

    def _type_custom_reply(self, draft_text: str) -> ControllerState:
        state = self.refresh_state()
        plan = self.planner.build_plan(
            classification=state.classification,
            action_type=ApprovalActionType.TYPE_DRAFT,
            draft_text=draft_text,
        )
        return self._execute_approved_plan(plan=plan, approval_note="type custom reply")

    def _selected_reply_sequence(self, state: ControllerState, suggestion_index: int) -> list[str]:
        if suggestion_index < len(state.reply_sequences) and state.reply_sequences[suggestion_index]:
            return state.reply_sequences[suggestion_index]
        if suggestion_index < len(state.reply_suggestions):
            return [state.reply_suggestions[suggestion_index]]
        return []

    def approve_send_current_draft(self) -> ControllerState:
        state = self.refresh_state()
        compose_text = (state.compose_text or "").strip()
        if not compose_text:
            raise HTTPException(status_code=409, detail="No draft text is currently in the compose box.")
        plan = self.planner.build_plan(
            classification=state.classification,
            action_type=ApprovalActionType.SEND_MESSAGE,
        )
        plan.allowed_package_names = [state.classification.package_name]
        plan.postcondition_text = compose_text
        plan.timeout_ms = 5000
        plan.retry = 1
        return self._execute_approved_plan(plan=plan, approval_note="send current draft")

    def approve_open_gallery(self) -> ControllerState:
        state = self.refresh_state()
        plan = self.planner.build_plan(
            classification=state.classification,
            action_type=ApprovalActionType.OPEN_GALLERY,
        )
        return self._execute_approved_plan(plan=plan, approval_note="open gallery")

    def approve_select_photo(self, photo_id: str | None) -> ControllerState:
        state = self.refresh_state()
        photo = self._resolve_photo(photo_id, state.approved_photo_suggestions)
        plan = self.planner.build_plan(
            classification=state.classification,
            action_type=ApprovalActionType.SELECT_APPROVED_PHOTO,
            approved_photo=photo,
        )
        return self._execute_approved_plan(plan=plan, approval_note=f"select photo {photo.photo_id}")

    def metrics_snapshot(self) -> MetricsSnapshot:
        history = self._metrics_history[-20:]
        average = (
            sum(metric.loop_time_ms for metric in history) / len(history)
            if history
            else 0.0
        )
        validation_total = len(self._compose_validation_runs)
        validation_success = sum(1 for item in self._compose_validation_runs if item.success)
        failure_counts: dict[str, int] = {}
        for item in self._compose_validation_runs:
            if item.failure_category is not None:
                key = item.failure_category.value
                failure_counts[key] = failure_counts.get(key, 0) + 1
        return MetricsSnapshot(
            total_iterations=len(self._metrics_history),
            halted_iterations=self._halted_iterations,
            last=self._metrics_history[-1] if self._metrics_history else None,
            rolling_average_ms=average,
            compose_validation={
                "total_runs": validation_total,
                "successful_runs": validation_success,
                "failed_runs": validation_total - validation_success,
                "success_rate": (validation_success / validation_total) if validation_total else 0.0,
                "keyboard_ambiguity_count": failure_counts.get(FailureCategory.KEYBOARD_STATE_AMBIGUOUS.value, 0),
                "failure_counts": failure_counts,
            },
        )

    def last_state(self) -> ControllerState | None:
        return self.latest_state

    def recent_logs(self) -> list[dict[str, object]]:
        return self.log_store.list_recent(limit=20)

    def _offline_state(self, reason: str) -> ControllerState:
        timestamp = datetime.now(timezone.utc)
        classification = ScreenClassification(
            app="Unknown",
            screen=ScreenName.UNKNOWN_SCREEN,
            confidence=0.0,
            visible_text=[],
            available_actions=[],
            package_name="unknown",
            activity_name="unknown",
            screenshot_width=0,
            screenshot_height=0,
            recent_messages=[],
            keyboard_visible=None,
            keyboard_height=None,
            keyboard_ambiguous=False,
            features_used=[],
            debug_info={"offline_reason": reason},
        )
        metrics = LoopMetrics(
            timestamp=timestamp,
            loop_time_ms=0.0,
            screenshot_time_ms=0.0,
            ocr_time_ms=0.0,
            classification_time_ms=0.0,
            planner_time_ms=0.0,
            execution_time_ms=0.0,
            screen=classification.screen.value,
            confidence=0.0,
        )
        state = ControllerState(
            captured_at=timestamp,
            classification=classification,
            planner_decision=PlannerDecision(status="review", reason=reason),
            summary="No live device state available.",
            compose_text=None,
            reply_suggestions=[],
            reply_sequences=[],
            draft_candidates=[],
            recommended_reply_index=None,
            auto_send_blocked_reason="No live device state available.",
            relationship_type="unknown",
            intent_type="unknown",
            retrieved_examples_count=0,
            retrieved_examples=[],
            final_decision="review",
            blocked_reason="No live device state available.",
            timings_ms={"offline": 0.0},
            prompt_preview=None,
            approved_photo_suggestions=[],
            halted=False,
            halt_reason=None,
            emergency_stop=False,
            screenshot_path=str(self.settings.screenshot_dir / "latest.png"),
            debug_image_path=None,
            metrics=metrics,
            last_action=self._last_action,
            last_action_result=reason,
            verification_errors=list(self._verification_errors[-8:]),
            failure_category=None,
            compose_validation=self._last_compose_validation_result,
            send_and_read=self._last_send_and_read_result,
            automation_mode=AutomationMode(
                enabled=self._automation_enabled,
                mode=self._current_automation_mode,
                confidence_threshold=self._automation_confidence,
                auto_send_enabled=self._auto_send_enabled,
            ),
            unread_queue=[item.model_copy() for item in self._unread_queue],
            thread_context=self.latest_state.thread_context.model_copy(deep=True) if self.latest_state and self.latest_state.thread_context else None,
            blacklisted=False,
        )
        return state

    def _public_media_url(self, raw_path: str | None) -> str | None:
        if not raw_path:
            return None
        path = Path(raw_path)
        if not path.name:
            return None
        try:
            resolved = path.resolve()
        except Exception:
            return None
        roots = {
            "screenshot": self.settings.screenshot_dir.resolve(),
            "debug": self.settings.debug_dir.resolve(),
        }
        for category, root in roots.items():
            candidate = resolved
            try:
                candidate.relative_to(root)
            except ValueError:
                candidate = (root / path.name).resolve()
                try:
                    candidate.relative_to(root)
                except ValueError:
                    continue
            if candidate.exists():
                return f"/api/media/{category}/{quote(candidate.name)}"
        return None

    def _state_payload(self, state: ControllerState | None = None) -> dict[str, Any] | None:
        state = state or self.latest_state
        if state is None:
            return None
        payload = state.model_dump(mode="json")
        payload["screenshot_url"] = self._public_media_url(state.screenshot_path)
        payload["debug_image_url"] = self._public_media_url(state.debug_image_path)
        return payload

    def _current_state_summary(self) -> dict[str, Any]:
        state = self.latest_state
        classification = state.classification if state is not None else None
        automation = state.automation_mode if state is not None else None
        candidate = self._recommended_candidate(state)
        return {
            "captured_at": state.captured_at.isoformat() if state else None,
            "connected": state is not None,
            "connection_status": "observed" if state else "unknown",
            "device": {
                "app": classification.app if classification else "Unknown",
                "package_name": classification.package_name if classification else None,
                "activity_name": classification.activity_name if classification else None,
                "screen": classification.screen.value if classification else "unknown",
                "keyboard_visible": classification.keyboard_visible if classification else None,
                "keyboard_ambiguous": classification.keyboard_ambiguous if classification else None,
                "screenshot_width": classification.screenshot_width if classification else None,
                "screenshot_height": classification.screenshot_height if classification else None,
            },
            "automation": {
                "enabled": automation.enabled if automation else False,
                "mode": automation.mode if automation else "unknown",
                "confidence_threshold": automation.confidence_threshold if automation else None,
                "auto_send_enabled": automation.auto_send_enabled if automation else False,
            },
            "halted": bool(state.halted) if state else None,
            "emergency_stop": bool(state.emergency_stop) if state else None,
            "halt_reason": state.halt_reason if state else None,
            "confidence": classification.confidence if classification else None,
            "risk_level": self._risk_level(state),
            "relationship_type": state.relationship_type if state else "unknown",
            "intent_type": state.intent_type if state else "unknown",
            "final_decision": state.final_decision if state else "unknown",
            "blocked_reason": state.blocked_reason if state else None,
            "last_action": state.last_action if state else self._last_action,
            "last_action_result": state.last_action_result if state else self._last_action_result,
            "recommended_reply_confidence": (
                candidate.score_breakdown.final_confidence if candidate is not None else None
            ),
        }

    def _risk_level(self, state: ControllerState | None) -> str:
        if state is None:
            return "unknown"
        if state.emergency_stop or state.halted:
            return "halted"
        if state.failure_category or state.blocked_reason or state.auto_send_blocked_reason:
            return "review"
        if state.classification.confidence < self.settings.confidence_threshold:
            return "high"
        if state.final_decision in {"reject", "blocked"}:
            return "high"
        if state.final_decision == "send":
            return "nominal"
        return "review"

    def _latest_incoming_message(self, state: ControllerState | None) -> str | None:
        if state is None:
            return None
        if state.thread_context and state.thread_context.full_conversation:
            for entry in reversed(state.thread_context.full_conversation):
                if entry.get("speaker") == "other" and entry.get("text"):
                    return redact_private_text(str(entry["text"]))
        if state.classification.recent_messages:
            return redact_private_text(state.classification.recent_messages[-1])
        return None

    def _latest_sent_reply(self) -> str | None:
        if self._last_send_and_read_result and self._last_send_and_read_result.sent_text:
            return redact_private_text(self._last_send_and_read_result.sent_text)
        for event in self.log_store.list_recent(limit=100):
            if event.get("event_type") not in {"approved_action", "send_and_read_completed"}:
                continue
            metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
            sent = metadata.get("sent_text") or metadata.get("postcondition_text")
            if sent:
                return redact_private_text(str(sent))
        return None

    def cyber_mission_control_status(self) -> dict[str, Any]:
        state = self.latest_state
        metrics = self.metrics_snapshot().model_dump(mode="json")
        training = self.training_stats()
        logs = self.log_store.list_recent(limit=40)
        candidate = self._recommended_candidate(state)
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "state": self._current_state_summary(),
            "messages": {
                "last_observed_incoming": self._latest_incoming_message(state),
                "last_drafted_reply": (
                    redact_private_text(candidate.text) if candidate is not None else None
                ),
                "last_sent_reply": self._latest_sent_reply(),
            },
            "action_queue": [item.model_dump(mode="json") for item in state.unread_queue] if state else [],
            "events": logs,
            "health_cards": [
                {
                    "label": "Loop latency",
                    "value": metrics["last"]["loop_time_ms"] if metrics.get("last") else None,
                    "unit": "ms",
                    "status": "observed" if metrics.get("last") else "unknown",
                },
                {
                    "label": "Classifier confidence",
                    "value": state.classification.confidence if state else None,
                    "unit": "",
                    "status": self._risk_level(state),
                },
                {
                    "label": "Vector DB",
                    "value": training.get("vector_index_count"),
                    "unit": "rows",
                    "status": "available" if training.get("vector_db_available") else "unknown",
                },
                {
                    "label": "Halted iterations",
                    "value": metrics.get("halted_iterations"),
                    "unit": "",
                    "status": "observed",
                },
            ],
            "pipeline": self._trace_stages(state),
            "metrics": metrics,
            "training": training,
        }

    def cyber_execution_trace_latest(self) -> dict[str, Any]:
        state = self.latest_state
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "trace_id": (
                f"trace-{state.captured_at.strftime('%Y%m%d%H%M%S')}" if state else None
            ),
            "stages": self._trace_stages(state),
            "tree": self._trace_tree(state),
            "events": self.log_store.list_recent(limit=30),
            "state": self._state_payload(state),
        }

    def _trace_stages(self, state: ControllerState | None) -> list[dict[str, Any]]:
        timings = state.timings_ms if state else {}
        confidence = state.classification.confidence if state else None
        screenshot_exists = bool(state and Path(state.screenshot_path).exists())
        candidate = self._recommended_candidate(state)
        stages = [
            self._trace_stage("observe", "Observe Screen", state is not None, confidence, timings.get("device_check"), state.summary if state else None),
            self._trace_stage("screenshot", "Capture Screenshot", screenshot_exists, confidence, timings.get("screenshot_capture"), state.screenshot_path if state else None),
            self._trace_stage("ocr", "OCR / Screen Parse", bool(state and state.classification.visible_text), confidence, timings.get("ocr"), f"{len(state.classification.visible_text)} text rows" if state else None),
            self._trace_stage("classification", "Screen Classification", state is not None, confidence, timings.get("classification"), state.classification.screen.value if state else None, failed=bool(state and confidence is not None and confidence < self.settings.confidence_threshold)),
            self._trace_stage("intent", "Intent Detection", bool(state and state.intent_type != "unknown"), None, timings.get("intent_detection"), state.intent_type if state else None),
            self._trace_stage("relationship", "Relationship Detection", bool(state and state.relationship_type != "unknown"), None, timings.get("relationship_detection"), state.relationship_type if state else None),
            self._trace_stage("retrieval", "Retrieval Search", bool(state and state.retrieved_examples_count), None, timings.get("retrieval"), f"{state.retrieved_examples_count} examples" if state else None),
            self._trace_stage("generation", "Candidate Generation", bool(state and state.reply_suggestions), None, timings.get("generation"), redact_private_text(candidate.text) if candidate else None),
            self._trace_stage("critic", "Critic / Policy Check", state is not None and candidate is not None, candidate.score_breakdown.final_confidence if candidate else None, timings.get("critic"), state.final_decision if state else None, failed=bool(state and state.blocked_reason)),
            self._trace_stage("approval", "Approval Gate", bool(state and state.last_action), None, None, state.last_action_result if state else None, halted=bool(state and state.halted)),
            self._trace_stage("execute", "Type / Send / Halt", bool(state and state.last_action), None, timings.get("execution"), state.last_action if state else None, halted=bool(state and state.halted)),
            self._trace_stage("verify", "Verification / Logging", bool(self.log_store.list_recent(limit=1)), None, timings.get("log_write"), state.failure_category if state else None, failed=bool(state and state.failure_category)),
        ]
        return stages

    def _trace_stage(
        self,
        stage_id: str,
        label: str,
        observed: bool,
        confidence: float | None,
        duration_ms: object,
        output_summary: object,
        *,
        failed: bool = False,
        halted: bool = False,
    ) -> dict[str, Any]:
        if halted:
            status = "halted"
        elif failed:
            status = "failed"
        elif observed:
            status = "passed"
        else:
            status = "unknown"
        return {
            "id": stage_id,
            "label": label,
            "status": status,
            "confidence": confidence,
            "duration_ms": duration_ms if isinstance(duration_ms, (int, float)) else None,
            "output_summary": output_summary if output_summary not in {"", []} else None,
            "failure_reason": output_summary if failed or halted else None,
        }

    def _trace_tree(self, state: ControllerState | None) -> list[dict[str, Any]]:
        stages = {stage["id"]: stage for stage in self._trace_stages(state)}
        nodes = [
            ("observation", ["observe", "screenshot", "ocr"]),
            ("classification", ["classification"]),
            ("retrieval", ["intent", "relationship", "retrieval"]),
            ("generation", ["generation"]),
            ("critic", ["critic"]),
            ("policy", ["approval"]),
            ("send", ["execute"]),
            ("verification/logging", ["verify"]),
        ]
        return [
            {
                "id": node_id,
                "status": next((stages[item]["status"] for item in children if item in stages), "unknown"),
                "children": [stages[item] for item in children if item in stages],
            }
            for node_id, children in nodes
        ]

    def cyber_signal_replay_runs(self, limit: int = 25) -> dict[str, Any]:
        safe_limit = max(1, min(int(limit or 25), 100))
        logs = self.log_store.list_recent(limit=safe_limit)
        return {
            "runs": [
                {
                    "run_id": str(event.get("id")),
                    "timestamp": event.get("timestamp"),
                    "event_type": event.get("event_type"),
                    "screen": event.get("detected_screen"),
                    "confidence": event.get("confidence"),
                    "outcome": event.get("execution_result"),
                    "has_before_screenshot": bool(self._public_media_url(event.get("before_screenshot_path"))),
                    "has_after_screenshot": bool(self._public_media_url(event.get("after_screenshot_path"))),
                }
                for event in logs
                if event.get("id") is not None
            ],
            "empty_state": "No replay data available" if not logs else None,
        }

    def cyber_signal_replay_run(self, run_id: str) -> dict[str, Any]:
        events = self.log_store.list_recent(limit=1000)
        selected = next((event for event in events if str(event.get("id")) == str(run_id)), None)
        if selected is None:
            raise HTTPException(status_code=404, detail="Replay run not found.")
        metadata = selected.get("metadata") if isinstance(selected.get("metadata"), dict) else {}
        event_cards = [
            {
                "id": f"{selected['id']}:observation",
                "timestamp": selected.get("timestamp"),
                "label": selected.get("event_type"),
                "summary": selected.get("execution_result"),
                "metadata": metadata,
            }
        ]
        return {
            "run_id": str(selected.get("id")),
            "timestamp": selected.get("timestamp"),
            "event_type": selected.get("event_type"),
            "screen": selected.get("detected_screen"),
            "confidence": selected.get("confidence"),
            "planner_decision": selected.get("planner_decision"),
            "approval_decision": selected.get("approval_decision"),
            "execution_result": selected.get("execution_result"),
            "before_screenshot_url": self._public_media_url(selected.get("before_screenshot_path")),
            "after_screenshot_url": self._public_media_url(selected.get("after_screenshot_path")),
            "classification": {
                "visible_text": metadata.get("visible_text", []),
                "recent_messages": metadata.get("recent_messages", []),
                "features_used": metadata.get("features_used", []),
                "debug_info": metadata.get("debug_info", {}),
            },
            "retrieved_memories": self.latest_state.retrieved_examples if self.latest_state else [],
            "generated_candidates": [candidate.model_dump(mode="json") for candidate in self.latest_state.draft_candidates] if self.latest_state else [],
            "selected_candidate": self._recommended_candidate(self.latest_state).model_dump(mode="json") if self._recommended_candidate(self.latest_state) else None,
            "policy_result": {
                "failure_category": metadata.get("failure_category"),
                "halted": selected.get("event_type") in {"execution_halt", "emergency_stop"},
            },
            "events": event_cards,
            "failure_point": selected.get("event_type") if metadata.get("failure_category") else None,
        }

    def cyber_vision_layer_latest(self) -> dict[str, Any]:
        state = self.latest_state
        if state is None:
            return {
                "screenshot_url": None,
                "classification": None,
                "ocr_regions": [],
                "ocr_text": [],
                "keyboard_region": None,
                "action_targets": [],
                "debug_image_url": None,
                "fixture_debug": {"status": "unknown"},
                "empty_state": "No trace recorded yet",
            }
        classification = state.classification
        keyboard_region = None
        if classification.keyboard_visible and classification.keyboard_height:
            keyboard_region = {
                "left": 0,
                "top": max(0, classification.screenshot_height - classification.keyboard_height),
                "right": classification.screenshot_width,
                "bottom": classification.screenshot_height,
                "label": "keyboard",
            }
        action_targets = []
        for action in classification.available_actions:
            if action.region:
                action_targets.append({**action.region.model_dump(mode="json"), "label": action.label, "type": action.action_type.value})
            elif action.point:
                action_targets.append(
                    {
                        "left": max(0, action.point.x - 36),
                        "top": max(0, action.point.y - 36),
                        "right": action.point.x + 36,
                        "bottom": action.point.y + 36,
                        "label": action.label,
                        "type": action.action_type.value,
                    }
                )
        return {
            "screenshot_url": self._public_media_url(state.screenshot_path),
            "debug_image_url": self._public_media_url(state.debug_image_path),
            "classification": classification.model_dump(mode="json"),
            "classification_confidence": classification.confidence,
            "ocr_regions": [],
            "ocr_text": classification.visible_text,
            "keyboard_region": keyboard_region,
            "action_targets": action_targets,
            "low_confidence_warning": classification.confidence < self.settings.confidence_threshold,
            "fixture_debug": {
                "status": "available" if state.debug_image_path else "unknown",
                "debug_info": classification.debug_info,
            },
        }

    def _memory_rows(self) -> list[dict[str, Any]]:
        rows = []
        for row in load_all_training_rows(self.settings.ai_reply_training_messages_dir):
            source = str(row.get("_source") or row.get("source") or "real")
            rows.append({**row, "_source_type": "synthetic" if row.get("is_synthetic") or source == "generated_safe_template" else "real"})
        for row in load_corrections(self.settings.ai_reply_training_messages_dir):
            rows.append({**row, "my_reply": row.get("user_final_reply"), "_source_type": "correction", "is_correction": True})
        for row in self._whatsapp_web_memory_rows_for_inspector():
            rows.append(row)
        return rows

    def _whatsapp_web_memory_rows_for_inspector(self) -> list[dict[str, Any]]:
        self._ensure_whatsapp_web_memory_db()
        with sqlite3.connect(self.whatsapp_web_memory_db_path) as connection:
            connection.row_factory = sqlite3.Row
            try:
                db_rows = connection.execute(
                    """
                    SELECT id, created_at, contact_name, relationship_type, question, answer, source
                    FROM whatsapp_web_memory
                    ORDER BY created_at DESC
                    LIMIT 500
                    """
                ).fetchall()
            except sqlite3.Error:
                return []
        rows: list[dict[str, Any]] = []
        for row in db_rows:
            question = str(row["question"] or "").strip()
            answer = str(row["answer"] or "").strip()
            if not question or not answer:
                continue
            rows.append(
                {
                    "id": str(row["id"]),
                    "incoming": question,
                    "my_reply": answer,
                    "context": [],
                    "contact_name": str(row["contact_name"] or ""),
                    "relationship_type": normalize_relationship_type(str(row["relationship_type"] or "unknown")),
                    "intent_type": "thread_note",
                    "memory_type": "web_thread_note",
                    "_source": "whatsapp_web_memory",
                    "_source_type": "web_thread_note",
                    "source": str(row["source"] or "whatsapp_web_extension"),
                    "style_authority": "context_note",
                    "created_at": str(row["created_at"] or ""),
                }
            )
        return rows

    def _memory_id(self, row: dict[str, Any]) -> str:
        return str(row.get("id") or hashlib.sha1(json.dumps(row, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12])

    def _memory_public_row(self, row: dict[str, Any]) -> dict[str, Any]:
        source_type = str(row.get("_source_type") or "unknown")
        authority = "high" if source_type == "correction" else ("low" if source_type == "synthetic" else str(row.get("style_authority") or "medium"))
        return {
            "id": self._memory_id(row),
            "incoming": redact_private_text(str(row.get("incoming") or "")),
            "my_reply": redact_private_text(str(row.get("my_reply") or row.get("user_final_reply") or "")),
            "context": [redact_private_text(str(item)) for item in row.get("context", []) if str(item).strip()] if isinstance(row.get("context"), list) else [],
            "source_type": source_type,
            "authority": authority,
            "relationship_type": normalize_relationship_type(str(row.get("relationship_type") or "unknown")),
            "intent_type": normalize_intent_label(str(row.get("intent_type") or "unknown")),
            "contact_name": redact_private_text(str(row.get("contact_name") or "")),
            "quarantined": bool(row.get("is_quarantined") or row.get("quarantined")),
            "metadata": {key: value for key, value in row.items() if key not in {"incoming", "my_reply", "user_final_reply", "context"}},
        }

    def cyber_memory_inspector_data(self, limit: int = 500) -> dict[str, Any]:
        safe_limit = max(1, min(int(limit or 500), 1000))
        rows = [self._memory_public_row(row) for row in self._memory_rows()]
        rows = rows[:safe_limit]
        return {
            "examples": rows,
            "counts": {
                "total": len(rows),
                "correction": sum(1 for row in rows if row["source_type"] == "correction"),
                "real": sum(1 for row in rows if row["source_type"] == "real"),
                "synthetic": sum(1 for row in rows if row["source_type"] == "synthetic"),
                "quarantined": sum(1 for row in rows if row["quarantined"]),
            },
            "empty_state": "No trace recorded yet" if not rows else None,
        }

    def cyber_memory_inspector_example(self, example_id: str) -> dict[str, Any]:
        rows = [self._memory_public_row(row) for row in self._memory_rows()]
        selected = next((row for row in rows if row["id"] == example_id), None)
        if selected is None:
            raise HTTPException(status_code=404, detail="Memory example not found.")
        backend = LexicalRetrievalBackend(self._memory_rows())
        neighbours = backend.search(
            selected["incoming"],
            selected["context"],
            selected["relationship_type"],
            selected["intent_type"],
            contact_name=selected.get("contact_name") or None,
            limit=8,
        )
        return {
            "example": selected,
            "nearest_neighbours": [item.model_dump(mode="json") for item in neighbours if item.retrieval_id != selected["id"]],
            "correction_lineage": selected["metadata"] if selected["source_type"] == "correction" else None,
        }

    def cyber_retrieval_battle(self, payload: dict[str, Any]) -> dict[str, Any]:
        incoming = str(payload.get("incoming") or "").strip()
        context = payload.get("context") if isinstance(payload.get("context"), list) else []
        contact_name = str(payload.get("contact_name") or "").strip() or None
        relationship = normalize_relationship_type(str(payload.get("relationship_type") or "unknown"))
        intent = normalize_intent_label(str(payload.get("intent_type") or classify_intent(incoming, [str(item) for item in context])))
        rows = self._memory_rows()
        lexical = LexicalRetrievalBackend(rows).search(incoming, [str(item) for item in context], relationship, intent, contact_name=contact_name, limit=6)
        vector = VectorChromaRetrievalBackend(
            index_dir=self.settings.data_dir / "vector_db" / "reply_examples_chroma",
            fallback_rows=rows,
        ).search(incoming, [str(item) for item in context], relationship, intent, contact_name=contact_name, limit=6)
        lexical_top = lexical[0] if lexical else None
        vector_top = vector[0] if vector else None
        winner = None
        if lexical_top or vector_top:
            winner = "vector" if (vector_top and (not lexical_top or vector_top.score >= lexical_top.score)) else "lexical"
        return {
            "query": {
                "incoming": redact_private_text(incoming),
                "context": [redact_private_text(str(item)) for item in context],
                "contact_name": redact_private_text(contact_name or ""),
                "relationship_type": relationship,
                "intent_type": intent,
            },
            "lexical": [item.model_dump(mode="json") for item in lexical],
            "vector": [item.model_dump(mode="json") for item in vector],
            "winner": winner,
            "reason": "highest available retrieval score" if winner else "Metric unavailable",
        }

    def cyber_contact_graph_data(self) -> dict[str, Any]:
        rows = [self._memory_public_row(row) for row in self._memory_rows()]
        contacts: dict[str, dict[str, Any]] = {}
        for row in rows:
            contact = row.get("contact_name") or "Unknown"
            bucket = contacts.setdefault(
                contact,
                {
                    "id": hashlib.sha1(contact.encode("utf-8")).hexdigest()[:12],
                    "contact_name": contact,
                    "relationship_type": row["relationship_type"],
                    "interaction_count": 0,
                    "training_examples": 0,
                    "corrections": 0,
                    "blacklisted": self._normalize_contact_name(contact) in self._blacklist if contact != "Unknown" else False,
                    "auto_send_eligible": None,
                    "style_strength": None,
                },
            )
            bucket["interaction_count"] += 1
            bucket["training_examples"] += 1
            if row["source_type"] == "correction":
                bucket["corrections"] += 1
        links = []
        grouped: dict[str, list[dict[str, Any]]] = {}
        for node in contacts.values():
            grouped.setdefault(str(node["relationship_type"]), []).append(node)
        for relationship, items in grouped.items():
            for left, right in zip(items, items[1:]):
                links.append({"source": left["id"], "target": right["id"], "relationship": relationship, "weight": 1})
        return {
            "nodes": list(contacts.values()),
            "links": links,
            "empty_state": "No trace recorded yet" if not contacts else None,
        }

    def cyber_failure_atlas_stats(self) -> dict[str, Any]:
        logs = self.log_store.list_recent(limit=500)
        categories = Counter()
        halt_reasons = Counter()
        timeline = []
        for event in logs:
            metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
            category = metadata.get("failure_category")
            if category:
                categories[str(category)] += 1
            if event.get("event_type") in {"execution_halt", "approval_denied", "compose_validation_failed", "send_and_read_failed", "emergency_stop"}:
                halt_reasons[str(event.get("execution_result") or "unknown")] += 1
                timeline.append(
                    {
                        "timestamp": event.get("timestamp"),
                        "event_type": event.get("event_type"),
                        "screen": event.get("detected_screen"),
                        "category": category or "unknown",
                        "reason": event.get("execution_result"),
                        "screenshot_url": self._public_media_url(event.get("before_screenshot_path")),
                    }
                )
        stage_map = {
            "classifier_low_confidence": "classification",
            "classifier_wrong": "classification",
            "keyboard_state_ambiguous": "approval",
            "keyboard_state_wrong": "approval",
            "tap_target_invalid": "execute",
            "ui_not_idle": "execute",
            "precondition_failed": "approval",
            "postcondition_timeout": "verification",
            "postcondition_mismatch": "verification",
            "response_timeout": "verification",
            "policy_blocked": "policy",
            "execution_retry_exhausted": "execute",
        }
        heatmap = Counter(stage_map.get(key, "unknown") for key in categories)
        metrics = self.metrics_snapshot().model_dump(mode="json")
        validation = metrics.get("compose_validation", {})
        return {
            "common_failure_categories": dict(categories.most_common()),
            "halt_reasons": dict(halt_reasons.most_common(12)),
            "classifier_failures": categories.get("classifier_low_confidence", 0) + categories.get("classifier_wrong", 0),
            "keyboard_ambiguity_failures": categories.get("keyboard_state_ambiguous", 0),
            "stale_or_duplicate_prevention_events": 0,
            "wrong_contact_prevention_events": 0,
            "recovery_outcomes": {},
            "success_rate_summary": {
                "compose_validation_success_rate": validation.get("success_rate"),
                "total_iterations": metrics.get("total_iterations"),
                "halted_iterations": metrics.get("halted_iterations"),
            },
            "pipeline_heatmap": dict(heatmap),
            "timeline": timeline[:100],
            "empty_state": "No trace recorded yet" if not logs else None,
        }

    def cyber_risk_engine_status(self) -> dict[str, Any]:
        state = self.latest_state
        candidate = self._recommended_candidate(state)
        scores = candidate.score_breakdown if candidate else None
        dimensions = {
            "screen_confidence": state.classification.confidence if state else None,
            "contact_confidence": scores.persona_confidence if scores else None,
            "wrong_thread_risk": 1.0 if state and state.blacklisted else None,
            "duplicate_reply_risk": scores.risk_score if scores else None,
            "reply_risk": scores.risk_score if scores else None,
            "ambiguity": 1.0 if state and state.classification.keyboard_ambiguous else None,
            "style_confidence": scores.style_confidence if scores else None,
            "policy_confidence": 0.0 if state and state.halted else (1.0 if state else None),
            "send_confidence": scores.final_confidence if scores else None,
            "verification_confidence": 0.0 if state and state.failure_category else (1.0 if state else None),
        }
        if state is None:
            decision = "UNKNOWN"
        elif state.emergency_stop or state.halted:
            decision = "HALTED"
        elif state.blocked_reason or state.final_decision in {"reject", "blocked"}:
            decision = "BLOCKED"
        elif state.final_decision == "send" and candidate and candidate.auto_send_allowed:
            decision = "SEND_ALLOWED"
        elif state.reply_suggestions:
            decision = "SAFE_TO_DRAFT"
        else:
            decision = "REVIEW_REQUIRED"
        waveform = [
            {
                "timestamp": metric.timestamp.isoformat(),
                "screen_confidence": metric.confidence,
                "intent_confidence": None,
                "reply_confidence": None,
                "send_confidence": None,
            }
            for metric in self._metrics_history[-80:]
        ]
        return {
            "decision": decision,
            "risk_level": self._risk_level(state),
            "dimensions": dimensions,
            "waveform": waveform,
            "state": self._current_state_summary(),
        }

    def cyber_memory_forge_status(self) -> dict[str, Any]:
        stats = self.training_stats()
        recent = self.training_recent(limit=50)
        rows = [self._memory_public_row(row) for row in self._memory_rows()]
        authority_distribution = Counter(str(row["authority"]) for row in rows)
        relationship_distribution = Counter(str(row["relationship_type"]) for row in rows)
        intent_distribution = Counter(str(row["intent_type"]) for row in rows)
        quarantine_rows = self._read_jsonl_tail(self.training_quarantine_path, limit=10000)
        return {
            **stats,
            "training_examples_count": stats.get("real_examples_count", 0),
            "synthetic_template_count": stats.get("synthetic_examples_count", 0),
            "quarantined_example_count": len(quarantine_rows),
            "last_rebuild_status": {
                "last_vector_rebuild": stats.get("last_vector_rebuild"),
                "rebuild_required": stats.get("vector_rebuild_required"),
            },
            "authority_distribution": dict(authority_distribution),
            "relationship_distribution": dict(relationship_distribution),
            "intent_distribution": dict(intent_distribution),
            "recent_sessions": recent.get("items", []),
            "ai_core": self.ai_core_status(),
        }

    def cyber_simulation_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        context = payload.get("context") if isinstance(payload.get("context"), list) else []
        incoming = str(payload.get("incoming") or "").strip()
        if not incoming:
            raise HTTPException(status_code=400, detail="incoming must not be empty.")
        force_low_confidence = bool(payload.get("force_low_confidence"))
        relationship = normalize_relationship_type(str(payload.get("relationship_type") or "unknown"))
        intent = normalize_intent_label(str(payload.get("intent_type") or classify_intent(incoming, [str(item) for item in context])))
        request = TrainingChatRequest(
            incoming=incoming,
            context=[str(item) for item in context],
            contact_name=str(payload.get("contact_name") or "") or None,
            relationship_type=relationship,
            intent_type=intent,
            diversity_mode=str(payload.get("diversity_mode") or "natural"),
        )
        bundle = self._training_bundle(request, regenerate=False)
        candidates = [candidate.model_dump(mode="json") for candidate in bundle.reply_candidates]
        selected = candidates[bundle.recommended_reply_index or 0] if candidates else None
        risk_breakdown = {
            "screen_confidence": 0.35 if force_low_confidence else None,
            "classifier_mismatch": bool(payload.get("force_classifier_mismatch")),
            "risk_tolerance": payload.get("risk_tolerance"),
            "policy_confidence": 0.0 if force_low_confidence or payload.get("force_classifier_mismatch") else None,
            "reply_risk": selected.get("score_breakdown", {}).get("risk_score") if selected else None,
        }
        if force_low_confidence or payload.get("force_classifier_mismatch"):
            decision = "REVIEW_REQUIRED"
        elif selected and selected.get("final_decision") == "send":
            decision = "SEND_ALLOWED" if payload.get("automation_mode") == "auto-send" else "SAFE_TO_DRAFT"
        else:
            decision = "REVIEW_REQUIRED"
        return {
            "offline": True,
            "phone_touched": False,
            "adb_used": False,
            "query": request.model_dump(mode="json"),
            "generated_candidates": candidates,
            "selected_candidate": selected,
            "selected_memories": bundle.retrieved_examples,
            "policy_result": {
                "decision": decision,
                "final_decision": bundle.final_decision,
                "blocked_reason": bundle.blocked_reason,
            },
            "risk_breakdown": risk_breakdown,
            "pipeline_trace": [
                {"stage": "retrieval", "status": "passed" if bundle.retrieved_examples else "unknown", "output_summary": f"{len(bundle.retrieved_examples)} examples"},
                {"stage": "generation", "status": "passed" if candidates else "unknown", "output_summary": f"{len(candidates)} candidates"},
                {"stage": "critic_policy", "status": "passed" if selected else "unknown", "output_summary": decision},
                {"stage": "send", "status": "halted", "output_summary": "Simulation only; no phone interaction."},
            ],
        }

    def run_compose_validation(self, text: str) -> ControllerState:
        validation_text = text.strip()
        if not validation_text:
            raise HTTPException(status_code=400, detail="Validation text must not be empty.")
        current_state = self.refresh_state()
        if current_state.classification.package_name != "com.google.android.apps.messaging":
            return self._fail_compose_validation(
                current_state=current_state,
                text=validation_text,
                reason="Compose validation only runs inside Google Messages.",
                category=FailureCategory.PRECONDITION_FAILED,
            )
        if current_state.classification.screen.value != "thread_view":
            return self._fail_compose_validation(
                current_state=current_state,
                text=validation_text,
                reason="Compose validation requires an existing Google Messages thread view.",
                category=FailureCategory.PRECONDITION_FAILED,
            )
        if current_state.classification.keyboard_ambiguous:
            return self._fail_compose_validation(
                current_state=current_state,
                text=validation_text,
                reason="Keyboard state is ambiguous; compose validation is blocked.",
                category=FailureCategory.KEYBOARD_STATE_AMBIGUOUS,
            )
        plan = self.planner.build_plan(
            classification=current_state.classification,
            action_type=ApprovalActionType.TYPE_DRAFT,
            draft_text=validation_text,
        )
        plan.allowed_package_names = ["com.google.android.apps.messaging"]
        plan.postcondition_text = validation_text
        plan.timeout_ms = 3500
        plan.retry = 1
        try:
            result_state = self._execute_approved_plan(plan=plan, approval_note="compose validation")
        except HTTPException as exc:
            category = self._last_failure_category or FailureCategory.EXECUTION_RETRY_EXHAUSTED
            return self._fail_compose_validation(
                current_state=self.latest_state or current_state,
                text=validation_text,
                reason=str(exc.detail),
                category=category,
            )
        self._last_compose_validation_result = ComposeValidationResult(
            success=True,
            validation_text=validation_text,
            observed_text=validation_text,
            reason="Validation text appeared in the compose box.",
        )
        self._compose_validation_runs.append(self._last_compose_validation_result)
        result_state.compose_validation = self._last_compose_validation_result
        return result_state

    def run_send_and_read(
        self,
        text: str,
        *,
        wait_timeout_seconds: float = 90.0,
        poll_interval_seconds: float = 1.5,
    ) -> ControllerState:
        message_text = text.strip()
        if not message_text:
            raise HTTPException(status_code=400, detail="Message text must not be empty.")
        if wait_timeout_seconds <= 0:
            raise HTTPException(status_code=400, detail="Reply wait timeout must be greater than zero.")

        current_state = self.refresh_state()
        preflight_error = self._validate_messages_thread_state(current_state, action_name="send-and-read")
        if preflight_error is not None:
            return self._fail_send_and_read(
                current_state=current_state,
                text=message_text,
                reason=preflight_error,
                category=FailureCategory.PRECONDITION_FAILED,
            )

        compose_text = self._read_compose_box_text()
        if compose_text:
            return self._fail_send_and_read(
                current_state=current_state,
                text=message_text,
                reason="Send-and-read requires an empty compose box before typing.",
                category=FailureCategory.PRECONDITION_FAILED,
            )

        type_plan = self.planner.build_plan(
            classification=current_state.classification,
            action_type=ApprovalActionType.TYPE_DRAFT,
            draft_text=message_text,
        )
        type_plan.allowed_package_names = ["com.google.android.apps.messaging"]
        type_plan.postcondition_text = message_text
        type_plan.timeout_ms = 3500
        type_plan.retry = 1
        try:
            self._execute_approved_plan(plan=type_plan, approval_note="send and read: type")
        except HTTPException as exc:
            category = self._last_failure_category or FailureCategory.EXECUTION_RETRY_EXHAUSTED
            return self._fail_send_and_read(
                current_state=self.latest_state or current_state,
                text=message_text,
                reason=str(exc.detail),
                category=category,
            )

        send_state = self.refresh_state()
        preflight_error = self._validate_messages_thread_state(send_state, action_name="send-and-read")
        if preflight_error is not None:
            return self._fail_send_and_read(
                current_state=send_state,
                text=message_text,
                reason=preflight_error,
                category=FailureCategory.PRECONDITION_FAILED,
            )
        if send_state.classification.keyboard_visible is not True:
            return self._fail_send_and_read(
                current_state=send_state,
                text=message_text,
                reason="Send-and-read requires the keyboard to remain visible after typing.",
                category=FailureCategory.KEYBOARD_STATE_WRONG,
            )

        send_plan = self.planner.build_plan(
            classification=send_state.classification,
            action_type=ApprovalActionType.SEND_MESSAGE,
        )
        send_plan.allowed_package_names = ["com.google.android.apps.messaging"]
        send_plan.postcondition_text = message_text
        send_plan.timeout_ms = 5000
        send_plan.retry = 1
        try:
            after_send_state = self._execute_approved_plan(plan=send_plan, approval_note="send and read: send")
        except HTTPException as exc:
            category = self._last_failure_category or FailureCategory.EXECUTION_RETRY_EXHAUSTED
            return self._fail_send_and_read(
                current_state=self.latest_state or send_state,
                text=message_text,
                reason=str(exc.detail),
                category=category,
            )

        try:
            response_text = self._wait_for_new_visible_reply(
                sent_text=message_text,
                baseline_state=after_send_state,
                wait_timeout_seconds=wait_timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
            )
        except HTTPException as exc:
            return self._fail_send_and_read(
                current_state=self.latest_state or after_send_state,
                text=message_text,
                reason=str(exc.detail),
                category=FailureCategory.PRECONDITION_FAILED,
            )
        if response_text is None:
            return self._fail_send_and_read(
                current_state=self.latest_state or after_send_state,
                text=message_text,
                reason=f"No new visible reply arrived within {wait_timeout_seconds:.1f} seconds.",
                category=FailureCategory.RESPONSE_TIMEOUT,
            )

        self._last_send_and_read_result = SendAndReadResult(
            success=True,
            sent_text=message_text,
            observed_sent_text=message_text,
            response_text=response_text,
            reason="Message was sent and a new visible reply was detected in the active thread.",
        )
        final_state = self.latest_state or after_send_state
        final_state.send_and_read = self._last_send_and_read_result
        final_state.failure_category = None
        final_state.last_action = "send and read"
        final_state.last_action_result = "Completed"
        self.latest_state = final_state
        self.log_store.record_event(
            event_type="send_and_read_completed",
            detected_screen=final_state.classification.screen.value,
            confidence=final_state.classification.confidence,
            planner_decision="send_and_read",
            approval_decision="send and read",
            execution_result="completed",
            before_screenshot_path=after_send_state.screenshot_path,
            after_screenshot_path=final_state.screenshot_path,
            metadata=self._last_send_and_read_result.model_dump(mode="json"),
        )
        return final_state

    def _resolve_photo(self, photo_id: str | None, suggestions: list[ApprovedPhoto]) -> ApprovedPhoto:
        if not suggestions:
            raise HTTPException(status_code=400, detail="No approved photos are currently suggested.")
        if photo_id is None:
            return suggestions[0]
        for suggestion in suggestions:
            if suggestion.photo_id == photo_id:
                return suggestion
        raise HTTPException(status_code=400, detail="Unknown approved photo selection.")

    def _capture_state(self, record_log: bool, execution_time_ms: float = 0.0) -> ControllerState:
        timestamp = datetime.now(timezone.utc)
        loop_started = time.perf_counter()
        timing_marks: dict[str, float] = {"request_start": 0.0}
        timing_marks["capture_start"] = 0.0
        screenshot_path = self.settings.screenshot_dir / f"capture-{timestamp.strftime('%Y%m%d-%H%M%S-%f')}.png"
        perception_result = self.pipeline.capture_and_classify(screenshot_path=screenshot_path)
        timing_marks["capture_end"] = (time.perf_counter() - loop_started) * 1000

        planner_started = time.perf_counter()
        timing_marks["planner_start"] = (planner_started - loop_started) * 1000
        draft_bundle = self.drafting.build_bundle(
            perception_result.classification.recent_messages,
            contact_name=self.latest_state.thread_context.contact_name if self.latest_state and self.latest_state.thread_context else None,
        )
        planner_decision = self.planner.decide(perception_result.classification)
        planner_time_ms = (time.perf_counter() - planner_started) * 1000
        timing_marks["planner_end"] = (time.perf_counter() - loop_started) * 1000

        policy_started = time.perf_counter()
        observation_policy = self.policy.evaluate_observation(
            classification=perception_result.classification,
            planner_decision=planner_decision,
            emergency_stop=self._emergency_stop,
            previous_step_failed=self._previous_step_failed,
        )
        timing_marks["policy_eval"] = (time.perf_counter() - policy_started) * 1000
        loop_time_ms = (time.perf_counter() - loop_started) * 1000
        metrics = LoopMetrics.from_perception(
            timestamp=timestamp,
            perception_timings=perception_result.timings,
            planner_time_ms=planner_time_ms,
            execution_time_ms=execution_time_ms,
            loop_time_ms=loop_time_ms,
            screen=perception_result.classification.screen.value,
            confidence=perception_result.classification.confidence,
        )
        state_started = time.perf_counter()
        state = ControllerState(
            captured_at=timestamp,
            classification=perception_result.classification,
            planner_decision=planner_decision,
            summary=draft_bundle.summary,
            compose_text=None,
            reply_suggestions=draft_bundle.reply_suggestions,
            reply_sequences=draft_bundle.reply_sequences,
            draft_candidates=[candidate.model_copy(deep=True) for candidate in draft_bundle.reply_candidates],
            recommended_reply_index=draft_bundle.recommended_reply_index,
            auto_send_blocked_reason=draft_bundle.auto_send_blocked_reason,
            relationship_type=draft_bundle.relationship_type,
            intent_type=draft_bundle.intent_type,
            retrieved_examples_count=draft_bundle.retrieved_examples_count,
            retrieved_examples=draft_bundle.retrieved_examples,
            final_decision=draft_bundle.final_decision,
            blocked_reason=draft_bundle.blocked_reason,
            timings_ms=dict(draft_bundle.timings_ms),
            prompt_preview=draft_bundle.prompt_preview,
            approved_photo_suggestions=draft_bundle.photo_suggestions,
            halted=not observation_policy.allowed,
            halt_reason=None if observation_policy.allowed else observation_policy.reason,
            emergency_stop=self._emergency_stop,
            screenshot_path=perception_result.screenshot_path,
            debug_image_path=perception_result.debug_image_path,
            metrics=metrics,
            last_action=self._last_action,
            last_action_result=self._last_action_result,
            verification_errors=list(self._verification_errors[-8:]),
            failure_category=(
                observation_policy.failure_category.value
                if observation_policy.failure_category
                else (self._last_failure_category.value if self._last_failure_category else None)
            ),
            compose_validation=self._last_compose_validation_result,
            send_and_read=self._last_send_and_read_result,
            automation_mode=AutomationMode(
                enabled=self._automation_enabled,
                mode=self._current_automation_mode,
                confidence_threshold=self._automation_confidence,
                auto_send_enabled=self._auto_send_enabled,
            ),
            unread_queue=[item.model_copy() for item in self._unread_queue],
            thread_context=self.latest_state.thread_context.model_copy(deep=True) if self.latest_state and self.latest_state.thread_context else None,
            blacklisted=self.latest_state.blacklisted if self.latest_state else False,
        )
        timing_marks["state_model_create"] = (time.perf_counter() - state_started) * 1000
        sync_started = time.perf_counter()
        self._sync_state_from_ui_hierarchy(state)
        timing_marks["context_hierarchy_sync"] = (time.perf_counter() - sync_started) * 1000
        redraft_started = time.perf_counter()
        if state.thread_context and state.thread_context.full_conversation:
            refreshed_drafts = self.drafting.build_bundle_with_context(
                recent_messages=state.classification.recent_messages,
                full_conversation=[
                    f"[{entry['speaker'].upper()}]: {entry['text']}"
                    for entry in state.thread_context.full_conversation
                ],
                contact_name=state.thread_context.contact_name,
            )
            self._apply_draft_bundle(state, refreshed_drafts)
        elif state.classification.recent_messages:
            refreshed_drafts = self.drafting.build_bundle(
                state.classification.recent_messages,
                contact_name=self._thread_contact_name_from_state(state),
            )
            self._apply_draft_bundle(state, refreshed_drafts)
        else:
            self._enrich_draft_candidates_with_runtime_confidence(state)
        timing_marks["context_aware_redraft"] = (time.perf_counter() - redraft_started) * 1000
        alias_started = time.perf_counter()
        self._write_latest_aliases(perception_result.screenshot_path, perception_result.debug_image_path)
        timing_marks["latest_alias_copy"] = (time.perf_counter() - alias_started) * 1000
        self._merge_perception_timings(state, metrics, timing_marks=timing_marks)
        self._metrics_history.append(metrics)
        if record_log and state.halted:
            self._halted_iterations += 1
        if record_log:
            log_started = time.perf_counter()
            self.log_store.record_event(
                event_type="observation",
                detected_screen=perception_result.classification.screen.value,
                confidence=perception_result.classification.confidence,
                planner_decision=planner_decision.reason,
                approval_decision="n/a",
                execution_result=observation_policy.reason,
                before_screenshot_path=perception_result.screenshot_path,
                metadata={
                    "app": perception_result.classification.app,
                    "visible_text": perception_result.classification.visible_text,
                    "recent_messages": perception_result.classification.recent_messages,
                    "available_actions": [
                        action.action_type.value for action in perception_result.classification.available_actions
                    ],
                    "features_used": perception_result.classification.features_used,
                    "debug_info": perception_result.classification.debug_info,
                    "metrics": metrics.model_dump(mode="json"),
                    "fixture_json_path": perception_result.fixture_json_path,
                    "fixture_png_path": perception_result.fixture_png_path,
                    "failure_category": state.failure_category,
                },
            )
            state.timings_ms["log_write"] = round((time.perf_counter() - log_started) * 1000, 2)
        self._finalize_timing_summary(state)
        return state

    def _write_latest_aliases(self, screenshot_path: str, debug_image_path: str | None) -> None:
        screenshot = Path(screenshot_path)
        if screenshot.exists():
            shutil.copyfile(screenshot, self.settings.screenshot_dir / "latest.png")
        if debug_image_path:
            debug_path = Path(debug_image_path)
            if debug_path.exists():
                shutil.copyfile(debug_path, self.settings.debug_dir / "latest-debug.png")

    def _execute_approved_plan(self, plan: ExecutionPlan, approval_note: str) -> ControllerState:
        current_state = self.latest_state or self.refresh_state()
        authorization = self.policy.authorize_plan(
            plan=plan,
            classification=current_state.classification,
            emergency_stop=self._emergency_stop,
            previous_step_failed=self._previous_step_failed,
        )
        if not authorization.allowed:
            self._previous_step_failed = True
            self._record_verification_error(authorization.reason, authorization.failure_category)
            self._last_action = approval_note
            self._last_action_result = authorization.reason
            self.log_store.record_event(
                event_type="approval_denied",
                detected_screen=current_state.classification.screen.value,
                confidence=current_state.classification.confidence,
                planner_decision=current_state.planner_decision.reason,
                approval_decision=approval_note,
                execution_result=authorization.reason,
                before_screenshot_path=current_state.screenshot_path,
                metadata={
                    "plan": plan.model_dump(mode="json"),
                    "failure_category": authorization.failure_category.value if authorization.failure_category else None,
                },
            )
            raise HTTPException(status_code=409, detail=authorization.reason)

        try:
            self._verify_precondition(plan, current_state.classification)
        except HTTPException as exc:
            self._previous_step_failed = True
            self._record_verification_error(str(exc.detail), FailureCategory.PRECONDITION_FAILED)
            self._last_action = approval_note
            self._last_action_result = str(exc.detail)
            raise
        execution_started = time.perf_counter()
        execution_timings: dict[str, float] = {"execution_start": 0.0, "sleep_ms": 0.0, "adb_total_ms": 0.0}
        failure_reason: str | None = None
        after_state: ControllerState | None = None
        failure_category: FailureCategory | None = None
        for attempt in range(plan.retry + 1):
            try:
                for step in plan.steps:
                    step_started = time.perf_counter()
                    self._execute_step(step=step, current_state=current_state)
                    if hasattr(self.pipeline, "invalidate_ocr_cache"):
                        self.pipeline.invalidate_ocr_cache()
                    step_ms = (time.perf_counter() - step_started) * 1000
                    execution_timings[f"execution_step_{step.kind}"] = execution_timings.get(f"execution_step_{step.kind}", 0.0) + step_ms
                    execution_timings["adb_total_ms"] += step_ms
                    idle_started = time.perf_counter()
                    self.adb.wait_for_idle(timeout_ms=min(2000, plan.timeout_ms), stable_cycles=1)
                    idle_ms = (time.perf_counter() - idle_started) * 1000
                    execution_timings["wait_for_idle"] = execution_timings.get("wait_for_idle", 0.0) + idle_ms
                    execution_timings["adb_total_ms"] += idle_ms
                postcondition_started = time.perf_counter()
                after_state = self._verify_postcondition(plan)
                execution_timings["postcondition_verify"] = (time.perf_counter() - postcondition_started) * 1000
                failure_reason = None
                failure_category = None
                break
            except (HTTPException, RuntimeError) as exc:
                failure_reason = str(exc)
                failure_category = self._categorize_execution_failure(exc)
                if attempt >= plan.retry:
                    break
                current_state = self.refresh_state()
                self._verify_precondition(plan, current_state.classification)
        execution_time_ms = (time.perf_counter() - execution_started) * 1000
        execution_timings["execution_end"] = execution_time_ms

        if failure_reason is not None or after_state is None:
            self._previous_step_failed = True
            if plan.retry > 0 and failure_category is not None:
                failure_reason = f"{failure_reason} Retries exhausted."
                failure_category = FailureCategory.EXECUTION_RETRY_EXHAUSTED
            self._record_verification_error(
                failure_reason or "Postcondition verification failed.",
                failure_category,
            )
            self._last_action = approval_note
            self._last_action_result = failure_reason or "Postcondition verification failed."
            halted_state = self._capture_state(record_log=True, execution_time_ms=execution_time_ms)
            halted_state.halted = True
            halted_state.halt_reason = failure_reason or "Postcondition verification failed."
            halted_state.failure_category = failure_category.value if failure_category else None
            self.latest_state = halted_state
            self.log_store.record_event(
                event_type="execution_halt",
                detected_screen=halted_state.classification.screen.value,
                confidence=halted_state.classification.confidence,
                planner_decision=plan.notes,
                approval_decision=approval_note,
                execution_result=halted_state.halt_reason,
                before_screenshot_path=current_state.screenshot_path,
                after_screenshot_path=halted_state.screenshot_path,
                metadata={
                    "plan": plan.model_dump(mode="json"),
                    "failure_category": failure_category.value if failure_category else None,
                },
            )
            raise HTTPException(status_code=409, detail=halted_state.halt_reason)

        self._previous_step_failed = False
        self._last_failure_category = None
        self._last_action = approval_note
        self._last_action_result = "Completed"
        after_state.metrics.execution_time_ms = execution_time_ms
        after_state.metrics.loop_time_ms += execution_time_ms
        for key, value in execution_timings.items():
            after_state.timings_ms[key] = round(value, 2)
        after_state.timings_ms["execution"] = round(execution_time_ms, 2)
        after_state.timings_ms["server_processing_ms"] = round(after_state.metrics.loop_time_ms, 2)
        self._finalize_timing_summary(after_state)
        after_state.last_action = self._last_action
        after_state.last_action_result = self._last_action_result
        after_state.failure_category = None
        self.latest_state = after_state
        self.log_store.record_event(
            event_type="approved_action",
            detected_screen=after_state.classification.screen.value,
            confidence=after_state.classification.confidence,
            planner_decision=plan.notes,
            approval_decision=approval_note,
            execution_result="completed",
            before_screenshot_path=current_state.screenshot_path,
            after_screenshot_path=after_state.screenshot_path,
            metadata={"plan": plan.model_dump(mode="json"), "metrics": after_state.metrics.model_dump(mode="json")},
        )
        return after_state

    def _verify_precondition(self, plan: ExecutionPlan, classification: ScreenClassification) -> None:
        if classification.screen != plan.expected_prev_screen:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Precondition failed: expected {plan.expected_prev_screen.value}, "
                    f"detected {classification.screen.value}."
                ),
            )

    def _verify_postcondition(self, plan: ExecutionPlan) -> ControllerState:
        deadline = time.monotonic() + (plan.timeout_ms / 1000.0)
        acceptable = set(plan.acceptable_next_screens or [plan.expected_next_screen])
        acceptable.add(plan.expected_next_screen)
        latest_observation: ControllerState | None = None
        last_text_mismatch = False
        while time.monotonic() < deadline:
            latest_observation = self._capture_state(record_log=False)
            keyboard_ok = (
                plan.expected_keyboard_after is None
                or latest_observation.classification.keyboard_visible == plan.expected_keyboard_after
            )
            if latest_observation.classification.screen in acceptable and keyboard_ok:
                if plan.postcondition_text:
                    observed = self._observe_compose_text(plan.postcondition_text, latest_observation)
                    if observed is not None:
                        return latest_observation
                    last_text_mismatch = True
                else:
                    return latest_observation
            time.sleep(0.25)
        expected = ", ".join(screen.value for screen in acceptable)
        detected = latest_observation.classification.screen.value if latest_observation else "unknown"
        if last_text_mismatch:
            raise HTTPException(
                status_code=409,
                detail=f"Postcondition mismatch: expected text '{plan.postcondition_text}' was not observed.",
            )
        raise HTTPException(
            status_code=409,
            detail=f"Postcondition timeout: expected {expected}, detected {detected}.",
        )

    def _record_verification_error(self, reason: str, category: FailureCategory | None = None) -> None:
        self._last_failure_category = category
        self._verification_errors.append(reason)
        self._verification_errors = self._verification_errors[-8:]

    def _execute_step(self, step: ExecutionStep, current_state: ControllerState) -> None:
        if step.kind == "tap":
            assert step.x is not None and step.y is not None
            tap_x = step.x
            tap_y = step.y
            if None not in {step.region_left, step.region_top, step.region_right, step.region_bottom}:
                assert step.region_left is not None
                assert step.region_top is not None
                assert step.region_right is not None
                assert step.region_bottom is not None
                if not (
                    step.region_left <= tap_x <= step.region_right
                    and step.region_top <= tap_y <= step.region_bottom
                ):
                    tap_x = (step.region_left + step.region_right) // 2
                    tap_y = (step.region_top + step.region_bottom) // 2
            before = self.settings.screenshot_dir / f"tap-before-{int(time.time() * 1000)}.png"
            after = self.settings.screenshot_dir / f"tap-after-{int(time.time() * 1000)}.png"
            self.adb.safe_tap(
                tap_x,
                tap_y,
                screen_context=SafeTapContext(
                    screen=current_state.classification.screen.value,
                    before_screenshot_path=str(before),
                    after_screenshot_path=str(after),
                    expected_region_left=step.region_left,
                    expected_region_top=step.region_top,
                    expected_region_right=step.region_right,
                    expected_region_bottom=step.region_bottom,
                ),
            )
            return
        if step.kind == "type_text":
            assert step.text is not None
            self.adb.type_text(step.text)
            return
        if step.kind == "keyevent":
            assert step.keycode is not None
            self.adb.keyevent(step.keycode)
            return
        if step.kind == "push_file":
            assert step.local_path is not None and step.remote_path is not None
            self.adb.push_file(Path(step.local_path), step.remote_path)
            return
        if step.kind == "media_scan":
            assert step.remote_path is not None
            self.adb.scan_media(step.remote_path)
            return
        raise HTTPException(status_code=500, detail=f"Unsupported execution step {step.kind}.")

    def _observe_compose_text(self, expected_text: str, state: ControllerState) -> str | None:
        normalized_expected = self._normalize_text(expected_text)
        for candidate in state.classification.visible_text + state.classification.recent_messages:
            if normalized_expected in self._normalize_text(candidate):
                return candidate
        try:
            hierarchy = self.adb.dump_ui_hierarchy()
        except RuntimeError:
            return None
        if normalized_expected in self._normalize_text(hierarchy):
            return expected_text
        return None

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.lower().split())

    def _categorize_execution_failure(self, exc: Exception) -> FailureCategory:
        detail = str(getattr(exc, "detail", exc)).lower()
        if isinstance(exc, ADBUIStabilityTimeout):
            return FailureCategory.UI_NOT_IDLE
        if isinstance(exc, ADBCommandError):
            if "keyboard state is ambiguous" in detail:
                return FailureCategory.KEYBOARD_STATE_AMBIGUOUS
            if "outside the expected region" in detail or "outside screen bounds" in detail:
                return FailureCategory.TAP_TARGET_INVALID
        if "precondition failed" in detail:
            return FailureCategory.PRECONDITION_FAILED
        if "postcondition mismatch" in detail:
            return FailureCategory.POSTCONDITION_MISMATCH
        if "postcondition timeout" in detail:
            return FailureCategory.POSTCONDITION_TIMEOUT
        if "keyboard state mismatch" in detail:
            return FailureCategory.KEYBOARD_STATE_WRONG
        return FailureCategory.POLICY_BLOCKED

    def _validate_messages_thread_state(self, state: ControllerState, action_name: str) -> str | None:
        if state.classification.package_name != "com.google.android.apps.messaging":
            return f"{action_name} only runs inside Google Messages."
        if state.classification.screen.value != "thread_view":
            return f"{action_name} requires an existing Google Messages thread view."
        if state.classification.keyboard_ambiguous:
            return f"{action_name} is blocked because keyboard state is ambiguous."
        return None

    def _read_compose_box_text(self) -> str | None:
        try:
            hierarchy = self.adb.dump_ui_hierarchy()
        except RuntimeError:
            return None
        return self._extract_compose_box_text_from_hierarchy(hierarchy)

    def _extract_compose_box_text_from_hierarchy(self, hierarchy: str) -> str | None:
        matches = list(re.finditer(r'<node\b[^>]*class="android\.widget\.EditText"[^>]*/?>', hierarchy))
        for match in reversed(matches):
            node = match.group(0)
            text_match = re.search(r'text="([^"]*)"', node)
            if text_match is None:
                continue
            value = html.unescape(text_match.group(1)).strip()
            if value and self._normalize_text(value) not in {"type a message", "rcs message", "text message"}:
                return value
        return None

    def _wait_for_new_visible_reply(
        self,
        *,
        sent_text: str,
        baseline_state: ControllerState,
        wait_timeout_seconds: float,
        poll_interval_seconds: float,
    ) -> str | None:
        baseline_counts = self._message_counts(baseline_state.classification.recent_messages)
        sent_normalized = self._normalize_text(sent_text)
        already_visible = self._message_after_sent(
            baseline_state.classification.recent_messages,
            sent_normalized=sent_normalized,
        )
        if already_visible is not None:
            return already_visible
        deadline = time.monotonic() + wait_timeout_seconds
        attempts = 0
        while time.monotonic() < deadline or attempts < 3:
            attempts += 1
            state = self.refresh_state()
            preflight_error = self._validate_messages_thread_state(state, action_name="reply wait")
            if preflight_error is not None:
                self.latest_state = state
                raise HTTPException(status_code=409, detail=preflight_error)
            response = self._extract_new_message(
                messages=state.classification.recent_messages,
                baseline_counts=baseline_counts,
                sent_normalized=sent_normalized,
            )
            if response is not None:
                self.latest_state = state
                return response
            time.sleep(max(0.0, poll_interval_seconds))
        return None

    def _message_counts(self, messages: list[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for message in messages:
            normalized = self._normalize_text(message)
            counts[normalized] = counts.get(normalized, 0) + 1
        return counts

    def _message_after_sent(self, messages: list[str], *, sent_normalized: str) -> str | None:
        seen_sent = False
        for message in messages:
            normalized = self._normalize_text(message)
            if normalized == sent_normalized:
                seen_sent = True
                continue
            if seen_sent and normalized:
                return message
        return None

    def _sync_state_from_ui_hierarchy(self, state: ControllerState) -> None:
        if state.classification.package_name != "com.google.android.apps.messaging":
            return
        try:
            hierarchy = self.adb.dump_ui_hierarchy()
        except RuntimeError:
            return

        self._stabilize_messages_screen_from_hierarchy(state, hierarchy)
        if not state.classification.recent_messages:
            state.classification.recent_messages = self._extract_recent_messages_from_ui(hierarchy)
        state.compose_text = self._extract_compose_box_text_from_hierarchy(hierarchy)
        if state.classification.screen.value == "thread_view":
            contact_name = self._thread_contact_name_from_hierarchy(hierarchy) or state.classification.activity_name
            self._collect_thread_context(state, contact_name, hierarchy=hierarchy)
        elif state.classification.screen.value == "app_inbox":
            self._build_unread_queue(state)

        action_specs = {
            ApprovalActionType.TYPE_DRAFT: (
                "com.google.android.apps.messaging:id/compose_message_text",
                "Approve type only",
                "Focus the compose field and type a suggested reply without sending.",
            ),
            ApprovalActionType.SEND_MESSAGE: (
                "Compose:Draft:Send",
                "Approve send message",
                "Send the current draft in the active thread.",
            ),
            ApprovalActionType.OPEN_GALLERY: (
                "ComposeRowIcon:Gallery",
                "Approve open gallery",
                "Open the attachment or gallery picker from the current thread.",
            ),
        }
        existing_actions = {action.action_type: action for action in state.classification.available_actions}
        for action_type, (resource_id, label, description) in action_specs.items():
            bounds = self._extract_node_bounds(hierarchy, resource_id)
            if bounds is None:
                continue
            left, top, right, bottom = bounds
            point = DevicePoint(x=(left + right) // 2, y=(top + bottom) // 2)
            region = ScreenRegion(left=left, top=top, right=right, bottom=bottom)
            action = existing_actions.get(action_type)
            if action is None:
                state.classification.available_actions.append(
                    AvailableAction(
                        action_type=action_type,
                        label=label,
                        point=point,
                        region=region,
                        description=description,
                    )
                )
            else:
                action.point = point
                action.region = region

    def _stabilize_messages_screen_from_hierarchy(self, state: ControllerState, hierarchy: str) -> None:
        if self._hierarchy_looks_like_thread_view(hierarchy):
            state.classification.screen = ScreenName.THREAD_VIEW
            state.classification.confidence = max(state.classification.confidence, 0.96)
            state.classification.debug_info["hierarchy_screen_override"] = "thread_view"
            state.classification.debug_info["hierarchy_override_reason"] = "compose/send/message list detected"
            return
        if self._hierarchy_looks_like_inbox(hierarchy):
            state.classification.screen = ScreenName.APP_INBOX
            state.classification.confidence = max(state.classification.confidence, 0.92)
            state.classification.debug_info["hierarchy_screen_override"] = "app_inbox"
            state.classification.debug_info["hierarchy_override_reason"] = "conversation list detected"

    def _hierarchy_looks_like_thread_view(self, hierarchy: str) -> bool:
        indicators = (
            'resource-id="ConversationScreenUi"',
            'resource-id="message_list"',
            'resource-id="com.google.android.apps.messaging:id/compose_message_text"',
            'resource-id="Compose:Draft:Send"',
        )
        score = sum(1 for indicator in indicators if indicator in hierarchy)
        has_message_node = 'resource-id="message_text"' in hierarchy
        return score >= 3 and has_message_node

    def _hierarchy_looks_like_inbox(self, hierarchy: str) -> bool:
        indicators = (
            'resource-id="com.google.android.apps.messaging:id/conversation_list_root_container"',
            'resource-id="com.google.android.apps.messaging:id/home_fragment_container"',
        )
        if 'resource-id="com.google.android.apps.messaging:id/compose_message_text"' in hierarchy:
            return False
        return all(indicator in hierarchy for indicator in indicators)

    def _extract_recent_messages_from_ui(self, hierarchy: str) -> list[str]:
        entries = self._extract_thread_message_entries_from_hierarchy(hierarchy)
        if entries:
            return [entry["text"] for entry in entries[-6:]]
        messages: list[str] = []
        for match in re.finditer(r'<node[^>]*resource-id="message_text"[^>]*>', hierarchy):
            node = match.group(0)
            text_match = re.search(r'text="([^"]+)"', node)
            if text_match is None:
                continue
            value = html.unescape(text_match.group(1)).strip()
            if value:
                messages.append(value)
        return messages[-6:]

    def _extract_node_bounds(self, hierarchy: str, resource_id: str) -> tuple[int, int, int, int] | None:
        match = re.search(
            rf'resource-id="{re.escape(resource_id)}"[^>]*bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"',
            hierarchy,
        )
        if not match:
            return None
        return tuple(int(group) for group in match.groups())

    def _extract_new_message(
        self,
        *,
        messages: list[str],
        baseline_counts: dict[str, int],
        sent_normalized: str,
    ) -> str | None:
        seen_counts: dict[str, int] = {}
        for message in messages:
            normalized = self._normalize_text(message)
            seen_counts[normalized] = seen_counts.get(normalized, 0) + 1
            if normalized == sent_normalized:
                continue
            if seen_counts[normalized] > baseline_counts.get(normalized, 0):
                return message
        return None

    def _fail_compose_validation(
        self,
        current_state: ControllerState,
        text: str,
        reason: str,
        category: FailureCategory,
    ) -> ControllerState:
        result = ComposeValidationResult(
            success=False,
            validation_text=text,
            reason=reason,
            failure_category=category,
        )
        self._compose_validation_runs.append(result)
        self._last_compose_validation_result = result
        self._last_failure_category = category
        self._last_action = "compose validation"
        self._last_action_result = reason
        current_state.halted = True
        current_state.halt_reason = reason
        current_state.failure_category = category.value
        current_state.compose_validation = result
        current_state.last_action = self._last_action
        current_state.last_action_result = self._last_action_result
        self.latest_state = current_state
        self.log_store.record_event(
            event_type="compose_validation_failed",
            detected_screen=current_state.classification.screen.value,
            confidence=current_state.classification.confidence,
            planner_decision="compose_validation",
            approval_decision="compose validation",
            execution_result=reason,
            before_screenshot_path=current_state.screenshot_path,
            metadata={"failure_category": category.value, "validation_text": text},
        )
        return current_state
    def _fail_send_and_read(
        self,
        current_state: ControllerState,
        text: str,
        reason: str,
        category: FailureCategory,
    ) -> ControllerState:
        result = SendAndReadResult(
            success=False,
            sent_text=text,
            reason=reason,
            failure_category=category,
        )
        self._last_send_and_read_result = result
        self._last_failure_category = category
        self._last_action = "send and read"
        self._last_action_result = reason
        current_state.halted = True
        current_state.halt_reason = reason
        current_state.failure_category = category.value
        current_state.send_and_read = result
        current_state.last_action = self._last_action
        current_state.last_action_result = self._last_action_result
        self.latest_state = current_state
        self.log_store.record_event(
            event_type="send_and_read_failed",
            detected_screen=current_state.classification.screen.value,
            confidence=current_state.classification.confidence,
            planner_decision="send_and_read",
            approval_decision="send and read",
            execution_result=reason,
            before_screenshot_path=current_state.screenshot_path,
            metadata={"failure_category": category.value, "sent_text": text},
        )
        return current_state
