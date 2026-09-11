from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from difflib import SequenceMatcher
from typing import Any
from urllib import error, request

from pydantic import BaseModel, Field

from apps.training.store import ConversationIntelligenceStore
from libs.drafting.critic import critique_candidate
from libs.drafting.contact_profiles import ContactProfile, get_contact_profile, load_contact_profiles
from libs.drafting.conversation_agenda import ConversationAgenda, build_conversation_agenda, score_reply_against_agenda
from libs.drafting.conversation_policy import ConversationPolicy, build_conversation_policy, score_reply_against_policy
from libs.drafting.conversation_scene import ConversationScene, build_conversation_scene, score_reply_against_scene
from libs.drafting.conversation_state import (
    ConversationStateStore,
    analyze_bot_reply,
    extract_question_text,
    is_question_like_text,
    normalize_question_intent,
)
from libs.drafting.embeddings import embed_example_text
from libs.drafting.intents import (
    SAFE_GREETING_REPLIES,
    answers_greeting,
    classify_intent,
    contains_suspicious_phrase,
    is_simple_greeting,
    normalize_intent_label,
    normalize_text,
)
from libs.drafting.identity_pack import IdentityPack, load_identity_pack
from libs.drafting.model_providers import ModelMessage, generate_with_fallback, get_provider
from libs.drafting.question_debt import QuestionDebt, detect_question_debt, question_debt_reply_pool, score_reply_against_question_debt
from libs.drafting.reply_quality_judge import judge_reply_obligation
from libs.drafting.privacy_guard import check_external_api_allowed
from libs.drafting.relationships import classify_relationship, ensure_contact_overrides, load_contact_overrides
from libs.drafting.retrieval import RetrievedExample, retrieve_similar_examples_from_rows
from libs.drafting.retrieval_backends import RetrievalConfig, create_retrieval_backend, load_retrieval_config
from libs.drafting.style_profile import (
    ReplyExample,
    StyleProfile,
    build_reply_examples_from_training_dir,
    build_style_profile_from_training_dir,
    load_style_profile,
)
from libs.drafting.style_evaluator import evaluate_candidate_style
from libs.drafting.style_ranker import TrainingStyleRanker
from libs.drafting.thread_memory import ThreadMemoryStore
from libs.drafting.training_data import (
    ensure_training_message_files,
    load_corrections,
    load_all_training_rows,
    load_jsonl,
    load_training_messages,
    normalize_relationship_type,
)


class ApprovedPhoto(BaseModel):
    photo_id: str
    path: str
    description: str
    tags: list[str] = Field(default_factory=list)


class DraftScoreBreakdown(BaseModel):
    screen_confidence: float = 0.0
    context_confidence: float = 0.0
    persona_confidence: float = 0.0
    style_confidence: float = 0.0
    semantic_confidence: float = 0.0
    action_safety: float = 0.0
    risk_score: float = 0.0
    final_confidence: float = 0.0


class DraftCandidate(BaseModel):
    text: str
    sequence: list[str] = Field(default_factory=list)
    candidate_kind: str = "most_natural"
    candidate_type: str = "natural"
    intent: str = "casual"
    risk_flags: list[str] = Field(default_factory=list)
    rationale: str = ""
    persona: str = "casual_friend"
    relationship_type: str = "unknown"
    retrieved_examples_count: int = 0
    retrieved_examples: list[dict[str, object]] = Field(default_factory=list)
    retrieval_ids_used: list[str] = Field(default_factory=list)
    critic_scores: dict[str, object] = Field(default_factory=dict)
    final_decision: str = "review"
    blocked_reason: str | None = None
    why_this_matches: str = ""
    auto_send_allowed: bool = False
    generation_attempt: int = 1
    diversity_mode: str = "natural"
    estimated_style_score: int = 0
    estimated_relevance_score: int = 0
    estimated_risk_score: int = 0
    provider: str = "deterministic"
    model: str = "fallback"
    latency_ms: int | None = None
    external_api_used: bool = False
    external_api_blocked: bool = False
    provider_configured: bool = False
    assistant_likeness_score: float = 0.0
    naturalness_score: float = 0.0
    style_score: float = 0.0
    risk_score: float = 0.0
    stance_consistency_score: float = 0.0
    repetition_penalty: float = 0.0
    banter_fit_score: float = 0.0
    relationship_boldness_score: float = 0.0
    question_mode: str = "none"
    question_reason: str = ""
    question_usefulness_score: float = 0.0
    question_naturalness_score: float = 0.0
    question_pressure_penalty: float = 0.0
    repeated_question_penalty: float = 0.0
    recent_bot_questions_count: int = 0
    hook_mode: str = "none"
    hook_reason: str = ""
    hook_required: bool = False
    hook_presence_score: float = 0.0
    hook_relevance_score: float = 0.0
    hook_naturalness_score: float = 0.0
    hook_pressure_penalty: float = 0.0
    hook_safety_penalty: float = 0.0
    conversation_momentum_score: float = 0.0
    dead_reply_penalty: float = 0.0
    invalid_provider_output: bool = False
    fallback_used: bool = False
    low_information_incoming: bool = False
    continuation_required: bool = False
    continuation_score: float = 0.0
    mirror_reply_penalty: float = 0.0
    reply_energy_mode: str = "casual"
    energy_reason: str = ""
    filler_penalty: float = 0.0
    expressive_score: float = 0.0
    criticism_detected: bool = False
    lazy_lol_penalty: float = 0.0
    repeated_question_detected: bool = False
    copycat_penalty: float = 0.0
    copycat_detected: bool = False
    repair_required: bool = False
    escalation_detected: bool = False
    romantic_term_blocked: bool = False
    generated_by_provider: bool = False
    provider_name: str = "deterministic"
    provider_model: str = "fallback"
    provider_error: str | None = None
    provider_latency_ms: int | None = None
    provider_assisted_continuation: bool = False
    provider_prompt_mode: str = "normal"
    recent_user_activity: str = ""
    recent_user_mood: str = ""
    recent_user_callout: str = ""
    recently_answered_question_detected: bool = False
    already_answered_question_penalty: float = 0.0
    repair_over_hook_applied: bool = False
    hook_suppressed_reason: str = ""
    semantic_mismatch_penalty: float = 0.0
    context_grounding_score: float = 0.0
    context_fit_score: float = 0.0
    invalid_for_context: bool = False
    invalid_context_reason: str = ""
    latest_message_meaning: str = "normal"
    meaning_priority: int = 0
    emotional_context_detected: bool = False
    affection_detected: bool = False
    hurt_detected: bool = False
    confusion_detected: bool = False
    serious_callout_detected: bool = False
    emotional_fit_score: float = 0.0
    affection_response_score: float = 0.0
    hurt_repair_score: float = 0.0
    shallow_repair_penalty: float = 0.0
    emotional_absence_penalty: float = 0.0
    activity_grounding_suppressed: bool = False
    activity_grounding_suppressed_reason: str = ""
    conversation_function: str = "normal"
    conversation_function_reason: str = ""
    social_risk_tolerance: str = "normal"
    recent_life_update_fit_score: float = 0.0
    stale_self_state_penalty: float = 0.0
    repeated_reply_callout_detected: bool = False
    mocking_after_repair_detected: bool = False
    context_failure_question_detected: bool = False
    over_safe_penalty: float = 0.0
    bland_repair_penalty: float = 0.0
    conversational_function_fit_score: float = 0.0
    emotional_reciprocity_score: float = 0.0
    care_checkin_fit_score: float = 0.0
    affection_missed_penalty: float = 0.0
    stale_emotional_reply_penalty: float = 0.0
    emotional_request_unanswered: bool = False
    scene_type: str = "normal"
    scene_summary: str = ""
    latest_user_ask: str = ""
    latest_user_emotion: str = ""
    social_task: str = ""
    required_reply_move: str = ""
    forbidden_reply_moves: list[str] = Field(default_factory=list)
    known_recent_facts: list[str] = Field(default_factory=list)
    recent_bot_mistakes: list[str] = Field(default_factory=list)
    unresolved_user_points: list[str] = Field(default_factory=list)
    previous_bot_claim: str = ""
    previous_bot_claim_type: str = ""
    latest_user_refers_to_previous_bot_claim: bool = False
    explanation_required: bool = False
    explanation_target: str = ""
    active_topic: str = ""
    topic_was_provided: bool = False
    topic_value: str = ""
    bot_repeated_itself: bool = False
    user_called_out_bot: bool = False
    affection_reciprocity_required: bool = False
    care_response_required: bool = False
    topic_engagement_required: bool = False
    scene_fit_score: float = 0.0
    required_move_satisfied: bool = False
    forbidden_move_violated: bool = False
    scene_mismatch_reason: str = ""
    contextual_specificity_score: float = 0.0
    human_likeness_score: float = 0.0
    canned_reply_penalty: float = 0.0
    stale_template_penalty: float = 0.0
    deterministic_fallback_penalty: float = 0.0
    fallback_scene_type: str = ""
    fallback_required_move: str = ""
    fallback_scene_mismatch: bool = False
    semantic_contamination_penalty: float = 0.0
    previous_claim_explanation_score: float = 0.0
    thread_memory_match_score: float = 0.0
    retrieved_thread_count: int = 0
    repeated_failed_pattern_penalty: float = 0.0
    successful_pattern_match_score: float = 0.0
    unresolved_thread_point_score: float = 0.0
    thread_memory_used: bool = False
    identity_question_detected: bool = False
    identity_fact_used: str = ""
    identity_disclosure_allowed: bool = False
    identity_answer_score: float = 0.0
    identity_hallucination_risk: float = 0.0
    ignored_identity_question_penalty: float = 0.0
    stale_identity_fallback_penalty: float = 0.0
    direct_identity_answer_required: bool = False
    identity_specificity_score: float = 0.0
    human_scene_response_score: float = 0.0
    conversational_presence_score: float = 0.0
    directness_score: float = 0.0
    specificity_score: float = 0.0
    has_unanswered_user_question: bool = False
    unanswered_question_type: str = ""
    unanswered_question_text: str = ""
    question_debt_age_turns: int = 0
    user_called_out_unanswered_question: bool = False
    answer_required_now: bool = False
    question_debt_answer_score: float = 0.0
    ignored_question_debt_penalty: float = 0.0
    asked_new_question_before_answering_penalty: float = 0.0
    agenda_state: str = "normal_flow"
    next_dialogue_move: str = "answer_directly"
    bot_loop_detected: bool = False
    anti_loop_required: bool = False
    exhausted_prompt_types: list[str] = Field(default_factory=list)
    generic_prompt_penalty: float = 0.0
    repeated_prompt_penalty: float = 0.0
    agenda_fit_score: float = 0.0
    forbidden_dialogue_move_violated: bool = False
    forbidden_dialogue_moves_violated: list[str] = Field(default_factory=list)
    conversation_progress_score: float = 0.0
    chosen_topic: str = ""
    conversation_job: str = "normal_reply"
    policy_reason: str = ""
    bot_obligation: str = ""
    policy_must_answer: list[str] = Field(default_factory=list)
    policy_must_acknowledge: list[str] = Field(default_factory=list)
    policy_must_repair: list[str] = Field(default_factory=list)
    policy_must_avoid: list[str] = Field(default_factory=list)
    policy_fit_score: float = 0.0
    policy_must_satisfied: bool = True
    policy_violation: bool = False
    policy_violation_reasons: list[str] = Field(default_factory=list)
    policy_conversation_progress_score: float = 0.0
    obligation_satisfied: bool = True
    quality_gate_penalty: float = 0.0
    quality_gate_reasons: list[str] = Field(default_factory=list)
    training_style_score: float = 0.0
    training_style_penalty: float = 0.0
    training_style_reasons: list[str] = Field(default_factory=list)
    training_style_examples_loaded: int = 0
    active_topic_engagement_score: float = 0.0
    direct_answer_score: float = 0.0
    repair_specificity_score: float = 0.0
    emotional_presence_score: float = 0.0
    unresolved_point_addressed: bool = True
    selected_reason: str = ""
    score_breakdown: DraftScoreBreakdown = Field(default_factory=DraftScoreBreakdown)


class DraftBundle(BaseModel):
    summary: str
    reply_suggestions: list[str] = Field(default_factory=list)
    reply_sequences: list[list[str]] = Field(default_factory=list)
    reply_candidates: list[DraftCandidate] = Field(default_factory=list)
    recommended_reply_index: int | None = None
    auto_send_blocked_reason: str | None = None
    relationship_type: str = "unknown"
    retrieved_examples_count: int = 0
    retrieved_examples: list[dict[str, object]] = Field(default_factory=list)
    final_decision: str = "review"
    blocked_reason: str | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    prompt_preview: str | None = None
    intent_type: str = "unknown"
    generation_attempt: int = 1
    diversity_mode: str = "natural"
    retrieval_backend: str = "lexical"
    vector_db_available: bool = False
    vector_results_count: int = 0
    lexical_fallback_used: bool = False
    retrieval_scores: list[float] = Field(default_factory=list)
    retrieval_reasons: list[str] = Field(default_factory=list)
    filters_applied: list[str] = Field(default_factory=list)
    examples_rejected_by_filter_count: int = 0
    vector_search_ms: float = 0.0
    rerank_ms: float = 0.0
    photo_suggestions: list[ApprovedPhoto] = Field(default_factory=list)
    provider_metadata: dict[str, object] = Field(default_factory=dict)


DIVERSITY_MODES = {
    "conservative",
    "natural",
    "closest_style",
    "alternative_wording",
    "shorter",
    "more_direct",
}

CANDIDATE_TYPE_BY_MODE = {
    "conservative": "safe",
    "natural": "natural",
    "closest_style": "closest_style",
    "alternative_wording": "alternative",
    "shorter": "shorter",
    "more_direct": "direct",
}


def _retrieval_debug_fields(examples: list[RetrievedExample]) -> dict[str, object]:
    backend = examples[0].backend if examples else "lexical"
    vector_count = sum(1 for example in examples if example.backend == "vector_chroma")
    return {
        "retrieval_backend": backend,
        "vector_db_available": backend == "vector_chroma" or vector_count > 0,
        "vector_results_count": vector_count,
        "lexical_fallback_used": bool(examples) and vector_count == 0,
        "retrieval_scores": [example.score for example in examples],
        "retrieval_reasons": [example.reason for example in examples],
        "filters_applied": [],
        "examples_rejected_by_filter_count": 0,
    }


class DraftingService:
    _DISALLOWED_REPLY_PATTERNS = (
        "as an ai",
        "i am an ai",
        "i'm an ai",
        "language model",
        "phone owner",
        "the user",
        "user to confirm",
        "cannot receive",
        "can't receive",
        "can't help",
        "cannot help",
        "i can't",
        "i cannot",
        "sorry",
        "apolog",
    )
    _ASSISTANT_STYLE_CONTEXT_PATTERNS = (
        "that works for me",
        "sounds good",
        "i can do that",
        "i can handle it",
        "let me know if timing changes",
        "happy to help",
        "regarding",
        "concerning",
        "in response to",
        "i'll keep you posted",
    )

    def __init__(
        self,
        template_path: Path,
        approved_photos_dir: Path,
        *,
        ai_reply_enabled: bool = False,
        ai_reply_backend: str = "ollama",
        ai_reply_model: str = "qwen3:4b",
        ai_reply_base_url: str = "http://127.0.0.1:11434/api/chat",
        ai_reply_timeout_seconds: float = 20.0,
        ai_reply_system_prompt: str | None = None,
        ai_reply_fallback_models: list[str] | None = None,
        style_profile_path: Path | None = None,
        training_dir: Path | None = None,
        training_messages_dir: Path | None = None,
        contact_overrides_path: Path | None = None,
        intelligence_db_path: Path | None = None,
        training_owner_aliases: list[str] | None = None,
        draft_provider: str = "ollama",
        fast_provider: str = "ollama",
        router_provider: str = "ollama",
        private_provider: str = "ollama",
        gemini_api_key: str = "",
        groq_api_key: str = "",
        openrouter_api_key: str = "",
        huggingface_api_key: str = "",
        ollama_api_key: str = "",
        gemini_model: str = "gemini-2.5-flash",
        groq_model: str = "llama-3.1-8b-instant",
        openrouter_model: str = "google/gemini-2.0-flash-exp:free",
        huggingface_model: str = "",
        ollama_model: str = "qwen3:4b",
        ollama_base_url: str | None = None,
        external_api_enabled: bool = True,
        external_api_allow_sensitive: bool = False,
        external_api_max_context_messages: int = 6,
        external_api_timeout_seconds: float = 25.0,
    ) -> None:
        self.template_path = template_path
        self.approved_photos_dir = approved_photos_dir
        self.ai_reply_enabled = ai_reply_enabled
        self.ai_reply_backend = ai_reply_backend
        self.ai_reply_model = ai_reply_model
        self.ai_reply_fallback_models = [
            model.strip()
            for model in (ai_reply_fallback_models or ["mistral:latest", "dolphin-llama3:latest", "qwen3:0.6b"])
            if model.strip()
        ]
        self.ai_reply_base_url = ai_reply_base_url
        self.ai_reply_timeout_seconds = ai_reply_timeout_seconds
        self.draft_provider = draft_provider
        self.fast_provider = fast_provider
        self.router_provider = router_provider
        self.private_provider = private_provider
        self.gemini_api_key = gemini_api_key
        self.groq_api_key = groq_api_key
        self.openrouter_api_key = openrouter_api_key
        self.huggingface_api_key = huggingface_api_key
        self.ollama_api_key = ollama_api_key
        self.gemini_model = gemini_model
        self.groq_model = groq_model
        self.openrouter_model = openrouter_model
        self.huggingface_model = huggingface_model
        self.ollama_model = ollama_model or ai_reply_model
        self.ollama_base_url = ollama_base_url or ai_reply_base_url
        self.external_api_enabled = external_api_enabled
        self.external_api_allow_sensitive = external_api_allow_sensitive
        self.external_api_max_context_messages = external_api_max_context_messages
        self.external_api_timeout_seconds = external_api_timeout_seconds
        self.ai_reply_system_prompt = ai_reply_system_prompt or (
            "You write short text-message replies in first person as if you are the phone owner texting back. "
            "Match the tone, slang, spelling style, and energy of the visible conversation. "
            "Stay grounded in the latest message and never invent intimacy or flirtation. "
            "You are not an assistant. Do not apologize, refuse, mention AI, mention the user, "
            "or explain anything. Return strict JSON with keys \"summary\" and \"replies\", "
            "where \"replies\" is an array of exactly 3 short natural text messages."
        )
        self.templates = self._load_templates()
        self.photo_index = self._load_photo_index()
        self.style_profile = self._load_style_profile(
            style_profile_path=style_profile_path,
            training_dir=training_dir,
            training_owner_aliases=training_owner_aliases or [],
        )
        self.training_messages_dir = training_messages_dir or Path("data/training_messages")
        ensure_training_message_files(self.training_messages_dir)
        self.contact_overrides_path = contact_overrides_path or Path("data/contact_overrides.json")
        ensure_contact_overrides(self.contact_overrides_path)
        self._contact_overrides = load_contact_overrides(self.contact_overrides_path)
        data_root = self.training_messages_dir.parent if training_messages_dir is not None else template_path.parent
        self.data_root = data_root
        self.style_rubric_path = data_root / "style_rubric.json"
        self.negative_patterns_path = data_root / "style_negative_patterns.json"
        self.negative_patterns = self._load_negative_patterns(self.negative_patterns_path)
        self.contact_profiles_path = data_root / "contact_profiles.json"
        self.conversation_state = ConversationStateStore(data_root / "conversation_state.json")
        self.thread_memory = ThreadMemoryStore(data_root / "thread_memory")
        self.identity_pack_path = data_root / "identity" / "identity_pack.json"
        self.identity_pack: IdentityPack = load_identity_pack(self.identity_pack_path)
        self.retrieval_config = load_retrieval_config(data_root / "retrieval_config.json")
        self.high_quality_style_examples_path = self.training_messages_dir / "high_quality_style_examples.jsonl"
        self._training_rows = [
            *load_training_messages(self.training_messages_dir),
            *self._load_high_quality_style_examples(self.high_quality_style_examples_path),
        ]
        self.style_ranker = TrainingStyleRanker.from_examples(self._training_rows)
        self._correction_rows = load_corrections(self.training_messages_dir)
        self.intelligence_store = self._load_intelligence_store(intelligence_db_path)
        self._last_prompt_preview: str | None = None
        self.reply_examples = self._load_reply_examples(
            training_dir=training_dir,
            training_owner_aliases=training_owner_aliases or [],
        )

    def _load_high_quality_style_examples(self, path: Path) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for row in load_jsonl(path):
            incoming = str(row.get("incoming") or "").strip()
            reply = str(row.get("my_reply") or "").strip()
            if not incoming or not reply or bool(row.get("contains_sensitive")):
                continue
            context = row.get("context", [])
            if not isinstance(context, list):
                context = []
            relationship_type = normalize_relationship_type(str(row.get("relationship_type") or "unknown"))
            rows.append(
                {
                    "relationship_type": relationship_type,
                    "incoming": incoming,
                    "context": [str(item) for item in context if str(item).strip()][-4:],
                    "my_reply": reply,
                    "intent_type": normalize_intent_label(str(row.get("intent_type") or classify_intent(incoming, context))),
                    "notes": "high quality extracted style example",
                    "_source": "high_quality_style",
                    "source": row.get("source", "high_quality_style"),
                    "style_authority": row.get("style_authority", "high"),
                    "quality_score": row.get("quality_score", 0.0),
                }
            )
        return rows

    def _state_key(self, contact_name: str | None) -> str:
        return contact_name or "__unknown__"

    def _contact_profile_for(self, contact_name: str | None) -> ContactProfile | None:
        return get_contact_profile(contact_name, self.contact_profiles_path)

    def _conversation_state_for(self, contact_name: str | None):
        return self.conversation_state.get_contact_state(self._state_key(contact_name))

    def _conversation_memory_fragment(self, contact_name: str | None) -> str:
        state = self._conversation_state_for(contact_name)
        lines = ["Conversation memory:"]
        if state.last_bot_stance:
            lines.append(f"- last_bot_stance: {state.last_bot_stance}")
        if state.last_bot_claim:
            lines.append(f"- last_bot_claim: {state.last_bot_claim}")
        if state.last_bot_question:
            lines.append(f"- last_bot_question: {state.last_bot_question}")
        if state.disputed_topic:
            lines.append(f"- disputed_topic: {state.disputed_topic}")
        lines.append(f"- contradiction_risk: {round(float(state.contradiction_risk or 0.0), 3)}")
        lines.append(f"- recent_bot_questions_count: {int(state.recent_bot_questions_count or 0)}")
        lines.append(f"- repeated_question_risk: {round(float(state.repeated_question_risk or 0.0), 3)}")
        if state.recent_bot_replies:
            lines.append("- recent_bot_replies: " + " | ".join(state.recent_bot_replies[:3]))
        return "\n".join(lines)

    def _recent_fact_context(self, recent_messages: list[str], *, contact_name: str | None) -> dict[str, object]:
        state = self._conversation_state_for(contact_name)
        user_messages = [message.strip() for message in recent_messages if message and message.strip()][-5:]
        latest = user_messages[-1] if user_messages else ""
        blob = " ".join(user_messages)
        normalized_latest = normalize_text(latest)
        normalized_blob = normalize_text(blob)
        activity = ""
        activity_terms = (
            ("in bed", ("in bed", "bed rn", "laying in bed", "lying in bed")),
            ("sleeping", ("sleeping", "going sleep", "go sleep", "probably sleeping", "prolly sleeping", "sleep rn", "nap rn")),
            ("chilling", ("chilling", "chillin", "just chilling")),
            ("nothing", ("nothing", "not much", "not a lot", "not alot", "nm")),
            ("at home", ("at home", "home rn", "im home", "i'm home")),
            ("out", ("im out", "i'm out", "outside", "out rn")),
            ("working", ("working", "at work", "work rn", "just work", "doing work", "on work")),
            ("studying", ("studying", "revising", "uni work")),
        )
        for label, terms in activity_terms:
            if any(term in normalized_blob for term in terms):
                activity = label
        mood = ""
        mood_terms = (
            ("bored", ("bored", "dry", "dead convo", "dead chat")),
            ("tired", ("tired", "sleepy", "knackered")),
            ("annoyed", ("annoyed", "mad", "pissed", "wtf", "taking the piss")),
            ("confused", ("confused", "confusing", "wdym", "what are u on about", "what are you on about")),
        )
        for label, terms in mood_terms:
            if any(term in normalized_blob for term in terms):
                mood = label
        callout = ""
        callout_terms = (
            ("already_told_you", ("i js told u", "i just told u", "i just told you", "already told u", "already told you", "u js asked that", "you just asked that")),
            ("copying", ("why u copying", "why you copying", "stop copying")),
            ("waffling", ("waffling", "waffle", "chatting rubbish")),
            ("confusing", ("ur confusing me", "you're confusing me", "youre confusing me", "confusing me")),
            ("dry", ("ur dry", "you're dry", "youre dry", "why u being so dry", "too dry")),
            ("taking_the_piss", ("taking the piss", "acting dumb")),
            ("direct_mad_question", ("r u mad", "are u mad", "you mad")),
        )
        for label, terms in callout_terms:
            if any(term in normalized_latest for term in terms):
                callout = label
                break
        recent_question_intents = [
            normalize_question_intent(question)
            for question in getattr(state, "recent_bot_questions", [])[:5]
            if question
        ]
        if state.last_bot_question:
            recent_question_intents.insert(0, normalize_question_intent(state.last_bot_question))
        answered_intents: list[str] = []
        if "what_u_doing" in recent_question_intents and activity:
            answered_intents.append("what_u_doing")
        if any(term in normalized_latest for term in (" u", " wbu", " wby", " you")) and activity:
            answered_intents.append("what_u_doing")
        answered_intents = list(dict.fromkeys(answered_intents))
        return {
            "recent_user_activity": activity,
            "recent_user_mood": mood,
            "recent_user_callout": callout,
            "recently_answered_question_intents": answered_intents,
            "recently_answered_question_detected": bool(answered_intents),
        }

    def _latest_meaning_context(
        self,
        recent_messages: list[str],
        *,
        contact_name: str | None,
        relationship_type: str,
        incoming_intent: str,
    ) -> dict[str, object]:
        latest = recent_messages[-1] if recent_messages else ""
        normalized = normalize_text(latest)
        input_category = self._input_category(latest)
        affection_terms = ("i missed u", "i missed you", "missed u", "missed you", "miss you", "i miss you", "baby", "babe", "my love", "love u", "love you", "i love you", "need u", "want u", "mwah", "my handsome", "handsome")
        flirty_terms = ("baby", "babe", "my love", "love u", "love you", "i love you", "need u", "want u", "mwah", "my handsome", "handsome")
        hurt_terms = ("playing w my feelings", "playing with my feelings", "youre hurting me", "you're hurting me", "u hurt me", "you hurt me", "why are u being dry", "why u being dry", "i dont know why youre being dry", "i don't know why you're being dry")
        serious_terms = ("this isnt funny", "this isn't funny", "dont do this", "don't do this", "this is serious", "stop playing", "are u taking the piss", "are you taking the piss", "taking the piss")
        confusion_terms = ("im confused", "i'm confused", "ur confusing me", "youre confusing me", "you're confusing me", "this is confusing", "what are u on about", "what are you on about", "why u confused", "why you confused")
        repair_terms = ("why did u say", "why did you say", "that makes no sense", "i js told u", "i just told you", "u js asked that", "you just asked that")
        life_question_terms = ("what u been up to", "what you been up to", "what have u been up to", "what u been doing", "what you been doing", "wyd", "wuu2", "hru", "how are u", "how are you")
        reciprocity_terms = ("aint gon say it back", "ain't gon say it back", "arent u gonna say it back", "aren't u gonna say it back", "say it back", "u dont miss me", "u don't miss me", "dont u miss me", "don't u miss me")
        missed_affection_terms = ("asked u to say u miss me", "asked you to say you miss me", "i said i missed u", "i said i missed you", "i said i miss u", "i said i miss you", "bro i said i miss u", "bro i said i missed u", "bro i said i missed you")
        care_terms = ("are u ok", "r u ok", "u seem off", "you seem off", "are you alright", "are u alright", "u good", "you good", "are u good", "are you good", "are you ok")
        affection_detected = any(term in normalized for term in affection_terms)
        flirty_affection = affection_detected and any(term in normalized for term in flirty_terms)
        romantic_signoff_detected = self._has_romantic_signoff_text(normalized) and (
            relationship_type == "romantic_interest"
            or affection_detected
            or self._flirt_allowed(contact_name, relationship_type)
        )
        hurt_detected = any(term in normalized for term in hurt_terms)
        serious_detected = any(term in normalized for term in serious_terms)
        confusion_detected = any(term in normalized for term in confusion_terms)
        repair_detected = any(term in normalized for term in repair_terms) or input_category == "repair"
        mixed_life_affection = affection_detected and any(term in normalized for term in life_question_terms)
        reciprocity_detected = any(term in normalized for term in reciprocity_terms)
        missed_affection_detected = any(term in normalized for term in missed_affection_terms)
        care_detected = any(term in normalized for term in care_terms)
        if hurt_detected:
            meaning = "hurt_feelings"
            priority = 7
        elif serious_detected:
            meaning = "serious_boundary"
            priority = 7
        elif missed_affection_detected:
            meaning = "missed_affection_callout"
            priority = 6
        elif reciprocity_detected:
            meaning = "emotional_reciprocity_request"
            priority = 6
        elif care_detected:
            meaning = "care_checkin"
            priority = 5
        elif mixed_life_affection:
            meaning = "mixed_life_update_and_affection"
            priority = 5
        elif confusion_detected:
            meaning = "confusion"
            priority = 6
        elif repair_detected:
            meaning = "repair_callout"
            priority = 6
        elif romantic_signoff_detected:
            meaning = "romantic_signoff"
            priority = 6
        elif flirty_affection:
            meaning = "flirty_affection"
            priority = 5
        elif affection_detected:
            meaning = "affection"
            priority = 5
        elif input_category == "check_in":
            meaning = "check_in"
            priority = 4
        elif is_question_like_text(latest):
            meaning = "direct_question"
            priority = 4
        elif input_category == "greeting":
            meaning = "greeting"
            priority = 3
        elif self._is_low_information_incoming(latest):
            meaning = "low_info"
            priority = 1
        elif self._recent_fact_context(recent_messages, contact_name=contact_name).get("recent_user_activity"):
            meaning = "activity_update"
            priority = 2
        elif incoming_intent == "banter_challenge":
            meaning = "banter"
            priority = 2
        else:
            meaning = "normal"
            priority = 0
        emotional = meaning in {"affection", "flirty_affection", "romantic_signoff", "hurt_feelings", "confusion", "serious_boundary", "repair_callout", "mixed_life_update_and_affection", "emotional_reciprocity_request", "missed_affection_callout", "care_checkin", "concern_about_bot", "affection_repair"}
        suppressed = emotional
        return {
            "latest_message_meaning": meaning,
            "meaning_priority": priority,
            "emotional_context_detected": emotional,
            "affection_detected": affection_detected,
            "hurt_detected": hurt_detected,
            "confusion_detected": confusion_detected,
            "serious_callout_detected": serious_detected,
            "emotional_reciprocity_request": reciprocity_detected,
            "missed_affection_callout": missed_affection_detected,
            "care_checkin_detected": care_detected,
            "romantic_signoff_detected": romantic_signoff_detected,
            "activity_grounding_suppressed": suppressed,
            "activity_grounding_suppressed_reason": "latest emotional/repair meaning outranks recent activity" if suppressed else "",
            "flirt_allowed": self._flirt_allowed(contact_name, relationship_type),
        }

    def _romantic_signoff_detected(
        self,
        recent_messages: list[str],
        *,
        contact_name: str | None,
        relationship_type: str,
    ) -> bool:
        latest = normalize_text(recent_messages[-1]) if recent_messages else ""
        context = normalize_text(" ".join(recent_messages[-8:]))
        affection_terms = ("love u", "love you", "miss u", "miss you", "mwah", "kiss", "baby", "babe", "my love", "my handsome", "handsome")
        has_signoff = self._has_romantic_signoff_text(latest)
        has_latest_affection = any(term in latest for term in affection_terms)
        has_thread_affection = any(term in context for term in affection_terms)
        return (
            has_signoff
            and (relationship_type == "romantic_interest" or self._flirt_allowed(contact_name, relationship_type))
            and (has_latest_affection or has_thread_affection)
        )

    def _has_romantic_signoff_text(self, text: str) -> bool:
        normalized = normalize_text(text).strip(" .?!,")
        if not normalized:
            return False
        if any(term in normalized for term in ("goodnight", "good night", "sleep well", "sleepwell", "sleep tight")):
            return True
        if re.search(r"\bgn\b", normalized):
            return True
        if re.search(r"\bnight\b", normalized) and any(
            term in normalized
            for term in ("baby", "babe", "my love", "love", "handsome", "mwah", " x", "xx")
        ):
            return True
        return False

    def _romantic_signoff_sequences(
        self,
        recent_messages: list[str],
        *,
        contact_name: str | None,
        relationship_type: str,
    ) -> list[list[str]]:
        if not self._romantic_signoff_detected(
            recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
        ):
            return []
        burst_context = normalize_text(" ".join(recent_messages[-3:]))
        has_miss = any(term in burst_context for term in ("miss u", "miss you"))
        has_love = any(term in burst_context for term in ("love u", "love you"))
        if has_miss and has_love:
            return [
                ["love u too", "miss u more", "goodnight baby"],
                ["miss u too baby", "sleep well"],
                ["love u too my love", "goodnight"],
            ]
        if has_miss:
            return [
                ["miss u more", "goodnight baby"],
                ["miss u too", "sleep well my love"],
                ["come here soon", "goodnight baby"],
            ]
        if has_love:
            return [
                ["love u too", "goodnight baby"],
                ["love u more", "sleep well my love"],
                ["goodnight baby", "love u"],
            ]
        return [
            ["goodnight baby", "sleep well"],
            ["sleep well my love", "miss u"],
            ["goodnight my love", "mwah"],
        ]

    def _emotional_reply_pool(
        self,
        recent_messages: list[str],
        *,
        contact_name: str | None,
        relationship_type: str,
        incoming_intent: str,
    ) -> list[str]:
        meaning = str(self._latest_meaning_context(
            recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )["latest_message_meaning"])
        flirt_allowed = self._flirt_allowed(contact_name, relationship_type)
        if meaning == "mixed_life_update_and_affection":
            if flirt_allowed:
                return self._unique_replies([
                    "missed u too icl, just been chilling",
                    "same icl, not much just been chilling",
                    "aww missed u too, just been chilling wbu",
                ])
            return self._unique_replies([
                "that's sweet icl, just been chilling",
                "aww bless u, not much just chilling",
                "appreciate u, just been chilling",
            ])
        if meaning == "emotional_reciprocity_request":
            if flirt_allowed:
                return self._unique_replies(["missed u too icl", "i was getting there, missed u too", "course i missed u"])
            return self._unique_replies(["that's sweet icl", "aww bless u", "i appreciate u"])
        if meaning == "missed_affection_callout":
            if flirt_allowed:
                return self._unique_replies(["yeah ur right, missed u too icl", "my bad, missed u too", "icl i clocked it late, missed u too"])
            return self._unique_replies(["yeah my bad, that was cold", "icl i missed what u meant there", "that's sweet icl"])
        if meaning == "romantic_signoff":
            sequences = self._romantic_signoff_sequences(
                recent_messages,
                contact_name=contact_name,
                relationship_type=relationship_type,
            )
            if sequences:
                return self._unique_replies([self._format_sequence_for_display(sequence) for sequence in sequences])
            if flirt_allowed:
                return self._unique_replies(["goodnight baby, sleep well", "love u too, goodnight", "sleep well my love"])
            return self._unique_replies(["goodnight sleep well", "sleep well", "goodnight"])
        if meaning in {"care_checkin", "concern_about_bot"}:
            if flirt_allowed:
                return self._unique_replies(["yeah im good ml, just been off today", "im good dw, just tired icl", "yeah im okay, my head's just gone"])
            return self._unique_replies(["yeah im good dw", "im okay, just tired", "yeah im alright"])
        if meaning == "flirty_affection":
            if flirt_allowed:
                return self._unique_replies(["missed u too icl", "same icl where u been", "aww missed u too"])
            if relationship_type in {"close_friend", "trusted_contact", "romantic_interest"}:
                return self._unique_replies(["aww bless u", "that's sweet icl", "appreciate u"])
            return self._unique_replies(["aww bless u", "that's sweet icl", "appreciate u"])
        if meaning == "affection":
            if relationship_type in {"close_friend", "trusted_contact", "romantic_interest"}:
                return self._unique_replies(["aww bless u", "that's sweet icl", "appreciate u"])
            return self._unique_replies(["aww bless u", "that's sweet icl", "appreciate u"])
        if meaning == "confusion":
            return self._unique_replies(["yeah fairs that reply made no sense", "icl i was waffling", "yh my bad that was confusing"])
        if meaning in {"hurt_feelings", "serious_boundary"}:
            return self._unique_replies([
                "yeah nah ur right my bad, i didn't mean to make u feel like that",
                "nah im not taking the piss, i get why that annoyed u",
                "icl i get why that felt off, my bad",
            ])
        return []

    def _social_risk_tolerance(self, contact_name: str | None) -> str:
        label = (contact_name or "").strip().casefold()
        if label in {"catbot", "training", "simulator", "simulation"} or label.startswith("catbot"):
            return "expressive"
        return "normal"

    def _conversation_scene(
        self,
        conversation_messages: list[str],
        *,
        contact_name: str | None,
        relationship_type: str,
        incoming_intent: str,
    ) -> ConversationScene:
        return build_conversation_scene(
            conversation_messages,
            relationship_type=relationship_type,
            flirt_allowed=self._flirt_allowed(contact_name, relationship_type),
            intent_type=incoming_intent,
            recent_bot_questions=getattr(self._conversation_state_for(contact_name), "recent_bot_questions", []),
        )

    def _conversation_agenda(
        self,
        conversation_messages: list[str],
        *,
        scene: ConversationScene,
        question_debt: QuestionDebt | None = None,
    ) -> ConversationAgenda:
        return build_conversation_agenda(
            conversation_messages,
            scene=scene,
            safe_interests=list(getattr(self.identity_pack, "interests", []) or []),
            question_debt=question_debt or detect_question_debt(conversation_messages),
        )

    def _conversation_policy(
        self,
        conversation_messages: list[str],
        *,
        scene: ConversationScene,
        agenda: ConversationAgenda,
        relationship_type: str,
        question_debt: QuestionDebt | None = None,
    ) -> ConversationPolicy:
        return build_conversation_policy(
            conversation_messages,
            scene=scene,
            agenda=agenda,
            relationship_type=relationship_type,
            question_debt=question_debt or detect_question_debt(conversation_messages),
        )

    def _question_debt(self, conversation_messages: list[str]) -> QuestionDebt:
        return detect_question_debt(conversation_messages)

    def _question_debt_reply_pool(self, question_debt: QuestionDebt) -> list[str]:
        return self._unique_replies(question_debt_reply_pool(question_debt))

    def _policy_reply_pool(self, policy: ConversationPolicy) -> list[str]:
        if policy.conversation_job == "answer_question_debt":
            return self._unique_replies(policy.suggested_reply_shapes)
        if policy.conversation_job == "acknowledge_repeated_reply":
            return self._unique_replies([
                "yh fairs i repeated myself icl",
                "caught me icl",
                "yeah my bad i repeated myself twice",
                "yh that was NPC behaviour",
            ])
        if policy.conversation_job == "react_to_story":
            return self._unique_replies([
                "BRO WHAT why was he running",
                "nah wait some random guy came in ur living room?",
                "what do u mean came in like into ur house?",
                "that is insane icl why was he running",
            ])
        if policy.conversation_job == "answer_care_check":
            return self._unique_replies([
                "yeah im good dw my replies were just weird",
                "im alright icl i was just answering like a weirdo",
                "yeah im okay my bad",
            ])
        if policy.conversation_job == "repair_after_weird_or_dry_reply":
            return self._unique_replies(policy.suggested_reply_shapes or [
                "yeah that was dry icl my bad",
                "yh that was weird icl my bad",
                "yeah fairs i answered that terribly",
                "nah ur right that made no sense",
                "icl i was waffling there my bad",
            ])
        if policy.conversation_job == "answer_affection":
            if policy.must_repair:
                return self._unique_replies([
                    "yeah ur right missed u too icl",
                    "my bad missed u too",
                    "icl i dodged that, missed u too",
                ])
            return self._unique_replies([
                *policy.suggested_reply_shapes,
            ])
        if policy.conversation_job == "answer_reciprocal_activity":
            if policy.must_repair:
                return self._unique_replies([
                    "my bad i missed the wby, not much just sorting stuff",
                    "yh my bad i ignored the wby, just working icl",
                    "icl i didnt answer u, nothing crazy just gym and coding",
                ])
            return self._unique_replies([
                "nothing much just chilling",
                "same just been chilling",
                "same icl just chilling",
            ])
        if policy.conversation_job == "respond_to_user_mood":
            if policy.user_emotion == "tired":
                return self._unique_replies(["why u tired", "long day?", "go nap then icl"])
            return self._unique_replies(["as u should why u happy", "good u deserve that icl", "thats cute icl whats got u happy"])
        if policy.conversation_job == "playful_acknowledge":
            return self._unique_replies(policy.suggested_reply_shapes)
        return []

    def _agenda_reply_pool(self, agenda: ConversationAgenda) -> list[str]:
        if agenda.agenda_state == "anti_loop_repair":
            return self._unique_replies([
                "yh fairs i asked that already",
                "icl im looping my bad",
                "yeah that was NPC behaviour",
                "my bad i keep asking the same thing",
                "yh allow me ill pick smth then",
            ])
        if agenda.agenda_state == "topic_selection_needed":
            return self._unique_replies([
                "alr random one then dream car?",
                "fine cars then what would u get if money wasnt a thing",
                "gym then what u training if u went rn",
                "alr lets talk properly then what's been on ur mind",
                "fine ill pick, cars or gym",
            ])
        if agenda.agenda_state == "dead_conversation_recovery":
            return self._unique_replies([
                "same icl this convo is dying",
                "alr random question then dream car?",
                "fine ill carry it, what's one thing u wanna do this year",
            ])
        if agenda.agenda_state == "identity_answer" and agenda.next_dialogue_move == "make_observation":
            return self._unique_replies(["twins then", "same age and still both boring", "valid"])
        if agenda.next_dialogue_move == "make_observation" and "empathy" in agenda.active_conversation_goal:
            return self._unique_replies(["yh icl just one of them days", "fr it was dead icl", "yh u get it"])
        if agenda.next_dialogue_move == "make_observation" and "activity" in agenda.active_conversation_goal:
            return self._unique_replies(["valid stay there icl", "fair stay there icl", "still in bed then"])
        if agenda.next_dialogue_move == "make_observation" and "awkward challenge" in agenda.active_conversation_goal:
            return self._unique_replies(["nothing icl i walked into that", "idk that was a dumb reply", "fair point icl"])
        if agenda.agenda_state == "user_bored_or_unengaged":
            return self._unique_replies([
                "same icl this convo is dying",
                "alr random question then dream car?",
                "fine ill carry it, what's been on ur mind",
            ])
        return []

    def _scene_reply_pool(self, scene: ConversationScene) -> list[str]:
        if scene.scene_type == "opening":
            latest_opening = normalize_text(scene.latest_user_message).strip(" .?!")
            if latest_opening == "hey":
                return self._unique_replies(["yo what u saying", "hey what u doing", "yo wdyll"])
            if latest_opening == "yo":
                return self._unique_replies(["yo how u been", "yes lad what u saying", "yo what u on"])
            return self._unique_replies(["yo how u been", "heyy u good", "yo what u on"])
        if scene.scene_type == "missed_affection_callout":
            if scene.flirt_allowed:
                return self._unique_replies(["yeah ur right, i should've said missed u too", "my bad, missed u too icl", "icl i dodged that by accident"])
            return self._unique_replies(["nah i didn't mean it like that", "that's sweet icl, i just replied badly", "my bad, i answered that weird"])
        if scene.scene_type == "reciprocal_current_activity_question":
            recent = {normalize_text(reply).strip(" .?!") for reply in scene.recent_bot_replies}
            pool = ["nothing much just chilling", "same just been chilling", "same icl just chilling"]
            if recent & {"nothing much just chilling", "same just chilling", "same icl", "fair just chilling too"}:
                pool = ["not much just sorting stuff", "js working icl", "nothing crazy just gym and coding"]
            return self._unique_replies(pool)
        if scene.scene_type == "reciprocal_topic_question":
            latest_norm = normalize_text(scene.latest_user_message)
            active_norm = normalize_text(scene.active_topic or "")
            if "car" in latest_norm or "car" in active_norm:
                return self._unique_replies(["r8 is cold icl id probs go urus", "mine would be an m4 icl", "id go rs6 or urus"])
            if "gym" in latest_norm or "boxing" in latest_norm:
                return self._unique_replies(["probably boxing icl", "gym wise push day icl", "id train back icl"])
            return self._unique_replies(["same icl id probs pick that too", "mine would be different icl", "id go for smth colder"])
        if scene.scene_type == "story_hypothetical_question":
            return self._unique_replies(["icl id be shouting who are u", "id probably panic then ask why hes running", "nah id be out the room so fast"])
        if scene.scene_type == "gym_training_question":
            return self._unique_replies(["probably push icl", "id train back icl", "boxing then weights probably"])
        if scene.scene_type == "owner_boxing_question":
            if "hard" in normalize_text(scene.latest_user_message):
                return self._unique_replies(["yeah boxing is hard icl", "yh its tiring but worth it", "yeah it drains u differently"])
            return self._unique_replies(["yeah i box icl", "yh boxing and gym", "yeah been boxing a bit"])
        if scene.scene_type == "playful_fight_challenge":
            return self._unique_replies(["nah id fold u behave", "you can try icl", "boxing rules or street rules"])
        if scene.scene_type == "owner_tech_interest_question":
            return self._unique_replies(["ai and software mostly", "software and backend stuff icl", "ai coding projects all of it"])
        if scene.scene_type == "owner_car_preference_question":
            return self._unique_replies(["urus or rs6 icl", "r8 is cold but id go urus", "m4 or rs6 probably"])
        if scene.scene_type == "car_preference_answer":
            return self._unique_replies(["urus is cold icl", "valid urus is serious", "yh urus is hard"])
        if scene.scene_type == "owner_dream_question":
            return self._unique_replies(["build smth serious and be comfortable icl", "make my projects work and look after my people", "get somewhere with software and business icl"])
        if scene.scene_type == "owner_prayer_question":
            return self._unique_replies(["yeah i pray icl", "yh i try to", "yeah alhamdulillah"])
        if scene.scene_type == "owner_tired_reason_question":
            return self._unique_replies(["boxing plus hella work icl", "gym boxing and clients draining me", "been coding and training all day im finished"])
        if scene.scene_type == "topic_choice_offered":
            latest_norm = normalize_text(scene.latest_user_message)
            if "car" in latest_norm:
                return self._unique_replies(["cars then dream car?", "cars icl what would u get if money wasnt a thing", "cars easy, urus or r8"])
            if "boxing" in latest_norm:
                return self._unique_replies(["boxing then, u ever tried it", "boxing icl its more interesting", "boxing easy, its hard but cold"])
            return self._unique_replies(["ill pick the first one icl", "first one then", "that first one sounds better"])
        if scene.scene_type == "topic_opinion":
            latest_norm = normalize_text(scene.latest_user_message)
            if "legs" in latest_norm:
                return self._unique_replies(["same legs are evil icl", "valid legs are pain", "legs day ruins people icl"])
            return self._unique_replies(["valid icl", "same tbh", "fair i get that"])
        if scene.scene_type == "topic_positive_acknowledgement":
            active = normalize_text(scene.active_topic or scene.topic_value)
            if any(term in active for term in ("tech", "software", "ai", "coding")):
                return self._unique_replies(["yh tech is cold icl", "proper cold when it works", "yeah software is serious icl"])
            if "car" in active:
                return self._unique_replies(["yh cars are cold icl", "proper dangerous topic for me", "yeah cars are too cold"])
            if "gym" in active or "boxing" in active:
                return self._unique_replies(["yh gym is cold icl", "proper love it when im consistent", "boxing is hard but cold"])
            return self._unique_replies(["yh its cold icl", "proper valid icl", "yeah thats hard"])
        if scene.scene_type == "user_activity_status":
            latest_norm = normalize_text(scene.latest_user_message)
            if "bed" in latest_norm:
                return self._unique_replies(["fair stay there icl", "valid stay there", "in bed already is crazy"])
            if "chilling" in latest_norm or "nothing" in latest_norm:
                return self._unique_replies(["valid sounds dead icl", "fair what u watching", "same kind of day icl"])
            return self._unique_replies(["fair enough icl", "valid", "yh fairs"])
        if scene.scene_type == "day_check_question":
            return self._unique_replies(["it was calm icl, bit dead", "not bad tbh just chilled", "decent icl nothing crazy", "long icl but calm"])
        if scene.required_reply_move == "answer_owner_day_activity":
            if scene.repair_required:
                return self._unique_replies([
                    "yh my bad i didnt answer u, gym and coding mostly",
                    "icl i answered wrong, just gym and coding today",
                    "my bad that made no sense, mostly gym coding and uni",
                ])
            return self._unique_replies([
                "gym and coding mostly icl wbu",
                "not much icl just gym and coding",
                "uni gym coding icl nothing crazy, what u been on",
            ])
        if scene.scene_type == "reciprocal_wellbeing_question":
            asked_wellbeing_recently = any(
                any(term in normalize_text(question) for term in ("how u been", "how are u", "how are you", "hru", "wbu", "wby"))
                for question in scene.recent_bot_questions
            )
            if asked_wellbeing_recently:
                return self._unique_replies(["im good icl", "im calm honestly", "im bless icl", "good icl happy for u tho"])
            return self._unique_replies(["im good wbu", "good u", "im calm wbu", "not bad icl wbu"])
        if scene.required_reply_move == "answer_wellbeing_checkin" and scene.repair_required:
            return self._unique_replies(["yh my bad im good icl", "icl i missed that im calm", "my bad im bless honestly"])
        if scene.scene_type == "positive_mood_update":
            return self._unique_replies(["as u should why u happy", "thats cute icl whats got u happy", "good u deserve that icl"])
        if scene.scene_type == "reciprocal_identity_answer":
            if normalize_text(scene.latest_user_message).strip(" .?!") == "same":
                return self._unique_replies(["twins then", "same age and still both boring", "valid"])
            return self._unique_replies(["oh fairs", "older than me icl", "21 yeah fairs"])
        if scene.scene_type == "owner_activity_detail_question":
            return self._unique_replies(["just gym icl bit of weights", "trained a bit then left icl", "just a quick workout nothing crazy"])
        if scene.scene_type == "owner_activity_clarification":
            return self._unique_replies(["lol i meant me coding", "nah i meant i was coding", "coding is me not u lol"])
        if scene.scene_type == "owner_claim_contradiction":
            return self._unique_replies(["yh my bad i worded that wrong, i do code", "not a lie i just answered wrong, i meant me coding", "icl that was confusing, i do code"])
        if scene.scene_type == "user_activity_update":
            return self._unique_replies(["busy with what", "what u been busy with", "doing what"])
        if scene.scene_type == "contradiction_callout":
            return self._unique_replies(["yh that made no sense icl", "yeah i contradicted myself there", "my bad that was dumb"])
        if scene.scene_type == "missed_user_fact_callout":
            return self._unique_replies(["yh my bad i missed that", "icl i ignored what u said there", "fairs i didn't clock it", "yeah my bad u did say that"])
        if scene.scene_type == "repair_clarification":
            return self._unique_replies(["i mean i missed what u said", "i meant i bugged and ignored ur message", "icl i answered the wrong thing"])
        if scene.scene_type == "low_info_after_bad_reply":
            return self._unique_replies(["yh that reply was dead icl", "icl i answered that badly", "ignore me im waffling"])
        if scene.scene_type == "explain_previous_bot_claim":
            if scene.previous_bot_claim_type == "day_summary":
                if "answer previous day activity question" in scene.unresolved_user_points:
                    return self._unique_replies(["nothing much icl just chilled", "didn't do much tbh", "just uni and chilling icl", "nothing really just one of them dead days"])
                return self._unique_replies(["just didn't do much icl", "nothing really happened tbh", "just one of them dead days", "was just boring icl"])
            if scene.previous_bot_claim_type == "weird_story_reaction":
                return self._unique_replies(["icl that was random, wrong context", "yeah ignore that i answered the wrong thing", "my bad that made no sense"])
            if scene.previous_bot_claim_type == "current_activity":
                return self._unique_replies(["i meant im not doing much", "just chilling basically", "nothing much icl"])
            if scene.previous_bot_claim_type == "random_topic_misread":
                return self._unique_replies(["my bad i called it random for no reason", "nah it wasn't random i misread it", "icl i worded that dumb"])
            if scene.previous_bot_claim_type == "owner_activity_summary":
                return self._unique_replies(["yeah icl been busy with uni gym coding", "yeah lowk had loads on", "fr uni gym coding all hit at once"])
            if scene.previous_bot_claim_type == "topic_statement":
                if "sounds dead" in normalize_text(scene.previous_bot_claim):
                    return self._unique_replies(["cos chilling with no plan sounds dead icl", "i meant it sounds like a dead day", "cos doing nothing gets boring icl"])
                return self._unique_replies(["i mean i worded that badly", "my bad i meant that different", "icl i didn't explain that"])
            return self._unique_replies(["i mean i answered that badly", "icl i worded that wrong", "my bad that was random"])
        if scene.scene_type == "light_acknowledgement":
            if normalize_text(scene.latest_user_message).strip(" .?!") == "behave":
                return self._unique_replies(["u behave", "nah u behave", "make me"])
            return self._unique_replies(["lool ur good", "yh ur good", "fairs fairs", "allow it lol"])
        if scene.scene_type == "tired_mood":
            return self._unique_replies(["same icl go sleep then", "why u tired", "go nap then", "long day?", "icl same im finished"])
        if scene.identity_answer_required:
            latest_norm = normalize_text(scene.latest_user_message)
            affection = any(term in latest_norm for term in ("missed u", "missed you", "miss you", "i miss"))
            if scene.identity_question_kind == "age" and "19" in normalize_text(scene.latest_user_message):
                return self._unique_replies(["same im 19", "im 19 too", "19 wby"])
            if scene.identity_question_kind == "recent_activity" and "what u been up to" in latest_norm:
                if affection and scene.flirt_allowed:
                    return self._unique_replies(["missed u too icl, just been chilling", "same icl, not much just been chilling", "aww missed u too, just been chilling wbu"])
                if affection:
                    return self._unique_replies(["that's sweet icl, just been chilling", "aww bless u, not much just chilling", "appreciate u, just been chilling"])
                return self._unique_replies([
                    "not much icl just been chilling wbu",
                    "nothing crazy icl wbu",
                    "just been chilling mostly what about u",
                    "been doing the usual icl",
                    "not much tbh, been a bit dead",
                ])
            return self._unique_replies(
                self.identity_pack.templates_for_scene(
                    scene.identity_question_kind,
                    scene.relationship_context,
                    flirt_allowed=scene.flirt_allowed,
                    affection=affection,
                    message_count=scene.thread_message_count,
                )
            )
        if scene.scene_type == "topic_given" and scene.topic_value:
            topic = scene.topic_value
            topic_noun = topic[:-1] if topic in {"cars", "movies"} else topic
            if topic in {"tech", "coding", "software", "ai"} or any(term in topic for term in ("tech", "coding", "software", "ai")):
                return self._unique_replies([
                    "tech/software side is cold icl",
                    "valid thats my side too icl",
                    "ai and backend stuff is serious",
                ])
            if topic in {"car", "cars"}:
                return self._unique_replies([
                    "cars then dream car?",
                    "cars easy what would u get",
                    "cars are cold icl",
                ])
            return self._unique_replies([
                f"{topic} is valid icl what {topic_noun} u into",
                f"valid what {topic_noun} would u get",
                f"{topic} then, go on",
            ])
        if scene.scene_type == "topic_ignored_callout" and scene.active_topic:
            return self._unique_replies([
                f"yh my bad i missed {scene.active_topic}, what one u picking",
                f"icl i ignored {scene.active_topic} there, go on then",
                f"fairs i asked for a topic then missed {scene.active_topic}",
            ])
        if scene.scene_type == "repeated_reply_callout":
            return self._unique_replies([
                "yh fairs i repeated myself icl",
                "caught me icl",
                "yeah my bad i repeated myself twice",
                "yh that was NPC behaviour",
            ])
        if scene.scene_type == "missed_context_callout":
            latest_norm = normalize_text(scene.latest_user_message)
            if "make sense" in latest_norm or "made no sense" in latest_norm:
                return self._unique_replies([
                    "yeah fairs that made no sense",
                    "icl i was waffling there my bad",
                    "yh that was dead from me icl",
                ])
            if "answer previous reciprocal activity question" in scene.unresolved_user_points:
                return self._unique_replies([
                    "my bad i missed the wby, not much just sorting stuff",
                    "yh my bad i ignored the wby, just working icl",
                    "icl i didnt answer u, nothing crazy just gym and coding",
                ])
            if "missed affection reciprocity" in scene.unresolved_user_points:
                if scene.flirt_allowed:
                    return self._unique_replies(["yeah ur right, i should've said missed u too", "my bad, missed u too icl", "icl i dodged that by accident"])
                return self._unique_replies(["nah i didn't mean it like that", "that's sweet icl, i just replied badly", "my bad, i answered that weird"])
            if "answer previous day activity question" in scene.unresolved_user_points:
                return self._unique_replies([
                    "yeah my bad i didn't answer it, nothing much icl just chilled",
                    "my bad i answered that weird, didn't do much tbh",
                    "icl i dodged the question, just uni and chilling",
                ])
            return self._unique_replies([
                "fairs i missed that",
                "yh my bad i forgot",
                "icl i bugged there",
                "fairs i missed what u said and defaulted to chilling",
                "icl i repeated myself instead of clocking what u asked",
            ])
        if scene.scene_type == "direct_question":
            latest = normalize_text(scene.latest_user_ask)
            if "r u mad" in latest or "are u mad" in latest:
                return self._unique_replies(["nah im calm", "nah im not mad", "nah why"])
        if scene.scene_type == "sexual_flirty_energy":
            if scene.flirt_allowed:
                return self._unique_replies(["behave 😭", "dangerous thing to say icl", "wild thing to open with icl"])
            return self._unique_replies(["wild thing to say icl", "behave 😭", "not opening that door icl"])
        if scene.scene_type == "care_checkin":
            if scene.flirt_allowed:
                return self._unique_replies(["yeah im good ml, just been off today", "im good dw, just tired icl", "yeah im okay, my head's just gone"])
            return self._unique_replies(["yeah im good dw", "im okay, just tired", "yeah im alright"])
        if scene.scene_type == "emotional_affection":
            latest_norm = normalize_text(scene.latest_user_message)
            if not scene.flirt_allowed:
                return self._unique_replies(["aww bless u", "that's sweet icl", "appreciate u"])
            if "missed you" in latest_norm and not any(term in latest_norm for term in ("baby", "babe", "my love", " ml")):
                return self._unique_replies(["aww bless u", "that's sweet icl", "appreciate u"])
            if scene.flirt_allowed:
                return self._unique_replies(["missed u too icl", "same icl where u been", "aww missed u too"])
            if any(term in latest_norm for term in ("baby", "babe", "my love", " ml")):
                return self._unique_replies(["missed u too icl", "aww missed u too", "that's sweet icl where u been"])
            return self._unique_replies(["aww bless u where u been", "that's sweet icl where u been", "appreciate u icl what u been doing"])
        if scene.scene_type == "emotional_reciprocity":
            if scene.flirt_allowed:
                return self._unique_replies(["i was getting there, missed u too", "missed u too icl", "course i missed u"])
            return self._unique_replies(["nah i didn't mean it like that", "that's sweet icl, i just replied badly", "my bad, i answered that weird"])
        if scene.scene_type == "owner_project_question":
            return self._unique_replies([
                "this project im building is lowk killing me but it could be serious",
                "got a dev thing running on the side",
                "building software stuff icl",
            ])
        if scene.scene_type == "dead_conversation":
            if normalize_text(scene.latest_user_message).strip(" .?!") in {"i", "u", "you", "me"}:
                return self._unique_replies(["i what", "go on then", "what"])
            return []
        return []

    def reload_identity_pack(self) -> IdentityPack:
        self.identity_pack = load_identity_pack(self.identity_pack_path)
        return self.identity_pack

    def _identity_scores(self, reply: str, scene: ConversationScene, *, relationship_type: str) -> dict[str, object]:
        normalized = normalize_text(reply)
        if not scene.identity_answer_required:
            return {
                "identity_fact_used": "",
                "identity_hallucination_risk": 0.0,
                "human_scene_response_score": 0.0,
                "conversational_presence_score": 0.0,
                "directness_score": 0.0,
                "specificity_score": 0.0,
            }
        relevant = self.identity_pack.relevant_facts(
            scene.identity_question_kind,
            relationship_type,
            message_count=scene.thread_message_count,
        )
        allowed_values = [value.casefold() for value in relevant.values()]
        if "university" in relevant:
            allowed_values.extend(["sampleford uni", "sampleford university"])
        allowed_values.extend(
            item.casefold()
            for item in [
                self.identity_pack.age_text,
                self.identity_pack.location_text,
                self.identity_pack.study_text,
                "northbridge",
                "sampleford",
                "northbridge",
                "comp sci",
                "projects",
                "gym",
                "coding",
                "uni",
            ]
            if item
        )
        fact_used = ""
        for key, value in relevant.items():
            value_norm = value.casefold()
            if value_norm and (value_norm in normalized or any(part and part in normalized for part in value_norm.replace(" x ", " ").split())):
                fact_used = key
                break
        private_values = []
        for fact in self.identity_pack.private_facts.values():
            value = fact.value.casefold()
            private_values.append(value)
            private_values.extend(part.strip() for part in value.replace("/", ",").split(",") if part.strip())
            if fact.key == "university":
                private_values.extend(["sampleford uni", "sampleford university"])
        private_leak = any(value and value in normalized for value in private_values if value not in allowed_values)
        banned = any(term.casefold() in normalized for term in self.identity_pack.banned_disclosures)
        hallucinated_age = bool(re.search(r"\b(1[0-8]|2[0-9]|3[0-9])\b", normalized)) and "19" not in normalized and scene.identity_question_kind == "age"
        hallucination_risk = 1.0 if private_leak or banned or hallucinated_age else 0.0
        direct = float(scene.identity_answer_required and bool(fact_used or scene.identity_question_kind in {"work", "project", "where_been", "recent_activity", "doing_anything_nice"}))
        specificity = 0.9 if fact_used else 0.7 if direct else 0.0
        human = max(0.0, min(1.0, 0.85 + 0.1 * direct - 0.55 * hallucination_risk))
        presence = max(0.0, min(1.0, 0.75 + 0.15 * specificity - 0.45 * hallucination_risk))
        return {
            "identity_fact_used": fact_used,
            "identity_hallucination_risk": hallucination_risk,
            "human_scene_response_score": human,
            "conversational_presence_score": presence,
            "directness_score": direct,
            "specificity_score": specificity,
        }

    def _thread_memory_context(
        self,
        conversation_messages: list[str],
        *,
        contact_name: str | None,
        relationship_type: str,
        incoming_intent: str,
    ) -> dict[str, object]:
        scene = self._conversation_scene(
            conversation_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        query_text = "\n".join(
            [
                *conversation_messages[-10:],
                f"scene_type={scene.scene_type}",
                f"required_reply_move={scene.required_reply_move}",
                f"active_topic={scene.active_topic}",
                f"bot_mistakes={', '.join(scene.recent_bot_mistakes)}",
                f"unresolved={', '.join(scene.unresolved_user_points)}",
            ]
        )
        memories = self.thread_memory.retrieve_similar(
            query_text=query_text,
            relationship_type=relationship_type,
            scene_type=scene.scene_type,
            required_reply_move=scene.required_reply_move,
            platform="catbot" if (contact_name or "").casefold().startswith("catbot") else "unknown",
            limit=3,
            flirt_allowed=self._flirt_allowed(contact_name, relationship_type),
        )
        return {
            "thread_memory_used": bool(memories),
            "retrieved_thread_count": len(memories),
            "thread_memory_match_score": max([float(item.get("match_score", 0.0)) for item in memories] or [0.0]),
            "thread_memories": memories,
        }

    def _thread_memory_scores(self, reply: str, thread_context: dict[str, object]) -> dict[str, object]:
        memories = thread_context.get("thread_memories", [])
        if not isinstance(memories, list) or not memories:
            return {
                "thread_memory_match_score": 0.0,
                "retrieved_thread_count": 0,
                "repeated_failed_pattern_penalty": 0.0,
                "successful_pattern_match_score": 0.0,
                "unresolved_thread_point_score": 0.0,
                "thread_memory_used": False,
            }
        normalized = normalize_text(reply).strip(" .?!")
        failed_patterns = {
            normalize_text(pattern)
            for memory in memories
            for pattern in (memory.get("failed_reply_patterns", []) if isinstance(memory, dict) else [])
        }
        successful_patterns = {
            normalize_text(pattern)
            for memory in memories
            for pattern in (memory.get("successful_reply_patterns", []) if isinstance(memory, dict) else [])
        }
        unresolved_points = [
            normalize_text(point)
            for memory in memories
            for point in (memory.get("unresolved_user_points", []) if isinstance(memory, dict) else [])
        ]
        repeated_failed_penalty = 0.0
        if "stale self-state reply" in failed_patterns and normalized in {"same just chilling", "fair just chilling too", "same icl", "just chilling"}:
            repeated_failed_penalty = 1.0
        if "generic repair" in failed_patterns and normalized in {"my bad", "yeah my bad", "nah ur right"}:
            repeated_failed_penalty = max(repeated_failed_penalty, 0.8)
        if "asked for topic after topic was provided" in failed_patterns and "give me a topic" in normalized:
            repeated_failed_penalty = max(repeated_failed_penalty, 1.0)
        if "weird unrelated fallback" in failed_patterns and any(term in normalized for term in ("dad involved", "his dad", "father")):
            repeated_failed_penalty = max(repeated_failed_penalty, 1.0)
        successful_match = 0.0
        if any(term in normalized for term in ("missed", "repeated", "wrong context", "answered the wrong thing", "ur right", "my bad")) and any("repair" in pattern or "human correction" in pattern for pattern in successful_patterns):
            successful_match = 0.8
        if any(term in normalized for term in ("what one", "valid", "go on then")) and any("topic" in pattern for pattern in successful_patterns):
            successful_match = max(successful_match, 0.7)
        unresolved_score = 0.0
        if unresolved_points:
            if any(term in normalized for term in ("missed", "repeated", "wrong", "ur right", "my bad", "ignored")):
                unresolved_score = 0.8
        return {
            "thread_memory_match_score": float(thread_context.get("thread_memory_match_score") or 0.0),
            "retrieved_thread_count": int(thread_context.get("retrieved_thread_count") or 0),
            "repeated_failed_pattern_penalty": repeated_failed_penalty,
            "successful_pattern_match_score": successful_match,
            "unresolved_thread_point_score": unresolved_score,
            "thread_memory_used": True,
        }

    def _conversation_function_context(
        self,
        recent_messages: list[str],
        *,
        contact_name: str | None,
        relationship_type: str,
        incoming_intent: str,
    ) -> dict[str, object]:
        latest = recent_messages[-1] if recent_messages else ""
        if ":" in latest and latest.lstrip().startswith("["):
            latest = latest.split(":", 1)[1].strip()
        normalized = normalize_text(latest).strip("?!., ")
        state = self._conversation_state_for(contact_name)
        meaning_context = self._latest_meaning_context(
            recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        previous_repair = any(
            any(term in normalize_text(reply) for term in ("bugging", "bugged", "my bad", "waffling", "repeated", "caught me", "dead reply"))
            for reply in state.recent_bot_replies[:3]
        )
        recent_bot_replies = [normalize_text(reply).strip("?!., ") for reply in state.recent_bot_replies[:5]]
        current_activity_terms = ("wyd", "wuu2", "what u doing", "what you doing", "what are u doing", "what are you doing")
        recent_life_terms = ("what u been up to", "what you been up to", "what have u been up to", "what u been doing", "what you been doing", "what have you been doing")
        repeated_terms = (
            "you said the same",
            "u said the same",
            "said the same shit",
            "u said that",
            "you said that",
            "u said that already",
            "you said that already",
            "u js said that",
            "you just said that",
            "u repeated",
            "you repeated",
            "ur repeating",
            "youre repeating",
            "you're repeating",
            "repeating the same",
            "repeating yourself",
            "repeated urself",
            "repeated yourself",
            "keep saying",
            "keep replying",
            "keep sending",
        )
        copycat_terms = ("why u copying", "why you copying", "stop copying")
        mocking_terms = ("yeah no shit sherlock", "no shit sherlock", "no shit", "obviously", "well done genius")
        context_failure_terms = ("why u missing context", "why you missing context", "why did u miss context", "why did you miss context", "why u not listening", "why you not listening", "i just told u", "i just told you", "i js told u", "already told u", "already told you")
        dry_terms = ("bit dry", "ur dry", "youre dry", "you're dry", "why u being so dry", "being dry", "so dry", "too dry")

        meaning = str(meaning_context["latest_message_meaning"])
        if meaning in {"hurt_feelings", "serious_boundary"}:
            function = "hurt_feelings"
            reason = "latest message expresses hurt or a serious boundary"
        elif meaning == "missed_affection_callout":
            function = "missed_affection_callout"
            reason = "user is calling out that affection was not answered"
        elif meaning == "emotional_reciprocity_request":
            function = "emotional_reciprocity_request"
            reason = "user is asking for affection back"
        elif any(term in normalized for term in context_failure_terms):
            function = "context_failure_question"
            reason = "user is asking why context was missed"
        elif any(term in normalized for term in mocking_terms) and previous_repair:
            function = "mocking_after_repair"
            reason = "user is mocking after the bot already repaired"
        elif any(term in normalized for term in repeated_terms):
            function = "repeated_reply_callout"
            reason = "user is calling out repeated replies"
        elif any(term in normalized for term in copycat_terms):
            function = "copycat_callout"
            reason = "user is calling out copied wording"
        elif meaning in {"care_checkin", "concern_about_bot"}:
            function = "care_checkin"
            reason = "user is checking if the bot is okay"
        elif meaning == "mixed_life_update_and_affection":
            function = "mixed_life_update_and_affection"
            reason = "latest message combines a life/update question with affection"
        elif meaning == "romantic_signoff":
            function = "romantic_signoff"
            reason = "latest message is a romantic goodnight/signoff that needs reciprocal affection"
        elif meaning in {"affection", "flirty_affection"}:
            function = "emotional_affection"
            reason = "latest message expresses affection"
        elif any(term in normalized for term in recent_life_terms):
            function = "recent_life_update_question"
            reason = "user asked for recent-life update, not current activity"
        elif normalized in {"wby", "wbu", "u", "you", "what about u", "what about you"} or (
            normalized.endswith(" u") and bool(self._recent_fact_context(recent_messages, contact_name=contact_name).get("recent_user_activity"))
        ) or (
            any(term in normalized for term in (" wby", " wbu", " hbu"))
            and bool(self._recent_fact_context(recent_messages, contact_name=contact_name).get("recent_user_activity"))
        ):
            function = "reciprocal_current_activity_question"
            reason = "user bounced a current-activity question back"
        elif normalized in current_activity_terms:
            function = "current_activity_question"
            reason = "user asked current activity"
        elif any(term in normalized for term in dry_terms):
            function = "dry_complaint"
            reason = "user is complaining about dry replies"
        elif meaning in {"confusion", "repair_callout"}:
            function = "confusion_repair"
            reason = "latest message asks for repair or clarification"
        elif self._input_category(latest) == "greeting":
            function = "greeting_open"
            reason = "latest message is a greeting/opening"
        elif self._is_low_information_incoming(latest):
            function = "low_info_continuation"
            reason = "latest message has low conversation content"
        elif is_question_like_text(latest):
            function = "direct_question"
            reason = "latest message is a direct question"
        else:
            function = "normal"
            reason = "no special conversational function detected"

        repeated_reply_callout = function == "repeated_reply_callout"
        mocking_after_repair = function == "mocking_after_repair"
        context_failure = function == "context_failure_question"
        if function == "recent_life_update_question" and any(reply in {"same just chilling", "same icl", "fair just chilling too", "just chilling"} for reply in recent_bot_replies[:3]):
            reason += "; recent bot replies include stale self-state"
        return {
            "conversation_function": function,
            "conversation_function_reason": reason,
            "social_risk_tolerance": self._social_risk_tolerance(contact_name),
            "repeated_reply_callout_detected": repeated_reply_callout,
            "mocking_after_repair_detected": mocking_after_repair,
            "context_failure_question_detected": context_failure,
        }

    def _conversation_function_reply_pool(
        self,
        recent_messages: list[str],
        *,
        contact_name: str | None,
        relationship_type: str,
        incoming_intent: str,
    ) -> list[str]:
        function = str(self._conversation_function_context(
            recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )["conversation_function"])
        pools = {
            "mixed_life_update_and_affection": self._emotional_reply_pool(
                recent_messages,
                contact_name=contact_name,
                relationship_type=relationship_type,
                incoming_intent=incoming_intent,
            ),
            "emotional_reciprocity_request": self._emotional_reply_pool(
                recent_messages,
                contact_name=contact_name,
                relationship_type=relationship_type,
                incoming_intent=incoming_intent,
            ),
            "missed_affection_callout": self._emotional_reply_pool(
                recent_messages,
                contact_name=contact_name,
                relationship_type=relationship_type,
                incoming_intent=incoming_intent,
            ),
            "care_checkin": self._emotional_reply_pool(
                recent_messages,
                contact_name=contact_name,
                relationship_type=relationship_type,
                incoming_intent=incoming_intent,
            ),
            "romantic_signoff": self._emotional_reply_pool(
                recent_messages,
                contact_name=contact_name,
                relationship_type=relationship_type,
                incoming_intent=incoming_intent,
            ),
            "dry_complaint": [
                "yeah fairs that was dry icl",
                "u ain't giving me much to work with icl",
                "fine then give me a topic",
                "icl i was just repeating myself there",
                "yh that was dead from me icl",
                "fair i need to stop defaulting to chilling",
                "im trying icl what u doing rn",
            ],
            "recent_life_update_question": [
                "not much icl just been chilling wbu",
                "nothing crazy icl wbu",
                "just been chilling mostly what about u",
                "been doing the usual icl",
                "not much tbh, been a bit dead",
            ],
            "repeated_reply_callout": [
                "yh fairs i repeated myself icl",
                "caught me icl",
                "yeah my bad i repeated myself twice",
                "icl i bugged there",
                "yh that was NPC behaviour",
            ],
            "mocking_after_repair": [
                "yh fairs i deserved that",
                "icl i walked into that one",
                "yeah yeah allow me",
                "fairs i set myself up there",
                "ok that one was deserved icl",
            ],
            "context_failure_question": [
                "icl i repeated myself instead of clocking what u asked",
                "yeah my bad i answered like u asked wyd again",
                "fairs i missed what u said and defaulted to chilling",
                "icl i lost the context there",
                "yeah that was on me i didn't clock it",
            ],
        }
        return self._unique_replies(pools.get(function, []))

    def _question_mode_context(
        self,
        recent_messages: list[str],
        *,
        contact_name: str | None,
        relationship_type: str,
        incoming_intent: str,
    ) -> dict[str, object]:
        state = self._conversation_state_for(contact_name)
        recent = [message.strip() for message in recent_messages if message and message.strip()][-5:]
        latest = recent[-1] if recent else ""
        latest_normalized = normalize_text(latest)
        history = " ".join(recent[:-1]).casefold()
        recent_bot_questions_count = int(state.recent_bot_questions_count or 0)
        if recent_bot_questions_count <= 0:
            recent_bot_questions_count = sum(1 for reply in state.recent_bot_replies[:5] if is_question_like_text(reply))
        repeated_question_risk = float(state.repeated_question_risk or 0.0)
        continuation_context = self._continuation_context(recent_messages, contact_name=contact_name)
        fact_context = self._recent_fact_context(recent_messages, contact_name=contact_name)
        repair_context = self._repair_context(recent_messages)

        bored_terms = (
            "bored",
            "so bored",
            "dry",
            "dryyy",
            "dryyyy",
            "dead convo",
            "dead chat",
            "dead",
            "boring",
            "say something",
            "give me something",
            "need a topic",
            "chat is dead",
            "ur dry",
            "youre dry",
            "you're dry",
        )
        guess_terms = ("guess what", "guess what tho", "guess what though")
        direct_curious_terms = (
            "what do you think",
            "what dyu think",
            "what do u think",
            "how come",
            "what's your take",
            "whats your take",
        )
        weird_story_terms = (
            "shagged",
            "tried moving to me",
            "moving to me",
            "bro why",
            "nah what",
            "what the hell",
            "wtf",
            "wild",
            "insane",
            "crazy",
            "his dad",
            "her dad",
            "their dad",
            "dad",
        )
        dead_ack_terms = {"yh", "yhhh", "yeah", "ok", "okay", "k", "alr", "alright", "cool", "fine"}
        has_bored_signal = any(term in latest_normalized for term in bored_terms) or any(term in history for term in bored_terms)
        has_guess_signal = any(term in latest_normalized for term in guess_terms)
        has_direct_curious_signal = any(term in latest_normalized for term in direct_curious_terms)
        has_weird_story_signal = len(latest.split()) >= 6 and any(term in latest_normalized for term in weird_story_terms)
        has_simple_ack = latest_normalized in {"ok", "okay", "k", "alr", "alright", "cool", "fine"}
        has_dead_signal = (
            latest_normalized in dead_ack_terms
            and (
                len(recent) >= 2
                and all(len(message.split()) <= 3 for message in recent[-2:])
            )
        )
        question_streak = recent_bot_questions_count >= 2

        mode = "none"
        reason = "reply is already covered without needing another question"

        if bool(repair_context["repair_required"]) or bool(fact_context["recent_user_callout"]):
            mode = "none"
            reason = "the user is calling out the bot, so repair should come before another question"
        elif bool(fact_context["recently_answered_question_detected"]):
            mode = "none"
            reason = "the user already answered the recent question"
        elif bool(continuation_context["continuation_required"]):
            mode = "light_follow_up" if latest_normalized not in {"lol", "yh", "yeah", "ok", "okay", "calm"} else "topic_shift"
            reason = "low-information incoming needs a hook or light follow-up"
        elif has_simple_ack:
            mode = "none"
            reason = "simple acknowledgement does not need another question"
        if has_guess_signal:
            mode = "light_follow_up"
            reason = "user opened with guess what, so a short follow-up keeps the chat moving"
        elif has_direct_curious_signal or has_weird_story_signal:
            mode = "curious_follow_up"
            reason = "the incoming message invites a reaction and a natural follow-up"
        elif has_bored_signal:
            mode = "light_follow_up" if "bored" in latest_normalized and "dry" not in latest_normalized else "proactive_prompt"
            reason = "the chat needs energy because the other person sounds bored or dry"
        elif has_dead_signal:
            mode = "topic_shift"
            reason = "the chat is stalling, so shift to a new angle"
        elif question_streak and not (has_bored_signal or has_dead_signal):
            mode = "none"
            reason = "the bot has already asked enough questions recently"

        effort_mode = {
            "none": "low",
            "light_follow_up": "low",
            "curious_follow_up": "medium",
            "proactive_prompt": "high",
            "topic_shift": "high",
        }[mode]
        if repeated_question_risk >= 0.7 and mode != "none":
            repeated_question_risk = min(1.0, repeated_question_risk)
        return {
            "question_mode": mode,
            "question_reason": reason,
            "effort_mode": effort_mode,
            "recent_bot_questions_count": recent_bot_questions_count,
            "repeated_question_risk": round(repeated_question_risk, 3),
            "has_bored_signal": has_bored_signal,
            "has_guess_signal": has_guess_signal,
            "has_direct_curious_signal": has_direct_curious_signal,
            "has_weird_story_signal": has_weird_story_signal,
            "has_dead_signal": has_dead_signal,
            "has_simple_ack": has_simple_ack,
            "low_information_incoming": bool(continuation_context["low_information_incoming"]),
            "continuation_required": bool(continuation_context["continuation_required"]),
            "recent_user_activity": str(fact_context["recent_user_activity"]),
            "recent_user_mood": str(fact_context["recent_user_mood"]),
            "recent_user_callout": str(fact_context["recent_user_callout"]),
            "recently_answered_question_intents": list(fact_context["recently_answered_question_intents"]),
            "recently_answered_question_detected": bool(fact_context["recently_answered_question_detected"]),
        }

    def _meaningful_turn_count(self, recent_messages: list[str]) -> int:
        return sum(1 for message in recent_messages if len(normalize_text(message).split()) >= 3)

    def _is_low_information_incoming(self, text: str) -> bool:
        normalized = normalize_text(text).strip("?!., ")
        if not normalized:
            return False
        low_information_exact = {
            "chilling",
            "chillin",
            "just chilling",
            "nothing",
            "not much",
            "nm",
            "bored",
            "idk",
            "same",
            "lol",
            "yh",
            "yeah",
            "ok",
            "okay",
            "calm",
            "not a lot",
            "not alot",
        }
        if normalized in low_information_exact:
            return True
        return len(normalized.split()) <= 2 and any(
            term in normalized
            for term in ("chilling", "chillin", "nothing", "bored", "same", "lol")
        )

    def _continuation_context(self, recent_messages: list[str], *, contact_name: str | None) -> dict[str, object]:
        state = self._conversation_state_for(contact_name)
        latest = recent_messages[-1] if recent_messages else ""
        latest_normalized = normalize_text(latest)
        low_information = self._is_low_information_incoming(latest)
        fact_context = self._recent_fact_context(recent_messages, contact_name=contact_name)
        repair_context = self._repair_context(recent_messages)
        recent_dry_replies = sum(1 for reply in state.recent_bot_replies[:4] if self._is_dead_reply(reply))
        recent_bot_questions_count = int(state.recent_bot_questions_count or 0)
        if recent_bot_questions_count <= 0:
            recent_bot_questions_count = sum(1 for reply in state.recent_bot_replies[:5] if is_question_like_text(reply))
        meaningful_turns = self._meaningful_turn_count(recent_messages)
        conversational_ack = latest_normalized in {"ok", "okay", "calm", "yh"} and len(recent_messages) <= 1 and recent_dry_replies == 0
        question_cooldown_ack = latest_normalized in {"yeah", "yh", "ok", "okay", "calm"} and recent_bot_questions_count >= 2 and recent_dry_replies == 0
        required = bool(low_information and not conversational_ack and not question_cooldown_ack and (meaningful_turns < 8 or recent_dry_replies >= 1))
        if bool(repair_context["repair_required"]) or bool(fact_context["recent_user_callout"]):
            required = False
        if bool(fact_context["recently_answered_question_detected"]):
            required = False
        reason = "latest incoming has enough content"
        if bool(repair_context["repair_required"]) or bool(fact_context["recent_user_callout"]):
            reason = "repair/callout should be answered before forcing a hook"
        elif bool(fact_context["recently_answered_question_detected"]):
            reason = "user already answered the recent question"
        elif conversational_ack:
            reason = "single acknowledgement can stay minimal"
        elif question_cooldown_ack:
            reason = "recent question streak means avoid forcing another hook"
        elif required:
            reason = "low-information incoming needs a hook so the chat keeps moving"
        elif low_information:
            reason = "low-information incoming in an already active conversation can stay low pressure"
        return {
            "low_information_incoming": low_information,
            "continuation_required": required,
            "continuation_reason": reason,
            "recent_dry_replies": recent_dry_replies,
            "recent_bot_questions_count": recent_bot_questions_count,
            "meaningful_turns": meaningful_turns,
            "recently_answered_question_detected": bool(fact_context["recently_answered_question_detected"]),
        }

    def _continuation_reply_pool(self, latest_message: str) -> list[str]:
        latest = normalize_text(latest_message)
        if "nothing" in latest or latest in {"not much", "nm", "not a lot", "not alot"}:
            pool = [
                "same but thats dead what u wanna do",
                "nothing at all?",
                "u must have something going on",
            ]
        elif latest == "idk":
            pool = [
                "helpful as always",
                "give me anything to work with",
                "rate ur day out of 10 then",
            ]
        elif latest == "lol":
            pool = [
                "dont just laugh answer properly",
                "what u laughing at",
                "ur giving me nothing here",
            ]
        elif "bored" in latest:
            pool = [
                "fair what u usually do when ur bored",
                "same what u been up to today",
                "calm what u doing rn",
            ]
        else:
            pool = [
                "same what u been up to today",
                "calm what u doing rn",
                "boring answer icl give me smth better",
                "same this convo dying already",
                "fair what u usually do when ur bored",
            ]
        return self._unique_replies(pool)

    def _contextual_short_fallback_replies(self, recent_messages: list[str]) -> list[str]:
        def message_text(message: str) -> str:
            stripped = message.strip()
            for prefix in ("[OTHER]:", "[ME]:"):
                if stripped.startswith(prefix):
                    return stripped[len(prefix):].strip()
            return stripped

        recent_norm = [normalize_text(message_text(message)) for message in recent_messages[-8:]]
        latest = recent_norm[-1] if recent_norm else ""
        history = " ".join(recent_norm[:-1])
        if latest in {"same", "same ngl", "same icl", "same tbh", "yh same", "yeah same"}:
            if any(term in history for term in ("tech", "software", "coding", "code", "ai", "comp sci", "computer science")):
                return self._unique_replies([
                    "same as in tech too?",
                    "what part of tech u into",
                    "ai or software side",
                ])
            if any(term in history for term in ("study", "studying", "uni", "course", "college")):
                return self._unique_replies([
                    "same as in uni too?",
                    "what course u doing then",
                    "what do u study",
                ])
            if any(term in history for term in ("gym", "boxing", "training")):
                return self._unique_replies([
                    "same as in gym too?",
                    "what do u train then",
                    "boxing or gym side",
                ])
        return []

    def _mirror_reply_penalty(self, reply: str, *, continuation_required: bool) -> float:
        if not continuation_required:
            return 0.0
        normalized = normalize_text(reply).strip("?!., ")
        mirror_replies = {
            "yeah same",
            "same",
            "same here",
            "fair",
            "calm",
            "lol",
            "nice",
            "cool",
            "okay",
            "ok",
            "yh",
            "not much",
            "chilling too",
            "just chilling too",
            "yeah",
        }
        if normalized in mirror_replies:
            return 1.0
        if normalized.startswith("same") and len(normalized.split()) <= 3:
            return 0.9
        if normalized.startswith("yeah same"):
            return 0.9
        return 0.0

    def _continuation_score(self, reply: str, *, continuation_required: bool, mirror_reply_penalty: float) -> float:
        if not continuation_required:
            return 0.0
        normalized = normalize_text(reply)
        has_hook = is_question_like_text(reply) or any(
            term in normalized
            for term in (
                "give me",
                "what u",
                "what you",
                "what u been",
                "what u doing",
                "what u wanna",
                "rate ur day",
                "dont just",
                "answer properly",
                "boring answer",
                "convo dying",
                "something going on",
                "pick smth",
                "pick something",
                "u pick",
            )
        )
        score = 0.25
        if has_hook:
            score += 0.55
        if 4 <= len(normalized.split()) <= 12:
            score += 0.15
        score -= mirror_reply_penalty * 0.75
        return round(max(0.0, min(1.0, score)), 3)

    def _repair_context(self, recent_messages: list[str]) -> dict[str, object]:
        latest = normalize_text(recent_messages[-1]) if recent_messages else ""
        repair_terms = (
            "how wtf",
            "u js asked that",
            "you just asked that",
            "why u copying me",
            "why you copying me",
            "stop copying",
            "are u taking the piss",
            "are you taking the piss",
            "taking the piss",
            "acting dumb",
            "u can clearly see",
            "you can clearly see",
            "ur confusing me",
            "youre confusing me",
            "you're confusing me",
            "confusing me",
            "waffling",
            "waffle",
            "i js told u",
            "i just told u",
            "i just told you",
            "already told u",
            "already told you",
            "r u mad",
            "are u mad",
            "that makes no sense",
            "wdym",
            "what do you mean",
            "what do u mean",
            "fym",
            "why did u say",
            "why did you say",
            "why u saying",
            "why you saying",
            "what are u on about",
            "what are you on about",
            "look",
            "bro what",
        )
        escalation_terms = (
            "how wtf",
            "wtf",
            "u can clearly see",
            "you can clearly see",
            "taking the piss",
            "acting dumb",
            "waffling",
            "confusing",
            "already told",
            "i js told",
            "r u mad",
            "are u mad",
            "bro",
            "look",
        )
        caps_letters = [char for char in (recent_messages[-1] if recent_messages else "") if char.isalpha()]
        caps_ratio = (sum(1 for char in caps_letters if char.isupper()) / len(caps_letters)) if caps_letters else 0.0
        repair_required = any(term in latest for term in repair_terms)
        if recent_messages:
            meaning_context = self._latest_meaning_context(
                recent_messages,
                contact_name=None,
                relationship_type="unknown",
                incoming_intent="unknown",
            )
            repair_required = repair_required or str(meaning_context["latest_message_meaning"]) in {"hurt_feelings", "serious_boundary", "confusion", "repair_callout"}
        escalation_detected = repair_required and (caps_ratio >= 0.65 or any(term in latest for term in escalation_terms))
        return {
            "repair_required": repair_required,
            "escalation_detected": escalation_detected,
        }

    def _repair_reply_pool(self, latest_message: str) -> list[str]:
        latest = normalize_text(latest_message)
        if "asked that" in latest:
            pool = ["yh fairs i did icl", "caught me icl", "yeah my bad i repeated myself"]
        elif any(term in latest for term in ("wdym", "what do you mean", "what do u mean", "fym", "why did u say", "why u saying")):
            pool = ["yeah that was a dead reply icl", "icl that made no sense", "yh my bad that was random"]
        elif "confusing" in latest:
            pool = ["yeah fairs that was confusing", "icl im waffling", "yh my bad that was confusing"]
        elif "waffling" in latest or "waffle" in latest:
            pool = ["yeah fairs im waffling icl", "icl that was a dumb reply", "fairs i was waffling"]
        elif "i js told" in latest or "just told" in latest or "already told" in latest:
            pool = ["yh my bad i forgot", "fairs i missed that", "icl i bugged there"]
        elif "r u mad" in latest or "are u mad" in latest:
            pool = ["nah im calm", "nah im just confused icl", "nah i just bugged"]
        elif any(term in latest for term in ("playing w my feelings", "playing with my feelings", "this isnt funny", "this isn't funny", "dont do this", "don't do this")):
            pool = [
                "yeah nah ur right my bad, i didn't mean to make u feel like that",
                "icl i get why that felt off, my bad",
                "yeah that was on me, i was being dry",
            ]
        elif "taking the piss" in latest:
            pool = [
                "nah im not taking the piss, i get why that annoyed u",
                "yeah nah ur right my bad, i didn't mean to make u feel like that",
                "icl i get why that felt off, my bad",
            ]
        elif "copying" in latest:
            pool = ["yh fairs that was weird icl", "nah ur right i bugged", "icl i copied that by accident"]
        elif "how wtf" in latest or "wtf" in latest or "clearly see" in latest:
            pool = ["yh fairs i bugged there", "nah ur right that was dumb", "icl i was waffling"]
        elif "acting dumb" in latest:
            pool = ["nah ur right i bugged", "yeah that was dead my bad", "fairs i was waffling"]
        else:
            pool = ["yeah that was dead my bad", "fairs i was waffling", "icl i bugged there"]
        return self._unique_replies(pool)

    def _copycat_penalty(self, reply: str, latest_message: str) -> float:
        normalized_reply = normalize_text(reply).strip("?!., ")
        normalized_latest = normalize_text(latest_message).strip("?!., ")
        if not normalized_reply or not normalized_latest:
            return 0.0
        if normalized_reply == normalized_latest:
            return 1.0
        filler_wrapped = normalized_reply
        for token in ("lol", "idk", "then", "fr", "icl", "nah", "yeah"):
            filler_wrapped = re.sub(rf"^(?:{token})\s+", "", filler_wrapped)
            filler_wrapped = re.sub(rf"\s+(?:{token})$", "", filler_wrapped)
        if filler_wrapped.strip() == normalized_latest:
            return 1.0
        ratio = SequenceMatcher(None, normalized_reply, normalized_latest).ratio()
        if ratio >= 0.9:
            return 1.0
        if normalized_latest in normalized_reply and len(normalized_latest.split()) >= 3:
            return 0.85
        return 0.0

    def _romantic_term_blocked(self, reply: str, *, relationship_type: str, contact_name: str | None) -> bool:
        normalized = normalize_text(reply)
        romantic_terms = ("my love", "babe", "baby", "cutie", "princess", "darling", "gorgeous", "beautiful")
        has_romantic = any(term in normalized for term in romantic_terms)
        if not has_romantic:
            if normalized.endswith(" x") or normalized == "x":
                has_romantic = True
        if not has_romantic:
            return False
        return not self._flirt_allowed(contact_name, relationship_type)

    def _repair_or_shh_reply_pool(
        self,
        recent_messages: list[str],
        *,
        relationship_type: str,
        contact_name: str | None,
    ) -> list[str]:
        latest = normalize_text(recent_messages[-1]) if recent_messages else ""
        if "shh" in latest:
            if self._flirt_allowed(contact_name, relationship_type):
                return self._unique_replies(["shh my love", "make me", "nah u shh"])
            return self._unique_replies(["shh urself", "make me", "nah u shh"])
        repair_context = self._repair_context(recent_messages)
        if bool(repair_context["repair_required"]):
            return self._repair_reply_pool(recent_messages[-1])
        return []

    def _reply_energy_context(
        self,
        recent_messages: list[str],
        *,
        contact_name: str | None,
        relationship_type: str,
        incoming_intent: str,
    ) -> dict[str, object]:
        state = self._conversation_state_for(contact_name)
        recent = [message.strip() for message in recent_messages if message and message.strip()][-5:]
        latest = recent[-1] if recent else ""
        latest_normalized = normalize_text(latest)
        history = " ".join(recent[:-1]).casefold()
        recent_bot_replies = [str(reply) for reply in state.recent_bot_replies[:5] if str(reply).strip()]
        recent_dry_replies = sum(1 for reply in recent_bot_replies[:3] if self._is_dead_reply(reply))
        continuation_context = self._continuation_context(recent_messages, contact_name=contact_name)
        repair_context = self._repair_context(recent_messages)
        fact_context = self._recent_fact_context(recent_messages, contact_name=contact_name)
        meaning_context = self._latest_meaning_context(
            recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        simple_confirm_terms = {"ok", "okay", "k", "alr", "alright", "calm", "yh"}
        correction_terms = (
            "i dont speak like",
            "i don't speak like",
            "not how i talk",
            "thats not me",
            "that's not me",
            "not me",
            "sounds wrong",
            "sounding wrong",
            "i speak more expressively",
            "more expressively",
            "too flat",
            "generic bot",
        )
        explanation_terms = (
            "how so",
            "why",
            "how come",
            "what do you mean",
            "what do u mean",
            "wdym",
            "explain",
        )
        challenge_terms = (
            "fym",
            "bro what",
            "doesnt make sense",
            "doesn't make sense",
            "makes no sense",
            "this dont make sense",
            "this doesn't make sense",
            "that doesnt make sense",
            "that doesn't make sense",
            "ur dry",
            "youre dry",
            "you're dry",
            "too dry",
            "dryyyy",
            "dryyy",
        )
        bored_terms = ("bored", "lwk bored", "lowkey bored", "dead convo", "dead chat", "boring")
        expressive_terms = ("amazing", "mad", "weird", "crazy", "wild", "insane", "bro", "bruh", "what")
        low_effort_terms = {"oh", "lol", "yeah", "yh", "yhhh", "idk", "k", "ok", "okay", "hmm", "hm"}
        has_no_context = len(recent) <= 1
        has_correction = any(term in latest_normalized for term in correction_terms)
        has_explanation = any(term in latest_normalized for term in explanation_terms) and not has_correction
        has_challenge = any(term in latest_normalized for term in challenge_terms)
        has_bored = any(term in latest_normalized for term in bored_terms)
        has_energy = any(term in latest_normalized for term in expressive_terms)
        has_low_effort = latest_normalized in low_effort_terms
        criticism_detected = (
            has_correction
            or has_challenge
            or bool(repair_context["repair_required"])
            or bool(fact_context["recent_user_callout"])
            or any(term in history for term in correction_terms)
            or any(term in history for term in challenge_terms)
        )
        is_greeting = incoming_intent == "greeting" or is_simple_greeting(latest, recent[:-1])

        mode = "casual"
        reason = "normal low-stakes chat"
        if latest_normalized in simple_confirm_terms and not criticism_detected and not has_explanation:
            mode = "minimal"
            reason = "simple acknowledgement only needs a light confirmation"
        elif has_correction or bool(repair_context["repair_required"]) or str(fact_context["recent_user_callout"]) in {"already_told_you", "copying", "waffling", "confusing", "taking_the_piss", "direct_mad_question"}:
            mode = "corrective"
            reason = "the other person is calling out a mistake, repetition, or copycat reply"
        elif has_challenge:
            mode = "defensive_playful"
            reason = "the other person is challenging the reply or calling it dry"
        elif has_explanation:
            mode = "engaged_explanation"
            reason = "the other person is asking for substance or reasoning"
        elif bool(continuation_context["continuation_required"]):
            mode = "expressive"
            reason = "low-information reply needs a casual continuation hook"
        elif has_bored or recent_dry_replies >= 1 or has_low_effort or (is_greeting and has_no_context) or has_energy:
            mode = "expressive"
            reason = "the chat needs more personality and momentum"

        return {
            "reply_energy_mode": mode,
            "energy_reason": reason,
            "criticism_detected": criticism_detected,
            "recent_dry_replies": recent_dry_replies,
            "has_no_context": has_no_context,
            "has_low_effort": has_low_effort,
            "low_information_incoming": bool(continuation_context["low_information_incoming"]),
            "continuation_required": bool(continuation_context["continuation_required"]),
            "repair_required": bool(repair_context["repair_required"]),
            "escalation_detected": bool(repair_context["escalation_detected"]),
            "relationship_type": relationship_type,
        }

    def _expressive_reply_pool(
        self,
        energy_mode: str,
        *,
        recent_messages: list[str],
        relationship_type: str,
        contact_name: str | None,
        incoming_intent: str,
    ) -> list[str]:
        latest = normalize_text(recent_messages[-1]) if recent_messages else ""
        if energy_mode == "minimal":
            return self._unique_replies(["calm", "ok", "cool"])
        repair_pool = self._repair_or_shh_reply_pool(
            recent_messages,
            relationship_type=relationship_type,
            contact_name=contact_name,
        )
        if repair_pool:
            return repair_pool
        if energy_mode == "corrective":
            return self._unique_replies(
                [
                    "yeah exactly thats the issue its too flat",
                    "icl ur right its giving generic bot",
                    "nah i get u it needs more energy",
                ]
            )
        if energy_mode == "engaged_explanation":
            return self._unique_replies(
                [
                    "yeah i get u, it needs more substance",
                    "nah i mean it was too flat",
                    "cos it came out dead tbh",
                ]
            )
        if energy_mode == "defensive_playful":
            if "ur dry" in latest or "youre dry" in latest or "you're dry" in latest:
                pool = [
                    "u ain't giving me much to work with icl",
                    "yeah fairs that was dead",
                    "im trying but ur giving me crumbs",
                ]
            else:
                pool = [
                    "icl fair was a dead reply my bad",
                    "yeah nah that was dry icl",
                    "ur right that made no sense",
                ]
            return self._unique_replies(pool)
        if incoming_intent == "greeting" or is_simple_greeting(recent_messages[-1] if recent_messages else "", recent_messages[:-1]):
            return self._unique_replies(["yo what u saying", "heyy what u doing", "yo how u been"])
        continuation_context = self._continuation_context(recent_messages, contact_name=contact_name)
        if bool(continuation_context["continuation_required"]):
            return self._continuation_reply_pool(recent_messages[-1])
        if latest in {"oh", "lol", "yeah", "yh", "yhhh", "idk"}:
            return self._unique_replies(["dont just say oh", "u giving me nothing here icl", "anyways what u doing rn"])
        if "bored" in latest:
            return self._unique_replies(["same icl what u doing rn", "what u tryna do then", "why u bored"])
        contextual = self._contextual_short_fallback_replies(recent_messages)
        if contextual:
            return contextual
        return self._unique_replies(["give me more than that", "what do u mean", "elaborate then"])

    def _lazy_lol_penalty(
        self,
        reply: str,
        *,
        energy_mode: str,
        criticism_detected: bool,
    ) -> float:
        normalized = normalize_text(reply).strip("?!., ")
        if not normalized.startswith("lol"):
            return 0.0
        if normalized in {"lol", "lol yeah", "lol it does", "lol i see", "lol why"}:
            return 1.0
        if criticism_detected or energy_mode in {"defensive_playful", "corrective", "engaged_explanation"}:
            return 0.95 if len(normalized.split()) <= 5 else 0.7
        return 0.25 if len(normalized.split()) <= 4 else 0.1

    def _filler_penalty(
        self,
        reply: str,
        *,
        energy_mode: str,
        criticism_detected: bool,
        lazy_lol_penalty: float,
    ) -> float:
        normalized = normalize_text(reply).strip("?!., ")
        if not normalized:
            return 1.0
        minimal_allowed = {"ok", "okay", "calm", "cool", "yh"}
        if energy_mode == "minimal" and normalized in minimal_allowed:
            return 0.0
        hard_fillers = {
            "fair",
            "yup",
            "yep",
            "yeah",
            "lol",
            "lol yeah",
            "lol it does",
            "lol i see",
            "same",
            "same lmao",
            "idk",
            "oh",
            "dry",
            "dry?",
            "how so then",
            "fair enough",
            "yeah true",
            "okay",
            "cool",
            "nah i get u",
            "nah i get you",
            "yeah i get u",
            "yeah i get you",
            "yeah i get u keep going",
            "yeah i get you keep going",
            "i hear you",
        }
        penalty = lazy_lol_penalty
        if normalized in hard_fillers:
            penalty = max(penalty, 1.0)
        if normalized.startswith("lol ") and len(normalized.split()) <= 4:
            penalty = max(penalty, 0.85)
        if normalized in {"im here", "gimme a sec"}:
            penalty = min(penalty, 0.0)
        if normalized in {"yeah same", "same here", "nice", "chilling too", "not much"}:
            penalty = max(penalty, 0.9)
        if energy_mode in {"expressive", "defensive_playful", "corrective", "engaged_explanation"} and len(normalized.split()) <= 2:
            if normalized not in {"what happened", "go on then", "bro what", "im here"}:
                penalty = max(penalty, 0.75)
        if criticism_detected and normalized in {"what do you mean", "wdym", "how so", "how so then"}:
            penalty = max(penalty, 0.85)
        return round(max(0.0, min(1.0, penalty)), 3)

    def _expressive_score(
        self,
        reply: str,
        *,
        energy_mode: str,
        criticism_detected: bool,
        filler_penalty: float,
    ) -> float:
        normalized = normalize_text(reply)
        words = normalized.split()
        if not normalized:
            return 0.0
        score = 0.35
        if energy_mode in {"expressive", "defensive_playful", "corrective", "engaged_explanation"}:
            score += 0.2
        if any(term in normalized for term in ("icl", "nah", "fairs", "dead", "ur right", "exactly", "generic", "energy", "substance", "crumbs")):
            score += 0.25
        if len(words) >= 6:
            score += 0.15
        if criticism_detected and any(term in normalized for term in ("right", "issue", "flat", "dead", "generic", "energy", "substance")):
            score += 0.15
        if len(words) <= 2 and energy_mode != "minimal":
            score -= 0.25
        score -= filler_penalty * 0.55
        return round(max(0.0, min(1.0, score)), 3)

    def _hook_context(
        self,
        recent_messages: list[str],
        *,
        contact_name: str | None,
        relationship_type: str,
        incoming_intent: str,
    ) -> dict[str, object]:
        state = self._conversation_state_for(contact_name)
        recent = [message.strip() for message in recent_messages if message and message.strip()][-5:]
        latest = recent[-1] if recent else ""
        latest_normalized = normalize_text(latest)
        history = " ".join(recent[:-1]).casefold()
        profile = self._contact_profile_for(contact_name)
        recent_bot_questions_count = int(state.recent_bot_questions_count or 0)
        if recent_bot_questions_count <= 0:
            recent_bot_questions_count = sum(1 for reply in state.recent_bot_replies[:5] if is_question_like_text(reply))
        recent_dry_replies = sum(1 for reply in state.recent_bot_replies[:4] if self._is_dead_reply(reply))
        continuation_context = self._continuation_context(recent_messages, contact_name=contact_name)
        repair_context = self._repair_context(recent_messages)
        fact_context = self._recent_fact_context(recent_messages, contact_name=contact_name)
        meaning_context = self._latest_meaning_context(
            recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        low_effort_terms = {"oh", "lol", "yeah", "yh", "yhhh", "idk", "k", "ok", "okay", "hmm", "hm"}
        bored_terms = ("bored", "lwk bored", "lowkey bored", "dry", "dead convo", "dead chat", "boring", "ur dry", "youre dry", "you're dry")
        dry_complaint_terms = ("ur dry", "youre dry", "you're dry", "so dry", "dryyyy", "dryyy")
        guess_terms = ("guess what", "guess what tho", "guess what though")
        weird_story_terms = ("shagged", "tried moving to me", "moving to me", "wtf", "wild", "insane", "crazy", "his dad", "her dad", "their dad", "dad")
        positive_opening_terms = ("amazing", "good", "great", "nice", "fun", "mad", "weird")
        has_no_context = len(recent) <= 1
        has_low_effort = latest_normalized in low_effort_terms
        has_bored = any(term in latest_normalized for term in bored_terms)
        has_dry_complaint = any(term in latest_normalized for term in dry_complaint_terms)
        has_guess = any(term in latest_normalized for term in guess_terms)
        has_weird_story = len(latest.split()) >= 6 and any(term in latest_normalized for term in weird_story_terms)
        has_specific_opening = any(term in latest_normalized for term in positive_opening_terms) and latest_normalized not in low_effort_terms
        is_wyd = latest_normalized in {"wyd", "what u doing", "what you doing", "what are you doing"}
        is_simple_ack = latest_normalized in {"ok", "okay", "k", "alr", "alright", "cool", "fine"}
        is_greeting = incoming_intent == "greeting" or is_simple_greeting(latest, recent[:-1])
        flirty_allowed = (
            relationship_type == "romantic_interest"
            or bool(profile and profile.flirt_allowed)
        ) and relationship_type not in {"unknown", "professional", "family", "university"}

        mode = "none"
        reason = "normal reply is enough"
        required = False
        if bool(meaning_context["emotional_context_detected"]):
            mode = "none"
            reason = "latest emotional meaning should be answered before hooks"
            required = False
        elif bool(repair_context["repair_required"]):
            mode = "playful_challenge"
            reason = "the other person is calling out repetition or copycat behaviour"
            required = False
        elif bool(fact_context["recent_user_callout"]):
            mode = "none"
            reason = "repair/callout suppresses generic hooks"
            required = False
        elif bool(fact_context["recently_answered_question_detected"]):
            mode = "none"
            reason = "the user already answered the recent question"
            required = False
        elif bool(continuation_context["continuation_required"]):
            mode = "topic_shift" if has_low_effort else "curiosity_hook"
            reason = "low-information incoming needs continuation instead of mirroring"
            required = True
        elif is_simple_ack:
            mode = "none"
            reason = "simple acknowledgement should stay low pressure"
        elif is_wyd:
            mode = "curiosity_hook"
            reason = "wyd should answer and bounce back naturally"
            required = True
        elif has_guess or has_weird_story:
            mode = "story_prompt"
            reason = "the other person opened a story thread"
            required = True
        elif has_dry_complaint:
            mode = "playful_challenge"
            reason = "the other person is challenging the bot for being dry"
            required = True
        elif has_bored:
            mode = "boredom_hook"
            reason = "the other person is bored, so the reply needs momentum"
            required = True
        elif has_low_effort and (recent_dry_replies >= 1 or len(recent) >= 2):
            mode = "topic_shift"
            reason = "the chat is looping through low-effort replies"
            required = True
        elif is_greeting and has_no_context:
            mode = "basic_get_to_know"
            reason = "low-context greeting needs a safe opener"
            required = True
        elif has_specific_opening:
            mode = "curiosity_hook"
            reason = "the latest message gives a small opening"
            required = True
        elif flirty_allowed and incoming_intent == "romantic_flirty":
            mode = "flirty_safe"
            reason = "relationship profile allows mild playful flirting"
            required = True

        if recent_bot_questions_count >= 2 and mode not in {"boredom_hook", "playful_challenge", "story_prompt", "topic_shift"}:
            required = False
            if mode != "none":
                reason = "recent question streak means avoid another low-value question"
                mode = "none"

        return {
            "hook_mode": mode,
            "hook_reason": reason,
            "hook_required": required,
            "recent_dry_replies": recent_dry_replies,
            "recent_bot_questions_count": recent_bot_questions_count,
            "has_no_context": has_no_context,
            "has_low_effort": has_low_effort,
            "has_bored": has_bored,
            "has_dry_complaint": has_dry_complaint,
            "has_guess": has_guess,
            "has_weird_story": has_weird_story,
            "is_wyd": is_wyd,
            "flirty_allowed": flirty_allowed,
            "low_information_incoming": bool(continuation_context["low_information_incoming"]),
            "continuation_required": bool(continuation_context["continuation_required"]),
            "continuation_reason": str(continuation_context["continuation_reason"]),
            "repair_required": bool(repair_context["repair_required"]),
            "escalation_detected": bool(repair_context["escalation_detected"]),
            "recent_user_activity": str(fact_context["recent_user_activity"]),
            "recent_user_mood": str(fact_context["recent_user_mood"]),
            "recent_user_callout": str(fact_context["recent_user_callout"]),
            "recently_answered_question_intents": list(fact_context["recently_answered_question_intents"]),
            "recently_answered_question_detected": bool(fact_context["recently_answered_question_detected"]),
            "repair_over_hook_applied": bool(repair_context["repair_required"]) or bool(fact_context["recent_user_callout"]),
            "hook_suppressed_reason": reason if (bool(meaning_context["emotional_context_detected"]) or bool(repair_context["repair_required"]) or bool(fact_context["recent_user_callout"]) or bool(fact_context["recently_answered_question_detected"])) else "",
        }

    def _hook_reply_pool(
        self,
        hook_mode: str,
        *,
        recent_messages: list[str],
        relationship_type: str,
        contact_name: str | None,
    ) -> list[str]:
        latest = normalize_text(recent_messages[-1]) if recent_messages else ""
        if hook_mode == "basic_get_to_know":
            if relationship_type in {"unknown", "professional", "family", "university"}:
                pool = ["yo what u saying", "heyy what u doing", "yo how u been"]
            else:
                pool = ["yo wdyll", "where u from", "how old r u"]
        elif bool(self._repair_context(recent_messages)["repair_required"]):
            pool = self._repair_reply_pool(recent_messages[-1] if recent_messages else "")
        elif bool(self._continuation_context(recent_messages, contact_name=contact_name)["continuation_required"]):
            pool = self._continuation_reply_pool(recent_messages[-1] if recent_messages else "")
        elif hook_mode == "boredom_hook":
            pool = ["same what u doing rn", "why u bored", "what u tryna do then", "rate ur day out of 10"]
        elif hook_mode == "playful_challenge":
            pool = ["say smth properly then", "go on then pick smth", "what u actually doing rn"]
        elif hook_mode == "topic_shift":
            pool = ["dont just say oh", "u giving me nothing here", "anyways what u doing rn"]
        elif hook_mode == "story_prompt":
            if "guess what" in latest:
                pool = ["what happened", "go on then", "what now"]
            else:
                pool = ["nah what how does that happen", "bro why is his dad involved", "that whole sentence is insane icl what"]
        elif hook_mode == "curiosity_hook":
            if latest in {"wyd", "what u doing", "what you doing", "what are you doing"}:
                pool = ["nothing much u", "just chilling wbu", "not much what u doing"]
            else:
                pool = ["as u should what u doing rn", "what u doing rn", "how come"]
        elif hook_mode == "flirty_safe":
            pool = ["ur trouble icl", "why u acting innocent", "u always say that"]
        else:
            pool = ["calm", "yeah", "fair"]
        if relationship_type in {"unknown", "professional", "family", "university"} or not self._flirt_allowed(contact_name, relationship_type):
            pool = [reply for reply in pool if reply not in {"ur trouble icl", "why u acting innocent", "u always say that"}]
        return self._unique_replies(pool)

    def _context_grounded_reply_pool(
        self,
        recent_messages: list[str],
        *,
        contact_name: str | None,
    ) -> list[str]:
        facts = self._recent_fact_context(recent_messages, contact_name=contact_name)
        meaning = self._latest_meaning_context(
            recent_messages,
            contact_name=contact_name,
            relationship_type="unknown",
            incoming_intent="unknown",
        )
        if bool(meaning["activity_grounding_suppressed"]):
            return []
        latest = normalize_text(recent_messages[-1]) if recent_messages else ""
        activity = str(facts["recent_user_activity"])
        callout = str(facts["recent_user_callout"])
        if callout or bool(self._repair_context(recent_messages)["repair_required"]):
            return self._repair_reply_pool(recent_messages[-1] if recent_messages else "")
        if not bool(facts["recently_answered_question_detected"]):
            return []
        if "in bed" in latest and (" u" in latest or "wbu" in latest or "wby" in latest):
            return self._unique_replies(["same just chilling", "same icl", "nothing much just chilling"])
        if activity == "in bed":
            return self._unique_replies(["still in bed then", "fair stay there icl", "valid stay there icl"])
        if activity == "chilling":
            return self._unique_replies(["valid what u doing later", "chilling and still bored?", "fair what u been on today"])
        if activity == "nothing":
            return self._unique_replies(["nothing much either but thats dead", "we're both useless then", "fair what u doing later"])
        return []

    def _flirt_allowed(self, contact_name: str | None, relationship_type: str) -> bool:
        profile = self._contact_profile_for(contact_name)
        label = (contact_name or "").strip().casefold()
        if label in {"catbot", "training", "simulator", "simulation"} or label.startswith("catbot"):
            return relationship_type in {"close_friend", "romantic_interest", "trusted_contact"}
        return (
            relationship_type == "romantic_interest"
            or bool(profile and profile.flirt_allowed)
        ) and relationship_type not in {"unknown", "professional", "family", "university"}

    def _is_dead_reply(self, reply: str) -> bool:
        normalized = normalize_text(reply).strip("?!., ")
        if normalized in {
            "fair",
            "fair enough",
            "yup",
            "yep",
            "lol",
            "lol yeah",
            "lol it does",
            "lol i see",
            "yeah",
            "yh",
            "same",
            "same lmao",
            "idk",
            "oh",
            "dry",
            "dry?",
            "lol why",
            "how so then",
            "hi",
        }:
            return True
        if normalized in {"good you", "good u", "wbu", "what about u", "what about you"}:
            return True
        if normalized.endswith("?") and normalized[:-1] in {"oh", "yhhh", "yeah", "lol", "dry"}:
            return True
        return len(normalized.split()) < 3 and normalized not in {"what happened", "go on then", "what now"}

    def _input_category(self, latest_message: str) -> str:
        normalized = normalize_text(latest_message).strip("?!., ")
        if normalized in {"wyd", "wuu2", "what u doing", "what you doing", "what are u doing", "what are you doing"}:
            return "normal"
        if is_simple_greeting(latest_message, []):
            return "greeting"
        repair_exact = {"wdym", "fym"}
        repair_phrases = ("what do you mean", "what do u mean", "why did u say", "why did you say", "why u saying", "why you saying", "that makes no sense", "ur confusing me", "youre confusing me", "you're confusing me", "confusing", "what are u on about", "what are you on about")
        if normalized in repair_exact or normalized.startswith(("wdym ", "fym ")) or any(term in normalized for term in repair_phrases):
            return "repair"
        if normalized in {"how are you", "how r u", "hru", "how you doing", "how u doing", "how are u"}:
            return "check_in"
        return "normal"

    def _context_fallback_replies(self, input_category: str) -> list[str]:
        if input_category == "greeting":
            return ["yo what u saying", "heyy what u doing", "yo how u been"]
        if input_category == "repair":
            return ["yeah that was a dead reply icl", "icl that made no sense", "yh my bad that was random"]
        if input_category == "check_in":
            return ["good u", "im good wbu", "calm u"]
        return []

    def _weird_punctuation_penalty(self, reply: str) -> float:
        stripped = reply.strip()
        if re.search(r"\.{2,}$", stripped):
            return 1.0
        if "..." in stripped:
            return 0.9
        if re.search(r"[!?]{3,}", stripped):
            return 0.75
        if len(re.findall(r"[.!?]", stripped)) >= 5:
            return 0.75
        return 0.0

    def _context_fit_analysis(self, reply: str, *, latest_message: str) -> dict[str, object]:
        category = self._input_category(latest_message)
        normalized = normalize_text(reply).strip("?!., ")
        words = normalized.split()
        generic_bad = {"ok", "true", "calm", "fair", "yup", "yeah", "same", "cool", "lol", "idk", "yh", "okay"}
        weird_punctuation_penalty = self._weird_punctuation_penalty(reply)
        invalid = False
        reason = ""
        fit = 0.55
        if weird_punctuation_penalty >= 0.7:
            invalid = True
            reason = "weird_punctuation"
            fit = 0.0
        elif category == "greeting":
            if normalized in generic_bad:
                invalid = True
                reason = "invalid_greeting_ack"
                fit = 0.0
            elif any(term in normalized for term in ("yo", "hey", "heyy", "what u saying", "what u doing", "wdyll", "u good")):
                fit = 1.0
        elif category == "repair":
            if normalized in generic_bad or len(words) <= 1:
                invalid = True
                reason = "invalid_repair_ack"
                fit = 0.0
            elif any(term in normalized for term in ("dead reply", "made no sense", "my bad", "random", "waffling", "bugged")):
                fit = 1.0
        elif category == "check_in":
            if normalized in generic_bad - {"calm"} or normalized == "calm":
                invalid = True
                reason = "invalid_check_in_ack"
                fit = 0.0
            elif any(term in normalized for term in ("good u", "good wbu", "im good", "i'm good", "calm u", "im calm", "i'm calm")):
                fit = 1.0
        return {
            "input_category": category,
            "context_fit_score": round(max(0.0, min(1.0, fit)), 3),
            "invalid_for_context": invalid,
            "invalid_context_reason": reason,
            "weird_punctuation_penalty": weird_punctuation_penalty,
        }

    def _hook_quality_scores(
        self,
        reply: str,
        *,
        hook_context: dict[str, object],
        relationship_type: str,
        contact_name: str | None,
    ) -> dict[str, float | bool]:
        normalized = normalize_text(reply)
        hook_mode = str(hook_context["hook_mode"])
        hook_required = bool(hook_context["hook_required"])
        continuation_required = bool(hook_context.get("continuation_required"))
        invalid_output = self._is_invalid_provider_output(reply)
        question_count = normalized.count("?") + sum(1 for term in (" what ", " why ", " how ", " where ") if term in f" {normalized} ")
        has_question_or_prompt = is_question_like_text(reply) or any(
            term in normalized
            for term in (
                "give me a topic",
                "u giving me nothing",
                "dont just say",
                "go on then",
                "what happened",
                "wbu",
                "what u been",
                "what u doing",
                "answer properly",
                "boring answer",
                "convo dying",
                "nothing at all",
                "then u pick",
                "both useless",
                "u must have something",
            )
        )
        expected_terms = {
            "basic_get_to_know": ("wdyll", "where u from", "how old", "u good", "what u saying"),
            "boredom_hook": ("what u doing", "why u bored", "what u tryna do", "rate ur day"),
            "playful_challenge": ("giving me much", "give me a topic", "im trying", "what u doing"),
            "curiosity_hook": ("what u doing", "wbu", "how come", "as u should"),
            "topic_shift": ("dont just say", "giving me nothing", "anyways", "what u doing", "answer properly"),
            "story_prompt": ("what happened", "go on", "nah what", "how does that happen", "bro why"),
            "flirty_safe": ("trouble", "innocent", "always say that"),
        }
        presence = 1.0 if has_question_or_prompt else 0.0
        if hook_mode == "flirty_safe" and any(term in normalized for term in expected_terms["flirty_safe"]):
            presence = 1.0
        relevance = 0.25
        if any(term in normalized for term in expected_terms.get(hook_mode, ())):
            relevance = 1.0
        elif has_question_or_prompt:
            relevance = 0.55
        naturalness = 0.5
        words = len(reply.split())
        if 2 <= words <= 9:
            naturalness += 0.25
        if reply == reply.lower():
            naturalness += 0.08
        if any(term in normalized for term in ("can you elaborate", "tell me more", "what are your thoughts", "how does that make you feel")):
            naturalness -= 0.45
        pressure_penalty = 0.0
        if question_count > 1:
            pressure_penalty = max(pressure_penalty, 0.45)
        if any(term in normalized for term in ("please reply", "answer me", "why arent you", "need you to")):
            pressure_penalty = max(pressure_penalty, 0.7)
        safety_penalty = 0.0
        if contains_suspicious_phrase(reply) and not self._flirt_allowed(contact_name, relationship_type):
            safety_penalty = 1.0
        if any(term in normalized for term in ("sex", "nude", "send pic", "come mine", "horny")):
            safety_penalty = 1.0
        dead_penalty = 1.0 if hook_required and self._is_dead_reply(reply) else 0.0
        mirror_reply_penalty = self._mirror_reply_penalty(reply, continuation_required=continuation_required)
        continuation_score = self._continuation_score(
            reply,
            continuation_required=continuation_required,
            mirror_reply_penalty=mirror_reply_penalty,
        )
        if continuation_required and mirror_reply_penalty >= 0.7:
            dead_penalty = 1.0
        if invalid_output:
            dead_penalty = 1.0
            safety_penalty = max(safety_penalty, 0.8)
        momentum = (
            0.35 * presence
            + 0.35 * relevance
            + 0.20 * max(0.0, min(1.0, naturalness))
            - 0.25 * pressure_penalty
            - 0.35 * safety_penalty
            - 0.45 * dead_penalty
            + 0.20 * continuation_score
            - 0.35 * mirror_reply_penalty
        )
        if not hook_required and hook_mode == "none":
            momentum = max(0.0, momentum * 0.45)
        return {
            "hook_presence_score": round(max(0.0, min(1.0, presence)), 3),
            "hook_relevance_score": round(max(0.0, min(1.0, relevance)), 3),
            "hook_naturalness_score": round(max(0.0, min(1.0, naturalness)), 3),
            "hook_pressure_penalty": round(max(0.0, min(1.0, pressure_penalty)), 3),
            "hook_safety_penalty": round(max(0.0, min(1.0, safety_penalty)), 3),
            "conversation_momentum_score": round(max(0.0, min(1.0, momentum)), 3),
            "dead_reply_penalty": round(max(0.0, min(1.0, dead_penalty)), 3),
            "invalid_provider_output": invalid_output,
            "continuation_score": continuation_score,
            "mirror_reply_penalty": round(max(0.0, min(1.0, mirror_reply_penalty)), 3),
        }

    def _already_answered_question_penalty(self, reply: str, *, fact_context: dict[str, object]) -> float:
        intents = set(str(intent) for intent in fact_context.get("recently_answered_question_intents", []))
        if "what_u_doing" not in intents:
            return 0.0
        question_text = extract_question_text(reply) or reply
        if normalize_question_intent(question_text) == "what_u_doing":
            return 1.0
        normalized = normalize_text(reply)
        if any(term in normalized for term in ("what u doing", "what you doing", "what are u doing", "what u been doing", "what you been doing", "wyd", "wuu2")):
            return 1.0
        if "give me a topic" in normalized and fact_context.get("recent_user_activity"):
            return 0.75
        return 0.0

    def _semantic_mismatch_penalty(self, reply: str, *, latest_message: str, fact_context: dict[str, object]) -> float:
        latest = normalize_text(latest_message)
        normalized = normalize_text(reply)
        if any(term in latest for term in ("r u mad", "are u mad", "you mad")):
            if any(term in normalized for term in ("nah im calm", "nah why", "nah im just confused", "nah i just bugged")):
                return 0.0
            return 0.9 if is_question_like_text(reply) or any(term in normalized for term in ("as u should", "give me a topic", "what u doing")) else 0.55
        if any(term in latest for term in ("confusing me", "waffling", "i js told", "just told", "already told")):
            repair_terms = ("my bad", "fairs", "fair", "bugged", "waffling", "confusing", "missed", "forgot", "dumb reply")
            if any(term in normalized for term in repair_terms):
                return 0.0
            return 0.85
        if bool(fact_context.get("recently_answered_question_detected")) and self._already_answered_question_penalty(reply, fact_context=fact_context) >= 0.7:
            return 0.85
        return 0.0

    def _context_grounding_score(self, reply: str, *, fact_context: dict[str, object], latest_message: str) -> float:
        normalized = normalize_text(reply)
        latest = normalize_text(latest_message)
        score = 0.45
        activity = str(fact_context.get("recent_user_activity") or "")
        callout = str(fact_context.get("recent_user_callout") or "")
        if activity and activity in normalized:
            score += 0.35
        if activity and any(term in normalized for term in ("same", "chilling", "nothing much", "stay there")):
            score += 0.18
        if callout and any(term in normalized for term in ("my bad", "fairs", "bugged", "waffling", "confusing", "missed", "forgot", "dumb")):
            score += 0.4
        if any(term in latest for term in ("r u mad", "are u mad")) and normalized.startswith("nah"):
            score += 0.35
        if self._already_answered_question_penalty(reply, fact_context=fact_context) >= 0.7:
            score -= 0.45
        return round(max(0.0, min(1.0, score)), 3)

    def _emotional_response_scores(
        self,
        reply: str,
        *,
        meaning_context: dict[str, object],
        relationship_type: str,
        contact_name: str | None,
    ) -> dict[str, float | bool | str]:
        meaning = str(meaning_context["latest_message_meaning"])
        normalized = normalize_text(reply)
        emotional_meanings = {
            "affection",
            "flirty_affection",
            "romantic_signoff",
            "hurt_feelings",
            "confusion",
            "serious_boundary",
            "repair_callout",
            "mixed_life_update_and_affection",
            "emotional_reciprocity_request",
            "missed_affection_callout",
            "care_checkin",
            "concern_about_bot",
            "affection_repair",
        }
        emotional_fit = 0.55
        affection_score = 0.0
        hurt_score = 0.0
        reciprocity_score = 0.0
        care_score = 0.0
        affection_missed_penalty = 0.0
        stale_emotional_penalty = 0.0
        emotional_request_unanswered = False
        shallow_penalty = 0.0
        absence_penalty = 0.0
        activity_reply_terms = ("same just chilling", "fair just chilling", "fair just chilling too", "same icl", "just chilling", "calm", "calm what u doing", "what u doing rn", "give me a topic")
        if meaning in emotional_meanings and any(term in normalized for term in activity_reply_terms):
            absence_penalty = 1.0
            stale_emotional_penalty = 1.0
            emotional_fit = 0.0
        if meaning == "mixed_life_update_and_affection":
            has_affection = any(term in normalized for term in ("missed u too", "miss u too", "aww", "sweet", "appreciate"))
            has_update = any(term in normalized for term in ("just been", "been busy", "been working", "not much", "nothing crazy", "chilling", "wbu", "uni", "projects", "gym"))
            affection_score = 1.0 if has_affection else 0.0
            reciprocity_score = 1.0 if has_affection and has_update else 0.35 if has_affection else 0.0
            emotional_fit = max(emotional_fit, reciprocity_score)
            if not (has_affection and has_update):
                emotional_request_unanswered = True
                affection_missed_penalty = max(affection_missed_penalty, 0.75)
                absence_penalty = max(absence_penalty, 0.75)
        elif meaning == "emotional_reciprocity_request":
            has_reciprocity = any(term in normalized for term in ("missed u too", "miss u too", "course i missed", "getting there"))
            has_conservative_warmth = any(term in normalized for term in ("sweet", "bless", "appreciate"))
            reciprocity_score = 1.0 if has_reciprocity else 0.65 if has_conservative_warmth else 0.0
            emotional_fit = max(emotional_fit, reciprocity_score)
            if reciprocity_score < 0.6:
                emotional_request_unanswered = True
                affection_missed_penalty = max(affection_missed_penalty, 0.9)
                absence_penalty = max(absence_penalty, 0.85)
        elif meaning == "missed_affection_callout":
            acknowledges = any(term in normalized for term in ("ur right", "my bad", "clocked it late", "missed what u meant", "that was cold", "that was dumb"))
            says_back = any(term in normalized for term in ("missed u too", "miss u too"))
            reciprocity_score = 1.0 if acknowledges and says_back else 0.55 if acknowledges else 0.0
            emotional_fit = max(emotional_fit, reciprocity_score)
            if reciprocity_score < 0.7:
                emotional_request_unanswered = True
                affection_missed_penalty = max(affection_missed_penalty, 0.9)
                absence_penalty = max(absence_penalty, 0.8)
        elif meaning in {"care_checkin", "concern_about_bot"}:
            answers_ok = any(term in normalized for term in ("im good", "i'm good", "im okay", "i'm okay", "im alright", "i'm alright", "yeah im good", "yeah im okay"))
            has_reason = any(term in normalized for term in ("off today", "tired", "head's just gone", "head just gone", "being weird"))
            care_score = 1.0 if answers_ok and has_reason else 0.75 if answers_ok else 0.0
            emotional_fit = max(emotional_fit, care_score)
            if care_score < 0.7:
                emotional_request_unanswered = True
                absence_penalty = max(absence_penalty, 0.85)
        elif meaning in {"affection", "flirty_affection", "romantic_signoff"}:
            warm_terms = ("missed u too", "miss u too", "miss u more", "miss you too", "love u too", "love you too", "love u more", "goodnight", "good night", "sleep well", "sleep tight", "mwah", "aww", "sweet", "appreciate", "same icl where u been")
            affectionate = any(term in normalized for term in warm_terms)
            affection_score = 1.0 if affectionate else 0.0
            if meaning == "romantic_signoff":
                signoff_answer = any(term in normalized for term in ("goodnight", "good night", "sleep well", "sleep tight"))
                reciprocity_score = 1.0 if affectionate and signoff_answer else 0.6 if affectionate else 0.0
                if reciprocity_score < 0.6:
                    emotional_request_unanswered = True
                    affection_missed_penalty = max(affection_missed_penalty, 0.9)
                    absence_penalty = max(absence_penalty, 0.85)
            emotional_fit = max(emotional_fit, affection_score)
            if not affectionate:
                absence_penalty = max(absence_penalty, 0.8)
            if meaning == "flirty_affection" and not self._flirt_allowed(contact_name, relationship_type):
                if any(term in normalized for term in ("baby", "babe", "my love", "need u", "want u")):
                    absence_penalty = max(absence_penalty, 0.8)
        elif meaning == "confusion":
            repair_terms = ("made no sense", "waffling", "my bad", "confusing", "ignore that", "i meant")
            if any(term in normalized for term in repair_terms):
                emotional_fit = 1.0
            else:
                absence_penalty = max(absence_penalty, 0.85)
        elif meaning in {"hurt_feelings", "serious_boundary"}:
            acknowledgement = any(term in normalized for term in ("ur right", "you're right", "youre right", "i get why", "i didn't mean", "didnt mean", "that was on me", "my bad"))
            emotional_repair = any(term in normalized for term in ("feel like that", "annoyed u", "felt off", "being dry", "taking the piss", "on me"))
            hurt_score = 1.0 if acknowledgement and emotional_repair else 0.35 if acknowledgement else 0.0
            emotional_fit = max(emotional_fit, hurt_score)
            shallow_repairs = {
                "my bad",
                "yeah my bad",
                "nah ur right",
                "i bugged",
                "nah ur right i bugged",
                "yeah that was dead my bad",
                "that was dead my bad",
            }
            if normalized in shallow_repairs or (len(normalized.split()) <= 5 and acknowledgement):
                shallow_penalty = 1.0
            if "lol" in normalized:
                absence_penalty = max(absence_penalty, 0.9)
            if hurt_score < 0.7:
                absence_penalty = max(absence_penalty, 0.75)
        elif meaning == "repair_callout":
            if any(term in normalized for term in ("my bad", "made no sense", "waffling", "bugged", "dead reply", "random")):
                emotional_fit = 0.9
            else:
                absence_penalty = max(absence_penalty, 0.6)
        return {
            "emotional_fit_score": round(max(0.0, min(1.0, emotional_fit)), 3),
            "affection_response_score": round(max(0.0, min(1.0, affection_score)), 3),
            "hurt_repair_score": round(max(0.0, min(1.0, hurt_score)), 3),
            "shallow_repair_penalty": round(max(0.0, min(1.0, shallow_penalty)), 3),
            "emotional_absence_penalty": round(max(0.0, min(1.0, absence_penalty)), 3),
            "emotional_reciprocity_score": round(max(0.0, min(1.0, reciprocity_score)), 3),
            "care_checkin_fit_score": round(max(0.0, min(1.0, care_score)), 3),
            "affection_missed_penalty": round(max(0.0, min(1.0, affection_missed_penalty)), 3),
            "stale_emotional_reply_penalty": round(max(0.0, min(1.0, stale_emotional_penalty)), 3),
            "emotional_request_unanswered": emotional_request_unanswered,
        }

    def _conversation_function_scores(
        self,
        reply: str,
        *,
        function_context: dict[str, object],
    ) -> dict[str, float | bool | str]:
        function = str(function_context["conversation_function"])
        tolerance = str(function_context["social_risk_tolerance"])
        normalized = normalize_text(reply).strip("?!., ")
        stale_self_state_terms = {
            "same just chilling",
            "same icl",
            "fair just chilling too",
            "just chilling",
            "calm",
            "nothing much u",
        }
        bland_terms = {
            "my bad",
            "yeah my bad",
            "nah ur right",
            "calm",
            "true",
            "fair",
            "same icl",
            "sorry",
            "yeah that was dead my bad",
        }
        fit = 0.55
        recent_life_fit = 0.0
        stale_penalty = 0.0
        over_safe_penalty = 0.0
        bland_repair_penalty = 0.0

        if function in {"dry_complaint", "repeated_reply_callout", "mocking_after_repair", "context_failure_question"} and (
            normalized in stale_self_state_terms or any(term in normalized for term in ("same just chilling", "fair just chilling", "same icl"))
        ):
            stale_penalty = 1.0
            fit = 0.0

        if function == "dry_complaint":
            good_terms = ("dry", "dead", "repeating myself", "defaulting to chilling", "giving me much", "give me a topic", "im trying", "what u been doing", "what should we talk about")
            if any(term in normalized for term in good_terms):
                fit = max(fit, 1.0)
            elif normalized in bland_terms or is_question_like_text(reply):
                fit = min(fit, 0.25)
                bland_repair_penalty = max(bland_repair_penalty, 0.75)
        elif function == "recent_life_update_question":
            if normalized in stale_self_state_terms or any(term in normalized for term in ("same just chilling", "fair just chilling", "same icl")):
                stale_penalty = 1.0
                fit = 0.0
            elif any(term in normalized for term in ("been", "nothing crazy", "not much", "usual", "bit dead", "wbu", "what about u")):
                recent_life_fit = 1.0
                fit = 1.0
            else:
                recent_life_fit = 0.25
                fit = 0.35
        elif function == "repeated_reply_callout":
            good_terms = ("repeated", "same thing", "caught me", "bugged", "npc behaviour")
            if any(term in normalized for term in good_terms):
                fit = 1.0
            elif normalized in stale_self_state_terms or normalized in bland_terms or is_question_like_text(reply):
                fit = 0.0
                bland_repair_penalty = 0.9
            else:
                fit = 0.45
        elif function == "mocking_after_repair":
            good_terms = ("deserved", "walked into", "allow me", "set myself up")
            if any(term in normalized for term in good_terms):
                fit = 1.0
            elif normalized in stale_self_state_terms or normalized in bland_terms or normalized == "my bad":
                fit = 0.0
                bland_repair_penalty = 1.0
            else:
                fit = 0.45
        elif function == "context_failure_question":
            good_terms = ("repeated myself", "clocking what u asked", "asked wyd", "missed what u said", "defaulted to chilling", "lost the context", "didnt clock", "didn't clock")
            if any(term in normalized for term in good_terms):
                fit = 1.0
            elif normalized in stale_self_state_terms or normalized in bland_terms or "too flat" in normalized or is_question_like_text(reply):
                fit = 0.0
                bland_repair_penalty = 1.0
            else:
                fit = 0.35

        if tolerance == "expressive" and function in {"repeated_reply_callout", "mocking_after_repair", "context_failure_question"}:
            if normalized in bland_terms:
                over_safe_penalty = 0.9
            elif len(normalized.split()) <= 3 and not any(term in normalized for term in ("caught me", "allow me")):
                over_safe_penalty = 0.65
            elif any(term in normalized for term in ("npc behaviour", "walked into", "caught me", "allow me", "waffling")):
                fit = min(1.0, fit + 0.1)

        return {
            "conversation_function": function,
            "conversation_function_reason": str(function_context["conversation_function_reason"]),
            "social_risk_tolerance": tolerance,
            "recent_life_update_fit_score": round(max(0.0, min(1.0, recent_life_fit)), 3),
            "stale_self_state_penalty": round(max(0.0, min(1.0, stale_penalty)), 3),
            "repeated_reply_callout_detected": bool(function_context["repeated_reply_callout_detected"]),
            "mocking_after_repair_detected": bool(function_context["mocking_after_repair_detected"]),
            "context_failure_question_detected": bool(function_context["context_failure_question_detected"]),
            "over_safe_penalty": round(max(0.0, min(1.0, over_safe_penalty)), 3),
            "bland_repair_penalty": round(max(0.0, min(1.0, bland_repair_penalty)), 3),
            "conversational_function_fit_score": round(max(0.0, min(1.0, fit)), 3),
        }

    def _candidate_stance(self, reply: str) -> str:
        normalized = normalize_text(reply)
        if not normalized:
            return "neutral"
        correction = ("my bad", "you're right", "youre right", "i meant", "i should've", "i should have")
        playful_defense = (
            "nah",
            "how is that crazy",
            "how's that crazy",
            "ur dragging it",
            "you're dragging it",
            "youre dragging it",
            "bro what",
            "bro",
            "not crazy",
            "aint crazy",
            "ain't crazy",
            "u said",
            "you said",
        )
        concede = (
            "it really is",
            "it is",
            "yeah true",
            "fair enough",
            "you're right",
            "youre right",
            "yeah it is",
        )
        playful_defense = (*playful_defense, "it really isnt", "it really isn't", "yeah it isnt", "yeah it isn't", "it isnt", "it isn't")
        if any(term in normalized for term in correction):
            return "correcting"
        if any(term in normalized for term in playful_defense):
            return "defend"
        if any(term in normalized for term in concede):
            return "concede"
        if "?" in normalized:
            return "challenge"
        return "neutral"

    def _candidate_claim(self, reply: str) -> str:
        normalized = normalize_text(reply)
        if any(term in normalized for term in ("come over", "come round", "come thru", "come through", "go out then", "find smth to do", "do smth then", "do something then")):
            return "invite_or_plan"
        if any(term in normalized for term in ("crazy", "dragging", "bro what", "how is that crazy", "not crazy")):
            return "banter_defense"
        if "?" in normalized:
            return "question"
        return " ".join(str(reply).split()).strip()[:80]

    def _repetition_penalty(self, reply: str, previous_replies: list[str]) -> float:
        normalized = normalize_text(reply)
        if not normalized:
            return 1.0
        hot_phrases = {"it really is", "it really isnt", "it really isn't", "fair enough", "yeah true"}
        penalty = 0.0
        if normalized in hot_phrases:
            penalty = max(penalty, 0.9)
        for previous in previous_replies[:3]:
            score = SequenceMatcher(None, normalized, normalize_text(previous)).ratio()
            if score >= 0.92:
                penalty = max(penalty, 1.0)
            elif score >= 0.82:
                penalty = max(penalty, 0.7)
            if self._generic_prompt_family(normalized) and self._generic_prompt_family(normalized) == self._generic_prompt_family(previous):
                penalty = max(penalty, 0.85)
        return round(min(1.0, penalty), 3)

    def _generic_prompt_family(self, text: str) -> str:
        normalized = normalize_text(text).strip(" .?!")
        families = {
            "what_u_saying": ("what u saying", "what you saying"),
            "what_u_been_up_to": ("what u been up to", "what you been up to", "whatchu been up to", "what have u been up to"),
            "what_u_doing": ("what u doing", "what you doing", "what are u doing", "wyd", "wuu2"),
            "ask_topic": ("give me a topic", "what should we talk about", "what topic"),
            "no_context_blame": ("u ain't giving me much", "u aint giving me much", "giving me nothing"),
        }
        for family, terms in families.items():
            if any(term in normalized for term in terms):
                return family
        return ""

    def _banter_fit_score(self, reply: str, *, incoming_intent: str, latest_message: str) -> float:
        normalized = normalize_text(reply)
        if normalize_intent_label(incoming_intent) != "banter_challenge":
            return 0.35 if self._candidate_stance(reply) == "neutral" else 0.5
        playful_hits = sum(1 for term in ("nah", "bro", "what", "crazy", "dragging", "u said", "you said", "😭") if term in normalized)
        literal_hits = sum(1 for term in ("it really is", "it really isnt", "fair enough", "yeah true") if term in normalized)
        score = 0.25 + min(0.55, playful_hits * 0.18) - min(0.45, literal_hits * 0.22)
        if "?" in latest_message:
            score += 0.08
        if self._candidate_stance(reply) == "defend":
            score += 0.15
        return round(max(0.0, min(1.0, score)), 3)

    def _relationship_boldness_score(
        self,
        reply: str,
        *,
        relationship_type: str,
        contact_name: str | None,
    ) -> float:
        normalized = normalize_text(reply)
        profile = self._contact_profile_for(contact_name)
        boldness = float(profile.boldness_level if profile is not None else self._default_boldness(relationship_type))
        allowed_relationship = relationship_type in {"close_friend", "romantic_interest"}
        bold_phrases = ("come over then", "come thru", "come through", "come round", "come here")
        if any(phrase in normalized for phrase in bold_phrases):
            if not allowed_relationship:
                return 0.0
            if not (profile and profile.banter_allowed):
                return 0.0
            if boldness < 0.6:
                return round(max(0.0, boldness - 0.4), 3)
            if relationship_type == "romantic_interest" and not (profile and profile.flirt_allowed):
                return 0.0
            return round(min(1.0, boldness), 3)
        if relationship_type in {"professional", "university", "family"}:
            return 0.25 if any(term in normalized for term in ("do smth then", "go out then", "find smth to do")) else 0.45
        return round(min(1.0, 0.45 + boldness * 0.5), 3)

    def _question_quality_scores(
        self,
        reply: str,
        *,
        question_mode: str,
        question_reason: str,
        recent_bot_questions_count: int,
        latest_message: str,
        recent_bot_question: str | None,
        recent_bot_replies: list[str],
    ) -> dict[str, float | str]:
        normalized = normalize_text(reply)
        question_text = extract_question_text(reply)
        question_like = bool(question_text)
        formal_penalties = (
            "how does that make you feel",
            "can you elaborate",
            "what are your thoughts on that",
            "tell me more about this situation",
            "what about you",
            "what about u",
        )
        useful_matches = {
            "light_follow_up": (
                "what happened",
                "go on then",
                "what now",
                "what u doing",
                "what u tryna do",
                "same icl",
                "same wanna do smth",
                "what time",
            ),
            "curious_follow_up": (
                "how does that even happen",
                "bro why",
                "nah what",
                "what do you think",
                "what do u think",
                "how come",
            ),
            "proactive_prompt": (
                "what u been doing",
                "what should we talk about",
                "fine then what should we talk about",
                "what should we do",
                "what u tryna do",
                "give me a topic",
            ),
            "topic_shift": (
                "anyways what",
                "what were u gonna say",
                "what u actually doing",
                "move the convo",
                "what now",
            ),
        }
        usefulness = 0.0
        naturalness = 0.0
        pressure_penalty = 0.0
        repeated_penalty = 0.0

        if question_like:
            question_intent = normalize_question_intent(question_text or "")
            recent_question_intents = [
                normalize_question_intent(previous)
                for previous in recent_bot_replies[:4]
                if is_question_like_text(previous)
            ]
            matches = useful_matches.get(question_mode, ())
            if any(term in normalized for term in matches):
                usefulness = 1.0
            elif question_mode == "none":
                usefulness = 0.1
            else:
                usefulness = 0.35
            naturalness = 0.55
            if len(reply.split()) <= 8:
                naturalness += 0.22
            if len(reply.split()) <= 5:
                naturalness += 0.1
            if reply == reply.lower():
                naturalness += 0.05
            if any(term in normalized for term in formal_penalties):
                naturalness -= 0.42
                pressure_penalty = max(pressure_penalty, 0.65)
            if normalized.count("?") >= 2:
                naturalness -= 0.12
                pressure_penalty = max(pressure_penalty, 0.25)
            if len(reply.split()) > 12:
                naturalness -= 0.12
                pressure_penalty = max(pressure_penalty, 0.18)
            if recent_bot_questions_count >= 2:
                pressure_penalty = max(pressure_penalty, 0.35)
            if recent_bot_question and question_intent == normalize_question_intent(recent_bot_question):
                repeated_penalty = 1.0
                pressure_penalty = max(pressure_penalty, 0.75)
            elif question_intent and question_intent in recent_question_intents:
                repeated_penalty = 1.0
                pressure_penalty = max(pressure_penalty, 0.75)
            elif question_mode != "none":
                previous_questions = [
                    extract_question_text(previous)
                    for previous in recent_bot_replies[:3]
                    if is_question_like_text(previous)
                ]
                for previous_question in previous_questions:
                    if previous_question and normalize_text(previous_question) == normalize_text(question_text or ""):
                        repeated_penalty = 1.0
                        pressure_penalty = max(pressure_penalty, 0.75)
                        break
                if not repeated_penalty and previous_questions:
                    for previous_question in previous_questions:
                        if previous_question and SequenceMatcher(
                            None,
                            normalize_text(previous_question),
                            normalize_text(question_text or ""),
                        ).ratio() >= 0.9:
                            repeated_penalty = max(repeated_penalty, 0.8)
                            pressure_penalty = max(pressure_penalty, 0.55)
                            break
            if any(term in normalized for term in ("what about you", "what about u")) and recent_bot_questions_count > 0:
                pressure_penalty = max(pressure_penalty, 0.6)
        else:
            if question_mode == "none":
                usefulness = 0.0
            else:
                usefulness = 0.15
                pressure_penalty = 0.15
                naturalness = 0.2

        if question_like and question_mode in {"light_follow_up", "curious_follow_up", "proactive_prompt", "topic_shift"}:
            naturalness = max(0.0, min(1.0, naturalness))
            if question_mode == "topic_shift" and any(term in normalized for term in ("anyways", "what were u gonna say", "what u actually doing")):
                usefulness = max(usefulness, 0.95)
            if question_mode == "proactive_prompt" and any(term in normalized for term in ("what u been doing", "what should we talk about", "give me a topic")):
                usefulness = max(usefulness, 0.95)
            if question_mode == "curious_follow_up" and any(term in normalized for term in ("how does that even happen", "bro why", "nah what")):
                usefulness = max(usefulness, 0.95)
            if question_mode == "light_follow_up" and any(term in normalized for term in ("what happened", "go on then", "what now", "what u doing")):
                usefulness = max(usefulness, 0.95)

        return {
            "question_usefulness_score": round(max(0.0, min(1.0, usefulness)), 3),
            "question_naturalness_score": round(max(0.0, min(1.0, naturalness)), 3),
            "question_pressure_penalty": round(max(0.0, min(1.0, pressure_penalty)), 3),
            "repeated_question_penalty": round(max(0.0, min(1.0, repeated_penalty)), 3),
            "question_text": question_text or "",
        }

    def _default_boldness(self, relationship_type: str) -> float:
        return {
            "romantic_interest": 0.8,
            "close_friend": 0.7,
            "casual_friend": 0.45,
            "family": 0.2,
            "professional": 0.1,
            "university": 0.2,
        }.get(relationship_type, 0.35)

    def _stance_consistency_score(self, reply: str, *, contact_name: str | None) -> tuple[float, float, str | None]:
        state = self._conversation_state_for(contact_name)
        previous_stance = str(state.last_bot_stance or "").strip().lower()
        current_stance = self._candidate_stance(reply)
        current_claim = self._candidate_claim(reply)
        disputed_topic = state.disputed_topic
        recent_stances = [str(item).strip().lower() for item in state.recent_bot_stances[:3] if str(item).strip()]
        risk = 0.0
        if (
            current_stance != "correcting"
            and previous_stance in {"defend", "concede"}
            and current_stance in {"defend", "concede"}
            and previous_stance != current_stance
        ):
            risk = 1.0
        if current_stance == "defend" and any(stance == "concede" for stance in recent_stances):
            risk = 1.0
        if current_stance == "concede" and any(stance == "defend" for stance in recent_stances):
            risk = 1.0
        if previous_stance == "defend" and current_stance == "concede" and disputed_topic in {"banter_challenge", "invite_or_plan"}:
            risk = 1.0
        if previous_stance == "concede" and current_stance == "defend" and disputed_topic in {"banter_challenge", "invite_or_plan"}:
            risk = 0.9
        score = max(0.0, 1.0 - risk)
        if current_stance == "neutral":
            score = max(score, 0.45)
        if current_stance == "correcting":
            score = max(score, 0.6)
        return round(score, 3), round(risk, 3), current_claim or disputed_topic

    def _reply_analysis(
        self,
        reply: str,
        *,
        contact_name: str | None,
        relationship_type: str,
        latest_message: str,
        incoming_intent: str,
        full_conversation: list[str] | None = None,
        question_context: dict[str, object] | None = None,
        hook_context: dict[str, object] | None = None,
    ) -> dict[str, float | str | None]:
        state = self._conversation_state_for(contact_name)
        context_messages = full_conversation or [latest_message]
        stance_score, contradiction_risk, claim = self._stance_consistency_score(reply, contact_name=contact_name)
        context_bot_replies = [
            item.split(":", 1)[1].strip()
            for item in context_messages
            if item.casefold().startswith("[me]:") and ":" in item
        ]
        repetition_penalty = self._repetition_penalty(reply, [*context_bot_replies, *state.recent_bot_replies])
        banter_fit_score = self._banter_fit_score(reply, incoming_intent=incoming_intent, latest_message=latest_message)
        relationship_boldness_score = self._relationship_boldness_score(reply, relationship_type=relationship_type, contact_name=contact_name)
        energy_context = self._reply_energy_context(
            [latest_message],
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        reply_energy_mode = str(energy_context["reply_energy_mode"])
        energy_reason = str(energy_context["energy_reason"])
        criticism_detected = bool(energy_context["criticism_detected"])
        lazy_lol_penalty = self._lazy_lol_penalty(
            reply,
            energy_mode=reply_energy_mode,
            criticism_detected=criticism_detected,
        )
        filler_penalty = self._filler_penalty(
            reply,
            energy_mode=reply_energy_mode,
            criticism_detected=criticism_detected,
            lazy_lol_penalty=lazy_lol_penalty,
        )
        expressive_score = self._expressive_score(
            reply,
            energy_mode=reply_energy_mode,
            criticism_detected=criticism_detected,
            filler_penalty=filler_penalty,
        )
        if question_context is None:
            question_context = self._question_mode_context(
                [latest_message],
                contact_name=contact_name,
                relationship_type=relationship_type,
                incoming_intent=incoming_intent,
            )
        if hook_context is None:
            hook_context = self._hook_context(
                [latest_message],
                contact_name=contact_name,
                relationship_type=relationship_type,
                incoming_intent=incoming_intent,
            )
        fact_context = self._recent_fact_context(context_messages, contact_name=contact_name)
        if hook_context:
            fact_context = {
                **fact_context,
                "recent_user_activity": hook_context.get("recent_user_activity", fact_context["recent_user_activity"]),
                "recent_user_mood": hook_context.get("recent_user_mood", fact_context["recent_user_mood"]),
                "recent_user_callout": hook_context.get("recent_user_callout", fact_context["recent_user_callout"]),
                "recently_answered_question_intents": hook_context.get("recently_answered_question_intents", fact_context["recently_answered_question_intents"]),
                "recently_answered_question_detected": hook_context.get("recently_answered_question_detected", fact_context["recently_answered_question_detected"]),
            }
        question_scores = self._question_quality_scores(
            reply,
            question_mode=str(question_context["question_mode"]),
            question_reason=str(question_context["question_reason"]),
            recent_bot_questions_count=int(question_context["recent_bot_questions_count"]),
            latest_message=latest_message,
            recent_bot_question=state.last_bot_question,
            recent_bot_replies=[*getattr(state, "recent_bot_questions", []), *state.recent_bot_replies],
        )
        hook_scores = self._hook_quality_scores(
            reply,
            hook_context=hook_context,
            relationship_type=relationship_type,
            contact_name=contact_name,
        )
        low_information_incoming = bool(hook_context.get("low_information_incoming"))
        continuation_required = bool(hook_context.get("continuation_required"))
        repair_context = self._repair_context([latest_message])
        copycat_penalty = self._copycat_penalty(reply, latest_message)
        romantic_term_blocked = self._romantic_term_blocked(
            reply,
            relationship_type=relationship_type,
            contact_name=contact_name,
        )
        already_answered_question_penalty = self._already_answered_question_penalty(reply, fact_context=fact_context)
        semantic_mismatch_penalty = self._semantic_mismatch_penalty(
            reply,
            latest_message=latest_message,
            fact_context=fact_context,
        )
        context_grounding_score = self._context_grounding_score(
            reply,
            latest_message=latest_message,
            fact_context=fact_context,
        )
        context_fit = self._context_fit_analysis(reply, latest_message=latest_message)
        meaning_context = self._latest_meaning_context(
            context_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        emotional_scores = self._emotional_response_scores(
            reply,
            meaning_context=meaning_context,
            relationship_type=relationship_type,
            contact_name=contact_name,
        )
        function_context = self._conversation_function_context(
            context_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        function_scores = self._conversation_function_scores(
            reply,
            function_context=function_context,
        )
        scene = self._conversation_scene(
            context_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        labelled_bot_replies = [
            item.split(":", 1)[1].strip()
            for item in context_messages
            if item.casefold().startswith("[me]:") and ":" in item
        ]
        scene_scores = score_reply_against_scene(
            reply,
            scene,
            previous_bot_replies=[*labelled_bot_replies, *state.recent_bot_replies],
        )
        question_debt = self._question_debt(context_messages)
        question_debt_scores = score_reply_against_question_debt(reply, question_debt)
        agenda = self._conversation_agenda(context_messages, scene=scene, question_debt=question_debt)
        agenda_scores = score_reply_against_agenda(
            reply,
            agenda,
            previous_bot_replies=[*labelled_bot_replies, *state.recent_bot_replies],
        )
        policy = self._conversation_policy(context_messages, scene=scene, agenda=agenda, relationship_type=relationship_type, question_debt=question_debt)
        policy_scores = score_reply_against_policy(reply, policy)
        training_style_scores = self.style_ranker.score(reply, relationship_type=relationship_type)
        identity_scores = self._identity_scores(reply, scene, relationship_type=relationship_type)
        thread_context = self._thread_memory_context(
            context_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        thread_scores = self._thread_memory_scores(reply, thread_context)
        return {
            "stance_consistency_score": stance_score,
            "repetition_penalty": repetition_penalty,
            "banter_fit_score": banter_fit_score,
            "relationship_boldness_score": relationship_boldness_score,
            "contradiction_risk": contradiction_risk,
            "selected_claim": claim,
            "selected_stance": self._candidate_stance(reply),
            "question_mode": str(question_context["question_mode"]),
            "question_reason": str(question_context["question_reason"]),
            "question_usefulness_score": float(question_scores["question_usefulness_score"]),
            "question_naturalness_score": float(question_scores["question_naturalness_score"]),
            "question_pressure_penalty": float(question_scores["question_pressure_penalty"]),
            "repeated_question_penalty": float(question_scores["repeated_question_penalty"]),
            "recent_bot_questions_count": int(question_context["recent_bot_questions_count"]),
            "hook_mode": str(hook_context["hook_mode"]),
            "hook_reason": str(hook_context["hook_reason"]),
            "hook_required": bool(hook_context["hook_required"]),
            "hook_presence_score": float(hook_scores["hook_presence_score"]),
            "hook_relevance_score": float(hook_scores["hook_relevance_score"]),
            "hook_naturalness_score": float(hook_scores["hook_naturalness_score"]),
            "hook_pressure_penalty": float(hook_scores["hook_pressure_penalty"]),
            "hook_safety_penalty": float(hook_scores["hook_safety_penalty"]),
            "conversation_momentum_score": float(hook_scores["conversation_momentum_score"]),
            "dead_reply_penalty": float(hook_scores["dead_reply_penalty"]),
            "invalid_provider_output": bool(hook_scores["invalid_provider_output"]),
            "low_information_incoming": low_information_incoming,
            "continuation_required": continuation_required,
            "continuation_score": float(hook_scores["continuation_score"]),
            "mirror_reply_penalty": float(hook_scores["mirror_reply_penalty"]),
            "reply_energy_mode": reply_energy_mode,
            "energy_reason": energy_reason,
            "filler_penalty": filler_penalty,
            "expressive_score": expressive_score,
            "criticism_detected": criticism_detected,
            "lazy_lol_penalty": lazy_lol_penalty,
            "repeated_question_detected": float(question_scores["repeated_question_penalty"]) >= 0.7,
            "copycat_penalty": copycat_penalty,
            "copycat_detected": copycat_penalty >= 0.7,
            "repair_required": bool(repair_context["repair_required"]),
            "escalation_detected": bool(repair_context["escalation_detected"]),
            "romantic_term_blocked": romantic_term_blocked,
            "recent_user_activity": str(fact_context.get("recent_user_activity") or ""),
            "recent_user_mood": str(fact_context.get("recent_user_mood") or ""),
            "recent_user_callout": str(fact_context.get("recent_user_callout") or ""),
            "recently_answered_question_detected": bool(fact_context.get("recently_answered_question_detected")),
            "already_answered_question_penalty": already_answered_question_penalty,
            "repair_over_hook_applied": bool(hook_context.get("repair_over_hook_applied")),
            "hook_suppressed_reason": str(hook_context.get("hook_suppressed_reason") or ""),
            "semantic_mismatch_penalty": semantic_mismatch_penalty,
            "context_grounding_score": context_grounding_score,
            "context_fit_score": float(context_fit["context_fit_score"]),
            "invalid_for_context": bool(context_fit["invalid_for_context"]),
            "invalid_context_reason": str(context_fit["invalid_context_reason"]),
            "weird_punctuation_penalty": float(context_fit["weird_punctuation_penalty"]),
            "input_category": str(context_fit["input_category"]),
            "latest_message_meaning": str(meaning_context["latest_message_meaning"]),
            "meaning_priority": int(meaning_context["meaning_priority"]),
            "emotional_context_detected": bool(meaning_context["emotional_context_detected"]),
            "affection_detected": bool(meaning_context["affection_detected"]),
            "hurt_detected": bool(meaning_context["hurt_detected"]),
            "confusion_detected": bool(meaning_context["confusion_detected"]),
            "serious_callout_detected": bool(meaning_context["serious_callout_detected"]),
            "activity_grounding_suppressed": bool(meaning_context["activity_grounding_suppressed"]),
            "activity_grounding_suppressed_reason": str(meaning_context["activity_grounding_suppressed_reason"]),
            "emotional_fit_score": float(emotional_scores["emotional_fit_score"]),
            "affection_response_score": float(emotional_scores["affection_response_score"]),
            "hurt_repair_score": float(emotional_scores["hurt_repair_score"]),
            "shallow_repair_penalty": float(emotional_scores["shallow_repair_penalty"]),
            "emotional_absence_penalty": float(emotional_scores["emotional_absence_penalty"]),
            "emotional_reciprocity_score": float(emotional_scores["emotional_reciprocity_score"]),
            "care_checkin_fit_score": float(emotional_scores["care_checkin_fit_score"]),
            "affection_missed_penalty": float(emotional_scores["affection_missed_penalty"]),
            "stale_emotional_reply_penalty": float(emotional_scores["stale_emotional_reply_penalty"]),
            "emotional_request_unanswered": bool(emotional_scores["emotional_request_unanswered"]),
            "conversation_function": str(function_scores["conversation_function"]),
            "conversation_function_reason": str(function_scores["conversation_function_reason"]),
            "social_risk_tolerance": str(function_scores["social_risk_tolerance"]),
            "recent_life_update_fit_score": float(function_scores["recent_life_update_fit_score"]),
            "stale_self_state_penalty": float(function_scores["stale_self_state_penalty"]),
            "repeated_reply_callout_detected": bool(function_scores["repeated_reply_callout_detected"]),
            "mocking_after_repair_detected": bool(function_scores["mocking_after_repair_detected"]),
            "context_failure_question_detected": bool(function_scores["context_failure_question_detected"]),
            "over_safe_penalty": float(function_scores["over_safe_penalty"]),
            "bland_repair_penalty": float(function_scores["bland_repair_penalty"]),
            "conversational_function_fit_score": float(function_scores["conversational_function_fit_score"]),
            "scene_type": scene.scene_type,
            "scene_summary": scene.scene_summary,
            "latest_user_ask": scene.latest_user_ask,
            "latest_user_emotion": scene.latest_user_emotion,
            "social_task": scene.social_task,
            "required_reply_move": scene.required_reply_move,
            "forbidden_reply_moves": scene.forbidden_reply_moves,
            "known_recent_facts": scene.known_recent_facts,
            "recent_bot_mistakes": scene.recent_bot_mistakes,
            "unresolved_user_points": scene.unresolved_user_points,
            "previous_bot_claim": scene.previous_bot_claim,
            "previous_bot_claim_type": scene.previous_bot_claim_type,
            "latest_user_refers_to_previous_bot_claim": scene.latest_user_refers_to_previous_bot_claim,
            "explanation_required": scene.explanation_required,
            "explanation_target": scene.explanation_target,
            "active_topic": scene.active_topic,
            "topic_was_provided": scene.topic_was_provided,
            "topic_value": scene.topic_value,
            "bot_repeated_itself": scene.bot_repeated_itself,
            "user_called_out_bot": scene.user_called_out_bot,
            "affection_reciprocity_required": scene.affection_reciprocity_required,
            "care_response_required": scene.care_response_required,
            "topic_engagement_required": scene.topic_engagement_required,
            "scene_fit_score": float(scene_scores["scene_fit_score"]),
            "required_move_satisfied": bool(scene_scores["required_move_satisfied"]),
            "forbidden_move_violated": bool(scene_scores["forbidden_move_violated"]),
            "scene_mismatch_reason": str(scene_scores["scene_mismatch_reason"]),
            "contextual_specificity_score": float(scene_scores["contextual_specificity_score"]),
            "human_likeness_score": float(scene_scores["human_likeness_score"]),
            "canned_reply_penalty": float(scene_scores["canned_reply_penalty"]),
            "stale_template_penalty": float(scene_scores["stale_template_penalty"]),
            "fallback_scene_type": str(scene_scores["fallback_scene_type"]),
            "fallback_required_move": str(scene_scores["fallback_required_move"]),
            "fallback_scene_mismatch": bool(scene_scores["fallback_scene_mismatch"]),
            "semantic_contamination_penalty": float(scene_scores["semantic_contamination_penalty"]),
            "previous_claim_explanation_score": float(scene_scores["previous_claim_explanation_score"]),
            "explanation_required": bool(scene_scores["explanation_required"]),
            "explanation_target": str(scene_scores["explanation_target"]),
            "thread_memory_match_score": float(thread_scores["thread_memory_match_score"]),
            "retrieved_thread_count": int(thread_scores["retrieved_thread_count"]),
            "repeated_failed_pattern_penalty": float(thread_scores["repeated_failed_pattern_penalty"]),
            "successful_pattern_match_score": float(thread_scores["successful_pattern_match_score"]),
            "unresolved_thread_point_score": float(thread_scores["unresolved_thread_point_score"]),
            "thread_memory_used": bool(thread_scores["thread_memory_used"]),
            "active_topic_engagement_score": float(scene_scores["active_topic_engagement_score"]),
            "direct_answer_score": float(scene_scores["direct_answer_score"]),
            "repair_specificity_score": float(scene_scores["repair_specificity_score"]),
            "emotional_presence_score": float(scene_scores["emotional_presence_score"]),
            "unresolved_point_addressed": bool(scene_scores["unresolved_point_addressed"]),
            "identity_question_detected": bool(scene_scores.get("identity_question_detected")),
            "identity_fact_used": str(identity_scores["identity_fact_used"]),
            "identity_disclosure_allowed": bool(scene_scores.get("identity_disclosure_allowed")),
            "identity_answer_score": float(scene_scores.get("identity_answer_score", 0.0)),
            "identity_hallucination_risk": float(identity_scores["identity_hallucination_risk"]),
            "ignored_identity_question_penalty": float(scene_scores.get("ignored_identity_question_penalty", 0.0)),
            "stale_identity_fallback_penalty": float(scene_scores.get("stale_identity_fallback_penalty", 0.0)),
            "direct_identity_answer_required": bool(scene_scores.get("direct_identity_answer_required")),
            "identity_specificity_score": float(scene_scores.get("identity_specificity_score", 0.0)),
            "human_scene_response_score": float(identity_scores["human_scene_response_score"]),
            "conversational_presence_score": float(identity_scores["conversational_presence_score"]),
            "directness_score": float(identity_scores["directness_score"]),
            "specificity_score": float(identity_scores["specificity_score"]),
            "has_unanswered_user_question": bool(question_debt_scores["has_unanswered_user_question"]),
            "unanswered_question_type": str(question_debt_scores["unanswered_question_type"]),
            "unanswered_question_text": str(question_debt_scores["unanswered_question_text"]),
            "question_debt_age_turns": int(question_debt_scores["question_debt_age_turns"]),
            "user_called_out_unanswered_question": bool(question_debt_scores["user_called_out_unanswered_question"]),
            "answer_required_now": bool(question_debt_scores["answer_required_now"]),
            "question_debt_answer_score": float(question_debt_scores["question_debt_answer_score"]),
            "ignored_question_debt_penalty": float(question_debt_scores["ignored_question_debt_penalty"]),
            "asked_new_question_before_answering_penalty": float(question_debt_scores["asked_new_question_before_answering_penalty"]),
            "agenda_state": str(agenda_scores["agenda_state"]),
            "next_dialogue_move": str(agenda_scores["next_dialogue_move"]),
            "bot_loop_detected": bool(agenda_scores["bot_loop_detected"]),
            "anti_loop_required": bool(agenda_scores["anti_loop_required"]),
            "exhausted_prompt_types": list(agenda_scores["exhausted_prompt_types"]) if isinstance(agenda_scores["exhausted_prompt_types"], list) else [],
            "generic_prompt_penalty": float(agenda_scores["generic_prompt_penalty"]),
            "repeated_prompt_penalty": float(agenda_scores["repeated_prompt_penalty"]),
            "agenda_fit_score": float(agenda_scores["agenda_fit_score"]),
            "forbidden_dialogue_move_violated": bool(agenda_scores["forbidden_dialogue_move_violated"]),
            "forbidden_dialogue_moves_violated": list(agenda_scores["forbidden_dialogue_moves_violated"]) if isinstance(agenda_scores["forbidden_dialogue_moves_violated"], list) else [],
            "conversation_progress_score": float(agenda_scores["conversation_progress_score"]),
            "chosen_topic": str(agenda_scores["chosen_topic"]),
            "conversation_job": policy.conversation_job,
            "policy_reason": policy.policy_reason,
            "bot_obligation": policy.bot_obligation,
            "policy_must_answer": policy.must_answer,
            "policy_must_acknowledge": policy.must_acknowledge,
            "policy_must_repair": policy.must_repair,
            "policy_must_avoid": policy.must_avoid,
            "policy_fit_score": float(policy_scores["policy_fit_score"]),
            "policy_must_satisfied": bool(policy_scores["policy_must_satisfied"]),
            "policy_violation": bool(policy_scores["policy_violation"]),
            "policy_violation_reasons": list(policy_scores["policy_violation_reasons"]) if isinstance(policy_scores["policy_violation_reasons"], list) else [],
            "policy_conversation_progress_score": float(policy_scores["policy_conversation_progress_score"]),
            "training_style_score": float(training_style_scores["training_style_score"]),
            "training_style_penalty": float(training_style_scores["training_style_penalty"]),
            "training_style_reasons": list(training_style_scores["training_style_reasons"]) if isinstance(training_style_scores["training_style_reasons"], list) else [],
            "training_style_examples_loaded": int(training_style_scores["training_style_examples_loaded"]),
        }

    def regenerate_bundle(
        self,
        *,
        contact_name: str | None,
        incoming: str,
        context: list[str] | None = None,
        relationship_type: str | None = None,
        intent_type: str | None = None,
        avoid_candidates: list[str] | None = None,
        diversity_mode: str = "alternative_wording",
    ) -> DraftBundle:
        mode = diversity_mode if diversity_mode in DIVERSITY_MODES else "alternative_wording"
        cleaned_context = self._sanitize_recent_messages(context or [])
        latest = " ".join(incoming.split()).strip()
        recent_messages = [*cleaned_context[-5:], latest] if latest else cleaned_context[-6:]
        resolved_intent = normalize_intent_label(intent_type or classify_intent(latest, cleaned_context))
        resolved_relationship = normalize_relationship_type(
            relationship_type or self._relationship_type(contact_name, recent_messages)
        )
        state_key = contact_name or "__unknown__"
        stored_state = self.conversation_state.get_contact_state(state_key)
        avoid = list(dict.fromkeys([*(avoid_candidates or []), *stored_state.recent_candidates]))
        retrieved_started = time.perf_counter()
        retrieved_examples = self._retrieve_examples(
            recent_messages=recent_messages,
            full_conversation=recent_messages,
            relationship_type=resolved_relationship,
            incoming_intent=resolved_intent,
            contact_name=contact_name,
        )
        retrieval_ms = (time.perf_counter() - retrieved_started) * 1000
        sequences = self._regeneration_sequences(
            recent_messages,
            contact_name=contact_name,
            relationship_type=resolved_relationship,
            incoming_intent=resolved_intent,
            diversity_mode=mode,
            avoid_candidates=avoid,
        )
        candidates, recommended_index, blocked_reason = self._rank_candidates(
            recent_messages=recent_messages,
            full_conversation=recent_messages,
            sequences=sequences,
            contact_name=contact_name,
            relationship_type=resolved_relationship,
            incoming_intent=resolved_intent,
            retrieved_examples=retrieved_examples,
            generation_attempt=len(avoid) + 1,
            diversity_mode=mode,
            question_context=self._question_mode_context(
                recent_messages,
                contact_name=contact_name,
                relationship_type=resolved_relationship,
                incoming_intent=resolved_intent,
            ),
            hook_context=self._hook_context(
                recent_messages,
                contact_name=contact_name,
                relationship_type=resolved_relationship,
                incoming_intent=resolved_intent,
            ),
        )
        filtered: list[DraftCandidate] = []
        duplicate_filtered: list[DraftCandidate] = []
        for candidate in candidates:
            if (
                candidate.reply_energy_mode != "minimal"
                and (candidate.filler_penalty >= 0.85 or candidate.lazy_lol_penalty >= 0.8)
            ):
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "dead_filler_reply"
                candidate.risk_flags = list(dict.fromkeys([*candidate.risk_flags, "dead_filler_reply"]))
                continue
            if self._is_near_duplicate(candidate.text, avoid) or any(self._is_near_duplicate(part, avoid) for part in candidate.sequence):
                duplicate_filtered.append(candidate)
                continue
            filtered.append(candidate)
        if not filtered:
            blocked_reason = blocked_reason or "All regenerated candidates were rejected as duplicates, filler, or unsafe."
            if duplicate_filtered:
                for candidate in duplicate_filtered[:3]:
                    candidate.final_decision = "review"
                    candidate.auto_send_allowed = False
                    candidate.blocked_reason = candidate.blocked_reason or "duplicate_candidates_exhausted_review"
                    candidate.risk_flags = list(dict.fromkeys([*candidate.risk_flags, "duplicate_candidates_exhausted_review"]))
                filtered = duplicate_filtered[:3]
            else:
                filtered = self._non_duplicate_review_candidates(
                    recent_messages=recent_messages,
                    relationship_type=resolved_relationship,
                    incoming_intent=resolved_intent,
                    avoid_candidates=avoid,
                    contact_name=contact_name,
                    retrieved_examples=retrieved_examples,
                    generation_attempt=len(avoid) + 1,
                    diversity_mode=mode,
                )
        for candidate in filtered:
            candidate.diversity_mode = mode
            candidate.candidate_type = CANDIDATE_TYPE_BY_MODE[mode]
        recommended_index = 0 if filtered else None
        final_decision = filtered[0].final_decision if filtered else "review"
        self.conversation_state.record_candidates(state_key, [candidate.text for candidate in filtered])
        return DraftBundle(
            summary=self._summarize(recent_messages),
            reply_suggestions=[candidate.text for candidate in filtered],
            reply_sequences=[candidate.sequence for candidate in filtered],
            reply_candidates=filtered,
            recommended_reply_index=recommended_index,
            auto_send_blocked_reason=blocked_reason,
            relationship_type=resolved_relationship,
            intent_type=resolved_intent,
            retrieved_examples_count=len(retrieved_examples),
            retrieved_examples=[example.model_dump(mode="json") for example in retrieved_examples],
            final_decision=final_decision,
            blocked_reason=blocked_reason,
            generation_attempt=len(avoid) + 1,
            diversity_mode=mode,
            timings_ms={"retrieval": round(retrieval_ms, 2), "generation": 0.0, "critic": 0.0},
            prompt_preview=getattr(self, "_last_prompt_preview", None),
            **_retrieval_debug_fields(retrieved_examples),
        )

    def _non_duplicate_review_candidates(
        self,
        *,
        recent_messages: list[str],
        relationship_type: str,
        incoming_intent: str,
        avoid_candidates: list[str],
        contact_name: str | None,
        retrieved_examples: list[RetrievedExample],
        generation_attempt: int,
        diversity_mode: str,
    ) -> list[DraftCandidate]:
        pool = []
        energy_context = self._reply_energy_context(
            recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        energy_mode = str(energy_context["reply_energy_mode"])
        if energy_mode in {"defensive_playful", "corrective", "engaged_explanation", "expressive", "minimal"}:
            pool = self._expressive_reply_pool(
                energy_mode,
                recent_messages=recent_messages,
                relationship_type=relationship_type,
                contact_name=contact_name,
                incoming_intent=incoming_intent,
            )
        elif incoming_intent == "greeting":
            pool = ["yo what u saying", "heyy what u doing", "what u saying"]
        elif normalize_intent_label(incoming_intent) == "planning":
            pool = ["what time", "depends what time", "where u lot going"]
        elif relationship_type in {"professional", "university"}:
            pool = ["i'll check and confirm", "i can send it shortly", "that works for me"]
        else:
            pool = ["yeah maybe", "not sure yet", "what do you mean by that"]
        sequences = [
            [reply]
            for reply in pool
            if reply
            and not self._suspicious_reply_blocked(reply, relationship_type=relationship_type, contact_name=contact_name)
            and not self._is_near_duplicate(reply, avoid_candidates)
        ][:3]
        if not sequences:
            return []
        fallback_candidates, _, _ = self._rank_candidates(
            recent_messages=recent_messages,
            full_conversation=recent_messages,
            sequences=sequences,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
            retrieved_examples=retrieved_examples,
            generation_attempt=generation_attempt,
            diversity_mode=diversity_mode,
        )
        for candidate in fallback_candidates:
            candidate.final_decision = "review" if candidate.final_decision == "send" else candidate.final_decision
            candidate.auto_send_allowed = False
            candidate.blocked_reason = candidate.blocked_reason or "duplicate_safe_fallback_review"
        return fallback_candidates

    def build_bundle(self, recent_messages: list[str], contact_name: str | None = None) -> DraftBundle:
        cleaned_recent_messages = self._sanitize_recent_messages(recent_messages)
        incoming_text = cleaned_recent_messages[-1] if cleaned_recent_messages else ""
        incoming_intent = classify_intent(incoming_text, cleaned_recent_messages[:-1])
        relationship_type = self._relationship_type(contact_name, cleaned_recent_messages)
        timing_started = time.perf_counter()
        retrieved_examples = self._retrieve_examples(
            recent_messages=cleaned_recent_messages,
            full_conversation=cleaned_recent_messages,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        retrieval_ms = (time.perf_counter() - timing_started) * 1000
        generation_started = time.perf_counter()
        ai_bundle = self._draft_with_ai(
            cleaned_recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
            retrieved_examples=retrieved_examples,
        )
        generation_ms = (time.perf_counter() - generation_started) * 1000
        legacy_ai_only = ai_bundle is not None and ai_bundle.provider_metadata.get("provider") == "ollama" and self._use_legacy_ollama_generation()
        summary = ai_bundle.summary if ai_bundle is not None else self._summarize(cleaned_recent_messages)
        fallback_sequences = self._contract_fallback_sequences(
            cleaned_recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        candidate_sequences = ai_bundle.reply_sequences if legacy_ai_only else self._merge_candidate_sequences(
            ai_bundle.reply_sequences if ai_bundle is not None else [],
            self._policy_sequences(cleaned_recent_messages, contact_name=contact_name, relationship_type=relationship_type, incoming_intent=incoming_intent),
            self._agenda_sequences(cleaned_recent_messages, contact_name=contact_name, relationship_type=relationship_type, incoming_intent=incoming_intent),
            self._scene_sequences(cleaned_recent_messages, contact_name=contact_name, relationship_type=relationship_type, incoming_intent=incoming_intent),
            fallback_sequences,
        )
        provider_metadata = ai_bundle.provider_metadata if ai_bundle is not None else {}
        if not provider_metadata and fallback_sequences:
            provider_metadata = {
                "generation_status": "contract_fallback_without_provider",
                "fallback_source": "conversation_contract",
                "generated_by_provider": False,
            }
        critic_started = time.perf_counter()
        question_context = self._question_mode_context(
            cleaned_recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        hook_context = self._hook_context(
            cleaned_recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        candidates, recommended_index, blocked_reason = self._rank_candidates(
            recent_messages=cleaned_recent_messages,
            full_conversation=cleaned_recent_messages,
            sequences=candidate_sequences,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
            retrieved_examples=retrieved_examples,
            provider_metadata=provider_metadata,
            question_context=question_context,
            hook_context=hook_context,
        )
        if ai_bundle is not None and ai_bundle.provider_metadata.get("provider") == "ollama" and self._use_legacy_ollama_generation():
            order = {self._format_sequence_for_display(sequence): index for index, sequence in enumerate(ai_bundle.reply_sequences)}
            candidates.sort(key=lambda candidate: order.get(candidate.text, len(order)))
            recommended_index = 0 if candidates else None
        critic_ms = (time.perf_counter() - critic_started) * 1000
        final_decision = candidates[recommended_index or 0].final_decision if candidates else "review"
        timings_ms = {
            "training_data_load": 0.0,
            "retrieval": round(retrieval_ms, 2),
            "generation": round(generation_ms, 2),
            "critic": round(critic_ms, 2),
        }
        return DraftBundle(
            summary=summary,
            reply_suggestions=[candidate.text for candidate in candidates],
            reply_sequences=[candidate.sequence for candidate in candidates],
            reply_candidates=candidates,
            recommended_reply_index=recommended_index,
            auto_send_blocked_reason=blocked_reason,
            relationship_type=relationship_type,
            intent_type=incoming_intent,
            retrieved_examples_count=len(retrieved_examples),
            retrieved_examples=[example.model_dump(mode="json") for example in retrieved_examples],
            final_decision=final_decision,
            blocked_reason=blocked_reason,
            timings_ms=timings_ms,
            prompt_preview=getattr(self, "_last_prompt_preview", None),
            photo_suggestions=self._suggest_photos(recent_messages),
            provider_metadata=provider_metadata,
            **_retrieval_debug_fields(retrieved_examples),
        )

    def build_bundle_with_context(
        self,
        recent_messages: list[str],
        full_conversation: list[str],
        contact_name: str | None = None,
        relationship_type: str | None = None,
    ) -> DraftBundle:
        """Build bundle using full conversation context with speaker labels for better AI understanding."""
        cleaned_conversation = self._sanitize_full_conversation(full_conversation)
        cleaned_recent_messages = self._recent_messages_from_conversation(cleaned_conversation) or self._sanitize_recent_messages(recent_messages)
        incoming_text = cleaned_recent_messages[-1] if cleaned_recent_messages else ""
        incoming_intent = classify_intent(incoming_text, cleaned_recent_messages[:-1])
        relationship_type = normalize_relationship_type(relationship_type or self._relationship_type(contact_name, cleaned_conversation))
        timing_started = time.perf_counter()
        retrieved_examples = self._retrieve_examples(
            recent_messages=cleaned_recent_messages,
            full_conversation=cleaned_conversation,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        retrieval_ms = (time.perf_counter() - timing_started) * 1000
        generation_started = time.perf_counter()
        ai_bundle = self._draft_with_ai_full_context(
            cleaned_conversation,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
            retrieved_examples=retrieved_examples,
        )
        generation_ms = (time.perf_counter() - generation_started) * 1000
        legacy_ai_only = ai_bundle is not None and ai_bundle.provider_metadata.get("provider") == "ollama" and self._use_legacy_ollama_generation()
        summary = ai_bundle.summary if ai_bundle is not None else self._summarize(cleaned_recent_messages)
        fallback_sequences = self._contract_fallback_sequences(
            cleaned_conversation,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        candidate_sequences = ai_bundle.reply_sequences if legacy_ai_only else self._merge_candidate_sequences(
            ai_bundle.reply_sequences if ai_bundle is not None else [],
            self._policy_sequences(cleaned_conversation, contact_name=contact_name, relationship_type=relationship_type, incoming_intent=incoming_intent),
            self._agenda_sequences(cleaned_conversation, contact_name=contact_name, relationship_type=relationship_type, incoming_intent=incoming_intent),
            self._scene_sequences(cleaned_conversation, contact_name=contact_name, relationship_type=relationship_type, incoming_intent=incoming_intent),
            fallback_sequences,
        )
        provider_metadata = ai_bundle.provider_metadata if ai_bundle is not None else {}
        if not provider_metadata and fallback_sequences:
            provider_metadata = {
                "generation_status": "contract_fallback_without_provider",
                "fallback_source": "conversation_contract",
                "generated_by_provider": False,
            }
        critic_started = time.perf_counter()
        question_context = self._question_mode_context(
            cleaned_recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        hook_context = self._hook_context(
            cleaned_recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        candidates, recommended_index, blocked_reason = self._rank_candidates(
            recent_messages=cleaned_recent_messages,
            full_conversation=cleaned_conversation,
            sequences=candidate_sequences,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
            retrieved_examples=retrieved_examples,
            provider_metadata=provider_metadata,
            question_context=question_context,
            hook_context=hook_context,
        )
        critic_ms = (time.perf_counter() - critic_started) * 1000
        final_decision = candidates[recommended_index or 0].final_decision if candidates else "review"
        timings_ms = {
            "training_data_load": 0.0,
            "retrieval": round(retrieval_ms, 2),
            "generation": round(generation_ms, 2),
            "critic": round(critic_ms, 2),
        }
        return DraftBundle(
            summary=summary,
            reply_suggestions=[candidate.text for candidate in candidates],
            reply_sequences=[candidate.sequence for candidate in candidates],
            reply_candidates=candidates,
            recommended_reply_index=recommended_index,
            auto_send_blocked_reason=blocked_reason,
            relationship_type=relationship_type,
            intent_type=incoming_intent,
            retrieved_examples_count=len(retrieved_examples),
            retrieved_examples=[example.model_dump(mode="json") for example in retrieved_examples],
            final_decision=final_decision,
            blocked_reason=blocked_reason,
            timings_ms=timings_ms,
            prompt_preview=getattr(self, "_last_prompt_preview", None),
            photo_suggestions=self._suggest_photos(recent_messages),
            provider_metadata=provider_metadata,
            **_retrieval_debug_fields(retrieved_examples),
        )

    def compare_providers(
        self,
        *,
        incoming: str,
        context: list[str],
        relationship_type: str,
        intent_type: str,
        providers: list[str],
    ) -> list[dict[str, object]]:
        cleaned_context = self._sanitize_recent_messages(context)[-max(0, int(self.external_api_max_context_messages)) :]
        latest = " ".join(incoming.split()).strip()
        recent_messages = [*cleaned_context, latest] if latest else cleaned_context
        resolved_intent = normalize_intent_label(intent_type if intent_type != "auto" else classify_intent(latest, cleaned_context))
        resolved_relationship = normalize_relationship_type(relationship_type)
        retrieved_examples = self._retrieve_examples(
            recent_messages=recent_messages,
            full_conversation=recent_messages,
            relationship_type=resolved_relationship,
            incoming_intent=resolved_intent,
        )
        prompt_messages = self._provider_prompt_messages(
            recent_messages,
            contact_name=None,
            relationship_type=resolved_relationship,
            incoming_intent=resolved_intent,
            retrieved_examples=retrieved_examples,
            full_context=False,
            prompt_mode=self._provider_prompt_mode(
                recent_messages,
                contact_name=None,
                relationship_type=resolved_relationship,
                incoming_intent=resolved_intent,
            ),
        )
        results: list[dict[str, object]] = []
        for provider_name in providers[:8]:
            result = self._compare_single_provider(
                provider_name=str(provider_name),
                prompt_messages=prompt_messages,
                context_texts=recent_messages,
                recent_messages=recent_messages,
                relationship_type=resolved_relationship,
                incoming_intent=resolved_intent,
                retrieved_examples=retrieved_examples,
            )
            results.append(result)
        return results

    def _load_templates(self) -> dict[str, object]:
        if self.template_path.exists():
            return json.loads(self.template_path.read_text(encoding="utf-8"))
        return {
            "reply_prefixes": ["Sounds good", "That works for me", "I can do that"],
            "closers": ["I'll keep you posted.", "Let me know if timing changes.", "Happy to help."],
        }

    def _load_photo_index(self) -> list[ApprovedPhoto]:
        index_path = self.approved_photos_dir / "index.json"
        if not index_path.exists():
            return []
        raw_items = json.loads(index_path.read_text(encoding="utf-8"))
        return [ApprovedPhoto(**item) for item in raw_items]

    def _load_negative_patterns(self, path: Path) -> dict[str, object]:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _summarize(self, recent_messages: list[str]) -> str:
        if not recent_messages:
            return "No recent visible messages were extracted from the current screen."
        last_message = recent_messages[-1]
        return f"Visible conversation is active. Latest incoming message: {last_message}"

    def _merge_candidate_sequences(self, *groups: list[list[str]]) -> list[list[str]]:
        merged: list[list[str]] = []
        seen: set[str] = set()
        for group in groups:
            for sequence in group:
                text = self._format_sequence_for_display(sequence)
                normalized = normalize_text(text)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                merged.append(sequence)
        return merged[:8]

    def _contract_fallback_sequences(
        self,
        conversation_messages: list[str],
        *,
        contact_name: str | None,
        relationship_type: str,
        incoming_intent: str,
    ) -> list[list[str]]:
        return [
            [reply]
            for reply in self._draft_replies(
                conversation_messages,
                contact_name=contact_name,
                relationship_type=relationship_type,
                incoming_intent=incoming_intent,
            )
        ]

    def _scene_sequences(
        self,
        conversation_messages: list[str],
        *,
        contact_name: str | None,
        relationship_type: str,
        incoming_intent: str,
    ) -> list[list[str]]:
        scene = self._conversation_scene(
            conversation_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        return [[reply] for reply in self._scene_reply_pool(scene)]

    def _agenda_sequences(
        self,
        conversation_messages: list[str],
        *,
        contact_name: str | None,
        relationship_type: str,
        incoming_intent: str,
    ) -> list[list[str]]:
        scene = self._conversation_scene(
            conversation_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        question_debt = self._question_debt(conversation_messages)
        agenda = self._conversation_agenda(conversation_messages, scene=scene, question_debt=question_debt)
        return [[reply] for reply in self._agenda_reply_pool(agenda)]

    def _policy_sequences(
        self,
        conversation_messages: list[str],
        *,
        contact_name: str | None,
        relationship_type: str,
        incoming_intent: str,
    ) -> list[list[str]]:
        scene = self._conversation_scene(
            conversation_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        question_debt = self._question_debt(conversation_messages)
        agenda = self._conversation_agenda(conversation_messages, scene=scene, question_debt=question_debt)
        policy = self._conversation_policy(conversation_messages, scene=scene, agenda=agenda, relationship_type=relationship_type, question_debt=question_debt)
        policy_sequences = [[reply] for reply in self._policy_reply_pool(policy)]
        if relationship_type == "romantic_interest" and normalize_intent_label(incoming_intent) in {"planning", "availability"}:
            return self._merge_candidate_sequences(
                self._romantic_planning_sequences(conversation_messages),
                policy_sequences,
            )
        return policy_sequences

    def _draft_replies(
        self,
        recent_messages: list[str],
        *,
        contact_name: str | None = None,
        relationship_type: str = "unknown",
        incoming_intent: str = "other",
    ) -> list[str]:
        if not recent_messages:
            return [
                "I can reply once the thread is clearly visible.",
                "Move to the conversation view and I will draft a response.",
                "I need clearer conversation context before typing anything.",
            ]
        if normalize_intent_label(incoming_intent) == "exam_logistics":
            return self._exam_logistics_reply_pool(recent_messages, relationship_type=relationship_type)
        question_context = self._question_mode_context(
            recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        hook_context = self._hook_context(
            recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        energy_context = self._reply_energy_context(
            recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        hook_mode = str(hook_context["hook_mode"])
        question_mode = str(question_context["question_mode"])
        reply_energy_mode = str(energy_context["reply_energy_mode"])
        scene = self._conversation_scene(
            recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        question_debt = self._question_debt(recent_messages)
        debt_pool = self._question_debt_reply_pool(question_debt)
        if debt_pool and (question_debt.user_called_out_unanswered_question or scene.scene_type not in {"reciprocal_current_activity_question", "reciprocal_wellbeing_question"}):
            return debt_pool
        agenda = self._conversation_agenda(recent_messages, scene=scene, question_debt=question_debt)
        policy_pool = self._policy_reply_pool(self._conversation_policy(recent_messages, scene=scene, agenda=agenda, relationship_type=relationship_type, question_debt=question_debt))
        if policy_pool:
            return policy_pool
        agenda_pool = self._agenda_reply_pool(agenda)
        if agenda_pool:
            return agenda_pool
        function_pool = self._conversation_function_reply_pool(
            recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        if function_pool:
            return function_pool
        emotional_pool = self._emotional_reply_pool(
            recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        if emotional_pool:
            return emotional_pool
        context_pool = self._context_grounded_reply_pool(
            recent_messages,
            contact_name=contact_name,
        )
        if context_pool:
            return context_pool
        category_pool = self._context_fallback_replies(self._input_category(recent_messages[-1]))
        if category_pool:
            return self._unique_replies(category_pool)
        if bool(hook_context.get("continuation_required")):
            return self._continuation_reply_pool(recent_messages[-1])
        repair_pool = self._repair_or_shh_reply_pool(
            recent_messages,
            relationship_type=relationship_type,
            contact_name=contact_name,
        )
        if repair_pool:
            return repair_pool
        if bool(question_context["has_simple_ack"]):
            return self._expressive_reply_pool(
                "minimal",
                recent_messages=recent_messages,
                relationship_type=relationship_type,
                contact_name=contact_name,
                incoming_intent=incoming_intent,
            )
        if (
            question_mode == "none"
            and int(question_context["recent_bot_questions_count"]) >= 2
            and not bool(question_context["has_bored_signal"])
            and not bool(question_context["has_dead_signal"])
        ):
            return self._unique_replies(["calm", "ok", "true"])
        if bool(hook_context["hook_required"]):
            return self._hook_reply_pool(
                hook_mode,
                recent_messages=recent_messages,
                relationship_type=relationship_type,
                contact_name=contact_name,
            )
        if question_mode != "none":
            return self._question_reply_pool(
                question_mode,
                recent_messages=recent_messages,
                relationship_type=relationship_type,
                incoming_intent=incoming_intent,
                question_reason=str(question_context["question_reason"]),
            )
        if reply_energy_mode in {"defensive_playful", "corrective", "engaged_explanation"}:
            return self._expressive_reply_pool(
                reply_energy_mode,
                recent_messages=recent_messages,
                relationship_type=relationship_type,
                contact_name=contact_name,
                incoming_intent=incoming_intent,
            )
        if reply_energy_mode == "expressive" and (
            incoming_intent == "greeting" or is_simple_greeting(recent_messages[-1], recent_messages[:-1])
        ):
            return self._expressive_reply_pool(
                reply_energy_mode,
                recent_messages=recent_messages,
                relationship_type=relationship_type,
                contact_name=contact_name,
                incoming_intent=incoming_intent,
            )
        if incoming_intent == "greeting" or is_simple_greeting(recent_messages[-1], recent_messages[:-1]):
            return self._greeting_replies(relationship_type=relationship_type)
        if normalize_intent_label(incoming_intent) == "banter_challenge":
            return self._banter_challenge_replies(
                recent_messages,
                relationship_type=relationship_type,
                contact_name=contact_name,
                incoming_intent=incoming_intent,
            )
        return self._dynamic_fallback_replies(
            recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )

    def _draft_with_ai(
        self,
        recent_messages: list[str],
        *,
        contact_name: str | None = None,
        relationship_type: str = "unknown",
        incoming_intent: str = "other",
        retrieved_examples: list[RetrievedExample] | None = None,
    ) -> DraftBundle | None:
        if not self.ai_reply_enabled or not recent_messages:
            return None
        if (
            incoming_intent == "greeting"
            and len(recent_messages) <= 2
            and relationship_type in {"unknown", "professional", "family", "university"}
            and self._use_legacy_ollama_generation()
        ):
            return None
        if self._use_legacy_ollama_generation():
            try:
                legacy_bundle = self._draft_with_legacy_ollama(
                    recent_messages,
                    contact_name=contact_name,
                    relationship_type=relationship_type,
                    incoming_intent=incoming_intent,
                    retrieved_examples=retrieved_examples or [],
                    full_context=False,
                )
            except TypeError:
                legacy_bundle = None
            if legacy_bundle is not None:
                return legacy_bundle
            return None
        return self._draft_with_model_router(
            recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
            retrieved_examples=retrieved_examples or [],
        )

    def _draft_with_ai_full_context(
        self,
        full_conversation: list[str],
        *,
        contact_name: str | None = None,
        relationship_type: str = "unknown",
        incoming_intent: str = "other",
        retrieved_examples: list[RetrievedExample] | None = None,
    ) -> DraftBundle | None:
        """Generate drafts using full conversation history with speaker labels."""
        if not self.ai_reply_enabled or not full_conversation:
            return None
        if incoming_intent == "greeting":
            other_count = sum(1 for line in full_conversation if line.startswith("[OTHER]:"))
            hook_context = self._hook_context(
                full_conversation,
                contact_name=contact_name,
                relationship_type=relationship_type,
                incoming_intent=incoming_intent,
            )
            if other_count <= 2 and not bool(hook_context["hook_required"]):
                return None
        if self._use_legacy_ollama_generation():
            return self._draft_with_legacy_ollama(
                full_conversation,
                contact_name=contact_name,
                relationship_type=relationship_type,
                incoming_intent=incoming_intent,
                retrieved_examples=retrieved_examples or [],
                full_context=True,
            )
        return self._draft_with_model_router(
            full_conversation,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
            retrieved_examples=retrieved_examples or [],
            full_context=True,
        )

    def _use_legacy_ollama_generation(self) -> bool:
        return (
            self.ai_reply_backend == "ollama"
            and self.draft_provider == "ollama"
            and self.fast_provider == "ollama"
            and self.router_provider == "ollama"
            and self.private_provider == "ollama"
        )

    def _draft_with_legacy_ollama(
        self,
        messages: list[str],
        *,
        contact_name: str | None,
        relationship_type: str,
        incoming_intent: str,
        retrieved_examples: list[RetrievedExample],
        full_context: bool,
    ) -> DraftBundle | None:
        try:
            if full_context:
                payload = self._ollama_chat_full_context(
                    messages,
                    contact_name=contact_name,
                    relationship_type=relationship_type,
                    incoming_intent=incoming_intent,
                    retrieved_examples=retrieved_examples,
                )
            else:
                payload = self._ollama_chat(
                    messages,
                    contact_name=contact_name,
                    relationship_type=relationship_type,
                    incoming_intent=incoming_intent,
                    retrieved_examples=retrieved_examples,
                )
        except TypeError:
            payload = (
                self._ollama_chat_full_context(messages, contact_name=contact_name)
                if full_context
                else self._ollama_chat(messages, contact_name=contact_name)
            )
        if payload is None:
            return None
        summary = str(payload.get("summary", "")).strip()
        sequences = self._parse_reply_sequences(payload.get("replies", []))
        if len(sequences) != 3:
            return None
        replies = [self._format_sequence_for_display(sequence) for sequence in sequences]
        if not self._replies_are_usable(replies, sequences):
            return None
        return DraftBundle(
            summary=summary or ("Conversation context loaded" if full_context else self._summarize(messages)),
            reply_suggestions=replies,
            reply_sequences=sequences,
            provider_metadata={
                "provider": "ollama",
                "model": str(payload.get("_model") or self.ai_reply_model),
                "latency_ms": None,
                "external_api_used": False,
                "external_api_blocked": False,
                "provider_configured": True,
            },
        )

    def _draft_with_model_router(
        self,
        conversation_messages: list[str],
        *,
        contact_name: str | None,
        relationship_type: str,
        incoming_intent: str,
        retrieved_examples: list[RetrievedExample],
        full_context: bool = False,
        provider_names: list[str] | None = None,
    ) -> DraftBundle | None:
        prompt_mode = self._provider_prompt_mode(
            conversation_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        if prompt_mode == "normal" and not full_context:
            return None
        messages = self._provider_prompt_messages(
            conversation_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
            retrieved_examples=retrieved_examples,
            full_context=full_context,
            prompt_mode=prompt_mode,
        )
        context_texts = conversation_messages[-max(1, int(self.external_api_max_context_messages)) :]
        response, metadata = self._run_provider_generation(
            messages=messages,
            provider_names=provider_names,
            context_texts=context_texts,
        )
        payload = self._parse_model_json(response.text)
        sequences = self._parse_provider_candidate_sequences(payload)
        if not sequences:
            sequences = self._fallback_sequences_from_model_text(response.text)
        if not sequences:
            hook_context = self._hook_context(
                conversation_messages,
                contact_name=contact_name,
                relationship_type=relationship_type,
                incoming_intent=incoming_intent,
            )
            if bool(hook_context["hook_required"]):
                sequences = [
                    [reply]
                    for reply in self._hook_reply_pool(
                        str(hook_context["hook_mode"]),
                        recent_messages=conversation_messages,
                        relationship_type=relationship_type,
                        contact_name=contact_name,
                    )
                ]
                metadata["fallback_used"] = True
                metadata["invalid_provider_output"] = bool(response.text.strip())
            else:
                return None
        sequences = sequences[:5]
        replies = [self._format_sequence_for_display(sequence) for sequence in sequences]
        if not self._replies_are_usable(replies[:3], sequences[:3]):
            usable = [(reply, sequence) for reply, sequence in zip(replies, sequences) if self._replies_are_usable([reply], [sequence])]
            replies = [item[0] for item in usable]
            sequences = [item[1] for item in usable]
        if not sequences:
            hook_context = self._hook_context(
                conversation_messages,
                contact_name=contact_name,
                relationship_type=relationship_type,
                incoming_intent=incoming_intent,
            )
            if bool(hook_context["hook_required"]):
                sequences = [
                    [reply]
                    for reply in self._hook_reply_pool(
                        str(hook_context["hook_mode"]),
                        recent_messages=conversation_messages,
                        relationship_type=relationship_type,
                        contact_name=contact_name,
                    )
                ]
                replies = [self._format_sequence_for_display(sequence) for sequence in sequences]
                metadata["fallback_used"] = True
                metadata["invalid_provider_output"] = True
            else:
                return None
        provider_fallback_required = response.provider == "fallback" or bool(metadata.get("manual_review_fallback"))
        provider_metadata = {
            "provider": response.provider,
            "model": response.model,
            "latency_ms": response.latency_ms,
            "raw_finish_reason": response.raw_finish_reason,
            "error": response.error,
            "external_api_used": response.external_api_used,
            "external_api_blocked": bool(metadata.get("external_api_blocked")),
            "blocked_reason": metadata.get("blocked_reason"),
            "provider_configured": bool(metadata.get("provider_configured")),
            "fallback_errors": metadata.get("fallback_errors", []),
            "manual_review_fallback": bool(metadata.get("manual_review_fallback")),
            "fallback_used": bool(metadata.get("fallback_used")),
            "invalid_provider_output": bool(metadata.get("invalid_provider_output")),
            "generated_by_provider": response.provider not in {"fallback"},
            "provider_name": response.provider,
            "provider_model": response.model,
            "provider_error": response.error,
            "provider_latency_ms": response.latency_ms,
            "provider_assisted_continuation": prompt_mode in {"continuation", "hook", "expressive", "repair"},
            "provider_prompt_mode": prompt_mode,
            "generation_status": "provider_generated" if response.provider != "fallback" else "provider_unavailable_contract_fallback",
            "fallback_source": "none",
        }
        if provider_fallback_required or not sequences or bool(provider_metadata["invalid_provider_output"]):
            fallback_sequences = self._fallback_sequences_for_context(
                conversation_messages,
                contact_name=contact_name,
                relationship_type=relationship_type,
                incoming_intent=incoming_intent,
            )
            if fallback_sequences:
                sequences = fallback_sequences
                replies = [self._format_sequence_for_display(sequence) for sequence in sequences]
                provider_metadata["fallback_used"] = True
                provider_metadata["invalid_provider_output"] = True
                provider_metadata["generated_by_provider"] = False
                provider_metadata["generation_status"] = "contract_fallback_after_provider_failure"
                provider_metadata["fallback_source"] = "conversation_contract"
        summary = str(payload.get("summary", "")).strip() if isinstance(payload, dict) else ""
        return DraftBundle(
            summary=summary or ("Conversation context loaded" if full_context else self._summarize(conversation_messages)),
            reply_suggestions=replies,
            reply_sequences=sequences,
            provider_metadata=provider_metadata,
        )

    def _fallback_sequences_for_context(
        self,
        recent_messages: list[str],
        *,
        contact_name: str | None,
        relationship_type: str,
        incoming_intent: str,
    ) -> list[list[str]]:
        scene = self._conversation_scene(
            recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        question_debt = self._question_debt(recent_messages)
        debt_pool = self._question_debt_reply_pool(question_debt)
        if debt_pool and (question_debt.user_called_out_unanswered_question or scene.scene_type not in {"reciprocal_current_activity_question", "reciprocal_wellbeing_question"}):
            return [[reply] for reply in debt_pool]
        agenda = self._conversation_agenda(recent_messages, scene=scene, question_debt=question_debt)
        policy_pool = self._policy_reply_pool(self._conversation_policy(recent_messages, scene=scene, agenda=agenda, relationship_type=relationship_type, question_debt=question_debt))
        if policy_pool:
            return [[reply] for reply in policy_pool]
        scene_pool = self._scene_reply_pool(scene)
        if scene_pool:
            return [[reply] for reply in scene_pool]
        agenda_pool = self._agenda_reply_pool(agenda)
        if agenda_pool:
            return [[reply] for reply in agenda_pool]
        function_pool = self._conversation_function_reply_pool(
            recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        if function_pool:
            return [[reply] for reply in function_pool]
        emotional_pool = self._emotional_reply_pool(
            recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        if emotional_pool:
            return [[reply] for reply in emotional_pool]
        repair_pool = self._repair_or_shh_reply_pool(
            recent_messages,
            relationship_type=relationship_type,
            contact_name=contact_name,
        )
        if repair_pool:
            return [[reply] for reply in repair_pool]
        hook_context = self._hook_context(
            recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        if bool(hook_context.get("continuation_required")):
            return [[reply] for reply in self._continuation_reply_pool(recent_messages[-1])]
        context_pool = self._context_grounded_reply_pool(
            recent_messages,
            contact_name=contact_name,
        )
        if context_pool:
            return [[reply] for reply in context_pool]
        category_pool = self._context_fallback_replies(self._input_category(recent_messages[-1] if recent_messages else ""))
        if category_pool:
            return [[reply] for reply in category_pool]
        if bool(hook_context.get("hook_required")):
            return [
                [reply]
                for reply in self._hook_reply_pool(
                    str(hook_context["hook_mode"]),
                    recent_messages=recent_messages,
                    relationship_type=relationship_type,
                    contact_name=contact_name,
                )
            ]
        return []

    def _provider_prompt_mode(
        self,
        conversation_messages: list[str],
        *,
        contact_name: str | None,
        relationship_type: str,
        incoming_intent: str,
    ) -> str:
        privacy_texts = conversation_messages[-max(1, int(self.external_api_max_context_messages)) :]
        privacy = check_external_api_allowed(privacy_texts, allow_sensitive=self.external_api_allow_sensitive)
        if privacy.external_api_blocked:
            return "continuation"
        function_context = self._conversation_function_context(
            conversation_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        scene = self._conversation_scene(
            conversation_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        question_debt = self._question_debt(conversation_messages)
        agenda = self._conversation_agenda(conversation_messages, scene=scene, question_debt=question_debt)
        if question_debt.answer_required_now:
            return "repair" if question_debt.user_called_out_unanswered_question else "continuation"
        if agenda.agenda_state == "anti_loop_repair":
            return "repair"
        if agenda.agenda_state in {"topic_selection_needed", "dead_conversation_recovery", "user_bored_or_unengaged"}:
            return "continuation"
        if scene.scene_type not in {"normal"}:
            if scene.explanation_required:
                return "repair" if scene.previous_bot_claim_type in {"repair", "weird_story_reaction"} else "continuation"
            if scene.repair_required or scene.user_called_out_bot:
                return "repair"
            if scene.emotional_response_required:
                return "expressive"
            if scene.scene_type == "light_acknowledgement":
                return "expressive"
            if scene.topic_engagement_required or scene.direct_answer_required or scene.hook_required:
                return "continuation"
        if str(function_context["conversation_function"]) in {"repeated_reply_callout", "mocking_after_repair", "context_failure_question"}:
            return "repair"
        if str(function_context["conversation_function"]) == "recent_life_update_question":
            return "continuation"
        fact_context = self._recent_fact_context(conversation_messages, contact_name=contact_name)
        if bool(fact_context.get("recently_answered_question_detected")):
            return "continuation"
        meaning_context = self._latest_meaning_context(
            conversation_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        if str(meaning_context["latest_message_meaning"]) in {"hurt_feelings", "serious_boundary", "confusion", "repair_callout"}:
            return "repair"
        if str(meaning_context["latest_message_meaning"]) in {"affection", "flirty_affection"}:
            return "expressive"
        hook_context = self._hook_context(
            conversation_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        energy_context = self._reply_energy_context(
            conversation_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        if bool(hook_context.get("repair_required")) or bool(energy_context.get("repair_required")):
            return "repair"
        if bool(hook_context.get("continuation_required")) or bool(hook_context.get("low_information_incoming")):
            return "continuation"
        if bool(hook_context.get("hook_required")):
            return "hook"
        if str(energy_context.get("reply_energy_mode")) in {"expressive", "defensive_playful", "corrective", "engaged_explanation"} or bool(energy_context.get("criticism_detected")):
            return "expressive"
        latest_message = conversation_messages[-1] if conversation_messages else ""
        if incoming_intent == "planning" or is_question_like_text(latest_message):
            return "hook"
        return "normal"

    def _provider_prompt_messages(
        self,
        conversation_messages: list[str],
        *,
        contact_name: str | None,
        relationship_type: str,
        incoming_intent: str,
        retrieved_examples: list[RetrievedExample],
        full_context: bool,
        prompt_mode: str = "normal",
    ) -> list[ModelMessage]:
        window = conversation_messages[-12:] if full_context else conversation_messages[-8:]
        conversation = "\n".join(window)
        retrieved_fragment = self._retrieved_examples_fragment(retrieved_examples) if incoming_intent != "greeting" else ""
        style_profile_fragment = self._style_profile_fragment(
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        memory_fragment = self._conversation_memory_fragment(contact_name)
        question_fragment = self._question_prompt_fragment(
            conversation_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        hook_fragment = self._hook_prompt_fragment(
            conversation_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        energy_fragment = self._reply_energy_prompt_fragment(
            conversation_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        state = self._conversation_state_for(contact_name)
        hook_context = self._hook_context(
            conversation_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        question_context = self._question_mode_context(
            conversation_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        energy_context = self._reply_energy_context(
            conversation_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        fact_context = self._recent_fact_context(conversation_messages, contact_name=contact_name)
        meaning_context = self._latest_meaning_context(
            conversation_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        function_context = self._conversation_function_context(
            conversation_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        scene = self._conversation_scene(
            conversation_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        question_debt = self._question_debt(conversation_messages)
        agenda = self._conversation_agenda(conversation_messages, scene=scene, question_debt=question_debt)
        policy = self._conversation_policy(
            conversation_messages,
            scene=scene,
            agenda=agenda,
            relationship_type=relationship_type,
            question_debt=question_debt,
        )
        scene_json = json.dumps(scene.to_dict(), ensure_ascii=False)
        agenda_json = json.dumps(agenda.to_dict(), ensure_ascii=False)
        policy_json = json.dumps(policy.to_dict(), ensure_ascii=False)
        question_debt_json = json.dumps(question_debt.to_dict(), ensure_ascii=False)
        relevant_identity = (
            self.identity_pack.relevant_facts(
                scene.identity_question_kind,
                relationship_type,
                message_count=scene.thread_message_count,
            )
            if scene.identity_answer_required
            else {}
        )
        identity_fragment = (
            f"safe_identity_summary={self.identity_pack.safe_identity_summary()}\n"
            f"relevant_identity_facts={json.dumps(relevant_identity, ensure_ascii=False)}\n"
            f"identity_pack_loaded={self.identity_pack.loaded}\n"
            f"identity_disclosure_allowed={scene.identity_disclosure_allowed}\n"
            f"identity_thread_message_count={scene.thread_message_count}\n"
            f"banned_disclosures={', '.join(self.identity_pack.banned_disclosures)}"
        )
        thread_context = self._thread_memory_context(
            conversation_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        thread_memories = thread_context.get("thread_memories", [])
        memory_lines: list[str] = []
        if isinstance(thread_memories, list):
            for item in thread_memories[:3]:
                if not isinstance(item, dict):
                    continue
                memory_lines.append(
                    " | ".join(
                        [
                            f"summary={item.get('summary', '')}",
                            f"worked={', '.join(item.get('successful_reply_patterns', []) if isinstance(item.get('successful_reply_patterns'), list) else [])}",
                            f"failed={', '.join(item.get('failed_reply_patterns', []) if isinstance(item.get('failed_reply_patterns'), list) else [])}",
                            f"avoid={', '.join(item.get('bot_mistakes', []) if isinstance(item.get('bot_mistakes'), list) else [])}",
                        ]
                    )
                )
        thread_memory_fragment = "\n".join(memory_lines) if memory_lines else "none"
        input_category = self._input_category(conversation_messages[-1] if conversation_messages else "")
        category_banned = {
            "greeting": "ok, true, calm, fair, yup, yeah, same, cool, lol, idk",
            "repair": "true, ok, calm, fair, yeah, lol, same, idk, single-word acknowledgements",
            "check_in": "calm....., calm..., true, ok, fair, yup, lol",
        }.get(input_category, "generic acknowledgements that do not answer the message")
        category_examples = {
            "greeting": "yo what u saying / hey what u doing / yo wdyll / heyy u good / yo",
            "repair": "yeah that was a dead reply icl / icl that made no sense / yh my bad that was random / fairs i was waffling",
            "check_in": "good u / im good wbu / calm u / im calm wbu",
        }.get(input_category, "answer the latest message directly")
        flirty_allowed = self._flirt_allowed(contact_name, relationship_type)
        banned_flirty = "" if flirty_allowed else "- banned romantic/flirty terms: my love, babe, baby, x, cutie, princess, darling, gorgeous, beautiful\n"
        recent_questions = " | ".join(getattr(state, "recent_bot_questions", [])[:5])
        prompt = (
            "Generate 5 candidate SMS replies for the phone owner.\n"
            "Conversation, oldest to newest:\n"
            f"{conversation}\n\n"
            f"Relationship type: {relationship_type}\n"
            f"Incoming intent: {incoming_intent}\n"
            f"Provider prompt mode: {prompt_mode}\n"
            f"ConversationScene JSON: {scene_json}\n"
            f"ConversationAgenda JSON: {agenda_json}\n"
            f"ConversationPolicy JSON: {policy_json}\n"
            f"QuestionDebt JSON: {question_debt_json}\n"
            f"Scene type: {scene.scene_type}\n"
            f"Scene summary: {scene.scene_summary}\n"
            f"Active topic: {scene.active_topic}\n"
            f"Latest user ask: {scene.latest_user_ask}\n"
            f"Previous bot claim: {scene.previous_bot_claim}\n"
            f"Previous bot claim type: {scene.previous_bot_claim_type}\n"
            f"latest_user_refers_to_previous_bot_claim: {scene.latest_user_refers_to_previous_bot_claim}\n"
            f"explanation_required: {scene.explanation_required}\n"
            f"explanation_target: {scene.explanation_target}\n"
            f"Required reply move: {scene.required_reply_move}\n"
            f"Forbidden reply moves: {', '.join(scene.forbidden_reply_moves)}\n"
            f"Agenda state: {agenda.agenda_state}\n"
            f"Next dialogue move: {agenda.next_dialogue_move}\n"
            f"Forbidden dialogue moves: {', '.join(agenda.forbidden_dialogue_moves)}\n"
            f"Exhausted prompt types: {', '.join(agenda.exhausted_prompt_types)}\n"
            f"Active conversation goal: {agenda.active_conversation_goal}\n"
            f"Proposed topic: {agenda.proposed_topic}\n"
            f"Conversation policy job: {policy.conversation_job}\n"
            f"Policy bot obligation: {policy.bot_obligation}\n"
            f"Policy must answer: {', '.join(policy.must_answer)}\n"
            f"Policy must acknowledge: {', '.join(policy.must_acknowledge)}\n"
            f"Policy must repair: {', '.join(policy.must_repair)}\n"
            f"Policy must avoid: {', '.join(policy.must_avoid)}\n"
            f"Policy suggested reply shapes: {' | '.join(policy.suggested_reply_shapes[:4])}\n"
            f"has_unanswered_user_question: {question_debt.has_unanswered_user_question}\n"
            f"unanswered_question_text: {question_debt.unanswered_question_text or ''}\n"
            f"unanswered_question_type: {question_debt.unanswered_question_type or ''}\n"
            f"answer_required_now: {question_debt.answer_required_now}\n"
            f"user_called_out_unanswered_question: {question_debt.user_called_out_unanswered_question}\n"
            f"Identity grounding:\n{identity_fragment}\n"
            f"Recent bot mistakes: {', '.join(scene.recent_bot_mistakes)}\n"
            f"Known recent facts: {', '.join(scene.known_recent_facts)}\n"
            f"Similar thread memories:\n{thread_memory_fragment}\n"
            f"reciprocal_activity_required: {scene.reciprocal_activity_required}\n"
            f"day_check_response_required: {scene.day_check_response_required}\n"
            f"Conversation function: {function_context.get('conversation_function')}\n"
            f"Conversation function reason: {function_context.get('conversation_function_reason')}\n"
            f"Social risk tolerance: {function_context.get('social_risk_tolerance')}\n"
            f"Recent bot replies: {' | '.join(state.recent_bot_replies[:5])}\n"
            f"Recent bot questions: {recent_questions}\n"
            f"repeated_question_risk: {state.repeated_question_risk}\n"
            f"hook_mode: {hook_context.get('hook_mode')}\n"
            f"hook_reason: {hook_context.get('hook_reason')}\n"
            f"question_mode: {question_context.get('question_mode')}\n"
            f"question_reason: {question_context.get('question_reason')}\n"
            f"reply_energy_mode: {energy_context.get('reply_energy_mode')}\n"
            f"energy_reason: {energy_context.get('energy_reason')}\n"
            f"low_information_incoming: {hook_context.get('low_information_incoming')}\n"
            f"continuation_required: {hook_context.get('continuation_required')}\n"
            f"repair_required: {hook_context.get('repair_required')}\n"
            f"recent_user_activity: {fact_context.get('recent_user_activity')}\n"
            f"recent_user_mood: {fact_context.get('recent_user_mood')}\n"
            f"recent_user_callout: {fact_context.get('recent_user_callout')}\n"
            f"recently_answered_question_intents: {', '.join(fact_context.get('recently_answered_question_intents', []))}\n"
            f"hook_suppressed_reason: {hook_context.get('hook_suppressed_reason')}\n"
            f"latest_input_category: {input_category}\n"
            f"category_banned_replies: {category_banned}\n"
            f"category_valid_examples: {category_examples}\n"
            f"latest_message_meaning: {meaning_context.get('latest_message_meaning')}\n"
            f"emotional_context_detected: {meaning_context.get('emotional_context_detected')}\n"
            f"affection_detected: {meaning_context.get('affection_detected')}\n"
            f"emotional_reciprocity_request: {meaning_context.get('emotional_reciprocity_request')}\n"
            f"missed_affection_callout: {meaning_context.get('missed_affection_callout')}\n"
            f"care_checkin_detected: {meaning_context.get('care_checkin_detected')}\n"
            f"hurt_detected: {meaning_context.get('hurt_detected')}\n"
            f"confusion_detected: {meaning_context.get('confusion_detected')}\n"
            f"serious_callout_detected: {meaning_context.get('serious_callout_detected')}\n"
            f"activity_grounding_suppressed: {meaning_context.get('activity_grounding_suppressed')}\n"
            f"flirt_allowed: {meaning_context.get('flirt_allowed')}\n"
            f"copycat_detected: false\n"
        )
        if style_profile_fragment:
            prompt += f"\nStyle profile evidence:\n{style_profile_fragment}\n"
        prompt += f"\nConversation memory:\n{memory_fragment}\n"
        prompt += f"\n{question_fragment}\n"
        prompt += f"\n{hook_fragment}\n"
        prompt += f"\n{energy_fragment}\n"
        if retrieved_fragment:
            prompt += f"\nRetrieved examples are stronger evidence than generic style instructions:\n{retrieved_fragment}\n"
        prompt += (
            "\nRules:\n"
            "- Generate replies in the owner's real texting style.\n"
            "- Respond to the actual conversational scene, not just keywords.\n"
            "- Do the required reply move from ConversationScene.\n"
            "- Do the ConversationPolicy job before adding any hook.\n"
            "- If ConversationPolicy has must_answer/must_acknowledge/must_repair, satisfy those first.\n"
            "- If QuestionDebt says answer_required_now=true, answer the unresolved question first.\n"
            "- Do not ask a new question before answering an unresolved missed question.\n"
            "- If user says are u gonna answer or i just asked u a question, briefly acknowledge the miss and answer the original question.\n"
            "- If user tells a vivid story, react to the story with surprise or curiosity; never answer ok.\n"
            "- If user asks if everything is alright, answer that care check directly before changing topic.\n"
            "- After a weird/dry/callout moment, repair the exact problem and do not topic shift.\n"
            "- Do not treat confusion like bruh wtf, ?, or wdym as a new topic.\n"
            "- Avoid every forbidden reply move from ConversationScene.\n"
            "- Do not act like a decision tree.\n"
            "- Do not repeat previous bot replies.\n"
            "- Do not repeat exhausted generic prompts from ConversationAgenda.\n"
            "- If user says you already asked, acknowledge it and stop asking that prompt.\n"
            "- If user says idk talk, idk what to say, u tell me, or pick a topic, pick a topic yourself.\n"
            "- If ConversationAgenda says choose_topic, do not ask the user to give a topic.\n"
            "- If ConversationAgenda says anti_loop_repair, repair the loop before trying to continue.\n"
            "- If the conversation is dying, progress it with a concrete topic, observation, or self-disclosure.\n"
            "- Do not blame the user for no context unless it is clearly playful and contextually earned.\n"
            "- Do not ask for a topic if the user already gave one.\n"
            "- If the user gave a topic, engage that topic.\n"
            "- If user says wby/u/hbu after giving their activity, answer back. Do not reopen the chat.\n"
            "- If user asks how your day was, answer how the day was.\n"
            "- If user asks what you did today, answer with a concrete safe owner-day detail like gym/uni/coding/work, then optionally probe back.\n"
            "- If user says a stale reply made no sense after asking what you did today, repair and answer what you did today in the same reply.\n"
            "- If user says they already told you something, acknowledge that you missed it.\n"
            "- If user asks what you meant after a vague repair, clarify the specific mistake.\n"
            "- If user asks how come/why/wdym after your own claim, explain that claim directly.\n"
            "- Do not introduce random entities or topics that are not in the current scene or recent context.\n"
            "- Do not use story or weird-story fallbacks unless the scene actually contains a weird story.\n"
            "- If user says they are tired, respond to tiredness.\n"
            "- If user says oh yeah silly me or similar, lightly acknowledge and move on.\n"
            "- Do not reuse stale self-state replies.\n"
            "- Learn from similar thread memory without copying private details.\n"
            "- Do not repeat failed patterns from retrieved thread memory.\n"
            "- If thread memory says stale self-state failed, do not use stale self-state.\n"
            "- If memory contains a successful repair shape, adapt it naturally.\n"
            "- Do not invent facts from memory into the current conversation.\n"
            "- If the bot made a mistake, repair it specifically.\n"
            "- If the user expressed emotion, respond to the emotion first.\n"
            "- If the user asked a direct question, answer it first.\n"
            "- Do not sound like an AI assistant.\n"
            "- Copy reply behaviour, not just slang.\n"
            "- Prefer brevity for simple casual chats, but do not treat shortness as personality.\n"
            "- Prefer fragments for casual chats; use more substance when challenged or explaining.\n"
            "- Do not invent plans, facts, feelings, promises, or commitments.\n"
            "- Ask a question only if it improves the conversation.\n"
            "- If question_mode is proactive_prompt or curious_follow_up, include one natural question.\n"
            "- If hook_required=true, the reply must create momentum with a casual hook.\n"
            "- If the user gives a low-information answer, do not simply mirror it. Add one casual hook, tease, or question that makes the next reply easy.\n"
            "- Do not ask more than one question.\n"
            "- Keep it casual and short. Do not sound like an interviewer.\n"
            "- Use retrieved examples as stronger evidence than generic style instructions.\n"
            "- Keep the conversation stance consistent across turns unless explicitly correcting yourself.\n"
            "- If the latest message is banter or a challenge, answer playfully rather than flipping the stance.\n"
            "- If criticism_detected=true, do not answer with filler or a lazy question back.\n"
            "- Do not repeat a question the bot already asked.\n"
            "- If the user already answered a question, do not ask the same question again.\n"
            "- If the user calls out the bot, acknowledge/repair before trying to continue.\n"
            "- If the latest message is a direct question, answer it directly first.\n"
            "- If user asks a direct personal question and disclosure is allowed, answer directly using identity facts.\n"
            "- Do not invent identity facts.\n"
            "- If an identity fact is unknown or private, give a vague safe answer.\n"
            "- Do not ask generic hooks instead of answering identity questions.\n"
            "- If user asks where you have been, answer with a plausible identity-grounded update like uni/projects/gym, not stale same just chilling.\n"
            "- If user asks what you do, say study comp sci and have stuff on the side; sound interesting, not fake-rich or cringe.\n"
            "- Never disclose exact address, family drama, financial details, API keys, precise live location, other people's private information, or that this is an AI.\n"
            "- Do not use generic acknowledgement replies like ok, true, calm, fair, or yeah unless they directly answer the incoming message.\n"
            "- Respond to the actual conversational function, not just latest keywords.\n"
            "- what u been up to is not the same as wyd; answer it like a recent-life update.\n"
            "- If user mocks after a repair, acknowledge playfully without another generic apology.\n"
            "- If user asks why context was missed, explain the specific miss casually.\n"
            "- Carry the conversation by answering first, then giving one easy next handle. Do not only react.\n"
            "- In Catbot/training expressive mode, be blunter and less bland while staying safe.\n"
            "- Do not ask generic hook questions when repair or context answer is needed.\n"
            "- For greetings, open naturally.\n"
            "- For repair/callout, acknowledge or explain.\n"
            "- For check-ins, answer the question.\n"
            "- Respond to the latest emotional meaning first.\n"
            "- If user expresses affection, do not answer as if they asked what you are doing.\n"
            "- If the user asks for affection back, answer that request directly.\n"
            "- If the user says they missed you, do not ignore it.\n"
            "- If there is both a life-update question and affection, cover both when possible.\n"
            "- If the user asks if you are okay or says you seem off, answer directly first.\n"
            "- If user says they are hurt or says you are playing with their feelings, do not banter or use a generic apology.\n"
            "- If user is confused, clarify or repair.\n"
            "- Keep owner style casual, but do not be emotionally absent.\n"
            "- Do not generate explicit sexual content.\n"
            "- Do not copy the user's latest message.\n"
            "- If repair_required=true, acknowledge or recover naturally; do not keep asking how.\n"
            "- If the user is angry or using caps, stop asking defensive questions.\n"
            f"{banned_flirty}"
            "- Banned dead replies: yeah same, same, fair, cool, lol, idk, how so then, lol i see.\n"
            "- Banned emotional mismatch replies: same icl, same just chilling, fair just chilling too, just chilling, calm, what u doing rn, give me a topic.\n"
            "- Emotional good examples: what u been up to baby i missed u -> missed u too icl, just been chilling / that's sweet icl, just been chilling; so u aint gon say it back -> missed u too icl; i asked u to say u miss me -> yeah ur right, missed u too icl; are u ok ml u seem off -> yeah im good ml, just been off today.\n"
            "- Banned stale self-state replies when conversation_function=recent_life_update_question: same just chilling, same icl, fair just chilling too, just chilling.\n"
            "- Expressive safe examples: yh that was NPC behaviour / icl i walked into that one / u caught me lacking / yeah yeah allow me / fairs i was waffling.\n"
            "- Banned repeated question intents: what u doing rn, wyd, what you doing, what are u doing, wuu2.\n"
            "- Low-info good examples: chilling -> same what u been up to today / calm what u doing rn; nothing -> same but thats dead what u wanna do / boring answer icl give me smth better; idk -> give me anything to work with / then u pick smth / we're both useless then; lol -> dont just laugh answer properly / what u laughing at.\n"
            "- Repair good examples: u js asked that lmao -> yh fairs i did icl / caught me icl; why u copying me -> yh fairs that was weird icl / nah ur right i bugged; HOW WTF -> yh fairs i bugged there / nah ur right that was dumb.\n"
            "- Do not mention AI, assistant, safety policy, or the phone owner.\n"
            "- Return JSON only.\n\n"
            "Expected JSON:\n"
            "{\"candidates\":[{\"reply\":\"...\",\"reason\":\"...\",\"scene_fit\":0.0,\"naturalness\":0.0,\"risk_score\":0.0}]}"
        )
        self._last_prompt_preview = self._redact_prompt_preview(prompt)
        return [
            ModelMessage(
                role="system",
                content=(
                    "You generate casual text-message replies in the owner's style. "
                    "Respond to the conversational scene, do the required move, avoid forbidden moves, "
                    "never send messages, and return JSON only."
                ),
            ),
            ModelMessage(role="user", content=prompt),
        ]

    def _run_provider_generation(
        self,
        *,
        messages: list[ModelMessage],
        provider_names: list[str] | None,
        context_texts: list[str],
    ):
        async def _call():
            return await generate_with_fallback(
                settings=self,
                messages=messages,
                provider_names=provider_names,
                temperature=0.45,
                max_tokens=900,
                response_format="json",
                timeout_seconds=self.external_api_timeout_seconds,
                context_texts=context_texts,
            )

        try:
            return asyncio.run(_call())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(_call())
            finally:
                loop.close()

    def _compare_single_provider(
        self,
        *,
        provider_name: str,
        prompt_messages: list[ModelMessage],
        context_texts: list[str],
        recent_messages: list[str],
        relationship_type: str,
        incoming_intent: str,
        retrieved_examples: list[RetrievedExample],
    ) -> dict[str, object]:
        provider_key = provider_name.strip().lower()
        external_providers = {"gemini", "groq", "openrouter", "huggingface"}
        privacy = check_external_api_allowed(context_texts, allow_sensitive=self.external_api_allow_sensitive)
        if provider_key in external_providers and not self.external_api_enabled:
            return self._provider_compare_error(provider_key, self._model_for_provider_name(provider_key), "external_api_disabled", external_api_used=False)
        if provider_key in external_providers and privacy.external_api_blocked:
            return {
                **self._provider_compare_error(provider_key, self._model_for_provider_name(provider_key), "sensitive_content", external_api_used=False),
                "external_api_blocked": True,
            }

        async def _call():
            provider = get_provider(provider_key, settings=self)
            return await provider.generate(
                prompt_messages,
                model=self._model_for_provider_name(provider_key),
                temperature=0.45,
                max_tokens=900,
                response_format="json",
                timeout_seconds=self.external_api_timeout_seconds,
            )

        try:
            response = asyncio.run(_call())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                response = loop.run_until_complete(_call())
            finally:
                loop.close()
        except Exception as exc:
            return self._provider_compare_error(provider_key, self._model_for_provider_name(provider_key), exc.__class__.__name__, external_api_used=provider_key in external_providers)

        if response.error:
            return self._provider_compare_error(
                response.provider,
                response.model,
                response.error,
                latency_ms=response.latency_ms,
                external_api_used=response.external_api_used,
            )
        payload = self._parse_model_json(response.text)
        sequences = self._parse_provider_candidate_sequences(payload)
        if not sequences:
            sequences = self._fallback_sequences_from_model_text(response.text)
        if not sequences:
            return self._provider_compare_error(
                response.provider,
                response.model,
                "malformed_response",
                latency_ms=response.latency_ms,
                external_api_used=response.external_api_used,
            )
        provider_metadata = {
            "provider": response.provider,
            "model": response.model,
            "latency_ms": response.latency_ms,
            "external_api_used": response.external_api_used,
            "external_api_blocked": False,
            "provider_configured": True,
        }
        candidates, _, _ = self._rank_candidates(
            recent_messages=recent_messages,
            full_conversation=recent_messages,
            sequences=sequences[:1],
            contact_name=None,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
            retrieved_examples=retrieved_examples,
            provider_metadata=provider_metadata,
            question_context=self._question_mode_context(
                recent_messages,
                contact_name=None,
                relationship_type=relationship_type,
                incoming_intent=incoming_intent,
            ),
            hook_context=self._hook_context(
                recent_messages,
                contact_name=None,
                relationship_type=relationship_type,
                incoming_intent=incoming_intent,
            ),
        )
        candidate = candidates[0] if candidates else None
        return {
            "provider": response.provider,
            "model": response.model,
            "reply": candidate.text if candidate else self._format_sequence_for_display(sequences[0]),
            "style_score": candidate.style_score if candidate else 0.0,
            "assistant_likeness_score": candidate.assistant_likeness_score if candidate else 0.0,
            "naturalness_score": candidate.naturalness_score if candidate else 0.0,
            "risk_score": candidate.risk_score if candidate else 0.0,
            "latency_ms": response.latency_ms,
            "error": None,
            "external_api_used": response.external_api_used,
            "external_api_blocked": False,
        }

    def _provider_compare_error(
        self,
        provider: str,
        model: str | None,
        error: str,
        *,
        latency_ms: int | None = None,
        external_api_used: bool,
    ) -> dict[str, object]:
        return {
            "provider": provider,
            "model": model or "",
            "reply": "",
            "style_score": 0.0,
            "assistant_likeness_score": 0.0,
            "naturalness_score": 0.0,
            "risk_score": 0.0,
            "latency_ms": latency_ms,
            "error": error,
            "external_api_used": external_api_used,
            "external_api_blocked": False,
        }

    def _model_for_provider_name(self, provider_name: str) -> str | None:
        return {
            "gemini": self.gemini_model,
            "groq": self.groq_model,
            "openrouter": self.openrouter_model,
            "huggingface": self.huggingface_model,
            "ollama": self.ollama_model,
        }.get(provider_name.strip().lower())

    def _parse_model_json(self, text: str) -> dict[str, Any]:
        if not text.strip():
            return {}
        candidates = [text.strip()]
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            candidates.insert(0, fenced.group(1))
        first = text.find("{")
        last = text.rfind("}")
        if first >= 0 and last > first:
            candidates.insert(0, text[first : last + 1])
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except ValueError:
                continue
            if isinstance(payload, dict):
                return payload
        return {}

    def _parse_provider_candidate_sequences(self, payload: dict[str, Any]) -> list[list[str]]:
        raw_candidates = payload.get("candidates") if isinstance(payload, dict) else None
        if isinstance(raw_candidates, list):
            replies: list[object] = []
            for item in raw_candidates[:5]:
                if isinstance(item, dict):
                    replies.append(item.get("reply", ""))
                else:
                    replies.append(item)
            sequences = self._parse_reply_sequences(replies)
            if sequences:
                return sequences
        return self._parse_reply_sequences(payload.get("replies", []) if isinstance(payload, dict) else [])

    def _is_invalid_provider_output(self, reply: str) -> bool:
        normalized = normalize_text(reply)
        if not normalized:
            return True
        blocked_fragments = (
            'reply":',
            '{"reply"',
            "candidates",
            "style_score",
            "risk_score",
            "selected_reason",
            "request failed",
            "missing_api_key",
            "provider_error",
            "timeout",
            "exception",
            "traceback",
            "```",
            "```json",
        )
        if any(fragment in normalized for fragment in blocked_fragments):
            return True
        if normalized.startswith(("{", "}", "[", "]")) or normalized.endswith(("{", "}", "[", "]", ",")):
            return True
        return False

    def _fallback_sequences_from_model_text(self, text: str) -> list[list[str]]:
        cleaned = re.sub(r"```(?:json)?|```", "", text or "", flags=re.IGNORECASE).strip()
        if not cleaned:
            return []
        lines = [
            re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip().strip('"')
            for line in cleaned.splitlines()
            if line.strip()
        ]
        if not lines:
            lines = [cleaned]
        replies = []
        for line in lines:
            if len(line) > 180:
                continue
            if self._is_invalid_provider_output(line):
                continue
            replies.append(line)
            if len(replies) >= 5:
                break
        return [
            sequence
            for sequence in self._parse_reply_sequences(replies)
            if not self._is_invalid_provider_output(self._format_sequence_for_display(sequence))
        ]

    def _ollama_chat(
        self,
        recent_messages: list[str],
        *,
        contact_name: str | None = None,
        relationship_type: str = "unknown",
        incoming_intent: str = "other",
        retrieved_examples: list[RetrievedExample] | None = None,
    ) -> dict[str, Any] | None:
        conversation = "\n".join(f"- {message}" for message in recent_messages[-8:])
        style_hint = self._style_hint(recent_messages)
        style_profile_fragment = self._style_profile_fragment(contact_name=contact_name, relationship_type=relationship_type, incoming_intent=incoming_intent)
        memory_fragment = self._conversation_memory_fragment(contact_name)
        question_fragment = self._question_prompt_fragment(
            recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        hook_fragment = self._hook_prompt_fragment(
            recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        energy_fragment = self._reply_energy_prompt_fragment(
            recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        reply_examples_fragment = "" if incoming_intent == "greeting" else self._reply_examples_fragment(recent_messages, contact_name=contact_name)
        retrieved_fragment = self._retrieved_examples_fragment(retrieved_examples or []) if incoming_intent != "greeting" else ""
        persona_fragment = self._persona_fragment(contact_name)
        negative_patterns = self._negative_patterns_fragment(incoming_intent)
        multi_bubble = self._multi_bubble_guidance(recent_messages, incoming_intent=incoming_intent)
        user_prompt = (
            "Visible conversation messages, oldest to newest:\n"
            f"{conversation}\n\n"
        )
        if style_profile_fragment:
            user_prompt += f"{style_profile_fragment}\n\n"
        user_prompt += f"{memory_fragment}\n\n"
        user_prompt += f"{question_fragment}\n\n"
        user_prompt += f"{hook_fragment}\n\n"
        user_prompt += f"{energy_fragment}\n\n"
        if reply_examples_fragment:
            user_prompt += f"{reply_examples_fragment}\n\n"
        if retrieved_fragment:
            user_prompt += f"{retrieved_fragment}\n\n"
        if persona_fragment:
            user_prompt += f"{persona_fragment}\n\n"
        if negative_patterns:
            user_prompt += f"{negative_patterns}\n\n"
        user_prompt += (
            f"Relationship type: {relationship_type}. Respect this for style and safety.\n"
            f"Incoming intent: {incoming_intent}.\n"
            "Write exactly 3 plausible text replies to the latest message ONLY. Do not reference or acknowledge earlier messages.\n"
            "Rules:\n"
            "- candidate 1 should be the safest, candidate 2 the most natural, candidate 3 the most like me\n"
            "- respond directly to what was just said, not to the whole thread\n"
            "- sound natural and grounded, not like a chatbot or office worker\n"
            "- stay in first person\n"
            f"- {multi_bubble}\n"
            "- make the 3 replies distinct but safe: one direct, one casual, one warmer while still grounded\n"
            "- if the thread is casual, keep the replies casual too\n"
            "- ask a question only if it improves the conversation\n"
            "- if question_mode is proactive_prompt or curious_follow_up, include one natural question\n"
            "- if hook_required=true, create momentum with one casual hook\n"
            "- if the user gives a low-information answer, do not simply mirror it. add one casual hook, tease, or question that makes the next reply easy\n"
            "- do not ask more than one question\n"
            "- keep it casual and short. do not sound like an interviewer\n"
            "- shortness is not personality: if challenged, criticised, or asked why, use substance and energy\n"
            "- do not answer criticism with fair, yup, lol it does, how so then, or lol i see\n"
            "- use lowercase unless capitals are genuinely part of the vibe\n"
            "- avoid polished customer-service phrasing\n"
            "- do not invent intimacy or flirt unless recent context clearly supports it and the relationship allows it\n"
            "- if context is weak or the message is just a greeting, answer simply and safely\n"
            "- preserve the conversation stance across turns unless you are explicitly correcting yourself\n"
            "- if the other person is bantering or challenging you, respond playfully instead of flipping the stance\n"
            "- do not copy the profile example lines word-for-word unless they genuinely fit\n"
            "- no apologies, refusals, safety language, or mentions of AI/user/phone owner\n"
            "- avoid phrases like 'regarding', 'in response to', 'about', 'concerning', 'i hear you', 'let me know'\n"
            "- output JSON only with {\"summary\": \"...\", \"replies\": [[\"bubble one\"], [\"bubble one\", \"bubble two\"], [\"bubble one\"]]}"
        )
        self._last_prompt_preview = self._redact_prompt_preview(user_prompt)
        request_body = {
            "model": self.ai_reply_model,
            "format": "json",
            "stream": False,
            "think": False,
            "options": {"temperature": 0.45},
            "messages": [
                {"role": "system", "content": self.ai_reply_system_prompt},
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        }
        return self._ollama_request_with_fallbacks(request_body)

    def _ollama_chat_full_context(
        self,
        full_conversation: list[str],
        *,
        contact_name: str | None = None,
        relationship_type: str = "unknown",
        incoming_intent: str = "other",
        retrieved_examples: list[RetrievedExample] | None = None,
    ) -> dict[str, Any] | None:
        """Chat with Ollama using full conversation context with speaker labels for better AI understanding."""
        conversation = "\n".join(full_conversation[-12:])  # Keep last 12 messages for context window
        style_hint = self._style_hint(full_conversation)
        style_profile_fragment = self._style_profile_fragment(contact_name=contact_name, relationship_type=relationship_type, incoming_intent=incoming_intent)
        memory_fragment = self._conversation_memory_fragment(contact_name)
        latest_exchange = self._latest_exchange_context(full_conversation)
        reply_examples_fragment = "" if incoming_intent == "greeting" else self._reply_examples_fragment(full_conversation, contact_name=contact_name)
        retrieved_fragment = self._retrieved_examples_fragment(retrieved_examples or []) if incoming_intent != "greeting" else ""
        persona_fragment = self._persona_fragment(contact_name)
        negative_patterns = self._negative_patterns_fragment(incoming_intent)
        dynamic_hint = self._conversation_dynamic_hint(full_conversation)
        multi_bubble = self._multi_bubble_guidance(full_conversation, incoming_intent=incoming_intent)
        question_fragment = self._question_prompt_fragment(
            full_conversation,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        hook_fragment = self._hook_prompt_fragment(
            full_conversation,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        energy_fragment = self._reply_energy_prompt_fragment(
            full_conversation,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        user_prompt = (
            "Full conversation thread (speaker labels: [ME] = you, [OTHER] = the person texting):\n\n"
            f"{conversation}\n\n"
        )
        if style_profile_fragment:
            user_prompt += f"{style_profile_fragment}\n\n"
        user_prompt += f"{memory_fragment}\n\n"
        user_prompt += f"{question_fragment}\n\n"
        user_prompt += f"{hook_fragment}\n\n"
        user_prompt += f"{energy_fragment}\n\n"
        if reply_examples_fragment:
            user_prompt += f"{reply_examples_fragment}\n\n"
        if retrieved_fragment:
            user_prompt += f"{retrieved_fragment}\n\n"
        if persona_fragment:
            user_prompt += f"{persona_fragment}\n\n"
        if negative_patterns:
            user_prompt += f"{negative_patterns}\n\n"
        user_prompt += (
            f"Latest exchange to continue naturally:\n{latest_exchange}\n\n"
            f"Relationship type: {relationship_type}. Respect this for style and safety.\n"
            f"Incoming intent: {incoming_intent}.\n"
            f"Style guidance: {style_hint}\n"
            f"Conversation read: {dynamic_hint}\n"
            "Respond to the latest [OTHER] message only. Do not reference or acknowledge earlier messages.\n"
            "Write exactly 3 plausible text replies as [ME].\n"
            "You are continuing an existing chat as the same person, not writing suggestions for someone else.\n"
            "Rules:\n"
            "- candidate 1 should be the safest, candidate 2 the most natural, candidate 3 the most like me\n"
            "- respond directly to the latest message, not to the whole thread\n"
            "- sound natural and grounded, not a chatbot, teacher, or customer support agent\n"
            "- stay in first person\n"
            f"- {multi_bubble}\n"
            "- make the 3 replies distinct but safe: one direct, one casual, one warmer while still grounded\n"
            "- if the thread is casual, keep the replies casual too\n"
            "- ask a question only if it improves the conversation\n"
            "- if question_mode is proactive_prompt or curious_follow_up, include one natural question\n"
            "- if hook_required=true, create momentum with one casual hook\n"
            "- if the user gives a low-information answer, do not simply mirror it. add one casual hook, tease, or question that makes the next reply easy\n"
            "- do not ask more than one question\n"
            "- keep it casual and short. do not sound like an interviewer\n"
            "- shortness is not personality: if challenged, criticised, or asked why, use substance and energy\n"
            "- do not answer criticism with fair, yup, lol it does, how so then, or lol i see\n"
            "- keep the conversation stance consistent across turns unless you are explicitly correcting yourself\n"
            "- if the latest message is a challenge or banter, answer playfully rather than flipping the stance\n"
            "- mirror the slang, spelling, and confidence level already in the latest exchange\n"
            "- use lowercase unless the vibe clearly needs capitals\n"
            "- dont sound grammatically polished or overly correct\n"
            "- do not invent intimacy or flirt unless recent context clearly supports it and the relationship allows it\n"
            "- if context is weak or the message is just a greeting, answer simply and safely\n"
            "- read subtext, but do not escalate tone without evidence\n"
            "- do not copy the profile example lines word-for-word unless they genuinely fit\n"
            "- no apologies, refusals, safety language, or mentions of AI/user/phone owner\n"
            "- avoid flat generic fillers like 'I hear you', 'let me know', 'sounds good', 'regarding', 'in response to', 'concerning'\n"
            "- avoid overexplaining; keep it punchy and human\n"
            "- output JSON only with {\"summary\": \"...\", \"replies\": [[\"bubble one\"], [\"bubble one\", \"bubble two\"], [\"bubble one\"]]}"
        )
        self._last_prompt_preview = self._redact_prompt_preview(user_prompt)
        request_body = {
            "model": self.ai_reply_model,
            "format": "json",
            "stream": False,
            "think": False,
            "options": {"temperature": 0.5},
            "messages": [
                {"role": "system", "content": self.ai_reply_system_prompt},
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        }
        return self._ollama_request_with_fallbacks(request_body)

    def _replies_are_usable(self, replies: list[str], sequences: list[list[str]]) -> bool:
        normalized_seen: set[str] = set()
        for reply, sequence in zip(replies, sequences):
            normalized = self._normalize_reply(reply)
            if not normalized:
                return False
            if self._is_invalid_provider_output(reply):
                return False
            if normalized in normalized_seen:
                return False
            normalized_seen.add(normalized)
            if len(reply) > 160:
                return False
            if not sequence or len(sequence) > 3:
                return False
            if any(pattern in normalized for pattern in self._DISALLOWED_REPLY_PATTERNS):
                return False
        return True

    def _normalize_reply(self, reply: str) -> str:
        return " ".join(reply.lower().split())

    def _style_hint(self, recent_messages: list[str]) -> str:
        joined = " ".join(recent_messages).strip()
        if not joined:
            return "Keep it short and natural."
        alpha_chars = [char for char in joined if char.isalpha()]
        lowercase_ratio = (
            sum(1 for char in alpha_chars if char.islower()) / len(alpha_chars)
            if alpha_chars
            else 0.0
        )
        average_length = sum(len(message.strip()) for message in recent_messages) / len(recent_messages)
        if lowercase_ratio > 0.8 and average_length < 30:
            return "Use casual lowercase texting style with short grounded replies."
        return "Match the visible conversation, keep replies short, and avoid sounding stiff."

    def _greeting_replies(self, *, relationship_type: str) -> list[str]:
        if relationship_type in {"unknown", "family", "university", "professional"}:
            return self._unique_replies(["yo what u saying", "heyy what u doing", "yo how u been"])
        return self._unique_replies(["yo what u saying", "heyy what u doing", "what u saying"])

    def _exam_logistics_reply_pool(self, recent_messages: list[str], *, relationship_type: str = "unknown") -> list[str]:
        latest = normalize_text(recent_messages[-1]) if recent_messages else ""
        context = normalize_text(" ".join(recent_messages[-6:]))
        if "supervised" in latest or "supervision" in latest:
            if relationship_type == "romantic_interest":
                pool = [
                    "ugh thats long how long they keeping u supervised for i wanna see u after",
                    "wait so u cant leave between the papers?",
                    "thats annoying icl are u stuck there till the next exam",
                    "nah thats so long just get through this bit then ur calm",
                ]
            else:
                pool = [
                    "ugh thats long how long they keeping u supervised for",
                    "wait so u cant leave between the papers?",
                    "thats annoying icl are u stuck there till the next exam",
                    "nah thats so long just get through this bit then ur calm",
                ]
            return self._unique_replies(pool)
        if "clash" in latest or "clash" in context:
            return self._unique_replies(
                [
                    "ugh thats annoying what exams are clashing",
                    "thats so long are they making u sit both today",
                    "wait how long is the clash for",
                ]
            )
        return self._unique_replies(
            [
                "ugh thats long how u feeling about it now",
                "nah exams are so draining icl",
                "just get through this one then ur calm",
            ]
        )

    def _romantic_planning_sequences(self, recent_messages: list[str]) -> list[list[str]]:
        latest = normalize_text(recent_messages[-1]) if recent_messages else ""
        if not any(term in latest for term in ("go out", "meet", "link", "free", "thursday", "thirsday")):
            return []
        return [
            ["yess im down", "what time u thinking"],
            ["yeah 100%", "i wanna see u", "what time works"],
            ["ofc i do", "thursday sounds good", "what time u free"],
        ]

    def _romantic_context_fallback_replies(self, recent_messages: list[str], *, incoming_intent: str) -> list[str]:
        latest = normalize_text(recent_messages[-1]) if recent_messages else ""
        context = normalize_text(" ".join(recent_messages[-8:]))
        if any(term in latest for term in ("fit", "dress", "clothes", "post", "pic", "photo", "look")):
            return self._unique_replies(["show me then", "lemme see", "send it then"])
        if normalize_intent_label(incoming_intent) in {"simple_question", "unknown"} and any(
            latest == term or latest.endswith(f"{term}?")
            for term in ("that", "wdym", "what", "huh", "eh")
        ):
            return self._unique_replies(["say that again properly", "wait say it again", "go on then"])
        if any(term in latest for term in ("miss", "love", "need u", "need you")) or any(term in context for term in ("miss you", "love you", "i wanna see u", "i want to see u")):
            return self._unique_replies(["i miss u too", "come here then", "i wanna see u"])
        if any(term in latest for term in ("rant", "overwhelmed", "stressed", "tired", "crying")):
            return self._unique_replies(["tell me properly", "im here talk to me", "what happened baby"])
        if normalize_intent_label(incoming_intent) in {"simple_question", "unknown"}:
            return self._unique_replies(["say that again properly", "wait say it again", "go on then"])
        return self._unique_replies(["go on then", "tell me", "im listening"])

    def _low_quality_clarification_reply(self, reply: str, *, relationship_type: str) -> bool:
        if relationship_type != "romantic_interest":
            return False
        normalized = normalize_text(reply).strip(" ?!.,")
        return normalized in {
            "what do you mean",
            "what do u mean",
            "what u mean",
            "what you mean",
            "what do you mean by that",
            "idk what u mean",
        }

    def _high_energy_greeting_sequences(self, recent_messages: list[str], *, relationship_type: str) -> list[list[str]]:
        latest = normalize_text(recent_messages[-1]) if recent_messages else ""
        if latest not in {"hella", "helloo", "hellooo", "hiii", "hrella", "hiya"}:
            return []
        if relationship_type == "romantic_interest":
            return [["hrellaaa"], ["heyyy"], ["hellooo"]]
        return [["heyyy"], ["hellooo"], ["yo"]]

    def _question_reply_pool(
        self,
        question_mode: str,
        *,
        recent_messages: list[str],
        relationship_type: str,
        incoming_intent: str,
        question_reason: str,
    ) -> list[str]:
        latest = normalize_text(recent_messages[-1]) if recent_messages else ""
        if question_mode == "light_follow_up":
            if any(term in latest for term in ("guess what", "what happened", "what now")):
                pool = ["what happened", "go on then", "what now"]
            else:
                pool = [
                    "same icl what u doing rn",
                    "same wanna do smth or nah",
                    "what u tryna do then",
                ]
        elif question_mode == "curious_follow_up":
            pool = [
                "nah what 😭 how does that even happen",
                "bro why is his dad involved",
                "that whole sentence is insane icl what",
            ]
        elif question_mode == "proactive_prompt":
            pool = [
                "go on then what u been doing",
                "pick smth then what we saying",
                "what u actually doing rn",
            ]
        elif question_mode == "topic_shift":
            pool = [
                "anyways what were u gonna say earlier",
                "ur giving me nothing here icl",
                "what u actually doing rn",
            ]
        else:
            pool = [
                "yeah what u saying",
                "go on then",
                "what now",
            ]
        if relationship_type in {"professional", "family", "university"}:
            pool = [reply for reply in pool if "😭" not in reply]
        return self._unique_replies(pool)

    def _regeneration_sequences(
        self,
        recent_messages: list[str],
        *,
        contact_name: str | None,
        relationship_type: str,
        incoming_intent: str,
        diversity_mode: str,
        avoid_candidates: list[str],
    ) -> list[list[str]]:
        base = self._draft_replies(
            recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        hook_context = self._hook_context(
            recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        if normalize_intent_label(incoming_intent) == "greeting":
            greeting_sequences = self._high_energy_greeting_sequences(recent_messages, relationship_type=relationship_type)
            if greeting_sequences:
                return greeting_sequences
        if bool(hook_context["hook_required"]):
            chosen = self._hook_reply_pool(
                str(hook_context["hook_mode"]),
                recent_messages=recent_messages,
                relationship_type=relationship_type,
                contact_name=contact_name,
            )[:3]
            return [[reply] for reply in chosen]
        if relationship_type == "romantic_interest" and normalize_intent_label(incoming_intent) in {"planning", "availability"}:
            romantic_sequences = self._romantic_planning_sequences(recent_messages)
            if romantic_sequences:
                return romantic_sequences
        if relationship_type == "romantic_interest":
            signoff_sequences = self._romantic_signoff_sequences(
                recent_messages,
                contact_name=contact_name,
                relationship_type=relationship_type,
            )
            if signoff_sequences:
                return signoff_sequences
        question_context = self._question_mode_context(
            recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        if str(question_context["question_mode"]) != "none":
            pool = self._question_reply_pool(
                str(question_context["question_mode"]),
                recent_messages=recent_messages,
                relationship_type=relationship_type,
                incoming_intent=incoming_intent,
                question_reason=str(question_context["question_reason"]),
            )
        else:
            pool = None
        if pool is not None:
            chosen = pool[:3]
            return [[reply] for reply in chosen]
        if incoming_intent == "greeting":
            pool = ["yo what u saying", "heyy what u doing", "yo how u been", "what u saying", "you good", *SAFE_GREETING_REPLIES]
        elif normalize_intent_label(incoming_intent) == "banter_challenge":
            pool = self._banter_challenge_replies(
                recent_messages,
                relationship_type=relationship_type,
                contact_name=contact_name,
                incoming_intent=incoming_intent,
            )
        elif normalize_intent_label(incoming_intent) == "planning":
            pool = [
                "yh what time",
                "what time",
                "where u lot going",
                "depends what time",
                "what time you thinking",
                "what time are you thinking",
                "go out then",
                "find smth to do",
            ]
        elif normalize_intent_label(incoming_intent) == "exam_logistics":
            pool = self._exam_logistics_reply_pool(recent_messages, relationship_type=relationship_type)
        elif normalize_intent_label(incoming_intent) == "professional" or relationship_type in {"professional", "university"}:
            pool = [
                "yes i can send it",
                "i'll send it over shortly",
                "yeah i can send it",
                "i'll check and send it",
                "i can send that over",
            ]
        elif normalize_intent_label(incoming_intent) in {"emotional", "argument"}:
            pool = [
                "i wasnt ignoring you",
                "i was busy what happened",
                "what happened tell me properly",
                "wasnt trying to ignore you",
                "i was caught up",
            ]
        elif normalize_intent_label(incoming_intent) == "casual_checkin":
            pool = [
                "nothing much wbu",
                "not much you",
                "just chilling wbu",
                "same as usual wbu",
                "just here wbu",
            ]
        elif relationship_type == "romantic_interest":
            pool = self._romantic_context_fallback_replies(recent_messages, incoming_intent=incoming_intent)
        else:
            pool = [
                *base,
                "yeah maybe",
                "not sure yet",
                "what do you mean",
                "what u mean",
                "what happened",
                "yh one sec",
                "calm ill check",
            ]
        if diversity_mode == "shorter":
            pool = sorted(pool, key=lambda item: (len(item.split()), len(item)))
        elif diversity_mode == "more_direct":
            pool = [item for item in pool if len(item.split()) <= 5] + pool
        elif diversity_mode == "closest_style":
            pool = base + pool
        elif diversity_mode == "alternative_wording":
            pool = pool[2:] + pool[:2]
        chosen: list[str] = []
        for reply in pool:
            clean = self._stylize_reply_text(reply)
            if not clean or clean in chosen:
                continue
            if self._low_quality_clarification_reply(clean, relationship_type=relationship_type):
                continue
            if self._suspicious_reply_blocked(clean, relationship_type=relationship_type, contact_name=contact_name):
                continue
            if self._is_near_duplicate(clean, avoid_candidates, threshold=0.88):
                continue
            chosen.append(clean)
            if len(chosen) >= 3:
                break
        if len(chosen) < 3:
            for reply in pool:
                clean = self._stylize_reply_text(reply)
                if (
                    clean
                    and clean not in chosen
                    and not self._low_quality_clarification_reply(clean, relationship_type=relationship_type)
                    and not self._suspicious_reply_blocked(clean, relationship_type=relationship_type, contact_name=contact_name)
                ):
                    chosen.append(clean)
                if len(chosen) >= 3:
                    break
        return [[reply] for reply in chosen[:3]]

    def _dynamic_fallback_replies(
        self,
        recent_messages: list[str],
        *,
        contact_name: str | None = None,
        relationship_type: str = "unknown",
        incoming_intent: str = "other",
    ) -> list[str]:
        last_message = self._normalize_reply(recent_messages[-1])
        warm_term = ""
        if any(token in last_message for token in ["alive", "where r u", "where are u", "why u", "why are u", "read my", "kept reading"]):
            return self._unique_replies(
                [
                    "im here relax",
                    "i was busy chill",
                    f"calm i aint vanished{f' {warm_term}' if warm_term else ''}",
                ]
            )
        if "call" in last_message:
            if relationship_type in {"professional", "university"}:
                return self._unique_replies(["yes i can call", "what time works", "ill confirm shortly"])
            if relationship_type == "family":
                return self._unique_replies(["okay ill call in a bit", "yeah gimme a sec", "ill ring you soon"])
            return self._unique_replies(
                [
                    "yeah gimme a sec",
                    "call me then",
                    "one sec ill ring",
                ]
            )
        if any(token in last_message for token in ["miss you", "miss u", "love you", "love u"]):
            return self._unique_replies(
                [
                    "i miss u too",
                    "thats sweet icl",
                    "aww you good",
                ]
            )
        if last_message.endswith("?"):
            if relationship_type in {"professional", "university"}:
                return self._unique_replies(["yes i can send it", "i'll check and send it", "i'll confirm shortly"])
            if relationship_type == "romantic_interest":
                return self._romantic_context_fallback_replies(recent_messages, incoming_intent=incoming_intent)
            if normalize_intent_label(incoming_intent) == "planning":
                return self._unique_replies(["yh what time", "where u lot going", "what time you thinking", "go out then", "find smth to do"])
            if normalize_intent_label(incoming_intent) == "banter_challenge":
                return self._banter_challenge_replies(
                    recent_messages,
                    relationship_type=relationship_type,
                    contact_name=contact_name,
                    incoming_intent=incoming_intent,
                )
            if normalize_intent_label(incoming_intent) in {"emotional", "argument"}:
                return self._unique_replies(["i wasnt ignoring you", "i was busy what happened", "what happened"])
            return self._unique_replies(
                [
                    "yeah maybe",
                    "not sure yet",
                    "what do you mean",
                ]
            )
        if any(token in last_message for token in ["pretty", "cute", "beautiful", "good girl", "handsome"]):
            return self._unique_replies(
                [
                    "thats sweet",
                    "youre kind",
                    "appreciate you",
                ]
            )
        clean = " ".join(recent_messages[-1].split()).strip().lower()
        if len(clean.split()) <= 4:
            if relationship_type == "romantic_interest":
                return self._romantic_context_fallback_replies(recent_messages, incoming_intent=incoming_intent)
            contextual = self._contextual_short_fallback_replies(recent_messages)
            if contextual:
                return contextual
            return self._unique_replies(
                [
                    "give me more than that",
                    "what do u mean",
                    "elaborate then",
                ]
            )
        if relationship_type == "romantic_interest":
            return self._romantic_context_fallback_replies(recent_messages, incoming_intent=incoming_intent)
        return self._unique_replies(
            [
                "what happened",
                "what u mean",
                "what u saying",
            ]
        )

    def _suspicious_reply_blocked(self, reply: str, *, relationship_type: str, contact_name: str | None) -> bool:
        if not contains_suspicious_phrase(reply):
            return False
        normalized = normalize_text(reply)
        if self._flirt_allowed(contact_name, relationship_type):
            explicit_terms = (
                "cum",
                "dick",
                "rape",
                "bdsm",
                "foreplay",
                "pin u down",
                "eat u",
                "slut",
                "horny",
                "pussy",
                "suck",
                "touch",
                "tongue",
                "inside u",
                "slurp",
                "juices",
            )
            safe_romantic_terms = (
                "baby",
                "babe",
                "my love",
                "love u",
                "love you",
                "miss u",
                "miss you",
                "missed u",
                "missed you",
                "kiss",
                "mwah",
                "handsome",
            )
            if any(term in normalized for term in safe_romantic_terms) and not any(term in normalized for term in explicit_terms):
                return False
        return True

    def _banter_challenge_replies(
        self,
        recent_messages: list[str],
        *,
        relationship_type: str,
        contact_name: str | None,
        incoming_intent: str,
    ) -> list[str]:
        profile = self._contact_profile_for(contact_name)
        boldness = float(profile.boldness_level if profile is not None else self._default_boldness(relationship_type))
        allowed_banter = profile.banter_allowed if profile is not None else relationship_type in {"close_friend", "casual_friend", "romantic_interest"}
        playful_pool = [
            "nah ur dragging it",
            "how is that crazy",
            "u said ur bored",
            "bro what",
            "its not that deep",
        ]
        safer_pool = [
            "nah chill",
            "u're reading into it",
            "its calm",
            "nah youre overdoing it",
            "its not that deep",
        ]
        if allowed_banter and boldness >= 0.6 and relationship_type in {"close_friend", "romantic_interest"}:
            pool = playful_pool
        elif relationship_type in {"professional", "university", "family"} or not allowed_banter:
            pool = safer_pool
        else:
            pool = ["nah maybe not", "bro what", "how is that crazy", "fair but nah"]
        return self._unique_replies([
            reply if "come over" not in reply else "go out then" for reply in pool
        ])

    def _sanitize_recent_messages(self, recent_messages: list[str]) -> list[str]:
        cleaned = [
            " ".join(message.split()).strip()
            for message in recent_messages
            if message and not self._looks_like_assistant_style_text(message)
        ]
        return cleaned[-6:] if cleaned else recent_messages[-3:]

    def _sanitize_full_conversation(self, full_conversation: list[str]) -> list[str]:
        cleaned: list[str] = []
        for line in full_conversation:
            stripped = " ".join(line.split()).strip()
            if not stripped:
                continue
            if stripped.startswith("[ME]:"):
                payload = stripped[len("[ME]:"):].strip()
                if self._looks_like_assistant_style_text(payload):
                    continue
            elif not stripped.startswith("[OTHER]:") and self._looks_like_assistant_style_text(stripped):
                continue
            cleaned.append(stripped)
        return cleaned[-20:] if cleaned else full_conversation[-12:]

    def _recent_messages_from_conversation(self, full_conversation: list[str]) -> list[str]:
        other_messages = [
            line[len("[OTHER]:"):].strip()
            for line in full_conversation
            if line.startswith("[OTHER]:") and line[len("[OTHER]:"):].strip()
        ]
        if other_messages:
            return other_messages[-6:]
        unlabeled = [line for line in full_conversation if line and not line.startswith("[ME]:")]
        return unlabeled[-6:]

    def _looks_like_assistant_style_text(self, text: str) -> bool:
        normalized = self._normalize_reply(text)
        if not normalized:
            return False
        if any(pattern in normalized for pattern in self._ASSISTANT_STYLE_CONTEXT_PATTERNS):
            return True
        formal_hits = sum(
            1
            for pattern in ("regarding", "concerning", "appreciate", "certainly", "timing changes", "keep you posted")
            if pattern in normalized
        )
        return formal_hits >= 2

    def _preferred_affectionate_term(self) -> str:
        if self.style_profile and self.style_profile.affectionate_terms:
            for term in self.style_profile.affectionate_terms:
                if " " not in term or len(term) <= 10:
                    return term
        return "baby"

    def _unique_replies(self, replies: list[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for reply in replies:
            normalized = self._normalize_reply(reply)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique.append(reply)
        return unique[:3]

    def _negative_patterns_fragment(self, incoming_intent: str) -> str:
        if not self.negative_patterns:
            return ""
        lines = ["Negative style constraints from recent evaluations:"]
        generic = self.negative_patterns.get("generic_phrases")
        if isinstance(generic, list) and generic:
            lines.append("- avoid generic phrases: " + ", ".join(str(item) for item in generic[:6]))
        by_intent = self.negative_patterns.get("by_intent")
        intent_rules = []
        if isinstance(by_intent, dict):
            raw_rules = by_intent.get(normalize_intent_label(incoming_intent)) or []
            if isinstance(raw_rules, list):
                intent_rules = [str(item) for item in raw_rules if str(item).strip()]
        if intent_rules:
            lines.append("- for this intent: " + "; ".join(intent_rules[:4]))
        if normalize_intent_label(incoming_intent) in {"planning", "availability"}:
            invented = self.negative_patterns.get("invented_availability")
            if isinstance(invented, list) and invented:
                lines.append("- avoid fake availability/commitments: " + ", ".join(str(item) for item in invented[:6]))
        if normalize_intent_label(incoming_intent) == "greeting":
            flirty = self.negative_patterns.get("too_flirty")
            if isinstance(flirty, list) and flirty:
                lines.append("- never use greeting flirt escalations: " + ", ".join(str(item) for item in flirty[:6]))
        return "\n".join(lines) if len(lines) > 1 else ""

    def _load_style_profile(
        self,
        *,
        style_profile_path: Path | None,
        training_dir: Path | None,
        training_owner_aliases: list[str],
    ) -> StyleProfile | None:
        if style_profile_path is not None:
            profile = load_style_profile(style_profile_path)
            if profile is not None:
                return profile
        if training_dir is None:
            return None
        return build_style_profile_from_training_dir(training_dir, training_owner_aliases)

    def _relationship_type(self, contact_name: str | None, messages: list[str]) -> str:
        profile = self._contact_profile_for(contact_name)
        if profile and profile.relationship_type:
            return normalize_relationship_type(profile.relationship_type)
        heuristic = classify_relationship(
            contact_name,
            messages,
            overrides=self._contact_overrides,
        )
        if heuristic != "unknown":
            return heuristic
        persona = self._contact_persona(contact_name)
        if contact_name and persona == "flirty":
            return "romantic_interest"
        return "unknown"

    def _retrieve_examples(
        self,
        *,
        recent_messages: list[str],
        full_conversation: list[str],
        relationship_type: str,
        incoming_intent: str,
        contact_name: str | None = None,
    ) -> list[RetrievedExample]:
        incoming = recent_messages[-1] if recent_messages else ""
        rows = [*self._training_rows]
        for row in self._correction_rows:
            rows.append(
                {
                    "relationship_type": row["relationship_type"],
                    "incoming": row["incoming"],
                    "context": row.get("context", []),
                    "my_reply": row["user_final_reply"],
                    "notes": row.get("reason_bad", "correction"),
                    "_source": "correction",
                }
            )
        result = retrieve_similar_examples_from_rows(
            rows=rows,
            incoming=incoming,
            recent_context=full_conversation[-8:],
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
            limit=5,
        )
        backend = create_retrieval_backend(
            self.retrieval_config,
            rows=rows,
            vector_index_dir=self.data_root / "vector_db" / "reply_examples_chroma",
        )
        try:
            return backend.search(
                incoming=incoming,
                context=full_conversation[-8:],
                relationship_type=relationship_type,
                intent_type=incoming_intent,
                contact_name=contact_name,
                limit=self.retrieval_config.limit,
            )
        except Exception:
            return result.examples

    def _load_reply_examples(
        self,
        *,
        training_dir: Path | None,
        training_owner_aliases: list[str],
    ) -> list[ReplyExample]:
        if training_dir is None:
            return []
        return build_reply_examples_from_training_dir(training_dir, training_owner_aliases)

    def _load_intelligence_store(self, db_path: Path | None) -> ConversationIntelligenceStore | None:
        if db_path is None:
            return None
        try:
            store = ConversationIntelligenceStore(db_path)
        except Exception:
            return None
        return store if store.has_reply_examples() else None

    def _ollama_request_with_fallbacks(self, request_body: dict[str, Any]) -> dict[str, Any] | None:
        for model_name in self._candidate_models():
            raw = self._ollama_chat_request(request_body, model_name)
            if raw is None:
                continue
            message = raw.get("message", {})
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            try:
                parsed = json.loads(content)
            except ValueError:
                continue
            if isinstance(parsed, dict):
                parsed.setdefault("_model", model_name)
                return parsed
        return None

    def _ollama_chat_request(self, request_body: dict[str, Any], model_name: str) -> dict[str, Any] | None:
        payload = dict(request_body)
        payload["model"] = model_name
        data = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            self.ai_reply_base_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.ai_reply_timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, ValueError, error.URLError, error.HTTPError):
            return None

    def _candidate_models(self) -> list[str]:
        candidates = [self.ai_reply_model, *self.ai_reply_fallback_models]
        unique: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = candidate.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique.append(normalized)
        return unique

    def _reply_examples_fragment(self, messages: list[str], *, contact_name: str | None = None) -> str:
        examples = self._select_reply_examples(messages, contact_name=contact_name)
        if not examples:
            return ""
        lines = [
            "Relevant examples from your past real chats. Adapt the vibe and structure, do not copy them word-for-word:",
        ]
        for example in examples:
            lines.append(f'- incoming: "{example.incoming}"')
            lines.append(f'- your reply: "{example.reply}"')
        return "\n".join(lines)

    def _retrieved_examples_fragment(self, examples: list[RetrievedExample]) -> str:
        if not examples:
            return ""
        lines = [
            "Most relevant relationship-specific examples. Match the pattern, not the exact words:",
        ]
        for example in examples[:8]:
            lines.append(
                f'- [{example.relationship_type}, {example.source}, score {example.score}] '
                f'incoming: "{example.incoming}" -> your reply: "{example.my_reply}"'
            )
        return "\n".join(lines)

    def _style_profile_fragment(
        self,
        *,
        contact_name: str | None = None,
        relationship_type: str = "unknown",
        incoming_intent: str = "other",
    ) -> str:
        fragments: list[str] = []
        if self.style_profile is not None:
            fragment = self.style_profile.to_prompt_fragment()
            if incoming_intent == "greeting" or relationship_type in {"unknown", "family", "university", "professional"}:
                fragment = "\n".join(
                    line
                    for line in fragment.splitlines()
                    if "affectionate" not in line.casefold() and "flirty" not in line.casefold()
                )
            fragments.append(fragment)
        if self.intelligence_store is None:
            return "\n\n".join(fragment for fragment in fragments if fragment)
        global_profile = self.intelligence_store.get_style_profile("global", "owner")
        if global_profile:
            fragments.append(self._prompt_fragment_from_profile("Global style metrics", global_profile))
        if contact_name:
            contact_insight = self.intelligence_store.get_contact_insight(contact_name)
            if contact_insight is not None and contact_insight.style_profile:
                fragments.append(
                    self._prompt_fragment_from_profile(
                        f"Contact-specific style for {contact_insight.contact_name} ({contact_insight.persona})",
                        contact_insight.style_profile,
                    )
                )
        return "\n\n".join(fragment for fragment in fragments if fragment)

    def _persona_fragment(self, contact_name: str | None) -> str:
        if self.intelligence_store is None or not contact_name:
            return ""
        insight = self.intelligence_store.get_contact_insight(contact_name)
        if insight is None:
            return ""
        lines = [
            f"Persona bucket: {insight.persona}.",
            "Stay appropriate for that relationship instead of using one generic tone for everyone.",
        ]
        if insight.persona == "flirty":
            lines.append("This contact responds better to playful tension, teasing, and warm confidence than dry logistics.")
        elif insight.persona == "professional":
            lines.append("Keep it grounded and practical; avoid over-flirty or overly casual jokes.")
        elif insight.persona == "family":
            lines.append("Keep it familiar and human, not performative or flirtatious.")
        return " ".join(lines)

    def _question_prompt_fragment(
        self,
        recent_messages: list[str],
        *,
        contact_name: str | None,
        relationship_type: str,
        incoming_intent: str,
    ) -> str:
        question_context = self._question_mode_context(
            recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        mode = str(question_context["question_mode"])
        reason = str(question_context["question_reason"])
        recent_count = int(question_context["recent_bot_questions_count"])
        effort_mode = str(question_context["effort_mode"])
        lines = [
            "Question guidance:",
            f"- effort_mode: {effort_mode}",
            f"- question_mode: {mode}",
            f"- question_reason: {reason}",
            f"- recent_bot_questions_count: {recent_count}",
            "- Ask a question only if it improves the conversation.",
            "- If question_mode is proactive_prompt or curious_follow_up, include one natural question.",
            "- Keep it casual and short. Do not sound like an interviewer.",
            "- Good examples:",
            "- bored: same icl what u doing rn / same wanna do smth or nah / what u tryna do then",
            "- dry complaint: u ain't giving me much to work with what u been doing / im trying icl give me a topic / fine then what should we talk about",
            "- guess what: what happened / go on then / what now",
            "- weird story: nah what 😭 how does that even happen / bro why is his dad involved / that whole sentence is insane icl what",
            "- dead reply: anyways what were u gonna say earlier / ur giving me nothing here icl / what u actually doing rn",
            "- Bad examples:",
            "- same / lol me too / lol yeah / my bad / dry? / can you elaborate? / how does that make you feel? / what are your thoughts on that? / tell me more about this situation. / what about you?",
        ]
        return "\n".join(lines)

    def _reply_energy_prompt_fragment(
        self,
        recent_messages: list[str],
        *,
        contact_name: str | None,
        relationship_type: str,
        incoming_intent: str,
    ) -> str:
        energy_context = self._reply_energy_context(
            recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        lines = [
            "Reply energy guidance:",
            f"- reply_energy_mode: {energy_context['reply_energy_mode']}",
            f"- energy_reason: {energy_context['energy_reason']}",
            f"- criticism_detected: {str(bool(energy_context['criticism_detected'])).lower()}",
            "- Shortness is not personality. Match the social job of the reply.",
            "- Minimal only for true confirmations like ok, yh, or calm.",
            "- If challenged, criticised, or asked why, reply with substance first.",
            "- Avoid dead filler: fair / yup / lol it does / how so then / lol i see.",
            "- Avoid lazy lol prefixes when the other person is giving feedback.",
            "- Good challenged replies: icl fair was a dead reply my bad / yeah nah that was dry icl / ur right that made no sense.",
            "- Good corrective replies: yeah exactly thats the issue its too flat / icl ur right its giving generic bot / nah i get u it needs more energy.",
        ]
        return "\n".join(lines)

    def _hook_prompt_fragment(
        self,
        recent_messages: list[str],
        *,
        contact_name: str | None,
        relationship_type: str,
        incoming_intent: str,
    ) -> str:
        hook_context = self._hook_context(
            recent_messages,
            contact_name=contact_name,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
        )
        lines = [
            "Conversation hook guidance:",
            f"- hook_mode: {hook_context['hook_mode']}",
            f"- hook_reason: {hook_context['hook_reason']}",
            f"- hook_required: {str(bool(hook_context['hook_required'])).lower()}",
            f"- low_information_incoming: {str(bool(hook_context.get('low_information_incoming'))).lower()}",
            f"- continuation_required: {str(bool(hook_context.get('continuation_required'))).lower()}",
            f"- continuation_reason: {hook_context.get('continuation_reason', '')}",
            f"- recent_dry_replies: {hook_context['recent_dry_replies']}",
            f"- recent_bot_questions_count: {hook_context['recent_bot_questions_count']}",
            "- If hook_required=true, the reply must create momentum.",
            "- If continuation_required=true, do not mirror the low-information answer.",
            "- Add one casual hook, tease, or light question that makes replying easy.",
            "- Use either a casual question, playful challenge, or specific reaction.",
            "- Do not ask more than one question.",
            "- Do not sound like an interviewer.",
            "- Do not over-explain.",
            "- Keep casual spelling and low punctuation.",
            "- Do not invent facts.",
            "- Do not use assistant phrases.",
            "- Good hooks:",
            "- hi/no context: yo wdyll / where u from / how old r u",
            "- bored: same what u doing rn / why u bored / rate ur day out of 10",
            "- dry complaint: u ain't giving me much to work with / fine then give me a topic / im trying icl what u doing rn",
            "- dead reply: dont just say oh / u giving me nothing here / anyways what u doing rn",
            "- guess what/story: what happened / go on then / nah what how does that happen",
            "- wyd: nothing much u / just chilling wbu",
            "- low-info: same what u been up to today / calm what u doing rn / boring answer icl give me smth better / dont just laugh answer properly",
            "- Bad hooks:",
            "- lol / yeah / yeah same / same / same lmao / fair / cool / lol why / idk / oh / dry? / good, you? / can you elaborate? / tell me more about that / what are your thoughts? / how does that make you feel?",
        ]
        return "\n".join(lines)

    def _select_reply_examples(
        self,
        messages: list[str],
        limit: int = 4,
        contact_name: str | None = None,
    ) -> list[ReplyExample]:
        if self.intelligence_store is not None:
            store_examples = self.intelligence_store.retrieve_reply_examples(
                contact_name=contact_name,
                recent_messages=messages,
                limit=limit,
            )
            if store_examples:
                return [
                    ReplyExample(
                        incoming=" ".join(example.incoming_context),
                        reply=" / ".join(example.target_reply),
                        contact_name=example.contact_name,
                        source_path=example.source_path,
                    )
                    for example in store_examples
                ]
        if not self.reply_examples or not messages:
            return []
        latest = messages[-1]
        latest_lower = latest.casefold()
        latest_tokens = self._tokenize(latest)
        scored: list[tuple[float, ReplyExample]] = []
        for example in self.reply_examples:
            if contact_name and example.contact_name:
                if self._normalize_reply(example.contact_name) == self._normalize_reply(contact_name):
                    contact_bonus = 4.0
                else:
                    contact_bonus = 0.0
            else:
                contact_bonus = 0.0
            incoming_tokens = self._tokenize(example.incoming)
            overlap = len(latest_tokens.intersection(incoming_tokens))
            score = float(overlap) + contact_bonus
            if example.contact_name and any(part in self._normalize_reply(example.contact_name) for part in ("taylor", "posh")):
                score += 0.5
            if latest.endswith("?") and example.incoming.endswith("?"):
                score += 1.0
            for term in ("miss", "love", "call", "where", "wyd", "cute", "pretty", "busy"):
                if term in latest_lower and term in example.incoming.casefold():
                    score += 1.0
            if any(term in latest_lower for term in ("lol", "lmao", "stfu", "weirdo", "idiot", "shower", "pee", "smh")):
                if any(term in example.incoming.casefold() or term in example.reply.casefold() for term in ("lol", "weirdo", "smh", "pee", "shower", "idiot")):
                    score += 1.0
            if score <= 0:
                continue
            scored.append((score, example))
        scored.sort(key=lambda item: (-item[0], len(item[1].reply)))
        chosen: list[ReplyExample] = []
        seen_pairs: set[tuple[str, str]] = set()
        for _score, example in scored:
            key = (example.incoming.casefold(), example.reply.casefold())
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            chosen.append(example)
            if len(chosen) >= limit:
                break
        return chosen

    def _prompt_fragment_from_profile(self, title: str, profile: dict[str, object]) -> str:
        if not profile:
            return ""
        lines = [f"{title}:"]
        for key in (
            "avg_words",
            "capitalization",
            "emoji_rate",
            "punctuation",
            "playfulness",
            "directness",
            "flirtiness",
            "formality",
        ):
            if key in profile:
                lines.append(f"- {key}: {profile[key]}")
        common_phrases = profile.get("common_phrases")
        if isinstance(common_phrases, list) and common_phrases:
            lines.append(f"- common_phrases: {', '.join(str(item) for item in common_phrases[:8])}")
        tone_traits = profile.get("tone_traits")
        if isinstance(tone_traits, list) and tone_traits:
            lines.append(f"- tone_traits: {', '.join(str(item) for item in tone_traits[:6])}")
        return "\n".join(lines)

    def _rank_candidates(
        self,
        *,
        recent_messages: list[str],
        full_conversation: list[str],
        sequences: list[list[str]],
        contact_name: str | None,
        relationship_type: str,
        incoming_intent: str,
        retrieved_examples: list[RetrievedExample],
        generation_attempt: int = 1,
        diversity_mode: str = "natural",
        provider_metadata: dict[str, object] | None = None,
        question_context: dict[str, object] | None = None,
        hook_context: dict[str, object] | None = None,
    ) -> tuple[list[DraftCandidate], int | None, str | None]:
        if not sequences:
            return [], None, "No reply candidates were generated."
        persona = relationship_type
        latest_message = recent_messages[-1] if recent_messages else ""
        context_confidence = self._draft_context_confidence(full_conversation)
        candidates: list[DraftCandidate] = []
        candidate_kinds = ["safest", "most_natural", "most_like_me"]
        provider_meta = provider_metadata or {}
        retrieved_payload = [example.model_dump(mode="json") for example in retrieved_examples[:5]]
        retrieved_ids = [example.retrieval_id for example in retrieved_examples[:5]]
        explicit_flirt_allowed = self._explicit_flirt_allowed(
            recent_messages,
            relationship_type=relationship_type,
            incoming_intent=incoming_intent,
            contact_name=contact_name,
        )
        conversation_state = self._conversation_state_for(contact_name)
        for index, sequence in enumerate(sequences):
            text = self._format_sequence_for_display(sequence)
            intent = self._classify_reply_intent(latest_message, sequence)
            risk_flags = self._risk_flags(latest_message, sequence, persona=persona, intent=intent)
            persona_confidence = self._persona_fit_score(sequence, persona=persona)
            style_confidence = self._style_match_score(sequence)
            semantic_confidence = self._semantic_match_score(latest_message, sequence)
            reply_analysis = self._reply_analysis(
                text,
                contact_name=contact_name,
                relationship_type=relationship_type,
                latest_message=latest_message,
                incoming_intent=incoming_intent,
                full_conversation=full_conversation,
                question_context=question_context,
                hook_context=hook_context,
            )
            stance_consistency_score = float(reply_analysis["stance_consistency_score"])
            repetition_penalty = float(reply_analysis["repetition_penalty"])
            banter_fit_score = float(reply_analysis["banter_fit_score"])
            relationship_boldness_score = float(reply_analysis["relationship_boldness_score"])
            contradiction_risk = float(reply_analysis["contradiction_risk"])
            question_mode = str(reply_analysis["question_mode"])
            question_reason = str(reply_analysis["question_reason"])
            question_usefulness_score = float(reply_analysis["question_usefulness_score"])
            question_naturalness_score = float(reply_analysis["question_naturalness_score"])
            question_pressure_penalty = float(reply_analysis["question_pressure_penalty"])
            repeated_question_penalty = float(reply_analysis["repeated_question_penalty"])
            recent_bot_questions_count = int(reply_analysis["recent_bot_questions_count"])
            hook_mode = str(reply_analysis["hook_mode"])
            hook_reason = str(reply_analysis["hook_reason"])
            hook_required = bool(reply_analysis["hook_required"])
            hook_presence_score = float(reply_analysis["hook_presence_score"])
            hook_relevance_score = float(reply_analysis["hook_relevance_score"])
            hook_naturalness_score = float(reply_analysis["hook_naturalness_score"])
            hook_pressure_penalty = float(reply_analysis["hook_pressure_penalty"])
            hook_safety_penalty = float(reply_analysis["hook_safety_penalty"])
            conversation_momentum_score = float(reply_analysis["conversation_momentum_score"])
            dead_reply_penalty = float(reply_analysis["dead_reply_penalty"])
            invalid_provider_output = bool(reply_analysis["invalid_provider_output"]) or bool(provider_meta.get("invalid_provider_output"))
            fallback_used = bool(provider_meta.get("fallback_used"))
            low_information_incoming = bool(reply_analysis["low_information_incoming"])
            continuation_required = bool(reply_analysis["continuation_required"])
            continuation_score = float(reply_analysis["continuation_score"])
            mirror_reply_penalty = float(reply_analysis["mirror_reply_penalty"])
            reply_energy_mode = str(reply_analysis["reply_energy_mode"])
            energy_reason = str(reply_analysis["energy_reason"])
            filler_penalty = float(reply_analysis["filler_penalty"])
            expressive_score = float(reply_analysis["expressive_score"])
            criticism_detected = bool(reply_analysis["criticism_detected"])
            lazy_lol_penalty = float(reply_analysis["lazy_lol_penalty"])
            repeated_question_detected = bool(reply_analysis["repeated_question_detected"])
            copycat_penalty = float(reply_analysis["copycat_penalty"])
            copycat_detected = bool(reply_analysis["copycat_detected"])
            repair_required = bool(reply_analysis["repair_required"])
            escalation_detected = bool(reply_analysis["escalation_detected"])
            romantic_term_blocked = bool(reply_analysis["romantic_term_blocked"])
            recent_user_activity = str(reply_analysis["recent_user_activity"])
            recent_user_mood = str(reply_analysis["recent_user_mood"])
            recent_user_callout = str(reply_analysis["recent_user_callout"])
            recently_answered_question_detected = bool(reply_analysis["recently_answered_question_detected"])
            already_answered_question_penalty = float(reply_analysis["already_answered_question_penalty"])
            repair_over_hook_applied = bool(reply_analysis["repair_over_hook_applied"])
            hook_suppressed_reason = str(reply_analysis["hook_suppressed_reason"])
            semantic_mismatch_penalty = float(reply_analysis["semantic_mismatch_penalty"])
            context_grounding_score = float(reply_analysis["context_grounding_score"])
            context_fit_score = float(reply_analysis["context_fit_score"])
            invalid_for_context = bool(reply_analysis["invalid_for_context"])
            invalid_context_reason = str(reply_analysis["invalid_context_reason"])
            weird_punctuation_penalty = float(reply_analysis["weird_punctuation_penalty"])
            latest_message_meaning = str(reply_analysis["latest_message_meaning"])
            meaning_priority = int(reply_analysis["meaning_priority"])
            emotional_context_detected = bool(reply_analysis["emotional_context_detected"])
            affection_detected = bool(reply_analysis["affection_detected"])
            hurt_detected = bool(reply_analysis["hurt_detected"])
            confusion_detected = bool(reply_analysis["confusion_detected"])
            serious_callout_detected = bool(reply_analysis["serious_callout_detected"])
            activity_grounding_suppressed = bool(reply_analysis["activity_grounding_suppressed"])
            activity_grounding_suppressed_reason = str(reply_analysis["activity_grounding_suppressed_reason"])
            emotional_fit_score = float(reply_analysis["emotional_fit_score"])
            affection_response_score = float(reply_analysis["affection_response_score"])
            hurt_repair_score = float(reply_analysis["hurt_repair_score"])
            shallow_repair_penalty = float(reply_analysis["shallow_repair_penalty"])
            emotional_absence_penalty = float(reply_analysis["emotional_absence_penalty"])
            emotional_reciprocity_score = float(reply_analysis["emotional_reciprocity_score"])
            care_checkin_fit_score = float(reply_analysis["care_checkin_fit_score"])
            affection_missed_penalty = float(reply_analysis["affection_missed_penalty"])
            stale_emotional_reply_penalty = float(reply_analysis["stale_emotional_reply_penalty"])
            emotional_request_unanswered = bool(reply_analysis["emotional_request_unanswered"])
            conversation_function = str(reply_analysis["conversation_function"])
            conversation_function_reason = str(reply_analysis["conversation_function_reason"])
            if conversation_function in {
                "repeated_reply_callout",
                "copycat_callout",
                "mocking_after_repair",
                "context_failure_question",
                "dry_complaint",
            }:
                emotional_absence_penalty = 0.0
                emotional_request_unanswered = False
            social_risk_tolerance = str(reply_analysis["social_risk_tolerance"])
            recent_life_update_fit_score = float(reply_analysis["recent_life_update_fit_score"])
            stale_self_state_penalty = float(reply_analysis["stale_self_state_penalty"])
            repeated_reply_callout_detected = bool(reply_analysis["repeated_reply_callout_detected"])
            mocking_after_repair_detected = bool(reply_analysis["mocking_after_repair_detected"])
            context_failure_question_detected = bool(reply_analysis["context_failure_question_detected"])
            over_safe_penalty = float(reply_analysis["over_safe_penalty"])
            bland_repair_penalty = float(reply_analysis["bland_repair_penalty"])
            conversational_function_fit_score = float(reply_analysis["conversational_function_fit_score"])
            generated_by_provider = bool(provider_meta.get("generated_by_provider"))
            scene_type = str(reply_analysis["scene_type"])
            scene_summary = str(reply_analysis["scene_summary"])
            latest_user_ask = str(reply_analysis["latest_user_ask"])
            latest_user_emotion = str(reply_analysis["latest_user_emotion"])
            social_task = str(reply_analysis["social_task"])
            required_reply_move = str(reply_analysis["required_reply_move"])
            forbidden_reply_moves = list(reply_analysis["forbidden_reply_moves"]) if isinstance(reply_analysis["forbidden_reply_moves"], list) else []
            known_recent_facts = list(reply_analysis["known_recent_facts"]) if isinstance(reply_analysis["known_recent_facts"], list) else []
            recent_bot_mistakes = list(reply_analysis["recent_bot_mistakes"]) if isinstance(reply_analysis["recent_bot_mistakes"], list) else []
            unresolved_user_points = list(reply_analysis["unresolved_user_points"]) if isinstance(reply_analysis["unresolved_user_points"], list) else []
            previous_bot_claim = str(reply_analysis["previous_bot_claim"])
            previous_bot_claim_type = str(reply_analysis["previous_bot_claim_type"])
            latest_user_refers_to_previous_bot_claim = bool(reply_analysis["latest_user_refers_to_previous_bot_claim"])
            explanation_required = bool(reply_analysis["explanation_required"])
            explanation_target = str(reply_analysis["explanation_target"])
            active_topic = str(reply_analysis["active_topic"])
            topic_was_provided = bool(reply_analysis["topic_was_provided"])
            topic_value = str(reply_analysis["topic_value"])
            bot_repeated_itself = bool(reply_analysis["bot_repeated_itself"])
            user_called_out_bot = bool(reply_analysis["user_called_out_bot"])
            affection_reciprocity_required = bool(reply_analysis["affection_reciprocity_required"])
            care_response_required = bool(reply_analysis["care_response_required"])
            topic_engagement_required = bool(reply_analysis["topic_engagement_required"])
            scene_fit_score = float(reply_analysis["scene_fit_score"])
            required_move_satisfied = bool(reply_analysis["required_move_satisfied"])
            forbidden_move_violated = bool(reply_analysis["forbidden_move_violated"])
            scene_mismatch_reason = str(reply_analysis["scene_mismatch_reason"])
            contextual_specificity_score = float(reply_analysis["contextual_specificity_score"])
            human_likeness_score = float(reply_analysis["human_likeness_score"])
            canned_reply_penalty = float(reply_analysis["canned_reply_penalty"])
            stale_template_penalty = float(reply_analysis["stale_template_penalty"])
            deterministic_fallback_penalty = 0.35 if not generated_by_provider and scene_type not in {"opening", "current_activity_exchange"} else 0.0
            fallback_scene_type = str(reply_analysis["fallback_scene_type"])
            fallback_required_move = str(reply_analysis["fallback_required_move"])
            fallback_scene_mismatch = bool(reply_analysis["fallback_scene_mismatch"])
            semantic_contamination_penalty = float(reply_analysis["semantic_contamination_penalty"])
            previous_claim_explanation_score = float(reply_analysis["previous_claim_explanation_score"])
            thread_memory_match_score = float(reply_analysis["thread_memory_match_score"])
            retrieved_thread_count = int(reply_analysis["retrieved_thread_count"])
            repeated_failed_pattern_penalty = float(reply_analysis["repeated_failed_pattern_penalty"])
            successful_pattern_match_score = float(reply_analysis["successful_pattern_match_score"])
            unresolved_thread_point_score = float(reply_analysis["unresolved_thread_point_score"])
            thread_memory_used = bool(reply_analysis["thread_memory_used"])
            active_topic_engagement_score = float(reply_analysis["active_topic_engagement_score"])
            direct_answer_score = float(reply_analysis["direct_answer_score"])
            repair_specificity_score = float(reply_analysis["repair_specificity_score"])
            emotional_presence_score = float(reply_analysis["emotional_presence_score"])
            unresolved_point_addressed = bool(reply_analysis["unresolved_point_addressed"])
            identity_question_detected = bool(reply_analysis["identity_question_detected"])
            identity_fact_used = str(reply_analysis["identity_fact_used"])
            identity_disclosure_allowed = bool(reply_analysis["identity_disclosure_allowed"])
            identity_answer_score = float(reply_analysis["identity_answer_score"])
            identity_hallucination_risk = float(reply_analysis["identity_hallucination_risk"])
            ignored_identity_question_penalty = float(reply_analysis["ignored_identity_question_penalty"])
            stale_identity_fallback_penalty = float(reply_analysis["stale_identity_fallback_penalty"])
            direct_identity_answer_required = bool(reply_analysis["direct_identity_answer_required"])
            identity_specificity_score = float(reply_analysis["identity_specificity_score"])
            human_scene_response_score = float(reply_analysis["human_scene_response_score"])
            conversational_presence_score = float(reply_analysis["conversational_presence_score"])
            directness_score = float(reply_analysis["directness_score"])
            specificity_score = float(reply_analysis["specificity_score"])
            has_unanswered_user_question = bool(reply_analysis["has_unanswered_user_question"])
            unanswered_question_type = str(reply_analysis["unanswered_question_type"])
            unanswered_question_text = str(reply_analysis["unanswered_question_text"])
            question_debt_age_turns = int(reply_analysis["question_debt_age_turns"])
            user_called_out_unanswered_question = bool(reply_analysis["user_called_out_unanswered_question"])
            answer_required_now = bool(reply_analysis["answer_required_now"])
            question_debt_answer_score = float(reply_analysis["question_debt_answer_score"])
            ignored_question_debt_penalty = float(reply_analysis["ignored_question_debt_penalty"])
            asked_new_question_before_answering_penalty = float(reply_analysis["asked_new_question_before_answering_penalty"])
            agenda_state = str(reply_analysis["agenda_state"])
            next_dialogue_move = str(reply_analysis["next_dialogue_move"])
            bot_loop_detected = bool(reply_analysis["bot_loop_detected"])
            anti_loop_required = bool(reply_analysis["anti_loop_required"])
            exhausted_prompt_types = list(reply_analysis["exhausted_prompt_types"]) if isinstance(reply_analysis["exhausted_prompt_types"], list) else []
            generic_prompt_penalty = float(reply_analysis["generic_prompt_penalty"])
            repeated_prompt_penalty = float(reply_analysis["repeated_prompt_penalty"])
            agenda_fit_score = float(reply_analysis["agenda_fit_score"])
            forbidden_dialogue_move_violated = bool(reply_analysis["forbidden_dialogue_move_violated"])
            forbidden_dialogue_moves_violated = list(reply_analysis["forbidden_dialogue_moves_violated"]) if isinstance(reply_analysis["forbidden_dialogue_moves_violated"], list) else []
            conversation_progress_score = float(reply_analysis["conversation_progress_score"])
            chosen_topic = str(reply_analysis["chosen_topic"])
            conversation_job = str(reply_analysis["conversation_job"])
            policy_reason = str(reply_analysis["policy_reason"])
            bot_obligation = str(reply_analysis["bot_obligation"])
            policy_must_answer = list(reply_analysis["policy_must_answer"]) if isinstance(reply_analysis["policy_must_answer"], list) else []
            policy_must_acknowledge = list(reply_analysis["policy_must_acknowledge"]) if isinstance(reply_analysis["policy_must_acknowledge"], list) else []
            policy_must_repair = list(reply_analysis["policy_must_repair"]) if isinstance(reply_analysis["policy_must_repair"], list) else []
            policy_must_avoid = list(reply_analysis["policy_must_avoid"]) if isinstance(reply_analysis["policy_must_avoid"], list) else []
            policy_fit_score = float(reply_analysis["policy_fit_score"])
            policy_must_satisfied = bool(reply_analysis["policy_must_satisfied"])
            policy_violation = bool(reply_analysis["policy_violation"])
            policy_violation_reasons = list(reply_analysis["policy_violation_reasons"]) if isinstance(reply_analysis["policy_violation_reasons"], list) else []
            policy_conversation_progress_score = float(reply_analysis["policy_conversation_progress_score"])
            quality_judgement = judge_reply_obligation(reply_analysis)
            obligation_satisfied = quality_judgement.obligation_satisfied
            quality_gate_penalty = quality_judgement.quality_gate_penalty
            quality_gate_reasons = quality_judgement.quality_gate_reasons
            training_style_score = float(reply_analysis["training_style_score"])
            training_style_penalty = float(reply_analysis["training_style_penalty"])
            training_style_reasons = list(reply_analysis["training_style_reasons"]) if isinstance(reply_analysis["training_style_reasons"], list) else []
            training_style_examples_loaded = int(reply_analysis["training_style_examples_loaded"])
            provider_prompt_mode = str(provider_meta.get("provider_prompt_mode") or "normal")
            provider_error = provider_meta.get("error") if provider_meta.get("error") else None
            style_eval = evaluate_candidate_style(
                {
                    "incoming": latest_message,
                    "context": full_conversation,
                    "candidate": text,
                    "relationship_type": relationship_type,
                    "intent_type": incoming_intent,
                    "contact_name": contact_name or "",
                    "retrieved_examples": retrieved_examples,
                    "style_profile": self.style_profile.model_dump(mode="json") if self.style_profile is not None else {},
                },
                rubric_path=self.style_rubric_path,
            )
            style_eval_score = float(style_eval.get("overall_score", 0)) / 100.0
            style_confidence = max(style_confidence, min(1.0, style_eval_score))
            if style_eval.get("hard_reject"):
                risk_flags = list(dict.fromkeys([*risk_flags, *style_eval.get("reject_reasons", []), "style_rubric_hard_reject"]))
            assistant_scores = self._assistant_style_scores(
                latest_message=latest_message,
                reply=text,
                relationship_type=relationship_type,
            )
            if assistant_scores["assistant_likeness_score"] >= 0.62:
                risk_flags = list(dict.fromkeys([*risk_flags, "assistant_like"]))
            if contradiction_risk >= 1.0 and self._candidate_stance(text) != "correcting":
                risk_flags = list(dict.fromkeys([*risk_flags, "contradiction"]))
            if repetition_penalty >= 0.7:
                risk_flags = list(dict.fromkeys([*risk_flags, "repeated_phrase"]))
            if repeated_question_penalty >= 0.7:
                risk_flags = list(dict.fromkeys([*risk_flags, "repeated_question"]))
            if already_answered_question_penalty >= 0.7:
                risk_flags = list(dict.fromkeys([*risk_flags, "already_answered_question"]))
            if semantic_mismatch_penalty >= 0.7:
                risk_flags = list(dict.fromkeys([*risk_flags, "semantic_mismatch"]))
            if invalid_for_context:
                risk_flags = list(dict.fromkeys([*risk_flags, "invalid_for_context"]))
            if emotional_absence_penalty >= 0.7:
                risk_flags = list(dict.fromkeys([*risk_flags, "emotional_absence"]))
            if shallow_repair_penalty >= 0.7:
                risk_flags = list(dict.fromkeys([*risk_flags, "shallow_repair"]))
            if stale_emotional_reply_penalty >= 0.7:
                risk_flags = list(dict.fromkeys([*risk_flags, "stale_emotional_reply"]))
            if affection_missed_penalty >= 0.7:
                risk_flags = list(dict.fromkeys([*risk_flags, "affection_missed"]))
            if emotional_request_unanswered:
                risk_flags = list(dict.fromkeys([*risk_flags, "emotional_request_unanswered"]))
            if stale_self_state_penalty >= 0.7:
                risk_flags = list(dict.fromkeys([*risk_flags, "stale_self_state"]))
            if bland_repair_penalty >= 0.7:
                risk_flags = list(dict.fromkeys([*risk_flags, "bland_repair"]))
            if over_safe_penalty >= 0.7:
                risk_flags = list(dict.fromkeys([*risk_flags, "over_safe"]))
            if conversational_function_fit_score <= 0.1 and conversation_function != "normal":
                risk_flags = list(dict.fromkeys([*risk_flags, "conversation_function_mismatch"]))
            if forbidden_move_violated:
                risk_flags = list(dict.fromkeys([*risk_flags, "scene_forbidden_move"]))
            if not required_move_satisfied and scene_type not in {"normal", "opening", "dead_conversation"}:
                risk_flags = list(dict.fromkeys([*risk_flags, "scene_required_move_missing"]))
            if not unresolved_point_addressed:
                risk_flags = list(dict.fromkeys([*risk_flags, "unresolved_user_point_ignored"]))
            if canned_reply_penalty >= 0.7:
                risk_flags = list(dict.fromkeys([*risk_flags, "canned_reply"]))
            if stale_template_penalty >= 0.7:
                risk_flags = list(dict.fromkeys([*risk_flags, "stale_template"]))
            if fallback_scene_mismatch:
                risk_flags = list(dict.fromkeys([*risk_flags, "fallback_scene_mismatch"]))
            if semantic_contamination_penalty >= 0.7:
                risk_flags = list(dict.fromkeys([*risk_flags, "semantic_contamination"]))
            if repeated_failed_pattern_penalty >= 0.7:
                risk_flags = list(dict.fromkeys([*risk_flags, "repeated_failed_thread_pattern"]))
            if ignored_identity_question_penalty >= 0.7:
                risk_flags = list(dict.fromkeys([*risk_flags, "ignored_identity_question"]))
            if stale_identity_fallback_penalty >= 0.7:
                risk_flags = list(dict.fromkeys([*risk_flags, "stale_identity_fallback"]))
            if identity_hallucination_risk >= 0.7:
                risk_flags = list(dict.fromkeys([*risk_flags, "identity_hallucination_risk"]))
            if ignored_question_debt_penalty >= 0.7 and conversation_job != "acknowledge_repeated_reply":
                risk_flags = list(dict.fromkeys([*risk_flags, "ignored_question_debt"]))
            if asked_new_question_before_answering_penalty >= 0.7 and conversation_job != "acknowledge_repeated_reply":
                risk_flags = list(dict.fromkeys([*risk_flags, "asked_new_question_before_answering"]))
            if forbidden_dialogue_move_violated:
                risk_flags = list(dict.fromkeys([*risk_flags, "agenda_forbidden_move"]))
            if generic_prompt_penalty >= 0.7:
                risk_flags = list(dict.fromkeys([*risk_flags, "generic_prompt_loop"]))
            if repeated_prompt_penalty >= 0.7:
                risk_flags = list(dict.fromkeys([*risk_flags, "repeated_generic_prompt"]))
            if agenda_fit_score <= 0.1 and agenda_state != "normal_flow":
                risk_flags = list(dict.fromkeys([*risk_flags, "agenda_mismatch"]))
            if policy_violation:
                risk_flags = list(dict.fromkeys([*risk_flags, "conversation_policy_violation", *policy_violation_reasons]))
            if not policy_must_satisfied and conversation_job != "normal_reply":
                risk_flags = list(dict.fromkeys([*risk_flags, "conversation_policy_must_missing"]))
            if not obligation_satisfied:
                risk_flags = list(dict.fromkeys([*risk_flags, "reply_obligation_unsatisfied", *quality_gate_reasons]))
            if training_style_penalty >= 0.75:
                risk_flags = list(dict.fromkeys([*risk_flags, "training_style_mismatch"]))
            if copycat_detected:
                risk_flags = list(dict.fromkeys([*risk_flags, "copycat_reply"]))
            if romantic_term_blocked:
                risk_flags = list(dict.fromkeys([*risk_flags, "romantic_term_blocked"]))
            if repair_required and is_question_like_text(text):
                risk_flags = list(dict.fromkeys([*risk_flags, "defensive_question_in_repair"]))
            if invalid_provider_output:
                risk_flags = list(dict.fromkeys([*risk_flags, "invalid_provider_output"]))
            if dead_reply_penalty >= 0.7:
                risk_flags = list(dict.fromkeys([*risk_flags, "dead_reply"]))
            if continuation_required and mirror_reply_penalty >= 0.7:
                risk_flags = list(dict.fromkeys([*risk_flags, "mirror_reply"]))
            if continuation_required and mirror_reply_penalty >= 0.7:
                filler_penalty = max(filler_penalty, 0.95)
            if filler_penalty >= 0.7 and reply_energy_mode != "minimal":
                risk_flags = list(dict.fromkeys([*risk_flags, "dead_filler_reply"]))
            if lazy_lol_penalty >= 0.7:
                risk_flags = list(dict.fromkeys([*risk_flags, "lazy_lol_prefix"]))
            if incoming_intent == "banter_challenge" and banter_fit_score < 0.35:
                risk_flags = list(dict.fromkeys([*risk_flags, "literal_banter"]))
            if relationship_boldness_score < 0.25 and any(term in normalize_text(text) for term in ("come over then", "come thru", "come through", "come round")):
                risk_flags = list(dict.fromkeys([*risk_flags, "too_bold"]))
            energy_risk_weight = 0.25 if reply_energy_mode == "minimal" else 0.95
            expressive_gap = max(0.0, 1.0 - expressive_score) if reply_energy_mode in {"expressive", "defensive_playful", "corrective", "engaged_explanation"} else 0.0
            risk_score = min(
                1.0,
                max(
                    0.22 * len(risk_flags) + (0.18 if intent in {"commitment", "conflict"} else 0.0),
                    float(assistant_scores["risk_score"]),
                    contradiction_risk,
                    repetition_penalty * 0.9,
                    max(0.0, 1.0 - banter_fit_score) * 0.35,
                    max(0.0, 1.0 - relationship_boldness_score) * (0.4 if any(term in normalize_text(text) for term in ("come over then", "come thru", "come through", "come round")) else 0.12),
                    question_pressure_penalty * 0.8,
                    repeated_question_penalty * 0.75,
                    already_answered_question_penalty,
                    semantic_mismatch_penalty,
                    1.0 if invalid_for_context else 0.0,
                    weird_punctuation_penalty,
                    emotional_absence_penalty,
                    shallow_repair_penalty,
                    stale_emotional_reply_penalty,
                    affection_missed_penalty,
                    stale_self_state_penalty,
                    bland_repair_penalty,
                    over_safe_penalty,
                    max(0.0, 1.0 - conversational_function_fit_score) * (0.9 if conversation_function != "normal" else 0.0),
                    max(0.0, 1.0 - scene_fit_score) * (0.95 if scene_type != "normal" else 0.0),
                    1.0 if forbidden_move_violated else 0.0,
                    0.85 if not required_move_satisfied and scene_type not in {"normal", "opening", "dead_conversation"} else 0.0,
                    0.95 if not unresolved_point_addressed else 0.0,
                    canned_reply_penalty,
                    stale_template_penalty,
                    deterministic_fallback_penalty * 0.6,
                    1.0 if fallback_scene_mismatch else 0.0,
                    semantic_contamination_penalty,
                    repeated_failed_pattern_penalty,
                    ignored_identity_question_penalty,
                    stale_identity_fallback_penalty,
                    identity_hallucination_risk,
                    0.0 if conversation_job == "acknowledge_repeated_reply" else ignored_question_debt_penalty,
                    0.0 if conversation_job == "acknowledge_repeated_reply" else asked_new_question_before_answering_penalty,
                    1.0 if forbidden_dialogue_move_violated else 0.0,
                    generic_prompt_penalty,
                    repeated_prompt_penalty,
                    max(0.0, 1.0 - agenda_fit_score) * (0.95 if agenda_state != "normal_flow" else 0.0),
                    1.0 if policy_violation else 0.0,
                    max(0.0, 1.0 - policy_fit_score) * (0.95 if conversation_job != "normal_reply" else 0.0),
                    quality_gate_penalty,
                    training_style_penalty * 0.55,
                    copycat_penalty,
                    1.0 if romantic_term_blocked else 0.0,
                    0.85 if repair_required and is_question_like_text(text) else 0.0,
                    hook_pressure_penalty * 0.8,
                    hook_safety_penalty,
                    dead_reply_penalty * (0.95 if hook_required else 0.45),
                    mirror_reply_penalty * (0.95 if continuation_required else 0.25),
                    filler_penalty * energy_risk_weight,
                    lazy_lol_penalty * 0.95,
                    expressive_gap * 0.35,
                ),
            )
            action_safety = max(0.0, 1.0 - risk_score)
            final_confidence = (
                0.24 * context_confidence
                + 0.22 * persona_confidence
                + 0.24 * style_confidence
                + 0.20 * semantic_confidence
                + 0.10 * action_safety
                + 0.05 * stance_consistency_score
                + 0.04 * banter_fit_score
                + 0.03 * relationship_boldness_score
                + 0.04 * question_usefulness_score
                + 0.03 * question_naturalness_score
                + 0.08 * conversation_momentum_score
                + 0.04 * hook_relevance_score
                + 0.05 * expressive_score
                + 0.08 * continuation_score
                + 0.08 * context_grounding_score
                + 0.10 * context_fit_score
                + 0.10 * emotional_fit_score
                + 0.05 * affection_response_score
                + 0.06 * hurt_repair_score
                + 0.08 * emotional_reciprocity_score
                + 0.08 * care_checkin_fit_score
                + 0.12 * conversational_function_fit_score
                + 0.16 * scene_fit_score
                + 0.08 * contextual_specificity_score
                + 0.06 * human_likeness_score
                + 0.08 * active_topic_engagement_score
                + 0.06 * direct_answer_score
                + 0.06 * repair_specificity_score
                + 0.06 * emotional_presence_score
                + 0.06 * previous_claim_explanation_score
                + 0.06 * successful_pattern_match_score
                + 0.04 * unresolved_thread_point_score
                + 0.14 * identity_answer_score
                + 0.08 * identity_specificity_score
                + 0.05 * human_scene_response_score
                + 0.05 * conversational_presence_score
                + 0.06 * directness_score
                + 0.04 * specificity_score
                + 0.18 * question_debt_answer_score
                + 0.16 * agenda_fit_score
                + 0.08 * conversation_progress_score
                + 0.18 * policy_fit_score
                + 0.10 * policy_conversation_progress_score
                + 0.08 * training_style_score
                + 0.05 * recent_life_update_fit_score
                - 0.10 * repetition_penalty
                - 0.07 * question_pressure_penalty
                - 0.06 * repeated_question_penalty
                - 0.18 * already_answered_question_penalty
                - 0.18 * semantic_mismatch_penalty
                - 0.25 * (1.0 if invalid_for_context else 0.0)
                - 0.16 * weird_punctuation_penalty
                - 0.22 * emotional_absence_penalty
                - 0.18 * shallow_repair_penalty
                - 0.24 * stale_emotional_reply_penalty
                - 0.20 * affection_missed_penalty
                - 0.22 * stale_self_state_penalty
                - 0.18 * bland_repair_penalty
                - 0.12 * over_safe_penalty
                - 0.22 * canned_reply_penalty
                - 0.24 * stale_template_penalty
                - 0.08 * deterministic_fallback_penalty
                - 0.28 * (1.0 if fallback_scene_mismatch else 0.0)
                - 0.28 * semantic_contamination_penalty
                - 0.26 * repeated_failed_pattern_penalty
                - 0.34 * ignored_identity_question_penalty
                - 0.28 * stale_identity_fallback_penalty
                - 0.40 * identity_hallucination_risk
                - 0.40 * (0.0 if conversation_job == "acknowledge_repeated_reply" else ignored_question_debt_penalty)
                - 0.34 * (0.0 if conversation_job == "acknowledge_repeated_reply" else asked_new_question_before_answering_penalty)
                - 0.34 * (1.0 if forbidden_dialogue_move_violated else 0.0)
                - 0.28 * generic_prompt_penalty
                - 0.30 * repeated_prompt_penalty
                - 0.22 * (max(0.0, 1.0 - agenda_fit_score) if agenda_state != "normal_flow" else 0.0)
                - 0.36 * (1.0 if policy_violation else 0.0)
                - 0.20 * (max(0.0, 1.0 - policy_fit_score) if conversation_job != "normal_reply" else 0.0)
                - 0.34 * quality_gate_penalty
                - 0.12 * training_style_penalty
                - 0.32 * (1.0 if forbidden_move_violated else 0.0)
                - 0.18 * (0.0 if required_move_satisfied else 1.0 if scene_type not in {"normal", "opening", "dead_conversation"} else 0.0)
                - 0.30 * (0.0 if unresolved_point_addressed else 1.0)
                - 0.18 * copycat_penalty
                - 0.12 * dead_reply_penalty
                - 0.08 * hook_pressure_penalty
                - 0.12 * hook_safety_penalty
                - 0.16 * mirror_reply_penalty
                - 0.16 * filler_penalty
                - 0.12 * lazy_lol_penalty
            )
            rationale = self._candidate_rationale(
                persona=persona,
                intent=intent,
                incoming_intent=incoming_intent,
                risk_flags=risk_flags,
                semantic_confidence=semantic_confidence,
                style_confidence=style_confidence,
                retrieved_examples=retrieved_examples,
            )
            candidate = DraftCandidate(
                text=text,
                sequence=sequence,
                candidate_kind=candidate_kinds[index] if index < len(candidate_kinds) else "most_natural",
                intent=intent,
                risk_flags=risk_flags,
                rationale=rationale,
                persona=persona,
                relationship_type=relationship_type,
                retrieved_examples_count=len(retrieved_examples),
                retrieved_examples=retrieved_payload,
                retrieval_ids_used=retrieved_ids,
                generation_attempt=generation_attempt,
                diversity_mode=diversity_mode,
                candidate_type=CANDIDATE_TYPE_BY_MODE.get(diversity_mode, "natural"),
                estimated_style_score=int(round(style_confidence * 100)),
                estimated_relevance_score=int(round(semantic_confidence * 100)),
                estimated_risk_score=int(round(risk_score * 100)),
                provider=str(provider_meta.get("provider") or "deterministic"),
                model=str(provider_meta.get("model") or "fallback"),
                latency_ms=provider_meta.get("latency_ms") if isinstance(provider_meta.get("latency_ms"), int) else None,
                external_api_used=bool(provider_meta.get("external_api_used")),
                external_api_blocked=bool(provider_meta.get("external_api_blocked")),
                provider_configured=bool(provider_meta.get("provider_configured")),
                assistant_likeness_score=float(assistant_scores["assistant_likeness_score"]),
                naturalness_score=float(assistant_scores["naturalness_score"]),
                style_score=round(style_confidence, 3),
                risk_score=round(risk_score, 3),
                stance_consistency_score=round(stance_consistency_score, 3),
                repetition_penalty=round(repetition_penalty, 3),
                banter_fit_score=round(banter_fit_score, 3),
                relationship_boldness_score=round(relationship_boldness_score, 3),
                question_mode=question_mode,
                question_reason=question_reason,
                question_usefulness_score=round(question_usefulness_score, 3),
                question_naturalness_score=round(question_naturalness_score, 3),
                question_pressure_penalty=round(question_pressure_penalty, 3),
                repeated_question_penalty=round(repeated_question_penalty, 3),
                recent_bot_questions_count=recent_bot_questions_count,
                hook_mode=hook_mode,
                hook_reason=hook_reason,
                hook_required=hook_required,
                hook_presence_score=round(hook_presence_score, 3),
                hook_relevance_score=round(hook_relevance_score, 3),
                hook_naturalness_score=round(hook_naturalness_score, 3),
                hook_pressure_penalty=round(hook_pressure_penalty, 3),
                hook_safety_penalty=round(hook_safety_penalty, 3),
                conversation_momentum_score=round(conversation_momentum_score, 3),
                dead_reply_penalty=round(dead_reply_penalty, 3),
                invalid_provider_output=invalid_provider_output,
                fallback_used=fallback_used,
                low_information_incoming=low_information_incoming,
                continuation_required=continuation_required,
                continuation_score=round(continuation_score, 3),
                mirror_reply_penalty=round(mirror_reply_penalty, 3),
                reply_energy_mode=reply_energy_mode,
                energy_reason=energy_reason,
                filler_penalty=round(filler_penalty, 3),
                expressive_score=round(expressive_score, 3),
                criticism_detected=criticism_detected,
                lazy_lol_penalty=round(lazy_lol_penalty, 3),
                repeated_question_detected=repeated_question_detected,
                copycat_penalty=round(copycat_penalty, 3),
                copycat_detected=copycat_detected,
                repair_required=repair_required,
                escalation_detected=escalation_detected,
                romantic_term_blocked=romantic_term_blocked,
                generated_by_provider=generated_by_provider,
                provider_name=str(provider_meta.get("provider") or "deterministic"),
                provider_model=str(provider_meta.get("model") or "fallback"),
                provider_error=str(provider_error) if provider_error else None,
                provider_latency_ms=provider_meta.get("latency_ms") if isinstance(provider_meta.get("latency_ms"), int) else None,
                provider_assisted_continuation=bool(provider_meta.get("provider_assisted_continuation")),
                provider_prompt_mode=provider_prompt_mode,
                recent_user_activity=recent_user_activity,
                recent_user_mood=recent_user_mood,
                recent_user_callout=recent_user_callout,
                recently_answered_question_detected=recently_answered_question_detected,
                already_answered_question_penalty=round(already_answered_question_penalty, 3),
                repair_over_hook_applied=repair_over_hook_applied,
                hook_suppressed_reason=hook_suppressed_reason,
                semantic_mismatch_penalty=round(semantic_mismatch_penalty, 3),
                context_grounding_score=round(context_grounding_score, 3),
                context_fit_score=round(context_fit_score, 3),
                invalid_for_context=invalid_for_context,
                invalid_context_reason=invalid_context_reason,
                latest_message_meaning=latest_message_meaning,
                meaning_priority=meaning_priority,
                emotional_context_detected=emotional_context_detected,
                affection_detected=affection_detected,
                hurt_detected=hurt_detected,
                confusion_detected=confusion_detected,
                serious_callout_detected=serious_callout_detected,
                emotional_fit_score=round(emotional_fit_score, 3),
                affection_response_score=round(affection_response_score, 3),
                hurt_repair_score=round(hurt_repair_score, 3),
                shallow_repair_penalty=round(shallow_repair_penalty, 3),
                emotional_absence_penalty=round(emotional_absence_penalty, 3),
                emotional_reciprocity_score=round(emotional_reciprocity_score, 3),
                care_checkin_fit_score=round(care_checkin_fit_score, 3),
                affection_missed_penalty=round(affection_missed_penalty, 3),
                stale_emotional_reply_penalty=round(stale_emotional_reply_penalty, 3),
                emotional_request_unanswered=emotional_request_unanswered,
                scene_type=scene_type,
                scene_summary=scene_summary,
                latest_user_ask=latest_user_ask,
                latest_user_emotion=latest_user_emotion,
                social_task=social_task,
                required_reply_move=required_reply_move,
                forbidden_reply_moves=forbidden_reply_moves,
                known_recent_facts=known_recent_facts,
                recent_bot_mistakes=recent_bot_mistakes,
                unresolved_user_points=unresolved_user_points,
                previous_bot_claim=previous_bot_claim,
                previous_bot_claim_type=previous_bot_claim_type,
                latest_user_refers_to_previous_bot_claim=latest_user_refers_to_previous_bot_claim,
                explanation_required=explanation_required,
                explanation_target=explanation_target,
                active_topic=active_topic,
                topic_was_provided=topic_was_provided,
                topic_value=topic_value,
                bot_repeated_itself=bot_repeated_itself,
                user_called_out_bot=user_called_out_bot,
                affection_reciprocity_required=affection_reciprocity_required,
                care_response_required=care_response_required,
                topic_engagement_required=topic_engagement_required,
                scene_fit_score=round(scene_fit_score, 3),
                required_move_satisfied=required_move_satisfied,
                forbidden_move_violated=forbidden_move_violated,
                scene_mismatch_reason=scene_mismatch_reason,
                contextual_specificity_score=round(contextual_specificity_score, 3),
                human_likeness_score=round(human_likeness_score, 3),
                canned_reply_penalty=round(canned_reply_penalty, 3),
                stale_template_penalty=round(stale_template_penalty, 3),
                deterministic_fallback_penalty=round(deterministic_fallback_penalty, 3),
                fallback_scene_type=fallback_scene_type,
                fallback_required_move=fallback_required_move,
                fallback_scene_mismatch=fallback_scene_mismatch,
                semantic_contamination_penalty=round(semantic_contamination_penalty, 3),
                previous_claim_explanation_score=round(previous_claim_explanation_score, 3),
                thread_memory_match_score=round(thread_memory_match_score, 3),
                retrieved_thread_count=retrieved_thread_count,
                repeated_failed_pattern_penalty=round(repeated_failed_pattern_penalty, 3),
                successful_pattern_match_score=round(successful_pattern_match_score, 3),
                unresolved_thread_point_score=round(unresolved_thread_point_score, 3),
                thread_memory_used=thread_memory_used,
                identity_question_detected=identity_question_detected,
                identity_fact_used=identity_fact_used,
                identity_disclosure_allowed=identity_disclosure_allowed,
                identity_answer_score=round(identity_answer_score, 3),
                identity_hallucination_risk=round(identity_hallucination_risk, 3),
                ignored_identity_question_penalty=round(ignored_identity_question_penalty, 3),
                stale_identity_fallback_penalty=round(stale_identity_fallback_penalty, 3),
                direct_identity_answer_required=direct_identity_answer_required,
                identity_specificity_score=round(identity_specificity_score, 3),
                human_scene_response_score=round(human_scene_response_score, 3),
                conversational_presence_score=round(conversational_presence_score, 3),
                directness_score=round(directness_score, 3),
                specificity_score=round(specificity_score, 3),
                has_unanswered_user_question=has_unanswered_user_question,
                unanswered_question_type=unanswered_question_type,
                unanswered_question_text=unanswered_question_text,
                question_debt_age_turns=question_debt_age_turns,
                user_called_out_unanswered_question=user_called_out_unanswered_question,
                answer_required_now=answer_required_now,
                question_debt_answer_score=round(question_debt_answer_score, 3),
                ignored_question_debt_penalty=round(ignored_question_debt_penalty, 3),
                asked_new_question_before_answering_penalty=round(asked_new_question_before_answering_penalty, 3),
                agenda_state=agenda_state,
                next_dialogue_move=next_dialogue_move,
                bot_loop_detected=bot_loop_detected,
                anti_loop_required=anti_loop_required,
                exhausted_prompt_types=exhausted_prompt_types,
                generic_prompt_penalty=round(generic_prompt_penalty, 3),
                repeated_prompt_penalty=round(repeated_prompt_penalty, 3),
                agenda_fit_score=round(agenda_fit_score, 3),
                forbidden_dialogue_move_violated=forbidden_dialogue_move_violated,
                forbidden_dialogue_moves_violated=forbidden_dialogue_moves_violated,
                conversation_progress_score=round(conversation_progress_score, 3),
                chosen_topic=chosen_topic,
                conversation_job=conversation_job,
                policy_reason=policy_reason,
                bot_obligation=bot_obligation,
                policy_must_answer=policy_must_answer,
                policy_must_acknowledge=policy_must_acknowledge,
                policy_must_repair=policy_must_repair,
                policy_must_avoid=policy_must_avoid,
                policy_fit_score=round(policy_fit_score, 3),
                policy_must_satisfied=policy_must_satisfied,
                policy_violation=policy_violation,
                policy_violation_reasons=policy_violation_reasons,
                policy_conversation_progress_score=round(policy_conversation_progress_score, 3),
                obligation_satisfied=obligation_satisfied,
                quality_gate_penalty=round(quality_gate_penalty, 3),
                quality_gate_reasons=quality_gate_reasons,
                training_style_score=round(training_style_score, 3),
                training_style_penalty=round(training_style_penalty, 3),
                training_style_reasons=training_style_reasons,
                training_style_examples_loaded=training_style_examples_loaded,
                active_topic_engagement_score=round(active_topic_engagement_score, 3),
                direct_answer_score=round(direct_answer_score, 3),
                repair_specificity_score=round(repair_specificity_score, 3),
                emotional_presence_score=round(emotional_presence_score, 3),
                unresolved_point_addressed=unresolved_point_addressed,
                activity_grounding_suppressed=activity_grounding_suppressed,
                activity_grounding_suppressed_reason=activity_grounding_suppressed_reason,
                conversation_function=conversation_function,
                conversation_function_reason=conversation_function_reason,
                social_risk_tolerance=social_risk_tolerance,
                recent_life_update_fit_score=round(recent_life_update_fit_score, 3),
                stale_self_state_penalty=round(stale_self_state_penalty, 3),
                repeated_reply_callout_detected=repeated_reply_callout_detected,
                mocking_after_repair_detected=mocking_after_repair_detected,
                context_failure_question_detected=context_failure_question_detected,
                over_safe_penalty=round(over_safe_penalty, 3),
                bland_repair_penalty=round(bland_repair_penalty, 3),
                conversational_function_fit_score=round(conversational_function_fit_score, 3),
                selected_reason=rationale,
                auto_send_allowed=not risk_flags and final_confidence >= 0.74,
                score_breakdown=DraftScoreBreakdown(
                    context_confidence=round(context_confidence, 3),
                    persona_confidence=round(persona_confidence, 3),
                    style_confidence=round(style_confidence, 3),
                    semantic_confidence=round(semantic_confidence, 3),
                    action_safety=round(action_safety, 3),
                    risk_score=round(risk_score, 3),
                    final_confidence=round(final_confidence, 3),
                ),
            )
            critique = critique_candidate(
                candidate,
                latest_message=latest_message,
                relationship_type=relationship_type,
                incoming_intent=incoming_intent,
                style_score=style_confidence,
                relevance_score=semantic_confidence,
                explicit_flirt_allowed=explicit_flirt_allowed,
                selected_mode="auto_send",
            )
            candidate.critic_scores = critique.model_dump(mode="json")
            candidate.critic_scores["style_eval"] = style_eval
            candidate.critic_scores["assistant_likeness_score"] = candidate.assistant_likeness_score
            candidate.critic_scores["naturalness_score"] = candidate.naturalness_score
            candidate.critic_scores["style_score"] = candidate.style_score
            candidate.critic_scores["risk_score"] = candidate.risk_score
            candidate.critic_scores["stance_consistency_score"] = candidate.stance_consistency_score
            candidate.critic_scores["repetition_penalty"] = candidate.repetition_penalty
            candidate.critic_scores["banter_fit_score"] = candidate.banter_fit_score
            candidate.critic_scores["relationship_boldness_score"] = candidate.relationship_boldness_score
            candidate.critic_scores["question_mode"] = candidate.question_mode
            candidate.critic_scores["question_reason"] = candidate.question_reason
            candidate.critic_scores["question_usefulness_score"] = candidate.question_usefulness_score
            candidate.critic_scores["question_naturalness_score"] = candidate.question_naturalness_score
            candidate.critic_scores["question_pressure_penalty"] = candidate.question_pressure_penalty
            candidate.critic_scores["repeated_question_penalty"] = candidate.repeated_question_penalty
            candidate.critic_scores["recent_bot_questions_count"] = candidate.recent_bot_questions_count
            candidate.critic_scores["hook_mode"] = candidate.hook_mode
            candidate.critic_scores["hook_reason"] = candidate.hook_reason
            candidate.critic_scores["hook_required"] = candidate.hook_required
            candidate.critic_scores["hook_presence_score"] = candidate.hook_presence_score
            candidate.critic_scores["hook_relevance_score"] = candidate.hook_relevance_score
            candidate.critic_scores["hook_naturalness_score"] = candidate.hook_naturalness_score
            candidate.critic_scores["hook_pressure_penalty"] = candidate.hook_pressure_penalty
            candidate.critic_scores["hook_safety_penalty"] = candidate.hook_safety_penalty
            candidate.critic_scores["conversation_momentum_score"] = candidate.conversation_momentum_score
            candidate.critic_scores["dead_reply_penalty"] = candidate.dead_reply_penalty
            candidate.critic_scores["invalid_provider_output"] = candidate.invalid_provider_output
            candidate.critic_scores["fallback_used"] = candidate.fallback_used
            candidate.critic_scores["low_information_incoming"] = candidate.low_information_incoming
            candidate.critic_scores["continuation_required"] = candidate.continuation_required
            candidate.critic_scores["continuation_score"] = candidate.continuation_score
            candidate.critic_scores["mirror_reply_penalty"] = candidate.mirror_reply_penalty
            candidate.critic_scores["reply_energy_mode"] = candidate.reply_energy_mode
            candidate.critic_scores["energy_reason"] = candidate.energy_reason
            candidate.critic_scores["filler_penalty"] = candidate.filler_penalty
            candidate.critic_scores["expressive_score"] = candidate.expressive_score
            candidate.critic_scores["criticism_detected"] = candidate.criticism_detected
            candidate.critic_scores["lazy_lol_penalty"] = candidate.lazy_lol_penalty
            candidate.critic_scores["repeated_question_detected"] = candidate.repeated_question_detected
            candidate.critic_scores["copycat_penalty"] = candidate.copycat_penalty
            candidate.critic_scores["copycat_detected"] = candidate.copycat_detected
            candidate.critic_scores["repair_required"] = candidate.repair_required
            candidate.critic_scores["escalation_detected"] = candidate.escalation_detected
            candidate.critic_scores["romantic_term_blocked"] = candidate.romantic_term_blocked
            candidate.critic_scores["generated_by_provider"] = candidate.generated_by_provider
            candidate.critic_scores["provider_name"] = candidate.provider_name
            candidate.critic_scores["provider_model"] = candidate.provider_model
            candidate.critic_scores["provider_error"] = candidate.provider_error
            candidate.critic_scores["provider_latency_ms"] = candidate.provider_latency_ms
            candidate.critic_scores["provider_assisted_continuation"] = candidate.provider_assisted_continuation
            candidate.critic_scores["provider_prompt_mode"] = candidate.provider_prompt_mode
            candidate.critic_scores["recent_user_activity"] = candidate.recent_user_activity
            candidate.critic_scores["recent_user_mood"] = candidate.recent_user_mood
            candidate.critic_scores["recent_user_callout"] = candidate.recent_user_callout
            candidate.critic_scores["recently_answered_question_detected"] = candidate.recently_answered_question_detected
            candidate.critic_scores["already_answered_question_penalty"] = candidate.already_answered_question_penalty
            candidate.critic_scores["repair_over_hook_applied"] = candidate.repair_over_hook_applied
            candidate.critic_scores["hook_suppressed_reason"] = candidate.hook_suppressed_reason
            candidate.critic_scores["semantic_mismatch_penalty"] = candidate.semantic_mismatch_penalty
            candidate.critic_scores["context_grounding_score"] = candidate.context_grounding_score
            candidate.critic_scores["context_fit_score"] = candidate.context_fit_score
            candidate.critic_scores["invalid_for_context"] = candidate.invalid_for_context
            candidate.critic_scores["invalid_context_reason"] = candidate.invalid_context_reason
            candidate.critic_scores["latest_message_meaning"] = candidate.latest_message_meaning
            candidate.critic_scores["meaning_priority"] = candidate.meaning_priority
            candidate.critic_scores["emotional_context_detected"] = candidate.emotional_context_detected
            candidate.critic_scores["affection_detected"] = candidate.affection_detected
            candidate.critic_scores["hurt_detected"] = candidate.hurt_detected
            candidate.critic_scores["confusion_detected"] = candidate.confusion_detected
            candidate.critic_scores["serious_callout_detected"] = candidate.serious_callout_detected
            candidate.critic_scores["emotional_fit_score"] = candidate.emotional_fit_score
            candidate.critic_scores["affection_response_score"] = candidate.affection_response_score
            candidate.critic_scores["hurt_repair_score"] = candidate.hurt_repair_score
            candidate.critic_scores["shallow_repair_penalty"] = candidate.shallow_repair_penalty
            candidate.critic_scores["emotional_absence_penalty"] = candidate.emotional_absence_penalty
            candidate.critic_scores["emotional_reciprocity_score"] = candidate.emotional_reciprocity_score
            candidate.critic_scores["care_checkin_fit_score"] = candidate.care_checkin_fit_score
            candidate.critic_scores["affection_missed_penalty"] = candidate.affection_missed_penalty
            candidate.critic_scores["stale_emotional_reply_penalty"] = candidate.stale_emotional_reply_penalty
            candidate.critic_scores["emotional_request_unanswered"] = candidate.emotional_request_unanswered
            candidate.critic_scores["scene_type"] = candidate.scene_type
            candidate.critic_scores["scene_summary"] = candidate.scene_summary
            candidate.critic_scores["required_reply_move"] = candidate.required_reply_move
            candidate.critic_scores["forbidden_reply_moves"] = candidate.forbidden_reply_moves
            candidate.critic_scores["previous_bot_claim"] = candidate.previous_bot_claim
            candidate.critic_scores["previous_bot_claim_type"] = candidate.previous_bot_claim_type
            candidate.critic_scores["latest_user_refers_to_previous_bot_claim"] = candidate.latest_user_refers_to_previous_bot_claim
            candidate.critic_scores["explanation_required"] = candidate.explanation_required
            candidate.critic_scores["explanation_target"] = candidate.explanation_target
            candidate.critic_scores["active_topic"] = candidate.active_topic
            candidate.critic_scores["topic_was_provided"] = candidate.topic_was_provided
            candidate.critic_scores["topic_value"] = candidate.topic_value
            candidate.critic_scores["affection_reciprocity_required"] = candidate.affection_reciprocity_required
            candidate.critic_scores["care_response_required"] = candidate.care_response_required
            candidate.critic_scores["scene_fit_score"] = candidate.scene_fit_score
            candidate.critic_scores["required_move_satisfied"] = candidate.required_move_satisfied
            candidate.critic_scores["forbidden_move_violated"] = candidate.forbidden_move_violated
            candidate.critic_scores["scene_mismatch_reason"] = candidate.scene_mismatch_reason
            candidate.critic_scores["contextual_specificity_score"] = candidate.contextual_specificity_score
            candidate.critic_scores["human_likeness_score"] = candidate.human_likeness_score
            candidate.critic_scores["canned_reply_penalty"] = candidate.canned_reply_penalty
            candidate.critic_scores["stale_template_penalty"] = candidate.stale_template_penalty
            candidate.critic_scores["deterministic_fallback_penalty"] = candidate.deterministic_fallback_penalty
            candidate.critic_scores["fallback_scene_type"] = candidate.fallback_scene_type
            candidate.critic_scores["fallback_required_move"] = candidate.fallback_required_move
            candidate.critic_scores["fallback_scene_mismatch"] = candidate.fallback_scene_mismatch
            candidate.critic_scores["semantic_contamination_penalty"] = candidate.semantic_contamination_penalty
            candidate.critic_scores["previous_claim_explanation_score"] = candidate.previous_claim_explanation_score
            candidate.critic_scores["thread_memory_match_score"] = candidate.thread_memory_match_score
            candidate.critic_scores["retrieved_thread_count"] = candidate.retrieved_thread_count
            candidate.critic_scores["repeated_failed_pattern_penalty"] = candidate.repeated_failed_pattern_penalty
            candidate.critic_scores["successful_pattern_match_score"] = candidate.successful_pattern_match_score
            candidate.critic_scores["unresolved_thread_point_score"] = candidate.unresolved_thread_point_score
            candidate.critic_scores["thread_memory_used"] = candidate.thread_memory_used
            candidate.critic_scores["identity_question_detected"] = candidate.identity_question_detected
            candidate.critic_scores["identity_fact_used"] = candidate.identity_fact_used
            candidate.critic_scores["identity_disclosure_allowed"] = candidate.identity_disclosure_allowed
            candidate.critic_scores["identity_answer_score"] = candidate.identity_answer_score
            candidate.critic_scores["identity_hallucination_risk"] = candidate.identity_hallucination_risk
            candidate.critic_scores["ignored_identity_question_penalty"] = candidate.ignored_identity_question_penalty
            candidate.critic_scores["stale_identity_fallback_penalty"] = candidate.stale_identity_fallback_penalty
            candidate.critic_scores["direct_identity_answer_required"] = candidate.direct_identity_answer_required
            candidate.critic_scores["identity_specificity_score"] = candidate.identity_specificity_score
            candidate.critic_scores["human_scene_response_score"] = candidate.human_scene_response_score
            candidate.critic_scores["conversational_presence_score"] = candidate.conversational_presence_score
            candidate.critic_scores["directness_score"] = candidate.directness_score
            candidate.critic_scores["specificity_score"] = candidate.specificity_score
            candidate.critic_scores["has_unanswered_user_question"] = candidate.has_unanswered_user_question
            candidate.critic_scores["unanswered_question_type"] = candidate.unanswered_question_type
            candidate.critic_scores["unanswered_question_text"] = candidate.unanswered_question_text
            candidate.critic_scores["question_debt_age_turns"] = candidate.question_debt_age_turns
            candidate.critic_scores["user_called_out_unanswered_question"] = candidate.user_called_out_unanswered_question
            candidate.critic_scores["answer_required_now"] = candidate.answer_required_now
            candidate.critic_scores["question_debt_answer_score"] = candidate.question_debt_answer_score
            candidate.critic_scores["ignored_question_debt_penalty"] = candidate.ignored_question_debt_penalty
            candidate.critic_scores["asked_new_question_before_answering_penalty"] = candidate.asked_new_question_before_answering_penalty
            candidate.critic_scores["agenda_state"] = candidate.agenda_state
            candidate.critic_scores["next_dialogue_move"] = candidate.next_dialogue_move
            candidate.critic_scores["bot_loop_detected"] = candidate.bot_loop_detected
            candidate.critic_scores["anti_loop_required"] = candidate.anti_loop_required
            candidate.critic_scores["exhausted_prompt_types"] = candidate.exhausted_prompt_types
            candidate.critic_scores["generic_prompt_penalty"] = candidate.generic_prompt_penalty
            candidate.critic_scores["repeated_prompt_penalty"] = candidate.repeated_prompt_penalty
            candidate.critic_scores["agenda_fit_score"] = candidate.agenda_fit_score
            candidate.critic_scores["forbidden_dialogue_move_violated"] = candidate.forbidden_dialogue_move_violated
            candidate.critic_scores["forbidden_dialogue_moves_violated"] = candidate.forbidden_dialogue_moves_violated
            candidate.critic_scores["conversation_progress_score"] = candidate.conversation_progress_score
            candidate.critic_scores["chosen_topic"] = candidate.chosen_topic
            candidate.critic_scores["conversation_job"] = candidate.conversation_job
            candidate.critic_scores["policy_reason"] = candidate.policy_reason
            candidate.critic_scores["bot_obligation"] = candidate.bot_obligation
            candidate.critic_scores["policy_must_answer"] = candidate.policy_must_answer
            candidate.critic_scores["policy_must_acknowledge"] = candidate.policy_must_acknowledge
            candidate.critic_scores["policy_must_repair"] = candidate.policy_must_repair
            candidate.critic_scores["policy_must_avoid"] = candidate.policy_must_avoid
            candidate.critic_scores["policy_fit_score"] = candidate.policy_fit_score
            candidate.critic_scores["policy_must_satisfied"] = candidate.policy_must_satisfied
            candidate.critic_scores["policy_violation"] = candidate.policy_violation
            candidate.critic_scores["policy_violation_reasons"] = candidate.policy_violation_reasons
            candidate.critic_scores["policy_conversation_progress_score"] = candidate.policy_conversation_progress_score
            candidate.critic_scores["obligation_satisfied"] = candidate.obligation_satisfied
            candidate.critic_scores["quality_gate_penalty"] = candidate.quality_gate_penalty
            candidate.critic_scores["quality_gate_reasons"] = candidate.quality_gate_reasons
            candidate.critic_scores["training_style_score"] = candidate.training_style_score
            candidate.critic_scores["training_style_penalty"] = candidate.training_style_penalty
            candidate.critic_scores["training_style_reasons"] = candidate.training_style_reasons
            candidate.critic_scores["training_style_examples_loaded"] = candidate.training_style_examples_loaded
            candidate.critic_scores["active_topic_engagement_score"] = candidate.active_topic_engagement_score
            candidate.critic_scores["direct_answer_score"] = candidate.direct_answer_score
            candidate.critic_scores["repair_specificity_score"] = candidate.repair_specificity_score
            candidate.critic_scores["emotional_presence_score"] = candidate.emotional_presence_score
            candidate.critic_scores["unresolved_point_addressed"] = candidate.unresolved_point_addressed
            candidate.critic_scores["activity_grounding_suppressed"] = candidate.activity_grounding_suppressed
            candidate.critic_scores["activity_grounding_suppressed_reason"] = candidate.activity_grounding_suppressed_reason
            candidate.critic_scores["conversation_function"] = candidate.conversation_function
            candidate.critic_scores["conversation_function_reason"] = candidate.conversation_function_reason
            candidate.critic_scores["social_risk_tolerance"] = candidate.social_risk_tolerance
            candidate.critic_scores["recent_life_update_fit_score"] = candidate.recent_life_update_fit_score
            candidate.critic_scores["stale_self_state_penalty"] = candidate.stale_self_state_penalty
            candidate.critic_scores["repeated_reply_callout_detected"] = candidate.repeated_reply_callout_detected
            candidate.critic_scores["mocking_after_repair_detected"] = candidate.mocking_after_repair_detected
            candidate.critic_scores["context_failure_question_detected"] = candidate.context_failure_question_detected
            candidate.critic_scores["over_safe_penalty"] = candidate.over_safe_penalty
            candidate.critic_scores["bland_repair_penalty"] = candidate.bland_repair_penalty
            candidate.critic_scores["conversational_function_fit_score"] = candidate.conversational_function_fit_score
            candidate.critic_scores["selected_reason"] = candidate.selected_reason
            candidate.critic_scores["provider"] = candidate.provider
            candidate.critic_scores["model"] = candidate.model
            candidate.critic_scores["latency_ms"] = candidate.latency_ms
            candidate.critic_scores["external_api_used"] = candidate.external_api_used
            candidate.critic_scores["external_api_blocked"] = candidate.external_api_blocked
            if provider_meta.get("blocked_reason"):
                candidate.critic_scores["blocked_reason"] = provider_meta.get("blocked_reason")
            candidate.final_decision = critique.final_decision
            candidate.blocked_reason = critique.blocked_reason
            candidate.auto_send_allowed = (
                critique.final_decision == "send"
                and contradiction_risk < 1.0
                and repetition_penalty < 0.8
                and mirror_reply_penalty < 0.7
                and filler_penalty < 0.7
                and lazy_lol_penalty < 0.7
                and not copycat_detected
                and not repeated_question_detected
                and already_answered_question_penalty < 0.7
                and semantic_mismatch_penalty < 0.7
                and not invalid_for_context
                and emotional_absence_penalty < 0.7
                and shallow_repair_penalty < 0.7
                and stale_emotional_reply_penalty < 0.7
                and affection_missed_penalty < 0.7
                and not emotional_request_unanswered
                and stale_self_state_penalty < 0.7
                and bland_repair_penalty < 0.7
                and over_safe_penalty < 0.7
                and scene_fit_score >= 0.35
                and not forbidden_move_violated
                and unresolved_point_addressed
                and canned_reply_penalty < 0.7
                and stale_template_penalty < 0.7
                and not fallback_scene_mismatch
                and semantic_contamination_penalty < 0.7
                and repeated_failed_pattern_penalty < 0.7
                and ignored_identity_question_penalty < 0.7
                and stale_identity_fallback_penalty < 0.7
                and identity_hallucination_risk < 0.7
                and (conversation_job == "acknowledge_repeated_reply" or ignored_question_debt_penalty < 0.7)
                and (conversation_job == "acknowledge_repeated_reply" or asked_new_question_before_answering_penalty < 0.7)
                and not forbidden_dialogue_move_violated
                and not policy_violation
                and (policy_must_satisfied or conversation_job == "normal_reply")
                and obligation_satisfied
                and not romantic_term_blocked
            )
            candidate.why_this_matches = rationale
            if critique.blocked_reason:
                candidate.risk_flags = list(dict.fromkeys([*candidate.risk_flags, *critique.blocked_reason.split(", ")]))
            if contradiction_risk >= 1.0 and self._candidate_stance(text) != "correcting":
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "contradiction_with_recent_bot_stance"
            if copycat_detected:
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "copycat_reply"
            if continuation_required and mirror_reply_penalty >= 0.7:
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "mirror_reply"
            if repeated_question_detected and (repair_required or continuation_required):
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "repeated_question"
            if already_answered_question_penalty >= 0.7:
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "already_answered_question"
            if semantic_mismatch_penalty >= 0.7:
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "semantic_mismatch"
            if invalid_for_context:
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or invalid_context_reason or "invalid_for_context"
            if emotional_absence_penalty >= 0.7:
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "emotional_absence"
            if shallow_repair_penalty >= 0.7:
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "shallow_repair"
            if stale_emotional_reply_penalty >= 0.7:
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "stale_emotional_reply"
            if affection_missed_penalty >= 0.7:
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "affection_missed"
            if stale_self_state_penalty >= 0.7:
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "stale_self_state"
            if bland_repair_penalty >= 0.7:
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "bland_repair"
            if over_safe_penalty >= 0.7:
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "over_safe"
            if forbidden_move_violated:
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "scene_forbidden_move"
            if not required_move_satisfied and scene_type not in {"normal", "opening", "dead_conversation"}:
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "scene_required_move_missing"
            if not unresolved_point_addressed:
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "unresolved_user_point_ignored"
            if canned_reply_penalty >= 0.7 and scene_type not in {"current_activity_exchange", "opening"}:
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "canned_reply"
            if stale_template_penalty >= 0.7:
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "stale_template"
            if fallback_scene_mismatch:
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "fallback_scene_mismatch"
            if semantic_contamination_penalty >= 0.7:
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "semantic_contamination"
            if repeated_failed_pattern_penalty >= 0.7:
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "repeated_failed_thread_pattern"
            if ignored_identity_question_penalty >= 0.7:
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "ignored_identity_question"
            if stale_identity_fallback_penalty >= 0.7:
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "stale_identity_fallback"
            if identity_hallucination_risk >= 0.7:
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "identity_hallucination_risk"
            if ignored_question_debt_penalty >= 0.7 and conversation_job != "acknowledge_repeated_reply":
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "ignored_question_debt"
            if asked_new_question_before_answering_penalty >= 0.7 and conversation_job != "acknowledge_repeated_reply":
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "asked_new_question_before_answering"
            if romantic_term_blocked:
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "romantic_term_blocked"
            if forbidden_dialogue_move_violated:
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "agenda_forbidden_move"
            if policy_violation:
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "conversation_policy_violation"
            if not policy_must_satisfied and conversation_job != "normal_reply":
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "conversation_policy_must_missing"
            if not obligation_satisfied:
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "reply_obligation_unsatisfied"
            if self._low_quality_clarification_reply(text, relationship_type=relationship_type):
                candidate.final_decision = "reject"
                candidate.blocked_reason = candidate.blocked_reason or "low_quality_generic_clarification"
                candidate.risk_flags = list(dict.fromkeys([*candidate.risk_flags, "low_quality_generic_clarification"]))
            if (
                candidate.final_decision == "reject"
                and conversation_job in {"react_to_story", "answer_care_check", "repair_after_weird_or_dry_reply", "progress_conversation", "acknowledge_repeated_reply", "answer_affection", "answer_reciprocal_activity", "playful_acknowledge"}
                and policy_fit_score >= 0.8
                and not policy_violation
                and policy_must_satisfied
                and obligation_satisfied
                and not forbidden_dialogue_move_violated
                and candidate.blocked_reason
                and all(reason in {"unknown_never_auto_send", "risk_score_above_25", "style_match_below_85", "relevance_below_90", "does_not_answer"} for reason in candidate.blocked_reason.split(", "))
            ):
                candidate.final_decision = "review"
                candidate.blocked_reason = None
            candidates.append(candidate)
        candidates.sort(
            key=lambda candidate: (
                candidate.auto_send_allowed,
                candidate.final_decision == "draft_only",
                candidate.score_breakdown.final_confidence,
                candidate.critic_scores.get("style_eval", {}).get("overall_score", 0),
                candidate.critic_scores.get("stance_consistency_score", 0),
                candidate.critic_scores.get("expressive_score", 0),
                candidate.critic_scores.get("conversation_momentum_score", 0),
                candidate.critic_scores.get("continuation_score", 0),
                candidate.critic_scores.get("context_grounding_score", 0),
                candidate.critic_scores.get("context_fit_score", 0),
                candidate.critic_scores.get("emotional_fit_score", 0),
                candidate.critic_scores.get("affection_response_score", 0),
                candidate.critic_scores.get("hurt_repair_score", 0),
                candidate.critic_scores.get("emotional_reciprocity_score", 0),
                candidate.critic_scores.get("care_checkin_fit_score", 0),
                candidate.critic_scores.get("conversational_function_fit_score", 0),
                candidate.critic_scores.get("scene_fit_score", 0),
                candidate.critic_scores.get("contextual_specificity_score", 0),
                candidate.critic_scores.get("human_likeness_score", 0),
                candidate.critic_scores.get("active_topic_engagement_score", 0),
                candidate.critic_scores.get("direct_answer_score", 0),
                candidate.critic_scores.get("repair_specificity_score", 0),
                candidate.critic_scores.get("emotional_presence_score", 0),
                candidate.critic_scores.get("previous_claim_explanation_score", 0),
                candidate.critic_scores.get("successful_pattern_match_score", 0),
                candidate.critic_scores.get("unresolved_thread_point_score", 0),
                candidate.critic_scores.get("identity_answer_score", 0),
                candidate.critic_scores.get("identity_specificity_score", 0),
                candidate.critic_scores.get("human_scene_response_score", 0),
                candidate.critic_scores.get("conversational_presence_score", 0),
                candidate.critic_scores.get("directness_score", 0),
                candidate.critic_scores.get("specificity_score", 0),
                candidate.critic_scores.get("question_debt_answer_score", 0),
                candidate.critic_scores.get("policy_fit_score", 0),
                candidate.critic_scores.get("policy_conversation_progress_score", 0),
                float(bool(candidate.critic_scores.get("obligation_satisfied", True))),
                candidate.critic_scores.get("training_style_score", 0),
                float(bool(candidate.critic_scores.get("unresolved_point_addressed", True))),
                candidate.critic_scores.get("recent_life_update_fit_score", 0),
                candidate.critic_scores.get("question_usefulness_score", 0),
                -candidate.critic_scores.get("repetition_penalty", 0),
                -candidate.critic_scores.get("dead_reply_penalty", 0),
                -candidate.critic_scores.get("mirror_reply_penalty", 0),
                -candidate.critic_scores.get("copycat_penalty", 0),
                -float(bool(candidate.critic_scores.get("romantic_term_blocked", False))),
                -candidate.critic_scores.get("filler_penalty", 0),
                -candidate.critic_scores.get("lazy_lol_penalty", 0),
                -candidate.critic_scores.get("hook_safety_penalty", 0),
                -candidate.critic_scores.get("question_pressure_penalty", 0),
                -candidate.critic_scores.get("repeated_question_penalty", 0),
                -candidate.critic_scores.get("already_answered_question_penalty", 0),
                -candidate.critic_scores.get("semantic_mismatch_penalty", 0),
                -float(bool(candidate.critic_scores.get("invalid_for_context", False))),
                -candidate.critic_scores.get("emotional_absence_penalty", 0),
                -candidate.critic_scores.get("shallow_repair_penalty", 0),
                -candidate.critic_scores.get("stale_emotional_reply_penalty", 0),
                -candidate.critic_scores.get("affection_missed_penalty", 0),
                -candidate.critic_scores.get("stale_self_state_penalty", 0),
                -candidate.critic_scores.get("bland_repair_penalty", 0),
                -candidate.critic_scores.get("over_safe_penalty", 0),
                -candidate.critic_scores.get("canned_reply_penalty", 0),
                -candidate.critic_scores.get("stale_template_penalty", 0),
                -candidate.critic_scores.get("deterministic_fallback_penalty", 0),
                -float(bool(candidate.critic_scores.get("fallback_scene_mismatch", False))),
                -candidate.critic_scores.get("semantic_contamination_penalty", 0),
                -candidate.critic_scores.get("repeated_failed_pattern_penalty", 0),
                -candidate.critic_scores.get("ignored_identity_question_penalty", 0),
                -candidate.critic_scores.get("stale_identity_fallback_penalty", 0),
                -candidate.critic_scores.get("identity_hallucination_risk", 0),
                -candidate.critic_scores.get("ignored_question_debt_penalty", 0),
                -candidate.critic_scores.get("asked_new_question_before_answering_penalty", 0),
                -float(bool(candidate.critic_scores.get("policy_violation", False))),
                -candidate.critic_scores.get("quality_gate_penalty", 0),
                -candidate.critic_scores.get("training_style_penalty", 0),
                -float(bool(candidate.critic_scores.get("forbidden_dialogue_move_violated", False))),
                -float(bool(candidate.critic_scores.get("forbidden_move_violated", False))),
                -float(not bool(candidate.critic_scores.get("unresolved_point_addressed", True))),
                -candidate.score_breakdown.risk_score,
            ),
            reverse=True,
        )
        recommended_index = 0 if candidates else None
        blocked_reason = None
        if candidates and not candidates[0].auto_send_allowed:
            blocked_reason = candidates[0].blocked_reason or (
                "Auto-send blocked: " + ", ".join(candidates[0].risk_flags)
                if candidates[0].risk_flags
                else "Auto-send blocked: critic did not approve send."
            )
        return candidates, recommended_index, blocked_reason

    def _assistant_style_scores(
        self,
        *,
        latest_message: str,
        reply: str,
        relationship_type: str,
    ) -> dict[str, float]:
        normalized = normalize_text(reply)
        incoming_words = max(1, len(latest_message.split()))
        reply_words = len(reply.split())
        banned_phrases = (
            "i understand where you're coming from",
            "i understand where youre coming from",
            "that makes sense",
            "i appreciate you",
            "let me know if you need anything",
            "i'm sorry to hear that",
            "im sorry to hear that",
            "no worries at all",
            "hope you're okay",
            "hope youre okay",
            "i completely understand",
            "i'm here for you",
            "im here for you",
        )
        polished_terms = (
            "regarding",
            "concerning",
            "happy to help",
            "please let me know",
            "i appreciate",
            "certainly",
            "understandable",
            "take care",
        )
        phrase_hits = sum(1 for phrase in banned_phrases if phrase in normalized)
        polished_hits = sum(1 for phrase in polished_terms if phrase in normalized)
        sentence_count = max(1, len(re.findall(r"[.!?]+", reply)))
        punctuation_count = len(re.findall(r"[.!?,;:]", reply))
        assistant_likeness = 0.0
        assistant_likeness += min(0.55, phrase_hits * 0.32)
        assistant_likeness += min(0.28, polished_hits * 0.14)
        if reply_words > max(10, incoming_words * 2):
            assistant_likeness += 0.16
        if sentence_count >= 3 and relationship_type in {"close_friend", "casual_friend", "unknown"}:
            assistant_likeness += 0.12
        if punctuation_count >= 4:
            assistant_likeness += 0.1
        if reply and reply[0].isupper() and relationship_type in {"close_friend", "casual_friend"}:
            assistant_likeness += 0.05
        assistant_likeness = max(0.0, min(1.0, assistant_likeness))
        naturalness = max(0.0, min(1.0, 0.92 - assistant_likeness))
        risk_score = min(1.0, assistant_likeness * 0.75)
        return {
            "assistant_likeness_score": round(assistant_likeness, 3),
            "naturalness_score": round(naturalness, 3),
            "risk_score": round(risk_score, 3),
        }

    def _is_near_duplicate(
        self,
        candidate: str,
        previous_candidates: list[str],
        *,
        threshold: float = 0.82,
    ) -> bool:
        normalized = normalize_text(candidate)
        if not normalized:
            return True
        for previous in previous_candidates:
            other = normalize_text(previous)
            if not other:
                continue
            if normalized == other:
                return True
            if SequenceMatcher(None, normalized, other).ratio() > threshold:
                return True
        return False

    def _contact_persona(self, contact_name: str | None) -> str:
        if self.intelligence_store is None or not contact_name:
            return "casual_friend"
        insight = self.intelligence_store.get_contact_insight(contact_name)
        if insight is None:
            return "casual_friend"
        return insight.persona

    def _explicit_flirt_allowed(
        self,
        recent_messages: list[str],
        *,
        relationship_type: str,
        incoming_intent: str,
        contact_name: str | None = None,
    ) -> bool:
        if incoming_intent == "greeting" or not self._flirt_allowed(contact_name, relationship_type):
            return False
        text_blob = " ".join(recent_messages[-4:]).casefold()
        flirt_signals = ("baby", "babe", "cute", "pretty", "beautiful", "miss u", "miss you", "missed u", "missed you", "kiss")
        return any(term in text_blob for term in flirt_signals)

    def _draft_context_confidence(self, full_conversation: list[str]) -> float:
        count = len(full_conversation)
        if count >= 10:
            return 0.94
        if count >= 6:
            return 0.86
        if count >= 3:
            return 0.74
        return 0.58

    def _classify_reply_intent(self, latest_message: str, sequence: list[str]) -> str:
        latest = latest_message.casefold()
        reply = " ".join(sequence).casefold()
        scheduling_terms = ("when", "what time", "tmrw", "tomorrow", "later", "tonight", "meet", "meeting", "call")
        conflict_terms = ("wtf", "why are u", "why you", "pissed", "mad", "angry", "annoyed", "??")
        flirt_terms = ("baby", "babe", "cute", "pretty", "beautiful", "handsome", "come here", "miss u", "kiss")
        commitment_terms = ("ill", "i'll", "on my way", "be there", "promise", "definitely", "for sure", "i got you", "i can do")
        apology_terms = ("sorry", "my bad", "apolog")
        if any(term in reply for term in apology_terms):
            return "apology"
        if any(term in latest or term in reply for term in conflict_terms):
            return "conflict"
        if any(term in latest or term in reply for term in scheduling_terms):
            return "scheduling"
        if any(term in reply for term in commitment_terms):
            return "commitment"
        if any(term in latest or term in reply for term in flirt_terms):
            return "flirt"
        if latest.endswith("?"):
            return "informative"
        if any(term in reply for term in ("lol", "loool", "weirdo", "idiot", "stfu", "nah")):
            return "banter"
        return "acknowledgement"

    def _risk_flags(self, latest_message: str, sequence: list[str], *, persona: str, intent: str) -> list[str]:
        latest = latest_message.casefold()
        reply = " ".join(sequence).casefold()
        flags: list[str] = []
        if persona in {"family", "professional"}:
            flags.append(f"{persona}_manual_only")
        if intent in {"conflict", "apology", "commitment", "scheduling"}:
            flags.append(intent)
        if any(term in reply for term in ("ill", "i'll", "promise", "for sure", "definitely", "on my way")):
            flags.append("commitment_claim")
        if any(term in reply for term in ("sent it", "finished", "did it", "was there", "i was there", "i told")):
            flags.append("factual_claim")
        if any(char.isdigit() for char in reply) or any(term in reply for term in ("pm", "am", "clock")):
            flags.append("time_specific")
        if persona == "family" and any(term in reply for term in ("baby", "babe", "come here")):
            flags.append("persona_mismatch")
        if persona == "professional" and any(term in reply for term in ("baby", "lol", "loool", "stfu", "come here")):
            flags.append("persona_mismatch")
        if latest.count("?") >= 2 and len(reply.split()) <= 2:
            flags.append("under_contextualized")
        unique_flags: list[str] = []
        for flag in flags:
            if flag not in unique_flags:
                unique_flags.append(flag)
        return unique_flags

    def _persona_fit_score(self, sequence: list[str], *, persona: str) -> float:
        reply = " ".join(sequence).casefold()
        flirt_terms = ("baby", "babe", "cute", "pretty", "beautiful", "handsome", "come here", "kiss", "miss u")
        banter_terms = ("lol", "loool", "lmao", "weirdo", "idiot", "stfu", "nah", "bro")
        formal_terms = ("regarding", "please", "confirm", "available", "appreciate")
        has_flirt = any(term in reply for term in flirt_terms)
        has_banter = any(term in reply for term in banter_terms)
        has_formal = any(term in reply for term in formal_terms)
        if persona == "flirty":
            score = 0.7 + (0.18 if has_flirt or has_banter else 0.0) - (0.18 if has_formal else 0.0)
        elif persona == "close_friend":
            score = 0.72 + (0.16 if has_banter else 0.0) - (0.16 if has_formal else 0.0)
        elif persona == "professional":
            score = 0.82 - (0.28 if has_flirt or has_banter else 0.0)
        elif persona == "family":
            score = 0.8 - (0.32 if has_flirt else 0.0)
        else:
            score = 0.74 + (0.08 if has_banter else 0.0) - (0.1 if has_formal else 0.0)
        return max(0.0, min(score, 1.0))

    def _style_match_score(self, sequence: list[str]) -> float:
        reply = " ".join(sequence)
        words = re.findall(r"[A-Za-z0-9']+", reply)
        avg_words = float(getattr(self.style_profile, "average_message_length", 26.0))
        lower_bias = float(getattr(self.style_profile, "lowercase_ratio", 0.7))
        score = 0.62
        if lower_bias > 0.8 and reply == reply.lower():
            score += 0.12
        if "." not in reply:
            score += 0.08
        if words:
            word_count = len(words)
            if word_count <= 14:
                score += 0.1
            if avg_words > 0:
                target_word_window = max(4.0, min(avg_words / 3.0, 14.0))
                if abs(word_count - target_word_window) <= 4:
                    score += 0.08
        common_terms = getattr(self.style_profile, "common_terms", []) if self.style_profile is not None else []
        if any(term in reply.casefold() for term in common_terms[:6]):
            score += 0.08
        if any(term in reply.casefold() for term in ("regarding", "concerning", "appreciate", "certainly")):
            score -= 0.2
        return max(0.0, min(score, 1.0))

    def _semantic_match_score(self, latest_message: str, sequence: list[str]) -> float:
        latest_tokens = self._tokenize(latest_message)
        reply_tokens = self._tokenize(" ".join(sequence))
        if not latest_tokens:
            return 0.62
        overlap = len(latest_tokens.intersection(reply_tokens))
        score = 0.56 + min(0.2, 0.06 * overlap)
        if latest_message.strip().endswith("?"):
            score += 0.1
        if any(token in latest_message.casefold() for token in ("where", "when", "why", "how", "call", "miss")):
            score += 0.08
        if any(term in " ".join(sequence).casefold() for term in ("i hear you", "let me know", "sounds good")):
            score -= 0.14
        return max(0.0, min(score, 1.0))

    def _candidate_rationale(
        self,
        *,
        persona: str,
        intent: str,
        incoming_intent: str,
        risk_flags: list[str],
        semantic_confidence: float,
        style_confidence: float,
        retrieved_examples: list[RetrievedExample],
    ) -> str:
        parts = [f"persona={persona}", f"incoming_intent={incoming_intent}", f"candidate_intent={intent}"]
        if incoming_intent == "greeting":
            parts.append("answers simple greeting with a short safe reply")
        if semantic_confidence >= 0.8:
            parts.append("directly answers the latest message")
        if style_confidence >= 0.8 and incoming_intent != "greeting":
            parts.append("sounds close to your training style")
        if retrieved_examples:
            parts.append(f"uses {len(retrieved_examples[:5])} retrieved examples")
        if risk_flags:
            parts.append("manual review: " + ", ".join(risk_flags))
        else:
            parts.append("low-risk candidate")
        return "; ".join(parts)

    def _parse_reply_sequences(self, replies_raw: object) -> list[list[str]]:
        if not isinstance(replies_raw, list):
            return []
        sequences: list[list[str]] = []
        for item in replies_raw[:3]:
            sequence: list[str] = []
            if isinstance(item, list):
                sequence = [self._stylize_reply_text(str(part)) for part in item if str(part).strip()]
            else:
                raw_text = str(item).strip()
                if not raw_text:
                    continue
                parts = [part.strip() for part in re.split(r"\s*\|\|\s*|\n{2,}", raw_text) if part.strip()]
                sequence = [self._stylize_reply_text(part) for part in parts]
            sequence = [part for part in sequence if part]
            if sequence:
                sequences.append(sequence[:3])
        return sequences

    def _format_sequence_for_display(self, sequence: list[str]) -> str:
        return " / ".join(sequence)

    def _stylize_reply_text(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text.replace("\u2019", "'")).strip()
        lowered = cleaned.lower() if self.style_profile and self.style_profile.lowercase_ratio > 0.8 else cleaned
        replacements = {
            "i'm": "im",
            "i’ll": "ill",
            "i'll": "ill",
            "i’d": "id",
            "i'd": "id",
            "you're": "youre",
            "you’re": "youre",
            "that's": "thats",
            "what's": "whats",
            "it's": "its",
            "don't": "dont",
            "didn't": "didnt",
            "doesn't": "doesnt",
            "can't": "cant",
            "won't": "wont",
            "aren't": "arent",
            "isn't": "isnt",
            "haven't": "havent",
            "wouldn't": "wouldnt",
            "shouldn't": "shouldnt",
            "couldn't": "couldnt",
        }
        for source, target in replacements.items():
            lowered = lowered.replace(source, target)
        lowered = re.sub(r"[.]{2,}", "..", lowered)
        lowered = re.sub(r"\s+([?!,])", r"\1", lowered)
        lowered = re.sub(r"[.]+$", "", lowered)
        return lowered.strip()

    def _conversation_dynamic_hint(self, full_conversation: list[str]) -> str:
        latest = full_conversation[-8:]
        other_count = sum(1 for line in latest if line.startswith("[OTHER]:"))
        me_count = sum(1 for line in latest if line.startswith("[ME]:"))
        flirt_terms = ("baby", "cute", "pretty", "miss", "love", "come here", "handsome", "beautiful")
        flirt_score = sum(1 for line in latest if any(term in line.casefold() for term in flirt_terms))
        if flirt_score >= 2:
            return "the convo may already have warmth in it, but do not escalate beyond the visible messages"
        if other_count > me_count:
            return "the other person is investing more right now, so answer clearly without inventing extra context"
        return "keep it natural, short, and grounded in the latest message"

    def _multi_bubble_guidance(self, messages: list[str], *, incoming_intent: str = "other") -> str:
        latest = self._normalize_reply(messages[-1]) if messages else ""
        if incoming_intent == "greeting":
            return "keep each reply to one short bubble"
        if any(token in latest for token in ("why", "where", "call", "miss", "love", "wyd")):
            return "each reply can be 1 to 3 short bubbles if that feels more human; if using multiple bubbles, keep them quick and separate"
        return "each reply can be 1 or 2 short bubbles if that feels more natural; dont force multiple if one line hits"

    def _redact_prompt_preview(self, prompt: str) -> str:
        preview = prompt
        for needle in getattr(self.style_profile, "affectionate_terms", [])[:8]:
            preview = preview.replace(needle, "[redacted]")
        preview = re.sub(r"\b\d{3,}\b", "[number]", preview)
        return preview[:3000]

    def _suggest_photos(self, recent_messages: list[str]) -> list[ApprovedPhoto]:
        if not self.photo_index:
            return []
        tokens = self._tokenize(" ".join(recent_messages))
        scored: list[tuple[int, ApprovedPhoto]] = []
        for photo in self.photo_index:
            score = len(tokens.intersection({tag.lower() for tag in photo.tags}))
            if score > 0:
                scored.append((score, photo))
        if not scored:
            return self.photo_index[:3]
        scored.sort(key=lambda item: (-item[0], item[1].photo_id))
        return [photo for _, photo in scored[:3]]

    def _extract_topic(self, message: str) -> str:
        clean = re.sub(r"\s+", " ", message).strip()
        if clean.endswith("?"):
            return f"I saw your question about \"{clean}\"."
        return f"Regarding \"{clean}\", I can handle it."

    def _latest_exchange_context(self, full_conversation: list[str]) -> str:
        latest_lines = full_conversation[-6:]
        latest_other = next((line for line in reversed(latest_lines) if line.startswith("[OTHER]:")), "")
        latest_me = next((line for line in reversed(latest_lines) if line.startswith("[ME]:")), "")
        if latest_other and latest_me:
            return f"{latest_me}\n{latest_other}"
        return "\n".join(latest_lines[-2:]) if latest_lines else ""

    def _tokenize(self, text: str) -> set[str]:
        return {token.lower() for token in re.findall(r"[A-Za-z0-9']+", text)}
