from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from libs.drafting import ApprovedPhoto, DraftCandidate
from libs.perception import PerceptionTimings
from libs.planners import PlannerDecision
from libs.screen_states import ScreenClassification
from libs.validation_models import ComposeValidationResult, SendAndReadResult


class LoopMetrics(BaseModel):
    timestamp: datetime
    loop_time_ms: float = 0.0
    screenshot_time_ms: float = 0.0
    ocr_time_ms: float = 0.0
    classification_time_ms: float = 0.0
    planner_time_ms: float = 0.0
    execution_time_ms: float = 0.0
    detailed_timings_ms: dict[str, float] = Field(default_factory=dict)
    screen: str
    confidence: float

    @classmethod
    def from_perception(
        cls,
        timestamp: datetime,
        perception_timings: PerceptionTimings,
        planner_time_ms: float,
        execution_time_ms: float,
        loop_time_ms: float,
        screen: str,
        confidence: float,
    ) -> "LoopMetrics":
        return cls(
            timestamp=timestamp,
            loop_time_ms=loop_time_ms,
            screenshot_time_ms=perception_timings.screenshot_time_ms,
            ocr_time_ms=perception_timings.ocr_time_ms,
            classification_time_ms=perception_timings.classification_time_ms,
            planner_time_ms=planner_time_ms,
            execution_time_ms=execution_time_ms,
            detailed_timings_ms=dict(perception_timings.detailed),
            screen=screen,
            confidence=confidence,
        )


class MetricsSnapshot(BaseModel):
    total_iterations: int = 0
    halted_iterations: int = 0
    last: LoopMetrics | None = None
    rolling_average_ms: float = 0.0
    compose_validation: dict[str, object] = Field(default_factory=dict)


class AutomationMode(BaseModel):
    enabled: bool = False
    mode: str = "review"  # "review", "auto-draft", "auto-send"
    confidence_threshold: float = 0.85
    auto_send_enabled: bool = False


class BlacklistUpdate(BaseModel):
    action: str  # "add" or "remove"
    contact_name: str | None = None
    contact_number: str | None = None


class AutomationModeUpdate(BaseModel):
    mode: str  # "review", "auto-draft", "auto-send"
    confidence: float | None = None


class ThreadOpenRequest(BaseModel):
    contact_name: str


class QueueItem(BaseModel):
    contact_name: str
    contact_number: str | None = None
    preview: str
    unread: bool = True
    timestamp: datetime


class ThreadContext(BaseModel):
    contact_name: str
    recent_messages: list[str] = Field(default_factory=list)  # Simple display format
    full_conversation: list[dict[str, str]] = Field(default_factory=list)  # [{"speaker": "me"|"other", "text": "message"}, ...]
    message_count: int = 0
    last_message_time: datetime | None = None
    scrolls_performed: int = 0
    stopped_reason: str | None = None


class ControllerState(BaseModel):
    captured_at: datetime
    classification: ScreenClassification
    planner_decision: PlannerDecision
    summary: str
    compose_text: str | None = None
    reply_suggestions: list[str] = Field(default_factory=list)
    reply_sequences: list[list[str]] = Field(default_factory=list)
    draft_candidates: list[DraftCandidate] = Field(default_factory=list)
    recommended_reply_index: int | None = None
    auto_send_blocked_reason: str | None = None
    relationship_type: str = "unknown"
    intent_type: str = "other"
    retrieved_examples_count: int = 0
    retrieved_examples: list[dict[str, object]] = Field(default_factory=list)
    final_decision: str = "review"
    blocked_reason: str | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    prompt_preview: str | None = None
    context_messages_count: int = 0
    scrolls_performed: int = 0
    context_stopped_reason: str | None = None
    approved_photo_suggestions: list[ApprovedPhoto] = Field(default_factory=list)
    halted: bool = False
    halt_reason: str | None = None
    emergency_stop: bool = False
    screenshot_path: str
    debug_image_path: str | None = None
    metrics: LoopMetrics | None = None
    last_action: str | None = None
    last_action_result: str | None = None
    verification_errors: list[str] = Field(default_factory=list)
    failure_category: str | None = None
    compose_validation: ComposeValidationResult | None = None
    send_and_read: SendAndReadResult | None = None
    # Automation state
    automation_mode: AutomationMode = Field(default_factory=AutomationMode)
    unread_queue: list[QueueItem] = Field(default_factory=list)
    thread_context: ThreadContext | None = None
    blacklisted: bool = False


class ApprovalRequest(BaseModel):
    suggestion_index: int | None = None
    photo_id: str | None = None


class ComposeValidationRequest(BaseModel):
    text: str


class SendAndReadRequest(BaseModel):
    text: str
    wait_timeout_seconds: float = Field(default=90.0, gt=0.0, le=300.0)


class FeedbackRequest(BaseModel):
    incoming: str
    context: list[str] = Field(default_factory=list)
    relationship_type: str
    bad_ai_reply: str = ""
    user_final_reply: str
    reason_bad: str = "other"
    selected_mode: str = "review"


class RegenerateRequest(BaseModel):
    contact_name: str | None = None
    incoming: str
    context: list[str] = Field(default_factory=list)
    relationship_type: str = "unknown"
    intent_type: str = "unknown"
    avoid_candidates: list[str] = Field(default_factory=list)
    diversity_mode: str = "alternative_wording"


class TrainingParameters(BaseModel):
    temperature: float = Field(default=0.7, ge=0.0, le=1.2)
    max_reply_length: int = Field(default=20, ge=3, le=80)
    slang_level: float = Field(default=0.6, ge=0.0, le=1.0)
    formality: float = Field(default=0.2, ge=0.0, le=1.0)
    directness: float = Field(default=0.7, ge=0.0, le=1.0)
    warmth: float = Field(default=0.5, ge=0.0, le=1.0)
    banter: float = Field(default=0.4, ge=0.0, le=1.0)
    emoji_allowed: bool = False
    question_bias: float = Field(default=0.5, ge=0.0, le=1.0)
    risk_tolerance: float = Field(default=0.0, ge=0.0, le=1.0)


class TrainingChatRequest(BaseModel):
    incoming: str
    context: list[str] = Field(default_factory=list)
    contact_name: str | None = None
    relationship_type: str = "unknown"
    intent_type: str = "auto"
    diversity_mode: str = "natural"
    avoid_candidates: list[str] = Field(default_factory=list)
    parameters: TrainingParameters = Field(default_factory=TrainingParameters)


class WhatsAppWebMessage(BaseModel):
    speaker: str = Field(default="unknown", pattern="^(me|other|unknown)$")
    text: str
    timestamp: str | None = None
    client_order: int | None = None
    message_id: str | None = None


class WhatsAppWebDraftRequest(BaseModel):
    request_id: str | None = None
    thread_id: str | None = None
    contact_name: str | None = None
    messages: list[WhatsAppWebMessage] = Field(default_factory=list)
    relationship_type: str = "unknown"
    intent_type: str = "auto"
    mode: str = Field(default="review", pattern="^(review|auto-review|auto-send|autonomous)$")
    diversity_mode: str = "natural"
    avoid_candidates: list[str] = Field(default_factory=list)
    parameters: TrainingParameters = Field(default_factory=TrainingParameters)


class WhatsAppWebMemoryRequest(BaseModel):
    contact_name: str | None = None
    relationship_type: str = "unknown"
    question: str
    answer: str
    messages: list[WhatsAppWebMessage] = Field(default_factory=list)
    source: str = "whatsapp_web_extension"


class CatbotChatRequest(BaseModel):
    incoming: str
    context: list[str] = Field(default_factory=list)
    contact_name: str | None = "Catbot"
    relationship_type: str = "close_friend"
    intent_type: str = "auto"
    diversity_mode: str = "natural"
    thread_id: str | None = None
    reset_thread: bool = False
    parameters: TrainingParameters = Field(default_factory=TrainingParameters)


class CatbotFeedbackRequest(BaseModel):
    session_id: str
    rating: str = Field(pattern="^(thumbs_up|thumbs_down)$")
    corrected_reply: str = ""
    notes: str = ""


class ProviderCompareRequest(BaseModel):
    incoming: str
    context: list[str] = Field(default_factory=list)
    relationship_type: str = "unknown"
    intent_type: str = "auto"
    providers: list[str] = Field(default_factory=lambda: ["ollama", "gemini", "groq", "openrouter"])


class TrainingFeedbackRequest(BaseModel):
    incoming: str
    context: list[str] = Field(default_factory=list)
    contact_name: str | None = None
    relationship_type: str
    intent_type: str
    ai_reply: str = ""
    correct_reply: str = ""
    feedback_label: str
    notes: str = ""
    parameters: TrainingParameters = Field(default_factory=TrainingParameters)
    retrieved_example_ids: list[str] = Field(default_factory=list)
    save_to_corrections: bool = True
    add_to_vector_db: bool = True


class TrainingQuarantineRequest(BaseModel):
    example_id: str
    reason: str = "bad training data"


class StyleReviewApproveRequest(BaseModel):
    row_id: str
    edited_reply: str | None = None


class StyleReviewRejectRequest(BaseModel):
    row_id: str
    reason: str = "rejected from style review queue"


class TrainingLoopConfigUpdate(BaseModel):
    enabled: bool | None = None
    interval_minutes: int | None = Field(default=None, ge=5, le=1440)
    max_cases_per_run: int | None = Field(default=None, ge=1, le=1000)
    max_review_candidates_per_run: int | None = Field(default=None, ge=1, le=500)
    auto_rebuild_after_approval: bool | None = None
    auto_approve: bool | None = None
    run_on_startup: bool | None = None
    run_eval: bool | None = None
    generate_improvements: bool | None = None
    build_active_learning_queue: bool | None = None
    rebuild_vector_if_required: bool | None = None
    quiet_hours: dict[str, object] | None = None


class NotificationPayload(BaseModel):
    package_name: str
    title: str | None = None
    text: str | None = None
    posted_at: datetime | None = None
