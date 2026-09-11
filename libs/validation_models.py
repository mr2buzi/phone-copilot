from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class FailureCategory(str, Enum):
    CLASSIFIER_WRONG = "classifier_wrong"
    CLASSIFIER_LOW_CONFIDENCE = "classifier_low_confidence"
    KEYBOARD_STATE_AMBIGUOUS = "keyboard_state_ambiguous"
    KEYBOARD_STATE_WRONG = "keyboard_state_wrong"
    TAP_TARGET_INVALID = "tap_target_invalid"
    UI_NOT_IDLE = "ui_not_idle"
    PRECONDITION_FAILED = "precondition_failed"
    POSTCONDITION_TIMEOUT = "postcondition_timeout"
    POSTCONDITION_MISMATCH = "postcondition_mismatch"
    RESPONSE_TIMEOUT = "response_timeout"
    POLICY_BLOCKED = "policy_blocked"
    EXECUTION_RETRY_EXHAUSTED = "execution_retry_exhausted"


class ValidationRunSummary(BaseModel):
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    success_rate: float = 0.0
    keyboard_ambiguity_count: int = 0
    failure_counts: dict[str, int] = Field(default_factory=dict)


class ComposeValidationResult(BaseModel):
    success: bool
    validation_text: str
    observed_text: str | None = None
    reason: str
    failure_category: FailureCategory | None = None


class SendAndReadResult(BaseModel):
    success: bool
    sent_text: str
    observed_sent_text: str | None = None
    response_text: str | None = None
    reason: str
    failure_category: FailureCategory | None = None
