from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from libs.planners import ExecutionPlan, PlannerDecision
from libs.screen_states import ApprovalActionType, ScreenClassification, ScreenName
from libs.validation_models import FailureCategory


class PolicyDecision(BaseModel):
    allowed: bool
    reason: str
    failure_category: FailureCategory | None = None


class PolicyEngine:
    def __init__(self, confidence_threshold: float, approved_photos_dir: Path) -> None:
        self.confidence_threshold = confidence_threshold
        self.approved_photos_dir = approved_photos_dir.resolve()

    def evaluate_observation(
        self,
        classification: ScreenClassification,
        planner_decision: PlannerDecision,
        emergency_stop: bool,
        previous_step_failed: bool = False,
    ) -> PolicyDecision:
        if emergency_stop:
            return PolicyDecision(allowed=False, reason="Emergency stop is engaged.", failure_category=FailureCategory.POLICY_BLOCKED)
        if previous_step_failed:
            return PolicyDecision(
                allowed=False,
                reason="Previous step failed. Manual review is required.",
                failure_category=FailureCategory.POLICY_BLOCKED,
            )
        if classification.confidence < self.confidence_threshold:
            return PolicyDecision(
                allowed=False,
                reason=f"Confidence {classification.confidence:.2f} is below the threshold.",
                failure_category=FailureCategory.CLASSIFIER_LOW_CONFIDENCE,
            )
        if classification.screen in {ScreenName.UNKNOWN_SCREEN, ScreenName.PERMISSION_POPUP}:
            return PolicyDecision(
                allowed=False,
                reason=planner_decision.reason,
                failure_category=FailureCategory.POLICY_BLOCKED,
            )
        return PolicyDecision(allowed=True, reason="Observation is within policy bounds.")

    def authorize_plan(
        self,
        plan: ExecutionPlan,
        classification: ScreenClassification,
        emergency_stop: bool,
        previous_step_failed: bool = False,
    ) -> PolicyDecision:
        observation = self.evaluate_observation(
            classification=classification,
            planner_decision=PlannerDecision(status="review", reason="preflight"),
            emergency_stop=emergency_stop,
            previous_step_failed=previous_step_failed,
        )
        if not observation.allowed:
            return observation
        if classification.screen != plan.expected_prev_screen:
            return PolicyDecision(
                allowed=False,
                reason=(
                    f"State mismatch: planner expected {plan.expected_prev_screen.value}, "
                    f"detected {classification.screen.value}."
                ),
                failure_category=FailureCategory.PRECONDITION_FAILED,
            )
        if classification.keyboard_ambiguous and plan.requested_action in {
            ApprovalActionType.TYPE_DRAFT,
            ApprovalActionType.SEND_MESSAGE,
        }:
            return PolicyDecision(
                allowed=False,
                reason="Keyboard state is ambiguous. Sensitive execution is blocked.",
                failure_category=FailureCategory.KEYBOARD_STATE_AMBIGUOUS,
            )
        if plan.expected_keyboard_before is not None and classification.keyboard_visible != plan.expected_keyboard_before:
            return PolicyDecision(
                allowed=False,
                reason=(
                    f"Keyboard state mismatch: expected visible={plan.expected_keyboard_before}, "
                    f"detected visible={classification.keyboard_visible}."
                ),
                failure_category=FailureCategory.KEYBOARD_STATE_WRONG,
            )
        if plan.allowed_package_names and classification.package_name not in plan.allowed_package_names:
            return PolicyDecision(
                allowed=False,
                reason=f"Execution is only allowed in {', '.join(plan.allowed_package_names)}.",
                failure_category=FailureCategory.POLICY_BLOCKED,
            )
        if plan.requested_action == ApprovalActionType.SELECT_APPROVED_PHOTO:
            push_steps = [step for step in plan.steps if step.kind == "push_file"]
            if not push_steps:
                return PolicyDecision(
                    allowed=False,
                    reason="Approved photo plan must push a vetted file.",
                    failure_category=FailureCategory.POLICY_BLOCKED,
                )
            photo_path = Path(push_steps[0].local_path or "").resolve()
            try:
                photo_path.relative_to(self.approved_photos_dir)
            except ValueError:
                return PolicyDecision(
                    allowed=False,
                    reason="Selected photo is outside the approved photo directory.",
                    failure_category=FailureCategory.POLICY_BLOCKED,
                )
            if not photo_path.exists():
                return PolicyDecision(
                    allowed=False,
                    reason=f"Approved photo not found: {photo_path}",
                    failure_category=FailureCategory.POLICY_BLOCKED,
                )
        forbidden = {"launch_app_camera", "send_message", "add_contact", "auto_send", "camera_access"}
        if any(step.kind in forbidden for step in plan.steps):
            return PolicyDecision(
                allowed=False,
                reason="Plan contains a forbidden action.",
                failure_category=FailureCategory.POLICY_BLOCKED,
            )
        return PolicyDecision(allowed=True, reason="Plan approved.")
