from pathlib import Path

from libs.planners import ExecutionPlan, ExecutionStep, PlannerDecision
from libs.policies import PolicyEngine
from libs.screen_states import ApprovalActionType, ScreenClassification, ScreenName
from libs.validation_models import FailureCategory


def test_policy_blocks_low_confidence() -> None:
    policy = PolicyEngine(confidence_threshold=0.8, approved_photos_dir=Path("data/approved_photos"))
    classification = ScreenClassification(
        app="Messages",
        screen=ScreenName.THREAD_VIEW,
        confidence=0.4,
        visible_text=[],
        available_actions=[],
        package_name="com.google.android.apps.messaging",
        activity_name=".ConversationActivity",
        screenshot_width=1080,
        screenshot_height=2340,
        recent_messages=[],
    )
    decision = policy.evaluate_observation(
        classification=classification,
        planner_decision=PlannerDecision(status="review", reason="thread view"),
        emergency_stop=False,
    )
    assert not decision.allowed


def test_policy_blocks_non_approved_photo_path(tmp_path: Path) -> None:
    policy = PolicyEngine(confidence_threshold=0.8, approved_photos_dir=tmp_path / "approved")
    (tmp_path / "approved").mkdir()
    outsider = tmp_path / "outsider.jpg"
    outsider.write_text("not an image", encoding="utf-8")
    plan = ExecutionPlan(
        requested_action=ApprovalActionType.SELECT_APPROVED_PHOTO,
        expected_prev_screen=ScreenName.GALLERY_PICKER,
        expected_next_screen=ScreenName.THREAD_VIEW,
        acceptable_next_screens=[ScreenName.THREAD_VIEW],
        steps=[ExecutionStep(kind="push_file", local_path=str(outsider), remote_path="/tmp/outsider.jpg")],
        notes="select",
    )
    classification = ScreenClassification(
        app="Gallery",
        screen=ScreenName.GALLERY_PICKER,
        confidence=0.9,
        visible_text=[],
        available_actions=[],
        package_name="com.google.android.apps.photos",
        activity_name=".picker",
        screenshot_width=1080,
        screenshot_height=2340,
        recent_messages=[],
    )
    decision = policy.authorize_plan(plan=plan, classification=classification, emergency_stop=False)
    assert not decision.allowed


def test_policy_blocks_previous_failure() -> None:
    policy = PolicyEngine(confidence_threshold=0.8, approved_photos_dir=Path("data/approved_photos"))
    classification = ScreenClassification(
        app="Messages",
        screen=ScreenName.THREAD_VIEW,
        confidence=0.9,
        visible_text=[],
        available_actions=[],
        package_name="com.google.android.apps.messaging",
        activity_name=".ConversationActivity",
        screenshot_width=1080,
        screenshot_height=2340,
        recent_messages=[],
        keyboard_visible=False,
    )
    decision = policy.evaluate_observation(
        classification=classification,
        planner_decision=PlannerDecision(status="review", reason="thread view"),
        emergency_stop=False,
        previous_step_failed=True,
    )
    assert not decision.allowed


def test_policy_allows_observation_when_keyboard_ambiguous_and_confident() -> None:
    policy = PolicyEngine(confidence_threshold=0.8, approved_photos_dir=Path("data/approved_photos"))
    classification = ScreenClassification(
        app="Messages",
        screen=ScreenName.THREAD_VIEW,
        confidence=0.95,
        visible_text=[],
        available_actions=[],
        package_name="com.google.android.apps.messaging",
        activity_name=".ConversationActivity",
        screenshot_width=1080,
        screenshot_height=2340,
        recent_messages=[],
        keyboard_visible=None,
        keyboard_height=None,
        keyboard_ambiguous=True,
    )
    decision = policy.evaluate_observation(
        classification=classification,
        planner_decision=PlannerDecision(status="review", reason="thread view"),
        emergency_stop=False,
    )
    assert decision.allowed
    assert decision.reason == "Observation is within policy bounds."


def test_policy_blocks_type_draft_when_keyboard_ambiguous() -> None:
    policy = PolicyEngine(confidence_threshold=0.8, approved_photos_dir=Path("data/approved_photos"))
    classification = ScreenClassification(
        app="Messages",
        screen=ScreenName.THREAD_VIEW,
        confidence=0.95,
        visible_text=[],
        available_actions=[],
        package_name="com.google.android.apps.messaging",
        activity_name=".ConversationActivity",
        screenshot_width=1080,
        screenshot_height=2340,
        recent_messages=[],
        keyboard_visible=None,
        keyboard_height=None,
        keyboard_ambiguous=True,
    )
    plan = ExecutionPlan(
        requested_action=ApprovalActionType.TYPE_DRAFT,
        expected_prev_screen=ScreenName.THREAD_VIEW,
        expected_next_screen=ScreenName.THREAD_VIEW,
        acceptable_next_screens=[ScreenName.THREAD_VIEW],
        steps=[ExecutionStep(kind="tap", x=100, y=100)],
        notes="type draft",
    )
    decision = policy.authorize_plan(plan=plan, classification=classification, emergency_stop=False)
    assert not decision.allowed
    assert decision.failure_category == FailureCategory.KEYBOARD_STATE_AMBIGUOUS


def test_policy_blocks_send_message_when_keyboard_ambiguous() -> None:
    policy = PolicyEngine(confidence_threshold=0.8, approved_photos_dir=Path("data/approved_photos"))
    classification = ScreenClassification(
        app="Messages",
        screen=ScreenName.THREAD_VIEW,
        confidence=0.95,
        visible_text=[],
        available_actions=[],
        package_name="com.google.android.apps.messaging",
        activity_name=".ConversationActivity",
        screenshot_width=1080,
        screenshot_height=2340,
        recent_messages=[],
        keyboard_visible=None,
        keyboard_height=None,
        keyboard_ambiguous=True,
    )
    plan = ExecutionPlan(
        requested_action=ApprovalActionType.SEND_MESSAGE,
        expected_prev_screen=ScreenName.THREAD_VIEW,
        expected_next_screen=ScreenName.THREAD_VIEW,
        acceptable_next_screens=[ScreenName.THREAD_VIEW],
        steps=[ExecutionStep(kind="tap", x=980, y=1600)],
        notes="send draft",
    )
    decision = policy.authorize_plan(plan=plan, classification=classification, emergency_stop=False)
    assert not decision.allowed
    assert decision.failure_category == FailureCategory.KEYBOARD_STATE_AMBIGUOUS


def test_policy_rejects_unknown_screen() -> None:
    policy = PolicyEngine(confidence_threshold=0.8, approved_photos_dir=Path("data/approved_photos"))
    classification = ScreenClassification(
        app="Unknown",
        screen=ScreenName.UNKNOWN_SCREEN,
        confidence=0.95,
        visible_text=[],
        available_actions=[],
        package_name="unknown",
        activity_name="unknown",
        screenshot_width=1080,
        screenshot_height=2340,
        recent_messages=[],
    )
    decision = policy.evaluate_observation(
        classification=classification,
        planner_decision=PlannerDecision(status="halt", reason="unknown"),
        emergency_stop=False,
    )
    assert not decision.allowed


def test_policy_blocks_forbidden_outbound_actions() -> None:
    policy = PolicyEngine(confidence_threshold=0.8, approved_photos_dir=Path("data/approved_photos"))
    classification = ScreenClassification(
        app="Messages",
        screen=ScreenName.THREAD_VIEW,
        confidence=0.9,
        visible_text=[],
        available_actions=[],
        package_name="com.google.android.apps.messaging",
        activity_name=".ConversationActivity",
        screenshot_width=1080,
        screenshot_height=2340,
        recent_messages=[],
    )
    for forbidden_kind in ["send_message", "add_contact", "camera_access", "auto_send"]:
        plan = ExecutionPlan(
            requested_action=ApprovalActionType.TYPE_DRAFT,
            expected_prev_screen=ScreenName.THREAD_VIEW,
            expected_next_screen=ScreenName.THREAD_VIEW,
            steps=[ExecutionStep(kind=forbidden_kind)],
            notes="forbidden",
        )
        decision = policy.authorize_plan(plan=plan, classification=classification, emergency_stop=False)
        assert not decision.allowed
