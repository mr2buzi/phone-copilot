from libs.drafting import ApprovedPhoto
from libs.planners import StateMachinePlanner
from libs.screen_states import (
    ApprovalActionType,
    AvailableAction,
    DevicePoint,
    ScreenClassification,
    ScreenName,
)


def _classification(screen: ScreenName) -> ScreenClassification:
    actions = []
    if screen == ScreenName.THREAD_VIEW:
        actions = [
            AvailableAction(
                action_type=ApprovalActionType.TYPE_DRAFT,
                label="Approve type only",
                point=DevicePoint(x=540, y=2200),
            ),
            AvailableAction(
                action_type=ApprovalActionType.OPEN_GALLERY,
                label="Approve open gallery",
                point=DevicePoint(x=120, y=2200),
            ),
            AvailableAction(
                action_type=ApprovalActionType.SEND_MESSAGE,
                label="Approve send message",
                point=DevicePoint(x=960, y=2200),
            ),
        ]
    if screen == ScreenName.GALLERY_PICKER:
        actions = [
            AvailableAction(
                action_type=ApprovalActionType.SELECT_APPROVED_PHOTO,
                label="Approve select approved photo",
                point=DevicePoint(x=240, y=680),
            )
        ]
    return ScreenClassification(
        app="Messages",
        screen=screen,
        confidence=0.91,
        visible_text=["Lunch tomorrow?"],
        available_actions=actions,
        package_name="com.google.android.apps.messaging",
        activity_name=".ConversationActivity",
        screenshot_width=1080,
        screenshot_height=2340,
        recent_messages=["Lunch tomorrow?"],
    )


def test_planner_builds_type_only_plan() -> None:
    planner = StateMachinePlanner(photo_push_dir="/sdcard/Pictures/PhoneCopilot")
    plan = planner.build_plan(
        classification=_classification(ScreenName.THREAD_VIEW),
        action_type=ApprovalActionType.TYPE_DRAFT,
        draft_text="Sounds good. I'll keep you posted.",
    )
    assert plan.expected_prev_screen == ScreenName.THREAD_VIEW
    assert plan.expected_next_screen == ScreenName.THREAD_VIEW
    assert [step.kind for step in plan.steps] == ["tap", "type_text"]


def test_planner_builds_photo_selection_plan() -> None:
    planner = StateMachinePlanner(photo_push_dir="/sdcard/Pictures/PhoneCopilot")
    plan = planner.build_plan(
        classification=_classification(ScreenName.GALLERY_PICKER),
        action_type=ApprovalActionType.SELECT_APPROVED_PHOTO,
        approved_photo=ApprovedPhoto(
            photo_id="sample-lunch",
            path="data/approved_photos/sample-lunch.jpg",
            description="Lunch photo",
            tags=["food"],
        ),
    )
    assert plan.expected_prev_screen == ScreenName.GALLERY_PICKER
    assert plan.expected_next_screen == ScreenName.THREAD_VIEW
    assert [step.kind for step in plan.steps] == ["push_file", "media_scan", "tap"]


def test_planner_builds_send_message_plan() -> None:
    planner = StateMachinePlanner(photo_push_dir="/sdcard/Pictures/PhoneCopilot")
    classification = _classification(ScreenName.THREAD_VIEW)
    classification.keyboard_visible = True
    plan = planner.build_plan(
        classification=classification,
        action_type=ApprovalActionType.SEND_MESSAGE,
    )
    assert plan.expected_prev_screen == ScreenName.THREAD_VIEW
    assert plan.expected_next_screen == ScreenName.THREAD_VIEW
    assert [step.kind for step in plan.steps] == ["tap"]


def test_planner_invalid_state_transition_raises() -> None:
    planner = StateMachinePlanner(photo_push_dir="/sdcard/Pictures/PhoneCopilot")
    classification = _classification(ScreenName.APP_INBOX)
    try:
        planner.build_plan(
            classification=classification,
            action_type=ApprovalActionType.TYPE_DRAFT,
            draft_text="Nope",
        )
    except ValueError as exc:
        assert "Unsupported planner request" in str(exc)
    else:
        raise AssertionError("Expected invalid planner transition to raise ValueError.")
