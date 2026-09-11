from __future__ import annotations

from datetime import datetime
from types import MethodType

import pytest
from fastapi import HTTPException

from apps.controller.models import ControllerState, LoopMetrics
from apps.controller.service import PhoneCopilotService
from libs.planners import ExecutionPlan, ExecutionStep, PlannerDecision
from libs.policies import PolicyDecision
from libs.screen_states import ApprovalActionType, ScreenClassification, ScreenName


def _state(
    screen: ScreenName,
    *,
    keyboard_visible: bool | None = False,
    keyboard_ambiguous: bool = False,
    halted: bool = False,
) -> ControllerState:
    classification = ScreenClassification(
        app="Messages",
        screen=screen,
        confidence=0.95,
        visible_text=[],
        available_actions=[],
        package_name="com.google.android.apps.messaging",
        activity_name=".ConversationActivity",
        screenshot_width=1080,
        screenshot_height=2340,
        recent_messages=[],
        keyboard_visible=keyboard_visible,
        keyboard_height=640 if keyboard_visible else None,
        keyboard_ambiguous=keyboard_ambiguous,
    )
    return ControllerState(
        captured_at=datetime.utcnow(),
        classification=classification,
        planner_decision=PlannerDecision(status="review", reason="ok"),
        summary="summary",
        screenshot_path="test.png",
        halted=halted,
        metrics=LoopMetrics(
            timestamp=datetime.utcnow(),
            loop_time_ms=1,
            screen=screen.value,
            confidence=classification.confidence,
        ),
    )


def _plan(*, retry: int = 0, expected_prev: ScreenName = ScreenName.THREAD_VIEW, expected_next: ScreenName = ScreenName.THREAD_VIEW) -> ExecutionPlan:
    return ExecutionPlan(
        requested_action=ApprovalActionType.TYPE_DRAFT,
        expected_prev_screen=expected_prev,
        expected_next_screen=expected_next,
        acceptable_next_screens=[expected_next],
        timeout_ms=25,
        retry=retry,
        expected_keyboard_after=True,
        steps=[ExecutionStep(kind="type_text", text="hello")],
        notes="verify",
    )


def test_postcondition_success(service: PhoneCopilotService) -> None:
    state = _state(ScreenName.THREAD_VIEW, keyboard_visible=True)
    service._capture_state = lambda record_log, execution_time_ms=0.0: state  # type: ignore[method-assign]
    result = service._verify_postcondition(_plan())
    assert result.classification.screen == ScreenName.THREAD_VIEW


def test_postcondition_timeout(service: PhoneCopilotService, monkeypatch) -> None:
    service._capture_state = lambda record_log, execution_time_ms=0.0: _state(ScreenName.GALLERY_PICKER)  # type: ignore[method-assign]
    monkeypatch.setattr("apps.controller.service.time.sleep", lambda _seconds: None)
    with pytest.raises(HTTPException) as exc:
        service._verify_postcondition(_plan(expected_next=ScreenName.THREAD_VIEW))
    assert "Postcondition timeout" in str(exc.value.detail)


def test_precondition_mismatch_blocks_execution(service: PhoneCopilotService, mock_adb) -> None:
    service.latest_state = _state(ScreenName.APP_INBOX)
    plan = _plan(expected_prev=ScreenName.THREAD_VIEW)
    with pytest.raises(HTTPException):
        service._execute_approved_plan(plan=plan, approval_note="type")
    assert not any(command[0] == "safe_tap" for command in mock_adb.commands)


def test_retry_logic_stops_after_configured_retries(service: PhoneCopilotService, monkeypatch) -> None:
    service.latest_state = _state(ScreenName.THREAD_VIEW)
    service.policy.authorize_plan = lambda **kwargs: PolicyDecision(allowed=True, reason="ok")  # type: ignore[assignment]
    attempts = {"count": 0}

    def fake_execute_step(self, step, current_state):
        attempts["count"] += 1

    service._execute_step = MethodType(fake_execute_step, service)
    service._verify_postcondition = lambda plan: (_ for _ in ()).throw(HTTPException(status_code=409, detail="boom"))  # type: ignore[method-assign]
    service.refresh_state = lambda: _state(ScreenName.THREAD_VIEW)  # type: ignore[method-assign]
    service._capture_state = lambda record_log, execution_time_ms=0.0: _state(ScreenName.THREAD_VIEW, halted=True)  # type: ignore[method-assign]
    with pytest.raises(HTTPException):
        service._execute_approved_plan(plan=_plan(retry=2), approval_note="type")
    assert attempts["count"] == 3


def test_halt_on_failure_sets_internal_failure_flag(service: PhoneCopilotService, monkeypatch) -> None:
    service.latest_state = _state(ScreenName.THREAD_VIEW)
    service.policy.authorize_plan = lambda **kwargs: PolicyDecision(allowed=True, reason="ok")  # type: ignore[assignment]
    service._execute_step = MethodType(lambda self, step, current_state: None, service)
    service._verify_postcondition = lambda plan: (_ for _ in ()).throw(HTTPException(status_code=409, detail="Postcondition timeout: expected thread_view, detected unknown."))  # type: ignore[method-assign]
    service._capture_state = lambda record_log, execution_time_ms=0.0: _state(ScreenName.THREAD_VIEW, halted=True)  # type: ignore[method-assign]
    monkeypatch.setattr("apps.controller.service.time.sleep", lambda _seconds: None)
    with pytest.raises(HTTPException):
        service._execute_approved_plan(plan=_plan(retry=0), approval_note="type")
    assert service._previous_step_failed is True
    assert service.latest_state is not None
    assert service.latest_state.halted is True
    assert service.latest_state.failure_category == "postcondition_timeout"


def test_compose_validation_blocks_ambiguous_keyboard(service: PhoneCopilotService) -> None:
    ambiguous_state = _state(ScreenName.THREAD_VIEW, keyboard_visible=None, keyboard_ambiguous=True)
    service.refresh_state = lambda: ambiguous_state  # type: ignore[method-assign]
    result = service.run_compose_validation("validation text")
    assert result.halted is True
    assert result.failure_category == "keyboard_state_ambiguous"


def test_compose_validation_postcondition_success(service: PhoneCopilotService) -> None:
    service.pipeline.ocr.extract_lines = lambda _image: ["Type a message", "Send"]
    service.latest_state = None
    result = service.run_compose_validation("validation text")
    assert result.compose_validation is not None
    assert result.compose_validation.success is True


def test_compose_validation_precondition_blocks_non_thread(service: PhoneCopilotService) -> None:
    non_thread = _state(ScreenName.APP_INBOX, keyboard_visible=False)
    service.refresh_state = lambda: non_thread  # type: ignore[method-assign]
    result = service.run_compose_validation("validation text")
    assert result.halted is True
    assert result.failure_category == "precondition_failed"


def test_metrics_include_compose_validation_reporting(service: PhoneCopilotService) -> None:
    service.refresh_state = lambda: _state(ScreenName.APP_INBOX, keyboard_visible=False)  # type: ignore[method-assign]
    service.run_compose_validation("validation text")
    metrics = service.metrics_snapshot()
    assert metrics.compose_validation["total_runs"] == 1
    assert metrics.compose_validation["failed_runs"] == 1


def test_send_current_draft_blocks_when_compose_is_empty(service: PhoneCopilotService) -> None:
    service.pipeline.ocr.extract_lines = lambda _image: ["Lunch tomorrow?", "Type a message", "Send"]
    service.adb.dump_ui_hierarchy = lambda: (  # type: ignore[method-assign]
        '<hierarchy><node class="android.widget.EditText" text="RCS message" /></hierarchy>'
    )

    with pytest.raises(HTTPException) as exc:
        service.approve_send_current_draft()

    assert "No draft text is currently in the compose box." in str(exc.value.detail)


def test_send_and_read_success(service: PhoneCopilotService, mock_adb, monkeypatch) -> None:
    observations = iter(
        [
            ["Lunch tomorrow?", "Type a message", "Send"],
            ["Lunch tomorrow?", "hi how are you", "Type a message", "Send"],
            ["Lunch tomorrow?", "hi how are you", "Type a message", "Send"],
            ["Lunch tomorrow?", "hi how are you", "Send"],
            ["Lunch tomorrow?", "hi how are you", "Send"],
            ["Lunch tomorrow?", "hi how are you", "I am good thanks", "Send"],
        ]
    )

    def fake_extract_lines(_image):
        try:
            return next(observations)
        except StopIteration:
            return ["Lunch tomorrow?", "hi how are you", "I am good thanks", "Send"]

    service.pipeline.ocr.extract_lines = fake_extract_lines
    monkeypatch.setattr("apps.controller.service.time.sleep", lambda _seconds: None)

    result = service.run_send_and_read("hi how are you", wait_timeout_seconds=1.0, poll_interval_seconds=0.0)

    assert result.send_and_read is not None
    assert result.send_and_read.success is True
    assert result.send_and_read.response_text == "I am good thanks"
    assert [command[0] for command in mock_adb.commands].count("safe_tap") == 2
    assert any(command[0] == "type_text" and command[1] == "hi how are you" for command in mock_adb.commands)


def test_send_and_read_blocks_non_empty_compose(service: PhoneCopilotService, mock_adb) -> None:
    service.pipeline.ocr.extract_lines = lambda _image: ["Lunch tomorrow?", "Type a message", "Send"]
    service.adb.dump_ui_hierarchy = lambda: '<hierarchy><node class="android.widget.EditText" text="draft already here" /></hierarchy>'  # type: ignore[method-assign]

    result = service.run_send_and_read("hi how are you", wait_timeout_seconds=1.0)

    assert result.halted is True
    assert result.failure_category == "precondition_failed"
    assert result.send_and_read is not None
    assert result.send_and_read.success is False
    assert not any(command[0] == "type_text" for command in mock_adb.commands)


def test_extract_compose_box_text_accepts_text_before_class(service: PhoneCopilotService) -> None:
    hierarchy = (
        '<hierarchy>'
        '<node index="0" text="im here, relax" '
        'resource-id="com.google.android.apps.messaging:id/compose_message_text" '
        'class="android.widget.EditText" hint="RCS message" />'
        '</hierarchy>'
    )

    assert service._extract_compose_box_text_from_hierarchy(hierarchy) == "im here, relax"
