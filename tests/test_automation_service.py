from __future__ import annotations

from datetime import datetime

from apps.controller.models import ControllerState, LoopMetrics, ThreadContext
from libs.drafting.service import DraftBundle, DraftCandidate, DraftScoreBreakdown
from libs.perception import PerceptionResult, PerceptionTimings
from libs.planners import PlannerDecision
from libs.screen_states import ScreenClassification, ScreenName


def _state(screen: ScreenName) -> ControllerState:
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
        keyboard_visible=False,
        keyboard_height=None,
        keyboard_ambiguous=False,
    )
    return ControllerState(
        captured_at=datetime.utcnow(),
        classification=classification,
        planner_decision=PlannerDecision(status="review", reason="ok"),
        summary="summary",
        screenshot_path="test.png",
        metrics=LoopMetrics(
            timestamp=datetime.utcnow(),
            loop_time_ms=1,
            screen=screen.value,
            confidence=classification.confidence,
        ),
    )


def test_extract_thread_message_entries_uses_content_desc_for_speaker(service) -> None:
    hierarchy = """
    <hierarchy>
      <node resource-id="message_text" text="hey" content-desc="You said  hey 4:40 p.m. ." bounds="[10,400][200,450]" />
      <node resource-id="message_text" text="how are you" content-desc="Alice said  how are you 4:41 p.m. ." bounds="[10,500][200,560]" />
    </hierarchy>
    """

    entries = service._extract_thread_message_entries_from_hierarchy(hierarchy)

    assert entries == [
      {"speaker": "me", "text": "hey"},
      {"speaker": "other", "text": "how are you"},
    ]


def test_merge_conversation_entries_uses_overlap_without_dropping_repeated_text(service) -> None:
    existing = [
        {"speaker": "other", "text": "hi"},
        {"speaker": "me", "text": "hey"},
        {"speaker": "other", "text": "hi"},
    ]
    new = [
        {"speaker": "other", "text": "hi"},
        {"speaker": "me", "text": "what you doing"},
    ]

    merged = service._merge_conversation_entries(existing, new)

    assert merged == [
        {"speaker": "other", "text": "hi"},
        {"speaker": "me", "text": "hey"},
        {"speaker": "other", "text": "hi"},
        {"speaker": "me", "text": "what you doing"},
    ]


def test_merge_conversation_entries_prepends_older_messages_when_scrolling_up(service) -> None:
    existing = [
        {"speaker": "other", "text": "hey"},
        {"speaker": "me", "text": "hi"},
        {"speaker": "other", "text": "how r u"},
        {"speaker": "me", "text": "im good"},
    ]
    older_view = [
        {"speaker": "other", "text": "yo"},
        {"speaker": "me", "text": "hey"},
        {"speaker": "other", "text": "hey"},
        {"speaker": "me", "text": "hi"},
        {"speaker": "other", "text": "how r u"},
    ]

    merged = service._merge_conversation_entries(existing, older_view)

    assert merged == [
        {"speaker": "other", "text": "yo"},
        {"speaker": "me", "text": "hey"},
        {"speaker": "other", "text": "hey"},
        {"speaker": "me", "text": "hi"},
        {"speaker": "other", "text": "how r u"},
        {"speaker": "me", "text": "im good"},
    ]


def test_build_unread_queue_prioritizes_detected_unread_threads(service) -> None:
    hierarchy = """
    <hierarchy>
      <node clickable="true" bounds="[0,220][1080,420]">
        <node text="Alice" />
        <node text="2" />
        <node content-desc="Unread conversation" />
      </node>
      <node clickable="true" bounds="[0,430][1080,630]">
        <node text="Bob" />
        <node text="see you later" />
      </node>
    </hierarchy>
    """
    state = _state(ScreenName.APP_INBOX)

    service._read_messages_hierarchy = lambda: hierarchy  # type: ignore[method-assign]
    service._build_unread_queue(state)

    assert [item.contact_name for item in state.unread_queue] == ["Alice"]
    assert state.unread_queue[0].unread is True


def test_capture_state_prefers_context_aware_drafting(service, monkeypatch) -> None:
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
        recent_messages=["u there"],
        keyboard_visible=False,
        keyboard_height=None,
        keyboard_ambiguous=False,
    )
    perception_result = PerceptionResult(
        classification=classification,
        screenshot_path="test.png",
        timings=PerceptionTimings(),
    )
    service.pipeline.capture_and_classify = lambda screenshot_path: perception_result  # type: ignore[method-assign]

    standard_bundle = DraftBundle(summary="recent-only", reply_suggestions=["a", "b", "c"], reply_sequences=[["a"], ["b"], ["c"]])
    context_bundle = DraftBundle(summary="full-context", reply_suggestions=["x", "y", "z"], reply_sequences=[["x"], ["y"], ["z"]])
    service.drafting.build_bundle = lambda recent_messages, contact_name=None: standard_bundle  # type: ignore[method-assign]
    service.drafting.build_bundle_with_context = lambda recent_messages, full_conversation, contact_name=None: context_bundle  # type: ignore[method-assign]
    service._write_latest_aliases = lambda screenshot_path, debug_image_path: None  # type: ignore[method-assign]

    def fake_sync(state: ControllerState) -> None:
        state.thread_context = ThreadContext(
            contact_name="Alice",
            recent_messages=["u there", "yeah"],
            full_conversation=[
                {"speaker": "other", "text": "u there"},
                {"speaker": "me", "text": "yeah"},
            ],
            message_count=2,
            last_message_time=datetime.utcnow(),
        )
        state.classification.recent_messages = ["u there", "yeah"]

    service._sync_state_from_ui_hierarchy = fake_sync  # type: ignore[method-assign]

    state = service._capture_state(record_log=False)

    assert state.summary == "full-context"
    assert state.reply_suggestions == ["x", "y", "z"]
    assert state.reply_sequences == [["x"], ["y"], ["z"]]


def test_sync_state_promotes_unknown_messages_screen_to_thread_view_from_hierarchy(service) -> None:
    state = _state(ScreenName.UNKNOWN_SCREEN)
    state.classification.confidence = 0.41
    hierarchy = """
    <hierarchy>
      <node resource-id="ConversationScreenUi" />
      <node resource-id="message_list" />
      <node resource-id="message_text" text="hey" content-desc="Alice said hey" bounds="[0,100][100,150]" />
      <node resource-id="com.google.android.apps.messaging:id/compose_message_text" text="RCS message" />
      <node resource-id="Compose:Draft:Send" />
      <node class="android.widget.TextView" text="Alice" bounds="[250,120][600,180]" />
    </hierarchy>
    """

    service.adb.dump_ui_hierarchy = lambda: hierarchy  # type: ignore[method-assign]
    service._sync_state_from_ui_hierarchy(state)

    assert state.classification.screen == ScreenName.THREAD_VIEW
    assert state.classification.confidence >= 0.96
    assert state.thread_context is not None
    assert state.thread_context.contact_name == "Alice"


def test_auto_send_reply_blocks_manual_review_candidate(service, mock_adb) -> None:
    state = _state(ScreenName.THREAD_VIEW)
    state.reply_suggestions = ["ill be there at 7"]
    state.reply_sequences = [["ill be there at 7"]]
    state.draft_candidates = [
        DraftCandidate(
            text="ill be there at 7",
            sequence=["ill be there at 7"],
            intent="scheduling",
            risk_flags=["commitment", "time_specific"],
            persona="casual_friend",
            auto_send_allowed=False,
            score_breakdown=DraftScoreBreakdown(final_confidence=0.91, risk_score=0.6),
        )
    ]
    state.recommended_reply_index = 0
    state.auto_send_blocked_reason = "Auto-send blocked: commitment, time_specific"
    service.latest_state = state

    result = service.auto_send_reply()

    assert "Auto-send blocked" in (result.last_action_result or "")
    assert not any(command[0] == "type_text" for command in mock_adb.commands)


def test_auto_send_reply_requires_auto_send_mode(service, mock_adb) -> None:
    state = _state(ScreenName.THREAD_VIEW)
    state.reply_suggestions = ["yh what time"]
    state.reply_sequences = [["yh what time"]]
    state.draft_candidates = [
        DraftCandidate(
            text="yh what time",
            sequence=["yh what time"],
            relationship_type="close_friend",
            final_decision="send",
            auto_send_allowed=True,
            score_breakdown=DraftScoreBreakdown(final_confidence=0.99, risk_score=0.0),
        )
    ]
    state.recommended_reply_index = 0
    service.latest_state = state

    result = service.auto_send_reply()

    assert "mode is not auto-send" in (result.last_action_result or "")
    assert not any(command[0] == "type_text" for command in mock_adb.commands)
