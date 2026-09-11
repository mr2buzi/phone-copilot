from pathlib import Path

from apps.controller.service import PhoneCopilotService
from libs.planners import ExecutionStep
from libs.drafting import ApprovedPhoto


def test_type_only_executes_tap_then_type(service: PhoneCopilotService, mock_adb) -> None:
    service.pipeline.ocr.extract_lines = lambda _image: [
        "Lunch tomorrow?",
        "Type a message",
        "Send",
    ]
    state = service.approve_type_only(suggestion_index=0)
    assert any(command[0] == "safe_tap" for command in mock_adb.commands)
    assert any(command[0] == "type_text" for command in mock_adb.commands)
    assert state.classification.screen.value == "thread_view"


def test_photo_selection_pushes_only_approved_path(service: PhoneCopilotService, mock_adb, tmp_path) -> None:
    approved_dir = tmp_path / "approved_photos"
    approved_dir.mkdir(parents=True, exist_ok=True)
    (approved_dir / "sample-cat.jpg").write_text("placeholder", encoding="utf-8")
    service.settings.approved_photos_dir = approved_dir
    service.policy.approved_photos_dir = approved_dir
    service.drafting.approved_photos_dir = approved_dir
    service.drafting.photo_index = [ApprovedPhoto(
        photo_id="sample-cat", path=str(approved_dir / "sample-cat.jpg"),
        description="Synthetic test image", tags=[])]
    service.pipeline.ocr.extract_lines = lambda _image: [
        "Gallery",
        "Recent",
        "Photos",
    ]
    mock_adb.foreground_app.package_name = "com.google.android.apps.photos"
    mock_adb.foreground_app.activity_name = ".picker"
    service.approve_select_photo(photo_id="sample-cat")
    push_commands = [command for command in mock_adb.commands if command[0] == "push_file"]
    assert push_commands
    local_path, remote_path = push_commands[0][1]
    assert Path(local_path).name == "sample-cat.jpg"
    assert remote_path == "/sdcard/Pictures/PhoneCopilot/sample-cat.jpg"
    assert any(command[0] == "wait_for_idle" for command in mock_adb.commands)


def test_send_current_draft_executes_send_tap(service: PhoneCopilotService, mock_adb) -> None:
    service.pipeline.ocr.extract_lines = lambda _image: [
        "Lunch tomorrow?",
        "Type a message",
        "Send",
    ]
    mock_adb.keyboard_visible = True
    service._observe_compose_text = lambda expected_text, state: expected_text  # type: ignore[method-assign]
    service.adb.dump_ui_hierarchy = lambda: (  # type: ignore[method-assign]
        '<hierarchy>'
        '<node class="android.widget.EditText" text="Sounds good" />'
        '</hierarchy>'
    )

    state = service.approve_send_current_draft()

    assert any(command[0] == "safe_tap" for command in mock_adb.commands)
    assert state.classification.screen.value == "thread_view"


def test_ai_draft_to_compose_uses_selected_suggestion(service: PhoneCopilotService, mock_adb) -> None:
    service.pipeline.ocr.extract_lines = lambda _image: [
        "Lunch tomorrow?",
        "Type a message",
        "Send",
    ]

    state = service.approve_ai_draft_to_compose(suggestion_index=1)

    typed = [command[1] for command in mock_adb.commands if command[0] == "type_text"]
    assert typed
    assert typed[-1] == state.reply_suggestions[1]


def test_execute_step_recenters_stale_tap_inside_expected_region(service: PhoneCopilotService, mock_adb) -> None:
    service.pipeline.ocr.extract_lines = lambda _image: [
        "Lunch tomorrow?",
        "Type a message",
        "Send",
    ]
    state = service.refresh_state()
    mock_adb.commands.clear()

    service._execute_step(
        ExecutionStep(
            kind="tap",
            x=372,
            y=1600,
            region_left=168,
            region_top=1122,
            region_right=576,
            region_bottom=1290,
        ),
        state,
    )

    safe_taps = [command for command in mock_adb.commands if command[0] == "safe_tap"]
    assert safe_taps
    x, y, _screen = safe_taps[-1][1]
    assert (x, y) == (372, 1206)
