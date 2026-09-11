from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from libs.adb.client import (
    ADBClient,
    ADBCommandError,
    ADBCommandTimeout,
    ADBDeviceDisconnectedError,
)
from libs.adb.models import KeyboardState, SafeTapContext


def test_adb_run_retries_until_success(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_run(command, **kwargs):
        calls["count"] += 1
        if command[1] == "version":
            return SimpleNamespace(stdout="Android Debug Bridge version", stderr="")
        if calls["count"] < 4:
            raise subprocess.CalledProcessError(returncode=1, cmd=command, stderr="temporary failure")
        return SimpleNamespace(stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    client = ADBClient(command_retries=2, retry_backoff_seconds=0.0)
    output = client._run("devices")
    assert output == "ok"


def test_adb_run_timeout_raises_structured_error(monkeypatch) -> None:
    def fake_run(command, **kwargs):
        if command[1] == "version":
            return SimpleNamespace(stdout="Android Debug Bridge version", stderr="")
        raise subprocess.TimeoutExpired(cmd=command, timeout=1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    client = ADBClient(command_retries=0, retry_backoff_seconds=0.0)
    with pytest.raises(ADBCommandTimeout):
        client._run("devices")


def test_adb_run_device_disconnected_error(monkeypatch) -> None:
    def fake_run(command, **kwargs):
        if command[1] == "version":
            return SimpleNamespace(stdout="Android Debug Bridge version", stderr="")
        raise subprocess.CalledProcessError(returncode=1, cmd=command, stderr="device offline")

    monkeypatch.setattr(subprocess, "run", fake_run)
    client = ADBClient(command_retries=0, retry_backoff_seconds=0.0)
    with pytest.raises(ADBDeviceDisconnectedError):
        client._run("devices")


def test_keyboard_state_parsing_prefers_visible_height(monkeypatch) -> None:
    monkeypatch.setattr(ADBClient, "_ensure_adb_available", lambda self: None)
    client = ADBClient()

    def fake_run(*args, **kwargs):
        if args[:3] == ("shell", "dumpsys", "input_method"):
            return "mInputShown=true\nimeVisibleHeight=640"
        if args[:3] == ("shell", "dumpsys", "window"):
            return "window dump"
        raise ADBCommandError("unexpected command")

    monkeypatch.setattr(client, "_run", fake_run)
    monkeypatch.setattr(client, "_read_window_dump", lambda: "window dump")
    state = client.get_keyboard_state()
    assert state.visible is True
    assert state.height == 640


def test_keyboard_state_parsing_can_be_ambiguous(monkeypatch) -> None:
    monkeypatch.setattr(ADBClient, "_ensure_adb_available", lambda self: None)
    client = ADBClient()

    def fake_run(*args, **kwargs):
        if args[:3] == ("shell", "dumpsys", "input_method"):
            return "mInputShown=true\nimeVisibleHeight=0"
        if args[:3] == ("shell", "dumpsys", "window"):
            return "window dump"
        raise ADBCommandError("unexpected command")

    monkeypatch.setattr(client, "_run", fake_run)
    monkeypatch.setattr(client, "_read_window_dump", lambda: "window dump")
    state = client.get_keyboard_state()
    assert state.ambiguous is True
    assert state.visible is None
    assert state.height is None


def test_foreground_app_parsing_handles_top_resumed_activity(monkeypatch) -> None:
    monkeypatch.setattr(ADBClient, "_ensure_adb_available", lambda self: None)
    client = ADBClient()

    def fake_run(*args, **kwargs):
        if args[:4] == ("shell", "dumpsys", "activity", "activities"):
            return (
                "ACTIVITY MANAGER ACTIVITIES\n"
                "  topResumedActivity=ActivityRecord{163244899 u0 "
                "com.google.android.apps.messaging/.ui.ConversationListActivity t801}\n"
            )
        raise ADBCommandError("unexpected command")

    monkeypatch.setattr(client, "_run", fake_run)
    monkeypatch.setattr(client, "_read_window_dump", lambda: "")

    foreground = client.get_foreground_app()
    assert foreground.package_name == "com.google.android.apps.messaging"
    assert foreground.activity_name == ".ui.ConversationListActivity"


def test_keyboard_state_parsing_uses_window_insets_frame(monkeypatch) -> None:
    monkeypatch.setattr(ADBClient, "_ensure_adb_available", lambda self: None)
    client = ADBClient()

    def fake_run(*args, **kwargs):
        if args[:3] == ("shell", "dumpsys", "input_method"):
            return "mInputShown=true"
        if args[:3] == ("shell", "wm", "size"):
            return "Physical size: 1080x2340"
        raise ADBCommandError("unexpected command")

    monkeypatch.setattr(client, "_run", fake_run)
    monkeypatch.setattr(
        client,
        "_read_window_dump",
        lambda: "InsetsSource id=3 type=ime frame=[0,1314][1080,2340] visibleFrame=[0,1314][1080,2340] visible=true",
    )

    state = client.get_keyboard_state()
    assert state.visible is True
    assert state.height == 1026
    assert state.source == "window_insets"


def test_keyboard_state_parsing_uses_input_method_window(monkeypatch) -> None:
    monkeypatch.setattr(ADBClient, "_ensure_adb_available", lambda self: None)
    client = ADBClient()

    def fake_run(*args, **kwargs):
        if args[:3] == ("shell", "dumpsys", "input_method"):
            return "mInputShown=true"
        if args[:3] == ("shell", "wm", "size"):
            return "Physical size: 1080x2340"
        raise ADBCommandError("unexpected command")

    monkeypatch.setattr(client, "_run", fake_run)
    monkeypatch.setattr(
        client,
        "_read_window_dump",
        lambda: (
            "Window #10 Window{1a40c40 u0 InputMethod}:\n"
            "  touchable region=SkRegion((0,1314,1080,2196))\n"
            "  isVisible=true\n"
        ),
    )

    state = client.get_keyboard_state()
    assert state.visible is True
    assert state.height == 1026
    assert state.source == "window_input_method"


def test_type_text_sends_words_with_space_keyevents(monkeypatch) -> None:
    monkeypatch.setattr(ADBClient, "_ensure_adb_available", lambda self: None)
    client = ADBClient()
    commands: list[tuple[str, ...]] = []

    def fake_run(*args, **kwargs):
        commands.append(tuple(args))
        return ""

    monkeypatch.setattr(client, "_run", fake_run)

    client.type_text("hi how are you")

    assert commands == [
        ("shell", "input", "text", "hi"),
        ("shell", "input", "keyevent", "62"),
        ("shell", "input", "text", "how"),
        ("shell", "input", "keyevent", "62"),
        ("shell", "input", "text", "are"),
        ("shell", "input", "keyevent", "62"),
        ("shell", "input", "text", "you"),
    ]


def test_safe_tap_prefers_original_point_when_keyboard_adjustment_leaves_expected_region(monkeypatch) -> None:
    monkeypatch.setattr(ADBClient, "_ensure_adb_available", lambda self: None)
    client = ADBClient()
    tapped: list[tuple[int, int]] = []

    monkeypatch.setattr(client, "_screen_size", lambda: (1080, 2340))
    monkeypatch.setattr(
        client,
        "get_keyboard_state",
        lambda: KeyboardState(visible=True, height=1100, source="test", ambiguous=False),
    )
    monkeypatch.setattr(client, "tap", lambda x, y: tapped.append((x, y)))

    point = client.safe_tap(
        372,
        1290,
        SafeTapContext(
            screen="thread_view",
            expected_region_left=168,
            expected_region_top=1250,
            expected_region_right=576,
            expected_region_bottom=1350,
        ),
    )

    assert point == (372, 1290)
    assert tapped == [(372, 1290)]


def test_safe_tap_still_rejects_target_outside_expected_region(monkeypatch) -> None:
    monkeypatch.setattr(ADBClient, "_ensure_adb_available", lambda self: None)
    client = ADBClient()

    monkeypatch.setattr(client, "_screen_size", lambda: (1080, 2340))
    monkeypatch.setattr(
        client,
        "get_keyboard_state",
        lambda: KeyboardState(visible=True, height=400, source="test", ambiguous=False),
    )

    with pytest.raises(ADBCommandError):
        client.safe_tap(
            900,
            1600,
            SafeTapContext(
                screen="thread_view",
                expected_region_left=168,
                expected_region_top=1122,
                expected_region_right=576,
                expected_region_bottom=1290,
            ),
        )
