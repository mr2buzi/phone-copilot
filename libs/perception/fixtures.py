from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from libs.screen_states import ScreenName


class FixtureLabel(str, Enum):
    HOME_SCREEN = "home_screen"
    MESSAGES_INBOX = "messages_inbox"
    THREAD_VIEW = "thread_view"
    THREAD_VIEW_KEYBOARD_OPEN = "thread_view_keyboard_open"
    PERMISSION_POPUP = "permission_popup"
    GALLERY_PICKER = "gallery_picker"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ExpectedFixtureState:
    label: FixtureLabel
    screen: ScreenName
    keyboard_visible: bool | None = None
    package_name: str | None = None


EXPECTED_FIXTURE_STATES: dict[FixtureLabel, ExpectedFixtureState] = {
    FixtureLabel.HOME_SCREEN: ExpectedFixtureState(FixtureLabel.HOME_SCREEN, ScreenName.HOME_SCREEN),
    FixtureLabel.MESSAGES_INBOX: ExpectedFixtureState(
        FixtureLabel.MESSAGES_INBOX,
        ScreenName.APP_INBOX,
        keyboard_visible=False,
        package_name="com.google.android.apps.messaging",
    ),
    FixtureLabel.THREAD_VIEW: ExpectedFixtureState(
        FixtureLabel.THREAD_VIEW,
        ScreenName.THREAD_VIEW,
        keyboard_visible=False,
        package_name="com.google.android.apps.messaging",
    ),
    FixtureLabel.THREAD_VIEW_KEYBOARD_OPEN: ExpectedFixtureState(
        FixtureLabel.THREAD_VIEW_KEYBOARD_OPEN,
        ScreenName.THREAD_VIEW,
        keyboard_visible=True,
        package_name="com.google.android.apps.messaging",
    ),
    FixtureLabel.PERMISSION_POPUP: ExpectedFixtureState(
        FixtureLabel.PERMISSION_POPUP,
        ScreenName.PERMISSION_POPUP,
    ),
    FixtureLabel.GALLERY_PICKER: ExpectedFixtureState(
        FixtureLabel.GALLERY_PICKER,
        ScreenName.GALLERY_PICKER,
    ),
    FixtureLabel.UNKNOWN: ExpectedFixtureState(FixtureLabel.UNKNOWN, ScreenName.UNKNOWN_SCREEN),
}


def normalize_fixture_label(label: str) -> FixtureLabel:
    cleaned = label.strip().lower()
    if cleaned == "app_inbox":
        cleaned = FixtureLabel.MESSAGES_INBOX.value
    if cleaned == "unknown_screen":
        cleaned = FixtureLabel.UNKNOWN.value
    return FixtureLabel(cleaned)


def expected_state_for_label(label: str) -> ExpectedFixtureState:
    return EXPECTED_FIXTURE_STATES[normalize_fixture_label(label)]


def fixture_label_from_classification(
    screen: ScreenName,
    package_name: str,
    keyboard_visible: bool | None,
) -> FixtureLabel:
    if screen == ScreenName.APP_INBOX and package_name == "com.google.android.apps.messaging":
        return FixtureLabel.MESSAGES_INBOX
    if screen == ScreenName.THREAD_VIEW and package_name == "com.google.android.apps.messaging":
        if keyboard_visible is True:
            return FixtureLabel.THREAD_VIEW_KEYBOARD_OPEN
        return FixtureLabel.THREAD_VIEW
    if screen == ScreenName.HOME_SCREEN:
        return FixtureLabel.HOME_SCREEN
    if screen == ScreenName.PERMISSION_POPUP:
        return FixtureLabel.PERMISSION_POPUP
    if screen == ScreenName.GALLERY_PICKER:
        return FixtureLabel.GALLERY_PICKER
    return FixtureLabel.UNKNOWN


def load_fixture_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
