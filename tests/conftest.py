from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from apps.controller.service import PhoneCopilotService
from apps.controller.settings import ControllerSettings
from libs.adb import ForegroundApp, KeyboardState


class MockADBClient:
    def __init__(self) -> None:
        self.commands: list[tuple[str, object]] = []
        self.foreground_app = ForegroundApp(
            package_name="com.google.android.apps.messaging",
            activity_name=".ConversationListActivity",
        )
        self.keyboard_visible = False

    def connect(self):
        return None

    def take_screenshot(self, save_path: Path | None = None) -> bytes:
        image = Image.new("RGB", (1080, 2340), color=(255, 255, 255))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_bytes(data)
        return data

    def get_foreground_app(self) -> ForegroundApp:
        return self.foreground_app

    def get_keyboard_state(self) -> KeyboardState:
        return KeyboardState(
            visible=self.keyboard_visible,
            height=640 if self.keyboard_visible else None,
            source="mock",
            ambiguous=False,
        )

    def wait_for_idle(self, timeout_ms: int = 2000, stable_cycles: int = 2) -> bool:
        self.commands.append(("wait_for_idle", (timeout_ms, stable_cycles)))
        return True

    def dump_ui_hierarchy(self) -> str:
        typed_texts = [command[1] for command in self.commands if command[0] == "type_text"]
        latest = typed_texts[-1] if typed_texts else ""
        return f"<hierarchy><node class=\"android.widget.EditText\" text=\"{latest}\" /></hierarchy>"

    def tap(self, x: int, y: int) -> None:
        self.commands.append(("tap", (x, y)))

    def safe_tap(self, x: int, y: int, screen_context=None) -> tuple[int, int]:
        self.commands.append(("safe_tap", (x, y, screen_context.screen if screen_context else None)))
        self.keyboard_visible = bool(screen_context and screen_context.screen == "thread_view")
        if screen_context and screen_context.before_screenshot_path:
            Path(screen_context.before_screenshot_path).write_text("before", encoding="utf-8")
        if screen_context and screen_context.after_screenshot_path:
            Path(screen_context.after_screenshot_path).write_text("after", encoding="utf-8")
        return x, y

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        self.commands.append(("swipe", (x1, y1, x2, y2, duration_ms)))

    def type_text(self, text: str) -> None:
        self.commands.append(("type_text", text))

    def keyevent(self, code: int) -> None:
        self.commands.append(("keyevent", code))
        if code == 4:
            self.keyboard_visible = False

    def launch_app(self, package_name: str) -> None:
        self.commands.append(("launch_app", package_name))

    def back(self) -> None:
        self.keyevent(4)

    def home(self) -> None:
        self.keyevent(3)

    def recent_apps(self) -> None:
        self.keyevent(187)

    def push_file(self, local_path: Path, remote_path: str) -> None:
        self.commands.append(("push_file", (str(local_path), remote_path)))

    def scan_media(self, remote_path: str) -> None:
        self.commands.append(("scan_media", remote_path))


@pytest.fixture()
def mock_adb() -> MockADBClient:
    return MockADBClient()


@pytest.fixture()
def service(tmp_path: Path, mock_adb: MockADBClient) -> PhoneCopilotService:
    settings = ControllerSettings(
        enable_ocr=False,
        confidence_threshold=0.7,
        ai_reply_enabled=False,
        log_db_path=tmp_path / "phone_copilot.db",
        screenshot_dir=tmp_path / "screenshots",
        debug_dir=tmp_path / "debug",
        fixtures_live_dir=tmp_path / "fixtures",
        approved_photos_dir=Path("data/approved_photos"),
        reply_template_path=Path("data/templates/reply_templates.json"),
        selector_path=Path("data/selectors/screen_signatures.json"),
    )
    return PhoneCopilotService(settings=settings, adb_client=mock_adb)
