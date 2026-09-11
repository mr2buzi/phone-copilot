from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import Protocol

from libs.adb.models import DeviceInfo, ForegroundApp, KeyboardState, SafeTapContext


class ADBError(RuntimeError):
    pass


class ADBConnectionError(ADBError):
    pass


class ADBCommandError(ADBError):
    pass


class ADBCommandTimeout(ADBCommandError):
    pass


class ADBDeviceDisconnectedError(ADBConnectionError):
    pass


class ADBUIStabilityTimeout(ADBCommandError):
    pass


class ADBClientProtocol(Protocol):
    def connect(self) -> DeviceInfo:
        ...

    def take_screenshot(self, save_path: Path | None = None) -> bytes:
        ...

    def get_foreground_app(self) -> ForegroundApp:
        ...

    def get_keyboard_state(self) -> KeyboardState:
        ...

    def dump_ui_hierarchy(self) -> str:
        ...

    def wait_for_idle(self, timeout_ms: int = 2000, stable_cycles: int = 2) -> bool:
        ...

    def tap(self, x: int, y: int) -> None:
        ...

    def safe_tap(self, x: int, y: int, screen_context: SafeTapContext | None = None) -> tuple[int, int]:
        ...

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        ...

    def type_text(self, text: str) -> None:
        ...

    def keyevent(self, code: int) -> None:
        ...

    def launch_app(self, package_name: str) -> None:
        ...

    def back(self) -> None:
        ...

    def home(self) -> None:
        ...

    def recent_apps(self) -> None:
        ...

    def push_file(self, local_path: Path, remote_path: str) -> None:
        ...

    def scan_media(self, remote_path: str) -> None:
        ...


class ADBClient:
    def __init__(
        self,
        adb_path: str = "adb",
        device_serial: str | None = None,
        command_retries: int = 2,
        retry_backoff_seconds: float = 0.4,
        default_timeout_seconds: float = 15.0,
    ) -> None:
        self.adb_path = adb_path
        self.device_serial = device_serial
        self.command_retries = command_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.default_timeout_seconds = default_timeout_seconds
        self._ensure_adb_available()

    def _base_command(self) -> list[str]:
        command = [self.adb_path]
        if self.device_serial:
            command.extend(["-s", self.device_serial])
        return command

    def _ensure_adb_available(self) -> None:
        try:
            subprocess.run(
                [self.adb_path, "version"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise ADBConnectionError(f"ADB executable not available: {self.adb_path}") from exc

    def _run(
        self,
        *args: str,
        timeout: float | None = None,
        expect_bytes: bool = False,
        retries: int | None = None,
    ) -> str | bytes:
        command = self._base_command() + list(args)
        attempts = (self.command_retries if retries is None else retries) + 1
        timeout = self.default_timeout_seconds if timeout is None else timeout
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                result = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=not expect_bytes,
                    encoding=None if expect_bytes else "utf-8",
                    errors=None if expect_bytes else "replace",
                    timeout=timeout,
                )
                return result.stdout.strip() if not expect_bytes else result.stdout
            except subprocess.TimeoutExpired as exc:
                last_error = ADBCommandTimeout(f"ADB command timed out: {' '.join(command)}")
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr or ""
                if any(token in stderr.lower() for token in ["device offline", "device not found", "no devices/emulators"]):
                    last_error = ADBDeviceDisconnectedError(stderr.strip() or "ADB device disconnected.")
                else:
                    last_error = ADBCommandError(f"ADB command failed: {' '.join(command)}\n{stderr}")
            if attempt < attempts:
                time.sleep(self.retry_backoff_seconds * attempt)
        assert last_error is not None
        raise last_error

    def _read_window_dump(self) -> str:
        return str(self._run("shell", "dumpsys", "window", "windows", timeout=25.0))

    def _screen_size(self) -> tuple[int, int]:
        output = str(self._run("shell", "wm", "size"))
        match = re.search(r"Physical size:\s*(\d+)x(\d+)", output)
        if not match:
            match = re.search(r"Override size:\s*(\d+)x(\d+)", output)
        if not match:
            raise ADBCommandError("Unable to determine device screen size.")
        return int(match.group(1)), int(match.group(2))

    def list_devices(self) -> list[str]:
        output = str(self._run("devices", retries=0))
        return [line.split()[0] for line in output.splitlines()[1:] if line.strip().endswith("device")]

    def connect(self) -> DeviceInfo:
        devices = self.list_devices()
        if self.device_serial:
            if self.device_serial not in devices:
                raise ADBDeviceDisconnectedError(f"Device {self.device_serial} is not connected.")
        elif len(devices) == 1:
            self.device_serial = devices[0]
        elif len(devices) > 1:
            raise ADBConnectionError("Multiple devices connected. Set PHONE_COPILOT_ADB_DEVICE_SERIAL.")
        else:
            raise ADBDeviceDisconnectedError("No ADB devices detected.")
        return DeviceInfo(
            serial=str(self._run("shell", "getprop", "ro.serialno")),
            model=str(self._run("shell", "getprop", "ro.product.model")),
            android_version=str(self._run("shell", "getprop", "ro.build.version.release")),
            product_name=str(self._run("shell", "getprop", "ro.product.name")),
        )

    def get_device_info(self) -> DeviceInfo:
        return self.connect()

    def take_screenshot(self, save_path: Path | None = None) -> bytes:
        data = bytes(self._run("exec-out", "screencap", "-p", expect_bytes=True, timeout=25.0))
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_bytes(data)
        return data

    def get_foreground_app(self) -> ForegroundApp:
        output = str(self._run("shell", "dumpsys", "activity", "activities", timeout=25.0))
        activity_markers = ("mResumedActivity", "topResumedActivity", "mFocusedApp")
        component_pattern = re.compile(r"\b([A-Za-z0-9._]+)/([A-Za-z0-9.$_/-]+)\b")
        for line in output.splitlines():
            if not any(marker in line for marker in activity_markers):
                continue
            match = component_pattern.search(line)
            if match:
                return ForegroundApp(package_name=match.group(1), activity_name=match.group(2))
        fallback = self._read_window_dump()
        match = re.search(r"mCurrentFocus=.*? ([A-Za-z0-9._]+)/([A-Za-z0-9.$_/-]+)", fallback)
        if not match:
            raise ADBCommandError("Unable to determine foreground package/activity.")
        return ForegroundApp(package_name=match.group(1), activity_name=match.group(2))

    def get_keyboard_state(self) -> KeyboardState:
        input_method = str(self._run("shell", "dumpsys", "input_method", timeout=20.0))
        window_dump = self._read_window_dump()
        visible_signal = any(
            token in input_method
            for token in ["mInputShown=true", "mIsInputViewShown=true", "imeWindowVis=0x1"]
        )
        visible: bool | None = True if visible_signal else None
        height: int | None = None
        source = "input_method"
        ambiguous = False
        height_match = re.search(r"imeVisibleHeight=(\d+)", input_method)
        if height_match:
            height = int(height_match.group(1))
            visible = height > 0
        if height == 0:
            inset_match = re.search(r"ime\(\)\s*=\s*Rect\(\d+,\s*\d+\s*-\s*\d+,\s*(\d+)\)", window_dump)
            if inset_match:
                _, screen_height = self._screen_size()
                inset_top = int(inset_match.group(1))
                inferred_height = max(0, screen_height - inset_top)
                height = inferred_height or None
                visible = inferred_height > 0
                source = "window_insets"
        if height is None:
            insets_match = re.search(
                r"InsetsSource id=3 type=ime frame=\[\d+,(\d+)\]\[(\d+),(\d+)\].*?visible=(true|false)",
                window_dump,
            )
            if insets_match:
                inset_top = int(insets_match.group(1))
                inset_bottom = int(insets_match.group(3))
                inferred_height = max(0, inset_bottom - inset_top)
                height = inferred_height or None
                visible = insets_match.group(4) == "true" and inferred_height > 0
                source = "window_insets"
        if height is None:
            input_window_match = re.search(
                r"Window #\d+ Window\{[^\n]+ InputMethod\}:.*?"
                r"touchable region=SkRegion\(\(\d+,(\d+),\d+,(\d+)\)\).*?"
                r"isVisible=(true|false)",
                window_dump,
                re.S,
            )
            if input_window_match:
                region_top = int(input_window_match.group(1))
                _, screen_height = self._screen_size()
                inferred_height = max(0, screen_height - region_top)
                height = inferred_height or None
                visible = input_window_match.group(3) == "true" and inferred_height > 0
                source = "window_input_method"
        if visible_signal and (height is None or height == 0):
            ambiguous = True
            visible = None
            height = None
        if visible is None and height is None:
            source = "ambiguous"
        return KeyboardState(visible=visible, height=height, source=source, ambiguous=ambiguous)

    def wait_for_idle(self, timeout_ms: int = 2000, stable_cycles: int = 2) -> bool:
        deadline = time.monotonic() + (timeout_ms / 1000.0)
        stable = 0
        last_signature: tuple[str, str, bool, int] | None = None
        while time.monotonic() < deadline:
            foreground = self.get_foreground_app()
            keyboard = self.get_keyboard_state()
            signature = (
                foreground.package_name,
                foreground.activity_name,
                keyboard.visible,
                keyboard.height,
            )
            if signature == last_signature:
                stable += 1
                if stable >= stable_cycles:
                    return True
            else:
                stable = 0
                last_signature = signature
            time.sleep(0.2)
        raise ADBUIStabilityTimeout("UI did not stabilize before timeout.")

    def tap(self, x: int, y: int) -> None:
        self._run("shell", "input", "tap", str(x), str(y))

    def safe_tap(self, x: int, y: int, screen_context: SafeTapContext | None = None) -> tuple[int, int]:
        width, height = self._screen_size()
        if x < 0 or y < 0 or x > width or y > height:
            raise ADBCommandError(f"Tap target {x},{y} is outside screen bounds {width}x{height}.")
        keyboard = self.get_keyboard_state()
        adjusted_y = y
        if keyboard.ambiguous:
            raise ADBCommandError("Keyboard state is ambiguous; refusing to tap.")
        expected_bounds: tuple[int, int, int, int] | None = None
        if screen_context is not None:
            left = screen_context.expected_region_left
            top = screen_context.expected_region_top
            right = screen_context.expected_region_right
            bottom = screen_context.expected_region_bottom
            if None not in {left, top, right, bottom}:
                assert left is not None and top is not None and right is not None and bottom is not None
                expected_bounds = (left, top, right, bottom)
        if keyboard.visible and keyboard.height is not None and keyboard.height > 0:
            max_safe_y = max(0, height - keyboard.height - 24)
            adjusted_y = min(y, max_safe_y)
            if expected_bounds is not None:
                left, top, right, bottom = expected_bounds
                original_in_region = left <= x <= right and top <= y <= bottom
                adjusted_in_region = left <= x <= right and top <= adjusted_y <= bottom
                if original_in_region and not adjusted_in_region:
                    adjusted_y = y
        if screen_context is not None:
            if expected_bounds is not None:
                left, top, right, bottom = expected_bounds
                if not (left <= x <= right and top <= adjusted_y <= bottom):
                    raise ADBCommandError(
                        f"Tap target {x},{adjusted_y} is outside the expected region for {screen_context.screen}."
                    )
            if screen_context.before_screenshot_path:
                self.take_screenshot(Path(screen_context.before_screenshot_path))
        self.tap(x, adjusted_y)
        if screen_context is not None and screen_context.after_screenshot_path:
            self.take_screenshot(Path(screen_context.after_screenshot_path))
        return x, adjusted_y

    def dump_ui_hierarchy(self) -> str:
        remote_path = "/sdcard/phone_copilot_window_dump.xml"
        self._run("shell", "uiautomator", "dump", remote_path, timeout=25.0)
        return str(self._run("shell", "cat", remote_path, timeout=25.0))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        self._run(
            "shell",
            "input",
            "swipe",
            str(x1),
            str(y1),
            str(x2),
            str(y2),
            str(duration_ms),
        )

    def type_text(self, text: str) -> None:
        parts = text.split(" ")
        for index, part in enumerate(parts):
            if part:
                self._run("shell", "input", "text", self._escape_text(part))
            if index < len(parts) - 1:
                self.keyevent(62)
                time.sleep(0.05)

    def _escape_text(self, text: str) -> str:
        return (
            text.replace("\\", "\\\\")
            .replace("&", "\\&")
            .replace("<", "\\<")
            .replace(">", "\\>")
            .replace("(", "\\(")
            .replace(")", "\\)")
            .replace("'", "\\'")
            .replace('"', '\\"')
        )

    def keyevent(self, code: int) -> None:
        self._run("shell", "input", "keyevent", str(code))

    def launch_app(self, package_name: str) -> None:
        self._run("shell", "monkey", "-p", package_name, "-c", "android.intent.category.LAUNCHER", "1")

    def back(self) -> None:
        self.keyevent(4)

    def home(self) -> None:
        self.keyevent(3)

    def recent_apps(self) -> None:
        self.keyevent(187)

    def push_file(self, local_path: Path, remote_path: str) -> None:
        if not local_path.exists():
            raise ADBCommandError(f"Local file does not exist: {local_path}")
        self._run("push", str(local_path), remote_path, timeout=60.0)

    def scan_media(self, remote_path: str) -> None:
        uri = f"file://{remote_path}"
        self._run(
            "shell",
            "am",
            "broadcast",
            "-a",
            "android.intent.action.MEDIA_SCANNER_SCAN_FILE",
            "-d",
            uri,
        )
