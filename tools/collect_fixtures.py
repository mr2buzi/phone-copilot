from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from apps.controller.settings import ControllerSettings
from libs.adb import ADBClient
from libs.perception import FixtureLabel, PerceptionPipeline

LABEL_KEYS = {
    "h": FixtureLabel.HOME_SCREEN.value,
    "i": FixtureLabel.MESSAGES_INBOX.value,
    "t": FixtureLabel.THREAD_VIEW.value,
    "k": FixtureLabel.THREAD_VIEW_KEYBOARD_OPEN.value,
    "g": FixtureLabel.GALLERY_PICKER.value,
    "p": FixtureLabel.PERMISSION_POPUP.value,
    "u": FixtureLabel.UNKNOWN.value,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuously capture labeled live fixtures from a connected Android device.")
    parser.add_argument("--screen", choices=[label.value for label in FixtureLabel], help="Pre-label the entire capture session.")
    parser.add_argument("--label", choices=[label.value for label in FixtureLabel], help="Deprecated alias for --screen.")
    parser.add_argument("--interval", type=float, default=1.0, help="Capture interval in seconds.")
    parser.add_argument("--count", type=int, default=0, help="Optional number of captures. Use 0 to run until interrupted.")
    parser.add_argument("--device-id", help="ADB serial for the target device.")
    parser.add_argument("--debug", action="store_true", help="Save annotated debug screenshots alongside raw screenshots.")
    return parser.parse_args()


def relabel_fixture(
    fixture_json_path: str | None,
    fixture_png_path: str | None,
    debug_image_path: str | None,
    screen_label: str,
) -> tuple[Path | None, Path | None, Path | None]:
    fixture_json = Path(fixture_json_path) if fixture_json_path else None
    fixture_png = Path(fixture_png_path) if fixture_png_path else None
    debug_image = Path(debug_image_path) if debug_image_path else None
    target_dir = None
    for candidate in [fixture_json, fixture_png, debug_image]:
        if candidate is not None:
            target_dir = candidate.parent.parent / screen_label
            break
    if target_dir is None:
        return None, None, None
    target_dir.mkdir(parents=True, exist_ok=True)

    moved_json = _move_to_dir(fixture_json, target_dir)
    moved_png = _move_to_dir(fixture_png, target_dir)
    moved_debug = _move_debug_to_dir(debug_image, target_dir)
    if moved_json and moved_json.exists():
        payload = json.loads(moved_json.read_text(encoding="utf-8"))
        payload["manual_label"] = screen_label
        payload["expected_label"] = screen_label
        moved_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return moved_json, moved_png, moved_debug


def main() -> None:
    args = parse_args()
    session_label = args.screen or args.label
    settings = ControllerSettings(debug_mode=args.debug, adb_device_serial=args.device_id)
    adb = ADBClient(
        adb_path=settings.adb_path,
        device_serial=settings.adb_device_serial,
        command_retries=settings.adb_command_retries,
        retry_backoff_seconds=settings.adb_retry_backoff_seconds,
        default_timeout_seconds=settings.adb_default_timeout_seconds,
    )
    device = adb.connect()
    pipeline = PerceptionPipeline(
        adb_client=adb,
        selector_path=settings.selector_path,
        ocr_enabled=settings.enable_ocr,
        fixtures_live_dir=settings.fixtures_live_dir,
        debug_dir=settings.debug_dir,
        debug_enabled=args.debug,
    )

    print(
        f"Connected to {device.model} ({device.serial}). "
        f"Capturing every {args.interval:.2f}s. Press Ctrl+C to stop."
    )
    if session_label is None:
        print("Label keys: [h]ome [i]nbox [t]hread [k]eyboard-open thread [g]allery [p]ermission [u]nknown [s]kip [q]uit")

    captured = 0
    try:
        while args.count == 0 or captured < args.count:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
            screenshot_path = settings.screenshot_dir / f"fixture-{timestamp}.png"
            result = pipeline.capture_and_classify(screenshot_path=screenshot_path)
            label = session_label or prompt_for_label(result.classification.screen.value)
            if label == "quit":
                break
            if label != "skip":
                moved_json, moved_png, moved_debug = relabel_fixture(
                    result.fixture_json_path,
                    result.fixture_png_path,
                    result.debug_image_path,
                    label,
                )
                print(
                    f"[{captured + 1}] labeled={label} classifier={result.classification.screen.value} "
                    f"confidence={result.classification.confidence:.2f} "
                    f"png={moved_png or result.fixture_png_path} debug={moved_debug or result.debug_image_path}"
                )
            else:
                print(
                    f"[{captured + 1}] skipped classifier={result.classification.screen.value} "
                    f"confidence={result.classification.confidence:.2f}"
                )
            captured += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Capture stopped.")


def prompt_for_label(predicted_screen: str) -> str:
    try:
        import msvcrt

        print(f"Predicted={predicted_screen}. Press label key: ", end="", flush=True)
        key = msvcrt.getwch().lower()
        print(key)
    except Exception:
        raw = input(f"Predicted={predicted_screen}. Enter label key or screen name: ").strip().lower()
        key = raw
    if key in {"q", "quit"}:
        return "quit"
    if key in {"s", "skip", ""}:
        return "skip"
    if key in LABEL_KEYS:
        return LABEL_KEYS[key]
    if key in {label.value for label in FixtureLabel}:
        return key
    print(f"Unknown label key '{key}', capture skipped.")
    return "skip"


def _move_to_dir(path: Path | None, directory: Path) -> Path | None:
    if path is None or not path.exists():
        return None
    destination = directory / path.name
    if path.resolve() == destination.resolve():
        return path
    shutil.move(str(path), destination)
    return destination


def _move_debug_to_dir(path: Path | None, directory: Path) -> Path | None:
    if path is None or not path.exists():
        return None
    destination = directory / path.name
    if path.resolve() == destination.resolve():
        return path
    shutil.move(str(path), destination)
    return destination


if __name__ == "__main__":
    main()
