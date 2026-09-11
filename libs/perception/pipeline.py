from __future__ import annotations

import json
import shutil
import hashlib
import time
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageStat

from libs.adb import ADBClientProtocol
from libs.perception.classifier import ScreenClassifier
from libs.perception.models import PerceptionResult, PerceptionTimings
from libs.perception.ocr import OCRExtractor


class PerceptionPipeline:
    def __init__(
        self,
        adb_client: ADBClientProtocol,
        selector_path: Path | None = None,
        ocr_enabled: bool = True,
        fixtures_live_dir: Path | None = None,
        debug_dir: Path | None = None,
        debug_enabled: bool = False,
    ) -> None:
        self.adb_client = adb_client
        self.ocr = OCRExtractor(enabled=ocr_enabled)
        self.classifier = ScreenClassifier(selector_path=selector_path)
        self.fixtures_live_dir = fixtures_live_dir
        self.debug_dir = debug_dir
        self.debug_enabled = debug_enabled
        self._last_screenshot_hash: str | None = None
        self._last_visible_text: list[str] | None = None

    def capture_and_classify(self, screenshot_path: Path) -> PerceptionResult:
        timings = PerceptionTimings()
        detailed: dict[str, float] = {}

        started = time.perf_counter()
        screenshot_bytes = self.adb_client.take_screenshot(save_path=screenshot_path)
        timings.screenshot_time_ms = (time.perf_counter() - started) * 1000
        detailed["screenshot_start"] = 0.0
        detailed["screenshot_end"] = timings.screenshot_time_ms
        screenshot_hash = hashlib.sha1(screenshot_bytes).hexdigest()

        device_started = time.perf_counter()
        started = time.perf_counter()
        foreground_app = self.adb_client.get_foreground_app()
        timings.foreground_app_time_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        keyboard_state = self.adb_client.get_keyboard_state()
        timings.keyboard_state_time_ms = (time.perf_counter() - started) * 1000
        timings.device_check_time_ms = (time.perf_counter() - device_started) * 1000

        started = time.perf_counter()
        image = Image.open(BytesIO(screenshot_bytes)).convert("RGB")
        timings.screenshot_decode_time_ms = (time.perf_counter() - started) * 1000

        if self.ocr.enabled and screenshot_hash == self._last_screenshot_hash and self._last_visible_text is not None:
            visible_text = list(self._last_visible_text)
            timings.ocr_time_ms = 0.0
            detailed["ocr_cache_hit"] = 1.0
        else:
            started = time.perf_counter()
            visible_text = self.ocr.extract_lines(screenshot_bytes)
            timings.ocr_time_ms = (time.perf_counter() - started) * 1000
            self._last_screenshot_hash = screenshot_hash
            self._last_visible_text = list(visible_text)
            detailed["ocr_cache_hit"] = 0.0

        started = time.perf_counter()
        visual_features = self._compute_visual_features(image)
        timings.visual_features_time_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        classification = self.classifier.classify(
            package_name=foreground_app.package_name,
            activity_name=foreground_app.activity_name,
            visible_text=visible_text,
            screenshot_width=image.width,
            screenshot_height=image.height,
            keyboard_visible=keyboard_state.visible,
            keyboard_height=keyboard_state.height,
            keyboard_ambiguous=keyboard_state.ambiguous,
            visual_features=visual_features,
        )
        timings.classification_time_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        fixture_json_path, fixture_png_path = self._persist_fixture(
            screenshot_path=screenshot_path,
            classification=classification,
        )
        timings.fixture_persist_time_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        debug_image_path = self._save_debug_image(
            image=image,
            screenshot_path=screenshot_path,
            classification=classification,
        )
        timings.debug_image_time_ms = (time.perf_counter() - started) * 1000
        timings.adb_total_time_ms = timings.screenshot_time_ms + timings.device_check_time_ms
        detailed.update(
            {
                "device_check": timings.device_check_time_ms,
                "foreground_app": timings.foreground_app_time_ms,
                "keyboard_state": timings.keyboard_state_time_ms,
                "screenshot_decode": timings.screenshot_decode_time_ms,
                "ocr": timings.ocr_time_ms,
                "visual_features": timings.visual_features_time_ms,
                "classification": timings.classification_time_ms,
                "fixture_persist": timings.fixture_persist_time_ms,
                "debug_image": timings.debug_image_time_ms,
                "adb_total": timings.adb_total_time_ms,
            }
        )
        timings.detailed = {key: round(value, 2) for key, value in detailed.items()}

        return PerceptionResult(
            classification=classification,
            screenshot_path=str(screenshot_path),
            fixture_json_path=str(fixture_json_path) if fixture_json_path else None,
            fixture_png_path=str(fixture_png_path) if fixture_png_path else None,
            debug_image_path=str(debug_image_path) if debug_image_path else None,
            timings=timings,
        )

    def invalidate_ocr_cache(self) -> None:
        self._last_screenshot_hash = None
        self._last_visible_text = None

    def _persist_fixture(self, screenshot_path: Path, classification) -> tuple[Path | None, Path | None]:
        if self.fixtures_live_dir is None:
            return None, None
        timestamp = screenshot_path.stem.replace("capture-", "")
        target_dir = self.fixtures_live_dir / classification.screen.value
        target_dir.mkdir(parents=True, exist_ok=True)
        fixture_png_path = target_dir / f"{timestamp}.png"
        fixture_json_path = target_dir / f"{timestamp}.json"
        shutil.copyfile(screenshot_path, fixture_png_path)
        fixture_json_path.write_text(
            json.dumps(
                {
                    "package_name": classification.package_name,
                    "activity_name": classification.activity_name,
                    "screen": classification.screen.value,
                    "confidence": classification.confidence,
                    "visible_text": classification.visible_text,
                    "keyboard_visible": classification.keyboard_visible,
                    "keyboard_height": classification.keyboard_height,
                    "keyboard_ambiguous": classification.keyboard_ambiguous,
                    "features_used": classification.features_used,
                    "debug_info": classification.debug_info,
                    "screenshot_width": classification.screenshot_width,
                    "screenshot_height": classification.screenshot_height,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return fixture_json_path, fixture_png_path

    def _save_debug_image(self, image: Image.Image, screenshot_path: Path, classification) -> Path | None:
        if not self.debug_enabled or self.debug_dir is None:
            return None
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        header = (
            f"{classification.screen.value} "
            f"confidence={classification.confidence:.2f} "
            f"keyboard={classification.keyboard_visible}"
        )
        draw.rectangle((12, 12, annotated.width - 12, 112), outline="orange", width=3)
        draw.text((24, 24), header, fill="orange")
        key_text = " | ".join(classification.visible_text[:3]) if classification.visible_text else "no text"
        draw.text((24, 62), key_text[:140], fill="orange")
        for action in classification.available_actions:
            if action.region is not None:
                draw.rectangle(
                    (
                        action.region.left,
                        action.region.top,
                        action.region.right,
                        action.region.bottom,
                    ),
                    outline="deepskyblue",
                    width=3,
                )
            if action.point is not None:
                draw.ellipse(
                    (
                        action.point.x - 12,
                        action.point.y - 12,
                        action.point.x + 12,
                        action.point.y + 12,
                    ),
                    outline="red",
                    width=3,
                )
        if classification.keyboard_visible and classification.keyboard_height:
            draw.rectangle(
                (
                    0,
                    max(0, annotated.height - classification.keyboard_height),
                    annotated.width,
                    annotated.height,
                ),
                outline="yellow",
                width=3,
            )
        debug_path = self.debug_dir / f"{screenshot_path.stem}-debug.png"
        annotated.save(debug_path)
        return debug_path

    def _compute_visual_features(self, image: Image.Image) -> dict[str, float]:
        grayscale = image.convert("L")
        width, height = grayscale.size
        top_band = grayscale.crop((0, 0, width, max(1, int(height * 0.12))))
        bottom_band = grayscale.crop((0, int(height * 0.82), width, height))
        center_band = grayscale.crop((int(width * 0.2), int(height * 0.2), int(width * 0.8), int(height * 0.8)))
        top_stats = ImageStat.Stat(top_band)
        bottom_stats = ImageStat.Stat(bottom_band)
        center_stats = ImageStat.Stat(center_band)
        return {
            "top_band_brightness": float(top_stats.mean[0]),
            "bottom_band_variance": float(bottom_stats.var[0]),
            "center_variance": float(center_stats.var[0]),
        }
