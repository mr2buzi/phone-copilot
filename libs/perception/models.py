from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from libs.screen_states import ScreenClassification


class PerceptionTimings(BaseModel):
    screenshot_time_ms: float = 0.0
    ocr_time_ms: float = 0.0
    classification_time_ms: float = 0.0
    screenshot_decode_time_ms: float = 0.0
    device_check_time_ms: float = 0.0
    foreground_app_time_ms: float = 0.0
    keyboard_state_time_ms: float = 0.0
    visual_features_time_ms: float = 0.0
    fixture_persist_time_ms: float = 0.0
    debug_image_time_ms: float = 0.0
    adb_total_time_ms: float = 0.0
    detailed: dict[str, float] = Field(default_factory=dict)


class PerceptionResult(BaseModel):
    classification: ScreenClassification
    screenshot_path: str
    fixture_json_path: str | None = None
    fixture_png_path: str | None = None
    debug_image_path: str | None = None
    timings: PerceptionTimings = Field(default_factory=PerceptionTimings)

    @property
    def screenshot_file(self) -> Path:
        return Path(self.screenshot_path)
