from __future__ import annotations

import json
from pathlib import Path

from libs.perception import evaluate_fixture_directory


def test_fixture_evaluation_generates_report(tmp_path: Path) -> None:
    source = Path("tests/fixtures/screens/thread_view.json")
    target_dir = tmp_path / "thread_view"
    target_dir.mkdir(parents=True)
    fixture_path = target_dir / "sample.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["expected_label"] = "thread_view"
    payload["keyboard_visible"] = False
    payload["keyboard_height"] = None
    payload["keyboard_ambiguous"] = False
    payload["debug_info"] = {
        "visual_features": {
            "top_band_brightness": 180.0,
            "bottom_band_variance": 600.0,
            "center_variance": 400.0,
        }
    }
    fixture_path.write_text(json.dumps(payload), encoding="utf-8")
    report = evaluate_fixture_directory(
        fixture_root=tmp_path,
        selector_path=Path("data/selectors/screen_signatures.json"),
        report_dir=tmp_path / "reports",
    )
    assert report["summary"]["total"] == 1
    assert report["summary"]["passed"] == 1
    assert Path(report["artifacts"]["json"]).exists()
    assert Path(report["artifacts"]["csv"]).exists()


def test_fixture_evaluation_flags_low_confidence(tmp_path: Path) -> None:
    source = Path("tests/fixtures/screens/thread_view.json")
    target_dir = tmp_path / "thread_view"
    target_dir.mkdir(parents=True)
    fixture_path = target_dir / "sample.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["expected_label"] = "thread_view"
    payload["visible_text"] = ["Lunch tomorrow?"]
    payload["keyboard_visible"] = False
    payload["keyboard_height"] = None
    payload["keyboard_ambiguous"] = False
    payload["debug_info"] = {"visual_features": {"top_band_brightness": 80.0, "bottom_band_variance": 0.0, "center_variance": 0.0}}
    fixture_path.write_text(json.dumps(payload), encoding="utf-8")
    report = evaluate_fixture_directory(
        fixture_root=tmp_path,
        selector_path=Path("data/selectors/screen_signatures.json"),
        report_dir=tmp_path / "reports",
    )
    assert report["summary"]["failed"] == 1
    assert report["low_confidence"]
