from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from libs.perception.classifier import ScreenClassifier
from libs.perception.fixtures import (
    FixtureLabel,
    expected_state_for_label,
    fixture_label_from_classification,
    load_fixture_metadata,
)


@dataclass
class FixtureEvaluationRecord:
    fixture_json_path: str
    expected_label: str
    predicted_label: str
    confidence: float
    package_name: str
    activity_name: str
    keyboard_visible: bool | None
    keyboard_height: int | None
    keyboard_ambiguous: bool
    passed: bool
    low_confidence: bool
    failure_category: str | None


def evaluate_fixture_directory(
    fixture_root: Path,
    selector_path: Path,
    report_dir: Path,
) -> dict[str, Any]:
    classifier = ScreenClassifier(selector_path=selector_path)
    records: list[FixtureEvaluationRecord] = []
    for json_path in sorted(fixture_root.rglob("*.json")):
        metadata = load_fixture_metadata(json_path)
        expected_label = metadata.get("expected_label") or json_path.parent.name
        expected_state = expected_state_for_label(str(expected_label))
        visual_features = metadata.get("debug_info", {}).get("visual_features", {})
        classification = classifier.classify(
            package_name=str(metadata["package_name"]),
            activity_name=str(metadata["activity_name"]),
            visible_text=list(metadata.get("visible_text", [])),
            screenshot_width=int(metadata["screenshot_width"]),
            screenshot_height=int(metadata["screenshot_height"]),
            keyboard_visible=metadata.get("keyboard_visible"),
            keyboard_height=metadata.get("keyboard_height"),
            keyboard_ambiguous=bool(metadata.get("keyboard_ambiguous", False)),
            visual_features=visual_features,
        )
        predicted_label = fixture_label_from_classification(
            screen=classification.screen,
            package_name=classification.package_name,
            keyboard_visible=classification.keyboard_visible,
        ).value
        low_confidence = classification.screen.value == "unknown_screen" or classification.confidence < 0.8
        passed = (
            predicted_label == expected_state.label.value
            and (
                expected_state.keyboard_visible is None
                or classification.keyboard_visible == expected_state.keyboard_visible
            )
        )
        failure_category = None
        if classification.keyboard_ambiguous:
            failure_category = "keyboard_state_ambiguous"
        elif not passed and low_confidence:
            failure_category = "classifier_low_confidence"
        elif not passed:
            failure_category = "classifier_wrong"
        records.append(
            FixtureEvaluationRecord(
                fixture_json_path=str(json_path),
                expected_label=expected_state.label.value,
                predicted_label=predicted_label,
                confidence=classification.confidence,
                package_name=classification.package_name,
                activity_name=classification.activity_name,
                keyboard_visible=classification.keyboard_visible,
                keyboard_height=classification.keyboard_height,
                keyboard_ambiguous=classification.keyboard_ambiguous,
                passed=passed,
                low_confidence=low_confidence,
                failure_category=failure_category,
            )
        )

    report = build_evaluation_report(records)
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    json_path = report_dir / f"fixture-evaluation-{timestamp}.json"
    csv_path = report_dir / f"fixture-evaluation-{timestamp}.csv"
    report["artifacts"] = {"json": str(json_path), "csv": str(csv_path)}
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(csv_path, records)
    return report


def build_evaluation_report(records: list[FixtureEvaluationRecord]) -> dict[str, Any]:
    total = len(records)
    by_screen: dict[str, dict[str, Any]] = {}
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    misclassified: list[dict[str, Any]] = []
    low_confidence: list[dict[str, Any]] = []
    ambiguity_count = 0
    grouped: dict[str, list[FixtureEvaluationRecord]] = defaultdict(list)
    for record in records:
        grouped[record.expected_label].append(record)
        confusion[record.expected_label][record.predicted_label] += 1
        if not record.passed:
            misclassified.append(record.__dict__)
        if record.low_confidence:
            low_confidence.append(record.__dict__)
        if record.keyboard_ambiguous:
            ambiguity_count += 1
    for expected_label, items in grouped.items():
        passes = sum(1 for item in items if item.passed)
        by_screen[expected_label] = {
            "count": len(items),
            "passes": passes,
            "accuracy": (passes / len(items)) if items else 0.0,
            "low_confidence": sum(1 for item in items if item.low_confidence),
        }
    failure_counts = Counter(record.failure_category for record in records if record.failure_category)
    return {
        "summary": {
            "total": total,
            "passed": sum(1 for record in records if record.passed),
            "failed": sum(1 for record in records if not record.passed),
            "keyboard_ambiguity_count": ambiguity_count,
            "failure_counts": dict(failure_counts),
        },
        "by_screen": by_screen,
        "confusion": {expected: dict(predicted) for expected, predicted in confusion.items()},
        "misclassified": misclassified,
        "low_confidence": low_confidence,
    }


def _write_csv(path: Path, records: list[FixtureEvaluationRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FixtureEvaluationRecord.__annotations__.keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)
