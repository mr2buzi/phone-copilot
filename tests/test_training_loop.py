from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from apps.controller.models import StyleReviewApproveRequest
from apps.controller.settings import ControllerSettings
from apps.controller.service import PhoneCopilotService
from libs.drafting.training_data import ensure_training_message_files
from libs.training import DEFAULT_TRAINING_LOOP_CONFIG, TrainingLoopService


@pytest.fixture()
def isolated_training_service(tmp_path: Path, mock_adb) -> PhoneCopilotService:
    training_messages = tmp_path / "training_messages"
    ensure_training_message_files(training_messages)
    settings = ControllerSettings(
        enable_ocr=False,
        ai_reply_enabled=False,
        data_dir=tmp_path,
        log_db_path=tmp_path / "phone_copilot.db",
        screenshot_dir=tmp_path / "screenshots",
        debug_dir=tmp_path / "debug",
        fixtures_live_dir=tmp_path / "fixtures",
        approved_photos_dir=tmp_path / "approved_photos",
        reply_template_path=Path("data/templates/reply_templates.json"),
        selector_path=Path("data/selectors/screen_signatures.json"),
        ai_reply_training_messages_dir=training_messages,
        ai_reply_training_dir=tmp_path / "training",
        ai_reply_style_profile_path=tmp_path / "style_profile.json",
        contact_overrides_path=tmp_path / "contact_overrides.json",
        ai_reply_intelligence_db_path=tmp_path / "conversation_intelligence.db",
    )
    return PhoneCopilotService(settings=settings, adb_client=mock_adb)


def test_training_loop_config_defaults_and_auto_approve_is_forced_false(tmp_path: Path) -> None:
    loop = TrainingLoopService(data_dir=tmp_path, repo_root=Path.cwd())

    config = loop.load_config()
    assert config["enabled"] is False
    assert config["mode"] == "review_only"
    assert config["auto_approve"] is False

    config_path = tmp_path / "training_loop_config.json"
    config_path.write_text(json.dumps({**DEFAULT_TRAINING_LOOP_CONFIG, "auto_approve": True}), encoding="utf-8")
    config = loop.load_config()

    assert config["auto_approve"] is False
    assert "ignored" in json.dumps(loop.status().get("warnings", []))


def test_training_loop_start_stop_and_status(tmp_path: Path) -> None:
    loop = TrainingLoopService(data_dir=tmp_path, repo_root=Path.cwd())

    started = loop.start()
    assert started["enabled"] is True
    assert started["config"]["auto_approve"] is False

    stopped = loop.stop()
    assert stopped["enabled"] is False


def test_training_loop_run_now_creates_status_without_adb(tmp_path: Path, monkeypatch) -> None:
    loop = TrainingLoopService(data_dir=tmp_path, repo_root=Path.cwd())
    calls: list[list[str]] = []

    def fake_run_command(args: list[str]) -> dict[str, object]:
        calls.append(args)
        if "evaluate_reply_style.py" in args:
            report = tmp_path / "reports" / "reply_style_eval_report.json"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps({"total_cases": 3, "fail_count": 2}), encoding="utf-8")
        if "build_active_learning_queue.py" in args:
            active = tmp_path / "training_messages" / "active_learning_queue.jsonl"
            active.parent.mkdir(parents=True, exist_ok=True)
            active.write_text(json.dumps({"id": "active_1"}) + "\n", encoding="utf-8")
        return {"status": "ok", "exit_code": 0, "stdout": "{}", "stderr": ""}

    monkeypatch.setattr(loop, "_run_command", fake_run_command)

    result = loop.run_now()

    assert result["status"] == "completed"
    assert loop.status()["total_runs"] == 1
    assert calls
    joined = " ".join(" ".join(call) for call in calls)
    assert "adb" not in joined.lower()


def test_training_loop_overlapping_runs_blocked(tmp_path: Path, monkeypatch) -> None:
    loop = TrainingLoopService(data_dir=tmp_path, repo_root=Path.cwd())

    def slow_run_command(args: list[str]) -> dict[str, object]:
        time.sleep(0.2)
        return {"status": "ok", "exit_code": 0, "stdout": "{}", "stderr": ""}

    monkeypatch.setattr(loop, "_run_command", slow_run_command)
    first = loop.run_now(background=True)
    second = loop.run_now()

    assert first["status"] == "started"
    assert second["status"] == "already_running"


def test_training_loop_errors_are_stored_not_raised(tmp_path: Path, monkeypatch) -> None:
    loop = TrainingLoopService(data_dir=tmp_path, repo_root=Path.cwd())

    def broken_command(args: list[str]) -> dict[str, object]:
        raise RuntimeError("eval exploded")

    monkeypatch.setattr(loop, "_run_command", broken_command)

    result = loop.run_once()

    assert result["status"] == "error"
    assert "eval exploded" in str(loop.status()["last_error"])


def test_training_loop_service_methods_on_controller(isolated_training_service: PhoneCopilotService) -> None:
    training_service = isolated_training_service
    config = training_service.training_loop_update_config({"auto_approve": True, "interval_minutes": 15})

    assert config["config"]["auto_approve"] is False
    assert config["config"]["interval_minutes"] == 15

    started = training_service.training_loop_start()
    assert started["enabled"] is True
    stopped = training_service.training_loop_stop()
    assert stopped["enabled"] is False


def test_style_review_approval_schedules_debounced_rebuild(isolated_training_service: PhoneCopilotService, monkeypatch) -> None:
    training_service = isolated_training_service
    review_path = training_service.style_review_queue_path
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(
        json.dumps(
            {
                "row_id": "review_rebuild_1",
                "timestamp": "2026-05-20T00:00:00Z",
                "incoming": "you coming later?",
                "context": [],
                "relationship_type": "close_friend",
                "intent_type": "planning",
                "contact_name": "Ali",
                "bad_ai_reply": "maybe later",
                "suggested_better_reply": "yh what time",
                "style_score": 92,
                "reason_bad": "invents availability or facts",
                "improvement_notes": [],
                "source": "auto_style_improvement",
                "status": "needs_human_review",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    scheduled: list[bool] = []
    monkeypatch.setattr(training_service.training_loop, "schedule_rebuild_after_approval", lambda: scheduled.append(True))

    response = training_service.training_style_review_approve(StyleReviewApproveRequest(row_id="review_rebuild_1"))

    assert response["vector_rebuild_required"] is True
    assert scheduled == [True]
