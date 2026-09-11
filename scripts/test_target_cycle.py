from __future__ import annotations

import json
from pathlib import Path

from apps.controller.service import PhoneCopilotService
from apps.controller.settings import ControllerSettings
from apps.controller.models import ControllerState, ThreadContext
from libs.planners import PlannerDecision
from libs.screen_states import ScreenClassification, ScreenName


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    write_json(Path("data/automation_targets.json"), {"enabled": False, "mode": "review_only", "targets": []})
    service = PhoneCopilotService(ControllerSettings(enable_ocr=False))
    disabled = service.run_target_cycle(dry_run=True)
    assert disabled["status"] == "skipped"

    write_json(
        Path("data/automation_targets.json"),
        {
            "enabled": True,
            "mode": "draft_only",
            "targets": [{"name": "Ali", "enabled": True, "auto_send": False, "max_messages_per_hour": 4}],
            "global_limits": {"max_contacts_per_cycle": 5, "max_total_sends_per_hour": 10, "cooldown_between_contacts_ms": 0},
        },
    )
    write_json(
        Path("data/contact_profiles.json"),
        {"Ali": {"relationship_type": "close_friend", "auto_send_allowed": False, "auto_draft_allowed": True, "max_reply_length": 20}},
    )
    state = ControllerState(
        captured_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        classification=ScreenClassification(
            app="Google Messages",
            screen=ScreenName.THREAD_VIEW,
            confidence=0.99,
            package_name="com.google.android.apps.messaging",
            activity_name="Messages",
            screenshot_width=1080,
            screenshot_height=2400,
            recent_messages=["you coming later?"],
        ),
        planner_decision=PlannerDecision(status="ok", reason="synthetic", allowed_actions=[]),
        summary="synthetic",
        screenshot_path="synthetic.png",
    )
    state.thread_context = ThreadContext(
        contact_name="Ali",
        recent_messages=["you coming later?"],
        full_conversation=[{"speaker": "other", "text": "you coming later?"}],
        message_count=1,
    )
    service.latest_state = state
    dry = service.run_target_cycle(dry_run=True)
    assert dry["dry_run"] is True
    assert dry["results"]
    assert dry["results"][0]["status"] in {"drafted", "review"}
    assert dry["results"][0]["selected_reply"]

    duplicate = service.run_target_cycle(dry_run=True)
    assert duplicate["results"][0]["status"] in {"drafted", "review", "skipped"}
    print("target_cycle_checks=passed")
    print(json.dumps(dry, indent=2))


if __name__ == "__main__":
    main()
