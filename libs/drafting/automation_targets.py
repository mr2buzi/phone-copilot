from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

DEFAULT_AUTOMATION_TARGETS_PATH = Path("data/automation_targets.json")


class AutomationTarget(BaseModel):
    name: str
    enabled: bool = True
    auto_send: bool = False
    max_messages_per_hour: int = Field(default=4, ge=1, le=50)


class AutomationTargetsConfig(BaseModel):
    enabled: bool = False
    mode: str = "review_only"
    targets: list[AutomationTarget] = Field(default_factory=list)
    global_limits: dict[str, Any] = Field(
        default_factory=lambda: {
            "max_contacts_per_cycle": 5,
            "max_total_sends_per_hour": 10,
            "cooldown_between_contacts_ms": 1500,
        }
    )


def ensure_automation_targets(path: Path = DEFAULT_AUTOMATION_TARGETS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        save_automation_targets(path, AutomationTargetsConfig())


def load_automation_targets(path: Path = DEFAULT_AUTOMATION_TARGETS_PATH) -> AutomationTargetsConfig:
    ensure_automation_targets(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return AutomationTargetsConfig()
    if not isinstance(payload, dict):
        return AutomationTargetsConfig()
    try:
        return AutomationTargetsConfig(**payload)
    except Exception:
        return AutomationTargetsConfig()


def save_automation_targets(path: Path, config: AutomationTargetsConfig | dict[str, Any]) -> None:
    payload = config.model_dump(mode="json") if isinstance(config, AutomationTargetsConfig) else config
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp_path.replace(path)
