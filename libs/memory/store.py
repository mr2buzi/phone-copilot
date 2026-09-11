from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlmodel import Field, Session, SQLModel, create_engine, select


class ActionLogEntry(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), index=True)
    event_type: str
    before_screenshot_path: str | None = None
    after_screenshot_path: str | None = None
    detected_screen: str
    confidence: float
    planner_decision: str
    approval_decision: str
    execution_result: str
    metadata_json: str = "{}"


class LogStore:
    def __init__(self, db_path: Path) -> None:
        self.engine = create_engine(f"sqlite:///{db_path}")
        SQLModel.metadata.create_all(self.engine)

    def record_event(
        self,
        event_type: str,
        detected_screen: str,
        confidence: float,
        planner_decision: str,
        approval_decision: str,
        execution_result: str,
        before_screenshot_path: str | None = None,
        after_screenshot_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with Session(self.engine) as session:
            session.add(
                ActionLogEntry(
                    event_type=event_type,
                    before_screenshot_path=before_screenshot_path,
                    after_screenshot_path=after_screenshot_path,
                    detected_screen=detected_screen,
                    confidence=confidence,
                    planner_decision=planner_decision,
                    approval_decision=approval_decision,
                    execution_result=execution_result,
                    metadata_json=json.dumps(metadata or {}, sort_keys=True),
                )
            )
            session.commit()

    def list_recent(self, limit: int) -> list[dict[str, Any]]:
        with Session(self.engine) as session:
            rows = session.exec(
                select(ActionLogEntry).order_by(ActionLogEntry.timestamp.desc()).limit(limit)
            ).all()
        return [
            {
                "id": row.id,
                "timestamp": row.timestamp.isoformat(),
                "event_type": row.event_type,
                "before_screenshot_path": row.before_screenshot_path,
                "after_screenshot_path": row.after_screenshot_path,
                "detected_screen": row.detected_screen,
                "confidence": row.confidence,
                "planner_decision": row.planner_decision,
                "approval_decision": row.approval_decision,
                "execution_result": row.execution_result,
                "metadata": json.loads(row.metadata_json),
            }
            for row in rows
        ]
