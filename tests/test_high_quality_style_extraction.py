from __future__ import annotations

import json
from pathlib import Path

from libs.drafting.service import DraftingService
from scripts.extract_high_quality_style_examples import extract_high_quality_examples, write_high_quality_examples


def test_extract_high_quality_examples_keeps_owner_reply_direction(tmp_path: Path) -> None:
    chat_dir = tmp_path / "training"
    training_messages = tmp_path / "training_messages"
    chat_dir.mkdir()
    training_messages.mkdir()
    (training_messages / "corrections.jsonl").write_text("", encoding="utf-8")
    (chat_dir / "WhatsApp Chat with Alice.txt").write_text(
        "\n".join(
            [
                "2026-05-01, 10:00 - Alice: guess what",
                "2026-05-01, 10:01 - Alex: what happened",
                "2026-05-01, 10:02 - Alice: someone ran in my living room",
                "2026-05-01, 10:03 - Alex: BRO WHAT why was he running",
            ]
        ),
        encoding="utf-8",
    )

    rows, summary = extract_high_quality_examples(
        chat_dir=chat_dir,
        training_messages_dir=training_messages,
        owner_aliases=["Alex"],
        limit=10,
    )

    replies = {row["incoming"]: row["my_reply"] for row in rows}
    assert replies["guess what"] == "what happened"
    assert replies["someone ran in my living room"] == "BRO WHAT why was he running"
    assert summary["selected_count"] == 2


def test_extract_high_quality_examples_skips_sensitive_rows(tmp_path: Path) -> None:
    chat_dir = tmp_path / "training"
    training_messages = tmp_path / "training_messages"
    chat_dir.mkdir()
    training_messages.mkdir()
    (training_messages / "corrections.jsonl").write_text("", encoding="utf-8")
    (chat_dir / "WhatsApp Chat with Alice.txt").write_text(
        "\n".join(
            [
                "2026-05-01, 10:00 - Alice: my otp is 123456",
                "2026-05-01, 10:01 - Alex: dont send that here",
                "2026-05-01, 10:02 - Alice: you coming later?",
                "2026-05-01, 10:03 - Alex: yh what time",
            ]
        ),
        encoding="utf-8",
    )

    rows, _summary = extract_high_quality_examples(
        chat_dir=chat_dir,
        training_messages_dir=training_messages,
        owner_aliases=["Alex"],
        limit=10,
    )

    assert all("otp" not in json.dumps(row).casefold() for row in rows)
    assert {row["my_reply"] for row in rows} == {"yh what time"}


def test_write_high_quality_examples_outputs_jsonl_and_summary(tmp_path: Path) -> None:
    chat_dir = tmp_path / "training"
    training_messages = tmp_path / "training_messages"
    out_dir = tmp_path / "out"
    chat_dir.mkdir()
    training_messages.mkdir()
    (training_messages / "corrections.jsonl").write_text("", encoding="utf-8")
    (chat_dir / "WhatsApp Chat with Alice.txt").write_text(
        "\n".join(
            [
                "2026-05-01, 10:00 - Alice: wyd",
                "2026-05-01, 10:01 - Alex: js working icl wby",
            ]
        ),
        encoding="utf-8",
    )

    summary = write_high_quality_examples(
        chat_dir=chat_dir,
        training_messages_dir=training_messages,
        out_dir=out_dir,
        owner_aliases=["Alex"],
        limit=10,
    )

    assert Path(summary["examples_path"]).exists()
    assert Path(summary["summary_path"]).exists()
    assert summary["selected_count"] == 1


def test_drafting_service_loads_high_quality_style_examples(tmp_path: Path) -> None:
    training_messages = tmp_path / "training_messages"
    training_messages.mkdir()
    (training_messages / "high_quality_style_examples.jsonl").write_text(
        json.dumps(
            {
                "relationship_type": "close_friend",
                "incoming": "guess what",
                "context": [],
                "my_reply": "what happened",
                "intent_type": "unknown",
                "quality_score": 0.91,
                "style_authority": "high",
                "contains_sensitive": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    service = DraftingService(
        template_path=tmp_path / "missing.json",
        approved_photos_dir=tmp_path / "approved",
        training_messages_dir=training_messages,
    )

    assert any(row.get("_source") == "high_quality_style" and row.get("my_reply") == "what happened" for row in service._training_rows)
