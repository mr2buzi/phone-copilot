from __future__ import annotations

from pathlib import Path

from tools.bootstrap_reply_quality_data import extract_rows


def test_whatsapp_bootstrap_keeps_incoming_and_my_reply_direction(tmp_path: Path) -> None:
    training_dir = tmp_path / "training"
    training_dir.mkdir()
    (training_dir / "WhatsApp Chat with Alice.txt").write_text(
        "\n".join(
            [
                "2026-05-01, 10:00 - Alice: you coming?",
                "2026-05-01, 10:01 - Alex: yh what time",
            ]
        ),
        encoding="utf-8",
    )

    rows = extract_rows(training_dir, {"alex"})
    generated = rows["casual_friend"][0]

    assert generated["incoming"] == "you coming?"
    assert generated["my_reply"] == "yh what time"


def test_whatsapp_bootstrap_skips_when_owner_reply_is_uncertain(tmp_path: Path) -> None:
    training_dir = tmp_path / "training"
    training_dir.mkdir()
    (training_dir / "WhatsApp Chat with Alice.txt").write_text(
        "\n".join(
            [
                "2026-05-01, 10:00 - Alice: you coming?",
                "2026-05-01, 10:01 - Bob: yh what time",
            ]
        ),
        encoding="utf-8",
    )
    skipped: dict[str, int] = {}

    rows = extract_rows(training_dir, {"alex"}, skipped=skipped)

    assert rows["casual_friend"] == []
    assert skipped["no_confident_owner_reply"] == 1
