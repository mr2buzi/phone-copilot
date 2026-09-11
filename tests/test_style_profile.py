from pathlib import Path

from libs.drafting.style_profile import build_style_profile_from_training_dir


def test_build_style_profile_from_training_dir_extracts_owner_style(tmp_path: Path) -> None:
    training_dir = tmp_path / "training"
    training_dir.mkdir()
    export_path = training_dir / "chat.txt"
    export_path.write_text(
        "\n".join(
            [
                "2026-04-10, 3:21 a.m. - Owner: i miss u baby",
                "2026-04-10, 3:22 a.m. - Friend: aww",
                "2026-04-10, 3:23 a.m. - Owner: ur such a cutie",
                "2026-04-10, 3:24 a.m. - Owner: dont worry ill make it work",
                "2026-04-10, 3:25 a.m. - Owner: BRO WHAT",
                "2026-04-10, 3:26 a.m. - Owner: this is line one",
                "line two",
            ]
        ),
        encoding="utf-8",
    )

    profile = build_style_profile_from_training_dir(training_dir, ["Owner"])

    assert profile is not None
    assert profile.source_message_count == 5
    assert "affectionate and lightly flirty" in profile.tone_traits
    assert "baby" in profile.affectionate_terms
    assert profile.safe_example_messages
    assert any("cutie" in example.casefold() or "dont worry" in example.casefold() for example in profile.safe_example_messages)
    assert "slight flirt" in profile.guidance_summary
