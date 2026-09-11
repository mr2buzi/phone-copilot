from __future__ import annotations

import argparse
from pathlib import Path

from libs.drafting.style_profile import build_style_profile_from_training_dir, write_style_profile


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a phone-copilot style profile from WhatsApp exports.")
    parser.add_argument("--training-dir", default="training", help="Directory containing WhatsApp .txt exports.")
    parser.add_argument("--output", default="data/style_profile.json", help="Destination JSON file.")
    parser.add_argument(
        "--owner-aliases",
        default="Owner",
        help="Comma-separated sender names that belong to the phone owner inside the exports.",
    )
    args = parser.parse_args()

    aliases = [alias.strip() for alias in args.owner_aliases.split(",") if alias.strip()]
    profile = build_style_profile_from_training_dir(Path(args.training_dir), aliases)
    if profile is None:
        raise SystemExit("No owner messages were found in the provided training exports.")

    output_path = Path(args.output)
    write_style_profile(profile, output_path)
    print(f"Wrote style profile to {output_path}")
    print(f"Messages analysed: {profile.source_message_count}")
    print(f"Tone traits: {', '.join(profile.tone_traits)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
