from __future__ import annotations

import argparse
from pathlib import Path

from libs.perception import FixtureLabel
from tools.collect_fixtures import relabel_fixture


def main() -> None:
    parser = argparse.ArgumentParser(description="Relabel existing fixture JSON/PNG pairs.")
    parser.add_argument("path", help="Path to a fixture JSON file or a directory of fixture JSON files.")
    parser.add_argument("label", choices=[label.value for label in FixtureLabel], help="New expected label.")
    args = parser.parse_args()

    target = Path(args.path)
    json_files = [target] if target.is_file() else sorted(target.rglob("*.json"))
    for json_file in json_files:
        png_file = json_file.with_suffix(".png")
        debug_file = json_file.with_name(f"{json_file.stem}-debug.png")
        relabel_fixture(
            fixture_json_path=str(json_file),
            fixture_png_path=str(png_file) if png_file.exists() else None,
            debug_image_path=str(debug_file) if debug_file.exists() else None,
            screen_label=args.label,
        )
        print(f"Relabeled {json_file} -> {args.label}")


if __name__ == "__main__":
    main()
