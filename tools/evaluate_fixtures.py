from __future__ import annotations

import argparse
import json
from pathlib import Path

from apps.controller.settings import ControllerSettings
from libs.perception import evaluate_fixture_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved live fixtures against the current classifier.")
    parser.add_argument("--fixtures", default="data/fixtures/live", help="Fixture root directory.")
    parser.add_argument("--out", default="data/logs/reports", help="Report output directory.")
    args = parser.parse_args()

    settings = ControllerSettings()
    report = evaluate_fixture_directory(
        fixture_root=Path(args.fixtures),
        selector_path=settings.selector_path,
        report_dir=Path(args.out),
    )
    print(json.dumps(report["summary"], indent=2))
    print(f"Reports saved to {report['artifacts']['json']} and {report['artifacts']['csv']}")


if __name__ == "__main__":
    main()
