from __future__ import annotations

import argparse
import json

from apps.controller.service import PhoneCopilotService
from apps.controller.settings import ControllerSettings


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a message in Google Messages and wait for a visible reply.")
    parser.add_argument("--text", required=True, help="Exact message text to send.")
    parser.add_argument(
        "--wait-timeout-seconds",
        type=float,
        default=90.0,
        help="How long to wait for a visible reply in the open thread.",
    )
    args = parser.parse_args()

    service = PhoneCopilotService(settings=ControllerSettings())
    state = service.run_send_and_read(args.text, wait_timeout_seconds=args.wait_timeout_seconds)
    print(json.dumps(state.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
