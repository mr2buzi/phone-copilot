from __future__ import annotations

import argparse
import json

from apps.controller.service import PhoneCopilotService
from apps.controller.settings import ControllerSettings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run supervised compose-only validation in Google Messages.")
    parser.add_argument("--text", required=True, help="Validation string to type into the compose box.")
    args = parser.parse_args()

    service = PhoneCopilotService(settings=ControllerSettings())
    state = service.run_compose_validation(args.text)
    print(json.dumps(state.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
