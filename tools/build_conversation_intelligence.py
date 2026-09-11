from __future__ import annotations

import argparse
from pathlib import Path

from apps.training import ConversationIntelligenceStore, rebuild_conversation_intelligence


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the phone-copilot conversation intelligence database.")
    parser.add_argument("--training-dir", default="training", help="Directory containing WhatsApp .txt exports.")
    parser.add_argument("--db-path", default="data/conversation_intelligence.db", help="SQLite output path.")
    parser.add_argument(
        "--owner-aliases",
        default="Owner",
        help="Comma-separated sender names that belong to the phone owner inside the exports.",
    )
    args = parser.parse_args()

    aliases = [alias.strip() for alias in args.owner_aliases.split(",") if alias.strip()]
    store = ConversationIntelligenceStore(Path(args.db_path))
    summary = rebuild_conversation_intelligence(
        training_dir=Path(args.training_dir),
        owner_aliases=aliases,
        store=store,
    )
    print(f"Wrote conversation intelligence to {args.db_path}")
    print(f"Chats imported: {summary.chats_imported}")
    print(f"Messages imported: {summary.messages_imported}")
    print(f"Reply examples built: {summary.reply_examples_built}")
    print(f"Personas: {summary.personas}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
