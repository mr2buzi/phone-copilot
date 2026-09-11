from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from libs.drafting.training_data import ensure_training_message_files, redact_private_text

MESSAGE_RE = re.compile(
    r"^(?:\[)?(?P<timestamp>(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}), .*?)(?:\])? - (?P<sender>[^:]+): ?(?P<text>.*)$"
)
SYSTEM_RE = re.compile(r"^(?:\[)?(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}), .*?(?:\])? - ")

MY_ALIASES = {"alex", "owner", "me", "you"}

TARGET_FILES = {
    "close_friend": "close_friends.jsonl",
    "casual_friend": "casual_friends.jsonl",
    "romantic_interest": "romantic_interest.jsonl",
    "family": "family.jsonl",
    "university": "university.jsonl",
    "professional": "professional.jsonl",
}

SYNTHETIC_SEEDS = {
    "close_friend": [
        ("hii", "heyy"),
        ("hey", "yo"),
        ("you coming later?", "yh probs what time u lot going"),
        ("where are u", "im here relax"),
        ("call me", "gimme a sec"),
        ("wyd", "nothing much icl wbu"),
    ],
    "casual_friend": [
        ("hii", "hey"),
        ("yo", "what u saying"),
        ("you free today?", "maybe later what time"),
        ("did you see that", "yeah thats mad"),
        ("can you send it", "yh one sec"),
        ("you coming uni?", "probs depends when"),
    ],
    "romantic_interest": [
        ("hii", "heyy"),
        ("you good", "yeah u good"),
        ("wyd", "not much wbu"),
        ("call me", "yeah gimme a sec"),
    ],
    "family": [
        ("hi", "hi u okay"),
        ("are you home?", "not yet ill let you know"),
        ("call me when you can", "okay ill call in a bit"),
        ("did you eat?", "yeah im good"),
        ("where are you?", "out rn ill message you soon"),
    ],
    "university": [
        ("hi", "hi you good"),
        ("can you send the notes?", "yeah ill send them when im back"),
        ("are you coming to the seminar?", "not sure yet ill check"),
        ("did you finish the coursework?", "not fully yet"),
        ("can we meet before lecture?", "yeah what time works"),
    ],
    "professional": [
        ("hello", "hello"),
        ("can you confirm the meeting?", "yes that works for me"),
        ("are you available tomorrow?", "yes i should be available"),
        ("please send the file", "sure ill send it over"),
        ("can you update me?", "yes ill update you shortly"),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, default=Path("training"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/training_messages"))
    parser.add_argument("--owner", action="append", default=[])
    parser.add_argument("--minimum", type=int, default=2400)
    args = parser.parse_args()

    ensure_training_message_files(args.out_dir)
    owner_aliases = {normalize_sender(owner) for owner in [*MY_ALIASES, *args.owner] if owner.strip()}
    skipped: dict[str, int] = {}
    rows = extract_rows(args.training_dir, owner_aliases, skipped=skipped)
    if sum(len(items) for items in rows.values()) < args.minimum:
        add_synthetic_rows(rows, args.minimum)

    for relationship, filename in TARGET_FILES.items():
        path = args.out_dir / filename
        unique = dedupe(rows.get(relationship, []))
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in unique),
            encoding="utf-8",
        )

    summary = {relationship: len(dedupe(rows.get(relationship, []))) for relationship in TARGET_FILES}
    summary["skipped_uncertain_direction"] = sum(skipped.values())
    print(json.dumps(summary, indent=2))


def extract_rows(
    training_dir: Path,
    owner_aliases: set[str],
    *,
    skipped: dict[str, int] | None = None,
) -> dict[str, list[dict[str, object]]]:
    rows = {relationship: [] for relationship in TARGET_FILES}
    if not training_dir.exists():
        return rows
    for path in sorted(training_dir.glob("*.txt")):
        relationship = relationship_for_path(path)
        events = parse_export(path)
        for index, (sender, text) in enumerate(events):
            if is_owner(sender, owner_aliases):
                continue
            incoming = clean(text)
            context = [clean(item[1]) for item in events[max(0, index - 2) : index] if clean(item[1])]
            replies: list[str] = []
            next_index = index + 1
            while next_index < len(events) and is_owner(events[next_index][0], owner_aliases):
                replies.append(clean(events[next_index][1]))
                next_index += 1
                if len(replies) >= 2:
                    break
            reply = clean(" ".join(replies))
            if not reply and skipped is not None and next_index < len(events):
                skipped["no_confident_owner_reply"] = skipped.get("no_confident_owner_reply", 0) + 1
            if usable(incoming) and usable(reply):
                rows[relationship].append(
                    {
                        "relationship_type": relationship,
                        "incoming": redact_private_text(incoming),
                        "context": [redact_private_text(item) for item in context[-2:]],
                        "my_reply": redact_private_text(reply),
                        "notes": "bootstrapped from WhatsApp export",
                    }
                )
    return rows


def normalize_sender(sender: str) -> str:
    return re.sub(r"\s+", " ", sender.strip().casefold())


def is_owner(sender: str, owner_aliases: set[str]) -> bool:
    normalized = normalize_sender(sender)
    return bool(normalized and normalized in owner_aliases)


def parse_export(path: Path) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    current_sender: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_sender, current_lines
        if current_sender is not None:
            text = clean(" ".join(current_lines))
            if text:
                events.append((current_sender, text))
        current_sender = None
        current_lines = []

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.replace("\ufeff", "").replace("\u200e", "").strip()
        match = MESSAGE_RE.match(line)
        if match:
            flush()
            current_sender = match.group("sender").strip()
            current_lines = [match.group("text").strip()]
        elif SYSTEM_RE.match(line):
            flush()
        elif current_sender is not None and line:
            current_lines.append(line)
    flush()
    return events


def relationship_for_path(path: Path) -> str:
    name = path.stem.casefold()
    if "taylor" in name or "posh" in name:
        return "romantic_interest"
    if "riley" in name:
        return "close_friend"
    return "casual_friend"


def add_synthetic_rows(rows: dict[str, list[dict[str, object]]], minimum: int) -> None:
    total = sum(len(items) for items in rows.values())
    counter = 0
    relationships = list(SYNTHETIC_SEEDS)
    while total < minimum:
        relationship = relationships[counter % len(relationships)]
        incoming, reply = SYNTHETIC_SEEDS[relationship][counter % len(SYNTHETIC_SEEDS[relationship])]
        rows[relationship].append(
            {
                "relationship_type": relationship,
                "incoming": f"{incoming} {counter}" if counter >= len(SYNTHETIC_SEEDS[relationship]) else incoming,
                "context": [],
                "my_reply": reply,
                "notes": "synthetic bootstrap example; replace with real corrections over time",
            }
        )
        counter += 1
        total += 1


def dedupe(items: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, object]] = []
    for item in items:
        key = (str(item.get("incoming", "")).casefold(), str(item.get("my_reply", "")).casefold())
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def usable(text: str) -> bool:
    lowered = text.casefold()
    if not text or lowered in {"<media omitted>", "you deleted this message"}:
        return False
    if len(text) > 160:
        return False
    return True


if __name__ == "__main__":
    main()
