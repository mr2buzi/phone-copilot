from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from libs.drafting.intents import contains_suspicious_phrase, normalize_text
from libs.drafting.training_data import row_quality_issues

TRAINING_DIR = Path("data/training_messages")
SAMPLE_SIZE = 20


def main() -> None:
    paths = sorted(path for path in TRAINING_DIR.glob("*.jsonl") if path.parent.name != "quarantine")
    if not paths:
        print(f"No JSONL files found in {TRAINING_DIR}")
        return

    grand_total = 0
    grand_flagged = 0
    for path in paths:
        report = audit_file(path)
        grand_total += report["total_rows"]
        grand_flagged += report["flagged_rows"]
        print_file_report(path, report)

    print("=" * 80)
    print(f"TOTAL rows={grand_total} flagged_rows={grand_flagged}")


def audit_file(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    replies = [reply_text(row) for row in rows if reply_text(row)]
    duplicate_replies = sum(count for _reply, count in Counter(normalize_text(reply) for reply in replies).items() if count > 1)
    issue_counts: Counter[str] = Counter()
    flagged_rows = 0
    samples: list[dict[str, Any]] = []

    for row in rows:
        issues = row_quality_issues(row, relationship_type=str(row.get("relationship_type", "")))
        if looks_like_other_person(row):
            issues.append("reply_looks_like_other_person")
        if issues:
            flagged_rows += 1
            issue_counts.update(issues)
        if contains_suspicious_phrase(reply_text(row)):
            issue_counts["suspicious_or_flirty_phrase"] += 1

    if rows:
        samples = random.Random(42).sample(rows, k=min(SAMPLE_SIZE, len(rows)))

    return {
        "total_rows": len(rows),
        "flagged_rows": flagged_rows,
        "empty_incoming": issue_counts.get("empty_incoming", 0),
        "empty_reply": issue_counts.get("empty_reply", 0),
        "duplicate_replies": duplicate_replies,
        "suspicious_replies": issue_counts.get("suspicious_phrase", 0),
        "unsafe_flirty_replies": issue_counts.get("unsafe_flirty_reply", 0),
        "very_long_replies": issue_counts.get("very_long_reply", 0),
        "looks_like_other_person": issue_counts.get("reply_looks_like_other_person", 0),
        "incoming_equals_reply": issue_counts.get("incoming_equals_reply", 0),
        "issue_counts": dict(issue_counts),
        "top_repeated_replies": Counter(replies).most_common(10),
        "samples": samples,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            rows.append({"_line_number": line_number, "_invalid_json": line})
            continue
        if isinstance(row, dict):
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def reply_text(row: dict[str, Any]) -> str:
    return str(row.get("my_reply") or row.get("user_final_reply") or "").strip()


def looks_like_other_person(row: dict[str, Any]) -> bool:
    reply = normalize_text(reply_text(row))
    incoming = normalize_text(str(row.get("incoming", "")))
    if not reply:
        return False
    other_starts = ("are you ", "r u ", "you coming", "can you ", "did you ", "where are you", "where r u")
    if reply.startswith(other_starts) and not incoming.startswith(other_starts):
        return True
    return bool(re.search(r"\b(reply to me|text me back|answer me)\b", reply))


def print_file_report(path: Path, report: dict[str, Any]) -> None:
    print("=" * 80)
    print(path)
    for key in (
        "total_rows",
        "flagged_rows",
        "empty_incoming",
        "empty_reply",
        "duplicate_replies",
        "suspicious_replies",
        "unsafe_flirty_replies",
        "very_long_replies",
        "looks_like_other_person",
        "incoming_equals_reply",
    ):
        print(f"{key}: {report[key]}")
    print("issue_counts:")
    for issue, count in sorted(report["issue_counts"].items()):
        print(f"  {issue}: {count}")
    print("top repeated replies:")
    for reply, count in report["top_repeated_replies"]:
        if count > 1:
            print(f"  {count}x {reply[:120]}")
    print("sample rows:")
    for row in report["samples"]:
        print(
            "  "
            + json.dumps(
                {
                    "line": row.get("_line_number"),
                    "relationship_type": row.get("relationship_type"),
                    "incoming": row.get("incoming"),
                    "my_reply": row.get("my_reply") or row.get("user_final_reply"),
                },
                ensure_ascii=True,
            )
        )


if __name__ == "__main__":
    main()
