from __future__ import annotations

import json
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from libs.drafting.training_data import load_all_training_rows, load_corrections

EVAL_REPORT_PATH = Path("data/reports/reply_style_eval_report.json")
OUTPUT_PATH = Path("data/training_messages/active_learning_queue.jsonl")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def make_row(case: dict[str, Any], reason: str, priority: int) -> dict[str, Any]:
    return {
        "id": f"active_{uuid.uuid4().hex[:12]}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "incoming": case.get("incoming", ""),
        "context": case.get("context", []) if isinstance(case.get("context"), list) else [],
        "relationship_type": case.get("relationship_type", "unknown"),
        "intent_type": case.get("intent_type", "unknown"),
        "reason": reason,
        "suggested_action": "add correction / approve better reply / add real example",
        "priority": max(1, min(5, priority)),
    }


def main() -> None:
    report = load_json(EVAL_REPORT_PATH)
    rows: list[dict[str, Any]] = []
    for summary in report.get("case_summaries", []):
        if not isinstance(summary, dict) or summary.get("pass"):
            continue
        case = {
            "incoming": summary.get("incoming", ""),
            "context": [],
            "relationship_type": summary.get("relationship_type", "unknown"),
            "intent_type": summary.get("intent_type", "unknown"),
        }
        best_score = int(summary.get("best_score", 0) or 0)
        bundle = summary.get("bundle") if isinstance(summary.get("bundle"), dict) else {}
        reasons = [str(item) for item in bundle.get("retrieval_reasons", [])]
        if best_score < 75:
            rows.append(make_row(case, "low style score", 5))
        if any("synthetic fallback" in reason for reason in reasons):
            rows.append(make_row(case, "synthetic fallback", 4))
        if not summary.get("generated"):
            rows.append(make_row(case, "no usable candidate", 5))

    training_rows = load_all_training_rows(Path("data/training_messages"))
    correction_rows = load_corrections(Path("data/training_messages"))
    coverage = Counter((str(row.get("relationship_type", "unknown")), str(row.get("intent_type", "unknown"))) for row in [*training_rows, *correction_rows])
    for (relationship, intent), count in coverage.items():
        if count < 3:
            rows.append(
                make_row(
                    {"incoming": "", "context": [], "relationship_type": relationship, "intent_type": intent},
                    "low coverage",
                    2,
                )
            )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = (str(row["incoming"]), str(row["relationship_type"]), str(row["intent_type"]), str(row["reason"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in deduped) + ("\n" if deduped else ""),
        encoding="utf-8",
    )
    print(json.dumps({"rows_written": len(deduped), "path": str(OUTPUT_PATH)}, indent=2))


if __name__ == "__main__":
    main()
