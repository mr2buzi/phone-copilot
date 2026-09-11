from __future__ import annotations

import argparse
import json
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from libs.drafting.style_evaluator import evaluate_candidate_style

FAILURES_PATH = Path("data/reports/reply_style_eval_failures.jsonl")
REVIEW_PATH = Path("data/training_messages/auto_style_corrections_review.jsonl")
HARD_SAFETY_REASONS = {
    "too flirty without context",
    "invents availability or facts",
    "does not answer the message",
    "generic filler",
    "cringe forced slang",
    "too casual for professional context",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def candidate_pool(case: dict[str, Any], failure: dict[str, Any]) -> list[str]:
    incoming = str(case.get("incoming", "")).casefold().strip()
    relationship = str(case.get("relationship_type", "unknown"))
    intent = str(case.get("intent_type", "unknown"))
    ideals = [str(item).strip() for item in case.get("ideal_replies", []) if str(item).strip()]
    pool: list[str] = [*ideals]
    if intent == "greeting":
        pool.extend(["yo", "heyy", "u good", "what u saying"])
    if intent in {"planning", "availability"} or any(term in incoming for term in ("coming", "going", "pulling up", "meeting", "plans", "free later")):
        pool.extend(["yh what time", "where u lot going", "what time you thinking", "who's going", "not sure yet what time?"])
    if relationship == "professional" or intent == "professional":
        pool.extend(["yes, i'll send it over", "yeah i can send it", "i'll send it over shortly"])
    if intent in {"argument", "emotional"}:
        pool.extend(["i wasnt ignoring you", "i was busy, whats happened", "i hear you, let me reply properly"])
    if intent == "casual_checkin" or incoming == "wyd":
        pool.extend(["nothing much wbu", "not much you", "just chilling wbu"])
    if not pool:
        pool.append("what do you mean")
    return list(dict.fromkeys(pool))


def existing_review_keys(rows: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    return {
        (
            str(row.get("incoming", "")),
            str(row.get("bad_ai_reply", "")),
            str(row.get("suggested_better_reply", "")),
        )
        for row in rows
        if str(row.get("status") or "needs_human_review") != "rejected"
    }


def hard_safety_reject(evaluation: dict[str, Any]) -> str | None:
    reasons = {str(reason) for reason in evaluation.get("reject_reasons", [])}
    for reason in HARD_SAFETY_REASONS:
        if reason in reasons:
            return reason
    return None


def best_safe_candidate(case: dict[str, Any], failure: dict[str, Any], min_review_score: int) -> tuple[dict[str, Any] | None, str | None]:
    best: dict[str, Any] | None = None
    discard_reason: str | None = None
    for candidate in candidate_pool(case, failure):
        evaluation = evaluate_candidate_style({**case, "candidate": candidate, "retrieved_examples": []})
        safety_reason = hard_safety_reject(evaluation)
        if safety_reason:
            discard_reason = safety_reason
            continue
        score = int(evaluation.get("overall_score", 0) or 0)
        if score < min_review_score:
            discard_reason = "low score"
            continue
        accepted = {
            "candidate": candidate,
            "acceptance": "strict_accept" if score >= 88 and not evaluation.get("hard_reject") else "review_accept",
            "evaluation": evaluation,
        }
        if best is None or score > int(best["evaluation"].get("overall_score", 0) or 0):
            best = accepted
    return best, discard_reason


def build_review_row(case: dict[str, Any], failure: dict[str, Any], accepted: dict[str, Any]) -> dict[str, Any]:
    evaluation = accepted["evaluation"]
    return {
        "id": f"auto_style_{uuid.uuid4().hex[:12]}",
        "row_id": f"auto_style_{uuid.uuid4().hex[:12]}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "incoming": case.get("incoming", ""),
        "context": case.get("context", []) if isinstance(case.get("context"), list) else [],
        "relationship_type": case.get("relationship_type", "unknown"),
        "intent_type": case.get("intent_type", "unknown"),
        "contact_name": case.get("contact_name", ""),
        "bad_ai_reply": failure.get("bad_ai_reply", ""),
        "suggested_better_reply": accepted["candidate"],
        "style_score": evaluation.get("overall_score", 0),
        "reason_bad": failure.get("reason_bad", "style score below threshold"),
        "improvement_notes": evaluation.get("improvement_notes", []),
        "acceptance": accepted["acceptance"],
        "source": "auto_style_improvement",
        "status": "needs_human_review",
    }


def run(*, limit: int | None, min_review_score: int, write: bool, max_write: int | None = None) -> dict[str, Any]:
    failures = load_jsonl(FAILURES_PATH)
    if limit is not None:
        failures = failures[: max(0, limit)]
    existing_rows = load_jsonl(REVIEW_PATH)
    seen = existing_review_keys(existing_rows)
    new_rows: list[dict[str, Any]] = []
    discard_reasons: Counter[str] = Counter()
    generated = 0
    for failure in failures:
        case = failure.get("case") if isinstance(failure.get("case"), dict) else {}
        accepted, discard_reason = best_safe_candidate(case, failure, min_review_score)
        if accepted is None:
            discard_reasons[discard_reason or "no safe candidate"] += 1
            continue
        generated += 1
        row = build_review_row(case, failure, accepted)
        key = (str(row["incoming"]), str(row["bad_ai_reply"]), str(row["suggested_better_reply"]))
        if key in seen:
            discard_reasons["duplicate review row"] += 1
            continue
        if max_write is not None and len(new_rows) >= max(0, max_write):
            discard_reasons["max review candidates reached"] += 1
            continue
        seen.add(key)
        new_rows.append(row)
    if write and new_rows:
        with REVIEW_PATH.open("a", encoding="utf-8") as handle:
            for row in new_rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return {
        "failures_read": len(failures),
        "improvements_generated": generated,
        "improvements_written": len(new_rows) if write else 0,
        "would_write": len(new_rows),
        "discarded_unsafe": sum(count for reason, count in discard_reasons.items() if reason in HARD_SAFETY_REASONS),
        "discarded_low_score": discard_reasons.get("low score", 0),
        "top_discard_reasons": dict(discard_reasons.most_common(12)),
        "review_path": str(REVIEW_PATH),
        "dry_run": not write,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite failed style-eval candidates into review-only correction suggestions.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-review-score", type=int, default=75)
    parser.add_argument("--max-write", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    write = bool(args.write and not args.dry_run)
    summary = run(limit=args.limit, min_review_score=args.min_review_score, write=write, max_write=args.max_write)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
