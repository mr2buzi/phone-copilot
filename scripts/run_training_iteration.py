from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPORT_PATH = Path("data/reports/training_iteration_report.json")
EVAL_REPORT_PATH = Path("data/reports/reply_style_eval_report.json")
REBUILD_MARKER = Path("data/vector_db/reply_examples_chroma/.rebuild_required")
REVIEW_PATH = Path("data/training_messages/auto_style_corrections_review.jsonl")


def run_command(args: list[str]) -> tuple[int, str]:
    completed = subprocess.run(args, cwd=Path.cwd(), text=True, capture_output=True)
    return completed.returncode, (completed.stdout + completed.stderr).strip()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_review_jsonl(path: Path) -> list[dict[str, Any]]:
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


def parse_json_output(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def eval_args(limit: int | None, intent: str | None, relationship: str | None) -> list[str]:
    args = [sys.executable, "scripts/evaluate_reply_style.py"]
    if limit is not None:
        args.extend(["--limit", str(limit)])
    if intent:
        args.extend(["--intent", intent])
    if relationship:
        args.extend(["--relationship", relationship])
    return args


def count_pending_review() -> int:
    return sum(1 for row in load_review_jsonl(REVIEW_PATH) if str(row.get("status") or "needs_human_review") == "needs_human_review")


def metric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    if not after:
        return {}
    return {
        "pass_count": int(after.get("pass_count", 0) or 0) - int(before.get("pass_count", 0) or 0),
        "fail_count": int(after.get("fail_count", 0) or 0) - int(before.get("fail_count", 0) or 0),
        "average_overall_score": round(float(after.get("average_overall_score", 0) or 0) - float(before.get("average_overall_score", 0) or 0), 2),
        "average_style_score": round(float(after.get("average_style_score", 0) or 0) - float(before.get("average_style_score", 0) or 0), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one safe reply-style training iteration.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--intent", default=None)
    parser.add_argument("--relationship", default=None)
    parser.add_argument("--auto-rebuild-if-approved", action="store_true")
    parser.add_argument("--no-rebuild", action="store_true")
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc).isoformat()
    before_code, before_output = run_command(eval_args(args.limit, args.intent, args.relationship))
    eval_before = load_json(EVAL_REPORT_PATH)
    improve_cmd = [sys.executable, "scripts/improve_failed_replies.py", "--write"]
    if args.limit is not None:
        improve_cmd.extend(["--limit", str(args.limit)])
    improve_code, improve_output = run_command(improve_cmd)
    improve_summary = parse_json_output(improve_output)
    if not improve_summary:
        improve_summary = {"raw_output": improve_output}

    vector_rebuild_run = False
    rebuild_output = ""
    approved_since_last_rebuild = 1 if REBUILD_MARKER.exists() else 0
    if args.auto_rebuild_if_approved and not args.no_rebuild and REBUILD_MARKER.exists():
        rebuild_code, rebuild_output = run_command([sys.executable, "scripts/build_reply_vector_db.py"])
        vector_rebuild_run = rebuild_code == 0
        if vector_rebuild_run:
            REBUILD_MARKER.unlink(missing_ok=True)

    eval_after: dict[str, Any] = {}
    after_output = ""
    if vector_rebuild_run:
        _, after_output = run_command(eval_args(args.limit, args.intent, args.relationship))
        eval_after = load_json(EVAL_REPORT_PATH)

    report = {
        "started_at": started_at,
        "eval_before": {
            key: eval_before.get(key)
            for key in ("total_cases", "pass_count", "fail_count", "average_overall_score", "average_style_score", "top_hard_reject_reasons")
        },
        "failures_found": int(eval_before.get("fail_count", 0) or 0),
        "review_candidates_generated": int(improve_summary.get("improvements_written", improve_summary.get("would_write", 0)) or 0),
        "review_queue_total": count_pending_review(),
        "approved_since_last_rebuild": approved_since_last_rebuild,
        "vector_rebuild_run": vector_rebuild_run,
        "eval_after": {
            key: eval_after.get(key)
            for key in ("total_cases", "pass_count", "fail_count", "average_overall_score", "average_style_score", "top_hard_reject_reasons")
        } if eval_after else {},
        "improvement_delta": metric_delta(eval_before, eval_after),
        "next_actions": [
            "Review pending style improvement rows in Training.",
            "Approve or edit only rows that actually sound like you.",
            "Rebuild vector DB after approvals.",
        ],
        "commands": {
            "eval_before_exit_code": before_code,
            "eval_before_output": before_output[-1000:],
            "improve_exit_code": improve_code,
            "improve_output": improve_output[-1000:],
            "rebuild_output": rebuild_output[-1000:],
            "eval_after_output": after_output[-1000:],
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
