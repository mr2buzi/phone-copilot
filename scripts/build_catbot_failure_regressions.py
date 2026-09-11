from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_FEEDBACK_PATH = Path("data/thread_memory/thread_feedback.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/reports/catbot_failure_regressions.jsonl")
REQUIRED_METADATA_FIELDS = [
    "scene_type",
    "required_reply_move",
    "forbidden_reply_moves",
    "scene_fit_score",
    "required_move_satisfied",
    "forbidden_move_violated",
]
STALE_DISALLOWED = {
    "ok",
    "okay",
    "same icl",
    "same just chilling",
    "fair just chilling too",
    "what u saying",
    "what u saying then",
    "yo what u saying",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    tmp_path.replace(path)


def _labelled_context(context: list[Any], incoming: str) -> list[str]:
    labelled: list[str] = []
    for index, item in enumerate(context[-12:]):
        speaker = "user" if index % 2 == 0 else "bot"
        labelled.append(f"{speaker}: {str(item)}")
    if incoming:
        labelled.append(f"user: {incoming}")
    return labelled


def _disallowed_replies(row: dict[str, Any]) -> list[str]:
    replies = set(STALE_DISALLOWED)
    selected = str(row.get("selected_candidate") or "").strip()
    if selected:
        replies.add(selected)
    for candidate in row.get("candidate_shortlist", []):
        if isinstance(candidate, dict):
            text = str(candidate.get("text") or "").strip()
            if text and candidate.get("final_decision") == "reject":
                replies.add(text)
    return sorted(replies)


def _case_id(row: dict[str, Any]) -> str:
    digest = hashlib.sha1(
        "|".join(
            [
                str(row.get("thread_id") or ""),
                str(row.get("session_id") or ""),
                str(row.get("timestamp") or ""),
                str(row.get("selected_candidate") or ""),
            ]
        ).encode("utf-8")
    ).hexdigest()[:12]
    return f"catbot_failure_{digest}"


def regression_case_from_feedback(row: dict[str, Any]) -> dict[str, Any] | None:
    if str(row.get("feedback") or "").lower() != "thumbs_down":
        return None
    failure_tags = [str(item) for item in row.get("failure_tags", []) if str(item).strip()] if isinstance(row.get("failure_tags"), list) else []
    if "contains_sensitive" in failure_tags:
        return None
    incoming = str(row.get("incoming") or "")
    context = row.get("context") if isinstance(row.get("context"), list) else []
    scene_type = str(row.get("scene_type") or "")
    required_reply_move = str(row.get("required_reply_move") or "")
    selected_metadata = row.get("selected_candidate_metadata") if isinstance(row.get("selected_candidate_metadata"), dict) else {}
    return {
        "id": _case_id(row),
        "source": "catbot_thread_feedback",
        "thread_id": str(row.get("thread_id") or ""),
        "session_id": str(row.get("session_id") or ""),
        "created_from_feedback_at": str(row.get("timestamp") or ""),
        "incoming": incoming,
        "context": context,
        "labelled_conversation": _labelled_context(context, incoming),
        "bad_reply": str(row.get("selected_candidate") or ""),
        "correction": str(row.get("correction") or ""),
        "expected_scene_type": scene_type,
        "expected_required_reply_move": required_reply_move,
        "disallowed_replies": _disallowed_replies(row),
        "failure_tags": failure_tags,
        "expected_metadata_fields": REQUIRED_METADATA_FIELDS,
        "selected_candidate_metadata": {
            key: selected_metadata.get(key)
            for key in REQUIRED_METADATA_FIELDS
            if key in selected_metadata
        },
        "assertions": {
            "reply_not_in_disallowed_replies": True,
            "scene_type_matches_if_present": bool(scene_type),
            "required_reply_move_matches_if_present": bool(required_reply_move),
            "provider_output_must_pass_local_validation": True,
            "auto_send_policy_unchanged": True,
        },
        "status": "needs_human_review",
    }


def build_catbot_failure_regressions(
    feedback_path: Path = DEFAULT_FEEDBACK_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    rows = _load_jsonl(feedback_path)
    cases = [case for row in rows if (case := regression_case_from_feedback(row)) is not None]
    unique: dict[str, dict[str, Any]] = {}
    for case in cases:
        unique[str(case["id"])] = case
    ordered = list(unique.values())
    _write_jsonl(output_path, ordered)
    return {
        "status": "written",
        "feedback_rows_read": len(rows),
        "regression_cases_written": len(ordered),
        "output_path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reviewable Catbot regression cases from thumbs-down thread feedback.")
    parser.add_argument("--feedback-path", type=Path, default=DEFAULT_FEEDBACK_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    print(json.dumps(build_catbot_failure_regressions(args.feedback_path, args.output_path), indent=2))


if __name__ == "__main__":
    main()
