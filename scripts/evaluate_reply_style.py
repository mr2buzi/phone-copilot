from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from libs.drafting.conversation_state import ConversationStateStore
from libs.drafting.service import DraftingService
from libs.drafting.style_evaluator import evaluate_candidate_style
from libs.drafting.training_data import load_all_training_rows, load_corrections, normalize_relationship_type

EVAL_CASES_PATH = Path("data/eval/reply_eval_cases.jsonl")
REPORT_PATH = Path("data/reports/reply_style_eval_report.json")
FAILURES_PATH = Path("data/reports/reply_style_eval_failures.jsonl")
HISTORY_PATH = Path("data/reports/style_eval_history.jsonl")


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


def load_eval_cases() -> list[dict[str, Any]]:
    cases = load_jsonl(EVAL_CASES_PATH)
    seen = {str(row.get("id")) for row in cases}
    training_dir = Path("data/training_messages")
    source_rows = []
    for row in load_corrections(training_dir):
        source_rows.append({**row, "my_reply": row.get("user_final_reply"), "_source": "correction", "style_authority": "high"})
    for row in load_all_training_rows(training_dir):
        if row.get("is_synthetic") or row.get("source") == "generated_safe_template":
            continue
        source_rows.append(row)
    added = 0
    for index, row in enumerate(source_rows):
        incoming = str(row.get("incoming", "")).strip()
        reply = str(row.get("my_reply") or row.get("user_final_reply") or "").strip()
        if not incoming or not reply:
            continue
        case_id = f"real_high_authority_{index:04d}"
        if case_id in seen:
            continue
        cases.append(
            {
                "id": case_id,
                "incoming": incoming,
                "context": row.get("context", []) if isinstance(row.get("context"), list) else [],
                "relationship_type": normalize_relationship_type(str(row.get("relationship_type", "unknown"))),
                "intent_type": str(row.get("intent_type", "unknown")),
                "contact_name": str(row.get("contact_name", "")),
                "ideal_replies": [reply],
                "bad_replies": [],
                "notes": f"Auto-added from {row.get('_source') or row.get('source') or 'real_training'} example.",
            }
        )
        added += 1
        if added >= 200:
            break
    return cases


def build_service() -> DraftingService:
    service = DraftingService(
        template_path=Path("data/templates/reply_templates.json"),
        approved_photos_dir=Path("data/approved_photos"),
        ai_reply_enabled=False,
        training_messages_dir=Path("data/training_messages"),
        contact_overrides_path=Path("data/contact_overrides.json"),
        style_profile_path=Path("data/style_profile.json"),
    )
    service.conversation_state = ConversationStateStore(Path("data/reports/.eval_conversation_state.json"))
    return service


def generate_candidates(service: DraftingService, case: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    bundle = service.regenerate_bundle(
        contact_name=str(case.get("contact_name") or "") or None,
        incoming=str(case.get("incoming", "")),
        context=[str(item) for item in case.get("context", [])] if isinstance(case.get("context"), list) else [],
        relationship_type=str(case.get("relationship_type", "unknown")),
        intent_type=str(case.get("intent_type", "unknown")),
        avoid_candidates=[],
        diversity_mode="natural",
    )
    return [candidate.text for candidate in bundle.reply_candidates], bundle.model_dump(mode="json")


def evaluate_case(service: DraftingService, case: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generated, bundle = generate_candidates(service, case)
    evaluations: list[dict[str, Any]] = []
    for candidate in generated:
        result = evaluate_candidate_style(
            {
                **case,
                "candidate": candidate,
                "retrieved_examples": bundle.get("retrieved_examples", []),
            }
        )
        evaluations.append({"candidate": candidate, **result})
    ideal_scores = [
        evaluate_candidate_style({**case, "candidate": reply, "retrieved_examples": bundle.get("retrieved_examples", [])})
        for reply in case.get("ideal_replies", [])
        if str(reply).strip()
    ]
    bad_scores = [
        evaluate_candidate_style({**case, "candidate": reply, "retrieved_examples": bundle.get("retrieved_examples", [])})
        for reply in case.get("bad_replies", [])
        if str(reply).strip()
    ]
    best = max(evaluations, key=lambda item: int(item.get("overall_score", 0)), default={})
    pass_case = bool(best) and int(best.get("overall_score", 0)) >= 75 and not best.get("hard_reject")
    if any(score.get("hard_reject") is False and int(score.get("overall_score", 0)) >= 70 for score in bad_scores):
        pass_case = False
    summary = {
        "case_id": case.get("id"),
        "incoming": case.get("incoming"),
        "relationship_type": case.get("relationship_type"),
        "intent_type": case.get("intent_type"),
        "generated": generated,
        "best_candidate": best.get("candidate", ""),
        "best_score": best.get("overall_score", 0),
        "pass": pass_case,
        "ideal_average": round(mean([score["overall_score"] for score in ideal_scores]), 2) if ideal_scores else None,
        "bad_average": round(mean([score["overall_score"] for score in bad_scores]), 2) if bad_scores else None,
        "evaluations": evaluations,
        "bundle": bundle,
    }
    return evaluations, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate offline reply candidates against the style rubric.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--intent", default=None)
    parser.add_argument("--relationship", default=None)
    parser.add_argument("--save-failures", action="store_true")
    args = parser.parse_args()

    cases = load_eval_cases()
    if args.intent:
        cases = [case for case in cases if str(case.get("intent_type")) == args.intent]
    if args.relationship:
        cases = [case for case in cases if str(case.get("relationship_type")) == args.relationship]
    if args.limit:
        cases = cases[: max(0, args.limit)]

    service = build_service()
    summaries: list[dict[str, Any]] = []
    all_evaluations: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case in cases:
        evaluations, summary = evaluate_case(service, case)
        summaries.append(summary)
        all_evaluations.extend(evaluations)
        if not summary["pass"]:
            best_eval = max(evaluations, key=lambda item: int(item.get("overall_score", 0)), default={})
            failures.append(
                {
                    "case": {k: v for k, v in case.items() if k != "bad_replies"},
                    "bad_ai_reply": best_eval.get("candidate") or (summary.get("generated") or [""])[0],
                    "style_score": best_eval.get("overall_score", 0),
                    "reason_bad": ", ".join(best_eval.get("reject_reasons", [])) or "style score below threshold",
                    "improvement_notes": best_eval.get("improvement_notes", []),
                    "evaluations": evaluations,
                }
            )

    generic_counter: Counter[str] = Counter()
    reject_counter: Counter[str] = Counter()
    for evaluation in all_evaluations:
        for reason in evaluation.get("reject_reasons", []):
            reject_counter[str(reason)] += 1
            if "generic" in str(reason) or "customer support" in str(reason):
                generic_counter[str(reason)] += 1

    failures_by_intent: dict[str, int] = defaultdict(int)
    failures_by_relationship: dict[str, int] = defaultdict(int)
    for failure in failures:
        case = failure["case"]
        failures_by_intent[str(case.get("intent_type", "unknown"))] += 1
        failures_by_relationship[str(case.get("relationship_type", "unknown"))] += 1

    report = {
        "total_cases": len(summaries),
        "pass_count": sum(1 for item in summaries if item["pass"]),
        "fail_count": sum(1 for item in summaries if not item["pass"]),
        "average_overall_score": round(mean([item["best_score"] for item in summaries]), 2) if summaries else 0,
        "average_style_score": round(mean([evaluation["owner_style"] for evaluation in all_evaluations]), 2) if all_evaluations else 0,
        "failures_by_intent_type": dict(sorted(failures_by_intent.items())),
        "failures_by_relationship_type": dict(sorted(failures_by_relationship.items())),
        "top_generic_phrases": dict(generic_counter.most_common(12)),
        "top_hard_reject_reasons": dict(reject_counter.most_common(12)),
        "too_formal_examples": [item for item in failures if "formal" in str(item.get("reason_bad", ""))][:10],
        "too_cringe_examples": [item for item in failures if "cringe" in str(item.get("reason_bad", "")) or "flirty" in str(item.get("reason_bad", ""))][:10],
        "invented_fact_examples": [item for item in failures if "invent" in str(item.get("reason_bad", ""))][:10],
        "did_not_answer_examples": [item for item in failures if "does not answer" in str(item.get("reason_bad", ""))][:10],
        "duplicate_examples": [item for item in failures if "duplicate" in str(item.get("reason_bad", ""))][:10],
        "case_summaries": summaries,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    history_row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_cases": report["total_cases"],
        "pass_count": report["pass_count"],
        "fail_count": report["fail_count"],
        "avg_overall_score": report["average_overall_score"],
        "avg_owner_style": report["average_style_score"],
        "avg_context_fit": round(mean([evaluation["context_fit"] for evaluation in all_evaluations]), 2) if all_evaluations else 0,
        "top_failure_reasons": report["top_hard_reject_reasons"],
    }
    with HISTORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(history_row, ensure_ascii=True) + "\n")
    if args.save_failures or True:
        FAILURES_PATH.write_text(
            "\n".join(json.dumps(row, ensure_ascii=True) for row in failures) + ("\n" if failures else ""),
            encoding="utf-8",
        )
    print(json.dumps({k: report[k] for k in ("total_cases", "pass_count", "fail_count", "average_overall_score", "average_style_score")}, indent=2))


if __name__ == "__main__":
    main()
