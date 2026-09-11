from __future__ import annotations

import argparse
import json
import os
import random
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Any

from libs.drafting import DraftingService
from libs.drafting.conversation_state import ConversationStateStore


SENSITIVE_RE = re.compile(
    r"(?i)(api[_ -]?key|password|passcode|otp|one[- ]?time|sort code|card number|\b\d{12,16}\b|\b\d{3,4}\s?\d{3}\s?\d{3,4}\b)"
)


@dataclass
class TurnCheck:
    incoming: str
    context: list[str]
    expected_jobs: set[str] = field(default_factory=set)
    expected_scenes: set[str] = field(default_factory=set)
    banned_replies: set[str] = field(default_factory=set)
    banned_substrings: tuple[str, ...] = ()
    required_substrings_any: tuple[str, ...] = ()


_WORKER_SERVICE: DraftingService | None = None


def _redact(text: str) -> str:
    return SENSITIVE_RE.sub("[REDACTED]", text)


def _safe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            safe[key] = _redact(value)
        elif isinstance(value, list):
            safe[key] = [_redact(item) if isinstance(item, str) else item for item in value]
        elif isinstance(value, dict):
            safe[key] = _safe_payload(value)
        else:
            safe[key] = value
    return safe


def _failure_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")


def _append_failure(path: Path, row: dict[str, Any]) -> None:
    _failure_path(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_safe_payload(row), ensure_ascii=False) + "\n")


def _base_scenarios() -> list[TurnCheck]:
    return [
        TurnCheck(
            incoming="i was at home yeah and showering and some guy came in and started running in my living room",
            context=["oi guess what", "what happened"],
            expected_jobs={"react_to_story"},
            banned_replies={"ok", "okay"},
            required_substrings_any=("what", "nah", "bro", "wait", "why", "how"),
        ),
        TurnCheck(
            incoming="its okay is everything alright?",
            context=["thats a bit dry mate", "yeah fairs that was dry icl", "yeah so why u being dry", "yeah nah ur right my bad"],
            expected_jobs={"answer_care_check"},
            banned_replies={"fine then what should we talk about", "what u saying", "ok"},
            required_substrings_any=("im good", "im okay", "im alright", "dw", "my bad"),
        ),
        TurnCheck(
            incoming="bruh wtf",
            context=["alex ur being weird", "u ain't giving me much to work with what u been doing"],
            expected_jobs={"repair_after_weird_or_dry_reply"},
            banned_substrings=("u ain't giving me much", "bruh wtf is valid"),
            required_substrings_any=("my bad", "weird", "made no sense", "waffling", "answered"),
        ),
        TurnCheck(
            incoming="u already asked that...",
            context=["yo", "yo what u been up to", "yhhh idk what to say tbh", "yh same what u been up to"],
            expected_scenes={"repeated_reply_callout", "missed_context_callout", "dead_conversation"},
            banned_replies={"what u saying", "what u been up to", "ok"},
            required_substrings_any=("asked", "loop", "my bad", "already"),
        ),
        TurnCheck(
            incoming="i missed u baby",
            context=["hi", "heyy what u doing", "chilling wby", "nothing much just chilling"],
            expected_jobs={"answer_affection"},
            expected_scenes={"emotional_affection"},
            banned_replies={"say less what u doing", "same icl", "same just chilling", "fair just chilling too"},
            required_substrings_any=("sweet", "bless", "appreciate", "missed", "where u been"),
        ),
        TurnCheck(
            incoming="honestly just work wby",
            context=["hi", "heyy what u doing", "chilling wby", "nothing much just chilling", "[OTHER]: wyd", "[ME]: fair what u been on today"],
            expected_jobs={"answer_reciprocal_activity"},
            expected_scenes={"reciprocal_current_activity_question"},
            banned_substrings=("what u saying", "what u been on today", "u ain't giving me much"),
            required_substrings_any=("not much", "working", "sorting", "gym", "coding", "chilling"),
        ),
        TurnCheck(
            incoming="prolly sleeping wby",
            context=["yeah im bored asf", "valid what u doing later"],
            expected_jobs={"answer_reciprocal_activity"},
            expected_scenes={"reciprocal_current_activity_question"},
            banned_substrings=("u ain't giving me much", "what u been doing", "what u saying"),
            required_substrings_any=("not much", "working", "sorting", "gym", "coding", "chilling"),
        ),
        TurnCheck(
            incoming="i told and asked u...",
            context=["honestly just work wby", "chilling and still bored?", "[OTHER]: prolly sleeping wby", "[ME]: u ain't giving me much to work with what u been doing"],
            expected_jobs={"answer_reciprocal_activity"},
            expected_scenes={"missed_context_callout"},
            banned_substrings=("dream car", "what's been on ur mind", "u ain't giving me much"),
            required_substrings_any=("my bad", "missed", "ignored", "didnt answer", "didn't answer"),
        ),
        TurnCheck(
            incoming="im so tired",
            context=["what u been up to", "nothing crazy icl wbu", "[OTHER]: i been busy asf", "[ME]: same just chilling"],
            expected_jobs={"respond_to_user_mood"},
            expected_scenes={"tired_mood"},
            banned_replies={"same icl", "same just chilling", "fair just chilling too"},
            required_substrings_any=("tired", "sleep", "nap", "long day"),
        ),
        TurnCheck(
            incoming="u tell me",
            context=["ur boring me", "what u tryna do then", "[OTHER]: idk talk", "[ME]: u ain't giving me much to work with what u been doing"],
            expected_jobs={"progress_conversation", "repair_after_weird_or_dry_reply"},
            banned_replies={"what u saying", "what u been up to", "give me a topic", "ok"},
            banned_substrings=("u ain't giving me much",),
            required_substrings_any=("dream car", "cars", "gym", "my bad", "weird", "made no sense"),
        ),
        TurnCheck(
            incoming="how old r u",
            context=["yo", "yo how u been"],
            expected_jobs={"answer_direct_question"},
            expected_scenes={"owner_age_question", "identity_question"},
            banned_replies={"what u saying", "ok", "same icl"},
            required_substrings_any=("19",),
        ),
        TurnCheck(
            incoming="what dyu study",
            context=["im into tech", "valid same icl what part of tech"],
            expected_jobs={"answer_direct_question"},
            expected_scenes={"owner_study_question", "identity_question"},
            banned_replies={"what u saying", "ok", "same icl", "same just chilling"},
            required_substrings_any=("comp sci", "computer science", "tech"),
        ),
    ]


def _randomized_scenarios(count: int, seed: int) -> list[TurnCheck]:
    rng = random.Random(seed)
    story_bodies = [
        "some guy came in and started running in my living room",
        "someone ran into my kitchen while i was showering",
        "a random guy came in my house and just started sprinting",
    ]
    callouts = ["bruh wtf", "wdym", "what?", "ur being weird", "thats dry mate"]
    scenarios = list(_base_scenarios())
    activities = [
        "chilling wby",
        "honestly just work wby",
        "prolly sleeping wby",
        "in bed u",
        "nothing wbu",
    ]
    affection_inputs = ["i missed u", "i missed u baby", "missed you icl", "dwdw i missed u anyway"]
    mood_inputs = ["im so tired", "im tired", "im drained", "im exhausted"]
    direct_questions = [
        ("how old r u", ("19",), {"owner_age_question", "identity_question"}),
        ("where u from", ("northbridge", "northbridge", "sampleford"), {"owner_location_question", "identity_question"}),
        ("what dyu study", ("comp sci", "computer science", "tech"), {"owner_study_question", "identity_question"}),
        ("how was ur day", ("calm", "dead", "not bad", "decent", "busy"), {"day_check_question"}),
        ("whatd u do today", ("gym", "coding", "uni", "work", "nothing"), {"owner_day_activity_question"}),
    ]
    for index in range(max(0, count - len(scenarios))):
        mode = rng.choice(["story", "care", "callout", "loop", "affection", "activity", "missed_wby", "mood", "topic_lead", "identity"])
        if mode == "story":
            body = rng.choice(story_bodies)
            scenarios.append(
                TurnCheck(
                    incoming=f"i was at home yeah and {body}",
                    context=[rng.choice(["oi guess what", "bro listen", "guess what"]), rng.choice(["what happened", "go on", "what"])],
                    expected_jobs={"react_to_story"},
                    banned_replies={"ok", "okay"},
                    required_substrings_any=("what", "nah", "bro", "wait", "why", "how"),
                )
            )
        elif mode == "care":
            scenarios.append(
                TurnCheck(
                    incoming=rng.choice(["is everything alright?", "are u okay?", "u good?"]),
                    context=["thats dry mate", "yeah fairs that was dry icl"],
                    expected_jobs={"answer_care_check"},
                    banned_replies={"fine then what should we talk about", "ok"},
                    required_substrings_any=("im good", "im okay", "im alright", "dw"),
                )
            )
        elif mode == "callout":
            scenarios.append(
                TurnCheck(
                    incoming=rng.choice(callouts),
                    context=["alex ur being weird", rng.choice(["u ain't giving me much to work with what u been doing", "fine then what should we talk about"])],
                    expected_jobs={"repair_after_weird_or_dry_reply", "progress_conversation"},
                    banned_substrings=("u ain't giving me much", "bruh wtf is valid"),
                )
            )
        else:
            if mode == "affection":
                incoming = rng.choice(affection_inputs)
                scenarios.append(
                    TurnCheck(
                        incoming=incoming,
                        context=["hi", rng.choice(["heyy what u doing", "yo how u been"]), "chilling wby", "nothing much just chilling"],
                        expected_jobs={"answer_affection"},
                        expected_scenes={"emotional_affection"},
                        banned_replies={"say less what u doing", "same icl", "same just chilling", "fair just chilling too"},
                        required_substrings_any=("sweet", "bless", "appreciate", "missed", "where u been"),
                    )
                )
            elif mode == "activity":
                incoming = rng.choice(activities)
                scenarios.append(
                    TurnCheck(
                        incoming=incoming,
                        context=["hi", "heyy what u doing"],
                        expected_jobs={"answer_reciprocal_activity"},
                        expected_scenes={"reciprocal_current_activity_question"},
                        banned_substrings=("what u saying", "u ain't giving me much", "give me a topic"),
                        required_substrings_any=("not much", "working", "sorting", "gym", "coding", "chilling"),
                    )
                )
            elif mode == "missed_wby":
                scenarios.append(
                    TurnCheck(
                        incoming=rng.choice(["i told and asked u...", "i told u and asked u", "i answered and asked u"]),
                        context=["honestly just work wby", rng.choice(["chilling and still bored?", "fair what u been on today"]), "[OTHER]: prolly sleeping wby", "[ME]: u ain't giving me much to work with what u been doing"],
                        expected_jobs={"answer_reciprocal_activity"},
                        expected_scenes={"missed_context_callout"},
                        banned_substrings=("dream car", "what's been on ur mind", "u ain't giving me much"),
                        required_substrings_any=("my bad", "missed", "ignored", "didnt answer", "didn't answer"),
                    )
                )
            elif mode == "mood":
                scenarios.append(
                    TurnCheck(
                        incoming=rng.choice(mood_inputs),
                        context=["what u been up to", "nothing crazy icl wbu"],
                        expected_jobs={"respond_to_user_mood"},
                        expected_scenes={"tired_mood"},
                        banned_replies={"same icl", "same just chilling", "fair just chilling too"},
                        required_substrings_any=("tired", "sleep", "nap", "long day"),
                    )
                )
            elif mode == "topic_lead":
                scenarios.append(
                    TurnCheck(
                        incoming=rng.choice(["idk talk", "u tell me", "idk what to say tbh", "entertain me"]),
                        context=["ur boring me", "what u tryna do then"],
                        expected_jobs={"progress_conversation"},
                        banned_replies={"what u saying", "give me a topic", "ok"},
                        banned_substrings=("u ain't giving me much",),
                        required_substrings_any=("dream car", "cars", "gym", "random", "ill pick", "i'll pick", "lets talk", "what's been on"),
                    )
                )
            elif mode == "identity":
                incoming, required, scenes = rng.choice(direct_questions)
                scenarios.append(
                    TurnCheck(
                        incoming=incoming,
                        context=["yo", "yo how u been"],
                        expected_jobs={"answer_direct_question"},
                        expected_scenes=scenes,
                        banned_replies={"what u saying", "ok", "same icl", "same just chilling"},
                        required_substrings_any=required,
                    )
                )
            else:
                scenarios.append(
                    TurnCheck(
                        incoming=rng.choice(["u asked this already", "u already asked that", "why u keep asking that"]),
                        context=["yo", "yo what u been up to", "yhhh idk what to say tbh", "yh same what u been up to"],
                        banned_replies={"what u saying", "what u been up to", "ok"},
                        required_substrings_any=("asked", "loop", "my bad", "already"),
                    )
                )
            continue
        if mode == "loop":
            scenarios.append(
                TurnCheck(
                    incoming=rng.choice(["u asked this already", "u already asked that", "why u keep asking that"]),
                    context=["yo", "yo what u been up to", "yhhh idk what to say tbh", "yh same what u been up to"],
                    banned_replies={"what u saying", "what u been up to", "ok"},
                    required_substrings_any=("asked", "loop", "my bad", "already"),
                )
            )
    rng.shuffle(scenarios)
    return scenarios[:count]


def _run_check(service: DraftingService, scenario: TurnCheck) -> dict[str, Any] | None:
    full_context = []
    for idx, text in enumerate(scenario.context):
        stripped = text.strip()
        if stripped.casefold().startswith(("[other]:", "[me]:")):
            full_context.append(stripped)
        else:
            speaker = "[OTHER]" if idx % 2 == 0 else "[ME]"
            full_context.append(f"{speaker}: {stripped}")
    full_context.append(f"[OTHER]: {scenario.incoming}")
    bundle = service.build_bundle_with_context(
        [*scenario.context[-5:], scenario.incoming],
        full_context,
        contact_name="Catbot",
        relationship_type="close_friend",
    )
    top = bundle.reply_candidates[0]
    reply_norm = top.text.casefold().strip(" .?!")
    reasons: list[str] = []
    if scenario.expected_jobs and top.conversation_job not in scenario.expected_jobs:
        reasons.append(f"expected_job={sorted(scenario.expected_jobs)} got={top.conversation_job}")
    if scenario.expected_scenes and top.scene_type not in scenario.expected_scenes:
        reasons.append(f"expected_scene={sorted(scenario.expected_scenes)} got={top.scene_type}")
    if reply_norm in {item.casefold().strip(" .?!") for item in scenario.banned_replies}:
        reasons.append(f"banned_reply={top.text}")
    for banned in scenario.banned_substrings:
        if banned.casefold() in reply_norm:
            reasons.append(f"banned_substring={banned}")
    if scenario.required_substrings_any and not any(item.casefold() in reply_norm for item in scenario.required_substrings_any):
        reasons.append(f"missing_any={scenario.required_substrings_any}")
    if top.final_decision == "reject":
        reasons.append(f"top_rejected={top.blocked_reason}")
    if not reasons:
        return None
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "incoming": scenario.incoming,
        "context": scenario.context,
        "reply": top.text,
        "scene_type": top.scene_type,
        "conversation_job": top.conversation_job,
        "policy_fit_score": top.policy_fit_score,
        "training_style_score": top.training_style_score,
        "blocked_reason": top.blocked_reason,
        "reasons": reasons,
    }


def _init_worker() -> None:
    global _WORKER_SERVICE
    service = DraftingService(
        template_path=Path("data/templates/reply_templates.json"),
        approved_photos_dir=Path("data/approved_photos"),
        training_messages_dir=Path("data/training_messages"),
        ai_reply_enabled=False,
    )
    state_dir = Path(tempfile.gettempdir()) / f"phone_copilot_policy_sim_{os.getpid()}"
    state_dir.mkdir(parents=True, exist_ok=True)
    service.conversation_state = ConversationStateStore(state_dir / "conversation_state.json")
    _WORKER_SERVICE = service


def _run_check_worker(index_and_scenario: tuple[int, TurnCheck]) -> dict[str, Any] | None:
    if _WORKER_SERVICE is None:
        _init_worker()
    index, scenario = index_and_scenario
    failure = _run_check(_WORKER_SERVICE, scenario)  # type: ignore[arg-type]
    if failure:
        failure["scenario_index"] = index
    return failure


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local ConversationPolicy simulation checks.")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--failures", type=Path, default=Path("data/eval/conversation_policy_failures.jsonl"))
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    scenarios = _randomized_scenarios(args.count, args.seed)
    failures = 0
    completed = 0
    workers = max(1, int(args.workers or 1))
    if workers == 1:
        _init_worker()
        service = _WORKER_SERVICE
        assert service is not None
        for index, scenario in enumerate(scenarios, start=1):
            failure = _run_check(service, scenario)
            completed += 1
            if completed % max(1, args.progress_every) == 0:
                print(json.dumps({"status": "progress", "completed": completed, "count": args.count, "failures": failures}, ensure_ascii=False), flush=True)
            if failure:
                failure["scenario_index"] = index
                _append_failure(args.failures, failure)
                failures += 1
                print(json.dumps({"status": "failed", **failure}, ensure_ascii=False), flush=True)
                if args.stop_on_fail:
                    return 1
    else:
        indexed = list(enumerate(scenarios, start=1))
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as executor:
            pending = {executor.submit(_run_check_worker, item) for item in indexed[:workers * 2]}
            next_submit = workers * 2
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    completed += 1
                    failure = future.result()
                    if failure:
                        _append_failure(args.failures, failure)
                        failures += 1
                        print(json.dumps({"status": "failed", **failure}, ensure_ascii=False), flush=True)
                        if args.stop_on_fail:
                            executor.shutdown(wait=False, cancel_futures=True)
                            return 1
                    if completed % max(1, args.progress_every) == 0:
                        print(json.dumps({"status": "progress", "completed": completed, "count": args.count, "failures": failures}, ensure_ascii=False), flush=True)
                    if next_submit < len(indexed):
                        pending.add(executor.submit(_run_check_worker, indexed[next_submit]))
                        next_submit += 1
    print(json.dumps({"status": "ok" if failures == 0 else "failed", "count": args.count, "failures": failures, "failure_path": str(args.failures)}, ensure_ascii=False))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
