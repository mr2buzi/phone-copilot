from __future__ import annotations

from dataclasses import asdict, dataclass
import re


def _norm(text: str) -> str:
    text = str(text or "").casefold().replace("`", "'").replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _strip_label(text: str) -> tuple[str, str]:
    raw = str(text or "").strip()
    lowered = raw.casefold()
    if lowered.startswith("[other]:"):
        return "other", raw.split(":", 1)[1].strip()
    if lowered.startswith("[me]:"):
        return "me", raw.split(":", 1)[1].strip()
    return "", raw


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


@dataclass
class QuestionDebt:
    has_unanswered_user_question: bool = False
    unanswered_question_text: str | None = None
    unanswered_question_type: str | None = None
    question_asked_turn_index: int | None = None
    question_debt_age_turns: int = 0
    user_called_out_unanswered_question: bool = False
    answer_required_now: bool = False
    question_debt_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


CALLOUT_TERMS = (
    "i js asked u a question",
    "i just asked u a question",
    "i asked u a question",
    "i asked you a question",
    "are u gonna answer",
    "are you gonna answer",
    "answer me",
    "u didn't answer",
    "you didn't answer",
    "u didnt answer",
    "you didnt answer",
    "u ignored my question",
    "you ignored my question",
    "bro answer",
)

ONE_WORD_OR_FILLER = {
    "ok",
    "okay",
    "k",
    "calm",
    "fair",
    "true",
    "yh",
    "yeah",
    "nah i get u",
    "nah i get you",
}

GENERIC_HOOK_TERMS = (
    "what u saying",
    "what you saying",
    "what u doing",
    "what you doing",
    "wyd",
    "what u been up to",
    "what you been up to",
    "what u been doing",
    "give me a topic",
    "what should we talk about",
    "u ain't giving me much",
    "u aint giving me much",
    "say less what u doing",
)


def _classify_user_question(text: str) -> tuple[str, str] | None:
    normalized = _norm(text).strip(" .?!")
    if not normalized:
        return None
    if _contains_any(normalized, ("how old r u", "how old are you", "what age are u", "age?")):
        return "direct_identity_question", "how old r u"
    if _contains_any(normalized, ("where u from", "where do u live", "where are u based")):
        return "location_question", "where u from"
    if _contains_any(normalized, ("what do u study", "what dyu study", "what course", "what uni")):
        return "study_question", "what do u study"
    if _contains_any(normalized, ("what project", "what u building", "what you building", "what u working on", "what you working on")):
        return "project_question", "what project"
    if _contains_any(normalized, ("dream car", " car", " cars", "gym", "boxing", "tech", "software", "ai ", "project", "business")) and re.search(r"\b(wby|wbu|hbu)\b", normalized):
        return "reciprocal_topic_question", "wby"
    has_reciprocal = bool(re.search(r"\b(wby|wbu|hbu)\b", normalized) or re.search(r"\b(?:u|you)\s*$", normalized))
    has_activity_answer = _contains_any(
        normalized,
        (
            "nth",
            "nothing",
            "chilling",
            "in bed",
            "at home",
            "work",
            "working",
            "busy",
            "sleeping",
            "sleep",
            "prolly",
            "good",
            "calm",
            "not much",
        ),
    )
    if has_reciprocal and has_activity_answer:
        if _contains_any(normalized, ("good", "calm", "alright", "happy")):
            return "reciprocal_checkin_question", "wby"
        return "reciprocal_activity_question", "wby"
    if re.search(r"\b(wby|wbu|hbu)\b", normalized):
        return "reciprocal_checkin_question", "wby"
    if _contains_any(normalized, ("hru", "how are u", "how are you", "how r u")):
        return "direct_status_question", "hru"
    if _contains_any(normalized, ("where u been", "what u been doing", "what u been up to", "what have u been up to")):
        return "direct_status_question", "what u been up to"
    if _contains_any(normalized, ("how was ur day", "how was your day", "how did ur day go")):
        return "day_question", "how was ur day"
    if _contains_any(normalized, ("are u ok", "are you ok", "u okay", "u good", "are u alright")):
        return "care_question", "are u ok"
    if _contains_any(normalized, ("wdym", "what do u mean", "what do you mean")):
        return "clarification_question", "wdym"
    if "?" in text or re.match(r"^(what|why|how|where|when|who)\b", normalized):
        return "generic_direct_question", normalized
    return None


def _bot_answered_question(bot_text: str, question_type: str) -> bool:
    normalized = _norm(bot_text).strip(" .?!")
    if not normalized or normalized in ONE_WORD_OR_FILLER:
        return False
    if _contains_any(normalized, GENERIC_HOOK_TERMS):
        return False
    if question_type == "reciprocal_activity_question":
        if _contains_any(normalized, ("im good", "i'm good", "im calm", "im alright", "im okay", "i'm okay")):
            return False
        return _contains_any(
            normalized,
            ("just", "nothing", "not much", "chilling", "working", "gym", "coding", "in bed", "sorting"),
        )
    if question_type in {"reciprocal_checkin_question", "direct_status_question"}:
        return _contains_any(normalized, ("im good", "i'm good", "im calm", "im alright", "im okay", "not bad", "good icl", "yeah im"))
    if question_type == "reciprocal_topic_question":
        return _contains_any(normalized, ("urus", "r8", "m4", "rs6", "porsche", "lambo", "audi", "bmw", "merc", "mine", "i'd", "id ", "probably", "gym", "boxing", "training"))
    if question_type == "direct_identity_question":
        return bool(re.search(r"\b19\b", normalized))
    if question_type == "location_question":
        return _contains_any(normalized, ("northbridge", "northbridge", "sampleford"))
    if question_type == "study_question":
        return _contains_any(normalized, ("comp sci", "computer science", "study"))
    if question_type == "project_question":
        return _contains_any(normalized, ("project", "building", "dev thing", "software", "side"))
    if question_type == "day_question":
        return _contains_any(normalized, ("calm", "dead", "busy", "long", "decent", "not bad", "good"))
    if question_type == "care_question":
        return _contains_any(normalized, ("im good", "i'm good", "im okay", "i'm okay", "dw", "alright"))
    if question_type == "clarification_question":
        return _contains_any(normalized, ("i mean", "i meant", "my bad", "ignored", "missed", "answered wrong"))
    return len(normalized.split()) >= 3 and not _contains_any(normalized, GENERIC_HOOK_TERMS)


def detect_question_debt(conversation: list[str]) -> QuestionDebt:
    labelled = [_strip_label(item) for item in conversation[-12:] if str(item).strip()]
    if not labelled:
        return QuestionDebt()

    latest_speaker, latest_text = labelled[-1]
    latest_norm = _norm(latest_text).strip(" .?!")
    latest_question = _classify_user_question(latest_text) if latest_speaker != "me" else None
    latest_is_callout = latest_speaker != "me" and _contains_any(latest_norm, CALLOUT_TERMS)

    if latest_question and not latest_is_callout and latest_question[0] in {"reciprocal_activity_question", "reciprocal_checkin_question"}:
        question_type, question_text = latest_question
        return QuestionDebt(
            has_unanswered_user_question=True,
            unanswered_question_text=question_text,
            unanswered_question_type=question_type,
            question_asked_turn_index=len(labelled) - 1,
            question_debt_age_turns=0,
            user_called_out_unanswered_question=False,
            answer_required_now=True,
            question_debt_reason="latest user message asks a direct question",
        )

    unresolved: QuestionDebt | None = None
    for index, (speaker, text) in enumerate(labelled[:-1]):
        if speaker == "me":
            continue
        classified = _classify_user_question(text)
        if not classified:
            continue
        question_type, question_text = classified
        if question_type in {"generic_direct_question", "clarification_question"}:
            continue
        next_bot = ""
        for follow_speaker, follow_text in labelled[index + 1 :]:
            if follow_speaker == "me":
                next_bot = follow_text
                break
            if follow_speaker != "me":
                break
        if not next_bot or not _bot_answered_question(next_bot, question_type):
            unresolved = QuestionDebt(
                has_unanswered_user_question=True,
                unanswered_question_text=question_text,
                unanswered_question_type=question_type,
                question_asked_turn_index=index,
                question_debt_age_turns=max(0, len(labelled) - index - 1),
                user_called_out_unanswered_question=latest_is_callout,
                answer_required_now=latest_is_callout,
                question_debt_reason="earlier user question was not answered",
            )

    if unresolved and latest_is_callout:
        if latest_is_callout:
            unresolved.question_debt_reason = "user called out an unanswered question"
        return unresolved
    if latest_is_callout:
        return QuestionDebt(
            has_unanswered_user_question=True,
            unanswered_question_text=None,
            unanswered_question_type="generic_direct_question",
            question_asked_turn_index=None,
            question_debt_age_turns=0,
            user_called_out_unanswered_question=True,
            answer_required_now=True,
            question_debt_reason="user called out an unanswered question but original question was not found",
        )
    return QuestionDebt()


def question_debt_reply_pool(question_debt: QuestionDebt) -> list[str]:
    if not question_debt.has_unanswered_user_question:
        return []
    question_type = question_debt.unanswered_question_type or "generic_direct_question"
    repair = question_debt.user_called_out_unanswered_question or question_debt.question_debt_age_turns > 0
    if question_type == "reciprocal_activity_question":
        return [
            "yh my bad im just chilling",
            "my bad, nothing much just chilling",
            "icl i missed that, im chilling",
        ] if repair else [
            "nothing much just chilling",
            "not much just sorting stuff",
            "js chilling icl",
        ]
    if question_type in {"reciprocal_checkin_question", "direct_status_question"}:
        return [
            "my bad im good icl",
            "yh my bad im good",
            "im good icl",
        ] if repair else [
            "im good wbu",
            "im calm wbu",
            "yeah im good dw",
        ]
    if question_type == "reciprocal_topic_question":
        return ["r8 is cold icl id probs go urus", "mine would be an m4 icl", "id go rs6 or urus"]
    if question_type == "direct_identity_question":
        return ["my bad im 19", "19 wby", "im 19"]
    if question_type == "location_question":
        return ["my bad northbridge n sampleford", "northbridge n sampleford wby", "northbridge but sampleford for uni"]
    if question_type == "study_question":
        return ["my bad comp sci", "computer science", "comp sci wby"]
    if question_type == "project_question":
        return ["this project im building is lowk killing me", "got a dev thing running on the side", "building software stuff icl"]
    if question_type == "day_question":
        return ["my bad it was calm icl", "busy icl but calm", "not bad tbh just chilled"]
    if question_type == "care_question":
        return ["yeah im good dw", "im okay, just tired icl", "yeah im alright"]
    if question_type == "clarification_question":
        return ["i mean i answered the wrong thing", "my bad i missed what u meant", "i meant i bugged and ignored the question"]
    return ["yh my bad i missed the question", "my bad i didnt answer u", "icl i dodged the question there"]


def score_reply_against_question_debt(reply: str, question_debt: QuestionDebt) -> dict[str, object]:
    normalized = _norm(reply).strip(" .?!")
    has_debt = question_debt.has_unanswered_user_question
    answer_required = question_debt.answer_required_now
    question_type = question_debt.unanswered_question_type or ""
    answer_score = 0.0
    ignored_penalty = 0.0
    new_question_penalty = 0.0

    if has_debt:
        answered = _bot_answered_question(normalized, question_type)
        repair_ack = not question_debt.user_called_out_unanswered_question or _contains_any(
            normalized,
            ("my bad", "missed", "ignored", "didnt answer", "didn't answer", "dodged"),
        )
        answer_score = 1.0 if answered and repair_ack else 0.45 if answered else 0.0
        if answer_required and answer_score < 0.7:
            ignored_penalty = 1.0
        if answer_required and _contains_any(normalized, GENERIC_HOOK_TERMS) and answer_score < 0.7:
            new_question_penalty = 1.0
        if answer_required and normalized in ONE_WORD_OR_FILLER:
            ignored_penalty = 1.0
    return {
        "has_unanswered_user_question": has_debt,
        "unanswered_question_type": question_debt.unanswered_question_type or "",
        "unanswered_question_text": question_debt.unanswered_question_text or "",
        "question_debt_age_turns": question_debt.question_debt_age_turns,
        "user_called_out_unanswered_question": question_debt.user_called_out_unanswered_question,
        "answer_required_now": answer_required,
        "question_debt_answer_score": round(answer_score, 3),
        "ignored_question_debt_penalty": round(ignored_penalty, 3),
        "asked_new_question_before_answering_penalty": round(new_question_penalty, 3),
    }
