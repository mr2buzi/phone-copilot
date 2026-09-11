from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re

from libs.drafting.intents import normalize_text


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _clean(text: str) -> str:
    return normalize_text(text).strip(" .?!")


def _without_trailing_discourse_fillers(text: str) -> str:
    words = text.split()
    fillers = {"tbh", "ngl", "icl", "lowk", "lowkey", "fr", "please", "pls"}
    while words and words[-1] in fillers:
        words.pop()
    return " ".join(words)


@dataclass(frozen=True)
class ConversationFunctionPrediction:
    function: str = "ordinary_reply"
    score: float = 0.0
    reason: str = ""
    slots: dict[str, str] = field(default_factory=dict)
    candidates: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _previous_bot(context: list[str]) -> str:
    return normalize_text(str(context[-1])) if context else ""


def _previous_claim_topic(previous_bot_norm: str) -> str:
    if not previous_bot_norm:
        return ""
    if _contains_any(previous_bot_norm, ("wanna kiss", "want to kiss", "kiss u", "kiss you", "really like u", "really like you", "proper like u", "proper like you", "want u", "want you", "miss u", "miss you", "thinking about u", "thinking about you", "obsessed with u", "obsessed with you", "can't get enough", "cant get enough")):
        return "romantic_affection"
    if _contains_any(previous_bot_norm, ("clinging", "outfit", "suited", "looked proper", "looked good", "tight on u", "tight on you")):
        return "physical_compliment"
    if _contains_any(previous_bot_norm, ("running out of things to say", "things to say", "get stuck", "got stuck", "forcing random questions", "looping back", "same questions", "fill the silence", "filling the silence", "real answer")):
        return "conversation_stuck"
    if _contains_any(previous_bot_norm, ("doing all the work", "do all the work", "carry the convo", "carry it", "making u carry", "making you carry", "making u do", "making you do")):
        return "conversation_effort"
    if _contains_any(previous_bot_norm, ("that was intense", "intense icl", "back and forth", "a lot icl")):
        return "intense_exchange"
    if _contains_any(previous_bot_norm, ("why is his dad involved", "wrong context", "made no sense", "random")):
        return "wrong_context"
    if _contains_any(previous_bot_norm, ("work", "coding", "project", "gym", "food", "shower", "bed", "watching", "chilling", "tired", "good", "calm", "dead day", "long day")):
        return "owner_state_or_activity"
    if len(previous_bot_norm.split()) >= 4:
        return "previous_statement_generic"
    return ""


def _story_context(context_blob: str) -> bool:
    return _contains_any(
        context_blob,
        ("random guy", "some guy", "living room", "came in", "ran in", "running in", "in ur home", "in your home"),
    )


def _prompt_topic(previous_bot_norm: str) -> str:
    if _contains_any(previous_bot_norm, ("best thing u ate", "best thing you ate", "weirdest thing u ate", "weirdest thing you ate", "weirdest thing youve eaten", "weirdest thing you've eaten", "food", "eat", "ate", "eaten", "eating", "burger", "craving")):
        return "food"
    if _contains_any(previous_bot_norm, ("film", "movie", "watched", "watch")):
        return "film"
    if _contains_any(previous_bot_norm, ("gym", "workout", "exercise", "training")):
        return "gym"
    if _contains_any(previous_bot_norm, ("dream", "dreamt")):
        return "dream"
    if _contains_any(previous_bot_norm, ("car", "range rover", "r8", "audi")):
        return "car"
    if _contains_any(previous_bot_norm, ("weirdest", "what happened", "story")):
        return "story"
    return "previous_prompt"


def _looks_like_prompt(previous_bot_norm: str) -> bool:
    if "?" in previous_bot_norm:
        return True
    return bool(
        re.search(
            r"\b(?:what|whats|what's|which|best|worst|last|least|favourite|favorite|weirdest|funniest)\b",
            previous_bot_norm,
        )
    )


def predict_conversation_function(*, incoming: str, context: list[str]) -> ConversationFunctionPrediction:
    incoming_norm = normalize_text(incoming)
    incoming_clean = _clean(incoming)
    incoming_function_clean = _without_trailing_discourse_fillers(incoming_clean)
    previous_bot_norm = _previous_bot(context)
    context_blob = " ".join(normalize_text(item) for item in context[-10:])

    scores: dict[str, float] = {}
    reasons: dict[str, str] = {}
    slots_by_function: dict[str, dict[str, str]] = {}

    def score(name: str, value: float, reason: str, slots: dict[str, str] | None = None) -> None:
        if value > scores.get(name, 0.0):
            scores[name] = value
            reasons[name] = reason
            slots_by_function[name] = slots or {}

    if incoming_clean in {"u tell me", "you tell me", "u tell me then", "you tell me then"} and _looks_like_prompt(previous_bot_norm):
        score(
            "bot_answer_own_prompt",
            0.91,
            "user bounced the bot's previous prompt back to the bot",
            {"prompt_topic": _prompt_topic(previous_bot_norm), "previous_prompt": previous_bot_norm[:120]},
        )

    if incoming_function_clean in {"go on", "go on then", "continue", "carry on", "and then", "say it then", "keep going", "mhmm keep going", "mhmmm keep going"} and previous_bot_norm and "?" not in previous_bot_norm:
        score(
            "continuation_prompt",
            0.86,
            "user asks the bot to continue or deepen its previous statement",
            {"continuation_topic": _previous_claim_topic(previous_bot_norm) or "previous_statement_generic", "previous_statement": previous_bot_norm[:120]},
        )

    if incoming_clean in {"wdym", "what do u mean", "what do you mean", "what you mean", "what u mean"}:
        topic = _previous_claim_topic(previous_bot_norm)
        if topic:
            score(
                "clarification_followup",
                0.92,
                "short confusion asks the bot to explain its previous claim",
                {"clarification_topic": topic, "previous_bot_claim": previous_bot_norm[:120]},
            )
        else:
            score("clarification_followup", 0.62, "short confusion with no concrete previous claim", {})

    if incoming_clean in {"yeah why", "yh why", "why", "why though", "but why", "how come"} or re.fullmatch(r"(?:yeah|yh|ok|okay)?\s*why(?:\s+(?:then|though))?", incoming_clean):
        score("reason_followup", 0.88, "short why asks for reason behind the previous statement", {"reason_target": "previous_bot"})

    if incoming_function_clean in {"no like actually", "like actually", "no actually"} and previous_bot_norm and not _story_context(context_blob):
        if _contains_any(previous_bot_norm, ("being silly", "bit silly", "got stuck", "stuck in my head", "nervous", "awkward", "burnt out", "trying to fill silence", "filling silence")):
            score("reason_followup", 0.87, "user asks for the real reason behind a weak meta explanation", {"reason_target": "previous_bot"})
        elif _previous_claim_topic(previous_bot_norm) in {"romantic_affection", "physical_compliment"}:
            score(
                "sincerity_challenge",
                0.9,
                "user is challenging whether the previous romantic or complimentary claim was actually meant",
                {"challenge_target": "sincerity_confirmation", "previous_bot_claim": previous_bot_norm[:120]},
            )

    if _contains_any(incoming_norm, ("be more specific", "more specific", "specific", "explain properly", "say it properly")):
        score("specificity_request", 0.93, "user asks for more specificity", {"repair_target": "specificity_request"})

    if _contains_any(incoming_norm, ("thats dry", "that's dry", "that is dry", "ur dry", "youre dry", "you're dry", "being dry", "boring me", "ur boring", "u boring", "youre boring", "you're boring", "dead reply", "too vague", "being vague")):
        score("quality_complaint", 0.94, "user complains the reply quality is dry or vague", {"repair_target": "dry_or_unclear_reply"})

    if incoming_clean in {"what would u do", "what would you do", "what would u do then", "what would you do then"} and _story_context(context_blob):
        score("story_hypothetical", 0.93, "user asks how the bot would react to the current story", {"story_function": "hypothetical_reaction"})

    if _contains_any(incoming_norm, ("what happened", "then what", "what did he do", "what did she do", "are u okay", "are you okay")) and _story_context(context_blob):
        score("story_reaction", 0.72, "user is continuing a story thread", {"story_function": "continuation"})

    if _contains_any(incoming_norm, ("what do u do", "what do you do", "what u do for work", "what you do for work", "u work", "you work")):
        score("identity_question", 0.9, "user asks owner identity/work fact", {"identity_fact": "work_status"})
    elif _contains_any(incoming_norm, ("what uni", "what course", "what do u study", "what do you study", "u at uni", "you at uni")):
        score("identity_question", 0.9, "user asks owner education fact", {"identity_fact": "education_status"})

    if _contains_any(incoming_norm, ("wyd", "what u doing", "what you doing", "what u up to", "what you up to", "where u", "where are u")):
        score("activity_question", 0.82, "user asks owner activity or whereabouts", {"activity_scope": "owner_activity"})

    if _contains_any(incoming_norm, ("im bored", "i'm bored", "entertain me", "carry the convo", "carry convo", "talk then", "talk to me", "say something then")):
        score("topic_initiative_request", 0.9, "user asks bot to lead conversation", {"conversation_need": "carry_or_pick_topic"})

    if incoming_function_clean in {"same", "nice", "oh fairs", "oh fair", "fairs", "fair", "okay then", "ok then", "trust me"}:
        score("low_effort_ack", 0.8, "user gave a short acknowledgement", {"ack_type": "light_ack"})

    if not scores:
        return ConversationFunctionPrediction(candidates={})

    function, top_score = max(scores.items(), key=lambda item: item[1])
    return ConversationFunctionPrediction(
        function=function,
        score=round(top_score, 3),
        reason=reasons.get(function, ""),
        slots=slots_by_function.get(function, {}),
        candidates={key: round(value, 3) for key, value in sorted(scores.items(), key=lambda item: item[1], reverse=True)},
    )
