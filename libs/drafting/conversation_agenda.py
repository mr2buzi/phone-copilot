from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re

from libs.drafting.conversation_scene import ConversationScene
from libs.drafting.question_debt import QuestionDebt


def _norm(text: str) -> str:
    text = text.casefold().replace("`", "'").replace("â€™", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _strip_label(text: str) -> tuple[str, str]:
    raw = str(text).strip()
    lowered = raw.casefold()
    if lowered.startswith("[other]:"):
        return "other", raw.split(":", 1)[1].strip()
    if lowered.startswith("[me]:"):
        return "me", raw.split(":", 1)[1].strip()
    return "", raw


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


GENERIC_PROMPT_TYPES: dict[str, tuple[str, ...]] = {
    "what_u_saying": ("what u saying", "what you saying"),
    "what_u_been_up_to": ("what u been up to", "what you been up to", "whatchu been up to", "what have u been up to"),
    "what_u_doing": ("what u doing", "what you doing", "what are u doing", "wyd", "wuu2"),
    "ask_for_topic": ("give me a topic", "what should we talk about", "what topic"),
    "what_u_tryna_do": ("what u tryna do", "what you tryna do", "what u trying to do"),
    "what_u_been_doing": ("what u been doing", "what you been doing"),
    "blame_user_for_no_context": ("u ain't giving me much", "u aint giving me much", "giving me nothing", "not giving me anything"),
}


FORBIDDEN_MOVE_TERMS: dict[str, tuple[str, ...]] = {
    "ask_what_u_saying": ("what u saying", "what you saying"),
    "ask_what_u_been_up_to": ("what u been up to", "what you been up to", "whatchu been up to", "what have u been up to"),
    "ask_what_u_doing": ("what u doing", "what you doing", "what are u doing", "wyd", "wuu2"),
    "ask_for_topic": ("give me a topic", "what should we talk about", "what topic"),
    "blame_user_for_no_context": ("u ain't giving me much", "u aint giving me much", "giving me nothing", "not giving me anything"),
    "stale_self_state": ("same just chilling", "fair just chilling too", "same icl"),
    "ask_new_question_before_answering": ("what u saying", "what u doing", "what you doing", "what u been up to", "what u been doing", "wyd", "give me a topic"),
    "topic_shift": ("dream car", "cars or gym", "random question", "what should we talk about"),
}


@dataclass
class ConversationAgenda:
    agenda_state: str = "normal_flow"
    agenda_reason: str = ""
    recent_generic_prompt_count: int = 0
    repeated_prompt_callout_count: int = 0
    last_generic_prompts: list[str] = field(default_factory=list)
    exhausted_prompt_types: list[str] = field(default_factory=list)
    active_conversation_goal: str = ""
    proposed_topic: str = ""
    topic_attempts: int = 0
    user_engagement_level: str = "medium"
    bot_loop_detected: bool = False
    anti_loop_required: bool = False
    next_dialogue_move: str = "answer_directly"
    forbidden_dialogue_moves: list[str] = field(default_factory=list)
    conversation_progress_score: float = 0.5

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _generic_prompt_type(text: str) -> str:
    normalized = _norm(text).strip(" .?!")
    for prompt_type, terms in GENERIC_PROMPT_TYPES.items():
        if _contains_any(normalized, terms):
            return prompt_type
    return ""


def _is_repeated_prompt_callout(text: str) -> bool:
    normalized = _norm(text)
    return _contains_any(
        normalized,
        (
            "u asked this already",
            "you asked this already",
            "u already asked",
            "you already asked",
            "u alrdy asked",
            "you alrdy asked",
            "alrdy asked",
            "asked that already",
            "keep asking",
            "stop asking",
            "ur repeating",
            "you're repeating",
            "ur boring me",
            "youre boring me",
            "you're boring me",
        ),
    )


def _needs_topic_choice(text: str) -> bool:
    normalized = _norm(text).strip(" .?!")
    normalized = re.sub(r"\b(?:icl|ngl|tbh|lowk|lol|btw)\b", "", normalized).strip()
    return (
        normalized in {"idk talk", "u tell me", "you tell me", "pick a topic", "talk about anything", "entertain me", "idk what to say", "idk what to say tbh"}
        or _contains_any(normalized, ("idk what to say", "dont know what to say", "don't know what to say", "u tell me", "you tell me", "idk talk", "entertain me"))
    )


def _last_bot_asked_age(last_bot: str) -> bool:
    normalized = _norm(last_bot)
    return _contains_any(normalized, ("19 wby", "19 wbu", "im 19", "i'm 19")) and _contains_any(normalized, ("wby", "wbu", "what about u", "what about you"))


def build_conversation_agenda(
    conversation: list[str],
    *,
    scene: ConversationScene,
    safe_interests: list[str] | None = None,
    question_debt: QuestionDebt | None = None,
) -> ConversationAgenda:
    labelled = [_strip_label(item) for item in conversation[-14:] if str(item).strip()]
    user_turns = [text for speaker, text in labelled if speaker != "me" and text]
    bot_turns = [text for speaker, text in labelled if speaker == "me" and text]
    latest = user_turns[-1] if user_turns else (labelled[-1][1] if labelled else "")
    latest_norm = _norm(latest).strip(" .?!")
    last_bot = bot_turns[-1] if bot_turns else ""
    recent_bot = bot_turns[-6:]

    prompt_types = [_generic_prompt_type(item) for item in recent_bot]
    prompt_types = [item for item in prompt_types if item]
    exhausted = sorted({item for item in prompt_types if prompt_types.count(item) > 1})
    generic_count = len(prompt_types)
    callout_count = sum(1 for item in user_turns[-4:] if _is_repeated_prompt_callout(item))
    loop_detected = bool(exhausted) or callout_count > 0
    explicit_loop_callout = callout_count > 0
    topic_attempts = sum(1 for item in prompt_types if item == "ask_for_topic")

    interests = [item for item in (safe_interests or []) if item]
    proposed_topic = "cars" if "cars" in interests else (interests[0] if interests else "cars")
    agenda = ConversationAgenda(
        recent_generic_prompt_count=generic_count,
        repeated_prompt_callout_count=callout_count,
        last_generic_prompts=prompt_types[-4:],
        exhausted_prompt_types=exhausted,
        proposed_topic=proposed_topic,
        topic_attempts=topic_attempts,
        bot_loop_detected=loop_detected,
        anti_loop_required=loop_detected,
        conversation_progress_score=max(0.0, min(1.0, 0.75 - 0.12 * generic_count - 0.2 * callout_count)),
    )

    if question_debt and question_debt.has_unanswered_user_question and question_debt.user_called_out_unanswered_question:
        agenda.agenda_state = "repairing_bot_error" if question_debt.user_called_out_unanswered_question else "answering_direct_question"
        agenda.next_dialogue_move = "answer_unresolved_question"
        agenda.active_conversation_goal = "answer the unresolved user question before any hook or topic shift"
        agenda.agenda_reason = question_debt.question_debt_reason or "unanswered user question exists"
        agenda.anti_loop_required = agenda.anti_loop_required or question_debt.user_called_out_unanswered_question
        agenda.forbidden_dialogue_moves.extend([
            "generic_hook",
            "ask_new_question_before_answering",
            "one_word_ack",
            "topic_shift",
            "blame_user_for_no_context",
            "ask_what_u_saying",
            "ask_what_u_been_up_to",
            "ask_what_u_doing",
            "ask_for_topic",
        ])
        agenda.conversation_progress_score = max(agenda.conversation_progress_score, 0.82)
        return agenda

    if _last_bot_asked_age(last_bot) and latest_norm == "same":
        agenda.agenda_state = "identity_answer"
        agenda.next_dialogue_move = "make_observation"
        agenda.active_conversation_goal = "acknowledge the same-age answer normally"
        agenda.agenda_reason = "user answered the reciprocal age question"
        agenda.forbidden_dialogue_moves.extend(["blame_user_for_no_context", "ask_what_u_saying", "ask_what_u_been_up_to", "ask_what_u_doing"])
        agenda.conversation_progress_score = max(agenda.conversation_progress_score, 0.75)
        return agenda
    if latest_norm in {"yeah i feel u", "yh i feel u", "i feel u", "yeah i get u", "yh i get u"}:
        agenda.agenda_state = "normal_flow"
        agenda.next_dialogue_move = "make_observation"
        agenda.active_conversation_goal = "acknowledge the user's empathy and add a small self-disclosure"
        agenda.agenda_reason = "user lightly acknowledged the previous explanation"
        agenda.forbidden_dialogue_moves.extend(["ask_what_u_saying", "ask_what_u_been_up_to", "ask_what_u_doing", "generic_hook"])
        agenda.conversation_progress_score = max(agenda.conversation_progress_score, 0.72)
        return agenda
    if latest_norm in {"in bed", "bed", "in my bed", "just in bed"}:
        agenda.agenda_state = "normal_flow"
        agenda.next_dialogue_move = "make_observation"
        agenda.active_conversation_goal = "respond to the user's activity without blaming them"
        agenda.agenda_reason = "user gave a short activity update"
        agenda.forbidden_dialogue_moves.extend(["ask_what_u_saying", "ask_what_u_been_up_to", "ask_what_u_doing", "blame_user_for_no_context", "generic_hook"])
        agenda.conversation_progress_score = max(agenda.conversation_progress_score, 0.72)
        return agenda
    if _contains_any(latest_norm, ("what do u want me to say", "what do you want me to say", "what dyu want me to say", "what d'you want me to say")):
        if _generic_prompt_type(last_bot):
            agenda.agenda_state = "anti_loop_repair"
            agenda.next_dialogue_move = "acknowledge_repetition"
            agenda.active_conversation_goal = "repair the repeated prompt loop and stop asking exhausted generic questions"
            agenda.agenda_reason = "user challenged a generic prompt"
        else:
            agenda.agenda_state = "playful_banter"
            agenda.next_dialogue_move = "make_observation"
            agenda.active_conversation_goal = "acknowledge the awkward challenge without asking another generic prompt"
            agenda.agenda_reason = "user challenged the bot's previous banter"
        agenda.anti_loop_required = True
        agenda.forbidden_dialogue_moves.extend(["ask_what_u_saying", "ask_what_u_been_up_to", "ask_what_u_doing", "ask_for_topic", "blame_user_for_no_context", "generic_hook"])
        return agenda
    if scene.identity_answer_required or scene.scene_type in {"owner_age_question", "reciprocal_identity_answer"}:
        agenda.agenda_state = "identity_answer"
        agenda.next_dialogue_move = "answer_directly"
        agenda.active_conversation_goal = "answer the identity point without turning it into an interview"
        agenda.agenda_reason = "scene requires identity/direct-answer priority"
        return agenda

    if _needs_topic_choice(latest):
        agenda.agenda_state = "topic_selection_needed" if "idk what to say" not in latest_norm else "dead_conversation_recovery"
        agenda.next_dialogue_move = "choose_topic"
        agenda.active_conversation_goal = "pick a concrete safe topic and make the next reply easy"
        agenda.agenda_reason = "user asked the bot to lead or said they do not know what to say"
        agenda.anti_loop_required = True
        agenda.forbidden_dialogue_moves.extend([
            "ask_what_u_saying",
            "ask_what_u_been_up_to",
            "ask_what_u_doing",
            "ask_for_topic",
            "blame_user_for_no_context",
            "generic_hook",
            "repeat_previous_prompt",
        ])
        return agenda

    if explicit_loop_callout:
        agenda.agenda_state = "anti_loop_repair"
        agenda.next_dialogue_move = "acknowledge_repetition"
        agenda.active_conversation_goal = "repair the repeated prompt loop and stop asking exhausted generic questions"
        agenda.agenda_reason = "generic prompt loop or user repeated-question callout detected"
        agenda.forbidden_dialogue_moves.extend([
            "ask_what_u_saying",
            "ask_what_u_been_up_to",
            "ask_what_u_doing",
            "ask_for_topic",
            "blame_user_for_no_context",
            "generic_hook",
            "one_word_ack",
            "repeat_previous_prompt",
        ])
        return agenda

    if scene.repair_required or scene.explanation_required:
        agenda.agenda_state = "repairing_bot_error"
        agenda.next_dialogue_move = "repair_then_move_on"
        agenda.active_conversation_goal = "repair before trying to continue"
        agenda.agenda_reason = "scene requires repair/explanation priority"
        agenda.forbidden_dialogue_moves.extend(["ask_what_u_saying", "ask_what_u_been_up_to", "ask_what_u_doing", "ask_for_topic", "blame_user_for_no_context"])
        return agenda

    if scene.emotional_response_required:
        agenda.agenda_state = "emotional_support"
        agenda.next_dialogue_move = "answer_directly"
        agenda.active_conversation_goal = "respond to the emotional point first"
        agenda.agenda_reason = "scene requires emotional response"
    elif scene.topic_engagement_required or scene.scene_type == "topic_given":
        agenda.agenda_state = "topic_engagement"
        agenda.next_dialogue_move = "continue_active_topic"
        agenda.active_conversation_goal = "continue the active topic"
        agenda.agenda_reason = "scene has an active topic"
    elif generic_count >= 2 or exhausted:
        agenda.agenda_state = "user_bored_or_unengaged"
        agenda.next_dialogue_move = "choose_topic"
        agenda.active_conversation_goal = "progress the chat without another generic prompt"
        agenda.agenda_reason = "too many generic prompts recently"
        agenda.forbidden_dialogue_moves.extend(["ask_what_u_saying", "ask_what_u_been_up_to", "ask_what_u_doing", "ask_for_topic", "blame_user_for_no_context", "generic_hook"])
    else:
        agenda.agenda_state = "normal_flow"
        agenda.next_dialogue_move = "answer_directly"
        agenda.active_conversation_goal = "answer the current turn naturally"
        agenda.agenda_reason = "no loop or agenda intervention required"
    return agenda


def score_reply_against_agenda(
    reply: str,
    agenda: ConversationAgenda,
    *,
    previous_bot_replies: list[str] | None = None,
) -> dict[str, object]:
    normalized = _norm(reply).strip(" .?!")
    previous = [_norm(item).strip(" .?!") for item in (previous_bot_replies or [])]
    forbidden: list[str] = []

    for move, terms in FORBIDDEN_MOVE_TERMS.items():
        if move in agenda.forbidden_dialogue_moves and _contains_any(normalized, terms):
            forbidden.append(move)
    if "one_word_ack" in agenda.forbidden_dialogue_moves and normalized in {"ok", "okay", "calm", "fair", "true", "yeah", "yh", "lol"}:
        forbidden.append("one_word_ack")
    if "generic_hook" in agenda.forbidden_dialogue_moves and _generic_prompt_type(normalized):
        forbidden.append("generic_hook")
    if "repeat_previous_prompt" in agenda.forbidden_dialogue_moves and normalized in previous:
        forbidden.append("repeat_previous_prompt")
    if agenda.agenda_state == "identity_answer" and agenda.next_dialogue_move == "make_observation" and _contains_any(normalized, ("boring answer", "give me smth better", "give me something better")):
        forbidden.append("badger_identity_answer")

    required_satisfied = True
    if agenda.next_dialogue_move == "acknowledge_repetition":
        required_satisfied = _contains_any(normalized, ("asked that already", "asked already", "looping", "keep asking", "same thing", "npc", "my bad", "allow me", "ill pick", "i'll pick"))
    elif agenda.next_dialogue_move == "choose_topic":
        required_satisfied = _contains_any(normalized, ("random", "dream car", "cars", "gym", "training", "what's been on", "whats been on", "ill pick", "i'll pick", "lets talk", "let's talk"))
    elif agenda.next_dialogue_move == "answer_unresolved_question":
        required_satisfied = not (
            normalized in {"ok", "okay", "calm", "fair", "true", "yeah", "yh", "lol", "nah i get u", "nah i get you"}
            or _generic_prompt_type(normalized)
            or _contains_any(normalized, ("give me a topic", "what should we talk about", "u ain't giving me much", "u aint giving me much"))
        )
    elif agenda.next_dialogue_move == "make_observation" and agenda.agenda_state == "identity_answer":
        required_satisfied = _contains_any(normalized, ("twins", "same age", "valid", "same")) and "boring answer" not in normalized

    generic_prompt_penalty = 1.0 if _generic_prompt_type(normalized) and (
        agenda.anti_loop_required
        or agenda.agenda_state in {"topic_selection_needed", "dead_conversation_recovery", "user_bored_or_unengaged"}
        or "generic_hook" in agenda.forbidden_dialogue_moves
    ) else 0.0
    repeated_prompt_penalty = 1.0 if normalized in previous and _generic_prompt_type(normalized) else 0.0
    if agenda.anti_loop_required and generic_prompt_penalty:
        generic_prompt_penalty = 1.0
    agenda_fit = 0.8 if required_satisfied else 0.25
    if forbidden:
        agenda_fit = min(agenda_fit, 0.05)
    else:
        agenda_fit = min(1.0, agenda_fit + 0.15 * agenda.conversation_progress_score)
    progress = agenda.conversation_progress_score
    if agenda.next_dialogue_move == "choose_topic" and required_satisfied:
        progress = max(progress, 0.9)
    elif agenda.next_dialogue_move == "acknowledge_repetition" and required_satisfied:
        progress = max(progress, 0.85)
    elif agenda.next_dialogue_move == "answer_unresolved_question" and required_satisfied:
        progress = max(progress, 0.9)
    elif generic_prompt_penalty:
        progress = min(progress, 0.2)

    return {
        "agenda_state": agenda.agenda_state,
        "next_dialogue_move": agenda.next_dialogue_move,
        "bot_loop_detected": agenda.bot_loop_detected,
        "anti_loop_required": agenda.anti_loop_required,
        "exhausted_prompt_types": agenda.exhausted_prompt_types,
        "generic_prompt_penalty": round(generic_prompt_penalty, 3),
        "repeated_prompt_penalty": round(repeated_prompt_penalty, 3),
        "agenda_fit_score": round(agenda_fit, 3),
        "forbidden_dialogue_move_violated": bool(forbidden),
        "forbidden_dialogue_moves_violated": sorted(set(forbidden)),
        "conversation_progress_score": round(progress, 3),
        "chosen_topic": agenda.proposed_topic if agenda.next_dialogue_move == "choose_topic" else "",
    }
