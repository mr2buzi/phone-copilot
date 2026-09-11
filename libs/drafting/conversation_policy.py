from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re

from libs.drafting.conversation_agenda import ConversationAgenda
from libs.drafting.conversation_scene import ConversationScene
from libs.drafting.question_debt import QuestionDebt


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").casefold()).strip()


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _strip_label(text: str) -> tuple[str, str]:
    raw = str(text).strip()
    lowered = raw.casefold()
    if lowered.startswith("[other]:"):
        return "other", raw.split(":", 1)[1].strip()
    if lowered.startswith("[me]:"):
        return "me", raw.split(":", 1)[1].strip()
    return "", raw


@dataclass
class ConversationPolicy:
    conversation_job: str = "normal_reply"
    policy_reason: str = ""
    user_emotion: str = ""
    user_intent: str = ""
    bot_obligation: str = ""
    must_answer: list[str] = field(default_factory=list)
    must_acknowledge: list[str] = field(default_factory=list)
    must_repair: list[str] = field(default_factory=list)
    must_continue_topic: list[str] = field(default_factory=list)
    must_avoid: list[str] = field(default_factory=list)
    suggested_reply_shapes: list[str] = field(default_factory=list)
    policy_priority: int = 40
    provider_assist_allowed: bool = True
    confidence: float = 0.55

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


GENERIC_BAD = (
    "ok",
    "okay",
    "what u saying",
    "what u saying then",
    "yo what u saying",
    "same icl",
    "same just chilling",
    "fair just chilling too",
    "fine then what should we talk about",
    "u ain't giving me much",
    "u aint giving me much",
    "alr random question then dream car",
)


def build_conversation_policy(
    conversation: list[str],
    *,
    scene: ConversationScene,
    agenda: ConversationAgenda,
    relationship_type: str = "unknown",
    question_debt: QuestionDebt | None = None,
) -> ConversationPolicy:
    labelled = [_strip_label(item) for item in conversation[-14:] if str(item).strip()]
    user_turns = [text for speaker, text in labelled if speaker != "me" and text]
    bot_turns = [text for speaker, text in labelled if speaker == "me" and text]
    latest = user_turns[-1] if user_turns else ""
    latest_norm = _norm(latest)
    last_bot = bot_turns[-1] if bot_turns else ""
    last_bot_norm = _norm(last_bot)
    recent_user_blob = " ".join(_norm(item) for item in user_turns[-4:])
    recent_bot_blob = " ".join(_norm(item) for item in bot_turns[-4:])

    policy = ConversationPolicy(
        user_intent=scene.user_intent,
        user_emotion=scene.latest_user_emotion,
        must_avoid=list(dict.fromkeys([*scene.forbidden_reply_moves, *agenda.forbidden_dialogue_moves])),
        provider_assist_allowed=relationship_type not in {"professional", "family", "university"},
    )

    story_setup = _contains_any(recent_user_blob, ("guess what", "guess", "story", "listen"))
    vivid_story = _contains_any(
        latest_norm,
        (
            "some guy",
            "someone",
            "came in",
            "ran in",
            "running in",
            "living room",
            "showering",
            "shower",
            "at home",
        ),
    ) and len(latest_norm.split()) >= 8
    dry_or_weird_callout = _contains_any(
        latest_norm,
        ("dry", "weird", "weirdo", "boring", "bruh wtf", "wtf", "what the fuck", "mate", "being weird"),
    )
    bare_confusion = latest_norm.strip(" .?!") in {"?", "??", "???", "what", "bruh", "bruh wtf", "wtf", "uhhhh", "uhh"} or _contains_any(
        latest_norm, ("wdym", "what do u mean", "what are u on")
    )
    care_question = _contains_any(
        latest_norm,
        ("is everything alright", "is everything okay", "are u alright", "are you alright", "u okay", "are u ok", "u good", "you good"),
    )
    bot_was_bad = _contains_any(last_bot_norm, GENERIC_BAD) or _contains_any(
        recent_bot_blob,
        ("bruh wtf is valid", "what bruh wtf u into", "u ain't giving me much", "fine then what should we talk about"),
    )

    missed_affection_callout = _contains_any(
        latest_norm,
        ("asked u to say u miss me", "asked you to say you miss me", "say u miss me", "say you miss me"),
    )

    if question_debt and question_debt.has_unanswered_user_question and question_debt.user_called_out_unanswered_question:
        policy.conversation_job = "answer_question_debt"
        policy.policy_reason = question_debt.question_debt_reason or "unanswered user question exists"
        policy.bot_obligation = "answer the unresolved user question before any new question or topic shift"
        policy.must_answer = [question_debt.unanswered_question_type or "unanswered_user_question"]
        if question_debt.user_called_out_unanswered_question:
            policy.must_repair = ["missed unanswered question"]
        policy.must_avoid.extend(["generic_hook", "ask_for_topic", "blame_user", "one_word_ack", "ask_new_question_before_answering", "generic_topic_shift"])
        policy.suggested_reply_shapes = ["yh my bad im just chilling", "my bad im good icl", "icl i missed that, im chilling"]
        policy.policy_priority = 98
        policy.confidence = 0.97
    elif scene.scene_type == "repeated_reply_callout":
        policy.conversation_job = "acknowledge_repeated_reply"
        policy.policy_reason = "user called out repeated bot replies"
        policy.bot_obligation = "acknowledge the repetition specifically"
        policy.must_repair = ["repeated reply"]
        policy.must_acknowledge = ["repetition callout"]
        policy.must_avoid.extend(["generic_hook", "ask_for_topic", "blame_user", "one_word_ack"])
        policy.suggested_reply_shapes = ["yh fairs i repeated myself icl", "caught me icl", "yh that was NPC behaviour"]
        policy.policy_priority = 96
        policy.confidence = 0.95
    elif scene.scene_type == "missed_affection_callout" or missed_affection_callout:
        policy.conversation_job = "answer_affection"
        policy.policy_reason = "user called out missed affection"
        policy.user_emotion = "affectionate"
        policy.bot_obligation = "repair the missed affection directly"
        policy.must_acknowledge = ["missed affection"]
        policy.must_repair = ["missed affection"]
        policy.must_avoid.extend(["generic_hook", "ask_for_topic", "blame_user", "stale_self_state", "one_word_ack"])
        policy.suggested_reply_shapes = ["yeah ur right missed u too icl", "my bad missed u too", "icl i dodged that, missed u too"]
        policy.policy_priority = 97
        policy.confidence = 0.96
    elif scene.scene_type == "weird_story" or vivid_story or (story_setup and scene.scene_type in {"normal", "direct_question"} and len(latest_norm.split()) >= 6):
        policy.conversation_job = "react_to_story"
        policy.policy_reason = "user gave a vivid story after a setup"
        policy.user_emotion = "surprised"
        policy.bot_obligation = "react to the story with surprise and ask one relevant detail"
        policy.must_acknowledge = ["the unusual story"]
        policy.must_answer = ["react to what happened"]
        policy.must_avoid.extend(["one_word_ack", "generic_topic_shift", "blame_user", "treat_confusion_as_topic"])
        policy.suggested_reply_shapes = [
            "BRO WHAT why was he running",
            "nah wait some random guy came in ur living room?",
            "what do u mean came in like into ur house?",
        ]
        policy.policy_priority = 92
        policy.confidence = 0.94
    elif care_question:
        policy.conversation_job = "answer_care_check"
        policy.policy_reason = "user asked if everything is alright"
        policy.user_emotion = "caring"
        policy.bot_obligation = "answer the care question directly before continuing"
        policy.must_answer = ["say whether the bot is alright"]
        policy.must_acknowledge = ["care"]
        policy.must_avoid.extend(["generic_topic_shift", "ask_for_topic", "blame_user", "one_word_ack"])
        policy.suggested_reply_shapes = ["yeah im good dw", "im alright icl my replies were just weird", "yeah im okay my bad"]
        policy.policy_priority = 95
        policy.confidence = 0.95
    elif (
        (dry_or_weird_callout and bot_was_bad)
        or (bare_confusion and bot_was_bad)
        or scene.scene_type in {"dry_complaint", "missed_context_callout", "topic_ignored_callout"}
    ) and scene.scene_type not in {
        "serious_boundary",
        "repeated_reply_callout",
        "owner_claim_contradiction",
        "owner_activity_clarification",
        "owner_activity_detail_question",
    } and scene.required_reply_move not in {"answer_owner_day_activity", "answer_reciprocal_activity", "answer_wellbeing_checkin"}:
        policy.conversation_job = "repair_after_weird_or_dry_reply"
        policy.policy_reason = "user called out weird/dry context or confusion"
        policy.user_emotion = "annoyed"
        policy.bot_obligation = "acknowledge the bad reply and stop topic shifting"
        policy.must_repair = ["previous bad reply"]
        policy.must_acknowledge = ["callout"]
        policy.must_avoid.extend(["generic_topic_shift", "ask_for_topic", "blame_user", "treat_confusion_as_topic", "one_word_ack"])
        if scene.scene_type == "topic_ignored_callout" and scene.active_topic:
            policy.suggested_reply_shapes = [
                f"yh my bad i missed {scene.active_topic}",
                f"icl i ignored {scene.active_topic} there",
                f"fairs i asked for a topic then missed {scene.active_topic}",
            ]
        elif scene.scene_type == "missed_context_callout" and _contains_any(_norm(scene.latest_user_message), ("why u missing context", "why you missing context")):
            policy.suggested_reply_shapes = [
                "icl i repeated myself instead of clocking what u asked",
                "yeah my bad i answered like u asked wyd again",
                "fairs i missed what u said and defaulted to chilling",
            ]
        elif scene.scene_type == "missed_context_callout":
            policy.suggested_reply_shapes = [
                "fairs i missed that",
                "yh my bad i forgot",
                "icl i bugged there",
            ]
        elif scene.scene_type == "dry_complaint":
            policy.suggested_reply_shapes = [
                "yh that was dead from me icl",
                "yeah fairs that was dry icl",
                "icl i was waffling there my bad",
            ]
        else:
            policy.suggested_reply_shapes = ["yeah that was dry icl my bad", "yh that was weird icl my bad", "nah ur right that made no sense"]
        policy.policy_priority = 94
        policy.confidence = 0.93
    elif scene.scene_type == "emotional_affection":
        policy.conversation_job = "answer_affection"
        policy.policy_reason = "user expressed affection"
        policy.user_emotion = "affectionate"
        policy.bot_obligation = "acknowledge the affection without using a stale hook"
        policy.must_acknowledge = ["affection"]
        policy.must_avoid.extend(["generic_hook", "ask_for_topic", "blame_user", "stale_self_state", "one_word_ack"])
        if not scene.flirt_allowed:
            policy.suggested_reply_shapes = ["aww bless u", "that's sweet icl", "appreciate u"]
        elif "missed you" in _norm(scene.latest_user_message) and not _contains_any(_norm(scene.latest_user_message), ("baby", "babe", "my love", " ml")):
            policy.suggested_reply_shapes = ["aww bless u", "that's sweet icl", "appreciate u"]
        else:
            policy.suggested_reply_shapes = ["aww bless u where u been", "that's sweet icl where u been", "missed u too icl"]
        policy.policy_priority = 88
        policy.confidence = 0.9
    elif scene.required_reply_move == "answer_reciprocal_activity":
        policy.conversation_job = "answer_reciprocal_activity"
        policy.policy_reason = "user answered an activity question and asked back"
        policy.bot_obligation = "answer the wby/u/hbu directly without reopening the chat"
        policy.must_answer = ["reciprocal activity question"]
        policy.must_avoid.extend(["generic_hook", "ask_for_topic", "blame_user", "one_word_ack"])
        if scene.repair_required:
            policy.must_repair = ["missed reciprocal question"]
        policy.suggested_reply_shapes = ["nothing much just chilling", "same just been chilling", "same icl just chilling"]
        policy.policy_priority = 90
        policy.confidence = 0.92
    elif scene.scene_type in {"tired_mood", "positive_mood_update"}:
        policy.conversation_job = "respond_to_user_mood"
        policy.policy_reason = "user gave a mood update"
        policy.user_emotion = scene.latest_user_emotion or "mood"
        policy.bot_obligation = "respond to the mood with presence and one relevant continuation"
        policy.must_acknowledge = ["user mood"]
        policy.must_avoid.extend(["generic_hook", "ask_for_topic", "blame_user", "one_word_ack", "stale_self_state"])
        policy.suggested_reply_shapes = ["why u tired", "long day?", "as u should why u happy"]
        policy.policy_priority = 86
        policy.confidence = 0.88
    elif scene.scene_type == "light_acknowledgement":
        policy.conversation_job = "playful_acknowledge"
        policy.policy_reason = "user gave a light acknowledgement or playful banter cue"
        policy.bot_obligation = "acknowledge lightly without reopening the chat"
        policy.must_acknowledge = ["light acknowledgement"]
        policy.must_avoid.extend(["generic_hook", "ask_for_topic", "blame_user", "stale_self_state"])
        if latest_norm.strip(" .?!") == "behave":
            policy.suggested_reply_shapes = ["nah u behave", "u behave", "make me"]
        else:
            policy.suggested_reply_shapes = ["lool ur good", "yh ur good", "fairs fairs"]
        policy.policy_priority = 74
        policy.confidence = 0.82
    elif agenda.agenda_state in {"topic_selection_needed", "dead_conversation_recovery", "user_bored_or_unengaged"}:
        policy.conversation_job = "progress_conversation"
        policy.policy_reason = "agenda says conversation needs progression"
        policy.bot_obligation = "choose a concrete direction without looping"
        policy.must_avoid.extend(["ask_for_topic", "repeat_previous_prompt", "blame_user"])
        policy.suggested_reply_shapes = ["alr random one then dream car?", "fine ill pick, cars or gym", "what's been on ur mind"]
        policy.policy_priority = 72
        policy.confidence = 0.8
    elif scene.identity_answer_required or scene.direct_answer_required:
        policy.conversation_job = "answer_direct_question"
        policy.policy_reason = "scene requires direct answer"
        policy.bot_obligation = "answer the direct question first"
        policy.must_answer = [scene.required_reply_move]
        policy.must_avoid.extend(["generic_hook", "ask_for_topic", "blame_user", "one_word_ack"])
        policy.policy_priority = 84
        policy.confidence = 0.86
    else:
        policy.conversation_job = "normal_reply"
        policy.policy_reason = "no policy intervention required"
        policy.bot_obligation = "reply naturally and keep context"
        policy.policy_priority = 40
        policy.confidence = 0.58

    policy.must_avoid = list(dict.fromkeys(policy.must_avoid))
    return policy


def score_reply_against_policy(reply: str, policy: ConversationPolicy) -> dict[str, object]:
    normalized = _norm(reply).strip(" .?!")
    violations: list[str] = []
    must_satisfied = True

    progress_topic_reply = policy.conversation_job == "progress_conversation" and _contains_any(
        normalized,
        ("dream car", "random question", "cars", "gym", "ill pick", "i'll pick", "lets talk", "let's talk"),
    )
    if normalized in GENERIC_BAD and not progress_topic_reply:
        violations.append("generic_bad_reply")
    if "one_word_ack" in policy.must_avoid and normalized in {"ok", "okay", "k", "calm", "fair", "true"}:
        violations.append("one_word_ack")
    if "generic_topic_shift" in policy.must_avoid and _contains_any(
        normalized,
        (
            "what should we talk about",
            "dream car",
            "random question",
            "cars or gym",
            "what's been on ur mind",
            "what one u picking",
            "would u get",
            "what u into",
        ),
    ):
        violations.append("generic_topic_shift")
    if "blame_user" in policy.must_avoid and _contains_any(normalized, ("u ain't giving me much", "u aint giving me much", "giving me nothing")):
        violations.append("blame_user")
    if "treat_confusion_as_topic" in policy.must_avoid and (
        _contains_any(normalized, ("bruh wtf is valid", "what bruh wtf u into", "meow is valid"))
        or (" is valid icl what " in normalized and " u into" in normalized)
    ):
        violations.append("treat_confusion_as_topic")
    if "ask_for_topic" in policy.must_avoid and _contains_any(normalized, ("give me a topic", "what should we talk about")):
        violations.append("ask_for_topic")
    if "generic_hook" in policy.must_avoid and _contains_any(
        normalized,
        (
            "what u saying",
            "what you saying",
            "what u been up to",
            "what you been up to",
            "what u doing",
            "what you doing",
            "what u been doing",
            "what you been doing",
        ),
    ):
        violations.append("generic_hook")
    if policy.conversation_job != "answer_reciprocal_activity" and "stale_self_state" in policy.must_avoid and _contains_any(normalized, ("same icl", "same just chilling", "fair just chilling", "just chilling")):
        violations.append("stale_self_state")
    if "ask_new_question_before_answering" in policy.must_avoid and _contains_any(
        normalized,
        ("what u saying", "what u doing", "what you doing", "what u been up to", "what u been doing", "wyd", "give me a topic"),
    ):
        violations.append("ask_new_question_before_answering")

    if policy.conversation_job == "react_to_story":
        must_satisfied = _contains_any(normalized, ("what", "nah", "bro", "wait", "how", "why", "living room", "came in", "running"))
    elif policy.conversation_job == "answer_question_debt":
        must_satisfied = not (
            normalized in {"ok", "okay", "k", "calm", "fair", "true", "nah i get u", "nah i get you"}
            or _contains_any(normalized, ("what u saying", "say less what u doing", "give me a topic", "u ain't giving me much", "u aint giving me much"))
        )
        if policy.must_repair:
            must_satisfied = must_satisfied and _contains_any(normalized, ("my bad", "missed", "ignored", "didnt answer", "didn't answer", "dodged"))
    elif policy.conversation_job == "answer_care_check":
        must_satisfied = _contains_any(normalized, ("im good", "i'm good", "im alright", "i'm alright", "im okay", "i'm okay", "dw", "my bad"))
    elif policy.conversation_job == "repair_after_weird_or_dry_reply":
        must_satisfied = _contains_any(normalized, ("my bad", "weird", "dry", "made no sense", "answered", "ur right", "you're right", "fairs"))
    elif policy.conversation_job == "acknowledge_repeated_reply":
        must_satisfied = _contains_any(normalized, ("repeated", "caught me", "bugged", "npc", "same thing"))
    elif policy.conversation_job == "answer_affection":
        must_satisfied = _contains_any(normalized, ("missed", "miss u", "sweet", "bless", "appreciate", "where u been", "where you been"))
    elif policy.conversation_job == "answer_reciprocal_activity":
        must_satisfied = _contains_any(normalized, ("same", "nothing much", "not much", "working", "sorting", "gym", "coding", "in bed", "sleep", "chilling"))
        if policy.must_repair:
            must_satisfied = must_satisfied and _contains_any(normalized, ("my bad", "missed", "ignored", "didn't answer", "didnt answer"))
    elif policy.conversation_job == "respond_to_user_mood":
        must_satisfied = _contains_any(normalized, ("tired", "sleep", "nap", "long day", "happy", "good", "cute", "deserve", "why u", "how come"))
    elif policy.conversation_job == "playful_acknowledge":
        must_satisfied = _contains_any(normalized, ("ur good", "lool", "lol", "fairs", "behave", "make me", "allow it"))
    elif policy.conversation_job == "answer_direct_question":
        must_satisfied = not _contains_any(normalized, ("what u saying", "give me a topic", "what should we talk about"))
    elif policy.conversation_job == "progress_conversation":
        must_satisfied = _contains_any(normalized, ("dream car", "cars", "gym", "random", "ill pick", "i'll pick", "lets talk", "let's talk", "what's been on", "whats been on"))

    if not must_satisfied:
        violations.append("policy_must_not_satisfied")

    fit = 0.9 if must_satisfied else 0.25
    if violations:
        fit = min(fit, 0.05)
    progress = 0.82 if must_satisfied and not violations else 0.15
    return {
        "policy_fit_score": round(fit, 3),
        "policy_must_satisfied": must_satisfied,
        "policy_violation": bool(violations),
        "policy_violation_reasons": sorted(dict.fromkeys(violations)),
        "policy_conversation_progress_score": round(progress, 3),
    }


def should_reply(
    contact_name: str,
    latest_incoming: str,
    context_entries: list[dict[str, object]] | list[object],
    contact_state: object,
    intent_type: str,
    *,
    allowlisted: bool = True,
    rate_limited: bool = False,
) -> dict[str, object]:
    """Local automation gate used by live send loops.

    This is intentionally conservative and separate from drafting policy. It
    decides whether the system should even draft/send, not what wording to use.
    """
    normalized = _norm(latest_incoming)
    if not allowlisted:
        return {"should_reply": False, "needs_review": True, "reason": "not_allowlisted"}
    if rate_limited:
        return {"should_reply": False, "needs_review": True, "reason": "rate_limited"}
    if not normalized:
        return {"should_reply": False, "needs_review": False, "reason": "empty_latest_incoming"}
    if normalized in {"👍", "👌", "🙏", "❤️", "😂", "🤣", "💀"}:
        return {"should_reply": False, "needs_review": False, "reason": "reaction_only"}
    last_hash = getattr(contact_state, "last_seen_message_hash", None)
    if last_hash:
        import hashlib

        current_hash = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        if current_hash == last_hash:
            return {"should_reply": False, "needs_review": False, "reason": "duplicate_latest_incoming"}
    needs_review = intent_type in {"emotional", "conflict", "romantic", "sexual"} or _contains_any(
        normalized,
        ("ignore me", "ignored me", "why did u ignore", "missed you", "missed u", "love you", "love u"),
    )
    return {
        "should_reply": True,
        "needs_review": bool(needs_review),
        "reason": "new_incoming",
        "contact_name": contact_name,
    }
