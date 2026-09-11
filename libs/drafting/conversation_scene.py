from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re


def _norm(text: str) -> str:
    text = text.casefold().replace("’", "'").replace("`", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _strip_label(text: str) -> tuple[str, str]:
    raw = text.strip()
    lowered = raw.casefold()
    if lowered.startswith("[other]:"):
        return "other", raw.split(":", 1)[1].strip()
    if lowered.startswith("[me]:"):
        return "me", raw.split(":", 1)[1].strip()
    return "", raw


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _contains_word_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)


def _is_tech_topic_text(text: str) -> bool:
    normalized = _norm(text)
    return _contains_word_any(normalized, ("ai", "software", "coding", "backend", "tech")) or _contains_any(
        normalized,
        ("projects", "building", "systems", "programming", "apps"),
    )


def _activity_from_text(text: str) -> str:
    normalized = _norm(text)
    if _contains_any(normalized, ("in bed", "bed rn", "lying in bed", "laying in bed")):
        return "in bed"
    if _contains_any(normalized, ("sleeping", "going sleep", "go sleep", "probably sleeping", "prolly sleeping", "sleep rn", "nap rn")):
        return "sleeping"
    if _contains_any(normalized, ("working", "at work", "work rn", "just work", "doing work", "on work")) or re.search(r"\bwork\s+(?:wby|wbu|hbu|u|you)\b", normalized):
        return "working"
    if _contains_any(normalized, ("just chilling", "chilling", "chillin")):
        return "chilling"
    if _contains_any(normalized, ("nothing", "not much", "not a lot", "not alot", "nm")):
        return "nothing"
    if _contains_any(normalized, ("at home", "home rn", "im home", "i'm home")):
        return "at home"
    return ""


def _looks_like_topic(text: str) -> bool:
    normalized = _norm(text).strip(" .?!")
    if re.fullmatch(r"\d+", normalized):
        return False
    if normalized in {"cars", "hm cars", "gym", "music", "food", "uni", "work", "football", "movies"}:
        return True
    if len(normalized.split()) <= 3 and re.fullmatch(r"(hm\s+)?[a-z0-9][a-z0-9\s&'-]{1,30}", normalized):
        low_info = {"hi", "hey", "yo", "ok", "okay", "yh", "yeah", "lol", "idk", "nothing", "same", "calm"}
        words = set(normalized.split())
        conversational_words = {"i", "im", "i'm", "me", "my", "u", "you", "ur", "your", "missed", "miss", "love", "need", "want"}
        return normalized not in low_info and not bool(words & conversational_words)
    return False


def _clean_topic_value(text: str) -> str:
    topic = _norm(text).removeprefix("hm ").strip(" .?!")
    topic = re.sub(r"\b(?:lowk|icl|ngl|tbh|lol|then)\b", "", topic).strip(" .?!")
    topic = re.sub(r"\s+", " ", topic).strip()
    return topic


def _topic_from_interest_answer(text: str, last_bot: str) -> str:
    normalized = _norm(text).strip(" .?!")
    last_bot_norm = _norm(last_bot)
    match = re.search(r"\b(?:i'?m|im|i am)\s+into\s+([a-z0-9][a-z0-9\s&'-]{1,30})$", normalized)
    if match:
        topic = _clean_topic_value(match.group(1))
        if topic not in {"it", "that", "this", "nothing"}:
            return topic
    if not _contains_any(last_bot_norm, ("what u into", "what you into", "what are u into", "what are you into")):
        return ""
    return ""


def _previous_bot_claim_type(text: str) -> str:
    normalized = _norm(text).strip(" .?!")
    if not normalized:
        return ""
    if _contains_any(normalized, ("uni", "gym", "coding", "clients", "projects", "project", "stuff")) and _contains_any(normalized, ("busy", "been", "working", "all of it")):
        return "owner_activity_summary"
    if _contains_any(normalized, ("it was calm", "bit dead", "long day", "decent", "not bad", "dead day", "boring")):
        return "day_summary"
    if _contains_any(normalized, ("just chilling", "nothing much", "nothing crazy", "same icl", "same just", "not much")):
        return "current_activity"
    if _contains_any(normalized, ("my bad", "i bugged", "bugged", "waffling", "dead reply", "wrong thing")):
        return "repair"
    if _contains_any(normalized, ("missed u", "missed you", "love u", "sweet", "bless")):
        return "emotional_statement"
    if _contains_any(normalized, ("dad involved", "his dad", "father involved")):
        return "weird_story_reaction"
    if "random" in normalized:
        return "random_topic_misread"
    return "topic_statement" if len(normalized.split()) >= 3 else ""


def _fallback_scene_for_reply(text: str) -> tuple[str, str]:
    normalized = _norm(text).strip(" .?!")
    mapping = {
        "bro why is his dad involved": ("weird_story", "ask_one_relevant_question"),
        "nah what how does that happen": ("weird_story", "ask_one_relevant_question"),
        "that whole sentence is insane icl what": ("weird_story", "ask_one_relevant_question"),
        "same just chilling": ("current_activity_exchange", "answer_directly"),
        "same icl": ("current_activity_exchange", "answer_directly"),
        "fair just chilling too": ("current_activity_exchange", "answer_directly"),
        "just chilling": ("current_activity_exchange", "answer_directly"),
        "what u doing rn": ("current_activity_exchange", "ask_one_relevant_question"),
        "what u saying then": ("opening", "ask_one_relevant_question"),
        "say less what u doing": ("dead_conversation", "ask_one_relevant_question"),
        "fine then give me a topic": ("dead_conversation", "ask_one_relevant_question"),
        "im trying icl give me a topic": ("dead_conversation", "ask_one_relevant_question"),
        "u ain't giving me much to work with": ("dead_conversation", "ask_one_relevant_question"),
        "u aint giving me much to work with": ("dead_conversation", "ask_one_relevant_question"),
    }
    return mapping.get(normalized, ("", ""))


def _identity_question_kind(text: str) -> str:
    normalized = _norm(text).strip(" .?!")
    if not normalized:
        return ""
    if _contains_any(normalized, ("how old", "what age", "age?")) or re.search(r"\b(age|wby age)\b", normalized):
        return "age"
    if re.search(r"\b(i'?m|im|like im|like i'm)\s+19\b.*\b(wby|hbu|what about u|what about you)\b", normalized):
        return "age"
    if _contains_any(normalized, ("where u from", "where you from", "where are u from", "where are you from", "where do u live", "where do you live", "where are u based", "where are you based")):
        return "location"
    if _contains_any(normalized, ("where u at uni", "where are u at uni", "where you at uni", "what uni")):
        return "study"
    if _contains_any(normalized, ("what do u study", "what do you study", "what dyu study", "what d'you study", "what d you study", "what dya study", "wdyu study", "what course", "u at uni", "you at uni")):
        return "study"
    if _contains_any(normalized, ("what project", "what u building", "what you building", "what u working on", "what you working on", "what side thing", "what business")):
        return "project"
    if _contains_any(normalized, ("what do u do", "what do you do", "what dyu do", "what d'you do", "what d you do", "what dya do", "what u do", "what you do", "u work", "you work", "what work do u do", "what work do you do")):
        return "work"
    if _contains_any(normalized, ("where u been", "where you been")):
        return "where_been"
    if _contains_any(normalized, ("what u been doing", "what you been doing", "what have u been doing", "what u been up to", "what you been up to", "what have u been up to", "whatchu been up to")):
        return "recent_activity"
    if _contains_any(normalized, ("doing anything nice", "u doing anything nice", "you doing anything nice", "wyd later", "what u doing later", "what you doing later", "what u doing tonight", "what you doing tonight")):
        return "doing_anything_nice"
    return ""


def _is_wellbeing_checkin(text: str) -> bool:
    normalized = _norm(text)
    return _contains_any(
        normalized,
        (
            "how are u",
            "how are you",
            "how r u",
            "hru",
            "how u been",
            "how you been",
            "how have u been",
            "how have you been",
            "how you doing",
            "how u doing",
            "how's u",
            "hows u",
        ),
    )


def _is_reciprocal_wellbeing_answer(text: str) -> bool:
    normalized = _norm(text).strip(" .?!")
    asks_back = _contains_any(
        normalized,
        (" wby", " wbu", " hbu", "what about u", "what about you", "how about u", "how about you"),
    ) or bool(re.search(r"\b(?:u|you)$", normalized))
    positive_state = _contains_any(
        normalized,
        (
            "im good",
            "i'm good",
            "i been good",
            "ive been good",
            "i've been good",
            "been good",
            "good",
            "im calm",
            "i'm calm",
            "im bless",
            "i'm bless",
            "alright",
            "happy",
        ),
    )
    return asks_back and positive_state


def _is_positive_mood_update(text: str) -> bool:
    normalized = _norm(text)
    return _contains_any(
        normalized,
        (
            "i been happy",
            "ive been happy",
            "i've been happy",
            "im happy",
            "i'm happy",
            "just been happy",
            "js been happy",
            "been happy",
            "i feel good",
        ),
    )


@dataclass
class ConversationScene:
    scene_type: str = "normal"
    scene_summary: str = ""
    latest_user_message: str = ""
    latest_user_ask: str = ""
    latest_user_emotion: str = ""
    user_intent: str = "other"
    social_task: str = ""
    required_reply_move: str = "answer_directly"
    forbidden_reply_moves: list[str] = field(default_factory=list)
    known_recent_facts: list[str] = field(default_factory=list)
    recent_bot_mistakes: list[str] = field(default_factory=list)
    recent_bot_replies: list[str] = field(default_factory=list)
    recent_bot_questions: list[str] = field(default_factory=list)
    unresolved_user_points: list[str] = field(default_factory=list)
    previous_bot_claim: str = ""
    previous_bot_claim_type: str = ""
    latest_user_refers_to_previous_bot_claim: bool = False
    explanation_required: bool = False
    explanation_target: str = ""
    active_topic: str = ""
    topic_was_provided: bool = False
    topic_value: str = ""
    bot_repeated_itself: bool = False
    user_called_out_bot: bool = False
    repair_required: bool = False
    emotional_response_required: bool = False
    affection_reciprocity_required: bool = False
    care_response_required: bool = False
    identity_answer_required: bool = False
    identity_question_kind: str = ""
    identity_disclosure_allowed: bool = False
    thread_message_count: int = 0
    direct_answer_required: bool = False
    topic_engagement_required: bool = False
    reciprocal_activity_required: bool = False
    day_check_response_required: bool = False
    hook_allowed: bool = True
    hook_required: bool = False
    relationship_context: str = "unknown"
    flirt_allowed: bool = False
    safety_mode: str = "normal"
    confidence: float = 0.5

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_conversation_scene(
    conversation: list[str],
    *,
    relationship_type: str = "unknown",
    flirt_allowed: bool = False,
    intent_type: str = "other",
    recent_bot_questions: list[str] | None = None,
) -> ConversationScene:
    labelled = [_strip_label(item) for item in conversation[-12:] if str(item).strip()]
    user_turns = [text for speaker, text in labelled if speaker != "me" and text]
    bot_turns = [text for speaker, text in labelled if speaker == "me" and text]
    latest = user_turns[-1] if user_turns else (labelled[-1][1] if labelled else "")
    latest_norm = _norm(latest)
    last_bot = bot_turns[-1] if bot_turns else ""
    last_bot_norm = _norm(last_bot)
    recent_bot_norms = [_norm(item).strip(" .?!") for item in bot_turns[-3:]]
    bot_repeated = len(recent_bot_norms) >= 2 and len(set(recent_bot_norms)) < len(recent_bot_norms)
    bot_questions = [item for item in bot_turns[-5:] if item.strip().endswith("?") or _contains_any(_norm(item), ("what ", "why ", "how ", "wyd", "wuu2"))]
    previous_claim_type = _previous_bot_claim_type(last_bot)
    recent_owner_activity_claim = ""
    for prior_bot in reversed(bot_turns[-5:]):
        if _previous_bot_claim_type(prior_bot) == "owner_activity_summary":
            recent_owner_activity_claim = prior_bot
            break
    recent_owner_coding_claim = recent_owner_activity_claim
    if not recent_owner_coding_claim:
        for prior_bot in reversed(bot_turns[-8:]):
            if _contains_any(_norm(prior_bot), ("coding", "code", "software", "project", "comp sci", "computer science")):
                recent_owner_coding_claim = prior_bot
                break

    topic_value = ""
    bot_asked_topic_recently = any("give me a topic" in _norm(item) or "what should we talk about" in _norm(item) for item in bot_turns[-3:])
    recent_interest_prompt = any(_contains_any(_norm(item), ("what u into", "what you into", "what are u into", "what are you into")) for item in [*user_turns[-4:-1], *bot_turns[-4:]])
    latest_explicit_interest = _contains_any(latest_norm, ("tech", "coding", "cars", "gym", "boxing", "business", "software", "ai "))
    if (bot_asked_topic_recently or recent_interest_prompt or latest_explicit_interest) and _looks_like_topic(latest):
        topic_value = _clean_topic_value(latest_norm)
    interest_topic = _topic_from_interest_answer(latest, last_bot)
    if interest_topic:
        topic_value = interest_topic
    active_topic = topic_value
    for prior in reversed(user_turns[:-1]):
        prior_norm = _norm(prior).strip(" .?!")
        if _looks_like_topic(prior_norm):
            active_topic = active_topic or _clean_topic_value(prior_norm)
            break
    recent_topic_context = _norm(" ".join([*user_turns[-5:], *bot_turns[-5:]]))
    if _contains_any(recent_topic_context, ("dream car", " car", "cars", "urus", "r8", "rs6", "m4", "porsche", "lambo", "audi", "bmw", "merc")):
        active_topic = "cars"
    elif _is_tech_topic_text(recent_topic_context):
        active_topic = active_topic or "tech"
    story_setup = any(_contains_any(_norm(item), ("guess what", "bro listen", "go on", "what happened")) for item in user_turns[-4:-1])
    vivid_story = _contains_any(
        latest_norm,
        ("some guy", "someone", "came in", "ran in", "running in", "living room", "showering", "shower", "at home"),
    ) and len(latest_norm.split()) >= 8
    recent_story_context = any(
        _contains_any(_norm(item), ("some guy", "someone", "came in", "running in", "living room", "showering", "shower", "at home"))
        for item in user_turns[-4:-1]
    )
    story_confirmation = recent_story_context and latest_norm.strip(" .?!") in {"nah like actually", "like actually", "actually", "fr", "deadass", "nah fr", "no like actually"}
    story_hypothetical = recent_story_context and _contains_any(latest_norm, ("what would u do", "what would you do", "what u doing there", "what would you have done"))

    repeated_callout = _contains_any(
        latest_norm,
        (
            "u said that",
            "you said that",
            "said the same",
            "repeating",
            "repeated",
            "same shit",
            "same thing",
            "keep saying",
            "keep replying",
            "keep sending",
            "u already asked",
            "you already asked",
            "u alrdy asked",
            "you alrdy asked",
            "alrdy asked",
        ),
    )
    missed_context = _contains_any(
        latest_norm,
        (
            "i just told",
            "i js told",
            "i just did",
            "i js did",
            "i told and asked",
            "told and asked",
            "i told u and asked",
            "i told you and asked",
            "told u and asked",
            "told you and asked",
            "i answered and asked",
            "i answered and asked u",
            "i answered and asked you",
            "i said and asked",
            "i said and asked u",
            "i said and asked you",
            "why u missing context",
            "why you missing context",
            "not listening",
            "what are u on about",
            "ur confusing me",
            "confusing me",
        ),
    )
    topic_ignored = missed_context and bool(active_topic) and ("give me a topic" in last_bot_norm or "topic" in last_bot_norm)
    dry = _contains_any(
        latest_norm,
        (
            "bit dry",
            "ur dry",
            "you're dry",
            "youre dry",
            "too dry",
            "being dry",
            "thats dry",
            "that's dry",
            "dry mate",
        ),
    )
    weird_or_confusion_callout = dry or _contains_any(
        latest_norm,
        (
            "ur being weird",
            "you're being weird",
            "youre being weird",
            "being weird",
            "ur weird",
            "you're weird",
            "youre weird",
            "weirdo",
            "bruh wtf",
            "wtf",
            "what the fuck",
        ),
    )
    affection = _contains_any(latest_norm, ("i missed u", "i missed you", "missed u", "missed you", "miss you", "i miss you", "love u", "love you", "i love you", "baby", "babe", "my love", " ml", "mwah", "my handsome", "handsome"))
    romantic_signoff = _contains_any(latest_norm, ("goodnight", "good night", "gn", "sleep well", "sleepwell", "sleep tight")) and (
        relationship_type == "romantic_interest"
        or affection
        or _contains_any(_norm(" ".join(user_turns[-4:])), ("love u", "love you", "miss u", "miss you", "baby", "babe", "my love", "mwah"))
    )
    conservative_affection = (
        not flirt_allowed
        or (
            _contains_any(latest_norm, ("i missed you", "missed you", "miss you", "i miss you"))
            and not _contains_any(latest_norm, ("baby", "babe", "my love", " ml"))
        )
    )
    reciprocity = _contains_any(latest_norm, ("say it back", "aint gon say it back", "ain't gon say it back", "dont u miss", "don't u miss", "u dont miss", "u don't miss", "didnt miss me", "didn't miss me", "you didnt miss me", "you didn't miss me"))
    ignored_question_callout = _contains_any(latest_norm, ("ignore my question", "ignored my question", "ignoring my question", "i asked u", "i asked you", "thats it ignore", "that's it ignore"))
    care = _contains_any(latest_norm, ("are u ok", "r u ok", "u seem off", "you seem off", "are you alright", "u good", "are u good"))
    serious = _contains_any(latest_norm, ("playing w my feelings", "playing with my feelings", "this isnt funny", "this isn't funny", "dont do this", "don't do this", "stop playing", "are u taking the piss", "are you taking the piss", "taking the piss"))
    sexual = _contains_any(latest_norm, ("im horny", "i'm horny", "need u", "want u", "i need u", "i want u"))
    statement_with_question_word = _contains_any(latest_norm, ("i dont know how", "i don't know how", "i do not know how", "idk how"))
    direct_question = (
        latest_norm.endswith("?")
        or _contains_any(latest_norm, ("r u mad", "are u mad", "why ", "what ", "whatd ", "what'd ", "what did "))
        or ("how " in latest_norm and not statement_with_question_word)
    )
    current_activity = _contains_any(latest_norm, ("wyd", "what u doing", "what you doing", "wuu2"))
    life_update = _contains_any(latest_norm, ("what u been up to", "what you been up to", "what have u been up to"))
    identity_kind = _identity_question_kind(latest_norm)

    scene = ConversationScene(
        user_intent=intent_type,
        relationship_context=relationship_type,
        flirt_allowed=flirt_allowed,
        safety_mode="expressive" if relationship_type in {"close_friend", "romantic_interest"} else "normal",
        thread_message_count=len(labelled),
        bot_repeated_itself=bot_repeated,
        active_topic=active_topic,
        topic_value=topic_value,
        topic_was_provided=bool(topic_value),
        known_recent_facts=[],
        recent_bot_mistakes=[],
        recent_bot_replies=bot_turns[-3:],
        recent_bot_questions=bot_questions,
        unresolved_user_points=[],
        previous_bot_claim=last_bot,
        previous_bot_claim_type=previous_claim_type,
    )

    if active_topic:
        scene.known_recent_facts.append(f"active_topic={active_topic}")
    if bot_repeated:
        scene.recent_bot_mistakes.append("bot repeated recent reply")
    previous_affection = any(_contains_any(_norm(item), ("i missed u", "i missed you", "missed u", "missed you", "miss you", "i miss you")) for item in user_turns[-4:-1])
    immediate_previous_affection = bool(user_turns[-2:-1]) and _contains_any(
        _norm(user_turns[-2]),
        ("i missed u", "i missed you", "missed u", "missed you", "miss you", "i miss you"),
    )
    previous_care = any(_contains_any(_norm(item), ("are u ok", "u seem off", "are you alright", "u good")) for item in user_turns[-4:-1])
    stale_last_bot = _contains_any(last_bot_norm, ("same just chilling", "fair just chilling", "same icl", "just chilling", "calm", "give me a topic", "ok", "okay"))
    latest_activity = _activity_from_text(latest_norm)
    reciprocal_activity = bool(latest_activity) and not identity_kind and (
        _contains_any(latest_norm, (" wby", " wbu", " hbu", " u", " you", "what about u", "what about you"))
        or current_activity
    )
    passive_user_activity_update = bool(latest_activity) and not identity_kind and not reciprocal_activity and not current_activity
    previous_reciprocal_activity = ""
    for prior_user in reversed(user_turns[-5:-1]):
        prior_norm = _norm(prior_user)
        prior_activity = _activity_from_text(prior_norm)
        if prior_activity and _contains_any(prior_norm, (" wby", " wbu", " hbu", " u", " you", "what about u", "what about you")):
            previous_reciprocal_activity = prior_activity
            break
    previous_reciprocal_unanswered = bool(previous_reciprocal_activity) and not _contains_any(
        last_bot_norm,
        (
            "same",
            "nothing much",
            "not much",
            "just chilling",
            "been chilling",
            "working",
            "in bed",
            "sleep",
            "sorting stuff",
            "gym",
            "coding",
            "uni",
            "projects",
        ),
    )
    day_check = _contains_any(latest_norm, ("how was ur day", "how was your day", "how's ur day", "hows ur day", "how was today", "how did ur day go", "how did your day go"))
    latest_simple = latest_norm.strip(" .?!")
    day_activity_question = _contains_any(
        latest_norm,
        (
            "whatd u do today",
            "what'd u do today",
            "what did u do today",
            "what did you do today",
            "what u do today",
            "what you do today",
            "what have u done today",
            "what did u get up to today",
            "what'd you get up to today",
        ),
    )
    reciprocal_topic_question = (
        not _is_reciprocal_wellbeing_answer(latest_norm)
        and not identity_kind
        and not bool(_activity_from_text(latest_norm))
        and _contains_any(latest_norm, (" wby", " wbu", " hbu", "what about u", "what about you"))
    ) and (
        _contains_any(
            latest_norm,
            ("dream car", " car", " cars", "gym", "boxing", "tech", "software", "ai ", "project", "business"),
        )
        or _contains_any(
            _norm(active_topic + " " + last_bot),
            ("car", "cars", "gym", "boxing", "tech", "software", "ai", "project", "business"),
        )
    )
    training_question = _contains_any(latest_norm, ("what u training", "what you training", "what would u train", "what would you train", "what u doing in gym", "what you doing in gym"))
    boxing_question = _contains_any(latest_norm, ("do u box", "do you box", "u box", "you box", "are u boxing", "are you boxing", "is boxing hard", "boxing hard"))
    playful_fight_challenge = _contains_any(latest_norm, ("would u fight me", "would you fight me", "could u fight me", "could you fight me", "fight me"))
    tech_interest_question = _contains_any(latest_norm, ("what part of tech", "what side of tech", "what bit of tech", "what tech"))
    owner_car_question = _contains_any(
        latest_norm,
        (
            "what car u like",
            "what car you like",
            "what car do u like",
            "what car do you like",
            "what cars u like",
            "what cars do u like",
            "what would u get if money wasnt a thing",
            "what would you get if money wasnt a thing",
            "what would u get if money wasn't a thing",
            "what would you get if money wasn't a thing",
        ),
    )
    owner_dream_question = _contains_any(latest_norm, ("whats ur dream", "what's ur dream", "what is ur dream", "what's your dream", "what is your dream"))
    owner_prayer_question = _contains_any(latest_norm, ("do u pray", "do you pray", "u pray", "you pray"))
    owner_tired_reason_question = _contains_any(latest_norm, ("why u tired", "why you tired", "why are u tired", "why are you tired"))
    car_answer = _contains_any(latest_norm, ("urus", "r8", "rs6", "m4", "porsche", "lambo", "audi", "bmw", "merc")) and _contains_any(
        _norm(active_topic + " " + last_bot),
        ("car", "dream car", "urus", "r8", "rs6", "m4"),
    )
    topic_choice_offered = bool(re.search(r"\b[a-z][a-z0-9&'-]+\s+or\s+[a-z][a-z0-9&'-]+\b", latest_simple)) and not direct_question
    topic_opinion = _contains_any(latest_norm, ("i hate ", "i love ", "i like ", "i don't like ", "i dont like "))
    previous_day_activity_question = previous_claim_type == "day_summary" and _contains_any(
        latest_norm,
        ("whatd u do", "what'd u do", "what did u do", "what did you do", "what u do then", "what you do then"),
    )
    confused_followup = latest_simple in {"what", "bro what", "huh", "what lol", "what wtf"}
    random_pushback = _contains_any(latest_norm, ("hows it random", "how's it random", "why random", "why is it random"))
    previous_claim_question = (
        latest_simple in {"how come", "why", "wdym", "how", "what", "what do u mean", "what do you mean"}
        or confused_followup
        or previous_day_activity_question
        or random_pushback
        or _contains_any(latest_norm, ("howd u mean", "how'd u mean", "how did u mean", "yeah so explain", "so explain", "yeah why", "yh why"))
    ) and bool(previous_claim_type)
    light_acknowledgement = latest_norm.strip(" .?!") in {
        "oh yeah silly me",
        "oh yh",
        "oh yeah true",
        "my bad",
        "oh fairs",
        "lol true",
        "oh yeah",
        "oh true",
        "yeah its ok",
        "yeah it's ok",
        "its ok",
        "it's ok",
        "no worries",
        "dw",
        "all good",
        "same tbh",
        "same icl tbh",
        "behave",
        "trust me",
        "same",
    }
    topic_positive_acknowledgement = bool(active_topic) and latest_norm.strip(" .?!") in {
        "thats sick",
        "that's sick",
        "sick",
        "thats cold",
        "that's cold",
        "cold",
        "thats cool",
        "that's cool",
        "cool",
        "hard",
        "thats hard",
        "that's hard",
        "valid",
        "yh valid",
        "yeah valid",
        "fair",
        "fairs",
    }
    tired_mood = _contains_any(latest_norm, ("im so tired", "i'm so tired", "im tired", "i'm tired", "im exhausted", "i'm exhausted", "im drained", "i'm drained", "sleepy"))
    missed_user_fact_callout = _contains_any(latest_norm, ("i js said", "i just said", "i told u", "i told you", "i just told u", "i just told you", "bro i js said", "i literally said"))
    vague_repair = _contains_any(
        last_bot_norm,
        (
            "yh fairs i did",
            "nah ur right i bugged",
            "i bugged",
            "my bad",
            "fairs",
            "made no sense",
            "dead reply",
            "answered badly",
            "that was dumb",
            "that was random",
        ),
    )
    repair_clarification = _contains_any(
        latest_norm,
        ("wdym u did", "wdym", "wym", "wytm", "what do u mean", "what do you mean", "what u mean", "what you mean", "how did u", "how did you", "how wtf"),
    ) and vague_repair
    low_info_after_bad_reply = latest_norm.strip(" .?!") in {"oh", "ok", "okay", "right", "lol", "nice"} and stale_last_bot
    wellbeing_checkin = _is_wellbeing_checkin(latest_norm) or _is_reciprocal_wellbeing_answer(latest_norm)
    positive_mood_update = _is_positive_mood_update(latest_norm)
    previous_wellbeing_unanswered = any(
        _is_wellbeing_checkin(_norm(item)) or _is_reciprocal_wellbeing_answer(_norm(item))
        for item in user_turns[-4:-1]
    ) and not _contains_any(
        last_bot_norm,
        ("im good", "i'm good", "im calm", "i'm calm", "good u", "good wbu", "not bad", "im bless", "i'm bless"),
    )
    direct_question_callout = _contains_any(
        latest_norm,
        ("i asked a question", "i asked u a question", "i asked you a question", "answer my question", "answer the question"),
    )
    owner_activity_followup = (bool(recent_owner_activity_claim) or previous_claim_type == "current_activity") and latest_simple in {"really", "rlly", "fr", "for real"}
    owner_gym_detail_question = bool(recent_owner_activity_claim) and _contains_any(
        latest_norm,
        ("what did u do in gym", "what did you do in gym", "whatd u do in gym", "what'd u do in gym", "what u do in gym", "what you do in gym"),
    )
    coding_user_fact = bool(recent_owner_coding_claim) and _contains_any(
        latest_norm,
        ("i dont know how to code", "i don't know how to code", "i do not know how to code", "idk how to code"),
    )
    coding_claim_callout = _contains_any(
        latest_norm,
        ("u lit do coding", "you lit do coding", "u literally do coding", "you literally do coding", "wdym u do coding", "wdym you do coding"),
    )
    false_claim_callout = _contains_any(
        latest_norm,
        ("it was a lie", "that was a lie", "was a lie", "u lied", "you lied", "it wasnt dead it was a lie", "it wasn't dead it was a lie"),
    )
    user_busy_update = _contains_any(
        latest_norm,
        ("i been busy", "i've been busy", "ive been busy", "been busy asf", "busy asf", "busy af"),
    )
    contradiction_callout = _contains_any(
        latest_norm,
        ("paradox", "contradiction", "contradict", "that makes no sense", "that made no sense", "doesnt make sense", "doesn't make sense"),
    ) and bool(bot_turns)
    last_bot_asked_age = _contains_any(last_bot_norm, ("19 wby", "19 wbu", "im 19", "i'm 19")) and _contains_any(last_bot_norm, ("wby", "wbu", "what about u", "what about you"))
    reciprocal_identity_answer = last_bot_asked_age and (
        bool(re.fullmatch(r"(?:i'?m\s+)?\d{1,2}", latest_simple)) or latest_simple == "same"
    )
    repair_pushback = _contains_any(
        latest_norm,
        (
            "didnt make sense",
            "didn't make sense",
            "doesnt make sense",
            "doesn't make sense",
            "not making sense",
            "im asking a question",
            "i'm asking a question",
            "not trying to be right",
        ),
    )
    unanswered_day_question = any(
        _contains_any(_norm(item), ("whatd u do", "what'd u do", "what did u do", "what did you do", "what u do today", "what you do today"))
        for item in user_turns[-5:-1]
    )
    if previous_wellbeing_unanswered and (ignored_question_callout or direct_question_callout or stale_last_bot):
        scene.unresolved_user_points.append("answer previous wellbeing question")
    if previous_affection and (ignored_question_callout or reciprocity or (stale_last_bot and immediate_previous_affection)):
        scene.unresolved_user_points.append("missed affection reciprocity")
    if previous_care and (ignored_question_callout or stale_last_bot):
        scene.unresolved_user_points.append("missed care check-in")
    if previous_reciprocal_unanswered and (missed_context or ignored_question_callout or direct_question_callout):
        scene.unresolved_user_points.append("answer previous reciprocal activity question")

    if direct_question_callout and previous_wellbeing_unanswered:
        scene.scene_type = "missed_context_callout"
        scene.required_reply_move = "answer_wellbeing_checkin"
        scene.social_task = "acknowledge the missed wellbeing question and answer it"
        scene.user_called_out_bot = True
        scene.repair_required = True
        scene.direct_answer_required = True
        scene.hook_allowed = False
        scene.unresolved_user_points.append("answer previous wellbeing question")
        scene.confidence = 0.96
    elif previous_reciprocal_unanswered and (missed_context or ignored_question_callout or direct_question_callout):
        scene.scene_type = "missed_context_callout"
        scene.required_reply_move = "answer_reciprocal_activity"
        scene.social_task = "acknowledge the missed wby and answer the reciprocal activity question"
        scene.user_called_out_bot = True
        scene.repair_required = True
        scene.direct_answer_required = True
        scene.reciprocal_activity_required = True
        scene.hook_allowed = False
        scene.known_recent_facts.append(f"recent_user_activity={previous_reciprocal_activity}")
        scene.confidence = 0.97
    elif repair_pushback and unanswered_day_question:
        scene.scene_type = "missed_context_callout"
        scene.required_reply_move = "answer_owner_day_activity"
        scene.social_task = "acknowledge the stale reply and answer what the bot did today"
        scene.user_called_out_bot = True
        scene.repair_required = True
        scene.direct_answer_required = True
        scene.hook_allowed = False
        scene.unresolved_user_points.append("answer previous day activity question")
        scene.confidence = 0.96
    elif repair_clarification and unanswered_day_question:
        scene.scene_type = "repair_clarification"
        scene.required_reply_move = "answer_owner_day_activity"
        scene.social_task = "explain the repair by answering the missed day activity question"
        scene.user_called_out_bot = True
        scene.repair_required = True
        scene.direct_answer_required = True
        scene.hook_allowed = False
        scene.unresolved_user_points.append("answer previous day activity question")
        scene.confidence = 0.96
    elif contradiction_callout:
        scene.scene_type = "contradiction_callout"
        scene.required_reply_move = "acknowledge_contradiction"
        scene.social_task = "acknowledge the contradiction before continuing"
        scene.user_called_out_bot = True
        scene.repair_required = True
        scene.hook_allowed = False
        scene.confidence = 0.92
    elif owner_gym_detail_question:
        scene.scene_type = "owner_activity_detail_question"
        scene.required_reply_move = "answer_owner_activity_detail"
        scene.social_task = "answer the user's follow-up about the earlier gym claim"
        scene.direct_answer_required = True
        scene.explanation_required = True
        scene.latest_user_refers_to_previous_bot_claim = True
        scene.previous_bot_claim = recent_owner_activity_claim
        scene.previous_bot_claim_type = "owner_activity_summary"
        scene.explanation_target = "gym"
        scene.hook_allowed = False
        scene.confidence = 0.96
    elif owner_activity_followup:
        scene.scene_type = "explain_previous_bot_claim"
        scene.required_reply_move = "explain_previous_bot_claim"
        scene.social_task = "confirm and lightly expand on the earlier activity claim"
        scene.direct_answer_required = True
        scene.explanation_required = True
        scene.latest_user_refers_to_previous_bot_claim = True
        scene.previous_bot_claim = recent_owner_activity_claim
        scene.previous_bot_claim_type = "owner_activity_summary"
        scene.explanation_target = "owner_activity_summary"
        scene.hook_allowed = False
        scene.confidence = 0.94
    elif coding_user_fact:
        scene.scene_type = "owner_activity_clarification"
        scene.required_reply_move = "clarify_owner_activity_claim"
        scene.social_task = "clarify that the earlier coding claim was about the bot/owner, not the user"
        scene.latest_user_refers_to_previous_bot_claim = True
        scene.previous_bot_claim = recent_owner_coding_claim
        scene.previous_bot_claim_type = "owner_activity_summary"
        scene.hook_allowed = False
        scene.confidence = 0.92
    elif coding_claim_callout or false_claim_callout:
        scene.scene_type = "owner_claim_contradiction"
        scene.required_reply_move = "acknowledge_and_correct_claim"
        scene.social_task = "acknowledge the contradiction and correct the owner activity claim"
        scene.user_called_out_bot = True
        scene.repair_required = True
        scene.latest_user_refers_to_previous_bot_claim = True
        scene.previous_bot_claim = recent_owner_activity_claim or last_bot
        scene.previous_bot_claim_type = "owner_activity_summary" if recent_owner_activity_claim else previous_claim_type
        scene.hook_allowed = False
        scene.confidence = 0.95
    elif user_busy_update:
        scene.scene_type = "user_activity_update"
        scene.required_reply_move = "respond_to_user_activity_update"
        scene.social_task = "respond to the user's busy update with a relevant follow-up"
        scene.latest_user_emotion = "busy"
        scene.hook_allowed = False
        scene.confidence = 0.88
    elif previous_claim_question:
        scene.scene_type = "explain_previous_bot_claim"
        scene.required_reply_move = "explain_previous_bot_claim"
        scene.social_task = "explain the previous bot claim or repair the random prior reply"
        scene.direct_answer_required = True
        scene.explanation_required = True
        scene.latest_user_refers_to_previous_bot_claim = True
        scene.explanation_target = previous_claim_type
        scene.hook_allowed = False
        if previous_claim_type == "weird_story_reaction":
            scene.repair_required = True
            scene.user_called_out_bot = True
            scene.recent_bot_mistakes.append("bot introduced random story context")
        if previous_day_activity_question:
            scene.unresolved_user_points.append("answer previous day activity question")
        scene.confidence = 0.96
    elif reciprocal_identity_answer:
        scene.scene_type = "reciprocal_identity_answer"
        scene.required_reply_move = "acknowledge_reciprocal_identity_answer"
        scene.social_task = "acknowledge the user's reciprocal identity answer without treating it as a random topic"
        scene.direct_answer_required = False
        scene.hook_allowed = False
        scene.confidence = 0.92
    elif topic_positive_acknowledgement:
        scene.scene_type = "topic_positive_acknowledgement"
        scene.required_reply_move = "acknowledge_topic_positive"
        scene.social_task = "acknowledge the user's positive reaction and keep the active topic moving lightly"
        scene.hook_allowed = False
        scene.confidence = 0.88
    elif light_acknowledgement:
        scene.scene_type = "light_acknowledgement"
        scene.required_reply_move = "playful_acknowledge_or_move_on"
        scene.social_task = "lightly acknowledge the user's self-correction"
        scene.hook_allowed = False
        scene.confidence = 0.9
    elif tired_mood:
        scene.scene_type = "tired_mood"
        scene.required_reply_move = "empathetic_casual_response"
        scene.social_task = "respond to tiredness with empathy or a light follow-up"
        scene.latest_user_emotion = "tired"
        scene.emotional_response_required = True
        scene.hook_allowed = False
        scene.confidence = 0.92
    elif serious:
        scene.scene_type = "serious_boundary"
        scene.required_reply_move = "deescalate"
        scene.social_task = "repair emotional harm and stop banter"
        scene.emotional_response_required = True
        scene.repair_required = True
        scene.hook_allowed = False
        scene.confidence = 0.95
        scene.latest_user_emotion = "hurt"
    elif reciprocity:
        scene.scene_type = "missed_affection_callout" if previous_affection or ignored_question_callout else "emotional_reciprocity"
        scene.required_reply_move = "reciprocate_affection" if flirt_allowed else "conservative_deflect"
        scene.social_task = "answer the affection request directly"
        scene.emotional_response_required = True
        scene.affection_reciprocity_required = True
        scene.hook_allowed = False
        scene.confidence = 0.95
    elif romantic_signoff:
        scene.scene_type = "romantic_signoff"
        scene.required_reply_move = "reciprocate_affection" if flirt_allowed else "conservative_deflect"
        scene.social_task = "reciprocate the romantic goodnight/signoff directly"
        scene.emotional_response_required = True
        scene.affection_reciprocity_required = flirt_allowed
        scene.hook_allowed = False
        scene.confidence = 0.94
    elif sexual:
        scene.scene_type = "sexual_flirty_energy"
        scene.required_reply_move = "safe_playful_flirty_response" if flirt_allowed else "conservative_deflect"
        scene.social_task = "respond playfully without explicit sexual content"
        scene.emotional_response_required = True
        scene.hook_allowed = False
        scene.confidence = 0.9
    elif ignored_question_callout and scene.unresolved_user_points:
        if "missed affection reciprocity" in scene.unresolved_user_points:
            scene.scene_type = "missed_affection_callout"
            scene.required_reply_move = "reciprocate_affection" if flirt_allowed else "conservative_deflect"
            scene.social_task = "repair the exact missed affection point"
            scene.emotional_response_required = True
            scene.affection_reciprocity_required = flirt_allowed
            scene.repair_required = True
            scene.user_called_out_bot = True
            scene.hook_allowed = False
            scene.confidence = 0.96
        else:
            scene.scene_type = "missed_context_callout"
            scene.required_reply_move = "acknowledge_and_repair"
            scene.social_task = "repair the exact missed user point"
            scene.repair_required = True
            scene.user_called_out_bot = True
            scene.hook_allowed = False
            scene.confidence = 0.9
    elif repair_pushback:
        scene.scene_type = "missed_context_callout"
        scene.required_reply_move = "acknowledge_and_repair"
        scene.social_task = "acknowledge the answer did not make sense and answer the unresolved question if possible"
        scene.user_called_out_bot = True
        scene.repair_required = True
        scene.hook_allowed = False
        if unanswered_day_question:
            scene.unresolved_user_points.append("answer previous day activity question")
        scene.confidence = 0.94
    elif repair_clarification:
        scene.scene_type = "repair_clarification"
        scene.required_reply_move = "clarify_previous_repair"
        scene.social_task = "explain the previous repair specifically"
        scene.user_called_out_bot = True
        scene.repair_required = True
        scene.hook_allowed = False
        scene.confidence = 0.94
    elif missed_user_fact_callout:
        scene.scene_type = "missed_user_fact_callout"
        scene.required_reply_move = "acknowledge_missed_fact"
        scene.social_task = "acknowledge the user already gave the fact"
        scene.user_called_out_bot = True
        scene.repair_required = True
        scene.hook_allowed = False
        if latest_activity:
            scene.known_recent_facts.append(f"recent_user_activity={latest_activity}")
        scene.confidence = 0.96
    elif low_info_after_bad_reply:
        scene.scene_type = "low_info_after_bad_reply"
        scene.required_reply_move = "repair_or_prompt_lightly"
        scene.social_task = "recover from the previous stale reply"
        scene.repair_required = True
        scene.hook_allowed = False
        scene.confidence = 0.9
    elif topic_ignored:
        scene.scene_type = "topic_ignored_callout"
        scene.required_reply_move = "acknowledge_and_repair"
        scene.social_task = "acknowledge missing the provided topic and engage it"
        scene.user_called_out_bot = True
        scene.repair_required = True
        scene.topic_engagement_required = True
        scene.hook_allowed = False
        scene.confidence = 0.95
    elif repeated_callout:
        scene.scene_type = "repeated_reply_callout"
        scene.required_reply_move = "acknowledge_and_repair"
        scene.social_task = "acknowledge repetition specifically"
        scene.user_called_out_bot = True
        scene.repair_required = True
        scene.hook_allowed = False
        scene.confidence = 0.95
    elif missed_context:
        scene.scene_type = "missed_context_callout"
        scene.required_reply_move = "acknowledge_and_repair"
        scene.social_task = "acknowledge missed context"
        scene.user_called_out_bot = True
        scene.repair_required = True
        scene.hook_allowed = False
        scene.confidence = 0.92
    elif weird_or_confusion_callout:
        scene.scene_type = "dry_complaint" if dry else "missed_context_callout"
        scene.required_reply_move = "acknowledge_and_repair"
        scene.social_task = "recover from the user's confusion or dry/weird callout without topic shifting"
        scene.user_called_out_bot = True
        scene.repair_required = True
        scene.hook_allowed = False
        scene.confidence = 0.9
    elif story_hypothetical:
        scene.scene_type = "story_hypothetical_question"
        scene.required_reply_move = "answer_story_hypothetical"
        scene.social_task = "answer what the bot would do in the user's story situation"
        scene.direct_answer_required = True
        scene.hook_allowed = False
        scene.confidence = 0.92
    elif vivid_story or story_confirmation or (story_setup and len(latest_norm.split()) >= 8):
        scene.scene_type = "weird_story"
        scene.required_reply_move = "ask_one_relevant_question"
        scene.social_task = "stay on the user's story with surprise and one relevant question"
        scene.emotional_response_required = True
        scene.hook_allowed = False
        scene.confidence = 0.92
    elif training_question:
        scene.scene_type = "gym_training_question"
        scene.required_reply_move = "answer_training_question"
        scene.social_task = "answer the gym/training question directly"
        scene.direct_answer_required = True
        scene.hook_allowed = False
        scene.confidence = 0.9
    elif boxing_question:
        scene.scene_type = "owner_boxing_question"
        scene.required_reply_move = "answer_boxing_question"
        scene.social_task = "answer whether the bot/owner boxes"
        scene.direct_answer_required = True
        scene.hook_allowed = False
        scene.confidence = 0.9
    elif playful_fight_challenge:
        scene.scene_type = "playful_fight_challenge"
        scene.required_reply_move = "answer_playful_fight_challenge"
        scene.social_task = "respond playfully to the user's fight challenge without escalating seriously"
        scene.hook_allowed = False
        scene.confidence = 0.88
    elif tech_interest_question:
        scene.scene_type = "owner_tech_interest_question"
        scene.required_reply_move = "answer_tech_interest_question"
        scene.social_task = "answer what part of tech the bot/owner is into"
        scene.direct_answer_required = True
        scene.hook_allowed = False
        scene.confidence = 0.9
    elif owner_car_question:
        scene.scene_type = "owner_car_preference_question"
        scene.required_reply_move = "answer_owner_car_preference"
        scene.social_task = "answer the car preference question with a specific car taste"
        scene.direct_answer_required = True
        scene.hook_allowed = False
        scene.active_topic = "cars"
        scene.confidence = 0.9
    elif owner_dream_question:
        scene.scene_type = "owner_dream_question"
        scene.required_reply_move = "answer_owner_dream"
        scene.social_task = "answer the owner's dream/ambition question with a safe grounded aspiration"
        scene.direct_answer_required = True
        scene.hook_allowed = False
        scene.confidence = 0.88
    elif owner_prayer_question:
        scene.scene_type = "owner_prayer_question"
        scene.required_reply_move = "answer_owner_prayer"
        scene.social_task = "answer the prayer/religious habit question directly without overdisclosing"
        scene.direct_answer_required = True
        scene.hook_allowed = False
        scene.confidence = 0.88
    elif owner_tired_reason_question:
        scene.scene_type = "owner_tired_reason_question"
        scene.required_reply_move = "answer_owner_tired_reason"
        scene.social_task = "answer why the owner is tired with a safe grounded reason"
        scene.direct_answer_required = True
        scene.hook_allowed = False
        scene.confidence = 0.88
    elif car_answer:
        scene.scene_type = "car_preference_answer"
        scene.required_reply_move = "acknowledge_car_preference"
        scene.social_task = "react to the user's car choice and keep the car topic alive"
        scene.active_topic = "cars"
        scene.hook_allowed = False
        scene.confidence = 0.88
    elif topic_choice_offered:
        scene.scene_type = "topic_choice_offered"
        scene.required_reply_move = "choose_from_topic_options"
        scene.social_task = "choose one of the user's offered topics and move the conversation forward"
        scene.topic_engagement_required = True
        scene.hook_allowed = False
        scene.topic_value = latest_simple
        scene.confidence = 0.9
    elif topic_opinion:
        scene.scene_type = "topic_opinion"
        scene.required_reply_move = "respond_to_topic_opinion"
        scene.social_task = "respond to the user's opinion on the active topic"
        scene.hook_allowed = False
        scene.confidence = 0.84
    elif reciprocal_topic_question:
        scene.scene_type = "reciprocal_topic_question"
        scene.required_reply_move = "answer_reciprocal_topic_question"
        scene.social_task = "answer the user's reciprocal topic question instead of treating wby as wellbeing"
        scene.direct_answer_required = True
        scene.hook_allowed = False
        scene.active_topic = active_topic or ("cars" if "car" in latest_norm else "")
        scene.confidence = 0.9
    elif reciprocal_activity:
        scene.scene_type = "reciprocal_current_activity_question"
        scene.required_reply_move = "answer_reciprocal_activity"
        scene.social_task = "answer the user's wby/hbu after they gave their activity"
        scene.direct_answer_required = True
        scene.reciprocal_activity_required = True
        scene.hook_allowed = False
        scene.known_recent_facts.append(f"recent_user_activity={latest_activity}")
        scene.confidence = 0.94
    elif passive_user_activity_update:
        scene.scene_type = "user_activity_status"
        scene.required_reply_move = "acknowledge_user_activity"
        scene.social_task = "acknowledge the user's current activity without asking what they are doing again"
        scene.hook_allowed = False
        scene.known_recent_facts.append(f"recent_user_activity={latest_activity}")
        scene.confidence = 0.86
    elif day_activity_question:
        scene.scene_type = "owner_day_activity_question"
        scene.required_reply_move = "answer_owner_day_activity"
        scene.social_task = "answer what the bot did today with a specific safe detail and carry the chat"
        scene.direct_answer_required = True
        scene.hook_allowed = False
        scene.confidence = 0.94
    elif day_check:
        scene.scene_type = "day_check_question"
        scene.required_reply_move = "answer_day_check"
        scene.social_task = "answer how the day was"
        scene.direct_answer_required = True
        scene.day_check_response_required = True
        scene.hook_allowed = False
        scene.confidence = 0.94
    elif wellbeing_checkin:
        scene.scene_type = "reciprocal_wellbeing_question"
        scene.required_reply_move = "answer_wellbeing_checkin"
        scene.social_task = "answer how the bot is doing"
        scene.direct_answer_required = True
        scene.hook_allowed = False
        scene.confidence = 0.9
    elif positive_mood_update:
        scene.scene_type = "positive_mood_update"
        scene.required_reply_move = "acknowledge_positive_mood"
        scene.social_task = "respond warmly to the user's good mood and carry it with a natural probe"
        scene.latest_user_emotion = "happy"
        scene.emotional_response_required = True
        scene.hook_allowed = False
        scene.confidence = 0.88
    elif care:
        scene.scene_type = "care_checkin"
        scene.required_reply_move = "respond_to_care"
        scene.social_task = "answer whether the bot is okay"
        scene.emotional_response_required = True
        scene.care_response_required = True
        scene.direct_answer_required = True
        scene.hook_allowed = False
        scene.confidence = 0.92
    elif affection and not identity_kind:
        scene.scene_type = "emotional_affection"
        scene.required_reply_move = "conservative_deflect" if conservative_affection else "reciprocate_affection"
        scene.social_task = "acknowledge affection before anything else"
        scene.emotional_response_required = True
        scene.affection_reciprocity_required = not conservative_affection
        scene.hook_allowed = False
        scene.confidence = 0.86
        if life_update or current_activity:
            scene.scene_type = "emotional_reciprocity"
            scene.required_reply_move = "conservative_deflect" if conservative_affection else "reciprocate_affection"
            scene.direct_answer_required = True
            scene.affection_reciprocity_required = not conservative_affection
            scene.social_task = "answer the life/activity question and acknowledge affection"
    elif identity_kind:
        scene.scene_type = {
            "age": "owner_age_question",
            "location": "owner_location_question",
            "study": "owner_study_question",
            "work": "owner_work_question",
            "project": "owner_project_question",
            "where_been": "owner_recent_activity_question",
            "recent_activity": "owner_recent_activity_question",
            "doing_anything_nice": "owner_status_question",
        }.get(identity_kind, "identity_question")
        scene.required_reply_move = {
            "age": "answer_age_question",
            "location": "answer_location_question",
            "study": "answer_study_question",
            "work": "answer_owner_work",
            "project": "answer_owner_projects",
            "where_been": "answer_where_been",
            "recent_activity": "answer_where_been",
            "doing_anything_nice": "answer_owner_status",
        }.get(identity_kind, "answer_identity_question")
        scene.social_task = "answer the personal/identity question directly using safe local owner facts"
        scene.identity_answer_required = True
        scene.identity_question_kind = identity_kind
        scene.identity_disclosure_allowed = relationship_type in {"unknown", "close_friend", "romantic_interest", "trusted_contact", "professional", "university"}
        if identity_kind == "study" and _contains_any(latest_norm, ("what uni", "where u at uni", "where are u at uni", "where you at uni")):
            scene.identity_disclosure_allowed = relationship_type in {"close_friend", "romantic_interest", "trusted_contact"} and len(labelled) >= 12
        scene.direct_answer_required = True
        scene.hook_allowed = False
        if affection:
            scene.emotional_response_required = True
            scene.affection_reciprocity_required = not conservative_affection
            scene.social_task = "acknowledge affection and answer the identity/status question"
        scene.confidence = 0.94
    elif topic_value:
        scene.scene_type = "topic_given"
        scene.required_reply_move = "continue_given_topic"
        scene.social_task = "engage the topic the user provided"
        scene.topic_engagement_required = True
        scene.hook_allowed = False
        scene.confidence = 0.95
    elif life_update:
        scene.scene_type = "recent_life_update"
        scene.required_reply_move = "give_short_life_update"
        scene.direct_answer_required = True
        scene.confidence = 0.82
    elif current_activity:
        scene.scene_type = "current_activity_exchange"
        scene.required_reply_move = "answer_directly"
        scene.direct_answer_required = True
        scene.confidence = 0.8
    elif direct_question:
        scene.scene_type = "direct_question"
        scene.required_reply_move = "answer_directly"
        scene.direct_answer_required = True
        scene.hook_allowed = False
        scene.confidence = 0.78
    elif _contains_any(latest_norm, ("hi", "hey", "yo", "hello")) and len(latest_norm.split()) <= 2:
        scene.scene_type = "opening"
        scene.required_reply_move = "ask_one_relevant_question"
        scene.hook_required = True
        scene.confidence = 0.75
    elif latest_simple in {"i", "u", "you", "me"} or _contains_any(latest_norm, ("bored", "nothing", "idk", "lol", "not much")):
        scene.scene_type = "dead_conversation"
        scene.required_reply_move = "ask_one_relevant_question"
        scene.hook_required = True
        scene.confidence = 0.72

    if not scene.social_task:
        scene.social_task = scene.required_reply_move
    scene.latest_user_message = latest
    scene.latest_user_ask = latest
    scene.scene_summary = f"{scene.scene_type}: {scene.social_task}"

    forbidden = {
        "repeat_previous_bot_reply",
        "mirror_user",
    }
    if not scene.hook_allowed:
        forbidden.add("generic_hook")
    if scene.repair_required:
        forbidden.update({"generic_repair", "ask_how", "low_info_fallback"})
    if scene.scene_type in {"reciprocal_current_activity_question", "reciprocal_topic_question", "reciprocal_wellbeing_question", "positive_mood_update", "topic_positive_acknowledgement", "day_check_question", "missed_user_fact_callout", "repair_clarification", "low_info_after_bad_reply", "explain_previous_bot_claim", "light_acknowledgement", "tired_mood", "user_activity_update", "user_activity_status", "story_hypothetical_question", "gym_training_question", "owner_boxing_question", "playful_fight_challenge", "owner_tech_interest_question", "owner_car_preference_question", "car_preference_answer", "owner_dream_question", "owner_prayer_question", "owner_tired_reason_question", "topic_choice_offered", "topic_opinion", "contradiction_callout", "owner_activity_detail_question", "owner_activity_clarification", "owner_claim_contradiction", "identity_question", "reciprocal_identity_question", "reciprocal_identity_answer", "owner_status_question", "owner_location_question", "owner_study_question", "owner_age_question", "owner_recent_activity_question", "owner_work_question", "owner_project_question"}:
        forbidden.update({"generic_hook", "activity_fallback", "stale_self_state"})
    if scene.explanation_required:
        forbidden.update({"ignore_direct_question", "generic_hook", "activity_fallback"})
    if scene.topic_engagement_required:
        forbidden.update({"ask_for_topic_when_topic_given", "activity_fallback", "stale_self_state"})
    if scene.emotional_response_required or scene.direct_answer_required:
        forbidden.update({"activity_fallback", "stale_self_state"})
    if scene.scene_type in {"emotional_affection", "emotional_reciprocity", "missed_affection_callout", "care_checkin", "sexual_flirty_energy", "serious_boundary"}:
        forbidden.update({"generic_hook", "ask_for_topic_when_topic_given"})
    if scene.affection_reciprocity_required:
        forbidden.add("ignore_affection")
    if scene.direct_answer_required:
        forbidden.add("ignore_direct_question")
    if scene.identity_answer_required:
        forbidden.update({"ignore_identity_question", "generic_hook", "activity_fallback", "stale_self_state"})
    if scene.user_called_out_bot:
        forbidden.add("ignore_user_callout")
    if not flirt_allowed:
        forbidden.add("flirty_reply_if_not_allowed")
    forbidden.add("explicit_sexual_content")
    scene.forbidden_reply_moves = sorted(forbidden)
    if scene.repair_required and active_topic:
        scene.unresolved_user_points.append(f"engage topic: {active_topic}")
    return scene


def score_reply_against_scene(reply: str, scene: ConversationScene, *, previous_bot_replies: list[str] | None = None) -> dict[str, object]:
    normalized = _norm(reply).strip(" .?!")
    previous = {_norm(item).strip(" .?!") for item in (previous_bot_replies or []) if item}
    fallback_scene_type, fallback_required_move = _fallback_scene_for_reply(normalized)
    canned = {
        "same icl",
        "same just chilling",
        "fair just chilling too",
        "fine then give me a topic",
        "im trying icl give me a topic",
        "u ain't giving me much to work with",
        "u aint giving me much to work with",
        "calm",
        "true",
        "fair",
        "lol",
        "what u saying",
        "yo what u saying",
        "hi what u saying then",
        "what u doing rn",
        "say less what u doing",
        "bro why is his dad involved",
        "nah ur dragging it",
        "what u saying then",
    }
    forbidden: list[str] = []
    fallback_scene_mismatch = bool(
        fallback_scene_type
        and fallback_scene_type not in {scene.scene_type, "universal"}
        and not (fallback_scene_type == "current_activity_exchange" and scene.scene_type == "reciprocal_current_activity_question")
    )
    if fallback_scene_mismatch:
        forbidden.append("fallback_scene_mismatch")
    if normalized in previous and scene.scene_type != "opening":
        forbidden.append("repeat_previous_bot_reply")
    opening_supported = scene.scene_type == "opening" and normalized in {"yo what u saying", "hi what u saying then"}
    if normalized in canned and not opening_supported:
        forbidden.append("stale_self_state" if "chilling" in normalized or normalized in {"same icl", "calm"} else "generic_repair")
    recent_stale = any(_contains_any(item, ("same just chilling", "fair just chilling", "same icl")) for item in previous)
    if recent_stale and _contains_any(normalized, ("same", "chilling", "fair just chilling")):
        forbidden.append("repeat_stale_self_state")
    if scene.scene_type == "reciprocal_current_activity_question" and _contains_any(normalized, ("yo what u saying", "hi what u saying", "what u doing rn", "give me a topic")):
        forbidden.append("generic_hook")
    if scene.scene_type == "reciprocal_wellbeing_question" and (
        normalized in {"ok", "okay", "calm", "fair", "true", "same icl", "same just chilling"}
        or _contains_any(normalized, ("what u saying", "give me a topic"))
    ):
        forbidden.append("ignore_direct_question")
    if scene.scene_type == "positive_mood_update" and (
        normalized in {"ok", "okay", "calm", "fair", "true", "thats good to hear", "that's good to hear"}
        or _contains_any(normalized, ("what u saying", "give me a topic", "same just chilling", "fair just chilling"))
    ):
        forbidden.append("dry_positive_mood_reply")
    if scene.scene_type == "reciprocal_identity_answer" and (
        normalized in {"ok", "okay", "what u saying", "what u saying then"}
        or _contains_any(normalized, ("is random", "what u into", "what you into", "what what u into", "what what you into"))
    ):
        forbidden.append("misread_reciprocal_identity_answer")
    if scene.scene_type == "day_check_question" and _contains_any(normalized, ("same just chilling", "fair just chilling", "same icl", "what u doing rn")):
        forbidden.append("activity_fallback")
    if scene.required_reply_move == "answer_owner_day_activity" and (
        _contains_any(normalized, ("same just chilling", "fair just chilling", "same icl", "what u saying", "give me a topic", "say less what u doing"))
        or normalized in {"ok", "okay", "calm", "fair", "true", "lol"}
    ):
        forbidden.append("ignore_day_activity_question")
    if scene.scene_type in {"user_activity_update", "contradiction_callout"} and _contains_any(normalized, ("same just chilling", "fair just chilling", "same icl", "what u saying", "give me a topic")):
        forbidden.append("activity_fallback")
    if scene.scene_type == "user_activity_status" and _contains_any(normalized, ("what u doing", "what you doing", "wyd", "say less what u doing", "what u saying", "give me a topic")):
        forbidden.append("asked_activity_already_answered")
    if scene.scene_type == "contradiction_callout" and normalized.startswith("yh yh but"):
        forbidden.append("ignore_user_callout")
    if scene.scene_type in {"owner_activity_detail_question", "owner_activity_clarification", "owner_claim_contradiction"} and _contains_any(normalized, ("same just chilling", "fair just chilling", "same icl", "what u saying", "give me a topic", "what should we talk about")):
        forbidden.append("ignore_previous_bot_claim")
    if scene.scene_type == "dead_conversation" and normalized in {"ok", "okay", "say less what u doing", "what u saying then"}:
        forbidden.append("low_info_fallback")
    if scene.scene_type == "missed_user_fact_callout" and normalized in {"yh fairs i did icl", "same icl", "calm", "fair"}:
        forbidden.append("generic_repair")
    if scene.scene_type == "repair_clarification" and (normalized in {"nah ur right i bugged", "same icl", "fair just chilling too"} or "same just chilling" in normalized):
        forbidden.append("generic_repair")
    if scene.scene_type == "low_info_after_bad_reply" and _contains_any(normalized, ("same just chilling", "fair just chilling", "same icl", "calm", "ok", "okay")):
        forbidden.append("activity_fallback")
    if scene.scene_type == "light_acknowledgement" and _contains_any(normalized, ("same just chilling", "fair just chilling", "same icl", "what u doing rn")):
        forbidden.append("activity_fallback")
    if scene.scene_type == "tired_mood" and normalized in {"same icl", "same", "fair just chilling too", "same just chilling", "give me a topic"}:
        forbidden.append("emotional_absence")
    if scene.scene_type == "explain_previous_bot_claim" and _contains_any(normalized, ("same just chilling", "fair just chilling", "same icl", "give me a topic", "what should we talk about")):
        forbidden.append("ignore_previous_bot_claim")
    if scene.scene_type == "explain_previous_bot_claim" and scene.previous_bot_claim_type in {"topic_statement", "random_topic_misread"} and _contains_any(normalized, ("that was dead", "dead reply", "reply was dead")):
        forbidden.append("generic_repair")
    if scene.identity_answer_required and (
        normalized in {"what u saying", "what u saying then", "same icl", "same just chilling", "fair just chilling too", "ok", "okay", "say less what u doing", "give me a topic", "calm"}
        or _contains_any(normalized, ("give me a topic", "what u doing rn"))
    ):
        forbidden.append("ignore_identity_question")
    if scene.topic_engagement_required and scene.scene_type != "topic_choice_offered" and scene.topic_value and scene.topic_value not in normalized and not (
        _is_tech_topic_text(scene.topic_value) and _is_tech_topic_text(normalized)
    ):
        forbidden.append("ignore_active_topic")
    if " is valid icl what " in normalized and " u into" in normalized:
        forbidden.append("parroted_user_as_topic")
    if scene.topic_engagement_required and ("give me a topic" in normalized or "what should we talk about" in normalized or "what u saying" in normalized):
        forbidden.append("ask_for_topic_when_topic_given")
    if scene.scene_type == "topic_choice_offered" and _contains_any(normalized, ("what one u picking", "what one you picking", "which one", "u pick", "you pick")):
        forbidden.append("refused_topic_choice")
    if scene.direct_answer_required and scene.scene_type == "direct_question" and ("what u doing" in normalized or "give me a topic" in normalized):
        forbidden.append("generic_hook")
    if scene.repair_required and (normalized in {"my bad", "yeah my bad", "nah ur right"} or normalized.startswith("how ")):
        forbidden.append("generic_repair")
    if scene.repair_required and _contains_any(normalized, ("u ain't giving me much", "u aint giving me much", "what should we talk about", "give me a topic")):
        forbidden.append("ignore_user_callout")
    if scene.emotional_response_required and any(term in normalized for term in ("same just chilling", "same icl", "fair just chilling", "give me a topic", "what u doing", "say less")):
        forbidden.append("activity_fallback")
    if any(term in normalized for term in ("nude", "send pics", "send nudes")):
        forbidden.append("explicit_sexual_content")
    scene_context = " ".join(
        [
            _norm(scene.latest_user_message),
            _norm(scene.previous_bot_claim),
            _norm(scene.active_topic),
            " ".join(_norm(item) for item in scene.known_recent_facts),
            " ".join(_norm(item) for item in scene.recent_bot_replies[-2:]),
        ]
    )
    has_family_context = _contains_any(scene_context, ("dad", "father", "parent", "mum", "mom", "family"))
    semantic_contamination_penalty = 0.0
    if _contains_any(normalized, ("dad", "his dad", "father", "involved")) and scene.scene_type not in {"weird_story", "story_prompt"} and not has_family_context:
        semantic_contamination_penalty = 1.0
        forbidden.append("semantic_contamination")

    required_satisfied = False
    if scene.required_reply_move == "continue_given_topic":
        if _is_tech_topic_text(scene.topic_value):
            required_satisfied = _is_tech_topic_text(normalized)
        else:
            required_satisfied = bool(scene.topic_value and scene.topic_value in normalized)
    elif scene.required_reply_move == "acknowledge_and_repair":
        required_satisfied = _contains_any(normalized, ("my bad", "fairs", "fair", "ur right", "repeated", "missed", "bugged", "waffling", "caught me", "npc behaviour"))
        if scene.topic_engagement_required and scene.topic_value:
            required_satisfied = required_satisfied and scene.topic_value in normalized
    elif scene.required_reply_move == "reciprocate_affection":
        if scene.affection_reciprocity_required:
            required_satisfied = _contains_any(normalized, ("missed u too", "miss u too", "miss u more", "miss you too", "course i missed", "love u too", "love you too", "love u more", "goodnight", "good night", "sleep well", "sleep tight", "mwah", "was getting there"))
        else:
            required_satisfied = _contains_any(normalized, ("sweet", "bless", "appreciate", "replied badly", "answered that weird", "didn't mean it like that", "didnt mean it like that"))
    elif scene.required_reply_move == "respond_to_care":
        required_satisfied = _contains_any(normalized, ("im good", "i'm good", "im okay", "i'm okay", "im alright", "i'm alright", "yeah im"))
    elif scene.required_reply_move == "answer_reciprocal_activity":
        required_satisfied = _contains_any(normalized, ("same", "nothing much", "not much", "just chilling", "been chilling", "working", "in bed", "sleep", "sorting stuff", "gym", "coding", "uni", "projects")) and not _contains_any(normalized, ("yo what u saying", "what u doing rn", "what u been on today"))
        if scene.repair_required:
            required_satisfied = required_satisfied and _contains_any(normalized, ("my bad", "missed", "ignored", "didn't answer", "didnt answer", "forgot"))
    elif scene.required_reply_move == "answer_reciprocal_topic_question":
        if _contains_any(_norm(scene.latest_user_message), ("dream car", " car", " cars")):
            required_satisfied = _contains_any(normalized, ("urus", "r8", "m4", "rs6", "porsche", "lambo", "audi", "bmw", "merc", "mine", "i'd", "id "))
        elif _contains_any(_norm(scene.latest_user_message), ("gym", "boxing")):
            required_satisfied = _contains_any(normalized, ("gym", "boxing", "training", "push", "pull", "legs", "fight"))
        else:
            required_satisfied = _contains_any(normalized, ("mine", "me", "i'd", "id ", "same", "probably")) and not _contains_any(normalized, ("im good", "i'm good", "good wbu"))
    elif scene.required_reply_move == "answer_story_hypothetical":
        required_satisfied = _contains_any(normalized, ("id ", "i'd", "shout", "run", "leave", "call", "ask", "swing", "panic", "icl"))
    elif scene.required_reply_move == "answer_training_question":
        required_satisfied = _contains_any(normalized, ("push", "pull", "legs", "back", "chest", "boxing", "gym", "train", "weights"))
    elif scene.required_reply_move == "answer_boxing_question":
        required_satisfied = _contains_any(normalized, ("yeah", "yh", "box", "boxing", "train", "spar", "hard", "tiring", "draining"))
    elif scene.required_reply_move == "answer_playful_fight_challenge":
        required_satisfied = _contains_any(normalized, ("fold", "win", "try", "behave", "easy", "fight", "boxing", "wouldn't", "wouldnt", "nah"))
    elif scene.required_reply_move == "answer_tech_interest_question":
        required_satisfied = _is_tech_topic_text(normalized)
    elif scene.required_reply_move == "answer_owner_car_preference":
        required_satisfied = _contains_any(normalized, ("urus", "r8", "rs6", "m4", "porsche", "lambo", "audi", "bmw", "merc", "car"))
    elif scene.required_reply_move == "acknowledge_car_preference":
        required_satisfied = _contains_any(normalized, ("urus", "r8", "rs6", "m4", "porsche", "lambo", "audi", "bmw", "merc", "cold", "hard", "valid"))
    elif scene.required_reply_move == "answer_owner_dream":
        required_satisfied = _contains_any(normalized, ("project", "business", "build", "building", "software", "family", "people", "comfortable", "serious", "make it"))
    elif scene.required_reply_move == "answer_owner_prayer":
        required_satisfied = _contains_any(normalized, ("yeah", "yh", "pray", "praying", "try", "alhamdulillah"))
    elif scene.required_reply_move == "answer_owner_tired_reason":
        required_satisfied = _contains_any(normalized, ("boxing", "gym", "work", "coding", "training", "clients", "draining", "tired", "finished"))
    elif scene.required_reply_move == "choose_from_topic_options":
        options = [part.strip() for part in re.split(r"\bor\b", _norm(scene.latest_user_message).strip(" .?!")) if part.strip()]
        required_satisfied = any(option in normalized for option in options) and not _contains_any(normalized, ("what one u picking", "which one", "u pick"))
    elif scene.required_reply_move == "respond_to_topic_opinion":
        required_satisfied = _contains_any(normalized, ("same", "valid", "legs", "hate", "love", "fair", "pain", "evil", "icl"))
    elif scene.required_reply_move == "answer_day_check":
        required_satisfied = _contains_any(normalized, ("day", "calm", "dead", "not bad", "decent", "nothing crazy", "long icl", "chilled"))
    elif scene.required_reply_move == "answer_owner_day_activity":
        required_satisfied = _contains_any(normalized, ("gym", "coding", "uni", "work", "boxing", "project", "chilled", "didn't do much", "didnt do much", "nothing much", "nothing crazy"))
        if scene.repair_required:
            required_satisfied = required_satisfied and _contains_any(normalized, ("my bad", "didn't answer", "didnt answer", "answered wrong", "i meant", "worded", "made no sense"))
    elif scene.required_reply_move == "answer_wellbeing_checkin":
        required_satisfied = _contains_any(normalized, ("im good", "i'm good", "good u", "good wbu", "im calm", "i'm calm", "calm u", "calm wbu", "not bad", "im bless", "i'm bless")) and normalized not in {"ok", "okay"}
        if scene.repair_required:
            required_satisfied = required_satisfied and _contains_any(normalized, ("my bad", "missed", "icl", "yh", "yeah"))
    elif scene.required_reply_move == "acknowledge_positive_mood":
        required_satisfied = _contains_any(normalized, ("as u should", "happy", "good", "cute", "love that", "deserve")) and normalized not in {"thats good to hear", "that's good to hear"}
    elif scene.required_reply_move == "acknowledge_topic_positive":
        required_satisfied = _contains_any(normalized, ("yh", "yeah", "cold", "sick", "hard", "icl", "proper", "serious", "valid"))
    elif scene.required_reply_move == "acknowledge_missed_fact":
        required_satisfied = _contains_any(normalized, ("missed that", "ignored what u said", "didn't clock", "didnt clock", "u did say", "you did say", "my bad"))
    elif scene.required_reply_move == "clarify_previous_repair":
        required_satisfied = _contains_any(normalized, ("i mean", "i meant", "missed what u said", "ignored ur message", "answered the wrong thing", "bugged and ignored"))
    elif scene.required_reply_move == "repair_or_prompt_lightly":
        required_satisfied = _contains_any(normalized, ("reply was dead", "answered that badly", "ignore me", "waffling", "my bad"))
    elif scene.required_reply_move == "acknowledge_reciprocal_identity_answer":
        if _norm(scene.latest_user_message).strip(" .?!") == "same":
            required_satisfied = _contains_any(normalized, ("twins", "same age", "valid")) and not _contains_any(normalized, ("random", "what u into", "what you into"))
        else:
            required_satisfied = _contains_any(normalized, ("fair", "fairs", "older", "21", "ohh", "oh ")) and not _contains_any(normalized, ("random", "what u into", "what you into"))
    elif scene.required_reply_move == "explain_previous_bot_claim":
        if scene.previous_bot_claim_type == "day_summary":
            required_satisfied = _contains_any(normalized, ("didn't do much", "didnt do much", "nothing really", "dead day", "boring", "not much happened", "one of them", "not much", "chilled", "uni", "gym", "coding", "work"))
        elif scene.previous_bot_claim_type == "weird_story_reaction":
            required_satisfied = _contains_any(normalized, ("random", "wrong context", "made no sense", "ignore that", "answered the wrong thing"))
        elif scene.previous_bot_claim_type == "current_activity":
            required_satisfied = _contains_any(normalized, ("i meant", "just chilling", "not doing much", "nothing much"))
        elif scene.previous_bot_claim_type == "repair":
            required_satisfied = _contains_any(normalized, ("i mean", "i meant", "missed", "answered the wrong thing", "bugged"))
        elif scene.previous_bot_claim_type == "random_topic_misread":
            required_satisfied = _contains_any(normalized, ("random", "misread", "worded that", "called it random", "my bad"))
        elif scene.previous_bot_claim_type == "owner_activity_summary":
            required_satisfied = _contains_any(normalized, ("busy", "uni", "gym", "coding", "clients", "projects", "all over", "loads"))
        else:
            if "sounds dead" in _norm(scene.previous_bot_claim):
                required_satisfied = _contains_any(normalized, ("sounds dead", "dead day", "boring", "doing nothing", "no plan", "gets boring"))
            else:
                required_satisfied = _contains_any(normalized, ("i mean", "i meant", "meant", "worded", "my bad", "didn't do much", "didnt do much")) and "give me a topic" not in normalized
    elif scene.required_reply_move == "answer_owner_activity_detail":
        required_satisfied = _contains_any(normalized, ("gym", "weights", "trained", "workout", "push", "pull", "legs", "boxing"))
    elif scene.required_reply_move == "clarify_owner_activity_claim":
        required_satisfied = _contains_any(normalized, ("i meant me", "me coding", "i do code", "i was coding", "coding is me", "not u", "not you"))
    elif scene.required_reply_move == "acknowledge_and_correct_claim":
        required_satisfied = _contains_any(normalized, ("my bad", "i do code", "i meant", "worded", "wasn't a lie", "wasnt a lie", "coding"))
    elif scene.required_reply_move == "respond_to_user_activity_update":
        required_satisfied = _contains_any(normalized, ("busy with what", "what u been busy", "what you been busy", "doing what", "how come", "what with"))
    elif scene.required_reply_move == "acknowledge_user_activity":
        required_satisfied = _contains_any(normalized, ("valid", "fair", "bed", "stay", "sleep", "chill", "comfy", "watching", "tired")) and not _contains_any(normalized, ("what u doing", "what you doing", "wyd"))
    elif scene.required_reply_move == "acknowledge_contradiction":
        required_satisfied = _contains_any(normalized, ("made no sense", "makes no sense", "contradicted", "contradiction", "paradox", "my bad", "that was dumb"))
    elif scene.required_reply_move == "playful_acknowledge_or_move_on":
        required_satisfied = _contains_any(normalized, ("ur good", "you're good", "lool", "lol", "fairs", "allow it", "all good", "behave", "make me"))
    elif scene.required_reply_move == "empathetic_casual_response":
        required_satisfied = _contains_any(normalized, ("tired", "sleep", "nap", "long day", "finished", "drained")) and normalized != "same icl"
    elif scene.required_reply_move == "answer_age_question":
        required_satisfied = _contains_any(normalized, ("19", "same im 19", "same i'm 19", "im 19", "i'm 19"))
    elif scene.required_reply_move == "answer_location_question":
        required_satisfied = _contains_any(normalized, ("northbridge", "northbridge", "sampleford"))
    elif scene.required_reply_move == "answer_study_question":
        required_satisfied = _contains_any(normalized, ("comp sci", "computer science", "study tech"))
    elif scene.required_reply_move == "answer_owner_work":
        required_satisfied = _contains_any(normalized, ("comp sci", "study", "uni", "projects", "building", "stuff on the side"))
    elif scene.required_reply_move == "answer_owner_projects":
        required_satisfied = _contains_any(normalized, ("project", "building", "dev thing", "side", "software", "stuff"))
    elif scene.required_reply_move == "answer_where_been":
        required_satisfied = _contains_any(normalized, ("busy", "uni", "projects", "gym", "coding", "work", "loads going on", "been around", "just been chilling", "not much", "nothing crazy", "doing the usual"))
    elif scene.required_reply_move == "answer_owner_status":
        required_satisfied = _contains_any(normalized, ("nothing crazy", "not really", "gym", "coding", "projects", "working", "uni"))
    elif scene.required_reply_move in {"answer_identity_question", "answer_reciprocal_identity_question"}:
        required_satisfied = bool(normalized) and not _contains_any(normalized, ("what u saying", "give me a topic", "same just chilling"))
    elif scene.required_reply_move in {"answer_directly", "give_short_life_update"}:
        latest = _norm(scene.latest_user_ask)
        if "r u mad" in latest or "are u mad" in latest:
            required_satisfied = _contains_any(normalized, ("im calm", "i'm calm", "not mad", "nah why"))
        else:
            required_satisfied = not _contains_any(normalized, ("give me a topic", "what u doing rn"))
    elif scene.required_reply_move in {"safe_playful_flirty_response", "conservative_deflect"}:
        required_satisfied = bool(normalized) and "give me a topic" not in normalized
    elif scene.required_reply_move == "deescalate":
        required_satisfied = _contains_any(normalized, ("my bad", "didn't mean", "didnt mean", "i get why", "not taking the piss", "on me"))
    else:
        required_satisfied = bool(normalized)

    contextual_specificity = 0.35
    if scene.topic_value and (
        scene.topic_value in normalized or (_is_tech_topic_text(scene.topic_value) and _is_tech_topic_text(normalized))
    ):
        contextual_specificity = 1.0
    elif "missed affection reciprocity" in scene.unresolved_user_points and _contains_any(normalized, ("missed u too", "miss u too", "replied badly", "answered that weird", "dodged that")):
        contextual_specificity = 1.0
    elif scene.scene_type in {"repeated_reply_callout", "missed_context_callout", "topic_ignored_callout"} and _contains_any(normalized, ("repeated", "missed", "bugged", "waffling", "caught me", "npc behaviour")):
        contextual_specificity = 0.9
    elif scene.scene_type in {"missed_user_fact_callout", "repair_clarification", "low_info_after_bad_reply"} and required_satisfied:
        contextual_specificity = 0.95
    elif scene.scene_type == "day_check_question" and required_satisfied:
        contextual_specificity = 0.9
    elif scene.required_reply_move == "answer_owner_day_activity" and required_satisfied:
        contextual_specificity = 0.95
    elif scene.scene_type == "reciprocal_wellbeing_question" and required_satisfied:
        contextual_specificity = 0.9
    elif scene.scene_type == "positive_mood_update" and required_satisfied:
        contextual_specificity = 0.9
    elif scene.scene_type == "topic_positive_acknowledgement" and required_satisfied:
        contextual_specificity = 0.9
    elif scene.scene_type == "reciprocal_current_activity_question" and required_satisfied:
        contextual_specificity = 0.85
    elif scene.scene_type == "explain_previous_bot_claim" and required_satisfied:
        contextual_specificity = 0.95
    elif scene.scene_type in {"owner_activity_detail_question", "owner_activity_clarification", "owner_claim_contradiction"} and required_satisfied:
        contextual_specificity = 0.95
    elif scene.scene_type in {"owner_car_preference_question", "car_preference_answer", "owner_dream_question", "owner_prayer_question", "owner_tired_reason_question", "topic_choice_offered"} and required_satisfied:
        contextual_specificity = 0.95
    elif scene.scene_type in {"user_activity_update", "contradiction_callout"} and required_satisfied:
        contextual_specificity = 0.9
    elif scene.scene_type == "reciprocal_identity_answer" and required_satisfied:
        contextual_specificity = 0.9
    elif scene.scene_type in {"light_acknowledgement", "tired_mood"} and required_satisfied:
        contextual_specificity = 0.9
    elif scene.identity_answer_required and required_satisfied:
        contextual_specificity = 0.95
    elif scene.emotional_response_required and required_satisfied:
        contextual_specificity = 0.85
    elif required_satisfied:
        contextual_specificity = 0.65

    canned_penalty = 1.0 if normalized in canned and not opening_supported else 0.0
    stale_template_penalty = 1.0 if any(term in normalized for term in ("same just chilling", "fair just chilling", "give me a topic")) else canned_penalty
    if recent_stale and _contains_any(normalized, ("same", "chilling", "fair just chilling")):
        stale_template_penalty = 1.0
    unresolved_point_addressed = True
    if "missed affection reciprocity" in scene.unresolved_user_points:
        unresolved_point_addressed = _contains_any(normalized, ("missed u too", "miss u too", "replied badly", "answered that weird", "dodged that", "didn't mean it like that", "didnt mean it like that"))
    elif "missed care check-in" in scene.unresolved_user_points:
        unresolved_point_addressed = _contains_any(normalized, ("im good", "i'm good", "im okay", "i'm okay", "im alright", "i'm alright"))
    elif "answer previous wellbeing question" in scene.unresolved_user_points:
        unresolved_point_addressed = _contains_any(normalized, ("im good", "i'm good", "im calm", "i'm calm", "not bad", "im bless", "i'm bless"))
    elif "answer previous reciprocal activity question" in scene.unresolved_user_points:
        unresolved_point_addressed = _contains_any(normalized, ("my bad", "missed", "ignored", "didn't answer", "didnt answer")) and _contains_any(normalized, ("nothing much", "not much", "just chilling", "working", "sorting stuff", "gym", "coding", "uni", "projects", "in bed", "sleep"))
    if not unresolved_point_addressed:
        forbidden.append("ignore_unresolved_user_point")
    identity_answer_score = 1.0 if scene.identity_answer_required and required_satisfied else 0.0
    ignored_identity_penalty = 1.0 if scene.identity_answer_required and not required_satisfied else 0.0
    stale_identity_penalty = 1.0 if scene.identity_answer_required and any(term in normalized for term in ("same icl", "same just chilling", "fair just chilling", "what u saying", "say less what u doing")) else 0.0
    scene_fit = 0.75 if required_satisfied else 0.25
    if forbidden:
        scene_fit = min(scene_fit, 0.05)
    else:
        scene_fit = min(1.0, scene_fit + contextual_specificity * 0.25)
    return {
        "scene_fit_score": round(scene_fit, 3),
        "required_move_satisfied": required_satisfied,
        "forbidden_move_violated": bool(forbidden),
        "scene_mismatch_reason": ", ".join(sorted(set(forbidden))),
        "contextual_specificity_score": round(contextual_specificity, 3),
        "human_likeness_score": round(max(0.0, min(1.0, 0.82 - 0.45 * canned_penalty - 0.25 * stale_template_penalty)), 3),
        "canned_reply_penalty": round(canned_penalty, 3),
        "stale_template_penalty": round(stale_template_penalty, 3),
        "fallback_scene_type": fallback_scene_type,
        "fallback_required_move": fallback_required_move,
        "fallback_scene_mismatch": fallback_scene_mismatch,
        "semantic_contamination_penalty": round(semantic_contamination_penalty, 3),
        "previous_claim_explanation_score": 1.0 if scene.explanation_required and required_satisfied else 0.0,
        "explanation_required": scene.explanation_required,
        "explanation_target": scene.explanation_target,
        "active_topic_engagement_score": 1.0 if scene.topic_value and (
            scene.topic_value in normalized or (_is_tech_topic_text(scene.topic_value) and _is_tech_topic_text(normalized))
        ) else 0.0,
        "direct_answer_score": 1.0 if scene.direct_answer_required and required_satisfied else 0.0,
        "repair_specificity_score": 1.0 if scene.repair_required and required_satisfied else 0.0,
        "emotional_presence_score": 1.0 if scene.emotional_response_required and required_satisfied else 0.0,
        "unresolved_point_addressed": unresolved_point_addressed,
        "identity_question_detected": scene.identity_answer_required,
        "identity_disclosure_allowed": scene.identity_disclosure_allowed,
        "identity_answer_score": round(identity_answer_score, 3),
        "ignored_identity_question_penalty": round(ignored_identity_penalty, 3),
        "stale_identity_fallback_penalty": round(stale_identity_penalty, 3),
        "direct_identity_answer_required": scene.identity_answer_required,
        "identity_specificity_score": round(contextual_specificity if scene.identity_answer_required else 0.0, 3),
    }
