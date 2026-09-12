from __future__ import annotations

import re
from dataclasses import dataclass, field

from libs.drafting.intents import normalize_text


ADULT_DETAIL_FAMILIES = {"neck_lips", "hips_press", "hands_thighs", "body_skin"}


@dataclass(frozen=True)
class ReplyPlan:
    move: str
    tone: str
    shape: str
    must_do: list[str]
    must_not_do: list[str]
    required_slots: dict[str, str] = field(default_factory=dict)
    forbidden_patterns: list[str] = field(default_factory=list)
    allowed_shapes: list[str] = field(default_factory=list)
    forbidden_shapes: list[str] = field(default_factory=list)
    max_bubbles: int = 2
    max_chars: int = 120


def detect_specific_topic(reply: str) -> str:
    reply_norm = normalize_text(reply)
    if re.search(r"\bhow\s+(?:was|is)\s+(?:ur|your)\s+day\b", reply_norm):
        return "day_status_question"
    if re.search(r"\b(?:did\s+u\s+get\s+up\s+to\s+much|what\s+(?:did\s+u|dya|do\s+u)\s+get\s+up\s+to|what\s+u\s+been\s+(?:up\s+to|on)\s+today|what\s+have\s+u\s+been\s+(?:up\s+to|on)\s+today)\b", reply_norm):
        return "day_status_question"
    if re.search(r"\bhow\s+(?:are|r)\s+(?:u|you)\b", reply_norm):
        return "wellbeing_status_question"
    if any(term in reply_norm for term in ("weirdest", "funniest", "what happened")) and any(term in reply_norm for term in ("day", "today", "happened")):
        return "day_story_prompt"
    if any(term in reply_norm for term in ("film", "movie", "show", "watch")):
        return "film_prompt"
    if any(term in reply_norm for term in ("food", "eat", "eaten", "craving", "craved")):
        return "food_prompt"
    if "dream" in reply_norm:
        return "dream_prompt"
    if "gym" in reply_norm:
        return "gym_prompt"
    if any(term in reply_norm for term in ("sleep", "bed")):
        return "sleep_prompt"
    if "work" in reply_norm:
        return "work_prompt"
    return ""


def detect_loop_acknowledgement(reply: str) -> bool:
    reply_norm = normalize_text(reply)
    return any(
        term in reply_norm
        for term in (
            "my bad",
            "i know",
            "i realised",
            "i realized",
            "too much",
            "ur right",
            "you're right",
            "u right",
            "yh fair",
            "yeah fair",
            "fair enough",
            "loop",
            "same thing",
            "keep asking",
            "kept asking",
            "i'll stop",
            "ill stop",
        )
    )


def detect_reason_answer(reply: str) -> bool:
    reply_norm = normalize_text(reply)
    if any(term in reply_norm for term in ("cos", "coz", "cuz", "because")):
        return True
    return any(
        term in reply_norm
        for term in (
            "trying",
            "tryna",
            "wanted",
            "wanna",
            "want to",
            "keep asking",
            "kept asking",
            "ran out of ideas",
            "like hearing",
            "like to hear",
            "like knowing",
            "hear about",
            "fantasise",
            "fantasize",
            "came to mind",
            "dunno what else",
            "don't know what else",
            "dont know what else",
            "can't think",
            "cant think",
            "lazy",
            "brain",
            "blank",
            "fried",
            "stuck",
            "fumbled",
            "panicked",
            "keep the convo going",
            "keep it going",
            "kept throwing questions",
            "throwing questions",
            "kept picking questions",
            "picking questions",
            "kept picking random",
            "picking random",
            "kept picking topics",
            "picking topics",
            "interrogating",
            "actually saying something",
        )
    )


def detect_reason_signatures(reply: str) -> set[str]:
    reply_norm = normalize_text(reply)
    signatures: set[str] = set()
    if any(term in reply_norm for term in ("tired", "half asleep", "sleepy", "brain's fried", "brains fried", "brain fried", "head's fried", "heads fried", "head fried")):
        signatures.add("tired_or_fried")
    if any(term in reply_norm for term in ("lazy", "got lazy", "being lazy")):
        signatures.add("lazy")
    if any(term in reply_norm for term in ("dodging", "avoiding", "instead of actually saying", "instead of saying something", "not actually saying")):
        signatures.add("avoidance")
    if any(term in reply_norm for term in ("couldnt think", "couldn't think", "cant think", "can't think", "blank", "ran out of ideas", "no ideas", "stuck")):
        signatures.add("blank_or_no_ideas")
    if any(term in reply_norm for term in ("keep the convo going", "keep it going", "keep the conversation going", "carry the convo", "carry conversation")):
        signatures.add("keep_convo_going")
    if any(term in reply_norm for term in ("like hearing", "like to hear", "hear u talk", "hear you talk", "hearing u talk", "hearing you talk", "hearing ur voice", "hearing your voice")):
        signatures.add("likes_hearing_them")
    if any(term in reply_norm for term in ("throwing questions", "random questions", "picking questions", "picking random", "picking topics", "interrogating", "kept asking", "keep asking")):
        signatures.add("random_questions")
    if any(term in reply_norm for term in ("buggy", "phone", "glitch")):
        signatures.add("technical_excuse")
    return signatures


def detect_detail_language(reply: str) -> bool:
    reply_norm = normalize_text(reply)
    return any(
        term in reply_norm
        for term in (
            "cock",
            "dick",
            "pussy",
            "clit",
            "hole",
            "lips",
            "neck",
            "thigh",
            "hips",
            "hands",
            "finger",
            "fingers",
            "mouth",
            "tongue",
            "throat",
            "body",
            "chest",
            "kiss",
            "tease",
            "taste",
            "press",
            "slide",
            "inside",
            "deep",
            "inch",
            "pinned",
            "wall",
            "wet",
            "hard",
            "throb",
            "puls",
            "drip",
            "leak",
            "whimper",
            "moan",
            "shake",
            "weak",
        )
    )


def detect_adult_detail_families(reply: str) -> set[str]:
    reply_norm = normalize_text(reply)
    families: set[str] = set()
    if any(term in reply_norm for term in ("neck", "lips", "kiss", "mouth", "tongue")):
        families.add("neck_lips")
    if any(term in reply_norm for term in ("hips", "press", "against", "wall", "closer")):
        families.add("hips_press")
    if any(term in reply_norm for term in ("thigh", "hands", "finger", "fingers", "waist", "slide", "tease")):
        families.add("hands_thighs")
    if any(term in reply_norm for term in ("chest", "body", "skin")):
        families.add("body_skin")
    return families


def detect_adult_followup_progression(reply: str) -> bool:
    reply_norm = normalize_text(reply)
    tokens = reply_norm.strip(" .?!").split()
    if len(tokens) < 8:
        return False
    families = detect_adult_detail_families(reply_norm)
    if len(families) >= 2:
        return True
    if any(
        marker in reply_norm
        for marker in (
            " and ",
            " while ",
            " till ",
            " until ",
            " then ",
            " against ",
            " all over ",
            " down to ",
            " keep ",
            " pull",
            " press",
        )
    ):
        return True
    return False


def detect_adult_scene_progression(reply: str) -> bool:
    reply_norm = normalize_text(reply)
    return any(
        marker in reply_norm
        for marker in (
            "chest",
            "back",
            "spine",
            "shoulder",
            "ear",
            "breath",
            "breathing",
            "shiver",
            "warm",
            "skin against",
            "skin on",
            "body against",
            "body press",
            "trail down",
            "down ur chest",
            "down your chest",
        )
    )


def detect_availability_plan_detail(reply: str) -> str:
    reply_norm = normalize_text(reply)
    if any(term in reply_norm for term in ("work", "coding", "project", "stuff", "busy")):
        return "work_stuff"
    if any(term in reply_norm for term in ("gym", "training", "cardio", "legs", "push", "pull")):
        return "gym"
    if "food" in reply_norm or "eat" in reply_norm:
        return "food"
    if any(term in reply_norm for term in ("bed", "sleep", "nap", "rest")):
        return "rest"
    if any(term in reply_norm for term in ("out", "night", "tonight", "plans")):
        return "going_out"
    if any(term in reply_norm for term in ("chill", "chilling", "nothing much", "not really", "staying in", "here i think")):
        return "low_key_chill"
    return ""


def detect_sensory_texture(reply: str) -> bool:
    reply_norm = normalize_text(reply)
    generic_flat_patterns = (
        "kiss u deep",
        "kiss you deep",
        "hand on ur thigh",
        "hand on your thigh",
        "hands on ur thigh",
        "hands on your thigh",
        "making u wet",
        "making you wet",
    )
    if any(pattern in reply_norm for pattern in generic_flat_patterns):
        return False
    return any(
        term in reply_norm
        for term in (
            "slow",
            "soft",
            "warm",
            "close",
            "closer",
            "against",
            "skin",
            "breath",
            "breathing",
            "shiver",
            "spine",
            "waist",
            "neck",
            "chest",
            "body",
            "hold",
            "holding",
            "pull",
            "trace",
            "tracing",
            "press",
            "pressing",
            "tease",
            "teasing",
            "linger",
            "feel",
            "mouth",
            "lips",
        )
    )


def detect_broad_question(reply: str) -> bool:
    reply_norm = normalize_text(reply)
    return bool(
        re.search(
            r"\bwhat\s+(?:do\s+)?(?:u|you|we)(?:\s+\w+){0,3}\s+(?:(?:wanna|gonna|should)\s+(?:do|talk|chat)|want\s+(?:to\s+)?(?:do|talk|chat))",
            reply_norm,
        )
    )


def detect_implied_question(reply: str) -> bool:
    reply_norm = normalize_text(reply)
    return bool(
        re.search(r"\b(?:want me to|do u want me to|do you want me to|should i|shall i|can i)\b", reply_norm)
        or re.search(r"\b(?:wanna|want to)\s+(?:hear|know|see)\b", reply_norm)
        or re.search(r"\b(?:u|you)\s+(?:wanna|want)\s+(?:hear|know|see)\b", reply_norm)
    )


def detect_reset_question(reply: str) -> bool:
    reply_norm = normalize_text(reply)
    if detect_broad_question(reply_norm):
        return False
    has_question = "?" in reply_norm or detect_implied_question(reply_norm) or bool(re.search(r"(?:^|[/\n.!]\s*)\b(?:what|why|how|who|when|where)\b", reply_norm))
    if not has_question:
        return False
    return bool(detect_specific_topic(reply_norm)) or any(
        term in reply_norm
        for term in (
            "best thing",
            "last thing",
            "go-to",
            "go to",
            "favourite",
            "favorite",
            "random question",
            "quick question",
        )
    )


def detect_bare_status_mirror(reply: str) -> bool:
    reply_norm = normalize_text(reply)
    reply_clean = reply_norm.strip(" .?!")
    parts = [part.strip(" .?!") for part in re.split(r"\s*/\s*|\n+", reply_norm) if part.strip()]
    mirror = r"(?:wby|wbu|hru|hbu|u(?:\?|$)|you(?:\?|$)|what about u|what about you)"
    bare_status = (
        r"(?:im|i'm|i am)\s+(?:good|fine|alright|okay|ok|calm|chilling|tired)"
        r"|(?:just\s+)?chilling"
        r"|(?:not much|nothing much|same|same here)"
    )
    if parts and re.fullmatch(rf"(?:{bare_status})(?:\s*/\s*|\s+){mirror}", parts[0]):
        return True
    return bool(re.fullmatch(rf"(?:{bare_status})(?:\s*/\s*|\s+){mirror}", reply_clean))


def detect_bare_return_question(reply: str) -> bool:
    reply_norm = normalize_text(reply)
    reply_clean = reply_norm.strip(" .")
    parts = [part.strip(" .?!") for part in re.split(r"\s*/\s*|\n+", reply_norm) if part.strip()]
    if any(part in {"u", "you", "wby", "wbu", "hru", "hbu", "what about u", "what about you"} for part in parts):
        return True
    return bool(re.search(r"(?:^|\s)(?:wby|wbu|hru|hbu)\??$", reply_clean))


def detect_repeat_apology_template(reply: str) -> bool:
    reply_norm = normalize_text(reply)
    return bool(
        re.search(r"\b(?:lol|lool)\s+yeah\s+i\s+did\s+didn'?t\s+i\b", reply_norm)
        or (
            "my bad" in reply_norm
            and "tired today" in reply_norm
            and any(term in reply_norm for term in ("how was ur day", "did u get up to much", "how are u", "how r u"))
        )
    )


def classify_reply_shape(reply: str) -> set[str]:
    reply_norm = normalize_text(reply)
    shapes: set[str] = set()
    if detect_loop_acknowledgement(reply_norm):
        shapes.add("acknowledgement")
    if detect_reason_answer(reply_norm):
        shapes.add("reason_answer")
    if detect_detail_language(reply_norm):
        shapes.add("detail_language")
    if detect_broad_question(reply_norm):
        shapes.add("broad_question")
    elif "?" in reply_norm or detect_implied_question(reply_norm) or re.search(r"(?:^|[/\n.!]\s*)\b(?:what|why|how|who|when|where)\b", reply_norm):
        shapes.add("question")
    if detect_reset_question(reply_norm):
        shapes.add("reset_question")
    topic = detect_specific_topic(reply_norm)
    if topic:
        shapes.add("specific_topic")
        shapes.add(topic)
    if any(term in reply_norm for term in ("tell me", "go on", "what happened", "what is it", "what's", "whats", "spill")):
        shapes.add("invite_reveal")
    if detect_bare_status_mirror(reply_norm):
        shapes.add("bare_status_mirror")
    if detect_bare_return_question(reply_norm):
        shapes.add("bare_return_question")
    if detect_repeat_apology_template(reply_norm):
        shapes.add("repeat_apology_template")
    return shapes


def _contains_forbidden_pattern(reply_norm: str, pattern: str) -> bool:
    pattern_norm = normalize_text(pattern)
    if not pattern_norm:
        return False
    if re.fullmatch(r"[a-z0-9']+", pattern_norm):
        return bool(re.search(rf"\b{re.escape(pattern_norm)}\b", reply_norm))
    return pattern_norm in reply_norm


def validate_reply_against_plan(reply: str, plan: ReplyPlan) -> str:
    reply_norm = normalize_text(reply)
    shapes = classify_reply_shape(reply_norm)
    forbidden = set(plan.forbidden_shapes)
    reply_clean = reply_norm.strip(" .?!")
    if len(reply.strip()) > plan.max_chars:
        return "violates_plan_too_long"
    bubbles = [part for part in re.split(r"\s*/\s*|\n+", reply) if part.strip()]
    if len(bubbles) > plan.max_bubbles:
        return "violates_plan_too_many_bubbles"
    for pattern in plan.forbidden_patterns:
        if _contains_forbidden_pattern(reply_norm, pattern):
            return f"violates_plan_forbidden_pattern:{pattern}"
    if "generic_ack" in forbidden and reply_clean in {"ok", "okay", "yh", "yeah", "fair", "calm", "nice"}:
        return "generic_ack"
    if "repeated_reset_topic" in forbidden and plan.shape != "answer_why_then_stop_loop":
        repeated_topic = plan.required_slots.get("repeated_reset_topic", "")
        if repeated_topic and detect_specific_topic(reply_norm) == repeated_topic:
            return "repeated_reset_topic"
    if plan.shape == "answer_status_then_continue":
        if detect_bare_status_mirror(reply_norm):
            return "bare_status_mirror"
        if detect_bare_return_question(reply_norm):
            return "bare_return_question"
    for shape in sorted(shapes):
        if shape in {"broad_question", "question", "reset_question"}:
            continue
        if shape in forbidden:
            return shape
    if "generic_tell_me_more" in forbidden and any(term in reply_norm for term in ("tell me more", "tell me everything")):
        return "generic_tell_me_more"
    if "what_kind_of_bored" in forbidden and any(term in reply_norm for term in ("what kind of bored", "what kinda bored", "kind of bored", "kinda bored")):
        return "what_kind_of_bored"
    if "what_kind_of_entertainment" in forbidden and any(term in reply_norm for term in ("what kind of entertainment", "what kinda entertainment", "kind of entertainment", "kinda entertainment")):
        return "what_kind_of_entertainment"
    if "ask_user_to_choose_topic" in forbidden and any(term in reply_norm for term in ("what kind", "what kinda", "what sort", "u craving", "you craving", "u craved", "you craved")):
        return "ask_user_to_choose_topic"
    if "what_do_you_want_to_talk_about" in forbidden and re.search(r"\bwhat\s+(?:do\s+)?(?:u|you)(?:\s+\w+){0,3}\s+wanna\s+(?:talk|chat)", reply_norm):
        return "what_do_you_want_to_talk_about"
    if "trailing_reset" in forbidden and any(reply_clean.endswith(term) for term in ("so instead of that", "instead of that", "so yeah")):
        return "trailing_reset"
    if "reset_question" in forbidden and "reset_question" in shapes:
        return "reset_question"
    if plan.move == "affectionate_greeting" and plan.shape == "short_reaction_plus_specific_continuation":
        if reply_clean in {"hey you", "hey u", "hi you", "hi u", "hey", "hi"}:
            return "bare_greeting"
        if len(reply_clean.split()) < 3:
            return "bare_greeting"
    if plan.shape == "multi_bubble_adult_escalation":
        if re.search(r"\bagain\b", reply_norm):
            return "repeated_adult_detail_cue"
        if len(bubbles) < 4:
            return "too_few_adult_bubbles"
        if "detail_language" not in shapes:
            return "missing_adult_detail"
        if not detect_sensory_texture(reply_norm):
            return "missing_sensory_texture"
        previous_families = {
            item
            for item in plan.required_slots.get("recent_adult_detail_families", "").split("|")
            if item
        }
        current_families = detect_adult_detail_families(reply_norm)
        if (
            plan.required_slots.get("escalation_source") == "adult_continuation_burst"
            and previous_families >= ADULT_DETAIL_FAMILIES
            and len(current_families & {"neck_lips", "hips_press", "hands_thighs"}) >= 2
            and not detect_adult_scene_progression(reply_norm)
        ):
            return "repeated_adult_detail_skeleton"
        if previous_families and current_families and current_families.issubset(previous_families):
            if not (previous_families >= ADULT_DETAIL_FAMILIES and len(current_families) >= 2):
                return "repeated_adult_detail_family"
        if reply_clean in {"i want u", "i need u", "i want you", "i need you", "same", "me too"}:
            return "vague_desire"
    if plan.shape == "specific_escalation_detail":
        if re.search(r"\bagain\b", reply_norm):
            return "repeated_adult_detail_cue"
        if not detect_sensory_texture(reply_norm):
            return "missing_sensory_texture"
        previous_families = {
            item
            for item in plan.required_slots.get("recent_adult_detail_families", "").split("|")
            if item
        }
        current_families = detect_adult_detail_families(reply_norm)
        if previous_families and current_families and current_families.issubset(previous_families):
            if not (previous_families >= ADULT_DETAIL_FAMILIES and len(current_families) >= 2):
                return "repeated_adult_detail_family"
        if plan.required_slots.get("escalation_source") in {"explicit_adult_followup", "sensual_ack_followup"}:
            if not detect_adult_followup_progression(reply_norm):
                return "flat_adult_followup_detail"
    if plan.shape == "answer_plan_status_then_continue":
        recent_plan_detail = plan.required_slots.get("recent_plan_detail")
        current_plan_detail = detect_availability_plan_detail(reply_norm)
        if recent_plan_detail and current_plan_detail == recent_plan_detail:
            return "repeated_plan_detail"
    if "ask_question" in forbidden and "question" in shapes:
        return "ask_question"
    if "repeat_apology_template" in forbidden and "repeat_apology_template" in shapes:
        return "repeat_apology_template"
    if plan.shape == "answer_why_then_stop_loop":
        if not {"reason_answer", "acknowledgement"}.issubset(shapes):
            return "missing_loop_reason"
        if "unrequested_reset_topic" in forbidden and ("question" in shapes or any(term in reply_norm for term in ("how bout", "how about", "tell me"))):
            return "unrequested_reset_topic"
    if plan.shape == "answer_properly_deepen_reason":
        if "question" in shapes:
            return "ask_question"
        if not detect_reason_answer(reply_norm):
            return "missing_deeper_reason"
        previous_signatures = {
            item
            for item in plan.required_slots.get("previous_reason_signatures", "").split("|")
            if item
        }
        current_signatures = detect_reason_signatures(reply_norm)
        if previous_signatures and current_signatures and current_signatures.issubset(previous_signatures):
            return "repeated_reason"
        if len(reply_clean.split()) < 8:
            return "missing_deeper_reason"
    if plan.shape == "age_fact_then_textured_continue":
        if "19" not in reply_norm:
            return "missing_age_fact"
        if "22" in reply_norm or "twenty two" in reply_norm:
            return "invent_identity_fact"
        dry_age_replies = {"19", "im 19", "i'm 19", "19 / wby", "19 wby", "19 / wbu", "im 19 / wby", "i'm 19 / wby"}
        if reply_clean in dry_age_replies:
            return "dry_age_mirror"
        if not any(
            term in reply_norm
            for term in (
                "why u asking",
                "why you asking",
                "what makes u ask",
                "what makes you ask",
                "dont make it sound like an interview",
                "don't make it sound like an interview",
                "same age",
                "wby then",
                "how old are u",
                "how old r u",
            )
        ):
            return "missing_textured_continuation"
    if plan.shape in {"identity_fact_then_continue", "identity_fact_restate_then_continue"} and plan.required_slots.get("identity_fact") == "education_status":
        if any(term in reply_norm for term in ("not at uni", "not studying", "no uni", "nothing atm")):
            return "deny_current_study"
        education_query = plan.required_slots.get("education_query", "")
        if education_query == "university_status":
            if "sampleford" not in reply_norm:
                return "missing_university_fact"
        elif not any(term in reply_norm for term in ("computer science", "comp sci")):
            return "missing_study_fact"
    if plan.shape == "work_status_fact_then_side_project":
        if any(term in reply_norm for term in ("uncle", "not at uni", "not studying", "no uni", "nothing atm")):
            return "invent_identity_fact"
        tokens = reply_clean.split()
        if not 5 <= len(tokens) <= 24:
            return "too_short" if len(tokens) < 5 else "violates_plan_too_long"
        has_study = any(term in reply_norm for term in ("computer science", "comp sci", "uni", "sampleford"))
        has_side = any(term in reply_norm for term in ("side", "running", "project", "building", "business", "software", "dev thing"))
        if not has_study:
            return "missing_study_fact"
        if not has_side:
            return "missing_side_project_fact"
    if plan.shape == "location_fact_then_light_return":
        if any(term in reply_norm for term in ("address", "postcode", "street", "accommodation", "building", "live location", "where u at", "where you at")):
            return "precise_location"
        if "sampleford for uni" in reply_norm:
            return "contradict_recent_education_status"
        tokens = reply_clean.split()
        if not 3 <= len(tokens) <= 16:
            return "too_short" if len(tokens) < 3 else "violates_plan_too_long"
        has_home_city = any(term in reply_norm for term in ("northbridge", "northbridge"))
        has_safe_context = any(term in reply_norm for term in ("sampleford", "mostly", "from", "wby", "wbu", "what about u", "what about you"))
        if not has_home_city or not has_safe_context:
            return "missing_location_fact"
    if plan.move == "unclassified" and plan.required_slots.get("unclassified_context") == "car_preference_question":
        tokens = reply_clean.split()
        if not 3 <= len(tokens) <= 18:
            return "too_short" if len(tokens) < 3 else "violates_plan_too_long"
        if any(term in reply_norm for term in ("what u mean", "what you mean", "what car", "idk", "dunno")):
            return "dodge_question"
        if not any(term in reply_norm for term in ("r8", "audi", "rs", "m4", "m3", "amg", "g wagon", "range rover")):
            return "missing_car_preference"
    if plan.move == "unclassified" and plan.required_slots.get("unclassified_context") == "story_reality_confirmation":
        tokens = reply_clean.split()
        if not 5 <= len(tokens) <= 24:
            return "too_short" if len(tokens) < 5 else "violates_plan_too_long"
        has_reaction = any(term in reply_norm for term in ("nah", "wtf", "mad", "serious", "actually", "thats crazy", "that's crazy", "are u okay", "u okay"))
        has_followup = "?" in reply_norm or any(term in reply_norm for term in ("did he", "what happened", "then what", "run out", "leave", "where did"))
        if not has_reaction:
            return "missing_story_reaction"
        if not has_followup:
            return "missing_story_followup"
    if plan.move == "unclassified" and plan.required_slots.get("unclassified_context") == "previous_comment_clarification":
        tokens = reply_clean.split()
        if not 7 <= len(tokens) <= 24:
            return "too_short" if len(tokens) < 7 else "violates_plan_too_long"
        has_clarifier = any(term in reply_norm for term in ("i mean", "meant", "meaning"))
        clarification_topic = str(plan.required_slots.get("clarification_topic") or "")
        if clarification_topic == "physical_compliment":
            has_content = any(term in reply_norm for term in ("outfit", "suited", "looked", "proper on u", "proper on you", "fit", "clinging", "tight"))
        elif clarification_topic == "wrong_context":
            has_content = any(term in reply_norm for term in ("random", "wrong context", "made no sense", "answered the wrong", "ignore that"))
        elif clarification_topic == "owner_state_or_activity":
            has_content = any(term in reply_norm for term in ("work", "coding", "project", "gym", "food", "shower", "bed", "watching", "chill", "switch off", "switching off", "week", "tired", "long day", "dead day", "drained", "what i was doing"))
        else:
            has_content = any(term in reply_norm for term in ("stuck", "running out", "forcing", "random questions", "looping", "things to say", "intense", "back and forth", "a lot", "reacting to", "what i said"))
        if not has_clarifier:
            return "missing_clarifier"
        if not has_content:
            return "missing_clarification_detail"
    if plan.move == "unclassified" and plan.required_slots.get("unclassified_context") == "answer_own_previous_prompt":
        tokens = reply_clean.split()
        if not 5 <= len(tokens) <= 24:
            return "too_short" if len(tokens) < 5 else "violates_plan_too_long"
        if "broad_question" in shapes or "question" in shapes:
            return "ask_question"
        prompt_topic = str(plan.required_slots.get("prompt_topic") or "")
        topic_terms = {
            "food": ("food", "ate", "burger", "pizza", "meal", "chicken", "place"),
            "film": ("film", "movie", "watch", "interstellar", "show"),
            "gym": ("gym", "legs", "push", "pull", "workout", "training", "exercise"),
            "dream": ("dream", "dreamt", "remember", "random"),
            "car": ("car", "r8", "audi", "range", "rover", "m4", "amg"),
            "story": ("story", "random", "weird", "happened"),
        }.get(prompt_topic, ("mine", "me", "my"))
        if not any(term in reply_norm for term in topic_terms):
            return "missing_prompt_answer"
    if plan.move == "unclassified" and plan.required_slots.get("unclassified_context") == "continue_previous_statement":
        tokens = reply_clean.split()
        if not 7 <= len(tokens) <= 24:
            return "too_short" if len(tokens) < 7 else "violates_plan_too_long"
        if "broad_question" in shapes or "question" in shapes:
            return "ask_question"
        has_continuation_detail = any(
            term in reply_norm
            for term in (
                "overthinking",
                "trying too hard",
                "sounding fake",
                "sounded fake",
                "went sideways",
                "forcing it",
                "trying to sound",
                "fill the silence",
                "real answer",
            )
        )
        if not has_continuation_detail:
            return "missing_continuation_detail"
    if plan.move == "unclassified" and plan.required_slots.get("unclassified_context") == "faith_prayer_question":
        tokens = reply_clean.split()
        if not 5 <= len(tokens) <= 22:
            return "too_short" if len(tokens) < 5 else "violates_plan_too_long"
        has_prayer_fact = any(term in reply_norm for term in ("pray", "prayer", "praying", "mosque", "religious", "muslim", "alhamdulillah"))
        has_human_qualifier = any(
            term in reply_norm
            for term in (
                "try",
                "trying",
                "not perfect",
                "when i can",
                "sometimes",
                "need to be better",
                "could be better",
                "alhamdulillah",
            )
        )
        if not has_prayer_fact:
            return "missing_prayer_fact"
        if not has_human_qualifier:
            return "too_dry"
    if plan.shape == "acknowledge_status_disclosure_plus_owner_detail":
        if "broad_question" in shapes or detect_bare_return_question(reply_norm):
            return "broad_question"
        if any(term in reply_norm for term in ("come here", "kiss", "lips", "neck", "hips", "thigh", "dirty", "wet", "hard")):
            return "adult_escalation"
        has_tired_ack = any(
            term in reply_norm
            for term in (
                "tired",
                "sleep",
                "nap",
                "rest",
                "long day",
                "fried",
                "finished",
                "drained",
                "knackered",
                "same",
                "feel u",
                "go lie down",
            )
        )
        has_detail = any(
            term in reply_norm
            for term in (
                "work",
                "gym",
                "coding",
                "brain",
                "head",
                "eyes",
                "bed",
                "day",
                "slept",
                "sleep",
                "half asleep",
            )
        )
        if not has_tired_ack:
            return "missing_status_acknowledgement"
        if not has_detail:
            return "missing_status_detail"
    if plan.shape == "answer_status_reason_then_continue":
        if "broad_question" in shapes or detect_bare_return_question(reply_norm):
            return "broad_question"
        if any(term in reply_norm for term in ("come here", "kiss", "lips", "neck", "hips", "thigh", "dirty", "wet", "hard")):
            return "adult_escalation"
        if reply_norm.strip(" .?!") in {"just am", "idk", "dunno", "because", "cos"}:
            return "missing_status_reason"
        if not any(
            term in reply_norm
            for term in (
                "work",
                "working",
                "coding",
                "code",
                "gym",
                "trained",
                "training",
                "legs",
                "sleep",
                "slept",
                "barely slept",
                "long day",
                "all day",
                "brain",
                "head",
                "eyes",
                "fried",
                "finished",
                "drained",
                "knackered",
            )
        ):
            return "missing_status_reason"
    if plan.shape == "reciprocate_affection_plus_specific_continuation":
        if (
            "broad_question" in shapes
            or detect_bare_return_question(reply_norm)
            or any(term in reply_norm for term in ("what u doing", "what you doing", "what are u doing", "what are you doing", "wyd"))
        ):
            return "broad_question"
        if any(term in reply_norm for term in ("tell me more", "what do you want to talk about", "what should we talk about")):
            return "broad_question"
        if not any(
            term in reply_norm
            for term in (
                "miss",
                "missed",
                "want",
                "wanted",
                "need",
                "crave",
                "thinking",
                "wish",
                "same",
            )
        ):
            return "missing_affection_reciprocity"
        if len(reply_norm.strip(" .?!").split()) < 4:
            return "too_short"
    if plan.shape == "playful_scold_acknowledge_then_soften":
        if (
            "broad_question" in shapes
            or detect_bare_return_question(reply_norm)
            or any(term in reply_norm for term in ("what u doing", "what you doing", "what are u doing", "what are you doing", "wyd"))
        ):
            return "broad_question"
        if any(term in reply_norm for term in ("lips", "neck", "hips", "thigh", "wet", "hard", "dirty", "moan")):
            return "adult_escalation"
        has_ack = any(
            term in reply_norm
            for term in (
                "okay",
                "ok",
                "fine",
                "alright",
                "behave",
                "behaving",
                "my bad",
                "u started",
                "you started",
                "i'll be good",
                "ill be good",
                "hands to myself",
            )
        )
        if not has_ack:
            return "missing_playful_scold_acknowledgement"
        if len(reply_norm.strip(" .?!").split()) < 4:
            return "too_short"
    if plan.shape == "acknowledge_dismissal_then_owner_side_reset":
        if "broad_question" in shapes:
            return "broad_question"
        has_ack = detect_loop_acknowledgement(reply_norm) or any(term in reply_norm for term in ("forget", "leave it", "leave that", "yh", "yeah", "fair"))
        has_owner_reset = any(
            term in reply_norm
            for term in (
                "my bad",
                "head went blank",
                "brain went blank",
                "brain lagged",
                "chatting rubbish",
                "talking rubbish",
                "lost the thread",
                "i was waffling",
                "i was wafflin",
                "i was chatting rubbish",
                "lost the thread",
                "lost my thread",
                "ignore me",
                "dead after the gym",
                "i'll stop",
                "ill stop",
            )
        )
        if not has_ack:
            return "missing_dismissal_acknowledgement"
        if not has_owner_reset:
            return "missing_owner_side_reset"
    if plan.shape == "fresh_status_detail_after_recent_status":
        if "question" in shapes:
            return "ask_question"
        if detect_bare_return_question(reply_norm):
            return "bare_return_question"
        if re.match(r"^(?:im|i'm|i am)\s+(?:good|fine|alright|okay|ok|calm|chilling|tired)\b", reply_clean):
            return "repeat_status_opener"
        if not any(
            term in reply_norm
            for term in (
                "gym",
                "legs",
                "nap",
                "bed",
                "work",
                "coding",
                "project",
                "phone",
                "shower",
                "food",
                "ate",
                "eating",
                "just got",
                "got in",
                "finished",
                "killed me",
                "tired",
                "half asleep",
                "on my feet",
                "brain",
                "lying",
                "laying",
                "relax",
                "relaxing",
                "switch off",
                "switching off",
                "watching",
                "watch",
                "game",
                "show",
                "clips",
                "film",
                "movie",
                "brothers",
                "chilling with",
                "thinking",
                "thinking about u",
                "thinking about you",
                "thinking of u",
                "thinking of you",
                "distracted",
                "mind",
            )
        ):
            return "missing_fresh_status_detail"
    if plan.shape == "answer_activity_detail_then_continue":
        if "question" in shapes and not any(term in reply_norm for term in ("i did", "i done", "did ", "hit ", "trained", "worked", "gym", "legs", "push", "pull", "weights", "cardio", "boxing", "box", "footwork")):
            return "mirror_question_without_answer"
        activity_scope = plan.required_slots.get("activity_scope", "")
        if activity_scope == "gym":
            gym_detail_terms = (
                "legs",
                "push",
                "pull",
                "chest",
                "back",
                "arms",
                "shoulder",
                "weights",
                "cardio",
                "bench",
                "squat",
                "deadlift",
                "machines",
                "workout",
                "trained",
                "hit ",
                "sets",
            )
            if not any(term in reply_norm for term in gym_detail_terms):
                return "missing_activity_detail"
        elif activity_scope == "low_activity":
            low_activity_terms = (
                "phone",
                "bed",
                "laid",
                "laying",
                "lying",
                "watching",
                "scrolling",
                "clips",
                "tiktok",
                "food",
                "ate",
                "eating",
                "thinking",
                "telly",
                "show",
                "film",
                "random stuff",
            )
            if not any(term in reply_norm for term in low_activity_terms):
                return "missing_activity_detail"
        elif activity_scope == "sport_skill":
            sport_skill_terms = (
                "boxing",
                "box",
                "cardio",
                "footwork",
                "pads",
                "rounds",
                "spar",
                "jab",
                "punch",
                "tiring",
                "hard",
                "killer",
                "humbles",
                "humbled",
                "trained",
                "used to",
            )
            if not any(term in reply_norm for term in sport_skill_terms):
                return "missing_activity_detail"
        elif not any(term in reply_norm for term in ("did", "been", "working", "coding", "gym", "work", "finished", "sorted", "made", "hit ")):
            return "missing_activity_detail"
        if len(reply_clean.split()) < 4:
            return "missing_activity_detail"
    if plan.shape == "acknowledge_ack_plus_specific_continuation":
        if "broad_question" in shapes:
            return "broad_question"
        bare_acks = {
            "same",
            "same tbh",
            "same icl",
            "same ngl",
            "nice",
            "oh fairs",
            "oh fair",
            "fair",
            "fairs",
            "okay then",
            "ok then",
        }
        if reply_clean in bare_acks:
            return "generic_ack"
        if len(reply_clean.split()) < 4:
            return "missing_specific_continuation"
        if not any(
            term in reply_norm
            for term in (
                "legs",
                "leg day",
                "stairs",
                "gym",
                "workout",
                "training",
                "tired",
                "finished",
                "killed",
                "humbled",
                "same",
                "exactly",
                "yh",
                "yeah",
                "fair",
                "icl",
                "ngl",
            )
        ):
            return "missing_specific_continuation"
    if plan.shape == "acknowledge_activity_opinion_plus_specific_comment":
        if "broad_question" in shapes or "question" in shapes:
            return "broad_question"
        if len(reply_clean.split()) < 5:
            return "missing_specific_activity_comment"
        if not any(term in reply_norm for term in ("legs", "leg day", "stairs", "gym", "workout", "training", "cardio", "humbles", "humbled", "evil", "killer", "walking")):
            return "missing_specific_activity_comment"
    if plan.shape == "acknowledge_loop_then_owner_detail":
        if "acknowledgement" not in shapes:
            return "missing_loop_acknowledgement"
        if "question" in shapes:
            return "ask_question"
        if len(reply_clean.split()) < 6:
            return "missing_specific_owner_detail"
    if plan.shape == "acknowledge_bad_question_then_owner_side_reset":
        if "question" in shapes or "invite_reveal" in shapes:
            return "ask_question"
        has_ack = "acknowledgement" in shapes or any(term in reply_norm for term in ("my bad", "yh", "yeah", "fair", "ur right", "u right"))
        has_owner_reset = any(
            term in reply_norm
            for term in (
                "dumb question",
                "bad question",
                "stupid",
                "came out dead",
                "came out stupid",
                "trying too hard",
                "moving lazy",
                "brain",
                "ignore me",
                "my head",
            )
        )
        if not has_ack:
            return "missing_repair_acknowledgement"
        if not has_owner_reset:
            return "missing_owner_side_reset"
    if plan.shape == "acknowledge_specific_question_family_without_repeat":
        if "acknowledgement" not in shapes:
            return "missing_loop_acknowledgement"
        if "question" in shapes:
            return "ask_question"
        if not any(
            term in reply_norm
            for term in (
                "how are u",
                "how r u",
                "how-are-u",
                "day",
                "asked",
                "interrogating",
                "interview",
                "defaulting",
                "no more",
                "i'll stop",
                "ill stop",
            )
        ):
            return "missing_specific_question_family_acknowledgement"
    if "broad_question" in forbidden and detect_broad_question(reply_norm):
        return "broad_question"
    if "mirror_question_without_answer" in forbidden and any(term in reply_norm for term in ("what u been up to", "what you been up to", "how u doing", "how you doing")):
        if not any(term in reply_norm for term in ("good", "calm", "alright", "busy", "work", "gym", "chilling", "tired", "not bad")):
            return "mirror_question_without_answer"
    return ""
