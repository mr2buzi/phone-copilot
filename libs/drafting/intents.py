from __future__ import annotations

import re

GREETING_TOKENS = {
    "hi",
    "hii",
    "hey",
    "heyy",
    "yo",
    "hello",
    "hella",
    "helloo",
    "hellooo",
    "hiii",
    "hrella",
    "hiya",
    "sup",
    "wagwan",
}

GREETING_PHRASES = {
    "you good",
    "u good",
    "you okay",
    "u okay",
    "what u saying",
    "whats good",
    "what's good",
    "wyd",
}

PLANNING_PHRASES = {
    "coming later",
    "you coming",
    "you coming later",
    "are you coming",
    "still coming",
    "what time",
    "what time works",
    "what time are you",
    "what time you",
    "later tonight",
    "tomorrow",
    "tonight",
    "meet up",
    "meeting",
    "go out",
    "wanna go out",
    "want to go out",
    "go out thursday",
    "go out thirsday",
    "thursday",
    "thirsday",
    "call later",
    "can you meet",
    "are you free",
    "free later",
}

QUESTION_STARTERS = ("what", "how", "when", "why", "where", "can", "could", "do", "did", "are", "is", "was", "will")

WORK_PHRASES = {
    "report",
    "file",
    "meeting",
    "client",
    "project",
    "invoice",
    "deadline",
    "email",
    "presentation",
    "work",
    "professional",
}

STUDY_PHRASES = {
    "coursework",
    "homework",
    "revision",
    "revise",
    "lecture",
    "seminar",
    "assignment",
    "notes",
    "study",
    "studying",
    "uni work",
    "exam",
    "exams",
}

EXAM_LOGISTICS_PHRASES = {
    "exam",
    "exams",
    "revision",
    "revise",
    "clash",
    "supervised",
    "supervision",
    "supervise",
    "paper",
}

EXAM_LOGISTICS_CONTEXT_PHRASES = {
    "exam",
    "exams",
    "revision",
    "revise",
    "clash",
    "supervised",
    "supervision",
    "study",
    "studying",
}

EMOTIONAL_PHRASES = {
    "ignore me",
    "ignored me",
    "upset",
    "annoyed",
    "hurt",
    "sad",
    "mad at me",
    "why are you ignoring",
    "why u ignore",
}

BANTER_CHALLENGE_PHRASES = {
    "bit of a crazy thing to say no",
    "bit of a crazy thing to say no?",
    "that's crazy",
    "thats crazy",
    "ur weird",
    "you're weird",
    "youre weird",
    "bruh",
    "bro",
    "what",
    "??",
    "?!",
}

ARGUMENT_PHRASES = {
    "wtf",
    "what the hell",
    "why are u",
    "why are you",
    "why you",
    "pissed",
    "angry",
    "annoyed",
    "mad",
    "argue",
    "argument",
}

URGENT_PHRASES = {
    "urgent",
    "asap",
    "now",
    "right now",
    "call me now",
    "ring me now",
    "need you now",
}

THANKS_PHRASES = {"thanks", "thank you", "ty", "cheers", "thx"}
DECLINE_PHRASES = {"can't", "cant", "no thanks", "not now", "can't make it", "cannot make it", "nope"}
DELAY_PHRASES = {"running late", "be late", "be there soon", "5 mins", "10 mins", "few mins", "delayed"}
APOLOGY_PHRASES = {"sorry", "my bad", "apolog", "apologies"}
CONFIRMATION_PHRASES = {"yh", "yes", "yeah", "okay", "ok", "sure", "done", "sorted", "confirmed"}
AVAILABILITY_PHRASES = {"free", "available", "when works", "what time works", "what time are you free"}
ROMANTIC_PHRASES = {
    "baby",
    "babe",
    "sexy",
    "love you",
    "love u",
    "miss you",
    "miss u",
    "my handsome",
    "handsome",
    "mwah",
    "kiss",
    "cute",
    "cutie",
    "naughty",
    "x",
    "xx",
}

FLIRTY_TOKENS = {
    "baby",
    "babe",
    "sexy",
    "love",
    "kiss",
    "miss u",
    "miss you",
    "trouble",
    "come mine",
    "send pic",
    "naughty",
    "x",
    "xx",
}

SUSPICIOUS_REPLY_PHRASES = (
    "baby",
    "keep talking like that",
    "youre trouble",
    "you're trouble",
    "x",
    "xx",
    "trouble",
    "sexy",
    "babe",
    "love",
    "miss you",
    "come mine",
    "send pic",
    "wyd then",
    "naughty",
    "cum",
    "dick",
    "rape",
    "bdsm",
    "foreplay",
    "pin u down",
    "eat u",
    "princess",
    "beautiful girl",
    "ur still mine",
    "you're mine",
    "ur mine",
    "as your man",
    "my man",
    "obsessed",
    "i really like u",
    "sexual",
    "slurp",
    "juices",
    "tongue",
    "act up",
    "missed u",
    "miss u",
    "cute",
    "cutie",
    "fat",
    "kiss",
    "pleasure",
    "adore",
    "inside u",
    "good boy",
    "slut",
    "cucked",
    "horny",
    "pussy",
    "suck",
    "touch",
    "toucher",
    "belly",
    "obesity",
)

SAFE_GREETING_REPLIES = ["yo", "heyy", "u good", "what u saying"]

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def normalize_text(value: str) -> str:
    return " ".join(value.replace("\u2019", "'").split()).casefold()


def tokenize(value: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(normalize_text(value))]


def is_simple_greeting(text: str, context_messages: list[str] | None = None) -> bool:
    normalized = normalize_text(text)
    normalized_clean = normalized.strip("?!., ")
    if not normalized:
        return False
    if normalized_clean in GREETING_TOKENS or normalized_clean in GREETING_PHRASES:
        return True
    tokens = set(tokenize(normalized_clean))
    if tokens and tokens <= GREETING_TOKENS:
        return True
    if len(tokens) <= 3 and any(phrase in normalized for phrase in GREETING_PHRASES):
        return True
    context_count = len(context_messages or [])
    return context_count <= 1 and normalized in {"hi", "hii", "hey", "heyy", "yo", "hello", "sup"}


def normalize_intent_label(intent: str | None) -> str:
    normalized = (intent or "").strip().lower().replace("-", "_")
    aliases = {
        "plans": "planning",
        "plan": "planning",
        "question": "simple_question",
        "short_ack": "confirmation",
        "banter": "joke_banter_safe",
        "flirt": "romantic_flirty",
        "other": "unknown",
    }
    return aliases.get(normalized, normalized or "unknown")


def classify_intent(text: str, context_messages: list[str] | None = None) -> str:
    normalized = normalize_text(text)
    context_blob = normalize_text(" ".join(context_messages or []))
    tokens = set(tokenize(normalized))
    if any(term in normalized for term in BANTER_CHALLENGE_PHRASES):
        return "banter_challenge"
    if is_simple_greeting(normalized, context_messages):
        return "greeting"
    if any(term in normalized for term in EMOTIONAL_PHRASES):
        return "emotional"
    if any(term in normalized for term in ARGUMENT_PHRASES):
        return "argument"
    if any(term in normalized for term in URGENT_PHRASES):
        return "urgent"
    if any(term in normalized for term in WORK_PHRASES):
        return "professional"
    if any(term in normalized for term in EXAM_LOGISTICS_PHRASES) or (
        any(term in normalized for term in ("supervised", "supervision", "clash", "paper"))
        and any(term in context_blob for term in EXAM_LOGISTICS_CONTEXT_PHRASES)
    ):
        return "exam_logistics"
    if any(term in normalized for term in STUDY_PHRASES):
        return "studying_work"
    if any(term in normalized for term in THANKS_PHRASES):
        return "thanks"
    if any(term in normalized for term in APOLOGY_PHRASES):
        return "apology"
    if any(term in normalized for term in DECLINE_PHRASES):
        return "decline"
    if any(term in normalized for term in PLANNING_PHRASES):
        return "planning"
    if any(term in normalized for term in DELAY_PHRASES):
        return "delay"
    if any(term in normalized for term in AVAILABILITY_PHRASES):
        return "availability"
    if any(term in normalized for term in ROMANTIC_PHRASES):
        return "romantic_flirty"
    if normalized.endswith("?") or normalized.startswith(QUESTION_STARTERS):
        return "simple_question"
    if any(term in normalized for term in ("bro", "lol", "lmao", "loool", "nah", "wtf", "ffs", "haha", "haha", "lmaoo")):
        return "joke_banter_safe"
    if tokens and len(tokens) <= 4:
        return "confirmation"
    return "unknown"


def looks_overfamiliar(text: str) -> bool:
    return contains_suspicious_phrase(text)


def contains_suspicious_phrase(text: str) -> bool:
    normalized = normalize_text(text)
    tokens = tokenize(normalized)
    token_set = set(tokens)
    for phrase in SUSPICIOUS_REPLY_PHRASES:
        if phrase in {"x", "xx"}:
            if phrase in token_set:
                return True
            continue
        if " " in phrase:
            if phrase in normalized:
                return True
            continue
        if phrase in token_set:
            return True
    return False


def answers_greeting(text: str) -> bool:
    normalized = normalize_text(text)
    tokens = tokenize(normalized)
    if not tokens:
        return False
    if normalized in {"hi", "hii", "hey", "heyy", "yo", "hello", "sup", "wagwan", "u good", "you good", "what u saying"}:
        return True
    if any(phrase in normalized for phrase in ("u good", "you good", "what u saying", "what you saying")):
        return True
    return len(tokens) <= 3 and tokens[0] in {"hi", "hey", "heyy", "yo", "hello", "sup", "wagwan", "wyd"}
