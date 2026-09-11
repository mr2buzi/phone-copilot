from __future__ import annotations

import json
from pathlib import Path

RELATIONSHIPS = ["close_friend", "casual_friend", "family", "university", "professional", "unknown"]
INTENTS = [
    "greeting",
    "planning",
    "simple_question",
    "casual_checkin",
    "joke_banter_safe",
    "apology",
    "thanks",
    "confirmation",
    "decline",
    "delay",
    "studying_work",
    "availability",
    "follow_up",
    "unknown_safe",
]

INCOMING_BY_INTENT = {
    "greeting": ["hii", "hey", "yo", "you good", "wyd"],
    "planning": ["you coming later?", "what time you coming?", "we still meeting?", "you free later?"],
    "simple_question": ["can you send it?", "did you see this?", "where is it?", "can you check?"],
    "casual_checkin": ["you alright?", "what you saying?", "how's your day?", "you awake?"],
    "joke_banter_safe": ["that was funny", "nah you're joking", "bro what was that", "loool fair"],
    "apology": ["sorry about that", "my bad", "sorry i missed it", "apologies for earlier"],
    "thanks": ["thanks", "cheers", "thank you", "appreciate it"],
    "confirmation": ["is that okay?", "all good?", "sorted?", "does that work?"],
    "decline": ["can you come now?", "want to go out?", "can you do it today?", "are you joining?"],
    "delay": ["where are you?", "you nearly here?", "how long?", "are you late?"],
    "studying_work": ["did you do the work?", "can you send the notes?", "when is the deadline?", "revision later?"],
    "availability": ["are you free?", "when works?", "you available today?", "what time works?"],
    "follow_up": ["any update?", "did you check?", "what happened with that?", "you sorted it?"],
    "unknown_safe": ["okay", "fair", "interesting", "not sure what you mean"],
}

REPLIES = {
    "close_friend": {
        "greeting": ["yo", "heyy", "u good", "what u saying"],
        "planning": ["yh what time", "maybe later", "where u lot going", "depends what time"],
        "simple_question": ["yeah one sec", "ill check", "not sure yet", "send it again"],
        "casual_checkin": ["im good wbu", "yeah calm", "not bad tbf", "just chilling"],
        "joke_banter_safe": ["loool fair", "nah behave", "bro relax", "thats funny icl"],
        "apology": ["its calm", "dont worry", "all good", "safe"],
        "thanks": ["safe", "no worries", "all good", "got you"],
        "confirmation": ["yeah thats fine", "yh calm", "works for me", "sorted"],
        "decline": ["nah not today", "cant today", "maybe another time", "not feeling it"],
        "delay": ["few mins", "nearly there", "leaving now", "ill be a bit late"],
        "studying_work": ["ill check notes", "not done yet", "send me yours", "we can do it later"],
        "availability": ["maybe later", "free later", "what time", "yh should be"],
        "follow_up": ["ill check now", "not yet", "one sec", "ill let you know"],
        "unknown_safe": ["fair", "yeah", "what do you mean", "calm"],
    },
    "casual_friend": {},
    "family": {},
    "university": {},
    "professional": {},
    "unknown": {},
}

REPLIES["casual_friend"] = {k: list(v) for k, v in REPLIES["close_friend"].items()}
REPLIES["family"] = {
    intent: ["okay", "yes thats fine", "ill check", "ill let you know"]
    for intent in INTENTS
}
REPLIES["university"] = {
    intent: ["yeah ill check", "not sure yet", "what time works", "ill send it later"]
    for intent in INTENTS
}
REPLIES["professional"] = {
    intent: ["yes i can confirm", "i'll check and confirm", "that works for me", "i'll send it shortly"]
    for intent in INTENTS
}
REPLIES["unknown"] = {
    intent: ["okay", "not sure yet", "what do you mean", "ill check"]
    for intent in INTENTS
}

OUTPUT_PATH = Path("data/training_messages/generated_safe_templates.jsonl")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    target = 15000
    index = 0
    while len(rows) < target:
        relationship = RELATIONSHIPS[index % len(RELATIONSHIPS)]
        intent = INTENTS[(index // len(RELATIONSHIPS)) % len(INTENTS)]
        incoming_options = INCOMING_BY_INTENT[intent]
        reply_options = REPLIES[relationship][intent]
        incoming = incoming_options[(index // (len(RELATIONSHIPS) * len(INTENTS))) % len(incoming_options)]
        reply_base = reply_options[(index + len(rows)) % len(reply_options)]
        suffix_options = ["", " now", " later", " in a bit", " if that works", " when you can"]
        suffix = suffix_options[(index // (len(RELATIONSHIPS) * len(INTENTS) * len(incoming_options))) % len(suffix_options)]
        reply = (reply_base + suffix).strip()
        rows.append(
            {
                "id": f"generated_safe_template_{len(rows)+1:05d}",
                "relationship_type": relationship,
                "intent_type": "unknown" if intent == "unknown_safe" else intent,
                "incoming": incoming,
                "context": [],
                "my_reply": reply,
                "notes": f"synthetic safe {intent} fallback",
                "is_synthetic": True,
                "source": "generated_safe_template",
                "style_authority": "low",
            }
        )
        index += 1
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"generated={len(rows)} path={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
