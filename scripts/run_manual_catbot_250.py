from __future__ import annotations

import argparse
import re
from pathlib import Path

from libs.drafting import DraftingService
from libs.drafting.conversation_state import ConversationStateStore
from libs.drafting.intents import normalize_text


SEED_MESSAGES = [
    "hi",
    "i been good wby",
    "i missed u baby",
    "wyd",
    "honestly just work wby",
    "im tired icl",
    "why u tired",
    "what did u do today",
    "how was ur day",
    "how come",
    "yeah i feel u",
    "im in bed",
    "wdym",
    "oh fairs",
    "idk what to say tbh",
    "u tell me",
    "cars then",
    "dream car is an r8 wby",
    "what do u study",
    "what do u do",
    "what project u working on",
    "where u from",
    "how old r u",
    "same",
    "nice",
    "what u been up to",
    "really",
    "what did u do in gym",
    "i dont know how to code lol",
    "wdym u lit do coding",
    "it wasnt dead it was a lie",
    "bro what",
    "thats dry mate",
    "is everything alright",
    "yeah its ok",
    "guess what",
    "i was at home yeah and showering and some guy came in and started running in my living room",
    "bruh wtf",
    "nah like actually",
    "what would u do",
    "im bored",
    "entertain me",
    "u already asked that",
    "ur boring me",
    "idk talk",
    "u pick",
    "gym then",
    "what u training",
    "i hate legs",
    "same tbh",
    "do u box",
    "is boxing hard",
    "would u fight me",
    "behave",
    "lol why",
    "where u been",
    "dwdw i missed u anyway where u been",
    "wyd later",
    "u doing anything nice",
    "trust me",
    "how are you",
    "u good",
    "are u okay",
    "why u being weird",
    "i asked u a question",
    "are u gonna answer",
    "nth wby",
    "i js asked u a question",
    "lol im chilling",
    "yeah why",
    "what have u been doing",
    "i been busy asf",
    "thats a paradox lol",
    "what dyu study",
    "im into tech",
    "same",
    "what part of tech",
    "ai and software",
    "thats sick",
    "what car u like",
    "u tell me a topic",
    "boxing or cars",
    "cars",
    "what would u get if money wasnt a thing",
    "probably urus",
    "valid",
    "what about u",
    "whats ur dream",
    "do u pray",
    "same",
]

BAD_EXACT = {
    "ok",
    "okay",
    "k",
    "calm",
    "same icl",
    "same just chilling",
    "fair just chilling too",
    "what u saying",
    "what u saying then",
    "say less what u doing",
    "fine then what should we talk about",
    "fine then give me a topic",
}
BAD_SUBSTRINGS = (
    "u ain't giving me much",
    "u aint giving me much",
    "give me a topic",
    "what should we talk about",
    "what bruh wtf u into",
    "what what u into",
    "bro why is his dad involved",
)
GENERIC_PROMPTS = ("what u saying", "what u been up to", "what u doing", "wyd", "what u been doing")
DEFAULT_TURN_COUNT = 550


def _contains_any(text: str, terms: tuple[str, ...] | list[str] | set[str]) -> bool:
    return any(term in text for term in terms)


def _messages(count: int) -> list[str]:
    variations = ["btw", "icl", "lol", "ngl", "tbh", "lowk", ""]
    messages: list[str] = []
    for index in range(count):
        message = SEED_MESSAGES[index % len(SEED_MESSAGES)]
        if index >= len(SEED_MESSAGES) and index % 17 == 0 and variations[index % len(variations)]:
            message = f"{message} {variations[index % len(variations)]}"
        messages.append(message)
    return messages


def _fail_reason(user: str, reply: str, metadata: dict[str, object], conversation: list[str]) -> str:
    user_norm = normalize_text(user).strip(" .?!")
    reply_norm = normalize_text(reply).strip(" .?!")
    scene_type = str(metadata.get("scene_type") or "")
    if metadata.get("_final_decision") == "reject":
        return "top candidate rejected"
    if reply_norm in BAD_EXACT:
        return "dead/stale exact reply"
    if _contains_any(reply_norm, BAD_SUBSTRINGS):
        return "bad generic/blame/random substring"
    recent_bots = [normalize_text(line[6:]).strip(" .?!") for line in conversation if line.startswith("[ME]:")][-8:]
    for prompt in GENERIC_PROMPTS:
        if prompt in reply_norm and sum(1 for bot in recent_bots if prompt in bot) >= 1:
            return "repeated generic prompt"
    if ("wby" in user_norm or "wbu" in user_norm or re.search(r"\bu\b$", user_norm)) and _contains_any(
        user_norm,
        ["chilling", "work", "good", "nothing", "nth", "bed", "busy", "r8", "urus"],
    ):
        if _contains_any(reply_norm, ["ok", "what u saying", "what u doing", "what u been", "give me a topic", "say less"]):
            return "ignored reciprocal question"
    if _contains_any(user_norm, ["missed u", "missed you", "i miss"]) and not _contains_any(
        reply_norm,
        ["missed", "sweet", "bless", "appreciate"],
    ):
        return "missed affection"
    if "what part of tech" in user_norm and not _contains_any(reply_norm, ["ai", "software", "coding", "backend", "project", "building"]):
        return "missed tech question"
    if _contains_any(user_norm, ["what car u like", "what car do u like"]) and not _contains_any(
        reply_norm,
        ["urus", "r8", "rs6", "m4", "porsche", "lambo", "audi", "bmw", "merc"],
    ):
        return "missed car preference"
    if "what about u" in user_norm and "cars" in normalize_text(" ".join(conversation[-10:])) and not _contains_any(
        reply_norm,
        ["urus", "r8", "rs6", "m4", "porsche", "lambo", "audi", "bmw", "merc", "mine", "id go"],
    ):
        return "missed reciprocal car topic"
    if "why u tired" in user_norm and not _contains_any(reply_norm, ["boxing", "gym", "work", "coding", "training", "clients", "draining"]):
        return "missed tired reason"
    if _contains_any(user_norm, ["what do u study", "what dyu study", "what course"]) and not _contains_any(
        reply_norm,
        ["comp sci", "computer science", "study tech"],
    ):
        return "missed study question"
    if "what do u do" in user_norm and not _contains_any(
        reply_norm,
        ["comp sci", "study", "uni", "project", "building", "stuff on the side"],
    ):
        return "missed work question"
    if _contains_any(user_norm, ["what project", "what u working on"]) and not _contains_any(
        reply_norm,
        ["project", "building", "software", "dev", "side"],
    ):
        return "missed project question"
    if "how old" in user_norm and "19" not in reply_norm:
        return "missed age question"
    if "where u from" in user_norm and not _contains_any(reply_norm, ["northbridge", "northbridge", "sampleford"]):
        return "missed location question"
    if _contains_any(user_norm, ["u already asked", "asked this already", "keep asking"]) and not _contains_any(
        reply_norm,
        ["asked", "loop", "repeated", "my bad", "npc", "same thing"],
    ):
        return "failed anti-loop repair"
    if _contains_any(user_norm, ["idk talk", "u tell me", "u pick", "entertain me", "tell me a topic"]) and _contains_any(
        reply_norm,
        ["what u saying", "give me a topic", "what should we talk about", "u ain"],
    ):
        return "failed to choose topic"
    if _contains_any(user_norm, ["i asked u a question", "are u gonna answer", "i js asked"]) and _contains_any(
        reply_norm,
        ["ok", "nah i get u", "say less", "what u saying", "what u doing"],
    ):
        return "ignored question debt callout"
    if _contains_any(user_norm, ["im tired", "tired icl"]) and reply_norm in {"same icl", "valid", "fair"}:
        return "tired mood underanswered"
    if _contains_any(user_norm, ["guess what", "some guy came in", "living room"]) and reply_norm in {"ok", "okay", "calm", "fair"}:
        return "story underreaction"
    direct_scenes = {
        "owner_tech_interest_question",
        "owner_study_question",
        "owner_work_question",
        "owner_project_question",
        "owner_age_question",
        "owner_location_question",
        "owner_car_preference_question",
        "owner_dream_question",
        "owner_prayer_question",
        "owner_tired_reason_question",
    }
    if scene_type in direct_scenes and not metadata.get("required_move_satisfied"):
        return "direct identity/preference scene not satisfied"
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=DEFAULT_TURN_COUNT)
    args = parser.parse_args()
    turn_count = max(1, int(args.count))
    state_path = Path(f"tmp_manual_{turn_count}_conversation_state.json")
    if state_path.exists():
        state_path.unlink()
    service = DraftingService(
        template_path=Path("tmp_manual_check_template.json"),
        approved_photos_dir=Path("tmp_debug_manual/approved"),
        ai_reply_enabled=False,
    )
    service.conversation_state = ConversationStateStore(state_path)
    conversation: list[str] = []
    transcript: list[tuple[object, ...]] = []
    for index, user in enumerate(_messages(turn_count), 1):
        conversation.append(f"[OTHER]: {user}")
        bundle = service.build_bundle_with_context([], conversation, contact_name="Catbot", relationship_type="close_friend")
        top = bundle.reply_candidates[bundle.recommended_reply_index or 0]
        metadata = dict(top.critic_scores)
        metadata["_final_decision"] = top.final_decision
        reason = _fail_reason(user, top.text, metadata, conversation)
        transcript.append((index, user, top.text, metadata.get("scene_type"), metadata.get("required_reply_move"), top.final_decision, reason))
        conversation.append(f"[ME]: {top.text}")
        if reason:
            print("FAIL", index, reason)
            print("USER:", user)
            print("BOT :", top.text)
            print(
                "scene=",
                metadata.get("scene_type"),
                "move=",
                metadata.get("required_reply_move"),
                "agenda=",
                metadata.get("agenda_state"),
                "job=",
                metadata.get("policy_conversation_job"),
                "decision=",
                top.final_decision,
                "blocked=",
                top.blocked_reason,
            )
            print("recent:")
            for line in conversation[-12:]:
                print(line)
            print("candidates:")
            for candidate in bundle.reply_candidates[:8]:
                print(
                    "-",
                    candidate.text,
                    candidate.final_decision,
                    round(candidate.score_breakdown.final_confidence, 3),
                    candidate.blocked_reason,
                    candidate.critic_scores.get("scene_type"),
                    candidate.critic_scores.get("required_move_satisfied"),
                    candidate.critic_scores.get("scene_mismatch_reason"),
                    candidate.critic_scores.get("policy_violation_reasons"),
                )
            return 1
    print(f"PASS {turn_count}")
    print("first15")
    for row in transcript[:15]:
        print(row)
    print("last15")
    for row in transcript[-15:]:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
