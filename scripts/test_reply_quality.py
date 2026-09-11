from __future__ import annotations

from pathlib import Path

from libs.drafting import DraftingService
from libs.drafting.critic import critique_candidate


def main() -> None:
    service = DraftingService(
        template_path=Path("data/templates/reply_templates.json"),
        approved_photos_dir=Path("data/approved_photos"),
        ai_reply_enabled=False,
        training_messages_dir=Path("data/training_messages"),
        contact_overrides_path=Path("data/contact_overrides.json"),
    )
    scenarios = [
        ("close_friend", "you coming later?", "Best mate"),
        ("professional", "Can you confirm the meeting?", "Boss"),
        ("family", "call me when you can", "Mom"),
        ("romantic_interest", "wyd", "Demo partner"),
        ("unknown", "hello?", "New number"),
        ("unknown", "did you do it?", "Someone"),
    ]
    for expected, incoming, contact in scenarios:
        relationship = service._relationship_type(contact, [incoming])
        bundle = service.build_bundle([incoming], contact_name=contact)
        print("=" * 72)
        print(f"contact={contact} expected={expected} classified={relationship}")
        print(f"retrieved_examples={bundle.retrieved_examples_count}")
        print(f"decision={bundle.final_decision} blocked={bundle.blocked_reason}")
        for candidate in bundle.reply_candidates:
            print(f"- {candidate.candidate_kind}: {candidate.text}")
            print(f"  critic={candidate.critic_scores}")

    generic = service.build_bundle(["can you confirm the meeting?"], contact_name="Boss").reply_candidates[0]
    generic.text = "Hope you're doing well, that sounds great!"
    generic.sequence = [generic.text]
    critique = critique_candidate(
        generic,
        latest_message="can you confirm the meeting?",
        relationship_type="professional",
        style_score=0.9,
        relevance_score=0.95,
    )
    print("=" * 72)
    print("generic phrase penalty")
    print(critique.model_dump(mode="json"))


if __name__ == "__main__":
    main()
