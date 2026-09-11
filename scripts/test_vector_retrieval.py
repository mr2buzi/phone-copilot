from __future__ import annotations

from libs.drafting.retrieval_backends.lexical import LexicalRetrievalBackend


def assert_true(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def main() -> None:
    rows = [
        {"relationship_type": "close_friend", "intent_type": "greeting", "incoming": "hii", "my_reply": "heyy"},
        {"relationship_type": "close_friend", "intent_type": "planning", "incoming": "you coming later?", "my_reply": "yh what time"},
        {"relationship_type": "romantic_interest", "intent_type": "greeting", "incoming": "wyd", "my_reply": "keep talking like that baby"},
        {"relationship_type": "professional", "intent_type": "simple_question", "incoming": "can you send the report", "my_reply": "yes i can send it shortly"},
        {"relationship_type": "close_friend", "intent_type": "planning", "incoming": "you coming later?", "my_reply": "yh what time", "_source": "correction"},
        {"relationship_type": "close_friend", "intent_type": "planning", "incoming": "you coming later?", "my_reply": "maybe later", "is_synthetic": True},
    ]
    backend = LexicalRetrievalBackend(rows)
    greeting = backend.search("hii", [], "unknown", "greeting")
    assert_true(greeting, "hii should retrieve greeting examples")
    assert_true(all(item.intent_type == "greeting" for item in greeting), "hii retrieved non-greeting")
    assert_true(all("baby" not in item.my_reply.casefold() for item in greeting), "hii retrieved unsafe text")
    professional = backend.search("can you send the report", [], "professional", "professional")
    assert_true(professional, "professional search should return examples")
    assert_true(all(item.relationship_type in {"professional", "university"} for item in professional), "professional retrieved slang relationship")
    planning = backend.search("you coming later?", [], "close_friend", "planning")
    assert_true(planning[0].source == "correction", "correction should outrank synthetic")
    print("vector_retrieval_checks=passed")
    print(f"hii_results={[item.my_reply for item in greeting[:3]]}")


if __name__ == "__main__":
    main()
