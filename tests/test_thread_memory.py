from __future__ import annotations

import json
from pathlib import Path

from libs.drafting.thread_memory import ThreadMemoryStore, load_indexable_thread_rows
from libs.drafting.training_data import ensure_training_message_files


def test_thread_memory_append_updates_summary_and_metadata(tmp_path: Path) -> None:
    store = ThreadMemoryStore(tmp_path / "thread_memory")

    store.append_turn("thread_1", session_id="session_1", role="user", text="hi", relationship_type="close_friend")
    store.append_turn(
        "thread_1",
        session_id="session_1",
        role="bot",
        text="same just chilling",
        relationship_type="close_friend",
        candidate_metadata={"scene_type": "current_activity_exchange", "required_reply_move": "answer_directly"},
    )
    episode = store.append_turn(
        "thread_1",
        session_id="session_1",
        role="bot",
        text="same just chilling",
        relationship_type="close_friend",
        candidate_metadata={"scene_type": "current_activity_exchange", "required_reply_move": "answer_directly"},
    )

    assert episode.thread_id == "thread_1"
    assert [turn.role for turn in episode.turns] == ["user", "bot", "bot"]
    assert episode.turns[-1].candidate_metadata["scene_type"] == "current_activity_exchange"
    assert "repeated same reply" in episode.bot_mistakes
    assert "stale activity fallback" in episode.bot_mistakes
    assert "stale self-state reply" in episode.failed_reply_patterns
    assert "stale self-state reply" in episode.summary


def test_thread_memory_summary_detects_ignored_topic_and_unresolved_context(tmp_path: Path) -> None:
    store = ThreadMemoryStore(tmp_path / "thread_memory")
    store.append_turn("thread_2", session_id="session_2", role="user", text="hm cars", relationship_type="close_friend")
    store.append_turn("thread_2", session_id="session_2", role="bot", text="im trying icl give me a topic", relationship_type="close_friend")
    episode = store.append_turn("thread_2", session_id="session_2", role="user", text="i just did bro", relationship_type="close_friend")

    assert episode.active_topic == "cars"
    assert "asked for topic after topic was provided" in episode.bot_mistakes
    assert "user called out missed context" in episode.unresolved_user_points


def test_thread_memory_retrieval_filters_sensitive_and_relationship(tmp_path: Path) -> None:
    store = ThreadMemoryStore(tmp_path / "thread_memory")
    store.append_turn("safe", session_id="s1", role="user", text="mate ur repeating", relationship_type="close_friend")
    safe = store.append_turn("safe", session_id="s1", role="bot", text="same just chilling", relationship_type="close_friend")
    safe.failed_reply_patterns.append("stale self-state reply")
    store.save_episode(safe)
    store.append_turn("sensitive", session_id="s2", role="user", text="my otp is 123456", relationship_type="close_friend")
    store.append_turn("romantic", session_id="s3", role="user", text="i missed u baby", relationship_type="romantic_interest")

    results = store.retrieve_similar(
        query_text="same just chilling repeated stale self-state",
        relationship_type="close_friend",
        scene_type="tired_mood",
        required_reply_move="empathetic_casual_response",
        limit=5,
    )

    ids = {item["thread_id"] for item in results}
    assert "safe" in ids
    assert "sensitive" not in ids
    assert "romantic" not in ids


def test_thread_memory_vector_rows_skip_sensitive(tmp_path: Path) -> None:
    store = ThreadMemoryStore(tmp_path / "thread_memory")
    store.append_turn("safe", session_id="s1", role="user", text="mate ur repeating", relationship_type="close_friend")
    safe = store.append_turn("safe", session_id="s1", role="bot", text="same just chilling", relationship_type="close_friend")
    safe.failed_reply_patterns.append("stale self-state reply")
    store.save_episode(safe)
    store.append_turn("sensitive", session_id="s2", role="user", text="my otp is 123456", relationship_type="close_friend")

    rows = load_indexable_thread_rows(tmp_path / "thread_memory")

    assert len(rows) == 1
    assert rows[0]["memory_type"] == "thread_episode"
    assert rows[0]["thread_id"] == "safe"
    assert "stale self-state reply" in rows[0]["embedding_text"]


def test_thread_memory_feedback_records_failure_context_and_tags(tmp_path: Path) -> None:
    store = ThreadMemoryStore(tmp_path / "thread_memory")
    candidate = {
        "text": "ok",
        "scene_type": "reciprocal_wellbeing_question",
        "required_reply_move": "answer_wellbeing_checkin",
        "required_move_satisfied": False,
        "forbidden_move_violated": True,
        "scene_fit_score": 0.1,
        "final_decision": "review",
    }
    store.append_turn("thread_feedback", session_id="s1", role="user", text="good how are u", relationship_type="close_friend")
    store.append_turn(
        "thread_feedback",
        session_id="s1",
        role="bot",
        text="ok",
        relationship_type="close_friend",
        candidate_metadata=candidate,
    )

    episode = store.record_feedback(
        "thread_feedback",
        "s1",
        "thumbs_down",
        correction="im good wbu",
        failure_context={
            "incoming": "good how are u",
            "context": ["hi", "yo how u been"],
            "selected_candidate": "ok",
            "selected_candidate_metadata": candidate,
            "candidates": [candidate, {"text": "im good wbu", "scene_type": "reciprocal_wellbeing_question"}],
        },
    )
    rows = [json.loads(line) for line in store.feedback_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert episode is not None
    assert "minimal_ok_reply" in episode.failed_reply_patterns
    assert "ignored_wellbeing_checkin" in episode.turns[-1].rejection_reasons
    assert rows[-1]["failure_tags"]
    assert "human_rejected_reply" in rows[-1]["failure_tags"]
    assert rows[-1]["scene_type"] == "reciprocal_wellbeing_question"
    assert rows[-1]["required_reply_move"] == "answer_wellbeing_checkin"
    assert rows[-1]["selected_candidate_metadata"]["text"] == "ok"
    assert rows[-1]["candidate_shortlist"][1]["text"] == "im good wbu"


def test_vector_db_build_writes_thread_episode_fallback(tmp_path: Path, monkeypatch) -> None:
    import scripts.build_reply_vector_db as builder

    training_dir = tmp_path / "training_messages"
    ensure_training_message_files(training_dir)
    index_dir = tmp_path / "vector_db" / "reply_examples_chroma"
    store = ThreadMemoryStore(tmp_path / "thread_memory")
    store.append_turn("thread_vector", session_id="s1", role="user", text="mate ur repeating", relationship_type="close_friend")
    episode = store.append_turn("thread_vector", session_id="s1", role="bot", text="same just chilling", relationship_type="close_friend")
    episode.failed_reply_patterns.append("stale self-state reply")
    store.save_episode(episode)

    monkeypatch.setattr(builder, "TRAINING_DIR", training_dir)
    monkeypatch.setattr(builder, "INDEX_DIR", index_dir)
    monkeypatch.setattr(builder, "FALLBACK_INDEX", index_dir / "reply_examples.jsonl")
    monkeypatch.setattr(builder, "load_indexable_thread_rows", lambda _path: load_indexable_thread_rows(tmp_path / "thread_memory"))

    result = builder.build_reply_vector_db()
    fallback_rows = [
        json.loads(line)
        for line in (index_dir / "reply_examples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert result["thread_episode_count"] == 1
    assert result["total_vector_count"] >= 1
    assert result["fallback_jsonl_written"] is True
    assert any(row.get("memory_type") == "thread_episode" for row in fallback_rows)


def test_catbot_failure_regression_builder_writes_review_cases(tmp_path: Path) -> None:
    from scripts.build_catbot_failure_regressions import build_catbot_failure_regressions

    feedback_path = tmp_path / "thread_feedback.jsonl"
    output_path = tmp_path / "catbot_failure_regressions.jsonl"
    feedback_path.write_text(
        json.dumps(
            {
                "thread_id": "thread_feedback",
                "session_id": "s1",
                "feedback": "thumbs_down",
                "timestamp": "2026-05-27T00:00:00+00:00",
                "incoming": "good how are u",
                "context": ["hi", "yo how u been"],
                "selected_candidate": "ok",
                "correction": "im good wbu",
                "scene_type": "reciprocal_wellbeing_question",
                "required_reply_move": "answer_wellbeing_checkin",
                "failure_tags": ["human_rejected_reply", "minimal_ok_reply"],
                "selected_candidate_metadata": {
                    "scene_type": "reciprocal_wellbeing_question",
                    "required_reply_move": "answer_wellbeing_checkin",
                    "scene_fit_score": 0.1,
                    "required_move_satisfied": False,
                    "forbidden_move_violated": True,
                },
                "candidate_shortlist": [{"text": "ok"}, {"text": "im good wbu"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = build_catbot_failure_regressions(feedback_path, output_path)
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert result["regression_cases_written"] == 1
    assert rows[0]["source"] == "catbot_thread_feedback"
    assert rows[0]["expected_scene_type"] == "reciprocal_wellbeing_question"
    assert rows[0]["expected_required_reply_move"] == "answer_wellbeing_checkin"
    assert "ok" in rows[0]["disallowed_replies"]
    assert rows[0]["correction"] == "im good wbu"
    assert rows[0]["status"] == "needs_human_review"
