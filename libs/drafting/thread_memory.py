from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from libs.drafting.intents import contains_suspicious_phrase, normalize_text
from libs.drafting.training_data import normalize_relationship_type


THREAD_MEMORY_DIR = Path("data/thread_memory")
THREAD_EPISODES_FILE = "thread_episodes.jsonl"
THREAD_FEEDBACK_FILE = "thread_feedback.jsonl"
THREAD_SUMMARIES_FILE = "thread_summaries.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9']+", normalize_text(text)) if len(token) > 1}


def _safe_thread_text(text: str) -> str:
    redacted = re.sub(r"\b\d{4,8}\b", "[redacted-code]", text)
    redacted = re.sub(r"\b(?:\d[ -]*?){13,19}\b", "[redacted-card]", redacted)
    redacted = re.sub(r"(?i)\b(password|otp|passcode|pin)\s*[:=]?\s*\S+", r"\1 [redacted]", redacted)
    return redacted.strip()


def _contains_sensitive(text: str) -> bool:
    lowered = normalize_text(text)
    return bool(
        re.search(r"\b\d{4,8}\b", text)
        or re.search(r"\b(?:\d[ -]*?){13,19}\b", text)
        or any(term in lowered for term in ("password", "passcode", "otp", "bank card", "sort code", "address is"))
        or contains_suspicious_phrase(text)
    )


def _safe_memory_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return "[truncated]"
    if isinstance(value, str):
        safe = _safe_thread_text(value)
        return safe[:1000]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, list):
        return [_safe_memory_value(item, depth=depth + 1) for item in value[:12]]
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.casefold() in {"api_key", "authorization", "token", "secret", "password"}:
                safe[key_text] = "[redacted]"
            else:
                safe[key_text] = _safe_memory_value(item, depth=depth + 1)
        return safe
    return str(value)[:500]


def _candidate_feedback_snapshot(candidate: dict[str, Any]) -> dict[str, Any]:
    critic_scores = candidate.get("critic_scores") if isinstance(candidate.get("critic_scores"), dict) else {}
    keys = (
        "text",
        "conversation_function",
        "scene_type",
        "required_reply_move",
        "forbidden_reply_moves",
        "scene_fit_score",
        "required_move_satisfied",
        "forbidden_move_violated",
        "scene_mismatch_reason",
        "contextual_specificity_score",
        "human_likeness_score",
        "direct_answer_score",
        "emotional_presence_score",
        "repair_specificity_score",
        "identity_answer_score",
        "canned_reply_penalty",
        "stale_template_penalty",
        "deterministic_fallback_penalty",
        "repeated_failed_pattern_penalty",
        "semantic_contamination_penalty",
        "fallback_used",
        "generated_by_provider",
        "provider_name",
        "final_decision",
        "blocked_reason",
        "risk_flags",
    )
    snapshot = {key: candidate.get(key) for key in keys if key in candidate}
    if critic_scores:
        snapshot["critic_scores"] = {
            key: critic_scores.get(key)
            for key in (
                "scene_type",
                "required_reply_move",
                "conversation_function",
                "scene_fit_score",
                "required_move_satisfied",
                "forbidden_move_violated",
                "fallback_used",
                "provider_error",
            )
            if key in critic_scores
        }
    return _safe_memory_value(snapshot)


def _derive_failure_tags(
    *,
    selected_reply: str,
    selected_candidate: dict[str, Any],
    incoming: str,
    context: list[str],
    correction: str | None,
) -> list[str]:
    tags = ["human_rejected_reply"]
    normalized_reply = normalize_text(selected_reply).strip(" .?!")
    normalized_incoming = normalize_text(incoming)
    scene_type = str(selected_candidate.get("scene_type") or "")
    required_move = str(selected_candidate.get("required_reply_move") or "")
    final_decision = str(selected_candidate.get("final_decision") or "")
    forbidden_violated = bool(selected_candidate.get("forbidden_move_violated"))
    required_satisfied = selected_candidate.get("required_move_satisfied")
    scene_fit = selected_candidate.get("scene_fit_score")

    if normalized_reply in {"ok", "okay", "k"}:
        tags.append("minimal_ok_reply")
    if normalized_reply in {"same icl", "same just chilling", "fair just chilling too", "just chilling"}:
        tags.append("stale_self_state_reply")
    if normalized_reply in {"what u saying", "what u saying then", "yo what u saying"}:
        tags.append("generic_hook_reply")
    if required_satisfied is False:
        tags.append("required_move_not_satisfied")
    if forbidden_violated:
        tags.append("forbidden_move_violated")
    if isinstance(scene_fit, (int, float)) and float(scene_fit) < 0.5:
        tags.append("low_scene_fit")
    if final_decision == "reject":
        tags.append("selected_candidate_was_rejected")
    if scene_type:
        tags.append(f"scene:{scene_type}")
    if required_move:
        tags.append(f"required_move:{required_move}")
    if any(term in normalized_incoming for term in ("how are u", "how are you", "hru", "how u doing")) and normalized_reply in {"ok", "okay", "k"}:
        tags.append("ignored_wellbeing_checkin")
    if any(term in normalized_incoming for term in ("why u keep", "why you keep", "u keep saying", "you keep saying")):
        tags.append("repetition_callout_failure")
    if "?" in incoming and required_move.startswith("answer") and required_satisfied is False:
        tags.append("ignored_direct_question")
    if correction and correction.strip():
        tags.append("human_correction_available")
    if any(_contains_sensitive(item) for item in [selected_reply, incoming, *context, correction or ""]):
        tags.append("contains_sensitive")
    return sorted(dict.fromkeys(tags))


@dataclass
class ThreadTurn:
    role: str
    text: str
    timestamp: str | None = None
    candidate_metadata: dict[str, Any] | None = None
    feedback: str | None = None
    rejection_reasons: list[str] = field(default_factory=list)
    correction: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ThreadTurn":
        return cls(
            role=str(payload.get("role") or "user"),
            text=str(payload.get("text") or ""),
            timestamp=str(payload.get("timestamp")) if payload.get("timestamp") else None,
            candidate_metadata=payload.get("candidate_metadata") if isinstance(payload.get("candidate_metadata"), dict) else None,
            feedback=str(payload.get("feedback")) if payload.get("feedback") else None,
            rejection_reasons=[str(item) for item in payload.get("rejection_reasons", []) if str(item).strip()] if isinstance(payload.get("rejection_reasons"), list) else [],
            correction=str(payload.get("correction")) if payload.get("correction") else None,
        )


@dataclass
class ThreadEpisode:
    thread_id: str
    session_id: str
    platform: str = "unknown"
    contact_alias: str | None = None
    relationship_type: str = "unknown"
    started_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    turns: list[ThreadTurn] = field(default_factory=list)
    summary: str = ""
    active_topic: str | None = None
    emotional_arc: str | None = None
    unresolved_user_points: list[str] = field(default_factory=list)
    resolved_user_points: list[str] = field(default_factory=list)
    bot_mistakes: list[str] = field(default_factory=list)
    repeated_bot_replies: list[str] = field(default_factory=list)
    successful_reply_patterns: list[str] = field(default_factory=list)
    failed_reply_patterns: list[str] = field(default_factory=list)
    dominant_scene_types: list[str] = field(default_factory=list)
    latest_scene_type: str | None = None
    latest_required_reply_move: str | None = None
    safety_flags: list[str] = field(default_factory=list)
    contains_sensitive: bool = False
    quarantined: bool = False
    style_authority: str = "medium"
    source: str = "catbot_thread"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["turns"] = [turn.to_dict() for turn in self.turns]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ThreadEpisode":
        turns = [ThreadTurn.from_dict(item) for item in payload.get("turns", []) if isinstance(item, dict)]
        return cls(
            thread_id=str(payload.get("thread_id") or ""),
            session_id=str(payload.get("session_id") or ""),
            platform=str(payload.get("platform") or "unknown"),
            contact_alias=str(payload.get("contact_alias")) if payload.get("contact_alias") else None,
            relationship_type=normalize_relationship_type(str(payload.get("relationship_type") or "unknown")),
            started_at=str(payload.get("started_at") or utc_now()),
            updated_at=str(payload.get("updated_at") or utc_now()),
            turns=turns,
            summary=str(payload.get("summary") or ""),
            active_topic=str(payload.get("active_topic")) if payload.get("active_topic") else None,
            emotional_arc=str(payload.get("emotional_arc")) if payload.get("emotional_arc") else None,
            unresolved_user_points=[str(item) for item in payload.get("unresolved_user_points", []) if str(item).strip()] if isinstance(payload.get("unresolved_user_points"), list) else [],
            resolved_user_points=[str(item) for item in payload.get("resolved_user_points", []) if str(item).strip()] if isinstance(payload.get("resolved_user_points"), list) else [],
            bot_mistakes=[str(item) for item in payload.get("bot_mistakes", []) if str(item).strip()] if isinstance(payload.get("bot_mistakes"), list) else [],
            repeated_bot_replies=[str(item) for item in payload.get("repeated_bot_replies", []) if str(item).strip()] if isinstance(payload.get("repeated_bot_replies"), list) else [],
            successful_reply_patterns=[str(item) for item in payload.get("successful_reply_patterns", []) if str(item).strip()] if isinstance(payload.get("successful_reply_patterns"), list) else [],
            failed_reply_patterns=[str(item) for item in payload.get("failed_reply_patterns", []) if str(item).strip()] if isinstance(payload.get("failed_reply_patterns"), list) else [],
            dominant_scene_types=[str(item) for item in payload.get("dominant_scene_types", []) if str(item).strip()] if isinstance(payload.get("dominant_scene_types"), list) else [],
            latest_scene_type=str(payload.get("latest_scene_type")) if payload.get("latest_scene_type") else None,
            latest_required_reply_move=str(payload.get("latest_required_reply_move")) if payload.get("latest_required_reply_move") else None,
            safety_flags=[str(item) for item in payload.get("safety_flags", []) if str(item).strip()] if isinstance(payload.get("safety_flags"), list) else [],
            contains_sensitive=bool(payload.get("contains_sensitive")),
            quarantined=bool(payload.get("quarantined")),
            style_authority=str(payload.get("style_authority") or "medium"),
            source=str(payload.get("source") or "catbot_thread"),
        )


def summarize_episode(episode: ThreadEpisode) -> ThreadEpisode:
    user_texts = [turn.text for turn in episode.turns if turn.role == "user"]
    bot_texts = [turn.text for turn in episode.turns if turn.role == "bot"]
    all_text = " ".join(user_texts + bot_texts)
    normalized = normalize_text(all_text)
    bot_norms = [normalize_text(text).strip(" .?!") for text in bot_texts if text.strip()]
    repeated = sorted({reply for reply in bot_norms if bot_norms.count(reply) > 1})
    mistakes: list[str] = []
    failed: list[str] = []
    unresolved: list[str] = []
    if repeated:
        mistakes.append("repeated same reply")
        failed.append("repeated bot reply")
    stale = [reply for reply in bot_norms if reply in {"same just chilling", "fair just chilling too", "same icl", "just chilling"}]
    if stale:
        mistakes.append("stale activity fallback")
        failed.append("stale self-state reply")
    if any(term in normalized for term in ("i just told", "i js told", "i just did", "i js did", "u said that", "repeating", "ignore my question", "missing context")):
        mistakes.append("missed user context")
        unresolved.append("user called out missed context")
    if any(term in normalized for term in ("give me a topic", "what should we talk about")) and any(term in normalized for term in (" cars", " gym", " music", " football", " movies")):
        mistakes.append("asked for topic after topic was provided")
        failed.append("asked for topic after topic was provided")
    if any(term in normalized for term in ("dad involved", "his dad", "father involved")):
        mistakes.append("weird unrelated entity introduced")
        failed.append("weird unrelated fallback")
    if any(term in normalized for term in ("missed u", "are u ok", "u seem off")) and stale:
        mistakes.append("ignored emotional or care context")
        unresolved.append("emotional or care point may be unresolved")

    successful: list[str] = []
    for turn in episode.turns:
        meta = turn.candidate_metadata or {}
        if turn.role == "bot" and str(turn.feedback or "") == "thumbs_up":
            successful.append(str(meta.get("required_reply_move") or meta.get("scene_type") or "accepted reply"))
        if turn.correction:
            successful.append("human correction")

    scene_types = [
        str((turn.candidate_metadata or {}).get("scene_type") or "")
        for turn in episode.turns
        if turn.role == "bot" and isinstance(turn.candidate_metadata, dict)
    ]
    episode.dominant_scene_types = sorted({scene for scene in scene_types if scene})
    if scene_types:
        episode.latest_scene_type = scene_types[-1]
    required_moves = [
        str((turn.candidate_metadata or {}).get("required_reply_move") or "")
        for turn in episode.turns
        if turn.role == "bot" and isinstance(turn.candidate_metadata, dict)
    ]
    if required_moves:
        episode.latest_required_reply_move = required_moves[-1] or None
    for turn in episode.turns:
        if _contains_sensitive(turn.text):
            episode.contains_sensitive = True
            if "contains_sensitive" not in episode.safety_flags:
                episode.safety_flags.append("contains_sensitive")
    episode.quarantined = episode.quarantined or episode.contains_sensitive
    episode.repeated_bot_replies = repeated
    episode.bot_mistakes = sorted(set([*episode.bot_mistakes, *mistakes]))
    episode.failed_reply_patterns = sorted(set([*episode.failed_reply_patterns, *failed]))
    episode.successful_reply_patterns = sorted(set([*episode.successful_reply_patterns, *successful]))
    episode.unresolved_user_points = sorted(set([*episode.unresolved_user_points, *unresolved]))
    episode.active_topic = _active_topic(user_texts)
    episode.emotional_arc = _emotional_arc(normalized)
    episode.summary = _summary_text(episode, user_texts, bot_texts)
    episode.updated_at = utc_now()
    return episode


def _active_topic(user_texts: list[str]) -> str | None:
    topics = ("cars", "gym", "music", "food", "uni", "work", "football", "movies")
    joined = normalize_text(" ".join(user_texts[-6:]))
    for topic in topics:
        if topic in joined:
            return topic
    return None


def _emotional_arc(normalized: str) -> str | None:
    if any(term in normalized for term in ("tired", "drained", "exhausted")):
        return "user tired"
    if any(term in normalized for term in ("annoyed", "repeating", "u said that", "missing context", "dry")):
        return "user annoyed by bot mistakes"
    if any(term in normalized for term in ("missed u", "love u", "baby")):
        return "affection/flirty"
    return None


def _summary_text(episode: ThreadEpisode, user_texts: list[str], bot_texts: list[str]) -> str:
    parts = [
        f"Thread has {len(episode.turns)} turns.",
        f"Relationship {episode.relationship_type}.",
    ]
    if user_texts:
        parts.append(f"Recent user: {' | '.join(user_texts[-3:])}.")
    if bot_texts:
        parts.append(f"Recent bot: {' | '.join(bot_texts[-3:])}.")
    if episode.active_topic:
        parts.append(f"Active topic: {episode.active_topic}.")
    if episode.emotional_arc:
        parts.append(f"Emotional arc: {episode.emotional_arc}.")
    if episode.bot_mistakes:
        parts.append(f"Bot mistakes: {', '.join(episode.bot_mistakes)}.")
    if episode.failed_reply_patterns:
        parts.append(f"Avoid: {', '.join(episode.failed_reply_patterns)}.")
    if episode.unresolved_user_points:
        parts.append(f"Unresolved: {', '.join(episode.unresolved_user_points)}.")
    return " ".join(parts)


def episode_embedding_text(episode: ThreadEpisode) -> str:
    turns = " | ".join(f"{turn.role}: {turn.text}" for turn in episode.turns[-8:])
    return (
        f"Thread summary: {episode.summary}\n"
        f"Recent turns: {turns}\n"
        f"Active topic: {episode.active_topic or ''}\n"
        f"Unresolved user points: {', '.join(episode.unresolved_user_points)}\n"
        f"Bot mistakes: {', '.join(episode.bot_mistakes)}\n"
        f"Successful patterns: {', '.join(episode.successful_reply_patterns)}\n"
        f"Failed patterns: {', '.join(episode.failed_reply_patterns)}\n"
        f"Relationship {episode.relationship_type}. Platform {episode.platform}. "
        f"Scene types {', '.join(episode.dominant_scene_types)}. "
        f"Latest required reply move {episode.latest_required_reply_move or ''}."
    )


def episode_to_index_row(episode: ThreadEpisode) -> dict[str, Any]:
    digest = hashlib.sha1(f"{episode.thread_id}|{episode.updated_at}".encode("utf-8")).hexdigest()[:12]
    incoming = episode.summary or episode_embedding_text(episode)
    reply = "; ".join(episode.successful_reply_patterns or episode.failed_reply_patterns or ["thread memory"])
    return {
        "id": f"thread_episode_{digest}",
        "thread_id": episode.thread_id,
        "session_id": episode.session_id,
        "memory_type": "thread_episode",
        "incoming": incoming,
        "my_reply": reply,
        "context": [f"{turn.role}: {turn.text}" for turn in episode.turns[-8:]],
        "relationship_type": episode.relationship_type,
        "intent_type": episode.latest_scene_type or "thread_episode",
        "platform": episode.platform,
        "contact_name": episode.contact_alias or "",
        "contact_alias": episode.contact_alias or "",
        "source": episode.source,
        "_source": episode.source,
        "style_authority": episode.style_authority,
        "latest_scene_type": episode.latest_scene_type or "",
        "latest_required_reply_move": episode.latest_required_reply_move or "",
        "dominant_scene_types": ",".join(episode.dominant_scene_types),
        "active_topic": episode.active_topic or "",
        "contains_sensitive": episode.contains_sensitive,
        "is_quarantined": episode.quarantined,
        "unsafe": episode.contains_sensitive or episode.quarantined,
        "summary": episode.summary,
        "embedding_text": episode_embedding_text(episode),
    }


def load_indexable_thread_rows(memory_dir: Path = THREAD_MEMORY_DIR) -> list[dict[str, Any]]:
    store = ThreadMemoryStore(memory_dir)
    rows: list[dict[str, Any]] = []
    for episode in store.load_episodes():
        if episode.quarantined or episode.contains_sensitive:
            continue
        rows.append(episode_to_index_row(episode))
    return rows


class ThreadMemoryStore:
    def __init__(self, memory_dir: Path = THREAD_MEMORY_DIR) -> None:
        self.memory_dir = memory_dir
        self.episodes_path = memory_dir / THREAD_EPISODES_FILE
        self.feedback_path = memory_dir / THREAD_FEEDBACK_FILE
        self.summaries_path = memory_dir / THREAD_SUMMARIES_FILE
        self._episodes_cache: list[ThreadEpisode] | None = None
        self.ensure_files()

    def ensure_files(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        for path in (self.episodes_path, self.feedback_path, self.summaries_path):
            path.touch(exist_ok=True)

    def load_episodes(self) -> list[ThreadEpisode]:
        if self._episodes_cache is not None:
            return self._episodes_cache
        episodes: list[ThreadEpisode] = []
        for line in self.episodes_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                episode = ThreadEpisode.from_dict(payload)
                if episode.thread_id:
                    episodes.append(episode)
        self._episodes_cache = episodes
        return episodes

    def get_episode(self, thread_id: str) -> ThreadEpisode | None:
        for episode in reversed(self.load_episodes()):
            if episode.thread_id == thread_id:
                return episode
        return None

    def get_or_create(
        self,
        thread_id: str,
        *,
        session_id: str,
        platform: str = "catbot",
        contact_alias: str | None = None,
        relationship_type: str = "unknown",
        source: str = "catbot_thread",
    ) -> ThreadEpisode:
        episode = self.get_episode(thread_id)
        if episode is not None:
            return episode
        now = utc_now()
        return ThreadEpisode(
            thread_id=thread_id,
            session_id=session_id,
            platform=platform,
            contact_alias=contact_alias,
            relationship_type=normalize_relationship_type(relationship_type),
            started_at=now,
            updated_at=now,
            source=source,
        )

    def append_turn(
        self,
        thread_id: str,
        *,
        session_id: str,
        role: str,
        text: str,
        platform: str = "catbot",
        contact_alias: str | None = None,
        relationship_type: str = "unknown",
        candidate_metadata: dict[str, Any] | None = None,
    ) -> ThreadEpisode:
        episode = self.get_or_create(
            thread_id,
            session_id=session_id,
            platform=platform,
            contact_alias=contact_alias,
            relationship_type=relationship_type,
        )
        safe_text = _safe_thread_text(text)
        turn = ThreadTurn(
            role="bot" if role == "bot" else "user",
            text=safe_text,
            timestamp=utc_now(),
            candidate_metadata=candidate_metadata,
            rejection_reasons=[str(item) for item in (candidate_metadata or {}).get("risk_flags", [])] if candidate_metadata else [],
        )
        episode.turns.append(turn)
        summarize_episode(episode)
        if platform == "catbot" and len(episode.turns) % 20 != 0:
            self._cache_episode(episode)
        else:
            self.save_episode(episode)
        return episode

    def record_feedback(
        self,
        thread_id: str,
        session_id: str,
        feedback: str,
        correction: str | None = None,
        failure_context: dict[str, Any] | None = None,
    ) -> ThreadEpisode | None:
        episode = self.get_episode(thread_id)
        if episode is None:
            return None
        feedback = str(feedback or "").strip().lower()
        failure_context = failure_context or {}
        selected_candidate = (
            failure_context.get("selected_candidate_metadata")
            if isinstance(failure_context.get("selected_candidate_metadata"), dict)
            else {}
        )
        incoming = str(failure_context.get("incoming") or "")
        context_value = failure_context.get("context")
        context = [str(item) for item in context_value] if isinstance(context_value, list) else []
        selected_reply = str(failure_context.get("selected_candidate") or "")
        failure_tags: list[str] = []
        if feedback == "thumbs_down":
            failure_tags = _derive_failure_tags(
                selected_reply=selected_reply,
                selected_candidate=selected_candidate,
                incoming=incoming,
                context=context,
                correction=correction,
            )
        for turn in reversed(episode.turns):
            if turn.role == "bot":
                turn.feedback = feedback
                turn.correction = _safe_thread_text(correction or "") or None
                if failure_tags:
                    turn.rejection_reasons = sorted(set([*turn.rejection_reasons, *failure_tags]))
                break
        if failure_tags:
            episode.bot_mistakes = sorted(set([*episode.bot_mistakes, *failure_tags]))
            episode.failed_reply_patterns = sorted(set([*episode.failed_reply_patterns, *failure_tags]))
        summarize_episode(episode)
        self.save_episode(episode)
        candidate_list = failure_context.get("candidates") if isinstance(failure_context.get("candidates"), list) else []
        feedback_row = {
            "thread_id": thread_id,
            "session_id": session_id,
            "feedback": feedback,
            "correction": _safe_thread_text(correction or ""),
            "timestamp": utc_now(),
            "failure_tags": failure_tags,
            "incoming": _safe_thread_text(incoming),
            "context": [_safe_thread_text(item) for item in context[-12:]],
            "selected_candidate": _safe_thread_text(selected_reply),
            "selected_candidate_metadata": _candidate_feedback_snapshot(selected_candidate) if selected_candidate else {},
            "candidate_shortlist": [
                _candidate_feedback_snapshot(item)
                for item in candidate_list[:8]
                if isinstance(item, dict)
            ],
            "scene_type": selected_candidate.get("scene_type") if selected_candidate else "",
            "required_reply_move": selected_candidate.get("required_reply_move") if selected_candidate else "",
            "thread_summary": episode.summary,
        }
        self._append_jsonl(self.feedback_path, feedback_row)
        return episode

    def save_episode(self, episode: ThreadEpisode) -> None:
        episodes = self._cache_episode(episode)
        self._write_jsonl_atomic(self.episodes_path, [item.to_dict() for item in episodes])
        self._write_jsonl_atomic(
            self.summaries_path,
            [
                {
                    "thread_id": item.thread_id,
                    "session_id": item.session_id,
                    "updated_at": item.updated_at,
                    "summary": item.summary,
                    "active_topic": item.active_topic,
                    "bot_mistakes": item.bot_mistakes,
                    "failed_reply_patterns": item.failed_reply_patterns,
                    "contains_sensitive": item.contains_sensitive,
                    "quarantined": item.quarantined,
                }
                for item in episodes
            ],
        )

    def _cache_episode(self, episode: ThreadEpisode) -> list[ThreadEpisode]:
        episodes = [item for item in self.load_episodes() if item.thread_id != episode.thread_id]
        episodes.append(episode)
        self._episodes_cache = episodes
        return episodes

    def retrieve_similar(
        self,
        *,
        query_text: str,
        relationship_type: str,
        scene_type: str = "",
        required_reply_move: str = "",
        platform: str = "catbot",
        limit: int = 3,
        flirt_allowed: bool = False,
    ) -> list[dict[str, Any]]:
        relationship_type = normalize_relationship_type(relationship_type)
        query_tokens = _tokens(query_text)
        scored: list[tuple[float, ThreadEpisode]] = []
        for episode in self.load_episodes():
            if episode.quarantined or episode.contains_sensitive:
                continue
            if episode.relationship_type == "romantic_interest" and not flirt_allowed:
                continue
            if relationship_type in {"professional", "university"} and episode.relationship_type not in {relationship_type, "professional", "university"}:
                continue
            episode_text = episode_embedding_text(episode)
            overlap = len(query_tokens & _tokens(episode_text))
            score = overlap * 0.08
            if episode.relationship_type == relationship_type:
                score += 0.3
            if platform and episode.platform == platform:
                score += 0.12
            if scene_type and scene_type in episode.dominant_scene_types:
                score += 0.3
            if required_reply_move and episode.latest_required_reply_move == required_reply_move:
                score += 0.25
            if episode.failed_reply_patterns:
                score += 0.08
            if score <= 0:
                continue
            scored.append((score, episode))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                **episode.to_dict(),
                "match_score": round(score, 3),
                "embedding_text": episode_embedding_text(episode),
            }
            for score, episode in scored[:limit]
        ]

    def stats(self) -> dict[str, Any]:
        episodes = self.load_episodes()
        return {
            "thread_episode_count": len(episodes),
            "recent_thread_count": sum(1 for episode in episodes if episode.source == "catbot_thread"),
            "thread_memory_path": str(self.memory_dir),
            "thread_memory_enabled": True,
        }

    def _append_jsonl(self, path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    def _write_jsonl_atomic(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        tmp_path.replace(path)
