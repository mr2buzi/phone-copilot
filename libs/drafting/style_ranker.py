from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re
from typing import Any

from libs.drafting.intents import normalize_text
from libs.drafting.training_data import normalize_relationship_type


GENERIC_BAD_PHRASES = (
    "ok",
    "same icl",
    "same just chilling",
    "fair just chilling too",
    "what u saying then",
    "fine then what should we talk about",
    "u ain't giving me much",
)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z']+", normalize_text(text))


@dataclass
class TrainingStyleRanker:
    reply_starters: Counter[str] = field(default_factory=Counter)
    reply_tokens: Counter[str] = field(default_factory=Counter)
    relationship_tokens: dict[str, Counter[str]] = field(default_factory=dict)
    average_words: float = 8.0
    examples_loaded: int = 0

    @classmethod
    def from_examples(cls, rows: list[dict[str, Any]]) -> "TrainingStyleRanker":
        starters: Counter[str] = Counter()
        tokens: Counter[str] = Counter()
        relationship_tokens: dict[str, Counter[str]] = {}
        lengths: list[int] = []
        count = 0
        for row in rows:
            reply = str(row.get("my_reply") or row.get("user_final_reply") or "").strip()
            if not reply:
                continue
            row_tokens = _tokens(reply)
            if not row_tokens:
                continue
            relationship = normalize_relationship_type(str(row.get("relationship_type") or "unknown"))
            starters[row_tokens[0]] += 1
            tokens.update(row_tokens)
            relationship_tokens.setdefault(relationship, Counter()).update(row_tokens)
            lengths.append(len(reply.split()))
            count += 1
        return cls(
            reply_starters=starters,
            reply_tokens=tokens,
            relationship_tokens=relationship_tokens,
            average_words=sum(lengths) / max(len(lengths), 1),
            examples_loaded=count,
        )

    def score(self, reply: str, *, relationship_type: str = "unknown") -> dict[str, object]:
        normalized = normalize_text(reply).strip(" .?!")
        words = _tokens(reply)
        if not words:
            return {
                "training_style_score": 0.0,
                "training_style_penalty": 1.0,
                "training_style_reasons": ["empty_reply"],
                "training_style_examples_loaded": self.examples_loaded,
            }
        reasons: list[str] = []
        score = 0.45
        if words[0] in self.reply_starters:
            score += min(0.18, self.reply_starters[words[0]] / max(sum(self.reply_starters.values()), 1) * 8)
            reasons.append("known_reply_starter")
        token_hits = sum(1 for token in words if token in self.reply_tokens)
        score += min(0.24, token_hits / max(len(words), 1) * 0.24)
        if token_hits:
            reasons.append("known_style_tokens")
        rel = normalize_relationship_type(relationship_type)
        rel_tokens = self.relationship_tokens.get(rel, Counter())
        if rel_tokens:
            rel_hits = sum(1 for token in words if token in rel_tokens)
            score += min(0.14, rel_hits / max(len(words), 1) * 0.14)
            if rel_hits:
                reasons.append("relationship_style_tokens")
        word_count = len(reply.split())
        if 2 <= word_count <= max(5, int(self.average_words * 2.2)):
            score += 0.1
            reasons.append("length_matches_training")
        if any(marker in normalized for marker in (" u", " ur", "icl", "lowk", "yh", "nah", "bro", "lool", "wbu", "wby")):
            score += 0.08
            reasons.append("casual_owner_marker")
        if any(phrase in normalized for phrase in GENERIC_BAD_PHRASES):
            score -= 0.45
            reasons.append("generic_bad_phrase")
        penalty = max(0.0, min(1.0, 1.0 - score))
        return {
            "training_style_score": round(max(0.0, min(1.0, score)), 3),
            "training_style_penalty": round(penalty, 3),
            "training_style_reasons": sorted(dict.fromkeys(reasons)),
            "training_style_examples_loaded": self.examples_loaded,
        }
