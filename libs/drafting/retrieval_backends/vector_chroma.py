from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from libs.drafting.embeddings import DEFAULT_EMBEDDING_MODEL, embed_text
import json
from libs.drafting.intents import normalize_intent_label
from libs.drafting.retrieval_backends.base import RetrievedExample, RetrievalBackend
from libs.drafting.retrieval_backends.lexical import LexicalRetrievalBackend, score_rows
from libs.drafting.training_data import normalize_relationship_type


@dataclass(slots=True)
class VectorIndexStats:
    rows_indexed: int = 0
    rows_skipped: int = 0


class VectorChromaRetrievalBackend(RetrievalBackend):
    def __init__(
        self,
        *,
        index_dir: Path,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        fallback_rows: list[dict[str, Any]] | None = None,
        min_score: float = 0.35,
        limit: int = 5,
        strict_greeting_filter: bool = True,
    ) -> None:
        self.index_dir = index_dir
        self.embedding_model = embedding_model
        self.min_score = min_score
        self.default_limit = limit
        self.strict_greeting_filter = strict_greeting_filter
        self._fallback = LexicalRetrievalBackend(fallback_rows or [])

    def build_index(self) -> None:
        return None

    def search(
        self,
        incoming: str,
        context: list[str],
        relationship_type: str,
        intent_type: str,
        contact_name: str | None = None,
        limit: int = 5,
    ) -> list[RetrievedExample]:
        relationship_type = normalize_relationship_type(relationship_type)
        intent_type = normalize_intent_label(intent_type)
        if not self.index_dir.exists():
            return self._fallback.search(
                incoming,
                context,
                relationship_type,
                intent_type,
                contact_name=contact_name,
                limit=limit,
            )

        fallback_index = self._load_fallback_index()
        if fallback_index:
            self._fallback = LexicalRetrievalBackend(fallback_index)
        try:
            import chromadb
        except Exception:
            return self._fallback.search(
                incoming,
                context,
                relationship_type,
                intent_type,
                contact_name=contact_name,
                limit=limit,
            )

        try:
            client = chromadb.PersistentClient(path=str(self.index_dir))
            collection = client.get_collection("reply_examples")
        except Exception:
            return self._fallback.search(
                incoming,
                context,
                relationship_type,
                intent_type,
                contact_name=contact_name,
                limit=limit,
            )

        query_text = f"incoming: {incoming}\ncontext: {' | '.join(context[-6:])}\nrelationship: {relationship_type}\nintent: {intent_type}\ncontact: {contact_name or ''}"
        try:
            query_embedding = embed_text(query_text, model_name=self.embedding_model)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=max(limit * 6, 20),
                include=["metadatas", "documents", "distances"],
            )
        except Exception:
            return self._fallback.search(
                incoming,
                context,
                relationship_type,
                intent_type,
                contact_name=contact_name,
                limit=limit,
            )

        rows: list[dict[str, Any]] = []
        vector_scores: dict[str, float] = {}
        documents = results.get("documents", [[]])[0] or []
        metadatas = results.get("metadatas", [[]])[0] or []
        distances = results.get("distances", [[]])[0] or []
        for document, metadata, distance in zip(documents, metadatas, distances):
            row = dict(metadata or {})
            row["incoming"] = row.get("incoming") or row.get("query_incoming") or ""
            row["my_reply"] = row.get("my_reply") or row.get("reply") or ""
            if document and not row["incoming"]:
                parts = str(document).split(" || ")
                for part in parts:
                    if part.startswith("incoming: "):
                        row["incoming"] = part[len("incoming: ") :]
                    elif part.startswith("reply: "):
                        row["my_reply"] = part[len("reply: ") :]
            row.setdefault("context", [])
            row.setdefault("relationship_type", relationship_type)
            row.setdefault("intent_type", intent_type)
            rows.append(row)
            if row.get("id"):
                vector_scores[str(row["id"])] = max(0.0, 1.0 - float(distance or 0.0))

        rows = [row for row in rows if self._metadata_allows(row, relationship_type, intent_type)]
        if not rows:
            return self._fallback.search(
                incoming,
                context,
                relationship_type,
                intent_type,
                contact_name=contact_name,
                limit=limit,
            )

        lexical_ranked = score_rows(
            rows,
            incoming=incoming,
            context=context,
            relationship_type=relationship_type,
            intent_type=intent_type,
            contact_name=contact_name,
            limit=limit,
            backend="vector_chroma",
            vector_scores=vector_scores,
            filters_applied=[f"vector_db={self.index_dir.name}"],
        )
        if lexical_ranked and lexical_ranked[0].score >= self.min_score:
            return lexical_ranked

        fallback = self._fallback.search(
            incoming,
            context,
            relationship_type,
            intent_type,
            contact_name=contact_name,
            limit=limit,
        )
        return lexical_ranked or fallback

    def _load_fallback_index(self) -> list[dict[str, Any]]:
        path = self.index_dir / "reply_examples.jsonl"
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows

    def _metadata_allows(self, row: dict[str, Any], relationship_type: str, intent_type: str) -> bool:
        if row.get("is_quarantined") or row.get("unsafe"):
            return False
        row_relationship = normalize_relationship_type(str(row.get("relationship_type", "")))
        row_intent = normalize_intent_label(str(row.get("intent_type", "")))
        if row_relationship == "romantic_interest" and relationship_type != "romantic_interest":
            return False
        if intent_type == "greeting" and row_intent != "greeting":
            return False
        if relationship_type in {"professional", "university"} and row_relationship not in {relationship_type, "professional", "university"}:
            return False
        if relationship_type == "unknown" and row_relationship not in {"unknown", "casual_friend", "close_friend"}:
            return False
        return True
