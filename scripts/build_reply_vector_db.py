from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from libs.drafting.embeddings import embed_example_text, embed_texts
from libs.drafting.intents import classify_intent, contains_suspicious_phrase, normalize_intent_label
from libs.drafting.thread_memory import load_indexable_thread_rows
from libs.drafting.training_data import load_all_training_rows, load_corrections, normalize_relationship_type

TRAINING_DIR = Path("data/training_messages")
INDEX_DIR = Path("data/vector_db/reply_examples_chroma")
FALLBACK_INDEX = INDEX_DIR / "reply_examples.jsonl"
CONVERSATION_INTELLIGENCE_DB = Path("data/conversation_intelligence.db")

CATBOT_EXPLICIT_STYLE_BLOCKLIST = (
    "pussy",
    "dick",
    "cum",
    "slut",
    "rape",
    "force",
    "choke",
    "pin you",
    "pin u",
    "naked",
    "under me",
    "scream my name",
    "faggot",
    "retard",
    "khs",
    "kill myself",
    "suicidal",
    "self harm",
    "bdsm",
    "foreplay",
    "eat u",
    "inside u",
    "naked body",
    "fuck the shit",
    "suck mine",
)


def _safe_row(row: dict[str, Any]) -> bool:
    if row.get("memory_type") == "thread_episode":
        return not bool(row.get("is_quarantined") or row.get("unsafe") or row.get("contains_sensitive"))
    if row.get("memory_type") == "web_thread_note":
        return bool(str(row.get("incoming") or "").strip() and str(row.get("my_reply") or "").strip())
    if row.get("is_quarantined") or row.get("unsafe"):
        return False
    reply = str(row.get("my_reply") or row.get("user_final_reply") or "").strip()
    incoming = str(row.get("incoming", "")).strip()
    if not incoming or not reply:
        return False
    relationship = normalize_relationship_type(str(row.get("relationship_type", "")))
    if relationship == "romantic_interest":
        blob = f"{incoming} {reply}".casefold()
        if any(term in blob for term in CATBOT_EXPLICIT_STYLE_BLOCKLIST):
            return False
        return len(reply.split()) <= 26
    if contains_suspicious_phrase(reply) or contains_suspicious_phrase(incoming):
        return False
    return True


def _load_web_thread_note_rows(db_path: Path = CONVERSATION_INTELLIGENCE_DB) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    try:
        with sqlite3.connect(db_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT id, created_at, contact_name, relationship_type, question, answer, context_json, source
                FROM whatsapp_web_memory
                ORDER BY created_at DESC
                """
            ).fetchall()
    except sqlite3.Error:
        return []
    output: list[dict[str, Any]] = []
    for row in rows:
        question = str(row["question"] or "").strip()
        answer = str(row["answer"] or "").strip()
        if not question or not answer:
            continue
        output.append(
            {
                "id": str(row["id"]),
                "memory_type": "web_thread_note",
                "contact_name": str(row["contact_name"] or ""),
                "relationship_type": normalize_relationship_type(str(row["relationship_type"] or "unknown")),
                "intent_type": "thread_note",
                "incoming": question,
                "my_reply": answer,
                "summary": f"{question}: {answer}",
                "embedding_text": f"WhatsApp thread note for {row['contact_name']}: {question}: {answer}",
                "_source": "whatsapp_web_memory",
                "source": str(row["source"] or "whatsapp_web_extension"),
                "style_authority": "context_note",
                "created_at": str(row["created_at"] or ""),
                "context": [],
            }
        )
    return output


def _rows() -> tuple[list[dict[str, Any]], Counter[str]]:
    skipped: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for row in load_all_training_rows(TRAINING_DIR):
        if row.get("source") == "generated_safe_template" or row.get("is_synthetic"):
            continue
        row["_source"] = "training"
        if not _safe_row(row):
            skipped["unsafe_or_incomplete"] += 1
            continue
        rows.append(row)
    for row in load_corrections(TRAINING_DIR):
        item = {
            **row,
            "my_reply": row["user_final_reply"],
            "_source": "correction",
            "is_correction": True,
        }
        if not _safe_row(item):
            skipped["unsafe_or_incomplete"] += 1
            continue
        rows.append(item)
    generated = TRAINING_DIR / "generated_safe_templates.jsonl"
    if generated.exists():
        for line in generated.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            item["_source"] = "synthetic"
            if not _safe_row(item):
                skipped["unsafe_or_incomplete"] += 1
                continue
            rows.append(item)
    for index, row in enumerate(rows, start=1):
        context = row.get("context", [])
        if not isinstance(context, list):
            context = []
        row["context"] = [str(item) for item in context if str(item).strip()]
        row["relationship_type"] = normalize_relationship_type(str(row.get("relationship_type", "")))
        row["intent_type"] = normalize_intent_label(
            str(row.get("intent_type") or classify_intent(str(row.get("incoming", "")), row["context"]))
        )
        row["id"] = str(row.get("id") or f"reply_example_{index:06d}")
        row["is_quarantined"] = False
        row["unsafe"] = False
        row.setdefault("is_correction", row.get("_source") == "correction")
        row.setdefault("is_synthetic", row.get("_source") == "synthetic")
        row.setdefault("is_template", row.get("source") == "generated_safe_template")
        row.setdefault("memory_type", "reply_example")
    thread_rows = load_indexable_thread_rows(Path("data/thread_memory"))
    for row in thread_rows:
        if not _safe_row(row):
            skipped["unsafe_or_incomplete"] += 1
            continue
        rows.append(row)
    for row in _load_web_thread_note_rows():
        if not _safe_row(row):
            skipped["unsafe_or_incomplete"] += 1
            continue
        rows.append(row)
    return rows, skipped


def _metadata(row: dict[str, Any], source_file: str = "") -> dict[str, Any]:
    return {
        "id": row["id"],
        "contact_name": str(row.get("contact_name", "")),
        "relationship_type": row["relationship_type"],
        "intent_type": row["intent_type"],
        "incoming": str(row.get("incoming", "")),
        "my_reply": str(row.get("my_reply", "")),
        "source_file": source_file,
        "is_correction": bool(row.get("is_correction")),
        "is_synthetic": bool(row.get("is_synthetic")),
        "is_template": bool(row.get("is_template")),
        "is_quarantined": False,
        "unsafe": False,
        "_source": str(row.get("_source", "training")),
        "source": str(row.get("source", "")),
        "style_authority": str(row.get("style_authority", "")),
        "memory_type": str(row.get("memory_type") or "reply_example"),
        "thread_id": str(row.get("thread_id", "")),
        "platform": str(row.get("platform", "")),
        "contact_alias": str(row.get("contact_alias", "")),
        "latest_scene_type": str(row.get("latest_scene_type", "")),
        "latest_required_reply_move": str(row.get("latest_required_reply_move", "")),
        "dominant_scene_types": str(row.get("dominant_scene_types", "")),
        "active_topic": str(row.get("active_topic", "")),
        "contains_sensitive": bool(row.get("contains_sensitive")),
        "summary": str(row.get("summary", "")),
        "question": str(row.get("incoming", "")) if row.get("memory_type") == "web_thread_note" else "",
        "answer": str(row.get("my_reply", "")) if row.get("memory_type") == "web_thread_note" else "",
    }


def build_reply_vector_db() -> dict[str, Any]:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    rows, skipped = _rows()
    by_relationship = Counter(row["relationship_type"] for row in rows)
    by_intent = Counter(row["intent_type"] for row in rows)
    chroma_error = ""
    indexed = 0
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(INDEX_DIR))
        try:
            client.delete_collection("reply_examples")
        except Exception:
            pass
        collection = client.create_collection("reply_examples")
        batch_size = 250
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            documents = [str(row.get("embedding_text") or embed_example_text(row)) for row in batch]
            collection.add(
                ids=[row["id"] for row in batch],
                embeddings=embed_texts(documents),
                documents=documents,
                metadatas=[_metadata(row) for row in batch],
            )
            indexed += len(batch)
    except Exception as exc:
        chroma_error = str(exc)
    with FALLBACK_INDEX.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({**row, "metadata": _metadata(row)}, ensure_ascii=True) + "\n")
    return {
        "rows_loaded": len(rows) + sum(skipped.values()),
        "rows_skipped": sum(skipped.values()),
        "skipped_reasons": dict(skipped),
        "rows_indexed": indexed or len(rows),
        "reply_example_count": sum(1 for row in rows if row.get("memory_type", "reply_example") == "reply_example"),
        "thread_episode_count": sum(1 for row in rows if row.get("memory_type") == "thread_episode"),
        "web_thread_note_count": sum(1 for row in rows if row.get("memory_type") == "web_thread_note"),
        "total_vector_count": indexed or len(rows),
        "fallback_jsonl_written": FALLBACK_INDEX.exists(),
        "counts_by_relationship_type": dict(sorted(by_relationship.items())),
        "counts_by_intent_type": dict(sorted(by_intent.items())),
        "corrections_indexed": sum(1 for row in rows if row.get("is_correction")),
        "synthetic_template_indexed": sum(1 for row in rows if row.get("is_synthetic") or row.get("is_template")),
        "vector_db_path": str(INDEX_DIR),
        "dependency_model_errors": chroma_error or "none",
    }


def main() -> None:
    result = build_reply_vector_db()
    print(f"rows_loaded={result['rows_loaded']}")
    print(f"rows_skipped={result['rows_skipped']} skipped_reasons={result['skipped_reasons']}")
    print(f"rows_indexed={result['rows_indexed']}")
    print(f"counts_by_relationship_type={result['counts_by_relationship_type']}")
    print(f"counts_by_intent_type={result['counts_by_intent_type']}")
    print(f"corrections_indexed={result['corrections_indexed']}")
    print(f"synthetic_template_indexed={result['synthetic_template_indexed']}")
    print(f"vector_db_path={result['vector_db_path']}")
    print(f"dependency_model_errors={result['dependency_model_errors']}")


if __name__ == "__main__":
    main()
