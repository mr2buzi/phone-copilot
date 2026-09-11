from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from typing import Any

from libs.drafting.intents import normalize_text

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=2)
def _load_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_text(text: str, *, model_name: str = DEFAULT_EMBEDDING_MODEL) -> list[float]:
    try:
        model = _load_model(model_name)
        vector = model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        return [float(value) for value in vector.tolist()]
    except Exception:
        return hash_embed_text(text)


def embed_texts(texts: list[str], *, model_name: str = DEFAULT_EMBEDDING_MODEL) -> list[list[float]]:
    try:
        model = _load_model(model_name)
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
        return [[float(value) for value in vector.tolist()] for vector in vectors]
    except Exception:
        return [hash_embed_text(text) for text in texts]


def hash_embed_text(text: str, *, dimensions: int = 384) -> list[float]:
    vector = [0.0] * dimensions
    tokens = normalize_text(text).split()
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def embed_example(example: Any, *, model_name: str = DEFAULT_EMBEDDING_MODEL) -> list[float]:
    return embed_text(embed_example_text(example), model_name=model_name)


def embed_example_text(example: Any) -> str:
    if isinstance(example, dict):
        incoming = str(example.get("incoming", "")).strip()
        context = " | ".join(str(item) for item in example.get("context", []) if str(item).strip())
        reply = str(example.get("my_reply") or example.get("user_final_reply") or "").strip()
        relationship_type = str(example.get("relationship_type", "unknown")).strip()
        intent_type = str(example.get("intent_type", "unknown")).strip()
        contact_name = str(example.get("contact_name", "")).strip()
    else:
        incoming = str(getattr(example, "incoming", "")).strip()
        context = " | ".join(str(item) for item in getattr(example, "context", []) if str(item).strip())
        reply = str(getattr(example, "my_reply", "")).strip()
        relationship_type = str(getattr(example, "relationship_type", "unknown")).strip()
        intent_type = str(getattr(example, "intent_type", "unknown")).strip()
        contact_name = str(getattr(example, "contact_name", "")).strip()
    parts = [
        f"incoming: {normalize_text(incoming)}",
        f"context: {normalize_text(context)}" if context else "",
        f"reply: {normalize_text(reply)}",
        f"relationship: {normalize_text(relationship_type)}",
        f"intent: {normalize_text(intent_type)}",
        f"contact: {normalize_text(contact_name)}" if contact_name else "",
    ]
    return " || ".join(part for part in parts if part)
