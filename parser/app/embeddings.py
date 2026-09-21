"""Локальные HF-эмбеддинги с безопасным fallback для разработки."""
from __future__ import annotations

import hashlib
import logging
import math
import os
from typing import List

logger = logging.getLogger("parser.embeddings")

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
EMBED_DIM = 384

_model = None


def _hash_embedding(text: str) -> List[float]:
    # Только dev fallback: детерминированный вектор, не заменяет HF-эмбеддинги на финальном стенде.
    values = []
    for i in range(EMBED_DIM):
        digest = hashlib.sha256((text + "#" + str(i)).encode("utf-8")).digest()
        values.append((int.from_bytes(digest[:4], "big") / 2**32) * 2.0 - 1.0)
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def embed(text: str) -> List[float]:
    global _model
    try:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(MODEL_NAME)
            logger.info("Загружена локальная HF-модель эмбеддингов: %s", MODEL_NAME)
        vec = _model.encode([text[:8000]], normalize_embeddings=True)[0]
        return [float(x) for x in vec]
    except Exception as exc:  # noqa: BLE001
        if os.getenv("EMBEDDING_ALLOW_HASH_FALLBACK", "true").lower() == "true":
            logger.warning("HF-эмбеддинг недоступен, dev fallback hash: %s", exc)
            return _hash_embedding(text)
        raise
