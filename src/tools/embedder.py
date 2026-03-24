"""
src/tools/embedder.py
---------------------
Singleton wrapper around bge-large-en (1024 dims).
Used by:
  - evaluator: Signal 1 cosine similarity, near-duplicate detection
  (Index embeddings are produced by Meilisearch's HuggingFace embedder via setup_index.)

Design:
  - Model loaded once at import time (heavy — ~1.3GB), reused everywhere.
  - Query vector normalised once per evaluator node entry, not per comparison.
  - Cosine similarity uses numpy dot product (BLAS-accelerated).
"""
from __future__ import annotations

import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """
    Load bge-large-en once, cache for the process lifetime.
    First call takes ~3–5s (model load). All subsequent calls are instant.
    """
    logger.info("loading_embedding_model", extra={"model": settings.embedding_model})
    model = SentenceTransformer(settings.embedding_model)
    logger.info("embedding_model_ready", extra={"model": settings.embedding_model})
    return model


def embed_query(text: str) -> np.ndarray:
    """
    Embed a single query string.
    Returns a normalised 1024-dim float32 array.
    Normalisation here means cosine similarity = dot product downstream.
    """
    model = get_embedder()
    vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return vec.astype(np.float32)


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """
    Embed multiple texts in batches.
    Returns (N, 1024) float32 array, each row L2-normalised.
    Used by seed script and evaluator Signal 1 (top-3 doc embeddings).
    """
    model = get_embedder()
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vecs.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two pre-normalised vectors.
    Since both vectors are L2-normalised: cosine_sim = dot(a, b).

    Both embed_query() and embed_texts() return normalised vectors,
    so normalisation here is a no-op safety guard only.

    BLAS-accelerated via numpy — ~100x faster than a Python loop
    for 1024-dim embeddings. Critical for P95 < 200ms budget.
    """
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a_norm, b_norm))


def cosine_similarity_batch(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """
    Cosine similarity of one query vector against N document vectors.
    query_vec: (1024,) normalised
    doc_vecs:  (N, 1024) normalised
    Returns:   (N,) float32 array of similarity scores.
    """
    # Normalise once — both already normalised but guard against drift
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    d = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)
    return (d @ q).astype(np.float32)