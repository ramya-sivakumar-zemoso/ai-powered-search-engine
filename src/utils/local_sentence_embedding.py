"""In-process SentenceTransformer embeddings for Nomic v2 MoE + Meilisearch ``userProvided`` embedders.

Used when ``EMBEDDER_SOURCE=local`` and ``EMBEDDING_MODEL`` is Nomic (MoE is not supported by
Meilisearch's built-in ``huggingFace`` runner). We embed in Python, store ``_vectors`` at index
time, and pass ``vector`` on hybrid/semantic search. Other local models (e.g. MiniLM, E5) use
Meilisearch's ``huggingFace`` embedder instead — see ``setup_index.configure_embedder``.
"""
from __future__ import annotations

import os
import re
import threading
from typing import Any

import numpy as np

from src.models.dataset_schema import DatasetSchema
from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.utils.triton_cpu_shim import ensure_triton_cpu_import_safe

logger = get_logger(__name__)

_NOMIC_MATRYOSHKA_DIMS = frozenset({256, 512})
_DOC_FIELD_RE = re.compile(r"\{\{doc\.(\w+)\}\}")

_model_lock = threading.Lock()
_model: Any = None


def is_nomic_embedding_model(model_id: str) -> bool:
    m = model_id.lower()
    return "nomic" in m or "nomic-embed" in m


def uses_python_sentence_embeddings() -> bool:
    s = get_settings()
    return s.embedder_source == "local" and is_nomic_embedding_model(s.embedding_model)


def render_meili_document_template(template: str, doc: dict[str, Any]) -> str:
    """Minimal subset of Meilisearch ``documentTemplate``: only ``{{doc.field}}`` placeholders."""

    def _repl(match: re.Match[str]) -> str:
        return str(doc.get(match.group(1), "") or "")

    return _DOC_FIELD_RE.sub(_repl, template).strip()


def _nomic_truncate_kwarg(embedding_dimensions: int) -> dict[str, int]:
    if embedding_dimensions == 768:
        return {}
    if embedding_dimensions in _NOMIC_MATRYOSHKA_DIMS:
        return {"truncate_dim": embedding_dimensions}
    raise ValueError(
        "For Nomic v2, EMBEDDING_DIMENSIONS must be 768 (full), 512, or 256; "
        f"got {embedding_dimensions}."
    )


def _configure_torch_for_speed(device: str) -> None:
    try:
        import torch

        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if device != "cpu":
        return
    try:
        import torch

        n = os.cpu_count() or 1
        try:
            torch.set_num_threads(n)
            torch.set_num_interop_threads(max(1, min(4, n // 4 or 1)))
        except Exception:
            torch.set_num_threads(n)
    except Exception:
        pass


def _default_encode_batch_size(device: str) -> int:
    raw = (os.getenv("EMBEDDING_ENCODE_BATCH_SIZE") or os.getenv("EMBED_SERVER_BATCH_SIZE") or "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return 128 if device == "cuda" else 32


def get_sentence_embedding_model() -> Any:
    global _model
    if not uses_python_sentence_embeddings():
        raise RuntimeError(
            "Local SentenceTransformer embedder is only used when "
            "EMBEDDER_SOURCE=local and EMBEDDING_MODEL is a Nomic model."
        )
    with _model_lock:
        if _model is None:
            ensure_triton_cpu_import_safe()
            import torch
            from sentence_transformers import SentenceTransformer

            settings = get_settings()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            override = (os.getenv("EMBEDDING_DEVICE") or "").strip().lower()
            if override in ("cpu", "cuda"):
                device = override
            _configure_torch_for_speed(device)
            extra = _nomic_truncate_kwarg(settings.embedding_dimensions)
            logger.info(
                "loading_local_nomic_embedder",
                extra={"model": settings.embedding_model, "device": device, "dims": settings.embedding_dimensions},
            )
            _model = SentenceTransformer(
                settings.embedding_model,
                trust_remote_code=True,
                device=device,
                **extra,
            )
        return _model


def warmup_local_embedder_if_needed() -> None:
    if not uses_python_sentence_embeddings():
        return
    try:
        m = get_sentence_embedding_model()
        bs = _default_encode_batch_size(str(m.device))
        m.encode(["warmup"], prompt_name="query", normalize_embeddings=True, batch_size=1, show_progress_bar=False)
        m.encode(["warmup"], prompt_name="passage", normalize_embeddings=True, batch_size=1, show_progress_bar=False)
        logger.info("local_nomic_embedder_warmup_ok", extra={"encode_batch_size": bs})
    except Exception as exc:
        logger.warning("local_nomic_embedder_warmup_failed", extra={"error": str(exc)[:200]})


def _encode_passage_batch(model: Any, texts: list[str]) -> np.ndarray:
    device = str(model.device)
    bs = _default_encode_batch_size(device)
    return model.encode(
        texts,
        prompt_name="passage",
        normalize_embeddings=True,
        batch_size=min(len(texts), bs) if texts else 1,
        show_progress_bar=False,
    )


def attach_document_vectors_for_meili(
    documents: list[dict[str, Any]],
    schema: DatasetSchema,
) -> list[dict[str, Any]]:
    """Return copies of documents with ``_vectors`` set when using local Nomic + ``userProvided``."""
    if not documents or not uses_python_sentence_embeddings():
        return documents
    settings = get_settings()
    name = settings.meili_embedder_name
    model = get_sentence_embedding_model()
    texts = [render_meili_document_template(schema.embedder_template, d) for d in documents]
    device = str(model.device)
    chunk_sz = _default_encode_batch_size(device)

    all_vecs: list[Any] = []
    for i in range(0, len(texts), chunk_sz):
        chunk = texts[i : i + chunk_sz]
        vecs = _encode_passage_batch(model, chunk)
        all_vecs.extend(list(vecs))

    out: list[dict[str, Any]] = []
    for doc, vec in zip(documents, all_vecs):
        row = dict(doc)
        arr = np.asarray(vec)
        row["_vectors"] = {name: arr.tolist()}
        out.append(row)
    return out


def encode_query_vector(query: str) -> list[float]:
    """Embedding for hybrid/semantic search (Nomic ``prompt_name=query``)."""
    model = get_sentence_embedding_model()
    q = (query or "").strip() or " "
    v = model.encode(
        [q],
        prompt_name="query",
        normalize_embeddings=True,
        batch_size=1,
        show_progress_bar=False,
    )[0]
    return np.asarray(v).tolist()
