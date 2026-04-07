"""Real-time ingest API — per-listing event-driven path (PRD: 5-minute ingest SLA).

Run:
    uvicorn src.tools.ingest_api:app --host 0.0.0.0 --port 8001

Endpoints
---------
POST /ingest/document   — upsert a single normalised document; blocks until indexed.
POST /ingest/batch      — upsert a small batch (≤ 100 docs); blocks until indexed.
DELETE /ingest/{id}     — hard-delete a document by id.
GET  /ingest/health     — readiness check (Meilisearch + SLA config).

Design notes
------------
- Idempotent: Meilisearch uses the ``id`` field as the primary key; re-sending the
  same payload updates in-place without duplication (safe for at-least-once streams).
- Soft-delete (tombstone): include ``deleted_at`` (ISO-8601) in the document payload
  and filter on ``deleted_at IS NULL`` at query time; this is the preferred pattern
  for streaming deletions with eventual visibility guarantees.
- SLA tracking: every response includes ``elapsed_seconds`` so callers and
  monitoring can assert that indexing (including embedder task completion) stays
  within the 5-minute contract.
- Streaming integration: this endpoint is the application-side webhook receiver for
  any upstream event bus (Kafka consumer, CDC hook, queue worker).  The caller is
  responsible for normalising the raw event into the ``IngestDocument`` schema before
  posting; domain-specific mapping lives in ``DatasetSchema.field_mappings``.
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.models.schema_registry import get_schema
from src.tools.meilisearch_client import (
    delete_document,
    health,
    upsert_documents,
)
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

app = FastAPI(
    title="AI Search — Real-time Ingest API",
    description=(
        "Event-driven per-listing ingest endpoint. "
        "Guarantees Meilisearch visibility within 5 minutes of document receipt."
    ),
    version="1.0.0",
)

_BATCH_LIMIT = 100
_SLA_SECONDS = 300


# ─────────────────────────────────────────────────────────────────────────────
#  Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class IngestDocument(BaseModel):
    """A single normalised document ready for indexing.

    The ``id`` field is mandatory and used as the Meilisearch primary key.
    All other fields map to the active ``DatasetSchema`` (title, description,
    category, brand, …).  Extra fields not in the schema are stored but not
    searched unless ``searchable_fields`` is updated.

    Tombstone pattern: set ``deleted_at`` to an ISO-8601 timestamp to mark a
    document as deleted without hard-removing it from the index.  Filter
    ``deleted_at IS NULL`` in search queries to hide deleted listings.
    """
    id: str = Field(..., description="Unique document identifier (primary key).")
    schema_name: str | None = Field(
        default=None,
        description="Override active DatasetSchema. Defaults to DATASET_SCHEMA env var.",
    )
    payload: dict[str, Any] = Field(
        ...,
        description="Normalised document fields (title, description, category, …).",
    )


class IngestBatchRequest(BaseModel):
    documents: list[IngestDocument] = Field(
        ..., max_length=_BATCH_LIMIT,
        description=f"Up to {_BATCH_LIMIT} documents per batch call.",
    )


class IngestResponse(BaseModel):
    task_uid: int | None
    status: str
    elapsed_seconds: float
    document_count: int
    sla_ok: bool
    index: str


class DeleteResponse(BaseModel):
    task_uid: int | None
    status: str
    elapsed_seconds: float
    document_id: str


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_meili_doc(doc: IngestDocument) -> dict[str, Any]:
    """Merge ``id`` + ``schema_name`` + ``payload`` into a flat Meilisearch document."""
    schema_name = doc.schema_name or settings.dataset_schema
    schema = get_schema(schema_name)
    flat = {"id": doc.id}
    flat.update(doc.payload)
    # Stamp ingest time so freshness tracking works end-to-end.
    if "indexed_at" not in flat:
        from datetime import datetime, timezone
        flat["indexed_at"] = datetime.now(timezone.utc).isoformat()
    if not hasattr(schema, "apply"):
        return flat
    # DatasetSchema.apply(raw, row_index) — row_index only affects fallback id / indexed_at jitter.
    applied = schema.apply(flat, 0)
    if applied is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Document rejected by schema rules (e.g. title shorter than 3 characters). "
                "Adjust payload or schema field_mappings."
            ),
        )
    return applied


# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/ingest/health", tags=["ops"])
def ingest_health() -> dict[str, Any]:
    """Readiness probe — confirms Meilisearch is reachable and shows SLA config."""
    try:
        meili_status = health()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Meilisearch unreachable: {exc}",
        ) from exc
    return {
        "status": "ok",
        "meilisearch": meili_status,
        "sla_seconds": _SLA_SECONDS,
        "batch_limit": _BATCH_LIMIT,
        "active_index": settings.meili_index_name,
        "active_schema": settings.dataset_schema,
    }


@app.post("/ingest/document", response_model=IngestResponse, tags=["ingest"])
def ingest_document(doc: IngestDocument) -> IngestResponse:
    """Upsert a single document and block until Meilisearch has indexed it.

    - Idempotent: re-posting the same ``id`` performs an in-place update.
    - Raises 408 if indexing does not complete within 5 minutes (SLA breach).
    - Raises 502 if Meilisearch rejects the request.
    """
    idx = settings.meili_index_name
    meili_doc = _build_meili_doc(doc)
    logger.info("ingest_document_received", extra={"id": doc.id, "index": idx})

    t0 = time.monotonic()
    try:
        result = upsert_documents([meili_doc], index_name=idx, wait=True, sla_seconds=_SLA_SECONDS)
    except TimeoutError as exc:
        logger.error("ingest_sla_breach", extra={"id": doc.id, "elapsed": time.monotonic() - t0})
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    elapsed = result["elapsed_seconds"]
    return IngestResponse(
        task_uid=result["task_uid"],
        status=result["status"],
        elapsed_seconds=elapsed,
        document_count=1,
        sla_ok=elapsed <= _SLA_SECONDS,
        index=idx,
    )


@app.post("/ingest/batch", response_model=IngestResponse, tags=["ingest"])
def ingest_batch(request: IngestBatchRequest) -> IngestResponse:
    """Upsert a batch of documents (≤ 100) and block until indexed.

    Use this for micro-batches from a stream consumer (e.g. Kafka consumer
    flushing on a 1-second window).  Each document is still upserted
    idempotently via its ``id``.
    """
    idx = settings.meili_index_name
    meili_docs = [_build_meili_doc(d) for d in request.documents]
    count = len(meili_docs)
    logger.info("ingest_batch_received", extra={"count": count, "index": idx})

    t0 = time.monotonic()
    try:
        result = upsert_documents(meili_docs, index_name=idx, wait=True, sla_seconds=_SLA_SECONDS)
    except TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    elapsed = result["elapsed_seconds"]
    return IngestResponse(
        task_uid=result["task_uid"],
        status=result["status"],
        elapsed_seconds=elapsed,
        document_count=count,
        sla_ok=elapsed <= _SLA_SECONDS,
        index=idx,
    )


@app.delete("/ingest/{document_id}", response_model=DeleteResponse, tags=["ingest"])
def delete_listing(document_id: str) -> DeleteResponse:
    """Hard-delete a document by id.

    Prefer the tombstone pattern (upsert with ``deleted_at`` set) when you need
    audit trails or the deletion must be visible in LangWatch traces.
    """
    idx = settings.meili_index_name
    logger.info("ingest_delete_received", extra={"id": document_id, "index": idx})
    t0 = time.monotonic()
    try:
        result = delete_document(document_id, index_name=idx, wait=True)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    return DeleteResponse(
        task_uid=result.get("task_uid"),
        status=result.get("status", "unknown"),
        elapsed_seconds=round(time.monotonic() - t0, 3),
        document_id=document_id,
    )
