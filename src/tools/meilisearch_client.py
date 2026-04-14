"""Meilisearch client — search (with retries + hybrid fallback), stats, tasks.

Uses the official ``meilisearch`` Python SDK for all API calls.
Retry logic and hybrid-to-keyword fallback are layered on top because
the SDK does not provide these out of the box.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

import meilisearch
import requests
from meilisearch.errors import (
    MeilisearchApiError,
    MeilisearchCommunicationError,
    MeilisearchTimeoutError,
)

import src.constants as C
from src.models.schema_registry import get_schema
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Lazy-initialised SDK client (singleton)
_client: meilisearch.Client | None = None


def _get_client() -> meilisearch.Client:
    """Return a lazily-created Meilisearch SDK client."""
    global _client
    if _client is None:
        _client = meilisearch.Client(
            settings.meili_url,
            settings.meili_master_key,
        )
    return _client


def _norm(data: dict[str, Any], strategy: str, partial: bool) -> dict[str, Any]:
    """Normalise a Meilisearch search response into our internal shape."""
    return {
        "hits": data.get("hits", []),
        "latency_ms": data.get("processingTimeMs", 0),
        "estimated_total_hits": data.get("estimatedTotalHits", 0),
        "strategy_used": strategy,
        "partial": partial,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Health
# ══════════════════════════════════════════════════════════════════════════════

def health() -> dict[str, Any]:
    """Check Meilisearch availability."""
    return _get_client().health()


# ══════════════════════════════════════════════════════════════════════════════
#  Internal: execute search with 3-retry + keyword fallback
# ══════════════════════════════════════════════════════════════════════════════

_RETRYABLE = (
    MeilisearchCommunicationError,
    MeilisearchTimeoutError,
    MeilisearchApiError,
    ConnectionError,
    TimeoutError,
)


def _execute_search(
    query: str,
    opt_params: dict[str, Any],
    strategy_label: str,
    index_name: str | None = None,
) -> dict[str, Any]:
    """Run a search with up to 3 retries, falling back to keyword on failure."""
    idx = index_name or settings.meili_index_name
    index = _get_client().index(idx)
    last_error: Exception | None = None

    max_attempts = C.MEILI_SEARCH_MAX_ATTEMPTS
    for attempt in range(max_attempts):
        try:
            result = index.search(query, opt_params)
            return _norm(result, strategy_label, False)
        except _RETRYABLE as exc:
            last_error = exc
            if attempt < max_attempts - 1:
                wait = C.MEILI_SEARCH_RETRY_BACKOFF_BASE_S * (2 ** (attempt + 1))
                logger.warning(
                    "meili_retry",
                    extra={"attempt": attempt + 1, "wait_s": wait, "error": str(exc)},
                )
                time.sleep(wait)

    if strategy_label != "KEYWORD" and "hybrid" in opt_params:
        logger.warning(
            "meili_hybrid_fallback",
            extra={"original_strategy": strategy_label, "error": str(last_error)},
        )
        fallback_params = {k: v for k, v in opt_params.items() if k != "hybrid"}
        try:
            result = index.search(query, fallback_params)
            return _norm(result, "KEYWORD_FALLBACK", True)
        except Exception as fallback_exc:
            logger.error("meili_hard_failure", extra={"error": str(fallback_exc)})
            raise RuntimeError(
                f"Meilisearch failed after fallback: {fallback_exc}"
            ) from fallback_exc

    raise RuntimeError(
        f"Meilisearch search failed after {max_attempts} retries: {last_error}"
    ) from last_error


# ══════════════════════════════════════════════════════════════════════════════
#  Default retrieve fields
# ══════════════════════════════════════════════════════════════════════════════

def _retrieve_fields(fields: list[str] | None) -> list[str]:
    if fields:
        return fields
    schema = get_schema(settings.dataset_schema)
    return schema.meilisearch_attributes_to_retrieve()


# ══════════════════════════════════════════════════════════════════════════════
#  Public search functions
# ══════════════════════════════════════════════════════════════════════════════

def keyword_search(
    query: str,
    limit: int = 20,
    filters: str | None = None,
    retrieve_fields: list[str] | None = None,
    index_name: str | None = None,
) -> dict[str, Any]:
    opt_params: dict[str, Any] = {
        "limit": limit,
        "showRankingScore": True,
        "attributesToRetrieve": _retrieve_fields(retrieve_fields),
    }
    if filters:
        opt_params["filter"] = filters
    return _execute_search(query, opt_params, "KEYWORD", index_name)


def hybrid_search(
    query: str,
    semantic_ratio: float = 0.6,
    limit: int = 20,
    filters: str | None = None,
    retrieve_fields: list[str] | None = None,
    index_name: str | None = None,
) -> dict[str, Any]:
    opt_params: dict[str, Any] = {
        "limit": limit,
        "showRankingScore": True,
        "attributesToRetrieve": _retrieve_fields(retrieve_fields),
        "hybrid": {
            "embedder": settings.meili_embedder_name,
            "semanticRatio": semantic_ratio,
        },
    }
    if filters:
        opt_params["filter"] = filters
    return _execute_search(query, opt_params, "HYBRID", index_name)


def semantic_search(
    query: str,
    limit: int = 20,
    filters: str | None = None,
    retrieve_fields: list[str] | None = None,
    index_name: str | None = None,
) -> dict[str, Any]:
    return hybrid_search(
        query=query,
        semantic_ratio=1.0,
        limit=limit,
        filters=filters,
        retrieve_fields=retrieve_fields,
        index_name=index_name,
    )


def filtered_search(
    query: str,
    filters: str,
    limit: int = 20,
    retrieve_fields: list[str] | None = None,
    index_name: str | None = None,
) -> dict[str, Any]:
    return keyword_search(
        query=query,
        limit=limit,
        filters=filters,
        retrieve_fields=retrieve_fields,
        index_name=index_name,
    )


def search(
    query: str,
    strategy: str,
    hybrid_weights: dict[str, float] | None = None,
    filters: str | None = None,
    limit: int = 20,
    retrieve_fields: list[str] | None = None,
    index_name: str | None = None,
) -> dict[str, Any]:
    """Route to the appropriate search function based on strategy."""
    alpha = (hybrid_weights or {}).get("semanticRatio", 0.6)
    if strategy == "KEYWORD":
        return keyword_search(
            query, limit=limit, filters=filters,
            retrieve_fields=retrieve_fields, index_name=index_name,
        )
    if strategy == "SEMANTIC":
        return semantic_search(
            query, limit=limit, filters=filters,
            retrieve_fields=retrieve_fields, index_name=index_name,
        )
    if strategy == "HYBRID":
        return hybrid_search(
            query, semantic_ratio=alpha, limit=limit,
            filters=filters, retrieve_fields=retrieve_fields,
            index_name=index_name,
        )
    logger.warning("unknown_strategy_fallback", extra={"strategy": strategy})
    return keyword_search(
        query, limit=limit, filters=filters,
        retrieve_fields=retrieve_fields, index_name=index_name,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Index metadata (PRD §4.3 — index freshness monitor)
# ══════════════════════════════════════════════════════════════════════════════


def _parse_meili_iso_datetime(raw: str | None) -> datetime | None:
    if not raw:
        return None
    s = str(raw).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def get_index_updated_at_meta(index_name: str | None = None) -> tuple[datetime | None, bool]:
    """Fetch index ``updatedAt`` from Meilisearch.

    Returns:
        (parsed_utc_datetime, meta_ok). If ``meta_ok`` is False, callers should set
        ``freshness_unknown`` on the response (PRD §4.3).
    """
    idx = index_name or settings.meili_index_name
    base = settings.meili_url.rstrip("/")
    headers = {"Authorization": f"Bearer {settings.meili_master_key}"}
    try:
        r = requests.get(f"{base}/indexes/{idx}", headers=headers, timeout=5)
        r.raise_for_status()
        data = r.json()
        raw = data.get("updatedAt") or data.get("updated_at")
        return (_parse_meili_iso_datetime(raw), True)
    except Exception as exc:
        logger.warning(
            "index_meta_unavailable",
            extra={"index": idx, "error": str(exc)[:200]},
        )
        return (None, False)


# ══════════════════════════════════════════════════════════════════════════════
#  Index stats + task helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_index_stats(index_name: str | None = None) -> dict[str, Any]:
    idx = index_name or settings.meili_index_name
    try:
        data = _get_client().index(idx).get_stats()
        # SDK returns IndexStats (attrs); older responses may be plain dicts.
        if isinstance(data, dict):
            return {
                "number_of_documents": int(
                    data.get("numberOfDocuments", data.get("number_of_documents", 0))
                ),
                "is_indexing": bool(
                    data.get("isIndexing", data.get("is_indexing", False))
                ),
            }
        return {
            "number_of_documents": int(getattr(data, "number_of_documents", 0)),
            "is_indexing": bool(getattr(data, "is_indexing", False)),
        }
    except Exception as exc:
        logger.error("stats_api_failure", extra={"error": str(exc)})
        return {"number_of_documents": 0, "is_indexing": False, "unavailable": True}


def drop_index(index_name: str | None = None) -> bool:
    """Delete an index. Returns True if a delete task ran, False if the index was already absent."""
    idx = index_name or settings.meili_index_name
    client = _get_client()
    try:
        task_info = client.delete_index(idx)
        wait_for_task(task_info.task_uid, interval=2)
        return True
    except Exception as exc:
        low = str(exc).lower()
        if "not found" in low or "index_not_found" in low:
            return False
        raise


def _task_to_dict(task: Any) -> dict[str, Any]:
    """Convert a Meilisearch Task object (or dict) to a plain dict."""
    if isinstance(task, dict):
        return task
    return {k: v for k, v in vars(task).items() if not k.startswith("_")}


def check_task(task_uid: int) -> dict[str, Any]:
    """Get the status of a Meilisearch task by UID."""
    return _task_to_dict(_get_client().get_task(task_uid))


def wait_for_task(
    task_uid: int,
    interval: int = 5,
    timeout: int = 1800,
) -> dict[str, Any]:
    """Poll a task until it succeeds or fails (with progress output)."""
    elapsed = 0
    while elapsed < timeout:
        task = check_task(task_uid)
        status = task.get("status", "unknown")
        logger.info(
            "meili_task_status",
            extra={"task_uid": task_uid, "status": status, "elapsed_seconds": elapsed},
        )
        if status == "succeeded":
            return task
        if status == "failed":
            err = task.get("error") or {}
            msg = err.get("message", "unknown error") if isinstance(err, dict) else str(err)
            raise RuntimeError(f"Task {task_uid} failed: {msg}")
        time.sleep(interval)
        elapsed += interval
    raise TimeoutError(f"Task {task_uid} timed out after {timeout}s")


# ══════════════════════════════════════════════════════════════════════════════
#  Real-time / streaming ingest (PRD: ingest SLA — default from INGEST_SLA_SECONDS)
# ══════════════════════════════════════════════════════════════════════════════


def upsert_documents(
    documents: list[dict[str, Any]],
    index_name: str | None = None,
    *,
    wait: bool = True,
    sla_seconds: int | None = None,
) -> dict[str, Any]:
    """Add or update documents in Meilisearch and optionally block until indexed.

    This is the per-listing ingest path (event-driven, not batch).  Meilisearch
    uses the document ``id`` field as the primary key so this call is fully
    **idempotent** — re-sending the same document updates in-place.

    Tombstone / soft-delete: include ``{"id": "<id>", "deleted_at": "<iso>"}``
    in the payload; the calling service should filter on ``deleted_at IS NULL``
    at query time.

    Args:
        documents:   One or more normalised documents (must contain ``id``).
        index_name:  Target index (defaults to ``MEILI_INDEX_NAME``).
        wait:        If True (default), poll until the indexing task succeeds or
                     the SLA expires.  Pass ``wait=False`` for fire-and-forget.
        sla_seconds: Max seconds to wait before raising ``TimeoutError``.
                     Default: ``INGEST_SLA_SECONDS`` from settings (env, typically 300).

    Returns:
        dict with ``task_uid``, ``status``, ``elapsed_seconds``, and
        ``document_count``.

    Raises:
        RuntimeError:  Meilisearch rejected the request.
        TimeoutError:  Task did not complete within ``sla_seconds``.
    """
    idx = index_name or settings.meili_index_name
    sla = sla_seconds if sla_seconds is not None else settings.ingest_sla_seconds
    if not documents:
        return {"task_uid": None, "status": "noop", "elapsed_seconds": 0.0, "document_count": 0}

    t0 = time.monotonic()
    try:
        task_info = _get_client().index(idx).add_documents(documents, primary_key="id")
    except Exception as exc:
        logger.error("upsert_documents_failed", extra={"index": idx, "error": str(exc)})
        raise RuntimeError(f"Meilisearch upsert failed: {exc}") from exc

    task_uid: int = task_info.task_uid
    logger.info(
        "upsert_documents_queued",
        extra={"index": idx, "task_uid": task_uid, "count": len(documents)},
    )

    if not wait:
        return {
            "task_uid": task_uid,
            "status": "queued",
            "elapsed_seconds": round(time.monotonic() - t0, 3),
            "document_count": len(documents),
        }

    # Poll until succeeded / failed — with SLA guard.
    elapsed = 0.0
    interval = 1  # start at 1 s, cap at 10 s
    while elapsed < sla:
        time.sleep(interval)
        elapsed = time.monotonic() - t0
        try:
            task = _task_to_dict(_get_client().get_task(task_uid))
        except Exception as exc:
            logger.warning("task_poll_error", extra={"task_uid": task_uid, "error": str(exc)})
            interval = min(interval * 2, 10)
            continue

        status = task.get("status", "unknown")
        if status == "succeeded":
            elapsed_final = round(time.monotonic() - t0, 3)
            logger.info(
                "upsert_documents_indexed",
                extra={
                    "task_uid": task_uid,
                    "elapsed_seconds": elapsed_final,
                    "count": len(documents),
                    "sla_ok": elapsed_final <= sla,
                },
            )
            return {
                "task_uid": task_uid,
                "status": "succeeded",
                "elapsed_seconds": elapsed_final,
                "document_count": len(documents),
            }
        if status == "failed":
            err = task.get("error") or {}
            error_msg = err.get("message", "unknown error") if isinstance(err, dict) else str(err)
            raise RuntimeError(f"Meilisearch indexing task {task_uid} failed: {error_msg}")
        interval = min(interval * 2, 10)

    raise TimeoutError(
        f"Meilisearch indexing task {task_uid} did not complete within {sla}s "
        f"(ingest SLA exceeded; see INGEST_SLA_SECONDS)."
    )


def delete_document(
    document_id: str,
    index_name: str | None = None,
    *,
    wait: bool = True,
) -> dict[str, Any]:
    """Hard-delete a single document by id (use tombstone upsert for soft-delete)."""
    idx = index_name or settings.meili_index_name
    t0 = time.monotonic()
    try:
        task_info = _get_client().index(idx).delete_document(document_id)
    except Exception as exc:
        raise RuntimeError(f"Meilisearch delete failed: {exc}") from exc

    if not wait:
        return {"task_uid": task_info.task_uid, "status": "queued"}

    task = wait_for_task(task_info.task_uid, interval=1, timeout=60)
    return {
        "task_uid": task_info.task_uid,
        "status": task.get("status"),
        "elapsed_seconds": round(time.monotonic() - t0, 3),
    }


async def poll_last_indexing_task(index_name: str | None = None) -> dict[str, Any]:
    """Poll the most recent document-indexing task (async wrapper)."""
    idx = index_name or settings.meili_index_name
    params = {
        "indexUids": [idx],
        "statuses": ["succeeded", "processing"],
        "types": ["documentAdditionOrUpdate"],
        "limit": 1,
    }
    for _ in range(2):
        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None,
                lambda: _get_client().get_tasks(params),
            )
            results = data.get("results", [])
            if not results:
                return {"status": "unknown", "finished_at": None, "task_uid": None}
            task = results[0]
            return {
                "status": task.get("status", "unknown"),
                "finished_at": task.get("finishedAt"),
                "started_at": task.get("startedAt"),
                "task_uid": task.get("uid"),
            }
        except Exception as exc:
            logger.error("task_poll_failure", extra={"error": str(exc)})
            await asyncio.sleep(2)
    return {"status": "unknown", "finished_at": None, "task_uid": None}
