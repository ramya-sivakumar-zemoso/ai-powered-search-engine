"""Meilisearch client — search (with retries + hybrid fallback), stats, tasks.

Uses the official ``meilisearch`` Python SDK for all API calls.
Retry logic and hybrid-to-keyword fallback are layered on top because
the SDK does not provide these out of the box.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import meilisearch
from meilisearch.errors import (
    MeilisearchApiError,
    MeilisearchCommunicationError,
    MeilisearchTimeoutError,
)

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

    for attempt in range(3):
        try:
            result = index.search(query, opt_params)
            return _norm(result, strategy_label, False)
        except _RETRYABLE as exc:
            last_error = exc
            if attempt < 2:
                wait = 0.1 * (2 ** (attempt + 1))
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
        f"Meilisearch search failed after 3 retries: {last_error}"
    ) from last_error


# ══════════════════════════════════════════════════════════════════════════════
#  Default retrieve fields
# ══════════════════════════════════════════════════════════════════════════════

def _retrieve_fields(fields: list[str] | None) -> list[str]:
    if fields:
        return fields
    return [
        "id", "title", "description", "overview",
        "category", "genres_all", "brand", "price",
        "rating", "in_stock", "indexed_at", "indexed_at_iso",
        "poster",
    ]


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
#  Index stats + task helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_index_stats(index_name: str | None = None) -> dict[str, Any]:
    idx = index_name or settings.meili_index_name
    try:
        data = _get_client().index(idx).get_stats()
        return {
            "number_of_documents": data.get("numberOfDocuments", 0),
            "is_indexing": data.get("isIndexing", False),
        }
    except Exception as exc:
        logger.error("stats_api_failure", extra={"error": str(exc)})
        return {"number_of_documents": 0, "is_indexing": False, "unavailable": True}


def check_task(task_uid: int) -> dict[str, Any]:
    """Get the status of a Meilisearch task by UID."""
    return _get_client().get_task(task_uid)


def wait_for_task(
    task_uid: int,
    interval: int = 5,
    timeout: int = 1800,
) -> dict[str, Any]:
    """Poll a task until it succeeds or fails (with progress output)."""
    elapsed = 0
    while elapsed < timeout:
        task = check_task(task_uid)
        status = task["status"]
        print(f"  [task {task_uid}] {status}  ({elapsed}s elapsed)")
        if status == "succeeded":
            return task
        if status == "failed":
            error = task.get("error", {}).get("message", "unknown error")
            raise RuntimeError(f"Task {task_uid} failed: {error}")
        time.sleep(interval)
        elapsed += interval
    raise TimeoutError(f"Task {task_uid} timed out after {timeout}s")


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
