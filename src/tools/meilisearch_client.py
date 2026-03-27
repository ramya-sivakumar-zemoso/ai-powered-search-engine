"""REST client for Meilisearch: search (with retries + hybrid fallback), stats, tasks."""
from __future__ import annotations

import asyncio
import time
from typing import Any

import requests
from requests.exceptions import ConnectionError, Timeout

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def _headers() -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.meili_master_key}",
    }


def _url(path: str) -> str:
    return f"{settings.meili_url}{path}"


def _norm(data: dict[str, Any], strategy: str, partial: bool) -> dict[str, Any]:
    return {
        "hits": data.get("hits", []),
        "latency_ms": data.get("processingTimeMs", 0),
        "estimated_total_hits": data.get("estimatedTotalHits", 0),
        "strategy_used": strategy,
        "partial": partial,
    }


def health() -> dict[str, Any]:
    resp = requests.get(_url("/health"), headers=_headers(), timeout=5)
    resp.raise_for_status()
    return resp.json()


def _execute_search(
    body: dict[str, Any],
    index_name: str | None = None,
) -> dict[str, Any]:
    idx = index_name or settings.meili_index_name
    url = _url(f"/indexes/{idx}/search")
    strategy_label = body.pop("_strategy", "KEYWORD")
    send_body = body
    last_error: Exception | None = None

    for attempt in range(3):
        try:
            resp = requests.post(url, headers=_headers(), json=send_body, timeout=10)
            resp.raise_for_status()
            return _norm(resp.json(), strategy_label, False)
        except (ConnectionError, Timeout, requests.HTTPError) as exc:
            last_error = exc
            if attempt < 2:
                wait = 0.1 * (2 ** (attempt + 1))
                logger.warning(
                    "meili_retry",
                    extra={"attempt": attempt + 1, "wait_s": wait, "error": str(exc)},
                )
                time.sleep(wait)

    if strategy_label != "KEYWORD" and "hybrid" in send_body:
        logger.warning(
            "meili_hybrid_fallback",
            extra={"original_strategy": strategy_label, "error": str(last_error)},
        )
        fallback_body = {k: v for k, v in send_body.items() if k != "hybrid"}
        try:
            resp = requests.post(url, headers=_headers(), json=fallback_body, timeout=10)
            resp.raise_for_status()
            return _norm(resp.json(), "KEYWORD_FALLBACK", True)
        except Exception as fallback_exc:
            logger.error("meili_hard_failure", extra={"error": str(fallback_exc)})
            raise RuntimeError(
                f"Meilisearch failed after fallback: {fallback_exc}"
            ) from fallback_exc

    raise RuntimeError(
        f"Meilisearch search failed after 3 retries: {last_error}"
    ) from last_error


def _retrieve_fields(fields: list[str] | None) -> list[str]:
    if fields:
        return fields
    return [
        "id", "title", "description", "overview",
        "category", "genres_all", "brand", "price",
        "rating", "in_stock", "indexed_at", "indexed_at_iso",
        "poster",
    ]


def keyword_search(
    query: str,
    limit: int = 20,
    filters: str | None = None,
    retrieve_fields: list[str] | None = None,
    index_name: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "q": query,
        "limit": limit,
        "showRankingScore": True,
        "attributesToRetrieve": _retrieve_fields(retrieve_fields),
        "_strategy": "KEYWORD",
    }
    if filters:
        body["filter"] = filters
    return _execute_search(body, index_name)


def hybrid_search(
    query: str,
    semantic_ratio: float = 0.6,
    limit: int = 20,
    filters: str | None = None,
    retrieve_fields: list[str] | None = None,
    index_name: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "q": query,
        "limit": limit,
        "showRankingScore": True,
        "attributesToRetrieve": _retrieve_fields(retrieve_fields),
        "hybrid": {
            "embedder": settings.meili_embedder_name,
            "semanticRatio": semantic_ratio,
        },
        "_strategy": "HYBRID",
    }
    if filters:
        body["filter"] = filters
    return _execute_search(body, index_name)


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


def get_index_stats(index_name: str | None = None) -> dict[str, Any]:
    idx = index_name or settings.meili_index_name
    try:
        resp = requests.get(
            _url(f"/indexes/{idx}/stats"), headers=_headers(), timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "number_of_documents": data.get("numberOfDocuments", 0),
            "is_indexing": data.get("isIndexing", False),
        }
    except Exception as exc:
        logger.error("stats_api_failure", extra={"error": str(exc)})
        return {"number_of_documents": 0, "is_indexing": False, "unavailable": True}


def check_task(task_uid: int) -> dict[str, Any]:
    resp = requests.get(_url(f"/tasks/{task_uid}"), headers=_headers(), timeout=5)
    resp.raise_for_status()
    return resp.json()


def wait_for_task(
    task_uid: int,
    interval: int = 5,
    timeout: int = 1800,
) -> dict[str, Any]:
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
    idx = index_name or settings.meili_index_name
    params = {
        "indexUids": idx,
        "statuses": "succeeded,processing",
        "types": "documentAdditionOrUpdate",
        "limit": "1",
    }
    for _ in range(2):
        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None,
                lambda: requests.get(
                    _url("/tasks"), headers=_headers(), params=params, timeout=3,
                ).json(),
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
