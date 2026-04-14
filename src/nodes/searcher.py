"""
Searcher node — calls Meilisearch and maps hits into SearchResult objects.

What this node does:
  1. Reads the retrieval strategy (KEYWORD / SEMANTIC / HYBRID) chosen by the router
  2. Builds a search query from parsed intent entities
  3. Builds Meilisearch filters from parsed intent filters (schema-driven aliases)
  4. Calls Meilisearch via the existing REST client (retries + fallback built-in)
  5. Maps each hit into a SearchResult for downstream nodes
  6. Checks freshness of results (are they stale?)
  7. If zero results with filters → retries WITHOUT filters (filter relaxation)
  8. If Meilisearch is completely down → writes ERROR to state (graph skips to reporter)

Uses ``MEILI_INDEX_NAME`` from settings for the target index.

No LLM call here → zero token cost.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import src.constants as C
from src.models.state import (
    SearchResult,
    FreshnessReport,
    ExtractionError,
    ErrorSeverity,
    PipelineEvent,
)
from src.models.schema_registry import get_schema
from src.tools.meilisearch_client import (
    search as meili_search,
    get_index_updated_at_meta,
)
from src.utils.config import get_settings
from src.utils.injection_guard import get_effective_user_query
from src.utils.logger import get_logger, log_node_exit
from src.utils.langwatch_tracker import annotate_node_span

logger = get_logger(__name__)
settings = get_settings()


def _stop_words_for(schema) -> frozenset:
    extra = {w.lower() for w in schema.query_stop_words_extra}
    return frozenset(C.BASE_QUERY_STOP_WORDS | extra)


def _query_stems(query: str, stop_words: frozenset) -> set[str]:
    """Extract query words and their root stems for fuzzy substring matching."""
    stems: set[str] = set()
    for w in query.split():
        w = w.lower()
        if len(w) < 3 or w in stop_words:
            continue
        stems.add(w)
        if len(w) > C.KEYWORD_STEM_LEN:
            stems.add(w[: C.KEYWORD_STEM_LEN])
    return stems


def _has_keyword_overlap(
    query: str,
    hits: list[dict],
    overlap_fields: list[str],
    stop_words: frozenset,
    top_n: int | None = None,
) -> bool:
    """Check whether enough of the top hits contain significant query keywords."""
    top_n = top_n if top_n is not None else C.KEYWORD_OVERLAP_TOP_N
    stems = _query_stems(query, stop_words)

    if not stems:
        return True

    checked = hits[:top_n]
    if not checked:
        return False

    matches = 0
    for hit in checked:
        parts = [(hit.get(f, "") or "").lower() for f in overlap_fields]
        text = " ".join(parts)

        if any(stem in text for stem in stems):
            matches += 1

    return (matches / len(checked)) >= C.KEYWORD_OVERLAP_MIN_RATIO


def _build_filter_string(intent_filters: dict, schema) -> str | None:
    """Converts parsed_intent.filters dict into a Meilisearch filter string."""
    if not intent_filters:
        return None

    filterable = frozenset(schema.filterable_fields)
    aliases = schema.filter_aliases_llm

    parts = []
    for key, value in intent_filters.items():
        meili_field = aliases.get(key.lower())
        if meili_field and meili_field in filterable:
            safe_value = str(value).replace('"', '\\"')
            parts.append(f'{meili_field} = "{safe_value}"')

    if not parts:
        return None

    return " AND ".join(parts)


def _build_search_query(state: dict) -> str:
    """Builds the query string to send to Meilisearch."""
    intent = state.get("parsed_intent", {})
    entities = intent.get("entities", [])

    if entities:
        return " ".join(entities)

    return get_effective_user_query(state)


def _hits_to_search_results(hits: list[dict]) -> list[SearchResult]:
    """Converts raw Meilisearch hit dicts into SearchResult models."""
    results = []
    for hit in hits:
        freshness_ts = None
        indexed_at = hit.get("indexed_at")
        if indexed_at:
            try:
                freshness_ts = datetime.fromtimestamp(int(indexed_at), tz=timezone.utc)
            except (ValueError, TypeError, OSError):
                pass

        results.append(
            SearchResult(
                id=str(hit.get("id", "")),
                title=hit.get("title", ""),
                score=float(hit.get("_rankingScore", 0.0)),
                source_fields={
                    k: v for k, v in hit.items()
                    if k not in ("id", "title", "_rankingScore")
                },
                freshness_timestamp=freshness_ts,
            )
        )

    return results


def _build_freshness_report(
    results: list[SearchResult],
    *,
    index_stats_updated_at: datetime | None,
    index_meta_ok: bool,
) -> FreshnessReport:
    """Checks how fresh/stale the search results are."""
    now = datetime.now(timezone.utc)
    stale_ids = []
    max_staleness = 0.0
    oldest_timestamp = None
    index_lag = 0.0

    for r in results:
        if r.freshness_timestamp:
            age_seconds = (now - r.freshness_timestamp).total_seconds()

            if age_seconds > settings.staleness_threshold_seconds:
                stale_ids.append({"id": r.id, "title": r.title})

            if age_seconds > max_staleness:
                max_staleness = age_seconds

            if oldest_timestamp is None or r.freshness_timestamp < oldest_timestamp:
                oldest_timestamp = r.freshness_timestamp

    if index_stats_updated_at is not None:
        index_lag = max((now - index_stats_updated_at).total_seconds(), 0.0)

    return FreshnessReport(
        index_last_updated=oldest_timestamp,
        index_stats_updated_at=index_stats_updated_at,
        staleness_flag=len(stale_ids) > 0,
        stale_result_ids=stale_ids,
        max_staleness_seconds=round(max_staleness, 2),
        index_lag=round(index_lag, 2),
        freshness_unknown=not index_meta_ok,
    )


def searcher_node(state: dict) -> dict:
    """Execute search against Meilisearch (``MEILI_INDEX_NAME``) and update state."""
    start = time.perf_counter()
    query_hash = state.get("query_hash", "")
    strategy = state.get("retrieval_strategy", "HYBRID")
    hybrid_weights = state.get("hybrid_weights", {"semanticRatio": 0.6})
    filter_relaxation = False

    updates: dict = {}
    new_errors: list[dict] = []
    partial_results = False

    schema = get_schema(settings.dataset_schema)
    retrieve_fields = schema.meilisearch_attributes_to_retrieve()
    stop_words = _stop_words_for(schema)

    search_query = _build_search_query(state)

    intent = state.get("parsed_intent", {})
    filter_string = _build_filter_string(intent.get("filters", {}), schema)

    limit = C.SEARCH_HITS_LIMIT

    logger.info(
        "searcher_executing",
        extra={
            "query_hash": query_hash,
            "search_query": search_query,
            "strategy": strategy,
            "filters": filter_string,
        },
    )

    try:
        raw_response = meili_search(
            query=search_query,
            strategy=strategy,
            hybrid_weights=hybrid_weights,
            filters=filter_string,
            limit=limit,
            retrieve_fields=retrieve_fields,
        )
        hits = raw_response.get("hits", [])
        partial_results = bool(raw_response.get("partial", False))

        if len(hits) == 0 and filter_string:
            logger.info(
                "searcher_filter_relaxation",
                extra={
                    "query_hash": query_hash,
                    "reason": "zero results with filters, retrying without",
                    "original_filters": filter_string,
                },
            )

            raw_response = meili_search(
                query=search_query,
                strategy=strategy,
                hybrid_weights=hybrid_weights,
                filters=None,
                limit=limit,
                retrieve_fields=retrieve_fields,
            )
            hits = raw_response.get("hits", [])
            filter_relaxation = True

            new_errors.append(
                ExtractionError(
                    node="searcher",
                    severity=ErrorSeverity.WARNING,
                    message="FILTER_RELAXATION_APPLIED",
                    fallback_applied=True,
                    fallback_description=(
                        f"Original filters ({filter_string}) returned 0 results. "
                        f"Retried without filters — got {len(hits)} results."
                    ),
                ).model_dump()
            )

        if (
            strategy in ("HYBRID", "SEMANTIC")
            and hits
            and not _has_keyword_overlap(
                search_query,
                hits,
                schema.keyword_overlap_fields,
                stop_words,
            )
        ):
            logger.warning(
                "searcher_semantic_degradation",
                extra={
                    "query_hash": query_hash,
                    "strategy": strategy,
                    "reason": "top results have no keyword overlap with query",
                },
            )

            keyword_response = meili_search(
                query=search_query,
                strategy="KEYWORD",
                filters=filter_string,
                limit=limit,
                retrieve_fields=retrieve_fields,
            )
            keyword_hits = keyword_response.get("hits", [])

            if keyword_hits:
                hits = keyword_hits
                raw_response = keyword_response
                partial_results = True

            new_errors.append(
                ExtractionError(
                    node="searcher",
                    severity=ErrorSeverity.WARNING,
                    message="SEMANTIC_DEGRADATION_FALLBACK",
                    fallback_applied=True,
                    fallback_description=(
                        f"{strategy} results had no keyword overlap with the "
                        f"query. Fell back to keyword search — got "
                        f"{len(keyword_hits)} results."
                    ),
                ).model_dump()
            )

        if partial_results:
            new_errors.append(
                ExtractionError(
                    node="searcher",
                    severity=ErrorSeverity.WARNING,
                    message=PipelineEvent.PARTIAL_RESULTS.value,
                    fallback_applied=True,
                    fallback_description=(
                        "Hybrid retrieval degraded to keyword fallback after retries."
                    ),
                ).model_dump()
            )

        search_results = _hits_to_search_results(hits)

        idx_stats_at, idx_meta_ok = get_index_updated_at_meta()
        freshness = _build_freshness_report(
            search_results,
            index_stats_updated_at=idx_stats_at,
            index_meta_ok=idx_meta_ok,
        )

        updates["search_results"] = [r.model_dump() for r in search_results]
        updates["freshness_metadata"] = freshness.model_dump()
        updates["filter_relaxation_applied"] = filter_relaxation
        updates["partial_results"] = partial_results

        result_count = len(search_results)
        strategy_used = raw_response.get("strategy_used", strategy)

        logger.info(
            "searcher_complete",
            extra={
                "query_hash": query_hash,
                "result_count": result_count,
                "strategy_used": strategy_used,
                "latency_ms": raw_response.get("latency_ms", 0),
                "filter_relaxation": filter_relaxation,
                "stale_results": len(freshness.stale_result_ids),
            },
        )

    except RuntimeError as exc:
        logger.error(
            "searcher_meili_failure",
            extra={"query_hash": query_hash, "error": str(exc)},
        )

        new_errors.append(
            ExtractionError(
                node="searcher",
                severity=ErrorSeverity.ERROR,
                message=f"MEILISEARCH_UNAVAILABLE: {str(exc)[:200]}",
                fallback_applied=False,
                fallback_description=(
                    "Meilisearch is unreachable after retries. "
                    "No search results available."
                ),
            ).model_dump()
        )
        updates["search_results"] = []
        idx_stats_at, idx_meta_ok = get_index_updated_at_meta()
        updates["freshness_metadata"] = FreshnessReport(
            index_stats_updated_at=idx_stats_at,
            freshness_unknown=not idx_meta_ok,
        ).model_dump()
        updates["partial_results"] = False

        result_count = 0
        strategy_used = strategy

    except Exception as exc:
        logger.error(
            "searcher_unexpected_error",
            extra={"query_hash": query_hash, "error": str(exc)},
        )

        new_errors.append(
            ExtractionError(
                node="searcher",
                severity=ErrorSeverity.ERROR,
                message=f"SEARCHER_UNEXPECTED: {str(exc)[:200]}",
                fallback_applied=False,
                fallback_description="Unexpected error in searcher node.",
            ).model_dump()
        )
        updates["search_results"] = []
        idx_stats_at, idx_meta_ok = get_index_updated_at_meta()
        updates["freshness_metadata"] = FreshnessReport(
            index_stats_updated_at=idx_stats_at,
            freshness_unknown=not idx_meta_ok,
        ).model_dump()
        updates["partial_results"] = False

        result_count = 0
        strategy_used = strategy

    duration_ms = (time.perf_counter() - start) * 1000

    log_node_exit(
        logger, "searcher", query_hash,
        result_count, strategy_used, duration_ms, 0.0,
        extra={"filter_relaxation": filter_relaxation},
    )
    annotate_node_span(
        "searcher", result_count, strategy_used, duration_ms,
        extra={"filter_relaxation": filter_relaxation},
    )

    if new_errors:
        updates["errors"] = new_errors

    return updates
