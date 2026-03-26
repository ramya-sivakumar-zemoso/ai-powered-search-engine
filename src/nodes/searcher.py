"""
Searcher node — calls Meilisearch and maps hits into SearchResult objects.

What this node does:
  1. Reads the retrieval strategy (KEYWORD / SEMANTIC / HYBRID) chosen by the router
  2. Builds a search query from parsed intent entities
  3. Builds Meilisearch filters from parsed intent filters (e.g. genre = "Action")
  4. Calls Meilisearch via the existing REST client (retries + fallback built-in)
  5. Maps each hit into a SearchResult for downstream nodes
  6. Checks freshness of results (are they stale?)
  7. If zero results with filters → retries WITHOUT filters (filter relaxation)
  8. If Meilisearch is completely down → writes ERROR to state (graph skips to reporter)

No LLM call here → zero token cost.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

from src.models.state import (
    SearchResult,
    FreshnessReport,
    ExtractionError,
    ErrorSeverity,
)
from src.tools.meilisearch_client import search as meili_search
from src.utils.config import get_settings
from src.utils.logger import get_logger, log_node_exit
from src.utils.langwatch_tracker import annotate_node_span

logger = get_logger(__name__)
settings = get_settings()


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Build filter string for Meilisearch
# ══════════════════════════════════════════════════════════════════════════════

# These are the fields Meilisearch can filter on (must match schema_registry.py)
FILTERABLE_FIELDS = {"category", "genres_all", "in_stock"}


def _build_filter_string(intent_filters: dict) -> str | None:
    """
    Converts parsed_intent.filters dict into a Meilisearch filter string.

    Example:
        {"genre": "Action", "year": "2020"}
        → 'category = "Action"'
        (year is not filterable in our schema, so it's skipped)

    Meilisearch filter docs: https://www.meilisearch.com/docs/learn/filtering_and_sorting/filter_expression

    Args:
        intent_filters: The filters dict from parsed_intent (e.g. {"genre": "Action"})

    Returns:
        A Meilisearch filter string, or None if no valid filters found.
    """
    if not intent_filters:
        return None

    # Map common LLM output field names to our actual Meilisearch filterable fields
    field_aliases = {
        "genre": "category",
        "genres": "genres_all",
        "category": "category",
        "in_stock": "in_stock",
    }

    parts = []
    for key, value in intent_filters.items():
        meili_field = field_aliases.get(key.lower())
        if meili_field and meili_field in FILTERABLE_FIELDS:
            safe_value = str(value).replace('"', '\\"')
            parts.append(f'{meili_field} = "{safe_value}"')

    if not parts:
        return None

    return " AND ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Build search query string from parsed intent
# ══════════════════════════════════════════════════════════════════════════════

def _build_search_query(state: dict) -> str:
    """
    Builds the query string to send to Meilisearch.

    Priority:
      1. Join parsed_intent.entities (e.g. ["sci-fi", "time travel"] → "sci-fi time travel")
      2. Fall back to the raw user query if no entities were extracted

    Args:
        state: The pipeline state dict.

    Returns:
        The search query string for Meilisearch.
    """
    intent = state.get("parsed_intent", {})
    entities = intent.get("entities", [])

    if entities:
        return " ".join(entities)

    return state.get("query", "")


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Map Meilisearch hits to SearchResult objects
# ══════════════════════════════════════════════════════════════════════════════

def _hits_to_search_results(hits: list[dict]) -> list[SearchResult]:
    """
    Converts raw Meilisearch hit dicts into SearchResult Pydantic models.

    Each hit from Meilisearch looks like:
        {"id": "11", "title": "Star Wars", "_rankingScore": 0.95, ...}

    We extract the core fields and store everything else in source_fields
    so downstream nodes (evaluator, reranker) can access any field they need.

    Args:
        hits: List of hit dicts from Meilisearch response.

    Returns:
        List of SearchResult objects.
    """
    results = []
    for hit in hits:
        # Parse freshness timestamp from indexed_at (unix seconds)
        freshness_ts = None
        indexed_at = hit.get("indexed_at")
        if indexed_at:
            try:
                freshness_ts = datetime.fromtimestamp(int(indexed_at), tz=timezone.utc)
            except (ValueError, TypeError, OSError):
                pass

        result = SearchResult(
            id=str(hit.get("id", "")),
            title=hit.get("title", ""),
            score=float(hit.get("_rankingScore", 0.0)),
            source_fields={
                k: v for k, v in hit.items()
                if k not in ("id", "title", "_rankingScore")
            },
            freshness_timestamp=freshness_ts,
        )
        results.append(result)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Build freshness report
# ══════════════════════════════════════════════════════════════════════════════

def _build_freshness_report(results: list[SearchResult]) -> FreshnessReport:
    """
    Checks how fresh/stale the search results are.

    PRD Section 4.3: "Surface index freshness metadata in every response."

    Uses two thresholds from .env:
      - FRESHNESS_THRESHOLD_SECONDS (default 300 = 5 min) — results newer than this are "fresh"
      - STALENESS_THRESHOLD_SECONDS (default 3600 = 1 hour) — results older than this are "stale"

    Args:
        results: List of SearchResult objects with freshness_timestamp set.

    Returns:
        FreshnessReport with staleness flags and stale result IDs.
    """
    now = datetime.now(timezone.utc)
    stale_ids = []
    max_staleness = 0.0
    oldest_timestamp = None

    for r in results:
        if r.freshness_timestamp:
            age_seconds = (now - r.freshness_timestamp).total_seconds()

            if age_seconds > settings.staleness_threshold_seconds:
                stale_ids.append(r.id)

            if age_seconds > max_staleness:
                max_staleness = age_seconds

            if oldest_timestamp is None or r.freshness_timestamp < oldest_timestamp:
                oldest_timestamp = r.freshness_timestamp

    return FreshnessReport(
        index_last_updated=oldest_timestamp,
        staleness_flag=len(stale_ids) > 0,
        stale_result_ids=stale_ids,
        max_staleness_seconds=round(max_staleness, 2),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN NODE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def searcher_node(state: dict) -> dict:
    """
    The searcher node — executes the actual search against Meilisearch.

    Flow:
      Step 1: Build search query from parsed intent entities
      Step 2: Build Meilisearch filter string from parsed intent filters
      Step 3: Call Meilisearch with the chosen retrieval strategy
      Step 4: If zero results with filters → retry without filters (filter relaxation)
      Step 5: Map hits to SearchResult objects
      Step 6: Build freshness report
      Step 7: Update state with results and metadata
      Step 8: Log structured node exit + LangWatch annotation

    If Meilisearch is completely unreachable (after retries + fallback in the client),
    we write an ERROR to state. The graph's route_after_searcher will send it
    straight to reporter, skipping evaluator/reranker.

    Args:
        state: The pipeline state dict.

    Returns:
        Updated state dict with search_results, freshness_metadata, etc.
    """
    start = time.perf_counter()
    query_hash = state.get("query_hash", "")
    strategy = state.get("retrieval_strategy", "HYBRID")
    hybrid_weights = state.get("hybrid_weights", {"semanticRatio": 0.6})
    filter_relaxation = False

    # ── Step 1: Build the search query ────────────────────────────────────────
    search_query = _build_search_query(state)

    # ── Step 2: Build Meilisearch filters ─────────────────────────────────────
    intent = state.get("parsed_intent", {})
    filter_string = _build_filter_string(intent.get("filters", {}))

    logger.info(
        "searcher_executing",
        extra={
            "query_hash": query_hash,
            "search_query": search_query,
            "strategy": strategy,
            "filters": filter_string,
        },
    )

    # ── Step 3: Call Meilisearch ──────────────────────────────────────────────
    try:
        raw_response = meili_search(
            query=search_query,
            strategy=strategy,
            hybrid_weights=hybrid_weights,
            filters=filter_string,
            limit=20,
        )
        hits = raw_response.get("hits", [])

        # ── Step 4: Filter relaxation ─────────────────────────────────────────
        # If we got zero results AND we had filters applied, retry without filters.
        # This prevents returning empty results when filters are too restrictive.
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
                limit=20,
            )
            hits = raw_response.get("hits", [])
            filter_relaxation = True

            errors = state.get("errors", [])
            errors.append(
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
            state["errors"] = errors

        # ── Step 5: Map hits to SearchResult objects ──────────────────────────
        search_results = _hits_to_search_results(hits)

        # ── Step 6: Build freshness report ────────────────────────────────────
        freshness = _build_freshness_report(search_results)

        # ── Step 7: Update state ──────────────────────────────────────────────
        state["search_results"] = [r.model_dump() for r in search_results]
        state["freshness_metadata"] = freshness.model_dump()
        state["filter_relaxation_applied"] = filter_relaxation

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
        # Meilisearch is completely unreachable (after 3 retries + fallback)
        # The meilisearch_client raises RuntimeError on hard failure.
        logger.error(
            "searcher_meili_failure",
            extra={"query_hash": query_hash, "error": str(exc)},
        )

        errors = state.get("errors", [])
        errors.append(
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
        state["errors"] = errors
        state["search_results"] = []
        state["freshness_metadata"] = FreshnessReport().model_dump()

        result_count = 0
        strategy_used = strategy

    except Exception as exc:
        # Unexpected error — catch-all so the pipeline doesn't crash
        logger.error(
            "searcher_unexpected_error",
            extra={"query_hash": query_hash, "error": str(exc)},
        )

        errors = state.get("errors", [])
        errors.append(
            ExtractionError(
                node="searcher",
                severity=ErrorSeverity.ERROR,
                message=f"SEARCHER_UNEXPECTED: {str(exc)[:200]}",
                fallback_applied=False,
                fallback_description="Unexpected error in searcher node.",
            ).model_dump()
        )
        state["errors"] = errors
        state["search_results"] = []
        state["freshness_metadata"] = FreshnessReport().model_dump()

        result_count = 0
        strategy_used = strategy

    # ── Step 8: Log and annotate node exit ────────────────────────────────────
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

    return state
