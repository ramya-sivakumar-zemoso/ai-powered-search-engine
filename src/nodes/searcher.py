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
    PipelineEvent,
)
from src.models.schema_registry import get_schema
from src.tools.meilisearch_client import (
    search as meili_search,
    get_index_updated_at_meta,
)
from src.utils.config import get_settings
from src.utils.logger import get_logger, log_node_exit
from src.utils.langwatch_tracker import annotate_node_span

logger = get_logger(__name__)
settings = get_settings()

_BASE_STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "in", "of", "to", "for", "with", "is",
    "on", "at", "by", "from", "this", "that", "it", "as", "are", "was",
    "be", "has", "had", "not", "but", "all", "can", "her", "his", "one",
    "our", "out", "you", "its", "my", "we", "do", "no", "so", "up", "if",
    "me", "what", "about", "which", "when", "how", "who", "where", "why",
})


def _stop_words_for(schema) -> frozenset:
    extra = {w.lower() for w in schema.query_stop_words_extra}
    return frozenset(_BASE_STOP_WORDS | extra)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Keyword overlap check (semantic degradation detector)
# ══════════════════════════════════════════════════════════════════════════════

_OVERLAP_MIN_RATIO = 0.5
_STEM_LEN = 5


def _query_stems(query: str, stop_words: frozenset) -> set[str]:
    """Extract query words and their root stems for fuzzy substring matching.

    For each word, we emit both the full word and a short stem (first 5 chars).
    The stem handles morphological variants: "pirates"→"pirat" matches "pirate",
    "romantic"→"roman" matches "romance", "dramatic"→"drama" matches "drama".
    """
    stems: set[str] = set()
    for w in query.split():
        w = w.lower()
        if len(w) < 3 or w in stop_words:
            continue
        stems.add(w)
        if len(w) > _STEM_LEN:
            stems.add(w[:_STEM_LEN])
    return stems


def _has_keyword_overlap(
    query: str,
    hits: list[dict],
    overlap_fields: list[str],
    stop_words: frozenset,
    top_n: int = 5,
) -> bool:
    """Check whether enough of the top hits contain significant query keywords.

    Returns True only when at least ``_OVERLAP_MIN_RATIO`` (50 %) of the top-N
    results contain a query keyword (or its stem) in the configured overlap
    fields. Returns True when there are no significant keywords to check.

    A single hit matching out of five (20 %) is NOT enough — that pattern
    indicates the embedder placed one genuine result among garbage and the
    keyword fallback should still fire.
    """
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

    return (matches / len(checked)) >= _OVERLAP_MIN_RATIO


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Build filter string for Meilisearch
# ══════════════════════════════════════════════════════════════════════════════

def _build_filter_string(intent_filters: dict, schema) -> str | None:
    """
    Converts parsed_intent.filters dict into a Meilisearch filter string.

    Example:
        {"genre": "Action", "year": "2020"}
        → 'category = "Action"' when genre aliases to category and year is not filterable

    Meilisearch filter docs: https://www.meilisearch.com/docs/learn/filtering_and_sorting/filter_expression

    Args:
        intent_filters: The filters dict from parsed_intent.
        schema: Active ``DatasetSchema`` (filterable fields + LLM aliases).

    Returns:
        A Meilisearch filter string, or None if no valid filters found.
    """
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
        {"id": "11", "title": "Example", "_rankingScore": 0.95, ...}

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

def _build_freshness_report(
    results: list[SearchResult],
    *,
    index_stats_updated_at: datetime | None,
    index_meta_ok: bool,
) -> FreshnessReport:
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
        Dict of only the keys this node changed.
    """
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

    # ── Step 1: Build the search query ────────────────────────────────────────
    search_query = _build_search_query(state)

    # ── Step 2: Build Meilisearch filters ─────────────────────────────────────
    intent = state.get("parsed_intent", {})
    filter_string = _build_filter_string(intent.get("filters", {}), schema)

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
            retrieve_fields=retrieve_fields,
        )
        hits = raw_response.get("hits", [])
        partial_results = bool(raw_response.get("is_fallback", False))

        # ── Step 4: Filter relaxation ─────────────────────────────────────────
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

        # ── Step 4b: Semantic degradation fallback ─────────────────────────
        # When hybrid/semantic search returns results that share zero keywords
        # with the query, the embedder is likely producing degenerate vectors.
        # Fall back to keyword search so relevant results are not silently lost.
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
                limit=20,
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

        # ── Step 5: Map hits to SearchResult objects ──────────────────────────
        search_results = _hits_to_search_results(hits)

        # ── Step 6: Build freshness report ────────────────────────────────────
        idx_stats_at, idx_meta_ok = get_index_updated_at_meta()
        freshness = _build_freshness_report(
            search_results,
            index_stats_updated_at=idx_stats_at,
            index_meta_ok=idx_meta_ok,
        )

        # ── Step 7: Update state ──────────────────────────────────────────────
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

    if new_errors:
        updates["errors"] = new_errors

    return updates
