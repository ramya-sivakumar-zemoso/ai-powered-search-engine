"""
Evaluator node — scores search result quality and decides accept / retry / exhausted.

What this node does:
  1. Takes the search_results from the searcher node
  2. Scores them on 4 signals:
     - semantic_relevance  (30%) — are results meaningfully relevant?
     - result_coverage     (22%) — did we get enough results?
     - ranking_stability   (10%) — are rankings diverse and stable?
     - freshness_signal    (20%) — are results recent or stale?
  3. Combines signals into a single quality_score (0.0 to 1.0)
  4. Decides:
     - "accept"    → results are good enough, send to reranker
     - "retry"     → results are poor, send back to router with a prescription
     - "exhausted" → we've hit the iteration limit or budget, send best results forward

Why 4 signals instead of 5:
  The PRD defines 5 quality signals. The 5th (confidence) comes from the
  reranker's cross-encoder output. But the evaluator runs BEFORE the reranker,
  so confidence is not available yet. We drop it and renormalize:
    30% + 22% + 10% + 20% = 82% → renormalized to 100%
    semantic_relevance: 36.6%, result_coverage: 26.8%,
    ranking_stability: 12.2%, freshness_signal: 24.4%
  The accept threshold drops from 0.72 to 0.65 to compensate.

Why these percentages (justification against PRD):
  - semantic_relevance (30%): Highest weight because relevance is the primary
    goal of any search system. If results don't match user intent semantically,
    nothing else matters. The Meilisearch _rankingScore captures both keyword
    and vector similarity, making it the most direct quality indicator.
  - result_coverage (22%): Second highest because a search that returns zero
    or very few results is a functional failure regardless of relevance.
    Coverage ensures the retrieval strategy is casting a wide enough net.
  - freshness_signal (20%): Weighted meaningfully because the PRD requires
    staleness awareness (Section 4.3). For time-sensitive domains (news,
    products), stale results degrade user trust. For static catalogs this
    signal may be mild, but the weight keeps behavior aligned with PRD 4.3
    for time-sensitive domains without code changes.
  - ranking_stability (10%): Lowest weight because near-duplicate results
    are an edge case, not the norm. When duplicates appear they hurt UX,
    but the reranker (downstream) further diversifies. This signal acts
    as an early warning rather than a primary quality gate.

No LLM call here → zero token cost. Pure Python scoring.
"""
from __future__ import annotations

import time
from difflib import SequenceMatcher
from typing import Any

from src.models.schema_registry import get_schema
from src.models.state import (
    RetryPrescription,
    RetrievalStrategy,
    SearchAttempt,
    ExtractionError,
    ErrorSeverity,
    PipelineEvent,
)
from src.utils.config import get_settings
from src.utils.logger import get_logger, log_node_exit
from src.utils.langwatch_tracker import annotate_node_span

logger = get_logger(__name__)
settings = get_settings()

# Quality threshold: results scoring above this are accepted.
# Lower than the original 0.72 because we score on 4 signals (no confidence).
ACCEPT_THRESHOLD = 0.65

# Signal weights — renormalized from the PRD's 5-signal plan after dropping
# the confidence signal (only available after reranker, not here).
# PRD original: semantic_relevance=30%, result_coverage=22%, confidence=18%,
#               ranking_stability=10%, freshness_signal=20%
# After dropping confidence: 30+22+10+20 = 82% → renormalize to 100%
DEFAULT_WEIGHTS = {
    "semantic_relevance": 0.366,   # 30/82
    "result_coverage": 0.268,      # 22/82
    "ranking_stability": 0.122,    # 10/82
    "freshness_signal": 0.244,     # 20/82
}

SIGNAL_KEYS = tuple(DEFAULT_WEIGHTS.keys())


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL 1: Semantic Relevance (30% → 36.6% renormalized)
# ══════════════════════════════════════════════════════════════════════════════

def _score_semantic_relevance(results: list[dict]) -> float:
    """
    Scores how semantically relevant the results are to the query.

    Uses the Meilisearch _rankingScore which combines keyword matching
    and vector similarity (when hybrid search is used). This is the most
    direct indicator of whether results actually match user intent.

    Logic:
      - Averages the _rankingScore of the top 5 results
      - Higher average = better semantic match

    Meilisearch _rankingScore ranges from 0.0 to 1.0.

    Args:
        results: List of search result dicts from state.

    Returns:
        Score between 0.0 and 1.0.
    """
    if not results:
        return 0.0

    top_scores = []
    for r in results[:5]:
        score = r.get("score", 0.0) if isinstance(r, dict) else 0.0
        top_scores.append(float(score))

    if not top_scores:
        return 0.0

    avg = sum(top_scores) / len(top_scores)
    return min(avg, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL 2: Result Coverage (22% → 26.8% renormalized)
# ══════════════════════════════════════════════════════════════════════════════

def _score_result_coverage(results: list[dict]) -> float:
    """
    Scores based on how many results the searcher returned.

    A search returning zero results is a functional failure. Too few results
    limit the reranker's ability to find the best matches. This signal
    ensures the retrieval strategy is casting a wide enough net.

    Logic:
      0 results     → 0.0  (nothing found — retrieval failed)
      1-2 results   → 0.4  (too few for meaningful ranking)
      3-5 results   → 0.7  (acceptable, reranker can work)
      6-10 results  → 0.9  (good coverage)
      11+ results   → 1.0  (full coverage)

    Args:
        results: List of search result dicts from state.

    Returns:
        Score between 0.0 and 1.0.
    """
    count = len(results)
    if count == 0:
        return 0.0
    if count <= 2:
        return 0.4
    if count <= 5:
        return 0.7
    if count <= 10:
        return 0.9
    return 1.0


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL 3: Ranking Stability (10% → 12.2% renormalized)
# ══════════════════════════════════════════════════════════════════════════════

def _score_ranking_stability(results: list[dict]) -> float:
    """
    Scores how stable and diverse the ranking is — flags near-duplicates.

    Near-duplicate results (e.g. two listings with almost the same title)
    waste result slots and degrade user experience. This signal penalizes
    result sets where too many titles are similar.

    Lowest weighted (10%) because duplicates are an edge case and the
    downstream reranker further diversifies. This acts as an early warning.

    Logic:
      - Compares titles of top 10 results pairwise using SequenceMatcher
      - Counts pairs exceeding the near_duplicate_threshold (default 0.92)
      - More duplicate pairs = lower score

    Args:
        results: List of search result dicts from state.

    Returns:
        Score between 0.0 and 1.0.
    """
    if len(results) <= 1:
        return 1.0

    titles = []
    for r in results[:10]:
        title = (r.get("title", "") or "").lower().strip()
        if title:
            titles.append(title)

    if len(titles) <= 1:
        return 1.0

    duplicate_pairs = 0
    total_pairs = 0
    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            total_pairs += 1
            similarity = SequenceMatcher(None, titles[i], titles[j]).ratio()
            if similarity > settings.near_duplicate_threshold:
                duplicate_pairs += 1

    if total_pairs == 0:
        return 1.0

    duplicate_ratio = duplicate_pairs / total_pairs
    return max(1.0 - duplicate_ratio, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL 4: Freshness Signal (20% → 24.4% renormalized)
# ══════════════════════════════════════════════════════════════════════════════

def _score_freshness_signal(freshness_metadata: dict) -> float:
    """
    Scores how fresh or stale the search results are.

    PRD Section 4.3 requires staleness awareness in every response.
    For time-sensitive domains (news, products, live inventory), stale
    results degrade user trust. Depending on the catalog, timestamps may be
    historical; tune thresholds via ``.env`` rather than hardcoding a vertical.

    The weight (20%) ensures the system is production-ready for any
    domain where freshness matters — no code change needed, just
    adjust FRESHNESS_THRESHOLD_SECONDS and STALENESS_THRESHOLD_SECONDS
    in .env.

    Logic:
      - No stale results → 1.0
      - Some stale results → proportionally lower
      - All stale → 0.3 (not zero because stale results are still usable)

    Args:
        freshness_metadata: The freshness_metadata dict from state.

    Returns:
        Score between 0.3 and 1.0.
    """
    stale_ids = freshness_metadata.get("stale_result_ids", [])
    if not stale_ids:
        return 1.0

    staleness_ratio = min(len(stale_ids) / 20.0, 1.0)
    return max(1.0 - (staleness_ratio * 0.7), 0.3)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Compute combined quality score
# ══════════════════════════════════════════════════════════════════════════════

def _weights_for_active_schema() -> dict[str, float]:
    """Resolve evaluator weights from active DatasetSchema with safe fallback."""
    try:
        schema = get_schema(settings.dataset_schema)
        return schema.normalized_evaluator_weights()
    except Exception:
        return dict(DEFAULT_WEIGHTS)


def _compute_quality_score(state: dict) -> dict[str, Any]:
    """
    Runs all 4 signals and returns a dict with individual + combined scores.

    Args:
        state: The pipeline state dict.

    Returns:
        Dict with keys: semantic_relevance, result_coverage,
        ranking_stability, freshness_signal, combined.
    """
    results = state.get("search_results", [])
    freshness = state.get("freshness_metadata", {})
    weights = _weights_for_active_schema()

    per_result_relevance = {}
    for r in results:
        if not isinstance(r, dict):
            continue
        rid = str(r.get("id", "")).strip()
        if not rid:
            continue
        per_result_relevance[rid] = round(float(r.get("score", 0.0)), 4)

    scores = {
        "semantic_relevance": _score_semantic_relevance(results),
        "result_coverage": _score_result_coverage(results),
        "ranking_stability": _score_ranking_stability(results),
        "freshness_signal": _score_freshness_signal(freshness),
        "per_result_relevance": per_result_relevance,
    }

    combined = sum(scores[signal] * weights[signal] for signal in SIGNAL_KEYS)
    scores["combined"] = round(combined, 4)
    scores["weights_used"] = {k: round(v, 4) for k, v in weights.items()}

    return scores


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Detect near-duplicate query variants via Jaccard similarity
# ══════════════════════════════════════════════════════════════════════════════

def _is_near_duplicate_variant(
    current_variant: str,
    search_history: list[dict],
) -> bool:
    """Check if the current query variant is a near-duplicate of a previous one.

    Uses token-level Jaccard similarity: ``|intersection| / |union|``.
    No model needed — fast set arithmetic on lowercased word tokens.

    If Jaccard >= ``near_duplicate_threshold`` (default 0.92 from config),
    the variant is considered a near-duplicate and retrying would produce
    the same results.

    Args:
        current_variant: The query variant for the current attempt.
        search_history: List of previous SearchAttempt dicts.

    Returns:
        True if a near-duplicate variant exists in history.
    """
    current_tokens = set(current_variant.lower().split())
    if not current_tokens:
        return False

    threshold = settings.near_duplicate_threshold

    for attempt in search_history:
        if not isinstance(attempt, dict):
            continue
        prev_variant = attempt.get("query_variant", "")
        prev_tokens = set(prev_variant.lower().split())
        if not prev_tokens:
            continue

        intersection = current_tokens & prev_tokens
        union = current_tokens | prev_tokens
        jaccard = len(intersection) / len(union) if union else 0.0

        if jaccard >= threshold:
            return True

    return False


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Build retry prescription
# ══════════════════════════════════════════════════════════════════════════════

def _build_retry_prescription(
    quality_scores: dict[str, float],
    current_strategy: str,
    search_history: list[dict],
) -> RetryPrescription:
    """
    Analyzes which signal was weakest and suggests what to change on retry.

    Logic:
      - semantic_relevance weak → suggest SEMANTIC (better meaning match)
      - result_coverage weak   → suggest HYBRID (broader search)
      - ranking_stability weak → suggest KEYWORD (more precise, less duplication)
      - freshness_signal weak  → no strategy change (can't fix via retrieval)

    Also avoids suggesting a strategy that was already tried.

    Args:
        quality_scores: Dict of individual signal scores.
        current_strategy: The strategy used in this attempt.
        search_history: List of previous SearchAttempt dicts.

    Returns:
        RetryPrescription with suggested changes.
    """
    signal_scores = {
        k: float(quality_scores.get(k, 0.0))
        for k in SIGNAL_KEYS
    }
    weakest = min(signal_scores, key=signal_scores.get)

    tried = {h.get("strategy", "") for h in search_history if isinstance(h, dict)}
    all_strategies = [
        RetrievalStrategy.KEYWORD,
        RetrievalStrategy.SEMANTIC,
        RetrievalStrategy.HYBRID,
    ]
    untried = [s for s in all_strategies if s.value not in tried]

    reason_map = {
        "semantic_relevance": ("LOW_RELEVANCE", "Ranking scores indicate poor semantic match"),
        "result_coverage": ("LOW_RESULT_COUNT", "Too few results returned"),
        "ranking_stability": ("NEAR_DUPLICATE", "Results are too similar to each other"),
        "freshness_signal": ("STALE_RESULTS", "Results are outdated"),
    }

    reason_code, explanation = reason_map.get(weakest, ("QUALITY_LOW", "Overall quality below threshold"))

    strategy_suggestion_map = {
        "semantic_relevance": RetrievalStrategy.SEMANTIC,
        "result_coverage": RetrievalStrategy.HYBRID,
        "ranking_stability": RetrievalStrategy.KEYWORD,
        "freshness_signal": None,
    }

    suggested = strategy_suggestion_map.get(weakest)

    if suggested and suggested.value in tried and untried:
        suggested = untried[0]
    elif suggested and suggested.value in tried:
        suggested = None

    return RetryPrescription(
        reason_code=reason_code,
        suggested_strategy=suggested,
        suggested_query_variant="",
        explanation=f"{explanation} (weakest signal: {weakest}={quality_scores[weakest]:.2f})",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN NODE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def evaluator_node(state: dict) -> dict:
    """
    The evaluator node — scores search quality and decides next action.

    Flow:
      Step 1: Increment iteration counter
      Step 2: Compute 4-signal quality score
      Step 3: Decide: accept / retry / exhausted
      Step 4: If retry → build RetryPrescription and record attempt in search_history
      Step 5: Update state with decision and scores
      Step 6: Log structured node exit + LangWatch annotation

    Decision rules:
      - quality_score >= 0.65           → "accept"  (results are good enough)
      - quality_score < 0.65 AND
        iteration < MAX_SEARCH_ITERATIONS AND
        budget not exceeded             → "retry"   (try a different strategy)
      - Otherwise                       → "exhausted" (send best results forward)

    Args:
        state: The pipeline state dict.

    Returns:
        Dict of only the keys this node changed.
    """
    start = time.perf_counter()
    query_hash = state.get("query_hash", "")

    updates: dict = {}
    new_errors: list[dict] = []
    new_search_history: list[dict] = []

    # ── Step 1: Increment iteration counter ───────────────────────────────
    iteration = state.get("iteration_count", 0) + 1
    updates["iteration_count"] = iteration

    # ── Step 2: Compute quality scores ────────────────────────────────────
    quality_scores = _compute_quality_score(state)
    updates["quality_scores"] = quality_scores
    combined = quality_scores["combined"]

    # ── Step 3: Make the decision ─────────────────────────────────────────
    results = state.get("search_results", [])
    strategy = state.get("retrieval_strategy", "HYBRID")
    budget_ok = state.get("cumulative_token_cost", 0.0) < settings.token_budget_usd
    can_retry = iteration < settings.max_search_iterations and budget_ok

    if combined >= ACCEPT_THRESHOLD:
        decision = "accept"
    elif can_retry and len(results) == 0:
        decision = "retry"
    elif can_retry and combined < ACCEPT_THRESHOLD:
        decision = "retry"
    else:
        decision = "exhausted"

    # ── Step 3b: Near-duplicate guard (PRD 4.5) ──────────────────────────
    # Before committing to a retry, check if the current query variant is
    # a near-duplicate of a previously tried variant.  If so, retrying would
    # produce the same results — force "exhausted" instead.
    intent = state.get("parsed_intent", {})
    entities = intent.get("entities", [])
    query_variant = " ".join(entities) if entities else state.get("query", "")
    search_history = state.get("search_history", [])

    if decision == "retry" and _is_near_duplicate_variant(query_variant, search_history):
        logger.info(
            "near_duplicate_variant_detected",
            extra={"query_variant": query_variant, "action": "forcing exhausted"},
        )
        decision = "exhausted"

    updates["evaluator_decision"] = decision

    # ── Step 4: If retry → build prescription + record attempt ────────────
    if decision == "retry":
        prescription = _build_retry_prescription(
            quality_scores,
            strategy,
            search_history,
        )
        updates["retry_prescription"] = prescription.model_dump()

    new_search_history.append(
        SearchAttempt(
            strategy=RetrievalStrategy(strategy) if strategy in ("KEYWORD", "SEMANTIC", "HYBRID") else RetrievalStrategy.HYBRID,
            query_variant=query_variant,
            quality_score=combined,
            result_count=len(results),
        ).model_dump()
    )

    # ── Step 5: Log exhausted as a warning ────────────────────────────────
    if decision == "exhausted":
        new_errors.append(
            ExtractionError(
                node="evaluator",
                severity=ErrorSeverity.WARNING,
                message=PipelineEvent.ITERATION_LIMIT.value,
                fallback_applied=True,
                fallback_description=(
                    f"Quality score {combined:.2f} is below threshold "
                    f"{ACCEPT_THRESHOLD} after {iteration} iteration(s). "
                    f"Sending best available results forward."
                ),
            ).model_dump()
        )

    # ── Step 6: Log and annotate node exit ────────────────────────────────
    duration_ms = (time.perf_counter() - start) * 1000

    log_node_exit(
        logger, "evaluator", query_hash,
        len(results), strategy, duration_ms, 0.0,
        extra={
            "decision": decision,
            "iteration": iteration,
            "quality_combined": combined,
            "semantic_relevance": quality_scores["semantic_relevance"],
            "result_coverage": quality_scores["result_coverage"],
            "ranking_stability": quality_scores["ranking_stability"],
            "freshness_signal": quality_scores["freshness_signal"],
        },
    )
    annotate_node_span(
        "evaluator", len(results), strategy, duration_ms,
        extra={
            "decision": decision,
            "iteration": iteration,
            "quality_combined": combined,
        },
    )

    if new_errors:
        updates["errors"] = new_errors
    if new_search_history:
        updates["search_history"] = new_search_history

    return updates
