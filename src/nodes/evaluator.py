"""
Evaluator node — scores search result quality and decides accept / retry / exhausted.

What this node does (plain English):
  1. Takes the search_results from the searcher node
  2. Scores them on 4 signals (result count, score quality, freshness, diversity)
  3. Combines signals into a single quality_score (0.0 to 1.0)
  4. Decides:
     - "accept"    → results are good enough, send to reranker
     - "retry"     → results are poor, send back to router with a prescription
     - "exhausted" → we've hit the iteration limit or budget, send best results forward

Why 4 signals instead of 5:
  The original design had a 5th signal (confidence from the reranker's
  cross-encoder). But the evaluator runs BEFORE the reranker, so confidence
  isn't available yet. We drop it and renormalize the remaining weights:
    30% + 22% + 10% + 20% = 82% → renormalized to 100%
    Result count: 36.6%, Score quality: 26.8%, Freshness: 12.2%, Diversity: 24.4%
  The accept threshold drops from 0.72 to 0.65 to compensate.

No LLM call here → zero token cost. Pure Python scoring.
"""
from __future__ import annotations

import time
from difflib import SequenceMatcher

from src.models.state import (
    RetryPrescription,
    RetrievalStrategy,
    SearchAttempt,
    ExtractionError,
    ErrorSeverity,
)
from src.utils.config import get_settings
from src.utils.logger import get_logger, log_node_exit
from src.utils.langwatch_tracker import annotate_node_span

logger = get_logger(__name__)
settings = get_settings()

# Quality threshold: results scoring above this are accepted.
# Lower than the original 0.72 because we score on 4 signals (no confidence).
ACCEPT_THRESHOLD = 0.65

# Signal weights — renormalized from original 5-signal plan after dropping
# the confidence signal (which comes from reranker, not available here).
# Original: count=30%, quality=22%, confidence=18%, freshness=10%, diversity=20%
# After drop: 30+22+10+20 = 82% → renormalize to 100%
WEIGHTS = {
    "result_count": 0.366,
    "score_quality": 0.268,
    "freshness": 0.122,
    "diversity": 0.244,
}


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL 1: Result Count
# ══════════════════════════════════════════════════════════════════════════════

def _score_result_count(results: list[dict]) -> float:
    """
    Scores based on how many results we got.

    Logic:
      0 results     → 0.0  (nothing found)
      1-2 results   → 0.4  (too few)
      3-5 results   → 0.7  (acceptable)
      6-10 results  → 0.9  (good)
      11+ results   → 1.0  (plenty)

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
#  SIGNAL 2: Score Quality (ranking scores from Meilisearch)
# ══════════════════════════════════════════════════════════════════════════════

def _score_quality(results: list[dict]) -> float:
    """
    Scores based on the Meilisearch ranking scores of the results.

    Logic:
      - Looks at the average _rankingScore of the top 5 results
      - Higher average = better quality match

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
#  SIGNAL 3: Freshness (are results stale?)
# ══════════════════════════════════════════════════════════════════════════════

def _score_freshness(freshness_metadata: dict) -> float:
    """
    Scores based on how fresh the results are.

    Logic:
      - No stale results → 1.0
      - Some stale results → proportionally lower
      - All stale → 0.3 (not zero because stale results are still usable)

    For a movie dataset, most results will be "stale" by timestamp since
    release dates are historical. This signal becomes more meaningful
    for live/news/product datasets.

    Args:
        freshness_metadata: The freshness_metadata dict from state.

    Returns:
        Score between 0.3 and 1.0.
    """
    stale_ids = freshness_metadata.get("stale_result_ids", [])
    if not stale_ids:
        return 1.0

    # Mild penalty — stale results are usable but not ideal
    staleness_ratio = min(len(stale_ids) / 20.0, 1.0)
    return max(1.0 - (staleness_ratio * 0.7), 0.3)


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL 4: Diversity (are results too similar to each other?)
# ══════════════════════════════════════════════════════════════════════════════

def _score_diversity(results: list[dict]) -> float:
    """
    Scores how diverse the results are (not all the same movie/genre).

    Logic:
      - Compares titles of top results pairwise using string similarity
      - High similarity between many results → low diversity score
      - All unique titles → high diversity score

    Args:
        results: List of search result dicts from state.

    Returns:
        Score between 0.0 and 1.0.
    """
    if len(results) <= 1:
        return 1.0  # single result is "diverse" by definition

    titles = []
    for r in results[:10]:
        title = (r.get("title", "") or "").lower().strip()
        if title:
            titles.append(title)

    if len(titles) <= 1:
        return 1.0

    # Count how many title pairs are too similar (> 0.8 similarity)
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
#  HELPER: Compute combined quality score
# ══════════════════════════════════════════════════════════════════════════════

def _compute_quality_score(state: dict) -> dict[str, float]:
    """
    Runs all 4 signals and returns a dict with individual + combined scores.

    Args:
        state: The pipeline state dict.

    Returns:
        Dict with keys: result_count, score_quality, freshness, diversity, combined.
    """
    results = state.get("search_results", [])
    freshness = state.get("freshness_metadata", {})

    scores = {
        "result_count": _score_result_count(results),
        "score_quality": _score_quality(results),
        "freshness": _score_freshness(freshness),
        "diversity": _score_diversity(results),
    }

    # Weighted sum (renormalized 4-signal weights)
    combined = sum(scores[signal] * WEIGHTS[signal] for signal in WEIGHTS)
    scores["combined"] = round(combined, 4)

    return scores


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
      - If result_count is the weakest → suggest HYBRID (broader search)
      - If score_quality is weak → suggest SEMANTIC (better meaning match)
      - If freshness is weak → no strategy change (can't fix via retrieval)
      - If diversity is weak → suggest KEYWORD (more precise matching)

    Also avoids suggesting a strategy that was already tried.

    Args:
        quality_scores: Dict of individual signal scores.
        current_strategy: The strategy used in this attempt.
        search_history: List of previous SearchAttempt dicts.

    Returns:
        RetryPrescription with suggested changes.
    """
    # Find the weakest signal (excluding combined)
    signal_scores = {k: v for k, v in quality_scores.items() if k != "combined"}
    weakest = min(signal_scores, key=signal_scores.get)

    # Strategies already tried
    tried = {h.get("strategy", "") for h in search_history if isinstance(h, dict)}
    all_strategies = [
        RetrievalStrategy.KEYWORD,
        RetrievalStrategy.SEMANTIC,
        RetrievalStrategy.HYBRID,
    ]
    untried = [s for s in all_strategies if s.value not in tried]

    reason_map = {
        "result_count": ("LOW_RESULT_COUNT", "Too few results returned"),
        "score_quality": ("LOW_RELEVANCE", "Ranking scores are too low"),
        "freshness": ("STALE_RESULTS", "Results are outdated"),
        "diversity": ("NEAR_DUPLICATE", "Results are too similar to each other"),
    }

    reason_code, explanation = reason_map.get(weakest, ("QUALITY_LOW", "Overall quality below threshold"))

    # Pick a suggested strategy
    strategy_suggestion_map = {
        "result_count": RetrievalStrategy.HYBRID,
        "score_quality": RetrievalStrategy.SEMANTIC,
        "freshness": None,  # freshness can't be fixed by changing strategy
        "diversity": RetrievalStrategy.KEYWORD,
    }

    suggested = strategy_suggestion_map.get(weakest)

    # If the suggested strategy was already tried, pick an untried one
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
        Updated state dict with evaluator_decision, quality_scores, etc.
    """
    start = time.perf_counter()
    query_hash = state.get("query_hash", "")

    # ── Step 1: Increment iteration counter ───────────────────────────────
    iteration = state.get("iteration_count", 0) + 1
    state["iteration_count"] = iteration

    # ── Step 2: Compute quality scores ────────────────────────────────────
    quality_scores = _compute_quality_score(state)
    state["quality_scores"] = quality_scores
    combined = quality_scores["combined"]

    # ── Step 3: Make the decision ─────────────────────────────────────────
    results = state.get("search_results", [])
    strategy = state.get("retrieval_strategy", "HYBRID")
    budget_ok = state.get("cumulative_token_cost", 0.0) < settings.token_budget_usd
    can_retry = iteration < settings.max_search_iterations and budget_ok

    if combined >= ACCEPT_THRESHOLD:
        decision = "accept"
    elif can_retry and len(results) == 0:
        # Zero results is always worth retrying if we can
        decision = "retry"
    elif can_retry and combined < ACCEPT_THRESHOLD:
        decision = "retry"
    else:
        decision = "exhausted"

    state["evaluator_decision"] = decision

    # ── Step 4: If retry → build prescription + record attempt ────────────
    if decision == "retry":
        prescription = _build_retry_prescription(
            quality_scores,
            strategy,
            state.get("search_history", []),
        )
        state["retry_prescription"] = prescription.model_dump()

    # Record this attempt in search_history (for near-duplicate detection)
    history = state.get("search_history", [])
    intent = state.get("parsed_intent", {})
    entities = intent.get("entities", [])
    query_variant = " ".join(entities) if entities else state.get("query", "")

    history.append(
        SearchAttempt(
            strategy=RetrievalStrategy(strategy) if strategy in ("KEYWORD", "SEMANTIC", "HYBRID") else RetrievalStrategy.HYBRID,
            query_variant=query_variant,
            quality_score=combined,
            result_count=len(results),
        ).model_dump()
    )
    state["search_history"] = history

    # ── Step 5: Log exhausted as a warning ────────────────────────────────
    if decision == "exhausted":
        errors = state.get("errors", [])
        errors.append(
            ExtractionError(
                node="evaluator",
                severity=ErrorSeverity.WARNING,
                message="ITERATIONS_EXHAUSTED",
                fallback_applied=True,
                fallback_description=(
                    f"Quality score {combined:.2f} is below threshold "
                    f"{ACCEPT_THRESHOLD} after {iteration} iteration(s). "
                    f"Sending best available results forward."
                ),
            ).model_dump()
        )
        state["errors"] = errors

    # ── Step 6: Log and annotate node exit ────────────────────────────────
    duration_ms = (time.perf_counter() - start) * 1000

    log_node_exit(
        logger, "evaluator", query_hash,
        len(results), strategy, duration_ms, 0.0,
        extra={
            "decision": decision,
            "iteration": iteration,
            "quality_combined": combined,
            "quality_result_count": quality_scores["result_count"],
            "quality_score_quality": quality_scores["score_quality"],
            "quality_freshness": quality_scores["freshness"],
            "quality_diversity": quality_scores["diversity"],
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

    return state
