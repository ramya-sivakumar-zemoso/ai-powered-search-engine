"""
Reporter node — assembles the final response payload for the end user.

What this node does:
  1. Picks the best available results (reranked if present, raw search otherwise)
  2. Builds a structured final_response dict containing:
     - results:           ordered result list (reranked or original)
     - result_count:      total number of results
     - quality_summary:   all quality signals from evaluator + reranker
     - freshness_report:  freshness metadata from the searcher (PRD Section 4.3)
     - cost_summary:      total tokens, total cost, per-node breakdown
     - pipeline_metadata: strategy, iterations, evaluator decision, query hash
     - warnings:          any errors/degradation messages
     - blocked:           True if injection was detected
  3. Writes final_response to state for any consumer (API, UI, CLI)

Incoming paths (3 possible routes into reporter):
  - Accept path:    evaluator → reranker → reporter  (has reranked_results)
  - Exhausted path: evaluator → reporter             (search_results only)
  - Error path:     query_understander → reporter     (injection detected)
                    searcher → reporter               (Meilisearch hard failure)

No LLM call here → zero token cost. Pure assembly.
"""
from __future__ import annotations

import time

import src.constants as C
from src.models.state import PipelineEvent
from src.utils import query_catalog_alignment as qca
from src.utils.config import get_settings
from src.utils.query_word_limit import query_word_limit_user_notice
from src.utils.logger import get_logger, log_node_exit
from src.utils.langwatch_tracker import annotate_node_span

logger = get_logger(__name__)

_RESULT_QUALITY_NEAREST_PICKS = (
    "It looks like we don't have exactly what you're looking for right now. Check out these related picks, or try a new search for more results."
)


def _error_messages(state: dict) -> list[str]:
    out: list[str] = []
    for error in state.get("errors", []):
        if isinstance(error, dict):
            out.append(str(error.get("message", "")))
    return out


def _float_q(qs: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(qs.get(key, default))
    except (TypeError, ValueError):
        return default


def _max_top_signal(
    results: list,
    result_source: str,
    *,
    top_n: int = 5,
) -> tuple[float, float]:
    """Max confidence (reranked) or max Meilisearch score among first ``top_n`` hits."""
    max_conf = 0.0
    max_score = 0.0
    for r in results[:top_n]:
        if not isinstance(r, dict):
            continue
        if result_source == "reranked":
            try:
                max_conf = max(
                    max_conf,
                    float(r.get("confidence", r.get("relevance_score", 0.0))),
                )
            except (TypeError, ValueError):
                pass
            try:
                max_score = max(
                    max_score,
                    float(r.get("meilisearch_ranking_score", r.get("score", 0.0))),
                )
            except (TypeError, ValueError):
                pass
        else:
            try:
                max_score = max(max_score, float(r.get("score", 0.0)))
            except (TypeError, ValueError):
                pass
    return max_conf, max_score


def _mean_rerank_confidence_head(results: list, *, top_n: int = 5) -> float:
    """Mean cross-encoder confidence over the first ``top_n`` reranked rows (visible SERP).

    ``quality_scores["rerank_confidence"]`` is averaged over *all* reranked rows; tail hits
    with higher scores can lift the global mean above the ambiguous band while the head rows
    the user actually sees are still near ~0.5 (*groceries* vs unrelated catalog neighbors).
    """
    confs: list[float] = []
    for r in results[:top_n]:
        if not isinstance(r, dict):
            continue
        try:
            confs.append(float(r.get("confidence", r.get("relevance_score", 0.0))))
        except (TypeError, ValueError):
            continue
    if not confs:
        return 0.0
    return sum(confs) / len(confs)


def _results_plausibly_on_topic(
    state: dict,
    results: list,
    result_source: str,
) -> bool:
    """True when evaluator / reranker / listing scores suggest hits are at least weakly relevant.

    Avoids false positives: ``partial_results`` / semantic-degradation can be set when an
    internal stem-overlap check fails, even though vector + reranker scores still show a
    reasonable fit (e.g. *fragrance products* vs *perfume* titles).
    """
    qs = state.get("quality_scores") or {}
    if not isinstance(qs, dict):
        qs = {}

    combined = _float_q(qs, "combined", 0.0)
    sem = _float_q(qs, "semantic_relevance", 0.0)
    r_mean = _float_q(qs, "rerank_confidence", 0.0)
    max_conf, max_score = _max_top_signal(results, result_source)

    # Evaluator already saw final ``search_results`` scores — strong combined = good enough.
    if combined >= 0.60:
        return True
    # Average Meilisearch ranking signal across top hits was not terrible.
    if sem >= 0.34:
        return True
    # Reranker is confident about the pool or the best hit.
    if result_source == "reranked":
        if r_mean >= 0.30 or max_conf >= 0.35:
            return True
        if max_score >= 0.28:
            return True
    else:
        if max_score >= 0.26:
            return True

    return False


def _scores_indicate_strong_fit(
    state: dict,
    results: list,
    result_source: str,
    *,
    trust_cross_encoder: bool = True,
) -> bool:
    """Stricter than :func:`_results_plausibly_on_topic` — clears weak-token warnings when scores are clearly good.

    When ``trust_cross_encoder`` is False (query words missing from visible hit text), we **do
    not** treat reranker mean/max confidence as proof of a good match — the cross-encoder often
    scores same-aisle catalog neighbors too high (*flying robot* vs generic toys).
    """
    qs = state.get("quality_scores") or {}
    if not isinstance(qs, dict):
        qs = {}
    combined = _float_q(qs, "combined", 0.0)
    sem = _float_q(qs, "semantic_relevance", 0.0)

    # When query terms are missing from visible hit text, Meilisearch `_rankingScore` and
    # reranker scores can still look "fine" for same-broad-category neighbors — do not use
    # ``max_score`` / cross-encoder here; rely on evaluator blend only.
    if not trust_cross_encoder:
        if combined >= 0.70:
            return True
        if sem >= 0.52:
            return True
        if combined >= 0.64 and sem >= 0.48:
            return True
        return False

    r_mean = _float_q(qs, "rerank_confidence", 0.0)
    max_conf, max_score = _max_top_signal(results, result_source)

    if combined >= 0.66:
        return True
    if sem >= 0.46:
        return True
    if result_source == "reranked":
        if max_conf >= 0.50 or r_mean >= 0.43:
            return True
        if max_score >= 0.42:
            return True
    else:
        if max_score >= 0.33:
            return True
    return False


# Phrases in VERIFIED reranker explanations that explicitly say a hit is a poor query match.
_EXPL_DISCLAIM_RELEVANCE_PHRASES: tuple[str, ...] = (
    "not relevant",
    "is not relevant",
    "isn't relevant",
    "not related to",
    "does not match",
    "doesn't match",
    "no connection to",
)


def _verified_explanations_disclaim_relevance(results: list, *, top_n: int = 8) -> bool:
    """True when several top hits carry audited explanations that say they do not match the query."""
    eligible = 0
    disclaim = 0
    for r in results[:top_n]:
        if not isinstance(r, dict):
            continue
        if str(r.get("explanation_status", "") or "") != "VERIFIED":
            continue
        expl = (r.get("explanation") or "").strip()
        if not expl:
            continue
        eligible += 1
        low = expl.lower()
        if any(p in low for p in _EXPL_DISCLAIM_RELEVANCE_PHRASES):
            disclaim += 1
    if eligible == 0:
        return False
    if disclaim >= 2:
        return True
    if eligible <= 3 and disclaim == eligible:
        return True
    if eligible >= 2 and disclaim >= eligible * 0.5:
        return True
    return False


def _rerank_pool_suggests_irrelevance(
    state: dict,
    result_source: str,
    results: list,
) -> bool:
    """True when reranker signals (scores and/or audited explanations) say the pool is a poor fit.

    Cross-encoder logits often sit near **0.5** for unrelated catalog neighbors, so the mean
    ``rerank_confidence`` alone can look “fine” while LLM explanations still VERIFIED‑say the hit
    is not relevant (e.g. *groceries* vs dish racks). Combine both.
    """
    if result_source != "reranked" or not results:
        return False
    qs = state.get("quality_scores") or {}
    if not isinstance(qs, dict):
        return False
    mean_r = _float_q(qs, "rerank_confidence", 1.0)
    low_ratio = _float_q(qs, "rerank_low_confidence_ratio", 0.0)
    mean_head = _mean_rerank_confidence_head(results, top_n=5)
    disclaim = _verified_explanations_disclaim_relevance(results)

    # Audited LLM explanations can disagree with a tail-inflated global CE mean — use SERP head.
    if disclaim and (mean_r <= 0.56 or mean_head < 0.58):
        return True
    if mean_r >= 0.58:
        if disclaim and mean_head < 0.62:
            return True
        return False

    if mean_r < C.RERANK_NOTICE_MAX_MEAN_CONFIDENCE:
        # Aggregate mean on a single-hit list is often the same signal as per-hit confidence;
        # let the max-confidence / semantic gates handle singletons to avoid double warnings.
        if len(results) < 2:
            return False
        return True
    if low_ratio >= C.RERANK_NOTICE_MIN_LOW_CONF_RATIO:
        return True
    return False


def _result_quality_notice(
    state: dict,
    *,
    results: list,
    result_source: str,
    partial_results: bool,
    blocked: bool,
) -> str | None:
    """End-user copy when results are likely a poor fit for the query (not only technical fallback)."""
    if blocked:
        return None

    messages = _error_messages(state)
    if PipelineEvent.QUERY_WORD_LIMIT.value in messages:
        return query_word_limit_user_notice(get_settings().max_query_words)

    if not results:
        return None

    degraded_retrieval = partial_results or any(
        m == "SEMANTIC_DEGRADATION_FALLBACK" for m in messages
    )
    if degraded_retrieval:
        # Fallback flags alone are not enough — confirm hits still look weak.
        if not _results_plausibly_on_topic(state, results, result_source):
            return (
                "No exact match. Here are the closest results we could find—try simpler or different "
                "words if these miss the mark."
            )
        # Scores look fine to the evaluator (e.g. vector similarity on unrelated catalog rows),
        # but do not exit early: reranker VERIFIED explanations can still say hits are not
        # relevant (groceries + SEMANTIC_DEGRADATION_FALLBACK + high combined).

    q_for_ux = (state.get("sanitized_query") or state.get("query") or "").strip()
    if q_for_ux and (
        qca.query_has_absurd_numeric_literal(q_for_ux)
        or qca.query_is_ultra_vague_lexical(q_for_ux)
    ):
        return _RESULT_QUALITY_NEAREST_PICKS

    # Searcher-side signal: Meilisearch returned mid-confidence neighbors vs literal query text.
    if state.get("retrieval_soft_match") and not _scores_indicate_strong_fit(
        state,
        results,
        result_source,
        trust_cross_encoder=False,
    ):
        return _RESULT_QUALITY_NEAREST_PICKS

    # Reranker aggregate: weak mean or many low-confidence hits, while no single standout result.
    disclaim = _verified_explanations_disclaim_relevance(results)
    pool_bad = _rerank_pool_suggests_irrelevance(state, result_source, results)
    if disclaim or pool_bad:
        # Do not let one high CE score in positions 3–5 suppress audited "not relevant" copy.
        if disclaim:
            return _RESULT_QUALITY_NEAREST_PICKS
        max_conf, _ = _max_top_signal(results, result_source)
        if max_conf < C.RERANK_NOTICE_TOP_CONFIDENCE_ESCAPE:
            return _RESULT_QUALITY_NEAREST_PICKS

    # Vector / rerank scores can stay "okay" for the wrong category (e.g. *flying robot* → toys).
    # If important query words never appear in top hits *and* scores are not clearly strong,
    # treat as a poor user-visible match (while synonym cases like *fragrance* → perfume stay
    # covered by strong scores or token hits like "products" in copy).
    tokens = qca.content_tokens_from_state(state)
    if tokens:
        coverage = qca.query_token_coverage_in_pipeline_results(tokens, results, state)
        if coverage < C.RETRIEVAL_WEAK_TOKEN_COVERAGE and not _scores_indicate_strong_fit(
            state,
            results,
            result_source,
            trust_cross_encoder=False,
        ):
            return _RESULT_QUALITY_NEAREST_PICKS

    qs = state.get("quality_scores") or {}
    try:
        sem = float(qs.get("semantic_relevance", 1.0))
    except (TypeError, ValueError):
        sem = 1.0
    try:
        combined = float(qs.get("combined", 1.0))
    except (TypeError, ValueError):
        combined = 1.0

    confs: list[float] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        if result_source == "reranked":
            raw = r.get("confidence", r.get("relevance_score", 0.0))
        else:
            raw = r.get("score", 0.0)
        try:
            confs.append(float(raw))
        except (TypeError, ValueError):
            confs.append(0.0)

    max_score = max(confs) if confs else 0.0
    low_rerank = result_source == "reranked" and max_score < 0.36
    low_search = result_source == "search" and max_score < 0.28
    low_sem = sem < 0.42

    if low_sem and (low_rerank or low_search or combined < 0.58):
        # Evaluator semantic can dip while titles still literally contain the query (*flying robot*).
        tokens = qca.content_tokens_from_state(state)
        if tokens:
            cov = qca.query_token_coverage_in_pipeline_results(tokens, results, state)
            if cov < C.RETRIEVAL_WEAK_TOKEN_COVERAGE:
                return _RESULT_QUALITY_NEAREST_PICKS
        else:
            return _RESULT_QUALITY_NEAREST_PICKS

    if low_rerank or low_search:
        return (
            "Hard to say if these fit your search. Try different words and search again if they "
            "don't help."
        )

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Build the final result list from the best available source
# ══════════════════════════════════════════════════════════════════════════════

def _pick_best_results(state: dict) -> tuple[list[dict], str]:
    """Choose the best available results and label the source.

    Priority:
      1. reranked_results (from reranker — cross-encoder scored + explained)
      2. search_results   (from searcher — Meilisearch raw results)
      3. empty list       (injection blocked or hard failure)

    Returns:
        (results_list, source_label)
    """
    reranked = state.get("reranked_results", [])
    if reranked:
        return reranked, "reranked"

    search = state.get("search_results", [])
    if search:
        return search, "search"

    return [], "none"


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Build cost summary from per-node token usage
# ══════════════════════════════════════════════════════════════════════════════

def _build_cost_summary(state: dict) -> dict:
    """Aggregate token usage across all nodes into a cost summary.

    Returns dict with:
      - total_prompt_tokens, total_completion_tokens, total_cost_usd
      - per_node: list of {node, prompt_tokens, completion_tokens, cost_usd}
    """
    token_usage = state.get("token_usage", [])

    total_prompt = 0
    total_completion = 0
    total_cost = 0.0
    per_node = []

    for entry in token_usage:
        if isinstance(entry, dict):
            p = entry.get("prompt_tokens", 0)
            c = entry.get("completion_tokens", 0)
            cost = entry.get("cost_usd", 0.0)
            total_prompt += p
            total_completion += c
            total_cost += cost
            per_node.append({
                "node": entry.get("node", "unknown"),
                "prompt_tokens": p,
                "completion_tokens": c,
                "cost_usd": round(cost, 8),
            })

    return {
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_cost_usd": round(total_cost, 8),
        "per_node": per_node,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Check if the query was blocked by injection detection
# ══════════════════════════════════════════════════════════════════════════════

def _is_blocked(state: dict) -> bool:
    """Return True if an INJECTION_DETECTED error exists in state."""
    for error in state.get("errors", []):
        msg = error.get("message", "") if isinstance(error, dict) else ""
        if msg == "INJECTION_DETECTED":
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Build warnings list from errors
# ══════════════════════════════════════════════════════════════════════════════

def _build_warnings(state: dict) -> list[dict]:
    """Convert state errors into a clean warnings list for the response.

    Each warning has: severity, node, message, fallback_description.
    """
    warnings = []
    for error in state.get("errors", []):
        if isinstance(error, dict):
            warnings.append({
                "severity": error.get("severity", "WARNING"),
                "node": error.get("node", "unknown"),
                "message": error.get("message", ""),
                "detail": error.get("fallback_description", ""),
            })
    return warnings


def _build_structured_text_response(final_response: dict) -> str:
    """Create a human-readable text view of the final response."""
    lines: list[str] = []
    lines.append(f"Query: {final_response.get('query', '')}")
    lines.append(f"Query Hash: {final_response.get('query_hash', '')}")
    lines.append(f"Session ID: {final_response.get('session_id', '')}")
    lines.append(f"Blocked: {final_response.get('blocked', False)}")
    lines.append(f"Partial Results: {final_response.get('partial_results', False)}")
    lines.append(f"Rerank Degraded: {final_response.get('rerank_degraded', False)}")
    nq = final_response.get("result_quality_notice")
    if nq:
        lines.append(f"Result quality notice: {nq}")
    lines.append(f"Result Count: {final_response.get('result_count', 0)}")
    lines.append(f"Result Source: {final_response.get('result_source', 'none')}")
    lines.append("")
    lines.append("Top Results:")

    results = final_response.get("results", [])
    for i, r in enumerate(results[:5], start=1):
        title = r.get("title") or "(untitled)"
        rid = r.get("id", "")
        conf = r.get("confidence")
        if conf is None:
            lines.append(f"{i}. [{rid}] {title}")
        else:
            lines.append(f"{i}. [{rid}] {title} (confidence={conf})")

    warnings = final_response.get("warnings", [])
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in warnings:
            lines.append(
                f"- {w.get('severity', 'WARNING')} {w.get('node', '')}: "
                f"{w.get('message', '')}"
            )

    return "\n".join(lines)


def _is_partial_results(state: dict) -> bool:
    """Return True when retrieval degraded and response is partial."""
    if bool(state.get("partial_results", False)):
        return True

    for error in state.get("errors", []):
        if not isinstance(error, dict):
            continue
        if error.get("message") in (
            PipelineEvent.PARTIAL_RESULTS.value,
            "HYBRID_FALLBACK",
        ):
            return True

    return any(
        isinstance(r, dict) and bool(r.get("is_fallback", False))
        for r in state.get("search_results", [])
    )


def _is_rerank_degraded(state: dict, results: list[dict]) -> bool:
    """Return True when reranker output is degraded and should be surfaced."""
    if bool(state.get("rerank_degraded", False)):
        return True

    for r in results:
        if not isinstance(r, dict):
            continue
        if r.get("explanation_status") in ("DEGRADED", "EXPLANATION_UNVERIFIED"):
            return True

    for error in state.get("errors", []):
        if not isinstance(error, dict):
            continue
        msg = str(error.get("message", ""))
        if msg == PipelineEvent.RERANK_DEGRADED.value or "RERANK_DEGRADED" in msg:
            return True

    return False


def assemble_final_response(state: dict) -> dict:
    """Build the final response payload from the current pipeline state."""
    # ── Step 1: Pick the best available results ───────────────────────────
    results, result_source = _pick_best_results(state)
    blocked = _is_blocked(state)
    partial_results = _is_partial_results(state)
    rerank_degraded = _is_rerank_degraded(state, results)
    session_id = state.get("session_id", "")
    strategy = state.get("retrieval_strategy", "HYBRID")
    explanation_status = state.get("explanation_job_status", "")
    explanations_pending = bool(state.get("explanations_pending", False))
    explanation_top_k = int(state.get("explanation_top_k", 0) or 0)
    explanation_job_id = state.get("explanation_job_id", "")

    settings = get_settings()
    meili_index = (state.get("meili_index_name") or "").strip() or settings.meili_index_name
    result_quality_notice = _result_quality_notice(
        state,
        results=results,
        result_source=result_source,
        partial_results=partial_results,
        blocked=blocked,
    )

    final_response = {
        "query": state.get("query", ""),
        "query_hash": state.get("query_hash", ""),
        "session_id": session_id,
        "blocked": blocked,
        "partial_results": partial_results,
        "rerank_degraded": rerank_degraded,
        "result_quality_notice": result_quality_notice,
        "results": results,
        "result_count": len(results),
        "result_source": result_source,
        "quality_summary": state.get("quality_scores", {}),
        "freshness_report": state.get("freshness_metadata", {}),
        "cost_summary": _build_cost_summary(state),
        "pipeline_metadata": {
            "session_id": session_id,
            "strategy": strategy,
            "iterations": state.get("iteration_count", 0),
            "evaluator_decision": state.get("evaluator_decision", "N/A"),
            "filter_relaxation_applied": state.get("filter_relaxation_applied", False),
            "router_reasoning": state.get("router_reasoning", ""),
            "meili_index_name": meili_index,
            "retrieval_soft_match": bool(state.get("retrieval_soft_match", False)),
            "explanations_pending": explanations_pending,
            "explanations_async": bool(state.get("explanations_async", False)),
            "explanation_status": explanation_status,
            "explanation_top_k": explanation_top_k,
        },
        "explanations": {
            "pending": explanations_pending,
            "async": bool(state.get("explanations_async", False)),
            "status": explanation_status,
            "job_id": explanation_job_id,
            "top_k": explanation_top_k,
        },
        "warnings": _build_warnings(state),
    }
    final_response["structured_text"] = _build_structured_text_response(final_response)
    return final_response


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN NODE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def reporter_node(state: dict) -> dict:
    """Assemble the final response payload from all pipeline outputs.

    Pipeline position: always the last node before END.

    This node never fails — it packages whatever the pipeline produced
    (full results, degraded results, or a blocked-query message) into
    a structured final_response dict.

    Args:
        state: The pipeline state dict.

    Returns:
        Dict of only the keys this node changed.
    """
    start = time.perf_counter()
    query_hash = state.get("query_hash", "")
    strategy = state.get("retrieval_strategy", "HYBRID")
    final_response = assemble_final_response(state)
    results = final_response.get("results", [])
    result_source = final_response.get("result_source", "none")
    blocked = bool(final_response.get("blocked", False))
    partial_results = bool(final_response.get("partial_results", False))
    rerank_degraded = bool(final_response.get("rerank_degraded", False))

    # ── Step 3: Log and annotate node exit ────────────────────────────────
    duration_ms = (time.perf_counter() - start) * 1000
    token_cost = state.get("cumulative_token_cost", 0.0)

    log_node_exit(
        logger, "reporter", query_hash,
        len(results), strategy, duration_ms, token_cost,
        extra={
            "result_source": result_source,
            "blocked": blocked,
            "partial_results": partial_results,
            "rerank_degraded": rerank_degraded,
            "warning_count": len(final_response["warnings"]),
        },
    )
    annotate_node_span(
        "reporter", len(results), strategy, duration_ms,
        extra={
            "result_source": result_source,
            "blocked": blocked,
        },
    )

    return {"final_response": final_response}
