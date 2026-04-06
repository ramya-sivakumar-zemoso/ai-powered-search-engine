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

from src.models.state import PipelineEvent
from src.utils.logger import get_logger, log_node_exit
from src.utils.langwatch_tracker import annotate_node_span

logger = get_logger(__name__)


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
        if isinstance(r, dict) and r.get("explanation_status") == "DEGRADED":
            return True

    for error in state.get("errors", []):
        if not isinstance(error, dict):
            continue
        msg = str(error.get("message", ""))
        if msg == PipelineEvent.RERANK_DEGRADED.value or "RERANK_DEGRADED" in msg:
            return True

    return False


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

    # ── Step 1: Pick the best available results ───────────────────────────
    results, result_source = _pick_best_results(state)
    blocked = _is_blocked(state)
    partial_results = _is_partial_results(state)
    rerank_degraded = _is_rerank_degraded(state, results)

    # ── Step 2: Build the final response payload ─────────────────────────
    session_id = state.get("session_id", "")

    final_response = {
        "query": state.get("query", ""),
        "query_hash": query_hash,
        "session_id": session_id,
        "blocked": blocked,
        "partial_results": partial_results,
        "rerank_degraded": rerank_degraded,

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
        },

        "warnings": _build_warnings(state),
    }
    final_response["structured_text"] = _build_structured_text_response(final_response)

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
