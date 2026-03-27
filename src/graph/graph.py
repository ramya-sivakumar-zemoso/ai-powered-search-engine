"""
LangGraph pipeline for NovaMart AI Search Engine.

What this file does :
  Defines the search pipeline as a graph of 6 nodes.
  Each node is a step in the search process:
    1. query_understander — parse the user's search query
    2. retrieval_router   — decide which search strategy to use
    3. searcher           — run the search on Meilisearch
    4. evaluator          — score the results and decide: accept, retry, or give up
    5. reranker           — re-rank results using AI for better ordering
    6. reporter           — package final results and send to user

  Nodes 1-4 are fully implemented. Nodes 5-6 (reranker, reporter)
  are stubs that will be replaced in later phases.

requirements covered:
  - LangGraph StateGraph with all 6 nodes
  - Loop prevention via iteration_count and evaluator routing
  - LangWatch trace wraps the entire pipeline
"""

from __future__ import annotations

import time
from typing import Any

import langwatch
from langgraph.graph import StateGraph, START, END

from src.nodes.query_understander import query_understander_node
from src.nodes.retrieval_router import retrieval_router_node
from src.nodes.searcher import searcher_node
from src.nodes.evaluator import evaluator_node
from src.utils.langwatch_tracker import annotate_node_span
from src.utils.logger import get_logger, log_node_exit
from src.utils.config import get_settings
from src.utils.state_display import state_delta

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  NODES — implemented nodes are imported, stubs remain for reranker/reporter
# ══════════════════════════════════════════════════════════════════════════════

# query_understander_node — imported from src.nodes.query_understander
# retrieval_router_node   — imported from src.nodes.retrieval_router
# searcher_node           — imported from src.nodes.searcher
# evaluator_node          — imported from src.nodes.evaluator


def reranker_node(state: dict) -> dict:
    """Phase 5 will add: cross-encoder ranking + LLM explanations."""
    start = time.perf_counter()

    duration_ms = (time.perf_counter() - start) * 1000
    log_node_exit(
        logger, "reranker", state.get("query_hash", ""),
        len(state.get("search_results", [])),
        state.get("retrieval_strategy", "HYBRID"), duration_ms, 0.0,
    )
    annotate_node_span(
        "reranker", len(state.get("search_results", [])),
        state.get("retrieval_strategy", "HYBRID"), duration_ms,
    )

    return state


def reporter_node(state: dict) -> dict:
    """Phase 6 will add: final response assembly + freshness report."""
    start = time.perf_counter()

    duration_ms = (time.perf_counter() - start) * 1000
    log_node_exit(
        logger, "reporter", state.get("query_hash", ""),
        len(state.get("reranked_results", state.get("search_results", []))),
        state.get("retrieval_strategy", "HYBRID"), duration_ms, 0.0,
    )
    annotate_node_span(
        "reporter",
        len(state.get("reranked_results", state.get("search_results", []))),
        state.get("retrieval_strategy", "HYBRID"), duration_ms,
    )

    return state


# ══════════════════════════════════════════════════════════════════════════════
#  CONDITIONAL EDGE FUNCTIONS — decide where to go next
# ══════════════════════════════════════════════════════════════════════════════

def route_after_query_understander(state: dict) -> str:
    """
    After query_understander finishes:
      - If injection detected (INJECTION_DETECTED error) → hard exit to reporter
      - Otherwise → proceed to retrieval_router

    PRD Section 4.1: "Hard exits to reporter on unrecoverable injection."
    """
    errors = state.get("errors", [])
    for error in errors:
        msg = error.get("message", "") if isinstance(error, dict) else getattr(error, "message", "")
        if msg == "INJECTION_DETECTED":
            return "reporter"
    return "retrieval_router"


def route_after_searcher(state: dict) -> str:
    """
    After searcher finishes:
      - If hard failure (errors with severity ERROR) → go to reporter (skip evaluator)
      - Otherwise → go to evaluator
    """
    errors = state.get("errors", [])
    for error in errors:
        if isinstance(error, dict) and error.get("severity") == "ERROR":
            return "reporter"
        if hasattr(error, "severity") and str(error.severity) == "ERROR":
            return "reporter"
    return "evaluator"


def route_after_evaluator(state: dict) -> str:
    """
    After evaluator scores the results:
      - "accept"    → results are good enough, send to reranker
      - "retry"     → results are poor, try a different strategy
      - "exhausted" → iteration limit / budget / near-duplicate hit, go to reporter
    """
    decision = state.get("evaluator_decision", "accept")

    if decision == "retry":
        return "retrieval_router"
    elif decision == "exhausted":
        return "reporter"
    else:
        return "reranker"


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH BUILDER — wires all nodes and edges together
# ══════════════════════════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    """
    Constructs the LangGraph StateGraph with all 6 nodes and edges.

    Graph flow:
        query_understander
              ↓ [conditional]
         ↙              ↘ (injection detected)
    retrieval_router     reporter → END
         ↓
       searcher
         ↓ [conditional]
    ↙              ↘ (hard failure)
    evaluator       reporter → END
      ↓ [conditional]
    ↙     ↓          ↘
(retry) reranker   reporter → END (exhausted)
  ↓       ↓
  ↑   reporter → END
  └── retrieval_router (loop back)

    Returns:
        A compiled LangGraph that can be invoked with a state dict.
    """
    # Create the graph using dict state (LangGraph works with dicts internally)
    graph = StateGraph(dict)

    # ── Add all 6 nodes ──────────────────────────────────────────────────────
    graph.add_node("query_understander", query_understander_node)
    graph.add_node("retrieval_router", retrieval_router_node)
    graph.add_node("searcher", searcher_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("reranker", reranker_node)
    graph.add_node("reporter", reporter_node)

    # ── Set the entry point (langgraph 1.x uses START constant) ────────────
    graph.add_edge(START, "query_understander")

    # ── Fixed edges (always go to the next node) ─────────────────────────────
    graph.add_edge("retrieval_router", "searcher")
    graph.add_edge("reranker", "reporter")
    graph.add_edge("reporter", END)

    # ── Conditional edges (decision points) ──────────────────────────────────

    # After query_understander: injection → reporter, otherwise → retrieval_router
    graph.add_conditional_edges(
        "query_understander",
        route_after_query_understander,
        {"retrieval_router": "retrieval_router", "reporter": "reporter"},
    )

    graph.add_conditional_edges(
        "searcher",
        route_after_searcher,
        {"evaluator": "evaluator", "reporter": "reporter"},
    )

    graph.add_conditional_edges(
        "evaluator",
        route_after_evaluator,
        {
            "reranker": "reranker",
            "retrieval_router": "retrieval_router",
            "reporter": "reporter",
        },
    )

    return graph


def compile_graph():
    """Build and compile the graph — ready to be invoked."""
    graph = build_graph()
    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT — wraps graph invocation with LangWatch tracing
# ══════════════════════════════════════════════════════════════════════════════

@langwatch.trace()
def run_search_with_trace(query: str) -> tuple[dict, list[dict]]:
    """
    The main entry point for running a search query through the pipeline.

    What happens:
      1. LangWatch starts a trace (the @langwatch.trace() decorator above)
      2. We build the initial state with the user's query
      3. The compiled graph runs the query through all 6 nodes
      4. LangWatch records the full trace on the dashboard
      5. The final state (with results) is returned

    Args:
        query: The raw search string from the user (e.g. "wireless earbuds under $50")


    Returns:
        ``(final_state, trace)`` where each trace item is
        ``{"node": str, "output": dict}`` — ``output`` is only the keys that
        node added or changed compared to the state before it ran (not the full state).
    """
    logger.info("search_started", extra={"query": query})

    initial_state = {"query": query}
    app = compile_graph()
    trace: list[dict] = []
    final_state: dict | None = None
    # LangGraph emits ``values`` (cumulative state) then ``updates`` (per node). Diff each
    # node output against the snapshot from the prior ``values`` event.
    last_snapshot: dict[str, Any] = dict(initial_state)

    for event in app.stream(initial_state, stream_mode=["updates", "values"]):
        mode, payload = event
        if mode == "values":
            final_state = payload
            last_snapshot = dict(payload)
        elif mode == "updates":
            for node_name, node_output in payload.items():
                diff = state_delta(last_snapshot, node_output)
                trace.append({"node": node_name, "output": diff})

    if final_state is None:
        final_state = {}

    logger.info(
        "search_completed",
        extra={
            "query_hash": final_state.get("query_hash", ""),
            "result_count": len(final_state.get("search_results", [])),
            "iteration_count": final_state.get("iteration_count", 0),
        },
    )

    return final_state, trace


def run_search(query: str) -> dict:
    """
    The main entry point for running a search query through the pipeline.

    Delegates to :func:`run_search_with_trace` (LangWatch-instrumented) and returns
    only the final state — same behavior as before for callers like ``main.py``.

    Args:
        query: The raw search string from the user (e.g. "wireless earbuds under $50")

    Returns:
        The final state dict containing search_results, quality_scores, errors, etc.
    """
    final_state, _ = run_search_with_trace(query)
    return final_state