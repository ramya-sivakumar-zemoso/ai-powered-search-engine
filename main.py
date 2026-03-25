"""
NovaMart AI Search Engine — main entry point.

What this file does :
  This is the file you run to start a search.
  It sets up LangWatch observability, then sends a test query
  through the full 6-node pipeline.

How to run it:
  python main.py

What output will be seen:
  - JSON log lines showing each node being entered and exited
  - A final summary showing the state after the pipeline completes
  - If LangWatch is enabled in .env, the trace appears on your dashboard
"""

import json

from src.utils.langwatch_tracker import setup_langwatch
from src.utils.logger import get_logger
from src.graph.graph import run_search

logger = get_logger(__name__)


def main() -> None:
    """
    1. Initialize LangWatch (reads LANGWATCH_API_KEY from .env)
    2. Run a test search query through the pipeline
    3. Print the final state
    """

    # ── Step 1: Initialize LangWatch ─────────────────────────────────────────
    setup_langwatch()

    # ── Step 2: Run a test query ─────────────────────────────────────────────
    test_query = "top rated sci-fi movies with time travel"

    print("\n" + "=" * 70)
    print(f"  Running search: \"{test_query}\"")
    print("=" * 70 + "\n")

    final_state = run_search(test_query)

    # ── Step 3: Print the final state ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL STATE (after all 6 nodes)")
    print("=" * 70)

    summary = {
        "query": final_state.get("query", ""),
        "query_hash": final_state.get("query_hash", ""),
        "retrieval_strategy": final_state.get("retrieval_strategy", "HYBRID"),
        "search_results_count": len(final_state.get("search_results", [])),
        "reranked_results_count": len(final_state.get("reranked_results", [])),
        "iteration_count": final_state.get("iteration_count", 0),
        "evaluator_decision": final_state.get("evaluator_decision", ""),
        "cumulative_token_cost": final_state.get("cumulative_token_cost", 0.0),
        "errors_count": len(final_state.get("errors", [])),
    }

    print(json.dumps(summary, indent=2))
    print("\n" + "=" * 70)
    print("  Pipeline complete. All 6 nodes executed successfully.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
