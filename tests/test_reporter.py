"""Tests for reporter node — result selection, cost summary, and response assembly."""

from src.nodes.reporter import (
    _pick_best_results,
    _build_cost_summary,
    _is_blocked,
    _build_warnings,
    reporter_node,
)


# ── Pick Best Results (priority: reranked > search > none) ─────────────────


def test_pick_reranked_over_search():
    state = {"reranked_results": [{"id": "1"}], "search_results": [{"id": "2"}]}
    results, source = _pick_best_results(state)
    assert source == "reranked" and results[0]["id"] == "1"


def test_pick_search_when_no_reranked():
    state = {"reranked_results": [], "search_results": [{"id": "2"}]}
    results, source = _pick_best_results(state)
    assert source == "search" and results[0]["id"] == "2"


def test_pick_none_when_empty():
    results, source = _pick_best_results({})
    assert source == "none" and results == []


# ── Cost Summary ───────────────────────────────────────────────────────────


def test_build_cost_summary_aggregates():
    state = {"token_usage": [
        {"node": "q_u", "prompt_tokens": 100, "completion_tokens": 50, "cost_usd": 0.001},
        {"node": "router", "prompt_tokens": 200, "completion_tokens": 80, "cost_usd": 0.002},
    ]}
    summary = _build_cost_summary(state)
    assert summary["total_prompt_tokens"] == 300
    assert summary["total_completion_tokens"] == 130
    assert len(summary["per_node"]) == 2


# ── Is Blocked ─────────────────────────────────────────────────────────────


def test_is_blocked_true():
    assert _is_blocked({"errors": [{"message": "INJECTION_DETECTED"}]}) is True


def test_is_blocked_false():
    assert _is_blocked({"errors": []}) is False


# ── Build Warnings ─────────────────────────────────────────────────────────


def test_build_warnings_maps_error_fields():
    state = {"errors": [{
        "severity": "WARNING", "node": "searcher",
        "message": "FILTER_RELAXATION", "fallback_description": "relaxed",
    }]}
    warnings = _build_warnings(state)
    assert len(warnings) == 1
    assert warnings[0]["node"] == "searcher"
    assert warnings[0]["detail"] == "relaxed"


# ── Reporter Node ──────────────────────────────────────────────────────────


def test_reporter_accept_path(state_factory):
    state = state_factory(
        reranked_results=[{"id": "1", "title": "Star Wars"}],
        quality_scores={"combined": 0.8},
    )
    result = reporter_node(state)
    resp = result["final_response"]
    assert resp["result_source"] == "reranked"
    assert resp["result_count"] == 1
    assert resp["blocked"] is False


def test_reporter_blocked_path(state_factory):
    state = state_factory(errors=[{
        "message": "INJECTION_DETECTED", "severity": "ERROR",
        "node": "query_understander", "fallback_description": "blocked",
    }])
    result = reporter_node(state)
    resp = result["final_response"]
    assert resp["blocked"] is True
    assert resp["result_source"] == "none"
    assert resp["result_count"] == 0
    assert len(resp["warnings"]) == 1
