"""Tests for the evaluator node — scoring signals and decision logic."""

import pytest

from src.nodes.evaluator import (
    _score_semantic_relevance,
    _score_result_coverage,
    _score_ranking_stability,
    _score_freshness_signal,
    _compute_quality_score,
    _build_retry_prescription,
    _is_near_duplicate_variant,
    evaluator_node,
)


# ── Signal 1: Semantic Relevance ────────────────────────────────────────────


@pytest.mark.parametrize("results, expected", [
    ([], 0.0),
    ([{"score": 0.9}, {"score": 0.8}], 0.85),
    ([{"score": 1.0}] * 5, 1.0),
    ([{}], 0.0),
])
def test_semantic_relevance(results, expected):
    assert _score_semantic_relevance(results) == pytest.approx(expected)


# ── Signal 2: Result Coverage ───────────────────────────────────────────────


@pytest.mark.parametrize("count, expected", [
    (0, 0.0), (1, 0.4), (3, 0.7), (7, 0.9), (15, 1.0),
])
def test_result_coverage(count, expected):
    assert _score_result_coverage([{}] * count) == expected


# ── Signal 3: Ranking Stability ─────────────────────────────────────────────


def test_ranking_stability_diverse_titles():
    results = [{"title": t} for t in ("Alien", "Coco", "Frozen", "Dune", "Jaws")]
    assert _score_ranking_stability(results) == 1.0


def test_ranking_stability_identical_titles():
    results = [{"title": "Star Wars"}, {"title": "Star Wars"}]
    assert _score_ranking_stability(results) == 0.0


# ── Signal 4: Freshness ────────────────────────────────────────────────────


@pytest.mark.parametrize("stale_ids, lo, hi", [
    ([], 1.0, 1.0),
    (["1", "2"], 0.9, 1.0),
    ([str(i) for i in range(20)], 0.3, 0.31),
])
def test_freshness_signal(stale_ids, lo, hi):
    score = _score_freshness_signal({"stale_result_ids": stale_ids})
    assert lo <= score <= hi


# ── Combined Quality Score ──────────────────────────────────────────────────


def test_compute_quality_score(results_factory, state_factory):
    state = state_factory(search_results=results_factory(10, score=0.9))
    scores = _compute_quality_score(state)
    expected_keys = {
        "semantic_relevance", "result_coverage",
        "ranking_stability", "freshness_signal", "combined",
        "per_result_relevance", "weights_used",
    }
    assert set(scores.keys()) == expected_keys
    assert 0.0 <= scores["combined"] <= 1.0


# ── Retry Prescription ─────────────────────────────────────────────────────


def test_retry_prescription_targets_weakest():
    scores = {
        "semantic_relevance": 0.9,
        "result_coverage": 0.1,
        "ranking_stability": 0.8,
        "freshness_signal": 0.7,
    }
    rx = _build_retry_prescription(scores, "KEYWORD", [])
    assert rx.reason_code == "LOW_RESULT_COUNT"


# ── Evaluator Node Decisions ───────────────────────────────────────────────


def test_evaluator_accept(results_factory, state_factory):
    state = state_factory(search_results=results_factory(10, score=0.95))
    assert evaluator_node(state)["evaluator_decision"] == "accept"


def test_evaluator_retry(state_factory):
    result = evaluator_node(state_factory())
    assert result["evaluator_decision"] == "retry"
    assert "retry_prescription" in result


def test_evaluator_exhausted(state_factory):
    # After increment: iteration=4 ⇒ retries_so_far=3 ⇒ not < MAX_SEARCH_ITERATIONS (3)
    state = state_factory(iteration_count=3)
    result = evaluator_node(state)
    assert result["evaluator_decision"] == "exhausted"
    assert any(e.get("message") == "ITERATION_LIMIT" for e in result["errors"])


def test_evaluator_budget_exhausted(state_factory):
    state = state_factory(cumulative_token_cost=0.02)
    result = evaluator_node(state)
    assert result["evaluator_decision"] == "exhausted"
    assert any(e.get("message") == "BUDGET_EXCEEDED" for e in result["errors"])


def test_near_duplicate_same_strategy_only():
    hist = [
        {"strategy": "HYBRID", "query_variant": "wireless earbuds pro max"},
    ]
    assert _is_near_duplicate_variant(
        "wireless earbuds pro max",
        "HYBRID",
        hist,
    )
    assert not _is_near_duplicate_variant(
        "wireless earbuds pro max",
        "SEMANTIC",
        hist,
    )
