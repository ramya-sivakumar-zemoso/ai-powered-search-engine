"""Tests for reranker node — citation audit, context builder, and node paths."""

import pytest

from src.nodes.reranker import _audit_citation, _build_results_context, reranker_node
from src.models.state import ExplanationStatus


# ── Citation Audit (4 statuses) ────────────────────────────────────────────


def test_audit_absent_when_no_explanation():
    status, ids = _audit_citation("", {"id": "1"}, 0.9)
    assert status == ExplanationStatus.ABSENT
    assert ids == []


def test_audit_degraded_when_low_confidence():
    status, ids = _audit_citation("some text", {"id": "1"}, 0.1)
    assert status == ExplanationStatus.DEGRADED
    assert "1" in ids


def test_audit_verified_by_title():
    result = {
        "id": "1",
        "title": "Star Wars Episode",
        "source_fields": {},
    }
    status, ids = _audit_citation(
        "This is about Star Wars Episode IV", result, 0.9,
    )
    assert status == ExplanationStatus.VERIFIED
    assert "1" in ids


def test_audit_verified_by_genre():
    result = {
        "id": "2",
        "title": "XY",
        "source_fields": {"genres_all": "Science Fiction, Drama"},
    }
    status, _ = _audit_citation("A science fiction masterpiece", result, 0.9)
    assert status == ExplanationStatus.VERIFIED


def test_audit_unverified_when_nothing_matches():
    result = {"id": "1", "title": "Star Wars", "source_fields": {}}
    status, ids = _audit_citation("A great space exploration documentary", result, 0.9)
    assert status == ExplanationStatus.UNVERIFIED
    assert ids == []


# ── Results Context Builder ────────────────────────────────────────────────


def test_build_results_context_format():
    results = [{"id": "1", "title": "Star Wars", "source_fields": {"category": "Sci-Fi"}}]
    context = _build_results_context(results)
    assert 'id="1"' in context
    assert 'title="Star Wars"' in context
    assert 'category="Sci-Fi"' in context


# ── Reranker Node Paths ────────────────────────────────────────────────────


def test_reranker_empty_results(state_factory):
    state = state_factory()
    result = reranker_node(state)
    assert result["reranked_results"] == []


def test_reranker_success(mocker, state_factory):
    mocker.patch(
        "src.nodes.reranker._score_with_cross_encoder",
        return_value=[(0, 0.95), (1, 0.80)],
    )
    mocker.patch("src.nodes.reranker.check_budget")
    mocker.patch(
        "src.nodes.reranker._generate_explanations",
        return_value=(
            [
                {"id": "1", "explanation": "Matches Star Wars Episode query"},
                {"id": "2", "explanation": "Related alien movie"},
            ],
            100, 50,
        ),
    )
    results = [
        {"id": "1", "title": "Star Wars Episode", "score": 0.9, "source_fields": {}},
        {"id": "2", "title": "Alien", "score": 0.8, "source_fields": {}},
    ]
    state = state_factory(
        query="space movies", search_results=results,
        token_usage=[], quality_scores={},
    )
    result = reranker_node(state)
    assert len(result["reranked_results"]) == 2
    assert result["reranked_results"][0]["id"] == "1"
    assert "rerank_confidence" in result["quality_scores"]
    assert result["cumulative_token_cost"] > 0


def test_reranker_cross_encoder_failure(mocker, state_factory):
    mocker.patch(
        "src.nodes.reranker._score_with_cross_encoder",
        side_effect=RuntimeError("model not found"),
    )
    mocker.patch("src.nodes.reranker.check_budget")
    mocker.patch(
        "src.nodes.reranker._generate_explanations",
        return_value=([], 0, 0),
    )
    results = [{"id": "1", "title": "Star Wars", "score": 0.9, "source_fields": {}}]
    state = state_factory(
        query="space", search_results=results,
        token_usage=[], quality_scores={},
    )
    result = reranker_node(state)
    assert len(result["reranked_results"]) == 1
    assert any("CROSS_ENCODER_FAILED" in e.get("message", "") for e in result["errors"])


def test_reranker_budget_exceeded(mocker, state_factory):
    mocker.patch(
        "src.nodes.reranker._score_with_cross_encoder",
        return_value=[(0, 0.9)],
    )
    mocker.patch(
        "src.nodes.reranker.check_budget",
        side_effect=ValueError("BUDGET_EXCEEDED"),
    )
    results = [{"id": "1", "title": "Star Wars", "score": 0.9, "source_fields": {}}]
    state = state_factory(
        query="space", search_results=results,
        token_usage=[], quality_scores={},
    )
    result = reranker_node(state)
    assert len(result["reranked_results"]) == 1
    assert result["reranked_results"][0]["explanation"] == ""
    assert any("BUDGET_EXCEEDED" in e.get("message", "") for e in result["errors"])
