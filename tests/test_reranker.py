"""Tests for reranker node — citation audit, context builder, and node paths."""

import pytest

from src.models.schema_registry import get_schema
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
    context = _build_results_context(results, get_schema("movies"))
    assert 'id="1"' in context
    assert "<doc_title>Star Wars</doc_title>" in context
    assert "<doc_category>Sci-Fi</doc_category>" in context


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
    results = [
        {"id": "1", "title": "Star Wars", "score": 0.7, "source_fields": {}},
        {"id": "2", "title": "Alien", "score": 0.9, "source_fields": {}},
    ]
    state = state_factory(
        query="space", search_results=results,
        token_usage=[], quality_scores={},
    )
    result = reranker_node(state)
    assert len(result["reranked_results"]) == 2
    assert result["reranked_results"][0]["id"] == "2"
    assert result["rerank_degraded"] is True
    assert any(e.get("message") == "RERANK_DEGRADED" for e in result["errors"])


def test_reranker_confidence_below_threshold_marks_degraded(mocker, state_factory):
    """PRD §4.3: confidence < 0.3 → explanation_status = DEGRADED."""
    mocker.patch(
        "src.nodes.reranker._score_with_cross_encoder",
        return_value=[(0, 0.15)],
    )
    mocker.patch("src.nodes.reranker.check_budget")
    mocker.patch(
        "src.nodes.reranker._generate_explanations",
        return_value=([{"id": "1", "explanation": "some text"}], 10, 5),
    )
    results = [{"id": "1", "title": "Test", "score": 0.5, "source_fields": {}}]
    state = state_factory(query="test", search_results=results, token_usage=[], quality_scores={})
    result = reranker_node(state)
    assert result["reranked_results"][0]["explanation_status"] == "DEGRADED"
    assert result["reranked_results"][0]["confidence"] < 0.3


def test_reranker_confidence_out_of_range_triggers_fallback(mocker, state_factory):
    """PRD §4.3: confidence > 1.0 → RERANK_DEGRADED event, native Meili order used."""
    mocker.patch(
        "src.nodes.reranker._score_with_cross_encoder",
        return_value=[(0, 1.8), (1, 0.9)],
    )
    mocker.patch("src.nodes.reranker.check_budget")
    mocker.patch(
        "src.nodes.reranker._generate_explanations",
        return_value=([], 0, 0),
    )
    results = [
        {"id": "1", "title": "Low score item", "score": 0.3, "source_fields": {}},
        {"id": "2", "title": "High score item", "score": 0.9, "source_fields": {}},
    ]
    state = state_factory(query="test", search_results=results, token_usage=[], quality_scores={})
    result = reranker_node(state)
    assert result["rerank_degraded"] is True
    assert any(e.get("message") == "RERANK_DEGRADED" for e in result["errors"])
    assert result["reranked_results"][0]["id"] == "2"


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
