"""Tests for reranker node — citation audit, context builder, and node paths."""

from dataclasses import replace

import pytest

import src.nodes.reranker as reranker_module
from src.models.schema_registry import get_schema
from src.nodes.reranker import (
    _audit_citation,
    _build_results_context,
    reranker_node,
    hydrate_async_explanations_in_state,
)
from src.models.state import ExplanationStatus


@pytest.fixture(autouse=True)
def _force_sync_explanations(mocker):
    """Keep reranker tests deterministic by disabling async explanations."""
    mocker.patch.object(
        reranker_module,
        "settings",
        replace(reranker_module.settings, reranker_explain_async=False),
    )


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


def test_audit_verified_when_doc_tags_match():
    result = {
        "id": "1",
        "title": "Star Wars Episode",
        "source_fields": {"description": "A space opera with lightsabers."},
    }
    expl = "Relevant because <doc_title>Star Wars Episode</doc_title> and <doc_description>space opera</doc_description>."
    status, ids = _audit_citation(expl, result, 0.9)
    assert status == ExplanationStatus.VERIFIED
    assert "1" in ids


def test_audit_explanation_unverified_when_doc_tag_not_in_field():
    result = {
        "id": "1",
        "title": "Star Wars",
        "source_fields": {"description": "A space adventure film."},
    }
    expl = "Cites <doc_description>quantum physics textbook</doc_description> which is not in the document."
    status, ids = _audit_citation(expl, result, 0.9)
    assert status == ExplanationStatus.EXPLANATION_UNVERIFIED
    assert ids == []


def test_audit_explanation_unverified_when_field_empty():
    result = {"id": "1", "title": "X", "source_fields": {}}
    expl = "See <doc_description>anything</doc_description>."
    status, ids = _audit_citation(expl, result, 0.9)
    assert status == ExplanationStatus.EXPLANATION_UNVERIFIED
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
    mocker.patch("src.nodes.reranker.check_budget_projected")
    mocker.patch(
        "src.nodes.reranker._generate_explanations",
        return_value=(
            [
                {"id": "1", "explanation": "Matches Star Wars Episode query"},
                {"id": "2", "explanation": "Related alien movie"},
            ],
            100, 50, 0.000045,
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
    assert result["reranked_results"][0]["meilisearch_ranking_score"] == 0.9
    assert "rerank_confidence" in result["quality_scores"]
    assert result["cumulative_token_cost"] > 0


def test_reranker_cross_encoder_failure(mocker, state_factory):
    mocker.patch(
        "src.nodes.reranker._score_with_cross_encoder",
        side_effect=RuntimeError("model not found"),
    )
    mocker.patch("src.nodes.reranker.check_budget_projected")
    mocker.patch(
        "src.nodes.reranker._generate_explanations",
        return_value=([], 0, 0, 0.0),
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
    mocker.patch("src.nodes.reranker.check_budget_projected")
    mocker.patch(
        "src.nodes.reranker._generate_explanations",
        return_value=([{"id": "1", "explanation": "some text"}], 10, 5, 0.000003),
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
    mocker.patch("src.nodes.reranker.check_budget_projected")
    mocker.patch(
        "src.nodes.reranker._generate_explanations",
        return_value=([], 0, 0, 0.0),
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
        "src.nodes.reranker.check_budget_projected",
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


def test_hydrate_async_explanations_applies_completed_job(mocker, state_factory):
    state = state_factory(
        search_results=[{"id": "1", "title": "Star Wars", "source_fields": {}}],
        reranked_results=[{
            "id": "1",
            "title": "Star Wars",
            "original_rank": 1,
            "new_rank": 1,
            "relevance_score": 0.9,
            "confidence": 0.9,
            "meilisearch_ranking_score": 0.8,
            "explanation": "",
            "explanation_status": "ABSENT",
            "explanation_citation_ids": [],
        }],
        explanations_pending=True,
        explanations_applied=False,
        explanation_job_id="job-1",
        explanation_job_status="PENDING",
        token_usage=[],
        cumulative_token_cost=0.0,
    )
    mocker.patch(
        "src.nodes.reranker.get_explanation_job",
        return_value={
            "status": "DONE",
            "explanations": [{"id": "1", "explanation": "This is Star Wars"}],
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "cost_usd": 0.00001,
        },
    )

    hydrated = hydrate_async_explanations_in_state(state)
    assert hydrated["explanations_pending"] is False
    assert hydrated["explanations_applied"] is True
    assert hydrated["explanation_job_status"] == "READY"
    assert hydrated["reranked_results"][0]["explanation"] == "This is Star Wars"
    assert hydrated["cumulative_token_cost"] > 0
