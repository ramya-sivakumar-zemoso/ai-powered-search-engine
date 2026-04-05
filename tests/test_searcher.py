"""Tests for searcher helpers and the node with mocked Meilisearch."""

import pytest
from datetime import datetime, timezone, timedelta

from src.nodes.searcher import (
    _build_filter_string,
    _build_search_query,
    _hits_to_search_results,
    _build_freshness_report,
    searcher_node,
)
from src.models.state import SearchResult


# ── Filter String Builder ──────────────────────────────────────────────────


@pytest.mark.parametrize("filters, expected", [
    ({}, None),
    ({"genre": "Action"}, 'category = "Action"'),
    ({"year": "2020"}, None),
])
def test_build_filter_string(filters, expected):
    assert _build_filter_string(filters) == expected


# ── Search Query Builder ────────────────────────────────────────────────────


def test_search_query_from_entities():
    state = {"parsed_intent": {"entities": ["sci-fi", "time travel"]}, "query": "raw"}
    assert _build_search_query(state) == "sci-fi time travel"


def test_search_query_fallback():
    state = {"parsed_intent": {"entities": []}, "query": "raw query"}
    assert _build_search_query(state) == "raw query"


# ── Hit Mapping ─────────────────────────────────────────────────────────────


def test_hits_to_search_results():
    hits = [{"id": "1", "title": "Star Wars", "_rankingScore": 0.95, "genre": "Sci-Fi"}]
    results = _hits_to_search_results(hits)
    assert len(results) == 1
    assert results[0].id == "1"
    assert results[0].score == 0.95
    assert "genre" in results[0].source_fields


# ── Freshness Report ───────────────────────────────────────────────────────


def test_freshness_all_fresh():
    now = datetime.now(timezone.utc)
    results = [SearchResult(id="1", freshness_timestamp=now - timedelta(minutes=1))]
    report = _build_freshness_report(results)
    assert not report.staleness_flag


def test_freshness_stale():
    old = datetime.now(timezone.utc) - timedelta(hours=2)
    results = [SearchResult(id="1", freshness_timestamp=old)]
    report = _build_freshness_report(results)
    assert report.staleness_flag
    assert {"id": "1", "title": ""} in report.stale_result_ids


# ── Searcher Node (mocked Meilisearch) ─────────────────────────────────────


def test_searcher_node_success(mocker, state_factory):
    mocker.patch(
        "src.nodes.searcher.meili_search",
        return_value={"hits": [{"id": "1", "title": "Alien", "_rankingScore": 0.9}]},
    )
    state = state_factory(
        query="alien",
        parsed_intent={"entities": ["alien"], "filters": {}},
    )
    result = searcher_node(state)
    assert len(result["search_results"]) == 1


def test_searcher_node_meili_down(mocker, state_factory):
    mocker.patch(
        "src.nodes.searcher.meili_search",
        side_effect=RuntimeError("connection refused"),
    )
    state = state_factory(query="test", parsed_intent={"entities": ["test"], "filters": {}})
    result = searcher_node(state)
    assert result["search_results"] == []
    assert any(e.get("severity") == "ERROR" for e in result["errors"])
