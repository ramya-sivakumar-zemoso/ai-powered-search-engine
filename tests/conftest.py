"""Shared fixtures for the test suite."""

import pytest

_TITLES = [
    "The Godfather", "Star Wars", "Jurassic Park", "The Matrix",
    "Inception", "Forrest Gump", "Pulp Fiction", "Fight Club",
    "Interstellar", "Gladiator", "Titanic", "Avatar", "Frozen",
    "Coco", "Jaws",
]


@pytest.fixture
def results_factory():
    """Factory that builds *n* search-result dicts with diverse titles."""

    def _make(n, score=0.8):
        return [
            {"title": _TITLES[i % len(_TITLES)], "score": score, "id": str(i)}
            for i in range(n)
        ]

    return _make


@pytest.fixture
def state_factory():
    """Factory that builds a minimal pipeline state dict."""

    def _make(**overrides):
        base = {
            "query": "test query",
            "query_hash": "abc123",
            "session_id": "test-session",
            "search_results": [],
            "freshness_metadata": {},
            "retrieval_strategy": "HYBRID",
            "hybrid_weights": {"semanticRatio": 0.6},
            "iteration_count": 0,
            "cumulative_token_cost": 0.0,
            "search_history": [],
            "parsed_intent": {},
            "errors": [],
        }
        base.update(overrides)
        return base

    return _make
