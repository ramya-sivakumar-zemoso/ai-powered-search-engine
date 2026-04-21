"""Tests for routing heuristics, retry overrides, graph edges, and injection."""

import pytest

from src.nodes.retrieval_router import _heuristic_route, _apply_retry_override
from src.graph.graph import (
    route_after_query_understander,
    route_after_searcher,
    route_after_evaluator,
)


# ── Heuristic Route (all 6 rules) ──────────────────────────────────────────


@pytest.mark.parametrize("intent, ambiguity, entities, want_strategy, want_rule", [
    ("NAVIGATIONAL", 0.3, ["star wars"], "KEYWORD", 1),
    ("TRANSACTIONAL", 0.2, ["earbuds"],  "HYBRID",  2),
    ("TRANSACTIONAL", 0.6, ["earbuds"],  "HYBRID",  3),
    ("INFORMATIONAL", 0.3, ["sci-fi"],   "HYBRID",  4),
    ("INFORMATIONAL", 0.7, ["cosmos"],   "SEMANTIC", 5),
    ("INFORMATIONAL", 0.5, [],           "SEMANTIC", 6),
])
def test_heuristic_route(intent, ambiguity, entities, want_strategy, want_rule):
    strategy, _, rule, _ = _heuristic_route(intent, ambiguity, entities)
    assert strategy == want_strategy
    assert rule == want_rule


# ── Retry Override ──────────────────────────────────────────────────────────


def test_retry_override_no_prescription():
    s, _, note = _apply_retry_override("HYBRID", 0.5, None, [])
    assert s == "HYBRID" and note == ""


def test_retry_override_applies_suggestion():
    rx = {"suggested_strategy": "SEMANTIC", "reason_code": "LOW_RELEVANCE"}
    s, _, note = _apply_retry_override("KEYWORD", 0.1, rx, [])
    assert s == "SEMANTIC"
    assert note != ""


# ── Graph Conditional Edges ─────────────────────────────────────────────────


def test_route_injection_to_reporter():
    assert route_after_query_understander(
        {"errors": [{"message": "INJECTION_DETECTED"}]}
    ) == "reporter"


def test_route_normal_to_router():
    assert route_after_query_understander({"errors": []}) == "retrieval_router"


def test_route_searcher_error_to_reporter():
    assert route_after_searcher(
        {
            "errors": [{
                "node": "searcher",
                "severity": "ERROR",
                "message": "MEILI_DOWN",
            }],
        }
    ) == "reporter"


def test_route_searcher_ok_to_evaluator():
    assert route_after_searcher({"errors": []}) == "evaluator"


@pytest.mark.parametrize("decision, target", [
    ("accept",    "reranker"),
    ("retry",     "retrieval_router"),
    ("exhausted", "reporter"),
])
def test_route_after_evaluator(decision, target):
    assert route_after_evaluator({"evaluator_decision": decision}) == target
