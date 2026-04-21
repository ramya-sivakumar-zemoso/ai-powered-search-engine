"""Tests for reporter node — result selection, cost summary, and response assembly."""

from src.nodes.reporter import (
    _pick_best_results,
    _build_cost_summary,
    _is_blocked,
    _build_warnings,
    assemble_final_response,
    reporter_node,
)
from src.utils.query_word_limit import query_word_limit_user_notice


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
        reranked_results=[{"id": "1", "title": "Star Wars", "confidence": 0.82}],
        quality_scores={"combined": 0.8},
    )
    result = reporter_node(state)
    resp = result["final_response"]
    assert resp["result_source"] == "reranked"
    assert resp["result_count"] == 1
    assert resp["blocked"] is False
    assert resp["partial_results"] is False
    assert resp["rerank_degraded"] is False
    assert "Top Results:" in resp["structured_text"]
    assert resp["session_id"] == "test-session"
    assert resp["pipeline_metadata"]["session_id"] == "test-session"
    assert resp.get("result_quality_notice") is None


def test_result_quality_notice_query_word_limit(state_factory, mocker):
    mocker.patch(
        "src.nodes.reporter.get_settings",
        return_value=mocker.Mock(max_query_words=25, meili_index_name="movies"),
    )
    state = state_factory(
        errors=[{
            "node": "query_understander",
            "severity": "WARNING",
            "message": "QUERY_WORD_LIMIT",
            "fallback_description": "detail",
        }],
        search_results=[{"id": "1", "title": "Hit", "score": 0.9}],
    )
    fr = assemble_final_response(state)
    assert fr["result_quality_notice"] == query_word_limit_user_notice(25)


def test_result_quality_notice_on_semantic_fallback_when_hits_weak(state_factory):
    """Fallback flag alone is not enough — scores must still look poor."""
    state = state_factory(
        errors=[{
            "node": "searcher",
            "severity": "WARNING",
            "message": "SEMANTIC_DEGRADATION_FALLBACK",
            "fallback_description": "no overlap",
        }],
        quality_scores={
            "semantic_relevance": 0.18,
            "combined": 0.48,
            "rerank_confidence": 0.14,
        },
        reranked_results=[
            {"id": "1", "title": "Unrelated", "confidence": 0.12, "meilisearch_ranking_score": 0.1},
        ],
    )
    fr = assemble_final_response(state)
    assert fr["result_quality_notice"]
    assert "exact" in fr["result_quality_notice"].lower()


def test_result_quality_notice_suppressed_when_fallback_but_on_topic(state_factory):
    """Stem/overlap heuristics can trigger fallback even when results are still relevant."""
    state = state_factory(
        partial_results=True,
        errors=[{
            "node": "searcher",
            "severity": "WARNING",
            "message": "SEMANTIC_DEGRADATION_FALLBACK",
            "fallback_description": "overlap heuristic",
        }],
        quality_scores={
            "semantic_relevance": 0.52,
            "combined": 0.7,
            "rerank_confidence": 0.48,
        },
        reranked_results=[
            {"id": "1", "title": "Maybelline Perfume", "confidence": 0.55, "meilisearch_ranking_score": 0.4},
        ],
    )
    fr = assemble_final_response(state)
    assert fr.get("result_quality_notice") is None


def test_notice_when_semantic_fallback_on_topic_but_verified_explanations_disclaim(state_factory):
    """Groceries: evaluator combined stays high after fallback; reranker text still trumps."""
    state = state_factory(
        query="groceries",
        errors=[{
            "node": "searcher",
            "severity": "WARNING",
            "message": "SEMANTIC_DEGRADATION_FALLBACK",
            "fallback_description": "no keyword overlap",
        }],
        quality_scores={
            "semantic_relevance": 0.6688,
            "combined": 0.6743,
            "rerank_confidence": 0.5,
            "rerank_low_confidence_ratio": 0.0,
        },
        reranked_results=[
            {
                "id": "1",
                "title": "Dish Racks",
                "confidence": 0.5,
                "explanation": "This result is not relevant to the query 'groceries' as it is not food.",
                "explanation_status": "VERIFIED",
            },
            {
                "id": "2",
                "title": "Sandals",
                "confidence": 0.5,
                "explanation": "This result is not relevant to the query 'groceries' since it is clothing.",
                "explanation_status": "VERIFIED",
            },
        ],
    )
    fr = assemble_final_response(state)
    assert fr.get("result_quality_notice")
    assert "might" in fr["result_quality_notice"].lower()


def test_notice_for_ultra_vague_query_thing_stuff(state_factory):
    state = state_factory(
        query="thing / stuff",
        sanitized_query="thing / stuff",
        quality_scores={
            "combined": 0.8,
            "semantic_relevance": 0.7,
            "rerank_confidence": 0.6,
        },
        reranked_results=[{"id": "1", "title": "Creative Art Supplies", "confidence": 0.6}],
    )
    fr = assemble_final_response(state)
    assert fr.get("result_quality_notice")


def test_notice_for_absurd_price_in_query(state_factory):
    state = state_factory(
        query="$999999999 laptop",
        sanitized_query="$999999999 laptop",
        quality_scores={
            "combined": 0.75,
            "semantic_relevance": 0.7,
            "rerank_confidence": 0.65,
        },
        reranked_results=[{"id": "1", "title": "Dell Laptop", "confidence": 0.62}],
    )
    fr = assemble_final_response(state)
    assert fr.get("result_quality_notice")


def test_notice_when_query_words_missing_from_hits_but_scores_mediocre(state_factory):
    """e.g. *flying robot* → generic toys: scores can look mid, text has no query words."""
    state = state_factory(
        query="flying robot",
        sanitized_query="flying robot",
        quality_scores={
            "semantic_relevance": 0.40,
            "combined": 0.62,
            "rerank_confidence": 0.36,
        },
        search_results=[
            {
                "id": "1",
                "title": "LeapFrog Forest Green Outdoor Toys",
                "score": 0.38,
                "source_fields": {
                    "description": "Outdoor play for kids.",
                    "brand": "LeapFrog",
                    "category": "Toys & Games",
                },
            },
        ],
        reranked_results=[
            {
                "id": "1",
                "title": "LeapFrog Forest Green Outdoor Toys",
                "confidence": 0.42,
                "meilisearch_ranking_score": 0.35,
                "relevance_score": 0.42,
            },
        ],
    )
    fr = assemble_final_response(state)
    assert fr.get("result_quality_notice")
    assert "might" in fr["result_quality_notice"].lower()


def test_notice_suppressed_when_scores_strongly_good_despite_rare_tokens(state_factory):
    """Synonym / vector-strong case: few literal query tokens in titles but scores are high."""
    state = state_factory(
        query="fragrance products",
        sanitized_query="fragrance products",
        quality_scores={
            "semantic_relevance": 0.52,
            "combined": 0.72,
            "rerank_confidence": 0.52,
        },
        search_results=[
            {
                "id": "1",
                "title": "Designer Eau de Parfum",
                "score": 0.55,
                "source_fields": {
                    "description": "Luxury scent for evening wear.",
                    "category": "Beauty & Personal Care",
                },
            },
        ],
        reranked_results=[
            {
                "id": "1",
                "title": "Designer Eau de Parfum",
                "confidence": 0.58,
                "meilisearch_ranking_score": 0.5,
            },
        ],
    )
    fr = assemble_final_response(state)
    assert fr.get("result_quality_notice") is None


def test_notice_when_rerank_pool_weak_despite_high_evaluator_scores(state_factory):
    """Groceries-style: evaluator can look fine; cross-encoder mean says the pool is off."""
    state = state_factory(
        quality_scores={
            "semantic_relevance": 0.55,
            "combined": 0.72,
            "rerank_confidence": 0.34,
            "rerank_low_confidence_ratio": 0.4,
        },
        reranked_results=[
            {"id": "1", "title": "Dish Rack", "confidence": 0.38},
            {"id": "2", "title": "Sandals", "confidence": 0.32},
        ],
    )
    fr = assemble_final_response(state)
    assert fr.get("result_quality_notice")
    assert "might" in fr["result_quality_notice"].lower()


def test_notice_when_verified_disclaimers_bypass_one_high_top_confidence(state_factory):
    """A single strong CE hit in the top 5 must not hide audited 'not relevant' explanations."""
    state = state_factory(
        quality_scores={
            "semantic_relevance": 0.55,
            "combined": 0.72,
            "rerank_confidence": 0.52,
            "rerank_low_confidence_ratio": 0.0,
        },
        reranked_results=[
            {
                "id": "1",
                "title": "Dish Rack",
                "confidence": 0.5,
                "explanation": "This result is not relevant to the query 'groceries'.",
                "explanation_status": "VERIFIED",
            },
            {
                "id": "2",
                "title": "Sandals",
                "confidence": 0.5,
                "explanation": "This result is not relevant to the query 'groceries'.",
                "explanation_status": "VERIFIED",
            },
            {
                "id": "3",
                "title": "Unrelated but high CE",
                "confidence": 0.62,
                "explanation": "Somewhat related category neighbor.",
                "explanation_status": "VERIFIED",
            },
        ],
    )
    fr = assemble_final_response(state)
    assert fr.get("result_quality_notice")
    assert "might" in fr["result_quality_notice"].lower()


def test_notice_when_verified_explanations_disclaim_mid_confidence_pool(state_factory):
    """Groceries case: mean confidence ~0.5 but LLM explanations VERIFIED-say not relevant."""
    state = state_factory(
        quality_scores={
            "semantic_relevance": 0.6,
            "combined": 0.75,
            "rerank_confidence": 0.5,
            "rerank_low_confidence_ratio": 0.0,
        },
        reranked_results=[
            {
                "id": "1",
                "title": "Dish Rack",
                "confidence": 0.5,
                "explanation": "This result is not relevant to the query 'groceries' as it is a kitchen accessory.",
                "explanation_status": "VERIFIED",
            },
            {
                "id": "2",
                "title": "Sandals",
                "confidence": 0.5,
                "explanation": "This result is not relevant to the query 'groceries' since it is clothing.",
                "explanation_status": "VERIFIED",
            },
        ],
    )
    fr = assemble_final_response(state)
    assert fr.get("result_quality_notice")
    assert "might" in fr["result_quality_notice"].lower()


def test_notice_suppressed_when_rerank_pool_weak_but_top_hit_clearly_strong(state_factory):
    state = state_factory(
        quality_scores={
            "semantic_relevance": 0.5,
            "combined": 0.7,
            "rerank_confidence": 0.35,
            "rerank_low_confidence_ratio": 0.55,
        },
        reranked_results=[
            {"id": "1", "title": "Organic Apples", "confidence": 0.62},
            {"id": "2", "title": "Other", "confidence": 0.2},
        ],
    )
    fr = assemble_final_response(state)
    assert fr.get("result_quality_notice") is None


def test_notice_when_searcher_flagged_retrieval_soft_match(state_factory):
    state = state_factory(
        retrieval_soft_match=True,
        quality_scores={"combined": 0.58, "semantic_relevance": 0.38},
        reranked_results=[
            {"id": "1", "title": "Toy", "confidence": 0.55, "meilisearch_ranking_score": 0.4},
        ],
    )
    fr = assemble_final_response(state)
    assert fr.get("result_quality_notice")
    assert "might" in fr["result_quality_notice"].lower()


def test_notice_suppressed_when_soft_match_but_evaluator_strong(state_factory):
    state = state_factory(
        retrieval_soft_match=True,
        quality_scores={"combined": 0.72, "semantic_relevance": 0.53},
        reranked_results=[
            {"id": "1", "title": "Perfume", "confidence": 0.6, "meilisearch_ranking_score": 0.55},
        ],
    )
    fr = assemble_final_response(state)
    assert fr.get("result_quality_notice") is None


def test_notice_suppressed_marginal_evaluator_when_low_token_coverage(state_factory):
    """Dual gate: decent combined + semantic without crossing into toy-vector false highs."""
    state = state_factory(
        query="designer scent gift",
        sanitized_query="designer scent gift",
        quality_scores={"semantic_relevance": 0.49, "combined": 0.65, "rerank_confidence": 0.4},
        search_results=[
            {
                "id": "1",
                "title": "Luxury Candle Set",
                "score": 0.4,
                "source_fields": {"description": "Home ambiance.", "category": "Home"},
            },
        ],
        reranked_results=[
            {"id": "1", "title": "Luxury Candle Set", "confidence": 0.41, "meilisearch_ranking_score": 0.4},
        ],
    )
    fr = assemble_final_response(state)
    assert fr.get("result_quality_notice") is None


def test_notice_suppressed_when_query_words_literal_in_titles(state_factory):
    state = state_factory(
        query="flying robot",
        sanitized_query="flying robot",
        quality_scores={"semantic_relevance": 0.38, "combined": 0.55},
        search_results=[
            {
                "id": "1",
                "title": "RC Flying Robot Drone",
                "score": 0.4,
                "source_fields": {"description": "Remote control flyer.", "category": "Toys"},
            },
        ],
        reranked_results=[
            {"id": "1", "title": "RC Flying Robot Drone", "confidence": 0.38, "meilisearch_ranking_score": 0.35},
        ],
    )
    fr = assemble_final_response(state)
    assert fr.get("result_quality_notice") is None


def test_result_quality_notice_low_semantic_and_confidence(state_factory):
    state = state_factory(
        quality_scores={"semantic_relevance": 0.25, "combined": 0.5},
        reranked_results=[
            {"id": "1", "title": "A", "confidence": 0.2},
            {"id": "2", "title": "B", "confidence": 0.15},
        ],
    )
    fr = assemble_final_response(state)
    assert fr["result_quality_notice"]
    assert "might" in fr["result_quality_notice"].lower()


def test_pipeline_metadata_includes_meili_index(state_factory):
    state = state_factory(
        meili_index_name="movies_alt",
        reranked_results=[{"id": "1", "title": "X", "confidence": 0.9}],
    )
    fr = assemble_final_response(state)
    assert fr["pipeline_metadata"]["meili_index_name"] == "movies_alt"


def test_reporter_zero_results(state_factory):
    """PRD §5: zero-result handling — reporter emits result_count=0, source=none."""
    state = state_factory(search_results=[], reranked_results=[])
    resp = reporter_node(state)["final_response"]
    assert resp["result_count"] == 0
    assert resp["result_source"] == "none"
    assert resp["blocked"] is False


def test_reporter_surfaces_structured_text(state_factory):
    """PRD §4.1 reporter: multi-format emission — structured_text must be present."""
    state = state_factory(
        reranked_results=[{"id": "1", "title": "Star Wars", "confidence": 0.9}],
    )
    resp = reporter_node(state)["final_response"]
    txt = resp.get("structured_text", "")
    assert "Top Results:" in txt
    assert "Star Wars" in txt


def test_reporter_error_propagation(state_factory):
    """PRD §5: errors set by upstream nodes must appear in final_response warnings."""
    state = state_factory(
        errors=[{
            "message": "FILTER_RELAXATION_APPLIED",
            "severity": "WARNING",
            "node": "searcher",
            "fallback_description": "relaxed filters",
        }],
        search_results=[{"id": "1", "title": "Item", "score": 0.8, "source_fields": {}}],
    )
    resp = reporter_node(state)["final_response"]
    messages = [w["message"] for w in resp["warnings"]]
    assert "FILTER_RELAXATION_APPLIED" in messages


def test_reporter_blocked_path(state_factory):
    state = state_factory(errors=[{
        "message": "INJECTION_DETECTED", "severity": "ERROR",
        "node": "query_understander", "fallback_description": "blocked",
    }])
    result = reporter_node(state)
    resp = result["final_response"]
    assert resp["blocked"] is True
    assert resp["partial_results"] is False
    assert resp["result_source"] == "none"
    assert resp["result_count"] == 0
    assert len(resp["warnings"]) == 1
