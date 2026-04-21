"""Tests for query ↔ catalog alignment helpers."""

from src.utils import query_catalog_alignment as qca


def test_content_tokens_filters_short_words():
    assert qca.content_tokens("a bc defg") == ["defg"]


def test_retrieval_soft_match_flying_robot_style():
    hits = [
        {
            "id": "1",
            "title": "LeapFrog Outdoor Toys",
            "_rankingScore": 0.4,
            "description": "Fun for kids.",
            "category": "Toys",
        },
    ]
    assert qca.retrieval_soft_match_from_meili_hits("flying robot", hits) is True


def test_retrieval_soft_match_not_triggered_for_strong_meili_score():
    hits = [
        {
            "id": "1",
            "title": "Desk Lamp",
            "_rankingScore": 0.62,
            "description": "LED light for your office.",
        },
    ]
    assert qca.retrieval_soft_match_from_meili_hits("flying robot", hits) is False


def test_looks_like_absurd_symbol_soup():
    assert qca.looks_like_absurd_query("@@@@@@@@@@@@@@@@@@@@") is True


def test_should_clear_hits_for_low_meili_scores():
    hits = [{"id": "1", "_rankingScore": 0.05}, {"id": "2", "_rankingScore": 0.04}]
    assert qca.should_clear_hits_for_low_meili_scores(hits) is True


def test_query_has_absurd_numeric_literal_currency():
    assert qca.query_has_absurd_numeric_literal("$999999999 laptop") is True


def test_query_has_absurd_numeric_literal_sane_prices():
    assert qca.query_has_absurd_numeric_literal("laptop under $50") is False
    assert qca.query_has_absurd_numeric_literal("macbook $4999") is False


def test_query_is_ultra_vague_lexical():
    assert qca.query_is_ultra_vague_lexical("thing / stuff") is True
    assert qca.query_is_ultra_vague_lexical("organic apples") is False
