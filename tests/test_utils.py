"""Tests for budget checking, state delta, and JSON serialization."""

import pytest
from datetime import datetime
from enum import Enum

from src.utils.langwatch_tracker import check_budget, make_budget_exceeded_error
from src.utils.state_display import state_delta, to_jsonable


def test_budget_within_limit():
    check_budget(0.001, "node", "hash")


def test_budget_exceeded():
    with pytest.raises(ValueError, match="BUDGET_EXCEEDED"):
        check_budget(1.0, "node", "hash")


def test_make_budget_error_fields():
    err = make_budget_exceeded_error("reranker")
    assert err.node == "reranker" and err.message == "BUDGET_EXCEEDED"


# ── State Delta ─────────────────────────────────────────────────────────────


def test_delta_detects_changes():
    delta = state_delta({"a": 1, "b": 2}, {"a": 1, "b": 3, "c": 4})
    assert "a" not in delta
    assert delta["b"] == 3 and delta["c"] == 4


def test_delta_detects_removals():
    delta = state_delta({"a": 1, "b": 2}, {"a": 1})
    assert "b" in delta.get("_removed_keys", [])


# ── to_jsonable ─────────────────────────────────────────────────────────────


class Color(Enum):
    RED = "red"


def test_to_jsonable_types():
    assert to_jsonable(None) is None
    assert to_jsonable(42) == 42
    assert to_jsonable(Color.RED) == "red"
    assert to_jsonable(datetime(2024, 1, 1)) == "2024-01-01T00:00:00"
    assert to_jsonable({"k": [1, 2]}) == {"k": [1, 2]}


def test_to_jsonable_truncates_large_result_lists():
    data = {"search_results": [{"id": str(i)} for i in range(30)]}
    result = to_jsonable(data, max_search_hits=5)
    assert result["search_results"]["_truncated"] is True
    assert result["search_results"]["_total"] == 30


def test_to_jsonable_pydantic_model():
    from src.models.state import TokenUsage

    model = TokenUsage(node="test", prompt_tokens=100)
    result = to_jsonable(model)
    assert result["node"] == "test"
    assert result["prompt_tokens"] == 100


# ── LLM Utilities ──────────────────────────────────────────────────────────

from src.utils.llm import strip_markdown_fences, extract_token_usage


@pytest.mark.parametrize("inp, expected", [
    ('{"key": "val"}', '{"key": "val"}'),
    ('```json\n{"key": "val"}\n```', '{"key": "val"}'),
    ('```\n{"key": "val"}\n```', '{"key": "val"}'),
])
def test_strip_markdown_fences(inp, expected):
    assert strip_markdown_fences(inp) == expected


def test_extract_token_usage_new_format():
    class Resp:
        usage_metadata = {"input_tokens": 100, "output_tokens": 50}
    p, c = extract_token_usage(Resp())
    assert p == 100 and c == 50


def test_extract_token_usage_legacy_format():
    class Resp:
        usage_metadata = None
        response_metadata = {"token_usage": {"prompt_tokens": 80, "completion_tokens": 40}}
    p, c = extract_token_usage(Resp())
    assert p == 80 and c == 40
