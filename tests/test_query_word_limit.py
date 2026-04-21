"""Tests for ``truncate_query_to_word_limit``."""

import pytest

from src.utils.query_word_limit import (
    OVERFLOW_WORDS_TOOLTIP,
    query_overflow_preview_html,
    query_word_limit_user_notice,
    truncate_query_to_word_limit,
)


@pytest.mark.parametrize(
    "raw,max_words,want,truncated",
    [
        ("", 25, "", False),
        ("  hello world  ", 25, "hello world", False),
        ("one two three", 2, "one two", True),
        ("x y z", 10, "x y z", False),
    ],
)
def test_truncate_query_to_word_limit(raw, max_words, want, truncated):
    got, was_trunc = truncate_query_to_word_limit(raw, max_words)
    assert got == want
    assert was_trunc is truncated


def test_query_word_limit_user_notice_text():
    msg = query_word_limit_user_notice(25)
    assert "25" in msg
    assert "shortening" in msg.lower()


def test_query_overflow_preview_html_wraps_tail():
    h = query_overflow_preview_html("one two three four five", 3)
    assert "query-overflow-words" in h
    assert "one" in h and "five" in h
    assert OVERFLOW_WORDS_TOOLTIP[:8] in h or "excluded" in h


def test_zero_or_negative_means_no_limit():
    long_q = " ".join([f"w{i}" for i in range(50)])
    got, was_trunc = truncate_query_to_word_limit(long_q, 0)
    assert got == long_q
    assert was_trunc is False
    got2, was_trunc2 = truncate_query_to_word_limit(long_q, -1)
    assert got2 == long_q
    assert was_trunc2 is False
