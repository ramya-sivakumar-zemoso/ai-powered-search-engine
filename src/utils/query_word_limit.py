"""Clamp user search queries to a maximum word count."""

from __future__ import annotations

import html as html_module

# Tooltip for UI previews of text beyond the word cap (Streamlit / HTML).
OVERFLOW_WORDS_TOOLTIP = "These words will be excluded to optimize your search."


def query_word_limit_user_notice(max_words: int) -> str:
    """Copy shown when the query exceeds or is shortened to the word limit."""
    return (
        f"To provide the best results, we support up to {max_words} words per search. "
        "Try shortening your query."
    )


def query_overflow_preview_html(raw: str, max_words: int) -> str:
    """HTML: first ``max_words`` tokens as plain text, remainder gray with tooltip. Empty if no overflow."""
    if max_words <= 0:
        return ""
    if not isinstance(raw, str):
        raw = str(raw or "")
    parts = raw.strip().split()
    if len(parts) <= max_words:
        return ""
    kept = " ".join(html_module.escape(w) for w in parts[:max_words])
    rest = " ".join(html_module.escape(w) for w in parts[max_words:])
    tip = html_module.escape(OVERFLOW_WORDS_TOOLTIP)
    return (
        f'<span class="query-kept-words">{kept}</span> '
        f'<span class="query-overflow-words" title="{tip}">{rest}</span>'
    )


def truncate_query_to_word_limit(text: str, max_words: int) -> tuple[str, bool]:
    """Return ``(query, truncated)`` with at most ``max_words`` whitespace-separated tokens.

    ``max_words <= 0`` disables limiting (returns stripped ``text``).
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.strip()
    if max_words <= 0:
        return text, False
    parts = text.split()
    if len(parts) <= max_words:
        return text, False
    return " ".join(parts[:max_words]), True
