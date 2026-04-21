"""Query ↔ catalog text alignment for UX warnings and retrieval gating.

Used by the searcher (Meilisearch hits + raw user query) and the reporter (final
pipeline results). Keeps one definition of tokenization / coverage so behavior
does not drift between nodes.
"""
from __future__ import annotations

import re

import src.constants as C

# Words dropped when checking whether the user's query appears in hit text.
ALIGNMENT_STOP_WORDS: frozenset[str] = frozenset({
    "that", "this", "with", "from", "your", "have", "some", "what", "when", "where",
    "which", "those", "these", "than", "then", "them", "very", "also", "just", "like",
    "best", "good", "free", "shop", "buy", "sale", "new", "top", "all", "only",
    "each", "every", "into", "over", "such", "more", "most", "much", "many",
})

# Literal numerals above this in the query are treated as unrealistic constraints (UX notice).
_MAX_SANE_QUERY_NUMERAL = 500_000

# When *every* meaningful token is one of these, the query is too vague to treat hits as confident.
_ULTRA_VAGUE_QUERY_TOKENS: frozenset[str] = frozenset({
    "thing", "things", "stuff", "item", "items", "something", "anything", "whatever",
    "misc", "miscellaneous", "goods", "products", "search",
})


def content_tokens(query_text: str) -> list[str]:
    """Meaningful words (length ≥ 4) from a user query string."""
    q = (query_text or "").strip()
    if not q:
        return []
    words = re.findall(r"[a-z0-9]+", q.lower())
    return [w for w in words if len(w) >= 4 and w not in ALIGNMENT_STOP_WORDS]


def query_has_absurd_numeric_literal(
    query_text: str,
    *,
    max_sane: int = _MAX_SANE_QUERY_NUMERAL,
) -> bool:
    """True when the query contains a currency amount or 6+ digit integer far beyond normal use."""
    t = (query_text or "").strip().lower()
    if not t:
        return False

    def _parse_int(num_str: str) -> int | None:
        s = num_str.replace(",", "")
        if not s:
            return None
        try:
            if "." in s:
                return int(float(s))
            return int(s)
        except ValueError:
            return None

    for m in re.finditer(r"[\$€£]\s*([\d,]+(?:\.\d+)?)", t):
        v = _parse_int(m.group(1))
        if v is not None and v > max_sane:
            return True
    for m in re.finditer(r"\b(\d{6,})\b", t):
        v = _parse_int(m.group(1))
        if v is not None and v > max_sane:
            return True
    return False


def query_is_ultra_vague_lexical(query_text: str) -> bool:
    """True when every meaningful token is ultra-generic filler (*thing* / *stuff* style)."""
    toks = content_tokens(query_text)
    if not toks or len(toks) > 3:
        return False
    return all(w in _ULTRA_VAGUE_QUERY_TOKENS for w in toks)


def meili_hit_text_blob(hit: dict) -> str:
    """Lowercased title + common listing fields from a raw Meilisearch hit."""
    parts: list[str] = [str(hit.get("title", "") or "")]
    for k in ("description", "brand", "category", "genres_all"):
        v = hit.get(k)
        if v is not None and v != "":
            parts.append(str(v))
    return " ".join(parts).lower()


def max_meilisearch_ranking_score(hits: list[dict]) -> float:
    scores: list[float] = []
    for h in hits:
        if not isinstance(h, dict):
            continue
        try:
            scores.append(float(h.get("_rankingScore", 0.0) or 0.0))
        except (TypeError, ValueError):
            continue
    return max(scores) if scores else 0.0


def mean_top_meilisearch_scores(hits: list[dict], *, top_n: int = 3) -> float:
    scores: list[float] = []
    for h in hits:
        if not isinstance(h, dict):
            continue
        try:
            scores.append(float(h.get("_rankingScore", 0.0) or 0.0))
        except (TypeError, ValueError):
            continue
    if not scores:
        return 0.0
    scores.sort(reverse=True)
    top = scores[: min(top_n, len(scores))]
    return sum(top) / len(top)


def token_coverage_in_meili_hits(
    tokens: list[str],
    hits: list[dict],
    *,
    top_n: int = 5,
) -> float:
    """Fraction of ``tokens`` found as a substring in any of the first ``top_n`` hit blobs."""
    if not tokens:
        return 1.0
    slice_h = [h for h in hits[:top_n] if isinstance(h, dict)]
    if not slice_h:
        return 0.0
    matched = 0
    for tok in tokens:
        for h in slice_h:
            if tok in meili_hit_text_blob(h):
                matched += 1
                break
    return matched / len(tokens)


def looks_like_absurd_query(query_text: str) -> bool:
    """Heuristic: keyboard mash / symbol soup / trivially empty — treat as no search."""
    t = (query_text or "").strip()
    if len(t) <= 1:
        return True
    if len(t) >= 36 and t.count(" ") < 2:
        letters = sum(1 for c in t if c.isalpha())
        if letters / len(t) < 0.55:
            return True
    if len(t) > 10 and len(set(t.lower())) <= 3:
        return True
    non_alnum = sum(1 for c in t if not c.isalnum() and not c.isspace())
    if len(t) >= 12 and non_alnum / len(t) > 0.5:
        return True
    return False


def should_clear_hits_for_low_meili_scores(hits: list[dict]) -> bool:
    """True when Meilisearch scores are noise-level — drop hits (caller handles absurd queries)."""
    if not hits:
        return False
    mx = max_meilisearch_ranking_score(hits)
    mn = mean_top_meilisearch_scores(hits, top_n=3)
    return mx < C.RETRIEVAL_COLLAPSE_MAX_SCORE and mn < C.RETRIEVAL_COLLAPSE_MEAN_TOP3


def retrieval_soft_match_from_meili_hits(query_text: str, hits: list[dict]) -> bool:
    """True when we keep hits but they are likely weak neighbors (warn in UI).

    Typical case: query and catalog are unrelated in wording, Meilisearch still returns
    same-broad-domain neighbors with mid ``_rankingScore`` (*flying robot* → generic toys).
    """
    if not hits or looks_like_absurd_query(query_text):
        return False
    tokens = content_tokens(query_text)
    if not tokens:
        return False
    cov = token_coverage_in_meili_hits(tokens, hits, top_n=5)
    mx = max_meilisearch_ranking_score(hits)
    return cov < C.RETRIEVAL_WEAK_TOKEN_COVERAGE and mx < C.RETRIEVAL_SOFT_MATCH_MAX_SCORE


def hit_text_blob_from_pipeline_result(r: dict, state: dict | None) -> str:
    """Title + listing fields for a pipeline result row (reranked or search)."""
    parts: list[str] = []
    rid = str(r.get("id", ""))
    if state and isinstance(state.get("search_results"), list):
        for s in state["search_results"]:
            if not isinstance(s, dict) or str(s.get("id", "")) != rid:
                continue
            parts.append(str(s.get("title", "") or ""))
            sf = s.get("source_fields") or {}
            if isinstance(sf, dict):
                for k in ("description", "brand", "category", "genres_all"):
                    parts.append(str(sf.get(k, "") or ""))
            break
        else:
            parts.append(str(r.get("title", "") or ""))
    else:
        parts.append(str(r.get("title", "") or ""))
    return " ".join(parts).lower()


def query_token_coverage_in_pipeline_results(
    tokens: list[str],
    results: list,
    state: dict | None,
    *,
    top_n: int = 5,
) -> float:
    """Same as :func:`token_coverage_in_meili_hits` but for final ``results`` dicts."""
    if not tokens:
        return 1.0
    hits = [r for r in results[:top_n] if isinstance(r, dict)]
    if not hits:
        return 0.0
    matched = 0
    for tok in tokens:
        for r in hits:
            blob = hit_text_blob_from_pipeline_result(r, state)
            if tok in blob:
                matched += 1
                break
    return matched / len(tokens)


def content_tokens_from_state(state: dict) -> list[str]:
    """Tokens using sanitized query when present (matches prior reporter behavior)."""
    q = (state.get("sanitized_query") or state.get("query") or "").strip()
    return content_tokens(q)
