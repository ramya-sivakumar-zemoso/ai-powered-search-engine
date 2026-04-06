"""
AI Search — end-user results + pipeline inspector.

Labels and demos follow ``DATASET_SCHEMA`` / ``MEILI_INDEX_NAME`` in ``.env``.

Run:
  streamlit run streamlit_app.py
"""

import html
import json
import uuid

import streamlit as st
from src.graph.graph import run_search_with_trace
from src.models.schema_registry import get_schema
from src.utils.config import get_settings
from src.utils.state_display import to_jsonable

_settings = get_settings()
_ui_schema = get_schema(_settings.dataset_schema)

if "pipeline_session_id" not in st.session_state:
    st.session_state.pipeline_session_id = str(uuid.uuid4())

st.set_page_config(
    page_title=_ui_schema.ui_product_title,
    layout="wide",
    page_icon="🔍",
)

# ═════════════════════════════════════════════════════════════════════════════
#  STYLES
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0d1117; }
.block-container { padding-top: 2rem; max-width: 1200px; }

/* ── header ─────────────────────────────────────── */
.page-title { font-size: 26px; font-weight: 700; color: #e6edf3; letter-spacing: -0.4px; }
.page-sub   { font-size: 13px; color: #484f58; margin-top: 2px; margin-bottom: 1.5rem; }

/* ── pipeline flow bar ──────────────────────────── */
.pipeline-flow {
    display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
    background: #161b22; border: 1px solid #21262d;
    border-radius: 8px; padding: 10px 14px; margin-bottom: 1.5rem;
}
.node-chip {
    font-size: 11px; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; padding: 3px 9px;
    border-radius: 4px; white-space: nowrap;
}
.arrow { color: #30363d; font-size: 13px; }
.n-qu  { background:#1c2d4a; color:#58a6ff; border:1px solid #1f6feb; }
.n-rr  { background:#1a2d1a; color:#3fb950; border:1px solid #238636; }
.n-s   { background:#2d1f0a; color:#d29922; border:1px solid #9e6a03; }
.n-ev  { background:#2d1a2d; color:#bc8cff; border:1px solid #8957e5; }
.n-rk  { background:#0f2d2d; color:#39d353; border:1px solid #196127; }
.n-rp  { background:#2d1616; color:#f85149; border:1px solid #b62324; }

/* ── inspector tabs (unchanged) ─────────────────── */
.section-label {
    font-size: 10px; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #484f58;
    border-bottom: 1px solid #21262d;
    padding-bottom: 6px; margin-bottom: 10px;
}
.node-tag {
    display: inline-block; font-size: 10px; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    padding: 2px 8px; border-radius: 3px; margin-bottom: 8px;
}

/* ── pipeline status badges ─────────────────────── */
.status-badge {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.05em;
    padding: 3px 10px; border-radius: 12px; margin-right: 6px;
    text-transform: uppercase;
}
.badge-warn {
    background: #2d2200; color: #d29922; border: 1px solid #9e6a03;
}
.badge-ok {
    background: #1a2d1a; color: #3fb950; border: 1px solid #238636;
}

/* ── search results page ────────────────────────── */
.facet-tag {
    font-size: 11px; padding: 2px 8px; border-radius: 10px;
    background: #1c2d4a; color: #58a6ff; border: 1px solid #1f6feb;
    display: inline-block; margin-right: 4px; margin-bottom: 4px;
}

.match-label {
    display: inline-block; font-size: 10px; font-weight: 700;
    letter-spacing: 0.06em; text-transform: uppercase;
    padding: 2px 8px; border-radius: 10px; margin-left: 8px;
    vertical-align: middle;
}
.match-best   { background: #1a2d1a; color: #3fb950; border: 1px solid #238636; }
.match-good   { background: #1c2d4a; color: #58a6ff; border: 1px solid #1f6feb; }

.ai-insight {
    margin-top: 8px; padding: 8px 12px;
    background: #0d1117; border-left: 3px solid #238636;
    border-radius: 0 6px 6px 0;
    font-size: 12px; color: #8b949e; line-height: 1.5;
}
.ai-insight-label {
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; color: #3fb950; margin-bottom: 2px;
}
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

NODE_COLORS = {
    "query_understander": "n-qu",
    "retrieval_router":   "n-rr",
    "searcher":           "n-s",
    "evaluator":          "n-ev",
    "reranker":           "n-rk",
    "reporter":           "n-rp",
}

NODE_LABELS = {
    "query_understander": "1 · query understander",
    "retrieval_router":   "2 · retrieval router",
    "searcher":           "3 · searcher",
    "evaluator":          "4 · evaluator",
    "reranker":           "5 · reranker",
    "reporter":           "6 · reporter",
}

PIPELINE_ORDER = list(NODE_LABELS.keys())

POSTER_PLACEHOLDER = "https://via.placeholder.com/80x120/21262d/484f58?text=No+Image"


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════


def _facet_tags_from_source(source: dict) -> list[str]:
    """First non-empty ``ui_tag_fields`` entry becomes chips (comma-separated OK)."""
    for field in _ui_schema.ui_tag_fields:
        val = source.get(field)
        if val is None or val == "":
            continue
        s = str(val).strip()
        if not s:
            continue
        if "," in s:
            return [x.strip() for x in s.split(",") if x.strip()][:5]
        return [s][:5]
    return []

def node_chip(name: str) -> str:
    css = NODE_COLORS.get(name, "")
    label = NODE_LABELS.get(name, name)
    return f'<span class="node-chip {css}">{label}</span>'


def render_pipeline_flow():
    chips = ' <span class="arrow">→</span> '.join(
        node_chip(n) for n in PIPELINE_ORDER
    )
    st.markdown(f'<div class="pipeline-flow">{chips}</div>', unsafe_allow_html=True)


_CROSS_CUTTING_KEYS = {
    "cumulative_token_cost", "token_usage", "errors",
    "search_history", "iteration_count",
}

_NODE_OWN_KEYS: dict[str, set[str]] = {
    "query_understander": {"query_hash", "parsed_intent"},
    "retrieval_router":   {"retrieval_strategy", "hybrid_weights", "router_reasoning"},
    "searcher":           {"search_results", "freshness_metadata", "filter_relaxation_applied"},
    "evaluator":          {"quality_scores", "evaluator_decision", "retry_prescription"},
    "reranker":           {"reranked_results"},
    "reporter":           {"final_response"},
}


def _filter_node_delta(node_name: str, delta: dict) -> dict:
    """Keep only the keys that are unique to this node, drop cross-cutting fields."""
    own = _NODE_OWN_KEYS.get(node_name)
    if own:
        return {k: v for k, v in delta.items() if k in own}
    return {k: v for k, v in delta.items() if k not in _CROSS_CUTTING_KEYS}


def _fmt_staleness(seconds: float) -> str:
    if seconds > 86400:
        return f"{seconds / 86400:.0f} days"
    if seconds > 3600:
        return f"{seconds / 3600:.1f} hrs"
    return f"{seconds:.0f}s"


def _render_reporter_tab(delta: dict, max_hits: int) -> None:
    """Render the reporter node's final_response as structured, readable sections."""
    fr_raw = delta.get("final_response", {})
    if not fr_raw:
        st.json(to_jsonable(delta, max_search_hits=max_hits))
        return

    fr = to_jsonable(fr_raw, max_search_hits=max_hits)

    # ── Warnings (show first so they're not missed) ──────────────────
    warnings_list = fr.get("warnings", [])
    for w in warnings_list:
        if isinstance(w, dict):
            node = w.get("node", "")
            msg = w.get("message", "")
            detail = w.get("detail", "")
            st.warning(f"**{node}** — {msg}" + (f": {detail}" if detail else ""))

    # ── Overview ─────────────────────────────────────────────────────
    meta = fr.get("pipeline_metadata", {})
    cost = fr.get("cost_summary", {})
    total_cost = cost.get("total_cost_usd", 0)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Results", fr.get("result_count", 0))
    c2.metric("Strategy", meta.get("strategy", "—"))
    c3.metric("Iterations", meta.get("iterations", 0))
    c4.metric("Decision", meta.get("evaluator_decision", "—"))
    c5.metric("Total Cost", f"${total_cost:.6f}" if total_cost else "$0")

    reasoning = meta.get("router_reasoning", "")
    if reasoning:
        st.caption(f"Router: {reasoning}")

    # ── Quality ──────────────────────────────────────────────────────
    quality = fr.get("quality_summary", {})
    if quality:
        st.divider()
        display_q = {
            "Semantic Relevance": quality.get("semantic_relevance"),
            "Result Coverage": quality.get("result_coverage"),
            "Ranking Stability": quality.get("ranking_stability"),
            "Freshness": quality.get("freshness_signal"),
            "Combined": quality.get("combined"),
            "Rerank Confidence": quality.get("rerank_confidence"),
        }
        q_items = [(k, v) for k, v in display_q.items() if v is not None]
        q_cols = st.columns(len(q_items))
        for col, (label, val) in zip(q_cols, q_items):
            col.metric(label, f"{val:.0%}" if isinstance(val, float) else val)

    # ── Cost breakdown (compact) ─────────────────────────────────────
    per_node = cost.get("per_node", [])
    billable = [e for e in per_node if isinstance(e, dict) and e.get("cost_usd", 0) > 0]
    if billable:
        st.divider()
        st.markdown("**Cost per node**")
        cost_cols = st.columns(len(billable))
        for col, entry in zip(cost_cols, billable):
            n = entry.get("node", "?")
            tokens = entry.get("prompt_tokens", 0) + entry.get("completion_tokens", 0)
            col.metric(n, f"${entry['cost_usd']:.6f}", delta=f"{tokens} tokens", delta_color="off")

    # ── Freshness ────────────────────────────────────────────────────
    freshness = fr.get("freshness_report", {})
    stale_ids = freshness.get("stale_result_ids", [])
    if freshness:
        st.divider()
        max_s = freshness.get("max_staleness_seconds", 0)
        f1, f2, f3 = st.columns(3)
        f1.metric("Data Staleness", _fmt_staleness(max_s))
        f2.metric("Stale Results", f"{len(stale_ids)} / {fr.get('result_count', 0)}")
        idx_api = freshness.get("index_stats_updated_at")
        f3.metric(
            "Index (API)",
            str(idx_api)[:19] if idx_api else "—",
        )
        if freshness.get("freshness_unknown"):
            st.caption(
                "Index freshness metadata unavailable from Meilisearch "
                "(PRD §4.3 `FRESHNESS_UNKNOWN`)."
            )

    # ── Results table ────────────────────────────────────────────────
    results = fr.get("results", [])
    if results:
        st.divider()
        items = results
        if isinstance(results, dict) and results.get("_truncated"):
            items = results.get("items", [])

        st.markdown(f"**Results ({len(items)})**")
        for r in items:
            if not isinstance(r, dict):
                continue
            title = r.get("title", r.get("id", "?"))
            rank = r.get("new_rank", "")
            conf = r.get("confidence", r.get("relevance_score", 0))
            conf_pct = f"{conf:.0%}" if isinstance(conf, float) else str(conf)
            expl = r.get("explanation", "")
            status = r.get("explanation_status", "ABSENT")

            header = f"#{rank}  {title}" if rank else title
            header += f"  —  {conf_pct} confidence"
            if status and status not in ("ABSENT", ""):
                header += f"  ({status})"

            if expl:
                with st.expander(header):
                    st.write(expl)
            else:
                st.text(header)


def _to_dict(obj):
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="python")
    return vars(obj) if hasattr(obj, "__dict__") else {}


def _build_result_lookup(search_results: list) -> dict:
    """Index search_results by id for fast cross-reference with reranked results."""
    lookup = {}
    for r in search_results:
        d = _to_dict(r)
        lookup[str(d.get("id", ""))] = d
    return lookup


_CONFIDENCE_BEST = 0.7
_CONFIDENCE_GOOD = 0.3
_NEGATIVE_PHRASES = frozenset({
    "not relevant", "no connection", "does not relate", "doesn't relate",
    "unrelated", "not related", "no relation", "does not match",
    "doesn't match", "not a match", "does not involve", "doesn't involve",
})


def _truncate(text: str, length: int = 250) -> str:
    if not text or len(text) <= length:
        return text or ""
    return text[:length].rsplit(" ", 1)[0] + "…"


def _match_label_html(confidence: float) -> str:
    if confidence >= _CONFIDENCE_BEST:
        return '<span class="match-label match-best">Best Match</span>'
    if confidence >= _CONFIDENCE_GOOD:
        return '<span class="match-label match-good">Good Match</span>'
    return ""


def _is_positive_explanation(explanation: str) -> bool:
    """Return False if the explanation says the result is NOT relevant."""
    if not explanation:
        return False
    lower = explanation.lower()
    return not any(phrase in lower for phrase in _NEGATIVE_PHRASES)


def _render_card_body(result: dict, search_hit: dict | None) -> None:
    """Render a single result card using native Streamlit layout."""
    rid = str(result.get("id", ""))
    title = search_hit.get("title", rid) if search_hit else rid
    source = search_hit.get("source_fields", {}) if search_hit else {}

    description = _truncate(source.get("description", ""), 280)
    category = source.get("category", "")
    img_key = _ui_schema.ui_image_field
    poster = source.get(img_key, "") if img_key else ""
    year = ""
    iso = source.get("indexed_at_iso", "")
    if iso:
        year = iso[:4]

    tags = _facet_tags_from_source(source)
    if not tags and category:
        tags = [category]

    confidence = result.get("confidence", 0.0)
    explanation = result.get("explanation", "")
    expl_status = result.get("explanation_status", "ABSENT")
    if not isinstance(expl_status, str):
        expl_status = getattr(expl_status, "value", "ABSENT")

    show_insight = (
        explanation
        and expl_status not in ("ABSENT", "DEGRADED")
        and confidence >= _CONFIDENCE_GOOD
        and _is_positive_explanation(explanation)
    )

    if img_key:
        col_poster, col_body = st.columns([1, 7])
        with col_poster:
            if poster:
                st.image(poster, width=90)
            else:
                st.image(POSTER_PLACEHOLDER, width=90)
    else:
        col_body = st.container()

    with col_body:
        year_str = f"  ({year})" if year else ""
        match_label = _match_label_html(confidence)
        st.markdown(
            f'<span style="font-size:17px;font-weight:700;color:#e6edf3;">'
            f'{html.escape(title)}</span>'
            f'<span style="font-size:13px;color:#484f58;">{html.escape(year_str)}</span>'
            f'{match_label}',
            unsafe_allow_html=True,
        )

        facet_html = " ".join(
            f'<span class="facet-tag">{html.escape(g)}</span>' for g in tags[:5]
        )
        if facet_html:
            st.markdown(facet_html, unsafe_allow_html=True)

        if description:
            st.markdown(
                f'<span style="font-size:13px;color:#8b949e;line-height:1.5;">'
                f'{html.escape(description)}</span>',
                unsafe_allow_html=True,
            )

        if show_insight:
            st.markdown(
                f'<div class="ai-insight">'
                f'<div class="ai-insight-label">Why this matches</div>'
                f'{html.escape(explanation)}'
                f'</div>',
                unsafe_allow_html=True,
            )


def _render_pipeline_status_badges(fr: dict) -> None:
    """Render partial_results / rerank_degraded status badges above results."""
    partial = fr.get("partial_results", False)
    degraded = fr.get("rerank_degraded", False)
    if not partial and not degraded:
        return
    badges_html = ""
    if partial:
        badges_html += (
            '<span class="status-badge badge-warn">'
            '⚠ Partial Results — keyword fallback active'
            '</span>'
        )
    if degraded:
        badges_html += (
            '<span class="status-badge badge-warn">'
            '⚠ Rerank Degraded — using original ranking order'
            '</span>'
        )
    st.markdown(
        f'<div style="margin-bottom:10px;">{badges_html}</div>',
        unsafe_allow_html=True,
    )


def render_serp(final_state: dict) -> None:
    """Render the consumer-facing search engine results page."""
    fr = final_state.get("final_response", {})
    if not fr:
        st.info("No final response available.")
        return

    blocked = fr.get("blocked", False)
    if blocked:
        st.error("This query was blocked by our safety system.Please try with another query")
        return

    _render_pipeline_status_badges(fr)

    results = fr.get("results", [])
    result_source = fr.get("result_source", "search")
    result_count = fr.get("result_count", len(results))

    if not results:
        st.info("No results matched your query. Try different keywords.")
        return

    search_lookup = _build_result_lookup(final_state.get("search_results", []))

    top_results = []
    other_results = []

    for idx, r in enumerate(results):
        rd = _to_dict(r)
        rid = str(rd.get("id", ""))
        search_hit = search_lookup.get(rid)

        if result_source == "reranked":
            confidence = rd.get("confidence", 0.0)
        else:
            confidence = rd.get("score", 0.0)
            rd = {"id": rid, "confidence": confidence}

        if confidence >= _CONFIDENCE_GOOD:
            top_results.append((rd, search_hit))
        else:
            other_results.append((rd, search_hit))

    total_label = f"{result_count} result{'s' if result_count != 1 else ''}"

    if not top_results:
        st.markdown(
            f'<span style="font-size:14px;color:#8b949e;">'
            f'<strong style="color:#e6edf3;">{total_label}</strong> found</span>',
            unsafe_allow_html=True,
        )
        for rd, search_hit in other_results:
            with st.container(border=True):
                _render_card_body(rd, search_hit)
    elif not other_results:
        st.markdown(
            f'<span style="font-size:14px;color:#8b949e;">'
            f'<strong style="color:#e6edf3;">{total_label}</strong> found</span>',
            unsafe_allow_html=True,
        )
        for rd, search_hit in top_results:
            with st.container(border=True):
                _render_card_body(rd, search_hit)
    else:
        shown = len(top_results)
        st.markdown(
            f'<span style="font-size:14px;color:#8b949e;">'
            f'Showing <strong style="color:#e6edf3;">{shown}</strong> '
            f'top result{"s" if shown != 1 else ""} of {total_label}</span>',
            unsafe_allow_html=True,
        )
        for rd, search_hit in top_results:
            with st.container(border=True):
                _render_card_body(rd, search_hit)
        with st.expander(f"More results ({len(other_results)})"):
            for rd, search_hit in other_results:
                with st.container(border=True):
                    _render_card_body(rd, search_hit)


# ═════════════════════════════════════════════════════════════════════════════
#  LAYOUT
# ═════════════════════════════════════════════════════════════════════════════

st.markdown(
    f'<div class="page-title">{html.escape(_ui_schema.ui_product_title)}</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="page-sub">{html.escape(_ui_schema.ui_product_subtitle)}</div>',
    unsafe_allow_html=True,
)

# ── Search input ──────────────────────────────────────────────────────────────

col_input, col_hits, col_btn = st.columns([5, 1, 1])

with col_input:
    query = st.text_input(
        "Query",
        placeholder=_ui_schema.ui_query_placeholder,
        label_visibility="collapsed",
    )
with col_hits:
    max_hits = st.number_input("Max hits", min_value=5, max_value=200, value=25)
with col_btn:
    run = st.button("Search", type="primary", use_container_width=True)

# ── Run pipeline ──────────────────────────────────────────────────────────────

if run and not query.strip():
    st.warning("Enter a search query.")
    st.stop()

if run and query.strip():
    with st.spinner("Searching…"):
        try:
            final_state, trace = run_search_with_trace(
                query.strip(),
                session_id=st.session_state.pipeline_session_id,
            )
        except Exception as exc:
            st.exception(exc)
            st.stop()

    # ── Tabs: Search Results (consumer) + Pipeline Inspector (debug) ───────

    tab_results, tab_pipeline = st.tabs(["Search Results", "Pipeline Inspector"])

    # ── Tab 1: Search Results (end-user SERP) ─────────────────────────────
    with tab_results:
        render_serp(final_state)

    # ── Tab 2: Pipeline Inspector (debug per-node + raw state) ────────────
    with tab_pipeline:
        render_pipeline_flow()

        node_counts: dict[str, int] = {}
        sub_labels: list[str] = []
        for step in trace:
            name = step["node"]
            node_counts[name] = node_counts.get(name, 0) + 1
            n = node_counts[name]
            label = NODE_LABELS.get(name, name)
            sub_labels.append(label if n == 1 else f"{label} ({n})")

        sub_labels.append("raw state")
        sub_tabs = st.tabs(sub_labels)

        for i, tab in enumerate(sub_tabs[:-1]):
            with tab:
                step = trace[i]
                name = step["node"]
                delta = step.get("output", {})

                css = NODE_COLORS.get(name, "")
                label = NODE_LABELS.get(name, name)
                st.markdown(f'<span class="node-tag {css}">{label}</span>', unsafe_allow_html=True)

                if not delta:
                    st.markdown(
                        '<p style="color:#484f58;font-size:13px;font-style:italic;">No state changes at this step.</p>',
                        unsafe_allow_html=True,
                    )
                elif name == "reporter":
                    _render_reporter_tab(delta, int(max_hits))
                else:
                    filtered = _filter_node_delta(name, to_jsonable(delta, max_search_hits=int(max_hits)))
                    if filtered:
                        st.json(filtered)
                    else:
                        st.markdown(
                            '<p style="color:#484f58;font-size:13px;font-style:italic;">No unique state changes at this step.</p>',
                            unsafe_allow_html=True,
                        )

        with sub_tabs[-1]:
            st.markdown(
                '<div class="section-label">complete merged state</div>',
                unsafe_allow_html=True,
            )
            st.json(to_jsonable(final_state, max_search_hits=int(max_hits)))