"""
NovaMart search pipeline inspector.

Run:
    streamlit run streamlit_app.py
"""

import streamlit as st
from src.graph.graph import run_search_with_trace
from src.utils.state_display import to_jsonable

st.set_page_config(page_title="NovaMart Search", layout="wide", page_icon="🔍")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0d1117; }
.block-container { padding-top: 2rem; max-width: 1200px; }

.page-title { font-size: 26px; font-weight: 700; color: #e6edf3; letter-spacing: -0.4px; }
.page-sub   { font-size: 13px; color: #484f58; margin-top: 2px; margin-bottom: 1.5rem; }

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

.section-label {
    font-size: 10px; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #484f58;
    border-bottom: 1px solid #21262d;
    padding-bottom: 6px; margin-bottom: 10px;
}

.stat-box {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 6px; padding: 12px 14px; text-align: center;
}
.stat-val   { font-size: 22px; font-weight: 700; color: #e6edf3; }
.stat-label { font-size: 11px; color: #484f58; text-transform: uppercase;
              letter-spacing: 0.08em; margin-top: 2px; }

.node-tag {
    display: inline-block; font-size: 10px; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    padding: 2px 8px; border-radius: 3px; margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

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


def node_chip(name: str) -> str:
    css = NODE_COLORS.get(name, "")
    label = NODE_LABELS.get(name, name)
    return f'<span class="node-chip {css}">{label}</span>'


def render_pipeline_flow():
    chips = f' <span class="arrow">→</span> '.join(
        node_chip(n) for n in PIPELINE_ORDER
    )
    st.markdown(f'<div class="pipeline-flow">{chips}</div>', unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown('<div class="page-title">NovaMart AI Search</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">Node-by-node pipeline inspector — each tab shows only what changed at that step.</div>',
    unsafe_allow_html=True,
)
render_pipeline_flow()

# ── Search input ──────────────────────────────────────────────────────────────

col_input, col_hits, col_btn = st.columns([5, 1, 1])

with col_input:
    query = st.text_input(
        "Query", placeholder="e.g. wireless earbuds under $50 with noise cancellation",
        label_visibility="collapsed",
    )
with col_hits:
    max_hits = st.number_input("Max hits", min_value=5, max_value=200, value=25)
with col_btn:
    run = st.button("Run", type="primary", use_container_width=True)

# ── Run pipeline ──────────────────────────────────────────────────────────────

if run and not query.strip():
    st.warning("Enter a search query.")
    st.stop()

if run and query.strip():
    with st.spinner("Running pipeline…"):
        try:
            final_state, trace = run_search_with_trace(query.strip())
        except Exception as exc:
            st.exception(exc)
            st.stop()


    # ── Node tabs ─────────────────────────────────────────────────────────────

    node_counts: dict[str, int] = {}
    tab_labels: list[str] = []
    for step in trace:
        name = step["node"]
        node_counts[name] = node_counts.get(name, 0) + 1
        n = node_counts[name]
        label = NODE_LABELS.get(name, name)
        tab_labels.append(label if n == 1 else f"{label} ({n})")

    tab_labels.append("final state")
    tabs = st.tabs(tab_labels)

    for i, tab in enumerate(tabs[:-1]):
        with tab:
            step = trace[i]
            name = step["node"]
            delta = step.get("output", {})

            css = NODE_COLORS.get(name, "")
            label = NODE_LABELS.get(name, name)
            st.markdown(f'<span class="node-tag {css}">{label}</span>', unsafe_allow_html=True)

            if not delta:
                st.markdown('<p style="color:#484f58;font-size:13px;font-style:italic;">No state changes at this step.</p>', unsafe_allow_html=True)
            else:
                st.json(to_jsonable(delta, max_search_hits=int(max_hits)))

    with tabs[-1]:
        st.markdown('<div class="section-label">complete merged state</div>', unsafe_allow_html=True)
        st.json(to_jsonable(final_state, max_search_hits=int(max_hits)))