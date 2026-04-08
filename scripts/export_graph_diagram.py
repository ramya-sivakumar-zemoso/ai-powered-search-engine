"""Export a programmatic LangGraph diagram (PNG/SVG/Mermaid text).

This script intentionally builds a lightweight topology-only graph so it can
run in environments missing runtime dependencies (e.g., meilisearch client).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _build_topology_only_graph():
    from langgraph.graph import END, START, StateGraph
    from src.models.state import SearchStateDict

    graph = StateGraph(SearchStateDict)

    # Keep node names identical to production graph.
    graph.add_node("query_understander", lambda state: {})
    graph.add_node("retrieval_router", lambda state: {})
    graph.add_node("searcher", lambda state: {})
    graph.add_node("evaluator", lambda state: {})
    graph.add_node("reranker", lambda state: {})
    graph.add_node("reporter", lambda state: {})

    graph.add_edge(START, "query_understander")
    graph.add_edge("retrieval_router", "searcher")
    graph.add_edge("reranker", "reporter")
    graph.add_edge("reporter", END)

    graph.add_conditional_edges(
        "query_understander",
        lambda state: "retrieval_router",
        {"retrieval_router": "retrieval_router", "reporter": "reporter"},
    )
    graph.add_conditional_edges(
        "searcher",
        lambda state: "evaluator",
        {"evaluator": "evaluator", "reporter": "reporter"},
    )
    graph.add_conditional_edges(
        "evaluator",
        lambda state: "reranker",
        {
            "reranker": "reranker",
            "retrieval_router": "retrieval_router",
            "reporter": "reporter",
        },
    )

    return graph.compile()


def _write_fallback_svg(path: Path) -> None:
    """Render a minimal static SVG for the LangGraph topology."""
    nodes = [
        ("START", 40, 50),
        ("query_understander", 220, 50),
        ("retrieval_router", 420, 50),
        ("searcher", 620, 50),
        ("evaluator", 820, 50),
        ("reranker", 1020, 50),
        ("reporter", 1220, 50),
        ("END", 1420, 50),
    ]
    node_lookup = {name: (x, y) for name, x, y in nodes}
    edges = [
        ("START", "query_understander"),
        ("query_understander", "retrieval_router"),
        ("query_understander", "reporter"),
        ("retrieval_router", "searcher"),
        ("searcher", "evaluator"),
        ("searcher", "reporter"),
        ("evaluator", "reranker"),
        ("evaluator", "retrieval_router"),
        ("evaluator", "reporter"),
        ("reranker", "reporter"),
        ("reporter", "END"),
    ]

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="220">',
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" '
        'markerWidth="8" markerHeight="8" orient="auto-start-reverse">',
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#333"/>',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#fafafa"/>',
    ]

    for src, dst in edges:
        x1, y1 = node_lookup[src]
        x2, y2 = node_lookup[dst]
        lines.append(
            f'<line x1="{x1+140}" y1="{y1+25}" x2="{x2}" y2="{y2+25}" '
            'stroke="#666" stroke-width="2" marker-end="url(#arrow)"/>'
        )

    for name, x, y in nodes:
        fill = "#fff4cc" if name not in ("START", "END") else "#d9f2e6"
        lines.append(
            f'<rect x="{x}" y="{y}" width="140" height="50" rx="8" '
            f'fill="{fill}" stroke="#444"/>'
        )
        lines.append(
            f'<text x="{x+70}" y="{y+30}" text-anchor="middle" '
            'font-family="Arial, sans-serif" font-size="12" fill="#111">'
            f"{name}</text>"
        )

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path} (fallback SVG)")


def main() -> None:
    from langchain_core.runnables.graph import MermaidDrawMethod

    out_dir = Path("src/graph")
    out_dir.mkdir(parents=True, exist_ok=True)

    app = _build_topology_only_graph()
    graph = app.get_graph()

    mmd_path = out_dir / "graph_programmatic.mmd"
    mmd_path.write_text(graph.draw_mermaid(), encoding="utf-8")
    print(f"Wrote {mmd_path}")

    png_path = out_dir / "graph_programmatic.png"
    try:
        png_bytes = graph.draw_png()
        png_path.write_bytes(png_bytes)
        print(f"Wrote {png_path} (draw_png)")
    except Exception as exc:
        print(f"PNG export via draw_png failed: {exc}")
        try:
            png_bytes = graph.draw_mermaid_png()
            png_path.write_bytes(png_bytes)
            print(f"Wrote {png_path} (mermaid API)")
        except Exception:
            try:
                png_bytes = graph.draw_mermaid_png(
                    draw_method=MermaidDrawMethod.PYPPETEER
                )
                png_path.write_bytes(png_bytes)
                print(f"Wrote {png_path} (pyppeteer)")
            except Exception as local_exc:
                print(f"PNG export skipped: {local_exc}")

    svg_path = out_dir / "graph_programmatic.svg"
    _write_fallback_svg(svg_path)


if __name__ == "__main__":
    main()
