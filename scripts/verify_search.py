"""
scripts/verify_search.py
------------------------
Smoke test: run queries against live Meilisearch to confirm setup worked.

Usage:
    python scripts/verify_search.py
    python scripts/verify_search.py --schema movies
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.tools.meilisearch_client import health as meili_health, search, get_index_stats
from src.utils.config import get_settings

settings = get_settings()

QUERY_SETS: dict[str, list[dict]] = {
    "movies": [
        {"label": "keyword — exact title",    "q": "Interstellar",               "strategy": "KEYWORD"},
        {"label": "semantic — mood/theme",    "q": "space exploration adventure", "strategy": "SEMANTIC", "alpha": 0.8},
        {"label": "hybrid — genre intent",    "q": "funny family animation",      "strategy": "HYBRID",   "alpha": 0.5},
        {"label": "zero result guard",        "q": "xyzzy_nonexistent_title_abc", "strategy": "KEYWORD"},
    ]
}

def run(schema_name: str) -> None:
    # ── Health ────────────────────────────────────────────────────────────────
    try:
        h = meili_health()
        print(f"✅  Meilisearch: {h}\n")
    except Exception as exc:
        print(f"❌  Meilisearch unreachable: {exc}")
        print(f"    Start the binary: ./meilisearch --master-key {settings.meili_master_key}")
        sys.exit(1)

    # ── Document count ────────────────────────────────────────────────────────
    stats = get_index_stats()
    count = stats.get("number_of_documents", 0)
    print(f"📊  Index '{settings.meili_index_name}': {count:,} documents")
    if stats.get("is_indexing"):
        print("    ⚠️  Embeddings still being generated — semantic results may be incomplete")
    if stats.get("unavailable"):
        print("    ⚠️  Stats API unavailable")
    print()

    # ── Query set ─────────────────────────────────────────────────────────────
    queries = QUERY_SETS.get(schema_name, [])
    if not queries:
        print(f"⚠️  No query set for schema '{schema_name}'. Add one to QUERY_SETS in verify_search.py")
        return

    all_passed = True
    for test in queries:
        label    = test["label"]
        query    = test["q"]
        strategy = test["strategy"]
        alpha    = test.get("alpha")
        weights  = {"semanticRatio": alpha} if alpha else None

        try:
            result = search(query=query, strategy=strategy, hybrid_weights=weights, limit=3)
            hits          = result.get("hits", [])
            latency       = result.get("latency_ms", "?")
            strategy_used = result.get("strategy_used", strategy)

            if query.startswith("xyzzy"):
                ok = "✅" if len(hits) == 0 else "⚠️ "
                print(f"{ok}  [{label}]  → {len(hits)} results (expected 0)  [{latency}ms]")
            else:
                ok = "✅" if hits else "❌"
                if not hits:
                    all_passed = False
                print(f"{ok}  [{label}]  '{query}'  → {len(hits)} results  [{latency}ms]  strategy={strategy_used}")
                if hits:
                    print(f"       Top result : {hits[0].get('title', 'N/A')!r}")
                    print(f"       Fields     : {list(hits[0].keys())}")
        except Exception as exc:
            print(f"❌  [{label}] FAILED: {exc}")
            all_passed = False

        print()

    if all_passed:
        print("✅  All checks passed. Ready for pipeline testing.")
    else:
        print("❌  Some checks failed. Review output above.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Meilisearch index after setup")
    parser.add_argument("--schema", default=os.getenv("DATASET_SCHEMA", "movies"))
    args = parser.parse_args()
    run(args.schema)


if __name__ == "__main__":
    main()