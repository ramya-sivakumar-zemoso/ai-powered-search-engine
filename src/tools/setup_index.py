"""Create index, set attributes + HuggingFace embedder, batch documents, wait for embeddings.

Uses the official ``meilisearch`` Python SDK for all API calls.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.models.schema_registry import get_schema
from src.tools.dataset_loader import load_and_normalise
from src.tools.meilisearch_client import (
    _get_client,
    get_index_stats,
    health,
    wait_for_task,
)
from src.utils.config import get_settings

settings = get_settings()


def create_index(index_name: str) -> dict:
    """Create a Meilisearch index (idempotent — skips if it already exists)."""
    client = _get_client()
    try:
        task_info = client.create_index(index_name, {"primaryKey": "id"})
        task_uid = task_info.task_uid
        wait_for_task(task_uid, interval=2)
        return {"taskUid": task_uid}
    except Exception as exc:
        if "already exists" in str(exc).lower() or "index_already_exists" in str(exc).lower():
            print(f"  Index '{index_name}' already exists — skipping.")
            return {}
        raise


def configure_attributes(index_name: str, schema) -> None:
    """Set searchable, filterable, and sortable attributes on the index."""
    index = _get_client().index(index_name)

    parts = [
        (index.update_searchable_attributes, schema.searchable_fields, "Searchable"),
        (index.update_filterable_attributes, schema.filterable_fields, "Filterable"),
        (index.update_sortable_attributes, schema.sortable_fields, "Sortable"),
    ]
    for update_fn, payload, label in parts:
        task_info = update_fn(payload)
        wait_for_task(task_info.task_uid, interval=2)
        print(f"  {label}: {payload}")


def configure_embedder(index_name: str, schema) -> dict:
    """Configure the HuggingFace embedder on the index."""
    index = _get_client().index(index_name)
    embedder_config = {
        settings.meili_embedder_name: {
            "source": "huggingFace",
            "model": settings.embedding_model,
            "dimensions": settings.embedding_dimensions,
            "documentTemplate": schema.embedder_template,
        }
    }
    task_info = index.update_embedders(embedder_config)
    print(f"  Embedder '{settings.meili_embedder_name}' configured.")
    print(f"  Template: {schema.embedder_template}")
    return {"taskUid": task_info.task_uid}


def add_documents(index_name: str, documents: list[dict]) -> None:
    """Add documents in batches of 500."""
    index = _get_client().index(index_name)
    batch_size = 500
    total = len(documents)
    for i in range(0, total, batch_size):
        batch = documents[i : i + batch_size]
        task_info = index.add_documents(batch)
        task_uid = task_info.task_uid
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        print(f"  Batch {batch_num}/{total_batches} → task {task_uid}")
        wait_for_task(task_uid, interval=3)


def setup(
    file_path: str,
    schema_name: str,
    limit: int | None = None,
    reset: bool = False,
) -> None:
    _ = reset
    index_name = settings.meili_index_name

    print("\n── Checking Meilisearch ──────────────────────────────────────")
    try:
        h = health()
        print(f"  Status: {h.get('status', 'ok')}")
    except Exception as exc:
        print(f"\n Cannot reach Meilisearch at {settings.meili_url}")
        print(f"    Error: {exc}")
        sys.exit(1)

    print(f"\n── Creating index '{index_name}' ──────────────────────────────")
    create_index(index_name)

    print("\n── Loading dataset ───────────────────────────────────────────")
    print(f"  File:   {file_path}")
    print(f"  Schema: {schema_name}")
    schema = get_schema(schema_name)
    documents = load_and_normalise(file_path, schema, limit=limit)
    print(f"  Normalised: {len(documents):,} documents")

    print("\n── Configuring index attributes ──────────────────────────────")
    configure_attributes(index_name, schema)

    print("\n── Configuring embedder ──────────────────────────────────────")
    embedder_task = configure_embedder(index_name, schema)

    print(f"\n── Indexing {len(documents):,} documents ─────────────────────")
    add_documents(index_name, documents)

    print("\n── Waiting for embedding generation ─────────────────────────")
    print("  Embedding model will be downloaded on first run.")
    print("  This can take 5–30 minutes depending on connection and CPU.")
    wait_for_task(embedder_task["taskUid"], interval=10, timeout=3600)

    print("\n── Done ──────────────────────────────────────────────────────")
    stats = get_index_stats(index_name)
    print(f"  Documents in index: {stats.get('number_of_documents', '?'):,}")
    print("\n Setup complete. Run verify:")
    print(f"    python scripts/verify_search.py --schema {schema_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set up Meilisearch index for the search pipeline."
    )
    parser.add_argument("--file", default=os.getenv("DATASET_FILE", "data/movies.json"))
    parser.add_argument("--schema", default=os.getenv("DATASET_SCHEMA", "movies"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the index before setup",
    )
    args = parser.parse_args()
    setup(
        file_path=args.file,
        schema_name=args.schema,
        limit=args.limit,
        reset=args.reset,
    )


if __name__ == "__main__":
    main()
