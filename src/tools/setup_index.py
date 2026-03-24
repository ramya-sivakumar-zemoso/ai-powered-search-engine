"""Create index, set attributes + HuggingFace embedder, batch documents, wait for embeddings."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.models.schema_registry import get_schema
from src.tools.dataset_loader import load_and_normalise
from src.tools.meilisearch_client import (
    _headers,
    _url,
    get_index_stats,
    health,
    wait_for_task,
)
from src.utils.config import get_settings

settings = get_settings()


def create_index(index_name: str) -> dict:
    resp = requests.post(
        _url("/indexes"),
        headers=_headers(),
        json={"uid": index_name, "primaryKey": "id"},
    )
    if resp.status_code == 409:
        print(f"  Index '{index_name}' already exists — skipping.")
        return {}
    resp.raise_for_status()
    task = resp.json()
    try:
        wait_for_task(task["taskUid"], interval=2)
    except RuntimeError as exc:
        if "already exists" in str(exc).lower():
            print(f"  Index '{index_name}' already exists — skipping.")
            return {}
        raise
    return task


def configure_attributes(index_name: str, schema) -> None:
    base = _url(f"/indexes/{index_name}/settings")
    parts = [
        ("searchable-attributes", schema.searchable_fields, "Searchable"),
        ("filterable-attributes", schema.filterable_fields, "Filterable"),
        ("sortable-attributes", schema.sortable_fields, "Sortable"),
    ]
    for suffix, payload, label in parts:
        resp = requests.put(
            f"{base}/{suffix}",
            headers=_headers(),
            json=payload,
        )
        resp.raise_for_status()
        wait_for_task(resp.json()["taskUid"], interval=2)
        print(f"  {label}: {payload}")


def configure_embedder(index_name: str, schema) -> dict:
    resp = requests.patch(
        _url(f"/indexes/{index_name}/settings/embedders"),
        headers=_headers(),
        json={
            settings.meili_embedder_name: {
                "source": "huggingFace",
                "model": settings.embedding_model,
                "documentTemplate": schema.embedder_template,
            }
        },
    )
    resp.raise_for_status()
    task = resp.json()
    print(f"  Embedder '{settings.meili_embedder_name}' configured.")
    print(f"  Template: {schema.embedder_template}")
    return task


def add_documents(index_name: str, documents: list[dict]) -> None:
    batch_size = 500
    total = len(documents)
    for i in range(0, total, batch_size):
        batch = documents[i : i + batch_size]
        resp = requests.post(
            _url(f"/indexes/{index_name}/documents"),
            headers=_headers(),
            json=batch,
        )
        resp.raise_for_status()
        task_uid = resp.json()["taskUid"]
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
    _ = reset  # CLI flag preserved; same as prior version (no drop-index step yet)
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
    print("  bge-large-en will be downloaded on first run (~1.3 GB).")
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
