"""Create index, set attributes + embedder (HuggingFace, OpenAI, Gemini, or self-hosted), batch documents.

Uses the official ``meilisearch`` Python SDK for all API calls.

Run from the repository root: ``python -m src.tools.setup_index`` (optional ``--file``, ``--schema``, ``--limit``, ``--reset``).

- Gemini: ``EMBEDDER_SOURCE=gemini`` + ``GEMINI_API_KEY`` (Meilisearch ``rest`` → batchEmbedContents).
- Self-hosted: ``EMBEDDER_SOURCE=self_hosted`` + ``EMBEDDING_SERVER_URL`` (OpenAI-compatible server, e.g. HuggingFace TEI).
"""
from __future__ import annotations

import argparse
import os
import sys

from src.models.schema_registry import get_schema
from src.tools.dataset_loader import load_and_normalise
from src.tools.meilisearch_client import (
    _get_client,
    drop_index,
    get_index_stats,
    health,
    wait_for_task,
)
from src.utils.config import get_settings
from src.utils.gemini_meili_embedder import build_gemini_batch_rest_embedder

settings = get_settings()

DOCUMENT_BATCH_SIZE = 500


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
    """Configure the Meilisearch embedder (HF local, OpenAI, or Gemini REST)."""
    index = _get_client().index(index_name)
    src = settings.embedder_source

    if src == "vertex":
        print(
            "  ERROR: EMBEDDER_SOURCE=vertex is no longer supported. "
            "Use local (HuggingFace), openai, or gemini."
        )
        sys.exit(1)

    if src == "openai":
        if not settings.openai_api_key:
            print("  ERROR: EMBEDDER_SOURCE=openai but OPENAI_API_KEY is empty.")
            sys.exit(1)
        embedder_config = {
            settings.meili_embedder_name: {
                "source": "openAi",
                "apiKey": settings.openai_api_key,
                "model": settings.embedding_model,
                "dimensions": settings.embedding_dimensions,
                "documentTemplate": schema.embedder_template,
            }
        }
        print("  Source: OpenAI Embeddings API")
    elif src == "gemini":
        if not settings.gemini_api_key:
            print(
                "  ERROR: EMBEDDER_SOURCE=gemini but GEMINI_API_KEY (or GOOGLE_API_KEY) is empty."
            )
            sys.exit(1)
        d = settings.embedding_dimensions
        cap = settings.gemini_embedding_max_output_dimension
        if d < 1 or d > cap:
            print(
                f"  ERROR: For Gemini embedder, EMBEDDING_DIMENSIONS must be 1..{cap} (got {d}). "
                "Raise GEMINI_EMBEDDING_MAX_OUTPUT_DIMENSION if Google raises the model limit."
            )
            sys.exit(1)
        task = settings.gemini_embedding_task_type or None
        embedder_config = build_gemini_batch_rest_embedder(
            embedder_name=settings.meili_embedder_name,
            api_key=settings.gemini_api_key,
            dimensions=d,
            document_template=schema.embedder_template,
            model_id=settings.embedding_model,
            task_type=task,
        )
        print("  Source: Gemini API (batchEmbedContents, x-goog-api-key)")
    elif src == "self_hosted":
        url = f"{settings.embedding_server_url}/v1/embeddings"
        embedder_config = {
            settings.meili_embedder_name: {
                "source": "rest",
                "apiKey": None,
                "url": url,
                "dimensions": settings.embedding_dimensions,
                "documentTemplate": schema.embedder_template,
                "request": {
                    "input": ["{{text}}", "{{..}}"],
                    "model": settings.embedding_model,
                },
                "response": {
                    "data": [
                        {"embedding": "{{embedding}}"},
                        "{{..}}",
                    ]
                },
            }
        }
        print(f"  Source: Self-hosted ({settings.embedding_server_url})")
    else:
        embedder_config = {
            settings.meili_embedder_name: {
                "source": "huggingFace",
                "model": settings.embedding_model,
                "documentTemplate": schema.embedder_template,
            }
        }
        print("  Source: local HuggingFace (downloaded on first use)")

    task_info = index.update_embedders(embedder_config)
    print(f"  Embedder '{settings.meili_embedder_name}' configured.")
    print(f"  Model: {settings.embedding_model}")
    print(f"  Dimensions: {settings.embedding_dimensions}")
    print(f"  Template: {schema.embedder_template}")
    return {"taskUid": task_info.task_uid}


def add_documents(index_name: str, documents: list[dict]) -> None:
    """Add documents in fixed-size batches."""
    index = _get_client().index(index_name)
    total = len(documents)
    for i in range(0, total, DOCUMENT_BATCH_SIZE):
        batch = documents[i : i + DOCUMENT_BATCH_SIZE]
        task_info = index.add_documents(batch)
        task_uid = task_info.task_uid
        batch_num = i // DOCUMENT_BATCH_SIZE + 1
        total_batches = (total + DOCUMENT_BATCH_SIZE - 1) // DOCUMENT_BATCH_SIZE
        print(f"  Batch {batch_num}/{total_batches} → task {task_uid}")
        wait_for_task(task_uid, interval=3)


def setup(
    file_path: str,
    schema_name: str,
    limit: int | None = None,
    reset: bool = False,
) -> None:
    index_name = settings.meili_index_name

    print("\n── Checking Meilisearch ──────────────────────────────────────")
    try:
        h = health()
        print(f"  Status: {h.get('status', 'ok')}")
    except Exception as exc:
        print(f"\n Cannot reach Meilisearch at {settings.meili_url}")
        print(f"    Error: {exc}")
        sys.exit(1)

    if reset:
        print(f"\n── Reset: drop index '{index_name}' if present ─────────────")
        dropped = drop_index(index_name)
        print(
            f"  {'Dropped' if dropped else 'No existing'} index '{index_name}'."
        )

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
    if settings.embedder_source == "openai":
        print("  OpenAI is embedding documents (API calls; no local model download).")
    elif settings.embedder_source == "gemini":
        print("  Gemini API is embedding documents (remote; rate limits apply).")
    elif settings.embedder_source == "self_hosted":
        print(f"  Self-hosted server at {settings.embedding_server_url} is embedding documents.")
    else:
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
