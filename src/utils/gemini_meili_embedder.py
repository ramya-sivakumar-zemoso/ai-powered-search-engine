"""Meilisearch ``rest`` embedder payload for Gemini API ``batchEmbedContents``.

Follows Meilisearch's Gemini guide and Google's REST shape:
https://www.meilisearch.com/docs/capabilities/hybrid_search/providers/gemini
https://ai.google.dev/api/rest/v1beta/models/batchEmbedContents
"""

from __future__ import annotations

from typing import Any


def _model_resource_name(model_id: str) -> tuple[str, str]:
    """Return (url_path_id, models/foo id) for gemini-embedding-001 style ids."""
    mid = model_id.strip()
    if mid.startswith("models/"):
        mid = mid[len("models/") :]
    return mid, f"models/{mid}"


def build_gemini_batch_rest_embedder(
    *,
    embedder_name: str,
    api_key: str,
    dimensions: int,
    document_template: str,
    model_id: str,
    task_type: str | None = None,
) -> dict[str, Any]:
    """Single-key embedders map for ``index.update_embedders`` (Gemini Developer API)."""
    path_id, model_ref = _model_resource_name(model_id)
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{path_id}:batchEmbedContents"
    )
    req_item: dict[str, Any] = {
        "model": model_ref,
        "content": {"parts": [{"text": "{{text}}"}]},
        "outputDimensionality": dimensions,
    }
    if task_type:
        req_item["taskType"] = task_type

    return {
        embedder_name: {
            "source": "rest",
            # Meilisearch adds Authorization: Bearer when apiKey is set; Gemini expects
            # x-goog-api-key only — a stale apiKey or MEILI_OPENAI_API_KEY causes 401
            # ACCESS_TOKEN_TYPE_UNSUPPORTED from Google. null clears Bearer auth.
            "apiKey": None,
            "url": url,
            "dimensions": dimensions,
            "documentTemplate": document_template,
            "headers": {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            "request": {
                "requests": [
                    req_item,
                    "{{..}}",
                ]
            },
            "response": {
                "embeddings": [
                    {"values": "{{embedding}}"},
                    "{{..}}",
                ]
            },
        }
    }
