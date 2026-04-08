"""Ingest SLA contract (health endpoint) and optional Meilisearch upsert check."""

from __future__ import annotations

import os

import pytest


def test_sla_contract_payload_structure():
    from src.utils.config import get_settings
    from src.utils.ingest_sla_contract import sla_contract_payload

    settings = get_settings()
    out = sla_contract_payload(settings)
    assert out["sla_seconds"] == settings.ingest_sla_seconds
    contract = out["sla_contract"]
    assert contract["sla_seconds"] == out["sla_seconds"]
    assert "measurement" in contract
    assert "breach_behavior" in contract
    assert "POST /ingest/document" in contract["live_paths"]
    assert "Kafka consumer" in contract["live_paths"][2]


def test_ingest_health_exposes_sla_contract(monkeypatch):
    pytest.importorskip("fastapi")

    def _fake_health():
        return {"status": "available"}

    monkeypatch.setattr("src.tools.ingest_api.health", _fake_health)
    from src.tools.ingest_api import ingest_health

    out = ingest_health()
    assert out["status"] == "ok"
    assert "sla_seconds" in out
    assert "sla_contract" in out


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_MEILI_INGEST_TEST") != "1",
    reason="Set RUN_MEILI_INGEST_TEST=1 with Meilisearch running and index configured.",
)
def test_upsert_document_then_retrievable():
    import uuid

    from src.models.schema_registry import get_schema
    from src.tools.meilisearch_client import delete_document, upsert_documents
    from src.utils.config import get_settings

    settings = get_settings()
    doc_id = f"ingest-contract-{uuid.uuid4().hex[:10]}"
    raw = {
        "id": doc_id,
        "title": "SLA Contract Test Title",
        "overview": "Synthetic row for ingest integration test.",
        "genres": ["IngestTest"],
        "poster": "",
        "release_date": 1_700_000_000,
    }
    schema = get_schema(settings.dataset_schema)
    normalised = schema.apply(raw, 0)
    assert normalised is not None

    try:
        result = upsert_documents([normalised], wait=True)
        assert result["status"] == "succeeded"
        assert result["elapsed_seconds"] <= float(settings.ingest_sla_seconds)

        import meilisearch

        client = meilisearch.Client(settings.meili_url, settings.meili_master_key)
        doc = client.index(settings.meili_index_name).get_document(doc_id)
        assert doc["id"] == doc_id
    finally:
        try:
            delete_document(doc_id, wait=True)
        except Exception:
            pass
