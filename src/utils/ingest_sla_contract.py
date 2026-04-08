"""Machine-readable ingest SLA contract (no FastAPI dependency)."""

from __future__ import annotations

from typing import Any

from src.utils.config import Settings


def sla_contract_payload(settings: Settings) -> dict[str, Any]:
    """Return ``sla_seconds`` plus ``sla_contract`` for GET /ingest/health and monitors."""
    sla = settings.ingest_sla_seconds
    return {
        "sla_seconds": sla,
        "sla_contract": {
            "sla_seconds": sla,
            "measurement": (
                "Time from this service accepting POST /ingest/document or "
                "/ingest/batch until the Meilisearch indexing task reports "
                "status succeeded (blocking wait). Excludes upstream latency "
                "and Kafka consumer queue lag before the message is processed."
            ),
            "breach_behavior": (
                "HTTP 408 on timeout; Kafka consumer skips offset commit for "
                "the failed batch so messages are redelivered."
            ),
            "live_paths": [
                "POST /ingest/document",
                "POST /ingest/batch",
                "Kafka consumer (KAFKA_ENABLED=true, python -m src.tools.kafka_consumer)",
            ],
            "bootstrap_only": (
                "python -m src.tools.setup_index — initial/bulk load, not the "
                "live streaming path."
            ),
        },
    }
