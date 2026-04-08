"""Kafka producer — publish listing events to the search-ingest topic.

Usage (programmatic):
    from src.tools.kafka_producer import SearchIngestProducer

    with SearchIngestProducer() as producer:
        producer.publish({"id": "123", "title": "Widget X", "category": "Electronics"})

Usage (CLI — quick test):
    python -m src.tools.kafka_producer --schema marketplace \\
        '{"id": "1", "title": "Test Item", "category": "Books"}'

Design notes
------------
- One producer instance per process; create once, reuse across requests.
- Messages are serialised as UTF-8 JSON.
- The document ``id`` field is used as the Kafka message *key* so that all
  updates for the same listing land on the same partition (ordering guarantee
  per listing).
- ``acks='all'`` ensures the broker acknowledges all in-sync replicas before
  confirming delivery — required for the 5-minute ingest SLA contract.
- Delivery callbacks log success/failure per message for full traceability.
- Compatible with: Apache Kafka, Redpanda (local + Serverless), Confluent Cloud.
  Switch ``KAFKA_SECURITY_PROTOCOL=SASL_SSL`` for managed brokers; no code change.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Any

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

try:
    from confluent_kafka import Producer, KafkaException
    _CONFLUENT_AVAILABLE = True
except ImportError:
    _CONFLUENT_AVAILABLE = False


def _build_producer_config() -> dict[str, Any]:
    """Build confluent-kafka producer config from Settings."""
    cfg: dict[str, Any] = {
        "bootstrap.servers": settings.kafka_bootstrap_servers,
        "acks": "all",
        "retries": 5,
        "retry.backoff.ms": 200,
        "delivery.timeout.ms": 30_000,
        "enable.idempotence": True,
    }
    if settings.kafka_security_protocol != "PLAINTEXT":
        cfg["security.protocol"] = settings.kafka_security_protocol
        cfg["sasl.mechanism"] = settings.kafka_sasl_mechanism
        cfg["sasl.username"] = settings.kafka_sasl_username
        cfg["sasl.password"] = settings.kafka_sasl_password
    return cfg


def _delivery_callback(err: Any, msg: Any) -> None:
    """Called by confluent-kafka after every message delivery attempt."""
    if err:
        logger.error(
            "kafka_delivery_failed",
            extra={
                "topic": msg.topic(),
                "partition": msg.partition(),
                "error": str(err),
            },
        )
    else:
        logger.debug(
            "kafka_delivery_ok",
            extra={
                "topic": msg.topic(),
                "partition": msg.partition(),
                "offset": msg.offset(),
                "key": msg.key().decode() if msg.key() else None,
            },
        )


class SearchIngestProducer:
    """Thread-safe Kafka producer for the search-ingest topic.

    Use as a context manager or call ``close()`` explicitly when done.

    Example::

        with SearchIngestProducer() as p:
            p.publish({"id": "abc", "title": "Laptop Pro", "category": "Electronics"})
    """

    def __init__(self) -> None:
        if not _CONFLUENT_AVAILABLE:
            raise ImportError(
                "confluent-kafka is required for Kafka ingestion. "
                "Install it with: pip install confluent-kafka"
            )
        self._producer = Producer(_build_producer_config())
        self._topic = settings.kafka_topic
        logger.info(
            "kafka_producer_created",
            extra={
                "bootstrap_servers": settings.kafka_bootstrap_servers,
                "topic": self._topic,
            },
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def publish(
        self,
        document: dict[str, Any],
        *,
        schema_name: str | None = None,
    ) -> None:
        """Publish a single listing event to the ingest topic.

        Args:
            document:    Normalised document dict.  Must contain ``id``.
            schema_name: Optional DatasetSchema override. Included in the
                         Kafka message header so the consumer can apply the
                         correct field mapping without extra configuration.

        Raises:
            ValueError:  ``document`` is missing the required ``id`` field.
            KafkaException: Producer configuration error.
        """
        doc_id = str(document.get("id", "")).strip()
        if not doc_id:
            raise ValueError("document must contain a non-empty 'id' field")

        # Stamp ingest time at the producer side for end-to-end latency tracking.
        if "indexed_at" not in document:
            document = {**document, "indexed_at": datetime.now(timezone.utc).isoformat()}

        payload = json.dumps(
            {"schema_name": schema_name or settings.dataset_schema, "document": document},
            ensure_ascii=False,
        ).encode("utf-8")

        headers = [("schema_name", (schema_name or settings.dataset_schema).encode())]

        try:
            self._producer.produce(
                topic=self._topic,
                key=doc_id.encode("utf-8"),  # key = id → same listing always hits same partition
                value=payload,
                headers=headers,
                on_delivery=_delivery_callback,
            )
            # Poll to trigger delivery callbacks without blocking indefinitely.
            self._producer.poll(0)
        except KafkaException as exc:
            logger.error("kafka_produce_error", extra={"id": doc_id, "error": str(exc)})
            raise

    def publish_batch(
        self,
        documents: list[dict[str, Any]],
        *,
        schema_name: str | None = None,
    ) -> int:
        """Publish a list of documents and flush once when done.

        Returns:
            Number of documents enqueued.
        """
        for doc in documents:
            self.publish(doc, schema_name=schema_name)
        self.flush()
        return len(documents)

    def publish_tombstone(self, document_id: str) -> None:
        """Publish a soft-delete (tombstone) event for a listing.

        The consumer will upsert the document with ``deleted_at`` set, marking
        it as deleted without hard-removing it from the Meilisearch index.
        Callers filter ``deleted_at IS NULL`` in search queries.
        """
        self.publish(
            {
                "id": document_id,
                "deleted_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def flush(self, timeout: float = 10.0) -> int:
        """Block until all enqueued messages are delivered or timeout expires.

        Returns:
            Number of messages still in the queue (0 = all delivered).
        """
        remaining = self._producer.flush(timeout=timeout)
        if remaining > 0:
            logger.warning(
                "kafka_flush_incomplete",
                extra={"remaining_messages": remaining, "timeout_s": timeout},
            )
        return remaining

    def close(self) -> None:
        self.flush()

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "SearchIngestProducer":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ── CLI helper ────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Publish a JSON document to the Kafka search-ingest topic."
    )
    parser.add_argument("document", help="JSON string of the document to publish")
    parser.add_argument(
        "--schema", default=None, help="DatasetSchema name (default: DATASET_SCHEMA env var)"
    )
    args = parser.parse_args()

    try:
        doc = json.loads(args.document)
    except json.JSONDecodeError as exc:
        logger.error("kafka_producer_invalid_json", extra={"error": str(exc)})
        sys.exit(1)

    with SearchIngestProducer() as producer:
        producer.publish(doc, schema_name=args.schema)
        producer.flush()
        logger.info(
            "kafka_document_published",
            extra={"document_id": doc.get("id"), "topic": settings.kafka_topic},
        )


if __name__ == "__main__":
    _cli()
