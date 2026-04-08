"""Kafka consumer — real-time ingest loop for the search-ingest topic.

This is the application-side Kafka consumer.  It:
  1. Subscribes to the configured topic (``KAFKA_TOPIC``).
  2. Polls for messages in a tight loop.
  3. Decodes each message and calls ``upsert_documents()`` — the same
     idempotent Meilisearch upsert used by the FastAPI ingest endpoint.
  4. Commits offsets **only after** Meilisearch confirms the document
     is indexed (``status = succeeded``), guaranteeing at-least-once
     delivery end-to-end.
  5. Emits structured logs per message for SLA monitoring and traceability.

Run:
    python -m src.tools.kafka_consumer          # foreground, Ctrl-C to stop
    python -m src.tools.kafka_consumer --once   # consume one batch then exit (CI/test)

Design decisions
----------------
**Offset commit after Meilisearch ACK (not Kafka ACK):**
  Kafka's auto-commit or immediate offset commit would mark a message as
  processed even if Meilisearch rejected it or timed out.  We commit
  manually only after ``upsert_documents()`` returns ``status=succeeded``.
  This guarantees that on consumer restart, unprocessed messages are
  re-delivered and retried.  Meilisearch upserts are idempotent (same ``id``
  = in-place update), so re-delivery is always safe.

**Micro-batching (KAFKA_MAX_POLL_RECORDS):**
  A single ``poll()`` returns up to ``KAFKA_MAX_POLL_RECORDS`` messages.
  These are sent to Meilisearch as one batch call (one task, one task_uid
  to poll).  This reduces Meilisearch API calls and improves throughput
  while keeping individual batch sizes small enough to complete within the
  5-minute SLA.  Default is 10 documents per poll cycle.

**Schema routing via Kafka message headers:**
  The producer stamps a ``schema_name`` header on each message.  The consumer
  reads this header to apply the correct ``DatasetSchema`` field mapping.
  Fallback: ``DATASET_SCHEMA`` env var.  This means multiple domains
  (movies, marketplace, sports) can share a single topic — partitioned by
  domain via the ``schema_name`` header — without needing separate topics.

**Tombstone (soft-delete) support:**
  Messages with a ``deleted_at`` field are upserted as-is.  The Meilisearch
  filter ``deleted_at IS NULL`` hides them at query time.  No hard-delete
  Kafka semantics (null-value tombstone) are used to keep the consumer logic
  simple and auditable.

**SASL / TLS support:**
  Set ``KAFKA_SECURITY_PROTOCOL=SASL_SSL``, ``KAFKA_SASL_MECHANISM``, and
  credentials in ``.env`` to connect to Redpanda Serverless, Confluent Cloud,
  or any managed Kafka broker.  No code change required.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from typing import Any

from src.models.schema_registry import get_schema
from src.tools.meilisearch_client import upsert_documents
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

try:
    from confluent_kafka import Consumer, KafkaError
    _CONFLUENT_AVAILABLE = True
except ImportError:
    _CONFLUENT_AVAILABLE = False

_POLL_TIMEOUT_S = 1.0  # how long to block waiting for messages per poll


# ── Config ────────────────────────────────────────────────────────────────────

def _build_consumer_config() -> dict[str, Any]:
    """Build confluent-kafka consumer config from Settings."""
    cfg: dict[str, Any] = {
        "bootstrap.servers": settings.kafka_bootstrap_servers,
        "group.id": settings.kafka_consumer_group,
        "auto.offset.reset": "earliest",          # pick up from the start on first run
        "enable.auto.commit": False,               # we commit manually after Meili ACK
        "max.poll.interval.ms": 600_000,           # 10 min — generous for slow Meili tasks
        "session.timeout.ms": 45_000,
        "heartbeat.interval.ms": 15_000,
    }
    if settings.kafka_security_protocol != "PLAINTEXT":
        cfg["security.protocol"] = settings.kafka_security_protocol
        cfg["sasl.mechanism"] = settings.kafka_sasl_mechanism
        cfg["sasl.username"] = settings.kafka_sasl_username
        cfg["sasl.password"] = settings.kafka_sasl_password
    return cfg


# ── Message decoding ──────────────────────────────────────────────────────────

def _decode_message(msg: Any) -> tuple[dict[str, Any], str]:
    """Decode a Kafka message into (document, schema_name).

    Message value format (produced by kafka_producer.py):
        {"schema_name": "marketplace", "document": {...}}

    Falls back to treating the value as a bare document dict with schema from headers.
    """
    raw = msg.value()
    if raw is None:
        return {}, settings.dataset_schema  # null-value tombstone — ignore

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.error(
            "kafka_decode_error",
            extra={"error": str(exc), "offset": msg.offset(), "partition": msg.partition()},
        )
        return {}, settings.dataset_schema

    # Envelope format from our producer
    if "document" in payload and isinstance(payload["document"], dict):
        schema_name = payload.get("schema_name", settings.dataset_schema)
        return payload["document"], schema_name

    # Bare document — schema from headers
    schema_name = settings.dataset_schema
    if msg.headers():
        for key, value in msg.headers():
            if key == "schema_name" and value:
                schema_name = value.decode("utf-8")
                break
    return payload, schema_name


def _apply_schema(document: dict[str, Any], schema_name: str) -> dict[str, Any] | None:
    """Apply DatasetSchema field mapping to normalise the document.

    Returns ``None`` when ``DatasetSchema.apply`` rejects the row (e.g. title too short).
    """
    try:
        schema = get_schema(schema_name)
        if hasattr(schema, "apply"):
            row_index = abs(hash(str(document.get("id", "")))) % (10**9)
            return schema.apply(document, row_index)
    except Exception as exc:
        logger.warning(
            "schema_apply_failed",
            extra={"schema": schema_name, "error": str(exc)},
        )
    return document


# ── Consumer loop ─────────────────────────────────────────────────────────────

class SearchIngestConsumer:
    """Long-running Kafka consumer for the search-ingest topic.

    Calls ``upsert_documents()`` for each micro-batch, commits Kafka offsets
    only after Meilisearch confirms successful indexing.
    """

    def __init__(self) -> None:
        if not _CONFLUENT_AVAILABLE:
            raise ImportError(
                "confluent-kafka is required for Kafka ingestion. "
                "Install it with: pip install confluent-kafka"
            )
        self._consumer = Consumer(_build_consumer_config())
        self._topic = settings.kafka_topic
        self._running = True
        self._stats = {"processed": 0, "failed": 0, "sla_breaches": 0}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self, *, once: bool = False) -> None:
        """Subscribe and run the consume loop.

        Args:
            once: If True, consume one poll batch then return (useful for
                  integration tests and CI smoke tests).
        """
        self._consumer.subscribe(
            [self._topic],
            on_assign=self._on_assign,
            on_revoke=self._on_revoke,
        )
        logger.info(
            "kafka_consumer_started",
            extra={
                "topic": self._topic,
                "group": settings.kafka_consumer_group,
                "bootstrap": settings.kafka_bootstrap_servers,
            },
        )

        try:
            while self._running:
                self._poll_and_process()
                if once:
                    break
        except KeyboardInterrupt:
            logger.info("kafka_consumer_interrupted")
        finally:
            self._consumer.close()
            logger.info("kafka_consumer_stopped", extra=self._stats)

    def stop(self) -> None:
        """Signal the consume loop to exit cleanly."""
        self._running = False

    # ── Core poll-and-process ─────────────────────────────────────────────────

    def _commit_single(self, msg: Any, reason: str) -> None:
        """Commit one message offset so bad payloads do not block the partition forever."""
        try:
            self._consumer.commit(msg, asynchronous=False)
        except Exception as exc:
            logger.error(
                "kafka_single_commit_failed",
                extra={"reason": reason, "error": str(exc)},
            )

    def _normalise_message(
        self,
        msg: Any,
    ) -> tuple[dict[str, Any], str] | None:
        """Return (normalised_document, schema_name) or None if message should be skipped."""
        if msg.error():
            err = msg.error()
            if err.code() == KafkaError._PARTITION_EOF:
                return None
            logger.error("kafka_message_error", extra={"code": err.code(), "error": str(err)})
            return None

        document, schema_name = _decode_message(msg)
        if not document or not document.get("id"):
            logger.warning(
                "kafka_message_skipped",
                extra={
                    "reason": "missing_id_or_empty",
                    "offset": msg.offset(),
                    "partition": msg.partition(),
                },
            )
            self._commit_single(msg, "skip_invalid_payload")
            return None

        normalised = _apply_schema(document, schema_name)
        if normalised is None:
            logger.warning(
                "kafka_message_skipped",
                extra={
                    "reason": "schema_rejected",
                    "offset": msg.offset(),
                    "partition": msg.partition(),
                    "id": document.get("id"),
                },
            )
            self._commit_single(msg, "skip_schema_rejected")
            return None
        return normalised, schema_name

    def _poll_and_process(self) -> None:
        """Poll up to KAFKA_MAX_POLL_RECORDS messages and upsert as one batch."""
        messages = self._consumer.consume(
            num_messages=settings.kafka_max_poll_records,
            timeout=_POLL_TIMEOUT_S,
        )
        if not messages:
            return

        # Decode all messages in this batch, group by schema_name for efficient
        # batching (same schema → same DatasetSchema.apply() call).
        batches: dict[str, list[dict[str, Any]]] = {}
        raw_messages = []

        for msg in messages:
            normalised_payload = self._normalise_message(msg)
            if normalised_payload is None:
                continue
            normalised, schema_name = normalised_payload
            batches.setdefault(schema_name, []).append(normalised)
            raw_messages.append(msg)

        if not batches:
            return

        # Upsert each schema group into Meilisearch, commit only on success.
        all_ok = True
        for schema_name, docs in batches.items():
            all_ok &= self._upsert_batch(docs, schema_name=schema_name)

        if all_ok:
            # Commit the latest offset for each partition in this batch.
            # confluent-kafka commit() without args commits the last consumed position.
            self._consumer.commit(asynchronous=False)
            logger.debug(
                "kafka_offsets_committed",
                extra={"message_count": len(raw_messages)},
            )
        else:
            # Do NOT commit — consumer will re-receive these messages on next start.
            logger.warning(
                "kafka_offsets_not_committed",
                extra={
                    "reason": "meilisearch_upsert_failed",
                    "message_count": len(raw_messages),
                },
            )

    def _upsert_batch(
        self,
        documents: list[dict[str, Any]],
        schema_name: str,
    ) -> bool:
        """Call upsert_documents() and return True on success, False on failure."""
        t0 = time.monotonic()
        doc_ids = [str(d.get("id", "?")) for d in documents]
        try:
            result = upsert_documents(
                documents,
                wait=True,
                sla_seconds=settings.ingest_sla_seconds,
            )
            elapsed = result["elapsed_seconds"]
            sla_ok = elapsed <= settings.ingest_sla_seconds
            self._stats["processed"] += len(documents)
            if not sla_ok:
                self._stats["sla_breaches"] += 1

            logger.info(
                "kafka_batch_indexed",
                extra={
                    "schema": schema_name,
                    "count": len(documents),
                    "elapsed_seconds": elapsed,
                    "sla_ok": sla_ok,
                    "task_uid": result.get("task_uid"),
                    "ids": doc_ids[:5],  # log first 5 ids for traceability
                },
            )
            return True

        except TimeoutError as exc:
            self._stats["failed"] += len(documents)
            self._stats["sla_breaches"] += 1
            logger.error(
                "kafka_batch_sla_breach",
                extra={
                    "schema": schema_name,
                    "count": len(documents),
                    "elapsed_s": round(time.monotonic() - t0, 2),
                    "error": str(exc),
                    "ids": doc_ids[:5],
                },
            )
            return False

        except Exception as exc:
            self._stats["failed"] += len(documents)
            logger.error(
                "kafka_batch_failed",
                extra={
                    "schema": schema_name,
                    "count": len(documents),
                    "error": str(exc)[:300],
                    "ids": doc_ids[:5],
                },
            )
            return False

    # ── Rebalance callbacks ───────────────────────────────────────────────────

    def _on_assign(self, consumer: Any, partitions: list[Any]) -> None:
        logger.info(
            "kafka_partitions_assigned",
            extra={"partitions": [p.partition for p in partitions]},
        )

    def _on_revoke(self, consumer: Any, partitions: list[Any]) -> None:
        logger.info(
            "kafka_partitions_revoked",
            extra={"partitions": [p.partition for p in partitions]},
        )


# ── Signal handling + CLI ─────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Start the Kafka search-ingest consumer."
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Consume one poll batch then exit (for CI / integration tests).",
    )
    args = parser.parse_args()

    if not settings.kafka_enabled:
        logger.error(
            "kafka_consumer_disabled",
            extra={
                "message": "KAFKA_ENABLED=false in .env — set KAFKA_ENABLED=true to activate the consumer."
            },
        )
        sys.exit(1)

    consumer = SearchIngestConsumer()

    def _handle_signal(sig: int, _frame: Any) -> None:
        logger.info("kafka_consumer_signal", extra={"signal": sig})
        consumer.stop()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    consumer.start(once=args.once)


if __name__ == "__main__":
    _cli()
