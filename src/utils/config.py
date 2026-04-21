"""Environment-backed settings. Use get_settings() everywhere; avoid os.getenv in callers."""
from __future__ import annotations

import os
from dotenv import load_dotenv
from dataclasses import dataclass
from functools import lru_cache

import src.constants as C

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # ── Meilisearch ──────────────────────────────────────
    meili_url: str
    meili_master_key: str
    meili_index_name: str
    meili_embedder_name: str

    # ── OpenAI ───────────────────────────────────────────
    openai_api_key: str
    openai_model: str

    # ── Embeddings (Meilisearch built-in huggingFace embedder in setup_index) ──
    # Model Hugging Face id and vector size must match the index (re-index after changes).
    embedding_model: str
    embedding_dimensions: int

    # ── LangWatch ───────────────────────────────────────
    langwatch_enabled: bool
    langwatch_api_key: str
    langwatch_project: str

    # ── Pipeline tuning ──────────────────────────────────
    # Max retry loops after the first search (retrieval_router → searcher → evaluator each).
    max_search_iterations: int
    # Max words per user query (extra words dropped before intent/search); 0 = unlimited.
    max_query_words: int
    # Per-query LLM spend cap (USD); cumulative + projected call estimates enforced in nodes.
    token_budget_usd: float
    reranker_top_n: int
    reranker_model: str
    confidence_threshold_degraded: float
    near_duplicate_threshold: float

    # ── Injection detection ──────────────────────────────
    injection_scan_threshold: float

    # ── Freshness thresholds ─────────────────────────────
    freshness_threshold_seconds: int
    staleness_threshold_seconds: int

    # ── Dataset ────────────────────────────────────────────
    dataset_file: str
    dataset_schema: str

    # ── Real-time ingest (Meilisearch task wait ceiling) ───
    ingest_sla_seconds: int

    # ── Latency / performance ──────────────────────────────
    # fast_mode skips LLM explanation generation in the reranker, reducing
    # end-to-end latency by ~400-800 ms at the cost of "Why this matches" text.
    # Use for latency-sensitive integrations; full AI path remains the default.
    fast_mode: bool
    # If true, reranker explanations are generated out-of-band (non-blocking).
    reranker_explain_async: bool
    # Number of top reranked hits to explain (reduces tail latency/cost).
    reranker_explain_top_k: int
    # Tight timeout/retry policy for explanation calls (interactive UX).
    reranker_explain_timeout_seconds: int
    reranker_explain_max_retries: int
    # Retrieval router policy: set false to use Python heuristic routing by default
    # (faster and zero-token). LLM routing remains available when true.
    router_use_llm: bool
    # Startup warmup: preload local scanner/reranker models in background so the
    # first user query avoids heavy cold-start latency.
    warmup_models_on_start: bool

    # ── Kafka / Redpanda streaming ingest ─────────────────
    # Set kafka_enabled=true to activate the Kafka consumer path.
    # Compatible with any Kafka-API broker: Apache Kafka, Redpanda, Confluent Cloud.
    # The FastAPI ingest endpoint (ingest_api.py) remains available alongside Kafka.
    kafka_enabled: bool
    kafka_bootstrap_servers: str   # e.g. localhost:9092 or seed.redpanda.cloud:9092
    kafka_topic: str               # topic name, e.g. search-ingest
    kafka_consumer_group: str      # consumer group id for offset tracking
    kafka_security_protocol: str   # PLAINTEXT | SASL_SSL
    kafka_sasl_mechanism: str      # PLAIN | SCRAM-SHA-256 | SCRAM-SHA-512 (for SASL_SSL)
    kafka_sasl_username: str       # leave empty for PLAINTEXT
    kafka_sasl_password: str       # leave empty for PLAINTEXT
    kafka_max_poll_records: int    # max records per poll (default 10)

    # ── LangGraph checkpointer (crash recovery) ───────────
    # Backend for persisting graph state between nodes so a mid-pipeline crash
    # can resume from the last completed node rather than starting over.
    # Options: sqlite (default) | postgres | memory | none
    checkpointer_type: str
    # Path to the SQLite database file (used when checkpointer_type=sqlite).
    checkpointer_sqlite_path: str
    # PostgreSQL connection string (used when checkpointer_type=postgres).
    # Format: postgresql://user:password@host:port/dbname
    checkpointer_postgres_url: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        # ── Meilisearch ──────────────────────────────────────
        meili_url=os.getenv("MEILI_URL", C.DEFAULT_MEILI_URL),
        meili_master_key=os.getenv("MEILI_MASTER_KEY", C.DEFAULT_MEILI_MASTER_KEY),
        meili_index_name=os.getenv("MEILI_INDEX_NAME", C.DEFAULT_MEILI_INDEX_NAME),
        meili_embedder_name=os.getenv("MEILI_EMBEDDER_NAME", C.DEFAULT_MEILI_EMBEDDER_NAME),
        # ── OpenAI ───────────────────────────────────────────
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", C.DEFAULT_OPENAI_MODEL),
        # ── Embeddings ───────────────────────────────────────
        embedding_model=os.getenv("EMBEDDING_MODEL", C.DEFAULT_EMBEDDING_MODEL),
        # Must match the chosen EMBEDDING_MODEL (e.g. 384 for default MiniLM).
        embedding_dimensions=int(
            os.getenv("EMBEDDING_DIMENSIONS", str(C.DEFAULT_EMBEDDING_DIMENSIONS)),
        ),
        # ── LangWatch ───────────────────────────────────────
        langwatch_enabled=os.getenv("LANGWATCH_ENABLED", "false").lower() == "true",
        langwatch_api_key=os.getenv("LANGWATCH_API_KEY", ""),
        langwatch_project=os.getenv("LANGWATCH_PROJECT", C.DEFAULT_LANGWATCH_PROJECT),
        # ── Pipeline tuning ──────────────────────────────────
        max_search_iterations=int(
            os.getenv("MAX_SEARCH_ITERATIONS", str(C.DEFAULT_MAX_SEARCH_ITERATIONS)),
        ),
        max_query_words=int(
            os.getenv("MAX_QUERY_WORDS", str(C.DEFAULT_MAX_QUERY_WORDS)),
        ),
        token_budget_usd=float(os.getenv("TOKEN_BUDGET_USD", str(C.DEFAULT_TOKEN_BUDGET_USD))),
        reranker_top_n=int(os.getenv("RERANKER_TOP_N", str(C.DEFAULT_RERANKER_TOP_N))),
        reranker_model=os.getenv("RERANKER_MODEL", C.DEFAULT_RERANKER_MODEL),
        confidence_threshold_degraded=float(
            os.getenv(
                "CONFIDENCE_THRESHOLD_DEGRADED",
                str(C.DEFAULT_CONFIDENCE_THRESHOLD_DEGRADED),
            ),
        ),
        near_duplicate_threshold=float(
            os.getenv("NEAR_DUPLICATE_THRESHOLD", str(C.DEFAULT_NEAR_DUPLICATE_THRESHOLD)),
        ),
        # ── Injection detection ──────────────────────────────
        injection_scan_threshold=float(
            os.getenv("INJECTION_SCAN_THRESHOLD", str(C.DEFAULT_INJECTION_SCAN_THRESHOLD)),
        ),
        # ── Freshness thresholds ─────────────────────────────
        freshness_threshold_seconds=int(
            os.getenv("FRESHNESS_THRESHOLD_SECONDS", str(C.DEFAULT_FRESHNESS_THRESHOLD_SECONDS)),
        ),
        staleness_threshold_seconds=int(
            os.getenv("STALENESS_THRESHOLD_SECONDS", str(C.DEFAULT_STALENESS_THRESHOLD_SECONDS)),
        ),
        # ── Dataset ─────────────────────────────────────────────
        dataset_file=os.getenv("DATASET_FILE", C.DEFAULT_DATASET_FILE),
        dataset_schema=os.getenv("DATASET_SCHEMA", C.DEFAULT_DATASET_SCHEMA),
        ingest_sla_seconds=int(os.getenv("INGEST_SLA_SECONDS", str(C.DEFAULT_INGEST_SLA_SECONDS))),
        # ── Latency / performance ─────────────────────────────────
        fast_mode=os.getenv("FAST_MODE", "false").lower() == "true",
        reranker_explain_async=os.getenv("RERANKER_EXPLAIN_ASYNC", "true").lower() == "true",
        reranker_explain_top_k=int(
            os.getenv("RERANKER_EXPLAIN_TOP_K", str(C.DEFAULT_RERANKER_EXPLAIN_TOP_K)),
        ),
        reranker_explain_timeout_seconds=int(
            os.getenv(
                "RERANKER_EXPLAIN_TIMEOUT_SECONDS",
                str(C.DEFAULT_RERANKER_EXPLAIN_TIMEOUT_SECONDS),
            ),
        ),
        reranker_explain_max_retries=int(
            os.getenv(
                "RERANKER_EXPLAIN_MAX_RETRIES",
                str(C.DEFAULT_RERANKER_EXPLAIN_MAX_RETRIES),
            ),
        ),
        router_use_llm=os.getenv("ROUTER_USE_LLM", "false").lower() == "true",
        warmup_models_on_start=os.getenv("WARMUP_MODELS_ON_START", "true").lower() == "true",
        # ── Kafka / Redpanda streaming ingest ─────────────────────
        kafka_enabled=os.getenv("KAFKA_ENABLED", "false").lower() == "true",
        kafka_bootstrap_servers=os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS",
            C.DEFAULT_KAFKA_BOOTSTRAP_SERVERS,
        ),
        kafka_topic=os.getenv("KAFKA_TOPIC", C.DEFAULT_KAFKA_TOPIC),
        kafka_consumer_group=os.getenv(
            "KAFKA_CONSUMER_GROUP",
            C.DEFAULT_KAFKA_CONSUMER_GROUP,
        ),
        kafka_security_protocol=os.getenv(
            "KAFKA_SECURITY_PROTOCOL",
            C.DEFAULT_KAFKA_SECURITY_PROTOCOL,
        ),
        kafka_sasl_mechanism=os.getenv("KAFKA_SASL_MECHANISM", C.DEFAULT_KAFKA_SASL_MECHANISM),
        kafka_sasl_username=os.getenv("KAFKA_SASL_USERNAME", ""),
        kafka_sasl_password=os.getenv("KAFKA_SASL_PASSWORD", ""),
        kafka_max_poll_records=int(
            os.getenv("KAFKA_MAX_POLL_RECORDS", str(C.DEFAULT_KAFKA_MAX_POLL_RECORDS)),
        ),
        # ── LangGraph checkpointer ─────────────────────────────────
        checkpointer_type=os.getenv("CHECKPOINTER_TYPE", "sqlite").lower().strip(),
        checkpointer_sqlite_path=os.getenv("CHECKPOINTER_SQLITE_PATH", "checkpoints.db"),
        checkpointer_postgres_url=os.getenv("CHECKPOINTER_POSTGRES_URL", ""),
    )
