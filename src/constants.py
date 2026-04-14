"""Non-secret defaults and tunables. Env vars in ``get_settings()`` should mirror these where applicable."""

from __future__ import annotations

# ── Meilisearch (defaults only; URLs/keys still from env in production) ─────
DEFAULT_MEILI_URL = "http://127.0.0.1:7700"
DEFAULT_MEILI_MASTER_KEY = "aSampleMasterKey"
DEFAULT_MEILI_INDEX_NAME = "movies"
DEFAULT_MEILI_EMBEDDER_NAME = "default"

# Search HTTP behaviour
MEILI_SEARCH_MAX_ATTEMPTS = 3
MEILI_SEARCH_RETRY_BACKOFF_BASE_S = 0.1

# ── OpenAI ───────────────────────────────────────────────────────────────────
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# ── Embeddings / reranker (HF ids) ──────────────────────────────────────────
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_DIMENSIONS = 384
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# ── LangWatch ────────────────────────────────────────────────────────────────
DEFAULT_LANGWATCH_PROJECT = "novamart-search"

# ── Pipeline tuning ──────────────────────────────────────────────────────────
DEFAULT_MAX_SEARCH_ITERATIONS = 3
DEFAULT_TOKEN_BUDGET_USD = 0.02
DEFAULT_RERANKER_TOP_N = 10
DEFAULT_CONFIDENCE_THRESHOLD_DEGRADED = 0.30
DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.92
DEFAULT_INJECTION_SCAN_THRESHOLD = 0.85
DEFAULT_FRESHNESS_THRESHOLD_SECONDS = 300
DEFAULT_STALENESS_THRESHOLD_SECONDS = 3600
DEFAULT_INGEST_SLA_SECONDS = 300

# ── Dataset ─────────────────────────────────────────────────────────────────
DEFAULT_DATASET_FILE = "data/movies.json"
DEFAULT_DATASET_SCHEMA = "movies"

# ── Reranker explanation defaults ────────────────────────────────────────────
DEFAULT_RERANKER_EXPLAIN_TOP_K = 5
DEFAULT_RERANKER_EXPLAIN_TIMEOUT_SECONDS = 10
DEFAULT_RERANKER_EXPLAIN_MAX_RETRIES = 0

# ── Kafka ────────────────────────────────────────────────────────────────────
DEFAULT_KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
DEFAULT_KAFKA_TOPIC = "search-ingest"
DEFAULT_KAFKA_CONSUMER_GROUP = "search-engine-consumer"
DEFAULT_KAFKA_SECURITY_PROTOCOL = "PLAINTEXT"
DEFAULT_KAFKA_SASL_MECHANISM = "PLAIN"
DEFAULT_KAFKA_MAX_POLL_RECORDS = 10

# ── Search pipeline (searcher node) ──────────────────────────────────────────
SEARCH_HITS_LIMIT = 20
KEYWORD_OVERLAP_MIN_RATIO = 0.5
KEYWORD_STEM_LEN = 5
KEYWORD_OVERLAP_TOP_N = 5

# Base English stop words for keyword-overlap heuristic (extended per schema).
BASE_QUERY_STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "in", "of", "to", "for", "with", "is",
    "on", "at", "by", "from", "this", "that", "it", "as", "are", "was",
    "be", "has", "had", "not", "but", "all", "can", "her", "his", "one",
    "our", "out", "you", "its", "my", "we", "do", "no", "so", "up", "if",
    "me", "what", "about", "which", "when", "how", "who", "where", "why",
})
