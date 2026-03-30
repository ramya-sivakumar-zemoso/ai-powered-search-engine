"""Environment-backed settings. Use get_settings() everywhere; avoid os.getenv in callers."""
from __future__ import annotations

import os
from dotenv import load_dotenv
from dataclasses import dataclass
from functools import lru_cache

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

    # ── Embeddings ───────────────────────────────────────
    embedding_model: str
    embedding_dimensions: int

    # ── LangWatch ───────────────────────────────────────
    langwatch_enabled: bool
    langwatch_api_key: str
    langwatch_project: str

    # ── Pipeline tuning ──────────────────────────────────
    max_search_iterations: int
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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        # ── Meilisearch ──────────────────────────────────────
        meili_url=os.getenv("MEILI_URL", "http://127.0.0.1:7700"),
        meili_master_key=os.getenv("MEILI_MASTER_KEY", "aSampleMasterKey"),
        meili_index_name=os.getenv("MEILI_INDEX_NAME", "movies"),
        meili_embedder_name=os.getenv("MEILI_EMBEDDER_NAME", "default"),
        # ── OpenAI ───────────────────────────────────────────
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        # ── Embeddings ───────────────────────────────────────
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en"),
        embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "1024")),
        # ── LangWatch ───────────────────────────────────────
        langwatch_enabled=os.getenv("LANGWATCH_ENABLED", "false").lower() == "true",
        langwatch_api_key=os.getenv("LANGWATCH_API_KEY", ""),
        langwatch_project=os.getenv("LANGWATCH_PROJECT", "novamart-search"),
        # ── Pipeline tuning ──────────────────────────────────
        max_search_iterations=int(os.getenv("MAX_SEARCH_ITERATIONS", "3")),
        token_budget_usd=float(os.getenv("TOKEN_BUDGET_USD", "0.02")),
        reranker_top_n=int(os.getenv("RERANKER_TOP_N", "10")),
        reranker_model=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        confidence_threshold_degraded=float(os.getenv("CONFIDENCE_THRESHOLD_DEGRADED", "0.30")),
        near_duplicate_threshold=float(os.getenv("NEAR_DUPLICATE_THRESHOLD", "0.92")),
        # ── Injection detection ──────────────────────────────
        injection_scan_threshold=float(os.getenv("INJECTION_SCAN_THRESHOLD", "0.85")),
        # ── Freshness thresholds ─────────────────────────────
        freshness_threshold_seconds=int(os.getenv("FRESHNESS_THRESHOLD_SECONDS", "300")),
        staleness_threshold_seconds=int(os.getenv("STALENESS_THRESHOLD_SECONDS", "3600")),
        # ── Dataset ─────────────────────────────────────────────
        dataset_file=os.getenv("DATASET_FILE", "data/movies.json"),
        dataset_schema=os.getenv("DATASET_SCHEMA", "movies"),
    )
    