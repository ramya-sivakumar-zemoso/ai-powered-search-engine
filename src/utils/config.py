"""Environment-backed settings. Use get_settings() everywhere; avoid os.getenv in callers."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _e(key: str, default: str) -> str:
    return os.getenv(key, default)


def _b(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).lower() == "true"


def _i(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _f(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


@dataclass(frozen=True)
class Settings:
    meili_url: str
    meili_master_key: str
    meili_index_name: str
    meili_embedder_name: str
    openai_api_key: str
    openai_model: str
    embedding_model: str
    embedding_dimensions: int
    langchain_tracing_v2: bool
    langchain_api_key: str
    langchain_project: str
    max_search_iterations: int
    token_budget_usd: float
    reranker_top_n: int
    confidence_threshold_degraded: float
    near_duplicate_threshold: float
    injection_scan_threshold: float
    freshness_threshold_seconds: int
    staleness_threshold_seconds: int
    dataset_file: str
    dataset_schema: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        meili_url=os.getenv("MEILI_URL", "http://127.0.0.1:7700"),
        meili_master_key=os.getenv("MEILI_MASTER_KEY", "aSampleMasterKey"),
        meili_index_name=os.getenv("MEILI_INDEX_NAME", "movies"),
        meili_embedder_name=os.getenv("MEILI_EMBEDDER_NAME", "default"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en"),
        embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "1024")),
        langchain_tracing_v2=os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true",
        langchain_api_key=os.getenv("LANGCHAIN_API_KEY", ""),
        langchain_project=os.getenv("LANGCHAIN_PROJECT", "novamart-search"),
        max_search_iterations=int(os.getenv("MAX_SEARCH_ITERATIONS", "3")),
        token_budget_usd=float(os.getenv("TOKEN_BUDGET_USD", "0.02")),
        reranker_top_n=int(os.getenv("RERANKER_TOP_N", "10")),
        confidence_threshold_degraded=float(os.getenv("CONFIDENCE_THRESHOLD_DEGRADED", "0.30")),
        near_duplicate_threshold=float(os.getenv("NEAR_DUPLICATE_THRESHOLD", "0.92")),
        injection_scan_threshold=float(os.getenv("INJECTION_SCAN_THRESHOLD", "0.85")),
        freshness_threshold_seconds=int(os.getenv("FRESHNESS_THRESHOLD_SECONDS", "300")),
        staleness_threshold_seconds=int(os.getenv("STALENESS_THRESHOLD_SECONDS", "3600")),
        dataset_file=os.getenv("DATASET_FILE", "data/movies.json"),
        dataset_schema=os.getenv("DATASET_SCHEMA", "movies"),
    )
    