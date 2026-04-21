from __future__ import annotations

import operator
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field

# ── Enums ─────────────────────────────────────────────────────────────────────

class ErrorSeverity(str, Enum):
    """How serious an error is. WARNING = degraded but usable. ERROR = node failed."""
    WARNING = "WARNING"
    ERROR = "ERROR"

class IntentType(str, Enum):
    """The type of user search intent (PRD Section 4.1 — query_understander output)."""
    NAVIGATIONAL = "NAVIGATIONAL"
    INFORMATIONAL = "INFORMATIONAL"
    TRANSACTIONAL = "TRANSACTIONAL"

class RetrievalStrategy(str, Enum):
    """Which search mode to use (PRD Section 4.2 — retrieval_strategy)."""
    KEYWORD = "KEYWORD"
    SEMANTIC = "SEMANTIC"
    HYBRID = "HYBRID"

class ExplanationStatus(str, Enum):
    """Status of a reranker explanation's citation audit (PRD Section 4.6)."""
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    # Explanation referenced a document field/snippet that is missing, redacted, or not in the hit.
    EXPLANATION_UNVERIFIED = "EXPLANATION_UNVERIFIED"
    DEGRADED = "DEGRADED"
    ABSENT = "ABSENT"


class PipelineEvent(str, Enum):
    """Canonical contract events surfaced to downstream consumers."""
    PARTIAL_RESULTS = "PARTIAL_RESULTS"
    RERANK_DEGRADED = "RERANK_DEGRADED"
    ITERATION_LIMIT = "ITERATION_LIMIT"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    NEAR_DUPLICATE_VARIANT = "NEAR_DUPLICATE_VARIANT"
    QUERY_WORD_LIMIT = "QUERY_WORD_LIMIT"

# ── Small models  ────────────────────────────────────────────
class ExtractionError(BaseModel):
    """A structured error recorded by any node that encounters a problem."""
    node: str
    severity: ErrorSeverity
    message: str
    fallback_applied: bool = False
    fallback_description: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class TokenUsage(BaseModel):
    """Token usage from a single LLM call — stored per-node for cost tracking."""
    node: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

class IntentModel(BaseModel):
    """Parsed intent output from query_understander (PRD Section 4.2)."""
    type: IntentType = IntentType.INFORMATIONAL
    entities: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    ambiguity_score: float = 0.0
    language: str = "en"

class SearchResult(BaseModel):
    """A single search hit from Meilisearch (PRD Section 4.2)."""
    id: str
    title: str = ""
    score: float = 0.0
    source_fields: dict[str, Any] = Field(default_factory=dict)
    freshness_timestamp: datetime | None = None

class RankedResult(BaseModel):
    """A re-ranked result with confidence and explanation (PRD Section 4.2)."""
    id: str
    title: str = ""
    original_rank: int = 0
    new_rank: int = 0
    relevance_score: float = 0.0
    confidence: float = 0.0
    # Meilisearch `_rankingScore` for this hit (unchanged by cross-encoder). Surfaced when
    # explanations are stripped or EXPLANATION_UNVERIFIED so clients retain ranking signal.
    meilisearch_ranking_score: float = 0.0
    explanation: str = ""
    explanation_citation_ids: list[str] = Field(default_factory=list)
    explanation_status: ExplanationStatus = ExplanationStatus.ABSENT

class FreshnessReport(BaseModel):
    """Freshness metadata for every response (PRD Section 4.3).

    ``index_stats_updated_at`` comes from Meilisearch index metadata when the API
    succeeds. ``index_last_updated`` remains the oldest hit timestamp among results
    (document-level proxy). ``freshness_unknown`` is True when index metadata could
    not be fetched (stats/index API failure).
    """
    index_last_updated: datetime | None = None
    index_stats_updated_at: datetime | None = None
    staleness_flag: bool = False
    stale_result_ids: list[dict] = Field(default_factory=list)
    max_staleness_seconds: float = 0.0
    index_lag: float = 0.0
    freshness_unknown: bool = False

class RetryPrescription(BaseModel):
    """Evaluator's recommendation for what to change on a retry attempt."""
    reason_code: str = ""
    suggested_strategy: RetrievalStrategy | None = None
    suggested_query_variant: str = ""
    explanation: str = ""

class SearchAttempt(BaseModel):
    """One entry in search_history — prevents near-duplicate searches (PRD 4.5)."""
    strategy: RetrievalStrategy
    query_variant: str
    quality_score: float = 0.0
    result_count: int = 0

# ── Top-level graph state ─────────────────────────────────────────────────────
class SearchState(BaseModel):
    """
    The master state object passed between all LangGraph nodes.
    Every field here is readable/writable by any node in the pipeline.
    """
    # Original user query
    query: str = ""
    # Heuristic-sanitised query (instruction-like fragments stripped) for LLM + retrieval fallback
    sanitized_query: str = ""
    query_hash: str = ""
    # Traceability (PRD Section 5 — LangWatch / observability)
    session_id: str = ""
    # Optional Meilisearch index UID (Streamlit / API override; default from settings).
    meili_index_name: str = ""
    # Searcher: mid-confidence neighbors vs query — reporter shows a gentle warning unless
    # downstream evaluator signals clearly contradict that.
    retrieval_soft_match: bool = False
    # query_understander output
    parsed_intent: IntentModel = Field(default_factory=IntentModel)
    # retrieval_router output
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    hybrid_weights: dict[str, float] = Field(default_factory=lambda: {"semanticRatio": 0.60})
    router_reasoning: str = ""
    # searcher output
    search_results: list[SearchResult] = Field(default_factory=list)
    filter_relaxation_applied: bool = False
    # reranker output
    reranked_results: list[RankedResult] = Field(default_factory=list)
    explanations_pending: bool = False
    explanations_applied: bool = False
    explanation_job_id: str = ""
    explanation_job_status: str = ""
    explanation_top_k: int = 0
    explanations_async: bool = False
    # evaluator output
    quality_scores: dict[str, Any] = Field(default_factory=dict)
    evaluator_decision: str = ""
    retry_prescription: RetryPrescription | None = None
    partial_results: bool = False
    rerank_degraded: bool = False
    # Loop prevention (PRD Section 4.5) — evaluator pass index (1 + retry loops completed).
    iteration_count: int = 0
    search_history: list[SearchAttempt] = Field(default_factory=list)
    # Token budget tracking (PRD Section 4.5)
    token_usage: list[TokenUsage] = Field(default_factory=list)
    cumulative_token_cost: float = 0.0
    # Freshness (PRD Section 4.3)
    freshness_metadata: FreshnessReport = Field(default_factory=FreshnessReport)
    # Errors (PRD Section 4.2)
    errors: list[ExtractionError] = Field(default_factory=list)


# ── TypedDict state for LangGraph ─────────────────────────────────────────────
# LangGraph 1.x requires TypedDict (not Pydantic BaseModel) as the state
# schema for StateGraph. Fields annotated with ``operator.add`` are
# *accumulated* (LangGraph merges new items via ``+``); all others are
# overwritten on each node return.  Nodes MUST return only the *new* items
# for accumulated list fields — not the full list.

class SearchStateDict(TypedDict, total=False):
    """Typed graph state consumed by ``StateGraph(SearchStateDict)``."""
    # ── identity ──
    query: str
    sanitized_query: str
    query_hash: str
    session_id: str
    meili_index_name: str
    retrieval_soft_match: bool
    # ── query_understander output ──
    parsed_intent: dict
    # ── retrieval_router output ──
    retrieval_strategy: str
    hybrid_weights: dict[str, float]
    router_reasoning: str
    # ── searcher output ──
    search_results: list[dict]
    filter_relaxation_applied: bool
    # ── reranker output ──
    reranked_results: list[dict]
    explanations_pending: bool
    explanations_applied: bool
    explanation_job_id: str
    explanation_job_status: str
    explanation_top_k: int
    explanations_async: bool
    # ── evaluator output ──
    quality_scores: dict[str, Any]
    evaluator_decision: str
    retry_prescription: dict | None
    iteration_count: int
    partial_results: bool
    rerank_degraded: bool
    # ── accumulated lists (operator.add reducer) ──
    search_history: Annotated[list, operator.add]
    token_usage: Annotated[list, operator.add]
    errors: Annotated[list, operator.add]
    # ── budget tracking ──
    cumulative_token_cost: float
    # ── freshness ──
    freshness_metadata: dict
    # ── reporter output ──
    final_response: dict