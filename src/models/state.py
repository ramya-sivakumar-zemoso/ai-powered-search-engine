from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

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
    DEGRADED = "DEGRADED"
    ABSENT = "ABSENT"

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
    original_rank: int = 0
    new_rank: int = 0
    relevance_score: float = 0.0
    confidence: float = 0.0
    explanation: str = ""
    explanation_citation_ids: list[str] = Field(default_factory=list)
    explanation_status: ExplanationStatus = ExplanationStatus.ABSENT

class FreshnessReport(BaseModel):
    """Index freshness metadata surfaced in every response (PRD Section 4.3)."""
    index_last_updated: datetime | None = None
    staleness_flag: bool = False
    stale_result_ids: list[str] = Field(default_factory=list)
    max_staleness_seconds: float = 0.0

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
    query_hash: str = ""
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
    # evaluator output
    quality_scores: dict[str, float] = Field(default_factory=dict)
    evaluator_decision: str = ""
    retry_prescription: RetryPrescription | None = None
    # Loop prevention (PRD Section 4.5)
    iteration_count: int = 0
    search_history: list[SearchAttempt] = Field(default_factory=list)
    # Token budget tracking (PRD Section 4.5)
    token_usage: list[TokenUsage] = Field(default_factory=list)
    cumulative_token_cost: float = 0.0
    # Freshness (PRD Section 4.3)
    freshness_metadata: FreshnessReport = Field(default_factory=FreshnessReport)
    # Errors (PRD Section 4.2)
    errors: list[ExtractionError] = Field(default_factory=list)