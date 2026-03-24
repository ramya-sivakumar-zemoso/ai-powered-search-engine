"""Pydantic models for LangGraph node contracts (SearchState and related types)."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class IntentType(str, Enum):
    NAVIGATIONAL = "NAVIGATIONAL"
    INFORMATIONAL = "INFORMATIONAL"
    TRANSACTIONAL = "TRANSACTIONAL"


class RetrievalStrategy(str, Enum):
    KEYWORD = "KEYWORD"
    SEMANTIC = "SEMANTIC"
    HYBRID = "HYBRID"


class ExplanationStatus(str, Enum):
    VERIFIED = "VERIFIED"          # all cited fields confirmed in source_fields
    UNVERIFIED = "UNVERIFIED"      # cited field absent or content mismatch
    DEGRADED = "DEGRADED"          # confidence < 0.30, no explanation generated
    ABSENT = "ABSENT"              # reranker failed for this result entirely
    RERANK_DEGRADED = "RERANK_DEGRADED"  # entire LLM response rejected (confidence > 1.0)


class ErrorSeverity(str, Enum):
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EvaluatorDecision(str, Enum):
    ACCEPT = "accept"
    RETRY = "retry"
    EXHAUSTED = "exhausted"


class RetryReasonCode(str, Enum):
    LOW_SEMANTIC = "LOW_SEMANTIC"
    LOW_COVERAGE = "LOW_COVERAGE"
    LOW_FRESHNESS = "LOW_FRESHNESS"
    LOW_STABILITY = "LOW_STABILITY"


# ── Sub-models ────────────────────────────────────────────────────────────────

class IntentModel(BaseModel):
    """Output of query_understander — parsed query intent."""
    type: IntentType
    entities: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    ambiguity_score: float = Field(ge=0.0, le=1.0)
    language: str = "en"
    sanitised_query: str = ""   # injection-stripped version used downstream


class StrategyConfig(BaseModel):
    """Output of retrieval_router — search strategy with weights."""
    strategy: RetrievalStrategy
    hybrid_weights: dict[str, float] = Field(default_factory=dict)
    # e.g. {"semanticRatio": 0.55}
    router_reasoning: str = ""


class SearchResult(BaseModel):
    """One result returned by Meilisearch searcher node."""
    id: str
    title: str
    ranking_score: float = Field(alias="_rankingScore", default=0.0)
    source_fields: dict[str, Any] = Field(default_factory=dict)
    # source_fields populated ONLY from attributesToRetrieve in Meilisearch response
    # Never from index schema — this is the citation audit ground truth
    freshness_timestamp: datetime | None = None
    filter_relaxation_applied: bool = False

    model_config = {"populate_by_name": True}


class RankedResult(BaseModel):
    """One result after reranker — extends SearchResult with ranking metadata."""
    id: str
    title: str
    original_rank: int
    new_rank: int
    relevance_score: float      # cross-encoder sigmoid score
    confidence: float           # same value, surfaced separately for gates
    explanation: str = ""
    explanation_citation_ids: list[str] = Field(default_factory=list)
    explanation_status: ExplanationStatus = ExplanationStatus.ABSENT
    native_ranking_score: float = 0.0  # preserved _rankingScore for UNVERIFIED/DEGRADED


class QualityScores(BaseModel):
    """
    Output of evaluator — 5-signal weighted quality score.
    Weights must sum to 1.0 (asserted at node entry).
    """
    semantic_relevance: float = 0.0     # weight 0.30
    coverage_score: float = 0.0         # weight 0.22
    ranking_stability: float = 0.0      # weight 0.12
    freshness_signal: float = 0.0       # weight 0.18
    rerank_confidence: float = 0.0      # weight 0.18 — deferred, set after reranker
    quality_score: float = 0.0          # final weighted sum


class RetryPrescription(BaseModel):
    """
    Structured retry recommendation from evaluator → retrieval_router.
    Router reads reason_code first — overrides rule table on retry attempts only.
    """
    recommended_strategy: RetrievalStrategy
    weight_adjustments: dict[str, float] = Field(default_factory=dict)
    reason_code: RetryReasonCode


class FreshnessReport(BaseModel):
    """Assembled by searcher (initial) and finalised by reporter."""
    index_last_updated: datetime | None = None
    staleness_flag: bool = False
    stale_result_ids: list[str] = Field(default_factory=list)
    max_staleness_seconds: float = 0.0
    freshness_unknown: bool = False     # True when stats API was unavailable


class ExtractionError(BaseModel):
    """Structured error written to state.errors by any node."""
    node: str
    severity: ErrorSeverity
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    message: str
    fallback_applied: bool = False
    fallback_description: str = ""


class SearchAttempt(BaseModel):
    """
    One entry in search_history.
    Used for near-duplicate detection — embedding stored for cosine comparison.
    """
    strategy: RetrievalStrategy
    query_variant: str
    query_embedding: list[float] = Field(default_factory=list)
    quality_score: float = 0.0
    freshness_reading: datetime | None = None


class TokenUsage(BaseModel):
    """Tracks cumulative token spend across all LLM nodes."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


# ── Top-level graph state ──────────────────────────────────────────────────────

class SearchState(BaseModel):
    """
    Complete state passed between all LangGraph nodes.
    Every field must be populated by its owning node before routing.
    """

    # ── Input ────────────────────────────────────────────────────────────────
    query: str                              # raw, unmodified user query string
    session_id: str = ""                    # for LangSmith traceability

    # ── query_understander output ─────────────────────────────────────────────
    parsed_intent: IntentModel | None = None
    injection_detected: bool = False

    # ── retrieval_router output ───────────────────────────────────────────────
    retrieval_strategy: StrategyConfig | None = None

    # ── searcher output ───────────────────────────────────────────────────────
    search_results: list[SearchResult] = Field(default_factory=list)
    partial_results: bool = False           # True when fallback triggered

    # ── evaluator output ──────────────────────────────────────────────────────
    quality_scores: QualityScores = Field(default_factory=QualityScores)
    evaluator_decision: EvaluatorDecision | None = None
    retry_prescription: RetryPrescription | None = None

    # ── reranker output ───────────────────────────────────────────────────────
    reranked_results: list[RankedResult] = Field(default_factory=list)
    reranker_gap_signal: str = ""           # set when mean confidence < 0.50

    # ── Loop prevention ───────────────────────────────────────────────────────
    iteration_count: int = 0
    search_history: list[SearchAttempt] = Field(default_factory=list)
    cumulative_token_cost: float = 0.0
    token_usage_per_node: dict[str, TokenUsage] = Field(default_factory=dict)

    # ── Freshness ─────────────────────────────────────────────────────────────
    freshness_metadata: FreshnessReport = Field(default_factory=FreshnessReport)

    # ── Error audit ───────────────────────────────────────────────────────────
    errors: list[ExtractionError] = Field(default_factory=list)

    # ── Reporter output (final) ───────────────────────────────────────────────
    query_hash: str = ""
    final_response: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def query_must_not_be_empty(self) -> "SearchState":
        if not self.query.strip():
            raise ValueError("query must not be empty")
        return self