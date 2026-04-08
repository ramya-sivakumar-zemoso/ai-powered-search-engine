from __future__ import annotations

import langwatch                                          # pip install langwatch==0.13.0
from langchain_core.runnables import RunnableConfig      # needed to pass callback to GPT

from src.models.state import ExtractionError, ErrorSeverity, PipelineEvent
from src.utils.config import get_settings
from src.utils.logger import get_logger

# Module-level logger — every event this file emits will appear in JSON stdout
logger = get_logger(__name__)

# Load settings once (reads .env file)
settings = get_settings()


# ── 1. SETUP ──────────────────────────────────────────────────────────────────

def setup_langwatch() -> None:
    if not settings.langwatch_enabled:
        logger.info("langwatch_disabled", extra={"reason": "LANGWATCH_ENABLED=false in .env"})
        return

    langwatch.setup(
        api_key=settings.langwatch_api_key,        # from LANGWATCH_API_KEY in .env
        base_attributes={
            "service.name": settings.langwatch_project,   # "novamart-search"
        },
    )

    logger.info(
        "langwatch_initialized",
        extra={"project": settings.langwatch_project},
    )

# ── 2. CALLBACK (for LLM nodes) ───────────────────────────────────────────────

def get_langwatch_callback() -> RunnableConfig:
    
    if not settings.langwatch_enabled:
        # Safe fallback — empty config, no LangWatch, no crash
        return RunnableConfig()

    try:
        # get_current_trace() gets the active trace started by @langwatch.trace()
        # get_langchain_callback() converts it into a LangChain-compatible callback
        callback = langwatch.get_current_trace().get_langchain_callback()
        return RunnableConfig(callbacks=[callback])

    except Exception as exc:
        # If LangWatch is unavailable, log and continue — never crash the pipeline
        logger.warning(
            "langwatch_callback_failed",
            extra={"error": str(exc), "fallback": "empty RunnableConfig"},
        )
        return RunnableConfig()


# ── 3. BUDGET GATE ────────────────────────────────────────────────────────────
# Conservative per-call USD ceilings for *projection* before an LLM invoke
# (gpt-4o-mini-style rates used post-hoc in nodes: input 0.15/M, output 0.60/M).
BUDGET_PROJECT_INTENT_PARSE_USD = 0.0008
BUDGET_PROJECT_RETRIEVAL_ROUTE_USD = 0.0008
BUDGET_PROJECT_RERANK_EXPLAIN_USD = 0.0012


def check_budget(
    cumulative_cost_usd: float,
    node_name: str,
    query_hash: str,
) -> None:
    """Abort if cumulative spend already meets or exceeds the per-query budget."""
    budget_limit = settings.token_budget_usd

    if cumulative_cost_usd >= budget_limit:
        _log_budget_breach(
            node_name, query_hash, cumulative_cost_usd, budget_limit, reason="cumulative",
        )
        raise ValueError(
            f"BUDGET_EXCEEDED: cumulative cost ${cumulative_cost_usd:.6f} "
            f"exceeds limit ${budget_limit:.4f} at node '{node_name}'"
        )


def check_budget_projected(
    cumulative_cost_usd: float,
    projected_additional_usd: float,
    node_name: str,
    query_hash: str,
) -> None:
    """Abort if cumulative spend already exceeds budget OR next call would exceed it."""
    check_budget(cumulative_cost_usd, node_name, query_hash)
    budget_limit = settings.token_budget_usd
    projected_total = cumulative_cost_usd + projected_additional_usd
    if projected_total > budget_limit:
        _log_budget_breach(
            node_name,
            query_hash,
            cumulative_cost_usd,
            budget_limit,
            reason="projected",
            projected_additional_usd=projected_additional_usd,
            projected_total_usd=projected_total,
        )
        raise ValueError(
            f"BUDGET_EXCEEDED: projected total ${projected_total:.6f} "
            f"(cumulative ${cumulative_cost_usd:.6f} + est. call ${projected_additional_usd:.6f}) "
            f"exceeds limit ${budget_limit:.4f} at node '{node_name}'"
        )


def _log_budget_breach(
    node_name: str,
    query_hash: str,
    cumulative_cost_usd: float,
    budget_limit: float,
    *,
    reason: str,
    projected_additional_usd: float | None = None,
    projected_total_usd: float | None = None,
) -> None:
    extra: dict = {
        "node": node_name,
        "query_hash": query_hash,
        "cumulative_cost_usd": round(cumulative_cost_usd, 6),
        "budget_limit_usd": budget_limit,
        "reason": reason,
    }
    if projected_additional_usd is not None:
        extra["projected_additional_usd"] = round(projected_additional_usd, 8)
    if projected_total_usd is not None:
        extra["projected_total_usd"] = round(projected_total_usd, 6)
    logger.warning("budget_exceeded", extra=extra)


def make_budget_exceeded_error(node_name: str) -> ExtractionError:
    return ExtractionError(
        node=node_name,
        severity=ErrorSeverity.ERROR,
        message=PipelineEvent.BUDGET_EXCEEDED.value,
        fallback_applied=True,
        fallback_description=(
            "Token budget exhausted or next LLM call would exceed budget. "
            "Skipped or fell back per node policy."
        ),
    )
    
    
# ── 4. NODE SPAN ANNOTATION ───────────────────────────────────────────────────
 
def annotate_node_span(
    node_name: str,
    result_count: int,
    strategy_used: str,
    duration_ms: float,
    extra: dict | None = None,
) -> None:

    if not settings.langwatch_enabled:
        # LangWatch is off — skip silently, do not crash
        return
 
    try:
        # Get the active trace that was opened by @langwatch.trace()
        # and attach our node metadata to it as custom attributes
        trace = langwatch.get_current_trace()
 
        # Build the metadata dict — start with required fields
        metadata: dict = {
            "node_name": node_name,
            "result_count": result_count,
            "strategy_used": strategy_used,
            "duration_ms": round(duration_ms, 2),
        }
 
        # Merge any optional extra fields the caller passed in
        if extra:
            metadata.update(extra)
 
        # update_metadata() attaches these key-value pairs to the current trace.
        # They are searchable and filterable on the LangWatch dashboard.
        trace.update(metadata=metadata)
 
    except Exception as exc:
        # If LangWatch is unavailable, log and continue — never crash the pipeline
        logger.warning(
            "langwatch_annotation_failed",
            extra={"node": node_name, "error": str(exc)},
        )