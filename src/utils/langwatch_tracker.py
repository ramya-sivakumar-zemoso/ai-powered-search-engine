from __future__ import annotations

import langwatch                                          # pip install langwatch==0.13.0
from langchain_core.runnables import RunnableConfig      # needed to pass callback to GPT

from src.models.state import ExtractionError, ErrorSeverity
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

def check_budget(
    cumulative_cost_usd: float,
    node_name: str,
    query_hash: str,
) -> None:
    budget_limit = settings.token_budget_usd   # default: 0.02 (from .env)

    if cumulative_cost_usd >= budget_limit:
        # Log to JSON stdout — visible in terminal and any log aggregator
        logger.warning(
            "budget_exceeded",
            extra={
                "node": node_name,
                "query_hash": query_hash,
                "cumulative_cost_usd": round(cumulative_cost_usd, 6),
                "budget_limit_usd": budget_limit,
            },
        )
        # Raise so the calling node catches it and writes the error to state
        raise ValueError(
            f"BUDGET_EXCEEDED: cumulative cost ${cumulative_cost_usd:.6f} "
            f"exceeds limit ${budget_limit:.4f} at node '{node_name}'"
        )


def make_budget_exceeded_error(node_name: str) -> ExtractionError:
    
    return ExtractionError(
        node=node_name,
        severity=ErrorSeverity.ERROR,
        message="BUDGET_EXCEEDED",
        fallback_applied=True,
        fallback_description=(
            "Token budget exhausted. Fell back to native Meilisearch ranking — "
            "no LLM reranking applied."
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