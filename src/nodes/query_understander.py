from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage

from src.models.state import (
    IntentModel,
    IntentType,
    ExtractionError,
    ErrorSeverity,
    TokenUsage,
    SearchAttempt,
    RetrievalStrategy,
)
from src.models.schema_registry import get_schema
from src.utils.config import get_settings
from src.utils.llm import get_llm, strip_markdown_fences, extract_token_usage
from src.utils.logger import get_logger, log_node_exit, log_injection_detection
from src.utils.langwatch_tracker import (
    get_langwatch_callback,
    check_budget,
    make_budget_exceeded_error,
    annotate_node_span,
)

logger = get_logger(__name__)
settings = get_settings()

# ── Load prompt template from file ────────────────────────────────────────────
# The prompt is stored as a text file so it can be version-controlled separately.
# Any change to this prompt is a version bump (Working Flow doc requirement).
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "intent_parse.txt"
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")


def _intent_system_prompt() -> str:
    """Base intent prompt plus optional domain appendix from ``DatasetSchema``."""
    schema = get_schema(settings.dataset_schema)
    extra = schema.intent_parse_appendix.strip()
    if not extra:
        return SYSTEM_PROMPT
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"## Domain-specific guidance ({schema.name})\n{extra}"
    )


# Lazy-cached injection scanner (avoids reloading ~400MB model on every call)
_injection_scanner = None


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Prompt Injection Scanner
# ══════════════════════════════════════════════════════════════════════════════

def _scan_for_injection(query: str) -> tuple[bool, float]:
    """
    Runs LLM Guard's PromptInjection scanner on the user query.

    What it does:
      - Loads a local ML model (deberta-v3-base) to detect injection patterns
      - Compares the detection score against the threshold from .env (default 0.85)
      - First run downloads the model (~400MB) — subsequent runs are fast

    Args:
        query: The raw user search string.

    Returns:
        (is_safe, risk_score)
        - is_safe: True if no injection detected (score below threshold)
        - risk_score: How confident the scanner is that this is an injection (0.0 to 1.0)
    """
    try:
        global _injection_scanner
        if _injection_scanner is None:
            from llm_guard.input_scanners import PromptInjection

            _injection_scanner = PromptInjection(
                threshold=settings.injection_scan_threshold,
            )

        _sanitized, is_valid, risk_score = _injection_scanner.scan(query)

        logger.info(
            "injection_scan_complete",
            extra={"is_safe": is_valid, "risk_score": round(risk_score, 4)},
        )

        return is_valid, risk_score

    except Exception as exc:
        # If LLM Guard itself fails (missing model, import error, etc.),
        # log a warning and allow the query through.
        # Defense in depth — scanner failure should not block legitimate searches.
        logger.warning(
            "injection_scan_failed",
            extra={"error": str(exc), "fallback": "allowing query through"},
        )
        return True, 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: GPT Intent Parser
# ══════════════════════════════════════════════════════════════════════════════

def _parse_intent_with_llm(query: str) -> tuple[dict, int, int]:
    """
    Calls GPT-4o-mini to parse the user query into structured intent.

    What it does:
      - Sends the query with the system prompt (intent_parse.txt + schema appendix)
      - GPT returns a JSON object with type, entities, filters, ambiguity_score, language
      - temperature=0 ensures the same query always produces the same output

    Args:
        query: The raw user search string.

    Returns:
        (parsed_dict, prompt_tokens, completion_tokens)
        - parsed_dict: The JSON parsed from GPT's response
        - prompt_tokens: Number of input tokens used (for cost tracking)
        - completion_tokens: Number of output tokens used (for cost tracking)
    """
    llm = get_llm()

    messages = [
        SystemMessage(content=_intent_system_prompt()),
        HumanMessage(content=query),
    ]

    response = llm.invoke(messages, config=get_langwatch_callback())

    prompt_tokens, completion_tokens = extract_token_usage(response)

    content = strip_markdown_fences(response.content)
    parsed = json.loads(content)
    return parsed, prompt_tokens, completion_tokens


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN NODE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def query_understander_node(state: dict) -> dict:
    """
    The query_understander node — entry point of the search pipeline.

    Flow:
      Step 1: Generate a unique query hash (for traceability — PRD Section 5)
      Step 2: Scan query for prompt injection (LLM Guard — PRD Section 4.7)
      Step 3: If injection → write error to state and return (hard exit)
      Step 4: If safe → call GPT-4o-mini to parse intent (PRD Section 4.1)
      Step 5: Record token usage for budget tracking (PRD Section 4.5)
      Step 6: Add sanitized query to search_history (PRD Section 4.5)
      Step 7: Log structured node exit (PRD Section 5: Observability)

    Args:
        state: The pipeline state dict (contains "query" at minimum).

    Returns:
        Dict of only the keys this node changed. List fields (errors,
        token_usage, search_history) contain only NEW items — the
        operator.add reducer in SearchStateDict merges them.
    """
    start = time.perf_counter()
    query = state.get("query", "")

    # Accumulate updates — only changed keys are returned
    updates: dict = {}
    new_errors: list[dict] = []
    new_token_usage: list[dict] = []
    new_search_history: list[dict] = []

    # ── Step 1: Generate query hash for traceability ─────────────────────────
    query_hash = hashlib.md5(query.encode()).hexdigest()[:12]
    updates["query_hash"] = query_hash

    # ── Step 2: Scan for prompt injection ────────────────────────────────────
    is_safe, risk_score = _scan_for_injection(query)

    # ── Step 3: If injection detected → hard exit to reporter ────────────────
    if not is_safe:
        log_injection_detection(
            logger, "user_query", None, "prompt_injection", risk_score,
        )

        new_errors.append(
            ExtractionError(
                node="query_understander",
                severity=ErrorSeverity.ERROR,
                message="INJECTION_DETECTED",
                fallback_applied=True,
                fallback_description=(
                    "Query blocked — prompt injection detected with risk score "
                    f"{risk_score:.4f}. No search results returned."
                ),
            ).model_dump()
        )

        duration_ms = (time.perf_counter() - start) * 1000
        log_node_exit(
            logger, "query_understander", query_hash,
            0, "BLOCKED", duration_ms, 0.0,
        )
        annotate_node_span(
            "query_understander", 0, "BLOCKED", duration_ms,
            extra={"injection_risk": round(risk_score, 4)},
        )

        updates["errors"] = new_errors
        return updates

    # ── Step 4: Parse intent with GPT-4o-mini ────────────────────────────────
    cost_usd = 0.0

    try:
        check_budget(
            state.get("cumulative_token_cost", 0.0),
            "query_understander",
            query_hash,
        )

        parsed, prompt_tokens, completion_tokens = _parse_intent_with_llm(query)

        intent = IntentModel(
            type=IntentType(parsed.get("type", "INFORMATIONAL")),
            entities=parsed.get("entities", []),
            filters=parsed.get("filters", {}),
            ambiguity_score=float(parsed.get("ambiguity_score", 0.5)),
            language=parsed.get("language", "en"),
        )
        updates["parsed_intent"] = intent.model_dump()

        # ── Step 5: Record token usage ───────────────────────────────────────
        cost_usd = (prompt_tokens * 0.000000150) + (completion_tokens * 0.000000600)

        new_token_usage.append(
            TokenUsage(
                node="query_understander",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=round(cost_usd, 8),
            ).model_dump()
        )

        updates["cumulative_token_cost"] = round(
            state.get("cumulative_token_cost", 0.0) + cost_usd, 8
        )

        # ── Step 6: Add sanitized query to search_history ────────────────────
        sanitized_variant = " ".join(intent.entities) if intent.entities else query

        new_search_history.append(
            SearchAttempt(
                strategy=RetrievalStrategy.HYBRID,
                query_variant=sanitized_variant,
            ).model_dump()
        )

    except json.JSONDecodeError as exc:
        logger.warning(
            "intent_parse_bad_json",
            extra={"error": str(exc), "fallback": "default IntentModel"},
        )
        updates["parsed_intent"] = IntentModel().model_dump()

        new_errors.append(
            ExtractionError(
                node="query_understander",
                severity=ErrorSeverity.WARNING,
                message=f"INTENT_PARSE_BAD_JSON: {str(exc)[:200]}",
                fallback_applied=True,
                fallback_description=(
                    "GPT returned invalid JSON. Using default intent "
                    "(INFORMATIONAL, no entities). Search will proceed with "
                    "basic settings."
                ),
            ).model_dump()
        )

    except ValueError:
        new_errors.append(make_budget_exceeded_error("query_understander").model_dump())

    except Exception as exc:
        logger.warning(
            "intent_parsing_failed",
            extra={"error": str(exc), "fallback": "default IntentModel"},
        )
        updates["parsed_intent"] = IntentModel().model_dump()

        new_errors.append(
            ExtractionError(
                node="query_understander",
                severity=ErrorSeverity.WARNING,
                message=f"INTENT_PARSE_FAILED: {str(exc)[:200]}",
                fallback_applied=True,
                fallback_description=(
                    "GPT call failed. Using default intent (INFORMATIONAL, "
                    "no entities). Search will proceed with basic settings."
                ),
            ).model_dump()
        )

    # ── Step 7: Log and annotate node exit ────────────────────────────────────
    duration_ms = (time.perf_counter() - start) * 1000
    strategy = state.get("retrieval_strategy", "HYBRID")

    log_node_exit(
        logger, "query_understander", query_hash,
        0, strategy, duration_ms,
        state.get("cumulative_token_cost", 0.0) + cost_usd,
    )
    annotate_node_span("query_understander", 0, strategy, duration_ms)

    # Only include list fields if they have new items (avoid no-op merges)
    if new_errors:
        updates["errors"] = new_errors
    if new_token_usage:
        updates["token_usage"] = new_token_usage
    if new_search_history:
        updates["search_history"] = new_search_history

    return updates
