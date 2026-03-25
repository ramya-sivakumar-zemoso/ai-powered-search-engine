from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from langchain_openai import ChatOpenAI
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
from src.utils.config import get_settings
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
        from llm_guard.input_scanners import PromptInjection

        # Create scanner with threshold from .env (INJECTION_SCAN_THRESHOLD=0.85)
        scanner = PromptInjection(threshold=settings.injection_scan_threshold)

        # scan() returns: (sanitized_output, is_valid, risk_score)
        # - sanitized_output: cleaned version of query (not used here)
        # - is_valid: True if query is safe (below threshold)
        # - risk_score: confidence that this is an injection attempt
        _sanitized, is_valid, risk_score = scanner.scan(query)

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
      - Sends the query with a fixed system prompt (from intent_parse.txt)
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
    # Create the LLM client with settings from .env
    llm = ChatOpenAI(
        model=settings.openai_model,       # "gpt-4o-mini-2024-07-18" from .env
        temperature=0,                      # Deterministic output (PRD Section 5)
        api_key=settings.openai_api_key,   # From OPENAI_API_KEY in .env
    )

    # Build the message list — system prompt defines the rules, human message is the query
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    # Invoke GPT with LangWatch callback so the call appears on the dashboard
    response = llm.invoke(messages, config=get_langwatch_callback())

    # Extract token counts from GPT's response metadata
    usage = response.response_metadata.get("token_usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    # Parse the JSON from GPT's response text
    content = response.content.strip()

    # GPT sometimes wraps JSON in markdown code fences — remove them
    if content.startswith("```"):
        # Remove opening fence (e.g. "```json\n")
        content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        # Remove closing fence
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

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
        Updated state dict with parsed_intent, query_hash, token_usage, etc.
    """
    start = time.perf_counter()
    query = state.get("query", "")

    # ── Step 1: Generate query hash for traceability ─────────────────────────
    # PRD Section 5: "every search traceable by session_id + query_hash"
    query_hash = hashlib.md5(query.encode()).hexdigest()[:12]
    state["query_hash"] = query_hash

    # ── Step 2: Scan for prompt injection ────────────────────────────────────
    # PRD Section 4.7: "detect and strip instruction-like patterns from user
    # queries before LLM processing"
    is_safe, risk_score = _scan_for_injection(query)

    # ── Step 3: If injection detected → hard exit to reporter ────────────────
    if not is_safe:
        # Log the detection (PRD Section 4.7: "log any query that matches
        # injection pattern signatures")
        log_injection_detection(
            logger, "user_query", None, "prompt_injection", risk_score,
        )

        # Write structured error to state (PRD Section 4.1: "errors" field)
        errors = state.get("errors", [])
        error = ExtractionError(
            node="query_understander",
            severity=ErrorSeverity.ERROR,
            message="INJECTION_DETECTED",
            fallback_applied=True,
            fallback_description=(
                "Query blocked — prompt injection detected with risk score "
                f"{risk_score:.4f}. No search results returned."
            ),
        )
        errors.append(error.model_dump())
        state["errors"] = errors

        # Log and annotate the node exit
        duration_ms = (time.perf_counter() - start) * 1000
        log_node_exit(
            logger, "query_understander", query_hash,
            0, "BLOCKED", duration_ms, 0.0,
        )
        annotate_node_span(
            "query_understander", 0, "BLOCKED", duration_ms,
            extra={"injection_risk": round(risk_score, 4)},
        )

        return state

    # ── Step 4: Parse intent with GPT-4o-mini ────────────────────────────────
    try:
        # Check token budget BEFORE making the LLM call (PRD Section 4.5)
        check_budget(
            state.get("cumulative_token_cost", 0.0),
            "query_understander",
            query_hash,
        )

        # Call GPT to parse the query
        parsed, prompt_tokens, completion_tokens = _parse_intent_with_llm(query)

        # Build IntentModel from GPT's response (validates types via Pydantic)
        intent = IntentModel(
            type=IntentType(parsed.get("type", "INFORMATIONAL")),
            entities=parsed.get("entities", []),
            filters=parsed.get("filters", {}),
            ambiguity_score=float(parsed.get("ambiguity_score", 0.5)),
            language=parsed.get("language", "en"),
        )
        state["parsed_intent"] = intent.model_dump()

        # ── Step 5: Record token usage ───────────────────────────────────────
        # Cost formula from Working Flow doc:
        # cost = (prompt_tokens × $0.000000150) + (completion_tokens × $0.000000600)
        cost_usd = (prompt_tokens * 0.000000150) + (completion_tokens * 0.000000600)

        token_record = TokenUsage(
            node="query_understander",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=round(cost_usd, 8),
        )

        token_list = state.get("token_usage", [])
        token_list.append(token_record.model_dump())
        state["token_usage"] = token_list

        # Update the running total cost (PRD Section 4.5: cumulative tracking)
        state["cumulative_token_cost"] = round(
            state.get("cumulative_token_cost", 0.0) + cost_usd, 8
        )

        # ── Step 6: Add sanitized query to search_history ────────────────────
        # PRD Section 4.5: "search_history in state must prevent semantic
        # near-duplicates"
        # Working Flow: "search_history[0] — sanitised query variant (never raw query)"
        sanitized_variant = " ".join(intent.entities) if intent.entities else query

        history = state.get("search_history", [])
        history.append(
            SearchAttempt(
                strategy=RetrievalStrategy.HYBRID,
                query_variant=sanitized_variant,
            ).model_dump()
        )
        state["search_history"] = history

    except ValueError:
        # Budget exceeded — check_budget() raised ValueError
        errors = state.get("errors", [])
        errors.append(make_budget_exceeded_error("query_understander").model_dump())
        state["errors"] = errors

    except json.JSONDecodeError as exc:
        # GPT returned invalid JSON — use safe defaults
        logger.warning(
            "intent_parse_bad_json",
            extra={"error": str(exc), "fallback": "default IntentModel"},
        )
        state["parsed_intent"] = IntentModel().model_dump()

        errors = state.get("errors", [])
        errors.append(
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
        state["errors"] = errors

    except Exception as exc:
        # Any other GPT failure — use safe defaults, do not crash
        logger.warning(
            "intent_parsing_failed",
            extra={"error": str(exc), "fallback": "default IntentModel"},
        )
        state["parsed_intent"] = IntentModel().model_dump()

        errors = state.get("errors", [])
        errors.append(
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
        state["errors"] = errors

    # ── Step 7: Log and annotate node exit ────────────────────────────────────
    duration_ms = (time.perf_counter() - start) * 1000
    strategy = state.get("retrieval_strategy", "HYBRID")

    log_node_exit(
        logger, "query_understander", query_hash,
        0, strategy, duration_ms,
        state.get("cumulative_token_cost", 0.0),
    )
    annotate_node_span("query_understander", 0, strategy, duration_ms)

    return state
