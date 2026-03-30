from __future__ import annotations

import json
import time
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage

from src.models.state import (
    RetrievalStrategy,
    TokenUsage,
    ExtractionError,
    ErrorSeverity,
)
from src.utils.config import get_settings
from src.utils.llm import get_llm, strip_markdown_fences, extract_token_usage
from src.utils.logger import get_logger, log_node_exit
from src.utils.langwatch_tracker import (
    get_langwatch_callback,
    check_budget,
    make_budget_exceeded_error,
    annotate_node_span,
)

logger = get_logger(__name__)
settings = get_settings()

# Load the routing prompt template from file
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "retrieval_routing.txt"
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
#  HEURISTIC FALLBACK — same 6 rules implemented in Python
# ══════════════════════════════════════════════════════════════════════════════

def _heuristic_route(
    intent_type: str,
    ambiguity_score: float,
    entities: list,
) -> tuple[str, float, int, str]:
    """
    Applies the 6 routing rules from the Working Flow doc in pure Python.
    Used as a fallback when GPT is unavailable or over budget.

    Returns:
        (strategy, semanticRatio, rule_number, reasoning)
    """
    # Rule 6 is checked first because it overrides intent-based rules
    # when there are no entities to search for
    if not entities:
        return (
            "SEMANTIC", 0.75, 6,
            "Rule 6: entities list is empty — using semantic search to find "
            "results by meaning rather than keywords.",
        )

    if intent_type == "NAVIGATIONAL":
        return (
            "KEYWORD", 0.10, 1,
            "Rule 1: navigational intent — user wants a specific known item, "
            "keyword search is most direct.",
        )

    if intent_type == "TRANSACTIONAL":
        if ambiguity_score < 0.40:
            return (
                "HYBRID", 0.30, 2,
                "Rule 2: transactional intent with low ambiguity — hybrid "
                "search with keyword emphasis for specific purchase intent.",
            )
        else:
            return (
                "HYBRID", 0.55, 3,
                "Rule 3: transactional intent with higher ambiguity — hybrid "
                "search with balanced weights to capture broader options.",
            )

    # INFORMATIONAL (default)
    if ambiguity_score < 0.55:
        return (
            "HYBRID", 0.65, 4,
            "Rule 4: informational intent with moderate clarity — hybrid "
            "search leaning toward semantic to capture related concepts.",
        )
    else:
        return (
            "SEMANTIC", 0.80, 5,
            "Rule 5: informational intent with high ambiguity — semantic "
            "search to find results by meaning when keywords are vague.",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  LLM ROUTING — GPT applies the same rules with reasoning
# ══════════════════════════════════════════════════════════════════════════════

def _route_with_llm(
    parsed_intent: dict,
    retry_prescription: dict | None,
    search_history: list,
) -> tuple[dict, int, int]:
    """
    Calls GPT-4o-mini to apply the routing rules and return strategy + reasoning.

    Args:
        parsed_intent: The IntentModel dict from query_understander.
        retry_prescription: Evaluator's retry recommendation (None on first attempt).
        search_history: List of previous SearchAttempt dicts.

    Returns:
        (parsed_response, prompt_tokens, completion_tokens)
    """
    llm = get_llm()

    user_content = json.dumps({
        "intent_type": parsed_intent.get("type", "INFORMATIONAL"),
        "ambiguity_score": parsed_intent.get("ambiguity_score", 0.5),
        "entities": parsed_intent.get("entities", []),
        "filters": parsed_intent.get("filters", {}),
        "language": parsed_intent.get("language", "en"),
    }, indent=2)

    if retry_prescription:
        user_content += "\n\n## Retry Context\n"
        user_content += json.dumps({
            "retry_prescription": retry_prescription,
            "previous_strategies": [
                attempt.get("strategy", "") for attempt in search_history
            ],
        }, indent=2)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    response = llm.invoke(messages, config=get_langwatch_callback())

    prompt_tokens, completion_tokens = extract_token_usage(response)

    content = strip_markdown_fences(response.content)
    parsed = json.loads(content)
    return parsed, prompt_tokens, completion_tokens


# ══════════════════════════════════════════════════════════════════════════════
#  RETRY OVERRIDE — honour evaluator's prescription
# ══════════════════════════════════════════════════════════════════════════════

def _apply_retry_override(
    strategy: str,
    semantic_ratio: float,
    retry_prescription: dict | None,
    search_history: list,
) -> tuple[str, float, str]:
    """
    If the evaluator provided a retry_prescription with a suggested_strategy,
    override the rule-based decision (unless it's a near-duplicate situation).

    Returns:
        (final_strategy, final_ratio, override_note)
    """
    if not retry_prescription:
        return strategy, semantic_ratio, ""

    suggested = retry_prescription.get("suggested_strategy")
    reason = retry_prescription.get("reason_code", "")

    if not suggested:
        return strategy, semantic_ratio, ""

    # For near-duplicate, pick a strategy NOT already tried
    if reason == "NEAR_DUPLICATE":
        tried = {attempt.get("strategy", "") for attempt in search_history}
        all_strategies = {"KEYWORD", "SEMANTIC", "HYBRID"}
        untried = all_strategies - tried

        if untried:
            new_strategy = untried.pop()
            ratio_map = {"KEYWORD": 0.10, "SEMANTIC": 0.80, "HYBRID": 0.50}
            return (
                new_strategy, ratio_map.get(new_strategy, 0.50),
                f"Near-duplicate override: switched to {new_strategy} "
                f"(already tried: {tried}).",
            )

    # Otherwise, honour the evaluator's suggestion
    if suggested != strategy:
        ratio_map = {"KEYWORD": 0.10, "SEMANTIC": 0.80, "HYBRID": 0.50}
        return (
            suggested, ratio_map.get(suggested, 0.50),
            f"Retry override: evaluator suggested {suggested} "
            f"(was {strategy}).",
        )

    return strategy, semantic_ratio, ""


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN NODE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def retrieval_router_node(state: dict) -> dict:
    """
    The retrieval_router node — decides search strategy based on parsed intent.

    Flow:
      Step 1: Read parsed_intent and check for retry context
      Step 2: Check token budget before LLM call
      Step 3: Try LLM routing (GPT applies rule table)
      Step 4: If LLM fails → fall back to Python heuristic (same rules)
      Step 5: Apply retry override if evaluator prescribed a different strategy
      Step 6: Write strategy + weights to state
      Step 7: Record token usage
      Step 8: Log and annotate
    """
    start = time.perf_counter()
    query_hash = state.get("query_hash", "")

    # ── Step 1: Read inputs from state ───────────────────────────────────────
    parsed_intent = state.get("parsed_intent", {})
    intent_type = parsed_intent.get("type", "INFORMATIONAL")
    ambiguity_score = parsed_intent.get("ambiguity_score", 0.5)
    entities = parsed_intent.get("entities", [])

    retry_prescription = state.get("retry_prescription")
    search_history = state.get("search_history", [])
    is_retry = retry_prescription is not None

    # ── Step 2–4: Route with LLM or heuristic fallback ───────────────────────
    strategy = "HYBRID"
    semantic_ratio = 0.50
    rule_applied = 0
    reasoning = ""
    used_llm = False

    try:
        # Check budget before LLM call
        check_budget(
            state.get("cumulative_token_cost", 0.0),
            "retrieval_router", query_hash,
        )

        # Try LLM routing
        parsed, prompt_tokens, completion_tokens = _route_with_llm(
            parsed_intent, retry_prescription, search_history,
        )

        strategy = parsed.get("strategy", "HYBRID")
        semantic_ratio = float(parsed.get("semanticRatio", 0.60))
        rule_applied = int(parsed.get("rule_applied", 0))
        reasoning = parsed.get("reasoning", "")
        used_llm = True

        # Record token usage
        cost_usd = (prompt_tokens * 0.000000150) + (completion_tokens * 0.000000600)

        token_list = state.get("token_usage", [])
        token_list.append(
            TokenUsage(
                node="retrieval_router",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=round(cost_usd, 8),
            ).model_dump()
        )
        state["token_usage"] = token_list
        state["cumulative_token_cost"] = round(
            state.get("cumulative_token_cost", 0.0) + cost_usd, 8
        )

    except ValueError:
        # Budget exceeded — fall back to heuristic (free, no LLM call)
        errors = state.get("errors", [])
        errors.append(make_budget_exceeded_error("retrieval_router").model_dump())
        state["errors"] = errors

        strategy, semantic_ratio, rule_applied, reasoning = _heuristic_route(
            intent_type, ambiguity_score, entities,
        )
        reasoning += " (heuristic fallback — budget exceeded)"

    except Exception as exc:
        # LLM failed — fall back to heuristic
        logger.warning(
            "routing_llm_failed",
            extra={"error": str(exc), "fallback": "heuristic rules"},
        )

        strategy, semantic_ratio, rule_applied, reasoning = _heuristic_route(
            intent_type, ambiguity_score, entities,
        )
        reasoning += " (heuristic fallback — LLM unavailable)"

        errors = state.get("errors", [])
        errors.append(
            ExtractionError(
                node="retrieval_router",
                severity=ErrorSeverity.WARNING,
                message=f"ROUTING_LLM_FAILED: {str(exc)[:200]}",
                fallback_applied=True,
                fallback_description="Used heuristic rule table instead of LLM.",
            ).model_dump()
        )
        state["errors"] = errors

    # ── Step 5: Apply retry override if applicable ───────────────────────────
    if is_retry:
        strategy, semantic_ratio, override_note = _apply_retry_override(
            strategy, semantic_ratio, retry_prescription, search_history,
        )
        if override_note:
            reasoning += f" | {override_note}"

    # ── Step 6: Write strategy + weights to state ────────────────────────────
    state["retrieval_strategy"] = strategy
    state["hybrid_weights"] = {"semanticRatio": round(semantic_ratio, 2)}
    state["router_reasoning"] = reasoning

    # Clear retry_prescription after it's been consumed
    if is_retry:
        state["retry_prescription"] = None

    # ── Step 7: Log and annotate ─────────────────────────────────────────────
    duration_ms = (time.perf_counter() - start) * 1000

    log_node_exit(
        logger, "retrieval_router", query_hash,
        0, strategy, duration_ms,
        state.get("cumulative_token_cost", 0.0),
        extra={
            "rule_applied": rule_applied,
            "semantic_ratio": semantic_ratio,
            "is_retry": is_retry,
            "used_llm": used_llm,
        },
    )
    annotate_node_span(
        "retrieval_router", 0, strategy, duration_ms,
        extra={
            "rule_applied": rule_applied,
            "semantic_ratio": semantic_ratio,
        },
    )

    return state
