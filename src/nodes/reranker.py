"""
Reranker node — cross-encoder re-ranking + LLM explanations with citation audit.

What this node does (plain English):
  1. Takes the accepted search_results from the evaluator
  2. Scores each result against the query using a cross-encoder model
     (a small neural network that reads query + result together)
  3. Re-sorts results by cross-encoder relevance score (top RERANKER_TOP_N)
  4. Asks GPT-4o-mini to explain why each result matches the query
  5. Audits each explanation to check it cites real result content
  6. Writes reranked_results to state as RankedResult objects

Cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Lightweight (~80MB), runs on CPU, no API key needed
  - Reads (query, result_text) pairs and outputs relevance logits
  - We apply sigmoid to convert logits to 0-1 confidence scores
  - Downloaded from HuggingFace on first use (one-time ~80MB)

LLM explanations (PRD Section 4.6):
  - One batch GPT-4o-mini call for all results (token-efficient)
  - Each explanation references why the result matches the user's query
  - Budget-aware: skips explanations if token budget is exceeded

Citation audit (PRD Section 4.6):
  - VERIFIED:   explanation references actual result content (title/description/genre)
  - DEGRADED:   confidence < CONFIDENCE_THRESHOLD_DEGRADED (cross-encoder unsure)
  - UNVERIFIED: explanation exists but doesn't reference result content
  - ABSENT:     no explanation generated (LLM failure or budget skip)

Graceful degradation:
  - Cross-encoder fails  → fall back to original Meilisearch ranking
  - LLM fails            → skip explanations (status = ABSENT)
  - Budget exceeded       → cross-encoder only, no LLM call
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.models.state import (
    RankedResult,
    ExplanationStatus,
    ExtractionError,
    ErrorSeverity,
    TokenUsage,
)
from src.utils.config import get_settings
from src.utils.logger import get_logger, log_node_exit
from src.utils.langwatch_tracker import (
    get_langwatch_callback,
    check_budget,
    make_budget_exceeded_error,
    annotate_node_span,
)

logger = get_logger(__name__)
settings = get_settings()

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Lazy-loaded cross-encoder instance (avoids slow import on every startup)
_cross_encoder_instance = None

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "reranker_explanation.txt"
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Load cross-encoder (lazy — first call downloads ~80MB model)
# ══════════════════════════════════════════════════════════════════════════════

def _get_cross_encoder():
    """Load the cross-encoder model on first use (lazy singleton).

    The model is downloaded from HuggingFace on first invocation (~80MB).
    Subsequent calls return the cached instance.
    """
    global _cross_encoder_instance
    if _cross_encoder_instance is None:
        from sentence_transformers import CrossEncoder

        logger.info("loading_cross_encoder", extra={"model": CROSS_ENCODER_MODEL})
        _cross_encoder_instance = CrossEncoder(CROSS_ENCODER_MODEL)
        logger.info("cross_encoder_loaded", extra={"model": CROSS_ENCODER_MODEL})
    return _cross_encoder_instance


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Score results with cross-encoder
# ══════════════════════════════════════════════════════════════════════════════

def _score_with_cross_encoder(
    query: str,
    results: list[dict],
) -> list[tuple[int, float]]:
    """Score each result against the query using the cross-encoder.

    How it works:
      1. For each result, build a text pair: (query, "title. description")
      2. The cross-encoder reads both together and outputs a relevance logit
      3. We apply sigmoid to convert the logit to a 0-1 probability
      4. Return sorted (original_index, score) pairs — highest score first

    Args:
        query: The user's search query.
        results: List of SearchResult dicts from the searcher.

    Returns:
        List of (original_index, sigmoid_score) tuples, sorted descending.
    """
    model = _get_cross_encoder()

    pairs = []
    for result in results:
        title = result.get("title", "")
        description = result.get("source_fields", {}).get("description", "")
        text = f"{title}. {description}" if description else title
        pairs.append((query, text))

    raw_scores = model.predict(pairs)

    sigmoid_scores = [1.0 / (1.0 + math.exp(-float(s))) for s in raw_scores]

    indexed_scores = [(i, score) for i, score in enumerate(sigmoid_scores)]
    indexed_scores.sort(key=lambda x: x[1], reverse=True)

    return indexed_scores


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Build context string for the LLM explanation prompt
# ══════════════════════════════════════════════════════════════════════════════

def _build_results_context(results: list[dict]) -> str:
    """Build a text block describing each result for the LLM prompt."""
    lines = []
    for i, r in enumerate(results, 1):
        result_id = r.get("id", "?")
        title = r.get("title", "Untitled")
        category = r.get("source_fields", {}).get("category", "")
        genres = r.get("source_fields", {}).get("genres_all", "")
        description = r.get("source_fields", {}).get("description", "")

        if len(description) > 250:
            description = description[:250] + "..."

        line = f'[Result {i}] id="{result_id}", title="{title}"'
        if category:
            line += f', category="{category}"'
        if genres:
            line += f', genres="{genres}"'
        if description:
            line += f', description="{description}"'
        lines.append(line)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Generate LLM explanations (one batch call for all results)
# ══════════════════════════════════════════════════════════════════════════════

def _generate_explanations(
    query: str,
    ranked_results: list[dict],
) -> tuple[list[dict], int, int]:
    """Ask GPT-4o-mini to explain why each result matches the query.

    One batch call covers all results (avoids N separate API calls).

    Returns:
        (explanations_list, prompt_tokens, completion_tokens)
    """
    results_context = _build_results_context(ranked_results)
    human_content = f"Query: {query}\n\nResults:\n{results_context}"

    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    callback_config = get_langwatch_callback()
    response = llm.invoke(messages, config=callback_config)

    text = response.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    explanations = json.loads(text)

    usage = response.usage_metadata or {}
    prompt_tokens = usage.get("input_tokens", 0)
    completion_tokens = usage.get("output_tokens", 0)

    return explanations, prompt_tokens, completion_tokens


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Citation audit — verify explanation references result content
# ══════════════════════════════════════════════════════════════════════════════

def _audit_citation(
    explanation: str,
    result: dict,
    confidence: float,
) -> tuple[ExplanationStatus, list[str]]:
    """Check if the explanation references actual content from the result.

    Logic (applied in order):
      1. No explanation text           → ABSENT
      2. Confidence < degraded threshold → DEGRADED
      3. Explanation cites title/description/genre → VERIFIED
      4. Otherwise                      → UNVERIFIED

    Args:
        explanation: The LLM-generated explanation text.
        result: The search result dict.
        confidence: The cross-encoder confidence score (0-1).

    Returns:
        (ExplanationStatus, list of cited result IDs)
    """
    if not explanation:
        return ExplanationStatus.ABSENT, []

    if confidence < settings.confidence_threshold_degraded:
        return ExplanationStatus.DEGRADED, [str(result.get("id", ""))]

    result_id = str(result.get("id", ""))
    title = result.get("title", "").lower()
    description = result.get("source_fields", {}).get("description", "").lower()
    category = result.get("source_fields", {}).get("category", "").lower()
    genres = result.get("source_fields", {}).get("genres_all", "").lower()

    explanation_lower = explanation.lower()
    cited = False

    # Title check: at least 2 significant title words found in explanation
    if title and len(title) >= 3:
        title_words = [w for w in title.split() if len(w) >= 3]
        matching = sum(1 for w in title_words if w in explanation_lower)
        if matching >= min(2, len(title_words)):
            cited = True

    # Description check: at least 3 significant words from description
    if not cited and description:
        desc_words = [w for w in description.split() if len(w) >= 5][:20]
        matching = sum(1 for w in desc_words if w in explanation_lower)
        if matching >= 3:
            cited = True

    # Category check: exact category name in explanation
    if not cited and category and len(category) >= 3 and category in explanation_lower:
        cited = True

    # Genre check: any genre name in explanation
    if not cited and genres:
        genre_list = [g.strip().lower() for g in genres.split(",") if len(g.strip()) >= 3]
        if any(g in explanation_lower for g in genre_list):
            cited = True

    if cited:
        return ExplanationStatus.VERIFIED, [result_id]
    return ExplanationStatus.UNVERIFIED, []


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN NODE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def reranker_node(state: dict) -> dict:
    """Re-rank search results using a cross-encoder and add LLM explanations.

    Pipeline position: evaluator → [accept] → reranker → reporter → END

    Steps:
      1. Take top RERANKER_TOP_N results from search_results
      2. Score each against the query with cross-encoder (local model, 0 tokens)
      3. Re-sort by cross-encoder score (highest first)
      4. Generate LLM explanations (if budget allows)
      5. Audit each explanation for citation quality (PRD Section 4.6)
      6. Write reranked_results to state

    Args:
        state: The pipeline state dict.

    Returns:
        Updated state dict with reranked_results.
    """
    start = time.perf_counter()
    query = state.get("query", "")
    query_hash = state.get("query_hash", "")
    results = state.get("search_results", [])
    strategy = state.get("retrieval_strategy", "HYBRID")

    top_n = settings.reranker_top_n
    candidates = results[:top_n]

    # ── Early exit: nothing to rerank ─────────────────────────────────────
    if not candidates:
        state["reranked_results"] = []
        duration_ms = (time.perf_counter() - start) * 1000
        log_node_exit(logger, "reranker", query_hash, 0, strategy, duration_ms, 0.0)
        annotate_node_span("reranker", 0, strategy, duration_ms)
        return state

    # ── Step 1: Cross-encoder scoring ─────────────────────────────────────
    try:
        scored = _score_with_cross_encoder(query, candidates)
    except Exception as exc:
        logger.warning(
            "cross_encoder_failed",
            extra={"error": str(exc)[:200], "fallback": "original_ranking"},
        )
        # Fallback: keep original Meilisearch order with descending scores
        scored = [(i, 1.0 - (i * 0.01)) for i in range(len(candidates))]

        errors = state.get("errors", [])
        errors.append(
            ExtractionError(
                node="reranker",
                severity=ErrorSeverity.WARNING,
                message=f"CROSS_ENCODER_FAILED: {str(exc)[:150]}",
                fallback_applied=True,
                fallback_description=(
                    "Cross-encoder model failed. Falling back to original "
                    "Meilisearch ranking order."
                ),
            ).model_dump()
        )
        state["errors"] = errors

    # ── Step 2: Build re-ranked list in cross-encoder order ───────────────
    reranked_candidates = []
    for new_rank, (original_idx, ce_score) in enumerate(scored):
        result = candidates[original_idx]
        reranked_candidates.append({
            "result": result,
            "original_rank": original_idx + 1,
            "new_rank": new_rank + 1,
            "relevance_score": round(ce_score, 4),
            "confidence": round(ce_score, 4),
        })

    # ── Step 3: Generate LLM explanations (if budget allows) ─────────────
    explanations_map: dict[str, str] = {}
    explanation_generated = False

    try:
        check_budget(
            state.get("cumulative_token_cost", 0.0),
            "reranker",
            query_hash,
        )

        ordered_for_llm = [rc["result"] for rc in reranked_candidates]
        raw_explanations, prompt_tokens, completion_tokens = _generate_explanations(
            query, ordered_for_llm,
        )

        for entry in raw_explanations:
            if isinstance(entry, dict):
                explanations_map[str(entry.get("id", ""))] = entry.get("explanation", "")

        cost_usd = (prompt_tokens * 0.000000150) + (completion_tokens * 0.000000600)

        token_list = state.get("token_usage", [])
        token_list.append(
            TokenUsage(
                node="reranker",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=round(cost_usd, 8),
            ).model_dump()
        )
        state["token_usage"] = token_list
        state["cumulative_token_cost"] = round(
            state.get("cumulative_token_cost", 0.0) + cost_usd, 8
        )

        explanation_generated = True

    except ValueError:
        logger.warning(
            "reranker_budget_exceeded",
            extra={"fallback": "cross_encoder_only"},
        )
        errors = state.get("errors", [])
        errors.append(make_budget_exceeded_error("reranker").model_dump())
        state["errors"] = errors

    except json.JSONDecodeError as exc:
        logger.warning(
            "explanation_parse_failed",
            extra={"error": str(exc)[:200], "fallback": "no_explanations"},
        )
        errors = state.get("errors", [])
        errors.append(
            ExtractionError(
                node="reranker",
                severity=ErrorSeverity.WARNING,
                message=f"EXPLANATION_PARSE_FAILED: {str(exc)[:150]}",
                fallback_applied=True,
                fallback_description=(
                    "GPT returned invalid JSON for explanations. "
                    "Results are re-ranked but without explanations."
                ),
            ).model_dump()
        )
        state["errors"] = errors

    except Exception as exc:
        logger.warning(
            "explanation_generation_failed",
            extra={"error": str(exc)[:200], "fallback": "no_explanations"},
        )
        errors = state.get("errors", [])
        errors.append(
            ExtractionError(
                node="reranker",
                severity=ErrorSeverity.WARNING,
                message=f"EXPLANATION_FAILED: {str(exc)[:150]}",
                fallback_applied=True,
                fallback_description=(
                    "LLM explanation call failed. Results are re-ranked "
                    "by cross-encoder but without explanations."
                ),
            ).model_dump()
        )
        state["errors"] = errors

    # ── Step 4: Citation audit + build RankedResult objects ───────────────
    reranked_results = []
    for rc in reranked_candidates:
        result = rc["result"]
        result_id = str(result.get("id", ""))
        confidence = rc["confidence"]

        explanation = explanations_map.get(result_id, "")

        if explanation_generated and explanation:
            status, citation_ids = _audit_citation(explanation, result, confidence)
        else:
            status = ExplanationStatus.ABSENT
            citation_ids = []

        ranked = RankedResult(
            id=result_id,
            original_rank=rc["original_rank"],
            new_rank=rc["new_rank"],
            relevance_score=rc["relevance_score"],
            confidence=confidence,
            explanation=explanation,
            explanation_citation_ids=citation_ids,
            explanation_status=status,
        )
        reranked_results.append(ranked.model_dump())

    state["reranked_results"] = reranked_results

    # ── Step 5: Compute aggregate rerank_confidence signal (PRD 5th signal)
    # The evaluator couldn't score this (runs before reranker). Now that we
    # have cross-encoder confidence, we append it to quality_scores so the
    # reporter and summary output show all 5 signals.
    quality_scores = state.get("quality_scores", {})
    if reranked_results:
        confidences = [r["confidence"] for r in reranked_results]
        mean_conf = sum(confidences) / len(confidences)
        low_conf_ratio = sum(1 for c in confidences if c < 0.5) / len(confidences)
        quality_scores["rerank_confidence"] = round(mean_conf, 4)
        quality_scores["rerank_low_confidence_ratio"] = round(low_conf_ratio, 4)
    else:
        quality_scores["rerank_confidence"] = 0.0
        quality_scores["rerank_low_confidence_ratio"] = 1.0
    state["quality_scores"] = quality_scores

    # ── Step 6: Log and annotate node exit ────────────────────────────────
    duration_ms = (time.perf_counter() - start) * 1000
    token_cost = state.get("cumulative_token_cost", 0.0)

    verified_count = sum(
        1 for r in reranked_results
        if r.get("explanation_status") == "VERIFIED"
    )

    log_node_exit(
        logger, "reranker", query_hash,
        len(reranked_results), strategy, duration_ms, token_cost,
        extra={
            "explanations_generated": explanation_generated,
            "verified_count": verified_count,
            "top_confidence": reranked_results[0]["confidence"] if reranked_results else 0.0,
            "mean_confidence": quality_scores.get("rerank_confidence", 0.0),
            "low_confidence_ratio": quality_scores.get("rerank_low_confidence_ratio", 0.0),
        },
    )
    annotate_node_span(
        "reranker", len(reranked_results), strategy, duration_ms,
        extra={
            "explanations_generated": explanation_generated,
            "verified_count": verified_count,
            "mean_confidence": quality_scores.get("rerank_confidence", 0.0),
        },
    )

    return state
