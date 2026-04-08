"""
Reranker node — cross-encoder re-ranking + LLM explanations with citation audit.

What this node does:
  1. Takes the accepted search_results from the evaluator
  2. Scores each result against the query using a cross-encoder model
     (a small neural network that reads query + result together)
  3. Re-sorts results by cross-encoder relevance score (top RERANKER_TOP_N)
  4. Asks GPT-4o-mini to explain why each result matches the query
  5. Audits each explanation to check it cites real result content
  6. Writes reranked_results to state as RankedResult objects

Cross-encoder: configurable via RERANKER_MODEL in .env
  (default: BAAI/bge-reranker-v2-m3)
  - Lightweight (~1.1GB), runs on CPU, no API key needed
  - Reads (query, result_text) pairs and outputs relevance logits
  - We apply sigmoid to convert logits to 0-1 confidence scores
  - Downloaded from Hugging Face on first use (one-time ~1.1GB)

LLM explanations (PRD Section 4.6):
  - One batch GPT-4o-mini call for all results (token-efficient)
  - Each explanation references why the result matches the user's query
  - Budget-aware: skips explanations if token budget is exceeded

Citation audit (PRD Section 4.6):
  - VERIFIED:   explanation references actual result content (title/description/facets)
  - DEGRADED:   explanation doesn't cite result content AND cross-encoder confidence
                 is below CONFIDENCE_THRESHOLD_DEGRADED
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
import re
import time
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage

from src.models.dataset_schema import DatasetSchema
from src.models.schema_registry import get_schema
from src.models.state import (
    RankedResult,
    ExplanationStatus,
    ExtractionError,
    ErrorSeverity,
    TokenUsage,
    PipelineEvent,
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

# Lazy-loaded cross-encoder instance (avoids slow import on every startup)
_cross_encoder_instance = None
_cross_encoder_model_name = None

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "reranker_explanation.txt"
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Load cross-encoder (lazy — first call downloads one-time ~1.1GB model)
# ══════════════════════════════════════════════════════════════════════════════

def _get_cross_encoder():
    """Load the cross-encoder model on first use (lazy singleton).

    The model is downloaded from Hugging Face on first invocation (one-time ~1.1GB).
    Subsequent calls return the cached instance. If RERANKER_MODEL changes
    in .env, a restart is needed (lru_cache on settings).

    If the network call fails (e.g. SSL errors), retries with
    ``local_files_only=True`` so a previously-cached model still works
    without affecting the global HF_HUB_OFFLINE setting.
    """
    global _cross_encoder_instance, _cross_encoder_model_name
    model_name = settings.reranker_model
    if _cross_encoder_instance is None or _cross_encoder_model_name != model_name:
        from sentence_transformers import CrossEncoder

        logger.info("loading_cross_encoder", extra={"model": model_name})
        try:
            _cross_encoder_instance = CrossEncoder(model_name)
        except Exception as exc:
            exc_str = str(exc)
            if "SSL" in exc_str or "MaxRetry" in exc_str or "ConnectionError" in exc_str:
                logger.warning(
                    "cross_encoder_network_error_retrying_offline",
                    extra={"error": str(exc)[:200]},
                )
                _cross_encoder_instance = CrossEncoder(
                    model_name, local_files_only=True,
                )
            else:
                raise
        _cross_encoder_model_name = model_name
        logger.info("cross_encoder_loaded", extra={"model": model_name})
    return _cross_encoder_instance


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Score results with cross-encoder
# ══════════════════════════════════════════════════════════════════════════════

def _score_with_cross_encoder(
    query: str,
    results: list[dict],
    schema: DatasetSchema | None = None,
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
    sch = schema or get_schema(settings.dataset_schema)
    model = _get_cross_encoder()

    pairs = []
    labels = sch.rerank_field_labels
    for result in results:
        title = (result.get("title", "") or "").strip()
        source = result.get("source_fields", {})
        parts: list[str] = []
        if title:
            parts.append(title)
        for field in sch.rerank_auxiliary_fields:
            raw = source.get(field, "")
            if raw is None or raw == "":
                continue
            if field == "description":
                parts.append(str(raw))
            else:
                label = labels.get(field, field.replace("_", " ").title())
                parts.append(f"{label}: {raw}")
        text = ". ".join(parts) if parts else title or "."
        pairs.append((query, text))

    raw_scores = model.predict(pairs)

    sigmoid_scores = [1.0 / (1.0 + math.exp(-float(s))) for s in raw_scores]

    indexed_scores = [(i, score) for i, score in enumerate(sigmoid_scores)]
    indexed_scores.sort(key=lambda x: x[1], reverse=True)

    return indexed_scores


def _native_meili_order(results: list[dict]) -> list[tuple[int, float]]:
    """Return candidates ordered by original Meilisearch score (desc)."""
    indexed_scores = [
        (i, float(r.get("score", 0.0)))
        for i, r in enumerate(results)
    ]
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    return indexed_scores


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Sanitize document fields before feeding to LLM (PRD 4.7)
# ══════════════════════════════════════════════════════════════════════════════

_INJECTION_PATTERNS = re.compile(
    r"(?i)^(SYSTEM\s*:|ASSISTANT\s*:|USER\s*:|"
    r"ignore\s+previous|forget\s+all|disregard\s+above|"
    r"override\s+instructions|you\s+must\s+always\s+rank|"
    r"always\s+rank\s+this|do\s+not\s+follow)"
)


def _sanitize_field(text: str) -> str:
    """Strip lines that look like prompt injection from a document field.

    Removes lines matching known injection prefixes while keeping
    legitimate content intact.
    """
    cleaned = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and _INJECTION_PATTERNS.match(stripped):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Build context string for the LLM explanation prompt
# ══════════════════════════════════════════════════════════════════════════════

def _build_results_context(
    results: list[dict],
    schema: DatasetSchema | None = None,
) -> str:
    """Build a text block with explicit content boundaries for each result.

    Each field is wrapped in `<doc_*>` tags (PRD 4.7). Dynamic fields come from
    ``schema.rerank_auxiliary_fields`` (plus ``doc_title`` always).
    """
    sch = schema or get_schema(settings.dataset_schema)
    lines = []
    for i, r in enumerate(results, 1):
        result_id = r.get("id", "?")
        title = _sanitize_field(r.get("title", "Untitled"))
        source = r.get("source_fields", {})

        line = f'[Result {i}] id="{result_id}"'
        line += f', <doc_title>{title}</doc_title>'
        for field in sch.rerank_auxiliary_fields:
            raw = source.get(field, "")
            if raw is None or raw == "":
                continue
            text = _sanitize_field(str(raw))
            if field == "description" and len(text) > 250:
                text = text[:250] + "..."
            tag = f"doc_{field}"
            line += f", <{tag}>{text}</{tag}>"
        lines.append(line)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Generate LLM explanations (one batch call for all results)
# ══════════════════════════════════════════════════════════════════════════════

def _generate_explanations(
    query: str,
    ranked_results: list[dict],
    schema: DatasetSchema | None = None,
) -> tuple[list[dict], int, int]:
    """Ask GPT-4o-mini to explain why each result matches the query.

    One batch call covers all results (avoids N separate API calls).

    Returns:
        (explanations_list, prompt_tokens, completion_tokens)
    """
    sch = schema or get_schema(settings.dataset_schema)
    results_context = _build_results_context(ranked_results, sch)
    human_content = f"Query: {query}\n\nResults:\n{results_context}"

    llm = get_llm()

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    response = llm.invoke(messages, config=get_langwatch_callback())

    text = strip_markdown_fences(response.content)
    explanations = json.loads(text)

    prompt_tokens, completion_tokens = extract_token_usage(response)

    return explanations, prompt_tokens, completion_tokens


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Citation audit — verify explanation references result content
# ══════════════════════════════════════════════════════════════════════════════

def _audit_citation(
    explanation: str,
    result: dict,
    confidence: float,
    schema: DatasetSchema | None = None,
) -> tuple[ExplanationStatus, list[str]]:
    """Check if the explanation references actual content from the result.

    Logic (applied in order):
      1. No explanation text                        → ABSENT
      2. Explanation cites title/description/facets → VERIFIED
      3. Confidence < degraded threshold (uncited)   → DEGRADED
      4. Otherwise                                   → UNVERIFIED

    Args:
        explanation: The LLM-generated explanation text.
        result: The search result dict.
        confidence: The cross-encoder confidence score (0-1).

    Returns:
        (ExplanationStatus, list of cited result IDs)
    """
    if not explanation:
        return ExplanationStatus.ABSENT, []

    sch = schema or get_schema(settings.dataset_schema)

    result_id = str(result.get("id", ""))
    title = result.get("title", "").lower()
    description = result.get("source_fields", {}).get("description", "").lower()
    category = result.get("source_fields", {}).get("category", "").lower()
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

    # Configurable tag fields (e.g. genres_all, comma-separated teams/tags)
    if not cited:
        for field in sch.citation_tag_fields:
            raw = result.get("source_fields", {}).get(field, "")
            blob = (raw or "").lower()
            if not blob:
                continue
            tokens = [g.strip().lower() for g in blob.split(",") if len(g.strip()) >= 3]
            if any(g in explanation_lower for g in tokens):
                cited = True
                break

    if cited:
        return ExplanationStatus.VERIFIED, [result_id]

    if confidence < settings.confidence_threshold_degraded:
        return ExplanationStatus.DEGRADED, [result_id]

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
        Dict of only the keys this node changed.
    """
    start = time.perf_counter()
    query = state.get("query", "")
    query_hash = state.get("query_hash", "")
    results = state.get("search_results", [])
    strategy = state.get("retrieval_strategy", "HYBRID")

    updates: dict = {}
    new_errors: list[dict] = []
    new_token_usage: list[dict] = []

    top_n = settings.reranker_top_n
    candidates = results[:top_n]
    schema = get_schema(settings.dataset_schema)
    rerank_degraded = False

    # ── Early exit: nothing to rerank ─────────────────────────────────────
    if not candidates:
        updates["reranked_results"] = []
        duration_ms = (time.perf_counter() - start) * 1000
        log_node_exit(logger, "reranker", query_hash, 0, strategy, duration_ms, 0.0)
        annotate_node_span("reranker", 0, strategy, duration_ms)
        return updates

    # ── Step 1: Cross-encoder scoring ─────────────────────────────────────
    try:
        scored = _score_with_cross_encoder(query, candidates, schema)
    except Exception as exc:
        logger.warning(
            "cross_encoder_failed",
            extra={"error": str(exc)[:200], "fallback": "original_ranking"},
        )
        scored = _native_meili_order(candidates)
        rerank_degraded = True

        new_errors.append(
            ExtractionError(
                node="reranker",
                severity=ErrorSeverity.WARNING,
                message=PipelineEvent.RERANK_DEGRADED.value,
                fallback_applied=True,
                fallback_description=(
                    "Cross-encoder model failed. Falling back to original "
                    "Meilisearch ranking order. "
                    f"Cause: {str(exc)[:150]}"
                ),
            ).model_dump()
        )

    if any(score < 0.0 or score > 1.0 for _, score in scored):
        logger.warning(
            "invalid_rerank_confidence",
            extra={"fallback": "original_ranking"},
        )
        scored = _native_meili_order(candidates)
        rerank_degraded = True
        new_errors.append(
            ExtractionError(
                node="reranker",
                severity=ErrorSeverity.WARNING,
                message=PipelineEvent.RERANK_DEGRADED.value,
                fallback_applied=True,
                fallback_description=(
                    "Reranker confidence out of valid range [0.0, 1.0]. "
                    "Falling back to native Meilisearch ranking."
                ),
            ).model_dump()
        )

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

    # ── Step 3: Generate LLM explanations (if budget and fast_mode allow) ──
    explanations_map: dict[str, str] = {}
    explanation_generated = False
    cost_usd = 0.0

    # fast_mode=True skips LLM explanation generation entirely to reduce
    # end-to-end latency by ~400-800 ms.  Cross-encoder scoring still runs.
    if settings.fast_mode:
        logger.info("reranker_fast_mode", extra={"query_hash": query_hash})
        rerank_degraded = True
        new_errors.append(
            ExtractionError(
                node="reranker",
                severity=ErrorSeverity.WARNING,
                message=PipelineEvent.RERANK_DEGRADED.value,
                fallback_applied=True,
                fallback_description=(
                    "fast_mode=true: LLM explanation generation skipped for lower latency. "
                    "Cross-encoder ranking is still applied."
                ),
            ).model_dump()
        )

    if not settings.fast_mode:
        try:
            check_budget(
                state.get("cumulative_token_cost", 0.0),
                "reranker",
                query_hash,
            )

            ordered_for_llm = [rc["result"] for rc in reranked_candidates]
            raw_explanations, prompt_tokens, completion_tokens = _generate_explanations(
                query, ordered_for_llm, schema,
            )

            for entry in raw_explanations:
                if isinstance(entry, dict):
                    explanations_map[str(entry.get("id", ""))] = entry.get("explanation", "")

            cost_usd = (prompt_tokens * 0.000000150) + (completion_tokens * 0.000000600)

            new_token_usage.append(
                TokenUsage(
                    node="reranker",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_usd=round(cost_usd, 8),
                ).model_dump()
            )
            updates["cumulative_token_cost"] = round(
                state.get("cumulative_token_cost", 0.0) + cost_usd, 8
            )

            explanation_generated = True

        except ValueError:
            logger.warning(
                "reranker_budget_exceeded",
                extra={"fallback": "cross_encoder_only"},
            )
            rerank_degraded = True
            new_errors.append(make_budget_exceeded_error("reranker").model_dump())

        except json.JSONDecodeError as exc:
            logger.warning(
                "explanation_parse_failed",
                extra={"error": str(exc)[:200], "fallback": "no_explanations"},
            )
            new_errors.append(
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

        except Exception as exc:
            logger.warning(
                "explanation_generation_failed",
                extra={"error": str(exc)[:200], "fallback": "no_explanations"},
            )
            new_errors.append(
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

    # ── Step 4: Citation audit + build RankedResult objects ───────────────
    reranked_results = []
    for rc in reranked_candidates:
        result = rc["result"]
        result_id = str(result.get("id", ""))
        confidence = rc["confidence"]

        explanation = explanations_map.get(result_id, "")

        if explanation_generated and explanation:
            status, citation_ids = _audit_citation(
                explanation, result, confidence, schema,
            )
        else:
            status = ExplanationStatus.ABSENT
            citation_ids = []

        ranked = RankedResult(
            id=result_id,
            title=str(result.get("title", "")),
            original_rank=rc["original_rank"],
            new_rank=rc["new_rank"],
            relevance_score=rc["relevance_score"],
            confidence=confidence,
            explanation=explanation,
            explanation_citation_ids=citation_ids,
            explanation_status=status,
        )
        reranked_results.append(ranked.model_dump())

    updates["reranked_results"] = reranked_results
    updates["rerank_degraded"] = rerank_degraded

    # ── Step 5: Compute aggregate rerank_confidence signal (PRD 5th signal)
    quality_scores = dict(state.get("quality_scores", {}))
    if reranked_results:
        confidences = [r["confidence"] for r in reranked_results]
        mean_conf = sum(confidences) / len(confidences)
        quality_scores["rerank_confidence"] = round(mean_conf, 4)
        quality_scores["rerank_low_confidence_ratio"] = round(
            sum(1 for c in confidences if c < settings.confidence_threshold_degraded)
            / len(confidences),
            4,
        )
        quality_scores["per_result_rerank_confidence"] = {
            str(r["id"]): round(float(r["confidence"]), 4)
            for r in reranked_results
        }
    else:
        quality_scores["rerank_confidence"] = 0.0
        quality_scores["rerank_low_confidence_ratio"] = 1.0
        quality_scores["per_result_rerank_confidence"] = {}
    updates["quality_scores"] = quality_scores

    if any(r.get("explanation_status") == "DEGRADED" for r in reranked_results):
        updates["rerank_degraded"] = True

    # ── Step 6: Log and annotate node exit ────────────────────────────────
    duration_ms = (time.perf_counter() - start) * 1000
    token_cost = state.get("cumulative_token_cost", 0.0) + cost_usd

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

    if new_errors:
        updates["errors"] = new_errors
    if new_token_usage:
        updates["token_usage"] = new_token_usage

    return updates
