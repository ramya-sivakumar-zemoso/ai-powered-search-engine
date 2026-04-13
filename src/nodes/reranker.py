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
  - VERIFIED:   explicit ``<doc_*>…</doc_*>`` verbatim claims all match the hit, or legacy overlap
  - EXPLANATION_UNVERIFIED: a `<doc_*>` claim references missing/redacted text or content
    not present in that field — explanation text is stripped (never patched with another field)
  - DEGRADED:   explanation doesn't cite result content AND cross-encoder confidence
                 is below CONFIDENCE_THRESHOLD_DEGRADED
  - UNVERIFIED: explanation exists but doesn't reference result content (legacy overlap miss)
  - ABSENT:     no explanation generated (LLM failure or budget skip)
  Each ranked hit always includes ``meilisearch_ranking_score`` (raw Meilisearch `_rankingScore`).

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
from src.utils.injection_guard import (
    collect_signature_hits,
    format_rerank_explanation_human_message,
    get_effective_user_query,
    log_injection_signature_hits,
    sanitize_document_field,
)
from src.utils.llm import get_llm, strip_markdown_fences, extract_token_usage
from src.utils.logger import get_logger, log_node_exit
from src.utils.langwatch_tracker import (
    BUDGET_PROJECT_RERANK_EXPLAIN_USD,
    get_langwatch_callback,
    check_budget_projected,
    make_budget_exceeded_error,
    annotate_node_span,
)
from src.utils.rerank_async import submit_explanation_job, get_explanation_job

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


def preload_cross_encoder() -> bool:
    """Warm up cross-encoder model at service startup."""
    try:
        _get_cross_encoder()
        return True
    except Exception as exc:
        logger.warning(
            "cross_encoder_warmup_failed",
            extra={"error": str(exc)[:200]},
        )
        return False


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
#  HELPER: Build context string for the LLM explanation prompt
# ══════════════════════════════════════════════════════════════════════════════

def _build_results_context(
    results: list[dict],
    schema: DatasetSchema | None = None,
    *,
    query_hash: str = "",
) -> str:
    """Build a text block with explicit content boundaries for each result.

    Each field is wrapped in `<doc_*>` tags (PRD 4.7). Vendor text is sanitised
    via ``injection_guard.sanitize_document_field``; signature hits are logged.
    """
    sch = schema or get_schema(settings.dataset_schema)
    lines = []
    for i, r in enumerate(results, 1):
        result_id = r.get("id", "?")
        raw_title = str(r.get("title", "Untitled") or "Untitled")
        hits_t = collect_signature_hits(raw_title)
        if hits_t:
            log_injection_signature_hits(
                logger,
                source="vendor_listing_field",
                doc_id=str(result_id),
                pattern_names=hits_t,
                query_hash=query_hash or None,
            )
        title, _ = sanitize_document_field(raw_title)

        source = r.get("source_fields", {})

        line = f'[Result {i}] id="{result_id}"'
        line += f', <doc_title>{title}</doc_title>'
        for field in sch.rerank_auxiliary_fields:
            raw = source.get(field, "")
            if raw is None or raw == "":
                continue
            raw_s = str(raw)
            hits = collect_signature_hits(raw_s)
            if hits:
                log_injection_signature_hits(
                    logger,
                    source="vendor_listing_field",
                    doc_id=str(result_id),
                    pattern_names=hits,
                    query_hash=query_hash or None,
                )
            text, strip_applied = sanitize_document_field(raw_s)
            if strip_applied:
                logger.warning(
                    "injection_listing_instruction_stripped",
                    extra={
                        "event": "injection_listing_instruction_stripped",
                        "query_hash": query_hash,
                        "doc_id": str(result_id),
                        "field": field,
                        "stripped_patterns": strip_applied,
                    },
                )
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
    sanitized_query: str,
    ranked_results: list[dict],
    schema: DatasetSchema | None = None,
    *,
    query_hash: str = "",
) -> tuple[list[dict], int, int, float]:
    """Ask GPT-4o-mini to explain why each result matches the query.

    One batch call covers all results (avoids N separate API calls).

    Returns:
        (explanations_list, prompt_tokens, completion_tokens, cost_usd)
    """
    sch = schema or get_schema(settings.dataset_schema)
    results_context = _build_results_context(
        ranked_results, sch, query_hash=query_hash,
    )
    human_content = format_rerank_explanation_human_message(
        sanitized_query,
        results_context,
    )

    llm = get_llm(
        request_timeout=settings.reranker_explain_timeout_seconds,
        max_retries=settings.reranker_explain_max_retries,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    response = llm.invoke(messages, config=get_langwatch_callback())

    text = strip_markdown_fences(response.content)
    explanations = json.loads(text)

    prompt_tokens, completion_tokens = extract_token_usage(response)
    cost_usd = (prompt_tokens * 0.000000150) + (completion_tokens * 0.000000600)
    return explanations, prompt_tokens, completion_tokens, cost_usd


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Citation audit — verify explanation references result content
# ══════════════════════════════════════════════════════════════════════════════

# Verbatim field citations copied by the LLM from the result context (see reranker_explanation.txt).
_DOC_TAG_RE = re.compile(r"<(doc_[a-z0-9_]+)>([\s\S]*?)</\1>", re.IGNORECASE)
_DOC_TAG_CLEAN_RE = re.compile(r"</?doc_[a-z0-9_]+>", re.IGNORECASE)


def _norm_citation_text(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", (s or "").lower())
    return re.sub(r"\s+", " ", s).strip()


def _sanitize_explanation_for_output(text: str) -> str:
    """Remove internal citation tags before returning explanation to clients."""
    if not text:
        return ""
    cleaned = _DOC_TAG_CLEAN_RE.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _field_text_for_doc_tag(tag_name: str, result: dict) -> str:
    """Map ``doc_*`` tag to the text available on this hit (post-retrieve)."""
    key = tag_name.lower()
    if key == "doc_title":
        return str(result.get("title") or "")
    if not key.startswith("doc_"):
        return ""
    field = key[4:]
    return str(result.get("source_fields", {}).get(field) or "")


def _claim_supported_by_actual(claimed: str, actual: str) -> bool:
    """True if the claimed snippet is substantively present in the field text."""
    nc = _norm_citation_text(claimed)
    na = _norm_citation_text(actual)
    if not nc or not na:
        return False
    if nc in na:
        return True
    words = [w for w in nc.split() if len(w) >= 4]
    if not words:
        shorter = [w for w in nc.split() if len(w) >= 3]
        return bool(shorter) and all(w in na for w in shorter)
    hits = sum(1 for w in words if w in na)
    return hits >= max(2, int(0.67 * len(words)))


def _audit_doc_tag_citations(
    explanation: str,
    result: dict,
) -> ExplanationStatus | None:
    """If ``<doc_*>...</doc_*>`` appear, validate each against the hit.

    Returns:
        ``VERIFIED`` if all tags match their fields, ``EXPLANATION_UNVERIFIED`` if any fail,
        ``None`` if no such tags (fall back to legacy overlap audit).
    """
    matches = list(_DOC_TAG_RE.finditer(explanation))
    if not matches:
        return None

    for m in matches:
        tag = m.group(1).lower()
        inner = (m.group(2) or "").strip()
        if not inner:
            return ExplanationStatus.EXPLANATION_UNVERIFIED
        actual = _field_text_for_doc_tag(tag, result)
        if not actual.strip():
            return ExplanationStatus.EXPLANATION_UNVERIFIED
        if not _claim_supported_by_actual(inner, actual):
            return ExplanationStatus.EXPLANATION_UNVERIFIED

    return ExplanationStatus.VERIFIED


def _audit_citation(
    explanation: str,
    result: dict,
    confidence: float,
    schema: DatasetSchema | None = None,
) -> tuple[ExplanationStatus, list[str]]:
    """Check if the explanation is grounded in the result (tags first, then legacy overlap).

    Explicit ``<doc_field>snippet</doc_field>`` claims are validated per field; we never
    rewrite the explanation — failures become ``EXPLANATION_UNVERIFIED`` and the client
    should drop the text downstream.

    Legacy path (no doc tags): token overlap over title/description/category/tag fields.
    """
    if not explanation:
        return ExplanationStatus.ABSENT, []

    sch = schema or get_schema(settings.dataset_schema)
    result_id = str(result.get("id", ""))

    tagged = _audit_doc_tag_citations(explanation, result)
    if tagged is not None:
        if tagged == ExplanationStatus.EXPLANATION_UNVERIFIED:
            return ExplanationStatus.EXPLANATION_UNVERIFIED, []
        return ExplanationStatus.VERIFIED, [result_id]

    title = result.get("title", "").lower()
    description = result.get("source_fields", {}).get("description", "").lower()
    category = result.get("source_fields", {}).get("category", "").lower()
    explanation_lower = explanation.lower()
    cited = False

    if title and len(title) >= 3:
        title_words = [w for w in title.split() if len(w) >= 3]
        matching = sum(1 for w in title_words if w in explanation_lower)
        if matching >= min(2, len(title_words)):
            cited = True

    if not cited and description:
        desc_words = [w for w in description.split() if len(w) >= 5][:20]
        matching = sum(1 for w in desc_words if w in explanation_lower)
        if matching >= 3:
            cited = True

    if not cited and category and len(category) >= 3 and category in explanation_lower:
        cited = True

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
    query_hash = state.get("query_hash", "")
    query = get_effective_user_query(state)
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
    explanations_pending = False
    explanation_job_id = ""
    explanation_job_status = "SKIPPED"
    explanation_top_k = 0
    explanations_async_mode = bool(settings.reranker_explain_async)
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
        explanation_top_k = min(
            max(settings.reranker_explain_top_k, 0),
            len(reranked_candidates),
        )
        try:
            check_budget_projected(
                state.get("cumulative_token_cost", 0.0),
                BUDGET_PROJECT_RERANK_EXPLAIN_USD,
                "reranker",
                query_hash,
            )

            ordered_for_llm = [rc["result"] for rc in reranked_candidates[:explanation_top_k]]

            if ordered_for_llm and settings.reranker_explain_async:
                def _job_fn() -> tuple[list[dict], int, int, float]:
                    return _generate_explanations(
                        query,
                        ordered_for_llm,
                        schema,
                        query_hash=query_hash,
                    )

                explanation_job_id = submit_explanation_job(_job_fn)
                explanations_pending = True
                explanation_job_status = "PENDING"

            elif ordered_for_llm:
                raw_explanations, prompt_tokens, completion_tokens, cost_usd = _generate_explanations(
                    query,
                    ordered_for_llm,
                    schema,
                    query_hash=query_hash,
                )

                for entry in raw_explanations:
                    if isinstance(entry, dict):
                        explanations_map[str(entry.get("id", ""))] = entry.get("explanation", "")

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
                explanation_job_status = "READY"
            else:
                explanation_job_status = "SKIPPED"

        except json.JSONDecodeError as exc:
            logger.warning(
                "explanation_parse_failed",
                extra={"error": str(exc)[:200], "fallback": "no_explanations"},
            )
            rerank_degraded = True
            explanation_job_status = "FAILED"
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

        except ValueError:
            logger.warning(
                "reranker_budget_exceeded",
                extra={"fallback": "cross_encoder_only"},
            )
            rerank_degraded = True
            explanation_job_status = "SKIPPED"
            new_errors.append(make_budget_exceeded_error("reranker").model_dump())

        except Exception as exc:
            logger.warning(
                "explanation_generation_failed",
                extra={"error": str(exc)[:200], "fallback": "no_explanations"},
            )
            rerank_degraded = True
            explanation_job_status = "FAILED"
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
        meili_score = float(result.get("score", 0.0))

        if explanation_generated and explanation:
            status, citation_ids = _audit_citation(
                explanation, result, confidence, schema,
            )
            if status == ExplanationStatus.EXPLANATION_UNVERIFIED:
                # Do not ship fabricated field claims; Meili score is exposed instead.
                explanation = ""
                citation_ids = []
            else:
                explanation = _sanitize_explanation_for_output(explanation)
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
            meilisearch_ranking_score=meili_score,
            explanation=explanation,
            explanation_citation_ids=citation_ids,
            explanation_status=status,
        )
        reranked_results.append(ranked.model_dump())

    updates["reranked_results"] = reranked_results
    updates["explanations_pending"] = explanations_pending
    updates["explanations_applied"] = explanation_generated and not explanations_pending
    updates["explanation_job_id"] = explanation_job_id
    updates["explanation_job_status"] = explanation_job_status
    updates["explanation_top_k"] = explanation_top_k
    updates["explanations_async"] = explanations_async_mode and not settings.fast_mode

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

    updates["rerank_degraded"] = rerank_degraded or any(
        r.get("explanation_status") in ("DEGRADED", "EXPLANATION_UNVERIFIED")
        for r in reranked_results
    )

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
            "explanations_pending": explanations_pending,
            "verified_count": verified_count,
            "top_confidence": reranked_results[0]["confidence"] if reranked_results else 0.0,
            "mean_confidence": quality_scores.get("rerank_confidence", 0.0),
        },
    )
    annotate_node_span(
        "reranker", len(reranked_results), strategy, duration_ms,
        extra={
            "explanations_generated": explanation_generated,
            "explanations_pending": explanations_pending,
            "verified_count": verified_count,
            "mean_confidence": quality_scores.get("rerank_confidence", 0.0),
        },
    )

    if new_errors:
        updates["errors"] = new_errors
    if new_token_usage:
        updates["token_usage"] = new_token_usage

    return updates


def hydrate_async_explanations_in_state(state: dict) -> dict:
    """Merge completed async explanation job output into a final pipeline state."""
    if not isinstance(state, dict):
        return state
    if not bool(state.get("explanations_pending", False)):
        return state
    if bool(state.get("explanations_applied", False)):
        return state

    job_id = str(state.get("explanation_job_id", "") or "")
    if not job_id:
        return state

    job = get_explanation_job(job_id)
    if not job:
        return state

    job_status = str(job.get("status", "PENDING"))
    state["explanation_job_status"] = "PENDING" if job_status == "PENDING" else job_status
    if job_status == "PENDING":
        return state

    state["explanations_pending"] = False

    if job_status == "FAILED":
        state["explanation_job_status"] = "FAILED"
        state["rerank_degraded"] = True
        errors = list(state.get("errors", []))
        if not any(
            isinstance(e, dict) and str(e.get("message", "")).startswith("EXPLANATION_FAILED_ASYNC")
            for e in errors
        ):
            errors.append(
                ExtractionError(
                    node="reranker",
                    severity=ErrorSeverity.WARNING,
                    message=f"EXPLANATION_FAILED_ASYNC: {str(job.get('error', 'unknown error'))[:150]}",
                    fallback_applied=True,
                    fallback_description=(
                        "Async LLM explanation generation failed. "
                        "Results remain reranked without explanations."
                    ),
                ).model_dump()
            )
            state["errors"] = errors
        return state

    explanations_map: dict[str, str] = {}
    for entry in job.get("explanations", []):
        if isinstance(entry, dict):
            explanations_map[str(entry.get("id", ""))] = str(entry.get("explanation", ""))

    schema = get_schema(settings.dataset_schema)
    search_lookup = {
        str(r.get("id", "")): r
        for r in state.get("search_results", [])
        if isinstance(r, dict)
    }

    reranked_results = []
    for rr in state.get("reranked_results", []):
        if not isinstance(rr, dict):
            continue
        item = dict(rr)
        rid = str(item.get("id", ""))
        explanation = explanations_map.get(rid, "")
        if explanation:
            source_hit = search_lookup.get(
                rid,
                {"id": rid, "title": item.get("title", ""), "source_fields": {}},
            )
            confidence = float(item.get("confidence", 0.0) or 0.0)
            status, citation_ids = _audit_citation(
                explanation,
                source_hit,
                confidence,
                schema,
            )
            if status == ExplanationStatus.EXPLANATION_UNVERIFIED:
                explanation = ""
                citation_ids = []
            else:
                explanation = _sanitize_explanation_for_output(explanation)
            item["explanation"] = explanation
            item["explanation_status"] = str(status.value if hasattr(status, "value") else status)
            item["explanation_citation_ids"] = citation_ids
        reranked_results.append(item)

    state["reranked_results"] = reranked_results
    state["explanations_applied"] = True
    state["explanation_job_status"] = "READY"
    state["rerank_degraded"] = bool(state.get("rerank_degraded", False)) or any(
        isinstance(r, dict) and r.get("explanation_status") in ("DEGRADED", "EXPLANATION_UNVERIFIED")
        for r in reranked_results
    )

    prompt_tokens = int(job.get("prompt_tokens", 0) or 0)
    completion_tokens = int(job.get("completion_tokens", 0) or 0)
    cost_usd = float(job.get("cost_usd", 0.0) or 0.0)
    if prompt_tokens or completion_tokens or cost_usd:
        token_usage = list(state.get("token_usage", []))
        token_usage.append(
            TokenUsage(
                node="reranker_async",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=round(cost_usd, 8),
            ).model_dump()
        )
        state["token_usage"] = token_usage
        state["cumulative_token_cost"] = round(
            float(state.get("cumulative_token_cost", 0.0) or 0.0) + cost_usd,
            8,
        )

    return state
