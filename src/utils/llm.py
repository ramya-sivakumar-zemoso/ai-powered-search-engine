"""Shared LLM utilities — cached client, token extraction, response cleaning."""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Lazy-cached ChatOpenAI instance (reused across all nodes)
_llm_instance: ChatOpenAI | None = None

# Max output tokens per LLM call — prevents runaway cost from long outputs
LLM_MAX_TOKENS = 1024


def get_llm() -> ChatOpenAI:
    """Return a cached ChatOpenAI client (created once, reused across calls).

    Uses settings from .env: OPENAI_MODEL, OPENAI_API_KEY.
    temperature=0 for deterministic output (PRD Section 5).
    max_tokens capped to prevent unexpected cost spikes.
    """
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
            api_key=settings.openai_api_key,
            max_tokens=LLM_MAX_TOKENS,
        )
        logger.info(
            "llm_client_initialized",
            extra={"model": settings.openai_model, "max_tokens": LLM_MAX_TOKENS},
        )
    return _llm_instance


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) from LLM responses.

    GPT sometimes wraps JSON output in code fences despite prompt instructions.
    This strips them so json.loads() can parse the content cleanly.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return text


def extract_token_usage(response) -> tuple[int, int]:
    """Extract prompt and completion token counts from a LangChain LLM response.

    Handles both attribute locations used by different LangChain versions:
      - response.usage_metadata (newer)
      - response.response_metadata["token_usage"] (older)

    Returns:
        (prompt_tokens, completion_tokens)
    """
    usage = getattr(response, "usage_metadata", None) or {}
    if usage:
        return (
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )

    meta = getattr(response, "response_metadata", {}) or {}
    token_usage = meta.get("token_usage", {})
    return (
        token_usage.get("prompt_tokens", 0),
        token_usage.get("completion_tokens", 0),
    )
