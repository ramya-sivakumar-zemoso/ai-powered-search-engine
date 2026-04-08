"""Shared LLM utilities — cached client, token extraction, response cleaning."""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Lazy-cached ChatOpenAI instances (by timeout/retries/tokens profile)
_llm_instances: dict[tuple[int, int, int], ChatOpenAI] = {}

# Max output tokens per LLM call — prevents runaway cost from long outputs
LLM_MAX_TOKENS = 1024
LLM_REQUEST_TIMEOUT = 30
LLM_MAX_RETRIES = 2


def get_llm(
    *,
    request_timeout: int | None = None,
    max_retries: int | None = None,
    max_tokens: int | None = None,
) -> ChatOpenAI:
    """Return a cached ChatOpenAI client (created once, reused across calls).

    Uses settings from .env: OPENAI_MODEL, OPENAI_API_KEY.
    temperature=0 for deterministic output (PRD Section 5).
    max_tokens capped to prevent unexpected cost spikes.
    request_timeout prevents hung requests in production.
    max_retries handles transient OpenAI API failures.
    """
    timeout = int(request_timeout if request_timeout is not None else LLM_REQUEST_TIMEOUT)
    retries = int(max_retries if max_retries is not None else LLM_MAX_RETRIES)
    tokens = int(max_tokens if max_tokens is not None else LLM_MAX_TOKENS)
    key = (timeout, retries, tokens)

    if key not in _llm_instances:
        _llm_instances[key] = ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
            api_key=settings.openai_api_key,
            max_tokens=tokens,
            request_timeout=timeout,
            max_retries=retries,
        )
        logger.info(
            "llm_client_initialized",
            extra={
                "model": settings.openai_model,
                "max_tokens": tokens,
                "request_timeout": timeout,
                "max_retries": retries,
            },
        )
    return _llm_instances[key]


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
