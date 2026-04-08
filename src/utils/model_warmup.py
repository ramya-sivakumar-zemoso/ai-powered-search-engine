"""Startup model warmup helpers to reduce first-query latency."""
from __future__ import annotations

import threading

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

_warmup_started = False
_warmup_lock = threading.Lock()


def _warmup_worker() -> None:
    """Best-effort warmup of expensive local models used in pipeline."""
    try:
        from src.nodes.query_understander import ensure_injection_scanner_loaded
        from src.nodes.reranker import preload_cross_encoder

        scanner_ok = ensure_injection_scanner_loaded()
        reranker_ok = preload_cross_encoder()
        logger.info(
            "model_warmup_completed",
            extra={"scanner_loaded": scanner_ok, "reranker_loaded": reranker_ok},
        )
    except Exception as exc:
        logger.warning(
            "model_warmup_failed",
            extra={"error": str(exc)[:200]},
        )


def start_background_warmup() -> None:
    """Kick off one-time background warmup when enabled in config."""
    global _warmup_started
    if not settings.warmup_models_on_start:
        return

    with _warmup_lock:
        if _warmup_started:
            return
        _warmup_started = True

    thread = threading.Thread(target=_warmup_worker, name="model-warmup", daemon=True)
    thread.start()
    logger.info("model_warmup_started")
