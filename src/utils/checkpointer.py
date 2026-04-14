"""
LangGraph checkpointer factory.

Builds and returns the configured persistence backend for graph state.
A checkpointer lets the pipeline resume from the last completed node after a
mid-pipeline crash, rather than restarting the full query from scratch.

Supported backends (set CHECKPOINTER_TYPE in .env):
  sqlite   — SqliteSaver; persistent file on disk, zero extra services required.
             Requires: pip install langgraph-checkpoint-sqlite
  postgres — PostgresSaver; shared, multi-process safe.
             Requires: pip install langgraph-checkpoint-postgres
             Set CHECKPOINTER_POSTGRES_URL to the connection string.
  memory   — MemorySaver; in-process only, state lost on restart.
             Useful for testing or single-process dev.
  none     — No checkpointing; graph.compile() receives checkpointer=None.

Fallback chain:
  sqlite (import fails) → MemorySaver + warning
  postgres (import fails OR no URL) → MemorySaver + warning
  unknown type → MemorySaver + warning
"""
from __future__ import annotations

import sqlite3
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_checkpointer() -> Any | None:
    """
    Instantiate and return the checkpointer backend selected by CHECKPOINTER_TYPE.

    Returns:
        A LangGraph-compatible checkpointer object, or ``None`` if
        ``CHECKPOINTER_TYPE=none`` (disables persistence).
    """
    cfg = get_settings()
    ctype = cfg.checkpointer_type

    if ctype == "none":
        logger.info("checkpointer_disabled", extra={"type": "none"})
        return None

    if ctype == "memory":
        logger.info("checkpointer_initialized", extra={"type": "memory"})
        return MemorySaver()

    if ctype == "sqlite":
        return _build_sqlite(cfg.checkpointer_sqlite_path)

    if ctype == "postgres":
        return _build_postgres(cfg.checkpointer_postgres_url)

    logger.warning(
        "checkpointer_unknown_type",
        extra={"type": ctype, "falling_back_to": "memory"},
    )
    return MemorySaver()


# ── Backend builders ──────────────────────────────────────────────────────────

def _build_sqlite(db_path: str) -> Any:
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore[import]

        conn = sqlite3.connect(db_path, check_same_thread=False)
        saver = SqliteSaver(conn)
        # Initialize schema (no-op if tables already exist).
        try:
            saver.setup()
        except Exception:
            pass
        logger.info("checkpointer_initialized", extra={"type": "sqlite", "path": db_path})
        return saver

    except ImportError:
        logger.warning(
            "checkpointer_fallback",
            extra={
                "requested": "sqlite",
                "reason": "langgraph-checkpoint-sqlite not installed; using MemorySaver",
                "fix": "pip install langgraph-checkpoint-sqlite",
            },
        )
        return MemorySaver()


def _build_postgres(postgres_url: str) -> Any:
    if not postgres_url:
        logger.warning(
            "checkpointer_fallback",
            extra={
                "requested": "postgres",
                "reason": "CHECKPOINTER_POSTGRES_URL is not set; using MemorySaver",
                "fix": "Set CHECKPOINTER_POSTGRES_URL=postgresql://user:pass@host:port/db",
            },
        )
        return MemorySaver()

    try:
        from langgraph.checkpoint.postgres import PostgresSaver  # type: ignore[import]

        saver = PostgresSaver.from_conn_string(postgres_url)
        try:
            saver.setup()
        except Exception:
            pass
        logger.info("checkpointer_initialized", extra={"type": "postgres"})
        return saver

    except ImportError:
        logger.warning(
            "checkpointer_fallback",
            extra={
                "requested": "postgres",
                "reason": "langgraph-checkpoint-postgres not installed; using MemorySaver",
                "fix": "pip install langgraph-checkpoint-postgres",
            },
        )
        return MemorySaver()
