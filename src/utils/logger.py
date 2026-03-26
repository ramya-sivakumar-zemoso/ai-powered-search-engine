"""JSON logs to stdout; helpers for node exit and injection warnings (stdlib only)."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    """One JSON object per line — no extra pip dependency."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "asctime": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "name": record.name,
            "levelname": record.levelname,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in (
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs", "message",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "taskName",
            ):
                continue
            if key not in payload:
                payload[key] = value
        return json.dumps(payload, default=str, ensure_ascii=False)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def log_node_exit(
    logger: logging.Logger,
    node_name: str,
    query_hash: str,
    result_count: int,
    strategy_used: str,
    duration_ms: float,
    token_cost: float,
    extra: dict | None = None,
) -> None:
    payload = {
        "node_name": node_name,
        "query_hash": query_hash,
        "result_count": result_count,
        "strategy_used": strategy_used,
        "duration_ms": round(duration_ms, 2),
        "token_cost_usd": round(token_cost, 8),
    }
    if extra:
        payload.update(extra)
    logger.info("node_exit", extra=payload)


def log_injection_detection(
    logger: logging.Logger,
    source: str,
    doc_id: str | None,
    pattern_matched: str,
    score: float,
) -> None:
    logger.warning(
        "injection_detected",
        extra={
            "source": source,
            "doc_id": doc_id,
            "pattern_matched": pattern_matched,
            "detection_score": round(score, 4),
        },
    )
