"""Serialize graph state for JSON display and compute state diffs between nodes."""

from __future__ import annotations

import json
from datetime import date, datetime
from enum import Enum
from typing import Any

_MISSING = object()
_DEFAULT_MAX_HITS = 25


def state_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Return only the keys that changed between two pipeline snapshots."""
    delta: dict[str, Any] = {}

    for key, val in after.items():
        if key not in before or not _equal(before[key], val):
            delta[key] = val

    removed = [k for k in before if k not in after]
    if removed:
        delta["_removed_keys"] = removed

    return delta


def _equal(a: Any, b: Any) -> bool:
    if a is b:
        return True
    try:
        return (
            json.dumps(to_jsonable(a), sort_keys=True, default=str)
            == json.dumps(to_jsonable(b), sort_keys=True, default=str)
        )
    except (TypeError, ValueError):
        return a == b


def to_jsonable(obj: Any, *, max_search_hits: int = _DEFAULT_MAX_HITS, _depth: int = 0) -> Any:
    """Convert any pipeline state value into something JSON-serializable."""
    if _depth > 32:
        return repr(obj)

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    if isinstance(obj, Enum):
        return obj.value

    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k in ("search_results", "reranked_results") and isinstance(v, list):
                result[k] = _maybe_truncate(v, max_search_hits, max_search_hits, _depth)
            else:
                result[str(k)] = to_jsonable(v, max_search_hits=max_search_hits, _depth=_depth + 1)
        return result

    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x, max_search_hits=max_search_hits, _depth=_depth + 1) for x in obj]

    if hasattr(obj, "model_dump"):
        try:
            return to_jsonable(obj.model_dump(mode="python"), max_search_hits=max_search_hits, _depth=_depth + 1)
        except Exception:
            return repr(obj)

    return repr(obj)


def _maybe_truncate(items: list, limit: int, max_search_hits: int, depth: int) -> Any:
    serialized = [to_jsonable(x, max_search_hits=max_search_hits, _depth=depth + 1) for x in items[:limit]]
    if len(items) <= limit:
        return serialized
    return {
        "_truncated": True,
        "_total": len(items),
        "_showing": limit,
        "items": serialized,
    }