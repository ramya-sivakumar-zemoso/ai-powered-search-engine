"""In-process async job manager for reranker explanation generation."""
from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from src.utils.logger import get_logger

logger = get_logger(__name__)

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rerank-explain")
_LOCK = threading.Lock()
_MAX_STORED_JOBS = 256
_MAX_JOB_AGE_SECONDS = 1800.0  # 30 minutes


@dataclass
class ExplanationJobRecord:
    status: str = "PENDING"  # PENDING | DONE | FAILED
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str = ""
    explanations: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


_JOBS: dict[str, ExplanationJobRecord] = {}


def _prune_jobs_locked(now_ts: float) -> None:
    stale_ids = [
        job_id
        for job_id, rec in _JOBS.items()
        if (rec.finished_at and (now_ts - rec.finished_at) > _MAX_JOB_AGE_SECONDS)
    ]
    for job_id in stale_ids:
        _JOBS.pop(job_id, None)

    if len(_JOBS) <= _MAX_STORED_JOBS:
        return

    # If still oversized, drop the oldest finished jobs first.
    finished = sorted(
        ((job_id, rec) for job_id, rec in _JOBS.items() if rec.finished_at is not None),
        key=lambda item: item[1].finished_at or item[1].created_at,
    )
    for job_id, _ in finished:
        _JOBS.pop(job_id, None)
        if len(_JOBS) <= _MAX_STORED_JOBS:
            break


def submit_explanation_job(
    fn: Callable[[], tuple[list[dict], int, int, float]],
) -> str:
    """Submit an explanation generation job and return a stable job id."""
    job_id = str(uuid.uuid4())
    now_ts = time.time()

    with _LOCK:
        _prune_jobs_locked(now_ts)
        _JOBS[job_id] = ExplanationJobRecord(status="PENDING", created_at=now_ts)

    def _runner() -> None:
        try:
            explanations, prompt_tokens, completion_tokens, cost_usd = fn()
            with _LOCK:
                rec = _JOBS.get(job_id)
                if rec is None:
                    return
                rec.status = "DONE"
                rec.finished_at = time.time()
                rec.explanations = explanations
                rec.prompt_tokens = int(prompt_tokens)
                rec.completion_tokens = int(completion_tokens)
                rec.cost_usd = float(cost_usd)
        except Exception as exc:
            with _LOCK:
                rec = _JOBS.get(job_id)
                if rec is None:
                    return
                rec.status = "FAILED"
                rec.finished_at = time.time()
                rec.error = str(exc)[:300]
            logger.warning(
                "rerank_async_job_failed",
                extra={"job_id": job_id, "error": str(exc)[:200]},
            )

    _EXECUTOR.submit(_runner)
    return job_id


def get_explanation_job(job_id: str) -> dict | None:
    """Read job snapshot by id (safe copy), or None when unknown."""
    if not job_id:
        return None
    with _LOCK:
        rec = _JOBS.get(job_id)
        if rec is None:
            return None
        return {
            "status": rec.status,
            "created_at": rec.created_at,
            "finished_at": rec.finished_at,
            "error": rec.error,
            "explanations": list(rec.explanations),
            "prompt_tokens": rec.prompt_tokens,
            "completion_tokens": rec.completion_tokens,
            "cost_usd": rec.cost_usd,
        }
