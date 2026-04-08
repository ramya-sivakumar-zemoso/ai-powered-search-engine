#!/usr/bin/env python3
"""End-to-end check: POST to the ingest API, then confirm the doc exists in Meilisearch.

Prerequisites
-------------
- Meilisearch running with index already created (e.g. ``python -m src.tools.setup_index`` once).
- Ingest API: ``uvicorn src.tools.ingest_api:app --host 0.0.0.0 --port 8001``
- Optional: ``pip install -e ".[streaming]"`` for FastAPI/uvicorn (server side only).

Environment
-----------
- ``INGEST_BASE_URL`` — default ``http://127.0.0.1:8001``
- ``MEILI_*``, ``DATASET_SCHEMA`` — same as the rest of the app (via ``.env``)

Usage
-----
.. code-block:: bash

   uvicorn src.tools.ingest_api:app --port 8001   # terminal 1
   python scripts/verify_ingest_sla.py           # terminal 2

Exit code 0 on success; 1 on failure.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools.meilisearch_client import delete_document  # noqa: E402
from src.utils.config import get_settings  # noqa: E402


def _poll_get_document(
    client: object,
    index_name: str,
    doc_id: str,
    *,
    max_wait_s: float,
    interval_s: float,
) -> bool:
    deadline = time.monotonic() + max_wait_s
    idx = client.index(index_name)
    while time.monotonic() < deadline:
        try:
            idx.get_document(doc_id)
            return True
        except Exception:
            time.sleep(interval_s)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("Prerequisites")[0].strip())
    parser.add_argument(
        "--ingest-url",
        default=os.getenv("INGEST_BASE_URL", "http://127.0.0.1:8001"),
        help="Base URL of the ingest FastAPI app (no trailing slash).",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Leave the test document in the index after success.",
    )
    args = parser.parse_args()
    settings = get_settings()

    health_r = requests.get(f"{args.ingest_url.rstrip('/')}/ingest/health", timeout=30)
    if health_r.status_code != 200:
        print(f"Ingest health failed: HTTP {health_r.status_code} {health_r.text}", file=sys.stderr)
        return 1
    health = health_r.json()
    print("ingest health:", health.get("status"), "sla_seconds=", health.get("sla_seconds"))

    doc_id = f"verify-ingest-{uuid.uuid4().hex[:12]}"
    unique_title = f"IngestVerify {uuid.uuid4().hex[:8]}"
    body = {
        "id": doc_id,
        "payload": {
            "title": unique_title,
            "overview": "Synthetic document from scripts/verify_ingest_sla.py.",
            "genres": ["VerifyIngest"],
            "poster": "",
            "release_date": 1_700_000_000,
        },
    }

    post_timeout = max(settings.ingest_sla_seconds + 60, 120)
    t0 = time.perf_counter()
    ingest_r = requests.post(
        f"{args.ingest_url.rstrip('/')}/ingest/document",
        json=body,
        timeout=post_timeout,
    )
    elapsed_ingest = time.perf_counter() - t0

    if ingest_r.status_code != 200:
        print(
            f"Ingest POST failed: HTTP {ingest_r.status_code} {ingest_r.text}",
            file=sys.stderr,
        )
        return 1

    payload = ingest_r.json()
    print(
        "ingest response:",
        "status=",
        payload.get("status"),
        "elapsed_seconds=",
        payload.get("elapsed_seconds"),
        "sla_ok=",
        payload.get("sla_ok"),
        "http_client_s=",
        round(elapsed_ingest, 3),
    )

    import meilisearch

    client = meilisearch.Client(settings.meili_url, settings.meili_master_key)
    visible = _poll_get_document(
        client,
        settings.meili_index_name,
        doc_id,
        max_wait_s=60.0,
        interval_s=0.5,
    )
    if not visible:
        print("Document not retrievable from Meilisearch after ingest success.", file=sys.stderr)
        return 1

    print("Meilisearch get_document OK for id=", doc_id)

    if not args.no_cleanup:
        try:
            delete_document(doc_id, wait=True)
            print("Cleaned up test document id=", doc_id)
        except Exception as exc:
            print(f"Cleanup failed (remove id={doc_id} manually): {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
