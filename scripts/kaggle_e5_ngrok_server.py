"""Kaggle helper: run multilingual-e5-large embedding server behind ngrok.

This script starts an OpenAI-compatible `/v1/embeddings` endpoint using
`sentence-transformers` and exposes it publicly using ngrok.

The server auto-detects CUDA GPUs and uses them by default. On a Kaggle T4
with batch_size=64, expect ~200-500 docs/sec vs ~10 docs/sec on CPU.

Typical Kaggle usage:

1) pip install sentence-transformers fastapi uvicorn pyngrok torch
2) Set NGROK_AUTHTOKEN (or NGROK_AUTH_TOKEN) as a Kaggle secret (recommended)
3) python scripts/kaggle_e5_ngrok_server.py --port 8080 --prefix-mode passage

GPU is auto-detected. Override with --device cpu/cuda. Tune throughput with
--batch-size (default 64; increase to 128/256 if VRAM allows).

Then set local project `.env` values to point Meilisearch to the printed ngrok URL:
  EMBEDDER_SOURCE=self_hosted
  EMBEDDING_SERVER_URL=https://<your-ngrok-domain>
  EMBEDDING_MODEL=intfloat/multilingual-e5-large
  EMBEDDING_DIMENSIONS=1024
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from pyngrok import conf, ngrok
from sentence_transformers import SentenceTransformer
import uvicorn
from dotenv import load_dotenv

logger = logging.getLogger("kaggle-e5-ngrok")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class EmbeddingsRequest(BaseModel):
    model: str | None = None
    input: str | list[str]
    encoding_format: str | None = "float"


def _load_ngrok_token() -> str:
    """Load ngrok token from env or Kaggle secrets."""
    load_dotenv()
    env_keys = ("NGROK_AUTHTOKEN", "NGROK_AUTH_TOKEN")
    for key in env_keys:
        token = (os.getenv(key) or "").strip()
        if token:
            return token

    # Optional fallback for Kaggle notebooks using the secrets UI.
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        client = UserSecretsClient()
        for key in env_keys:
            token = (client.get_secret(key) or "").strip()
            if token:
                return token
    except Exception:
        pass

    raise RuntimeError(
        "Missing ngrok token. Set NGROK_AUTHTOKEN or NGROK_AUTH_TOKEN "
        "(env var or Kaggle secret)."
    )


def _prefix_text(text: str, prefix_mode: str) -> str:
    t = text.strip()
    if prefix_mode == "none":
        return t
    if prefix_mode == "query":
        return f"query: {t}"
    return f"passage: {t}"


def build_app(
    model: SentenceTransformer,
    model_id: str,
    prefix_mode: str,
    batch_size: int,
) -> FastAPI:
    app = FastAPI(title="Kaggle E5 Embedding Server", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": model_id,
            "prefix_mode": prefix_mode,
            "device": str(model.device),
            "batch_size": batch_size,
        }

    @app.post("/v1/embeddings")
    def embeddings(body: EmbeddingsRequest) -> dict[str, Any]:
        texts = body.input if isinstance(body.input, list) else [body.input]
        prefixed = [_prefix_text(t, prefix_mode=prefix_mode) for t in texts]

        t0 = time.perf_counter()
        vectors = model.encode(
            prefixed,
            normalize_embeddings=True,
            batch_size=batch_size,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        docs_per_sec = len(texts) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
        logger.info(
            "Encoded %d texts in %.1fms (%.1f docs/sec)",
            len(texts), elapsed_ms, docs_per_sec,
        )

        data = [
            {
                "object": "embedding",
                "embedding": vec.tolist(),
                "index": idx,
            }
            for idx, vec in enumerate(vectors)
        ]
        return {
            "object": "list",
            "data": data,
            "model": body.model or model_id,
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run intfloat/multilingual-e5-large server in Kaggle + ngrok."
    )
    parser.add_argument("--model-id", default="intfloat/multilingual-e5-large")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--prefix-mode",
        choices=["passage", "query", "none"],
        default="passage",
        help="Prefix strategy for E5-compatible embeddings.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="SentenceTransformer device. Auto-detects CUDA when omitted.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for model.encode(). Higher = faster on GPU, more VRAM.",
    )
    parser.add_argument(
        "--print-env",
        action="store_true",
        help="Print .env lines for this ngrok endpoint and exit.",
    )
    return parser.parse_args()


def _resolve_device(requested: str | None) -> str:
    """Return the best available device, preferring CUDA when available."""
    if requested is not None:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _log_device_info(device: str) -> None:
    """Log hardware info so the operator can verify GPU is actually in use."""
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("GPU detected: %s (%.1f GB VRAM)", gpu_name, vram_gb)
    else:
        import multiprocessing
        logger.info(
            "Running on CPU (%d cores). Pass --device cuda for GPU acceleration.",
            multiprocessing.cpu_count(),
        )


def main() -> None:
    args = parse_args()
    token = _load_ngrok_token()
    conf.get_default().auth_token = token

    device = _resolve_device(args.device)
    _log_device_info(device)

    logger.info("Loading model: %s on device=%s (batch_size=%d)", args.model_id, device, args.batch_size)
    model = SentenceTransformer(args.model_id, device=device)

    # Warm up the model with a dummy encode so the first real request isn't slow.
    model.encode(["warmup"], batch_size=1)
    logger.info("Model warm-up complete")

    app = build_app(
        model=model,
        model_id=args.model_id,
        prefix_mode=args.prefix_mode,
        batch_size=args.batch_size,
    )

    tunnel = ngrok.connect(addr=args.port, bind_tls=True)
    public_url = tunnel.public_url.rstrip("/")
    logger.info("ngrok public URL: %s", public_url)
    logger.info(
        "Use this in local .env: EMBEDDING_SERVER_URL=%s and EMBEDDER_SOURCE=self_hosted",
        public_url,
    )
    if args.print_env:
        print("\n# Copy to local .env")
        print("EMBEDDER_SOURCE=self_hosted")
        print(f"EMBEDDING_SERVER_URL={public_url}")
        print(f"EMBEDDING_MODEL={args.model_id}")
        print("EMBEDDING_DIMENSIONS=1024")

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
