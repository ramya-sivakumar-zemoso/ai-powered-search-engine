"""Kaggle helper: run multilingual-e5-large embedding server behind ngrok.

This script starts an OpenAI-compatible `/v1/embeddings` endpoint using
`sentence-transformers` and exposes it publicly using ngrok.

Typical Kaggle usage:

1) pip install sentence-transformers fastapi uvicorn pyngrok
2) Set NGROK_AUTHTOKEN as a Kaggle secret (recommended)
3) python scripts/kaggle_e5_ngrok_server.py --port 8080 --prefix-mode passage

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
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel
from pyngrok import conf, ngrok
from sentence_transformers import SentenceTransformer
import uvicorn

logger = logging.getLogger("kaggle-e5-ngrok")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class EmbeddingsRequest(BaseModel):
    model: str | None = None
    input: str | list[str]
    encoding_format: str | None = "float"


def _load_ngrok_token() -> str:
    """Load ngrok token from env or Kaggle secrets."""
    token = (os.getenv("NGROK_AUTHTOKEN") or "").strip()
    if token:
        return token

    # Optional fallback for Kaggle notebooks using the secrets UI.
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        token = UserSecretsClient().get_secret("NGROK_AUTHTOKEN") or ""
        token = token.strip()
    except Exception:
        token = ""

    if not token:
        raise RuntimeError(
            "Missing NGROK_AUTHTOKEN. Set env var or Kaggle secret 'NGROK_AUTHTOKEN'."
        )
    return token


def _prefix_text(text: str, prefix_mode: str) -> str:
    t = text.strip()
    if prefix_mode == "none":
        return t
    if prefix_mode == "query":
        return f"query: {t}"
    return f"passage: {t}"


def build_app(model: SentenceTransformer, model_id: str, prefix_mode: str) -> FastAPI:
    app = FastAPI(title="Kaggle E5 Embedding Server", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model": model_id, "prefix_mode": prefix_mode}

    @app.post("/v1/embeddings")
    def embeddings(body: EmbeddingsRequest) -> dict[str, Any]:
        texts = body.input if isinstance(body.input, list) else [body.input]
        prefixed = [_prefix_text(t, prefix_mode=prefix_mode) for t in texts]
        vectors = model.encode(prefixed, normalize_embeddings=True)
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
        default="cpu",
        help="SentenceTransformer device (cpu, cuda).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = _load_ngrok_token()
    conf.get_default().auth_token = token

    logger.info("Loading model: %s on device=%s", args.model_id, args.device)
    model = SentenceTransformer(args.model_id, device=args.device)
    app = build_app(model=model, model_id=args.model_id, prefix_mode=args.prefix_mode)

    tunnel = ngrok.connect(addr=args.port, bind_tls=True)
    public_url = tunnel.public_url.rstrip("/")
    logger.info("ngrok public URL: %s", public_url)
    logger.info(
        "Use this in local .env: EMBEDDING_SERVER_URL=%s and EMBEDDER_SOURCE=self_hosted",
        public_url,
    )

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
