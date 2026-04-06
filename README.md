# AI-Powered Search Engine

Hybrid search on **Meilisearch** (keyword + semantic) with **multilingual E5** embeddings (`intfloat/multilingual-e5-large` via `EMBEDDING_MODEL`) on the index, and a **BGE** cross-encoder reranker (`BAAI/bge-reranker-v2-m3` via `RERANKER_MODEL`) in the pipeline.

## Requirements

- Python 3.12+
- Meilisearch (see `.env.example` for `MEILI_*` vars)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # or: pip install -e .
cp .env.example .env                # then edit secrets
```

Start Meilisearch (example):

```bash
docker run -d -p 7700:7700 -e MEILI_MASTER_KEY=your_key getmeili/meilisearch:latest
```

## Index data (production-style)

Create/configure the index, load documents, and wait for embeddings:

```bash
python -m src.tools.setup_index
python -m src.tools.setup_index --file data/movies.json --schema movies --limit 1000
```

Optional env defaults: `DATASET_FILE`, `DATASET_SCHEMA` (see `.env.example`).

## Verify

```bash
python scripts/verify_search.py --schema movies
```

## Layout

- `src/tools/setup_index.py` — index lifecycle + batch ingest
- `src/tools/meilisearch_client.py` — search, stats, tasks (REST)
- `src/models/schema_registry.py` — dataset schemas
- `src/tools/dataset_loader.py` — load files → normalise via schema
