# AI-Powered Search Engine

Hybrid search on **Meilisearch** (keyword + semantic) with **paraphrase-multilingual-MiniLM-L12-v2** by default (`EMBEDDING_MODEL`, 384-d via Meilisearch’s built-in Hugging Face embedder). Swap models by changing `EMBEDDING_MODEL` and `EMBEDDING_DIMENSIONS` in `.env`, then re-index. The pipeline uses a **cross-encoder** reranker (`RERANKER_MODEL`, default `BAAI/bge-reranker-v2-m3`).

The codebase is **domain-agnostic**: movies, e-commerce, sports (and others) are configured by a **`DatasetSchema`** in `src/models/schema_registry.py`, selected with **`DATASET_SCHEMA`** in `.env`. Ingest maps raw columns → the shared internal document shape (`title`, `description`, `category`, `brand`, …) via **`FieldMapping`**. The pipeline (intent parsing, filters, retrieve fields, reranker text, Streamlit labels) reads that schema so behavior stays consistent across verticals.

**Architecture, PRD mapping, and trade-offs:** see **[DESIGN.md](DESIGN.md)**. **Trace correlation:** pass `--session-id` to `main.py` or use Streamlit (session id in app state); responses include `session_id` and `query_hash`.

**Prompt-injection defenses** (query sanitisation, delimiter boundaries for user vs vendor text in human messages only, structured detection logs) are described in **[DESIGN.md §8](DESIGN.md#8-prompt-injection-defenses)**.

## Requirements

- Python 3.10+ (see `pyproject.toml`; 3.12+ recommended)
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

After indexing, run a pipeline query (needs `OPENAI_API_KEY` for LLM nodes):

```bash
python main.py --query "science fiction adventure"
# or: streamlit run streamlit_app.py
```

## Embeddings and reranker models

Indexing uses Meilisearch’s **Hugging Face** embedder. Defaults are in `.env.example` (`EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`). Reranking uses **`RERANKER_MODEL`** (sentence-transformers `CrossEncoder`). Change either set of variables to point at another compatible Hugging Face id, then restart services and **re-index** after embedding model changes:

Re-index after changing the embedding model or dimensions:

```bash
python -m src.tools.setup_index --reset --schema movies
```

## Response flags

Every pipeline response includes the following top-level flags for downstream monitoring:

| Flag | Type | Description |
|---|---|---|
| `partial_results` | `bool` | `true` when hybrid retrieval degraded to keyword-only fallback |
| `rerank_degraded` | `bool` | `true` when cross-encoder failed or produced out-of-range confidence; native Meilisearch ranking was used instead |
| `structured_text` | `str` | Human-readable text summary of results for logging / non-JSON consumers |

Quality scores returned per query also include:

- `per_result_relevance` — `{id: score}` mapping of raw Meilisearch scores
- `per_result_rerank_confidence` — `{id: confidence}` mapping of cross-encoder confidence per result
- `weights_used` — the actual evaluator signal weights applied (from the active `DatasetSchema`)

## Evaluator signal weights

Each `DatasetSchema` carries `evaluator_signal_weights` (normalised at runtime) to tune how much each signal influences the combined quality score:

```python
evaluator_signal_weights = {
    "semantic_relevance": 0.42,   # query–result embedding similarity
    "result_coverage":    0.31,   # fraction of results above score threshold
    "ranking_stability":  0.14,   # rank order consistency across retries
    "freshness_signal":   0.13,   # recency of indexed documents
}
```

Override in `schema_registry.py` per domain (marketplace, sports, movies, …).

## Kafka streaming ingest (recommended for live data)

The primary real-time path uses **Kafka / Redpanda** as a durable event buffer. The consumer commits Kafka offsets only after Meilisearch confirms the document is indexed — guaranteeing at-least-once delivery end-to-end.

### Quick start (local — Redpanda in Docker)

```bash
# 1. Start Redpanda (Kafka-compatible, zero config)
docker run -d -p 9092:9092 --name redpanda \
  redpandadata/redpanda:latest redpanda start \
  --overprovisioned --smp 1 --memory 1G --reserve-memory 0M \
  --node-id 0 --check=false --kafka-addr 0.0.0.0:9092

# 2. Install the Kafka client
pip install confluent-kafka

# 3. Start the consumer (terminal 1)
KAFKA_ENABLED=true python -m src.tools.kafka_consumer

# 4. Publish a listing event (terminal 2)
KAFKA_ENABLED=true python -m src.tools.kafka_producer \
  --schema marketplace '{"id": "1", "title": "Widget X", "category": "Electronics"}'
```

### Managed broker (Redpanda Serverless / Confluent Cloud)

```bash
KAFKA_ENABLED=true \
KAFKA_BOOTSTRAP_SERVERS=seed.redpanda.cloud:9092 \
KAFKA_SECURITY_PROTOCOL=SASL_SSL \
KAFKA_SASL_USERNAME=your-username \
KAFKA_SASL_PASSWORD=your-password \
python -m src.tools.kafka_consumer
```

### Publish from Python

```python
from src.tools.kafka_producer import SearchIngestProducer

with SearchIngestProducer() as producer:
    producer.publish({"id": "123", "title": "Air Max 90", "category": "Shoes"}, schema_name="marketplace")
    producer.publish_tombstone("123")   # soft-delete
```

**Key design choices:**
- Partition key = document `id` → all updates for the same listing hit the same partition (ordered)
- `schema_name` header routes each message to the correct `DatasetSchema` — one topic serves all domains
- Offset committed **after** Meilisearch ACK — safe replay if Meilisearch is temporarily down

## Real-time ingest (5-minute SLA)

The batch `setup_index.py` path is for initial data load. For live listings use the **ingest API**:

```bash
pip install fastapi uvicorn
uvicorn src.tools.ingest_api:app --host 0.0.0.0 --port 8001
```

| Endpoint | Purpose |
|---|---|
| `POST /ingest/document` | Upsert a single document; blocks until Meilisearch indexes it (HTTP 408 on SLA breach) |
| `POST /ingest/batch` | Upsert up to 100 documents in one call |
| `DELETE /ingest/{id}` | Hard-delete a document |
| `GET /ingest/health` | Readiness probe + SLA config |

Every response includes `elapsed_seconds` and `sla_ok` for monitoring. The endpoint is idempotent — re-posting the same `id` performs an in-place update (safe for at-least-once streams).

**Soft-delete (tombstone):** include `deleted_at` in the payload and filter `deleted_at IS NULL` at query time.

## Latency modes

| Mode | End-to-end P95 | Trade-off |
|---|---|---|
| Default (`FAST_MODE=false`) | ~2 700 ms | Full AI quality with LLM explanations |
| Fast mode (`FAST_MODE=true`) | ~1 100 ms | Skips LLM explanations; cross-encoder ranking still runs |
| Meilisearch-only | ~80 ms | No AI enrichment; raw keyword/hybrid results |

Every response includes `pipeline_latency_ms` for SLA tracking. See `DESIGN.md §13` for the full latency breakdown.

## Graph diagram

Export a Mermaid + SVG diagram of the pipeline topology:

```bash
python scripts/export_graph_diagram.py
# outputs: src/graph/graph_topology.mmd  and  src/graph/graph_programmatic.svg
```

## Layout

- `src/tools/setup_index.py` — index lifecycle + batch ingest
- `src/tools/meilisearch_client.py` — search, stats, tasks (REST)
- `src/models/schema_registry.py` — dataset schemas (`movies`, `marketplace`, `sports`, …)
- `src/models/dataset_schema.py` — `FieldMapping`, transforms, `meilisearch_attributes_to_retrieve()`
- `src/tools/dataset_loader.py` — load files → normalise via schema
- `scripts/export_graph_diagram.py` — programmatic LangGraph topology export

### Adding a new domain

1. Copy an existing schema in `schema_registry.py` and adjust `field_mappings` (source column names), `searchable_fields`, `filterable_fields`, `filter_aliases_llm`, `embedder_template`, and UI fields (`ui_*`, `demo_queries`).
2. Register it in `SCHEMA_REGISTRY`.
3. Set `DATASET_SCHEMA` and `MEILI_INDEX_NAME` (and `DATASET_FILE`) in `.env`, then run `python -m src.tools.setup_index`.
