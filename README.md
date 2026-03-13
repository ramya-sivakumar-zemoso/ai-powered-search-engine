# AI-Powered Search Engine

A simple search engine using **Meilisearch** and **OpenAI embeddings** for hybrid (keyword + semantic) search.

## Requirements

- Python 3
- [Meilisearch](https://www.meilisearch.com/) installed and running (default: `http://127.0.0.1:7700`)
- OpenAI API key

## Setup

1. **Install and run Meilisearch**:

   **Linux / macOS (binary):**
   ```bash
   curl -L https://install.meilisearch.com | sh
   sudo mv meilisearch /usr/local/bin/
   meilisearch --master-key="aSampleMasterKey"
   ```

   **Docker:**
   ```bash
   docker run -d -p 7700:7700 getmeili/meilisearch:latest
   ```

   Use the same `--master-key` (or `MEILI_MASTER_KEY` in Docker) as in your `.env`.

2. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with:

   - `OPENAI_API_KEY` – your OpenAI API key (required for embeddings)
   - `MEILI_URL` – (optional) Meilisearch URL, default `http://127.0.0.1:7700`
   - `MEILI_MASTER_KEY` – (optional) Meilisearch master key, default `aSampleMasterKey`

4. With Meilisearch running, start the app:

   ```bash
   python main.py
   ```

   The first run can call `setup()` in `main.py` to create the index, load `movies.json`, and configure the OpenAI embedder. After that, comment out `setup()` and use `run_search("your query")` to search.

## Usage

- **Setup (once):** Uncomment `setup()` in `main.py` to create the index, add documents, and create the embedder.
- **Search:** Use `run_search("your query")` for hybrid search results.
