"""Setup: create index, load documents, configure embedder, and task utilities."""
import json
import time
import requests

from config import MEILI_URL, MASTER_HEADERS, OPENAI_KEY


def create_index(index_name: str, primary_key: str = "id"):
    """Create a new index with a given primary key."""
    response = requests.post(
        f"{MEILI_URL}/indexes",
        headers=MASTER_HEADERS,
        json={"uid": index_name, "primaryKey": primary_key},
    )
    response.raise_for_status()
    return response.json()


def add_documents(index_name: str, documents: list):
    """Add a list of document dicts to an index."""
    response = requests.post(
        f"{MEILI_URL}/indexes/{index_name}/documents",
        headers=MASTER_HEADERS,
        json=documents,
    )
    response.raise_for_status()
    return response.json()


def add_documents_from_file(index_name: str, filepath: str):
    """Load documents from a JSON file and add to index."""
    with open(filepath, "r") as f:
        documents = json.load(f)
    return add_documents(index_name, documents)


def create_openai_embedder(
    index_name: str,
    embedder_name: str,
    document_template: str,
    model: str = "text-embedding-ada-002",
):
    """
    Create an OpenAI embedder on a given index.
    document_template uses Liquid syntax e.g.:
      "A movie titled {{doc.title}} about {{doc.overview}}"
    """
    response = requests.patch(
        f"{MEILI_URL}/indexes/{index_name}/settings/embedders",
        headers=MASTER_HEADERS,
        json={
            embedder_name: {
                "source": "openAi",
                "apiKey": OPENAI_KEY,
                "model": model,
                "documentTemplate": document_template,
            }
        },
    )
    response.raise_for_status()
    return response.json()


def list_keys():
    """List all API keys. Requires master key."""
    response = requests.get(
        f"{MEILI_URL}/keys",
        headers=MASTER_HEADERS,
    )
    response.raise_for_status()
    return response.json()


def check_task(task_uid: int):
    """Get the current status of a task by its UID."""
    response = requests.get(
        f"{MEILI_URL}/tasks/{task_uid}",
        headers=MASTER_HEADERS,
    )
    response.raise_for_status()
    return response.json()


def wait_for_task(task_uid: int, interval: int = 5, timeout: int = 1800):
    """
    Poll a task until it succeeds or fails.
    interval: seconds between polls (default 5)
    timeout: max seconds to wait (default 30 min)
    Raises TimeoutError if task doesn't complete in time.
    """
    elapsed = 0
    while elapsed < timeout:
        task = check_task(task_uid)
        status = task["status"]
        print(f"[Task {task_uid}] status: {status} ({elapsed}s elapsed)")

        if status == "succeeded":
            print(f"[Task {task_uid}] completed successfully.")
            return task

        if status == "failed":
            error = task.get("error", {}).get("message", "Unknown error")
            raise RuntimeError(f"[Task {task_uid}] failed: {error}")

        time.sleep(interval)
        elapsed += interval

    raise TimeoutError(f"[Task {task_uid}] timed out after {timeout}s")


# Default index and embedder names (must match main.py)
INDEX_NAME = "movies"
EMBEDDER_NAME = "movies-openai-embedder-2"


def setup():
    """Full setup: create index, load documents, configure embedder. Run once before using main.py."""
    print("--- Creating index ---")
    result = create_index(INDEX_NAME, primary_key="id")
    print(result)

    print("\n--- Adding documents ---")
    task = add_documents_from_file(INDEX_NAME, "movies.json")
    wait_for_task(task["taskUid"])

    print("\n--- Creating embedder ---")
    task = create_openai_embedder(
        index_name=INDEX_NAME,
        embedder_name=EMBEDDER_NAME,
        document_template="A movie titled {{doc.title}} about {{doc.overview}}",
    )
    print("Waiting for embeddings to generate (this may take a while)...")
    wait_for_task(task["taskUid"], interval=10)
    print("Setup complete.")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    setup()
