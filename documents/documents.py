import json
import requests
from config.settings import MEILI_URL, MASTER_HEADERS


def add_documents_from_file(index_name: str, filepath: str):
    """Load documents from a JSON file and add to index."""
    with open(filepath, "r") as f:
        documents = json.load(f)
    return add_documents(index_name, documents)


def add_documents(index_name: str, documents: list):
    """Add a list of document dicts to an index."""
    response = requests.post(
        f"{MEILI_URL}/indexes/{index_name}/documents",
        headers=MASTER_HEADERS,
        json=documents
    )
    response.raise_for_status()
    return response.json()


def get_documents(index_name: str, limit: int = 20, offset: int = 0):
    """Fetch documents from an index with pagination."""
    response = requests.get(
        f"{MEILI_URL}/indexes/{index_name}/documents",
        headers=MASTER_HEADERS,
        params={"limit": limit, "offset": offset}
    )
    response.raise_for_status()
    return response.json()


def get_document(index_name: str, document_id: str):
    """Fetch a single document by its ID."""
    response = requests.get(
        f"{MEILI_URL}/indexes/{index_name}/documents/{document_id}",
        headers=MASTER_HEADERS
    )
    response.raise_for_status()
    return response.json()


def delete_document(index_name: str, document_id: str):
    """Delete a single document by its ID."""
    response = requests.delete(
        f"{MEILI_URL}/indexes/{index_name}/documents/{document_id}",
        headers=MASTER_HEADERS
    )
    response.raise_for_status()
    return response.json()


def delete_all_documents(index_name: str):
    """Delete all documents in an index (keeps the index itself)."""
    response = requests.delete(
        f"{MEILI_URL}/indexes/{index_name}/documents",
        headers=MASTER_HEADERS
    )
    response.raise_for_status()
    return response.json()