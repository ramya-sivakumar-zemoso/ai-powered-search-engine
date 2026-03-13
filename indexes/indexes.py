import requests
from config.settings import MEILI_URL, MASTER_HEADERS


def create_index(index_name: str, primary_key: str = "id"):
    """Create a new index with a given primary key."""
    response = requests.post(
        f"{MEILI_URL}/indexes",
        headers=MASTER_HEADERS,
        json={"uid": index_name, "primaryKey": primary_key}
    )
    response.raise_for_status()
    return response.json()


def list_indexes():
    """List all indexes."""
    response = requests.get(
        f"{MEILI_URL}/indexes",
        headers=MASTER_HEADERS
    )
    response.raise_for_status()
    return response.json()


def get_index(index_name: str):
    """Get details of a specific index."""
    response = requests.get(
        f"{MEILI_URL}/indexes/{index_name}",
        headers=MASTER_HEADERS
    )
    response.raise_for_status()
    return response.json()


def delete_index(index_name: str):
    """Delete an index and all its documents."""
    response = requests.delete(
        f"{MEILI_URL}/indexes/{index_name}",
        headers=MASTER_HEADERS
    )
    response.raise_for_status()
    return response.json()