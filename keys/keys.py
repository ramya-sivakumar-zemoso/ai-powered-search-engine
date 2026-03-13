import requests
from config.settings import MEILI_URL, MASTER_HEADERS


def list_keys():
    """List all API keys. Requires master key."""
    response = requests.get(
        f"{MEILI_URL}/keys",
        headers=MASTER_HEADERS
    )
    response.raise_for_status()
    return response.json()


def get_key(key_uid: str):
    """Get a specific API key by its UID."""
    response = requests.get(
        f"{MEILI_URL}/keys/{key_uid}",
        headers=MASTER_HEADERS
    )
    response.raise_for_status()
    return response.json()


def create_key(name: str, actions: list, indexes: list, expires_at: str = None):
    """
    Create a new API key.
    actions: e.g. ["search", "documents.add"]
    indexes: e.g. ["movies"] or ["*"] for all
    expires_at: ISO 8601 string or None for no expiry
    """
    payload = {
        "name":      name,
        "actions":   actions,
        "indexes":   indexes,
        "expiresAt": expires_at
    }
    response = requests.post(
        f"{MEILI_URL}/keys",
        headers=MASTER_HEADERS,
        json=payload
    )
    response.raise_for_status()
    return response.json()


def delete_key(key_uid: str):
    """Delete an API key by its UID."""
    response = requests.delete(
        f"{MEILI_URL}/keys/{key_uid}",
        headers=MASTER_HEADERS
    )
    response.raise_for_status()
    return response.status_code