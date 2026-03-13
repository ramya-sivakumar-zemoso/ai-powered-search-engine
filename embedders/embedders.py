import requests
from config.settings import MEILI_URL, MASTER_HEADERS, OPENAI_KEY


def create_openai_embedder(
    index_name: str,
    embedder_name: str,
    document_template: str,
    model: str = "text-embedding-ada-002"
):
    """
    Create an OpenAI embedder on a given index.
    document_template uses Liquid syntax e.g:
      "A movie titled {{doc.title}} about {{doc.overview}}"
    """
    response = requests.patch(
        f"{MEILI_URL}/indexes/{index_name}/settings/embedders",
        headers=MASTER_HEADERS,
        json={
            embedder_name: {
                "source":           "openAi",
                "apiKey":           OPENAI_KEY,
                "model":            model,
                "documentTemplate": document_template
            }
        }
    )
    response.raise_for_status()
    return response.json()


def get_embedders(index_name: str):
    """Get all embedder configurations for an index."""
    response = requests.get(
        f"{MEILI_URL}/indexes/{index_name}/settings/embedders",
        headers=MASTER_HEADERS
    )
    response.raise_for_status()
    return response.json()


def delete_embedder(index_name: str, embedder_name: str):
    """Remove an embedder from an index by setting it to null."""
    response = requests.patch(
        f"{MEILI_URL}/indexes/{index_name}/settings/embedders",
        headers=MASTER_HEADERS,
        json={embedder_name: None}
    )
    response.raise_for_status()
    return response.json()