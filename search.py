"""Search: keyword, hybrid, semantic, and filtered search."""
import requests

from config import MEILI_URL, MASTER_HEADERS


def keyword_search(index_name: str, query: str, limit: int = 10):
    """
    Standard keyword search.
    Fast, no embedder required.
    """
    response = requests.post(
        f"{MEILI_URL}/indexes/{index_name}/search",
        headers=MASTER_HEADERS,
        json={"q": query, "limit": limit},
    )
    response.raise_for_status()
    return response.json()


def hybrid_search(
    index_name: str,
    query: str,
    embedder_name: str,
    semantic_ratio: float = 0.5,
    limit: int = 10,
):
    """
    Hybrid search combining keyword + vector search.
    semantic_ratio: 0.0 = pure keyword, 1.0 = pure vector, 0.5 = balanced
    Requires an embedder to be configured on the index first.
    """
    response = requests.post(
        f"{MEILI_URL}/indexes/{index_name}/search",
        headers=MASTER_HEADERS,
        json={
            "q": query,
            "limit": limit,
            "hybrid": {
                "embedder": embedder_name,
                "semanticRatio": semantic_ratio,
            },
        },
    )
    response.raise_for_status()
    return response.json()


def semantic_search(
    index_name: str,
    query: str,
    embedder_name: str,
    limit: int = 10,
):
    """
    Pure vector/semantic search (no keyword matching).
    Wraps hybrid_search with semanticRatio=1.0.
    """
    return hybrid_search(
        index_name, query, embedder_name,
        semantic_ratio=1.0, limit=limit,
    )


def filtered_search(
    index_name: str,
    query: str,
    filters: str,
    limit: int = 10,
):
    """
    Keyword search with filters.
    filters: MeiliSearch filter expression e.g. "genres = Action AND release_date > 2020"
    """
    response = requests.post(
        f"{MEILI_URL}/indexes/{index_name}/search",
        headers=MASTER_HEADERS,
        json={"q": query, "filter": filters, "limit": limit},
    )
    response.raise_for_status()
    return response.json()
