"""AI-powered search engine: run search operations (run setup.py first if needed)."""
from dotenv import load_dotenv

load_dotenv()

from search import hybrid_search, keyword_search

# Must match INDEX_NAME and EMBEDDER_NAME in setup.py
INDEX_NAME = "movies"
EMBEDDER_NAME = "movies-openai-embedder-2"


def run_search(query: str):
    """Run a hybrid search and print results."""
    print(f"\n--- Hybrid search: '{query}' ---")
    results = hybrid_search(INDEX_NAME, query, EMBEDDER_NAME)
    for hit in results["hits"]:
        print(f"  {hit['title']} ({hit.get('release_date', 'N/A')})")


if __name__ == "__main__":
    run_search("animated movies for kids")
    # run_search("space exploration sci-fi")
