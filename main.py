from dotenv import load_dotenv
load_dotenv()

from indexes.indexes       import create_index, get_index, list_indexes
from documents.documents   import add_documents_from_file
from embedders.embedders   import create_openai_embedder, get_embedders
from search.search         import hybrid_search, keyword_search
from tasks.tasks           import wait_for_task
from keys.keys             import list_keys

INDEX_NAME    = "movies"
EMBEDDER_NAME = "movies-openai-embedder-2"


def setup():
    """Full setup: create index, load documents, configure embedder."""
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
        document_template="A movie titled {{doc.title}} about {{doc.overview}}"
    )
    print("Waiting for embeddings to generate (this may take a while)...")
    wait_for_task(task["taskUid"], interval=10)
    print("Setup complete.")


def run_search(query: str):
    """Run a hybrid search and print results."""
    print(f"\n--- Hybrid search: '{query}' ---")
    results = hybrid_search(INDEX_NAME, query, EMBEDDER_NAME)
    for hit in results["hits"]:
        print(f"  {hit['title']} ({hit.get('release_date', 'N/A')})")


if __name__ == "__main__":
    # Uncomment to run full setup (only needed once):
    #setup()

    run_search("animated movies for kids")
    run_search("space exploration sci-fi")