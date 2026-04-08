"""Unit tests for Gemini → Meilisearch REST embedder payload."""

from src.utils.gemini_meili_embedder import build_gemini_batch_rest_embedder


def test_build_gemini_rest_embedder_default():
    cfg = build_gemini_batch_rest_embedder(
        embedder_name="default",
        api_key="test-key",
        dimensions=3072,
        document_template="{{doc.title}}",
        model_id="gemini-embedding-001",
        task_type=None,
    )
    body = cfg["default"]
    assert body["source"] == "rest"
    assert body.get("apiKey") is None
    assert body["url"].endswith("models/gemini-embedding-001:batchEmbedContents")
    assert body["dimensions"] == 3072
    assert body["headers"]["x-goog-api-key"] == "test-key"
    req0 = body["request"]["requests"][0]
    assert req0["model"] == "models/gemini-embedding-001"
    assert req0["content"]["parts"][0]["text"] == "{{text}}"
    assert req0["outputDimensionality"] == 3072
    assert "taskType" not in req0
    assert body["request"]["requests"][1] == "{{..}}"


def test_build_gemini_rest_embedder_strips_models_prefix_and_task_type():
    cfg = build_gemini_batch_rest_embedder(
        embedder_name="g",
        api_key="k",
        dimensions=768,
        document_template="x",
        model_id="models/gemini-embedding-001",
        task_type="RETRIEVAL_DOCUMENT",
    )
    req0 = cfg["g"]["request"]["requests"][0]
    assert req0["taskType"] == "RETRIEVAL_DOCUMENT"
    assert req0["outputDimensionality"] == 768
