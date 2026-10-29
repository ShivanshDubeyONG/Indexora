"""Tests for semantic search pipeline and integration (Phase 12B).

All tests are completely offline by default, using mock deterministic embedding functions
so that the test suite never performs network requests or downloads pretrained model weights.
"""

from __future__ import annotations

import hashlib
import numpy as np
import pytest

from indexora.semantic import (
    DEFAULT_EMBEDDING_MODEL,
    Document,
    SemanticSearchEngine,
    SemanticSearchResult,
    SentenceTransformersNotAvailableError,
    build_semantic_engine,
    encode_texts,
    get_embedding_model,
    is_sentence_transformers_available,
    load_demo_corpus,
)


def _mock_embedder(texts: list[str] | str, dim: int = 16) -> np.ndarray:
    """Deterministic offline mock embedder producing unit-normalized vectors."""
    is_single = isinstance(texts, str)
    items = [texts] if is_single else list(texts)
    out = np.zeros((len(items), dim), dtype=np.float32)
    for i, t in enumerate(items):
        digest = hashlib.sha256(t.encode("utf-8")).hexdigest()
        seed = int(digest[:8], 16)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(dim).astype(np.float32)
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        out[i] = v
    return out[0] if is_single else out


class MockModel:
    """Mock sentence-transformers model wrapper."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def encode(
        self,
        sentences: list[str] | str,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        return _mock_embedder(sentences, dim=self.dim)


# ==============================================================================
# Document & Corpus Tests
# ==============================================================================


def test_document_dataclass():
    doc = Document(
        doc_id=42,
        title="Test Document",
        text="This is a test body.",
        category="Testing",
        metadata={"author": "Indexora"},
    )
    assert doc.doc_id == 42
    assert doc.title == "Test Document"
    assert doc.text == "This is a test body."
    assert doc.category == "Testing"
    assert doc.metadata == {"author": "Indexora"}

    # Frozen dataclass immutability
    with pytest.raises(Exception):
        doc.title = "New Title"  # type: ignore


def test_load_demo_corpus():
    docs = load_demo_corpus()
    assert len(docs) == 25, "Expected 25 documents in the curated demo corpus"

    doc_ids = [d.doc_id for d in docs]
    assert len(set(doc_ids)) == len(docs), "All document IDs must be unique"

    categories = {d.category for d in docs}
    expected_categories = {
        "Machine Learning",
        "Databases",
        "Distributed Systems",
        "Cybersecurity",
        "Software Engineering",
    }
    assert categories == expected_categories

    for d in docs:
        assert isinstance(d.doc_id, int)
        assert len(d.title.strip()) > 0
        assert len(d.text.strip()) > 20
        assert "tags" in d.metadata
        assert len(d.metadata["tags"]) > 0


# ==============================================================================
# Embedding Encoding Tests
# ==============================================================================


def test_encode_texts_with_callable_and_mock_model():
    mock_model = MockModel(dim=16)

    # Single text with callable
    single_vec = encode_texts("Machine learning", model=mock_model.encode, normalize=True)
    assert isinstance(single_vec, np.ndarray)
    assert single_vec.shape == (16,)
    assert single_vec.dtype == np.float32
    assert np.isclose(np.linalg.norm(single_vec), 1.0, atol=1e-5)

    # Batch texts with mock model instance
    batch_texts = ["First query", "Second query", "Third query"]
    batch_vecs = encode_texts(batch_texts, model=mock_model, normalize=True)
    assert isinstance(batch_vecs, np.ndarray)
    assert batch_vecs.shape == (3, 16)
    assert batch_vecs.dtype == np.float32
    for i in range(3):
        assert np.isclose(np.linalg.norm(batch_vecs[i]), 1.0, atol=1e-5)


def test_encode_texts_missing_model_raises(monkeypatch):
    """When sentence-transformers is not available and no model is passed, raises helpful error."""
    import indexora.semantic as sem

    monkeypatch.setattr(sem, "is_sentence_transformers_available", lambda: False)
    with pytest.raises(SentenceTransformersNotAvailableError) as exc_info:
        encode_texts("some text", model=None)
    assert "pip install" in str(exc_info.value)


# ==============================================================================
# SemanticSearchEngine Tests
# ==============================================================================


@pytest.mark.parametrize("index_type", ["Exact BLAS", "Exact Flat (Ref)", "IVF-Flat", "HNSW"])
def test_semantic_search_engine_build_and_query(index_type: str):
    docs = load_demo_corpus()[:10]  # first 10 docs for rapid test
    mock_model = MockModel(dim=16)

    engine = build_semantic_engine(
        corpus=docs,
        index_type=index_type,
        model=mock_model,
        ivf_nlist=2,
        ivf_nprobe=2,
        hnsw_m=4,
        hnsw_ef_construction=16,
        hnsw_ef_search=16,
    )

    assert len(engine) == 10
    assert engine.dimension == 16
    assert engine.index_type == index_type

    # Perform query
    res = engine.search("LSM-tree write amplification", k=3)
    assert isinstance(res, SemanticSearchResult)
    assert res.query == "LSM-tree write amplification"
    assert len(res.retrieved_documents) == 3
    assert len(res.distances) == 3
    assert res.latency_ms >= 0.0
    assert res.embed_time_ms >= 0.0

    # Distances must be non-negative squared-L2 distances
    for d in res.distances:
        assert d >= -1e-6
        # Cosine distance for unit vectors is between 0 and 4
        assert d <= 4.0 + 1e-6

    # Test cosine similarity conversion
    cos_sims = res.cosine_similarities
    assert len(cos_sims) == 3
    for s in cos_sims:
        assert -1.0 - 1e-5 <= s <= 1.0 + 1e-5

    # Ascending distance corresponds to descending similarity
    for i in range(len(res.distances) - 1):
        assert res.distances[i] <= res.distances[i + 1] + 1e-6
        assert cos_sims[i] >= cos_sims[i + 1] - 1e-6


def test_semantic_search_filtering():
    docs = load_demo_corpus()
    mock_model = MockModel(dim=16)

    engine = build_semantic_engine(
        corpus=docs,
        index_type="Exact BLAS",
        model=mock_model,
    )

    # 1. Filter by category
    target_category = "Databases"
    res_cat = engine.search(
        "btree and indexing",
        k=5,
        filter_category=target_category,
    )
    assert len(res_cat.retrieved_documents) == 5
    for doc in res_cat.retrieved_documents:
        assert doc.category == target_category

    # 2. Filter by explicit callable
    res_callable = engine.search(
        "distributed consensus",
        k=5,
        filter_ids=lambda doc_id: doc_id % 2 == 0,
    )
    for doc in res_callable.retrieved_documents:
        assert doc.doc_id % 2 == 0

    # 3. Filter with set of IDs
    allowed_ids = {101, 102, 103}
    res_set = engine.search("transformers", k=5, filter_ids=allowed_ids)
    assert len(res_set.retrieved_documents) == 3
    for doc in res_set.retrieved_documents:
        assert doc.doc_id in allowed_ids

    # 4. Filter resulting in empty eligible set
    res_empty = engine.search("test", k=5, filter_ids={999999})
    assert len(res_empty.retrieved_documents) == 0
    assert len(res_empty.distances) == 0


def test_semantic_search_exact_match_flag():
    docs = load_demo_corpus()
    mock_model = MockModel(dim=16)

    # Build Exact BLAS engine
    engine_exact = build_semantic_engine(
        corpus=docs,
        index_type="Exact BLAS",
        model=mock_model,
    )

    res = engine_exact.search("zero trust security", k=5)
    # For Exact BLAS, every retrieved result is an exact match against the exact baseline
    assert len(res.is_exact_match) == 5
    assert all(res.is_exact_match)


def test_semantic_search_empty_query():
    docs = load_demo_corpus()[:5]
    mock_model = MockModel(dim=16)
    engine = build_semantic_engine(corpus=docs, model=mock_model)

    res = engine.search("   ", k=3)
    assert len(res.retrieved_documents) == 0
    assert len(res.distances) == 0
    assert res.latency_ms == 0.0


def test_get_document_and_contains():
    docs = load_demo_corpus()
    mock_model = MockModel(dim=16)
    engine = build_semantic_engine(corpus=docs, model=mock_model)

    assert 101 in engine
    assert 9999 not in engine

    doc101 = engine.get_document(101)
    assert doc101 is not None
    assert doc101.doc_id == 101

    assert engine.get_document(9999) is None


def test_mock_embed_texts():
    from indexora.semantic import mock_embed_texts

    v_single = mock_embed_texts("hello world", dim=32)
    assert isinstance(v_single, np.ndarray)
    assert v_single.shape == (32,)
    assert np.isclose(np.linalg.norm(v_single), 1.0, atol=1e-5)

    v_batch = mock_embed_texts(["hello", "world"], dim=32)
    assert isinstance(v_batch, np.ndarray)
    assert v_batch.shape == (2, 32)
    for row in v_batch:
        assert np.isclose(np.linalg.norm(row), 1.0, atol=1e-5)


def test_is_sentence_transformers_available_returns_bool():
    """Verify boolean contract of availability check in any environment."""
    assert isinstance(is_sentence_transformers_available(), bool)


def test_is_sentence_transformers_available_true_path(monkeypatch):
    """Verify True path when sentence-transformers is present in sys.modules."""
    import sys
    from unittest.mock import MagicMock

    monkeypatch.setitem(sys.modules, "sentence_transformers", MagicMock())
    assert is_sentence_transformers_available() is True


def test_is_sentence_transformers_available_false_path(monkeypatch):
    """Verify False path when sentence-transformers is absent/unimportable."""
    import sys

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    assert is_sentence_transformers_available() is False


def test_get_embedding_model_missing_dependency_raises(monkeypatch):
    """Verify get_embedding_model raises SentenceTransformersNotAvailableError when absent."""
    import sys

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(SentenceTransformersNotAvailableError, match="pip install"):
        get_embedding_model(model_name="uncached-test-model")


@pytest.mark.integration
def test_real_pretrained_model_integration_opt_in():
    """Opt-in integration test using real pretrained weights.

    Only executed when explicitly requested with: pytest -m integration
    """
    if not is_sentence_transformers_available():
        pytest.skip("sentence-transformers is not installed")

    res = encode_texts("vector search engine", normalize=True)
    assert isinstance(res, np.ndarray)
    assert res.shape == (384,)
    assert np.isclose(np.linalg.norm(res), 1.0, atol=1e-5)
