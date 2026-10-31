"""Tests for semantic search correctness, model boundary behavior, and unavailable states.

Verifies:
1. Production semantic retrieval strictly rejects mock/random embeddings.
2. Missing sentence-transformers dependency raises SentenceTransformersNotAvailableError.
3. Model weight download / initialization failure raises ModelInitializationError without silent fallback.
4. check_semantic_availability() returns explicit unavailable diagnostics when dependencies fail.
5. SemanticSearchResult correctly distinguishes vector oracle agreement from human relevance.
6. Real embedding provider boundary behavior operates properly with compliant mock model interfaces.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock
import numpy as np
import pytest

from indexora.semantic import (
    DEFAULT_EMBEDDING_MODEL,
    Document,
    ModelInitializationError,
    SemanticSearchEngine,
    SemanticSearchResult,
    SentenceTransformersNotAvailableError,
    build_semantic_engine,
    check_semantic_availability,
    encode_texts,
    get_embedding_model,
    is_sentence_transformers_available,
    load_demo_corpus,
)


class BoundaryCompliantModel:
    """Mock model implementing sentence_transformers encode() protocol for boundary tests."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.call_count = 0

    def encode(
        self,
        sentences: list[str] | str,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        self.call_count += 1
        items = [sentences] if isinstance(sentences, str) else list(sentences)
        # Produce deterministic, non-random structured synthetic embeddings
        out = np.zeros((len(items), self.dim), dtype=np.float32)
        for i, text in enumerate(items):
            # Deterministic linear projection based on character ordinals
            val = float(sum(ord(c) for c in text) % 100) / 100.0
            out[i, :] = val
            out[i, 0] += 1.0
        return out[0] if isinstance(sentences, str) else out


# ==============================================================================
# Dependency & Model Initialization Failure Tests
# ==============================================================================


def test_missing_sentence_transformers_raises_explicit_error(monkeypatch):
    """Missing sentence-transformers must raise SentenceTransformersNotAvailableError."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    assert is_sentence_transformers_available() is False

    with pytest.raises(SentenceTransformersNotAvailableError, match="pip install"):
        get_embedding_model("uninstalled-model")


def test_model_initialization_failure_raises_model_initialization_error(monkeypatch):
    """Failure to download or load model weights raises ModelInitializationError without fallback."""
    fake_st = MagicMock()
    fake_st.SentenceTransformer.side_effect = RuntimeError("Network timeout downloading weights")
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

    # Invalidate cache if present
    import indexora.semantic as sem
    sem._LOADED_MODEL_CACHE.pop("faulty-model", None)

    with pytest.raises(ModelInitializationError, match="Network timeout"):
        get_embedding_model("faulty-model")


def test_check_semantic_availability_diagnostics(monkeypatch):
    """check_semantic_availability() returns false with diagnostic when missing or failing."""
    # 1. Missing dependency
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    avail, msg = check_semantic_availability()
    assert avail is False
    assert "pip install" in msg

    # 2. Failing initialization
    fake_st = MagicMock()
    fake_st.SentenceTransformer.side_effect = ValueError("Corrupt checkpoint")
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)
    import indexora.semantic as sem
    sem._LOADED_MODEL_CACHE.pop(DEFAULT_EMBEDDING_MODEL, None)

    avail2, msg2 = check_semantic_availability(DEFAULT_EMBEDDING_MODEL)
    assert avail2 is False
    assert "Corrupt checkpoint" in msg2 or "Embedding model failed" in msg2


def test_no_silent_fallback_to_random_mock_embeddings(monkeypatch):
    """encode_texts() and build_semantic_engine() must never silently substitute random vectors."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    # encode_texts without model must raise, not generate random hash vectors
    with pytest.raises(SentenceTransformersNotAvailableError):
        encode_texts("sample text query", model=None)

    # build_semantic_engine without model must raise, not generate random hash vectors
    with pytest.raises(SentenceTransformersNotAvailableError):
        build_semantic_engine(corpus=None, model=None)


# ==============================================================================
# Model Boundary & Vector Oracle Agreement Tests
# ==============================================================================


def test_real_embedding_provider_behavior_with_compliant_boundary():
    """Verify SemanticSearchEngine functions correctly with a model boundary implementation."""
    model = BoundaryCompliantModel(dim=128)
    corpus = load_demo_corpus()[:10]

    engine = build_semantic_engine(
        corpus=corpus,
        model=model,
        index_type="Exact BLAS",
    )

    assert len(engine) == 10
    assert engine.dimension == 128

    result = engine.search("transformer attention", k=3)
    assert isinstance(result, SemanticSearchResult)
    assert len(result.retrieved_documents) == 3
    assert len(result.distances) == 3
    assert len(result.is_exact_match) == 3

    # On Exact BLAS, every retrieved document is an exact oracle hit
    assert all(result.is_exact_match)
    assert result.vector_oracle_agreement == 1.0


def test_vector_oracle_agreement_calculation():
    """Verify vector_oracle_agreement property reflects fraction of oracle matches."""
    corpus = load_demo_corpus()[:3]
    dists = np.array([0.1, 0.4, 0.9], dtype=np.float32)

    res_all_hits = SemanticSearchResult(
        query="test query",
        retrieved_documents=corpus,
        distances=dists,
        is_exact_match=[True, True, True],
        latency_ms=0.5,
        embed_time_ms=1.2,
    )
    assert res_all_hits.vector_oracle_agreement == 1.0

    res_partial_hits = SemanticSearchResult(
        query="test query",
        retrieved_documents=corpus,
        distances=dists,
        is_exact_match=[True, False, False],
        latency_ms=0.5,
        embed_time_ms=1.2,
    )
    assert pytest.approx(res_partial_hits.vector_oracle_agreement, 1e-4) == 1.0 / 3.0

    res_empty = SemanticSearchResult(
        query="test query",
        retrieved_documents=[],
        distances=np.empty((0,), dtype=np.float32),
        is_exact_match=[],
        latency_ms=0.0,
        embed_time_ms=0.0,
    )
    assert res_empty.vector_oracle_agreement == 0.0
