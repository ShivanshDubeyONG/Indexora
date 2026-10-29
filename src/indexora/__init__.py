"""Indexora: A learning-first vector search engine built from scratch in Python and NumPy."""

from indexora.clustering import KMeansResult, kmeans, kmeans_plus_plus
from indexora.data import (
    ClusteredDataset,
    VectorDataset,
    compute_file_sha256,
    generate_clustered_dataset,
    generate_synthetic_dataset,
    load_dataset,
    load_sift_dataset,
    normalize_l2,
    read_fvecs,
    save_dataset,
    write_fvecs,
)
from indexora.eval import (
    compute_distance_aware_recall_at_k,
    compute_recall_at_k,
    count_distance_computations,
)
from indexora.exact import (
    ExactBlasIndex,
    ExactFlatIndex,
    exact_search,
    exact_search_blas,
    generate_ground_truth,
)
from indexora.hnsw import (
    HNSWIndex,
    hnsw_build,
    hnsw_search,
)
from indexora.ivf import (
    InvertedList,
    IVFFlatIndex,
    ivf_flat_build,
    ivf_flat_search,
)
from indexora.ivfpq import (
    InvertedPQList,
    IVFPQIndex,
    ivf_pq_build,
    ivf_pq_search,
)
from indexora.kernels import pairwise_squared_l2_blas, pairwise_squared_l2_ref
from indexora.lab import (
    IndexCache,
    LabDataset,
    LabSession,
    compute_exact_ground_truth,
    generate_lab_dataset,
)
from indexora.persistence import load_index, save_index
from indexora.pq import ProductQuantizer
from indexora.topk import SearchResult, select_top_k

__version__ = "0.1.0.dev0"

__all__ = [
    "__version__",
    "pairwise_squared_l2_ref",
    "pairwise_squared_l2_blas",
    "SearchResult",
    "select_top_k",
    "ExactFlatIndex",
    "ExactBlasIndex",
    "exact_search",
    "exact_search_blas",
    "generate_ground_truth",
    "VectorDataset",
    "ClusteredDataset",
    "generate_synthetic_dataset",
    "generate_clustered_dataset",
    "normalize_l2",
    "save_dataset",
    "load_dataset",
    "read_fvecs",
    "write_fvecs",
    "load_sift_dataset",
    "compute_file_sha256",
    "KMeansResult",
    "kmeans_plus_plus",
    "kmeans",
    "InvertedList",
    "IVFFlatIndex",
    "ivf_flat_build",
    "ivf_flat_search",
    "ProductQuantizer",
    "HNSWIndex",
    "hnsw_build",
    "hnsw_search",
    "InvertedPQList",
    "IVFPQIndex",
    "ivf_pq_build",
    "ivf_pq_search",
    "save_index",
    "load_index",
    "LabDataset",
    "LabSession",
    "IndexCache",
    "compute_exact_ground_truth",
    "generate_lab_dataset",
    "compute_recall_at_k",
    "compute_distance_aware_recall_at_k",
    "count_distance_computations",
    "BenchmarkConfig",
    "RunMetrics",
    "BenchmarkResult",
    "run_benchmark",
    "FaissFlatIndex",
    "FaissIVFFlatIndex",
    "HnswlibIndex",
    "is_faiss_available",
    "is_hnswlib_available",
    "Document",
    "SemanticSearchResult",
    "SemanticSearchEngine",
    "build_semantic_engine",
    "load_demo_corpus",
    "encode_texts",
    "mock_embed_texts",
    "is_sentence_transformers_available",
]


def __getattr__(name: str):
    """Lazy import benchmark, adapter, and semantic symbols to avoid unneeded dependency overhead."""
    if name in {"BenchmarkConfig", "BenchmarkResult", "RunMetrics", "run_benchmark"}:
        import indexora.benchmark as _bench

        return getattr(_bench, name)

    if name in {
        "FaissFlatIndex",
        "FaissIVFFlatIndex",
        "HnswlibIndex",
        "is_faiss_available",
        "is_hnswlib_available",
    }:
        import indexora.adapters as _adapters

        return getattr(_adapters, name)

    if name in {
        "Document",
        "SemanticSearchResult",
        "SemanticSearchEngine",
        "build_semantic_engine",
        "load_demo_corpus",
        "encode_texts",
        "mock_embed_texts",
        "is_sentence_transformers_available",
    }:
        import indexora.semantic as _semantic

        return getattr(_semantic, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
