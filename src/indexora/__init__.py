"""Indexora: A learning-first vector search engine built from scratch in Python and NumPy."""

from indexora.clustering import KMeansResult, kmeans, kmeans_plus_plus
from indexora.data import (
    ClusteredDataset,
    VectorDataset,
    generate_clustered_dataset,
    generate_synthetic_dataset,
    load_dataset,
    normalize_l2,
    save_dataset,
)
from indexora.exact import ExactFlatIndex, exact_search, generate_ground_truth
from indexora.ivf import (
    InvertedList,
    IVFFlatIndex,
    ivf_flat_build,
    ivf_flat_search,
)
from indexora.kernels import pairwise_squared_l2_ref
from indexora.topk import SearchResult, select_top_k

__version__ = "0.1.0.dev0"

__all__ = [
    "__version__",
    "pairwise_squared_l2_ref",
    "SearchResult",
    "select_top_k",
    "ExactFlatIndex",
    "exact_search",
    "generate_ground_truth",
    "VectorDataset",
    "ClusteredDataset",
    "generate_synthetic_dataset",
    "generate_clustered_dataset",
    "normalize_l2",
    "save_dataset",
    "load_dataset",
    "KMeansResult",
    "kmeans_plus_plus",
    "kmeans",
    "InvertedList",
    "IVFFlatIndex",
    "ivf_flat_build",
    "ivf_flat_search",
]
