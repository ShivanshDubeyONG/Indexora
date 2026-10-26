"""Indexora: A learning-first vector search engine built from scratch in Python and NumPy."""

from indexora.exact import ExactFlatIndex, exact_search, generate_ground_truth
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
]
