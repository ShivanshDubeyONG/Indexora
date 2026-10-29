"""Benchmark harness v0.1 for Indexora.

This module provides reproducible benchmarking of exact search and IVF-Flat sweeps:
- Ground-truth generation via exact flat search.
- Multi-probe IVF-Flat sweeps over configurable `nlist` and `nprobe`.
- Separate measurement of individual per-query latency percentiles and full-batch throughput (QPS).
- Index storage memory reporting (logical bytes, MiB, and bytes per vector).
- Actual search work instrumentation (distance computations executed by the search path).
- Support for both synthetic clustered datasets and real SIFT .fvecs datasets.
- Machine-readable JSON and CSV export with full environment and reproducibility metadata.
- Automated plot generation for Recall vs Per-Query Latency and Recall vs Distance Computations.
- CLI entry point with a fast `--smoke` test mode.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import tracemalloc
from typing import Any, Callable, Sequence

import numpy as np

from indexora.data import (
    VectorDataset,
    compute_file_sha256,
    generate_clustered_dataset,
    load_sift_dataset,
)
from indexora.eval import compute_distance_aware_recall_at_k, compute_recall_at_k
from indexora.exact import ExactBlasIndex, ExactFlatIndex, generate_ground_truth
from indexora.ivf import IVFFlatIndex
from indexora.ivfpq import IVFPQIndex


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration parameters for a vector search benchmark sweep."""

    name: str = "benchmark"
    k: int = 10
    warmup_runs: int = 2
    repeat_runs: int = 5
    latency_samples: int = 500
    seed: int = 42
    nlist_values: tuple[int, ...] = (10, 50, 100)
    nprobe_values: tuple[int, ...] = (1, 2, 5, 10, 20)
    smoke: bool = False
    dataset_type: str = "synthetic"
    base_path: str | None = None
    query_path: str | None = None
    compare_faiss: bool = False
    compare_hnswlib: bool = False
    include_ivf_pq: bool = False
    ivf_pq_m: int = 8
    ivf_pq_k_sub: int = 256

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")
        if self.warmup_runs < 0:
            raise ValueError(f"warmup_runs must be non-negative, got {self.warmup_runs}")
        if self.repeat_runs < 1:
            raise ValueError(f"repeat_runs must be at least 1, got {self.repeat_runs}")
        if self.latency_samples < 1:
            raise ValueError(f"latency_samples must be at least 1, got {self.latency_samples}")
        if not self.nlist_values:
            raise ValueError("nlist_values cannot be empty")
        if not self.nprobe_values:
            raise ValueError("nprobe_values cannot be empty")


@dataclass(frozen=True)
class RunMetrics:
    """Measured performance, storage, and accuracy metrics for an index configuration."""

    index_name: str
    parameters: dict[str, Any]
    build_time_seconds: float
    build_peak_tracemalloc_bytes: int
    index_memory_bytes: int
    index_memory_mib: float
    bytes_per_vector: float
    per_query_latencies_ms: tuple[float, ...]
    per_query_p50_ms: float
    per_query_p95_ms: float
    per_query_p99_ms: float
    per_query_mean_ms: float
    per_query_min_ms: float
    per_query_max_ms: float
    per_query_sample_count: int
    batch_qps: float
    batch_mean_latency_ms: float
    batch_repeat_runs: int
    recall_at_k: float
    distance_computations_per_query: float
    per_query_std_ms: float = 0.0
    distance_aware_recall_at_k: float = 0.0
    kmeans_n_iter: int | None = None
    kmeans_converged: bool | None = None
    kmeans_inertia: float | None = None
    peak_process_rss_bytes: int | None = None
    total_distance_computations: int = 0
    total_centroid_distance_computations: int = 0
    total_candidate_distance_computations: int = 0
    centroid_distance_computations_per_query: float = 0.0
    candidate_distance_computations_per_query: float = 0.0

    @property
    def distance_computations(self) -> float:
        """Backward-compatible alias for distance_computations_per_query."""
        return self.distance_computations_per_query

    @property
    def p50_latency_ms(self) -> float:
        """Backward-compatible alias for per_query_p50_ms."""
        return self.per_query_p50_ms

    @property
    def p95_latency_ms(self) -> float:
        """Backward-compatible alias for per_query_p95_ms."""
        return self.per_query_p95_ms

    @property
    def p99_latency_ms(self) -> float:
        """Backward-compatible alias for per_query_p99_ms."""
        return self.per_query_p99_ms

    @property
    def mean_latency_ms(self) -> float:
        """Backward-compatible alias for per_query_mean_ms."""
        return self.per_query_mean_ms

    @property
    def std_latency_ms(self) -> float:
        """Backward-compatible alias for per_query_std_ms."""
        return self.per_query_std_ms

    @property
    def min_latency_ms(self) -> float:
        """Backward-compatible alias for per_query_min_ms."""
        return self.per_query_min_ms

    @property
    def max_latency_ms(self) -> float:
        """Backward-compatible alias for per_query_max_ms."""
        return self.per_query_max_ms

    @property
    def qps(self) -> float:
        """Backward-compatible alias for batch_qps."""
        return self.batch_qps

    @property
    def query_latencies_ms(self) -> tuple[float, ...]:
        """Backward-compatible alias for per_query_latencies_ms."""
        return self.per_query_latencies_ms

    @property
    def distance_aware_recall(self) -> float:
        """Backward-compatible alias for distance_aware_recall_at_k."""
        return self.distance_aware_recall_at_k

    @property
    def peak_rss_bytes(self) -> int | None:
        """Process peak RSS in bytes."""
        return self.peak_process_rss_bytes

    @property
    def peak_rss_mib(self) -> float:
        """Process peak RSS in MiB."""
        return (
            float(self.peak_process_rss_bytes / (1024 * 1024))
            if self.peak_process_rss_bytes is not None
            else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a standard serializable dictionary."""
        d = asdict(self)
        d["per_query_latencies_ms"] = list(self.per_query_latencies_ms)
        # Ensure backward-compatible keys exist in serialized manifest
        d["p50_latency_ms"] = self.p50_latency_ms
        d["p95_latency_ms"] = self.p95_latency_ms
        d["p99_latency_ms"] = self.p99_latency_ms
        d["mean_latency_ms"] = self.mean_latency_ms
        d["std_latency_ms"] = self.std_latency_ms
        d["min_latency_ms"] = self.min_latency_ms
        d["max_latency_ms"] = self.max_latency_ms
        d["qps"] = self.qps
        d["query_latencies_ms"] = list(self.per_query_latencies_ms)
        d["distance_aware_recall"] = self.distance_aware_recall_at_k
        d["peak_rss_mib"] = self.peak_rss_mib
        return d


@dataclass
class BenchmarkResult:
    """Complete results of a benchmark run, including environment metadata and raw metrics."""

    environment: dict[str, Any]
    dataset_metadata: dict[str, Any]
    config: dict[str, Any]
    timestamp: str
    runs: list[RunMetrics] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete benchmark result to dictionary."""
        return {
            "environment": self.environment,
            "dataset_metadata": self.dataset_metadata,
            "config": self.config,
            "timestamp": self.timestamp,
            "runs": [r.to_dict() for r in self.runs],
        }

    def to_json(self, path: str | Path) -> None:
        """Write benchmark results to a formatted JSON manifest."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_csv(self, path: str | Path) -> None:
        """Write tabular summary of runs to a CSV file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "index_name",
            "nlist",
            "nprobe",
            "build_time_sec",
            "build_peak_mem_mb",
            "peak_rss_mib",
            "index_mem_mib",
            "bytes_per_vec",
            "per_query_p50_ms",
            "per_query_p95_ms",
            "per_query_mean_ms",
            "per_query_std_ms",
            "batch_qps",
            "recall_at_k",
            "dist_aware_recall",
            "kmeans_n_iter",
            "kmeans_converged",
            "distance_computations",
            "total_distance_computations",
            "centroid_dist_comps_per_query",
            "candidate_dist_comps_per_query",
        ]

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.runs:
                params = r.parameters
                writer.writerow(
                    {
                        "index_name": r.index_name,
                        "nlist": params.get("nlist", "-"),
                        "nprobe": params.get("nprobe", "-"),
                        "build_time_sec": f"{r.build_time_seconds:.4f}",
                        "build_peak_mem_mb": f"{r.build_peak_tracemalloc_bytes / (1024 * 1024):.2f}",
                        "peak_rss_mib": (
                            f"{r.peak_process_rss_bytes / (1024 * 1024):.2f}"
                            if r.peak_process_rss_bytes is not None
                            else "-"
                        ),
                        "index_mem_mib": f"{r.index_memory_mib:.2f}",
                        "bytes_per_vec": f"{r.bytes_per_vector:.1f}",
                        "per_query_p50_ms": f"{r.per_query_p50_ms:.3f}",
                        "per_query_p95_ms": f"{r.per_query_p95_ms:.3f}",
                        "per_query_mean_ms": f"{r.per_query_mean_ms:.3f}",
                        "per_query_std_ms": f"{r.per_query_std_ms:.3f}",
                        "batch_qps": f"{r.batch_qps:.1f}",
                        "recall_at_k": f"{r.recall_at_k:.4f}",
                        "dist_aware_recall": f"{r.distance_aware_recall_at_k:.4f}",
                        "kmeans_n_iter": str(r.kmeans_n_iter) if r.kmeans_n_iter is not None else "-",
                        "kmeans_converged": str(r.kmeans_converged) if r.kmeans_converged is not None else "-",
                        "distance_computations": f"{r.distance_computations_per_query:.1f}",
                        "total_distance_computations": f"{r.total_distance_computations}",
                        "centroid_dist_comps_per_query": f"{r.centroid_distance_computations_per_query:.1f}",
                        "candidate_dist_comps_per_query": f"{r.candidate_distance_computations_per_query:.1f}",
                    }
                )

    def plot(self, output_dir: str | Path) -> list[str]:
        """Generate headline benchmark plots (Recall vs Latency and Recall vs Distance Computations).

        Returns
        -------
        list[str]
            List of generated plot image file paths.

        Raises
        ------
        RuntimeError
            If matplotlib is not installed.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "Matplotlib is required for generating benchmark plots. "
                "Install it via 'pip install matplotlib' or pass '--no-plot' to skip."
            ) from exc

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        generated_plots = []

        exact_ref_runs = [r for r in self.runs if r.index_name in ("exact", "exact_reference")]
        exact_blas_runs = [r for r in self.runs if r.index_name == "exact_blas"]
        faiss_flat_runs = [r for r in self.runs if r.index_name == "faiss_flat"]
        faiss_ivf_runs = [r for r in self.runs if r.index_name == "faiss_ivf_flat"]
        hnsw_runs = [r for r in self.runs if r.index_name == "hnswlib"]
        ivf_runs = [r for r in self.runs if r.index_name == "ivf_flat"]
        ivf_pq_runs = [r for r in self.runs if r.index_name == "ivf_pq"]

        nlist_groups: dict[int, list[RunMetrics]] = {}
        for r in ivf_runs:
            nl = int(r.parameters.get("nlist", 0))
            nlist_groups.setdefault(nl, []).append(r)

        pq_nlist_groups: dict[int, list[RunMetrics]] = {}
        for r in ivf_pq_runs:
            nl = int(r.parameters.get("nlist", 0))
            pq_nlist_groups.setdefault(nl, []).append(r)

        ds_name = self.dataset_metadata.get("name", "Dataset")
        n_base = self.dataset_metadata.get("n_base", 0)
        d_dim = self.dataset_metadata.get("d", 0)
        k_val = self.config.get("k", 10)

        # 1. Plot: Recall@k vs Per-Query p50 Latency (ms)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for nl, group in sorted(nlist_groups.items()):
            group_sorted = sorted(group, key=lambda x: int(x.parameters.get("nprobe", 0)))
            latencies = [x.per_query_p50_ms for x in group_sorted]
            recalls = [x.recall_at_k for x in group_sorted]
            ax.plot(latencies, recalls, marker="o", label=f"IVF-Flat (nlist={nl})")
            for x, y, run in zip(latencies, recalls, group_sorted):
                p = run.parameters.get("nprobe", "")
                ax.annotate(f"p={p}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

        for nl, group in sorted(pq_nlist_groups.items()):
            group_sorted = sorted(group, key=lambda x: int(x.parameters.get("nprobe", 0)))
            latencies = [x.per_query_p50_ms for x in group_sorted]
            recalls = [x.recall_at_k for x in group_sorted]
            m_val = group_sorted[0].parameters.get("M", "")
            ax.plot(latencies, recalls, marker="^", linestyle="-.", label=f"IVF-PQ (M={m_val}, nlist={nl})")
            for x, y, run in zip(latencies, recalls, group_sorted):
                p = run.parameters.get("nprobe", "")
                ax.annotate(f"pq p={p}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)

        if exact_ref_runs:
            ex = exact_ref_runs[0]
            ax.scatter(
                [ex.per_query_p50_ms],
                [ex.recall_at_k],
                color="red",
                s=130,
                marker="*",
                label=f"Exact Flat (Ref) (p50={ex.per_query_p50_ms:.2f}ms)",
                zorder=5,
            )

        if exact_blas_runs:
            ex_b = exact_blas_runs[0]
            ax.scatter(
                [ex_b.per_query_p50_ms],
                [ex_b.recall_at_k],
                color="purple",
                s=100,
                marker="D",
                label=f"Exact Flat (BLAS) (p50={ex_b.per_query_p50_ms:.2f}ms)",
                zorder=5,
            )

        if faiss_flat_runs:
            ff = faiss_flat_runs[0]
            ax.scatter(
                [ff.per_query_p50_ms],
                [ff.recall_at_k],
                color="darkorange",
                s=110,
                marker="^",
                label=f"FAISS Flat (p50={ff.per_query_p50_ms:.2f}ms)",
                zorder=5,
            )

        if faiss_ivf_runs:
            faiss_nl_groups: dict[int, list[RunMetrics]] = {}
            for r in faiss_ivf_runs:
                nl = int(r.parameters.get("nlist", 0))
                faiss_nl_groups.setdefault(nl, []).append(r)
            for nl, group in sorted(faiss_nl_groups.items()):
                group_sorted = sorted(group, key=lambda x: int(x.parameters.get("nprobe", 0)))
                lats = [x.per_query_p50_ms for x in group_sorted]
                recs = [x.recall_at_k for x in group_sorted]
                ax.plot(lats, recs, marker="v", linestyle="--", label=f"FAISS IVF-Flat (nlist={nl})")
                for x, y, run in zip(lats, recs, group_sorted):
                    p = run.parameters.get("nprobe", "")
                    ax.annotate(f"faiss p={p}", (x, y), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=7)

        if hnsw_runs:
            hnsw_sorted = sorted(hnsw_runs, key=lambda x: int(x.parameters.get("ef_search", 0)))
            lats = [x.per_query_p50_ms for x in hnsw_sorted]
            recs = [x.recall_at_k for x in hnsw_sorted]
            ax.plot(lats, recs, marker="x", linestyle=":", color="green", label="hnswlib")
            for x, y, run in zip(lats, recs, hnsw_sorted):
                ef = run.parameters.get("ef_search", "")
                ax.annotate(f"ef={ef}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)

        ax.set_title(f"Recall@{k_val} vs. Per-Query p50 Latency\n{ds_name} (N={n_base:,}, d={d_dim})")
        ax.set_xlabel("p50 Individual Query Latency (ms)")
        ax.set_ylabel(f"Recall@{k_val}")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        plt.tight_layout()

        plot1_path = out_dir / "recall_vs_latency.png"
        fig.savefig(plot1_path, dpi=150)
        plt.close(fig)
        generated_plots.append(str(plot1_path))

        # 2. Plot: Recall@k vs Distance Computations per Query
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for nl, group in sorted(nlist_groups.items()):
            group_sorted = sorted(group, key=lambda x: int(x.parameters.get("nprobe", 0)))
            dcs = [x.distance_computations_per_query for x in group_sorted]
            recalls = [x.recall_at_k for x in group_sorted]
            ax.plot(dcs, recalls, marker="s", label=f"IVF-Flat (nlist={nl})")
            for x, y, run in zip(dcs, recalls, group_sorted):
                p = run.parameters.get("nprobe", "")
                ax.annotate(f"p={p}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

        for nl, group in sorted(pq_nlist_groups.items()):
            group_sorted = sorted(group, key=lambda x: int(x.parameters.get("nprobe", 0)))
            dcs = [x.distance_computations_per_query for x in group_sorted]
            recalls = [x.recall_at_k for x in group_sorted]
            m_val = group_sorted[0].parameters.get("M", "")
            ax.plot(dcs, recalls, marker="^", linestyle="-.", label=f"IVF-PQ (M={m_val}, nlist={nl})")
            for x, y, run in zip(dcs, recalls, group_sorted):
                p = run.parameters.get("nprobe", "")
                ax.annotate(f"pq p={p}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)

        if exact_ref_runs:
            ex = exact_ref_runs[0]
            ax.scatter(
                [ex.distance_computations_per_query],
                [ex.recall_at_k],
                color="red",
                s=130,
                marker="*",
                label=f"Exact Flat (Ref) (N={int(ex.distance_computations_per_query):,})",
                zorder=5,
            )

        if exact_blas_runs:
            ex_b = exact_blas_runs[0]
            ax.scatter(
                [ex_b.distance_computations_per_query],
                [ex_b.recall_at_k],
                color="purple",
                s=100,
                marker="D",
                label=f"Exact Flat (BLAS) (N={int(ex_b.distance_computations_per_query):,})",
                zorder=5,
            )

        if faiss_flat_runs:
            ff = faiss_flat_runs[0]
            ax.scatter(
                [ff.distance_computations_per_query],
                [ff.recall_at_k],
                color="darkorange",
                s=110,
                marker="^",
                label=f"FAISS Flat (N={int(ff.distance_computations_per_query):,})",
                zorder=5,
            )

        ax.set_title(f"Recall@{k_val} vs. Distance Computations / Query\n{ds_name} (N={n_base:,}, d={d_dim})")
        ax.set_xlabel("Actual Distance Computations per Query")
        ax.set_ylabel(f"Recall@{k_val}")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        plt.tight_layout()

        plot2_path = out_dir / "recall_vs_distance_computations.png"
        fig.savefig(plot2_path, dpi=150)
        plt.close(fig)
        generated_plots.append(str(plot2_path))

        return generated_plots


def get_process_peak_rss_bytes() -> int | None:
    """Return the peak resident set size (RSS) in bytes for the current process, or None."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            fn = ctypes.windll.psapi.GetProcessMemoryInfo
            fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
            fn.restype = wintypes.BOOL
            if fn(handle, ctypes.byref(counters), counters.cb):
                return int(counters.PeakWorkingSetSize)
        except Exception:
            return None
        return None
    else:
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                return int(usage)
            else:
                return int(usage * 1024)
        except Exception:
            return None


def _resolve_latency_sample(queries: np.ndarray, sample_count: int, seed: int = 42) -> np.ndarray:
    """Deterministically select a subset of queries for per-query latency measurements."""
    n_queries = queries.shape[0]
    if sample_count >= n_queries:
        return queries
    rng = np.random.default_rng(seed)
    chosen_indices = np.sort(rng.choice(n_queries, size=sample_count, replace=False))
    return queries[chosen_indices]


def get_environment_metadata() -> dict[str, Any]:
    """Collect platform, Python, NumPy, git, and thread environment metadata."""
    git_commit = "unknown"
    try:
        git_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=Path(__file__).parent,
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        pass

    processor = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or platform.machine() or "unknown"

    thread_env_vars: dict[str, str] = {}
    for var in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        val = os.environ.get(var)
        if val is not None:
            thread_env_vars[var] = val

    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": processor,
        "git_commit": git_commit,
        "thread_env_vars": thread_env_vars,
        "measurement_methodology": (
            "Individual per-query timing with warmup for latency percentiles; "
            "repeated batched timing for throughput (QPS)."
        ),
    }


def _measure_per_query_latency(
    search_single_fn: Callable[[np.ndarray, int], Any],
    queries_subset: np.ndarray,
    k: int,
    warmup_runs: int,
) -> tuple[tuple[float, ...], float, float, float, float, float, float, float, int]:
    """Execute warmup and timed individual query searches to collect per-query latency percentiles."""
    num_samples = queries_subset.shape[0]

    # Warmup individual queries
    n_warm = min(warmup_runs, num_samples)
    for i in range(n_warm):
        _ = search_single_fn(queries_subset[i], k)

    # Measured individual query runs
    latencies_ms: list[float] = []
    for i in range(num_samples):
        t0 = time.perf_counter()
        _ = search_single_fn(queries_subset[i], k)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    lat_arr = np.array(latencies_ms, dtype=np.float64)
    p50 = float(np.percentile(lat_arr, 50))
    p95 = float(np.percentile(lat_arr, 95))
    p99 = float(np.percentile(lat_arr, 99))
    mean_val = float(np.mean(lat_arr))
    std_val = float(np.std(lat_arr))
    min_val = float(np.min(lat_arr))
    max_val = float(np.max(lat_arr))

    return tuple(latencies_ms), p50, p95, p99, mean_val, std_val, min_val, max_val, num_samples


def _measure_batch_throughput(
    search_batch_fn: Callable[[np.ndarray, int], Any],
    queries: np.ndarray,
    k: int,
    warmup_runs: int,
    repeat_runs: int,
) -> tuple[float, float]:
    """Execute warmup and repeated full-batch searches to measure overall batch throughput (QPS)."""
    # Warmup batch runs
    for _ in range(warmup_runs):
        _ = search_batch_fn(queries, k)

    # Measured batch runs
    batch_times_sec: list[float] = []
    for _ in range(repeat_runs):
        t0 = time.perf_counter()
        _ = search_batch_fn(queries, k)
        batch_times_sec.append(time.perf_counter() - t0)

    total_batch_sec = sum(batch_times_sec)
    total_queries = queries.shape[0] * repeat_runs
    batch_qps = float(total_queries / total_batch_sec) if total_batch_sec > 0.0 else 0.0
    mean_batch_ms = float(np.mean(batch_times_sec) * 1000.0)

    return batch_qps, mean_batch_ms


def run_benchmark(
    dataset: VectorDataset,
    config: BenchmarkConfig,
    verbose: bool = True,
    dataset_name: str | None = None,
    dataset_extra_metadata: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """Run exact search and IVF-Flat benchmark sweeps on a given dataset."""
    N, d = dataset.base.shape
    Q = dataset.queries.shape[0]

    env_meta = get_environment_metadata()
    effective_name = dataset_name or config.name

    dataset_meta = {
        "name": effective_name,
        "n_base": N,
        "n_queries": Q,
        "d": d,
        "dtype": str(dataset.base.dtype),
        "metric": "squared_l2",
        "seed": config.seed,
        "stable_id_policy": "int64_unique",
        "ground_truth_method": "exact_flat_reference",
    }
    if dataset_extra_metadata:
        dataset_meta.update(dataset_extra_metadata)

    result = BenchmarkResult(
        environment=env_meta,
        dataset_metadata=dataset_meta,
        config=asdict(config),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    if verbose:
        print("=" * 80)
        print(f"INDEXORA BENCHMARK HARNESS v0.1 -- {effective_name.upper()}")
        print(f"Dataset: N={N:,}, Q={Q:,}, d={d} | Seed: {config.seed} | k: {config.k}")
        print("=" * 80)

    # 1. Ground Truth Generation
    if verbose:
        print("\n[1/3] Generating exact ground-truth baseline...")
    t0_gt = time.perf_counter()
    gt_result = generate_ground_truth(
        queries=dataset.queries,
        database=dataset.base,
        ids=dataset.base_ids,
        k=config.k,
    )
    t_gt = time.perf_counter() - t0_gt
    if verbose:
        print(f"  Ground truth ready in {t_gt:.2f}s ({Q} queries against {N:,} vectors).")

    # Resolve latency query sample deterministically
    queries_latency_sample = _resolve_latency_sample(
        dataset.queries, config.latency_samples, seed=config.seed
    )

    # 2. Exact Flat Baselines Benchmark (Reference and BLAS)
    if verbose:
        print("\n[2/4] Benchmarking Exact Flat Reference Index...")
    tracemalloc.start()
    t0_ref_b = time.perf_counter()
    exact_ref_index = ExactFlatIndex(database=dataset.base, ids=dataset.base_ids)
    exact_ref_build_time = time.perf_counter() - t0_ref_b
    _, exact_ref_build_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    lats_ref, p50_ref, p95_ref, p99_ref, mean_ref, std_ref, min_ref, max_ref, sample_cnt_ref = (
        _measure_per_query_latency(
            search_single_fn=lambda q, k: exact_ref_index.search(q, k=k),
            queries_subset=queries_latency_sample,
            k=config.k,
            warmup_runs=config.warmup_runs,
        )
    )

    batch_qps_ref, batch_mean_ms_ref = _measure_batch_throughput(
        search_batch_fn=lambda q, k: exact_ref_index.search(q, k=k),
        queries=dataset.queries,
        k=config.k,
        warmup_runs=config.warmup_runs,
        repeat_runs=config.repeat_runs,
    )

    exact_ref_res = exact_ref_index.search(dataset.queries, k=config.k)
    exact_ref_recall = compute_recall_at_k(exact_ref_res, gt_result, k=config.k)
    exact_ref_dist_recall = compute_distance_aware_recall_at_k(exact_ref_res, gt_result, k=config.k)
    ref_total_dc = int(exact_ref_res.distance_computations) if exact_ref_res.distance_computations is not None else Q * N
    ref_total_centroid_dc = int(exact_ref_res.centroid_distance_computations) if exact_ref_res.centroid_distance_computations is not None else 0
    ref_total_cand_dc = int(exact_ref_res.candidate_distance_computations) if exact_ref_res.candidate_distance_computations is not None else ref_total_dc
    ref_dc = float(ref_total_dc / Q) if Q > 0 else 0.0
    ref_centroid_dc = float(ref_total_centroid_dc / Q) if Q > 0 else 0.0
    ref_cand_dc = float(ref_total_cand_dc / Q) if Q > 0 else 0.0
    peak_rss_ref = get_process_peak_rss_bytes()

    exact_ref_metrics = RunMetrics(
        index_name="exact_reference",
        parameters={},
        build_time_seconds=exact_ref_build_time,
        build_peak_tracemalloc_bytes=exact_ref_build_mem,
        index_memory_bytes=exact_ref_index.index_memory_bytes,
        index_memory_mib=exact_ref_index.index_memory_mib,
        bytes_per_vector=exact_ref_index.bytes_per_vector,
        per_query_latencies_ms=lats_ref,
        per_query_p50_ms=p50_ref,
        per_query_p95_ms=p95_ref,
        per_query_p99_ms=p99_ref,
        per_query_mean_ms=mean_ref,
        per_query_min_ms=min_ref,
        per_query_max_ms=max_ref,
        per_query_sample_count=sample_cnt_ref,
        batch_qps=batch_qps_ref,
        batch_mean_latency_ms=batch_mean_ms_ref,
        batch_repeat_runs=config.repeat_runs,
        recall_at_k=exact_ref_recall,
        distance_computations_per_query=ref_dc,
        per_query_std_ms=std_ref,
        distance_aware_recall_at_k=exact_ref_dist_recall,
        peak_process_rss_bytes=peak_rss_ref,
        total_distance_computations=ref_total_dc,
        total_centroid_distance_computations=ref_total_centroid_dc,
        total_candidate_distance_computations=ref_total_cand_dc,
        centroid_distance_computations_per_query=ref_centroid_dc,
        candidate_distance_computations_per_query=ref_cand_dc,
    )
    result.runs.append(exact_ref_metrics)

    if verbose:
        print(
            f"  Exact Flat (Ref):  p50={p50_ref:.2f}ms (std={std_ref:.2f}ms) | QPS={batch_qps_ref:.1f} | "
            f"Recall={exact_ref_recall:.4f} | DistRecall={exact_ref_dist_recall:.4f} | "
            f"DC/Q={ref_dc:.0f} | Storage={exact_ref_index.index_memory_mib:.2f}MiB"
        )

    if verbose:
        print("\n[3/4] Benchmarking Exact Flat BLAS Index...")
    tracemalloc.start()
    t0_blas_b = time.perf_counter()
    exact_blas_index = ExactBlasIndex(database=dataset.base, ids=dataset.base_ids)
    exact_blas_build_time = time.perf_counter() - t0_blas_b
    _, exact_blas_build_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    lats_blas, p50_blas, p95_blas, p99_blas, mean_blas, std_blas, min_blas, max_blas, sample_cnt_blas = (
        _measure_per_query_latency(
            search_single_fn=lambda q, k: exact_blas_index.search(q, k=k),
            queries_subset=queries_latency_sample,
            k=config.k,
            warmup_runs=config.warmup_runs,
        )
    )

    batch_qps_blas, batch_mean_ms_blas = _measure_batch_throughput(
        search_batch_fn=lambda q, k: exact_blas_index.search(q, k=k),
        queries=dataset.queries,
        k=config.k,
        warmup_runs=config.warmup_runs,
        repeat_runs=config.repeat_runs,
    )

    exact_blas_res = exact_blas_index.search(dataset.queries, k=config.k)
    exact_blas_recall = compute_recall_at_k(exact_blas_res, gt_result, k=config.k)
    exact_blas_dist_recall = compute_distance_aware_recall_at_k(exact_blas_res, gt_result, k=config.k)
    blas_total_dc = int(exact_blas_res.distance_computations) if exact_blas_res.distance_computations is not None else Q * N
    blas_total_centroid_dc = int(exact_blas_res.centroid_distance_computations) if exact_blas_res.centroid_distance_computations is not None else 0
    blas_total_cand_dc = int(exact_blas_res.candidate_distance_computations) if exact_blas_res.candidate_distance_computations is not None else blas_total_dc
    blas_dc = float(blas_total_dc / Q) if Q > 0 else 0.0
    blas_centroid_dc = float(blas_total_centroid_dc / Q) if Q > 0 else 0.0
    blas_cand_dc = float(blas_total_cand_dc / Q) if Q > 0 else 0.0
    peak_rss_blas = get_process_peak_rss_bytes()

    exact_blas_metrics = RunMetrics(
        index_name="exact_blas",
        parameters={},
        build_time_seconds=exact_blas_build_time,
        build_peak_tracemalloc_bytes=exact_blas_build_mem,
        index_memory_bytes=exact_blas_index.index_memory_bytes,
        index_memory_mib=exact_blas_index.index_memory_mib,
        bytes_per_vector=exact_blas_index.bytes_per_vector,
        per_query_latencies_ms=lats_blas,
        per_query_p50_ms=p50_blas,
        per_query_p95_ms=p95_blas,
        per_query_p99_ms=p99_blas,
        per_query_mean_ms=mean_blas,
        per_query_min_ms=min_blas,
        per_query_max_ms=max_blas,
        per_query_sample_count=sample_cnt_blas,
        batch_qps=batch_qps_blas,
        batch_mean_latency_ms=batch_mean_ms_blas,
        batch_repeat_runs=config.repeat_runs,
        recall_at_k=exact_blas_recall,
        distance_computations_per_query=blas_dc,
        per_query_std_ms=std_blas,
        distance_aware_recall_at_k=exact_blas_dist_recall,
        peak_process_rss_bytes=peak_rss_blas,
        total_distance_computations=blas_total_dc,
        total_centroid_distance_computations=blas_total_centroid_dc,
        total_candidate_distance_computations=blas_total_cand_dc,
        centroid_distance_computations_per_query=blas_centroid_dc,
        candidate_distance_computations_per_query=blas_cand_dc,
    )
    result.runs.append(exact_blas_metrics)

    if verbose:
        print(
            f"  Exact Flat (BLAS): p50={p50_blas:.2f}ms (std={std_blas:.2f}ms) | QPS={batch_qps_blas:.1f} | "
            f"Recall={exact_blas_recall:.4f} | DistRecall={exact_blas_dist_recall:.4f} | "
            f"DC/Q={blas_dc:.0f} | Storage={exact_blas_index.index_memory_mib:.2f}MiB"
        )

    # 3. IVF-Flat Sweeps
    if verbose:
        print("\n[4/4] Benchmarking IVF-Flat Parameter Sweeps...")

    for nlist in config.nlist_values:
        if nlist > N:
            continue

        if verbose:
            print(f"  Building IVF-Flat (nlist={nlist})...")

        tracemalloc.start()
        t0_ivf_b = time.perf_counter()
        ivf_index = IVFFlatIndex(
            database=dataset.base,
            nlist=nlist,
            ids=dataset.base_ids,
            seed=config.seed,
        )
        ivf_build_time = time.perf_counter() - t0_ivf_b
        _, ivf_build_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        km_iter = ivf_index.kmeans_n_iter
        km_conv = ivf_index.kmeans_converged
        km_inertia = ivf_index.kmeans_inertia

        for nprobe in config.nprobe_values:
            if nprobe > nlist:
                continue

            # Per-query latency percentiles
            lats, p50, p95, p99, mean_l, std_l, min_l, max_l, sample_count = _measure_per_query_latency(
                search_single_fn=lambda q, k: ivf_index.search(q, k=k, nprobe=nprobe),
                queries_subset=queries_latency_sample,
                k=config.k,
                warmup_runs=config.warmup_runs,
            )

            # Batch throughput
            batch_qps, batch_mean_ms = _measure_batch_throughput(
                search_batch_fn=lambda q, k: ivf_index.search(q, k=k, nprobe=nprobe),
                queries=dataset.queries,
                k=config.k,
                warmup_runs=config.warmup_runs,
                repeat_runs=config.repeat_runs,
            )

            # Evaluate search accuracy against ground truth using actual search execution
            search_res = ivf_index.search(dataset.queries, k=config.k, nprobe=nprobe)
            recall = compute_recall_at_k(search_res, gt_result, k=config.k)
            dist_aware_rec = compute_distance_aware_recall_at_k(search_res, gt_result, k=config.k)
            ivf_total_dc = int(search_res.distance_computations) if search_res.distance_computations is not None else 0
            ivf_total_centroid_dc = int(search_res.centroid_distance_computations) if search_res.centroid_distance_computations is not None else 0
            ivf_total_cand_dc = int(search_res.candidate_distance_computations) if search_res.candidate_distance_computations is not None else 0
            ivf_dc = float(ivf_total_dc / Q) if Q > 0 else 0.0
            ivf_centroid_dc = float(ivf_total_centroid_dc / Q) if Q > 0 else 0.0
            ivf_cand_dc = float(ivf_total_cand_dc / Q) if Q > 0 else 0.0
            peak_rss_ivf = get_process_peak_rss_bytes()

            run_m = RunMetrics(
                index_name="ivf_flat",
                parameters={"nlist": nlist, "nprobe": nprobe},
                build_time_seconds=ivf_build_time,
                build_peak_tracemalloc_bytes=ivf_build_mem,
                index_memory_bytes=ivf_index.index_memory_bytes,
                index_memory_mib=ivf_index.index_memory_mib,
                bytes_per_vector=ivf_index.bytes_per_vector,
                per_query_latencies_ms=lats,
                per_query_p50_ms=p50,
                per_query_p95_ms=p95,
                per_query_p99_ms=p99,
                per_query_mean_ms=mean_l,
                per_query_min_ms=min_l,
                per_query_max_ms=max_l,
                per_query_sample_count=sample_count,
                batch_qps=batch_qps,
                batch_mean_latency_ms=batch_mean_ms,
                batch_repeat_runs=config.repeat_runs,
                recall_at_k=recall,
                distance_computations_per_query=ivf_dc,
                per_query_std_ms=std_l,
                distance_aware_recall_at_k=dist_aware_rec,
                kmeans_n_iter=km_iter,
                kmeans_converged=km_conv,
                kmeans_inertia=km_inertia,
                peak_process_rss_bytes=peak_rss_ivf,
                total_distance_computations=ivf_total_dc,
                total_centroid_distance_computations=ivf_total_centroid_dc,
                total_candidate_distance_computations=ivf_total_cand_dc,
                centroid_distance_computations_per_query=ivf_centroid_dc,
                candidate_distance_computations_per_query=ivf_cand_dc,
            )
            result.runs.append(run_m)

            if verbose:
                print(
                    f"    nprobe={nprobe:<3d} -> Recall@{config.k}={recall:.4f} (DistRec={dist_aware_rec:.4f}) | "
                    f"p50={p50:6.2f}ms (std={std_l:5.2f}ms) | QPS={batch_qps:7.1f} | "
                    f"DC/Q={ivf_dc:6.0f} (Cent={ivf_centroid_dc:4.0f}, Cand={ivf_cand_dc:4.0f})"
                )

    # 4. IVF-PQ Sweeps (Phase 9)
    if config.include_ivf_pq:
        if verbose:
            print("\n[5/5] Benchmarking IVF-PQ Parameter Sweeps...")

        d_val = dataset.base.shape[1]
        target_m = config.ivf_pq_m
        if d_val % target_m != 0:
            divisors = [m for m in range(1, target_m + 1) if d_val % m == 0]
            pq_m = max(divisors) if divisors else 1
        else:
            pq_m = target_m

        pq_k_sub = min(config.ivf_pq_k_sub, max(4, N // 2))

        for nlist in config.nlist_values:
            if nlist > N:
                continue

            if verbose:
                print(f"  Building IVF-PQ (nlist={nlist}, M={pq_m}, k_sub={pq_k_sub})...")

            tracemalloc.start()
            t0_pq_b = time.perf_counter()
            ivf_pq_index = IVFPQIndex(
                database=dataset.base,
                nlist=nlist,
                M=pq_m,
                k_sub=pq_k_sub,
                ids=dataset.base_ids,
                seed=config.seed,
            )
            pq_build_time = time.perf_counter() - t0_pq_b
            _, pq_build_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            for nprobe in config.nprobe_values:
                if nprobe > nlist:
                    continue

                lats, p50, p95, p99, mean_l, std_l, min_l, max_l, sample_count = _measure_per_query_latency(
                    search_single_fn=lambda q, k: ivf_pq_index.search(q, k=k, nprobe=nprobe),
                    queries_subset=queries_latency_sample,
                    k=config.k,
                    warmup_runs=config.warmup_runs,
                )

                batch_qps, batch_mean_ms = _measure_batch_throughput(
                    search_batch_fn=lambda q, k: ivf_pq_index.search(q, k=k, nprobe=nprobe),
                    queries=dataset.queries,
                    k=config.k,
                    warmup_runs=config.warmup_runs,
                    repeat_runs=config.repeat_runs,
                )

                pq_eval_res = ivf_pq_index.search(dataset.queries, k=config.k, nprobe=nprobe)
                recall = compute_recall_at_k(pq_eval_res, gt_result, k=config.k)
                dist_aware_rec = compute_distance_aware_recall_at_k(pq_eval_res, gt_result, k=config.k)

                pq_total_dc = int(pq_eval_res.distance_computations) if pq_eval_res.distance_computations is not None else 0
                pq_total_centroid_dc = int(pq_eval_res.centroid_distance_computations) if pq_eval_res.centroid_distance_computations is not None else 0
                pq_total_cand_dc = int(pq_eval_res.candidate_distance_computations) if pq_eval_res.candidate_distance_computations is not None else 0
                pq_dc = float(pq_total_dc / Q) if Q > 0 else 0.0
                pq_centroid_dc = float(pq_total_centroid_dc / Q) if Q > 0 else 0.0
                pq_cand_dc = float(pq_total_cand_dc / Q) if Q > 0 else 0.0
                peak_rss_pq = get_process_peak_rss_bytes()

                run_pq = RunMetrics(
                    index_name="ivf_pq",
                    parameters={
                        "nlist": nlist,
                        "nprobe": nprobe,
                        "M": pq_m,
                        "k_sub": pq_k_sub,
                        "compression": f"{ivf_pq_index.compression_ratio:.1f}x",
                    },
                    build_time_seconds=pq_build_time,
                    build_peak_tracemalloc_bytes=pq_build_mem,
                    index_memory_bytes=ivf_pq_index.index_memory_bytes,
                    index_memory_mib=ivf_pq_index.index_memory_mib,
                    bytes_per_vector=ivf_pq_index.bytes_per_vector,
                    per_query_latencies_ms=lats,
                    per_query_p50_ms=p50,
                    per_query_p95_ms=p95,
                    per_query_p99_ms=p99,
                    per_query_mean_ms=mean_l,
                    per_query_min_ms=min_l,
                    per_query_max_ms=max_l,
                    per_query_sample_count=sample_count,
                    batch_qps=batch_qps,
                    batch_mean_latency_ms=batch_mean_ms,
                    batch_repeat_runs=config.repeat_runs,
                    recall_at_k=recall,
                    distance_computations_per_query=pq_dc,
                    per_query_std_ms=std_l,
                    distance_aware_recall_at_k=dist_aware_rec,
                    kmeans_n_iter=ivf_pq_index.kmeans_n_iter,
                    kmeans_converged=ivf_pq_index.kmeans_converged,
                    kmeans_inertia=ivf_pq_index.kmeans_inertia,
                    peak_process_rss_bytes=peak_rss_pq,
                    total_distance_computations=pq_total_dc,
                    total_centroid_distance_computations=pq_total_centroid_dc,
                    total_candidate_distance_computations=pq_total_cand_dc,
                    centroid_distance_computations_per_query=pq_centroid_dc,
                    candidate_distance_computations_per_query=pq_cand_dc,
                )
                result.runs.append(run_pq)

                if verbose:
                    print(
                        f"    IVF-PQ nprobe={nprobe:<3d} -> Recall@{config.k}={recall:.4f} (DistRec={dist_aware_rec:.4f}) | "
                        f"p50={p50:6.2f}ms (std={std_l:5.2f}ms) | QPS={batch_qps:7.1f} | "
                        f"Mem={ivf_pq_index.index_memory_mib:.2f}MB ({ivf_pq_index.compression_ratio:.1f}x compression)"
                    )

    # 5. Reference Library Comparison Adapters (Phase 7)
    if config.compare_faiss:
        from indexora.adapters import FaissFlatIndex, FaissIVFFlatIndex, is_faiss_available

        if not is_faiss_available():
            if verbose:
                print("\n[Warning] FAISS requested for comparison but is not installed. Skipping.")
        else:
            if verbose:
                print("\n[Comparison 1/2] Benchmarking FAISS Flat Index...")
            t0_ff_b = time.perf_counter()
            faiss_flat_idx = FaissFlatIndex(database=dataset.base, ids=dataset.base_ids)
            ff_build_time = time.perf_counter() - t0_ff_b

            lats_ff, p50_ff, p95_ff, p99_ff, mean_ff, std_ff, min_ff, max_ff, cnt_ff = (
                _measure_per_query_latency(
                    search_single_fn=lambda q, k: faiss_flat_idx.search(q, k=k),
                    queries_subset=queries_latency_sample,
                    k=config.k,
                    warmup_runs=config.warmup_runs,
                )
            )

            qps_ff, mean_batch_ff = _measure_batch_throughput(
                search_batch_fn=lambda q, k: faiss_flat_idx.search(q, k=k),
                queries=dataset.queries,
                k=config.k,
                warmup_runs=config.warmup_runs,
                repeat_runs=config.repeat_runs,
            )

            ff_res = faiss_flat_idx.search(dataset.queries, k=config.k)
            recall_ff = compute_recall_at_k(ff_res, gt_result, k=config.k)
            dist_rec_ff = compute_distance_aware_recall_at_k(ff_res, gt_result, k=config.k)
            peak_rss_ff = get_process_peak_rss_bytes()

            ff_metrics = RunMetrics(
                index_name="faiss_flat",
                parameters={},
                build_time_seconds=ff_build_time,
                build_peak_tracemalloc_bytes=0,
                index_memory_bytes=faiss_flat_idx.index_memory_bytes,
                index_memory_mib=faiss_flat_idx.index_memory_mib,
                bytes_per_vector=faiss_flat_idx.bytes_per_vector,
                per_query_latencies_ms=lats_ff,
                per_query_p50_ms=p50_ff,
                per_query_p95_ms=p95_ff,
                per_query_p99_ms=p99_ff,
                per_query_mean_ms=mean_ff,
                per_query_min_ms=min_ff,
                per_query_max_ms=max_ff,
                per_query_sample_count=cnt_ff,
                batch_qps=qps_ff,
                batch_mean_latency_ms=mean_batch_ff,
                batch_repeat_runs=config.repeat_runs,
                recall_at_k=recall_ff,
                distance_computations_per_query=float(N),
                per_query_std_ms=std_ff,
                distance_aware_recall_at_k=dist_rec_ff,
                peak_process_rss_bytes=peak_rss_ff,
                total_distance_computations=Q * N,
                candidate_distance_computations_per_query=float(N),
            )
            result.runs.append(ff_metrics)
            if verbose:
                print(
                    f"  FAISS Flat:        p50={p50_ff:.2f}ms (std={std_ff:.2f}ms) | QPS={qps_ff:.1f} | "
                    f"Recall={recall_ff:.4f} | DistRec={dist_rec_ff:.4f}"
                )

            # FAISS IVF sweeps
            for nlist in config.nlist_values:
                if nlist > N:
                    continue
                if verbose:
                    print(f"  Building FAISS IVF-Flat (nlist={nlist})...")
                t0_fivf_b = time.perf_counter()
                faiss_ivf_idx = FaissIVFFlatIndex(
                    database=dataset.base,
                    nlist=nlist,
                    ids=dataset.base_ids,
                    seed=config.seed,
                )
                fivf_build_time = time.perf_counter() - t0_fivf_b

                for nprobe in config.nprobe_values:
                    if nprobe > nlist:
                        continue
                    lats_fivf, p50_fivf, p95_fivf, p99_fivf, mean_fivf, std_fivf, min_fivf, max_fivf, cnt_fivf = (
                        _measure_per_query_latency(
                            search_single_fn=lambda q, k: faiss_ivf_idx.search(q, k=k, nprobe=nprobe),
                            queries_subset=queries_latency_sample,
                            k=config.k,
                            warmup_runs=config.warmup_runs,
                        )
                    )

                    qps_fivf, mean_batch_fivf = _measure_batch_throughput(
                        search_batch_fn=lambda q, k: faiss_ivf_idx.search(q, k=k, nprobe=nprobe),
                        queries=dataset.queries,
                        k=config.k,
                        warmup_runs=config.warmup_runs,
                        repeat_runs=config.repeat_runs,
                    )

                    fivf_res = faiss_ivf_idx.search(dataset.queries, k=config.k, nprobe=nprobe)
                    recall_fivf = compute_recall_at_k(fivf_res, gt_result, k=config.k)
                    dist_rec_fivf = compute_distance_aware_recall_at_k(fivf_res, gt_result, k=config.k)
                    peak_rss_fivf = get_process_peak_rss_bytes()

                    fivf_metrics = RunMetrics(
                        index_name="faiss_ivf_flat",
                        parameters={"nlist": nlist, "nprobe": nprobe},
                        build_time_seconds=fivf_build_time,
                        build_peak_tracemalloc_bytes=0,
                        index_memory_bytes=faiss_ivf_idx.index_memory_bytes,
                        index_memory_mib=faiss_ivf_idx.index_memory_mib,
                        bytes_per_vector=faiss_ivf_idx.bytes_per_vector,
                        per_query_latencies_ms=lats_fivf,
                        per_query_p50_ms=p50_fivf,
                        per_query_p95_ms=p95_fivf,
                        per_query_p99_ms=p99_fivf,
                        per_query_mean_ms=mean_fivf,
                        per_query_min_ms=min_fivf,
                        per_query_max_ms=max_fivf,
                        per_query_sample_count=cnt_fivf,
                        batch_qps=qps_fivf,
                        batch_mean_latency_ms=mean_batch_fivf,
                        batch_repeat_runs=config.repeat_runs,
                        recall_at_k=recall_fivf,
                        distance_computations_per_query=0.0,
                        per_query_std_ms=std_fivf,
                        distance_aware_recall_at_k=dist_rec_fivf,
                        peak_process_rss_bytes=peak_rss_fivf,
                    )
                    result.runs.append(fivf_metrics)
                    if verbose:
                        print(
                            f"    FAISS IVF nprobe={nprobe:<3d} -> Recall@{config.k}={recall_fivf:.4f} | "
                            f"p50={p50_fivf:6.2f}ms | QPS={qps_fivf:7.1f}"
                        )

    if config.compare_hnswlib:
        from indexora.adapters import HnswlibIndex, is_hnswlib_available

        if not is_hnswlib_available():
            if verbose:
                print("\n[Warning] hnswlib requested for comparison but is not installed. Skipping.")
        else:
            if verbose:
                print("\n[Comparison 2/2] Benchmarking hnswlib Graph Index...")
            t0_hnsw_b = time.perf_counter()
            hnsw_idx = HnswlibIndex(
                database=dataset.base,
                ids=dataset.base_ids,
                M=16,
                ef_construction=200,
                ef_search=50,
                random_seed=config.seed,
            )
            hnsw_build_time = time.perf_counter() - t0_hnsw_b

            for ef in (10, 20, 50, 100):
                hnsw_idx.set_ef_search(ef)
                lats_h, p50_h, p95_h, p99_h, mean_h, std_h, min_h, max_h, cnt_h = (
                    _measure_per_query_latency(
                        search_single_fn=lambda q, k: hnsw_idx.search(q, k=k, ef_search=ef),
                        queries_subset=queries_latency_sample,
                        k=config.k,
                        warmup_runs=config.warmup_runs,
                    )
                )

                qps_h, mean_batch_h = _measure_batch_throughput(
                    search_batch_fn=lambda q, k: hnsw_idx.search(q, k=k, ef_search=ef),
                    queries=dataset.queries,
                    k=config.k,
                    warmup_runs=config.warmup_runs,
                    repeat_runs=config.repeat_runs,
                )

                h_res = hnsw_idx.search(dataset.queries, k=config.k, ef_search=ef)
                recall_h = compute_recall_at_k(h_res, gt_result, k=config.k)
                dist_rec_h = compute_distance_aware_recall_at_k(h_res, gt_result, k=config.k)
                peak_rss_h = get_process_peak_rss_bytes()

                h_metrics = RunMetrics(
                    index_name="hnswlib",
                    parameters={"ef_search": ef, "M": 16, "ef_construction": 200},
                    build_time_seconds=hnsw_build_time,
                    build_peak_tracemalloc_bytes=0,
                    index_memory_bytes=hnsw_idx.index_memory_bytes,
                    index_memory_mib=hnsw_idx.index_memory_mib,
                    bytes_per_vector=hnsw_idx.bytes_per_vector,
                    per_query_latencies_ms=lats_h,
                    per_query_p50_ms=p50_h,
                    per_query_p95_ms=p95_h,
                    per_query_p99_ms=p99_h,
                    per_query_mean_ms=mean_h,
                    per_query_min_ms=min_h,
                    per_query_max_ms=max_h,
                    per_query_sample_count=cnt_h,
                    batch_qps=qps_h,
                    batch_mean_latency_ms=mean_batch_h,
                    batch_repeat_runs=config.repeat_runs,
                    recall_at_k=recall_h,
                    distance_computations_per_query=0.0,
                    per_query_std_ms=std_h,
                    distance_aware_recall_at_k=dist_rec_h,
                    peak_process_rss_bytes=peak_rss_h,
                )
                result.runs.append(h_metrics)
                if verbose:
                    print(
                        f"    hnswlib ef={ef:<3d} -> Recall@{config.k}={recall_h:.4f} | "
                        f"p50={p50_h:6.2f}ms | QPS={qps_h:7.1f}"
                    )

    if verbose:
        print("\n" + "=" * 105)
        print("BENCHMARK SWEEP COMPLETE -- SUMMARY TABLE")
        print("=" * 105)
        print(
            f"{'Index':<16} {'nlist':<6} {'nprobe':<7} {'Recall@k':<9} {'DistRec':<9} "
            f"{'p50(ms)':<8} {'std(ms)':<8} {'QPS':<7} {'Dist/Q':<8} {'PeakMB':<7}"
        )
        print("-" * 105)
        for r in result.runs:
            nl_str = str(r.parameters.get("nlist", "-"))
            np_str = str(r.parameters.get("nprobe", "-"))
            peak_mb_str = f"{r.peak_rss_mib:.1f}" if r.peak_process_rss_bytes is not None else "-"
            print(
                f"{r.index_name:<16} {nl_str:<6} {np_str:<7} {r.recall_at_k:<9.4f} {r.distance_aware_recall_at_k:<9.4f} "
                f"{r.per_query_p50_ms:<8.2f} {r.per_query_std_ms:<8.2f} {r.batch_qps:<7.1f} "
                f"{r.distance_computations_per_query:<8.1f} {peak_mb_str:<7}"
            )
        print("=" * 105)

    return result


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for running the Indexora benchmark harness."""
    parser = argparse.ArgumentParser(description="Indexora Vector Search Benchmark Harness v0.1")
    parser.add_argument("--smoke", action="store_true", help="Run a quick synthetic smoke test sweep")
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "sift_fvecs"],
        help="Dataset type to benchmark ('synthetic' or 'sift_fvecs')",
    )
    parser.add_argument("--base-path", type=str, default=None, help="Path to base .fvecs file for sift_fvecs")
    parser.add_argument("--query-path", type=str, default=None, help="Path to query .fvecs file for sift_fvecs")
    parser.add_argument("--output-dir", type=str, default="benchmark_output", help="Directory for JSON/CSV/plots")
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors to retrieve")
    parser.add_argument("--n-base", type=int, default=10000, help="Number of base database vectors")
    parser.add_argument("--n-queries", type=int, default=100, help="Number of query vectors")
    parser.add_argument("--d", type=int, default=32, help="Vector feature dimension (for synthetic dataset)")
    parser.add_argument("--nlist", type=str, default="10,50,100", help="Comma-separated nlist values")
    parser.add_argument("--nprobe", type=str, default="1,2,5,10,20", help="Comma-separated nprobe values")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--repeats", type=int, default=5, help="Number of batch timing iterations")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup iterations")
    parser.add_argument(
        "--latency-samples",
        type=int,
        default=500,
        help="Number of individual queries sampled for per-query latency percentiles",
    )
    parser.add_argument(
        "--compare-faiss",
        action="store_true",
        help="Include FAISS flat and IVF-flat baselines if faiss is installed",
    )
    parser.add_argument(
        "--compare-hnswlib",
        action="store_true",
        help="Include hnswlib baseline if hnswlib is installed",
    )
    parser.add_argument(
        "--ivf-pq",
        action="store_true",
        help="Include IVF-PQ index sweep in benchmark",
    )
    parser.add_argument("--pq-m", type=int, default=8, help="Number of subspaces M for IVF-PQ")
    parser.add_argument("--pq-ksub", type=int, default=256, help="Centroids per subspace k_sub for IVF-PQ")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating plot images")

    args = parser.parse_args(argv)

    extra_metadata: dict[str, Any] = {}

    if args.smoke:
        # Fast smoke configuration: N=200, Q=20, d=8, nlist=5, nprobe=[1, 5]
        dataset = generate_clustered_dataset(
            n_base=200,
            n_queries=20,
            d=8,
            n_clusters=5,
            seed=args.seed,
        ).dataset
        config = BenchmarkConfig(
            name="smoke_test",
            k=5,
            warmup_runs=1,
            repeat_runs=2,
            latency_samples=10,
            seed=args.seed,
            nlist_values=(5,),
            nprobe_values=(1, 5),
            smoke=True,
            dataset_type="synthetic",
            compare_faiss=args.compare_faiss,
            compare_hnswlib=args.compare_hnswlib,
            include_ivf_pq=args.ivf_pq,
            ivf_pq_m=args.pq_m,
            ivf_pq_k_sub=args.pq_ksub,
        )
        dataset_name = "Synthetic Smoke Clustered"
    elif args.dataset == "sift_fvecs":
        if not args.base_path or not args.query_path:
            sys.stderr.write(
                "Error: --dataset sift_fvecs requires both --base-path and --query-path.\n"
                "Example:\n"
                "  python -m indexora.benchmark --dataset sift_fvecs --base-path data/sift_base.fvecs "
                "--query-path data/sift_query.fvecs --n-base 100000 --n-queries 1000\n"
            )
            return 1

        n_base_req = args.n_base if args.n_base != 10000 else 100000
        n_queries_req = args.n_queries if args.n_queries != 100 else 1000

        try:
            dataset = load_sift_dataset(
                base_path=args.base_path,
                query_path=args.query_path,
                n_base=n_base_req,
                n_queries=n_queries_req,
            )
        except (FileNotFoundError, ValueError) as err:
            sys.stderr.write(f"Error loading SIFT dataset:\n{err}\n")
            return 1
        extra_metadata["base_sha256"] = compute_file_sha256(args.base_path)
        extra_metadata["query_sha256"] = compute_file_sha256(args.query_path)
        extra_metadata["source_type"] = "sift_fvecs"
        extra_metadata["provenance"] = "Texmex SIFT1M dataset subset prefix"

        nl_vals = tuple(int(x.strip()) for x in args.nlist.split(",") if x.strip())
        np_vals = tuple(int(x.strip()) for x in args.nprobe.split(",") if x.strip())
        config = BenchmarkConfig(
            name="sift_100k_benchmark",
            k=args.k,
            warmup_runs=args.warmup,
            repeat_runs=args.repeats,
            latency_samples=args.latency_samples,
            seed=args.seed,
            nlist_values=nl_vals,
            nprobe_values=np_vals,
            smoke=False,
            dataset_type="sift_fvecs",
            base_path=args.base_path,
            query_path=args.query_path,
            compare_faiss=args.compare_faiss,
            compare_hnswlib=args.compare_hnswlib,
            include_ivf_pq=args.ivf_pq,
            ivf_pq_m=args.pq_m,
            ivf_pq_k_sub=args.pq_ksub,
        )
        dataset_name = f"SIFT-{n_base_req // 1000}k Subset"
    else:
        dataset = generate_clustered_dataset(
            n_base=args.n_base,
            n_queries=args.n_queries,
            d=args.d,
            n_clusters=max(5, int(args.n_base // 100)),
            seed=args.seed,
        ).dataset
        extra_metadata["base_hash"] = hashlib.sha256(dataset.base.tobytes()).hexdigest()
        extra_metadata["query_hash"] = hashlib.sha256(dataset.queries.tobytes()).hexdigest()
        extra_metadata["source_type"] = "synthetic_clustered"

        nl_vals = tuple(int(x.strip()) for x in args.nlist.split(",") if x.strip())
        np_vals = tuple(int(x.strip()) for x in args.nprobe.split(",") if x.strip())
        config = BenchmarkConfig(
            name="standard_sweep",
            k=args.k,
            warmup_runs=args.warmup,
            repeat_runs=args.repeats,
            latency_samples=args.latency_samples,
            seed=args.seed,
            nlist_values=nl_vals,
            nprobe_values=np_vals,
            smoke=False,
            dataset_type="synthetic",
            compare_faiss=args.compare_faiss,
            compare_hnswlib=args.compare_hnswlib,
            include_ivf_pq=args.ivf_pq,
            ivf_pq_m=args.pq_m,
            ivf_pq_k_sub=args.pq_ksub,
        )
        dataset_name = f"Synthetic Clustered (N={args.n_base:,})"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_benchmark(
        dataset=dataset,
        config=config,
        verbose=True,
        dataset_name=dataset_name,
        dataset_extra_metadata=extra_metadata,
    )

    # Save outputs
    json_path = out_dir / "benchmark_results.json"
    csv_path = out_dir / "benchmark_summary.csv"
    result.to_json(json_path)
    result.to_csv(csv_path)
    print("\n[Artifacts Saved]")
    print(f"  JSON manifest: {json_path}")
    print(f"  CSV summary:   {csv_path}")

    if not args.no_plot:
        try:
            plots = result.plot(out_dir)
            for p in plots:
                print(f"  Plot:          {p}")
        except Exception as e:
            sys.stderr.write(f"Error generating benchmark plots: {e}\n")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
