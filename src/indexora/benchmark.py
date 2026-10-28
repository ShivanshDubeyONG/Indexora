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
from indexora.eval import compute_recall_at_k
from indexora.exact import ExactFlatIndex, generate_ground_truth
from indexora.ivf import IVFFlatIndex


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration parameters for a vector search benchmark sweep."""

    name: str = "benchmark"
    k: int = 10
    warmup_runs: int = 2
    repeat_runs: int = 5
    latency_samples: int = 100
    seed: int = 42
    nlist_values: tuple[int, ...] = (10, 50, 100)
    nprobe_values: tuple[int, ...] = (1, 2, 5, 10, 20)
    smoke: bool = False
    dataset_type: str = "synthetic"
    base_path: str | None = None
    query_path: str | None = None

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

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a standard serializable dictionary."""
        d = asdict(self)
        d["per_query_latencies_ms"] = list(self.per_query_latencies_ms)
        # Ensure backward-compatible keys exist in serialized manifest
        d["p50_latency_ms"] = self.p50_latency_ms
        d["p95_latency_ms"] = self.p95_latency_ms
        d["p99_latency_ms"] = self.p99_latency_ms
        d["mean_latency_ms"] = self.mean_latency_ms
        d["min_latency_ms"] = self.min_latency_ms
        d["max_latency_ms"] = self.max_latency_ms
        d["qps"] = self.qps
        d["query_latencies_ms"] = list(self.per_query_latencies_ms)
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
            "index_mem_mib",
            "bytes_per_vec",
            "per_query_p50_ms",
            "per_query_p95_ms",
            "per_query_mean_ms",
            "batch_qps",
            "recall_at_k",
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
                        "index_mem_mib": f"{r.index_memory_mib:.2f}",
                        "bytes_per_vec": f"{r.bytes_per_vector:.1f}",
                        "per_query_p50_ms": f"{r.per_query_p50_ms:.3f}",
                        "per_query_p95_ms": f"{r.per_query_p95_ms:.3f}",
                        "per_query_mean_ms": f"{r.per_query_mean_ms:.3f}",
                        "batch_qps": f"{r.batch_qps:.1f}",
                        "recall_at_k": f"{r.recall_at_k:.4f}",
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

        exact_runs = [r for r in self.runs if r.index_name == "exact"]
        ivf_runs = [r for r in self.runs if r.index_name == "ivf_flat"]

        nlist_groups: dict[int, list[RunMetrics]] = {}
        for r in ivf_runs:
            nl = int(r.parameters.get("nlist", 0))
            nlist_groups.setdefault(nl, []).append(r)

        ds_name = self.dataset_metadata.get("name", "Dataset")
        n_base = self.dataset_metadata.get("n_base", 0)
        d_dim = self.dataset_metadata.get("d", 0)
        k_val = self.config.get("k", 10)

        # 1. Plot: Recall@k vs Per-Query p50 Latency (ms)
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for nl, group in sorted(nlist_groups.items()):
            group_sorted = sorted(group, key=lambda x: int(x.parameters.get("nprobe", 0)))
            latencies = [x.per_query_p50_ms for x in group_sorted]
            recalls = [x.recall_at_k for x in group_sorted]
            ax.plot(latencies, recalls, marker="o", label=f"IVF-Flat (nlist={nl})")
            for x, y, run in zip(latencies, recalls, group_sorted):
                p = run.parameters.get("nprobe", "")
                ax.annotate(f"p={p}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

        if exact_runs:
            ex = exact_runs[0]
            ax.scatter(
                [ex.per_query_p50_ms],
                [ex.recall_at_k],
                color="red",
                s=130,
                marker="*",
                label=f"Exact Flat (p50={ex.per_query_p50_ms:.2f}ms)",
                zorder=5,
            )

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
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for nl, group in sorted(nlist_groups.items()):
            group_sorted = sorted(group, key=lambda x: int(x.parameters.get("nprobe", 0)))
            dcs = [x.distance_computations_per_query for x in group_sorted]
            recalls = [x.recall_at_k for x in group_sorted]
            ax.plot(dcs, recalls, marker="s", label=f"IVF-Flat (nlist={nl})")
            for x, y, run in zip(dcs, recalls, group_sorted):
                p = run.parameters.get("nprobe", "")
                ax.annotate(f"p={p}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

        if exact_runs:
            ex = exact_runs[0]
            ax.scatter(
                [ex.distance_computations_per_query],
                [ex.recall_at_k],
                color="red",
                s=130,
                marker="*",
                label=f"Exact Flat (N={int(ex.distance_computations_per_query):,})",
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
) -> tuple[tuple[float, ...], float, float, float, float, float, float, int]:
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
    min_val = float(np.min(lat_arr))
    max_val = float(np.max(lat_arr))

    return tuple(latencies_ms), p50, p95, p99, mean_val, min_val, max_val, num_samples


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

    # Resolve latency query sample
    n_latency_samples = min(config.latency_samples, Q)
    queries_latency_sample = dataset.queries[:n_latency_samples]

    # 2. Exact Flat Baseline Benchmark
    if verbose:
        print("\n[2/3] Benchmarking Exact Flat Index...")
    tracemalloc.start()
    t0_b = time.perf_counter()
    exact_index = ExactFlatIndex(database=dataset.base, ids=dataset.base_ids)
    exact_build_time = time.perf_counter() - t0_b
    _, exact_build_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Per-query latency percentiles
    lats, p50, p95, p99, mean_l, min_l, max_l, sample_count = _measure_per_query_latency(
        search_single_fn=lambda q, k: exact_index.search(q, k=k),
        queries_subset=queries_latency_sample,
        k=config.k,
        warmup_runs=config.warmup_runs,
    )

    # Batch throughput
    batch_qps, batch_mean_ms = _measure_batch_throughput(
        search_batch_fn=lambda q, k: exact_index.search(q, k=k),
        queries=dataset.queries,
        k=config.k,
        warmup_runs=config.warmup_runs,
        repeat_runs=config.repeat_runs,
    )

    # Exact search evaluation against ground truth
    exact_res = exact_index.search(dataset.queries, k=config.k)
    exact_recall = compute_recall_at_k(exact_res, gt_result, k=config.k)
    exact_total_dc = int(exact_res.distance_computations) if exact_res.distance_computations is not None else Q * N
    exact_total_centroid_dc = int(exact_res.centroid_distance_computations) if exact_res.centroid_distance_computations is not None else 0
    exact_total_cand_dc = int(exact_res.candidate_distance_computations) if exact_res.candidate_distance_computations is not None else exact_total_dc
    exact_dc = float(exact_total_dc / Q) if Q > 0 else 0.0
    exact_centroid_dc = float(exact_total_centroid_dc / Q) if Q > 0 else 0.0
    exact_cand_dc = float(exact_total_cand_dc / Q) if Q > 0 else 0.0

    exact_metrics = RunMetrics(
        index_name="exact",
        parameters={},
        build_time_seconds=exact_build_time,
        build_peak_tracemalloc_bytes=exact_build_mem,
        index_memory_bytes=exact_index.index_memory_bytes,
        index_memory_mib=exact_index.index_memory_mib,
        bytes_per_vector=exact_index.bytes_per_vector,
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
        recall_at_k=exact_recall,
        distance_computations_per_query=exact_dc,
        total_distance_computations=exact_total_dc,
        total_centroid_distance_computations=exact_total_centroid_dc,
        total_candidate_distance_computations=exact_total_cand_dc,
        centroid_distance_computations_per_query=exact_centroid_dc,
        candidate_distance_computations_per_query=exact_cand_dc,
    )
    result.runs.append(exact_metrics)

    if verbose:
        print(
            f"  Exact Flat: p50={p50:.2f}ms | QPS={batch_qps:.1f} | Recall={exact_recall:.4f} | "
            f"DC/Q={exact_dc:.0f} (Cand={exact_cand_dc:.0f}) | Storage={exact_index.index_memory_mib:.2f}MiB"
        )

    # 3. IVF-Flat Sweeps
    if verbose:
        print("\n[3/3] Benchmarking IVF-Flat Parameter Sweeps...")

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

        for nprobe in config.nprobe_values:
            if nprobe > nlist:
                continue

            # Per-query latency percentiles
            lats, p50, p95, p99, mean_l, min_l, max_l, sample_count = _measure_per_query_latency(
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
            ivf_total_dc = int(search_res.distance_computations) if search_res.distance_computations is not None else 0
            ivf_total_centroid_dc = int(search_res.centroid_distance_computations) if search_res.centroid_distance_computations is not None else 0
            ivf_total_cand_dc = int(search_res.candidate_distance_computations) if search_res.candidate_distance_computations is not None else 0
            ivf_dc = float(ivf_total_dc / Q) if Q > 0 else 0.0
            ivf_centroid_dc = float(ivf_total_centroid_dc / Q) if Q > 0 else 0.0
            ivf_cand_dc = float(ivf_total_cand_dc / Q) if Q > 0 else 0.0

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
                total_distance_computations=ivf_total_dc,
                total_centroid_distance_computations=ivf_total_centroid_dc,
                total_candidate_distance_computations=ivf_total_cand_dc,
                centroid_distance_computations_per_query=ivf_centroid_dc,
                candidate_distance_computations_per_query=ivf_cand_dc,
            )
            result.runs.append(run_m)

            if verbose:
                print(
                    f"    nprobe={nprobe:<3d} -> Recall@{config.k}={recall:.4f} | "
                    f"p50={p50:6.2f}ms | QPS={batch_qps:7.1f} | DC/Q={ivf_dc:6.0f} (Cent={ivf_centroid_dc:4.0f}, Cand={ivf_cand_dc:4.0f})"
                )

    if verbose:
        print("\n" + "=" * 90)
        print("BENCHMARK SWEEP COMPLETE -- SUMMARY TABLE")
        print("=" * 90)
        print(
            f"{'Index':<10} {'nlist':<6} {'nprobe':<7} {'Recall@k':<9} "
            f"{'p50 (ms)':<9} {'QPS':<8} {'Dist/Q':<8} {'Cent/Q':<7} {'Cand/Q':<7} {'Index MiB':<9}"
        )
        print("-" * 90)
        for r in result.runs:
            nl_str = str(r.parameters.get("nlist", "-"))
            np_str = str(r.parameters.get("nprobe", "-"))
            print(
                f"{r.index_name:<10} {nl_str:<6} {np_str:<7} {r.recall_at_k:<9.4f} "
                f"{r.per_query_p50_ms:<9.2f} {r.batch_qps:<8.1f} "
                f"{r.distance_computations_per_query:<8.1f} {r.centroid_distance_computations_per_query:<7.1f} "
                f"{r.candidate_distance_computations_per_query:<7.1f} {r.index_memory_mib:<9.2f}"
            )
        print("=" * 90)

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
        default=100,
        help="Number of individual queries sampled for per-query latency percentiles",
    )
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
