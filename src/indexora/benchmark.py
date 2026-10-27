"""Benchmark harness v0.1 for Indexora.

This module provides reproducible benchmarking of exact search and IVF-Flat sweeps:
- Ground-truth generation via exact flat search.
- Multi-probe IVF-Flat sweeps over configurable `nlist` and `nprobe`.
- Standard metrics: Recall@k, p50/p95/mean latency, QPS, distance computations, build time.
- Machine-readable JSON and CSV export with full environment and reproducibility metadata.
- Automated plot generation for Recall vs Latency and Recall vs Distance Computations.
- CLI entry point with a fast `--smoke` test mode.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import tracemalloc
from typing import Any, Sequence

import numpy as np

from indexora.data import VectorDataset, generate_clustered_dataset, generate_synthetic_dataset
from indexora.eval import compute_recall_at_k, count_distance_computations
from indexora.exact import ExactFlatIndex, generate_ground_truth
from indexora.ivf import IVFFlatIndex


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration parameters for a vector search benchmark sweep."""

    name: str = "benchmark"
    k: int = 10
    warmup_runs: int = 1
    repeat_runs: int = 5
    seed: int = 42
    nlist_values: tuple[int, ...] = (10, 50, 100)
    nprobe_values: tuple[int, ...] = (1, 2, 5, 10, 20)
    smoke: bool = False

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")
        if self.warmup_runs < 0:
            raise ValueError(f"warmup_runs must be non-negative, got {self.warmup_runs}")
        if self.repeat_runs < 1:
            raise ValueError(f"repeat_runs must be at least 1, got {self.repeat_runs}")
        if not self.nlist_values:
            raise ValueError("nlist_values cannot be empty")
        if not self.nprobe_values:
            raise ValueError("nprobe_values cannot be empty")


@dataclass(frozen=True)
class RunMetrics:
    """Measured performance and accuracy metrics for an individual index configuration."""

    index_name: str
    parameters: dict[str, Any]
    build_time_seconds: float
    build_peak_tracemalloc_bytes: int
    query_latencies_ms: tuple[float, ...]
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    mean_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    qps: float
    recall_at_k: float
    distance_computations_per_query: float

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a standard serializable dictionary."""
        d = asdict(self)
        d["query_latencies_ms"] = list(self.query_latencies_ms)
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
            "p50_latency_ms",
            "p95_latency_ms",
            "mean_latency_ms",
            "qps",
            "recall_at_k",
            "distance_computations",
        ]

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.runs:
                params = r.parameters
                writer.writerow(
                    {
                        "index_name": r.index_name,
                        "nlist": params.get("nlist", ""),
                        "nprobe": params.get("nprobe", ""),
                        "build_time_sec": f"{r.build_time_seconds:.4f}",
                        "build_peak_mem_mb": f"{r.build_peak_tracemalloc_bytes / (1024 * 1024):.2f}",
                        "p50_latency_ms": f"{r.p50_latency_ms:.3f}",
                        "p95_latency_ms": f"{r.p95_latency_ms:.3f}",
                        "mean_latency_ms": f"{r.mean_latency_ms:.3f}",
                        "qps": f"{r.qps:.1f}",
                        "recall_at_k": f"{r.recall_at_k:.4f}",
                        "distance_computations": f"{r.distance_computations_per_query:.1f}",
                    }
                )

    def plot(self, output_dir: str | Path) -> list[str]:
        """Generate headline benchmark plots (Recall vs Latency and Recall vs Distance Computations).

        Returns
        -------
        list[str]
            List of generated plot image file paths. Returns empty list if matplotlib
            is not installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return []

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        generated_plots = []

        # Separate runs by index type
        exact_runs = [r for r in self.runs if r.index_name == "exact"]
        ivf_runs = [r for r in self.runs if r.index_name == "ivf_flat"]

        # Group IVF runs by nlist
        nlist_groups: dict[int, list[RunMetrics]] = {}
        for r in ivf_runs:
            nl = int(r.parameters.get("nlist", 0))
            nlist_groups.setdefault(nl, []).append(r)

        # 1. Plot: Recall@k vs p50 Latency (ms)
        fig, ax = plt.subplots(figsize=(8, 5))
        for nl, group in sorted(nlist_groups.items()):
            # Sort group by nprobe
            group_sorted = sorted(group, key=lambda x: int(x.parameters.get("nprobe", 0)))
            latencies = [x.p50_latency_ms for x in group_sorted]
            recalls = [x.recall_at_k for x in group_sorted]
            ax.plot(latencies, recalls, marker="o", label=f"IVF-Flat (nlist={nl})")
            for x, y, run in zip(latencies, recalls, group_sorted):
                p = run.parameters.get("nprobe", "")
                ax.annotate(f"p={p}", (x, y), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)

        if exact_runs:
            ex = exact_runs[0]
            ax.scatter(
                [ex.p50_latency_ms],
                [ex.recall_at_k],
                color="red",
                s=120,
                marker="*",
                label="Exact Flat (Baseline)",
                zorder=5,
            )

        ax.set_title(f"Recall@{self.config.get('k', 10)} vs. p50 Query Latency")
        ax.set_xlabel("p50 Batch Query Latency (ms)")
        ax.set_ylabel(f"Recall@{self.config.get('k', 10)}")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        plt.tight_layout()

        plot1_path = out_dir / "recall_vs_latency.png"
        fig.savefig(plot1_path, dpi=150)
        plt.close(fig)
        generated_plots.append(str(plot1_path))

        # 2. Plot: Recall@k vs Distance Computations per Query
        fig, ax = plt.subplots(figsize=(8, 5))
        for nl, group in sorted(nlist_groups.items()):
            group_sorted = sorted(group, key=lambda x: int(x.parameters.get("nprobe", 0)))
            dcs = [x.distance_computations_per_query for x in group_sorted]
            recalls = [x.recall_at_k for x in group_sorted]
            ax.plot(dcs, recalls, marker="s", label=f"IVF-Flat (nlist={nl})")
            for x, y, run in zip(dcs, recalls, group_sorted):
                p = run.parameters.get("nprobe", "")
                ax.annotate(f"p={p}", (x, y), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)

        if exact_runs:
            ex = exact_runs[0]
            ax.scatter(
                [ex.distance_computations_per_query],
                [ex.recall_at_k],
                color="red",
                s=120,
                marker="*",
                label=f"Exact Flat (N={int(ex.distance_computations_per_query)})",
                zorder=5,
            )

        ax.set_title(f"Recall@{self.config.get('k', 10)} vs. Distance Computations / Query")
        ax.set_xlabel("Distance Computations per Query")
        ax.set_ylabel(f"Recall@{self.config.get('k', 10)}")
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
    """Collect platform, Python, NumPy, and git environment metadata."""
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

    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "git_commit": git_commit,
    }


def _measure_search_latency(
    search_fn: Any,
    queries: np.ndarray,
    k: int,
    warmup_runs: int,
    repeat_runs: int,
) -> tuple[tuple[float, ...], float, float, float, float, float, float, float]:
    """Execute warmup and repeated search runs, computing detailed summary statistics."""
    # Warmup runs
    for _ in range(warmup_runs):
        _ = search_fn(queries, k=k)

    # Measured runs
    latencies_ms: list[float] = []
    for _ in range(repeat_runs):
        t0 = time.perf_counter()
        _ = search_fn(queries, k=k)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    lat_arr = np.array(latencies_ms, dtype=np.float64)
    p50 = float(np.percentile(lat_arr, 50))
    p95 = float(np.percentile(lat_arr, 95))
    p99 = float(np.percentile(lat_arr, 99))
    mean_val = float(np.mean(lat_arr))
    min_val = float(np.min(lat_arr))
    max_val = float(np.max(lat_arr))

    # QPS = total queries in batch / mean batch execution time in seconds
    num_queries = queries.shape[0] if queries.ndim == 2 else 1
    mean_sec = mean_val / 1000.0
    qps = float(num_queries / mean_sec) if mean_sec > 0.0 else 0.0

    return tuple(latencies_ms), p50, p95, p99, mean_val, min_val, max_val, qps


def run_benchmark(
    dataset: VectorDataset,
    config: BenchmarkConfig,
    verbose: bool = True,
) -> BenchmarkResult:
    """Run exact search and IVF-Flat benchmark sweeps on a given dataset.

    Parameters
    ----------
    dataset : VectorDataset
        Input dataset containing `.base` and `.queries` splits.
    config : BenchmarkConfig
        Benchmark configuration and hyperparameter sweep parameters.
    verbose : bool, default=True
        Whether to print progress and ASCII results table to stdout.

    Returns
    -------
    BenchmarkResult
        Container with complete metrics, metadata, and JSON/CSV/plot exporters.
    """
    N, d = dataset.base.shape
    Q = dataset.queries.shape[0]

    env_meta = get_environment_metadata()
    dataset_meta = {
        "n_base": N,
        "n_queries": Q,
        "d": d,
        "seed": config.seed,
    }

    result = BenchmarkResult(
        environment=env_meta,
        dataset_metadata=dataset_meta,
        config=asdict(config),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    if verbose:
        print("=" * 80)
        print(f"INDEXORA BENCHMARK HARNESS v0.1 -- {config.name.upper()}")
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

    # 2. Exact Flat Baseline Benchmark
    if verbose:
        print("\n[2/3] Benchmarking Exact Flat Index...")
    tracemalloc.start()
    t0_b = time.perf_counter()
    exact_index = ExactFlatIndex(database=dataset.base, ids=dataset.base_ids)
    exact_build_time = time.perf_counter() - t0_b
    _, exact_build_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    lats, p50, p95, p99, mean_l, min_l, max_l, qps = _measure_search_latency(
        search_fn=exact_index.search,
        queries=dataset.queries,
        k=config.k,
        warmup_runs=config.warmup_runs,
        repeat_runs=config.repeat_runs,
    )

    exact_dc = count_distance_computations(exact_index, dataset.queries)
    exact_metrics = RunMetrics(
        index_name="exact",
        parameters={},
        build_time_seconds=exact_build_time,
        build_peak_tracemalloc_bytes=exact_build_mem,
        query_latencies_ms=lats,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        p99_latency_ms=p99,
        mean_latency_ms=mean_l,
        min_latency_ms=min_l,
        max_latency_ms=max_l,
        qps=qps,
        recall_at_k=1.0,
        distance_computations_per_query=exact_dc,
    )
    result.runs.append(exact_metrics)

    if verbose:
        print(
            f"  Exact Flat: p50={p50:.2f}ms | QPS={qps:.1f} | Recall={exact_metrics.recall_at_k:.4f} | DC={exact_dc:.0f}"
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

            # Query latency measurement
            def ivf_search_wrapper(q: np.ndarray, k: int) -> Any:
                return ivf_index.search(q, k=k, nprobe=nprobe)

            lats, p50, p95, p99, mean_l, min_l, max_l, qps = _measure_search_latency(
                search_fn=ivf_search_wrapper,
                queries=dataset.queries,
                k=config.k,
                warmup_runs=config.warmup_runs,
                repeat_runs=config.repeat_runs,
            )

            # Evaluate search accuracy against ground truth
            search_res = ivf_index.search(dataset.queries, k=config.k, nprobe=nprobe)
            recall = compute_recall_at_k(search_res.ids, gt_result.ids, k=config.k)
            dc = count_distance_computations(ivf_index, dataset.queries, nprobe=nprobe)

            run_m = RunMetrics(
                index_name="ivf_flat",
                parameters={"nlist": nlist, "nprobe": nprobe},
                build_time_seconds=ivf_build_time,
                build_peak_tracemalloc_bytes=ivf_build_mem,
                query_latencies_ms=lats,
                p50_latency_ms=p50,
                p95_latency_ms=p95,
                p99_latency_ms=p99,
                mean_latency_ms=mean_l,
                min_latency_ms=min_l,
                max_latency_ms=max_l,
                qps=qps,
                recall_at_k=recall,
                distance_computations_per_query=dc,
            )
            result.runs.append(run_m)

            if verbose:
                print(
                    f"    nprobe={nprobe:<3d} -> Recall@{config.k}={recall:.4f} | "
                    f"p50={p50:6.2f}ms | QPS={qps:7.1f} | DC={dc:6.0f}"
                )

    if verbose:
        print("\n" + "=" * 80)
        print("BENCHMARK SWEEP COMPLETE -- SUMMARY TABLE")
        print("=" * 80)
        print(
            f"{'Index':<12} {'nlist':<7} {'nprobe':<7} {'Recall@k':<10} "
            f"{'p50 (ms)':<10} {'p95 (ms)':<10} {'QPS':<8} {'Dist Comps':<10}"
        )
        print("-" * 80)
        for r in result.runs:
            nl_str = str(r.parameters.get("nlist", "-"))
            np_str = str(r.parameters.get("nprobe", "-"))
            print(
                f"{r.index_name:<12} {nl_str:<7} {np_str:<7} {r.recall_at_k:<10.4f} "
                f"{r.p50_latency_ms:<10.2f} {r.p95_latency_ms:<10.2f} {r.qps:<8.1f} "
                f"{r.distance_computations_per_query:<10.0f}"
            )
        print("=" * 80)

    return result


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for running the Indexora benchmark harness."""
    parser = argparse.ArgumentParser(description="Indexora Vector Search Benchmark Harness v0.1")
    parser.add_argument("--smoke", action="store_true", help="Run a quick synthetic smoke test sweep")
    parser.add_argument("--output-dir", type=str, default="benchmark_output", help="Directory for JSON/CSV/plots")
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors to retrieve")
    parser.add_argument("--n-base", type=int, default=10000, help="Number of base database vectors")
    parser.add_argument("--n-queries", type=int, default=100, help="Number of query vectors")
    parser.add_argument("--d", type=int, default=32, help="Vector feature dimension")
    parser.add_argument("--nlist", type=str, default="10,50,100", help="Comma-separated nlist values")
    parser.add_argument("--nprobe", type=str, default="1,2,5,10,20", help="Comma-separated nprobe values")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--repeats", type=int, default=5, help="Number of timing iterations")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup iterations")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating plot images")

    args = parser.parse_args(argv)

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
            seed=args.seed,
            nlist_values=(5,),
            nprobe_values=(1, 5),
            smoke=True,
        )
    else:
        dataset = generate_clustered_dataset(
            n_base=args.n_base,
            n_queries=args.n_queries,
            d=args.d,
            n_clusters=max(5, int(args.n_base // 100)),
            seed=args.seed,
        ).dataset
        nl_vals = tuple(int(x.strip()) for x in args.nlist.split(",") if x.strip())
        np_vals = tuple(int(x.strip()) for x in args.nprobe.split(",") if x.strip())
        config = BenchmarkConfig(
            name="standard_sweep",
            k=args.k,
            warmup_runs=args.warmup,
            repeat_runs=args.repeats,
            seed=args.seed,
            nlist_values=nl_vals,
            nprobe_values=np_vals,
            smoke=False,
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_benchmark(dataset=dataset, config=config, verbose=True)

    # Save outputs
    json_path = out_dir / "benchmark_results.json"
    csv_path = out_dir / "benchmark_summary.csv"
    result.to_json(json_path)
    result.to_csv(csv_path)
    print(f"\n[Artifacts Saved]")
    print(f"  JSON manifest: {json_path}")
    print(f"  CSV summary:   {csv_path}")

    if not args.no_plot:
        plots = result.plot(out_dir)
        for p in plots:
            print(f"  Plot:          {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
