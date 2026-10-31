# Indexora — Benchmark Methodology & Provenance

This document provides a comprehensive, transparent reference for the benchmarking methodology, mathematical formulations, measurement protocols, environment specifications, and provenance data behind Indexora's evaluation suites.

---

## 1. Executive Summary & Conceptual Distinction

Indexora maintains two distinct evaluation tiers. It is critical not to confuse or conflate their operational purposes:

| Evaluation Tier | Dataset & Scale | Primary Objective | Environment & Execution |
| :--- | :--- | :--- | :--- |
| **Interactive Index Lab** | Synthetic Clustered ($N=120$, $d=8$, $c=4$) | Interactive algorithm inspection, 2D PCA visualization, mutation verification, and immediate pedagogical feedback. | Executes dynamically in-browser via Streamlit (`src/indexora/ui.py`). Parameters are user-controlled and results are not representative of large-scale production workloads. |
| **SIFT-100K Benchmark** | Canonical SIFT1M Prefix ($N=100,000$, $Q=1,000$, $d=128$) | Rigorous empirical measurement of recall, per-query latency percentiles, throughput (QPS), distance operations, and memory footprint. | Precomputed offline batch runs under strictly recorded hardware and environment specifications. |

> [!IMPORTANT]
> The Streamlit Interactive Lab does **not** execute the SIFT-100K benchmark in real time. The SIFT-100K tab visualizes precomputed, cryptographically verified historical benchmark artifacts recorded in `results/sift100k_p66/`.

---

## 2. Dataset Specifications & Cryptographic Provenance

The primary scale benchmark is conducted on the standard **SIFT-100k** subset, derived from the canonical INRIA Texmex SIFT1M benchmark dataset (128-dimensional SIFT descriptors extracted from real-world photographic images).

### Dataset Metadata (Verified from `results/sift100k_p66/benchmark_results.json`)

- **Dataset Name:** SIFT-100k Subset (`sift_fvecs`)
- **Provenance:** First 100,000 base vectors and 1,000 queries from the Texmex SIFT1M corpus
- **Base Vector Count ($N$):** 100,000
- **Query Vector Count ($Q$):** 1,000
- **Dimensionality ($d$):** 128
- **Element Data Type:** `np.float32` (little-endian IEEE 754 single precision)
- **Distance Metric:** Squared Euclidean Distance ($\|q - x\|_2^2$)
- **Base Vectors SHA-256 Digest:**
  `21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816`
- **Query Vectors SHA-256 Digest:**
  `f7fc9be140accdfd64116c2fa2365ecdb69b8f084970c6b0532db5ff79ac8fdc`
- **Entity ID Policy:** Sequential 64-bit signed integers (`np.int64`), $0 \le \text{id} < 100,000$

---

## 3. Ground-Truth Generation & Deterministic Tie-Breaking

To evaluate Approximate Nearest Neighbor (ANN) search without bias, ground truth is established using exhaustive brute-force search over all $N=100,000$ vectors for each of the $Q=1,000$ queries.

1. **Exhaustive Scan:** Computed using Indexora's `exact_reference` kernel with float64 accumulation to prevent numerical catastrophic cancellation.
2. **Top-k Extraction:** Configured at $k=10$.
3. **Deterministic Tie-Breaking (ADR-0001 / ADR-0002):**
   When multiple candidates produce identical squared Euclidean distances (or differ by less than IEEE 754 precision $\epsilon$), ties are resolved strictly in ascending numerical order of their 64-bit external entity IDs:
   $$\text{sort\_key} = (\text{distance}, \text{external\_id})$$
   This guarantees 100% deterministic, bit-for-bit reproducible ground truth across diverse platforms and OS architectures.

---

## 4. Evaluation Metrics & Mathematical Formulation

### 4.1 Standard Recall@k
Measures the proportion of true ground-truth top-$k$ nearest neighbors successfully retrieved by an index:
$$\text{Recall}@k = \frac{|\text{Retrieved IDs} \cap \text{Ground Truth IDs}|}{k}$$
- Evaluated strictly using result validity masks (`SearchResult.mask`).
- Padding sentinel slots (used when fewer than $k$ candidates exist) are excluded from the denominator.
- Valid external entity IDs (including negative integers) are treated as eligible hits.

### 4.2 Distance-Aware Recall@k
Standard set-intersection recall can artificially penalize an approximate index when multiple vectors share identical distances at the rank-$k$ boundary. Distance-Aware Recall counts a retrieved candidate as a valid hit if its distance satisfies:
$$d_i \le d_k^{\text{GT}} + \epsilon \quad (\epsilon = 1 \times 10^{-5})$$
where $d_k^{\text{GT}}$ is the distance of the $k$-th ground-truth neighbor.

> [!NOTE]
> Distance-Aware Recall applies to indexes that calculate exact candidate distances (Exact Flat, IVF-Flat). It is not applicable to un-reranked quantized codes (IVF-PQ ADC distances) where quantization distortion alters distance thresholds.

### 4.3 Per-Query Latency Percentiles
- **Sampling Methodology:** Individual single-query executions measured over 500 query invocations using Python's high-resolution monotonic timer `time.perf_counter()`.
- **Warm-Up Behavior:** 2 initial warmup queries precede timing to ensure processor instruction cache and memory page translation tables are primed.
- **Reported Statistics:** Median ($p_{50}$), 95th percentile ($p_{95}$), arithmetic mean ($\mu$), and sample standard deviation ($\sigma$).

### 4.4 Batch Throughput (QPS)
- **Measurement Protocol:** Measured over 5 full-batch search repetitions across all $Q=1,000$ queries simultaneously.
- **Formula:**
  $$\text{QPS} = \frac{Q \times \text{repetitions}}{\sum_{r=1}^{\text{repetitions}} t_{\text{batch}}^{(r)}}$$

### 4.5 Distance Computations (DC)
Tracks the exact number of vector-vector distance calculations performed along the query execution path:
- **Exact Flat:** Strictly $N$ distance computations per query ($100,000$ for SIFT-100K).
- **IVF-Flat:** Divided into centroid distance operations and candidate list distance operations:
  $$\text{DC}_{\text{query}} = n_{\text{list}} + \sum_{p=1}^{n_{\text{probe}}} |L_{\pi(p)}|$$
  where $L_{\pi(p)}$ is the $p$-th closest inverted posting list.

### 4.6 Memory Footprint
- **Index Storage Memory (MiB):** Sum of underlying contiguous NumPy array allocations (vectors, centroids, posting lists, and ID buffers) divided by $1024^2$.
- **Peak Process RSS (MiB):** Peak resident set size of the benchmark process sampled directly from operating system memory counters (`tracemalloc` / OS page counters).

---

## 5. Recorded Execution Environment & Provenance

The official Phase 6.6 SIFT-100K benchmark artifacts were generated under the following verified environment:

- **Operating System:** Microsoft Windows 10 Pro / Enterprise (Build 10.0.26300, 64-bit AMD64)
- **Processor:** Intel64 Family 6 Model 186 Stepping 2, GenuineIntel (13th Gen Intel Core Architecture)
- **Python Version:** 3.10.11 (CPython, 64-bit)
- **NumPy Version:** 2.2.6 (linked with optimized OpenBLAS / MKL matrix kernels)
- **Git Commit Hash:** `c84a3dc777eb624b09a0bc220515697710c264d3`
- **Execution Timestamp:** `2026-10-29T12:24:12.728500+00:00` (UTC)
- **Artifact Manifest File:** `results/sift100k_p66/benchmark_results.json`
- **Summary Table File:** `results/sift100k_p66/benchmark_summary.csv`

---

## 6. Official SIFT-100K Benchmark Findings

The table below reproduces the exact recorded results from `results/sift100k_p66/benchmark_summary.csv`:

| Index Architecture | $n_{\text{list}}$ | $n_{\text{probe}}$ | Recall@10 | DistAware | $p_{50}$ Latency (ms) | $p_{95}$ Latency (ms) | QPS (Throughput) | Distance Ops / Query | Peak RSS (MiB) | Build Time (s) | K-means Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exact (Reference)** | — | — | **1.0000** | 1.0000 | 150.69 | 198.22 | 6.2 | 100,000.0 | 145.58 | 0.012 | — |
| **Exact (BLAS)** | — | — | **1.0000** | 1.0000 | **99.74** | 146.96 | **19.5** | 100,000.0 | 145.58 | 0.031 | — |
| **IVF-Flat** | 10 | 1 | 0.7882 | 0.7882 | 15.41 | 23.45 | 64.0 | 10,542.0 | 467.71 | 16.38 | Converged (83 iter) |
| **IVF-Flat** | 10 | 2 | 0.9507 | 0.9507 | 33.36 | 45.51 | 31.9 | 20,347.6 | 467.71 | 16.38 | Converged (83 iter) |
| **IVF-Flat** | 10 | 5 | **1.0000** | 1.0000 | 82.32 | 100.52 | 12.0 | 50,173.8 | 467.71 | 16.38 | Converged (83 iter) |
| **IVF-Flat** | 10 | 10 | **1.0000** | 1.0000 | 175.20 | 214.04 | 5.9 | 100,010.0 | 467.71 | 16.38 | Converged (83 iter) |
| **IVF-Flat** | 50 | 1 | 0.5976 | 0.5976 | 3.84 | 5.45 | 263.0 | 2,275.9 | 517.36 | 36.71 | Max iter (100) |
| **IVF-Flat** | 50 | 2 | 0.7940 | 0.7940 | 7.15 | 9.87 | 141.4 | 4,461.2 | 517.36 | 36.71 | Max iter (100) |
| **IVF-Flat** | 50 | 5 | 0.9587 | 0.9587 | 17.55 | 21.97 | 55.9 | 11,015.0 | 517.36 | 36.71 | Max iter (100) |
| **IVF-Flat** | 50 | 10 | 0.9961 | 0.9961 | 36.19 | 44.08 | 8.2 | 21,753.6 | 517.36 | 36.71 | Max iter (100) |
| **IVF-Flat** | 50 | 20 | **1.0000** | 1.0000 | 67.90 | 92.15 | 16.4 | 40,973.3 | 517.36 | 36.71 | Max iter (100) |
| **IVF-Flat** | 100 | 1 | 0.5287 | 0.5287 | **0.72** | 1.22 | **1,534.6** | 1,212.6 | 577.99 | 22.77 | Max iter (100) |
| **IVF-Flat** | 100 | 2 | 0.7176 | 0.7176 | 1.70 | 2.53 | 565.8 | 2,325.6 | 577.99 | 22.77 | Max iter (100) |
| **IVF-Flat** | 100 | 5 | 0.9126 | 0.9126 | 4.53 | 7.05 | 220.7 | 5,588.6 | 577.99 | 22.77 | Max iter (100) |
| **IVF-Flat** | 100 | 10 | 0.9791 | 0.9791 | 8.32 | 11.14 | 116.8 | 10,928.1 | 577.99 | 22.77 | Max iter (100) |
| **IVF-Flat** | 100 | 20 | **0.9980** | 0.9980 | 17.75 | 21.18 | 55.9 | 21,480.9 | 577.99 | 22.77 | Max iter (100) |

---

## 7. Scope & Index Coverage Disclosures

> [!IMPORTANT]
> **Scope Boundaries of this Specific Historical Run:**
> 1. **Evaluated Architectures:** The SIFT-100K Phase 6.6 benchmark specifically evaluated `exact_reference`, `exact_blas`, and `ivf_flat` parameter sweeps.
> 2. **HNSW & IVF-PQ Status:** Neither HNSW nor IVF-PQ were executed in this specific historical SIFT-100K batch run. They were evaluated in the synthetic interactive lab suite, property-based tests, and smoke test sweeps. Their performance characteristics are documented separately.
> 3. **Search vs. Embedding Decoupling:** All reported latencies measure purely vector retrieval time from memory. Neural embedding generation is decoupled and not measured in index search benchmarks.
> 4. **Build Time vs. Query Time:** Index build times (e.g. K-means clustering) are recorded independently and never conflated with search latencies.

---

## 8. Systematic Technical Takeaways

1. **BLAS Acceleration:** Exact BLAS achieves a **$3.15\times$ throughput speedup** over the coordinate-differencing reference kernel (19.5 vs. 6.2 QPS) and reduces $p_{50}$ latency from 150.7ms to 99.7ms.
2. **IVF Throughput Speedup:** At $n_{\text{list}}=100, n_{\text{probe}}=1$, IVF-Flat achieves **1,534.6 QPS** (0.72ms median latency), representing a **$78.7\times$ speedup** over the Exact BLAS baseline at 52.9% Recall@10.
3. **Balanced Operating Point:** At $n_{\text{list}}=100, n_{\text{probe}}=5$, IVF-Flat delivers **91.3% Recall@10** at **4.53ms median latency** (220.7 QPS), reducing distance computations by **$17.9\times$** (5,589 vs. 100,000 operations per query).
4. **Near-Lossless Operating Point:** At $n_{\text{list}}=100, n_{\text{probe}}=20$, IVF-Flat achieves **99.8% Recall@10** at **17.75ms median latency**, evaluating only 21.5% of database vectors.
5. **Batch QPS Anomaly & Contention Hypothesis:** As documented in ADR-0008, batch QPS at $n_{\text{list}}=50, n_{\text{probe}}=10$ dropped to 8.2 QPS while individual query latencies and distance computations scaled strictly monotonically. Comparative baseline runs (`results/sift100k`) exhibited strictly monotonic throughput scaling (329.3 $\to$ 173.5 $\to$ 63.3 $\to$ 29.4 $\to$ 12.9 QPS), indicating that transient OS/CPU thread scheduling contention during that multi-minute batch execution is a plausible explanation, though without low-level kernel tracing it remains an empirical hypothesis rather than a conclusively proven root cause.

---

## 9. Reproducing the Benchmarks

### 9.1 Fast Offline Smoke Benchmark (< 2 seconds)
```bash
python -m indexora.benchmark --smoke --no-plot
```
Generates verified machine-readable JSON and CSV manifests in `benchmark_output/`.

### 9.2 Full SIFT-100K Benchmark Run
1. Download Texmex SIFT1M dataset:
   ```bash
   curl -O ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
   tar -xzf sift.tar.gz
   ```
2. Execute the sweep:
   ```bash
   python -m indexora.benchmark \
       --dataset sift_fvecs \
       --base-path sift/sift_base.fvecs \
       --query-path sift/sift_query.fvecs \
       --n-base 100000 \
       --n-queries 1000 \
       --k 10 \
       --nlist 10,50,100 \
       --nprobe 1,2,5,10,20 \
       --output-dir results/my_sift_run
   ```
