# Indexora — Technical Interview Preparation Guide

This guide compiles essential architectural concepts, mathematical proofs, implementation details, and trade-offs for technical interviews, systems discussions, and live code walkthroughs of **Indexora**.

---

## 1. Core Vector Search Concepts

### Q1: Why build a vector search engine from scratch in Python and NumPy?
**Answer:**
Most developers interact with vector databases (e.g., Pinecone, Milvus, Qdrant) as opaque black boxes. Building a vector search engine from first principles provides direct, hands-on mastery over:
1. Low-level numerical precision and distance kernel mechanics.
2. Approximate nearest-neighbor (ANN) algorithms (Voronoi partitions, product quantization codebooks, hierarchical graphs).
3. The real-world friction between theoretical algorithmic complexity (e.g., $O(\log N)$ graph search) and practical hardware realities (cache locality, BLAS SIMD vectorization, and language interpreter overhead).

Indexora isolates these core algorithmic mechanics from massive distributed infrastructure, creating a verifiable learning platform where every design decision is measured and tested.

---

### Q2: What is the fundamental difference between Exact Search and Approximate Nearest-Neighbor (ANN) Search?
**Answer:**
- **Exact Search (Brute Force / Flat):** Computes distances between the query vector and every single vector in the database ($O(N \cdot d)$ operations), guaranteeing 100% recall. However, latency scales linearly with dataset size $N$, making it computationally prohibitive for millions of vectors.
- **ANN Search:** Sacrifices a small fraction of recall (e.g., achieving 95–99% Recall@k) to achieve sub-linear query time ($O(\sqrt{N})$ or $O(\log N)$) and reduced distance evaluations. ANN algorithms structure the search space using space partitioning (IVF), vector compression (PQ), or navigable proximity graphs (HNSW).

---

### Q3: Why does Indexora use Squared Euclidean Distance ($d^2$) instead of standard Euclidean Distance or Cosine Similarity?
**Answer:**
1. **Computational Efficiency:** Standard Euclidean distance requires computing a square root: $d(q, x) = \sqrt{\sum (q_j - x_j)^2}$. Since the square root function is strictly monotonically increasing for non-negative values:
   $$d(q, x_A) < d(q, x_B) \iff d^2(q, x_A) < d^2(q, x_B)$$
   Omitting $\sqrt{\cdot}$ saves millions of CPU cycles per query without altering candidate ranking.
2. **Equivalence with Cosine Similarity:** For unit-normalized vectors ($\|q\|_2 = \|x\|_2 = 1$):
   $$\|q - x\|^2 = \|q\|^2 + \|x\|^2 - 2(q \cdot x) = 2 - 2 \cos(q, x)$$
   Minimizing squared Euclidean distance $\|q - x\|^2 \in [0, 4]$ is mathematically identical to maximizing cosine similarity $\cos(q, x) \in [-1, 1]$. Indexora normalizes embeddings and uses squared L2 to natively power cosine search without specialized kernels.

---

### Q4: Why does BLAS GEMM (`np.dot`) dramatically outperform coordinate-difference loops in NumPy?
**Answer:**
In a reference coordinate-difference implementation:
```python
diff = database - query  # Allocates an (N, d) temporary array
dist = np.sum(diff * diff, axis=1)  # Reads (N, d) twice from memory
```
This approach is strictly memory-bandwidth bound: it allocates large intermediate memory buffers and saturates CPU cache lines.

In contrast, the binomial expansion $\|q - x\|^2 = \|q\|^2 + \|x\|^2 - 2(q \cdot x)$ delegates the heavy cross-term $-2(q \cdot x)$ to `np.dot`, which invokes Level-3 BLAS GEMM (OpenBLAS/MKL). BLAS libraries are written in hand-tuned assembly utilizing:
1. AVX-512 / AVX2 SIMD vector instructions.
2. Cache-blocking tiles that keep matrix blocks inside L1/L2 CPU caches.
3. Multi-threaded parallelization.

On SIFT-100k ($N=100,000, d=128$), Exact BLAS achieves **19.5 QPS (103.5 ms mean latency)** compared to **6.2 QPS (155.8 ms mean latency)** for the reference kernel—a **3.1x throughput speedup**.

---

## 2. Inverted File & Clustering (IVF-Flat)

### Q5: How does IVF-Flat work, and what role do `nlist` and `nprobe` play?
**Answer:**
IVF-Flat partitions the vector space into `nlist` Voronoi cells using $K$-means clustering:
1. **Build Phase:** $K$-means identifies `nlist` centroids. Each vector is assigned to its closest centroid and appended to that centroid's inverted posting list.
2. **Query Phase:** The query is compared against all `nlist` centroids. The search explores only the posting lists of the closest `nprobe` centroids.

**Trade-offs:**
- Higher `nlist`: Creates smaller posting lists, reducing candidate distance computations, but increases coarse centroid search cost and cluster training time.
- Higher `nprobe`: Scans more candidate vectors, increasing Recall@k at the expense of higher query latency.
- **Invariant:** When `nprobe == nlist`, IVF-Flat evaluates all database vectors, guaranteeing bit-for-bit equivalence with exact search.

---

### Q6: What happens if a $K$-means cluster becomes empty during training?
**Answer:**
In naive $K$-means, if no vectors are assigned to a centroid during Lloyd's iteration, dividing by cluster size causes a division-by-zero error. Indexora implements three configurable empty-cluster policies:
1. `reseed_furthest` (default): Identifies the database vector with the highest residual error (greatest distance from its assigned centroid) and reseeds the empty cluster center at that vector.
2. `keep`: Leaves the empty centroid at its previous position.
3. `error`: Raises an explicit runtime exception.

---

## 3. Product Quantization (PQ & IVF-PQ)

### Q7: How does Product Quantization achieve 15x–64x memory compression?
**Answer:**
Standard vector storage requires $4 \times d$ bytes per vector (e.g., 512 bytes for $d=128$). Product Quantization decomposes the vector into $M$ orthogonal subspaces of dimension $d_{\text{sub}} = d / M$:
1. For each subspace $m \in \{1, \dots, M\}$, a codebook of $k_{\text{sub}}$ centroids (typically 256) is trained via $K$-means.
2. Each sub-vector is replaced by the integer index of its closest sub-centroid.
3. For $k_{\text{sub}} = 256$, each sub-index fits in an 8-bit integer (`uint8`).
4. A 128-dimensional vector is represented by just $M$ bytes. For $M=8$, storage drops from 512 bytes to 8 bytes—an exact **64x compression ratio**.

In IVF-PQ, quantization is performed on the *residuals* ($r_i = x_i - C_{\text{coarse}}$) relative to coarse IVF centroids, preserving fine-grained geometric variance.

---

### Q8: What is Asymmetric Distance Computation (ADC), and why is it preferred over Symmetric Distance Computation (SDC)?
**Answer:**
- **Symmetric Distance Computation (SDC):** Both query $q$ and database vector $x$ are quantized. Distances are computed between codebook centroids: $d(q, x) \approx d(\hat{q}, \hat{x})$. SDC suffers from two sources of quantization error ($q \to \hat{q}$ and $x \to \hat{x}$).
- **Asymmetric Distance Computation (ADC):** Only database vectors $x$ are quantized; the query $q$ remains uncompressed.
  1. The engine computes a small lookup table of size $(M, k_{\text{sub}})$ containing squared Euclidean distances between each query sub-vector and all $k_{\text{sub}}$ codebook centroids.
  2. The distance to any quantized database vector is computed by summing $M$ table lookups:
     $$d_{\text{ADC}}^2(q, x_i) = \sum_{m=1}^M \text{Table}[m, \text{code}_{i, m}]$$
  3. ADC eliminates $q$ quantization error and evaluates candidate distances using fast table lookups and integer additions without reconstructing original vectors.

---

## 4. Hierarchical Navigable Small World (HNSW)

### Q9: How does HNSW achieve logarithmic $O(\log N)$ search complexity?
**Answer:**
HNSW builds a multi-layer proximity graph inspired by the Skip List data structure:
1. **Hierarchical Layers:** Nodes are assigned a maximum layer $l$ via exponential decay: $l = \lfloor -\ln(\text{uniform}(0, 1)) / \ln(M) \rfloor$. Layer 0 contains all vectors; layer $L_{\max}$ contains very few nodes.
2. **Coarse-to-Fine Routing:** Search begins at the top layer. At each upper layer ($L_{\max} \dots 1$), a greedy 1-recall search navigates toward the query, jumping across long-range express links.
3. **Beam Search at Layer 0:** Upon reaching layer 0 (where the target local minimum resides), the search switches to a bounded priority-queue beam search exploring candidates up to depth `ef_search`.
4. This top-down hierarchy avoids getting trapped in distant local minima and reduces graph exploration steps to $O(\log N)$.

---

### Q10: What is Malkov & Yashunin's Algorithm 4 (Heuristic Diversity), and why is it crucial for HNSW?
**Answer:**
When selecting up to $M$ neighbors for a node $u$ from a candidate set, a simple greedy policy chooses the $M$ closest neighbors. However, closest neighbors often cluster in the same spatial direction, creating redundant co-linear edges and failing to connect different regions of the graph.

Algorithm 4 enforces **geometric diversity**:
- Candidates are evaluated in ascending order of distance from $u$.
- A candidate $e$ is connected to $u$ **if and only if** $e$ is closer to $u$ than to any neighbor already selected for $u$:
  $$d(u, e) < \min_{v \in \text{Selected}} d(e, v)$$
- This ensures neighbors span diverse angular directions, preventing clustering, preserving graph connectivity, and guaranteeing the Small-World navigable property.

---

## 5. Lifecycle, Filtering, Persistence & Semantic Search

### Q11: How does Indexora handle vector deletions in HNSW without breaking graph connectivity?
**Answer:**
Directly deleting a node and removing its edges can partition the graph or sever crucial routing pathways between clusters.

Indexora implements **tombstone routing preservation**:
1. When `delete(doc_id)` is invoked, the ID is recorded in an in-memory deletion mask.
2. The node remains physically present in the HNSW adjacency graph, continuing to serve as a routing waypoint for traversals across layers.
3. During query evaluation, tombstoned nodes are traversed during graph routing but are strictly excluded from the final top-$k$ result set.
4. Memory is reclaimed later during a maintenance compaction pass (`compact()`).

---

### Q12: How does single-stage filtered search work in Indexora?
**Answer:**
Indexora supports explicit ID sets (`filter_ids={1, 2, 3}`) and callable predicates (`filter_ids=lambda id: id % 2 == 0`).
- **Post-filtering Flaw:** Traditional post-filtering searches for top-$k$ nearest neighbors and then filters out invalid items. If only 1% of the database is eligible, top-$k$ search frequently returns zero valid results.
- **Indexora Single-Stage Filtering:** Eligibility is evaluated during candidate consideration. In HNSW and IVF, candidate vectors that fail the filter predicate are never evaluated for top-$k$ insertion, ensuring the top-$k$ heap is filled exclusively with valid, eligible documents.

---

### Q13: How does Indexora persist indexes to disk without Python pickle?
**Answer:**
Python's `pickle` module poses severe security vulnerabilities (arbitrary code execution) and lacks cross-version/cross-language stability. Indexora implements atomic, pickle-free persistence:
1. **Format:** Metadata, hyperparameters, and index state are saved in human-readable `manifest.json`. Numerical vectors, IDs, and centroids are saved in standard NumPy `.npy` binary arrays.
2. **HNSW CSR Serialization:** Graph adjacency lists are flattened into Compressed Sparse Row (CSR) arrays (`indptr` and `indices`), enabling zero-rebuilding deserialization.
3. **Streaming SHA-256 Checksums:** During save, every `.npy` and `.json` file is hashed in streaming 64 KiB chunks. On `load_index(..., verify_checksums=True)`, file digests are recalculated. Any truncated file or byte corruption aborts loading with a descriptive integrity error.

---

### Q14: How are text queries translated into vector search, and why is embedding latency separated from search latency?
**Answer:**
1. Text embeddings are generated using a pretrained Transformer model (`sentence-transformers/all-MiniLM-L6-v2`), producing 384-dimensional dense vectors.
2. Embeddings are $L_2$-normalized so that squared Euclidean distance ordering directly equals cosine similarity ordering.
3. **Latency Separation:** Generating deep-learning embeddings on CPU requires 20–50 ms per query (neural forward pass). In contrast, Indexora's vector index retrieval requires 0.1–2.0 ms. Conflating the two hides the true performance of the vector search engine. Indexora explicitly tracks and reports $t_{\text{embed}}$ and $t_{\text{search}}$ separately.

---

## 6. Benchmarking & Performance Disclosures

### Q15: Why does Exact BLAS sometimes beat HNSW or IVF on small datasets ($N \le 1,000$)?
**Answer:**
This is an essential systems insight:
- `ExactBlasIndex` executes via OpenBLAS/MKL GEMM in C/assembly. For $N=1,000, d=128$, the entire dataset is 512 KiB—fitting comfortably inside L2/L3 CPU cache. The CPU streams the matrix multiplication at hardware memory bandwidth with zero Python interpreter overhead.
- In contrast, Python implementations of IVF and HNSW execute priority queues, dictionary lookups, dynamic array indexing, and loop iterations in Python bytecode.
- For small $N$, the interpreter overhead of graph traversal exceeds the brute-force cost of vectorized matrix multiplication. HNSW and IVF prove their wall-clock advantage when $N \ge 50,000$, where sub-linear distance evaluations ($O(\log N)$) outweigh interpreter constants.

---

### Q16: What does the current Indexora implementation NOT yet support? (Honest Limitations)
**Answer:**
A credible engineer knows their system's boundaries:
1. **Single-Node In-Memory:** Indexora does not support distributed sharding, partition replication, or consensus protocols (Raft/Paxos).
2. **Single-Writer Concurrency:** Mutations are thread-safe only under single-writer locks; it does not support lock-free multi-writer concurrent graph insertions.
3. **Memory Mapped Storage (mmap):** Indexes reside in RAM. Large datasets exceeding host RAM require paging or out-of-core storage which is not yet implemented.
4. **Pure Python Loops in Graph Search:** While distance kernels use C-BLAS, HNSW graph priority-queue traversal is pure Python. Rewriting graph traversal in C/Cython or Rust would yield 5x–15x speedups.

---

## 7. Two-Minute Live Demonstration Script

When presenting Indexora in an interview or demo, follow this 4-step walkthrough:

```bash
# Step 1: Verify Algorithmic Correctness (15s)
pytest -q
# Point out: 325 passing tests covering mathematical kernels, IVF, PQ, HNSW, mutations, and persistence.

# Step 2: Run Offline Smoke Benchmark (10s)
python -m indexora.benchmark --smoke --no-plot
# Point out: Tracks Recall@k, distance-aware recall, QPS, and exact distance computations.

# Step 3: Launch Interactive Index Lab (30s)
streamlit run src/indexora/ui.py
# Point out: 
# - Tab 2 (Playground): 2D PCA spatial projection highlighting query vs nearest neighbors.
# - Tab 4 (Mutations): Real-time vector additions, tombstone deletions, and SHA-256 persistence round-trips.

# Step 4: Live Semantic Text Search (45s)
# Point out:
# - Tab 5 (Semantic Text Search): Type "LSM-tree disk flushing" or click a preset.
# - Highlight the latency breakdown: text embedding (42 ms) vs Indexora retrieval (0.29 ms).
# - Show that squared L2 distance directly mirrors cosine similarity.
# - Demonstrate category filtering restricting results exclusively to Databases.
```
