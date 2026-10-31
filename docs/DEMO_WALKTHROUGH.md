# Indexora — 2–3 Minute Technical Interview Demonstration Script

This walkthrough provides a crisp, confident demonstration of Indexora for technical interviewers, hiring managers, and engineers. It showcases the core architectural concepts, interactive application, and systems engineering trade-offs.

---

## Preparation (Before the Interview)

Ensure the virtual environment is activated and dependencies are installed:

```bash
# Windows
.\venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# Launch the lab in the background or in a separate terminal:
streamlit run src/indexora/ui.py
```

Have two terminal tabs ready:
- **Terminal 1:** Running the Streamlit UI (or ready to execute `pytest -q`).
- **Terminal 2:** Ready to run the offline smoke benchmark.

---

## 0:00–0:25 — Introduction: The Problem & The Core Trade-Off

> *"Hi! Today I want to show you Indexora, a vector search engine I built from first principles in Python and NumPy.*
>
> *Every modern AI application—from semantic search to RAG—relies on vector databases to find nearest neighbors in high-dimensional embedding spaces. But finding exact nearest neighbors via brute-force scan requires evaluating distance against every single vector in the database: an $O(N \cdot d)$ operation that becomes completely unscalable as datasets reach hundreds of thousands or millions of vectors.*
>
> *The entire vector search industry is built on Approximate Nearest Neighbor (ANN) search: trading a tiny amount of mathematical accuracy—measured by Recall@k—in exchange for orders-of-magnitude faster query latency and candidate pruning. In Indexora, I implemented the foundational algorithms behind this trade-off from scratch."*

---

## 0:25–1:00 — Interactive Demo: Spatial Indexing in Action

> *(Open browser to `http://localhost:8501`)*
>
> *"Here is the Indexora Interactive Lab. In the sidebar, we can configure our vector space—here we have 120 vectors in 8 dimensions clustered into 4 Gaussian centroids.*
>
> *If we go to the **Search Playground** tab, you can see our vector space projected into 2D using PCA. When I select a query vector—represented by the bright cyan star—Indexora instantly computes nearest neighbors.*
>
> *We can toggle between different index architectures:*
> - *First, **Exact BLAS**: which evaluates all 120 vectors using Level-3 GEMM matrix multiplication and acts as our mathematical ground-truth oracle.*
> - *Second, **IVF-Flat**: an Inverted File index that partitions vectors into Voronoi cells using $K$-means.*
> - *And third, **HNSW**: a Hierarchical Navigable Small World proximity graph that routes queries through a multi-layer skip-list hierarchy.*
>
> *Notice that for every approximate query, the results table clearly indicates whether each neighbor is a `100% Exact Match` or an approximation, verified against our ground-truth oracle."*

---

## 1:00–1:40 — Algorithm Comparison: The Recall vs. Latency Pareto Frontier

> *(Click the **Accuracy & Performance Lab** tab)*
>
> *"Next, let's look at the core systems trade-off in the Accuracy & Benchmarks tab. Here, Indexora executes a live comparative benchmark across all 5 index implementations on identical vectors:*
>
> 1. *Notice the **Recall vs. Median Latency Curve**: As we increase `nprobe` from 1 to 2 to 3 in IVF-Flat, Recall@5 climbs from 70% to 100%. At the same time, look at the second chart: **Candidate Distance Computations per Query**.*
> 2. *Exact scan must compute all 120 pairwise distances. But IVF with `nprobe=1` only evaluates coarse centroid comparisons plus the vectors in that single list—cutting distance operations by more than 60%.*
> 3. *And for memory-constrained environments, look at **IVF-PQ (Product Quantization)**: It decomposes our 8-dimensional space into 2 orthogonal subspaces and quantizes residuals into 8-bit codes. Using Asymmetric Distance Computation lookup tables, it scores candidates in microseconds with up to 16x memory compression."*

---

## 1:40–2:15 — Engineering: Systems Rigor, Correctness & Security

> *(Switch to Terminal 1)*
>
> *"Building this purely in Python and NumPy meant paying intense attention to systems-level correctness:*
>
> *Let me run our test suite:*
> ```bash
> pytest -q
> ```
> *(Runs in ~25 seconds: **329 passed**)*
>
> *Every single component is tested under strict mathematical contracts:*
> 1. **Numerical Stability:** We accumulate squared L2 coordinate differences in 64-bit float precision before casting to float32 to prevent floating-point catastrophic cancellation.
> 2. **Deterministic Tie-Breaking:** In `topk.py`, we enforce composite sorting on `(distance, external_id)` tuples so queries return identical result orderings on all runs.
> 3. **Persistence & Security:** Indexora serializes indexes to disk using atomic directory renames, streaming SHA-256 manifest verification, and zero pickle usage. In our security audit, we added strict path traversal checks (`is_relative_to`) so untrusted manifests cannot escape the index directory.
> 4. **Reproducible Benchmarks:** Let's run our offline smoke benchmark:
> ```bash
> python -m indexora.benchmark --smoke --no-plot
> ```
> *It runs in under a second and reproduces our baseline recall, QPS, and distance accounting."*

---

## 2:15–2:45 — Conclusion & End-to-End Applications

> *(Quickly switch back to the browser, Tab 5: Semantic Text Search)*
>
> *"Finally, vector indexes are metric-agnostic: by normalizing vectors to unit length, squared Euclidean distance is mathematically equivalent to cosine similarity:*
> $$\|q - x\|^2 = 2 - 2 \cos(q, x)$$
>
> *Here on the **Semantic Text Search** tab, I hooked up our native vector engine to a pretrained transformer (`all-MiniLM-L6-v2`). When I click a technical query like 'LSM-tree disk flushing', look at the latency breakdown:*
> - *Text embedding took 42 ms in PyTorch.*
> - *Vector retrieval took just 0.28 ms in Indexora.*
>
> *Indexora isn't meant to compete with billion-scale C++ engines like FAISS; it is a learning-first, transparent reference architecture designed to demystify vector retrieval, understand hardware bottlenecks, and explore the math from first principles.*
>
> *I'd love to dive deeper into any specific kernel, graph routing heuristic, or benchmark trade-off you'd like to explore!"*

---

## Key Interview Talking Points (Cheat Sheet)

| Topic | Key Concept | Where in Code |
| :--- | :--- | :--- |
| **Exact Oracle** | Coordinate difference in float64 accumulation vs BLAS Level-3 GEMM $\|q-x\|^2 = \|q\|^2 + \|x\|^2 - 2(q \cdot x)$ | `src/indexora/kernels.py` |
| **Top-k Tie-Breaking** | Partial sort (`argpartition`) + ascending `(distance, external_id)` composite ordering | `src/indexora/topk.py` |
| **IVF Clustering** | Vectorized $K$-means++ initialization, multi-probe candidate list traversal | `src/indexora/clustering.py`, `src/indexora/ivf.py` |
| **Product Quantization** | Subspace decomposition ($d = M \times d_{\text{sub}}$), 8-bit codebooks, Asymmetric Distance Computation (ADC) LUTs | `src/indexora/pq.py`, `src/indexora/ivfpq.py` |
| **HNSW Proximity Graph** | Multi-layer skip list, exponential level decay ($m_L = 1/\ln M$), Malkov & Yashunin Algorithm 4 diversity pruning | `src/indexora/hnsw.py` |
| **Mutations & Routing** | Dynamic `add()`, `delete()` with routing tombstones preserving reachability | `src/indexora/hnsw.py` (`_deleted_nodes`) |
| **Persistence & Security** | Pickle-free `.npy` storage, streaming SHA-256 manifests, strict path traversal defense | `src/indexora/persistence.py` |
