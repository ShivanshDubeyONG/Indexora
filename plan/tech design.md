# Indexora --- Technical Design

**Version:** 3.0 \| **Purpose:** Architecture, module contracts,
algorithms, and evaluation rules

## 1. Architecture

Keep the project layered: 1. Data: vectors, stable IDs, metadata,
dataset splits. 2. Correctness: reference distances, deterministic
ranking, exact search, ground truth. 3. Algorithms: k-means, IVF-Flat,
PQ, IVF-PQ, HNSW. 4. Evaluation: recall, latency, throughput, distance
computations, memory, build time. 5. Operations: mutations, filters,
persistence, diagnostics. 6. Demo: semantic search through the same
public Indexora API.

The demo must not become the only way to exercise the engine.

## 2. Suggested repository layout

``` text
indexora/
  pyproject.toml
  README.md
  src/indexora/
    __init__.py
    data.py
    metrics.py
    kernels.py
    topk.py
    exact.py
    clustering.py
    ivf_flat.py
    pq.py
    ivf_pq.py
    hnsw.py
    filters.py
    evaluation.py
    persistence.py
    diagnostics.py
    accel.py
    adapters/
  tests/
  benchmarks/
    datasets.py
    run_exact_ivf.py
    run_pq.py
    run_hnsw.py
    run_comparison.py
    plots.py
    results/
  examples/semantic_search/
  docs/
```

Create modules when their phase begins; this is a target layout, not a
requirement to create empty files on day one.

## 3. Data model and metric contracts

-   Vectors: 2D contiguous NumPy `float32` array of shape `(N, d)`.
-   IDs: 1D `int64` array of length `N`, unique among live records.
-   Query: `(d,)`, with batched `(Q, d)` support added explicitly.
-   Metadata: columnar arrays or JSON-compatible records keyed by stable
    ID; do not treat row position as durable identity.
-   Validate dimensions, finite values, ID length/uniqueness, and
    supported dtypes.
-   Raise clear errors; do not silently truncate dimensions, normalize,
    replace NaNs, or mutate IDs.

Initial canonical metric is squared L2:

\\\[ d^2(q,x)=\\sum\_{j=1}^{d}(q_j-x_j)\^2 \\\]

Squared L2 preserves L2 ranking without square roots. Cosine behavior is
explicit: normalize vectors/queries through a documented helper,
including a zero-vector policy. For normalized nonzero vectors, cosine
and squared-L2 rankings are monotonically related.

## 4. Distance kernels and top-k

### Reference kernel

Implement direct coordinate differences and careful accumulation (for
example, `float64` accumulation for supported sizes). This is the
reference semantics used by tests and ground-truth generation.

### Fast kernel

A separate BLAS-friendly path may use:

\\\[ \\\|q-x\\\|^2=\\\|q\\\|^2-2q\^\\top x+\\\|x\\\|\^2 \\\]

Floating-point cancellation can cause small errors or slightly negative
values. Clamp small negative values only as a numerical correction and
test it.

**Critical caveat:** selecting `k + slack` candidates using fast
distances and reranking them with the reference kernel does not
guarantee exact top-k. A true neighbor may be omitted from the candidate
pool due to numerical ranking error or ties at the selection boundary.
Keep the reference kernel as oracle. Treat fast candidate generation
plus reranking as an optimization experiment unless a stronger
correctness method is implemented and proved. Measure disagreement
against the reference path; do not label it exact by assumption.

### Deterministic ties

Use a final deterministic order such as `(distance, ID)`. `argpartition`
alone does not define which equal-distance items survive at the
k-boundary. Handle boundary ties explicitly or use a stable method. Test
duplicate vectors, equal distances, `k=1`, `k=N`, and `k>N`.

## 5. Exact Flat search and ground truth

Exact search scans all eligible vectors using the reference semantics.
Generate ground truth with this path, never with the ANN method being
evaluated.

Test hand-computable vectors, duplicate vectors, ties, empty indexes,
invalid `k`, non-row-position IDs, non-contiguous inputs if accepted,
and filtered exact search when filtering is introduced. Benchmark the
reference path and any optimized path separately.

## 6. k-means and IVF-Flat

### k-means

Implement k-means++ and Lloyd iterations. Expose `n_clusters`,
`max_iter`, `tol`, and `seed`. Handle empty clusters explicitly, with a
documented policy. Record inertia where practical. For large data,
training may use a reproducible sample; record sample policy and seed.

### IVF-Flat

-   Train coarse centroids with k-means.
-   Assign each vector to one inverted list.
-   Store row references and stable IDs consistently.
-   Rank centroids, probe nearest `nprobe` lists, and exact-score all
    candidates in those lists.
-   Validate `nprobe` and document its range.

**Invariant:** probing every populated list includes every live vector,
so IVF-Flat should agree with exact search under the same score and tie
policy. If not, debug list membership, ID mapping, scoring, and ties.
Report `nlist`, `nprobe`, list-size distribution, build time, and
memory.

## 7. PQ and IVF-PQ

### PQ

Split each vector into `m` equal subspaces; initially require
`d % m == 0`. Train a codebook per subspace. With 256 centroids per
subspace, each subspace code is one byte; codes take about `m`
bytes/vector before IDs, codebooks, and metadata.

Test shape constraints, codebook/code-array dimensions, code
range/dtype, encode/decode behavior, approximation error, and fixed-seed
behavior.

### ADC

Build query-to-codeword distance lookup tables for each subspace, then
sum entries for each candidate (asymmetric distance computation).
Compare ADC scores with direct distances and report approximation
quality.

### IVF-PQ residuals

Encode residuals relative to the assigned coarse centroid:

\\\[ r=x-c\_{\\operatorname{assign}(x)} \\\]

At query time, score encoded residual candidates consistently relative
to each probed centroid. Test explicitly for confusion between
raw-vector PQ and residual IVF-PQ. Optional exact reranking requires
retained original vectors and its memory cost must be stated.

## 8. HNSW

Implement only after exact, IVF, and the benchmark harness are
dependable. Expose `M`, `efConstruction`, and `efSearch`. A common
level-sampling design uses a logarithmic distribution with a multiplier
related to `1 / ln(M)`; document and test the exact formula chosen. Keep
search-layer heaps, insertion, neighbor selection, and link pruning
independently testable.

Check: - No self-links; all neighbor IDs valid. - Degree limits and
entry-point/max-level consistency. - Tombstones do not break
traversal. - Seeded builds are repeatable to the intended extent. -
Recall/latency are measured across parameters.

**Do not assume `efSearch >= N` makes ordinary graph search
exhaustive.** It does not guarantee every node is visited. Test exact
equality only for an explicitly implemented exhaustive fallback. Normal
HNSW graph search is approximate; report measured recall. Compare custom
Python HNSW with a compiled reference such as hnswlib where available,
including distance computations and matched-recall latency.

## 9. Mutations, filtering, and lifecycle

These features come after core search and evaluation.

-   IVF-Flat insert: assign to nearest existing centroid; do not
    silently retrain.
-   PQ insert: encode with existing codebooks.
-   IVF-PQ insert: assign to existing centroid and encode its residual
    with existing codebooks.
-   HNSW insert: normal graph insertion.
-   Delete: start with tombstones and document whether deleted nodes
    remain traversable.
-   Update: define whether it is delete-plus-insert and whether an ID
    can be reused.
-   Compaction: rebuild index structures and validate against live
    records.

Filtering semantics must be consistent: only live records matching the
predicate may appear in results. Exact search can filter before scoring.
IVF may adaptively increase `nprobe` under a documented budget. HNSW may
need to traverse nodes that fail the filter while excluding them from
final results. Very selective filters may benefit from an exact filtered
fallback; measure rather than assume. Test no matches, all matches, high
selectivity, deleted records, missing metadata, and combined predicates.

## 10. Persistence and validation

Use `header.json` for schema/version/configuration/file manifest and
`.npy`/`.npz` for numeric arrays. Load with `allow_pickle=False`.
Validate schema, shapes/dtypes, IDs, list offsets, code ranges, graph
links/levels, entry point, and checksums where implemented. Write to a
temporary location and atomically replace the destination where
supported. Never unpickle arbitrary objects. Reject corrupt or
unsupported state clearly.

## 11. Benchmark methodology

Measure: - Recall@k against exact ground truth. - p50/p95 query latency
and QPS under a documented timing/batching setup. - Distance
computations per query, with the counting definition documented. -
Training/build time and memory, separating array bytes from process
memory where possible. - Index bytes per vector, stating
included/excluded overhead.

Protocol: - Same dataset, split, metric, `k`, and query set for
comparisons. - Fixed seeds and full parameter metadata. - Warm up before
timing; separate JIT compilation. - Repeat runs; retain raw records and
aggregate summaries. - Compare latency/memory at matched recall. -
Separate download, build, query, and plotting stages. - Label
reference-library versions and results clearly.

### Experiment matrix

  -----------------------------------------------------------------------
  ID                      Experiment              Question
  ----------------------- ----------------------- -----------------------
  E1                      Reference vs fast       Speed and numerical
                          kernel                  disagreement?

  E2                      IVF-Flat sweep          Effect of
                                                  `nlist`/`nprobe` on
                                                  recall, latency, DC?

  E3                      PQ sweep                Compression/accuracy
                                                  trade-off by `m`?

  E4                      IVF-PQ                  How does residual
                                                  coding compare with PQ?

  E5                      HNSW sweep              Effect of `M`,
                                                  `efConstruction`,
                                                  `efSearch`?

  E6                      Matched-recall          Best measured
                          comparison              latency/memory at
                                                  target recall?

  E7                      Mutation/compaction     Cost of writes,
                                                  deletes, rebuilds?

  E8                      Filter selectivity      Effect on latency and
                                                  recall?

  E9                      SIFT vs text embeddings How does workload
                                                  change trade-offs?

  E10                     Numba                   Does one profiled hot
                                                  path benefit?

  E11                     Library vs HTTP         How much overhead does
                                                  the demo/API add?
  -----------------------------------------------------------------------

Headline plots: recall@10 vs p50 latency; recall@10 vs distance
computations; recall@10 vs index bytes/vector; filter selectivity vs
latency/recall; separate standard-vector and text-embedding results.
Never present placeholder numbers as real results.

## 12. Dataset and demo

Use a suitable public vector dataset (e.g. a SIFT-derived set) for
algorithm benchmarks. Separately use a modest text corpus and cached
embeddings from a small sentence-embedding model. Record source,
model/version, dimension, preprocessing, and corpus identity.

The demo can be a CLI or lightweight web page. It should query
Indexora's public API, display top results and metadata/snippets, cache
embeddings, and separate embedding-generation time from index-search
latency. Explain that semantic relevance depends on the embedding model
and corpus, and ANN recall is not the same thing as semantic quality.

## 13. Dependencies and implementation order

NumPy is required. Numba, FAISS, and hnswlib are optional and must be
isolated so the core imports/tests without them. Profile first; try one
Numba hot path at most initially and report compilation separately.

Order: exact correctness → k-means/IVF-Flat → benchmark harness →
PQ/IVF-PQ → HNSW → operations → semantic demo. If time is short, finish
a correct v0.1 and then choose one advanced algorithm to complete before
starting another.
