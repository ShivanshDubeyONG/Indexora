# Indexora --- Phased Implementation Plan

**Version:** 3.0 \| **Planning model:** Acceptance gates, not calendar
promises

## 1. How to work

Work on one phase at a time. A phase ends when its acceptance gate
passes, not when an estimated date arrives. After each phase, record:
changed files, design decisions, tests and actual outputs, what was
learned, limitations, and next step.

### Owner/agent split

The owner implements and understands the core algorithm bodies. For
human-owned phases, the agent may explain, propose interfaces, write
tests, give pseudocode/hints, and review the owner's attempt, but must
not silently deliver the finished core algorithm. If agent-written core
code is used as a reference, the owner must re-derive, explain, rewrite,
or materially validate it before claiming ownership.

Agent-assisted tasks include packaging/CI, dataset cache utilities, test
scaffolds, benchmark orchestration/plots, reference adapters, demo
plumbing, persistence plumbing, and documentation drafts.

### Rules for every agent task

1.  Inspect the repository before editing.
2.  Work only on the named phase.
3.  Preserve unrelated work and avoid scope creep.
4.  Add/update tests with meaningful changes.
5.  Run tests and report actual commands and outputs.
6.  Never invent benchmark results, test output, or completion claims.
7.  Keep optional dependencies optional.
8.  End with a concise handoff: files changed, tests, results,
    limitations, next step.

## 2. Phase overview

  -------------------------------------------------------------------------------------------
  Phase             Deliverable                     Ownership         Acceptance gate
  ----------------- ------------------------------- ----------------- -----------------------
  0                 Repository setup                Agent-assisted    Clean install and tests
                                                                      work

  1                 Reference distance +            Human core        Numerical/tie tests
                    deterministic top-k                               pass

  2                 Exact search + ground truth     Human core        Hand-computed cases
                                                                      pass

  3                 Data/dataset utilities          Agent-assisted    Stable IDs and offline
                                                                      tests

  4                 k-means++/Lloyd                 Human core        Clustering and
                                                                      empty-cluster tests
                                                                      pass

  5                 IVF-Flat                        Human core        All-probe results equal
                                                                      exact

  6                 Benchmark harness v0.1          Agent-assisted    Reproducible raw
                                                                      results/plots

  7                 Reference adapters              Agent-assisted    Optional, labeled
                                                                      comparisons

  8                 Profile/one acceleration        Shared            Measured benefit or
                    experiment                                        documented no-go

  9                 PQ + IVF-PQ                     Human core        Encoding/ADC/residual
                                                                      tests pass

  10                Custom HNSW                     Human core        Invariants pass; recall
                                                                      measured

  11                Mutations/filters/persistence   Shared            Lifecycle tests pass

  12                Semantic-search demo            Agent-assisted    Documented run uses
                                                                      Indexora API

  13                Release/interview prep          Shared            Reproducible release
                                                                      and evidence-based
                                                                      claims
  -------------------------------------------------------------------------------------------

## 3. Detailed phases

### Phase 0 --- Repository setup (agent-assisted)

Create a minimal package layout, Python/NumPy dependency, test runner,
optional formatting/linting, basic CI, `.gitignore`, license choice, and
short README. Do not implement search algorithms.

**Prompt:** "Inspect the current repository first. Set up only the
minimal packaging, test, and documentation foundation for Indexora. Do
not implement vector-search algorithms. Keep optional dependencies
optional. Run checks and report actual outputs."

**Gate:** clean install works, `import indexora` works, test command is
documented.

### Phase 1 --- Distance kernels and top-k (HUMAN CORE)

Implement the direct-difference squared-L2 reference kernel and
deterministic top-k. Test duplicate vectors, equal-distance boundary
ties, dimensions, empty arrays, non-finite values, and `k` edge cases.
Add a BLAS-style fast kernel only after the reference is correct.

**Prompt:** "Do not write the final bodies of the human-owned distance
or top-k functions. Refine edge-case tests, explain expected semantics,
provide hints/pseudocode when asked, and review my implementation for
numerical and tie-handling bugs."

**Gate:** hand-computable results pass; final ordering is deterministic
by the agreed tie policy; fast-path disagreement is measured and it is
not called exact without proof.

### Phase 2 --- Exact Flat and ground truth (HUMAN CORE)

Implement scan-based exact search using reference semantics, stable IDs,
and deterministic ranking. Generate ground truth through this path. Test
duplicate vectors, empty data, invalid `k`, non-row-position IDs, and
malformed input.

**Prompt:** "Help define the exact-search contract and tests, but do not
implement the core search loop for me. Ground truth must use the
reference implementation, not an ANN index."

**Gate:** exact outputs match hand-computed cases and ground truth is
reproducible.

### Phase 3 --- Data and dataset utilities (AGENT-ASSISTED)

Add dataset validation, splits, stable IDs, download/cache helpers,
provenance, and checksums where practical. Keep unit tests offline.

**Prompt:** "Implement only dataset/cache utilities. Do not change
kernel or exact-search semantics. Add offline tests with generated
arrays and document dataset acquisition."

**Gate:** invalid shapes/IDs fail clearly; unit tests need no network.

### Phase 4 --- k-means (HUMAN CORE)

Implement k-means++ and Lloyd updates. Define empty-cluster handling,
convergence criteria, seeds, and inertia reporting. Test synthetic
clusters and fixed-seed behavior.

**Prompt:** "Do not implement the core k-means++ or Lloyd update bodies.
Help define tests and pseudocode, then review my attempt, especially
empty clusters, seeds, and convergence."

**Gate:** synthetic tests pass and empty clusters follow an explicit
policy.

### Phase 5 --- IVF-Flat (HUMAN CORE)

Train coarse centroids, assign each vector to one list, preserve stable
IDs, and implement centroid ranking plus `nprobe` search. Test
unbalanced/empty lists and invalid parameters.

**Prompt:** "Help with interfaces and tests, but do not implement core
IVF assignment/search. Add an invariant test: probing every populated
list must match exact search under the same metric and tie policy."

**Gate:** each live vector belongs to one list; all-probe IVF equals
exact. Partial probing is evaluated, not assumed to have a particular
recall.

### Phase 6 --- Benchmark harness v0.1 (AGENT-ASSISTED)

Add a command to run exact and IVF sweeps. Save raw CSV/JSON and
metadata. Measure recall@k, p50/p95 latency, QPS, distance computations,
build time, and memory where available. Record dataset/split,
parameters, seed, commit, environment, warm-up, repeats, and timing
policy. Plot recall vs latency and recall vs distance computations.

**Prompt:** "Build a reproducible harness for existing exact and IVF
APIs. Adapt to current code instead of rewriting algorithms. Save raw
measurements and metadata. Do not use placeholders as results. Add a
tiny synthetic smoke test."

**Gate:** documented command creates results and plots; smoke test
works. Tag v0.1 only after the gate passes.

### Phase 7 --- Reference adapters (AGENT-ASSISTED)

Add optional adapters for suitable libraries such as FAISS or hnswlib.
Keep imports lazy, skip tests cleanly when unavailable, and record
library version and parameters.

**Gate:** core tests pass without optional libraries; comparisons share
dataset, metric, query set, and recall definition.

### Phase 8 --- Profiling/acceleration (SHARED)

Profile current code. Select one hot path only if profiling justifies
it; try Numba only where suitable. Separate compilation from
steady-state timings and compare against reference behavior.

**Gate:** same-workload before/after evidence exists. A measured no-go
is valid; do not retain acceleration just for appearance.

### Phase 9 --- PQ and IVF-PQ (HUMAN CORE)

Implement subspace splitting, PQ codebooks, encoding, ADC
lookup/scoring, IVF-PQ residual encoding, and optional exact reranking.
Measure memory, approximation error, recall, and latency.

**Prompt:** "Do not implement core PQ training, encoding, ADC, or IVF-PQ
residual logic for me. Help with interfaces, tests, and pseudocode. Test
shape/code constraints, fixed seeds, and raw PQ vs residual semantics;
review my attempt."

**Gate:** tiny ADC cases agree with direct calculations; code
shapes/ranges are valid; residual behavior is tested; memory accounting
states what is included. Tag v0.2 only when advertised scope is actually
complete.

### Phase 10 --- Custom HNSW (HUMAN CORE)

Implement level sampling, graph storage, greedy descent, search-layer
heaps, insertion, neighbor selection, and pruning. Test graph invariants
and seeded builds. Sweep `M`, `efConstruction`, and `efSearch`; compare
with hnswlib where available.

**Prompt:** "Do not implement core HNSW level assignment, insertion,
neighbor selection, or search-layer algorithms. Help define invariants,
tests, and pseudocode, then review my attempt. Do not assume ordinary
HNSW is exhaustive when `efSearch >= N`; use a separately named
exhaustive fallback if exact equality is needed."

**Gate:** valid links/degrees/entry point; search returns valid live
IDs; recall and latency are measured. Tag v0.3 only after tests and
evidence.

### Phase 11 --- Mutations, filters, persistence (SHARED)

First decide and document semantics for insert/delete/update, ID reuse,
tombstones, filtering, and compaction. Then implement approved scope.
Use `header.json` plus numeric `.npy`/`.npz` files, load with
`allow_pickle=False`, validate all structures, and reject corrupt files.

**Prompt:** "Propose ambiguous mutation/filter/persistence semantics for
review before coding. After approval, implement only the agreed scope.
Test corrupted state, deleted IDs, no-match filters, duplicate IDs, and
rebuild consistency."

**Gate:** operations match a live-record exact baseline; valid state
round-trips; invalid state fails clearly. Tag v0.4 only when advertised
operations pass.

### Phase 12 --- Semantic-search demo (AGENT-ASSISTED)

Use a modest corpus and cached sentence embeddings. Provide a CLI or
lightweight web page that calls Indexora's public API, displays
results/snippets/metadata, and separates embedding generation from
search latency.

**Prompt:** "Build a simple semantic-search demo using the existing
Indexora API. Cache embeddings, document corpus/model/preprocessing, and
do not introduce a separate retrieval engine. Distinguish embedding
relevance from ANN recall."

**Gate:** documented clean run works and uses Indexora rather than
bypassing it.

### Phase 13 --- Portfolio and interview prep (SHARED)

Audit install/tests/benchmarks/demo from a clean checkout. Add
architecture and experiment diagrams, methodology, limitations, and
interview notes. Draft resume bullets only from measured evidence.

**Prompt:** "Audit the repository against the brief and phase gates.
Find missing tests, unsupported claims, and reproducibility gaps. Do not
invent results. Draft claims only from measurements present in the
repository."

**Gate:** new users can install, test, run a benchmark, and launch the
demo; all advertised features are implemented/tested or clearly marked
experimental/planned.

## 4. Release gates

-   **v0.1:** exact search, ground truth, k-means, IVF-Flat, tests,
    benchmark harness, initial plots.
-   **v0.2:** reference adapters, one justified acceleration experiment,
    PQ and IVF-PQ.
-   **v0.3:** custom HNSW, invariant tests, sweeps, reference
    comparison.
-   **v0.4:** mutations, filters, compaction, persistence, diagnostics.
-   **v1.0:** semantic demo, reproducible commands, report,
    evidence-based claims.

Do not tag a release because a target date arrived. Defer or revise
scope transparently.

## 5. If time gets tight

1.  Correct reference distances, deterministic top-k, exact search, and
    ground truth.
2.  k-means and IVF-Flat with all-probe equivalence.
3.  Reproducible benchmark harness and one honest plot.
4.  Choose **one** next advanced algorithm---PQ/IVF-PQ or HNSW---and
    finish it before starting the other.
5.  Add the second algorithm only if the first is stable.
6.  Add mutations/filtering/persistence after core search is reliable.
7.  Finish the small demo and documentation.

This is a prioritization strategy, not a universal claim that one
algorithm is more valuable. A correct narrow project beats a broad
project with weak tests and untrustworthy numbers.

## 6. Weekly review checklist

-   What did I implement myself versus with agent help?
-   Can I explain the algorithm without reading the code?
-   Which tests demonstrate correctness?
-   What limitations or edge cases remain?
-   What did the benchmark actually measure?
-   Are README/resume claims stronger than the evidence?
-   What is the smallest next step to clear the current gate?

**Final rule:** optimize for algorithms you can explain, code you can
trust, and experiments another person can reproduce---not for the number
of algorithms in the README.
