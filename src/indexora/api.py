"""FastAPI service layer for Indexora — High-Performance First-Principles Vector Engine.

Exposes clean RESTful endpoints for interactive query execution, multi-index benchmark sweeps,
2D PCA spatial exploration, incremental mutations, sandboxed persistence, and neural semantic search.
Serves the React + TypeScript frontend statically when built in production.
"""

from __future__ import annotations

from dataclasses import asdict
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure src directory is on sys.path
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from indexora.exact import ExactBlasIndex, ExactFlatIndex
from indexora.hnsw import HNSWIndex
from indexora.ivf import IVFFlatIndex
from indexora.ivfpq import IVFPQIndex
from indexora.lab import (
    IndexBenchmarkMetrics,
    IndexCache,
    LabDataset,
    LabSession,
    compute_exact_ground_truth,
    evaluate_index_search,
    generate_lab_dataset,
    resolve_safe_storage_path,
    sanitize_index_name,
)

try:
    from indexora.semantic import (
        DEFAULT_EMBEDDING_MODEL,
        Document,
        SemanticSearchEngine,
        SemanticSearchResult,
        build_semantic_engine,
        encode_texts,
        get_embedding_model,
        is_sentence_transformers_available,
        load_demo_corpus,
    )
except ImportError:
    DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    Document = Any  # type: ignore[misc,assignment]
    SemanticSearchEngine = Any  # type: ignore[misc,assignment]
    SemanticSearchResult = Any  # type: ignore[misc,assignment]
    build_semantic_engine = None  # type: ignore[assignment]
    encode_texts = None  # type: ignore[assignment]
    get_embedding_model = None  # type: ignore[assignment]
    is_sentence_transformers_available = lambda: False  # type: ignore[assignment]
    load_demo_corpus = lambda: []  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# Global Application State
# -----------------------------------------------------------------------------
class EngineState:
    """Thread-safe engine container maintaining active lab dataset and stateful index session."""

    def __init__(self) -> None:
        self.dataset: LabDataset = generate_lab_dataset(
            n_base=120,
            n_queries=8,
            d=8,
            n_clusters=4,
            cluster_std=0.25,
            seed=42,
        )
        idx = ExactBlasIndex(database=self.dataset.base, ids=self.dataset.base_ids)
        self.session: LabSession = LabSession(index=idx, index_name="ExactBlasIndex", d=self.dataset.d)
        self.cache: IndexCache = IndexCache()
        self.corpus_docs = load_demo_corpus()
        self.cached_semantic_embeddings: Optional[np.ndarray] = None
        self.semantic_model: Any = None


state = EngineState()

app = FastAPI(
    title="Indexora Vector Engine API",
    description="High-performance, first-principles vector search engine API and interactive lab services.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Pydantic Request & Response Schemas
# -----------------------------------------------------------------------------
class DatasetGenerateRequest(BaseModel):
    n_base: int = Field(120, ge=20, le=2000, description="Number of base vectors")
    n_queries: int = Field(8, ge=1, le=50, description="Number of query vectors")
    d: int = Field(8, ge=4, le=64, description="Vector dimensionality")
    n_clusters: int = Field(4, ge=2, le=16, description="Number of clusters")
    cluster_std: float = Field(0.25, ge=0.05, le=1.5, description="Cluster dispersion std")
    seed: int = Field(42, ge=1, le=999999, description="RNG Seed")


class SearchRequest(BaseModel):
    index_type: str = Field("Exact BLAS", description="Index Architecture")
    query_id_index: Optional[int] = Field(0, description="Query index (0 to Q-1) or None if custom")
    custom_vector: Optional[List[float]] = Field(None, description="Custom vector of dimension d")
    k: int = Field(5, ge=1, le=50, description="Top-k items to retrieve")
    nprobe: int = Field(2, ge=1, le=16, description="Probed centroids for IVF-Flat and IVF-PQ")
    ef_search: int = Field(32, ge=4, le=256, description="Beam exploration depth for HNSW")
    filter_mode: str = Field("none", description="'none', 'even_ids', 'exclude_top1', or 'custom'")
    custom_filter_ids: Optional[List[int]] = Field(None, description="Allowed IDs when filter_mode is 'custom'")


class BenchmarkSweepRequest(BaseModel):
    k: int = Field(5, ge=1, le=30)
    repetitions: int = Field(3, ge=1, le=10)
    ivf_nlist: int = Field(4, ge=2, le=32)
    ivf_nprobe: int = Field(2, ge=1, le=16)
    pq_m: int = Field(2, ge=1, le=16)
    pq_ksub: int = Field(4, ge=2, le=256)
    hnsw_m: int = Field(8, ge=4, le=64)
    hnsw_ef_construction: int = Field(32, ge=8, le=128)
    hnsw_ef_search: int = Field(32, ge=4, le=128)


class MutationAddRequest(BaseModel):
    ext_id: int = Field(..., ge=1, le=99999999)
    constant_val: Optional[float] = Field(None, description="Scalar value to fill vector")
    vector: Optional[List[float]] = Field(None, description="Explicit d-dimensional vector")


class MutationDeleteRequest(BaseModel):
    ext_id: int = Field(..., ge=1, le=99999999)


class PersistenceSaveRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    overwrite: bool = True


class PersistenceLoadRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    verify_checksums: bool = True


class SemanticSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=20)
    category: Optional[str] = None
    index_type: str = Field("Exact BLAS")
    nprobe: int = Field(2, ge=1, le=16)
    ef_search: int = Field(32, ge=4, le=128)


# -----------------------------------------------------------------------------
# Internal Index Helper
# -----------------------------------------------------------------------------
def _get_or_build_index(
    index_type: str,
    ds: LabDataset,
    ivf_nlist: int = 4,
    pq_m: int = 2,
    pq_ksub: int = 4,
    hnsw_m: int = 8,
    hnsw_ef_construction: int = 32,
) -> tuple[Any, float, bool]:
    """Retrieve index from cache or build it if parameters changed."""
    if index_type == "Exact Flat (Ref)" or index_type == "Exact Flat (Reference)":
        build_params: dict[str, Any] = {}
        builder = lambda: ExactFlatIndex(database=ds.base, ids=ds.base_ids)
    elif index_type == "Exact BLAS":
        build_params = {}
        builder = lambda: ExactBlasIndex(database=ds.base, ids=ds.base_ids)
    elif index_type == "IVF-Flat":
        build_params = {"nlist": ivf_nlist}
        builder = lambda: IVFFlatIndex(
            database=ds.base,
            ids=ds.base_ids,
            nlist=ivf_nlist,
            nprobe=1,
            distance_method="blas",
            seed=ds.seed,
        )
    elif index_type == "IVF-PQ":
        build_params = {"nlist": ivf_nlist, "M": pq_m, "k_sub": pq_ksub}
        builder = lambda: IVFPQIndex(
            database=ds.base,
            ids=ds.base_ids,
            nlist=ivf_nlist,
            M=pq_m,
            k_sub=pq_ksub,
            nprobe=1,
            seed=ds.seed,
        )
    elif index_type == "HNSW":
        build_params = {"M": hnsw_m, "ef_construction": hnsw_ef_construction}
        builder = lambda: HNSWIndex(
            database=ds.base,
            ids=ds.base_ids,
            M=hnsw_m,
            ef_construction=hnsw_ef_construction,
            random_seed=ds.seed,
        )
    else:
        build_params = {}
        builder = lambda: ExactBlasIndex(database=ds.base, ids=ds.base_ids)

    return state.cache.get_or_build(index_type, ds, build_params, builder)


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/status")
def get_system_status() -> dict[str, Any]:
    """Return platform metadata, engine status, active dataset and session stats."""
    ds = state.dataset
    session = state.session
    return {
        "engine_version": "1.0.0",
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "dataset": {
            "n_base": ds.N,
            "n_queries": ds.Q,
            "d": ds.d,
            "n_clusters": ds.n_clusters,
            "seed": ds.seed,
            "id_range": [int(ds.base_ids[0]), int(ds.base_ids[-1])],
            "dataset_id": ds.dataset_id,
        },
        "session": {
            "index_name": session.index_name,
            "active_count": session.active_count,
            "d": session.d,
            "added_count": len(session.added_ids),
            "deleted_count": len(session.deleted_ids),
        },
        "semantic_available": is_sentence_transformers_available(),
    }


@app.post("/api/dataset/generate")
def generate_dataset(req: DatasetGenerateRequest) -> dict[str, Any]:
    """Regenerate deterministic synthetic clustered dataset and refresh session."""
    state.dataset = generate_lab_dataset(
        n_base=req.n_base,
        n_queries=req.n_queries,
        d=req.d,
        n_clusters=req.n_clusters,
        cluster_std=req.cluster_std,
        seed=req.seed,
    )
    ds = state.dataset
    idx = ExactBlasIndex(database=ds.base, ids=ds.base_ids)
    state.session = LabSession(index=idx, index_name="ExactBlasIndex", d=ds.d)
    state.cache.clear()

    return {
        "status": "success",
        "message": f"Generated dataset with N={ds.N}, d={ds.d}, clusters={ds.n_clusters}",
        "dataset": {
            "n_base": ds.N,
            "n_queries": ds.Q,
            "d": ds.d,
            "n_clusters": ds.n_clusters,
            "seed": ds.seed,
            "id_range": [int(ds.base_ids[0]), int(ds.base_ids[-1])],
        },
    }


@app.get("/api/dataset/current")
def get_current_dataset() -> dict[str, Any]:
    """Return detailed active dataset parameters, sample vectors, and cluster information."""
    ds = state.dataset
    sample_preview = [
        {"id": int(ds.base_ids[i]), "cluster": int(ds.cluster_labels[i]), "vector": [float(x) for x in ds.base[i]]}
        for i in range(min(12, ds.N))
    ]
    query_preview = [
        {"id": int(ds.query_ids[i]), "vector": [float(x) for x in ds.queries[i]]}
        for i in range(min(8, ds.Q))
    ]
    return {
        "n_base": ds.N,
        "n_queries": ds.Q,
        "d": ds.d,
        "n_clusters": ds.n_clusters,
        "seed": ds.seed,
        "base_samples": sample_preview,
        "query_samples": query_preview,
    }


@app.post("/api/search")
def search_index(req: SearchRequest) -> dict[str, Any]:
    """Execute vector retrieval against specified index with full ground-truth validation."""
    ds = state.dataset

    # Resolve query vector
    if req.custom_vector is not None:
        if len(req.custom_vector) != ds.d:
            raise HTTPException(
                status_code=400,
                detail=f"Custom vector dimension {len(req.custom_vector)} does not match dataset dimension {ds.d}",
            )
        query_vec = np.array(req.custom_vector, dtype=np.float32)
        q_label = "Custom Vector"
    else:
        q_idx = req.query_id_index if req.query_id_index is not None else 0
        if q_idx < 0 or q_idx >= ds.Q:
            raise HTTPException(status_code=400, detail=f"Query index must be between 0 and {ds.Q - 1}")
        query_vec = ds.queries[q_idx]
        q_label = f"Query #{q_idx} (ID {int(ds.query_ids[q_idx])})"

    # Filter predicate setup
    applied_filter: Any = None
    if req.filter_mode == "even_ids":
        applied_filter = lambda eid: eid % 2 == 0
    elif req.filter_mode == "exclude_top1":
        unfiltered_gt = compute_exact_ground_truth(database=ds.base, ids=ds.base_ids, query=query_vec, k=1)
        top1_id = int(unfiltered_gt.ids[0]) if len(unfiltered_gt.ids) > 0 else -1
        applied_filter = lambda eid, excluded=top1_id: eid != excluded
    elif req.filter_mode == "custom" and req.custom_filter_ids:
        applied_filter = set(req.custom_filter_ids)

    # Compute exact ground truth for this query and filter
    gt_res = compute_exact_ground_truth(
        database=ds.base,
        ids=ds.base_ids,
        query=query_vec,
        k=req.k,
        filter_ids=applied_filter,
    )
    eligible_gt_ids = set(gt_res.ids.tolist())
    num_eligible = len(eligible_gt_ids)

    # Build or retrieve index
    idx_instance, build_time_ms, was_cached = _get_or_build_index(
        index_type=req.index_type,
        ds=ds,
        ivf_nlist=4,
        pq_m=2,
        pq_ksub=4,
        hnsw_m=8,
        hnsw_ef_construction=32,
    )

    # Execute search
    search_kwargs: dict[str, Any] = {}
    if "IVF" in req.index_type:
        search_kwargs["nprobe"] = req.nprobe
    elif "HNSW" in req.index_type:
        search_kwargs["ef_search"] = req.ef_search

    t0 = time.perf_counter()
    search_res = idx_instance.search(query_vec, k=req.k, filter_ids=applied_filter, **search_kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    retrieved_ids = search_res.ids.tolist()
    distances = search_res.distances.tolist()
    hits = len(set(retrieved_ids) & eligible_gt_ids)
    recall = (hits / num_eligible) if num_eligible > 0 else (1.0 if len(retrieved_ids) == 0 else 0.0)

    results_table = []
    for rank, (dist, id_val) in enumerate(zip(distances, retrieved_ids), start=1):
        is_exact = id_val in eligible_gt_ids
        results_table.append({
            "rank": rank,
            "id": int(id_val),
            "squared_l2": float(dist),
            "is_exact_match": is_exact,
        })

    return {
        "index_type": req.index_type,
        "query_label": q_label,
        "k": req.k,
        "latency_ms": latency_ms,
        "distance_computations": int(search_res.distance_computations) if search_res.distance_computations is not None else ds.N,
        "recall_at_k": float(recall),
        "num_eligible_ground_truth": num_eligible,
        "was_cached": was_cached,
        "build_time_ms": build_time_ms,
        "results": results_table,
    }


@app.post("/api/benchmark/sweep")
def run_benchmark_sweep(req: BenchmarkSweepRequest) -> dict[str, Any]:
    """Execute live microsecond multi-index benchmark sweep across all 5 architectures."""
    ds = state.dataset
    exact_gt = compute_exact_ground_truth(database=ds.base, ids=ds.base_ids, query=ds.queries, k=req.k)

    bench_configs = [
        ("Exact Flat (Reference)", lambda: ExactFlatIndex(database=ds.base, ids=ds.base_ids), {}),
        ("Exact BLAS", lambda: ExactBlasIndex(database=ds.base, ids=ds.base_ids), {}),
        (
            f"IVF-Flat (nlist={req.ivf_nlist}, nprobe={req.ivf_nprobe})",
            lambda: IVFFlatIndex(
                database=ds.base,
                ids=ds.base_ids,
                nlist=req.ivf_nlist,
                nprobe=req.ivf_nprobe,
                distance_method="blas",
                seed=ds.seed,
            ),
            {"nprobe": req.ivf_nprobe},
        ),
        (
            f"IVF-PQ (M={req.pq_m}, k_sub={req.pq_ksub}, nprobe={req.ivf_nprobe})",
            lambda: IVFPQIndex(
                database=ds.base,
                ids=ds.base_ids,
                nlist=req.ivf_nlist,
                M=req.pq_m,
                k_sub=req.pq_ksub,
                nprobe=req.ivf_nprobe,
                seed=ds.seed,
            ),
            {"nprobe": req.ivf_nprobe},
        ),
        (
            f"HNSW (M={req.hnsw_m}, ef={req.hnsw_ef_search})",
            lambda: HNSWIndex(
                database=ds.base,
                ids=ds.base_ids,
                M=req.hnsw_m,
                ef_construction=req.hnsw_ef_construction,
                random_seed=ds.seed,
            ),
            {"ef_search": req.hnsw_ef_search},
        ),
    ]

    metrics: list[dict[str, Any]] = []
    for name, factory, search_kwargs in bench_configs:
        t0 = time.perf_counter()
        idx = factory()
        build_time_ms = (time.perf_counter() - t0) * 1000.0

        m: IndexBenchmarkMetrics = evaluate_index_search(
            index=idx,
            index_name=name,
            queries=ds.queries,
            ground_truth=exact_gt,
            k=req.k,
            repetitions=req.repetitions,
            build_time_ms=build_time_ms,
            search_kwargs=search_kwargs,
        )
        metrics.append({
            "index_name": m.index_name,
            "build_time_ms": float(m.build_time_ms),
            "median_latency_ms": float(m.median_latency_ms),
            "p95_latency_ms": float(m.p95_latency_ms),
            "qps": float(m.qps),
            "recall_at_k": float(m.recall_at_k),
            "distance_computations_per_query": float(m.distance_computations_per_query) if m.distance_computations_per_query is not None else ds.N,
            "memory_kb": float(m.memory_kb),
            "memory_method": m.memory_method,
        })

    return {"k": req.k, "n_queries": ds.Q, "metrics": metrics}


@app.get("/api/benchmark/sift")
def get_sift_evidence() -> dict[str, Any]:
    """Return precomputed, verified canonical SIFT-100K and SIFT-10K benchmark tables and metadata."""
    evidence: dict[str, Any] = {
        "provenance": {
            "platform": "Windows 10 AMD64",
            "cpu": "Intel64 Family 6 Model 186 Stepping 2, GenuineIntel",
            "python": "3.10.11",
            "numpy": "2.2.6 (linked to optimized BLAS)",
            "git_commit": "c84a3dc",
            "dataset_sift100k_sha256": "21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816",
            "queries_sift100k_sha256": "f7fc9be140accdfd64116c2fa2365ecdb69b8f084970c6b0532db5ff79ac8fdc",
        },
        "sift100k_baseline": [],
        "sift100k_ivfpq": [],
        "sift10k_hnsw": [],
    }

    def _read_csv(p: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if not p.exists():
            return records
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed: dict[str, Any] = {}
                for k, v in row.items():
                    if v is None:
                        parsed[k] = None
                        continue
                    v_str = str(v).strip()
                    try:
                        if "." in v_str:
                            parsed[k] = float(v_str)
                        else:
                            parsed[k] = int(v_str)
                    except ValueError:
                        parsed[k] = v_str
                records.append(parsed)
        return records

    # 1. SIFT-100K Baseline & IVF-Flat
    evidence["sift100k_baseline"] = _read_csv(Path("results/sift100k_p66/benchmark_summary.csv"))

    # 2. SIFT-100K IVF-PQ
    evidence["sift100k_ivfpq"] = _read_csv(Path("results/sift100k_ivfpq/benchmark_summary.csv"))

    # 3. SIFT-10K HNSW
    evidence["sift10k_hnsw"] = _read_csv(Path("results/sift10k_hnsw/benchmark_summary.csv"))

    return evidence


@app.get("/api/explorer/pca")
def get_pca_projection(query_id_index: int = 0, retrieved_ids: Optional[str] = None) -> dict[str, Any]:
    """Compute and return 2D PCA projected points for base vectors, query, centroids, and highlights."""
    ds = state.dataset
    base_proj = ds.project_2d(ds.base)
    centroids_proj = ds.project_2d(ds.centroids)

    q_idx = max(0, min(query_id_index, ds.Q - 1))
    q_proj = ds.project_2d(ds.queries[q_idx])

    retrieved_set: set[int] = set()
    if retrieved_ids:
        try:
            retrieved_set = set(int(x.strip()) for x in retrieved_ids.split(",") if x.strip())
        except ValueError:
            pass

    points = []
    for i in range(ds.N):
        eid = int(ds.base_ids[i])
        points.append({
            "id": eid,
            "x": float(base_proj[i, 0]),
            "y": float(base_proj[i, 1]),
            "cluster": int(ds.cluster_labels[i]),
            "is_retrieved": eid in retrieved_set,
        })

    centroids = [
        {"cluster": i, "x": float(centroids_proj[i, 0]), "y": float(centroids_proj[i, 1])}
        for i in range(ds.n_clusters)
    ]

    return {
        "points": points,
        "centroids": centroids,
        "query": {
            "query_index": q_idx,
            "id": int(ds.query_ids[q_idx]),
            "x": float(q_proj[0]),
            "y": float(q_proj[1]),
        },
        "d": ds.d,
        "n_clusters": ds.n_clusters,
    }


@app.post("/api/mutations/add")
def add_vector(req: MutationAddRequest) -> dict[str, Any]:
    """Incrementally add a vector to active index with external entity ID."""
    session = state.session
    if req.vector is not None:
        if len(req.vector) != session.d:
            raise HTTPException(status_code=400, detail=f"Vector length {len(req.vector)} != {session.d}")
        vec = np.array(req.vector, dtype=np.float32)
    elif req.constant_val is not None:
        vec = np.full(session.d, req.constant_val, dtype=np.float32)
    else:
        raise HTTPException(status_code=400, detail="Must provide either 'vector' or 'constant_val'")

    try:
        assigned = session.add_vector(vec, ext_id=req.ext_id)
        return {
            "status": "success",
            "assigned_id": int(assigned),
            "active_count": session.active_count,
            "total_added": len(session.added_ids),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/mutations/delete")
def delete_vector(req: MutationDeleteRequest) -> dict[str, Any]:
    """Delete vector external entity ID from active index."""
    session = state.session
    try:
        del_count = session.delete_vector(req.ext_id)
        return {
            "status": "success" if del_count > 0 else "not_found",
            "deleted_id": req.ext_id,
            "deleted_count": del_count,
            "active_count": session.active_count,
            "total_deleted": len(session.deleted_ids),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/persistence/list")
def list_saved_indexes() -> dict[str, Any]:
    """List valid saved indexes inside sandboxed storage directory."""
    return {"saved_indexes": state.session.list_saved()}


@app.post("/api/persistence/save")
def save_named_index(req: PersistenceSaveRequest) -> dict[str, Any]:
    """Safely persist active index inside storage_root using a validated index name."""
    try:
        manifest = state.session.save_named(req.name, overwrite=req.overwrite)
        return {"status": "success", "name": req.name, "manifest": manifest}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/persistence/load")
def load_named_index(req: PersistenceLoadRequest) -> dict[str, Any]:
    """Safely load and checksum-validate an index from storage_root."""
    try:
        restored = state.session.load_named(req.name, verify_checksums=req.verify_checksums)
        # Run test verification search
        q_test = np.zeros(state.session.d, dtype=np.float32)
        v_res = state.session.search(q_test, k=3)
        return {
            "status": "success",
            "name": req.name,
            "index_type": type(restored).__name__,
            "active_count": state.session.active_count,
            "verification_top3_ids": v_res.ids.tolist(),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/semantic/status")
def get_semantic_status() -> dict[str, Any]:
    """Return neural semantic search availability and document corpus metadata."""
    has_st = is_sentence_transformers_available()
    corpus_docs = state.corpus_docs
    categories = sorted({d.category for d in corpus_docs}) if corpus_docs else []
    return {
        "available": has_st,
        "model_name": DEFAULT_EMBEDDING_MODEL,
        "document_count": len(corpus_docs),
        "categories": categories,
        "corpus_sample": [
            {"doc_id": d.doc_id, "category": d.category, "title": d.title, "text": d.text, "tags": d.metadata.get("tags", [])}
            for d in corpus_docs
        ],
    }


@app.post("/api/semantic/search")
def search_semantic(req: SemanticSearchRequest) -> dict[str, Any]:
    """Execute neural semantic text retrieval using dense embeddings and Indexora index."""
    if not is_sentence_transformers_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neural semantic retrieval requires 'sentence-transformers' and 'PyTorch'. Install via: pip install -e .[semantic]",
        )

    corpus_docs = state.corpus_docs
    if not corpus_docs:
        raise HTTPException(status_code=400, detail="Corpus document set is empty")

    if state.semantic_model is None:
        state.semantic_model = get_embedding_model(DEFAULT_EMBEDDING_MODEL)

    if state.cached_semantic_embeddings is None:
        texts = [d.text for d in corpus_docs]
        state.cached_semantic_embeddings = encode_texts(texts, model=state.semantic_model, normalize=True)

    engine = SemanticSearchEngine(
        corpus=corpus_docs,
        embeddings=state.cached_semantic_embeddings,
        index_type=req.index_type,
        embedder=state.semantic_model,
        model_name=DEFAULT_EMBEDDING_MODEL,
        ivf_nlist=4,
        ivf_nprobe=req.nprobe,
        hnsw_m=8,
        hnsw_ef_construction=32,
        hnsw_ef_search=req.ef_search,
    )

    cat_filter = None if req.category in ("All Categories", None, "") else req.category
    res: SemanticSearchResult = engine.search(
        query=req.query,
        k=req.k,
        filter_category=cat_filter,
        nprobe=req.nprobe,
        ef_search=req.ef_search,
    )

    ranked_docs = []
    for rank, (doc, dist, is_match, cos_sim) in enumerate(
        zip(res.retrieved_documents, res.distances, res.is_exact_match, res.cosine_similarities),
        start=1,
    ):
        ranked_docs.append({
            "rank": rank,
            "doc_id": doc.doc_id,
            "category": doc.category,
            "title": doc.title,
            "text": doc.text,
            "tags": doc.metadata.get("tags", []),
            "cosine_similarity": float(cos_sim),
            "squared_l2": float(dist),
            "is_oracle_hit": bool(is_match),
        })

    oracle_agreement_pct = (sum(res.is_exact_match) / len(res.is_exact_match) * 100.0) if res.is_exact_match else 0.0

    return {
        "query": req.query,
        "k": req.k,
        "category_filter": cat_filter,
        "total_latency_ms": float(res.latency_ms + res.embed_time_ms),
        "text_embed_time_ms": float(res.embed_time_ms),
        "vector_search_time_ms": float(res.latency_ms),
        "distance_computations": int(res.distance_computations) if res.distance_computations is not None else len(corpus_docs),
        "oracle_agreement_pct": float(oracle_agreement_pct),
        "ranked_documents": ranked_docs,
    }


@app.get("/healthz", tags=["Health"])
def health_check() -> dict[str, str]:
    """Lightweight health-check endpoint for Render / Kubernetes."""
    return {"status": "ok", "engine": "Indexora", "version": "1.0.0"}


# -----------------------------------------------------------------------------
# Static Frontend Mount & SPA Routing Fallback
# -----------------------------------------------------------------------------
_FRONTEND_DIST = _SRC_DIR.parent / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting Indexora Production Server on {host}:{port}")
    uvicorn.run("indexora.api:app", host=host, port=port, reload=False)
