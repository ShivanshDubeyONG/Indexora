"""Index persistence, serialization, and deserialization for Indexora.

This module provides atomic, safe local persistence (save and load) for Indexora vector indexes:
- ExactFlatIndex
- ExactBlasIndex
- IVFFlatIndex
- IVFPQIndex
- HNSWIndex

Persistence guarantees:
- Safe NumPy storage (no unsafe Python pickle deserialization).
- Atomic directory writes via temporary staging and safe replacement.
- Provenance and integrity manifests (`manifest.json`) with file SHA-256 checksums.
- Hyperparameter, dimension, and dtype validation on load.
- Exact graph topology restoration for HNSW (no graph rebuilds).
- Exact codebook and posting restoration for IVF-PQ (no retraining).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any
import uuid

import numpy as np


SCHEMA_VERSION = "1.0"
FORMAT_IDENTIFIER = "indexora_index"


def _compute_sha256(filepath: Path) -> str:
    """Compute the SHA-256 checksum of a file in streaming chunks."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _save_array(staging_dir: Path, filename: str, arr: np.ndarray) -> dict[str, Any]:
    """Save a numpy array safely without pickle and return its metadata with sha256."""
    filepath = staging_dir / filename
    np.save(filepath, arr, allow_pickle=False)
    sha = _compute_sha256(filepath)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": sha,
    }


def _load_and_validate_array(
    directory: Path,
    filename: str,
    meta: dict[str, Any],
    verify_checksums: bool = True,
) -> np.ndarray:
    """Load and validate a numpy array from an index directory."""
    dir_resolved = directory.resolve()
    filepath = (directory / filename).resolve()
    if not filepath.is_relative_to(dir_resolved):
        raise ValueError(f"Path traversal detected in manifest filename: '{filename}'")

    if not filepath.exists():
        raise FileNotFoundError(f"Required index array file '{filename}' not found in {directory}")

    if verify_checksums:
        actual_sha = _compute_sha256(filepath)
        expected_sha = meta.get("sha256")
        if expected_sha and actual_sha != expected_sha:
            raise ValueError(
                f"Integrity check failed for '{filename}': expected SHA-256 {expected_sha}, got {actual_sha}"
            )

    arr = np.load(filepath, allow_pickle=False)
    expected_shape = tuple(meta["shape"])
    expected_dtype = meta["dtype"]

    if arr.shape != expected_shape:
        raise ValueError(
            f"Shape mismatch in '{filename}': expected {expected_shape}, got {arr.shape}"
        )
    if str(arr.dtype) != expected_dtype:
        raise TypeError(
            f"Dtype mismatch in '{filename}': expected {expected_dtype}, got {arr.dtype}"
        )

    return arr


# -----------------------------------------------------------------------------
# Save Dispatchers
# -----------------------------------------------------------------------------

def _serialize_exact_flat(index: Any, staging_dir: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "format": FORMAT_IDENTIFIER,
        "schema_version": SCHEMA_VERSION,
        "index_type": "ExactFlatIndex",
        "d": index.d,
        "ntotal": index.n,
        "arrays": {},
    }
    manifest["arrays"]["database.npy"] = _save_array(staging_dir, "database.npy", index.database)
    manifest["arrays"]["ids.npy"] = _save_array(staging_dir, "ids.npy", index.ids)
    return manifest


def _serialize_exact_blas(index: Any, staging_dir: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "format": FORMAT_IDENTIFIER,
        "schema_version": SCHEMA_VERSION,
        "index_type": "ExactBlasIndex",
        "d": index.d,
        "ntotal": index.n,
        "arrays": {},
    }
    manifest["arrays"]["database.npy"] = _save_array(staging_dir, "database.npy", index.database)
    manifest["arrays"]["ids.npy"] = _save_array(staging_dir, "ids.npy", index.ids)
    return manifest


def _serialize_ivf_flat(index: Any, staging_dir: Path) -> dict[str, Any]:
    # Flatten inverted lists into lengths, flat_vectors, flat_ids
    nlist = index.nlist
    list_lengths = np.array([lst.size for lst in index.invlists], dtype=np.int64)

    non_empty_vecs = [lst.vectors for lst in index.invlists if lst.size > 0]
    non_empty_ids = [lst.ids for lst in index.invlists if lst.size > 0]

    flat_vectors = np.concatenate(non_empty_vecs, axis=0) if non_empty_vecs else np.empty((0, index.d), dtype=np.float32)
    flat_ids = np.concatenate(non_empty_ids, axis=0) if non_empty_ids else np.empty((0,), dtype=np.int64)

    manifest: dict[str, Any] = {
        "format": FORMAT_IDENTIFIER,
        "schema_version": SCHEMA_VERSION,
        "index_type": "IVFFlatIndex",
        "d": index.d,
        "ntotal": index.ntotal,
        "hyperparameters": {
            "nlist": nlist,
            "nprobe": index.nprobe,
            "seed": getattr(index, "_seed", None),
            "distance_method": getattr(index, "_distance_method", "blas"),
            "kmeans_n_iter": getattr(index, "kmeans_n_iter", None),
            "kmeans_converged": getattr(index, "kmeans_converged", None),
            "kmeans_inertia": getattr(index, "kmeans_inertia", None),
        },
        "arrays": {},
    }
    manifest["arrays"]["centroids.npy"] = _save_array(staging_dir, "centroids.npy", index.centroids)
    manifest["arrays"]["list_lengths.npy"] = _save_array(staging_dir, "list_lengths.npy", list_lengths)
    manifest["arrays"]["flat_vectors.npy"] = _save_array(staging_dir, "flat_vectors.npy", flat_vectors)
    manifest["arrays"]["flat_ids.npy"] = _save_array(staging_dir, "flat_ids.npy", flat_ids)
    return manifest


def _serialize_ivf_pq(index: Any, staging_dir: Path) -> dict[str, Any]:
    nlist = index.nlist
    list_lengths = np.array([lst.size for lst in index.invlists], dtype=np.int64)

    non_empty_codes = [lst.codes for lst in index.invlists if lst.size > 0]
    non_empty_ids = [lst.ids for lst in index.invlists if lst.size > 0]

    flat_codes = np.concatenate(non_empty_codes, axis=0) if non_empty_codes else np.empty((0, index.M), dtype=index.code_dtype)
    flat_ids = np.concatenate(non_empty_ids, axis=0) if non_empty_ids else np.empty((0,), dtype=np.int64)

    manifest: dict[str, Any] = {
        "format": FORMAT_IDENTIFIER,
        "schema_version": SCHEMA_VERSION,
        "index_type": "IVFPQIndex",
        "d": index.d,
        "ntotal": index.ntotal,
        "hyperparameters": {
            "nlist": nlist,
            "nprobe": index.nprobe,
            "M": index.M,
            "d_sub": index.d_sub,
            "k_sub": index.k_sub,
            "use_residuals": index.use_residuals,
            "seed": getattr(index, "_seed", None),
            "code_dtype": str(index.code_dtype),
            "kmeans_n_iter": getattr(index, "kmeans_n_iter", None),
            "kmeans_converged": getattr(index, "kmeans_converged", None),
            "kmeans_inertia": getattr(index, "kmeans_inertia", None),
        },
        "arrays": {},
    }
    manifest["arrays"]["coarse_centroids.npy"] = _save_array(staging_dir, "coarse_centroids.npy", index.coarse_centroids)
    manifest["arrays"]["codebooks.npy"] = _save_array(staging_dir, "codebooks.npy", index.pq.codebooks)
    manifest["arrays"]["list_lengths.npy"] = _save_array(staging_dir, "list_lengths.npy", list_lengths)
    manifest["arrays"]["flat_codes.npy"] = _save_array(staging_dir, "flat_codes.npy", flat_codes)
    manifest["arrays"]["flat_ids.npy"] = _save_array(staging_dir, "flat_ids.npy", flat_ids)
    return manifest


def _serialize_hnsw(index: Any, staging_dir: Path) -> dict[str, Any]:
    # Extract CSR graph representation per level
    max_level = index.max_level
    graphs = getattr(index, "_graphs", [])

    level_meta: list[dict[str, Any]] = []
    manifest_arrays: dict[str, Any] = {}

    for lvl in range(max_level + 1):
        lvl_graphs = graphs[lvl] if lvl < len(graphs) else []
        node_count = len(lvl_graphs)
        offsets = np.zeros(node_count + 1, dtype=np.int64)
        if node_count > 0:
            lengths = [len(adj) for adj in lvl_graphs]
            offsets[1:] = np.cumsum(lengths)
            flat_edges = np.array([v for adj in lvl_graphs for v in adj], dtype=np.int64)
        else:
            flat_edges = np.empty((0,), dtype=np.int64)

        offsets_file = f"graph_lvl_{lvl}_offsets.npy"
        edges_file = f"graph_lvl_{lvl}_edges.npy"

        manifest_arrays[offsets_file] = _save_array(staging_dir, offsets_file, offsets)
        manifest_arrays[edges_file] = _save_array(staging_dir, edges_file, flat_edges)

        level_meta.append({
            "level": lvl,
            "node_count": node_count,
            "offsets_file": offsets_file,
            "edges_file": edges_file,
        })

    node_levels = np.array(getattr(index, "_node_levels", []), dtype=np.int32)
    deleted_nodes = np.array(sorted(getattr(index, "_deleted_nodes", set())), dtype=np.int64)

    manifest: dict[str, Any] = {
        "format": FORMAT_IDENTIFIER,
        "schema_version": SCHEMA_VERSION,
        "index_type": "HNSWIndex",
        "d": index.d,
        "ntotal": index.ntotal,
        "n_active": getattr(index, "_n_active", index.ntotal),
        "hyperparameters": {
            "M": index.M,
            "M0": index.M0,
            "ef_construction": index.ef_construction,
            "ef_search": index.ef_search,
            "random_seed": getattr(index, "_random_seed", None),
            "keep_pruned": getattr(index, "_keep_pruned", True),
            "max_level": max_level,
            "entry_point": index.entry_point,
            "level_meta": level_meta,
        },
        "arrays": manifest_arrays,
    }

    manifest["arrays"]["database.npy"] = _save_array(staging_dir, "database.npy", index.database)
    manifest["arrays"]["ids.npy"] = _save_array(staging_dir, "ids.npy", index.ids)
    manifest["arrays"]["node_levels.npy"] = _save_array(staging_dir, "node_levels.npy", node_levels)
    manifest["arrays"]["deleted_nodes.npy"] = _save_array(staging_dir, "deleted_nodes.npy", deleted_nodes)
    return manifest


# -----------------------------------------------------------------------------
# Public Save API
# -----------------------------------------------------------------------------

def save_index(
    index: Any,
    directory: str | Path,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Atomically save an Indexora vector index to disk with manifest and checksums.

    Parameters
    ----------
    index : Any
        An instance of `ExactFlatIndex`, `ExactBlasIndex`, `IVFFlatIndex`,
        `IVFPQIndex`, or `HNSWIndex`.
    directory : str or Path
        Target directory to write index files.
    overwrite : bool, default=True
        Whether to overwrite an existing directory.

    Returns
    -------
    dict[str, Any]
        The written manifest metadata.

    Raises
    ------
    FileExistsError
        If `directory` exists and `overwrite=False`.
    TypeError
        If `index` is not a supported Indexora index type.
    """
    dest = Path(directory).resolve()
    if dest.exists() and not overwrite:
        raise FileExistsError(f"Directory {dest} already exists and overwrite=False")

    dest.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = dest.parent / f".tmp_{dest.name}_{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        index_type_name = type(index).__name__
        if index_type_name == "ExactFlatIndex":
            manifest = _serialize_exact_flat(index, staging_dir)
        elif index_type_name == "ExactBlasIndex":
            manifest = _serialize_exact_blas(index, staging_dir)
        elif index_type_name == "IVFFlatIndex":
            manifest = _serialize_ivf_flat(index, staging_dir)
        elif index_type_name == "IVFPQIndex":
            manifest = _serialize_ivf_pq(index, staging_dir)
        elif index_type_name == "HNSWIndex":
            manifest = _serialize_hnsw(index, staging_dir)
        else:
            raise TypeError(f"Index type '{index_type_name}' does not support persistence.")

        manifest_path = staging_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Atomic replacement of dest
        if dest.exists():
            backup_dir = dest.parent / f".backup_{dest.name}_{uuid.uuid4().hex}"
            try:
                dest.rename(backup_dir)
                staging_dir.rename(dest)
                shutil.rmtree(backup_dir, ignore_errors=True)
            except Exception:
                if backup_dir.exists() and not dest.exists():
                    backup_dir.rename(dest)
                raise
        else:
            staging_dir.rename(dest)

        return manifest

    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise


# -----------------------------------------------------------------------------
# Load Dispatchers
# -----------------------------------------------------------------------------

def _deserialize_exact_flat(directory: Path, manifest: dict[str, Any], verify_checksums: bool) -> Any:
    from indexora.exact import ExactFlatIndex
    arrays = manifest["arrays"]
    database = _load_and_validate_array(directory, "database.npy", arrays["database.npy"], verify_checksums)
    ids = _load_and_validate_array(directory, "ids.npy", arrays["ids.npy"], verify_checksums)
    return ExactFlatIndex(database=database, ids=ids)


def _deserialize_exact_blas(directory: Path, manifest: dict[str, Any], verify_checksums: bool) -> Any:
    from indexora.exact import ExactBlasIndex
    arrays = manifest["arrays"]
    database = _load_and_validate_array(directory, "database.npy", arrays["database.npy"], verify_checksums)
    ids = _load_and_validate_array(directory, "ids.npy", arrays["ids.npy"], verify_checksums)
    return ExactBlasIndex(database=database, ids=ids)


def _deserialize_ivf_flat(directory: Path, manifest: dict[str, Any], verify_checksums: bool) -> Any:
    from indexora.ivf import IVFFlatIndex, InvertedList
    arrays = manifest["arrays"]
    hp = manifest.get("hyperparameters", {})

    centroids = _load_and_validate_array(directory, "centroids.npy", arrays["centroids.npy"], verify_checksums)
    list_lengths = _load_and_validate_array(directory, "list_lengths.npy", arrays["list_lengths.npy"], verify_checksums)
    flat_vectors = _load_and_validate_array(directory, "flat_vectors.npy", arrays["flat_vectors.npy"], verify_checksums)
    flat_ids = _load_and_validate_array(directory, "flat_ids.npy", arrays["flat_ids.npy"], verify_checksums)

    nlist = int(hp["nlist"])
    nprobe = int(hp.get("nprobe", 1))
    seed = hp.get("seed", 42)

    # Reconstruct inverted lists
    invlists: list[InvertedList] = []
    offset = 0
    d = int(manifest["d"])
    for c in range(nlist):
        sz = int(list_lengths[c])
        if sz > 0:
            v_slice = flat_vectors[offset : offset + sz]
            id_slice = flat_ids[offset : offset + sz]
            offset += sz
        else:
            v_slice = np.empty((0, d), dtype=np.float32)
            id_slice = np.empty((0,), dtype=np.int64)
        invlists.append(InvertedList(ids=id_slice, vectors=v_slice))

    # Instantiate index without re-running kmeans
    index = IVFFlatIndex.__new__(IVFFlatIndex)
    index._centroids = centroids
    index._nlist = nlist
    index._nprobe = nprobe
    index._d = d
    index._ntotal = int(manifest["ntotal"])
    index._distance_method = hp.get("distance_method", "blas")
    index._invlists = invlists
    index._kmeans_n_iter = hp.get("kmeans_n_iter")
    index._kmeans_converged = hp.get("kmeans_converged")
    index._kmeans_inertia = hp.get("kmeans_inertia")

    return index


def _deserialize_ivf_pq(directory: Path, manifest: dict[str, Any], verify_checksums: bool) -> Any:
    from indexora.ivfpq import IVFPQIndex, InvertedPQList
    from indexora.pq import ProductQuantizer
    arrays = manifest["arrays"]
    hp = manifest.get("hyperparameters", {})

    coarse_centroids = _load_and_validate_array(
        directory, "coarse_centroids.npy", arrays["coarse_centroids.npy"], verify_checksums
    )
    codebooks = _load_and_validate_array(
        directory, "codebooks.npy", arrays["codebooks.npy"], verify_checksums
    )
    list_lengths = _load_and_validate_array(
        directory, "list_lengths.npy", arrays["list_lengths.npy"], verify_checksums
    )
    flat_codes = _load_and_validate_array(
        directory, "flat_codes.npy", arrays["flat_codes.npy"], verify_checksums
    )
    flat_ids = _load_and_validate_array(
        directory, "flat_ids.npy", arrays["flat_ids.npy"], verify_checksums
    )

    d = int(manifest["d"])
    nlist = int(hp["nlist"])
    nprobe = int(hp.get("nprobe", 1))
    M = int(hp["M"])
    d_sub = int(hp["d_sub"])
    k_sub = int(hp["k_sub"])
    use_residuals = bool(hp.get("use_residuals", hp.get("use_residual", True)))
    seed = hp.get("seed", 42)

    # Reconstruct ProductQuantizer with exact codebooks restored (no retraining!)
    pq = ProductQuantizer.__new__(ProductQuantizer)
    pq._d = d
    pq._M = M
    pq._k_sub = k_sub
    pq._d_sub = d_sub
    pq._seed = seed
    pq._codebooks = codebooks
    pq._is_trained = True

    # Reconstruct inverted lists
    invlists: list[InvertedPQList] = []
    offset = 0
    for c in range(nlist):
        sz = int(list_lengths[c])
        if sz > 0:
            c_slice = flat_codes[offset : offset + sz]
            id_slice = flat_ids[offset : offset + sz]
            offset += sz
        else:
            c_slice = np.empty((0, M), dtype=flat_codes.dtype)
            id_slice = np.empty((0,), dtype=np.int64)
        invlists.append(InvertedPQList(ids=id_slice, codes=c_slice))

    index = IVFPQIndex.__new__(IVFPQIndex)
    index._d = d
    index._nlist = nlist
    index._M = M
    index._k_sub = k_sub
    index._d_sub = d_sub
    index._nprobe = nprobe
    index._use_residuals = use_residuals
    index._ntotal = int(manifest["ntotal"])
    index._centroids = coarse_centroids
    index._pq = pq
    index._invlists = invlists
    index._kmeans_n_iter = hp.get("kmeans_n_iter")
    index._kmeans_converged = hp.get("kmeans_converged")
    index._kmeans_inertia = hp.get("kmeans_inertia")

    return index


def _deserialize_hnsw(directory: Path, manifest: dict[str, Any], verify_checksums: bool) -> Any:
    from indexora.hnsw import HNSWIndex
    import math

    arrays = manifest["arrays"]
    hp = manifest.get("hyperparameters", {})

    database = _load_and_validate_array(directory, "database.npy", arrays["database.npy"], verify_checksums)
    ids = _load_and_validate_array(directory, "ids.npy", arrays["ids.npy"], verify_checksums)
    node_levels = _load_and_validate_array(directory, "node_levels.npy", arrays["node_levels.npy"], verify_checksums)
    deleted_nodes = _load_and_validate_array(directory, "deleted_nodes.npy", arrays["deleted_nodes.npy"], verify_checksums)

    d = int(manifest["d"])
    M = int(hp["M"])
    M0 = int(hp.get("M0", 2 * M))
    ef_construction = int(hp.get("ef_construction", 64))
    ef_search = int(hp.get("ef_search", 32))
    random_seed = hp.get("random_seed")
    keep_pruned = bool(hp.get("keep_pruned", True))
    max_level = int(hp.get("max_level", -1))
    entry_point = hp.get("entry_point")
    if entry_point is not None:
        entry_point = int(entry_point)

    level_meta = hp.get("level_meta", [])
    graphs: list[list[list[int]]] = []

    for l_info in level_meta:
        lvl = int(l_info["level"])
        node_count = int(l_info["node_count"])
        offsets_file = l_info["offsets_file"]
        edges_file = l_info["edges_file"]

        offsets = _load_and_validate_array(directory, offsets_file, arrays[offsets_file], verify_checksums)
        flat_edges = _load_and_validate_array(directory, edges_file, arrays[edges_file], verify_checksums)

        lvl_graphs: list[list[int]] = []
        for i in range(node_count):
            start = int(offsets[i])
            end = int(offsets[i + 1])
            lvl_graphs.append(list(flat_edges[start:end]))

        while len(graphs) <= lvl:
            graphs.append([])
        graphs[lvl] = lvl_graphs

    index = HNSWIndex.__new__(HNSWIndex)
    index._d = d
    index._M = M
    index._M0 = M0
    index._ef_construction = ef_construction
    index._ef_search = ef_search
    index._mL = 1.0 / math.log(float(M))
    index._random_seed = random_seed
    index._keep_pruned = keep_pruned
    index._data = database
    index._ids = ids
    index._ntotal = int(manifest["ntotal"])
    index._n_active = int(manifest.get("n_active", index._ntotal - len(deleted_nodes)))
    index._graphs = graphs
    index._node_levels = [int(lvl) for lvl in node_levels]
    index._enter_node = entry_point
    index._enter_level = max_level
    index._deleted_nodes = set(int(x) for x in deleted_nodes)
    index._id_to_node = {
        int(ids[i]): i for i in range(len(ids)) if i not in index._deleted_nodes
    }
    index._rng = np.random.default_rng(random_seed)

    return index


# -----------------------------------------------------------------------------
# Public Load API
# -----------------------------------------------------------------------------

def load_index(
    directory: str | Path,
    verify_checksums: bool = True,
) -> Any:
    """Load an Indexora index from disk, verifying manifest and data integrity.

    Parameters
    ----------
    directory : str or Path
        Directory containing the saved index manifest and arrays.
    verify_checksums : bool, default=True
        Whether to verify SHA-256 checksums of all array files against the manifest.

    Returns
    -------
    Any
        The reconstructed index instance.

    Raises
    ------
    FileNotFoundError
        If directory or manifest.json does not exist.
    ValueError
        If manifest is invalid, corrupt, or checksum verification fails.
    TypeError
        If index_type in manifest is unsupported.
    """
    path = Path(directory).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Index directory '{path}' does not exist.")
    if not path.is_dir():
        raise ValueError(f"Index path '{path}' is not a directory.")

    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Index manifest '{manifest_path}' does not exist.")

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except json.JSONDecodeError as err:
        raise ValueError(f"Corrupt manifest.json in {path}: {err}") from err

    if manifest.get("format") != FORMAT_IDENTIFIER:
        raise ValueError(
            f"Invalid index format: expected '{FORMAT_IDENTIFIER}', got '{manifest.get('format')}'"
        )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema version: expected '{SCHEMA_VERSION}', got '{manifest.get('schema_version')}'"
        )

    # Path traversal protection across all declared array filenames
    path_resolved = path.resolve()
    for fname in manifest.get("arrays", {}):
        if not (path / fname).resolve().is_relative_to(path_resolved):
            raise ValueError(f"Path traversal detected in manifest filename: '{fname}'")

    index_type = manifest.get("index_type")
    if index_type == "ExactFlatIndex":
        return _deserialize_exact_flat(path, manifest, verify_checksums)
    elif index_type == "ExactBlasIndex":
        return _deserialize_exact_blas(path, manifest, verify_checksums)
    elif index_type == "IVFFlatIndex":
        return _deserialize_ivf_flat(path, manifest, verify_checksums)
    elif index_type == "IVFPQIndex":
        return _deserialize_ivf_pq(path, manifest, verify_checksums)
    elif index_type == "HNSWIndex":
        return _deserialize_hnsw(path, manifest, verify_checksums)
    else:
        raise TypeError(f"Unknown or unsupported index_type: '{index_type}'")
