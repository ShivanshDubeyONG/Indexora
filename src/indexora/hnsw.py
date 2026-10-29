"""Hierarchical Navigable Small World (HNSW) vector search index for Indexora.

This module provides the Phase 10 custom HNSW graph index built from scratch in Python
and NumPy:
- Hierarchical multi-layer proximity graphs.
- Random exponential layer assignment.
- Bounded best-first greedy beam search (`search_layer`).
- Heuristic diverse neighbor selection (Algorithm 4 in Malkov & Yashunin 2018).
- Deterministic ascending `(distance, external_id)` tie-breaking.
- Immutable `SearchResult` return containers with mask and distance-computation tracking.
"""

from __future__ import annotations

import heapq
import math
from typing import Sequence
import numpy as np

from indexora.topk import SearchResult


def _validate_hnsw_params(
    d: int,
    M: int,
    ef_construction: int,
    ef_search: int,
) -> tuple[int, int, int, int]:
    """Validate HNSW graph structural hyperparameters."""
    if not isinstance(d, (int, np.integer)) or d <= 0:
        raise ValueError(f"Vector dimension d must be a positive integer, got {d}")
    if not isinstance(M, (int, np.integer)) or M < 2:
        raise ValueError(f"Connectivity M must be an integer >= 2, got {M}")
    if not isinstance(ef_construction, (int, np.integer)) or ef_construction < 1:
        raise ValueError(f"ef_construction must be an integer >= 1, got {ef_construction}")
    if not isinstance(ef_search, (int, np.integer)) or ef_search < 1:
        raise ValueError(f"ef_search must be an integer >= 1, got {ef_search}")
    return int(d), int(M), int(ef_construction), int(ef_search)


def _validate_database_vectors(
    database: np.ndarray,
    ids: np.ndarray | None,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Validate database vectors and IDs array."""
    if not isinstance(database, np.ndarray):
        raise TypeError(f"database must be a numpy ndarray, got {type(database).__name__}")
    if database.dtype != np.float32:
        raise TypeError(
            f"database must have strict dtype float32, got {database.dtype}. Silent casting is not permitted."
        )
    if database.ndim != 2:
        raise ValueError(f"database must be a 2D array of shape (N, d), got ndim={database.ndim}")
    if not database.flags.c_contiguous:
        raise ValueError("database must be C-contiguous in memory.")
    if not np.all(np.isfinite(database)):
        raise ValueError("database must contain only finite numbers (no NaN or Inf).")

    N, d = database.shape

    if ids is not None:
        if not isinstance(ids, np.ndarray):
            raise TypeError(f"ids must be a numpy ndarray, got {type(ids).__name__}")
        if ids.dtype != np.int64:
            raise TypeError(
                f"ids must have strict dtype int64, got {ids.dtype}. Silent casting is not permitted."
            )
        if ids.ndim != 1 or ids.shape[0] != N:
            raise ValueError(f"ids must be a 1D array of shape ({N},), got shape {ids.shape}")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("ids must be strictly unique among live database vectors.")
        valid_ids = np.ascontiguousarray(ids, dtype=np.int64)
    else:
        valid_ids = np.arange(N, dtype=np.int64)

    return N, d, database, valid_ids


def _squared_l2_dist(u: np.ndarray, v: np.ndarray) -> float:
    """Compute exact squared Euclidean distance between two 1D float32 vectors."""
    diff = u - v
    return float(np.dot(diff, diff))


class HNSWIndex:
    """Hierarchical Navigable Small World (HNSW) vector search index.

    Constructs a multi-layer proximity graph where upper layers feature long-range
    navigable links for fast logarithmic routing, and lower layers provide dense
    local connectivity for accurate approximate nearest neighbor search.

    Parameters
    ----------
    database : np.ndarray
        Database vectors of shape `(N, d)` with strict `dtype=np.float32`. Must be C-contiguous.
    M : int, default=16
        Maximum number of bidirectional connections per node at layers > 0.
        At layer 0, the maximum degree is `M0 = 2 * M`.
    ef_construction : int, default=64
        Size of dynamic candidate list during graph construction.
    ef_search : int, default=32
        Default size of dynamic candidate list during search.
    ids : np.ndarray | None, default=None
        Durable identifiers of shape `(N,)` with strict `dtype=np.int64`. Must be unique.
        If None, defaults to `np.arange(N, dtype=np.int64)`. Supports negative IDs and `-1`.
    random_seed : int | None, default=42
        Seed for reproducible random layer sampling.
    keep_pruned_connections : bool, default=True
        Whether neighbor selection heuristic adds discarded candidates back if degree < M_max.
    """

    def __init__(
        self,
        database: np.ndarray,
        M: int = 16,
        ef_construction: int = 64,
        ef_search: int = 32,
        ids: np.ndarray | None = None,
        random_seed: int | None = 42,
        keep_pruned_connections: bool = True,
    ) -> None:
        N, d, valid_db, valid_ids = _validate_database_vectors(database, ids)
        self._d, self._M, self._ef_construction, self._ef_search = _validate_hnsw_params(
            d, M, ef_construction, ef_search
        )
        self._M0 = 2 * self._M
        self._mL = 1.0 / math.log(float(self._M))
        self._random_seed = random_seed
        self._keep_pruned = bool(keep_pruned_connections)

        self._data = valid_db
        self._ids = valid_ids
        self._ntotal = N

        # Graph representations:
        # self._graphs[level] is a list of lists of neighbor node indices
        # self._graphs[level][node_idx] -> list[int]
        self._graphs: list[list[list[int]]] = []
        self._node_levels: list[int] = []
        self._enter_node: int | None = None
        self._enter_level: int = -1

        self._rng = np.random.default_rng(self._random_seed)

        # Build graph incrementally
        if N > 0:
            self._build_index()

    # ------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------

    @property
    def d(self) -> int:
        """Vector dimensionality."""
        return self._d

    @property
    def ntotal(self) -> int:
        """Total number of indexed vectors."""
        return self._ntotal

    @property
    def M(self) -> int:
        """Maximum neighbor count per node at layer > 0."""
        return self._M

    @property
    def M0(self) -> int:
        """Maximum neighbor count per node at layer 0 (`2 * M`)."""
        return self._M0

    @property
    def ef_construction(self) -> int:
        """Dynamic exploration depth during index construction."""
        return self._ef_construction

    @property
    def ef_search(self) -> int:
        """Default dynamic exploration depth during search."""
        return self._ef_search

    @ef_search.setter
    def ef_search(self, value: int) -> None:
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"ef_search must be an integer >= 1, got {value}")
        self._ef_search = int(value)

    @property
    def max_level(self) -> int:
        """Current maximum level in the graph hierarchy (-1 if empty)."""
        return self._enter_level

    @property
    def entry_point(self) -> int | None:
        """Internal node index of the top-layer entry point."""
        return self._enter_node

    @property
    def database(self) -> np.ndarray:
        """Indexed database vectors array of shape `(N, d)`."""
        return self._data

    @property
    def ids(self) -> np.ndarray:
        """Durable external integer IDs array of shape `(N,)`."""
        return self._ids

    def node_level(self, node_idx: int) -> int:
        """Return the maximum level assigned to node `node_idx`."""
        if not (0 <= node_idx < self._ntotal):
            raise IndexError(f"node_idx {node_idx} out of range [0, {self._ntotal})")
        return self._node_levels[node_idx]

    def get_neighbors(self, node_idx: int, level: int) -> tuple[int, ...]:
        """Return the neighbor node indices for `node_idx` at `level`."""
        if not (0 <= level <= self._enter_level):
            raise IndexError(f"level {level} out of range [0, {self._enter_level}]")
        if node_idx >= len(self._graphs[level]):
            return ()
        return tuple(self._graphs[level][node_idx])

    # ------------------------------------------------------------------------
    # Memory Accounting
    # ------------------------------------------------------------------------

    @property
    def vector_memory_bytes(self) -> int:
        """Memory in bytes of stored float32 database vectors."""
        return int(self._data.nbytes)

    @property
    def id_memory_bytes(self) -> int:
        """Memory in bytes of stored int64 external IDs."""
        return int(self._ids.nbytes)

    @property
    def graph_edge_count(self) -> int:
        """Total directed edge count across all levels."""
        total = 0
        for lvl_graph in self._graphs:
            for adj in lvl_graph:
                total += len(adj)
        return total

    @property
    def graph_memory_bytes(self) -> int:
        """Logical storage memory in bytes of all graph edges (assuming 64-bit node indices)."""
        return int(self.graph_edge_count * 8)

    @property
    def index_memory_bytes(self) -> int:
        """Total logical index memory in bytes (vectors + IDs + graph edges)."""
        return self.vector_memory_bytes + self.id_memory_bytes + self.graph_memory_bytes

    @property
    def index_memory_mib(self) -> float:
        """Total logical index memory in mebibytes (MiB)."""
        return float(self.index_memory_bytes / (1024.0 * 1024.0))

    @property
    def bytes_per_vector(self) -> float:
        """Average logical storage bytes per indexed vector."""
        if self._ntotal == 0:
            return 0.0
        return float(self.index_memory_bytes / self._ntotal)

    # ------------------------------------------------------------------------
    # Construction Algorithms
    # ------------------------------------------------------------------------

    def _sample_level(self) -> int:
        """Sample maximum graph level using the exponential decay distribution."""
        unif = float(self._rng.uniform(0.0, 1.0))
        # Clamp to avoid math.log(0)
        unif = max(unif, 1e-15)
        return int(math.floor(-math.log(unif) * self._mL))

    def _build_index(self) -> None:
        """Construct the HNSW index by inserting all database vectors sequentially."""
        for i in range(self._ntotal):
            self._insert_vector(i)

    def _ensure_graph_levels(self, max_lvl: int) -> None:
        """Ensure adjacency list structures exist up to level `max_lvl`."""
        while len(self._graphs) <= max_lvl:
            self._graphs.append([])

    def _insert_vector(self, node_idx: int) -> None:
        """Insert a single node into the HNSW graph hierarchy (Algorithm 1)."""
        q_vec = self._data[node_idx]
        l_node = self._sample_level()
        self._node_levels.append(l_node)

        self._ensure_graph_levels(l_node)
        for lvl in range(l_node + 1):
            while len(self._graphs[lvl]) <= node_idx:
                self._graphs[lvl].append([])

        # First node inserted becomes the entry point
        if self._enter_node is None:
            self._enter_node = node_idx
            self._enter_level = l_node
            return

        curr_obj = self._enter_node
        curr_dist = _squared_l2_dist(q_vec, self._data[curr_obj])

        # 1. Greedy routing down to layer l_node + 1 (using ef=1)
        for lvl in range(self._enter_level, l_node, -1):
            changed = True
            while changed:
                changed = False
                for neighbor in self._graphs[lvl][curr_obj]:
                    d_nbr = _squared_l2_dist(q_vec, self._data[neighbor])
                    if d_nbr < curr_dist:
                        curr_dist = d_nbr
                        curr_obj = neighbor
                        changed = True

        # 2. Multilayer insertion from min(enter_level, l_node) down to layer 0
        top_insert_lvl = min(self._enter_level, l_node)
        for lvl in range(top_insert_lvl, -1, -1):
            m_max = self._M0 if lvl == 0 else self._M

            # Search layer to find candidates pool W (Algorithm 2)
            candidates_w, _ = self._search_layer_internal(
                query=q_vec,
                enter_points=[curr_obj],
                ef=self._ef_construction,
                level=lvl,
            )

            # Heuristic neighbor selection (Algorithm 4)
            chosen_neighbors = self._select_neighbors_heuristic(
                query=q_vec,
                candidates=candidates_w,
                m_max=m_max,
            )

            # Add bidirectional connections
            self._graphs[lvl][node_idx] = chosen_neighbors
            for nbr in chosen_neighbors:
                self._graphs[lvl][nbr].append(node_idx)
                # Prune if degree exceeds m_max
                if len(self._graphs[lvl][nbr]) > m_max:
                    nbr_vec = self._data[nbr]
                    nbr_cands = [
                        (nbr_nbr, _squared_l2_dist(nbr_vec, self._data[nbr_nbr]))
                        for nbr_nbr in self._graphs[lvl][nbr]
                    ]
                    self._graphs[lvl][nbr] = self._select_neighbors_heuristic(
                        query=nbr_vec,
                        candidates=nbr_cands,
                        m_max=m_max,
                    )

            # Next layer entry point is the closest candidate in W
            if candidates_w:
                closest_cand = min(candidates_w, key=lambda c: (c[1], self._ids[c[0]]))
                curr_obj = closest_cand[0]
                curr_dist = closest_cand[1]

        # Update global entry point if new node exceeded entry level
        if l_node > self._enter_level:
            self._enter_node = node_idx
            self._enter_level = l_node

    def _select_neighbors_heuristic(
        self,
        query: np.ndarray,
        candidates: list[tuple[int, float]],
        m_max: int,
    ) -> list[int]:
        """Select diverse nearest neighbors using Malkov & Yashunin's Algorithm 4 heuristic.

        Parameters
        ----------
        query : np.ndarray
            Base point coordinates.
        candidates : list[tuple[int, float]]
            List of `(node_idx, distance_to_query)` candidate pairs.
        m_max : int
            Maximum number of neighbors to select.

        Returns
        -------
        list[int]
            Selected neighbor node indices of length <= m_max.
        """
        if len(candidates) <= m_max:
            return [c[0] for c in candidates]

        # Sort candidates ascending by (distance, external_id) for deterministic ordering
        sorted_candidates = sorted(candidates, key=lambda c: (c[1], self._ids[c[0]]))

        result_nodes: list[int] = []
        result_vecs: list[np.ndarray] = []
        pruned_nodes: list[int] = []

        for cand_idx, d_q_cand in sorted_candidates:
            cand_vec = self._data[cand_idx]
            # Shrinkage check: cand must be closer to query than to any already-chosen neighbor
            is_diverse = True
            for r_vec in result_vecs:
                if _squared_l2_dist(cand_vec, r_vec) < d_q_cand:
                    is_diverse = False
                    break

            if is_diverse:
                result_nodes.append(cand_idx)
                result_vecs.append(cand_vec)
                if len(result_nodes) == m_max:
                    break
            else:
                pruned_nodes.append(cand_idx)

        # Fallback: if heuristic resulted in fewer than m_max connections, add best pruned
        if self._keep_pruned and len(result_nodes) < m_max:
            for p_idx in pruned_nodes:
                result_nodes.append(p_idx)
                if len(result_nodes) == m_max:
                    break

        return result_nodes

    def _search_layer_internal(
        self,
        query: np.ndarray,
        enter_points: list[int],
        ef: int,
        level: int,
    ) -> tuple[list[tuple[int, float]], int]:
        """Perform bounded best-first candidate exploration at a specific layer (Algorithm 2).

        Returns
        -------
        candidates : list[tuple[int, float]]
            List of `(node_idx, distance)` pairs in dynamic candidate pool.
        dist_comps : int
            Number of distance computations executed.
        """
        visited: set[int] = set()
        dist_comps = 0

        # C (candidates min-heap): (dist, ext_id, node_idx)
        # W (dynamic pool max-heap): (-dist, -ext_id, node_idx)
        c_min_heap: list[tuple[float, int, int]] = []
        w_max_heap: list[tuple[float, int, int]] = []

        for ep in enter_points:
            visited.add(ep)
            d = _squared_l2_dist(query, self._data[ep])
            dist_comps += 1
            ext_id = int(self._ids[ep])
            heapq.heappush(c_min_heap, (d, ext_id, ep))
            heapq.heappush(w_max_heap, (-d, -ext_id, ep))

        furthest_w_dist = -w_max_heap[0][0]

        while c_min_heap:
            closest_dist, _, closest_node = heapq.heappop(c_min_heap)
            if closest_dist > furthest_w_dist:
                break

            for neighbor in self._graphs[level][closest_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    d_nbr = _squared_l2_dist(query, self._data[neighbor])
                    dist_comps += 1
                    ext_id_nbr = int(self._ids[neighbor])

                    if d_nbr < furthest_w_dist or len(w_max_heap) < ef:
                        heapq.heappush(c_min_heap, (d_nbr, ext_id_nbr, neighbor))
                        heapq.heappush(w_max_heap, (-d_nbr, -ext_id_nbr, neighbor))

                        if len(w_max_heap) > ef:
                            heapq.heappop(w_max_heap)

                        furthest_w_dist = -w_max_heap[0][0]

        # Extract elements from W pool
        pool = [(node_idx, -neg_dist) for neg_dist, _, node_idx in w_max_heap]
        return pool, dist_comps

    # ------------------------------------------------------------------------
    # Search Execution
    # ------------------------------------------------------------------------

    def _search_single(
        self,
        query: np.ndarray,
        k: int,
        eff_ef: int,
    ) -> SearchResult:
        """Search nearest neighbors for a single query vector of shape `(d,)`."""
        if self._ntotal == 0 or self._enter_node is None:
            return SearchResult(
                distances=np.empty((0,), dtype=np.float32),
                ids=np.empty((0,), dtype=np.int64),
                mask=np.empty((0,), dtype=bool),
                distance_computations=0,
                centroid_distance_computations=0,
                candidate_distance_computations=0,
            )

        total_dc = 0
        curr_obj = self._enter_node
        curr_dist = _squared_l2_dist(query, self._data[curr_obj])
        total_dc += 1

        # 1. Greedy routing down to layer 1
        for lvl in range(self._enter_level, 0, -1):
            changed = True
            while changed:
                changed = False
                for neighbor in self._graphs[lvl][curr_obj]:
                    d_nbr = _squared_l2_dist(query, self._data[neighbor])
                    total_dc += 1
                    if d_nbr < curr_dist:
                        curr_dist = d_nbr
                        curr_obj = neighbor
                        changed = True

        # 2. Best-first search at layer 0 with candidate pool ef
        pool, layer0_dc = self._search_layer_internal(
            query=query,
            enter_points=[curr_obj],
            ef=eff_ef,
            level=0,
        )
        total_dc += layer0_dc

        # 3. Deterministic top-k selection by ascending (distance, external_id)
        # pool is list of (node_idx, distance)
        sorted_pool = sorted(pool, key=lambda c: (c[1], self._ids[c[0]]))
        k_eff = min(k, len(sorted_pool))
        top_k = sorted_pool[:k_eff]

        res_dists = np.array([c[1] for c in top_k], dtype=np.float32)
        res_ids = np.array([self._ids[c[0]] for c in top_k], dtype=np.int64)
        res_mask = np.ones(k_eff, dtype=bool)

        return SearchResult(
            distances=res_dists,
            ids=res_ids,
            mask=res_mask,
            distance_computations=total_dc,
            centroid_distance_computations=0,
            candidate_distance_computations=total_dc,
        )

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
        ef_search: int | None = None,
    ) -> SearchResult:
        """Search approximate nearest neighbors using HNSW graph traversal.

        Parameters
        ----------
        queries : np.ndarray
            Query vector(s) of shape `(d,)` or `(Q, d)` with strict `dtype=np.float32`.
        k : int, default=10
            Number of nearest neighbors to retrieve per query (`k >= 1`).
        ef_search : int | None, default=None
            Dynamic exploration depth during search. If None, uses `self.ef_search`.

        Returns
        -------
        SearchResult
            SearchResult container conforming to Indexora's standard API.
        """
        # Validate k
        if not isinstance(k, (int, np.integer)):
            raise TypeError(f"k must be an integer, got {type(k).__name__}")
        if k <= 0:
            raise ValueError("k must be positive")

        # Validate ef_search
        if ef_search is None:
            eff_ef = max(k, self._ef_search)
        else:
            if not isinstance(ef_search, (int, np.integer)):
                raise TypeError(f"ef_search must be an integer, got {type(ef_search).__name__}")
            if ef_search < 1:
                raise ValueError(f"ef_search must be at least 1, got {ef_search}")
            eff_ef = max(k, int(ef_search))

        # Validate queries
        if not isinstance(queries, np.ndarray):
            raise TypeError(f"queries must be a numpy ndarray, got {type(queries).__name__}")
        if queries.dtype != np.float32:
            raise TypeError(
                f"queries must have strict dtype float32, got {queries.dtype}. Silent casting is not permitted."
            )
        if not np.all(np.isfinite(queries)):
            raise ValueError("queries must contain only finite numbers (no NaN or Inf).")

        is_single_query = queries.ndim == 1
        if is_single_query:
            if queries.shape[0] != self._d:
                raise ValueError(
                    f"Query feature dimension {queries.shape[0]} does not match index dimension {self._d}"
                )
            if not queries.flags.c_contiguous:
                raise ValueError("queries must be C-contiguous in memory.")
            return self._search_single(queries, k=k, eff_ef=eff_ef)

        if queries.ndim != 2:
            raise ValueError(f"queries must be 1D or 2D array, got ndim={queries.ndim}")
        if queries.shape[1] != self._d:
            raise ValueError(
                f"Query feature dimension {queries.shape[1]} does not match index dimension {self._d}"
            )
        if not queries.flags.c_contiguous:
            raise ValueError("queries must be C-contiguous in memory.")

        num_queries = queries.shape[0]
        if num_queries == 0:
            return SearchResult(
                distances=np.empty((0, min(k, self._ntotal)), dtype=np.float32),
                ids=np.empty((0, min(k, self._ntotal)), dtype=np.int64),
                mask=np.empty((0, min(k, self._ntotal)), dtype=bool),
                distance_computations=0,
                centroid_distance_computations=0,
                candidate_distance_computations=0,
            )

        per_query_results = [
            self._search_single(queries[i], k=k, eff_ef=eff_ef)
            for i in range(num_queries)
        ]

        total_dc = sum(
            r.distance_computations for r in per_query_results if r.distance_computations is not None
        )

        lengths = [len(r.ids) for r in per_query_results]
        if len(set(lengths)) == 1:
            batch_distances = np.stack([r.distances for r in per_query_results], axis=0)
            batch_ids = np.stack([r.ids for r in per_query_results], axis=0)
            batch_mask = np.stack([r.mask for r in per_query_results], axis=0)
        else:
            target_k = max(lengths) if lengths else 0
            batch_distances = np.full((num_queries, target_k), np.float32(np.inf), dtype=np.float32)
            batch_ids = np.full((num_queries, target_k), np.int64(-1), dtype=np.int64)
            batch_mask = np.zeros((num_queries, target_k), dtype=bool)
            for i, r in enumerate(per_query_results):
                L = len(r.ids)
                batch_distances[i, :L] = r.distances
                batch_ids[i, :L] = r.ids
                batch_mask[i, :L] = r.mask

        return SearchResult(
            distances=batch_distances,
            ids=batch_ids,
            mask=batch_mask,
            distance_computations=total_dc,
            centroid_distance_computations=0,
            candidate_distance_computations=total_dc,
        )


def hnsw_build(
    database: np.ndarray,
    M: int = 16,
    ef_construction: int = 64,
    ef_search: int = 32,
    ids: np.ndarray | None = None,
    random_seed: int | None = 42,
    keep_pruned_connections: bool = True,
) -> HNSWIndex:
    """Construct an HNSWIndex from database vectors."""
    return HNSWIndex(
        database=database,
        M=M,
        ef_construction=ef_construction,
        ef_search=ef_search,
        ids=ids,
        random_seed=random_seed,
        keep_pruned_connections=keep_pruned_connections,
    )


def hnsw_search(
    index: HNSWIndex,
    queries: np.ndarray,
    k: int = 10,
    ef_search: int | None = None,
) -> SearchResult:
    """Search an HNSWIndex using query vectors."""
    if not isinstance(index, HNSWIndex):
        raise TypeError(f"index must be an HNSWIndex instance, got {type(index).__name__}")
    return index.search(queries=queries, k=k, ef_search=ef_search)
