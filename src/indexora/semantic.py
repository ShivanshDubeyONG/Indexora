"""Semantic text-search integration and pipeline for Indexora (Phase 12B).

Connects pretrained sentence-embedding models with Indexora's native vector indexes
(Exact BLAS, IVF-Flat, and HNSW) to perform end-to-end semantic retrieval over
structured text documents.

Key Principles:
- Core Indexora package remains lightweight: sentence-transformers is an optional dependency.
- Text embedding is strictly decoupled from vector indexing and top-k retrieval.
- Embeddings are L2-normalized so that squared Euclidean distance ranking:
      ||q - x||² = ||q||² + ||x||² - 2 (q · x) = 2 - 2 cos(q, x)
  corresponds identically to descending cosine similarity ordering, while preserving
  Indexora's squared-L2 distance metric contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import numpy as np

from indexora.data import normalize_l2
from indexora.exact import ExactBlasIndex, ExactFlatIndex
from indexora.hnsw import HNSWIndex
from indexora.ivf import IVFFlatIndex
from indexora.topk import SearchResult


@dataclass(frozen=True)
class Document:
    """A structured text document with durable entity ID and metadata.

    Parameters
    ----------
    doc_id : int
        Durable integer ID (must be unique within the corpus).
    title : str
        Human-readable title of the document.
    text : str
        Document text content used for embedding generation.
    category : str
        Topic category (e.g. 'Machine Learning', 'Databases', 'Security').
    metadata : dict[str, Any], default=empty dict
        Additional metadata (author, tags, keywords).
    """

    doc_id: int
    title: str
    text: str
    category: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticSearchResult:
    """Result of a semantic search query against an Indexora index.

    Parameters
    ----------
    query : str
        Raw input query string.
    retrieved_documents : list[Document]
        List of documents matching the top-k nearest neighbor IDs.
    distances : np.ndarray
        Squared Euclidean distances (float32) from query embedding to document vectors.
    is_exact_match : list[bool]
        Whether each retrieved document matches the true exact BLAS top-k ground truth.
    latency_ms : float
        Query search latency in milliseconds (excluding text embedding generation).
    embed_time_ms : float
        Time in milliseconds spent generating the query embedding.
    distance_computations : int | None
        Number of vector distance computations performed during index search.
    """

    query: str
    retrieved_documents: list[Document]
    distances: np.ndarray
    is_exact_match: list[bool]
    latency_ms: float
    embed_time_ms: float
    distance_computations: int | None = None

    @property
    def cosine_similarities(self) -> np.ndarray:
        """Convert squared L2 distances to cosine similarities for unit-normalized vectors:
        cos(q, x) = 1 - 0.5 * ||q - x||^2.
        """
        sims = 1.0 - 0.5 * self.distances
        return np.clip(sims, -1.0, 1.0)

    @property
    def vector_oracle_agreement(self) -> float:
        """Fraction of retrieved candidates matching the exact brute-force vector search oracle.

        Note: Measures vector index recall against an exhaustive scan in embedding space.
        Does not independently evaluate human or semantic relevance.
        """
        if not self.is_exact_match:
            return 0.0
        return float(sum(self.is_exact_match) / len(self.is_exact_match))


# -----------------------------------------------------------------------------
# Demonstration Technical Corpus
# -----------------------------------------------------------------------------

DEMO_CORPUS: list[Document] = [
    # Category: Machine Learning & Deep Learning
    Document(
        doc_id=101,
        title="Transformer Architecture and Self-Attention",
        text=(
            "The Transformer architecture replaces recurrent neural networks with multi-head self-attention mechanisms, "
            "allowing sequence tokens to attend to all other positions in parallel and dramatically scaling large language models."
        ),
        category="Machine Learning",
        metadata={"tags": ["transformers", "attention", "nlp", "deep learning"]},
    ),
    Document(
        doc_id=102,
        title="Dense Vector Embeddings and Representation Learning",
        text=(
            "Dense vector embeddings map complex categorical, text, or graph entities into a continuous metric space "
            "where geometric proximity (such as inner product or Euclidean distance) captures semantic similarity."
        ),
        category="Machine Learning",
        metadata={"tags": ["embeddings", "vector space", "representation", "metric"]},
    ),
    Document(
        doc_id=103,
        title="Stochastic Gradient Descent and Optimization",
        text=(
            "Gradient descent optimizes neural network parameters by iteratively computing the gradient of the loss "
            "function with respect to model weights using backpropagation and adaptive learning rates."
        ),
        category="Machine Learning",
        metadata={"tags": ["optimization", "gradient descent", "backprop", "training"]},
    ),
    Document(
        doc_id=104,
        title="Contrastive Learning and Triplet Loss",
        text=(
            "Contrastive learning trains encoders by pulling positive pairs closer together in embedding space while "
            "pushing negative pairs apart using objective functions like InfoNCE or triplet margin loss."
        ),
        category="Machine Learning",
        metadata={"tags": ["contrastive", "triplet loss", "metric learning", "encoders"]},
    ),
    Document(
        doc_id=105,
        title="Convolutional Neural Networks for Computer Vision",
        text=(
            "Convolutional neural networks use localized spatial filters with parameter sharing to extract shift-invariant "
            "visual features from images across hierarchical abstraction layers."
        ),
        category="Machine Learning",
        metadata={"tags": ["cnn", "vision", "convolution", "feature extraction"]},
    ),

    # Category: Databases & Storage Engines
    Document(
        doc_id=201,
        title="Log-Structured Merge-Trees vs. B-Trees",
        text=(
            "Log-Structured Merge-trees (LSM-trees) optimize write throughput by appending sequentially to a memtable "
            "and flushing sorted immutable SSTables to disk, whereas B-trees provide predictable in-place page reads."
        ),
        category="Databases",
        metadata={"tags": ["lsm tree", "b tree", "storage engine", "writes"]},
    ),
    Document(
        doc_id=202,
        title="Vector Indexing with Hierarchical Navigable Small Worlds",
        text=(
            "HNSW builds a multi-layer graph of vectors where upper layers act as an exponential skip-list for greedy "
            "logarithmic routing, and layer zero conducts bounded beam exploration for high-recall nearest-neighbor search."
        ),
        category="Databases",
        metadata={"tags": ["hnsw", "vector search", "ann", "graph index"]},
    ),
    Document(
        doc_id=203,
        title="Columnar Storage and Parquet Encoding",
        text=(
            "Columnar formats such as Apache Parquet store data by attribute rather than row, enabling high dictionary "
            "compression, run-length encoding, and vectorized execution of analytical OLAP queries."
        ),
        category="Databases",
        metadata={"tags": ["columnar", "parquet", "olap", "compression"]},
    ),
    Document(
        doc_id=204,
        title="ACID Transaction Isolation and Write-Ahead Logging",
        text=(
            "Database management systems ensure atomicity, consistency, isolation, and durability through write-ahead logs (WAL) "
            "and multiversion concurrency control (MVCC) to prevent dirty reads and lost updates."
        ),
        category="Databases",
        metadata={"tags": ["acid", "transactions", "wal", "concurrency", "mvcc"]},
    ),
    Document(
        doc_id=205,
        title="Inverted Indexes and BM25 Lexical Ranking",
        text=(
            "Inverted indexes map lexicon tokens to posting lists containing document IDs and term frequencies, "
            "supporting sparse keyword search ranked by tf-idf scoring or Okapi BM25 algorithms."
        ),
        category="Databases",
        metadata={"tags": ["inverted index", "lexical search", "bm25", "information retrieval"]},
    ),

    # Category: Distributed Systems & Cloud
    Document(
        doc_id=301,
        title="Raft Consensus Algorithm and Leader Election",
        text=(
            "Raft manages replicated state machines across distributed clusters using randomized election timers, "
            "strong leader invariants, and replicated log entry consistency checks."
        ),
        category="Distributed Systems",
        metadata={"tags": ["raft", "consensus", "leader election", "replication"]},
    ),
    Document(
        doc_id=302,
        title="CAP Theorem and Eventual Consistency",
        text=(
            "Brewer's CAP theorem establishes that a distributed data store can guarantee at most two of Consistency, "
            "Availability, and Partition tolerance, driving modern NoSQL designs toward tunable eventual consistency."
        ),
        category="Distributed Systems",
        metadata={"tags": ["cap theorem", "consistency", "partitions", "nosql"]},
    ),
    Document(
        doc_id=303,
        title="MapReduce and Large-Scale Batch Processing",
        text=(
            "MapReduce partitions computation into parallel map steps that process input chunks followed by "
            "shuffle-sort phases and reduce steps that aggregate intermediate keys across compute nodes."
        ),
        category="Distributed Systems",
        metadata={"tags": ["mapreduce", "batch processing", "distributed compute", "hadoop"]},
    ),
    Document(
        doc_id=304,
        title="Distributed Tracing and OpenTelemetry",
        text=(
            "Distributed tracing tracks asynchronous requests through microservice boundaries by propagating trace IDs "
            "and span contexts, revealing bottleneck latency and error propagation."
        ),
        category="Distributed Systems",
        metadata={"tags": ["tracing", "opentelemetry", "observability", "microservices"]},
    ),
    Document(
        doc_id=305,
        title="Consistent Hashing in Distributed Caching",
        text=(
            "Consistent hashing maps both keys and cluster cache servers to a circular hash ring, ensuring that adding "
            "or removing a node only requires remapping a minimal fraction (K/N) of cached keys."
        ),
        category="Distributed Systems",
        metadata={"tags": ["consistent hashing", "caching", "sharding", "hash ring"]},
    ),

    # Category: Cybersecurity & Cryptography
    Document(
        doc_id=401,
        title="Asymmetric Public-Key Cryptography and RSA",
        text=(
            "Public-key cryptography uses mathematically linked key pairs—a public key for encryption and a private key "
            "for decryption—relying on the computational intractability of prime factorization or elliptic curve discrete logs."
        ),
        category="Cybersecurity",
        metadata={"tags": ["cryptography", "rsa", "public key", "encryption"]},
    ),
    Document(
        doc_id=402,
        title="Zero Trust Security Architecture",
        text=(
            "Zero Trust eliminates perimeter-based implicit trust, requiring continuous authentication, strict role-based "
            "access control (RBAC), and mutual TLS encryption for every network request regardless of origin."
        ),
        category="Cybersecurity",
        metadata={"tags": ["zero trust", "rbac", "security architecture", "authentication"]},
    ),
    Document(
        doc_id=403,
        title="SQL Injection Vulnerabilities and Prepared Statements",
        text=(
            "SQL injection occurs when untrusted user input is directly concatenated into database queries. Parameterized "
            "queries and prepared statements eliminate injection by strictly separating SQL syntax from user data."
        ),
        category="Cybersecurity",
        metadata={"tags": ["sql injection", "appsec", "prepared statements", "vulnerabilities"]},
    ),
    Document(
        doc_id=404,
        title="TLS Handshake and Transport Layer Security",
        text=(
            "The TLS handshake establishes secure end-to-end communication by validating server X.509 certificates, "
            "negotiating cipher suites, and deriving ephemeral symmetric session keys via Elliptic Curve Diffie-Hellman."
        ),
        category="Cybersecurity",
        metadata={"tags": ["tls", "ssl", "certificates", "diffie hellman", "handshake"]},
    ),
    Document(
        doc_id=405,
        title="OAuth 2.0 and OpenID Connect Token Authorization",
        text=(
            "OAuth 2.0 delegates user authorization using cryptographically signed JSON Web Tokens (JWT) and bearer scopes, "
            "while OpenID Connect (OIDC) adds an identity layer for federated single sign-on."
        ),
        category="Cybersecurity",
        metadata={"tags": ["oauth2", "oidc", "jwt", "authorization", "tokens"]},
    ),

    # Category: Software Engineering & Architecture
    Document(
        doc_id=501,
        title="Clean Architecture and Dependency Inversion",
        text=(
            "Clean Architecture enforces inward dependency rules where core business logic and domain entities remain "
            "completely isolated from frameworks, databases, and external I/O protocols."
        ),
        category="Software Engineering",
        metadata={"tags": ["clean architecture", "domain driven", "solid", "decoupling"]},
    ),
    Document(
        doc_id=502,
        title="Event-Driven Architecture and Pub/Sub Messaging",
        text=(
            "Event-driven architectures decouple producers from consumers using distributed message brokers (such as Kafka "
            "or RabbitMQ) to broadcast domain state changes as asynchronous event streams."
        ),
        category="Software Engineering",
        metadata={"tags": ["event driven", "pub sub", "kafka", "messaging", "queues"]},
    ),
    Document(
        doc_id=503,
        title="Microservices vs. Modular Monolith",
        text=(
            "While microservices offer independent deployment and technology diversity at the cost of distributed systems "
            "complexity, modular monoliths provide strong domain encapsulation with in-process function call performance."
        ),
        category="Software Engineering",
        metadata={"tags": ["microservices", "monolith", "software architecture", "scalability"]},
    ),
    Document(
        doc_id=504,
        title="Hexagonal Architecture and Ports and Adapters",
        text=(
            "Hexagonal architecture structures software into an application core surrounded by ports (interfaces) and "
            "adapters that translate outside inputs (HTTP controllers, message handlers) and outputs (repositories, clients)."
        ),
        category="Software Engineering",
        metadata={"tags": ["hexagonal", "ports adapters", "domain design", "architecture"]},
    ),
    Document(
        doc_id=505,
        title="Continuous Integration and Automated Testing Pipelines",
        text=(
            "Continuous Integration (CI) validates code changes automatically on push through unit tests, linting, "
            "static analysis, and hermetic build workflows to detect regressions rapidly before deployment."
        ),
        category="Software Engineering",
        metadata={"tags": ["ci cd", "testing", "devops", "automation", "quality"]},
    ),
]


def load_demo_corpus() -> list[Document]:
    """Return the curated technical demonstration document corpus."""
    return list(DEMO_CORPUS)


# -----------------------------------------------------------------------------
# Sentence-Transformers Wrapper with Graceful Degradation
# -----------------------------------------------------------------------------

DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
_LOADED_MODEL_CACHE: dict[str, Any] = {}


class SemanticSearchUnavailableError(RuntimeError):
    """Base exception raised when semantic search capabilities are unavailable."""

    pass


class SentenceTransformersNotAvailableError(SemanticSearchUnavailableError, ImportError):
    """Raised when sentence-transformers or its dependencies are not installed."""

    pass


class ModelInitializationError(SemanticSearchUnavailableError, RuntimeError):
    """Raised when pretrained embedding model weights fail to load or initialize."""

    pass


def is_sentence_transformers_available() -> bool:
    """Check whether sentence-transformers is installed in the current environment."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def check_semantic_availability(model_name: str = DEFAULT_EMBEDDING_MODEL) -> tuple[bool, str]:
    """Check whether semantic search can be run in the current environment.

    Returns
    -------
    tuple[bool, str]
        (is_available, diagnostic_message)
    """
    if not is_sentence_transformers_available():
        return False, (
            "Optional dependency 'sentence-transformers' is not installed. "
            "Install it with: pip install -e \".[semantic]\""
        )
    try:
        get_embedding_model(model_name)
        return True, "Pretrained embedding model is available."
    except Exception as exc:
        return False, f"Embedding model failed to initialize: {exc}"


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Any:
    """Load and cache a pretrained SentenceTransformer model on CPU.

    Raises
    ------
    SentenceTransformersNotAvailableError
        If sentence-transformers is not installed.
    ModelInitializationError
        If the model weights cannot be downloaded or loaded.
    """
    if not is_sentence_transformers_available():
        raise SentenceTransformersNotAvailableError(
            "sentence-transformers is not installed. "
            "Install optional semantic dependencies with: pip install -e \".[semantic]\""
        )

    if model_name in _LOADED_MODEL_CACHE:
        return _LOADED_MODEL_CACHE[model_name]

    from sentence_transformers import SentenceTransformer
    try:
        model = SentenceTransformer(model_name, device="cpu")
        _LOADED_MODEL_CACHE[model_name] = model
        return model
    except Exception as exc:
        raise ModelInitializationError(
            f"Failed to load sentence-transformers model '{model_name}': {exc}. "
            "A working internet connection is required on the initial run to download model weights."
        ) from exc


def mock_embed_texts(texts: Sequence[str] | str, dim: int = 64) -> np.ndarray:
    """Deterministic offline test-fixture vector generator for unit tests.

    WARNING: This helper hashes text into pseudo-random vectors. These vectors do
    NOT capture linguistic or semantic meaning and must NEVER be substituted in
    place of real neural embeddings for semantic search or relevance evaluation.
    """
    import hashlib

    is_single = isinstance(texts, str)
    items = [texts] if is_single else list(texts)
    out = np.zeros((len(items), dim), dtype=np.float32)
    for i, t in enumerate(items):
        digest = hashlib.sha256(t.encode("utf-8")).hexdigest()
        seed = int(digest[:8], 16)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(dim).astype(np.float32)
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        out[i] = v
    return out[0] if is_single else out


def encode_texts(
    texts: Sequence[str] | str,
    model: Any | None = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    normalize: bool = True,
) -> np.ndarray:
    """Generate dense float32 vector embeddings for input text strings.

    Parameters
    ----------
    texts : Sequence[str] | str
        Single text string or sequence of text strings to embed.
    model : Any | None, default=None
        Pre-loaded model instance or callable `texts -> np.ndarray`. If None, loads `model_name`.
    model_name : str, default='sentence-transformers/all-MiniLM-L6-v2'
        Model identifier to load if `model` is not supplied.
    normalize : bool, default=True
        Whether to L2-normalize vectors to unit length so that squared Euclidean distance
        directly corresponds to cosine similarity ordering.

    Returns
    -------
    np.ndarray
        Array of embeddings with shape `(N, d)` or `(d,)` and `dtype=np.float32`.
    """
    is_single = isinstance(texts, str)
    text_list = [texts] if is_single else list(texts)

    if len(text_list) == 0:
        return np.empty((0, 384), dtype=np.float32)

    if model is None:
        model = get_embedding_model(model_name=model_name)

    if callable(model) and not hasattr(model, "encode"):
        embeddings = model(text_list)
    else:
        # SentenceTransformer instance
        embeddings = model.encode(
            text_list,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )

    arr = np.ascontiguousarray(embeddings, dtype=np.float32)

    if normalize:
        arr = normalize_l2(arr, eps=1e-12)

    return arr[0] if is_single else arr


# -----------------------------------------------------------------------------
# Semantic Search Engine
# -----------------------------------------------------------------------------

class SemanticSearchEngine:
    """End-to-end semantic search engine backed by Indexora's native vector indexes.

    Integrates document text, sentence embeddings, Indexora vector indexing,
    and document metadata lookup into a unified retrieval pipeline.
    """

    def __init__(
        self,
        corpus: list[Document],
        embeddings: np.ndarray,
        index_type: str = "Exact BLAS",
        index_instance: Any | None = None,
        embedder: Any | None = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        ivf_nlist: int = 4,
        ivf_nprobe: int = 2,
        hnsw_m: int = 8,
        hnsw_ef_construction: int = 32,
        hnsw_ef_search: int = 32,
        build_time_ms: float = 0.0,
    ) -> None:
        self.corpus = list(corpus)
        self.doc_map: dict[int, Document] = {d.doc_id: d for d in corpus}
        self.doc_ids = np.array([d.doc_id for d in corpus], dtype=np.int64)
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.d = int(self.embeddings.shape[1])
        self.index_type = index_type
        self.embedder = embedder
        self.model_name = model_name
        self.build_time_ms = build_time_ms

        # Configuration parameters
        self.ivf_nlist = ivf_nlist
        self.ivf_nprobe = ivf_nprobe
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search

        # Build index if not provided
        if index_instance is not None:
            self.index = index_instance
        else:
            self.index, self.build_time_ms = self._build_index(index_type)

        # Baseline ground truth index
        self._exact_oracle = ExactBlasIndex(database=self.embeddings, ids=self.doc_ids)

    def _build_index(self, index_type: str) -> tuple[Any, float]:
        """Construct the chosen Indexora index on corpus embeddings, timing build duration."""
        t0 = time.perf_counter()
        if index_type == "Exact Flat (Ref)":
            idx = ExactFlatIndex(database=self.embeddings, ids=self.doc_ids)
        elif index_type == "Exact BLAS":
            idx = ExactBlasIndex(database=self.embeddings, ids=self.doc_ids)
        elif index_type == "IVF-Flat":
            idx = IVFFlatIndex(
                database=self.embeddings,
                ids=self.doc_ids,
                nlist=self.ivf_nlist,
                nprobe=self.ivf_nprobe,
            )
        elif index_type == "HNSW":
            idx = HNSWIndex(
                database=self.embeddings,
                ids=self.doc_ids,
                M=self.hnsw_m,
                ef_construction=self.hnsw_ef_construction,
                ef_search=self.hnsw_ef_search,
                random_seed=42,
            )
        else:
            idx = ExactBlasIndex(database=self.embeddings, ids=self.doc_ids)
        build_time_ms = (time.perf_counter() - t0) * 1000.0
        return idx, build_time_ms

    def search(
        self,
        query: str,
        k: int = 5,
        filter_category: str | None = None,
        filter_ids: Sequence[int] | set[int] | Callable[[int], bool] | None = None,
        nprobe: int | None = None,
        ef_search: int | None = None,
    ) -> SemanticSearchResult:
        """Execute semantic text retrieval for a user query.

        Parameters
        ----------
        query : str
            User search query string.
        k : int, default=5
            Number of nearest documents to retrieve.
        filter_category : str | None, default=None
            Optional category restriction (e.g. 'Databases').
        filter_ids : Sequence[int] | set[int] | Callable[[int], bool] | None, default=None
            Explicit external ID eligibility filter.
        nprobe : int | None, default=None
            Query-time probed centroids for IVF-Flat.
        ef_search : int | None, default=None
            Query-time exploration depth for HNSW.

        Returns
        -------
        SemanticSearchResult
            Structured search result with mapped documents, distances, and exact match status.
        """
        if not query or not query.strip():
            return SemanticSearchResult(
                query=query,
                retrieved_documents=[],
                distances=np.empty((0,), dtype=np.float32),
                is_exact_match=[],
                latency_ms=0.0,
                embed_time_ms=0.0,
                distance_computations=0,
            )

        # 1. Generate Query Vector Embedding
        t_embed_0 = time.perf_counter()
        query_vec = encode_texts(
            query,
            model=self.embedder,
            model_name=self.model_name,
            normalize=True,
        )
        embed_time_ms = (time.perf_counter() - t_embed_0) * 1000.0

        # 2. Determine combined eligibility filter
        applied_filter = filter_ids
        if filter_category is not None:
            cat_allowed = {d.doc_id for d in self.corpus if d.category == filter_category}
            if applied_filter is None:
                applied_filter = cat_allowed
            elif callable(applied_filter):
                fn = applied_filter
                applied_filter = lambda eid: bool(eid in cat_allowed and fn(eid))
            else:
                applied_filter = set(int(x) for x in applied_filter) & cat_allowed

        # 3. Compute Exact Reference Ground Truth over eligible live documents
        gt_result = self._exact_oracle.search(query_vec, k=k, filter_ids=applied_filter)
        true_topk_ids = set(gt_result.ids.tolist())

        # 4. Search the active index with query-time parameters
        search_kwargs: dict[str, Any] = {}
        if self.index_type == "IVF-Flat" and nprobe is not None:
            search_kwargs["nprobe"] = nprobe
        elif self.index_type == "HNSW" and ef_search is not None:
            search_kwargs["ef_search"] = ef_search

        t_search_0 = time.perf_counter()
        search_res = self.index.search(query_vec, k=k, filter_ids=applied_filter, **search_kwargs)
        search_latency_ms = (time.perf_counter() - t_search_0) * 1000.0

        # 5. Map retrieved IDs to original Document records and evaluate exact top-k match
        retrieved_docs: list[Document] = []
        is_exact_matches: list[bool] = []

        for ret_id in search_res.ids:
            doc_obj = self.doc_map.get(int(ret_id))
            if doc_obj is not None:
                retrieved_docs.append(doc_obj)
                is_exact_matches.append(int(ret_id) in true_topk_ids)

        return SemanticSearchResult(
            query=query,
            retrieved_documents=retrieved_docs,
            distances=search_res.distances,
            is_exact_match=is_exact_matches,
            latency_ms=search_latency_ms,
            embed_time_ms=embed_time_ms,
            distance_computations=search_res.distance_computations,
        )

    def __len__(self) -> int:
        """Return the number of documents in the index."""
        return len(self.corpus)

    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        return self.d

    def __contains__(self, doc_id: int) -> bool:
        """Check whether a document ID exists in the index."""
        return int(doc_id) in self.doc_map

    def get_document(self, doc_id: int) -> Document | None:
        """Retrieve a document by its integer ID, or None if not found."""
        return self.doc_map.get(int(doc_id))


def build_semantic_engine(
    corpus: list[Document] | None = None,
    index_type: str = "Exact BLAS",
    model: Any | None = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    ivf_nlist: int = 4,
    ivf_nprobe: int = 2,
    hnsw_m: int = 8,
    hnsw_ef_construction: int = 32,
    hnsw_ef_search: int = 32,
) -> SemanticSearchEngine:
    """Build and initialize an end-to-end SemanticSearchEngine from a document corpus.

    Parameters
    ----------
    corpus : list[Document] | None, default=None
        List of documents to index. If None, loads default DEMO_CORPUS.
    index_type : str, default='Exact BLAS'
        Index architecture ('Exact BLAS', 'Exact Flat (Ref)', 'IVF-Flat', 'HNSW').
    model : Any | None, default=None
        Pre-loaded model or embedding callable.
    model_name : str, default='sentence-transformers/all-MiniLM-L6-v2'
        Pretrained model identifier.
    ivf_nlist : int, default=4
        Voronoi centroid count for IVF-Flat.
    ivf_nprobe : int, default=2
        Probed centroid count for IVF-Flat.
    hnsw_m : int, default=8
        Out-degree connectivity for HNSW.
    hnsw_ef_construction : int, default=32
        Construction beam depth for HNSW.
    hnsw_ef_search : int, default=32
        Query exploration depth for HNSW.

    Returns
    -------
    SemanticSearchEngine
        Initialized search engine ready for semantic queries.
    """
    docs = load_demo_corpus() if corpus is None else list(corpus)
    texts = [d.text for d in docs]
    embeddings = encode_texts(texts, model=model, model_name=model_name, normalize=True)

    return SemanticSearchEngine(
        corpus=docs,
        embeddings=embeddings,
        index_type=index_type,
        embedder=model,
        model_name=model_name,
        ivf_nlist=ivf_nlist,
        ivf_nprobe=ivf_nprobe,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
        hnsw_ef_search=hnsw_ef_search,
    )
