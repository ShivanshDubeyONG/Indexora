"""Indexora Interactive Index Lab — Web-based exploration UI.

Built with Streamlit and backed directly by Indexora's core Python vector engine.
Demonstrates exact search, IVF-Flat, IVF-PQ, HNSW, filtered retrieval, mutations,
and atomic disk persistence with interactive 2D projections and performance metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

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
)
from indexora.semantic import (
    DEFAULT_EMBEDDING_MODEL,
    Document,
    SemanticSearchEngine,
    SemanticSearchResult,
    build_semantic_engine,
    is_sentence_transformers_available,
    load_demo_corpus,
    mock_embed_texts,
)


def _apply_custom_styling() -> None:
    """Inject clean developer-tool CSS styling inspired by modern tech dashboards into Streamlit."""
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Montserrat:wght@500;600;700;800&family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">

        <style>
        /* Global Typography: Poppins for clean, modern readability */
        html, body, [class*="css"], .stApp, p, span, div, label, li, .stMarkdown, .stSelectbox, .stSlider {
            font-family: 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            letter-spacing: -0.01em;
        }

        /* Headings: Montserrat for bold, geometric, tech-forward authority */
        h1, h2, h3, h4, h5, h6, .main-header, [data-testid="stHeader"] {
            font-family: 'Montserrat', -apple-system, BlinkMacSystemFont, sans-serif !important;
            font-weight: 700 !important;
            letter-spacing: -0.025em !important;
        }

        .main-header {
            font-family: 'Montserrat', sans-serif !important;
            font-size: 2.25rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.035em !important;
            margin-bottom: 0.25rem !important;
            line-height: 1.2 !important;
        }
        .sub-header {
            font-family: 'Poppins', sans-serif !important;
            font-size: 0.98rem !important;
            font-weight: 400 !important;
            color: #64748b !important;
            margin-bottom: 1.5rem !important;
        }

        /* Technical Monospace: JetBrains Mono for Code, Coordinates, Numbers, Distances */
        code, pre, .font-mono, .stMetric [data-testid="stMetricValue"], [data-testid="stDataFrame"], .stCode, kbd {
            font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
        }

        /* Metric Cards: Crisp, rounded borders with JetBrains Mono numbers */
        [data-testid="stMetric"] {
            background: rgba(248, 250, 252, 0.85);
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
            transition: border-color 0.15s ease, box-shadow 0.15s ease;
        }
        [data-testid="stMetric"]:hover {
            border-color: #cbd5e1;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }
        [data-testid="stMetricLabel"] {
            font-family: 'Poppins', sans-serif !important;
            font-size: 0.75rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.04em !important;
            color: #64748b !important;
        }
        [data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 1.5rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
            color: #0f172a !important;
        }

        /* Navigation Tabs: Montserrat font with subtle pills */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.75rem;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            font-family: 'Montserrat', sans-serif !important;
            font-size: 0.92rem !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em !important;
            padding: 0.45rem 0.9rem !important;
            border-radius: 6px !important;
            color: #475569 !important;
            transition: all 0.15s ease-in-out;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #0f172a !important;
            background-color: #f1f5f9 !important;
        }
        .stTabs [aria-selected="true"] {
            background-color: #e0f2fe !important;
            color: #0284c7 !important;
        }

        /* Buttons: Rounded with modern hover feel */
        .stButton > button {
            font-family: 'Poppins', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            border-radius: 6px !important;
            letter-spacing: -0.01em !important;
            transition: all 0.15s ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(2, 132, 199, 0.15);
        }

        /* Form Inputs & Selects */
        .stTextInput input, .stSelectbox [data-baseweb="select"], .stNumberInput input {
            font-family: 'Poppins', sans-serif !important;
            border-radius: 6px !important;
            font-size: 0.9rem !important;
        }

        /* Info & Callout Boxes */
        .callout-disclaimer {
            background-color: #f8fafc;
            border-left: 3px solid #0284c7;
            padding: 0.75rem 1rem;
            font-size: 0.88rem;
            color: #334155;
            border-radius: 0 6px 6px 0;
            margin-bottom: 1rem;
        }

        /* Dark mode compatibility */
        @media (prefers-color-scheme: dark) {
            .sub-header {
                color: #94a3b8 !important;
            }
            [data-testid="stMetric"] {
                background: rgba(30, 41, 59, 0.6) !important;
                border-color: #334155 !important;
            }
            [data-testid="stMetricLabel"] {
                color: #94a3b8 !important;
            }
            [data-testid="stMetricValue"] {
                color: #f8fafc !important;
            }
            .stTabs [data-baseweb="tab"] {
                color: #94a3b8 !important;
            }
            .stTabs [data-baseweb="tab"]:hover {
                color: #f8fafc !important;
                background-color: #1e293b !important;
            }
            .stTabs [aria-selected="true"] {
                background-color: rgba(2, 132, 199, 0.2) !important;
                color: #38bdf8 !important;
            }
            .callout-disclaimer {
                background-color: #1e293b;
                color: #cbd5e1;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_session_state() -> None:
    """Initialize persistent session state variables for dataset and session."""
    if "dataset" not in st.session_state:
        st.session_state.dataset = generate_lab_dataset(
            n_base=120,
            n_queries=6,
            d=8,
            n_clusters=4,
            cluster_std=0.25,
            seed=42,
        )
    if "lab_session" not in st.session_state:
        ds: LabDataset = st.session_state.dataset
        idx = ExactBlasIndex(database=ds.base, ids=ds.base_ids)
        st.session_state.lab_session = LabSession(index=idx, index_name="ExactBlasIndex", d=ds.d)
    if "benchmark_results" not in st.session_state:
        st.session_state.benchmark_results = None
    if "index_cache" not in st.session_state:
        st.session_state.index_cache = IndexCache()


def _get_or_build_index(
    index_type: str,
    ds: LabDataset,
    cache: IndexCache,
    ivf_nlist: int = 4,
    pq_m: int = 2,
    pq_ksub: int = 4,
    hnsw_m: int = 8,
    hnsw_ef_construction: int = 32,
) -> tuple[Any, float, bool]:
    """Retrieve index from cache or build it if build-time parameters or dataset changed.

    Separates build-time hyperparameters (e.g. nlist, M, ef_construction) from
    query-time exploration parameters (e.g. nprobe, ef_search) to prevent expensive
    re-indexing on ordinary UI rerenders.
    """
    if index_type == "Exact Flat (Reference)":
        build_params: dict[str, Any] = {}
        builder = lambda: ExactFlatIndex(database=ds.base, ids=ds.base_ids)
    elif index_type == "Exact BLAS":
        build_params = {}
        builder = lambda: ExactBlasIndex(database=ds.base, ids=ds.base_ids)
    elif index_type == "IVF-Flat":
        build_params = {"nlist": ivf_nlist}
        builder = lambda: IVFFlatIndex(database=ds.base, ids=ds.base_ids, nlist=ivf_nlist)
    elif index_type == "IVF-PQ":
        build_params = {"nlist": ivf_nlist, "M": pq_m, "k_sub": pq_ksub}
        builder = lambda: IVFPQIndex(database=ds.base, ids=ds.base_ids, nlist=ivf_nlist, M=pq_m, k_sub=pq_ksub)
    elif index_type == "HNSW":
        build_params = {"M": hnsw_m, "ef_construction": hnsw_ef_construction, "seed": ds.seed}
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

    return cache.get_or_build(index_type, ds, build_params, builder)


def main() -> None:
    st.set_page_config(
        page_title="Indexora — Interactive Index Lab",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_custom_styling()
    _init_session_state()

    # -------------------------------------------------------------------------
    # Sidebar: Dataset Generator Controls
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("⚡ Lab Configuration")
        st.markdown("**Synthetic Clustered Dataset**")

        sb_n_base = st.slider("Base Vectors (N)", min_value=30, max_value=400, value=120, step=10)
        sb_d = st.slider("Dimensionality (d)", min_value=4, max_value=32, value=8, step=4)
        sb_clusters = st.slider("Clusters", min_value=2, max_value=8, value=4, step=1)
        sb_std = st.slider("Cluster Dispersion (std)", min_value=0.1, max_value=0.8, value=0.25, step=0.05)
        sb_seed = st.number_input("RNG Seed", min_value=1, max_value=999999, value=42, step=1)

        if st.button("🔄 Regenerate Dataset", use_container_width=True):
            st.session_state.dataset = generate_lab_dataset(
                n_base=sb_n_base,
                n_queries=8,
                d=sb_d,
                n_clusters=sb_clusters,
                cluster_std=sb_std,
                seed=int(sb_seed),
            )
            ds = st.session_state.dataset
            idx = ExactBlasIndex(database=ds.base, ids=ds.base_ids)
            st.session_state.lab_session = LabSession(index=idx, index_name="ExactBlasIndex", d=ds.d)
            st.session_state.benchmark_results = None
            if "index_cache" in st.session_state:
                st.session_state.index_cache.clear()
            st.success("New synthetic dataset generated!")

        st.markdown("---")
        ds_curr: LabDataset = st.session_state.dataset
        st.caption(
            f"**Current Dataset Info:**\n"
            f"- Base: `{ds_curr.N}` vectors\n"
            f"- Queries: `{ds_curr.Q}` vectors\n"
            f"- Dimension: `{ds_curr.d}`-D float32\n"
            f"- Clusters: `{ds_curr.n_clusters}` centers\n"
            f"- IDs: `[{ds_curr.base_ids[0]} ... {ds_curr.base_ids[-1]}]`"
        )
        st.markdown(
            """
            <div class="callout-disclaimer">
            <strong>Data Disclosure:</strong> This lab uses synthetic numeric vector coordinates in ℝᵈ for index mechanics demonstration. It is <em>not</em> semantic text search or embedding retrieval.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # Header Banner
    # -------------------------------------------------------------------------
    st.markdown('<div class="main-header">⚡ Indexora — Interactive Index Lab</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Inspect, benchmark, and mutate vector index architectures from first principles.</div>',
        unsafe_allow_html=True,
    )

    tab_overview, tab_playground, tab_benchmark, tab_mutations, tab_semantic = st.tabs(
        [
            "📘 Overview & Concepts",
            "🔍 Search Playground",
            "📊 Accuracy & Performance Lab",
            "⚙️ Mutations & Persistence",
            "📚 Semantic Text Search",
        ]
    )

    ds: LabDataset = st.session_state.dataset

    # =========================================================================
    # Tab 1: Overview & Concepts
    # =========================================================================
    with tab_overview:
        st.subheader("Architecture Breakdown")
        st.markdown(
            "Indexora implements foundational nearest-neighbor search algorithms directly in Python and NumPy. "
            "Explore how each index partitions the vector space and trades accuracy for candidate evaluation speed:"
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                ### 1. Exact Flat / Exact BLAS (Baseline)
                - **Concept:** Brute-force exhaustive scan over all $N$ database vectors.
                - **Complexity:** $O(N \cdot d)$ distance computations per query.
                - **Guarantee:** 100% Recall@k. Serves as the correctness oracle.
                - **BLAS Acceleration:** Exact BLAS evaluates $\|x - y\|^2 = \|x\|^2 + \|y\|^2 - 2(x \cdot y)$ via optimized matrix multiplication.
                """
            )
            st.markdown(
                """
                ### 2. IVF-Flat (Inverted File Flat)
                - **Concept:** Partitions vector space into `nlist` Voronoi cells using $k$-means clustering.
                - **Query:** Ranks coarse centroids, probes only the nearest `nprobe` inverted posting lists.
                - **Trade-off:** Evaluates a fraction of total vectors. Higher `nprobe` recovers lost boundary recall.
                """
            )

        with col2:
            st.markdown(
                """
                ### 3. IVF-PQ (Product Quantization)
                - **Concept:** Decomposes $d$-dimensional space into $M$ orthogonal subspaces. Quantizes residuals into compact `uint8` codes.
                - **Query:** Asymmetric Distance Computation (ADC) builds a small lookup table (LUT) and scores candidates in microseconds.
                - **Trade-off:** Compresses memory up to 16x. Distances are quantized approximations.
                """
            )
            st.markdown(
                """
                ### 4. HNSW (Hierarchical Navigable Small World)
                - **Concept:** Multi-layer proximity graph with exponential level decay.
                - **Query:** Logarithmic greedy skip-list routing on upper layers, bounded beam search at layer 0.
                - **Trade-off:** High recall with very few distance evaluations (~15% of database).
                """
            )

        st.markdown("---")
        st.subheader("Current Dataset Distribution Preview")
        df_preview = pd.DataFrame(
            ds.base[:8],
            columns=[f"dim_{i}" for i in range(ds.d)],
            index=[f"ID {int(x)}" for x in ds.base_ids[:8]],
        )
        st.dataframe(df_preview, use_container_width=True)

    # =========================================================================
    # Tab 2: Search Playground
    # =========================================================================
    with tab_playground:
        st.subheader("Interactive Query Search")

        c_idx, c_q, c_k = st.columns([1.5, 1.2, 1.0])
        with c_idx:
            selected_index_name = st.selectbox(
                "Select Index Architecture",
                ["Exact BLAS", "Exact Flat (Reference)", "IVF-Flat", "IVF-PQ", "HNSW"],
            )
        with c_q:
            query_opts = [f"Query #{i} (ID {int(ds.query_ids[i])})" for i in range(ds.Q)]
            query_opts.append("Custom Random Vector")
            selected_query_label = st.selectbox("Select Query Vector", query_opts)
        with c_k:
            k_val = st.slider("Top-k", min_value=1, max_value=min(ds.N, 30), value=5)

        # Dynamic Hyperparameters per index
        nprobe_val = 2
        ef_search_val = 32
        if selected_index_name in ("IVF-Flat", "IVF-PQ"):
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                nprobe_val = st.slider("nprobe (Probed Centroids)", min_value=1, max_value=4, value=2)
            with c_p2:
                st.caption(f"Index built with `nlist=4`. Probing {nprobe_val}/4 Voronoi cells per query.")
        elif selected_index_name == "HNSW":
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                ef_search_val = st.slider("ef_search (Beam Exploration Depth)", min_value=4, max_value=64, value=32, step=4)
            with c_p2:
                st.caption(f"Layer 0 search explores candidate beam of size `ef={ef_search_val}`.")

        # Optional ID filter
        with st.expander("🔎 Optional ID Eligibility Filter (filter_ids)"):
            filter_mode = st.radio(
                "Filter Preset",
                ["None (All Vectors Eligible)", "Even IDs Only", "Exclude True Top-1 ID", "Custom ID List"],
                horizontal=True,
            )
            custom_filter_input = ""
            if filter_mode == "Custom ID List":
                custom_filter_input = st.text_input(
                    "Comma-separated External IDs allowed:",
                    value=f"{ds.base_ids[0]}, {ds.base_ids[1]}, {ds.base_ids[2]}",
                )

        # Prepare Query Vector
        if selected_query_label == "Custom Random Vector":
            rng_q = np.random.default_rng(999)
            query_vec = rng_q.normal(loc=0.0, scale=1.0, size=(ds.d,)).astype(np.float32)
        else:
            q_idx = int(selected_query_label.split("#")[1].split(" ")[0])
            query_vec = ds.queries[q_idx]

        # Parse Filter IDs before ground truth and index search
        applied_filter = None
        if filter_mode == "Even IDs Only":
            applied_filter = lambda eid: eid % 2 == 0
        elif filter_mode == "Exclude True Top-1 ID":
            unfiltered_gt = compute_exact_ground_truth(database=ds.base, ids=ds.base_ids, query=query_vec, k=1)
            top1_id = int(unfiltered_gt.ids[0]) if len(unfiltered_gt.ids) > 0 else -1
            applied_filter = lambda eid, excluded=top1_id: eid != excluded
        elif filter_mode == "Custom ID List":
            try:
                applied_filter = set(int(x.strip()) for x in custom_filter_input.split(",") if x.strip())
            except ValueError:
                st.error("Invalid ID format in custom filter. Enter comma-separated integers.")
                applied_filter = set()

        # Compute exact ground truth specifically for this query and the active filter
        filtered_gt = compute_exact_ground_truth(
            database=ds.base,
            ids=ds.base_ids,
            query=query_vec,
            k=k_val,
            filter_ids=applied_filter,
        )
        eligible_gt_ids = set(filtered_gt.ids.tolist())
        num_eligible_live = len(eligible_gt_ids)

        # Retrieve or Build Index from Cache (separating build params from query params)
        idx_instance, build_time_ms, was_cached = _get_or_build_index(
            index_type=selected_index_name,
            ds=ds,
            cache=st.session_state.index_cache,
            ivf_nlist=4,
            pq_m=2,
            pq_ksub=4,
            hnsw_m=8,
            hnsw_ef_construction=32,
        )

        if was_cached:
            st.caption(f"⚡ Reused cached index instance ({selected_index_name}).")
        else:
            st.caption(f"🔨 Built index ({selected_index_name}) in {build_time_ms:.2f} ms.")

        # Execute Search with Query-Time Parameters
        t_start = time.perf_counter()
        search_kwargs: dict[str, Any] = {}
        if selected_index_name in ("IVF-Flat", "IVF-PQ"):
            search_kwargs["nprobe"] = nprobe_val
        elif selected_index_name == "HNSW":
            search_kwargs["ef_search"] = ef_search_val

        search_res = idx_instance.search(query_vec, k=k_val, filter_ids=applied_filter, **search_kwargs)
        search_time_ms = (time.perf_counter() - t_start) * 1000.0

        # Display Metrics
        retrieved_ids = set(search_res.ids.tolist())
        hits = len(retrieved_ids & eligible_gt_ids)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Query Latency", f"{search_time_ms:.3f} ms")
        with m2:
            dc_val = search_res.distance_computations if search_res.distance_computations is not None else ds.N
            st.metric("Distances Evaluated", f"{dc_val} / {ds.N}")
        with m3:
            if num_eligible_live == 0:
                match_label = "100.0% (Empty Filter Set)" if len(retrieved_ids) == 0 else "0.0%"
            else:
                match_label = f"{(hits / num_eligible_live) * 100.0:.1f}% ({hits}/{num_eligible_live})"
            st.metric("Filtered Exact Match Rate", match_label)
        with m4:
            st.metric("Retrieved Candidates", f"{len(search_res.ids)} / {k_val}")

        # Display Results Table
        st.markdown("#### Retrieved Nearest Neighbors")
        if len(search_res.ids) == 0:
            if num_eligible_live == 0:
                st.info("ℹ️ Filter restriction resulted in 0 eligible database vectors. Empty result correctly matches filtered ground truth.")
            else:
                st.warning("No eligible candidates retrieved by the index matching the query and filter criteria.")
        else:
            table_rows = []
            for rank_i, (dist_val, id_val) in enumerate(zip(search_res.distances, search_res.ids), start=1):
                is_exact_match = id_val in eligible_gt_ids
                table_rows.append({
                    "Rank": rank_i,
                    "External ID": int(id_val),
                    "Squared L2 Distance (||q - x||²)": f"{dist_val:.5f}",
                    "Filtered Ground Truth Status": "✅ Exact Match" if is_exact_match else "⚡ Approximation",
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        # 2D PCA Visualization
        st.markdown("#### 2D PCA Projection Visualizer")
        st.caption(
            "Visualizing the synthetic vector distribution projected to 2D via deterministic SVD PCA. "
            "Note: Distances on this 2D plane can be visually distorted relative to true ℝᵈ squared-L2 distances."
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        # 1. Base vectors
        base_proj = ds.project_2d(ds.base)
        scatter = ax.scatter(
            base_proj[:, 0],
            base_proj[:, 1],
            c=ds.cluster_labels,
            cmap="viridis",
            alpha=0.45,
            s=40,
            label="Database Vectors",
        )

        # 2. Query point
        q_proj = ds.project_2d(query_vec)
        ax.scatter(
            q_proj[0],
            q_proj[1],
            color="red",
            marker="*",
            s=220,
            zorder=10,
            label="Query Point",
        )

        # 3. Retrieved neighbors
        if len(search_res.ids) > 0:
            id_to_idx = {int(eid): i for i, eid in enumerate(ds.base_ids)}
            retrieved_indices = [id_to_idx[int(eid)] for eid in search_res.ids if int(eid) in id_to_idx]
            if retrieved_indices:
                ax.scatter(
                    base_proj[retrieved_indices, 0],
                    base_proj[retrieved_indices, 1],
                    facecolors="none",
                    edgecolors="crimson",
                    s=140,
                    linewidths=2.0,
                    zorder=9,
                    label="Retrieved Nearest Neighbors",
                )
                for rank_i, r_idx in enumerate(retrieved_indices, start=1):
                    ax.annotate(
                        f"#{rank_i}",
                        (base_proj[r_idx, 0] + 0.05, base_proj[r_idx, 1] + 0.05),
                        fontsize=9,
                        fontweight="bold",
                        color="#991b1b",
                    )

        ax.set_title(f"2D PCA Map — {selected_index_name} Search (Top-{k_val})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Principal Component 1", fontsize=9)
        ax.set_ylabel("Principal Component 2", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

    # =========================================================================
    # Tab 3: Accuracy & Performance Lab
    # =========================================================================
    with tab_benchmark:
        st.subheader("Accuracy vs. Speed Benchmark Sweep")
        st.markdown(
            "Compare all 5 core Indexora implementations on identical query vectors under the same top-$k$. "
            "Examines the trade-offs among build cost, query latency percentiles, throughput (QPS), and Recall@k."
        )

        b_k = st.slider("Benchmark Top-k", min_value=1, max_value=min(ds.N, 20), value=5, key="bench_k")

        if st.button("🚀 Run Comparative Benchmark", use_container_width=True):
            with st.spinner("Executing comparative benchmark across all 5 index implementations..."):
                gt = ds.get_ground_truth(k=b_k)
                bench_configs = [
                    ("Exact BLAS", lambda: ExactBlasIndex(database=ds.base, ids=ds.base_ids), {}),
                    ("Exact Flat (Ref)", lambda: ExactFlatIndex(database=ds.base, ids=ds.base_ids), {}),
                    ("IVF-Flat (nprobe=1)", lambda: IVFFlatIndex(database=ds.base, ids=ds.base_ids, nlist=4, nprobe=1), {"nprobe": 1}),
                    ("IVF-Flat (nprobe=2)", lambda: IVFFlatIndex(database=ds.base, ids=ds.base_ids, nlist=4, nprobe=2), {"nprobe": 2}),
                    ("IVF-Flat (nprobe=4)", lambda: IVFFlatIndex(database=ds.base, ids=ds.base_ids, nlist=4, nprobe=4), {"nprobe": 4}),
                    ("IVF-PQ (M=2, nprobe=1)", lambda: IVFPQIndex(database=ds.base, ids=ds.base_ids, nlist=4, M=2, k_sub=4, nprobe=1), {"nprobe": 1}),
                    ("IVF-PQ (M=2, nprobe=2)", lambda: IVFPQIndex(database=ds.base, ids=ds.base_ids, nlist=4, M=2, k_sub=4, nprobe=2), {"nprobe": 2}),
                    ("HNSW (ef=8)", lambda: HNSWIndex(database=ds.base, ids=ds.base_ids, M=8, ef_construction=32, ef_search=8, random_seed=ds.seed), {"ef_search": 8}),
                    ("HNSW (ef=32)", lambda: HNSWIndex(database=ds.base, ids=ds.base_ids, M=8, ef_construction=32, ef_search=32, random_seed=ds.seed), {"ef_search": 32}),
                ]

                results: list[IndexBenchmarkMetrics] = []
                for name, builder_fn, skwargs in bench_configs:
                    t_b0 = time.perf_counter()
                    inst = builder_fn()
                    b_time_ms = (time.perf_counter() - t_b0) * 1000.0
                    m = evaluate_index_search(
                        index=inst,
                        index_name=name,
                        queries=ds.queries,
                        ground_truth=gt,
                        k=b_k,
                        build_time_ms=b_time_ms,
                        search_kwargs=skwargs,
                        warmup=1,
                        repetitions=3,
                    )
                    results.append(m)

                st.session_state.benchmark_results = results

        if st.session_state.benchmark_results:
            bench_data: list[IndexBenchmarkMetrics] = st.session_state.benchmark_results
            summary_table = [
                {
                    "Index Configuration": r.index_name,
                    "Build Time (ms)": f"{r.build_time_ms:.2f}",
                    "Median Latency (ms)": f"{r.median_latency_ms:.4f}",
                    "p95 Latency (ms)": f"{r.p95_latency_ms:.4f}",
                    "Throughput (QPS)": f"{r.qps:.1f}",
                    "Recall@k": f"{r.recall_at_k:.4f}",
                    "Distances / Query": f"{r.distance_computations_per_query:.1f}" if r.distance_computations_per_query else "N/A",
                    "Storage (KiB)": f"{r.memory_kb:.2f}",
                    "Storage Method": r.memory_method,
                }
                for r in bench_data
            ]
            st.dataframe(pd.DataFrame(summary_table), use_container_width=True)

            # Charts
            c_c1, c_c2 = st.columns(2)
            with c_c1:
                st.markdown("#### Recall@k vs. Median Latency")
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                for r in bench_data:
                    ax1.scatter(r.median_latency_ms, r.recall_at_k, s=120, alpha=0.85)
                    ax1.annotate(r.index_name, (r.median_latency_ms, r.recall_at_k + 0.015), fontsize=8)
                ax1.set_xlabel("Median Query Latency (ms)", fontsize=9)
                ax1.set_ylabel("Recall@k", fontsize=9)
                ax1.set_ylim(-0.05, 1.15)
                ax1.grid(True, linestyle="--", alpha=0.5)
                ax1.set_title("Recall vs. Latency Trade-Off", fontsize=10, fontweight="bold")
                st.pyplot(fig1)
                plt.close(fig1)

            with c_c2:
                st.markdown("#### Distance Computations per Query")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                names = [r.index_name for r in bench_data if r.distance_computations_per_query is not None]
                dcs = [r.distance_computations_per_query for r in bench_data if r.distance_computations_per_query is not None]
                y_pos = np.arange(len(names))
                ax2.barh(y_pos, dcs, color="#3b82f6", alpha=0.8)
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(names, fontsize=8)
                ax2.invert_yaxis()
                ax2.set_xlabel("Average Distance Computations per Query", fontsize=9)
                ax2.grid(True, axis="x", linestyle="--", alpha=0.5)
                ax2.set_title("Candidate Evaluation Efficiency", fontsize=10, fontweight="bold")
                st.pyplot(fig2)
                plt.close(fig2)

            st.info(
                "💡 **Performance & Scalability Insights:**\n\n"
                "- **Small-Scale Overheads:** On small synthetic datasets (e.g. $N \\approx 120$), Exact BLAS often "
                "reports lower query latency than HNSW or IVF because BLAS matrix multiplication runs in optimized, "
                "compiled C/Fortran libraries, whereas Python-level graph traversals and Voronoi cell lookups incur "
                "interpreter loop overhead.\n"
                "- **Algorithmic vs. Wall-Clock Trade-Offs:** While HNSW ($O(\\log N)$ graph exploration) and IVF "
                "(Voronoi candidate pruning) evaluate significantly fewer vector distances per query, wall-clock "
                "speedups depend heavily on vector dimensionality, clustering quality, target recall ($k$, `nprobe`, "
                "`ef_search`), BLAS multi-threading, and implementation language overhead.\n"
                "- **Evidence-Based Evaluation:** Rather than assuming asymptotic speedups at any fixed dataset size, "
                "use Indexora's offline benchmark harness (`python -m indexora.benchmark`) to measure empirical "
                "recall-vs-latency curves for your specific hardware and workload."
            )

    # =========================================================================
    # Tab 4: Mutations & Persistence (Phase 11 Lab)
    # =========================================================================
    with tab_mutations:
        st.subheader("Index Mutations & Atomic Persistence Playground")
        st.markdown(
            "Test Indexora Phase 11 features: dynamically add new vectors with external IDs, tombstone/delete "
            "items, verify search exclusion, and persist the index to disk with SHA-256 integrity checks."
        )

        session: LabSession = st.session_state.lab_session

        stat1, stat2, stat3, stat4 = st.columns(4)
        with stat1:
            st.metric("Active Index Type", session.index_name)
        with stat2:
            st.metric("Searchable Vectors", session.active_count)
        with stat3:
            st.metric("Incrementally Added", len(session.added_ids))
        with stat4:
            st.metric("Tombstoned / Deleted", len(session.deleted_ids))

        st.markdown("---")
        m_col1, m_col2 = st.columns(2)

        # Section 1: Add Vector
        with m_col1:
            st.markdown("#### 1. Incrementally Add Vector (`add`)")
            add_id = st.number_input("External ID to assign:", min_value=1, max_value=999999, value=9999, step=1)
            add_val = st.slider("Vector Constant Value (fills d dimensions):", min_value=-5.0, max_value=5.0, value=3.5, step=0.5)

            if st.button("➕ Add Vector to Index", use_container_width=True):
                vec_to_add = np.full(session.d, add_val, dtype=np.float32)
                try:
                    assigned = session.add_vector(vec_to_add, ext_id=int(add_id))
                    st.success(f"Successfully added vector with ID `{assigned}`. Active count is now `{session.active_count}`.")
                except Exception as exc:
                    st.error(f"Failed to add vector: {exc}")

        # Section 2: Delete Vector
        with m_col2:
            st.markdown("#### 2. Delete Vector (`delete`)")
            del_id = st.number_input("External ID to delete:", min_value=1, max_value=999999, value=int(add_id), step=1)

            if st.button("🗑️ Delete Vector from Index", use_container_width=True):
                try:
                    del_cnt = session.delete_vector(int(del_id))
                    if del_cnt > 0:
                        st.success(f"Successfully deleted ID `{del_id}` (1 item removed). Active count is now `{session.active_count}`.")
                    else:
                        st.warning(f"ID `{del_id}` was not found in the active index (0 deleted).")
                except Exception as exc:
                    st.error(f"Failed to delete vector: {exc}")

        st.markdown("---")
        st.markdown("#### 3. Atomic Disk Persistence (`save_index` / `load_index`)")
        st.markdown(
            "Persists index hyperparameters and numerical array weights to disk without Python pickle. "
            "Generates a versioned `manifest.json` with streaming SHA-256 file checksums."
        )

        p_col1, p_col2 = st.columns(2)
        demo_save_dir = Path("demo_indexes/lab_session")

        with p_col1:
            st.markdown("**Save Active Index**")
            save_path_str = st.text_input("Target Directory:", value=str(demo_save_dir))
            if st.button("💾 Save Index to Disk", use_container_width=True):
                try:
                    target_p = Path(save_path_str)
                    manifest_data = session.save(target_p, overwrite=True)
                    st.success(f"Index successfully saved to `{target_p}`!")
                    st.json(manifest_data)
                except Exception as exc:
                    st.error(f"Save failed: {exc}")

        with p_col2:
            st.markdown("**Restore Saved Index**")
            load_path_str = st.text_input("Source Directory:", value=str(demo_save_dir), key="load_dir")
            if st.button("📂 Load & Verify Index", use_container_width=True):
                try:
                    source_p = Path(load_path_str)
                    restored_idx = session.load(source_p, verify_checksums=True)
                    st.success(f"Successfully restored `{type(restored_idx).__name__}` with full SHA-256 verification!")

                    # Verification query
                    q_test = np.zeros(session.d, dtype=np.float32)
                    v_res = session.search(q_test, k=3)
                    st.info(f"Test query on restored index returned top-3 IDs: `{v_res.ids.tolist()}`.")
                except Exception as exc:
                    st.error(f"Load failed: {exc}")

    # =========================================================================
    # Tab 5: Semantic Text Search (Phase 12B)
    # =========================================================================
    with tab_semantic:
        st.subheader("📚 End-to-End Semantic Text Search")
        st.markdown(
            "Query a curated corpus of 25 technical engineering documents using dense sentence embeddings "
            "and Indexora's native vector retrieval engines (**Exact BLAS**, **IVF-Flat**, and **HNSW**)."
        )

        has_st = is_sentence_transformers_available()
        use_mock = False

        if not has_st:
            st.warning(
                "⚠️ **Optional Semantic Dependencies Not Found**\n\n"
                "Install `sentence-transformers` and `torch` to enable pretrained neural embeddings:\n\n"
                "```bash\n"
                "pip install -e \".[semantic]\"\n"
                "```\n\n"
                "You can still explore the retrieval pipeline using deterministic offline mock embeddings."
            )
            use_mock = st.checkbox("Enable offline mock embeddings", value=True)
            if not use_mock:
                st.stop()
        else:
            st.success("✅ `sentence-transformers` active. Using pretrained model `all-MiniLM-L6-v2` (384 dimensions, L2-normalized).")

        corpus_docs = load_demo_corpus()

        # Engine caching in session state
        embed_cache_key = "semantic_doc_embeddings_mock" if use_mock else "semantic_doc_embeddings_real"
        if embed_cache_key not in st.session_state:
            with st.spinner("Generating document corpus embeddings..."):
                texts = [d.text for d in corpus_docs]
                if use_mock:
                    st.session_state[embed_cache_key] = mock_embed_texts(texts, dim=64)
                else:
                    from indexora.semantic import encode_texts
                    st.session_state[embed_cache_key] = encode_texts(texts, normalize=True)

        cached_embeddings = st.session_state[embed_cache_key]

        # UI Controls
        c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([2, 1, 1])
        with c_ctrl1:
            sem_index_choice = st.selectbox(
                "Retrieval Engine Architecture:",
                options=["Exact BLAS", "Exact Flat (Ref)", "IVF-Flat", "HNSW"],
                index=0,
                key="sem_idx_choice",
            )
        with c_ctrl2:
            all_categories = ["All Categories"] + sorted({d.category for d in corpus_docs})
            selected_cat = st.selectbox("Category Filter:", options=all_categories, index=0)
        with c_ctrl3:
            sem_k = st.slider("Top-k Results:", min_value=1, max_value=10, value=5, step=1)

        # Index Tuning Parameters
        sem_nprobe = 2
        sem_ef = 32
        if sem_index_choice == "IVF-Flat":
            sem_nprobe = st.slider("IVF Probed Centroids (`nprobe`):", min_value=1, max_value=4, value=2, step=1)
        elif sem_index_choice == "HNSW":
            sem_ef = st.slider("HNSW Query Exploration Depth (`ef_search`):", min_value=8, max_value=64, value=32, step=8)

        # Build / retrieve engine
        embedder_fn = (lambda t: mock_embed_texts(t, dim=64)) if use_mock else None
        engine = SemanticSearchEngine(
            corpus=corpus_docs,
            embeddings=cached_embeddings,
            index_type=sem_index_choice,
            embedder=embedder_fn,
            model_name=DEFAULT_EMBEDDING_MODEL,
            ivf_nlist=4,
            ivf_nprobe=sem_nprobe,
            hnsw_m=8,
            hnsw_ef_construction=32,
            hnsw_ef_search=sem_ef,
        )

        st.markdown("---")
        st.markdown("#### Query Input")

        example_queries = [
            "How do attention mechanisms in language models work?",
            "LSM-tree write performance and disk flushing",
            "Consensus algorithms and fault-tolerant leader election",
            "Zero trust architecture and network encryption",
            "Decoupling services with asynchronous message queues",
        ]

        preset_cols = st.columns(len(example_queries))
        preset_clicked: str | None = None
        for i, q_text in enumerate(example_queries):
            short_label = q_text.split()[0] + " " + q_text.split()[1] + "..."
            if preset_cols[i].button(f"💡 {short_label}", key=f"ex_q_{i}", help=q_text, use_container_width=True):
                preset_clicked = q_text

        default_query_text = preset_clicked or "How do attention mechanisms in language models work?"
        user_query = st.text_input("Enter natural language search query:", value=default_query_text)

        filter_category_val = None if selected_cat == "All Categories" else selected_cat

        if user_query.strip():
            result: SemanticSearchResult = engine.search(
                query=user_query,
                k=sem_k,
                filter_category=filter_category_val,
                nprobe=sem_nprobe,
                ef_search=sem_ef,
            )

            # Metrics row
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("Total Latency", f"{result.latency_ms + result.embed_time_ms:.2f} ms")
            with m2:
                st.metric("Text Embed Time", f"{result.embed_time_ms:.2f} ms")
            with m3:
                st.metric("Index Search Time", f"{result.latency_ms:.3f} ms")
            with m4:
                dist_str = str(result.distance_computations) if result.distance_computations is not None else "N/A"
                st.metric("Distance Computations", dist_str)
            with m5:
                match_pct = (sum(result.is_exact_match) / len(result.is_exact_match) * 100) if result.is_exact_match else 0.0
                st.metric("Exact Top-k Match", f"{match_pct:.0f}%")

            st.markdown("#### Retrieved Documents")
            if not result.retrieved_documents:
                st.info("No documents found matching the query and category filter.")
            else:
                cos_sims = result.cosine_similarities
                for rank, (doc, dist, is_match, cos_sim) in enumerate(
                    zip(result.retrieved_documents, result.distances, result.is_exact_match, cos_sims),
                    start=1,
                ):
                    match_badge = "🎯 Exact Match" if is_match else "⚡ Approximation"
                    tags_badge = " ".join([f"`{t}`" for t in doc.metadata.get("tags", [])])

                    with st.container():
                        st.markdown(
                            f"**#{rank}. {doc.title}** &nbsp;•&nbsp; `{doc.category}` &nbsp;•&nbsp; "
                            f"<span style='color: {'#22c55e' if is_match else '#f59e0b'}; font-weight: 600;'>{match_badge}</span>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"📐 **Squared L2 Distance:** `{dist:.4f}` &nbsp;|&nbsp; "
                            f"**Cosine Similarity:** `{cos_sim:.4f}` &nbsp;|&nbsp; "
                            f"**Document ID:** `{doc.doc_id}`"
                        )
                        st.markdown(f"> *{doc.text}*")
                        if tags_badge:
                            st.markdown(f"🏷️ Tags: {tags_badge}")
                        st.markdown("---")

        st.markdown("#### Mathematical Equivalence: Squared Euclidean Distance & Cosine Similarity")
        st.info(
            "**Why Squared L2 Distance is used for Cosine Ranking:**\n\n"
            "Because text embeddings are unit-normalized ($||q||_2 = ||x||_2 = 1$):\n\n"
            "$$||q - x||^2 = ||q||^2 + ||x||^2 - 2(q \\cdot x) = 2 - 2 \\cos(q, x)$$\n\n"
            "Thus, minimizing squared Euclidean distance $||q - x||^2 \\in [0, 4]$ directly maximizes "
            "cosine similarity $\\cos(q, x) \\in [-1, 1]$. Indexora's native squared-L2 kernels, IVF centroids, "
            "and HNSW graph exploration seamlessly execute cosine ranking with zero algorithm alterations."
        )


if __name__ == "__main__":
    import streamlit.runtime as st_runtime

    if st_runtime.exists():
        main()
    else:
        from streamlit.web import cli as stcli

        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())
