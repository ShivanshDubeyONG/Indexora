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
    LabDataset,
    LabSession,
    evaluate_index_search,
    generate_lab_dataset,
)


def _apply_custom_styling() -> None:
    """Inject clean developer-tool CSS styling into the Streamlit app."""
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin-bottom: 0.2rem;
            color: #1e293b;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #64748b;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }
        .metric-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
        }
        .metric-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #0f172a;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 1.05rem;
            font-weight: 600;
        }
        .callout-disclaimer {
            background-color: #f1f5f9;
            border-left: 4px solid #3b82f6;
            padding: 0.75rem 1rem;
            font-size: 0.9rem;
            color: #334155;
            border-radius: 0 4px 4px 0;
            margin-bottom: 1rem;
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


def _build_selected_index(
    index_type: str,
    ds: LabDataset,
    ivf_nlist: int = 4,
    ivf_nprobe: int = 2,
    pq_m: int = 2,
    pq_ksub: int = 4,
    hnsw_m: int = 8,
    hnsw_ef_construction: int = 32,
    hnsw_ef_search: int = 32,
) -> tuple[Any, float]:
    """Instantiate and build a chosen Indexora index on dataset vectors, timing build duration."""
    t0 = time.perf_counter()
    if index_type == "Exact Flat (Reference)":
        idx = ExactFlatIndex(database=ds.base, ids=ds.base_ids)
    elif index_type == "Exact BLAS":
        idx = ExactBlasIndex(database=ds.base, ids=ds.base_ids)
    elif index_type == "IVF-Flat":
        idx = IVFFlatIndex(
            database=ds.base,
            ids=ds.base_ids,
            nlist=ivf_nlist,
            nprobe=ivf_nprobe,
        )
    elif index_type == "IVF-PQ":
        idx = IVFPQIndex(
            database=ds.base,
            ids=ds.base_ids,
            nlist=ivf_nlist,
            M=pq_m,
            k_sub=pq_ksub,
            nprobe=ivf_nprobe,
        )
    elif index_type == "HNSW":
        idx = HNSWIndex(
            database=ds.base,
            ids=ds.base_ids,
            M=hnsw_m,
            ef_construction=hnsw_ef_construction,
            ef_search=hnsw_ef_search,
            random_seed=ds.seed,
        )
    else:
        idx = ExactBlasIndex(database=ds.base, ids=ds.base_ids)
    t1 = time.perf_counter()
    build_time_ms = (t1 - t0) * 1000.0
    return idx, build_time_ms


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

    tab_overview, tab_playground, tab_benchmark, tab_mutations = st.tabs(
        ["📘 Overview & Concepts", "🔍 Search Playground", "📊 Accuracy & Performance Lab", "⚙️ Mutations & Persistence"]
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

        # Compute ground truth for this query
        gt_single = ds.get_ground_truth(k=k_val)
        gt_ids_for_q = set(gt_single.ids[q_idx if selected_query_label != "Custom Random Vector" else 0].tolist())

        # Construct and Search Index
        idx_instance, _ = _build_selected_index(
            index_type=selected_index_name,
            ds=ds,
            ivf_nlist=4,
            ivf_nprobe=nprobe_val,
            hnsw_ef_search=ef_search_val,
        )

        # Parse Filter IDs
        applied_filter = None
        if filter_mode == "Even IDs Only":
            applied_filter = lambda eid: eid % 2 == 0
        elif filter_mode == "Exclude True Top-1 ID":
            top1_id = gt_single.ids[q_idx if selected_query_label != "Custom Random Vector" else 0][0]
            applied_filter = lambda eid, excluded=top1_id: eid != excluded
        elif filter_mode == "Custom ID List":
            try:
                applied_filter = set(int(x.strip()) for x in custom_filter_input.split(",") if x.strip())
            except ValueError:
                st.error("Invalid ID format in custom filter. Enter comma-separated integers.")

        # Execute Search
        t_start = time.perf_counter()
        search_kwargs: dict[str, Any] = {}
        if selected_index_name in ("IVF-Flat", "IVF-PQ"):
            search_kwargs["nprobe"] = nprobe_val
        elif selected_index_name == "HNSW":
            search_kwargs["ef_search"] = ef_search_val

        search_res = idx_instance.search(query_vec, k=k_val, filter_ids=applied_filter, **search_kwargs)
        search_time_ms = (time.perf_counter() - t_start) * 1000.0

        # Display Metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Query Latency", f"{search_time_ms:.3f} ms")
        with m2:
            dc_val = search_res.distance_computations if search_res.distance_computations is not None else ds.N
            st.metric("Distances Evaluated", f"{dc_val} / {ds.N}")
        with m3:
            retrieved_ids = set(search_res.ids.tolist())
            hits = len(retrieved_ids & gt_ids_for_q)
            rec = hits / max(1, len(gt_ids_for_q)) if filter_mode == "None (All Vectors Eligible)" else None
            st.metric("Exact Top-k Match Rate", f"{rec * 100:.1f}%" if rec is not None else "Filtered")
        with m4:
            st.metric("Retrieved Neighbors", f"{len(search_res.ids)} / {k_val}")

        # Display Results Table
        st.markdown("#### Retrieved Nearest Neighbors")
        if len(search_res.ids) == 0:
            st.warning("No eligible candidates found matching the query and filter criteria.")
        else:
            table_rows = []
            for rank_i, (dist_val, id_val) in enumerate(zip(search_res.distances, search_res.ids), start=1):
                is_exact_match = id_val in gt_ids_for_q
                table_rows.append({
                    "Rank": rank_i,
                    "External ID": int(id_val),
                    "Squared L2 Distance (||q - x||²)": f"{dist_val:.5f}",
                    "Exact Top-k Status": "✅ Exact Match" if is_exact_match else "⚡ Approximation",
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
                "💡 **Performance Insight:** On small synthetic datasets ($N \\approx 120$), Exact BLAS can report "
                "lower query latency than HNSW or IVF because BLAS leverages hardware-optimized C routines over "
                "a tiny matrix, while Python graph traversals incur interpreter loop overhead. As dataset size "
                "grows to $N=100{,}000+$, HNSW's logarithmic $O(\\log N)$ scaling and IVF's clustering dramatically "
                "outperform $O(N)$ exhaustive scans."
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


if __name__ == "__main__":
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", __file__]
    sys.exit(stcli.main())
