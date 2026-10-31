"""Indexora Interactive Index Lab — Production Developer Tool UI.

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

# Ensure 'src' package directory is available across cloud and local execution
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

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
try:
    from indexora.semantic import (
        DEFAULT_EMBEDDING_MODEL,
        Document,
        SemanticSearchEngine,
        SemanticSearchResult,
        build_semantic_engine,
        is_sentence_transformers_available,
        load_demo_corpus,
    )
except ImportError:
    DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    Document = Any  # type: ignore[misc,assignment]
    SemanticSearchEngine = Any  # type: ignore[misc,assignment]
    SemanticSearchResult = Any  # type: ignore[misc,assignment]
    build_semantic_engine = None  # type: ignore[assignment]
    is_sentence_transformers_available = lambda: False  # type: ignore[assignment]
    load_demo_corpus = lambda: []  # type: ignore[assignment]


def _apply_custom_styling() -> None:
    """Inject clean scientific computing platform design system with refined light blue palette."""
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Montserrat:wght@600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

        <script>
        // Ensure Streamlit localStorage theme is forced to light so browser doesn't apply cached dark CSS
        try {
            if (window.localStorage && window.localStorage.getItem("stActiveTheme") !== "light") {
                window.localStorage.setItem("stActiveTheme", "light");
            }
        } catch (e) {}
        </script>

        <style>
        /* =========================================================================
           INDEXORA DESIGN SYSTEM: SCIENTIFIC COMPUTING LIGHT PALETTE
           Canvas Background: #F2F8FF (very light blue, almost white)
           Secondary Surface / Sidebar / Cards: #E5F1FC (soft pale blue)
           Elevated Surface: #FFFFFF (crisp white)
           Primary Accent: #3986D9 (clean medium blue)
           Dark / Headings / Text: #303943 (graphite)
           Secondary / Muted Text: #687583 (muted graphite)
           Borders: #D5E2EF (subtle blue-grey)
        ========================================================================= */

        /* 1. Global CSS Variables & Canvas Reset */
        :root, html, body, .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > .main,
        section.main,
        .main,
        .block-container,
        [data-testid="stMainBlockContainer"],
        [data-testid="stBottom"],
        div[class*="stAppViewContainer"],
        section[class*="main"] {
            --primary-color: #3986D9 !important;
            --background-color: #F2F8FF !important;
            --secondary-background-color: #E5F1FC !important;
            --text-color: #303943 !important;
            --font: 'Inter', sans-serif !important;
            background-color: #F2F8FF !important;
            background: #F2F8FF !important;
            color: #303943 !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            letter-spacing: -0.01em;
        }

        .stApp {
            max-width: 100vw;
            overflow-x: hidden;
        }

        /* Ambient soft blue gradient top wash */
        .stApp::before {
            content: "";
            position: fixed;
            top: 0; left: 0; right: 0; height: 280px;
            background: radial-gradient(circle 900px at 50% -140px, rgba(57, 134, 217, 0.08), transparent 70%);
            pointer-events: none;
            z-index: 0;
        }

        /* Hide default Streamlit footer & watermark */
        #MainMenu, footer {
            display: none !important;
        }
        [data-testid="stHeader"], [data-testid="stToolbar"] {
            background-color: rgba(242, 248, 255, 0.95) !important;
            background: rgba(242, 248, 255, 0.95) !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            border-bottom: 1px solid #D5E2EF !important;
        }

        /* 2. Sidebar Shell & Sidebar Components */
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div,
        [data-testid="stSidebarContent"],
        [data-testid="stSidebarUserContent"],
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div {
            background-color: #E5F1FC !important;
            background: #E5F1FC !important;
            border-right: 1px solid #D5E2EF !important;
            color: #303943 !important;
        }
        [data-testid="stSidebar"] hr {
            border-color: #D5E2EF !important;
            margin: 1.25rem 0 !important;
        }
        .sidebar-section-title {
            font-size: 0.72rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.08em !important;
            text-transform: uppercase !important;
            color: #687583 !important;
            margin-bottom: 0.85rem !important;
        }

        /* Sidebar collapse / expand buttons */
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stExpandSidebarButton"] {
            z-index: 100 !important;
        }
        [data-testid="stSidebarCollapseButton"] button,
        [data-testid="stExpandSidebarButton"] button {
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 2.25rem !important;
            height: 2.25rem !important;
            border-radius: 8px !important;
            background: #FFFFFF !important;
            border: 1px solid #D5E2EF !important;
            color: #303943 !important;
            cursor: pointer !important;
            transition: all 0.15s ease !important;
            pointer-events: auto !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
        }
        [data-testid="stSidebarCollapseButton"] button:hover,
        [data-testid="stExpandSidebarButton"] button:hover {
            color: #3986D9 !important;
            background: #E5F1FC !important;
            border-color: #3986D9 !important;
            box-shadow: 0 2px 6px rgba(57, 134, 217, 0.15) !important;
        }

        /* 3. Typography & Material Icons Preservation */
        html, body, .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown div, label,
        [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            color: #303943 !important;
        }
        /* CRITICAL: Preserve Material Symbols & Icons font ligatures so icons render cleanly */
        [data-testid*="stIcon"],
        [data-testid*="Icon"],
        [data-testid="stExpanderToggleIcon"],
        [data-testid="stExpanderToggleIcon"] *,
        .material-symbols-rounded,
        .material-symbols-sharp,
        .material-symbols-outlined,
        .material-icons {
            font-family: 'Material Symbols Rounded', 'Material Icons', 'Material Symbols Outlined', sans-serif !important;
            font-style: normal !important;
            font-weight: normal !important;
            text-transform: none !important;
            letter-spacing: normal !important;
            word-wrap: normal !important;
            white-space: nowrap !important;
            direction: ltr !important;
            -webkit-font-smoothing: antialiased !important;
            text-rendering: optimizeLegibility !important;
        }
        h1, h2, h3, h4, h5, h6, .idx-title, .idx-card-title {
            font-family: 'Montserrat', sans-serif !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em !important;
            color: #303943 !important;
        }
        h1, .idx-title {
            font-weight: 700 !important;
            letter-spacing: -0.03em !important;
        }
        strong, b {
            color: #1E293B !important;
            font-weight: 600 !important;
        }
        .stMarkdown p {
            color: #303943 !important;
            line-height: 1.6 !important;
        }
        .stMarkdown small, .stCaption, [data-testid="stCaptionContainer"] {
            color: #687583 !important;
            font-size: 0.82rem !important;
        }

        /* Monospace Elements */
        code, pre, .font-mono, [data-testid="stDataFrame"], .stCode, kbd {
            font-family: 'JetBrains Mono', ui-monospace, monospace !important;
        }
        code {
            background: #E5F1FC !important;
            color: #1D4ED8 !important;
            border: 1px solid #D5E2EF !important;
            padding: 0.12rem 0.38rem !important;
            border-radius: 4px !important;
            font-size: 0.84em !important;
            white-space: nowrap !important;
        }

        /* 4. Product Header Banner */
        .idx-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 0 1.25rem 0;
            border-bottom: 1px solid #D5E2EF;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
            gap: 1rem;
        }
        .idx-brand {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }
        .idx-logo-badge {
            width: 38px;
            height: 38px;
            background: linear-gradient(135deg, #3986D9 0%, #2563EB 100%) !important;
            border-radius: 9px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            color: #FFFFFF !important;
            box-shadow: 0 2px 10px rgba(57, 134, 217, 0.28);
        }
        .idx-title {
            font-size: 1.4rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: #303943 !important;
            margin: 0;
            line-height: 1.2;
        }
        .idx-tagline {
            font-size: 0.82rem;
            color: #687583 !important;
            margin: 0;
            line-height: 1.3;
        }
        .idx-status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.35rem 0.75rem;
            background: #ECFDF5 !important;
            border: 1px solid #A7F3D0 !important;
            border-radius: 9999px;
            font-size: 0.76rem;
            font-weight: 600;
            color: #047857 !important;
            font-family: 'JetBrains Mono', monospace;
        }
        .idx-status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #10B981 !important;
            box-shadow: 0 0 6px #10B981;
        }
        .idx-status-pill-amber {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.35rem 0.75rem;
            background: #FFFBEB !important;
            border: 1px solid #FDE68A !important;
            border-radius: 9999px;
            font-size: 0.76rem;
            font-weight: 600;
            color: #B45309 !important;
            font-family: 'JetBrains Mono', monospace;
        }
        .idx-status-dot-amber {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #F59E0B !important;
            box-shadow: 0 0 6px #F59E0B;
        }

        /* 5. Navigation Tabs - Scientific Light Pill Design */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem !important;
            border-bottom: 1px solid #D5E2EF !important;
            padding: 0 0 10px 0 !important;
            background: transparent !important;
            margin-bottom: 1.75rem !important;
            overflow-x: auto !important;
            flex-wrap: nowrap !important;
        }
        .stTabs [data-baseweb="tab-highlight"],
        .stTabs [data-baseweb="tab-border"] {
            display: none !important;
        }
        .stTabs [data-baseweb="tab"] {
            font-family: 'Inter', sans-serif !important;
            font-size: 0.88rem !important;
            font-weight: 500 !important;
            padding: 0.55rem 1.25rem !important;
            border-radius: 8px !important;
            color: #687583 !important;
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
            border: 1px solid #D5E2EF !important;
            transition: all 0.15s ease-in-out !important;
            white-space: nowrap !important;
            flex-shrink: 0 !important;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #303943 !important;
            background-color: #E5F1FC !important;
            background: #E5F1FC !important;
            border-color: #CBD5E1 !important;
        }
        .stTabs [aria-selected="true"] {
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
            color: #3986D9 !important;
            font-weight: 600 !important;
            border: 1px solid #3986D9 !important;
            border-bottom: 2px solid #3986D9 !important;
            box-shadow: 0 2px 6px rgba(57, 134, 217, 0.12) !important;
        }

        /* 6. Technical Metric Cards */
        [data-testid="stMetric"] {
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
            border: 1px solid #D5E2EF !important;
            border-radius: 10px !important;
            padding: 1rem 1.25rem !important;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.03) !important;
        }
        [data-testid="stMetricLabel"] p, [data-testid="stMetricLabel"] {
            font-family: 'Inter', sans-serif !important;
            font-size: 0.74rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            color: #687583 !important;
        }
        [data-testid="stMetricValue"] div, [data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 1.35rem !important;
            font-weight: 700 !important;
            color: #303943 !important;
            letter-spacing: -0.02em !important;
        }

        /* 7. Structured Technical Cards */
        .idx-card {
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
            border: 1px solid #D5E2EF !important;
            border-radius: 12px !important;
            padding: 1.35rem 1.5rem !important;
            margin-bottom: 1.25rem !important;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.03) !important;
            transition: all 0.2s ease !important;
        }
        .idx-card:hover {
            border-color: #3986D9 !important;
            box-shadow: 0 4px 16px rgba(57, 134, 217, 0.09) !important;
        }
        .idx-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.5rem;
        }
        .idx-card-title {
            font-size: 0.98rem !important;
            font-weight: 600 !important;
            color: #303943 !important;
            margin: 0 !important;
        }
        .idx-card-desc {
            font-size: 0.85rem !important;
            color: #687583 !important;
            line-height: 1.6 !important;
            margin: 0 0 0.75rem 0 !important;
        }
        .idx-card-meta {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: #495460 !important;
            display: flex;
            align-items: center;
            gap: 1rem;
            flex-wrap: wrap;
        }

        /* Pipeline Flow Grid (Multi-Stage Architecture Map) */
        .idx-pipeline-grid {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 2rem;
            overflow-x: auto;
            padding: 0.3rem 0;
        }
        .idx-step-card {
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
            border: 1px solid #D5E2EF !important;
            border-radius: 8px !important;
            padding: 0.8rem 1.1rem !important;
            min-width: 155px !important;
            flex: 1 !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
        }
        .idx-step-card-active {
            border-color: #3986D9 !important;
            background-color: #E5F1FC !important;
            background: #E5F1FC !important;
        }
        .idx-step-num {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            font-weight: 700;
            color: #3986D9 !important;
            letter-spacing: 0.06em;
            margin-bottom: 0.2rem;
        }
        .idx-step-name {
            font-family: 'Montserrat', sans-serif;
            font-size: 0.85rem;
            font-weight: 600;
            color: #303943 !important;
            white-space: nowrap;
            margin-bottom: 0.15rem;
        }
        .idx-step-tech {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            color: #687583 !important;
            white-space: nowrap;
        }
        .idx-step-arrow {
            color: #94A3B8 !important;
            font-size: 1.1rem;
            font-weight: 600;
            flex-shrink: 0;
        }

        /* 8. Sliders */
        [data-baseweb="slider"] div[class*="StyledTrack"],
        [data-baseweb="slider"] div[class*="track"] {
            background-color: #D5E2EF !important;
            background: #D5E2EF !important;
        }
        [data-baseweb="slider"] [role="slider"] {
            background-color: #3986D9 !important;
            background: #3986D9 !important;
            border: 2px solid #FFFFFF !important;
            box-shadow: 0 1px 4px rgba(57, 134, 217, 0.4) !important;
        }
        [data-baseweb="slider"] div[data-testid="stThumbValue"] {
            color: #3986D9 !important;
            font-weight: 700 !important;
            font-family: 'JetBrains Mono', monospace !important;
        }
        [data-testid="stSlider"] [data-testid="stWidgetLabel"] p {
            color: #303943 !important;
            font-weight: 600 !important;
        }
        [data-testid="stTickBar"] span {
            color: #687583 !important;
        }

        /* 9. Modern Buttons */
        .stButton > button,
        [data-testid="baseButton-secondary"],
        [data-testid="baseButton-primary"] {
            font-family: 'Inter', sans-serif !important;
            font-weight: 500 !important;
            font-size: 0.85rem !important;
            border-radius: 6px !important;
            letter-spacing: -0.01em !important;
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
            color: #303943 !important;
            border: 1px solid #D5E2EF !important;
            padding: 0.45rem 0.95rem !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
            transition: all 0.15s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }
        .stButton > button:hover,
        [data-testid="baseButton-secondary"]:hover {
            background-color: #E5F1FC !important;
            background: #E5F1FC !important;
            border-color: #3986D9 !important;
            color: #3986D9 !important;
            transform: translateY(-1px);
            box-shadow: 0 2px 6px rgba(57, 134, 217, 0.12) !important;
        }
        .stButton > button:active {
            transform: translateY(0);
        }

        /* 10. Form Inputs & Number Inputs */
        [data-testid="stNumberInput"] input,
        .stTextInput input,
        .stSelectbox [data-baseweb="select"],
        .stSelectbox [data-baseweb="select"] > div,
        [data-baseweb="input"],
        [data-baseweb="input"] input,
        [data-baseweb="base-input"] {
            font-family: 'Inter', sans-serif !important;
            border-radius: 6px !important;
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
            border: 1px solid #D5E2EF !important;
            color: #303943 !important;
            font-size: 0.86rem !important;
        }
        [data-testid="stNumberInput"] button,
        [data-testid="stNumberInputStepDown"],
        [data-testid="stNumberInputStepUp"] {
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
            color: #303943 !important;
            border: 1px solid #D5E2EF !important;
        }
        [data-testid="stNumberInput"] button:hover {
            background-color: #E5F1FC !important;
            color: #3986D9 !important;
        }
        .stTextInput input:focus, .stSelectbox [data-baseweb="select"]:focus, .stNumberInput input:focus {
            border-color: #3986D9 !important;
            box-shadow: 0 0 0 2px rgba(57, 134, 217, 0.2) !important;
        }

        /* 11. Badges & Status Pills */
        .badge-indigo, .badge-blue {
            display: inline-flex;
            align-items: center;
            background: #E0EEFD !important;
            color: #1D4ED8 !important;
            border: 1px solid #BFDBFE !important;
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
            font-size: 0.74rem;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }
        .badge-emerald {
            display: inline-flex;
            align-items: center;
            background: #ECFDF5 !important;
            color: #047857 !important;
            border: 1px solid #A7F3D0 !important;
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
            font-size: 0.74rem;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }
        .badge-amber {
            display: inline-flex;
            align-items: center;
            background: #FFFBEB !important;
            color: #B45309 !important;
            border: 1px solid #FDE68A !important;
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
            font-size: 0.74rem;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }

        /* Callout / Disclaimer */
        .idx-callout {
            background-color: #E5F1FC !important;
            background: #E5F1FC !important;
            border-left: 3px solid #3986D9 !important;
            padding: 0.85rem 1.15rem !important;
            font-size: 0.83rem !important;
            color: #303943 !important;
            border-radius: 0 8px 8px 0 !important;
            margin-bottom: 1.15rem !important;
            line-height: 1.6 !important;
        }
        .idx-callout strong {
            color: #1E293B !important;
        }

        /* Dataframe styling */
        [data-testid="stDataFrame"] {
            border: 1px solid #D5E2EF !important;
            border-radius: 8px !important;
            overflow: hidden !important;
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
        }

        /* Expanders */
        [data-testid="stExpander"] {
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
            border: 1px solid #D5E2EF !important;
            border-radius: 8px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
        }
        [data-testid="stExpander"] summary {
            font-weight: 600 !important;
            color: #303943 !important;
        }
        [data-testid="stExpanderDetails"] {
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
            color: #303943 !important;
        }

        /* Engineering Footer */
        .idx-footer {
            margin-top: 3.5rem;
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
            border-top: 1px solid #D5E2EF;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.80rem;
            color: #687583;
            flex-wrap: wrap;
            gap: 1rem;
        }
        .idx-footer-brand {
            font-weight: 600;
            color: #303943 !important;
        }
        .idx-footer a {
            color: #3986D9 !important;
            text-decoration: none !important;
            font-weight: 500;
        }
        .idx-footer a:hover {
            text-decoration: underline !important;
        }

        /* Responsive Mobile Layout */
        @media (max-width: 768px) {
            .idx-header {
                flex-direction: column;
                align-items: flex-start;
                gap: 0.75rem;
            }
            .idx-pipeline-grid {
                flex-direction: column;
                align-items: stretch;
            }
            .idx-step-arrow {
                transform: rotate(90deg);
                text-align: center;
                margin: -0.25rem 0;
            }
            .idx-card {
                padding: 1rem !important;
            }
            .idx-footer {
                flex-direction: column;
                align-items: flex-start;
                gap: 0.5rem;
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
    """Retrieve index from cache or build it if build-time parameters or dataset changed."""
    if index_type == "Exact Flat (Reference)":
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

    return cache.get_or_build(index_type, ds, build_params, builder)


def main() -> None:
    st.set_page_config(
        page_title="Indexora — Vector Search Engine",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_custom_styling()
    _init_session_state()

    # -------------------------------------------------------------------------
    # Header Banner (Linear/Vercel App Shell Header)
    # -------------------------------------------------------------------------
    session_val = st.session_state.get("lab_session")
    active_cnt = getattr(session_val, "active_count", 0) if session_val else 0
    if session_val is not None and active_cnt > 0:
        status_label = f"Demo Session Ready ({active_cnt} active)"
        status_pill_cls = "idx-status-pill"
        status_dot_cls = "idx-status-dot"
    else:
        status_label = "Initializing Session"
        status_pill_cls = "idx-status-pill-amber"
        status_dot_cls = "idx-status-dot-amber"

    st.markdown(
        f"""
        <div class="idx-header">
            <div class="idx-brand">
                <div class="idx-logo-badge">⚡</div>
                <div>
                    <h1 class="idx-title">Indexora</h1>
                    <p class="idx-tagline">Vector Search Engine & Interactive Lab</p>
                </div>
            </div>
            <div style="display:flex; align-items:center; gap: 0.75rem;">
                <span class="{status_pill_cls}">
                    <span class="{status_dot_cls}"></span> {status_label}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # Sidebar: Lab Configuration & Dataset Status
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown('<div class="sidebar-section-title">Environment & Dataset</div>', unsafe_allow_html=True)

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
            st.success("Dataset regenerated!")

        st.markdown("---")
        st.markdown('<div class="sidebar-section-title">Active Dataset State</div>', unsafe_allow_html=True)
        ds_curr: LabDataset = st.session_state.dataset
        st.markdown(
            f"""
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #687583; line-height: 1.7; background: #FFFFFF; padding: 0.85rem; border-radius: 8px; border: 1px solid #D5E2EF; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                <div>Base Vectors: <span style="color:#303943; font-weight:600;">{ds_curr.N}</span></div>
                <div>Queries: <span style="color:#303943; font-weight:600;">{ds_curr.Q}</span></div>
                <div>Dimension: <span style="color:#303943; font-weight:600;">{ds_curr.d}-D float32</span></div>
                <div>Clusters: <span style="color:#303943; font-weight:600;">{ds_curr.n_clusters}</span></div>
                <div>ID Range: <span style="color:#303943; font-weight:600;">[{ds_curr.base_ids[0]} ... {ds_curr.base_ids[-1]}]</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="idx-callout" style="margin-top: 1rem;">
                <strong>Synthetic Demo Scope:</strong> Uses geometric coordinates in ℝᵈ for algorithm validation. Real text retrieval is active in Tab 5.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # Main Navigation: 5 Primary Product Tabs
    # -------------------------------------------------------------------------
    tab_overview, tab_playground, tab_benchmark, tab_mutations, tab_semantic = st.tabs(
        [
            "Overview",
            "Search Lab",
            "Benchmarks",
            "Mutations",
            "Semantic Search",
        ]
    )

    ds: LabDataset = st.session_state.dataset

    # =========================================================================
    # Tab 1: Overview & Concepts
    # =========================================================================
    with tab_overview:
        st.markdown("### Vector Index Architectures")
        st.markdown(
            "Indexora implements foundational nearest-neighbor retrieval algorithms directly in Python and NumPy. "
            "Examine how each architecture partitions the search space and balances candidate evaluation cost with recall guarantees:"
        )

        # Visual Retrieval Pipeline Flowchart
        st.markdown(
            """
            <div class="idx-pipeline-grid">
                <div class="idx-step-card">
                    <div class="idx-step-num">STAGE 01</div>
                    <div class="idx-step-name">Query Vector</div>
                    <div class="idx-step-tech">q ∈ ℝ<sup>d</sup> (float32)</div>
                </div>
                <div class="idx-step-arrow">→</div>
                <div class="idx-step-card">
                    <div class="idx-step-num">STAGE 02</div>
                    <div class="idx-step-name">Partitioning</div>
                    <div class="idx-step-tech">Voronoi / Skip-Graph</div>
                </div>
                <div class="idx-step-arrow">→</div>
                <div class="idx-step-card">
                    <div class="idx-step-num">STAGE 03</div>
                    <div class="idx-step-name">Candidate Scoring</div>
                    <div class="idx-step-tech">BLAS GEMM / ADC LUT</div>
                </div>
                <div class="idx-step-arrow">→</div>
                <div class="idx-step-card">
                    <div class="idx-step-num">STAGE 04</div>
                    <div class="idx-step-name">Top-k Selection</div>
                    <div class="idx-step-tech">Partial Selection + (Dist, ID) Sort</div>
                </div>
                <div class="idx-step-arrow">→</div>
                <div class="idx-step-card idx-step-card-active">
                    <div class="idx-step-num">STAGE 05</div>
                    <div class="idx-step-name">Search Result</div>
                    <div class="idx-step-tech">SearchResult(ids, dists)</div>
                </div>
            </div>
            <div class="idx-callout" style="margin-top: 1.25rem; margin-bottom: 1.5rem;">
                <strong>Top-k Selection Architecture:</strong> Indexora avoids an expensive full <code>O(N log N)</code> array sort by executing <code>np.argpartition</code> to perform <code>O(N)</code> average partial selection for the top-<code>k</code> candidates, followed by <code>np.lexsort</code> on <code>(distance, external_id)</code> pairs to guarantee strictly deterministic tie-breaking.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                <div class="idx-card">
                    <div class="idx-card-header">
                        <div class="idx-card-title">1. Exact Flat / Exact BLAS</div>
                        <span class="badge-emerald">100% Recall</span>
                    </div>
                    <div class="idx-card-desc">
                        Brute-force exhaustive search evaluating every vector in the dataset. Serves as the mathematical correctness oracle.
                    </div>
                    <div class="idx-card-meta">
                        <span>Complexity: <code>O(N · d)</code></span>
                        <span>Kernel: <code>GEMM d²(q, x)</code></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="idx-card">
                    <div class="idx-card-header">
                        <div class="idx-card-title">2. IVF-Flat (Inverted File)</div>
                        <span class="badge-indigo">Voronoi Partitioning</span>
                    </div>
                    <div class="idx-card-desc">
                        Partitions database vectors into <code>nlist</code> centroids via <em>K</em>-means++. At query time, evaluates only the closest <code>nprobe</code> inverted posting lists.
                    </div>
                    <div class="idx-card-meta">
                        <span>Complexity: <code>O(nprobe · N / nlist · d)</code></span>
                        <span>Trade-off: High Recall / Low DC</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="idx-card">
                    <div class="idx-card-header">
                        <div class="idx-card-title">3. Exact Reference Kernel</div>
                        <span class="badge-amber">Precision Oracle</span>
                    </div>
                    <div class="idx-card-desc">
                        Direct coordinate-by-coordinate differencing with float64 accumulation to guarantee numerical stability against catastrophic cancellation.
                    </div>
                    <div class="idx-card-meta">
                        <span>Precision: <code>float64 accumulation</code></span>
                        <span>Validation Oracle</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
                <div class="idx-card">
                    <div class="idx-card-header">
                        <div class="idx-card-title">4. IVF-PQ (Product Quantization)</div>
                        <span class="badge-indigo">15x–64x Compression</span>
                    </div>
                    <div class="idx-card-desc">
                        Decomposes <em>d</em>-dimensional space into <em>M</em> orthogonal subspaces. Quantizes residuals into 8-bit codes and evaluates distances via Asymmetric Distance Computation (ADC) lookup tables.
                    </div>
                    <div class="idx-card-meta">
                        <span>Complexity: <code>O(N_cand · M) LUT</code></span>
                        <span>Footprint: <code>M bytes/vec</code></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="idx-card">
                    <div class="idx-card-header">
                        <div class="idx-card-title">5. HNSW (Hierarchical Small World)</div>
                        <span class="badge-indigo">Graph ANN</span>
                    </div>
                    <div class="idx-card-desc">
                        Multi-layer proximity graph with exponential level decay and Algorithm 4 diversity neighbor selection. Navigates logarithmic expressways on upper layers and performs greedy beam search at layer 0. Traversal latency is empirical sublinear under fixed search parameters. On smaller datasets, Exact BLAS may achieve lower wall-clock latency due to Python interpreter overhead in graph traversal.
                    </div>
                    <div class="idx-card-meta">
                        <span>Traversal: <code>Empirical Sublinear (fixed M, ef)</code></span>
                        <span>Key Factors: <code>M, ef_search & graph degree</code></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("#### Live Dataset Vector Distribution Sample")
        df_preview = pd.DataFrame(
            ds.base[:8],
            columns=[f"d{i}" for i in range(ds.d)],
            index=[f"id_{int(x)}" for x in ds.base_ids[:8]],
        )
        st.dataframe(df_preview, use_container_width=True)

    # =========================================================================
    # Tab 2: Search Playground
    # =========================================================================
    with tab_playground:
        st.markdown("### Interactive Query Search")
        st.markdown("Execute nearest-neighbor retrieval on live index instances with real-time ground truth validation.")

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
            c_p1, c_p2 = st.columns([1, 2])
            with c_p1:
                nprobe_val = st.slider("nprobe (Probed Centroids)", min_value=1, max_value=4, value=2)
            with c_p2:
                st.caption(f"Index built with `nlist=4`. Probing {nprobe_val}/4 Voronoi cells per query.")
        elif selected_index_name == "HNSW":
            c_p1, c_p2 = st.columns([1, 2])
            with c_p1:
                ef_search_val = st.slider("ef_search (Beam Exploration Depth)", min_value=4, max_value=64, value=32, step=4)
            with c_p2:
                st.caption(f"Layer 0 search explores candidate beam of size `ef={ef_search_val}`.")

        # Optional ID filter
        with st.expander("Filter Eligibility Settings (filter_ids)"):
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

        # Retrieve or Build Index from Cache
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

        build_note = f"⚡ Reused cached index ({selected_index_name})" if was_cached else f"🔨 Built index in {build_time_ms:.2f} ms"
        st.markdown(f'<div style="font-size:0.78rem; color:#64748b; margin-bottom:0.75rem; font-family:\'JetBrains Mono\';">{build_note}</div>', unsafe_allow_html=True)

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
            st.metric("Filtered Exact Match", match_label)
        with m4:
            st.metric("Retrieved Candidates", f"{len(search_res.ids)} / {k_val}")

        # Display Results Table
        st.markdown("#### Retrieved Nearest Neighbors")
        if len(search_res.ids) == 0:
            if num_eligible_live == 0:
                st.info("Filter restriction resulted in 0 eligible database vectors. Empty result correctly matches filtered ground truth.")
            else:
                st.warning("No eligible candidates retrieved matching the query and filter criteria.")
        else:
            table_rows = []
            for rank_i, (dist_val, id_val) in enumerate(zip(search_res.distances, search_res.ids), start=1):
                is_exact_match = id_val in eligible_gt_ids
                table_rows.append({
                    "Rank": rank_i,
                    "External ID": int(id_val),
                    "Squared Distance (d²)": f"{dist_val:.5f}",
                    "Ground Truth Status": "Exact Match" if is_exact_match else "Approximation",
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        # 2D PCA Visualization (Scientific Light Palette)
        st.markdown("#### 2D Spatial Projection (PCA)")
        st.caption("Dimensionality reduction via deterministic SVD PCA. Illustrates candidate proximity and cluster assignment.")

        fig, ax = plt.subplots(figsize=(9, 4.4), facecolor="#F2F8FF")
        ax.set_facecolor("#FFFFFF")

        # 1. Base vectors
        base_proj = ds.project_2d(ds.base)
        ax.scatter(
            base_proj[:, 0],
            base_proj[:, 1],
            c=ds.cluster_labels,
            cmap="viridis",
            alpha=0.65,
            s=44,
            edgecolors="none",
            label="Database Vectors",
        )

        # 2. Query point
        q_proj = ds.project_2d(query_vec)
        ax.scatter(
            q_proj[0],
            q_proj[1],
            color="#DC2626",
            edgecolors="#FFFFFF",
            linewidths=1.5,
            marker="*",
            s=260,
            zorder=10,
            label="Query Vector",
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
                    edgecolors="#3986D9",
                    s=180,
                    linewidths=2.2,
                    zorder=9,
                    label="Retrieved Neighbors",
                )
                for rank_i, r_idx in enumerate(retrieved_indices, start=1):
                    ax.annotate(
                        f"#{rank_i}",
                        (base_proj[r_idx, 0] + 0.05, base_proj[r_idx, 1] + 0.05),
                        fontsize=8.5,
                        fontweight="bold",
                        color="#1D4ED8",
                    )

        ax.set_title(f"2D Projection: {selected_index_name} (Top-{k_val})", fontsize=10.5, fontweight="600", color="#303943", pad=10)
        ax.set_xlabel("PC 1", fontsize=9, color="#495460")
        ax.set_ylabel("PC 2", fontsize=9, color="#495460")
        ax.tick_params(colors="#687583", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#D5E2EF")
        ax.grid(True, linestyle="--", alpha=0.6, color="#E2E8F0")
        legend = ax.legend(loc="upper right", fontsize=8, facecolor="#FFFFFF", edgecolor="#D5E2EF")
        for text in legend.get_texts():
            text.set_color("#303943")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # =========================================================================
    # Tab 3: Accuracy & Performance Lab
    # =========================================================================
    with tab_benchmark:
        subtab_lab, subtab_sift = st.tabs(
            [
                "🧪 Interactive Lab Sweep (Synthetic N=120)",
                "📊 SIFT-100K Benchmark Evidence (Recorded Scale Run)",
            ]
        )

        with subtab_lab:
            st.markdown("### Accuracy vs. Speed Benchmark Sweep (Interactive Synthetic)")
            st.markdown(
                "Execute an empirical sweep comparing all 5 index implementations on identical queries with "
                "deterministic ground-truth verification."
            )

        b_k = st.slider("Benchmark Top-k", min_value=1, max_value=min(ds.N, 20), value=5, key="bench_k")

        if st.button("🚀 Run Comparative Benchmark", use_container_width=True):
            with st.spinner("Benchmarking 5 index architectures..."):
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

            # Polished Scientific Light Charts
            c_c1, c_c2 = st.columns(2)
            with c_c1:
                st.markdown("#### Recall@k vs. Median Latency")
                fig1, ax1 = plt.subplots(figsize=(6, 3.6), facecolor="#F2F8FF")
                ax1.set_facecolor("#FFFFFF")

                fam_meta = {
                    "Exact": {"color": "#16A34A", "marker": "o", "label": "Exact (Flat / BLAS)"},
                    "IVF-Flat": {"color": "#0284C7", "marker": "s", "label": "IVF-Flat"},
                    "IVF-PQ": {"color": "#D97706", "marker": "^", "label": "IVF-PQ"},
                    "HNSW": {"color": "#7C3AED", "marker": "D", "label": "HNSW"},
                }

                rendered_fams: set[str] = set()
                for r in bench_data:
                    if "Exact" in r.index_name:
                        fam = "Exact"
                    elif "IVF-Flat" in r.index_name:
                        fam = "IVF-Flat"
                    elif "IVF-PQ" in r.index_name:
                        fam = "IVF-PQ"
                    else:
                        fam = "HNSW"

                    cfg = fam_meta[fam]
                    plot_label = cfg["label"] if fam not in rendered_fams else None
                    rendered_fams.add(fam)

                    ax1.scatter(
                        r.median_latency_ms,
                        r.recall_at_k,
                        color=cfg["color"],
                        marker=cfg["marker"],
                        s=95,
                        alpha=0.9,
                        edgecolors="#FFFFFF",
                        linewidths=1.2,
                        label=plot_label,
                        zorder=5,
                    )

                ax1.set_xlabel("Median Query Latency (ms)", fontsize=8.5, color="#495460")
                ax1.set_ylabel("Recall@k", fontsize=8.5, color="#495460")
                ax1.set_ylim(-0.05, 1.15)
                ax1.tick_params(colors="#687583", labelsize=7.5)
                for spine in ax1.spines.values():
                    spine.set_color("#D5E2EF")
                ax1.grid(True, linestyle="--", alpha=0.6, color="#E2E8F0")
                ax1.legend(loc="lower right", fontsize=7.5, facecolor="#FFFFFF", edgecolor="#D5E2EF", labelcolor="#303943")
                fig1.tight_layout()
                st.pyplot(fig1)
                plt.close(fig1)

            with c_c2:
                st.markdown("#### Candidate Evaluations per Query")
                fig2, ax2 = plt.subplots(figsize=(6, 3.6), facecolor="#F2F8FF")
                ax2.set_facecolor("#FFFFFF")
                names = [r.index_name for r in bench_data if r.distance_computations_per_query is not None]
                dcs = [r.distance_computations_per_query for r in bench_data if r.distance_computations_per_query is not None]

                bar_colors = []
                for n in names:
                    if "Exact" in n:
                        bar_colors.append("#16A34A")
                    elif "IVF-Flat" in n:
                        bar_colors.append("#0284C7")
                    elif "IVF-PQ" in n:
                        bar_colors.append("#D97706")
                    else:
                        bar_colors.append("#7C3AED")

                y_pos = np.arange(len(names))
                bars = ax2.barh(y_pos, dcs, color=bar_colors, alpha=0.9, height=0.60)
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(names, fontsize=7.5, color="#303943")
                ax2.invert_yaxis()
                ax2.set_xlabel("Mean Distance Operations / Query", fontsize=8.5, color="#495460")
                ax2.tick_params(colors="#687583", labelsize=7.5)
                for spine in ax2.spines.values():
                    spine.set_color("#D5E2EF")
                ax2.grid(True, axis="x", linestyle="--", alpha=0.6, color="#E2E8F0")
                fig2.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)

            st.markdown(
                """
                <div class="idx-callout">
                    <strong>Systems Insight:</strong> On small synthetic datasets (e.g. N=120), BLAS exact scan can report lower wall-clock latency than HNSW or IVF because Level-3 GEMM runs in optimized C/Fortran libraries, whereas Python-level graph traversals incur bytecode interpreter overhead. Approximate methods prove their throughput advantage at scale (N ≥ 50,000).
                </div>
                """,
                unsafe_allow_html=True,
            )

        with subtab_sift:
            st.markdown("### SIFT-100K Canonical Benchmark Evidence")
            st.markdown(
                "Empirical retrieval performance, latency distributions, and throughput measured on the "
                "standard Texmex SIFT-100k benchmark ($N=100,000$ base vectors, $Q=1,000$ queries, $d=128$, squared L2)."
            )

            st.markdown(
                """
                <div class="idx-callout" style="margin-bottom: 1.25rem;">
                    <strong>Benchmarking Provenance Notice:</strong>
                    <ul style="margin: 0.35rem 0 0 1.2rem; padding: 0;">
                        <li><strong>Interactive Index Lab (Subtab 1):</strong> Executes live in-browser on small synthetic clustered datasets (<code>N=120, d=8</code>). Designed for visual intuition and algorithmic debugging; not representative of large-scale throughput.</li>
                        <li><strong>SIFT-100K Benchmark (Subtab 2):</strong> Visualizes precomputed, cryptographically verified benchmark artifacts recorded under strictly controlled hardware conditions. <em>The Streamlit application does not execute SIFT-100K in real time.</em></li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Highlight KPI Metrics
            kpi1, kpi2, kpi3 = st.columns(3)
            with kpi1:
                st.metric(
                    "Max Speedup vs. BLAS",
                    "78.7× QPS",
                    delta="IVF-Flat (nlist=100, nprobe=1): 1,534.6 QPS vs 19.5 QPS",
                )
            with kpi2:
                st.metric(
                    "Balanced Operating Point",
                    "91.3% Recall@10",
                    delta="IVF-Flat (nlist=100, nprobe=5): 4.53ms p50 (17.9× distance reduction)",
                )
            with kpi3:
                st.metric(
                    "Near-Lossless Retrieval",
                    "99.8% Recall@10",
                    delta="IVF-Flat (nlist=100, nprobe=20): 17.75ms p50 (21.5% distance scan)",
                )

            st.markdown("---")

            # Environment & Provenance Expander
            with st.expander("Hardware, Software Environment & Provenance Details", expanded=False):
                env_col1, env_col2 = st.columns(2)
                with env_col1:
                    st.markdown(
                        """
                        **Hardware & Runtime Environment:**
                        - **Platform:** Windows 10 AMD64
                        - **CPU:** Intel64 Family 6 Model 186 Stepping 2, GenuineIntel
                        - **Python:** 3.10.11 (64-bit CPython)
                        - **NumPy:** 2.2.6 (linked to optimized BLAS)
                        - **Git Commit:** `c84a3dc`
                        - **Timestamp:** `2026-10-29T12:24:12Z` (UTC)
                        """
                    )
                with env_col2:
                    st.markdown(
                        """
                        **Dataset Specification & Cryptographic Checksums:**
                        - **Dataset:** Canonical SIFT-100k subset (Texmex SIFT1M prefix)
                        - **Vectors:** N=100,000 base, Q=1,000 queries, d=128, float32
                        - **Distance Metric:** Squared Euclidean Distance ($||q - x||_2^2$)
                        - **Base Vectors SHA-256:** `21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816`
                        - **Query Vectors SHA-256:** `f7fc9be140accdfd64116c2fa2365ecdb69b8f084970c6b0532db5ff79ac8fdc`
                        - **Timing Protocol:** 500 individual per-query samples (2 warmups); 5 full-batch repetitions for QPS
                        """
                    )
                st.info(
                    "**Scope Disclosure:** This historical SIFT-100K benchmark evaluated Exact Reference, Exact BLAS, "
                    "and IVF-Flat sweeps. HNSW and IVF-PQ were evaluated in the interactive lab and smoke suites, and "
                    "were not included in this specific historical SIFT-100K batch run."
                )

            # Benchmark Plots
            st.markdown("#### SIFT-100K Benchmark Visualizations")
            plot_col1, plot_col2 = st.columns(2)
            lat_plot_path = Path("results/sift100k_p66/recall_vs_latency.png")
            dc_plot_path = Path("results/sift100k_p66/recall_vs_distance_computations.png")

            with plot_col1:
                if lat_plot_path.exists():
                    st.image(str(lat_plot_path), caption="SIFT-100k: Recall@10 vs. Median Query Latency (p50)", use_container_width=True)
                else:
                    st.warning("Latency plot artifact not found at results/sift100k_p66/recall_vs_latency.png")

            with plot_col2:
                if dc_plot_path.exists():
                    st.image(str(dc_plot_path), caption="SIFT-100k: Recall@10 vs. Distance Computations / Query", use_container_width=True)
                else:
                    st.warning("Distance plot artifact not found at results/sift100k_p66/recall_vs_distance_computations.png")

            # Benchmark Summary Table
            st.markdown("#### SIFT-100K Baseline & IVF-Flat Benchmark (`results/sift100k_p66/`)")
            summary_csv_path = Path("results/sift100k_p66/benchmark_summary.csv")
            if summary_csv_path.exists():
                df_sift = pd.read_csv(summary_csv_path)
                display_cols = [
                    "index_name",
                    "nlist",
                    "nprobe",
                    "recall_at_k",
                    "per_query_p50_ms",
                    "per_query_p95_ms",
                    "batch_qps",
                    "distance_computations",
                    "peak_rss_mib",
                    "build_time_sec",
                ]
                df_display = df_sift[display_cols].copy()
                df_display.columns = [
                    "Index Architecture",
                    "nlist",
                    "nprobe",
                    "Recall@10",
                    "p50 Latency (ms)",
                    "p95 Latency (ms)",
                    "Throughput (QPS)",
                    "Distance Ops / Query",
                    "Peak RSS (MiB)",
                    "Build Time (s)",
                ]
                st.dataframe(df_display, use_container_width=True)
            else:
                st.warning("Benchmark CSV artifact not found at results/sift100k_p66/benchmark_summary.csv")

            # SIFT-100K IVF-PQ Compression Benchmark
            st.markdown("#### SIFT-100K IVF-PQ Memory Compression Benchmark (`results/sift100k_ivfpq/`)")
            st.caption(
                "Evaluates IVFPQIndex on all 100,000 SIFT base vectors ($d=128$, $M=16$, $k_{\\text{sub}}=256$, $n_{\\text{list}}=100$) "
                "with 1,000 queries. Index storage is **2.46 MiB (25.8 bytes/vector)** vs. 51.2 MiB for raw float32 vectors (**20.8× memory compression**)."
            )
            ivfpq_csv_path = Path("results/sift100k_ivfpq/benchmark_summary.csv")
            if ivfpq_csv_path.exists():
                df_ivfpq = pd.read_csv(ivfpq_csv_path)
                ivfpq_cols = [
                    "index_name",
                    "nlist",
                    "M",
                    "k_sub",
                    "nprobe",
                    "recall_at_k",
                    "per_query_p50_ms",
                    "batch_qps",
                    "distance_computations",
                    "index_mem_mib",
                    "bytes_per_vec",
                ]
                df_ivfpq_disp = df_ivfpq[ivfpq_cols].copy()
                df_ivfpq_disp.columns = [
                    "Index",
                    "nlist",
                    "M Subspaces",
                    "k_sub",
                    "nprobe",
                    "Recall@10",
                    "p50 Latency (ms)",
                    "Throughput (QPS)",
                    "LUT Lookups / Query",
                    "Index RAM (MiB)",
                    "Bytes / Vec",
                ]
                st.dataframe(df_ivfpq_disp, use_container_width=True)
            else:
                st.info("Run SIFT benchmark suite to generate IVF-PQ scale results.")

            # SIFT-10K HNSW Proximity Graph Benchmark
            st.markdown("#### SIFT-10K HNSW Proximity Graph Benchmark (`results/sift10k_hnsw/`)")
            st.caption(
                "Evaluates HNSWIndex on the **SIFT-10K subset** (first 10,000 base vectors, $d=128$, $M=16$, $ef_{\\text{construction}}=64$) "
                "with 1,000 queries against exact subset ground truth. HNSW is evaluated on 10K to respect pure-Python interpreter graph-traversal constraints "
                "while demonstrating sub-linear distance evaluations and high recall (>94%–99%)."
            )
            hnsw_csv_path = Path("results/sift10k_hnsw/benchmark_summary.csv")
            if hnsw_csv_path.exists():
                df_hnsw = pd.read_csv(hnsw_csv_path)
                hnsw_cols = [
                    "index_name",
                    "M",
                    "ef_construction",
                    "ef_search",
                    "recall_at_k",
                    "per_query_p50_ms",
                    "batch_qps",
                    "distance_computations",
                    "index_mem_mib",
                    "bytes_per_vec",
                ]
                df_hnsw_disp = df_hnsw[hnsw_cols].copy()
                df_hnsw_disp.columns = [
                    "Index",
                    "M Degree",
                    "ef_construction",
                    "ef_search",
                    "Recall@10",
                    "p50 Latency (ms)",
                    "Throughput (QPS)",
                    "Distance Ops / Query",
                    "Index RAM (MiB)",
                    "Bytes / Vec",
                ]
                st.dataframe(df_hnsw_disp, use_container_width=True)
            else:
                st.info("Run SIFT benchmark suite to generate HNSW scale results.")

            st.markdown(
                """
                <div class="idx-callout" style="margin-top: 1rem;">
                    <strong>Documentation & Reproducibility Reference:</strong> For detailed mathematical formulations,
                    metric definitions, and reproduction commands, see <code>docs/BENCHMARK_METHODOLOGY.md</code>.
                    To reproduce offline smoke benchmarks, run: <code>python -m indexora.benchmark --smoke --no-plot</code>.
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =========================================================================
    # Tab 4: Mutations & Persistence (Admin Console Feel)
    # =========================================================================
    with tab_mutations:
        st.markdown("### Database Administration & Storage Console")
        st.markdown(
            "Manage real-time vector mutations (incremental additions, vector deletions) and execute "
            "atomic disk persistence with SHA-256 integrity validation."
        )

        session: LabSession = st.session_state.lab_session

        # Immediate flash message display for post-mutation synchronization
        if "mutation_flash" in st.session_state:
            f_type, f_msg = st.session_state.pop("mutation_flash")
            if f_type == "success":
                st.success(f_msg)
            elif f_type == "warning":
                st.warning(f_msg)
            elif f_type == "error":
                st.error(f_msg)

        if "load_verification_note" in st.session_state:
            st.info(st.session_state.pop("load_verification_note"))

        stat1, stat2, stat3, stat4 = st.columns(4)
        with stat1:
            st.metric("Active Index Type", session.index_name)
        with stat2:
            st.metric("Searchable Vectors", session.active_count)
        with stat3:
            st.metric("Incrementally Added", len(session.added_ids))
        with stat4:
            st.metric("Deleted Vectors", len(session.deleted_ids))

        st.markdown(
            """
            <div class="idx-callout" style="margin-top: 1rem; margin-bottom: 1.25rem;">
                <strong>Index-Specific Deletion Mechanics:</strong>
                <ul style="margin: 0.35rem 0 0 1.2rem; padding: 0;">
                    <li><strong>HNSW</strong>: Applies <em>logical tombstones</em> (<code>_deleted_nodes</code> set). Vectors remain in the multi-layer proximity graph to preserve graph connectivity and routing paths for other vectors, but are filtered out from search results.</li>
                    <li><strong>Exact Flat / BLAS</strong>: Performs <em>physical removal</em> by deleting vectors and IDs directly from memory arrays.</li>
                    <li><strong>IVF-Flat / IVF-PQ</strong>: Removes matching vector records directly from inverted posting lists.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        m_col1, m_col2 = st.columns(2)

        # Section 1: Add Vector
        with m_col1:
            st.markdown(
                """
                <div class="idx-card">
                    <div class="idx-card-title">➕ Incremental Mutation (add)</div>
                    <div class="idx-card-desc">Insert a new vector with a durable external entity ID into the active index.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            add_id = st.number_input("External ID to assign:", min_value=1, max_value=999999, value=9999, step=1)
            add_val = st.slider("Vector Constant Value (fills d dimensions):", min_value=-5.0, max_value=5.0, value=3.5, step=0.5)

            if st.button("Commit Vector Addition", use_container_width=True):
                vec_to_add = np.full(session.d, add_val, dtype=np.float32)
                try:
                    assigned = session.add_vector(vec_to_add, ext_id=int(add_id))
                    st.session_state["mutation_flash"] = (
                        "success",
                        f"Committed vector ID {assigned}. Active searchable records: {session.active_count}",
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Addition rejected: {exc}")

        # Section 2: Delete Vector
        with m_col2:
            st.markdown(
                """
                <div class="idx-card">
                    <div class="idx-card-title">🗑️ Vector Deletion (delete)</div>
                    <div class="idx-card-desc">Remove an external entity ID from the active index.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            del_id = st.number_input("External ID to delete:", min_value=1, max_value=999999, value=int(add_id), step=1)

            if st.button("Commit Vector Deletion", use_container_width=True):
                try:
                    del_cnt = session.delete_vector(int(del_id))
                    if del_cnt > 0:
                        st.session_state["mutation_flash"] = (
                            "success",
                            f"Deleted vector ID {del_id}. Searchable records remaining: {session.active_count}",
                        )
                        st.rerun()
                    else:
                        st.warning(f"Vector ID {del_id} not found in active live set (or already deleted).")
                except Exception as exc:
                    st.error(f"Deletion failed: {exc}")

        st.markdown("---")
        st.markdown("#### Sandboxed Atomic Persistence Engine (`save_index` / `load_index`)")
        st.markdown(
            "Serializes numerical array weights to pickle-free `.npy` files and generates a versioned "
            "`manifest.json` with streaming SHA-256 file digests. Operations are strictly sandboxed within "
            "`storage/lab_indexes/` to prevent arbitrary filesystem access or path traversal."
        )

        p_col1, p_col2 = st.columns(2)

        with p_col1:
            st.markdown("**Save Active Index State**")
            save_name_str = st.text_input(
                "Index Identifier:",
                value="my_lab_session",
                help="Alphanumeric name (1-64 characters). Stored in dedicated storage/lab_indexes/.",
            )
            if st.button("💾 Write to Sandboxed Storage", use_container_width=True):
                try:
                    manifest_data = session.save_named(save_name_str, overwrite=True)
                    st.success(f"Storage committed to sandbox: `{save_name_str}`")
                    st.json(manifest_data)
                except Exception as exc:
                    st.error(f"Save failed: {exc}")

        with p_col2:
            st.markdown("**Restore & Validate Checksums**")
            available_saved = session.list_saved()
            if available_saved:
                load_choice_str = st.selectbox(
                    "Select Saved Index:",
                    options=available_saved,
                    help="Detected valid saved indexes in storage/lab_indexes/.",
                )
            else:
                load_choice_str = st.text_input(
                    "Index Identifier to Load:",
                    value="my_lab_session",
                    key="load_ident_txt",
                    help="Name of saved index in storage/lab_indexes/.",
                )

            if st.button("📂 Load & Verify Storage", use_container_width=True):
                try:
                    restored_idx = session.load_named(load_choice_str, verify_checksums=True)
                    # Verification query
                    q_test = np.zeros(session.d, dtype=np.float32)
                    v_res = session.search(q_test, k=3)
                    st.session_state["mutation_flash"] = (
                        "success",
                        f"Successfully verified and loaded {type(restored_idx).__name__} ('{load_choice_str}')! Searchable records: {session.active_count}",
                    )
                    st.session_state["load_verification_note"] = f"Verification query on restored index returned top-3 IDs: {v_res.ids.tolist()}"
                    st.rerun()
                except Exception as exc:
                    st.error(f"Load verification failed: {exc}")

    # =========================================================================
    # Tab 5: Semantic Text Search (Product Quality)
    # =========================================================================
    with tab_semantic:
        st.markdown("### Natural Language Semantic Retrieval")
        st.markdown(
            "Search across a curated technical corpus of 25 engineering documents using dense sentence embeddings "
            "and Indexora's native vector engines (**Exact BLAS**, **IVF-Flat**, and **HNSW**)."
        )

        corpus_docs = load_demo_corpus()
        has_st = is_sentence_transformers_available()

        if not has_st:
            st.markdown(
                """
                <div class="idx-callout" style="margin-bottom: 1.5rem;">
                    <h4 style="margin-top:0; color:#1E293B; font-size: 1.05rem;">⚠️ Semantic Text Search Unavailable in this Environment</h4>
                    <p style="margin-bottom:0.6rem; color:#495460; font-size:0.9rem;">
                        Real semantic retrieval requires neural sentence embeddings (<code>sentence-transformers</code> and <code>PyTorch</code>).
                        These optional deep-learning packages are not installed in the current environment.
                    </p>
                    <p style="margin-bottom:0.6rem; color:#495460; font-size:0.9rem;">
                        <strong>Why pseudo-random mock search is disabled:</strong><br/>
                        Indexora strictly avoids substituting pseudo-random or hash-derived vectors for semantic retrieval.
                        Random vectors lack semantic structure, and ranking them generates misleading similarity scores
                        and false ground-truth claims.
                    </p>
                    <p style="margin-bottom:0.6rem; color:#495460; font-size:0.9rem;">
                        <strong>To enable real semantic search locally:</strong><br/>
                        <code>pip install -e ".[semantic]"</code><br/>
                        <code>streamlit run src/indexora/ui.py</code>
                    </p>
                    <p style="margin-bottom:0; color:#495460; font-size:0.9rem;">
                        <strong>Core Engine Status:</strong><br/>
                        Tabs 1–4 (Overview & Concepts, Search Playground, Accuracy & Benchmarks, Mutations & Sandboxed Persistence)
                        remain 100% operational using pure Python and NumPy with zero deep-learning dependencies.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("📄 Preview Technical Document Corpus (25 Documents Across 5 Domains)", expanded=True):
                for d in corpus_docs:
                    st.markdown(f"**[{d.category}] {d.title}** (ID: `{d.doc_id}`)\n\n_{d.text}_")
        else:
            # Model initialization and embedding generation with strict failure isolation
            model = None
            try:
                from indexora.semantic import get_embedding_model, encode_texts
                model = get_embedding_model(DEFAULT_EMBEDDING_MODEL)
            except Exception as exc:
                st.error(
                    f"**Neural embedding model initialization failed:** {exc}\n\n"
                    "Semantic search requires valid pretrained weights. Indexora will not silently substitute "
                    "random vectors. Please verify network access for the initial download or check local PyTorch setup."
                )
                st.stop()

            # Cache document corpus embeddings in session state
            if "semantic_doc_embeddings_real" not in st.session_state:
                with st.spinner("Generating dense sentence embeddings with all-MiniLM-L6-v2..."):
                    texts = [d.text for d in corpus_docs]
                    st.session_state["semantic_doc_embeddings_real"] = encode_texts(texts, model=model, normalize=True)

            cached_embeddings = st.session_state["semantic_doc_embeddings_real"]

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
                sem_ef = st.slider("HNSW Exploration Depth (`ef_search`):", min_value=8, max_value=64, value=32, step=8)

            engine = SemanticSearchEngine(
                corpus=corpus_docs,
                embeddings=cached_embeddings,
                index_type=sem_index_choice,
                embedder=model,
                model_name=DEFAULT_EMBEDDING_MODEL,
                ivf_nlist=4,
                ivf_nprobe=sem_nprobe,
                hnsw_m=8,
                hnsw_ef_construction=32,
                hnsw_ef_search=sem_ef,
            )

            st.markdown("---")
            st.markdown("#### Search Input")

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
                if preset_cols[i].button(short_label, key=f"ex_q_{i}", help=q_text, use_container_width=True):
                    preset_clicked = q_text

            default_query_text = preset_clicked or "How do attention mechanisms in language models work?"
            user_query = st.text_input("Enter natural language query:", value=default_query_text)

            filter_category_val = None if selected_cat == "All Categories" else selected_cat

            if user_query.strip():
                result: SemanticSearchResult = engine.search(
                    query=user_query,
                    k=sem_k,
                    filter_category=filter_category_val,
                    nprobe=sem_nprobe,
                    ef_search=sem_ef,
                )

                # Metrics row with clear latency segregation
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1:
                    st.metric("Total Latency", f"{result.latency_ms + result.embed_time_ms:.2f} ms")
                with m2:
                    st.metric("Text Embed Time (PyTorch)", f"{result.embed_time_ms:.2f} ms")
                with m3:
                    st.metric("Vector Search Time (Indexora)", f"{result.latency_ms:.3f} ms")
                with m4:
                    dist_str = str(result.distance_computations) if result.distance_computations is not None else "N/A"
                    st.metric("Distance Computations", dist_str)
                with m5:
                    match_pct = (sum(result.is_exact_match) / len(result.is_exact_match) * 100) if result.is_exact_match else 0.0
                    st.metric(
                        "Vector Oracle Agreement",
                        f"{match_pct:.0f}%",
                        help="Percentage of retrieved documents matching the exact brute-force BLAS vector search oracle. Measures vector index recall; does not evaluate human relevance.",
                    )

                st.markdown(
                    """
                    <div class="idx-callout" style="margin-bottom: 1rem; font-size: 0.85rem;">
                        <strong>Pipeline Clarity:</strong>
                        <strong>Text Embed Time</strong> is deep-learning inference.
                        <strong>Vector Search Time</strong> is Indexora nearest-neighbor search.
                        <strong>Vector Oracle Agreement</strong> indicates index fidelity against exhaustive vector scan; it does not independently establish semantic or human relevance.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("#### Ranked Documents")
                if not result.retrieved_documents:
                    st.info("No documents found matching the query and category filter.")
                else:
                    cos_sims = result.cosine_similarities
                    for rank, (doc, dist, is_match, cos_sim) in enumerate(
                        zip(result.retrieved_documents, result.distances, result.is_exact_match, cos_sims),
                        start=1,
                    ):
                        match_badge = (
                            '<span class="badge-emerald">Oracle Hit</span>'
                            if is_match
                            else '<span class="badge-indigo">ANN Candidate</span>'
                        )
                    tags_html = " ".join([f'<code>{t}</code>' for t in doc.metadata.get("tags", [])])

                    st.markdown(
                        f"""
                        <div class="idx-card">
                            <div class="idx-card-header">
                                <div class="idx-card-title">#{rank}. {doc.title}</div>
                                <div>{match_badge}</div>
                            </div>
                            <div class="idx-card-meta" style="margin-bottom: 0.5rem;">
                                <span>Category: <strong style="color: #303943;">{doc.category}</strong></span>
                                <span>Cosine Sim: <strong style="color: #3986D9;">{cos_sim:.4f}</strong></span>
                                <span>Squared L2: <strong style="color: #303943;">{dist:.4f}</strong></span>
                                <span>Doc ID: <code>{doc.doc_id}</code></span>
                            </div>
                            <p style="font-size: 0.88rem; color: #303943; margin-bottom: 0.5rem; line-height: 1.5;">"{doc.text}"</p>
                            <div style="font-size: 0.76rem; color: #687583;">
                                Tags: {tags_html}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        st.markdown(
            """
            <div class="idx-callout" style="margin-top: 1.5rem;">
                <strong>Mathematical Equivalence:</strong> Because text embeddings are unit-normalized (||q|| = ||x|| = 1), minimizing squared Euclidean distance <code>||q - x||² = 2 - 2·cos(q, x)</code> strictly maximizes cosine similarity. Indexora's native squared-L2 kernels power cosine ranking with zero algorithm alterations.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # Engineering Footer
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="idx-footer">
            <div class="idx-footer-left">
                <span class="idx-footer-brand">⚡ Indexora</span> — First-Principles Vector Search Engine & Interactive Architecture Lab
            </div>
            <div class="idx-footer-center">
                <span>Pure Python & NumPy Core • Zero External Vector DB Dependencies</span>
            </div>
            <div class="idx-footer-right">
                <span>Engineered by <strong>Shivansh</strong> • <a href="mailto:shivansh1.mitblr2024@learner.manipal.edu">shivansh1.mitblr2024@learner.manipal.edu</a></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    import streamlit.runtime as st_runtime

    if st_runtime.exists():
        main()
    else:
        from streamlit.web import cli as stcli

        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())
