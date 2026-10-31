"""Integration tests for Indexora FastAPI service layer."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from indexora.api import app

client = TestClient(app)


def test_api_status() -> None:
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["engine_version"] == "1.0.0"
    assert "dataset" in data
    assert "session" in data
    assert data["dataset"]["n_base"] > 0


def test_api_dataset_generate() -> None:
    req = {
        "n_base": 100,
        "n_queries": 6,
        "d": 8,
        "n_clusters": 3,
        "cluster_std": 0.2,
        "seed": 99,
    }
    res = client.post("/api/dataset/generate", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["dataset"]["n_base"] == 100
    assert data["dataset"]["d"] == 8


def test_api_search_exact_and_filtered() -> None:
    req = {
        "index_type": "Exact BLAS",
        "query_id_index": 0,
        "k": 5,
        "filter_mode": "even_ids",
    }
    res = client.post("/api/search", json=req)
    assert res.status_code == 200
    data = res.json()
    assert len(data["results"]) <= 5
    for item in data["results"]:
        assert item["id"] % 2 == 0


def test_api_benchmark_sweep() -> None:
    req = {
        "k": 3,
        "repetitions": 1,
        "ivf_nlist": 4,
        "ivf_nprobe": 2,
        "pq_m": 2,
        "pq_ksub": 4,
        "hnsw_m": 8,
        "hnsw_ef_construction": 16,
        "hnsw_ef_search": 16,
    }
    res = client.post("/api/benchmark/sweep", json=req)
    assert res.status_code == 200
    data = res.json()
    assert len(data["metrics"]) == 5
    assert any("Exact BLAS" in m["index_name"] for m in data["metrics"])


def test_api_explorer_pca() -> None:
    res = client.get("/api/explorer/pca?query_id_index=0")
    assert res.status_code == 200
    data = res.json()
    assert "points" in data
    assert len(data["points"]) > 0
    assert "centroids" in data
    assert "query" in data


def test_api_mutations_and_persistence() -> None:
    # Add vector
    add_req = {"ext_id": 88888, "constant_val": 2.5}
    res_add = client.post("/api/mutations/add", json=add_req)
    assert res_add.status_code == 200
    assert res_add.json()["assigned_id"] == 88888

    # Save index
    save_req = {"name": "test_api_session", "overwrite": True}
    res_save = client.post("/api/persistence/save", json=save_req)
    assert res_save.status_code == 200
    assert "manifest" in res_save.json()

    # List saved
    res_list = client.get("/api/persistence/list")
    assert res_list.status_code == 200
    assert "test_api_session" in res_list.json()["saved_indexes"]

    # Delete vector
    del_req = {"ext_id": 88888}
    res_del = client.post("/api/mutations/delete", json=del_req)
    assert res_del.status_code == 200
    assert res_del.json()["deleted_count"] == 1


def test_api_semantic_status() -> None:
    res = client.get("/api/semantic/status")
    assert res.status_code == 200
    data = res.json()
    assert "available" in data
    assert "document_count" in data


def test_api_serves_frontend_html() -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "Indexora" in res.text
    assert "<div id=\"root\"></div>" in res.text


def test_api_health_check() -> None:
    res = client.get("/healthz")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["engine"] == "Indexora"
