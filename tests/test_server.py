import pytest
from fastapi.testclient import TestClient
from app.server import app

@pytest.fixture
def client():
    return TestClient(app)

def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data

def test_get_presets(client):
    response = client.get("/api/presets")
    assert response.status_code == 200
    data = response.json()
    assert "presets" in data
    assert len(data["presets"]) >= 3
    ids = [p["id"] for p in data["presets"]]
    assert "macro" in ids
    assert "housing" in ids
    assert "wages" in ids

def test_run_audit_endpoint(client):
    payload = {
        "dataset_id": "macro",
        "target_col": "GDP_Growth",
        "feature_cols": ["Interest_Rate", "Money_Growth", "Unemployment_Rate"],
        "add_constant": True,
        "alpha": 0.05
    }
    response = client.post("/api/audit", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "metrics" in data
    assert "params" in data
    assert "diagnostics" in data
    assert "charts" in data
    assert "scorecard" in data
    assert "audit_summary" in data

def test_static_index(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Econometrix Cockpit" in response.text

