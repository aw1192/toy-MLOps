"""
Step 5 (CI/CD input): basic tests. Nothing fraud-specific -- the point is
that these run automatically on every push (see .github/workflows/ci.yml),
catching a broken API before it ever reaches deployment.

Run locally:
    pytest
"""
import pytest
from fastapi.testclient import TestClient

from app import app


@pytest.fixture
def client():
    # Using TestClient as a context manager triggers the lifespan startup
    # (model loading) -- without `with`, startup/shutdown events never fire.
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid_input(client):
    features = [0.0] * 30  # dummy input, just checking the pipeline runs end to end
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["prediction"] <= 1.0
    assert body["latency_ms"] >= 0


def test_predict_rejects_wrong_feature_count(client):
    response = client.post("/predict", json={"features": [1.0, 2.0]})
    assert response.status_code == 422  # Pydantic validation error, not a 500
