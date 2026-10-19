from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

valid_patient = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3,
    "slope": 0, "ca": 0, "thal": 1,
}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_prediction_and_confidence():
    response = client.post("/predict", json=valid_patient)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert "label" in body


def test_predict_rejects_missing_field():
    incomplete = {k: v for k, v in valid_patient.items() if k != "age"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_rejects_out_of_range_value():
    bad = dict(valid_patient, age=500)
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_metrics_endpoint_exposes_counters():
    client.post("/predict", json=valid_patient)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "api_requests_total" in response.text