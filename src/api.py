import json
import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Response
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Histogram,
                               generate_latest)
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------- Logging (console + file) ----------
(BASE_DIR / "logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(BASE_DIR / "logs" / "api.log"),
    ],
)
logger = logging.getLogger("heart-api")

# ---------- Model ----------
model = joblib.load(BASE_DIR / "models" / "model.joblib")
with open(BASE_DIR / "models" / "feature_columns.json") as f:
    feature_columns = json.load(f)

# ---------- Prometheus metrics ----------
REQUESTS = Counter("api_requests_total", "Total API requests", ["endpoint", "status"])
PREDICTIONS = Counter("api_predictions_total", "Predictions made", ["result"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])

app = FastAPI(title="Heart Disease Prediction API", version="1.0")


class Patient(BaseModel):
    age: int = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=4)
    trestbps: int = Field(..., ge=50, le=300)
    chol: int = Field(..., ge=50, le=700)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: int = Field(..., ge=40, le=250)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=10)
    slope: int = Field(..., ge=0, le=3)
    ca: int = Field(..., ge=0, le=4)
    thal: int = Field(..., ge=0, le=7)


@app.get("/health")
def health():
    REQUESTS.labels(endpoint="/health", status="200").inc()
    return {"status": "ok"}


@app.post("/predict")
def predict(patient: Patient):
    start = time.time()
    row = pd.DataFrame([patient.model_dump()])[feature_columns]
    proba = float(model.predict_proba(row)[0][1])
    prediction = int(proba >= 0.5)
    confidence = proba if prediction == 1 else 1 - proba

    duration = time.time() - start
    LATENCY.labels(endpoint="/predict").observe(duration)
    REQUESTS.labels(endpoint="/predict", status="200").inc()
    PREDICTIONS.labels(result=str(prediction)).inc()
    logger.info(
        "predict input=%s prediction=%d confidence=%.3f latency_ms=%.1f",
        patient.model_dump(), prediction, confidence, duration * 1000,
    )

    return {
        "prediction": prediction,
        "label": "heart disease" if prediction == 1 else "no heart disease",
        "confidence": round(confidence, 4),
        "probability_disease": round(proba, 4),
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
