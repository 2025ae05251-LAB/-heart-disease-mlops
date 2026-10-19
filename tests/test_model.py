import json
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "model.joblib")
with open(BASE_DIR / "models" / "feature_columns.json") as f:
    feature_columns = json.load(f)

sample = pd.DataFrame([{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3,
    "slope": 0, "ca": 0, "thal": 1,
}])[feature_columns]


def test_model_predicts_binary_label():
    assert model.predict(sample)[0] in (0, 1)


def test_model_probabilities_are_valid():
    proba = model.predict_proba(sample)[0]
    assert len(proba) == 2
    assert abs(proba.sum() - 1.0) < 1e-6
    assert (proba >= 0).all() and (proba <= 1).all()


def test_model_handles_batch_input():
    batch = pd.concat([sample, sample, sample], ignore_index=True)
    assert len(model.predict(batch)) == 3


def test_model_quality_is_reasonable():
    results = pd.read_csv(BASE_DIR / "models" / "model_comparison.csv")
    assert results["test_roc_auc"].max() > 0.75