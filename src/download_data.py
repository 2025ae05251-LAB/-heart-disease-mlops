from pathlib import Path

import pandas as pd

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

out_path = Path("data/raw/heart.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)

try:
    df = pd.read_csv(URL, header=None, names=COLUMNS, na_values="?")
except Exception as exc:
    print(f"Direct download failed ({exc}); trying ucimlrepo...")
    from ucimlrepo import fetch_ucirepo

    data = fetch_ucirepo(id=45)
    df = data.data.features.copy()
    df["target"] = data.data.targets.values.ravel()

df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows and {df.shape[1]} columns to {out_path}")
