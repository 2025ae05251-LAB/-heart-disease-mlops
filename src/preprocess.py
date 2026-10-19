from pathlib import Path

import pandas as pd

raw_path = Path("data/raw/heart.csv")
clean_path = Path("data/processed/heart_clean.csv")
clean_path.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(raw_path)
print("Raw shape:", df.shape)
print("Missing values before:")
print(df.isna().sum()[df.isna().sum() > 0])

# Remove exact duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

# Fill missing values: ca is a count (median), thal is categorical (most frequent)
df["ca"] = df["ca"].fillna(df["ca"].median())
df["thal"] = df["thal"].fillna(df["thal"].mode()[0])

# Original target is 0-4 (severity). Convert to binary: 0 = no disease, 1 = disease
df["target"] = (df["target"] > 0).astype(int)

# Make sure column types are sensible
int_cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "slope", "ca", "thal", "target"]
df[int_cols] = df[int_cols].astype(int)

df.to_csv(clean_path, index=False)

print("Missing values after:", int(df.isna().sum().sum()))
print("Class balance:")
print(df["target"].value_counts())
print(f"Saved {len(df)} rows and {df.shape[1]} columns to {clean_path}")