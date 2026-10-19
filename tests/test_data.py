from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

EXPECTED_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

raw_df = pd.read_csv(BASE_DIR / "data" / "raw" / "heart.csv")
clean_df = pd.read_csv(BASE_DIR / "data" / "processed" / "heart_clean.csv")


def test_raw_has_expected_columns():
    assert list(raw_df.columns) == EXPECTED_COLUMNS


def test_clean_has_no_missing_values():
    assert clean_df.isna().sum().sum() == 0


def test_clean_target_is_binary():
    assert set(clean_df["target"].unique()) == {0, 1}


def test_clean_keeps_all_columns_and_rows():
    assert list(clean_df.columns) == EXPECTED_COLUMNS
    assert len(clean_df) > 250


def test_clean_values_are_in_plausible_ranges():
    assert clean_df["age"].between(1, 120).all()
    assert clean_df["trestbps"].between(50, 300).all()
    assert clean_df["sex"].isin([0, 1]).all()