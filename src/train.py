import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

model_dir = Path("models")
model_dir.mkdir(exist_ok=True)
plot_dir = Path("screenshots/training")
plot_dir.mkdir(parents=True, exist_ok=True)

# ---------- Data ----------
df = pd.read_csv("data/processed/heart_clean.csv")
X = df.drop(columns="target")
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# ---------- Feature engineering (inside the pipeline) ----------
numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
onehot_cols = ["cp", "restecg", "slope", "thal"]
binary_cols = ["sex", "fbs", "exang"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), onehot_cols),
        ("bin", "passthrough", binary_cols),
    ]
)

# ---------- Models and tuning grids ----------
candidates = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        {"model__C": [0.01, 0.1, 1, 10], "model__penalty": ["l2"]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5, None],
            "model__min_samples_split": [2, 5],
        },
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("heart-disease-classification")

results = []
best_auc = -1
best_name = None
best_model = None

for name, (estimator, grid) in candidates.items():
    pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])

    with mlflow.start_run(run_name=name):
        # Hyperparameter tuning with cross-validation (train set only)
        search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        search.fit(X_train, y_train)
        model = search.best_estimator_

        # Cross-validated metrics of the tuned model
        cv_scores = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring=["accuracy", "precision", "recall", "roc_auc"],
        )

        # Hold-out test metrics
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        test_metrics = {
            "test_accuracy": accuracy_score(y_test, pred),
            "test_precision": precision_score(y_test, pred),
            "test_recall": recall_score(y_test, pred),
            "test_roc_auc": roc_auc_score(y_test, proba),
        }
        cv_metrics = {
            "cv_accuracy": cv_scores["test_accuracy"].mean(),
            "cv_precision": cv_scores["test_precision"].mean(),
            "cv_recall": cv_scores["test_recall"].mean(),
            "cv_roc_auc": cv_scores["test_roc_auc"].mean(),
        }

        # Log parameters and metrics
        mlflow.log_param("model_type", name)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(cv_metrics)
        mlflow.log_metrics(test_metrics)

        # Plots
        RocCurveDisplay.from_estimator(model, X_test, y_test)
        plt.title(f"ROC curve - {name}")
        roc_file = plot_dir / f"roc_{name}.png"
        plt.savefig(roc_file, dpi=150, bbox_inches="tight")
        plt.close()

        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
        plt.title(f"Confusion matrix - {name}")
        cm_file = plot_dir / f"confusion_{name}.png"
        plt.savefig(cm_file, dpi=150, bbox_inches="tight")
        plt.close()

        mlflow.log_artifact(str(roc_file))
        mlflow.log_artifact(str(cm_file))

        # Log the full pipeline (preprocessing + model)
        mlflow.sklearn.log_model(
            model, "model", skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        print(f"\n{name}")
        print("  best params:", search.best_params_)
        print("  CV  :", {k: round(v, 3) for k, v in cv_metrics.items()})
        print("  Test:", {k: round(v, 3) for k, v in test_metrics.items()})

        results.append({"model": name, **cv_metrics, **test_metrics})

        if test_metrics["test_roc_auc"] > best_auc:
            best_auc = test_metrics["test_roc_auc"]
            best_name = name
            best_model = model

# ---------- Save the final model ----------
joblib.dump(best_model, model_dir / "model.joblib")
with open(model_dir / "feature_columns.json", "w") as f:
    json.dump(list(X.columns), f)
pd.DataFrame(results).to_csv(model_dir / "model_comparison.csv", index=False)

print(f"\nBest model: {best_name} (test ROC-AUC = {best_auc:.3f})")
print("Saved:", model_dir / "model.joblib")
