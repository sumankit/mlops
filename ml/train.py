"""
Part 2 — Model training pipeline for dropout/performance-risk prediction.

Prediction target: `risk_flag` from data/analytics/student_master.csv,
converted to a binary label `is_high_risk` (1 = High Risk / needs early
intervention, 0 = Low Risk). This target was engineered in Part 1
(src/transform.py) from final grade, attendance, past failures, and grade
decline -- see docs/data_dictionary.md.

Data leakage note: G3 (final grade) is EXCLUDED from the feature set because
risk_flag is partly derived from G3. Features use only information available
before/at the point of intervention: G1, G2, attendance, engagement,
demographics, and behavioural attributes.

Split strategy: the UCI dataset is a single cross-sectional snapshot per
student (no per-student time series beyond the G1->G2->G3 term progression,
which is already encoded as features) -- so a chronological split is not
applicable. A stratified random train/validation/test split (70/15/15,
stratified on the target to preserve class balance) is used instead, which
is the problem-appropriate choice for cross-sectional tabular data.

Tracked with MLflow: params, metrics, confusion-matrix artifact, and the
fitted model for every run; the best run (by validation ROC-AUC) is
registered in the MLflow Model Registry and also exported as a joblib file
for the FastAPI service (ml/models/best_model.joblib).

Usage:
    python ml/train.py
"""
import os
import sys
import json
import hashlib
import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import ANALYTICS_DIR, PROJECT_ROOT, LOG_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "training.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("train")

MLFLOW_TRACKING_DIR = os.path.join(PROJECT_ROOT, "mlruns")
MLFLOW_DB_PATH = os.path.join(PROJECT_ROOT, "mlflow.db")
MODEL_DIR = os.path.join(PROJECT_ROOT, "ml", "models")
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "ml", "artifacts")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

EXPERIMENT_NAME = "student_dropout_risk_prediction"
REGISTERED_MODEL_NAME = "student_dropout_risk_model"
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel",
    "freetime", "goout", "Dalc", "Walc", "health", "absences",
    "attendance_percentage", "avg_internal_marks", "assignment_completion_rate",
    "G1", "G2",
]
CATEGORICAL_FEATURES = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
    "guardian", "schoolsup", "famsup", "paid", "activities", "nursery",
    "higher", "internet", "romantic", "subject",
]
TARGET_COL = "is_high_risk"


def _dataset_checksum(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def load_dataset():
    path = os.path.join(ANALYTICS_DIR, "student_master.csv")
    df = pd.read_csv(path)
    df[TARGET_COL] = (df["risk_flag"] == "High Risk").astype(int)
    checksum = _dataset_checksum(path)
    log.info(f"Loaded {len(df)} rows from {path} (sha256:{checksum})")
    return df, checksum


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def evaluate(model, X, y):
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1_score": f1_score(y, preds, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
    }, preds


def log_confusion_matrix(y_true, y_pred, run_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Risk", "High Risk"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {run_name}")
    path = os.path.join(ARTIFACT_DIR, f"confusion_matrix_{run_name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def run_training():
    log.info("=== Starting Part 2 model training pipeline ===")
    # MLflow >=3.x deprecates the plain filesystem backend for tracking
    # metadata; a local SQLite backend store is used instead, with run
    # artifacts (models, confusion matrices) still saved to the local
    # ./mlruns directory. This keeps everything file-based / server-free for
    # local grading while giving a fully functional MLflow Model Registry.
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    os.makedirs(MLFLOW_TRACKING_DIR, exist_ok=True)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=f"file://{MLFLOW_TRACKING_DIR}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df, dataset_checksum = load_dataset()
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df[TARGET_COL]

    # 70/15/15 stratified split -> train / validation / test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    log.info(f"Split sizes -> train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    candidates = {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
            {"max_iter": 1000, "class_weight": "balanced"},
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=300, max_depth=8, class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {"n_estimators": 300, "max_depth": 8, "class_weight": "balanced"},
        ),
        "xgboost": (
            XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.08,
                eval_metric="logloss", random_state=RANDOM_STATE,
            ),
            {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.08},
        ),
    }

    results = []
    for model_name, (estimator, params) in candidates.items():
        with mlflow.start_run(run_name=model_name) as run:
            pipeline = Pipeline([
                ("preprocess", build_preprocessor()),
                ("model", estimator),
            ])
            pipeline.fit(X_train, y_train)

            train_metrics, _ = evaluate(pipeline, X_train, y_train)
            val_metrics, val_preds = evaluate(pipeline, X_val, y_val)

            mlflow.log_param("model_type", model_name)
            mlflow.log_param("dataset_checksum", dataset_checksum)
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_param("val_rows", len(X_val))
            for k, v in params.items():
                mlflow.log_param(k, v)
            for k, v in train_metrics.items():
                mlflow.log_metric(f"train_{k}", v)
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", v)

            cm_path = log_confusion_matrix(y_val, val_preds, model_name)
            mlflow.log_artifact(cm_path)

            if model_name == "xgboost":
                mlflow.xgboost.log_model(pipeline.named_steps["model"], name="xgb_raw_model")
                # The full pipeline includes an XGBClassifier, which sklearn's
                # skops-based serializer flags as an untrusted type by default;
                # it is safe here since we produced it ourselves in this run.
                mlflow.sklearn.log_model(
                    pipeline, name="model",
                    skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"],
                )
            else:
                mlflow.sklearn.log_model(pipeline, name="model")

            log.info(f"[{model_name}] val_roc_auc={val_metrics['roc_auc']:.4f} val_f1={val_metrics['f1_score']:.4f}")
            results.append({
                "model_name": model_name,
                "run_id": run.info.run_id,
                "pipeline": pipeline,
                "val_metrics": val_metrics,
            })

    # Select best model by validation ROC-AUC
    best = max(results, key=lambda r: r["val_metrics"]["roc_auc"])
    log.info(f"Best model: {best['model_name']} (val_roc_auc={best['val_metrics']['roc_auc']:.4f})")

    # Final unbiased evaluation on the held-out test set
    test_metrics, test_preds = evaluate(best["pipeline"], X_test, y_test)
    log.info(f"Test-set metrics for {best['model_name']}: {test_metrics}")

    with mlflow.start_run(run_id=best["run_id"]):
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        model_uri = f"runs:/{best['run_id']}/model"
        registered = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
        log.info(f"Registered model '{REGISTERED_MODEL_NAME}' version {registered.version}")

        client = mlflow.MlflowClient()
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "production", registered.version)
        log.info(f"Aliased version {registered.version} as 'production'")

    # Export for the FastAPI service (avoids requiring a live MLflow server at inference time)
    model_path = os.path.join(MODEL_DIR, "best_model.joblib")
    joblib.dump(best["pipeline"], model_path)

    model_info = {
        "model_name": best["model_name"],
        "mlflow_run_id": best["run_id"],
        "registered_model_name": REGISTERED_MODEL_NAME,
        "registered_version": registered.version,
        "dataset_checksum": dataset_checksum,
        "trained_at": datetime.now().isoformat(),
        "feature_columns": feature_cols,
        "val_metrics": best["val_metrics"],
        "test_metrics": test_metrics,
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
    }
    with open(os.path.join(MODEL_DIR, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)

    # Leaderboard for the report
    leaderboard = pd.DataFrame([
        {"model": r["model_name"], **r["val_metrics"]} for r in results
    ]).sort_values("roc_auc", ascending=False)
    leaderboard.to_csv(os.path.join(ARTIFACT_DIR, "model_leaderboard.csv"), index=False)
    log.info(f"\n{leaderboard.to_string(index=False)}")

    log.info(f"=== Training complete. Best model exported to {model_path} ===")
    return model_info


if __name__ == "__main__":
    run_training()
