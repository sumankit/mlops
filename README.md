# Student Performance and Dropout-Risk Prediction System

Individual Project | Data Engineering and MLOps | Part 1: Data Pipeline Implementation + Part 2: MLOps Pipeline Extension

## What this is

An end-to-end data pipeline that ingests the UCI Student Performance Dataset,
validates and cleans it, engineers analytical features (attendance %, risk
score, pass/fail, etc.), loads the result into a relational warehouse, and
exposes it through an interactive Streamlit dashboard. Orchestration is
provided both as a lightweight DAG-style Python orchestrator (used for the
local demo) and as a native Apache Airflow DAG (for deployment on an Airflow
instance).

## 1. Setup

```bash
# from the project root
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Dataset

Already downloaded and extracted under `data/raw/student_performance_uci/`.
To re-download from scratch:

```bash
curl -sL "https://archive.ics.uci.edu/static/public/320/student+performance.zip" \
  -o data/raw/student_performance_source.zip
cd data/raw && unzip -o student_performance_source.zip -d source_extract \
  && unzip -o source_extract/student.zip -d student_performance_uci
```
Source: https://archive.ics.uci.edu/dataset/320/student+performance (free, public, no login required).

## 3. Run the full pipeline

```bash
source venv/bin/activate
python src/pipeline_dag.py
```

This runs, in order: `ingest_extract` → `validate_and_clean` → `transform_features` →
`load_to_warehouse`, with per-task logging (`logs/`), retries, and a run-history
log (`logs/pipeline_run_history.csv`).

Individual stages can also be run standalone:
```bash
python src/ingest.py
python src/validate_clean.py
python src/transform.py
python src/load_db.py
```

## 4. Run on Apache Airflow (optional deployment path)

Copy `airflow_dag/student_pipeline_dag.py` into your Airflow `dags/` folder
(e.g. after `pip install apache-airflow` and `airflow standalone`, or via
Astronomer's `astro dev start`). It defines the same four tasks as Airflow
`PythonOperator`s with a daily schedule and 2 retries.

## 5. Launch the dashboard

```bash
source venv/bin/activate
streamlit run dashboard/app.py
```
Open http://localhost:8501. The dashboard has 5 views: attendance-vs-marks
scatter, subject-wise pass/fail, school & subject performance, a ranked
high-risk student list, and a per-student profile/intervention drill-down.

## 6. Project structure

```
mlops/
├── data/
│   ├── raw/            # immutable raw source + extracted UCI CSVs
│   ├── staging/         # timestamped ingestion snapshots
│   ├── cleaned/         # validated, deduplicated per-subject tables
│   ├── analytics/       # feature-engineered marts (student_master, summaries)
│   └── rejected/        # rejected_records.csv with rejection reasons
├── src/
│   ├── config.py         # paths, DB URI, business-rule thresholds
│   ├── ingest.py          # Task 1: extraction + ingestion logging
│   ├── validate_clean.py  # Task 2: validation, dedup, ID standardization
│   ├── transform.py       # Task 3: feature engineering, risk scoring
│   ├── load_db.py         # Task 4: load into SQL warehouse
│   └── pipeline_dag.py    # DAG-style orchestrator (local demo)
├── airflow_dag/
│   └── student_pipeline_dag.py   # native Airflow DAG definition
├── dashboard/
│   └── app.py             # Streamlit dashboard
├── db/
│   └── student_performance.db   # SQLite warehouse (generated)
├── ml/
│   ├── train.py             # Part 2: training + MLflow tracking + registry
│   ├── models/               # best_model.joblib, model_info.json (+ .dvc pointers)
│   └── artifacts/            # confusion matrices, model leaderboard
├── api/
│   ├── main.py               # FastAPI inference service
│   └── schemas.py            # Pydantic request/response models
├── monitoring/
│   ├── monitor.py            # drift / class-shift / latency / input-quality checks
│   └── reports/               # timestamped JSON monitoring reports
├── mlruns/, mlflow.db         # MLflow tracking store (generated, gitignored)
├── docs/
│   ├── architecture.md    # architecture diagram (Mermaid) + layer table
│   ├── data_dictionary.md # column definitions, validation rules, SQL DDL
│   ├── model_lifecycle.md  # Part 2: retraining criteria & model lifecycle
│   ├── versioning.md        # DVC + Git dataset/model versioning workflow
│   └── report.md           # project report
├── logs/                   # ingestion/validation/transform/load/pipeline/prediction logs
├── Dockerfile, .dockerignore
├── requirements.txt
├── .env.example
└── README.md
```

## 7. Data quality & error handling

- Every ingestion run is logged to `logs/ingestion_log.csv` (source, status, row count, timestamp, error).
- Validation rejects out-of-range grades/ages/absences and null required fields; rejects are logged with a reason to `data/rejected/rejected_records.csv`.
- The orchestrator retries a failing task up to 2 times and halts the DAG (logging the failure) if a task cannot recover, so a bad run never silently loads partial/corrupt data.
- No manual edits are ever made to `data/analytics/` — it is fully regenerated by re-running the pipeline.

## 8. Environment variables

Credentials are never hardcoded. See `.env.example` — copy to `.env` and load via
`python-dotenv` (or your shell) before pointing `src/config.py`'s `DB_URI` at PostgreSQL.

## 9. Part 2 — MLOps Pipeline Extension

Part 2 extends the Part 1 warehouse into a full MLOps loop: model training,
experiment tracking, a registry, an inference API, containerization, and
monitoring.

### 9.1 Train the model

```bash
source venv/bin/activate
python ml/train.py
```

Trains Logistic Regression, Random Forest, and XGBoost on
`data/analytics/student_master.csv` (target: `risk_flag` → binary
`is_high_risk`), tracks every run with MLflow (SQLite backend `mlflow.db`,
artifacts in `mlruns/`), registers the best model (by validation ROC-AUC) in
the MLflow Model Registry as `student_dropout_risk_model` (alias
`production`), and exports it to `ml/models/best_model.joblib` for serving.
Reference run: **Random Forest**, val ROC-AUC 0.98, test ROC-AUC 0.99 (see
`ml/models/model_info.json` and `ml/artifacts/model_leaderboard.csv`).

View the MLflow UI:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### 9.2 Serve predictions (FastAPI)

```bash
uvicorn api.main:app --reload --port 8000
```
- `GET /health` — model load status
- `POST /predict` — student features → `{risk_prediction, risk_probability, ...}`
- Interactive docs: http://localhost:8000/docs
- Every request is logged to `logs/prediction_log.csv` for monitoring.

### 9.3 Containerize with Docker

```bash
docker build -t student-risk-api .
docker run -p 8000:8000 student-risk-api
```

### 9.4 Monitor drift & performance

```bash
python monitoring/monitor.py
```
Compares the live prediction log against the training reference
distribution: feature drift (PSI), class-distribution shift, latency (p95),
and input-quality checks. Writes a timestamped JSON report to
`monitoring/reports/` and flags `retraining_recommended: true` when any
threshold is breached. Retraining criteria and the full model lifecycle are
documented in `docs/model_lifecycle.md`.

### 9.5 Dataset & model versioning

`data/analytics/student_master.csv` and `ml/models/best_model.joblib` are
version-tracked with **DVC** (pointer files committed to Git; see
`docs/versioning.md` for the reproduce/rollback workflow).

## 10. Development-only tools

`dvc` (dataset/model versioning) is a dev-time tool and is intentionally
**not** included in `requirements.txt` / the Docker image (the API only
needs the already-exported `ml/models/best_model.joblib`). Install it
separately for development: `pip install dvc`.
