# Architecture Diagram — Student Performance & Dropout-Risk Pipeline

```mermaid
flowchart LR
    subgraph SRC["Source Layer"]
        A1[UCI student-mat.csv]
        A2[UCI student-por.csv]
    end

    subgraph ING["Ingestion Layer (src/ingest.py)"]
        B1[Extract via Pandas]
        B2[Stamp extraction timestamp]
        B3[Write ingestion_log.csv]
    end

    subgraph STG["Raw / Staging Layer (data/staging)"]
        C1[Staged CSV snapshots]
    end

    subgraph TRF["Transformation Layer"]
        D1[validate_clean.py: dedupe, rule checks, ID standardization]
        D2[rejected_records.csv]
        D3[transform.py: feature engineering, risk scoring]
    end

    subgraph CLN["Cleaned & Analytical Layer (data/cleaned, data/analytics)"]
        E1[student_master.csv]
        E2[subject_summary.csv]
        E3[department_summary.csv]
    end

    subgraph STO["Storage Layer"]
        F1[(SQLite / PostgreSQL warehouse\ndim_student, fact_assessment,\nagg_subject_summary, agg_department_summary)]
    end

    subgraph ANL["Analytics / Serving Layer"]
        G1[Streamlit Dashboard]
    end

    subgraph ORCH["Orchestration"]
        H1[Apache Airflow DAG /\nDAG-style Python orchestrator\n(src/pipeline_dag.py)]
    end

    subgraph MLOPS["MLOps Layer (Part 2 — future)"]
        I1[Feature engineering pipeline]
        I2[MLflow experiment tracking]
        I3[FastAPI inference service]
        I4[Docker container]
        I5[Drift & performance monitoring]
    end

    A1 --> B1
    A2 --> B1
    B1 --> B2 --> B3
    B2 --> C1
    C1 --> D1
    D1 -- invalid rows --> D2
    D1 -- valid rows --> D3
    D3 --> E1 --> E2
    E1 --> E3
    E1 --> F1
    E2 --> F1
    E3 --> F1
    F1 --> G1
    H1 -. schedules & orchestrates .-> B1
    H1 -. schedules & orchestrates .-> D1
    H1 -. schedules & orchestrates .-> D3
    H1 -. schedules & orchestrates .-> F1
    F1 -. Part 2 .-> I1 --> I2 --> I3 --> I4 --> I5
```

## Layer descriptions

| Layer | Purpose | Implementation |
|---|---|---|
| Source | Original public dataset files | UCI Student Performance Dataset (student-mat.csv, student-por.csv) |
| Ingestion | Extract source files, stamp metadata, log status | `src/ingest.py` |
| Raw/Staging | Immutable timestamped snapshots of extracted data | `data/staging/*.csv` |
| Transformation (validation) | Standardize IDs, remove duplicates/invalid rows, log rejects | `src/validate_clean.py` |
| Cleaned | Rule-validated, deduplicated per-subject tables | `data/cleaned/*.csv` |
| Transformation (feature engineering) | Derive attendance %, risk score, pass/fail, aggregates | `src/transform.py` |
| Analytical | Student-level and aggregate marts | `data/analytics/*.csv` |
| Storage | Relational warehouse (star-ish schema) | `db/student_performance.db` (SQLite, Postgres-portable) — `src/load_db.py` |
| Orchestration | Schedules & sequences all tasks with retries/logging | `src/pipeline_dag.py` (local demo) + `airflow_dag/student_pipeline_dag.py` (Airflow deployment) |
| Analytics/Serving | Interactive dashboard for stakeholders | `dashboard/app.py` (Streamlit) |
| MLOps (Part 2) | Model training, tracking, serving, monitoring | To be implemented in Part 2 |
