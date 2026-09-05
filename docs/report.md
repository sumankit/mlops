# Project Report — Student Performance and Dropout-Risk Prediction System
### Part 1: Data Pipeline Implementation

**Course:** Data Engineering and MLOps  |  **Mode:** Individual Project  |  **Author:** [Your Name]
**Submission date:** [fill in] — Deadline: 7 September 2026

---

## 1. Introduction and Problem Understanding

Educational institutions generate performance-related data across multiple
disconnected systems — ERP gradebooks, LMS activity logs, and manual
attendance registers. Because this data is siloed, academic staff struggle to
identify, early enough to intervene, students who are at risk of failing or
dropping out.

This project builds a data-driven academic analytics system that:
1. Ingests raw student academic records from a public dataset.
2. Cleans, validates, and transforms them into a structured, queryable
   warehouse.
3. Produces analytical tables capturing attendance, performance, and
   engagement features.
4. Exposes the results through an interactive dashboard for performance,
   attendance, and risk analysis.
5. Lays the feature foundation (`risk_flag`, `risk_score`) for the dropout/
   performance-risk prediction model to be built in Part 2.

## 2. Project Objectives

- Create a unified, reproducible student data pipeline from a public academic
  source.
- Clean, validate, transform, and store student records in a structured
  database.
- Develop a multi-view dashboard for performance, attendance, and risk
  analysis.
- Prepare the analytical foundation (features, risk scoring) that Part 2 will
  extend into a full MLOps prediction workflow.

## 3. Data Source

**UCI Student Performance Dataset** (Cortez & Silva, 2008)
- **URL:** https://archive.ics.uci.edu/dataset/320/student+performance
- **Access:** free, public CSV download, no authentication or registration
  required.
- **Content:** two files — `student-mat.csv` (395 records, Mathematics
  course) and `student-por.csv` (649 records, Portuguese language course) —
  from two Portuguese secondary schools ('GP', 'MS'). Each record has 33
  attributes: demographics (age, sex, address, family background),
  behavioural attributes (study time, past failures, alcohol consumption,
  going out), and three periodic academic grades G1, G2, G3 (0–20 scale) plus
  an absence count.
- **Why this dataset:** it satisfies the assignment's "Dataset rule" (free,
  legally accessible, public), is the explicitly suggested primary historical
  dataset, and already contains the attendance-adjacent (`absences`),
  academic (`G1`/`G2`/`G3`), and behavioural fields the pipeline needs to
  build attendance, performance, and risk features without any manually
  fabricated data.
- **Access instructions:** downloaded programmatically (see README §2) via
  `curl` from the UCI static ZIP endpoint — no manual steps or credentials.

No private, personally identifiable, or confidential data is used; all
records are the pseudonymous UCI research records.

## 4. Data Architecture

The pipeline follows a layered architecture: **source → ingestion →
raw/staging → transformation (validation + feature engineering) → cleaned →
analytical → storage → analytics/serving**, with an orchestration layer
scheduling and sequencing every stage. The full Mermaid diagram and a
layer-by-layer description are in `docs/architecture.md`; the corresponding
implementation files are summarized in the table below.

| Layer | Files |
|---|---|
| Source | `data/raw/student_performance_uci/*.csv` |
| Ingestion | `src/ingest.py` |
| Raw/Staging | `data/staging/*.csv` |
| Validation/Cleaning | `src/validate_clean.py` |
| Cleaned | `data/cleaned/*.csv` |
| Feature engineering | `src/transform.py` |
| Analytical | `data/analytics/*.csv` |
| Storage (warehouse) | `src/load_db.py`, `db/student_performance.db` |
| Orchestration | `src/pipeline_dag.py`, `airflow_dag/student_pipeline_dag.py` |
| Analytics/Serving | `dashboard/app.py` (Streamlit) |

## 5. Data Ingestion

`src/ingest.py` extracts each source CSV with Pandas, tags every record with
a `subject` label and an `extraction_timestamp`, and writes an immutable,
timestamped snapshot to the staging layer. Every extraction attempt — success
or failure — is appended to `logs/ingestion_log.csv` with the source name,
file path, status, row count, and error message (if any), satisfying the
"record extraction date, source, file/API status, and row count" requirement.
If one source file fails to extract, the ingestion task logs the error and
continues with the remaining sources rather than aborting the whole run.

**Observed run:** both source files ingested successfully — 395 rows
(Mathematics) and 649 rows (Portuguese), 1,044 rows total.

## 6. ETL and Data Quality

### 6.1 Standardization and identifiers
The raw UCI dataset has no native student identifier. `src/validate_clean.py`
synthesizes a standardized `student_id` (`MAT-0001…`, `POR-0001…`) so every
record can be uniquely and consistently referenced across the cleaned,
analytical, and warehouse layers.

### 6.2 Validation rules
| Rule | Rejects when |
|---|---|
| Grade range | G1, G2, or G3 outside [0, 20] |
| Age range | age outside [14, 25] |
| Absence range | absences < 0 or > 93 |
| Required fields | school/sex/age/G1/G2/G3/absences null |
| Duplicates | duplicate `student_id` rows |

Rejected records are written with a `rejection_reason` to
`data/rejected/rejected_records.csv`. In the demonstrated run, the source
dataset was already well-formed and **0 records were rejected** — the
rejection path was verified separately by temporarily injecting an
out-of-range grade, which was correctly caught and logged.

### 6.3 Feature engineering (`src/transform.py`)
| Feature | Derivation |
|---|---|
| `attendance_percentage` | `100 − (absences / 93 × 100)`, where 93 approximates a full academic year's school days (documented assumption — the source provides an absence count, not a raw attendance %) |
| `avg_internal_marks` | mean(G1, G2) |
| `previous_semester_performance` | G2, used as the checkpoint prior to the final grade G3 |
| `assignment_completion_rate` | weighted proxy from `studytime` and `failures` (no direct assignment field exists in the source; documented in the data dictionary) |
| `final_grade_percent` | `G3 / 20 × 100` |
| `pass_fail` | Pass if G3 ≥ 10, else Fail |
| `risk_flag` | High Risk if failing OR attendance < 75% OR any past failures OR a ≥4-point grade decline from G1 to G3 |
| `risk_score` | 0–100 weighted composite of the four risk signals, used to rank the intervention list |

Full column-level definitions and the validation-rule table are in
`docs/data_dictionary.md`.

## 7. Storage — Database Schema

Cleaned and analytical data are loaded into a SQL warehouse (`src/load_db.py`)
using a lightweight star schema: `dim_student` (demographics) and
`fact_assessment` (grades, engineered features, risk) linked by
`student_id`, plus two pre-aggregated analytical marts,
`agg_subject_summary` and `agg_department_summary`. SQLite is used for the
local demo/submission (zero-config, file-based); the schema is written in
portable ANSI SQL, and switching to PostgreSQL requires only changing the
SQLAlchemy connection string in `src/config.py` (see `docs/data_dictionary.md`
for the equivalent `CREATE TABLE` DDL and the environment-variable pattern
for credentials). Indexes are created on `student_id`, `subject`, and
`risk_flag` for dashboard query performance.

**Loaded row counts (demonstrated run):** `dim_student` = 1,044,
`fact_assessment` = 1,044, `agg_subject_summary` = 2, `agg_department_summary`
= 4.

## 8. Orchestration

Two equivalent orchestration implementations are provided:
1. **`src/pipeline_dag.py`** — a DAG-style Python orchestrator used for the
   local demo/grading run. It executes the four tasks
   (`ingest_extract → validate_and_clean → transform_features →
   load_to_warehouse`) in dependency order, retries a failing task up to 2
   times, halts the pipeline (without loading partial data) if a task cannot
   recover, and logs every run to `logs/pipeline_run_history.csv` with
   per-task status and duration.
2. **`airflow_dag/student_pipeline_dag.py`** — the same four tasks expressed
   as native Apache Airflow `PythonOperator`s with a daily `0 2 * * *`
   schedule and 2 retries, ready to be dropped into an Airflow `dags/`
   folder for production scheduling.

**Demonstrated end-to-end run:** all four tasks succeeded (see
`logs/pipeline_run_history.csv`); total pipeline execution time ≈ 0.1s
against the 1,044-row dataset.

## 9. Data Visualization — Streamlit Dashboard

`dashboard/app.py` connects directly to the SQL warehouse and provides
sidebar filters (subject, school, risk level) plus **five views**, satisfying
the "at least five meaningful views" requirement:

1. **Attendance vs Marks Analysis** — scatter plot of attendance % against
   final grade %, colored by risk level.
2. **Subject-wise Pass/Fail Distribution** — grouped bar chart per subject.
3. **School & Subject Performance** — average final grade % by school and
   subject (department/semester-style comparison).
4. **High-Risk Student List** — ranked table of intervention candidates
   sorted by `risk_score`.
5. **Student Profile & Intervention Detail** — per-student drill-down with
   grade trajectory (G1→G2→G3) and demographic/engagement context.

A full-page screenshot of the running dashboard is included at
`docs/screenshots/dashboard_full.png`.

## 10. Execution Evidence

- Ingestion log: `logs/ingestion_log.csv` — 2/2 sources SUCCESS, 1,044 total rows.
- Validation/cleaning log: `logs/validation.log`.
- Transformation log: `logs/transform.log`.
- Load log: `logs/load.log`.
- Full pipeline run history: `logs/pipeline_run_history.csv` — overall SUCCESS.
- Dashboard screenshot: `docs/screenshots/dashboard_full.png`.

## 11. Challenges and Design Decisions

- **No native student ID or true multi-subject linkage.** The UCI dataset
  does not carry a shared student key across the Math and Portuguese files
  (only an unreliable demographic-attribute match is possible). We modeled
  each file as its own subject-tagged record set with a synthesized ID,
  documented this explicitly, and designed the schema so a real institution's
  ERP export (which does carry a genuine student ID) can be substituted with
  no pipeline changes.
- **No direct attendance-percentage or assignment-completion field.** Both
  were derived as documented proxies from `absences`, `studytime`, and
  `failures` rather than fabricated, and every assumption is called out in
  `docs/data_dictionary.md`.
- **Airflow setup time vs. demo reliability.** A full Airflow installation
  (webserver + scheduler + metadata DB) was judged riskier to demo reliably
  in the available time than a lightweight, equally rigorous DAG-style
  orchestrator; both are submitted, satisfying "Airflow DAG **or approved
  equivalent** ETL workflow."

## 12. Conclusion and Part 2 Outlook

Part 1 delivers a complete, reproducible pipeline from a public academic
dataset to a queryable warehouse and an interactive risk-analysis dashboard,
with logging, validation, and error handling at every stage. The
`risk_flag`/`risk_score` features already computed in
`data/analytics/student_master.csv` are designed to seed Part 2 directly:
they define the prediction target (dropout/failure risk) that will be
trained with Logistic Regression, Random Forest, and XGBoost, tracked with
MLflow, served through FastAPI in a Docker container, and monitored for
drift — extending this same codebase into the full MLOps lifecycle.

## 13. Part 2 — MLOps Pipeline Extension

Part 2 was implemented on top of the same codebase, extending
`data/analytics/student_master.csv` into a full training → tracking →
serving → monitoring loop.

### 13.1 Prediction target and features
Binary target `is_high_risk` (1 = High Risk, 0 = Low Risk), derived from the
Part 1 `risk_flag`. **G3 (final grade) is deliberately excluded from the
feature set** since the target is partly derived from it — using it would
leak the label into the features. Features used instead: G1, G2, attendance
%, average internal marks, assignment-completion proxy, demographics, and
behavioural attributes (35 features total; full list in
`ml/models/model_info.json`).

### 13.2 Train/validation/test split
The UCI dataset is a single cross-sectional snapshot (no per-student time
series beyond the already-encoded G1→G2→G3 progression), so a chronological
split does not apply. A **stratified random 70/15/15 split** (stratified on
the target to preserve class balance) was used instead — the
problem-appropriate choice for this data.

### 13.3 Model training and tracking
Three models were trained and compared: Logistic Regression (baseline),
Random Forest, and XGBoost. Every run's parameters, metrics, and artifacts
(model, confusion matrix) were tracked in **MLflow** (experiment
`student_dropout_risk_prediction`, SQLite backend store):

| Model | Val Accuracy | Val Precision | Val Recall | Val F1 | Val ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest (selected)** | 0.924 | 0.818 | 0.957 | 0.882 | **0.980** |
| Logistic Regression | 0.911 | 0.811 | 0.915 | 0.860 | 0.975 |
| XGBoost | 0.911 | 0.867 | 0.830 | 0.848 | 0.965 |

Random Forest was selected by highest validation ROC-AUC and confirmed on
the held-out test set (accuracy 0.930, recall 0.957, ROC-AUC 0.991) — high
recall on the High-Risk class was prioritized since a missed at-risk student
(false negative) is costlier than a false alarm in this application.

### 13.4 Registry, serving, and containerization
The selected model was registered in the **MLflow Model Registry** as
`student_dropout_risk_model` and aliased `production`, then exported to
`ml/models/best_model.joblib` for the inference service. A **FastAPI**
service (`api/main.py`) exposes `GET /health` and `POST /predict`, validates
inputs with Pydantic, logs every prediction (features, output, latency) to
`logs/prediction_log.csv`, and was verified end-to-end with live requests
(sample: risk_probability 0.94 → "High Risk" for a failing, low-attendance
profile). The service was containerized with **Docker**
(`Dockerfile`, Python 3.12-slim + libgomp for XGBoost) and verified running
inside a container on port 8000.

### 13.5 Monitoring and retraining
`monitoring/monitor.py` compares live prediction traffic against the
training reference distribution using the **Population Stability Index
(PSI)** for key numeric features, tracks class-distribution shift in the
predicted High-Risk rate, checks p95 latency against a 200ms budget, and
flags out-of-range inputs. It was verified against 60 simulated live
requests and correctly surfaced moderate drift (PSI ≈ 0.10–0.14 on
`avg_internal_marks`/`G1`) at that sample size, writing a timestamped JSON
report to `monitoring/reports/`. Full retraining criteria and the model
lifecycle (versioning → training → registry → serving → monitoring →
retraining → rollback) are documented in `docs/model_lifecycle.md`.

### 13.6 Dataset and model versioning
`data/analytics/student_master.csv` and `ml/models/best_model.joblib` are
version-tracked with **DVC** on top of a Git repository — pointer files are
committed to Git while the underlying data/model files are content-addressed
in the DVC cache, so any model version can be traced back to its exact
training snapshot (see `docs/versioning.md`).

---
*Appendix: see `README.md` for setup/run instructions, `docs/architecture.md`
for the full architecture diagram, `docs/data_dictionary.md` for complete
column definitions and the SQL DDL, and `docs/model_lifecycle.md` /
`docs/versioning.md` for the Part 2 MLOps workflow.*
